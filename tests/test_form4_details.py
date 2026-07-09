"""Tests for Form 4 detail ingestion (SUBMISSION/DERIV_TRANS/FOOTNOTES bulk
TSVs) and footnote classification.

Covers three areas: parsing individual SUBMISSION/DERIV_TRANS/FOOTNOTES rows
into ingestion-ready dicts (including SEC's quirky field names and blank/
missing values), classifying footnote text for 10b5-1 trading-plan language
and mechanical-acquisition patterns (dividend reinvestment, 401(k),
employee stock purchase plans, ownership guidelines, employment
agreements), and end-to-end bulk ingestion from a quarterly ZIP into SQLite
— including idempotency (a second pass over the same file adds no
duplicate rows).
"""
import zipfile
from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.data.form4_details_fetcher import (
    classify_filing_footnotes,
    classify_footnote,
    ingest_quarter_details,
    load_target_accessions,
    parse_deriv_row,
    parse_footnote_row,
    parse_submission_row,
)


# ---------------------------------------------------------------------------
# Row parsers
# ---------------------------------------------------------------------------

class TestRowParsers:
    def test_deriv_row_full(self):
        row = {
            "ACCESSION_NUMBER": "0001213900-18-008553", "DERIV_TRANS_SK": "419566",
            "SECURITY_TITLE": "Stock Option (Right to Buy)",
            "CONV_EXERCISE_PRICE": "6.01", "TRANS_DATE": "27-JUN-2018",
            "TRANS_CODE": "M", "TRANS_TIMELINESS": "E",
            "TRANS_SHARES": "250000.0", "TRANS_PRICEPERSHARE": "",
            "TRANS_ACQUIRED_DISP_CD": "D",
            "EXCERCISE_DATE": "01-JAN-2019", "EXPIRATION_DATE": "27-JUN-2026",
            "UNDLYNG_SEC_SHARES": "250000.0", "SHRS_OWND_FOLWNG_TRANS": "0.0",
        }
        p = parse_deriv_row(row)
        assert p["accession_number"] == "0001213900-18-008553"
        assert p["deriv_trans_sk"] == 419566
        assert p["conv_exercise_price"] == pytest.approx(6.01)
        assert p["trans_date"] == date(2018, 6, 27)
        assert p["exercisable_date"] == date(2019, 1, 1)  # SEC's EXCERCISE_DATE [sic]
        assert p["expiration_date"] == date(2026, 6, 27)
        assert p["trans_price_per_share"] is None  # empty string -> None
        assert p["timeliness"] == "E"

    def test_deriv_row_missing_keys_returns_none(self):
        assert parse_deriv_row({"ACCESSION_NUMBER": "", "DERIV_TRANS_SK": "1"}) is None
        assert parse_deriv_row({"ACCESSION_NUMBER": "X", "DERIV_TRANS_SK": ""}) is None

    def test_deriv_row_blank_dates(self):
        p = parse_deriv_row({
            "ACCESSION_NUMBER": "A", "DERIV_TRANS_SK": "7",
            "TRANS_DATE": "", "EXPIRATION_DATE": "  ",
        })
        assert p["trans_date"] is None
        assert p["expiration_date"] is None

    def test_submission_row(self):
        p = parse_submission_row({
            "ACCESSION_NUMBER": "0001209191-18-034404",
            "FILING_DATE": "31-MAY-2018", "PERIOD_OF_REPORT": "30-MAY-2018",
            "REMARKS": "",
        })
        assert p["filing_date"] == date(2018, 5, 31)
        assert p["period_of_report"] == date(2018, 5, 30)
        assert p["remarks"] is None
        assert p["source"] == "bulk_tsv"

    def test_footnote_row(self):
        assert parse_footnote_row(
            {"ACCESSION_NUMBER": "A", "FOOTNOTE_ID": "F1", "FOOTNOTE_TXT": "hi"}
        ) == {"accession_number": "A", "footnote_id": "F1", "text": "hi"}
        assert parse_footnote_row(
            {"ACCESSION_NUMBER": "A", "FOOTNOTE_ID": "F1", "FOOTNOTE_TXT": ""}
        ) is None


# ---------------------------------------------------------------------------
# Footnote classifier
# ---------------------------------------------------------------------------

class TestFootnoteClassifier:
    def test_plan_positive(self):
        c = classify_footnote(
            "The reported transactions were effected pursuant to a Rule 10b5-1 "
            "trading plan adopted on March 15, 2019."
        )
        assert c["is_plan"] is True
        assert c["plan_negated"] is False
        assert c["plan_adoption_date"] == "March 15, 2019"

    def test_plan_negated(self):
        c = classify_footnote(
            "This purchase was not made pursuant to a Rule 10b5-1 trading plan."
        )
        assert c["is_plan"] is False
        assert c["plan_negated"] is True

    def test_plan_other_than_negation(self):
        c = classify_footnote(
            "Shares acquired other than pursuant to the issuer's 10b5-1 plan."
        )
        assert c["is_plan"] is False
        assert c["plan_negated"] is True

    def test_mechanical_kinds(self):
        assert classify_footnote(
            "Shares acquired under the issuer's dividend reinvestment plan."
        )["mechanical_kinds"] == ["drip"]
        assert classify_footnote(
            "Represents shares held in the reporting person's 401(k) account."
        )["mechanical_kinds"] == ["retirement_401k"]
        assert classify_footnote(
            "Purchased under the Employee Stock Purchase Plan."
        )["mechanical_kinds"] == ["espp"]
        assert classify_footnote(
            "Purchase made to satisfy the company's stock ownership guidelines."
        )["mechanical_kinds"] == ["ownership_guideline"]
        assert classify_footnote(
            "Shares purchased pursuant to his employment agreement."
        )["mechanical_kinds"] == ["employment_agreement"]

    def test_plain_text_negative(self):
        c = classify_footnote(
            "Includes shares held indirectly through a family trust."
        )
        assert c["is_plan"] is False
        assert c["mechanical_kinds"] == []

    def test_filing_aggregation(self):
        flags = classify_filing_footnotes([
            ("ACC1", "Effected pursuant to a Rule 10b5-1 plan."),
            ("ACC1", "Includes 401(k) plan shares."),
            ("ACC2", "Held in a family trust."),
        ])
        assert flags["ACC1"]["fn_plan"] is True
        assert flags["ACC1"]["fn_mechanical"] is True
        assert "retirement_401k" in flags["ACC1"]["fn_mechanical_kinds"]
        assert flags["ACC2"]["fn_plan"] is False
        assert flags["ACC2"]["fn_mechanical"] is False


# ---------------------------------------------------------------------------
# Ingestion (synthetic ZIP, in-memory SQLite, idempotency)
# ---------------------------------------------------------------------------

def _tsv(rows: list[dict], cols: list[str]) -> bytes:
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines).encode()


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_zip(tmp_path, accession="0001-23-000111"):
    sub_cols = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT",
                "DOCUMENT_TYPE", "REMARKS"]
    deriv_cols = ["ACCESSION_NUMBER", "DERIV_TRANS_SK", "SECURITY_TITLE",
                  "CONV_EXERCISE_PRICE", "TRANS_DATE", "TRANS_CODE",
                  "TRANS_TIMELINESS", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                  "TRANS_ACQUIRED_DISP_CD", "EXCERCISE_DATE", "EXPIRATION_DATE",
                  "UNDLYNG_SEC_SHARES", "SHRS_OWND_FOLWNG_TRANS"]
    fn_cols = ["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"]
    path = tmp_path / "2020q1_form345.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv([
            {"ACCESSION_NUMBER": accession, "FILING_DATE": "02-MAR-2020",
             "PERIOD_OF_REPORT": "01-MAR-2020", "DOCUMENT_TYPE": "4"},
            {"ACCESSION_NUMBER": "0009-99-999999", "FILING_DATE": "02-MAR-2020",
             "PERIOD_OF_REPORT": "01-MAR-2020", "DOCUMENT_TYPE": "4"},  # not a target
        ], sub_cols))
        zf.writestr("DERIV_TRANS.tsv", _tsv([
            {"ACCESSION_NUMBER": accession, "DERIV_TRANS_SK": "11",
             "SECURITY_TITLE": "Stock Option", "CONV_EXERCISE_PRICE": "10.0",
             "TRANS_DATE": "01-MAR-2020", "TRANS_CODE": "M",
             "TRANS_SHARES": "1000.0", "TRANS_ACQUIRED_DISP_CD": "D",
             "EXPIRATION_DATE": "01-MAR-2028"},
            {"ACCESSION_NUMBER": "0009-99-999999", "DERIV_TRANS_SK": "12",
             "TRANS_DATE": "01-MAR-2020", "TRANS_CODE": "M"},  # not a target
        ], deriv_cols))
        zf.writestr("FOOTNOTES.tsv", _tsv([
            {"ACCESSION_NUMBER": accession, "FOOTNOTE_ID": "F1",
             "FOOTNOTE_TXT": "Pursuant to a Rule 10b5-1 plan adopted on January 2, 2020."},
        ], fn_cols))
    return path


class TestIngestion:
    def test_ingest_filters_and_is_idempotent(self, db, tmp_path):
        acc = "0001-23-000111"
        # a transaction whose synthetic accession prefixes to our target
        db.execute(text(
            "INSERT INTO companies (cik, name, ticker) VALUES ('123', 'T Corp', 'TC')"))
        db.execute(text(
            "INSERT INTO insiders (cik, name, is_institution) VALUES ('456', 'A B', 0)"))
        db.execute(text(f"""
            INSERT INTO transactions (insider_id, company_id, accession_number,
                filing_date, transaction_date, transaction_code, shares,
                acquired_or_disposed, is_discretionary)
            VALUES (1, 1, '{acc}_M_2020-03-01_1000.0', '2020-03-02', '2020-03-01',
                    'M', 1000.0, 'A', 0)
        """))
        db.commit()

        targets = load_target_accessions(db)
        assert targets == {acc: 1}

        zip_path = _make_zip(tmp_path, accession=acc)
        existing: set = set()
        stats = ingest_quarter_details(zip_path, targets, db, existing)
        assert stats == {"meta": 1, "deriv": 1, "footnotes": 1}

        # the non-target filing was filtered out
        n_meta = db.execute(text("SELECT COUNT(*) FROM form4_filing_meta")).scalar()
        assert n_meta == 1
        exp = db.execute(text(
            "SELECT expiration_date FROM form4_deriv_txns")).scalar()
        assert str(exp).startswith("2028-03-01")

        # idempotent: second pass with a fresh `existing` set adds nothing
        stats2 = ingest_quarter_details(
            zip_path, targets, db, {r[0] for r in db.execute(
                text("SELECT accession_number FROM form4_filing_meta"))})
        assert stats2 == {"meta": 0, "deriv": 0, "footnotes": 0}
