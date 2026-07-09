"""Tests for form4lab.data.bulk_fetcher: URL generation, date parsing, ZIP parsing."""

import csv
import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from form4lab.data.bulk_fetcher import (
    generate_quarter_urls,
    _generate_quarter_urls_from,
    download_quarter_zip,
    _parse_date_ddmonyyyy,
    _parse_relationship,
    parse_quarter_zip,
    backfill_10b5_1_flags,
    BULK_DIR,
)
from form4lab.database import Base
from form4lab.models.transaction import Transaction
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """In-memory SQLite DB with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_tsv(rows: list[dict]) -> str:
    """Create a TSV string from a list of dicts."""
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _make_zip(files: dict[str, str], path: Path) -> Path:
    """Create a ZIP file with named TSV contents."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


# ---------------------------------------------------------------------------
# generate_quarter_urls
# ---------------------------------------------------------------------------

class TestGenerateQuarterUrls:
    @patch("form4lab.data.bulk_fetcher.date")
    def test_generates_correct_urls(self, mock_date):
        mock_date.today.return_value = date(2026, 2, 21)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        urls = generate_quarter_urls(years=1)

        assert len(urls) == 5  # 2025 Q1-Q4 + 2026 Q1
        assert urls[0] == (
            "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2025q1_form345.zip",
            "2025q1_form345.zip",
        )
        assert urls[-1] == (
            "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2026q1_form345.zip",
            "2026q1_form345.zip",
        )

    @patch("form4lab.data.bulk_fetcher.date")
    def test_ten_years_generates_many_quarters(self, mock_date):
        mock_date.today.return_value = date(2026, 2, 21)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        urls = generate_quarter_urls(years=10)
        # 2016 Q1-Q4 through 2025 Q4 = 40, plus 2026 Q1 = 41
        assert len(urls) == 41

    @patch("form4lab.data.bulk_fetcher.date")
    def test_q4_boundary(self, mock_date):
        """In Q4, should include Q1-Q4 of current year."""
        mock_date.today.return_value = date(2025, 12, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        urls = generate_quarter_urls(years=1)
        filenames = [f for _, f in urls]
        assert "2025q4_form345.zip" in filenames
        assert "2024q1_form345.zip" in filenames


# ---------------------------------------------------------------------------
# _parse_date_ddmonyyyy
# ---------------------------------------------------------------------------

class TestParseDateDdmonyyyy:
    def test_valid_date(self):
        assert _parse_date_ddmonyyyy("31-MAR-2025") == date(2025, 3, 31)

    def test_valid_date_lowercase(self):
        assert _parse_date_ddmonyyyy("15-jan-2020") == date(2020, 1, 15)

    def test_single_digit_day(self):
        assert _parse_date_ddmonyyyy("1-FEB-2024") == date(2024, 2, 1)

    def test_none(self):
        assert _parse_date_ddmonyyyy(None) is None

    def test_empty(self):
        assert _parse_date_ddmonyyyy("") is None

    def test_whitespace(self):
        assert _parse_date_ddmonyyyy("  ") is None

    def test_invalid_month(self):
        assert _parse_date_ddmonyyyy("01-XYZ-2025") is None

    def test_invalid_format(self):
        assert _parse_date_ddmonyyyy("2025-03-31") is None

    def test_with_surrounding_whitespace(self):
        assert _parse_date_ddmonyyyy(" 15-JUN-2023 ") == date(2023, 6, 15)


# ---------------------------------------------------------------------------
# _parse_relationship
# ---------------------------------------------------------------------------

class TestParseRelationship:
    def test_officer_only(self):
        r = _parse_relationship("Officer")
        assert r["is_officer"] is True
        assert r["is_director"] is False
        assert r["is_ten_pct_owner"] is False

    def test_director_only(self):
        r = _parse_relationship("Director")
        assert r["is_director"] is True
        assert r["is_officer"] is False

    def test_director_and_officer(self):
        r = _parse_relationship("Director,Officer")
        assert r["is_director"] is True
        assert r["is_officer"] is True
        assert r["is_ten_pct_owner"] is False

    def test_all_three(self):
        r = _parse_relationship("Director,Officer,TenPercentOwner")
        assert r["is_director"] is True
        assert r["is_officer"] is True
        assert r["is_ten_pct_owner"] is True

    def test_ten_percent_owner_only(self):
        r = _parse_relationship("TenPercentOwner")
        assert r["is_ten_pct_owner"] is True
        assert r["is_officer"] is False

    def test_none(self):
        r = _parse_relationship(None)
        assert r["is_officer"] is False
        assert r["is_director"] is False
        assert r["is_ten_pct_owner"] is False

    def test_empty_string(self):
        r = _parse_relationship("")
        assert r["is_officer"] is False

    def test_with_spaces(self):
        r = _parse_relationship("Director , Officer")
        assert r["is_director"] is True
        assert r["is_officer"] is True


# ---------------------------------------------------------------------------
# parse_quarter_zip — end-to-end with synthetic ZIP
# ---------------------------------------------------------------------------

class TestParseQuarterZip:
    def _make_test_zip(self, tmp_path, **overrides):
        """Create a synthetic ZIP with default test data."""
        submissions = overrides.get("submissions", [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000789",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TEST",
            "FILING_DATE": "15-MAR-2025",
        }])
        owners = overrides.get("owners", [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "RPTOWNERCIK": "0000123",
            "RPTOWNERNAME": "John Doe",
            "RPTOWNER_RELATIONSHIP": "Officer",
            "RPTOWNER_TITLE": "CEO",
        }])
        nonderiv = overrides.get("nonderiv", [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "TRANS_CODE": "P",
            "TRANS_DATE": "14-MAR-2025",
            "TRANS_SHARES": "1000",
            "TRANS_PRICEPERSHARE": "50.00",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "5000",
            "SECURITY_TITLE": "Common Stock",
        }])
        deriv = overrides.get("deriv", [])

        # Ensure deriv has the right columns even if empty
        if not deriv:
            deriv_content = "ACCESSION_NUMBER\tTRANS_CODE\tTRANS_DATE\n"
        else:
            deriv_content = _make_tsv(deriv)

        zip_path = tmp_path / "test_quarter.zip"
        return _make_zip({
            "SUBMISSION.tsv": _make_tsv(submissions),
            "REPORTINGOWNER.tsv": _make_tsv(owners),
            "NONDERIV_TRANS.tsv": _make_tsv(nonderiv),
            "DERIV_TRANS.tsv": deriv_content,
        }, zip_path)

    def test_basic_parse(self, tmp_path):
        zip_path = self._make_test_zip(tmp_path)
        results = parse_quarter_zip(zip_path, {"TEST"})

        assert len(results) == 1
        txn = results[0]
        assert txn["insider_cik"] == "123"
        assert txn["insider_name"] == "John Doe"
        assert txn["company_cik"] == "789"
        assert txn["company_ticker"] == "TEST"
        assert txn["transaction_code"] == "P"
        assert txn["shares"] == 1000.0
        assert txn["price_per_share"] == 50.0
        assert txn["filing_date"] == date(2025, 3, 15)
        assert txn["transaction_date"] == date(2025, 3, 14)
        assert txn["acquired_or_disposed"] == "A"
        assert txn["shares_owned_after"] == 5000.0
        assert txn["is_discretionary"] is True
        assert txn["is_common_stock"] is True
        assert txn["is_officer"] is True
        assert txn["officer_title"] == "CEO"

    def test_filters_by_ticker(self, tmp_path):
        zip_path = self._make_test_zip(tmp_path)
        results = parse_quarter_zip(zip_path, {"OTHER"})
        assert len(results) == 0

    def test_ticker_case_insensitive(self, tmp_path):
        zip_path = self._make_test_zip(tmp_path)
        results = parse_quarter_zip(zip_path, {"test"})
        assert len(results) == 1

    def test_filters_non_form4(self, tmp_path):
        submissions = [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "DOCUMENT_TYPE": "3",  # Form 3, not 4
            "ISSUERCIK": "0000789",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TEST",
            "FILING_DATE": "15-MAR-2025",
        }]
        zip_path = self._make_test_zip(tmp_path, submissions=submissions)
        results = parse_quarter_zip(zip_path, {"TEST"})
        assert len(results) == 0

    def test_accession_format_matches_xml_parser(self, tmp_path):
        """Verify accession_number format is compatible with existing dedup."""
        zip_path = self._make_test_zip(tmp_path)
        results = parse_quarter_zip(zip_path, {"TEST"})
        acc = results[0]["accession_number"]
        # Format: {base_accession}_{code}_{date}_{shares}
        assert acc == "0001234567-25-000001_P_2025-03-14_1000.0"

    def test_total_value_computed(self, tmp_path):
        zip_path = self._make_test_zip(tmp_path)
        results = parse_quarter_zip(zip_path, {"TEST"})
        assert results[0]["total_value"] == 50000.0

    def test_missing_price(self, tmp_path):
        nonderiv = [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "TRANS_CODE": "P",
            "TRANS_DATE": "14-MAR-2025",
            "TRANS_SHARES": "1000",
            "TRANS_PRICEPERSHARE": "",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "",
            "SECURITY_TITLE": "Common Stock",
        }]
        zip_path = self._make_test_zip(tmp_path, nonderiv=nonderiv)
        results = parse_quarter_zip(zip_path, {"TEST"})
        assert results[0]["price_per_share"] is None
        assert results[0]["total_value"] is None
        assert results[0]["shares_owned_after"] is None


# ---------------------------------------------------------------------------
# Multiple owners
# ---------------------------------------------------------------------------

class TestMultipleOwners:
    def test_two_owners_per_filing(self, tmp_path):
        """Two reporting owners on the same filing → two transaction dicts."""
        submissions = [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000789",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TEST",
            "FILING_DATE": "15-MAR-2025",
        }]
        owners = [
            {
                "ACCESSION_NUMBER": "0001234567-25-000001",
                "RPTOWNERCIK": "0000111",
                "RPTOWNERNAME": "Alice",
                "RPTOWNER_RELATIONSHIP": "Director",
                "RPTOWNER_TITLE": "",
            },
            {
                "ACCESSION_NUMBER": "0001234567-25-000001",
                "RPTOWNERCIK": "0000222",
                "RPTOWNERNAME": "Bob",
                "RPTOWNER_RELATIONSHIP": "Officer",
                "RPTOWNER_TITLE": "VP",
            },
        ]
        nonderiv = [{
            "ACCESSION_NUMBER": "0001234567-25-000001",
            "TRANS_CODE": "P",
            "TRANS_DATE": "14-MAR-2025",
            "TRANS_SHARES": "100",
            "TRANS_PRICEPERSHARE": "10.00",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "500",
            "SECURITY_TITLE": "Common Stock",
        }]

        zip_path = _make_zip({
            "SUBMISSION.tsv": _make_tsv(submissions),
            "REPORTINGOWNER.tsv": _make_tsv(owners),
            "NONDERIV_TRANS.tsv": _make_tsv(nonderiv),
            "DERIV_TRANS.tsv": "ACCESSION_NUMBER\tTRANS_CODE\tTRANS_DATE\n",
        }, tmp_path / "multi_owner.zip")

        results = parse_quarter_zip(zip_path, {"TEST"})
        assert len(results) == 2
        names = {r["insider_name"] for r in results}
        assert names == {"Alice", "Bob"}

        alice = [r for r in results if r["insider_name"] == "Alice"][0]
        assert alice["is_director"] is True
        assert alice["is_officer"] is False

        bob = [r for r in results if r["insider_name"] == "Bob"][0]
        assert bob["is_officer"] is True
        assert bob["is_director"] is False
        assert bob["officer_title"] == "VP"


# ---------------------------------------------------------------------------
# Download behavior
# ---------------------------------------------------------------------------

class TestDownload:
    def test_skips_cached_file(self, tmp_path):
        """Cached ZIP should be reused without network call."""
        # Create a fake cached file
        cached = tmp_path / "2025q1_form345.zip"
        cached.write_bytes(b"fake zip content")

        with patch("form4lab.data.bulk_fetcher.BULK_DIR", tmp_path):
            result = download_quarter_zip(
                "https://example.com/2025q1_form345.zip",
                "2025q1_form345.zip",
                redownload=False,
            )
        assert result == cached

    def test_redownload_forces_fetch(self, tmp_path):
        """With redownload=True, should fetch even if cached."""
        cached = tmp_path / "2025q1_form345.zip"
        cached.write_bytes(b"old content")

        mock_response = MagicMock()
        mock_response.content = b"new content"

        with patch("form4lab.data.bulk_fetcher.BULK_DIR", tmp_path), \
             patch("form4lab.data.bulk_fetcher._sec_get", return_value=mock_response):
            result = download_quarter_zip(
                "https://example.com/2025q1_form345.zip",
                "2025q1_form345.zip",
                redownload=True,
            )

        assert result == cached
        assert cached.read_bytes() == b"new content"

    def test_handles_404_gracefully(self, tmp_path):
        """404 should return None, not raise."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404

        error = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("form4lab.data.bulk_fetcher.BULK_DIR", tmp_path), \
             patch("form4lab.data.bulk_fetcher._sec_get", side_effect=error):
            result = download_quarter_zip(
                "https://example.com/2026q2_form345.zip",
                "2026q2_form345.zip",
            )

        assert result is None


# ---------------------------------------------------------------------------
# Integration with persist_transaction
# ---------------------------------------------------------------------------

class TestIntegrationWithPersist:
    def test_parsed_dict_persists_successfully(self, tmp_path, db_session):
        """Verify parsed transaction dicts work with persist_transaction."""
        from form4lab.data.utils import persist_transaction

        submissions = [{
            "ACCESSION_NUMBER": "0009999999-25-000099",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000999",
            "ISSUERNAME": "Integration Test Corp",
            "ISSUERTRADINGSYMBOL": "INTG",
            "FILING_DATE": "20-FEB-2025",
        }]
        owners = [{
            "ACCESSION_NUMBER": "0009999999-25-000099",
            "RPTOWNERCIK": "0000888",
            "RPTOWNERNAME": "Test Insider",
            "RPTOWNER_RELATIONSHIP": "Director,Officer",
            "RPTOWNER_TITLE": "CEO",
        }]
        nonderiv = [{
            "ACCESSION_NUMBER": "0009999999-25-000099",
            "TRANS_CODE": "P",
            "TRANS_DATE": "19-FEB-2025",
            "TRANS_SHARES": "2000",
            "TRANS_PRICEPERSHARE": "35.50",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "10000",
            "SECURITY_TITLE": "Common Stock",
        }]

        zip_path = _make_zip({
            "SUBMISSION.tsv": _make_tsv(submissions),
            "REPORTINGOWNER.tsv": _make_tsv(owners),
            "NONDERIV_TRANS.tsv": _make_tsv(nonderiv),
            "DERIV_TRANS.tsv": "ACCESSION_NUMBER\tTRANS_CODE\tTRANS_DATE\n",
        }, tmp_path / "integration.zip")

        results = parse_quarter_zip(zip_path, {"INTG"})
        assert len(results) == 1

        txn = persist_transaction(results[0], db_session)
        db_session.commit()

        assert txn is not None
        assert txn.shares == 2000.0
        assert txn.price_per_share == 35.5
        assert txn.transaction_code == "P"
        assert txn.filing_date == date(2025, 2, 20)
        assert txn.transaction_date == date(2025, 2, 19)

        # Verify dedup — second persist should return None
        txn2 = persist_transaction(results[0], db_session)
        assert txn2 is None

    def test_common_stock_classification(self, tmp_path):
        """Verify security_title flows through to classify_common_stock."""
        submissions = [{
            "ACCESSION_NUMBER": "0009999999-25-000099",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000999",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "CLSF",
            "FILING_DATE": "20-FEB-2025",
        }]
        owners = [{
            "ACCESSION_NUMBER": "0009999999-25-000099",
            "RPTOWNERCIK": "0000888",
            "RPTOWNERNAME": "Test Insider",
            "RPTOWNER_RELATIONSHIP": "Officer",
            "RPTOWNER_TITLE": "CFO",
        }]
        nonderiv = [
            {
                "ACCESSION_NUMBER": "0009999999-25-000099",
                "TRANS_CODE": "P",
                "TRANS_DATE": "19-FEB-2025",
                "TRANS_SHARES": "100",
                "TRANS_PRICEPERSHARE": "10.00",
                "TRANS_ACQUIRED_DISP_CD": "A",
                "SHRS_OWND_FOLWNG_TRANS": "200",
                "SECURITY_TITLE": "Series A Preferred Stock",
            },
            {
                "ACCESSION_NUMBER": "0009999999-25-000099",
                "TRANS_CODE": "P",
                "TRANS_DATE": "19-FEB-2025",
                "TRANS_SHARES": "200",
                "TRANS_PRICEPERSHARE": "20.00",
                "TRANS_ACQUIRED_DISP_CD": "A",
                "SHRS_OWND_FOLWNG_TRANS": "400",
                "SECURITY_TITLE": "Common Stock",
            },
        ]

        zip_path = _make_zip({
            "SUBMISSION.tsv": _make_tsv(submissions),
            "REPORTINGOWNER.tsv": _make_tsv(owners),
            "NONDERIV_TRANS.tsv": _make_tsv(nonderiv),
            "DERIV_TRANS.tsv": "ACCESSION_NUMBER\tTRANS_CODE\tTRANS_DATE\n",
        }, tmp_path / "classify.zip")

        results = parse_quarter_zip(zip_path, {"CLSF"})
        preferred = [r for r in results if r["shares"] == 100.0][0]
        common = [r for r in results if r["shares"] == 200.0][0]

        assert preferred["is_common_stock"] is False
        assert common["is_common_stock"] is True


# ---------------------------------------------------------------------------
# parse_10b5_1_flag
# ---------------------------------------------------------------------------

from form4lab.data.utils import parse_10b5_1_flag

class TestParse10b51Flag:
    def test_true_numeric(self):
        assert parse_10b5_1_flag("1") is True

    def test_true_string(self):
        assert parse_10b5_1_flag("true") is True

    def test_true_string_uppercase(self):
        assert parse_10b5_1_flag("True") is True

    def test_false_numeric(self):
        assert parse_10b5_1_flag("0") is False

    def test_false_string(self):
        assert parse_10b5_1_flag("false") is False

    def test_empty_string(self):
        assert parse_10b5_1_flag("") is None

    def test_none(self):
        assert parse_10b5_1_flag(None) is None

    def test_whitespace(self):
        assert parse_10b5_1_flag("  1  ") is True

    def test_unexpected_value(self):
        assert parse_10b5_1_flag("maybe") is None


# ---------------------------------------------------------------------------
# 10b5-1 plan indicator from bulk TSV
# ---------------------------------------------------------------------------

class TestBulk10b51Plan:
    """Tests for 10b5-1 plan indicator from bulk TSV data."""

    def test_plan_flag_true_from_submission(self, db_session, tmp_path):
        submissions = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000001",
            "FILING_DATE": "2024-06-15",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000001234",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TST",
            "AFF10B5ONE": "1",
        }])
        owners = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000001",
            "RPTOWNERCIK": "0000005678",
            "RPTOWNERNAME": "Jane Doe",
            "RPTOWNER_RELATIONSHIP": "Officer",
            "RPTOWNER_TITLE": "CEO",
        }])
        nonderiv = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000001",
            "SECURITY_TITLE": "Common Stock",
            "TRANS_DATE": "2024-06-15",
            "TRANS_CODE": "P",
            "TRANS_SHARES": "500",
            "TRANS_PRICEPERSHARE": "50.00",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "1500",
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
            "REPORTINGOWNER.tsv": owners,
            "NONDERIV_TRANS.tsv": nonderiv,
            "DERIV_TRANS.tsv": _make_tsv([]),
        }, tmp_path / "test.zip")
        results = parse_quarter_zip(zip_path, {"TST"})
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is True

    def test_plan_flag_false_from_submission(self, db_session, tmp_path):
        submissions = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000002",
            "FILING_DATE": "2024-06-15",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000001234",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TST",
            "AFF10B5ONE": "false",
        }])
        owners = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000002",
            "RPTOWNERCIK": "0000005678",
            "RPTOWNERNAME": "Jane Doe",
            "RPTOWNER_RELATIONSHIP": "Officer",
            "RPTOWNER_TITLE": "CEO",
        }])
        nonderiv = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-24-000002",
            "SECURITY_TITLE": "Common Stock",
            "TRANS_DATE": "2024-06-15",
            "TRANS_CODE": "S",
            "TRANS_SHARES": "200",
            "TRANS_PRICEPERSHARE": "55.00",
            "TRANS_ACQUIRED_DISP_CD": "D",
            "SHRS_OWND_FOLWNG_TRANS": "1300",
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
            "REPORTINGOWNER.tsv": owners,
            "NONDERIV_TRANS.tsv": nonderiv,
            "DERIV_TRANS.tsv": _make_tsv([]),
        }, tmp_path / "test.zip")
        results = parse_quarter_zip(zip_path, {"TST"})
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is False

    def test_plan_flag_missing_from_submission(self, db_session, tmp_path):
        submissions = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-22-000003",
            "FILING_DATE": "2022-01-15",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000001234",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TST",
        }])
        owners = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-22-000003",
            "RPTOWNERCIK": "0000005678",
            "RPTOWNERNAME": "Jane Doe",
            "RPTOWNER_RELATIONSHIP": "Director",
            "RPTOWNER_TITLE": "",
        }])
        nonderiv = _make_tsv([{
            "ACCESSION_NUMBER": "0001234-22-000003",
            "SECURITY_TITLE": "Common Stock",
            "TRANS_DATE": "2022-01-15",
            "TRANS_CODE": "P",
            "TRANS_SHARES": "100",
            "TRANS_PRICEPERSHARE": "30.00",
            "TRANS_ACQUIRED_DISP_CD": "A",
            "SHRS_OWND_FOLWNG_TRANS": "100",
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
            "REPORTINGOWNER.tsv": owners,
            "NONDERIV_TRANS.tsv": nonderiv,
            "DERIV_TRANS.tsv": _make_tsv([]),
        }, tmp_path / "test.zip")
        results = parse_quarter_zip(zip_path, {"TST"})
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is None


# ---------------------------------------------------------------------------
# persist_transaction with is_10b5_1_plan
# ---------------------------------------------------------------------------

class TestPersistTransaction10b51:
    """Test that persist_transaction stores is_10b5_1_plan."""

    def test_persist_plan_trade(self, db_session):
        from form4lab.data.utils import persist_transaction
        txn_data = {
            "insider_cik": "999", "insider_name": "Plan Trader",
            "is_officer": True, "is_director": False, "is_ten_pct_owner": False,
            "officer_title": "CFO", "company_cik": "888",
            "company_name": "Plan Corp", "company_ticker": "PLAN",
            "accession_number": "ACC-PLAN-001",
            "filing_date": date(2024, 6, 15), "transaction_date": date(2024, 6, 15),
            "transaction_code": "S", "shares": 1000.0, "price_per_share": 50.0,
            "total_value": 50000.0, "shares_owned_after": 5000.0,
            "acquired_or_disposed": "D", "is_discretionary": False,
            "security_title": "Common Stock",
            "is_common_stock": True, "is_10b5_1_plan": True,
        }
        txn = persist_transaction(txn_data, db_session)
        db_session.flush()
        assert txn is not None
        assert txn.is_10b5_1_plan is True

    def test_persist_non_plan_trade(self, db_session):
        from form4lab.data.utils import persist_transaction
        txn_data = {
            "insider_cik": "999", "insider_name": "Disc Buyer",
            "is_officer": True, "is_director": False, "is_ten_pct_owner": False,
            "officer_title": "CEO", "company_cik": "888",
            "company_name": "Plan Corp", "company_ticker": "PLAN",
            "accession_number": "ACC-PLAN-002",
            "filing_date": date(2024, 6, 15), "transaction_date": date(2024, 6, 15),
            "transaction_code": "P", "shares": 500.0, "price_per_share": 50.0,
            "total_value": 25000.0, "shares_owned_after": 5500.0,
            "acquired_or_disposed": "A", "is_discretionary": True,
            "security_title": "Common Stock",
            "is_common_stock": True, "is_10b5_1_plan": False,
        }
        txn = persist_transaction(txn_data, db_session)
        db_session.flush()
        assert txn is not None
        assert txn.is_10b5_1_plan is False

    def test_persist_unknown_plan_status(self, db_session):
        from form4lab.data.utils import persist_transaction
        txn_data = {
            "insider_cik": "999", "insider_name": "Old Buyer",
            "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
            "officer_title": "", "company_cik": "888",
            "company_name": "Plan Corp", "company_ticker": "PLAN",
            "accession_number": "ACC-PLAN-003",
            "filing_date": date(2022, 1, 15), "transaction_date": date(2022, 1, 15),
            "transaction_code": "P", "shares": 100.0, "price_per_share": 30.0,
            "total_value": 3000.0, "shares_owned_after": 100.0,
            "acquired_or_disposed": "A", "is_discretionary": True,
            "security_title": "Common Stock",
            "is_common_stock": True, "is_10b5_1_plan": None,
        }
        txn = persist_transaction(txn_data, db_session)
        db_session.flush()
        assert txn is not None
        assert txn.is_10b5_1_plan is None


# ---------------------------------------------------------------------------
# _generate_quarter_urls_from
# ---------------------------------------------------------------------------

class TestGenerateQuarterUrlsFrom:
    @patch("form4lab.data.bulk_fetcher.date")
    def test_from_2023_q2(self, mock_date):
        mock_date.today.return_value = date(2026, 2, 24)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        urls = _generate_quarter_urls_from(2023, 2)
        filenames = [f for _, f in urls]
        assert filenames[0] == "2023q2_form345.zip"
        assert "2023q1_form345.zip" not in filenames
        assert "2026q1_form345.zip" in filenames

    @patch("form4lab.data.bulk_fetcher.date")
    def test_count(self, mock_date):
        mock_date.today.return_value = date(2026, 2, 24)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        urls = _generate_quarter_urls_from(2023, 2)
        # 2023 Q2-Q4 (3) + 2024 Q1-Q4 (4) + 2025 Q1-Q4 (4) + 2026 Q1 (1) = 12
        assert len(urls) == 12


# ---------------------------------------------------------------------------
# backfill_10b5_1_flags — retroactive update
# ---------------------------------------------------------------------------

class TestBackfill10b51Flags:
    def _insert_transaction(self, db_session, accession_prefix, code="P",
                            trans_date="2024-06-15", shares=100.0):
        """Insert a transaction with NULL is_10b5_1_plan for testing."""
        from form4lab.data.utils import persist_transaction
        acc_key = f"{accession_prefix}_{code}_{trans_date}_{shares}"
        txn_data = {
            "insider_cik": "999", "insider_name": "Test Insider",
            "is_officer": True, "is_director": False, "is_ten_pct_owner": False,
            "officer_title": "CEO", "company_cik": "888",
            "company_name": "Test Corp", "company_ticker": "TST",
            "accession_number": acc_key,
            "filing_date": date(2024, 6, 16),
            "transaction_date": date(2024, 6, 15),
            "transaction_code": code, "shares": shares,
            "price_per_share": 50.0, "total_value": shares * 50.0,
            "shares_owned_after": 5000.0,
            "acquired_or_disposed": "A", "is_discretionary": code == "P",
            "security_title": "Common Stock",
            "is_common_stock": True, "is_10b5_1_plan": None,
        }
        txn = persist_transaction(txn_data, db_session)
        db_session.commit()
        return txn

    def test_updates_null_flags(self, db_session, tmp_path):
        """Transactions with NULL is_10b5_1_plan get updated from TSV data."""
        acc = "0001234-24-000001"
        txn = self._insert_transaction(db_session, acc)
        assert txn.is_10b5_1_plan is None

        # Create a ZIP with AFF10B5ONE=1 for this accession
        submissions = _make_tsv([{
            "ACCESSION_NUMBER": acc,
            "FILING_DATE": "15-JUN-2024",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000888",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TST",
            "AFF10B5ONE": "1",
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
        }, tmp_path / "2024q2_form345.zip")

        with patch("form4lab.data.bulk_fetcher._generate_quarter_urls_from") as mock_urls, \
             patch("form4lab.data.bulk_fetcher.download_quarter_zip") as mock_dl:
            mock_urls.return_value = [
                ("https://example.com/2024q2_form345.zip", "2024q2_form345.zip"),
            ]
            mock_dl.return_value = zip_path

            stats = backfill_10b5_1_flags(db_session)

        assert stats["transactions_updated"] == 1
        assert stats["accessions_with_flag"] == 1

        # Verify the transaction was updated
        db_session.expire_all()
        updated = db_session.query(Transaction).filter(
            Transaction.accession_number.like(f"{acc}_%")
        ).first()
        assert updated.is_10b5_1_plan is True

    def test_skips_already_set_flags(self, db_session, tmp_path):
        """Transactions that already have is_10b5_1_plan set are not overwritten."""
        from form4lab.data.utils import persist_transaction
        acc = "0001234-24-000010"
        acc_key = f"{acc}_P_2024-06-15_100.0"
        txn_data = {
            "insider_cik": "997", "insider_name": "Already Set",
            "is_officer": True, "is_director": False, "is_ten_pct_owner": False,
            "officer_title": "CEO", "company_cik": "886",
            "company_name": "Set Corp", "company_ticker": "SET",
            "accession_number": acc_key,
            "filing_date": date(2024, 6, 16),
            "transaction_date": date(2024, 6, 15),
            "transaction_code": "P", "shares": 100.0,
            "price_per_share": 50.0, "total_value": 5000.0,
            "shares_owned_after": 5000.0,
            "acquired_or_disposed": "A", "is_discretionary": True,
            "security_title": "Common Stock",
            "is_common_stock": True, "is_10b5_1_plan": False,  # Already set
        }
        persist_transaction(txn_data, db_session)
        db_session.commit()

        submissions = _make_tsv([{
            "ACCESSION_NUMBER": acc,
            "FILING_DATE": "15-JUN-2024",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000886",
            "ISSUERNAME": "Set Corp",
            "ISSUERTRADINGSYMBOL": "SET",
            "AFF10B5ONE": "1",  # Would set to True, but should be skipped
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
        }, tmp_path / "2024q2_form345.zip")

        with patch("form4lab.data.bulk_fetcher._generate_quarter_urls_from") as mock_urls, \
             patch("form4lab.data.bulk_fetcher.download_quarter_zip") as mock_dl:
            mock_urls.return_value = [
                ("https://example.com/2024q2_form345.zip", "2024q2_form345.zip"),
            ]
            mock_dl.return_value = zip_path

            stats = backfill_10b5_1_flags(db_session)

        assert stats["transactions_updated"] == 0

        db_session.expire_all()
        txn = db_session.query(Transaction).filter(
            Transaction.accession_number == acc_key
        ).first()
        assert txn.is_10b5_1_plan is False  # Unchanged

    def test_updates_multiple_transactions_same_accession(self, db_session, tmp_path):
        """Multiple transactions from the same filing all get updated."""
        acc = "0001234-24-000020"
        self._insert_transaction(db_session, acc, code="P", shares=100.0)
        self._insert_transaction(db_session, acc, code="S", shares=200.0)

        submissions = _make_tsv([{
            "ACCESSION_NUMBER": acc,
            "FILING_DATE": "15-JUN-2024",
            "DOCUMENT_TYPE": "4",
            "ISSUERCIK": "0000888",
            "ISSUERNAME": "Test Corp",
            "ISSUERTRADINGSYMBOL": "TST",
            "AFF10B5ONE": "0",
        }])
        zip_path = _make_zip({
            "SUBMISSION.tsv": submissions,
        }, tmp_path / "2024q2_form345.zip")

        with patch("form4lab.data.bulk_fetcher._generate_quarter_urls_from") as mock_urls, \
             patch("form4lab.data.bulk_fetcher.download_quarter_zip") as mock_dl:
            mock_urls.return_value = [
                ("https://example.com/2024q2_form345.zip", "2024q2_form345.zip"),
            ]
            mock_dl.return_value = zip_path

            stats = backfill_10b5_1_flags(db_session)

        assert stats["transactions_updated"] == 2

        db_session.expire_all()
        txns = db_session.query(Transaction).filter(
            Transaction.accession_number.like(f"{acc}_%")
        ).all()
        assert all(t.is_10b5_1_plan is False for t in txns)
