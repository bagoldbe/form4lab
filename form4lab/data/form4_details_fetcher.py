"""Form 4 detail ingestion from the SEC bulk insider datasets.

Re-parses the quarterly ZIPs already cached in data/bulk/ for the detail the
live pipeline drops on the floor: derivative Table II rows (option expiration
dates → early-exercise research; conversion/exercise prices) and footnote text
(→ pre-2023 10b5-1 recovery + mechanical-buy taxonomy). Targets ONLY filings
already present in the `transactions` table (joined via the synthetic
accession prefix), so volume is bounded by the configured universe.

Research-only: writes to form4_filing_meta / form4_deriv_txns / form4_footnotes;
the live transactions pipeline never reads these tables.
"""
import logging
import zipfile
from pathlib import Path

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from form4lab.data.bulk_fetcher import (
    BULK_DIR,
    BULK_URL_BASE,
    _generate_quarter_urls_from,
    _parse_date_ddmonyyyy,
    _read_tsv,
    _safe_float,
    download_quarter_zip,
)
from form4lab.models.form4_detail import Form4DerivTxn, Form4FilingMeta, Form4Footnote

logger = logging.getLogger(__name__)

# Earliest year of bulk data to pull: transaction history in this dataset
# starts ~2015-2016, and a full year of prior-year lookback is enough to
# cover any earlier related event a 2016 transaction might reference.
EARLIEST_YEAR = 2015


# ---------------------------------------------------------------------------
# Pure row parsers (unit-testable without ZIPs)
# ---------------------------------------------------------------------------

def parse_submission_row(row: dict) -> dict | None:
    """SUBMISSION.tsv row -> form4_filing_meta dict (sans company_id)."""
    acc = (row.get("ACCESSION_NUMBER") or "").strip()
    if not acc:
        return None
    return {
        "accession_number": acc,
        "filing_date": _parse_date_ddmonyyyy(row.get("FILING_DATE")),
        "period_of_report": _parse_date_ddmonyyyy(row.get("PERIOD_OF_REPORT")),
        "remarks": (row.get("REMARKS") or "").strip() or None,
        "source": "bulk_tsv",
    }


def parse_deriv_row(row: dict) -> dict | None:
    """DERIV_TRANS.tsv row -> form4_deriv_txns dict."""
    acc = (row.get("ACCESSION_NUMBER") or "").strip()
    sk = (row.get("DERIV_TRANS_SK") or "").strip()
    if not acc or not sk:
        return None
    return {
        "accession_number": acc,
        "deriv_trans_sk": int(sk),
        "security_title": (row.get("SECURITY_TITLE") or "").strip() or None,
        "trans_date": _parse_date_ddmonyyyy(row.get("TRANS_DATE")),
        "trans_code": (row.get("TRANS_CODE") or "").strip() or None,
        "conv_exercise_price": _safe_float(row.get("CONV_EXERCISE_PRICE")),
        "trans_shares": _safe_float(row.get("TRANS_SHARES")),
        "trans_price_per_share": _safe_float(row.get("TRANS_PRICEPERSHARE")),
        "acquired_disposed": (row.get("TRANS_ACQUIRED_DISP_CD") or "").strip() or None,
        # SEC's own header misspells it: EXCERCISE_DATE = first-exercisable date
        "exercisable_date": _parse_date_ddmonyyyy(row.get("EXCERCISE_DATE")),
        "expiration_date": _parse_date_ddmonyyyy(row.get("EXPIRATION_DATE")),
        "underlying_shares": _safe_float(row.get("UNDLYNG_SEC_SHARES")),
        "shares_owned_after": _safe_float(row.get("SHRS_OWND_FOLWNG_TRANS")),
        "timeliness": (row.get("TRANS_TIMELINESS") or "").strip() or None,
    }


def parse_footnote_row(row: dict) -> dict | None:
    """FOOTNOTES.tsv row -> form4_footnotes dict."""
    acc = (row.get("ACCESSION_NUMBER") or "").strip()
    fid = (row.get("FOOTNOTE_ID") or "").strip()
    txt = (row.get("FOOTNOTE_TXT") or "").strip()
    if not acc or not fid or not txt:
        return None
    return {"accession_number": acc, "footnote_id": fid, "text": txt}


# ---------------------------------------------------------------------------
# Footnote classification (pure; regexes evolve, raw text is the store)
# ---------------------------------------------------------------------------

_RE_PLAN = re.compile(r"10b5[-‐-―]?1", re.IGNORECASE)
_RE_PLAN_NEGATED = re.compile(
    r"(?:not|other\s+than)[^.]{0,80}?(?:pursuant\s+to|under)[^.]{0,40}?10b5[-‐-―]?1",
    re.IGNORECASE,
)
_RE_PLAN_ADOPTED = re.compile(
    r"(?:adopted|entered\s+into|dated)\s+(?:on\s+|as\s+of\s+)?"
    r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
)
_RE_MECHANICAL = [
    ("drip", re.compile(r"dividend\s+reinvestment", re.IGNORECASE)),
    ("retirement_401k", re.compile(r"401\s*\(\s*k\s*\)", re.IGNORECASE)),
    ("espp", re.compile(r"employee\s+stock\s+purchase\s+plan|\bESPP\b", re.IGNORECASE)),
    ("deferred_comp", re.compile(r"deferred\s+compensation", re.IGNORECASE)),
    ("ownership_guideline", re.compile(
        r"(?:stock|share)\s+ownership\s+(?:guideline|requirement)", re.IGNORECASE)),
    ("employment_agreement", re.compile(
        r"(?:pursuant\s+to|under)\s+(?:his|her|the|an)\s+employment\s+agreement",
        re.IGNORECASE)),
]


def classify_footnote(text_: str) -> dict:
    """Classify a single footnote's text.

    Returns {is_plan, plan_negated, plan_adoption_date, mechanical_kinds}.
    `is_plan` means an affirmative 10b5-1 mention (negated mentions like
    "not made pursuant to a 10b5-1 plan" set plan_negated instead).
    Filing-level caveat: a footnote can describe a different transaction in
    the same filing — per-field *_FN attribution is a future refinement.
    """
    mentions_plan = bool(_RE_PLAN.search(text_))
    negated = bool(_RE_PLAN_NEGATED.search(text_)) if mentions_plan else False
    adoption = None
    if mentions_plan and not negated:
        m = _RE_PLAN_ADOPTED.search(text_)
        adoption = m.group(1) if m else None
    kinds = [kind for kind, rx in _RE_MECHANICAL if rx.search(text_)]
    return {
        "is_plan": mentions_plan and not negated,
        "plan_negated": negated,
        "plan_adoption_date": adoption,
        "mechanical_kinds": kinds,
    }


def classify_filing_footnotes(footnotes: list[tuple[str, str]]) -> dict[str, dict]:
    """Aggregate footnote classifications to filing level.

    Args:
        footnotes: (accession_number, text) pairs.
    Returns:
        accession -> {fn_plan, fn_plan_negated, fn_mechanical, fn_mechanical_kinds}
    """
    out: dict[str, dict] = {}
    for acc, txt in footnotes:
        c = classify_footnote(txt)
        agg = out.setdefault(acc, {
            "fn_plan": False, "fn_plan_negated": False,
            "fn_mechanical": False, "fn_mechanical_kinds": set(),
        })
        agg["fn_plan"] = agg["fn_plan"] or c["is_plan"]
        agg["fn_plan_negated"] = agg["fn_plan_negated"] or c["plan_negated"]
        if c["mechanical_kinds"]:
            agg["fn_mechanical"] = True
            agg["fn_mechanical_kinds"].update(c["mechanical_kinds"])
    return out


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_target_accessions(db: Session) -> dict[str, int]:
    """RAW accession -> company_id for every filing in our transactions table.

    transactions.accession_number is synthetic ({acc}_{code}_{date}_{shares});
    the split happens Python-side so this works on SQLite (tests) and Postgres.
    """
    rows = db.execute(text(
        "SELECT DISTINCT accession_number, company_id FROM transactions"
    )).fetchall()
    out: dict[str, int] = {}
    for acc, cid in rows:
        if acc:
            out[acc.split("_", 1)[0]] = cid
    return out


def _existing_meta_accessions(db: Session) -> set[str]:
    rows = db.execute(text("SELECT accession_number FROM form4_filing_meta")).fetchall()
    return {r[0] for r in rows}


def ingest_quarter_details(zip_path: Path, targets: dict[str, int],
                           db: Session, existing: set[str]) -> dict:
    """Ingest one quarter's filing meta + derivative rows + footnotes.

    Idempotent at filing granularity: accessions already in form4_filing_meta
    are skipped entirely (a filing's children are written in the same commit
    as its meta row). Returns per-quarter counts.
    """
    stats = {"meta": 0, "deriv": 0, "footnotes": 0}
    with zipfile.ZipFile(zip_path) as zf:
        new_meta: dict[str, dict] = {}
        for row in _read_tsv(zf, "SUBMISSION.tsv"):
            parsed = parse_submission_row(row)
            if parsed is None:
                continue
            acc = parsed["accession_number"]
            if acc in targets and acc not in existing:
                parsed["company_id"] = targets[acc]
                new_meta[acc] = parsed

        if not new_meta:
            return stats

        deriv_rows = []
        for row in _read_tsv(zf, "DERIV_TRANS.tsv"):
            parsed = parse_deriv_row(row)
            if parsed and parsed["accession_number"] in new_meta:
                deriv_rows.append(parsed)

        footnote_rows = []
        for row in _read_tsv(zf, "FOOTNOTES.tsv"):
            parsed = parse_footnote_row(row)
            if parsed and parsed["accession_number"] in new_meta:
                footnote_rows.append(parsed)

    db.execute(Form4FilingMeta.__table__.insert(), list(new_meta.values()))
    if deriv_rows:
        db.execute(Form4DerivTxn.__table__.insert(), deriv_rows)
    if footnote_rows:
        db.execute(Form4Footnote.__table__.insert(), footnote_rows)
    db.commit()
    existing.update(new_meta.keys())

    stats["meta"] = len(new_meta)
    stats["deriv"] = len(deriv_rows)
    stats["footnotes"] = len(footnote_rows)
    return stats


def backfill_form4_details(db: Session, redownload: bool = False) -> dict:
    """Orchestrate the bulk re-parse: 2015q1 → current quarter.

    Uses the cached ZIPs in data/bulk/ (downloads only the few missing ones,
    e.g. 2015 quarters and the latest). Filing-level idempotent — safe to
    re-run after a partial failure.
    """
    targets = load_target_accessions(db)
    existing = _existing_meta_accessions(db)
    logger.info("form4-details backfill: %d target filings, %d already ingested",
                len(targets), len(existing))

    totals = {"meta": 0, "deriv": 0, "footnotes": 0, "quarters": 0, "missing": 0}
    for url, filename in _generate_quarter_urls_from(EARLIEST_YEAR, 1):
        zip_path = download_quarter_zip(url, filename, redownload=redownload)
        if zip_path is None:
            totals["missing"] += 1
            continue
        stats = ingest_quarter_details(zip_path, targets, db, existing)
        totals["meta"] += stats["meta"]
        totals["deriv"] += stats["deriv"]
        totals["footnotes"] += stats["footnotes"]
        totals["quarters"] += 1
        logger.info("  %s: +%d filings, +%d deriv rows, +%d footnotes",
                    filename, stats["meta"], stats["deriv"], stats["footnotes"])

    logger.info("form4-details backfill complete: %s", totals)
    return totals
