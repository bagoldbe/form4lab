"""Bulk backfill using SEC quarterly data sets.

The SEC publishes quarterly ZIP files containing ALL Form 4 data at:
  https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{Q}_form345.zip

Each ZIP contains TSVs: SUBMISSION.tsv, REPORTINGOWNER.tsv, NONDERIV_TRANS.tsv, DERIV_TRANS.tsv.
Downloading ~41 ZIPs replaces 13,000+ individual XML fetches for a full 10-year backfill.
"""

import csv
import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from sqlalchemy import text

from form4lab.data.sec_fetcher import _sec_get
from form4lab.data.utils import (
    classify_common_stock,
    parse_10b5_1_flag,
    determine_is_discretionary,
    persist_transaction,
)

logger = logging.getLogger(__name__)

BULK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bulk"
BULK_URL_BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"

# Months for DD-MON-YYYY parsing
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def generate_quarter_urls(years: int) -> list[tuple[str, str]]:
    """Generate (url, filename) pairs for quarterly data sets.

    Covers from Q1 of (current_year - years) through the current quarter.
    """
    today = date.today()
    start_year = today.year - years
    current_quarter = (today.month - 1) // 3 + 1

    results = []
    for year in range(start_year, today.year + 1):
        max_q = current_quarter if year == today.year else 4
        for q in range(1, max_q + 1):
            filename = f"{year}q{q}_form345.zip"
            url = f"{BULK_URL_BASE}/{filename}"
            results.append((url, filename))
    return results


def download_quarter_zip(
    url: str, filename: str, redownload: bool = False
) -> Path | None:
    """Download a quarterly ZIP to data/bulk/, skip if cached.

    Returns the local path, or None if the file doesn't exist (404).
    """
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    local_path = BULK_DIR / filename

    if local_path.exists() and not redownload:
        logger.debug("Cached: %s", filename)
        return local_path

    try:
        import httpx
        response = _sec_get(url)
        local_path.write_bytes(response.content)
        size_mb = len(response.content) / (1024 * 1024)
        logger.info("Downloaded %s (%.1f MB)", filename, size_mb)
        return local_path
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info("Not yet published: %s", filename)
            return None
        raise


def _parse_date_ddmonyyyy(date_str: str | None) -> date | None:
    """Parse SEC bulk date format: '31-MAR-2025' -> date(2025, 3, 31)."""
    if not date_str or not date_str.strip():
        return None
    date_str = date_str.strip()
    try:
        parts = date_str.split("-")
        if len(parts) != 3:
            return None
        day = int(parts[0])
        month = _MONTH_MAP.get(parts[1].upper())
        year = int(parts[2])
        if month is None:
            return None
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def _parse_relationship(rel_str: str | None) -> dict:
    """Parse RPTOWNER_RELATIONSHIP like 'Director,Officer' into booleans."""
    result = {"is_officer": False, "is_director": False, "is_ten_pct_owner": False}
    if not rel_str:
        return result
    parts = [p.strip().lower() for p in rel_str.split(",")]
    for part in parts:
        if part == "officer":
            result["is_officer"] = True
        elif part == "director":
            result["is_director"] = True
        elif part == "tenpercentowner":
            result["is_ten_pct_owner"] = True
    return result


def _safe_float(val: str | None) -> float | None:
    """Convert string to float, returning None on failure."""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _read_tsv(zip_file: zipfile.ZipFile, name: str) -> list[dict]:
    """Read a TSV file from a ZIP, returning list of row dicts."""
    try:
        with zip_file.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text, delimiter="\t")
            return list(reader)
    except KeyError:
        logger.warning("Missing %s in ZIP", name)
        return []


def parse_quarter_zip(
    zip_path: Path, target_tickers: set[str]
) -> list[dict]:
    """Parse a quarterly ZIP, filtering to target tickers.

    Returns list of dicts compatible with persist_transaction().
    """
    target_upper = {t.upper() for t in target_tickers}
    results = []

    with zipfile.ZipFile(zip_path) as zf:
        # --- Step 1: SUBMISSION.tsv — filter to Form 4 + target tickers ---
        submissions = _read_tsv(zf, "SUBMISSION.tsv")
        # accession -> submission row
        accession_submissions: dict[str, dict] = {}
        for row in submissions:
            doc_type = (row.get("DOCUMENT_TYPE") or row.get("FORM_TYPE") or "").strip()
            ticker = (row.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
            if doc_type == "4" and ticker in target_upper:
                acc = (row.get("ACCESSION_NUMBER") or "").strip()
                if acc:
                    accession_submissions[acc] = row

        if not accession_submissions:
            return []

        matching_accessions = set(accession_submissions.keys())
        logger.debug(
            "  %s: %d matching submissions from %d total",
            zip_path.name, len(matching_accessions), len(submissions),
        )

        # --- Step 2: REPORTINGOWNER.tsv — build owner lookup ---
        owners_tsv = _read_tsv(zf, "REPORTINGOWNER.tsv")
        # accession -> list of owner dicts
        accession_owners: dict[str, list[dict]] = {}
        for row in owners_tsv:
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if acc not in matching_accessions:
                continue
            owner = {
                "cik": (row.get("RPTOWNERCIK") or "").strip().lstrip("0") or "0",
                "name": (row.get("RPTOWNERNAME") or "").strip(),
                "officer_title": (row.get("RPTOWNER_TITLE") or "").strip(),
                **_parse_relationship(row.get("RPTOWNER_RELATIONSHIP")),
            }
            accession_owners.setdefault(acc, []).append(owner)

        # --- Step 3: NONDERIV_TRANS.tsv — build transaction dicts ---
        nonderiv_trans = _read_tsv(zf, "NONDERIV_TRANS.tsv")
        nonderiv_rows_by_acc: dict[str, list[dict]] = {}
        for row in nonderiv_trans:
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if acc not in matching_accessions:
                continue
            nonderiv_rows_by_acc.setdefault(acc, []).append(row)

        # Now build transaction dicts
        for acc, nd_rows in nonderiv_rows_by_acc.items():
            sub = accession_submissions[acc]
            owners = accession_owners.get(acc, [])
            if not owners:
                continue

            filing_date = _parse_date_ddmonyyyy(sub.get("FILING_DATE"))
            company_cik = (sub.get("ISSUERCIK") or "").strip().lstrip("0") or "0"
            company_name = (sub.get("ISSUERNAME") or "").strip()
            company_ticker = (sub.get("ISSUERTRADINGSYMBOL") or "").strip().upper()

            for row in nd_rows:
                code = (row.get("TRANS_CODE") or "").strip()
                if not code:
                    continue

                trans_date_str = (row.get("TRANS_DATE") or "").strip()
                trans_date = _parse_date_ddmonyyyy(trans_date_str)
                shares = _safe_float(row.get("TRANS_SHARES"))
                price = _safe_float(row.get("TRANS_PRICEPERSHARE"))
                acq_disp = (row.get("TRANS_ACQUIRED_DISP_CD") or "").strip()
                shares_after = _safe_float(row.get("SHRS_OWND_FOLWNG_TRANS"))
                security_title = (row.get("SECURITY_TITLE") or "").strip() or None

                shares_float = shares if shares is not None else 0.0
                total_value = (
                    (shares_float * price) if (shares_float and price) else None
                )

                is_discretionary = determine_is_discretionary(code)

                # Match accession format from XML parser
                acc_key = f"{acc}_{code}_{trans_date or ''}_{shares_float}"

                for owner in owners:
                    results.append({
                        "insider_cik": owner["cik"],
                        "insider_name": owner["name"],
                        "is_officer": owner["is_officer"],
                        "is_director": owner["is_director"],
                        "is_ten_pct_owner": owner["is_ten_pct_owner"],
                        "officer_title": owner["officer_title"],
                        "company_cik": company_cik,
                        "company_name": company_name,
                        "company_ticker": company_ticker,
                        "accession_number": acc_key,
                        "filing_date": filing_date,
                        "transaction_date": trans_date,
                        "transaction_code": code,
                        "shares": shares_float,
                        "price_per_share": price,
                        "total_value": total_value,
                        "shares_owned_after": shares_after,
                        "acquired_or_disposed": acq_disp,
                        "is_discretionary": is_discretionary,
                        "security_title": security_title,
                        "is_common_stock": classify_common_stock(security_title),
                        "is_10b5_1_plan": parse_10b5_1_flag(sub.get("AFF10B5ONE")),
                    })

    return results


def backfill_from_bulk(
    ticker_file: str,
    years: int,
    db: Session,
    redownload: bool = False,
) -> dict:
    """Orchestrate bulk backfill: load tickers, download ZIPs, parse, persist.

    Returns stats dict with counts.
    """
    with open(ticker_file) as f:
        tickers = {
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        }

    logger.info("Bulk backfill: %d tickers, %d years", len(tickers), years)

    urls = generate_quarter_urls(years)
    logger.info("Will process %d quarterly files", len(urls))

    stats = {
        "tickers": len(tickers),
        "quarters_downloaded": 0,
        "quarters_cached": 0,
        "quarters_missing": 0,
        "transactions_parsed": 0,
        "transactions_persisted": 0,
        "errors": 0,
    }

    batch_count = 0

    for url, filename in urls:
        zip_path = download_quarter_zip(url, filename, redownload=redownload)
        if zip_path is None:
            stats["quarters_missing"] += 1
            continue

        if (BULK_DIR / filename).exists() and not redownload:
            stats["quarters_cached"] += 1
        else:
            stats["quarters_downloaded"] += 1

        try:
            txn_dicts = parse_quarter_zip(zip_path, tickers)
        except Exception as e:
            logger.error("Error parsing %s: %s", filename, e)
            stats["errors"] += 1
            continue

        stats["transactions_parsed"] += len(txn_dicts)
        logger.info(
            "  %s: %d transactions for target tickers", filename, len(txn_dicts)
        )

        for txn_data in txn_dicts:
            try:
                if persist_transaction(txn_data, db):
                    stats["transactions_persisted"] += 1
                    batch_count += 1

                if batch_count >= 500:
                    db.commit()
                    batch_count = 0
            except Exception as e:
                db.rollback()
                stats["errors"] += 1
                logger.warning("Error persisting transaction: %s", e)
                batch_count = 0

        # Commit remaining batch after each quarter
        if batch_count > 0:
            db.commit()
            batch_count = 0

    logger.info(
        "Bulk backfill complete: %d parsed, %d persisted, %d errors",
        stats["transactions_parsed"],
        stats["transactions_persisted"],
        stats["errors"],
    )
    return stats


def _generate_quarter_urls_from(start_year: int, start_quarter: int) -> list[tuple[str, str]]:
    """Generate (url, filename) pairs from a specific start quarter to now."""
    today = date.today()
    current_quarter = (today.month - 1) // 3 + 1

    results = []
    for year in range(start_year, today.year + 1):
        min_q = start_quarter if year == start_year else 1
        max_q = current_quarter if year == today.year else 4
        for q in range(min_q, max_q + 1):
            filename = f"{year}q{q}_form345.zip"
            url = f"{BULK_URL_BASE}/{filename}"
            results.append((url, filename))
    return results


def backfill_10b5_1_flags(db: Session, redownload: bool = False) -> dict:
    """Retroactively populate is_10b5_1_plan on existing transactions.

    Downloads 2023 Q2+ quarterly ZIPs (when SEC started requiring 10b5-1
    disclosure), reads SUBMISSION.tsv for the AFF10B5ONE column, and updates
    existing transactions that have is_10b5_1_plan IS NULL.

    Optimized in three ways:
    1. Queries all (id, accession_number) for NULL rows upfront
    2. Builds a Python-side prefix→id mapping so TSV lookups are O(1)
    3. Batches updates by ID in chunks (2 queries per chunk instead of 1 per accession)

    Returns stats dict with counts.
    """
    # 10b5-1 data available from 2023 Q2 onward (SEC rule effective April 2023)
    urls = _generate_quarter_urls_from(2023, 2)
    logger.info("10b5-1 backfill: processing %d quarters (2023Q2 onward)", len(urls))

    stats = {
        "quarters_processed": 0,
        "quarters_missing": 0,
        "accessions_with_flag": 0,
        "accessions_matched": 0,
        "transactions_updated": 0,
        "errors": 0,
    }

    # Step 1: Load all transactions needing updates into a Python-side lookup.
    # accession_number format is "{acc}_{code}_{date}_{shares}"
    logger.info("Querying DB for transactions needing 10b5-1 flags...")
    rows = db.execute(text("""
        SELECT id, accession_number
        FROM transactions
        WHERE is_10b5_1_plan IS NULL
          AND accession_number IS NOT NULL
    """)).fetchall()

    # prefix → list of transaction IDs
    prefix_to_ids: dict[str, list[int]] = {}
    for txn_id, acc_num in rows:
        if "_" in acc_num:
            prefix = acc_num.split("_", 1)[0]
            prefix_to_ids.setdefault(prefix, []).append(txn_id)

    logger.info(
        "Found %d transactions (%d distinct accession prefixes) needing 10b5-1 flags",
        len(rows), len(prefix_to_ids),
    )

    if not prefix_to_ids:
        logger.info("No transactions need updating — all already have 10b5-1 flags")
        return stats

    # Step 2: Process each quarter's ZIP, collecting flag assignments
    # Accumulate all (id → flag) mappings across quarters, then batch update
    id_to_flag: dict[int, bool] = {}

    for url, filename in urls:
        zip_path = download_quarter_zip(url, filename, redownload=redownload)
        if zip_path is None:
            stats["quarters_missing"] += 1
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                submissions = _read_tsv(zf, "SUBMISSION.tsv")
        except Exception as e:
            logger.error("Error reading %s: %s", filename, e)
            stats["errors"] += 1
            continue

        # Match TSV accessions to our DB prefixes
        total_with_flag = 0
        quarter_matched = 0
        for row in submissions:
            acc = (row.get("ACCESSION_NUMBER") or "").strip()
            if not acc:
                continue
            flag = parse_10b5_1_flag(row.get("AFF10B5ONE"))
            if flag is not None:
                total_with_flag += 1
                if acc in prefix_to_ids:
                    quarter_matched += 1
                    for txn_id in prefix_to_ids[acc]:
                        id_to_flag[txn_id] = flag

        stats["accessions_with_flag"] += total_with_flag
        stats["accessions_matched"] += quarter_matched
        stats["quarters_processed"] += 1
        logger.info(
            "  %s: %d with flag, %d matched our DB (of %d submissions)",
            filename, total_with_flag, quarter_matched, len(submissions),
        )

    # Step 3: Batch update by flag value, chunked by 1000 IDs per query.
    # Use ORM .update() for portable IN clause (works on SQLite + PostgreSQL).
    from form4lab.models.transaction import Transaction

    CHUNK_SIZE = 1000
    true_ids = [tid for tid, flag in id_to_flag.items() if flag is True]
    false_ids = [tid for tid, flag in id_to_flag.items() if flag is False]

    logger.info("Updating %d transactions (True: %d, False: %d)",
                len(id_to_flag), len(true_ids), len(false_ids))

    total_updated = 0
    for flag_val, ids in [(True, true_ids), (False, false_ids)]:
        for i in range(0, len(ids), CHUNK_SIZE):
            chunk = ids[i:i + CHUNK_SIZE]
            try:
                count = db.query(Transaction).filter(
                    Transaction.id.in_(chunk)
                ).update(
                    {"is_10b5_1_plan": flag_val},
                    synchronize_session=False,
                )
                total_updated += count
            except Exception as e:
                logger.warning("Error updating chunk: %s", e)
                db.rollback()
                stats["errors"] += 1

    db.commit()
    stats["transactions_updated"] = total_updated

    logger.info(
        "10b5-1 backfill complete: %d quarters, %d accessions matched, %d transactions updated",
        stats["quarters_processed"],
        stats["accessions_matched"],
        stats["transactions_updated"],
    )
    return stats
