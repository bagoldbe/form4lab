"""SEC EDGAR direct API client with rate limiting.

Direct HTTP client for the SEC EDGAR submissions/filing APIs.
SEC allows max 10 requests/second with a 10-minute cooldown on violation.
"""

import logging
import threading
import time
from datetime import date, datetime, timedelta

import httpx
from lxml import etree

from sqlalchemy.orm import Session

from form4lab.config import settings
from form4lab.data.utils import (
    classify_common_stock,
    parse_10b5_1_flag,
    determine_is_discretionary,
    get_or_create_insider,
    get_or_create_company,
    get_or_create_role,
    is_plausible_cik,
    persist_transaction,
)
from form4lab.models.company import Company as CompanyModel
from form4lab.models.transaction import Transaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEC_BASE = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"

SEC_HEADERS = {
    "User-Agent": settings.sec_identity,
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Enforces max N requests per second across threads."""

    def __init__(self, max_per_second: float = 9.0):
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.monotonic()


_sec_cfg = settings.sec
_rate_limiter = RateLimiter(max_per_second=_sec_cfg.max_requests_per_second)

MAX_RETRIES = _sec_cfg.max_retries
RATE_LIMIT_WAIT = _sec_cfg.rate_limit_wait_seconds


# ---------------------------------------------------------------------------
# HTTP client (persistent connection pool)
# ---------------------------------------------------------------------------
_client = httpx.Client(
    headers=SEC_HEADERS,
    timeout=_sec_cfg.http_timeout_seconds,
    follow_redirects=True,
    limits=httpx.Limits(
        max_connections=_sec_cfg.max_connections,
        max_keepalive_connections=_sec_cfg.max_connections,
    ),
)


def close_client():
    """Close the persistent HTTP client. Called during app shutdown."""
    _client.close()


def _sec_get(url: str) -> httpx.Response:
    """Rate-limited GET with persistent connection and retry on 429/503/network errors."""
    for attempt in range(MAX_RETRIES + 1):
        _rate_limiter.wait()
        try:
            response = _client.get(url)
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES:
                logger.warning("Network error on %s (attempt %d/%d): %s", url, attempt + 1, MAX_RETRIES, e)
                time.sleep(5)
                continue
            raise
        if response.status_code in (429, 503) and attempt < MAX_RETRIES:
            wait = RATE_LIMIT_WAIT if response.status_code == 429 else 5
            logger.warning(
                "Got %d from %s, waiting %ds (attempt %d/%d)",
                response.status_code, url, wait, attempt + 1, MAX_RETRIES,
            )
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------
_cik_cache: dict[str, str] = {}
_cik_cache_lock = threading.Lock()


def _fetch_company_tickers() -> dict:
    """Download company_tickers.json from SEC."""
    url = f"{SEC_BASE}/files/company_tickers.json"
    response = _sec_get(url)
    return response.json()


def resolve_cik(ticker: str) -> str | None:
    """Resolve a stock ticker to a zero-padded 10-digit CIK string.

    On first call, downloads company_tickers.json and populates the cache.
    Subsequent calls use the cache. Lookup is case-insensitive.
    Thread-safe via double-checked locking.

    Returns None if the ticker is not found.
    """
    ticker_upper = ticker.upper()
    if not _cik_cache:
        with _cik_cache_lock:
            if not _cik_cache:  # double-checked locking
                data = _fetch_company_tickers()
                for entry in data.values():
                    t = entry["ticker"].upper()
                    cik = str(entry["cik_str"]).zfill(10)
                    _cik_cache[t] = cik
    return _cik_cache.get(ticker_upper)


# ---------------------------------------------------------------------------
# Submissions API
# ---------------------------------------------------------------------------
def _extract_filings(
    filing_data: dict, form_filter: set[str], cutoff_date: date | None = None
) -> list[dict]:
    """Extract filings of the given form types from parallel-array filing data.

    Args:
        filing_data: Dict with parallel arrays: form, accessionNumber,
                     filingDate, primaryDocument (and optionally reportDate,
                     items — populated for 8-Ks).
        form_filter: Form types to keep (e.g. {"4"} or {"8-K", "SC 13D"}).
        cutoff_date: If provided, skip filings with filingDate < cutoff_date.

    Returns:
        List of dicts with keys: accession_number, filing_date,
        primary_document, form, report_date, items.
    """
    results: list[dict] = []
    forms = filing_data.get("form", [])
    accession_numbers = filing_data.get("accessionNumber", [])
    filing_dates = filing_data.get("filingDate", [])
    primary_documents = filing_data.get("primaryDocument", [])
    report_dates = filing_data.get("reportDate", [])
    items_list = filing_data.get("items", [])

    for i, form in enumerate(forms):
        if form not in form_filter:
            continue

        filing_date_str = filing_dates[i]

        if cutoff_date is not None:
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            if filing_date < cutoff_date:
                continue

        results.append(
            {
                "accession_number": accession_numbers[i],
                "filing_date": filing_date_str,
                "primary_document": primary_documents[i] if i < len(primary_documents) else "",
                "form": form,
                "report_date": report_dates[i] if i < len(report_dates) else "",
                "items": items_list[i] if i < len(items_list) else "",
            }
        )

    return results


def _extract_form4_filings(
    filing_data: dict, cutoff_date: date | None = None
) -> list[dict]:
    """Extract Form 4 filings (thin wrapper over _extract_filings, preserving
    the original 4-key contract — tests assert the exact key set)."""
    keep = ("accession_number", "filing_date", "primary_document", "form")
    return [{k: f[k] for k in keep}
            for f in _extract_filings(filing_data, {"4"}, cutoff_date=cutoff_date)]


def fetch_submissions_filings(
    cik: str, form_filter: set[str], cutoff_date: date | None = None
) -> list[dict]:
    """Fetch filing metadata of arbitrary form types for a company from the
    SEC Submissions API (one primary request + pagination files). Same shape
    as fetch_submissions but generalized — used by the filing-events harvester
    (8-K / SC 13D / SC 13G)."""
    url = f"{SEC_DATA}/submissions/CIK{cik}.json"
    response = _sec_get(url)
    data = response.json()

    filings = _extract_filings(
        data["filings"]["recent"], form_filter, cutoff_date=cutoff_date
    )
    for file_entry in data["filings"].get("files", []):
        paginated_url = f"{SEC_DATA}/submissions/{file_entry['name']}"
        paginated_response = _sec_get(paginated_url)
        paginated_data = paginated_response.json()
        filings.extend(
            _extract_filings(paginated_data, form_filter, cutoff_date=cutoff_date)
        )
    return filings


def fetch_submissions(
    cik: str, cutoff_date: date | None = None
) -> list[dict]:
    """Fetch all Form 4 filing metadata for a company from the SEC Submissions API.

    Makes one primary request to get the submissions JSON, then follows any
    pagination files referenced in data["filings"]["files"].

    Args:
        cik: Zero-padded 10-digit CIK string.
        cutoff_date: If provided, only include filings on or after this date.

    Returns:
        Combined list of all Form 4 filing metadata dicts.
    """
    url = f"{SEC_DATA}/submissions/CIK{cik}.json"
    response = _sec_get(url)
    data = response.json()

    # Extract Form 4s from the "recent" filings
    filings = _extract_form4_filings(
        data["filings"]["recent"], cutoff_date=cutoff_date
    )

    # Handle pagination: each entry in "files" has a "name" key
    for file_entry in data["filings"].get("files", []):
        paginated_url = f"{SEC_DATA}/submissions/{file_entry['name']}"
        paginated_response = _sec_get(paginated_url)
        paginated_data = paginated_response.json()
        filings.extend(
            _extract_form4_filings(paginated_data, cutoff_date=cutoff_date)
        )

    return filings


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
def _xml_text(element, path: str) -> str | None:
    """Extract text from an XML element at the given path."""
    node = element.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return None


def _xml_float(element, path: str) -> float | None:
    """Extract a float from an XML element at the given path."""
    text = _xml_text(element, path)
    if text:
        try:
            return float(text)
        except (ValueError, TypeError):
            return None
    return None


def _xml_bool(element, path: str) -> bool:
    """Extract a boolean from an XML element at the given path."""
    text = _xml_text(element, path)
    if text is None:
        return False
    return text in ("1", "true", "True")


# ---------------------------------------------------------------------------
# Form 4 XML parsing
# ---------------------------------------------------------------------------
def parse_form4_xml(
    xml_content: str, accession_number: str, filing_date_str: str
) -> list[dict]:
    """Parse raw Form 4 XML into transaction dicts.

    Output dict format matches the standard transaction dict layout,
    so all downstream code (DB persistence, scoring) works unchanged.

    Args:
        xml_content: Raw XML string of the Form 4 document.
        accession_number: The SEC accession number for the filing.
        filing_date_str: Filing date as a string in YYYY-MM-DD format.

    Returns:
        List of transaction dicts, one per transaction per owner.
    """
    root = etree.fromstring(xml_content.encode("utf-8"))

    # --- 10b5-1 plan indicator (filing-level, post-April 2023) ---
    aff_10b5 = _xml_text(root, "aff10b5One")
    is_10b5_1_plan = parse_10b5_1_flag(aff_10b5)


    # --- Issuer ---
    issuer_cik = (_xml_text(root, ".//issuer/issuerCik") or "").lstrip("0")
    issuer_name = _xml_text(root, ".//issuer/issuerName")
    issuer_ticker = _xml_text(root, ".//issuer/issuerTradingSymbol")

    if not issuer_cik:
        return []

    # --- Filing date ---
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        filing_date = date.today()

    # --- Owners ---
    owners: list[dict] = []
    for owner_el in root.findall(".//reportingOwner"):
        cik_raw = _xml_text(owner_el, "reportingOwnerId/rptOwnerCik") or ""
        owners.append(
            {
                "cik": cik_raw.lstrip("0"),
                "name": _xml_text(owner_el, "reportingOwnerId/rptOwnerName") or "",
                "is_officer": _xml_bool(
                    owner_el, "reportingOwnerRelationship/isOfficer"
                ),
                "is_director": _xml_bool(
                    owner_el, "reportingOwnerRelationship/isDirector"
                ),
                "is_ten_pct_owner": _xml_bool(
                    owner_el, "reportingOwnerRelationship/isTenPercentOwner"
                ),
                "officer_title": _xml_text(
                    owner_el, "reportingOwnerRelationship/officerTitle"
                )
                or "",
            }
        )

    if not owners:
        return []

    # --- Parse transactions ---
    results: list[dict] = []
    for txn_el in root.findall(
        ".//nonDerivativeTable/nonDerivativeTransaction"
    ):
        code = _xml_text(txn_el, "transactionCoding/transactionCode") or ""
        if not code:
            continue

        security_title = _xml_text(txn_el, "securityTitle/value")

        txn_date_str = _xml_text(txn_el, "transactionDate/value") or ""
        shares = _xml_float(txn_el, "transactionAmounts/transactionShares/value")
        price = _xml_float(
            txn_el, "transactionAmounts/transactionPricePerShare/value"
        )
        acq_disp = (
            _xml_text(
                txn_el,
                "transactionAmounts/transactionAcquiredDisposedCode/value",
            )
            or ""
        )
        shares_after = _xml_float(
            txn_el,
            "postTransactionAmounts/sharesOwnedFollowingTransaction/value",
        )

        # Numeric defaults
        shares_float = shares if shares is not None else 0.0
        price_float = price if price is not None else None

        is_discretionary = determine_is_discretionary(code)

        # Total value
        total_value = (
            (shares_float * price_float)
            if (shares_float and price_float)
            else None
        )

        # Transaction date
        try:
            txn_date = (
                datetime.strptime(txn_date_str, "%Y-%m-%d").date()
                if txn_date_str
                else None
            )
        except ValueError:
            txn_date = None

        for owner in owners:
            results.append(
                {
                    "insider_cik": owner["cik"],
                    "insider_name": owner["name"],
                    "is_officer": owner["is_officer"],
                    "is_director": owner["is_director"],
                    "is_ten_pct_owner": owner["is_ten_pct_owner"],
                    "officer_title": owner["officer_title"],
                    "company_cik": issuer_cik,
                    "company_name": issuer_name,
                    "company_ticker": issuer_ticker,
                    "accession_number": f"{accession_number}_{code}_{txn_date_str}_{shares_float}",
                    "filing_date": filing_date,
                    "transaction_date": txn_date,
                    "transaction_code": code,
                    "shares": shares_float,
                    "price_per_share": price_float,
                    "total_value": total_value,
                    "shares_owned_after": shares_after,
                    "acquired_or_disposed": acq_disp,
                    "is_discretionary": is_discretionary,
                    "security_title": security_title,
                    "is_common_stock": classify_common_stock(security_title),
                    "is_10b5_1_plan": is_10b5_1_plan,
                }
            )

    return results


def fetch_and_parse_form4(
    cik: str,
    accession_number: str,
    primary_document: str,
    filing_date_str: str,
) -> list[dict]:
    """Fetch a Form 4 XML document from SEC EDGAR and parse it.

    Args:
        cik: Company CIK (used to build the URL path).
        accession_number: The SEC accession number (e.g. "0000320193-24-000123").
        primary_document: Filename of the primary XML document.
        filing_date_str: Filing date as YYYY-MM-DD string.

    Returns:
        List of transaction dicts, or [] on failure.
    """
    acc_no_dashes = accession_number.replace("-", "")
    # Strip XSL directory prefix (e.g. "xslF345X05/") to get raw XML instead
    # of the XSLT-rendered HTML version that SEC returns for prefixed paths.
    doc_name = primary_document.split("/")[-1] if "/" in primary_document else primary_document
    url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{acc_no_dashes}/{doc_name}"

    try:
        response = _sec_get(url)
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP %d fetching %s: %s", e.response.status_code, url, e)
        return []
    except httpx.HTTPError as e:
        logger.warning("Network error fetching %s: %s", url, e)
        return []

    try:
        return parse_form4_xml(response.text, accession_number, filing_date_str)
    except etree.XMLSyntaxError as e:
        logger.warning("XML parse error for %s: %s", accession_number, e)
        return []
    except Exception as e:
        logger.warning("Unexpected parse error for %s: %s", accession_number, e)
        return []


# ---------------------------------------------------------------------------
# Backfill pipeline
# ---------------------------------------------------------------------------
def backfill_company_fast(ticker: str, years: int, db: Session) -> int:
    """Backfill Form 4 data for a company using direct SEC APIs. Returns count of transactions inserted."""
    cik = resolve_cik(ticker)
    if not cik:
        logger.error(f"Could not resolve CIK for {ticker}")
        return 0

    cutoff = date.today() - timedelta(days=years * 365)
    filings = fetch_submissions(cik, cutoff_date=cutoff)
    logger.info(f"{ticker}: {len(filings)} Form 4 filings found since {cutoff}")

    count = 0
    for filing_meta in filings:
        try:
            txns = fetch_and_parse_form4(
                cik,
                filing_meta["accession_number"],
                filing_meta["primary_document"],
                filing_meta["filing_date"],
            )
            for txn_data in txns:
                if persist_transaction(txn_data, db):
                    count += 1

            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(f"{ticker}: Error on filing {filing_meta['accession_number']}: {e}")

    logger.info(f"{ticker}: {count} transactions inserted")
    return count


# ---------------------------------------------------------------------------
# Daily ingest
# ---------------------------------------------------------------------------
def ingest_daily_filings(db: Session, days_back: int = 1) -> int:
    """Check all tracked companies for new Form 4 filings.

    Fetches submissions for each tracked company, finds filings newer than
    the latest we have, and persists new transactions.

    Args:
        db: Database session
        days_back: How many days back to check (default 1, increase to catch up)
    Returns:
        Count of new transactions inserted.
    """
    cutoff = date.today() - timedelta(days=days_back)

    # Get all tracked companies with CIKs
    companies = (
        db.query(CompanyModel.cik)
        .filter(CompanyModel.cik.isnot(None))
        .distinct()
        .all()
    )
    all_ciks = [row[0].zfill(10) for row in companies]
    ciks = [c for c in all_ciks if is_plausible_cik(c)]
    if len(ciks) < len(all_ciks):
        logger.warning(
            "Skipping %d implausible CIKs (likely test fixtures) out of %d tracked",
            len(all_ciks) - len(ciks), len(all_ciks),
        )
    logger.info(f"Checking {len(ciks)} companies for new filings since {cutoff}")

    total_count = 0
    for cik in ciks:
        try:
            filings = fetch_submissions(cik, cutoff_date=cutoff)
        except Exception as e:
            logger.warning(f"Failed to fetch submissions for CIK {cik}: {e}")
            continue

        for filing_meta in filings:
            try:
                txns = fetch_and_parse_form4(
                    cik,
                    filing_meta["accession_number"],
                    filing_meta["primary_document"],
                    filing_meta["filing_date"],
                )
            except Exception as e:
                logger.warning(f"Failed to fetch Form 4 {filing_meta['accession_number']}: {e}")
                continue

            try:
                for txn_data in txns:
                    if persist_transaction(txn_data, db):
                        total_count += 1
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning("Failed to persist filing %s: %s", filing_meta["accession_number"], e)

    logger.info(f"Daily ingest complete: {total_count} new transactions")
    return total_count


# ---------------------------------------------------------------------------
# Continuous polling via SEC EDGAR EFTS (full-text search)
# ---------------------------------------------------------------------------

EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
EFTS_PAGE_SIZE = 100  # SEC EFTS default and maximum per-request hit count
EFTS_MAX_OFFSET = 10000  # EFTS deep-paging limit; beyond this SEC returns 400


def _build_efts_url(since: datetime, from_offset: int = 0) -> str:
    """Build EFTS search URL for Form 4 filings since a given timestamp.

    `from_offset` advances through result pages; EFTS caps each response at
    EFTS_PAGE_SIZE, so busy days (>100 Form 4 filings) require pagination.
    """
    start_str = since.strftime("%Y-%m-%d")
    end_str = date.today().strftime("%Y-%m-%d")
    url = (
        f"{EFTS_BASE}?forms=4"
        f"&dateRange=custom&startdt={start_str}&enddt={end_str}"
    )
    if from_offset:
        url += f"&from={from_offset}"
    return url


def poll_recent_filings(db: Session, since: datetime) -> int:
    """Poll SEC EFTS for Form 4 filings filed since `since`.

    Returns count of new transactions inserted.
    Designed to be called every 20 seconds during market hours.

    Iterates all EFTS result pages so filings past rank 100 are not silently
    lost on busy days (a busy filing day can exceed 800 Form 4s).
    """
    hits: list[dict] = []
    offset = 0
    while offset < EFTS_MAX_OFFSET:
        url = _build_efts_url(since, from_offset=offset)
        try:
            response = _sec_get(url)
            data = response.json()
        except Exception as e:
            # ERROR (not warning): a mid-pagination failure drops filings past
            # this offset. `_last_poll_time` in the caller won't advance on raise,
            # but we break here to process what we have — rely on the daily
            # ingest (days_back=7) as the backstop for any filings lost.
            logger.error(
                "EFTS poll failed at offset %d (%d hits collected): %s",
                offset, len(hits), e,
            )
            break

        page_hits = data.get("hits", {}).get("hits", [])
        if not page_hits:
            break
        hits.extend(page_hits)
        if len(page_hits) < EFTS_PAGE_SIZE:
            break
        offset += EFTS_PAGE_SIZE
    else:
        # `while/else` fires only when the condition becomes false (offset cap
        # reached) — every inner `break` skips it. See EFTS_MAX_OFFSET above.
        logger.error(
            "EFTS poll hit max offset %d; filings past this may be lost "
            "(daily ingest at days_back=7 is the backstop)",
            EFTS_MAX_OFFSET,
        )

    if not hits:
        return 0

    # Collect unique CIKs from hits — EFTS returns a "ciks" array per hit
    # containing both insider and company CIKs
    efts_ciks = set()
    for hit in hits:
        source = hit.get("_source", {})
        for cik in source.get("ciks", []):
            if cik:
                efts_ciks.add(cik.lstrip("0"))

    # Filter to only tracked company CIKs to avoid wasting SEC API calls
    # on insider personal CIKs or companies we don't track
    tracked_ciks = {
        row[0]
        for row in db.query(CompanyModel.cik)
        .filter(CompanyModel.cik.in_(efts_ciks))
        .all()
    }

    implausible = {c for c in tracked_ciks if not is_plausible_cik(c)}
    if implausible:
        logger.warning(
            "Skipping %d implausible tracked CIKs (likely test fixtures)",
            len(implausible),
        )
        tracked_ciks -= implausible

    if not tracked_ciks:
        return 0

    logger.info(
        "EFTS poll: %d filings, %d CIKs, %d tracked",
        len(hits), len(efts_ciks), len(tracked_ciks),
    )

    new_txn_count = 0
    for cik in tracked_ciks:
        padded_cik = cik.zfill(10)
        try:
            filings = fetch_submissions(padded_cik, cutoff_date=since.date())
        except Exception as e:
            logger.warning("Failed to fetch submissions for CIK %s: %s", padded_cik, e)
            continue

        for filing_meta in filings:
            try:
                txns = fetch_and_parse_form4(
                    padded_cik,
                    filing_meta["accession_number"],
                    filing_meta["primary_document"],
                    filing_meta["filing_date"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse Form 4 %s: %s",
                    filing_meta.get("accession_number"), e,
                )
                continue

            try:
                for txn_data in txns:
                    if persist_transaction(txn_data, db):
                        new_txn_count += 1
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning("Failed to persist filing %s: %s", filing_meta.get("accession_number"), e)

    logger.info("Continuous poll complete: %d new transactions", new_txn_count)
    return new_txn_count
