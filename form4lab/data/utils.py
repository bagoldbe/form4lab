"""Shared data utilities for SEC filing ingestion.

Contains entity lookup/creation helpers and classification functions
used by sec_fetcher.py and bulk_fetcher.py.
"""
import logging
import re

from sqlalchemy.orm import Session

from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company as CompanyModel
from form4lab.models.transaction import Transaction

logger = logging.getLogger(__name__)

_EXCLUDE_KEYWORDS = ["preferred", "depositary", "warrant", "unit", "debt", "note", "right", "option"]

# Real SEC CIKs top out around 2.2M today; 10M leaves decades of headroom
# while excluding uuid-generated test fixtures (always > 10M).
MAX_PLAUSIBLE_CIK = 10_000_000


def is_plausible_cik(cik: str | None) -> bool:
    """True iff `cik` looks like a real SEC CIK: ASCII digits only, <= 10,000,000.

    Guards SEC fetch loops against rows that can only 404 (e.g. test fixtures
    that leaked into the DB) so they don't burn the 10 req/s budget.
    """
    if not cik:
        return False
    s = str(cik).strip()
    if not re.fullmatch(r"[0-9]+", s):
        return False
    return int(s) <= MAX_PLAUSIBLE_CIK


def classify_common_stock(security_title: str | None) -> bool | None:
    """Classify whether a security title refers to common stock.

    Returns True for common stock, False for preferred/derivative/other,
    None if security_title is None (unknown).
    """
    if security_title is None:
        return None
    title_lower = security_title.lower()
    for keyword in _EXCLUDE_KEYWORDS:
        if keyword in title_lower:
            return False
    if "stock" in title_lower or "share" in title_lower:
        return True
    return False



def parse_10b5_1_flag(value: str | None) -> bool | None:
    """Parse SEC 10b5-1 plan indicator into a boolean.

    SEC filings use inconsistent encodings: "1", "true", "0", "false", or empty.
    Returns True/False/None.
    """
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in ("1", "true"):
        return True
    if cleaned in ("0", "false"):
        return False
    return None

def determine_is_discretionary(code: str) -> bool:
    """Only open-market purchases (code P) are considered discretionary."""
    return code == "P"


def get_or_create_insider(db: Session, cik: str, name: str) -> Insider:
    """Get an existing insider by CIK or create a new one."""
    from form4lab.models.insider import detect_is_institution

    insider = db.query(Insider).filter(Insider.cik == cik).first()
    if not insider:
        insider = Insider(cik=cik, name=name, is_institution=detect_is_institution(name))
        db.add(insider)
        db.flush()
    return insider


def get_or_create_company(
    db: Session, cik: str, ticker: str | None, name: str
) -> CompanyModel:
    """Get an existing company by CIK or create a new one."""
    company = db.query(CompanyModel).filter(CompanyModel.cik == cik).first()
    if not company:
        company = CompanyModel(cik=cik, ticker=ticker, name=name)
        db.add(company)
        db.flush()
    return company


def get_or_create_role(
    db: Session,
    insider_id: int,
    company_id: int,
    txn_data: dict,
    filing_date,
) -> InsiderRole:
    """Get an existing insider role or create a new one. Updates last_filing_date if newer."""
    role = (
        db.query(InsiderRole)
        .filter(
            InsiderRole.insider_id == insider_id,
            InsiderRole.company_id == company_id,
        )
        .first()
    )
    if not role:
        title = txn_data.get("officer_title", "")
        if not title:
            if txn_data.get("is_director"):
                title = "Director"
            elif txn_data.get("is_ten_pct_owner"):
                title = "10% Owner"
            else:
                title = "Other"
        role = InsiderRole(
            insider_id=insider_id,
            company_id=company_id,
            role_title=title,
            is_officer=txn_data.get("is_officer", False),
            is_director=txn_data.get("is_director", False),
            is_ten_percent_owner=txn_data.get("is_ten_pct_owner", False),
            first_filing_date=filing_date,
            last_filing_date=filing_date,
        )
        db.add(role)
        db.flush()
    else:
        if filing_date and (
            role.last_filing_date is None or filing_date > role.last_filing_date
        ):
            role.last_filing_date = filing_date
    return role


def company_already_backfilled(ticker: str, db: Session) -> bool:
    """Check if a company already has transaction data in the DB."""
    company = db.query(CompanyModel).filter(CompanyModel.ticker == ticker).first()
    if not company:
        return False
    return db.query(Transaction.id).filter(Transaction.company_id == company.id).first() is not None


def persist_transaction(txn_data: dict, db: Session) -> Transaction | None:
    """Persist a single parsed transaction dict to the database.

    Handles dedup check, entity creation (insider/company/role),
    and Transaction record creation. Returns the Transaction if
    created, None if it already existed.
    """
    exists = (
        db.query(Transaction.id)
        .filter(Transaction.accession_number == txn_data["accession_number"])
        .first()
    )
    if exists:
        return None

    insider = get_or_create_insider(db, txn_data["insider_cik"], txn_data["insider_name"])
    company = get_or_create_company(db, txn_data["company_cik"], txn_data.get("company_ticker"), txn_data["company_name"])
    get_or_create_role(db, insider.id, company.id, txn_data, txn_data.get("filing_date"))

    txn = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number=txn_data["accession_number"],
        filing_date=txn_data["filing_date"],
        transaction_date=txn_data["transaction_date"] or txn_data["filing_date"],
        transaction_code=txn_data["transaction_code"],
        shares=txn_data["shares"],
        price_per_share=txn_data.get("price_per_share"),
        total_value=txn_data.get("total_value"),
        shares_owned_after=txn_data.get("shares_owned_after"),
        acquired_or_disposed=txn_data["acquired_or_disposed"],
        is_discretionary=txn_data["is_discretionary"],
        security_title=txn_data.get("security_title"),
        is_common_stock=txn_data.get("is_common_stock"),
        is_10b5_1_plan=txn_data.get("is_10b5_1_plan"),
    )
    db.add(txn)
    return txn
