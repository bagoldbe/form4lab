"""Filing-events harvester: 8-K + SC 13D/13G per company.

One pass over the SEC submissions API per company (the same endpoint the Form 4
pipeline already uses, same rate limiter) collecting:
  - 8-K filings WITH their item codes — Item 2.02 ("Results of Operations") is
    the free, full-history earnings-release-date proxy that extends
    earnings-release-date coverage beyond yfinance's;
  - SC 13D / 13D/A / 13G / 13G/A — activist/beneficial-ownership context
    (initial 13Ds are the activist-confluence signal; filing_date is the
    public-knowledge date).

Idempotent at accession granularity. Research-only table.
"""
import logging
from datetime import date, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from form4lab.data.sec_fetcher import fetch_submissions_filings
from form4lab.models.filing_event import CompanyFilingEvent

logger = logging.getLogger(__name__)

HARVEST_FORMS = {"8-K", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}
CUTOFF = date(2014, 1, 1)  # a year before transaction history typically starts


def _parse_iso(s: str | None) -> date | None:
    if not s or not str(s).strip():
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def harvest_filing_events(db: Session, forms: set[str] | None = None,
                          cutoff: date = CUTOFF) -> dict:
    """Harvest filing events for every company with a CIK. Skips accessions
    already stored (idempotent; safe to re-run after a partial failure)."""
    forms = forms or HARVEST_FORMS
    companies = db.execute(text(
        "SELECT id, cik, ticker FROM companies WHERE cik IS NOT NULL"
    )).fetchall()
    existing = {r[0] for r in db.execute(text(
        "SELECT accession_number FROM company_filing_events")).fetchall()}
    logger.info("filing-events harvest: %d companies, %d events already stored",
                len(companies), len(existing))

    stats = {"companies": 0, "events": 0, "errors": 0}
    for company_id, cik, ticker in companies:
        try:
            filings = fetch_submissions_filings(
                str(cik).zfill(10), forms, cutoff_date=cutoff)
        except Exception as e:
            logger.warning("submissions fetch failed for %s (cik %s): %s",
                           ticker, cik, e)
            stats["errors"] += 1
            continue

        rows = []
        for f in filings:
            acc = f["accession_number"]
            if not acc or acc in existing:
                continue
            existing.add(acc)
            rows.append({
                "company_id": company_id,
                "form_type": f["form"],
                "filing_date": _parse_iso(f["filing_date"]),
                "report_date": _parse_iso(f.get("report_date")),
                "items": (f.get("items") or "").strip() or None,
                "accession_number": acc,
                "primary_document": (f.get("primary_document") or "").strip() or None,
            })
        rows = [r for r in rows if r["filing_date"] is not None]
        if rows:
            db.execute(CompanyFilingEvent.__table__.insert(), rows)
            db.commit()
        stats["companies"] += 1
        stats["events"] += len(rows)
        if stats["companies"] % 100 == 0:
            logger.info("  %d/%d companies, %d events",
                        stats["companies"], len(companies), stats["events"])

    logger.info("filing-events harvest complete: %s", stats)
    return stats
