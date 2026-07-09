"""Backfill point-in-time fundamentals from SEC EDGAR XBRL company-facts.

For each company (by CIK) we pull data.sec.gov company-facts and store the raw
facts for a fixed set of concepts. Every fact keeps its `filed` date, so
downstream features are look-ahead-free (use only facts filed before a trade).
Idempotent: existing (company, concept, period_end, filed) rows are skipped.
Reuses the rate-limited SEC client from sec_fetcher (9 req/s, proper User-Agent).
"""
import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from form4lab.data.sec_fetcher import _sec_get
from form4lab.models.company import Company
from form4lab.models.fundamentals import Fundamental

logger = logging.getLogger(__name__)

# Concepts we need, by XBRL namespace. Stored as "ns:Concept".
CONCEPTS = {
    "us-gaap": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss", "OperatingIncomeLoss",
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "Assets", "Liabilities", "StockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue",
        "LongTermDebtNoncurrent", "LongTermDebt",
    ],
    "dei": ["EntityCommonStockSharesOutstanding"],
}


def fetch_facts_for_cik(cik: str) -> list[dict]:
    """Return [{concept, period_end, filed_date, fiscal_period, form, value}, ...]."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    try:
        resp = _sec_get(url)
        if resp.status_code != 200:
            return []
        facts = resp.json().get("facts", {})
    except Exception as e:
        logger.warning("companyfacts failed for CIK %s: %s", cik, e)
        return []

    out = []
    for ns, names in CONCEPTS.items():
        ns_facts = facts.get(ns, {})
        for name in names:
            entry = ns_facts.get(name)
            if not entry:
                continue
            for unit_key, rows in entry.get("units", {}).items():
                for r in rows:
                    end = r.get("end")
                    filed = r.get("filed")
                    val = r.get("val")
                    if end is None or filed is None or val is None:
                        continue
                    out.append({
                        "concept": f"{ns}:{name}",
                        "period_end": end,
                        "filed_date": filed,
                        "fiscal_period": r.get("fp"),
                        "form": r.get("form"),
                        "value": float(val),
                    })
    return out


def backfill_fundamentals(db: Session, tickers: list[str] | None = None) -> dict:
    """Backfill fundamentals for companies (optionally restricted to tickers)."""
    companies = db.execute(
        select(Company).where(Company.ticker.isnot(None), Company.cik.isnot(None))
    ).scalars().all()
    if tickers is not None:
        tset = {t.upper() for t in tickers}
        companies = [c for c in companies if (c.ticker or "").upper() in tset]

    inserted = skipped = no_data = 0
    for i, c in enumerate(companies, 1):
        facts = fetch_facts_for_cik(c.cik)
        if not facts:
            no_data += 1
            continue
        existing = set(db.execute(
            select(Fundamental.concept, Fundamental.period_end, Fundamental.filed_date)
            .where(Fundamental.company_id == c.id)
        ).all())
        new_objs = []
        for f in facts:
            pe = datetime.strptime(f["period_end"], "%Y-%m-%d").date()
            fd = datetime.strptime(f["filed_date"], "%Y-%m-%d").date()
            if (f["concept"], pe, fd) in existing:
                skipped += 1
                continue
            existing.add((f["concept"], pe, fd))  # guard intra-batch dupes
            new_objs.append(Fundamental(
                company_id=c.id, concept=f["concept"], period_end=pe, filed_date=fd,
                fiscal_period=f["fiscal_period"], form=f["form"], value=f["value"]))
        if new_objs:
            db.add_all(new_objs)
            db.commit()
            inserted += len(new_objs)
        if i % 50 == 0:
            logger.info("fundamentals backfill: %d/%d companies, %d rows inserted",
                        i, len(companies), inserted)

    return {"companies": len(companies), "inserted": inserted,
            "skipped_existing": skipped, "no_data": no_data}
