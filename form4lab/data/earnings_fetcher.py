"""Fetch historical earnings report dates from yfinance into earnings_dates.

yfinance's get_earnings_dates returns past + scheduled report datetimes (history
goes back ~15-20yr for large caps, varies by ticker). We store the calendar date
and BMO/AMC hint. Idempotent: existing (company_id, earnings_date) rows are
skipped, so re-runs are safe.
"""
import contextlib
import io
import logging

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.orm import Session

from form4lab.models.company import Company
from form4lab.models.earnings import EarningsDate

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def _classify_report_time(ts: pd.Timestamp) -> str | None:
    """Crude BMO/AMC from the timestamp hour (ET-ish). None if unknown."""
    try:
        hour = ts.hour
    except Exception:
        return None
    if hour == 0:
        return None  # midnight = date-only, time unknown
    return "BMO" if hour < 12 else "AMC"


def fetch_earnings_for_ticker(ticker: str, limit: int = 80) -> list[tuple]:
    """Return [(date, report_time), ...] for a ticker, or [] on failure."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            t = yf.Ticker(ticker)
            df = t.get_earnings_dates(limit=limit)
    except Exception as e:
        logger.warning("earnings fetch failed for %s: %s", ticker, e)
        return []
    if df is None or len(df) == 0:
        return []
    out = []
    for idx in df.index:
        ts = pd.Timestamp(idx)
        if pd.isna(ts):
            continue
        out.append((ts.date(), _classify_report_time(ts)))
    # dedupe by date (keep first report_time)
    seen = {}
    for d, rt in out:
        seen.setdefault(d, rt)
    return sorted(seen.items())


def backfill_earnings(db: Session, tickers: list[str] | None = None,
                      limit: int = 80) -> dict:
    """Backfill earnings dates for companies (optionally restricted to tickers).

    Returns {"companies": n, "inserted": m, "skipped_existing": k, "no_data": j}.
    Idempotent — re-running only inserts missing (company_id, date) rows.
    """
    q = select(Company).where(Company.ticker.isnot(None))
    companies = db.execute(q).scalars().all()
    if tickers is not None:
        tset = {t.upper() for t in tickers}
        companies = [c for c in companies if (c.ticker or "").upper() in tset]

    inserted = skipped = no_data = 0
    for i, c in enumerate(companies, 1):
        rows = fetch_earnings_for_ticker(c.ticker, limit=limit)
        if not rows:
            no_data += 1
            continue
        existing = set(db.execute(
            select(EarningsDate.earnings_date).where(EarningsDate.company_id == c.id)
        ).scalars().all())
        new_objs = []
        for d, rt in rows:
            if d in existing:
                skipped += 1
                continue
            new_objs.append(EarningsDate(company_id=c.id, earnings_date=d, report_time=rt))
        if new_objs:
            db.add_all(new_objs)
            db.commit()
            inserted += len(new_objs)
        if i % 50 == 0:
            logger.info("earnings backfill: %d/%d companies, %d inserted", i, len(companies), inserted)

    return {"companies": len(companies), "inserted": inserted,
            "skipped_existing": skipped, "no_data": no_data}
