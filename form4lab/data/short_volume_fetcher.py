"""FINRA Reg SHO daily short-sale volume backfill.

Free daily files (published the same evening) at
  https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
pipe-delimited: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market.
Coverage is FINRA-facility (off-exchange) volume only — a biased-but-consistent
short-pressure proxy; short VOLUME is not short INTEREST (no free history for
the latter). Filtered to our universe tickers; idempotent at day granularity.

Uses its OWN polite rate limiter (FINRA's CDN is not the SEC — never consume
the SEC limiter for this).
"""
import logging
import time
from datetime import date, timedelta

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from form4lab.models.short_volume import ShortVolume

logger = logging.getLogger(__name__)

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
# FINRA's CDN serves daily files only from ~2018-09 onward (older dates
# return 403 for everyone, per the vendor); the default start is set just
# before that floor.
DEFAULT_START = date(2018, 8, 1)
REQUEST_INTERVAL_S = 0.25  # ~4 req/s, polite


def parse_short_volume_text(content: str, tickers: set[str]) -> list[dict]:
    """Parse one CNMS daily file into row dicts, filtered to `tickers`.

    A symbol can appear on more than one line in a file (facility/format
    quirks — observed: CPK twice on 2018-08-01); volumes for the same
    (ticker, date) are SUMMED so the unique constraint holds.
    """
    agg: dict[tuple, dict] = {}
    for line in content.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5 or parts[0] == "Date":
            continue
        sym = parts[1].strip().upper()
        if sym not in tickers:
            continue
        try:
            d = date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:8]))
            sv = int(float(parts[2]))
            sev = int(float(parts[3]))
            tv = int(float(parts[4]))
        except (ValueError, IndexError):
            continue
        key = (sym, d)
        if key in agg:
            agg[key]["short_volume"] += sv
            agg[key]["short_exempt_volume"] += sev
            agg[key]["total_volume"] += tv
        else:
            agg[key] = {
                "ticker": sym,
                "date": d,
                "short_volume": sv,
                "short_exempt_volume": sev,
                "total_volume": tv,
            }
    return list(agg.values())


def backfill_short_volume(db: Session, start: date = DEFAULT_START,
                          end: date | None = None) -> dict:
    """Download + ingest daily files from `start` to `end` (default today).

    Skips weekends and dates already present (idempotent at day granularity —
    a day with ANY stored rows is considered done). 404s (holidays) skipped.
    """
    end = end or date.today()
    tickers = {r[0].upper() for r in db.execute(text(
        "SELECT ticker FROM companies WHERE ticker IS NOT NULL")).fetchall()}
    done_days = {r[0] for r in db.execute(text(
        "SELECT DISTINCT date FROM short_volume")).fetchall()}
    logger.info("short-volume backfill: %s -> %s, %d tickers, %d days already stored",
                start, end, len(tickers), len(done_days))

    stats = {"days": 0, "rows": 0, "missing": 0, "skipped": 0, "errors": 0}
    client = httpx.Client(timeout=30.0)
    d = start
    try:
        while d <= end:
            if d.weekday() >= 5 or d in done_days:
                stats["skipped"] += 1
                d += timedelta(days=1)
                continue
            url = FINRA_URL.format(ymd=d.strftime("%Y%m%d"))
            try:
                resp = client.get(url)
                if resp.status_code in (403, 404):
                    # 404 = market holiday; 403 = CloudFront's "no such object"
                    # (the CDN's history floor) — both mean "not available".
                    stats["missing"] += 1
                    d += timedelta(days=1)
                    continue
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("fetch failed for %s: %s", d, e)
                stats["errors"] += 1
                d += timedelta(days=1)
                continue

            rows = parse_short_volume_text(resp.text, tickers)
            if rows:
                db.execute(ShortVolume.__table__.insert(), rows)
                db.commit()
            stats["days"] += 1
            stats["rows"] += len(rows)
            if stats["days"] % 100 == 0:
                logger.info("  %s: %d days, %d rows", d, stats["days"], stats["rows"])
            time.sleep(REQUEST_INTERVAL_S)
            d += timedelta(days=1)
    finally:
        client.close()

    logger.info("short-volume backfill complete: %s", stats)
    return stats
