"""Portfolio equity vs SPY benchmark for the dashboard chart.

Pulls live equity series from Alpaca (`get_portfolio_history`) and SPY closes
from the local PriceData cache. Both series are aligned on actual trading days
and the SPY series is rebased to start at the same value as the portfolio so
the lines are directly comparable in dollars. Optional event markers can tag
operator-defined dates on the chart (see EVENTS below).

Output is JSON-serializable; the dashboard fetches this via /api/v1/performance/equity-vs-benchmark.
"""
from datetime import date, datetime, timedelta
from typing import TypedDict

from sqlalchemy.orm import Session

from form4lab.config import settings
from form4lab.models.price import PriceData


class _EventMarker(TypedDict):
    date: str
    label: str
    color: str


class EquityVsBenchmark(TypedDict):
    dates: list[str]
    equity: list[float]
    benchmark: list[float]
    events: list[_EventMarker]
    summary: dict


# Operator-defined chart annotations. Add entries here to mark dates on the
# equity chart (e.g. deploys, config changes) — empty by default.
EVENTS: list[_EventMarker] = []


def _alpaca_equity_series(period: str = "3M") -> list[tuple[date, float]]:
    """Pull daily equity from Alpaca portfolio history. Returns [(date, equity), ...]."""
    cfg = settings.alpaca
    if not cfg.enabled:
        return []
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    client = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
    ph = client.get_portfolio_history(
        GetPortfolioHistoryRequest(period=period, timeframe="1D")
    )
    if not ph.equity or not ph.timestamp:
        return []
    series = []
    for ts, eq in zip(ph.timestamp, ph.equity):
        if eq is None:
            continue
        series.append((datetime.fromtimestamp(ts).date(), float(eq)))
    return series


def _spy_series(db: Session, start: date, end: date) -> dict[date, float]:
    """SPY close prices from the local PriceData cache, keyed by date."""
    rows = (
        db.query(PriceData.date, PriceData.close)
        .filter(
            PriceData.ticker == "SPY",
            PriceData.date >= start,
            PriceData.date <= end,
        )
        .order_by(PriceData.date)
        .all()
    )
    return {r.date: float(r.close) for r in rows}


def compute(db: Session, period: str = "3M") -> EquityVsBenchmark:
    """Build the chart payload.

    Returns an empty payload when Alpaca is disabled or no equity history is available.
    """
    equity_pts = _alpaca_equity_series(period)
    if not equity_pts:
        return {"dates": [], "equity": [], "benchmark": [], "events": [], "summary": {}}

    start = equity_pts[0][0]
    end = equity_pts[-1][0]
    spy_by_date = _spy_series(db, start, end)
    if not spy_by_date:
        return {"dates": [], "equity": [], "benchmark": [], "events": [], "summary": {}}

    # Anchor the SPY series to the first day's portfolio equity so both lines
    # start at the same dollar value. SPY return from that anchor is then
    # applied to give a "what if it had all been SPY" comparison.
    base_equity = equity_pts[0][1]
    spy_anchor_date = min(d for d in spy_by_date if d >= start) if spy_by_date else None
    if spy_anchor_date is None:
        return {"dates": [], "equity": [], "benchmark": [], "events": [], "summary": {}}
    spy_anchor = spy_by_date[spy_anchor_date]

    dates: list[str] = []
    equity: list[float] = []
    benchmark: list[float] = []
    last_spy = spy_anchor
    for d, eq in equity_pts:
        # Use the most recent SPY close on or before this date.
        candidates = [sd for sd in spy_by_date if sd <= d]
        if candidates:
            last_spy = spy_by_date[max(candidates)]
        dates.append(d.isoformat())
        equity.append(round(eq, 2))
        benchmark.append(round(base_equity * (last_spy / spy_anchor), 2))

    portfolio_return = (equity[-1] - equity[0]) / equity[0] * 100 if equity[0] else 0.0
    benchmark_return = (benchmark[-1] - benchmark[0]) / benchmark[0] * 100 if benchmark[0] else 0.0

    visible_events = [e for e in EVENTS if start.isoformat() <= e["date"] <= end.isoformat()]

    return {
        "dates": dates,
        "equity": equity,
        "benchmark": benchmark,
        "events": visible_events,
        "summary": {
            "start_date": dates[0],
            "end_date": dates[-1],
            "start_equity": equity[0],
            "end_equity": equity[-1],
            "portfolio_return_pct": round(portfolio_return, 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "alpha_pp": round(portfolio_return - benchmark_return, 2),
        },
    }
