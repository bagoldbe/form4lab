import json
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache

from fastapi import APIRouter, Request, Depends
from pathlib import Path
from sqlalchemy import or_
from sqlalchemy.orm import Session

from form4lab.config import settings
from form4lab.database import get_db
from form4lab.models.alert import Alert
from form4lab.models.company import Company
from form4lab.models.insider import Insider
from form4lab.models.outcome import TradeOutcome
from form4lab.models.price import PriceData
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.templating import templates


def _sig_sets():
    """Derive tradeable and sell signal types from the active strategy registry."""
    from form4lab.strategy.registry import get_active
    r = get_active()[1]
    tradeable = sorted(r.tradeable_names())
    sells = sorted(r.sell_names())
    return tradeable, sells


router = APIRouter()


def _get_drawdowns(db: Session, tickers: list[str]) -> dict[str, float | None]:
    """Compute 52-week drawdown from cached price data for each ticker.

    Returns dict of ticker -> drawdown float (e.g. -0.30 means 30% below high),
    or None if insufficient price data.  Uses a single batched query to avoid
    N+1 database round-trips.
    """

    if not tickers:
        return {}

    cutoff_252 = (datetime.now() - timedelta(days=370)).date()  # ~252 trading days

    # Single query for all tickers, ordered by ticker then date
    all_prices = (
        db.query(PriceData.ticker, PriceData.adj_close, PriceData.date)
        .filter(
            PriceData.ticker.in_(tickers),
            PriceData.date >= cutoff_252,
        )
        .order_by(PriceData.ticker, PriceData.date.asc())
        .all()
    )

    # Group by ticker
    prices_by_ticker: dict[str, list] = defaultdict(list)
    for row in all_prices:
        prices_by_ticker[row.ticker].append(row)

    result: dict[str, float | None] = {}
    for ticker in tickers:
        prices = prices_by_ticker.get(ticker, [])
        if len(prices) < 20:
            result[ticker] = None
            continue

        high_52w = max(p.adj_close for p in prices)
        latest_price = prices[-1].adj_close
        if high_52w <= 0:
            result[ticker] = None
            continue

        result[ticker] = (latest_price - high_52w) / high_52w

    return result


@lru_cache(maxsize=1)
def _load_perf_metrics() -> dict | None:
    """Load backtest metrics from the static performance export."""
    metrics_path = Path(__file__).resolve().parent.parent / "static" / "data" / "performance_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _build_summary(db: Session, days: int):
    """Performance scorecard filtered to the strategy's tradeable signals."""
    from form4lab.models.insider import InsiderRole
    from form4lab.utils import is_csuite

    tradeable, _ = _sig_sets()
    cutoff = (datetime.now() - timedelta(days=days)).date()
    scored_outcomes = (
        db.query(TradeOutcome, Alert, InsiderRole)
        .join(Alert, TradeOutcome.transaction_id == Alert.transaction_id)
        .join(Insider, Alert.insider_id == Insider.id)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .outerjoin(
            InsiderRole,
            (InsiderRole.insider_id == Alert.insider_id)
            & (InsiderRole.company_id == Alert.company_id),
        )
        .filter(
            Alert.trade_date >= cutoff,
            Alert.alert_type.in_(tradeable),
            Insider.is_institution == False,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .all()
    )

    def _empty_bucket():
        return {"total": 0, "hit_60": 0, "has_60": 0,
                "excess_60_sum": 0.0, "excess_60_count": 0}

    all_perf = _empty_bucket()
    csuite_perf = _empty_bucket()
    non_csuite_perf = _empty_bucket()

    for outcome, alert, role in scored_outcomes:
        is_cs = is_csuite(role.role_title if role else None)
        buckets = [all_perf, csuite_perf if is_cs else non_csuite_perf]
        for perf in buckets:
            perf["total"] += 1
            if outcome.hit_60d is not None:
                perf["has_60"] += 1
                if outcome.hit_60d:
                    perf["hit_60"] += 1
            if outcome.excess_return_60d is not None:
                perf["excess_60_sum"] += outcome.excess_return_60d
                perf["excess_60_count"] += 1

    def _make_scorecard(perf):
        return {
            "total_signals": perf["total"],
            "hit_rate_60d": perf["hit_60"] / perf["has_60"] if perf["has_60"] else None,
            "avg_excess_60d": perf["excess_60_sum"] / perf["excess_60_count"] if perf["excess_60_count"] else None,
        }

    return {
        "scorecard": _make_scorecard(all_perf),
        "csuite_scorecard": _make_scorecard(csuite_perf),
        "non_csuite_scorecard": _make_scorecard(non_csuite_perf),
        "days": days,
    }


def _build_recommendations(db: Session):
    """Get recent high-alpha signals for actionable recommendations."""
    from form4lab.models.insider import InsiderRole

    cutoff = (datetime.now() - timedelta(days=30)).date()
    high_alpha_types, _ = _sig_sets()

    rows = (
        db.query(Alert, Insider, Company)
        .join(Insider, Alert.insider_id == Insider.id)
        .join(Company, Alert.company_id == Company.id)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .filter(
            Alert.trade_date >= cutoff,
            Alert.alert_type.in_(high_alpha_types),
            Insider.is_institution == False,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Alert.conviction_score.desc())
        .limit(10)
        .all()
    )

    if not rows:
        return []

    # Batch-fetch scores and roles to avoid N+1 queries
    insider_ids = [insider.id for _, insider, _ in rows]
    company_ids = [company.id for _, _, company in rows]

    scores_by_insider = {
        s.insider_id: s
        for s in db.query(InsiderScore)
        .filter(InsiderScore.insider_id.in_(insider_ids), InsiderScore.company_id == None)  # noqa: E711
        .all()
    }
    roles_by_key = {
        (r.insider_id, r.company_id): r
        for r in db.query(InsiderRole)
        .filter(InsiderRole.insider_id.in_(insider_ids), InsiderRole.company_id.in_(company_ids))
        .all()
    }

    recommendations = []
    for alert, insider, company in rows:
        score = scores_by_insider.get(insider.id)
        role = roles_by_key.get((insider.id, company.id))

        recommendations.append({
            "ticker": company.ticker,
            "company_name": company.name,
            "alert_type": alert.alert_type,
            "insider_name": insider.name,
            "insider_cik": insider.cik,
            "role": role.role_title if role else "Other",
            "tier": score.credibility_tier if score else "Insufficient",
            "skill_score": score.skill_score if score else 0,
            "hit_rate": score.bayesian_hit_rate if score else None,
            "transaction_value": alert.transaction_value,
            "conviction": alert.conviction_score,
            "trade_date": alert.trade_date,
            "summary": alert.summary,
        })

    # Enrich with drawdown data
    tickers = [r["ticker"] for r in recommendations]
    drawdowns = _get_drawdowns(db, tickers)
    threshold = settings.alpaca.drawdown_threshold  # None means disabled

    perf = _load_perf_metrics()
    try:
        win_rate_str = "{:.0f}%".format(perf["win_rate"] * 100) if perf else "—"
        by_signal = perf.get("by_signal", {}) if perf else {}
        signal_perf = next((by_signal[t] for t in high_alpha_types if t in by_signal), None)
        avg_return_str = "{:+.0f}%".format(signal_perf["avg_return"] * 100) if signal_perf else "—"
    except (KeyError, TypeError):
        win_rate_str = "—"
        avg_return_str = "—"

    for rec in recommendations:
        dd = drawdowns.get(rec["ticker"])
        rec["drawdown"] = dd
        rec["passes_drawdown"] = threshold is None or dd is None or dd <= threshold
        rec["backtest_win_rate"] = win_rate_str
        rec["backtest_avg_return"] = avg_return_str

    # Sort: drawdown-passing first, then by conviction
    recommendations.sort(key=lambda r: (not r["passes_drawdown"], -(r["conviction"] or 0)))

    return recommendations


@router.get("/partials/recommendations")
async def recommendations_partial(
    request: Request, db: Session = Depends(get_db)
):
    recommendations = _build_recommendations(db)
    return templates.TemplateResponse(
        request, "partials/recommendations.html",
        {"recommendations": recommendations, "drawdown_threshold": abs(int(settings.alpaca.drawdown_threshold * 100)) if settings.alpaca.drawdown_threshold is not None else None},
    )


@router.get("/partials/action-summary")
async def action_summary_partial(
    request: Request, days: int = 30, db: Session = Depends(get_db)
):
    days = max(1, min(days, 365))
    summary = _build_summary(db, days)
    return templates.TemplateResponse(
        request, "partials/action_summary.html", summary
    )


@router.get("/partials/sell-warnings")
async def sell_warnings_partial(
    request: Request, db: Session = Depends(get_db)
):
    """Sell warnings from credible insiders in the last 30 days."""
    from form4lab.models.insider import InsiderRole

    _, sells = _sig_sets()
    cutoff = (datetime.now() - timedelta(days=30)).date()
    rows = (
        db.query(Alert, Insider, Company)
        .join(Insider, Alert.insider_id == Insider.id)
        .join(Company, Alert.company_id == Company.id)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .filter(
            Alert.trade_date >= cutoff,
            Alert.alert_type.in_(sells),
            Insider.is_institution == False,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Alert.trade_date.desc())
        .limit(15)
        .all()
    )

    # Batch-fetch roles
    if rows:
        insider_ids = [insider.id for _, insider, _ in rows]
        company_ids = [company.id for _, _, company in rows]
        roles_by_key = {
            (r.insider_id, r.company_id): r
            for r in db.query(InsiderRole)
            .filter(InsiderRole.insider_id.in_(insider_ids), InsiderRole.company_id.in_(company_ids))
            .all()
        }
    else:
        roles_by_key = {}

    warnings = []
    for alert, insider, company in rows:
        role = roles_by_key.get((insider.id, company.id))
        warnings.append({
            "trade_date": alert.trade_date.strftime("%b %d") if alert.trade_date else "—",
            "insider_name": insider.name,
            "insider_cik": insider.cik,
            "role": role.role_title if role else "Other",
            "ticker": company.ticker or "—",
            "alert_type": alert.alert_type,
            "value": alert.transaction_value,
            "summary": alert.summary,
        })

    return templates.TemplateResponse(
        request, "partials/sell_warnings.html",
        {"warnings": warnings},
    )


@router.get("/partials/this-week")
async def this_week_partial(
    request: Request, db: Session = Depends(get_db)
):
    """Recent tradeable buys and sell warnings — last 7 days, falling back to 30 if empty."""
    from form4lab.models.insider import InsiderRole

    tradeable, sells = _sig_sets()
    eh_and_sell_types = tradeable + sells
    cutoff_7 = (datetime.now() - timedelta(days=7)).date()
    cutoff_30 = (datetime.now() - timedelta(days=30)).date()

    def _fetch_trades(cutoff):
        return (
            db.query(Alert, Insider, Company, Transaction)
            .join(Insider, Alert.insider_id == Insider.id)
            .join(Company, Alert.company_id == Company.id)
            .join(Transaction, Alert.transaction_id == Transaction.id)
            .filter(
                Alert.trade_date >= cutoff,
                Alert.alert_type.in_(eh_and_sell_types),
                Insider.is_institution == False,  # noqa: E712
                or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
            )
            .order_by(Alert.conviction_score.desc())
            .limit(25)
            .all()
        )

    rows = _fetch_trades(cutoff_7)
    recent_count = len(rows)
    if recent_count == 0:
        rows = _fetch_trades(cutoff_30)

    # Batch-fetch scores and roles
    if rows:
        insider_ids = [insider.id for _, insider, _, _ in rows]
        company_ids = [company.id for _, _, company, _ in rows]
        scores_by_insider = {
            s.insider_id: s
            for s in db.query(InsiderScore)
            .filter(InsiderScore.insider_id.in_(insider_ids), InsiderScore.company_id == None)  # noqa: E711
            .all()
        }
        roles_by_key = {
            (r.insider_id, r.company_id): r
            for r in db.query(InsiderRole)
            .filter(InsiderRole.insider_id.in_(insider_ids), InsiderRole.company_id.in_(company_ids))
            .all()
        }
    else:
        scores_by_insider = {}
        roles_by_key = {}

    trades = []
    for alert, insider, company, txn in rows:
        score = scores_by_insider.get(insider.id)
        role = roles_by_key.get((insider.id, company.id))
        trades.append({
            "trade_date": alert.trade_date.strftime("%b %d") if alert.trade_date else "—",
            "insider_name": insider.name,
            "insider_cik": insider.cik,
            "role": role.role_title if role else "Other",
            "ticker": company.ticker or "—",
            "tier": score.credibility_tier if score else "Insufficient",
            "hit_rate": score.bayesian_hit_rate if score else None,
            "price": txn.price_per_share,
            "value": alert.transaction_value,
            "conviction": alert.conviction_score,
            "alert_type": alert.alert_type,
        })

    return templates.TemplateResponse(
        request, "partials/this_week.html",
        {"trades": trades, "recent_count": recent_count},
    )


@router.get("/partials/portfolio")
async def portfolio_partial(
    request: Request, db: Session = Depends(get_db)
):
    """Paper trading portfolio status partial."""
    from form4lab.services.alpaca_service import get_portfolio_summary

    summary = get_portfolio_summary(db)
    return templates.TemplateResponse(
        request, "partials/portfolio.html", summary
    )
