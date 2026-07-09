import re
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session
from form4lab.database import get_db
from form4lab.models.alert import Alert
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.score import InsiderScore
from form4lab.models.price import PriceData
from form4lab.models.transaction import Transaction
from form4lab.services.alert_service import enrich_alerts

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker format."""
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail=f"Invalid ticker format: {ticker}")
    return t


@router.get("/alerts")
async def get_alerts(
    days_back: int = Query(default=7, ge=1, le=365),
    min_conviction: float = Query(default=0, ge=0, le=1),
    alert_types: str = "",
    ticker: str = "",
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(Alert).join(
        Transaction, Alert.transaction_id == Transaction.id
    ).filter(
        Alert.created_at >= datetime.now() - timedelta(days=days_back),
        Alert.alert_type != "filtered_out",
        or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
    )
    if min_conviction > 0:
        query = query.filter(Alert.conviction_score >= min_conviction)
    if alert_types:
        types = [t.strip() for t in alert_types.split(",")]
        query = query.filter(Alert.alert_type.in_(types))
    if ticker:
        t = _validate_ticker(ticker)
        company = db.query(Company).filter(Company.ticker == t).first()
        if company:
            query = query.filter(Alert.company_id == company.id)
        else:
            return []

    alerts = query.order_by(Alert.conviction_score.desc()).offset(skip).limit(limit).all()
    enriched = enrich_alerts(alerts, db)

    return [
        {
            "id": e["alert"].id,
            "alert_type": e["alert"].alert_type,
            "conviction_score": e["alert"].conviction_score,
            "insider_skill_score": e["alert"].insider_skill_score,
            "transaction_value": e["alert"].transaction_value,
            "cluster_id": e["alert"].cluster_id,
            "summary": e["alert"].summary,
            "created_at": str(e["alert"].created_at),
            "insider": {
                "name": e["insider"].name if e["insider"] else None,
                "cik": e["insider"].cik if e["insider"] else None,
            },
            "company": {
                "name": e["company"].name if e["company"] else None,
                "ticker": e["company"].ticker if e["company"] else None,
            },
        }
        for e in enriched
    ]


@router.get("/insider/{cik}/score")
async def get_insider_score(cik: str, db: Session = Depends(get_db)):
    insider = db.query(Insider).filter(Insider.cik == cik).first()
    if not insider:
        raise HTTPException(status_code=404, detail="Insider not found")

    score = (
        db.query(InsiderScore)
        .filter(
            InsiderScore.insider_id == insider.id,
            InsiderScore.company_id == None,  # noqa: E711
        )
        .first()
    )
    if not score:
        raise HTTPException(status_code=404, detail="Score not found")

    return {
        "cik": cik,
        "name": insider.name,
        "skill_score": score.skill_score,
        "tier": score.credibility_tier,
        "hit_rate": score.bayesian_hit_rate,
        "raw_hit_rate": score.raw_hit_rate,
        "confidence_above_baseline": score.confidence_above_baseline,
        "excess_return": score.shrunk_excess_return,
        "avg_excess_return": score.avg_excess_return,
        "momentum_adjusted_excess": score.momentum_adjusted_excess,
        "credibility_weight": score.credibility_weight,
        "num_buys": score.num_discretionary_buys,
        "computed_at": str(score.computed_at),
    }


@router.get("/performance/equity-vs-benchmark")
async def equity_vs_benchmark(
    period: str = Query(default="3M", pattern=r"^(1M|3M|6M|1A)$"),
    db: Session = Depends(get_db),
):
    """Live portfolio equity vs SPY benchmark for the dashboard chart."""
    from form4lab.services.portfolio_history import compute
    return compute(db, period=period)


@router.get("/prices/{ticker}")
async def get_prices(ticker: str, days: int = Query(default=0, ge=0, le=3650), db: Session = Depends(get_db)):
    t = _validate_ticker(ticker)
    query = db.query(PriceData).filter(PriceData.ticker == t)
    if days > 0:
        cutoff = datetime.now() - timedelta(days=days)
        query = query.filter(PriceData.date >= cutoff.date())
    prices = query.order_by(PriceData.date).all()
    return [
        {
            "time": str(p.date),
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        }
        for p in prices
    ]
