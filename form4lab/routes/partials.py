from datetime import datetime, timedelta
from fastapi import APIRouter, Request, Depends
from sqlalchemy import or_
from sqlalchemy.orm import Session
from form4lab.database import get_db
from form4lab.models.alert import Alert
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.services.alert_service import (
    build_person_alerts_query,
    enrich_alerts,
    normalize_conviction,
)
from form4lab.templating import templates

router = APIRouter()


@router.get("/partials/alerts")
async def alerts_partial(
    request: Request, filter: str = "all", days: int = 30, db: Session = Depends(get_db)
):
    days = max(1, min(days, 365))
    cutoff = (datetime.now() - timedelta(days=days)).date()

    exclude_filtered = filter != "unfiltered"
    query = build_person_alerts_query(db, cutoff, exclude_filtered=exclude_filtered)

    if filter == "elite":
        elite_insider_ids = (
            db.query(InsiderScore.insider_id)
            .filter(
                InsiderScore.credibility_tier == "Elite",
                InsiderScore.company_id == None,  # noqa: E711
            )
            .scalar_subquery()
        )
        query = query.filter(Alert.insider_id.in_(elite_insider_ids))
    elif filter == "cluster":
        query = query.filter(Alert.cluster_id != None)  # noqa: E711
    elif filter == "first_time":
        query = query.filter(Alert.alert_type == "first_time_buy")

    alerts = query.order_by(Alert.conviction_score.desc()).limit(50).all()
    enriched = enrich_alerts(alerts, db)
    enriched = normalize_conviction(enriched)

    return templates.TemplateResponse(
        request, "partials/alert_feed.html", {"alerts": enriched}
    )


@router.get("/partials/insider-detail/{cik}")
async def insider_detail_partial(
    cik: str, request: Request, db: Session = Depends(get_db)
):
    insider = db.query(Insider).filter(Insider.cik == cik).first()
    if not insider:
        return templates.TemplateResponse(
            request,
            "partials/insider_detail.html",
            {"insider": None, "score": None, "recent_trades": []},
        )

    # Get global score
    score = (
        db.query(InsiderScore)
        .filter(
            InsiderScore.insider_id == insider.id,
            InsiderScore.company_id == None,  # noqa: E711
        )
        .first()
    )

    # Get recent 5 trades with outcomes
    recent_trades = (
        db.query(Transaction, TradeOutcome, Company)
        .outerjoin(TradeOutcome)
        .join(Company)
        .filter(
            Transaction.insider_id == insider.id,
            Transaction.is_discretionary == True,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Transaction.transaction_date.desc())
        .limit(5)
        .all()
    )

    return templates.TemplateResponse(
        request,
        "partials/insider_detail.html",
        {
            "insider": insider,
            "score": score,
            "recent_trades": recent_trades,
        },
    )






