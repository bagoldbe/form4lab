from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session
from form4lab.database import get_db
from form4lab.models.company import Company
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.templating import templates

router = APIRouter()


@router.get("/company/{ticker}")
async def company_page(ticker: str, request: Request, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.ticker == ticker).first()
    if not company:
        return templates.TemplateResponse(
            request,
            "404.html",
            {"message": f"Company with ticker {ticker} not found"},
            status_code=404,
        )

    # Get all insiders with roles at this company + their global scores
    insider_data = (
        db.query(InsiderRole, Insider, InsiderScore)
        .select_from(InsiderRole)
        .join(Insider, InsiderRole.insider_id == Insider.id)
        .outerjoin(
            InsiderScore,
            (InsiderScore.insider_id == Insider.id)
            & (InsiderScore.company_id == None),  # noqa: E711
        )
        .filter(InsiderRole.company_id == company.id)
        .all()
    )

    # Get recent transactions (last 90 days)
    cutoff = datetime.now() - timedelta(days=90)
    recent_transactions = (
        db.query(Transaction, Insider)
        .join(Insider)
        .filter(
            Transaction.company_id == company.id,
            Transaction.transaction_date >= cutoff.date(),
            Transaction.is_discretionary == True,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Transaction.transaction_date.desc())
        .all()
    )

    # Get ALL discretionary buys for chart markers (not just 90 days)
    all_transactions = (
        db.query(Transaction, Insider)
        .join(Insider)
        .filter(
            Transaction.company_id == company.id,
            Transaction.is_discretionary == True,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Transaction.transaction_date.desc())
        .all()
    )

    # Detect clusters: 2+ insiders buying within the same week
    clusters = []
    if recent_transactions:
        weekly_buys = defaultdict(list)
        for txn, ins in recent_transactions:
            # Group by ISO week
            week_key = txn.transaction_date.isocalendar()[:2]
            weekly_buys[week_key].append({"transaction": txn, "insider": ins})

        for week_key, trades in weekly_buys.items():
            unique_insiders = set(t["insider"].id for t in trades)
            if len(unique_insiders) >= 2:
                total_value = sum(
                    t["transaction"].total_value or 0 for t in trades
                )
                clusters.append(
                    {
                        "week": f"{week_key[0]}-W{week_key[1]:02d}",
                        "num_insiders": len(unique_insiders),
                        "num_trades": len(trades),
                        "total_value": total_value,
                        "trades": trades,
                    }
                )

    return templates.TemplateResponse(
        request,
        "company.html",
        {
            "company": company,
            "ticker": ticker,
            "insider_data": insider_data,
            "recent_transactions": recent_transactions,
            "all_transactions": all_transactions,
            "clusters": clusters,
        },
    )
