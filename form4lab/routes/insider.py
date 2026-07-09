from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session
from form4lab.database import get_db
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.company import Company
from form4lab.templating import templates

router = APIRouter()


@router.get("/insider/{cik}")
async def insider_profile(cik: str, request: Request, db: Session = Depends(get_db)):
    insider = db.query(Insider).filter(Insider.cik == cik).first()
    if not insider:
        return templates.TemplateResponse(
            request,
            "404.html",
            {"message": f"Insider with CIK {cik} not found"},
            status_code=404,
        )

    # Get global score (company_id is None)
    score = (
        db.query(InsiderScore)
        .filter(
            InsiderScore.insider_id == insider.id,
            InsiderScore.company_id == None,  # noqa: E711
        )
        .first()
    )

    # Get roles with company info
    roles = (
        db.query(InsiderRole, Company)
        .join(Company)
        .filter(InsiderRole.insider_id == insider.id)
        .all()
    )

    # Get transactions with outcomes and company info
    transactions = (
        db.query(Transaction, TradeOutcome, Company)
        .outerjoin(TradeOutcome)
        .join(Company)
        .filter(
            Transaction.insider_id == insider.id,
            Transaction.is_discretionary == True,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
        .order_by(Transaction.transaction_date.desc())
        .all()
    )

    return templates.TemplateResponse(
        request,
        "insider_profile.html",
        {
            "insider": insider,
            "cik": cik,
            "score": score,
            "roles": roles,
            "transactions": transactions,
        },
    )
