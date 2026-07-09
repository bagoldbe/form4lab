import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from form4lab.models.transaction import Transaction
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.score import InsiderScore
from form4lab.models.alert import Alert
from form4lab.strategy.base import TxnView

logger = logging.getLogger(__name__)


def score_sell_transaction(transaction_id: int, db: Session) -> Alert | None:
    """Score a sell transaction and generate a sell-avoidance alert.

    Only alerts when the active strategy's evaluate_sell finds this sell
    worth flagging; returns None otherwise.
    """
    txn = db.get(Transaction, transaction_id)
    if not txn or txn.transaction_code != "S":
        return None
    if txn.is_common_stock is False:
        return None

    insider = db.get(Insider, txn.insider_id)
    company = db.get(Company, txn.company_id)
    if not insider or not company:
        return None

    if insider.is_institution:
        return None

    # Dedup: skip if an alert already exists for this sell event. Sell alert
    # types are whatever the active strategy declares as its sell-direction,
    # visible signal types — see SignalRegistry.sell_names().
    from form4lab.strategy.registry import get_active
    existing_alert = db.query(Alert).filter(
        Alert.insider_id == insider.id,
        Alert.company_id == company.id,
        Alert.trade_date == txn.transaction_date,
        Alert.alert_type.in_(sorted(get_active()[1].sell_names())),
    ).first()
    if existing_alert:
        return None

    # Get insider score
    score = db.query(InsiderScore).filter(
        InsiderScore.insider_id == insider.id,
        InsiderScore.company_id == None,  # noqa: E711 — global score
    ).first()

    tier = score.credibility_tier if score else "Insufficient"
    skill_score = score.skill_score if score else 0.0

    # Sum same-day sell value
    same_day_total = (
        db.query(func.sum(Transaction.total_value))
        .filter(
            Transaction.insider_id == insider.id,
            Transaction.company_id == company.id,
            Transaction.transaction_date == txn.transaction_date,
            Transaction.transaction_code == "S",
            Transaction.total_value.isnot(None),
        )
        .scalar()
    )
    txn_value = same_day_total or txn.total_value or (txn.shares * (txn.price_per_share or 0))

    # Get role
    role = db.query(InsiderRole).filter(
        InsiderRole.insider_id == insider.id,
        InsiderRole.company_id == company.id,
    ).first()
    role_title = role.role_title if role else "Other"

    from form4lab.strategy.features import LiveFeatureView
    from form4lab.strategy.registry import get_active

    strategy = get_active()[0]
    txn_view = TxnView(insider_id=insider.id, company_id=company.id,
                       ticker=company.ticker, transaction_date=txn.transaction_date,
                       txn_value=float(txn_value or 0.0), role_title=role_title)
    f = LiveFeatureView(db, txn, tier, skill_score, role_title, score=score)
    evaluation = strategy.evaluate_sell(txn_view, f)
    if evaluation is None:
        return None

    alert = Alert(
        transaction_id=txn.id,
        insider_id=insider.id,
        company_id=company.id,
        alert_type=evaluation.alert_type,
        conviction_score=evaluation.conviction,
        insider_skill_score=float(skill_score),
        transaction_value=float(txn_value or 0.0),
        cluster_id=None,
        summary=evaluation.summary,
        trade_date=txn.transaction_date,
    )
    db.add(alert)
    db.commit()
    return alert


def score_new_transaction(transaction_id: int, db: Session) -> Alert | None:
    """Score a new transaction and generate an alert."""
    txn = db.get(Transaction, transaction_id)
    if not txn or not txn.is_discretionary:
        return None
    # TODO: optionally exclude plan-based trades; off pending operator research
    # if txn.is_10b5_1_plan is True:
    #     return None
    if txn.is_common_stock is False:
        return None

    insider = db.get(Insider, txn.insider_id)
    company = db.get(Company, txn.company_id)
    if not insider or not company:
        return None

    # Skip institutions -- they're not individual insiders
    if insider.is_institution:
        return None

    # Dedup: skip if an alert already exists for the same buy event
    existing_alert = db.query(Alert).filter(
        Alert.insider_id == insider.id,
        Alert.company_id == company.id,
        Alert.trade_date == txn.transaction_date,
    ).first()
    if existing_alert:
        return None

    # Sum all same-day lots for this insider+company to get true event value
    same_day_total = (
        db.query(func.sum(Transaction.total_value))
        .filter(
            Transaction.insider_id == insider.id,
            Transaction.company_id == company.id,
            Transaction.transaction_date == txn.transaction_date,
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.total_value.isnot(None),
        )
        .scalar()
    )
    txn_value = same_day_total or txn.total_value or (txn.shares * (txn.price_per_share or 0))

    # Skip micro-buys (< $500) and absurdly large values (> $10B) — engine
    # noise filter, configurable later.
    if txn_value < 500 or txn_value > 10_000_000_000:
        return None

    # Get insider score
    score = db.query(InsiderScore).filter(
        InsiderScore.insider_id == insider.id,
        InsiderScore.company_id == None,  # noqa: E711 — global score
    ).first()

    skill_score = score.skill_score if score else 0.0
    tier = score.credibility_tier if score else "Insufficient"

    # Get role
    role = db.query(InsiderRole).filter(
        InsiderRole.insider_id == insider.id,
        InsiderRole.company_id == company.id,
    ).first()
    role_title = role.role_title if role else "Other"

    from form4lab.strategy.features import LiveFeatureView
    from form4lab.strategy.registry import get_active

    strategy = get_active()[0]
    txn_view = TxnView(insider_id=insider.id, company_id=company.id,
                       ticker=company.ticker, transaction_date=txn.transaction_date,
                       txn_value=float(txn_value or 0.0), role_title=role_title)
    f = LiveFeatureView(db, txn, tier, skill_score, role_title, score=score)
    evaluation = strategy.evaluate_buy(txn_view, f)
    if evaluation is None:
        return None

    # Create alert
    alert = Alert(
        transaction_id=txn.id,
        insider_id=insider.id,
        company_id=company.id,
        alert_type=evaluation.alert_type,
        conviction_score=evaluation.conviction,
        insider_skill_score=skill_score,
        transaction_value=txn_value or 0.0,
        cluster_id=evaluation.cluster_id,
        summary=evaluation.summary,
        trade_date=txn.transaction_date,
    )
    db.add(alert)
    db.commit()
    return alert
