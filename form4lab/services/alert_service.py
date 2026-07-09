"""Shared alert query, enrichment, and batch generation logic.

Eliminates N+1 queries and duplicated alert query patterns across routes,
CLI commands, and scheduler jobs.
"""
import logging
from datetime import date, timedelta
from scipy.stats import percentileofscore

from sqlalchemy import exists as sa_exists, or_
from sqlalchemy.orm import Session

from form4lab.models.alert import Alert
from form4lab.models.company import Company
from form4lab.models.insider import Insider
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction

logger = logging.getLogger(__name__)


def _registry():
    from form4lab.strategy.registry import get_active
    return get_active()[1]


def build_person_alerts_query(db: Session, cutoff_date: date, exclude_filtered: bool = True):
    """Build base query for non-institutional, common-stock alerts.

    This is the standard filter used across dashboard, partials, summary,
    and API routes. Returns a SQLAlchemy query that callers can further
    filter, order, and limit.
    """
    query = (
        db.query(Alert)
        .join(Insider, Alert.insider_id == Insider.id)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .filter(
            Alert.trade_date >= cutoff_date,
            Insider.is_institution == False,  # noqa: E712
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        )
    )
    if exclude_filtered:
        hidden = _registry().hidden_names()
        if hidden:
            query = query.filter(Alert.alert_type.notin_(hidden))
    return query


def enrich_alerts(alerts: list[Alert], db: Session) -> list[dict]:
    """Batch-load related data for a list of alerts.

    Replaces the N+1 pattern of calling db.get() per alert with
    batch lookups for insiders, companies, and scores.
    """
    if not alerts:
        return []

    # Collect unique IDs
    insider_ids = set(a.insider_id for a in alerts)
    company_ids = set(a.company_id for a in alerts)

    # Batch load insiders
    insiders = {i.id: i for i in db.query(Insider).filter(Insider.id.in_(insider_ids)).all()}

    # Batch load companies
    companies = {c.id: c for c in db.query(Company).filter(Company.id.in_(company_ids)).all()}

    # Batch load global scores (company_id == None)
    scores_list = (
        db.query(InsiderScore)
        .filter(
            InsiderScore.insider_id.in_(insider_ids),
            InsiderScore.company_id == None,  # noqa: E711
        )
        .all()
    )
    scores = {s.insider_id: s for s in scores_list}

    return [
        {
            "alert": alert,
            "insider": insiders.get(alert.insider_id),
            "company": companies.get(alert.company_id),
            "score": scores.get(alert.insider_id),
        }
        for alert in alerts
    ]


_CONVICTION_LABELS = {5: "Very Strong", 4: "Strong", 3: "Moderate", 2: "Weak", 1: "Noise"}


def normalize_conviction(alerts: list[dict]) -> list[dict]:
    """Add conviction_display (1-5) and conviction_label to enriched alert dicts.

    Uses percentile buckets based on raw conviction_score:
      Top  5% (pct > 0.95) → 5 "Very Strong"
      Top 15% (pct > 0.85) → 4 "Strong"
      Top 40% (pct > 0.60) → 3 "Moderate"
      Top 70% (pct > 0.30) → 2 "Weak"
      Bottom 30%            → 1 "Noise"

    Special cases:
      - Empty list returns []
      - Single item returns with display=5, label="Very Strong"
    """
    if not alerts:
        return []

    if len(alerts) == 1:
        alerts[0]["conviction_display"] = 5
        alerts[0]["conviction_label"] = _CONVICTION_LABELS[5]
        return alerts

    scores = [a["alert"].conviction_score for a in alerts]

    for alert_dict in alerts:
        raw = alert_dict["alert"].conviction_score
        pct = percentileofscore(scores, raw, kind="rank") / 100.0

        if pct > 0.95:
            level = 5
        elif pct > 0.85:
            level = 4
        elif pct > 0.60:
            level = 3
        elif pct > 0.30:
            level = 2
        else:
            level = 1

        alert_dict["conviction_display"] = level
        alert_dict["conviction_label"] = _CONVICTION_LABELS[level]

    return alerts


def generate_missing_alerts(db: Session, max_age_days: int = 90) -> int:
    """Generate alerts for recent discretionary buys that don't have one yet.

    Only considers transactions from the last `max_age_days` days to prevent
    generating alerts for ancient transactions during bulk regeneration.

    Shared by the generate-signals CLI command and scheduler jobs.
    Returns count of alerts generated.
    """
    from form4lab.scoring.signal_generator import score_new_transaction

    cutoff = date.today() - timedelta(days=max_age_days)
    pending_ids = [
        row[0] for row in db.query(Transaction.id).filter(
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.transaction_date >= cutoff,
            ~sa_exists().where(Alert.transaction_id == Transaction.id),
        ).all()
    ]

    generated = 0
    for txn_id in pending_ids:
        try:
            alert = score_new_transaction(txn_id, db)
            if alert:
                generated += 1
        except Exception as e:
            db.rollback()
            logger.warning("Error scoring txn %d: %s", txn_id, e)

    return generated


def generate_and_execute_alerts(db: Session, max_age_days: int = 90) -> tuple[int, int]:
    """Generate missing alerts AND execute tradeable ones on Alpaca.

    Only considers transactions from the last `max_age_days` days to prevent
    executing ancient transactions that surface during bulk regeneration.

    Returns (alerts_generated, orders_placed).
    """
    from form4lab.scoring.signal_generator import score_new_transaction

    cutoff = date.today() - timedelta(days=max_age_days)
    pending_ids = [
        row[0] for row in db.query(Transaction.id).filter(
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.transaction_date >= cutoff,
            ~sa_exists().where(Alert.transaction_id == Transaction.id),
        ).all()
    ]

    generated = 0
    executed = 0
    for txn_id in pending_ids:
        try:
            alert = score_new_transaction(txn_id, db)
            if alert:
                generated += 1
                if _registry().is_tradeable(alert.alert_type):
                    try:
                        from form4lab.services.alpaca_service import execute_signal
                        position = execute_signal(alert, db)
                        if position:
                            executed += 1
                    except Exception as e:
                        logger.warning("Alpaca execution failed for alert %d: %s", alert.id, e)
        except Exception as e:
            db.rollback()
            logger.warning("Error scoring txn %d: %s", txn_id, e)

    return generated, executed
