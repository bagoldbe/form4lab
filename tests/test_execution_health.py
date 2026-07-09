from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.alert import Alert
from form4lab.models.broker import BrokerOrder, BrokerPosition
from form4lab.models.company import Company
from form4lab.models.insider import Insider
from form4lab.models.transaction import Transaction
from form4lab.services.execution_health import (
    LOG_MARKER,
    WINDOW_DAYS,
    check_execution_health,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_alert(db, *, days_ago: int, alert_type: str = "cluster_buy", suffix: str = "a"):
    """Create an alert dated `days_ago` calendar days back."""
    created = datetime.utcnow() - timedelta(days=days_ago)
    insider = Insider(cik=f"cik-{suffix}", name=f"insider-{suffix}")
    company = Company(cik=f"comp-{suffix}", ticker=f"TST{suffix.upper()}", name=f"Co {suffix}")
    db.add_all([insider, company])
    db.flush()
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number=f"acc-{suffix}",
        filing_date=created.date(), transaction_date=created.date(),
        transaction_code="P", shares=10, price_per_share=10.0,
        total_value=100.0, shares_owned_after=20, acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.flush()
    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type=alert_type, conviction_score=1.0,
        insider_skill_score=0.0, transaction_value=100.0,
        summary="test", trade_date=created.date(),
        created_at=created,
    )
    db.add(alert)
    db.commit()
    return alert


def _make_position(db, alert_id: int, *, days_ago: int, suffix: str = "a"):
    """Create a BrokerPosition with entry_date `days_ago` back."""
    order = BrokerOrder(
        alert_id=alert_id, alpaca_order_id=f"ord-{suffix}",
        symbol=f"TST{suffix.upper()}", side="buy", notional=100.0,
        order_type="market", extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=alert_id, entry_order_id=order.id,
        symbol=f"TST{suffix.upper()}", shares=10, entry_price=10.0,
        entry_date=date.today() - timedelta(days=days_ago),
        exit_target_date=date.today() + timedelta(days=60),
        status="open", insider_name="x", insider_role="Director",
    )
    db.add(pos)
    db.commit()
    return pos


def test_no_alerts_in_window_is_healthy(db):
    """Quiet weeks (e.g., holidays, low signal volume) should not alert."""
    report = check_execution_health(db)
    assert report.healthy is True
    assert report.tradeable_alerts == 0
    assert report.positions_opened == 0
    assert "no tradeable alerts" in report.reason


def test_alerts_with_positions_is_healthy(db):
    """Normal operation: alerts arrive and positions get opened."""
    alert = _make_alert(db, days_ago=3)
    _make_position(db, alert.id, days_ago=3)
    report = check_execution_health(db)
    assert report.healthy is True
    assert report.tradeable_alerts == 1
    assert report.positions_opened == 1


def test_alerts_without_positions_is_unhealthy(db):
    """Silent-execution-freeze pattern: alerts generated, nothing executed."""
    _make_alert(db, days_ago=5, suffix="a")
    _make_alert(db, days_ago=2, suffix="b")
    report = check_execution_health(db)
    assert report.healthy is False
    assert report.tradeable_alerts == 2
    assert report.positions_opened == 0
    assert "may be silently frozen" in report.reason


def test_old_alerts_outside_window_ignored(db):
    """Alerts older than the window must not skew the verdict."""
    _make_alert(db, days_ago=WINDOW_DAYS + 5, suffix="old")
    report = check_execution_health(db)
    assert report.healthy is True
    assert report.tradeable_alerts == 0


def test_non_tradeable_alert_types_ignored(db):
    """filtered_out / first_time_buy etc. are dashboard-only and must not trip the gate."""
    _make_alert(db, days_ago=3, alert_type="filtered_out", suffix="c")
    _make_alert(db, days_ago=3, alert_type="first_time_buy", suffix="f")
    report = check_execution_health(db)
    assert report.healthy is True
    assert report.tradeable_alerts == 0


def test_log_marker_is_stable():
    """Greppable from log streams — should not change without intent."""
    assert LOG_MARKER == "EXEC_HEALTH"
