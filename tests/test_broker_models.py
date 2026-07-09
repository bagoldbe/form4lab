import pytest
from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.broker import BrokerOrder, BrokerPosition  # noqa: F401 — register models before create_all


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_broker_order_create(db):
    from form4lab.models.broker import BrokerOrder

    order = BrokerOrder(
        alert_id=1,
        alpaca_order_id="abc-123",
        symbol="AAPL",
        side="buy",
        qty=10.0,
        order_type="market",
        extended_hours=False,
        status="submitted",
    )
    db.add(order)
    db.flush()
    assert order.id is not None
    assert order.alpaca_order_id == "abc-123"
    assert order.filled_avg_price is None


def test_broker_position_create(db):
    from form4lab.models.broker import BrokerOrder, BrokerPosition

    order = BrokerOrder(
        alert_id=1, alpaca_order_id="abc-123", symbol="AAPL",
        side="buy", qty=10.0, order_type="market",
        extended_hours=False, status="filled",
        filled_avg_price=150.0, filled_qty=10.0,
    )
    db.add(order)
    db.flush()

    position = BrokerPosition(
        alert_id=1,
        entry_order_id=order.id,
        symbol="AAPL",
        shares=10.0,
        entry_price=150.0,
        entry_date=date(2026, 2, 21),
        exit_target_date=date(2026, 5, 15),
        status="open",
        insider_name="Jane Smith",
        insider_role="CEO",
    )
    db.add(position)
    db.flush()
    assert position.id is not None
    assert position.status == "open"
    assert position.pnl is None


def test_broker_order_exit_links_to_entry(db):
    from form4lab.models.broker import BrokerOrder

    entry = BrokerOrder(
        alert_id=1, alpaca_order_id="entry-1", symbol="AAPL",
        side="buy", qty=10.0, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(entry)
    db.flush()

    exit_order = BrokerOrder(
        alert_id=1, alpaca_order_id="exit-1", symbol="AAPL",
        side="sell", qty=10.0, order_type="market",
        extended_hours=False, status="submitted",
        entry_order_id=entry.id,
    )
    db.add(exit_order)
    db.flush()
    assert exit_order.entry_order_id == entry.id
