"""Integration test: signal -> alert -> Alpaca order (mocked).

Fixtures use alert_type "cluster_buy" (the only tradeable rung
ClusterBuyStrategy declares) and expected notional amounts sized at the
shipped default Strategy.size() (form4lab/strategy/base.py): a flat 5%
that does not read role at all. test_non_tradeable_signal_not_executed
covers the opposite path — the strategy tradeable-gates on cluster_buy,
rejecting its own non-tradeable "filtered_out" rung.
"""
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.alert import Alert
from form4lab.models.broker import BrokerOrder, BrokerPosition


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_full_signal_to_order_flow(db):
    """End-to-end: create a cluster_buy alert -> execute_signal -> BrokerOrder + BrokerPosition created."""
    insider = Insider(cik="integ-001", name="Integration Test CEO")
    company = Company(cik="integ-002", ticker="INTG", name="Integration Corp")
    db.add_all([insider, company])
    db.flush()

    role = InsiderRole(
        insider_id=insider.id, company_id=company.id,
        role_title="CEO", is_officer=True, is_director=False,
        is_ten_percent_owner=False,
    )
    db.add(role)

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="0001-integ-test", filing_date=date(2026, 2, 21),
        transaction_date=date(2026, 2, 20), transaction_code="P",
        shares=500, price_per_share=50.0, total_value=25000.0,
        shares_owned_after=1000, acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.flush()

    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type="cluster_buy", conviction_score=3.0,
        insider_skill_score=1.5, transaction_value=25000.0,
        summary="Integration test cluster buy", trade_date=date(2026, 2, 20),
    )
    db.add(alert)
    db.commit()

    # Mock Alpaca client
    mock_account = MagicMock()
    mock_account.equity = "100000.00"
    mock_account.cash = "100000.00"
    mock_account.buying_power = "200000.00"
    mock_account.long_market_value = "0"

    mock_order = MagicMock()
    mock_order.id = "mock-order-uuid-123"
    mock_order.status = "accepted"

    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client.submit_order.return_value = mock_order

    # Mock data client for last close price
    mock_data_client = MagicMock()
    mock_bar = MagicMock()
    mock_bar.close = 50.0
    mock_data_client.get_stock_latest_bar.return_value = {"INTG": mock_bar}

    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg, \
         patch("form4lab.services.alpaca_service._get_trading_client", return_value=mock_client), \
         patch("form4lab.services.alpaca_service._get_data_client", return_value=mock_data_client):

        mock_cfg.enabled = True
        mock_cfg.paper = True
        mock_cfg.api_key = "test-key"
        mock_cfg.secret_key = "test-secret"
        mock_cfg.hold_days = 60
        mock_cfg.max_positions_per_insider_ticker = 1
        mock_cfg.max_positions_per_ticker = 3
        mock_cfg.drawdown_threshold = None  # disable for integration test
        mock_cfg.margin_multiplier = 1.0
        mock_cfg.vol_targeting_enabled = False
        mock_cfg.vol_targeting_shadow = False
        mock_cfg.vol_target_k = None
        mock_cfg.spy_parking_enabled = False

        from form4lab.services.alpaca_service import execute_signal
        position = execute_signal(alert, db)

    # Verify position was created
    assert position is not None
    assert position.symbol == "INTG"
    assert position.insider_name == "Integration Test CEO"
    assert position.insider_role == "CEO"
    assert position.status == "open"

    # Verify BrokerOrder was created
    orders = db.query(BrokerOrder).all()
    assert len(orders) == 1
    assert orders[0].alpaca_order_id == "mock-order-uuid-123"
    assert orders[0].side == "buy"
    assert orders[0].symbol == "INTG"

    # Verify position sizing: shipped default is a flat 5% of $100k = $5k
    assert orders[0].notional == pytest.approx(5000.0)
    assert orders[0].sizing_method == "role"


def test_director_role_recorded_with_flat_sizing(db):
    """A Director's alert still flows role_title through to the persisted
    position correctly, and the shipped flat-5% default sizes it
    identically to a CEO — role does not affect position size."""
    insider = Insider(cik="integ-003", name="Director Bob")
    company = Company(cik="integ-004", ticker="DIRB", name="Director Corp")
    db.add_all([insider, company])
    db.flush()

    role = InsiderRole(
        insider_id=insider.id, company_id=company.id,
        role_title="Director", is_officer=False, is_director=True,
        is_ten_percent_owner=False,
    )
    db.add(role)

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="0001-integ-dir", filing_date=date(2026, 2, 21),
        transaction_date=date(2026, 2, 20), transaction_code="P",
        shares=200, price_per_share=75.0, total_value=15000.0,
        shares_owned_after=500, acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.flush()

    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type="cluster_buy", conviction_score=2.0,
        insider_skill_score=0.8, transaction_value=15000.0,
        summary="Director cluster buy", trade_date=date(2026, 2, 20),
    )
    db.add(alert)
    db.commit()

    mock_account = MagicMock()
    mock_account.equity = "100000.00"
    mock_account.cash = "100000.00"
    mock_account.buying_power = "200000.00"
    mock_account.long_market_value = "0"

    mock_order = MagicMock()
    mock_order.id = "mock-order-dir-456"

    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client.submit_order.return_value = mock_order

    mock_data_client = MagicMock()
    mock_bar = MagicMock()
    mock_bar.close = 75.0
    mock_data_client.get_stock_latest_bar.return_value = {"DIRB": mock_bar}

    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg, \
         patch("form4lab.services.alpaca_service._get_trading_client", return_value=mock_client), \
         patch("form4lab.services.alpaca_service._get_data_client", return_value=mock_data_client):

        mock_cfg.enabled = True
        mock_cfg.paper = True
        mock_cfg.api_key = "test-key"
        mock_cfg.secret_key = "test-secret"
        mock_cfg.hold_days = 60
        mock_cfg.max_positions_per_insider_ticker = 1
        mock_cfg.max_positions_per_ticker = 3
        mock_cfg.drawdown_threshold = None
        mock_cfg.margin_multiplier = 1.0
        mock_cfg.vol_targeting_enabled = False
        mock_cfg.vol_targeting_shadow = False
        mock_cfg.vol_target_k = None
        mock_cfg.spy_parking_enabled = False

        from form4lab.services.alpaca_service import execute_signal
        position = execute_signal(alert, db)

    assert position is not None
    assert position.insider_role == "Director"
    orders = db.query(BrokerOrder).all()
    assert len(orders) == 1
    # Flat 5% of $100k = $5k — identical to the CEO case above.
    assert orders[0].notional == pytest.approx(5000.0)


def test_non_tradeable_signal_not_executed(db):
    """filtered_out (ClusterBuyStrategy's hidden, non-tradeable rung) must
    never trigger Alpaca execution."""
    insider = Insider(cik="integ-005", name="Filtered Insider")
    company = Company(cik="integ-006", ticker="FLTR", name="Filtered Corp")
    db.add_all([insider, company])
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="0001-integ-fltr", filing_date=date(2026, 2, 21),
        transaction_date=date(2026, 2, 20), transaction_code="P",
        shares=100, price_per_share=50.0, total_value=5000.0,
        shares_owned_after=200, acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.flush()

    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type="filtered_out", conviction_score=1.5,
        insider_skill_score=0.5, transaction_value=5000.0,
        summary="Filtered out", trade_date=date(2026, 2, 20),
    )
    db.add(alert)
    db.commit()

    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = True
        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

    assert result is None
    assert db.query(BrokerOrder).count() == 0
