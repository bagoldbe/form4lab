"""Tests for form4lab.services.alpaca_service: order timing, exits, 52-week
drawdown computation, SPY parking, reconciliation, and order sync.

Strategy.size() ships as a flat percentage that ignores role (see
form4lab/strategy/base.py). _make_alert's default alert_type is "cluster_buy"
(the shipped default's only tradeable rung) so execute_signal's
registry.is_tradeable() check passes.
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


def _make_alert(db, role_title="CEO", alert_type="cluster_buy"):
    """Helper to create a fully-linked alert for testing."""
    insider = Insider(cik="test-001", name="Jane Smith")
    company = Company(cik="test-002", ticker="AAPL", name="Apple Inc")
    db.add_all([insider, company])
    db.flush()

    role = InsiderRole(
        insider_id=insider.id, company_id=company.id,
        role_title=role_title, is_officer=True, is_director=False,
        is_ten_percent_owner=False,
    )
    db.add(role)
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="0001-test", filing_date=date(2026, 2, 21),
        transaction_date=date(2026, 2, 20), transaction_code="P",
        shares=100, price_per_share=150.0, total_value=15000.0,
        shares_owned_after=200, acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.flush()

    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type=alert_type, conviction_score=2.5,
        insider_skill_score=1.2, transaction_value=15000.0,
        summary="Test buy", trade_date=date(2026, 2, 20),
    )
    db.add(alert)
    db.commit()
    return alert


def test_execute_signal_disabled(db):
    """Should return None when Alpaca is disabled."""
    from form4lab.services.alpaca_service import execute_signal

    alert = _make_alert(db)
    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = False
        result = execute_signal(alert, db)
    assert result is None


def test_execute_signal_wrong_type(db):
    """Should return None for non-tradeable alert types."""
    from form4lab.services.alpaca_service import execute_signal

    alert = _make_alert(db, alert_type="cluster_buy")
    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = True
        result = execute_signal(alert, db)
    assert result is None


def test_execute_signal_concentration_limit(db):
    """Should skip when insider+ticker already has an open position."""
    from form4lab.services.alpaca_service import execute_signal

    alert = _make_alert(db)
    # Create existing open position for same insider+ticker
    order = BrokerOrder(
        alert_id=alert.id, alpaca_order_id="existing-1", symbol="AAPL",
        side="buy", qty=10, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=alert.id, entry_order_id=order.id, symbol="AAPL",
        shares=10, entry_price=150.0, entry_date=date(2026, 2, 1),
        exit_target_date=date(2026, 4, 25), status="open",
        insider_name="Jane Smith", insider_role="CEO",
    )
    db.add(pos)
    db.commit()

    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = True
        mock_cfg.max_positions_per_insider_ticker = 1
        mock_cfg.max_positions_per_ticker = 3
        result = execute_signal(alert, db)
    assert result is None


def test_determine_order_params_market_hours():
    """During RTH: market order, not queued, no extended hours."""
    from form4lab.services.alpaca_service import _determine_order_params

    # 10:30 AM ET on a weekday
    market_time = datetime(2026, 2, 23, 10, 30)  # Monday
    params = _determine_order_params(market_time)
    assert params["order_type"] == "market"
    assert params["extended_hours"] is False
    assert params["queued"] is False


def test_determine_order_params_after_hours():
    """After hours: still a market order, but queued for the next open."""
    from form4lab.services.alpaca_service import _determine_order_params

    after_hours = datetime(2026, 2, 23, 17, 30)  # 5:30 PM ET
    params = _determine_order_params(after_hours)
    assert params["order_type"] == "market"
    assert params["extended_hours"] is False
    assert params["queued"] is True


def test_determine_order_params_premarket():
    """Pre-market: queued market order for the regular open."""
    from form4lab.services.alpaca_service import _determine_order_params

    pre_market = datetime(2026, 2, 23, 8, 0)  # 8:00 AM ET
    params = _determine_order_params(pre_market)
    assert params["order_type"] == "market"
    assert params["extended_hours"] is False
    assert params["queued"] is True


def test_determine_order_params_overnight():
    """Overnight (post-extended): queued market order."""
    from form4lab.services.alpaca_service import _determine_order_params

    overnight = datetime(2026, 2, 23, 23, 0)  # 11 PM ET
    params = _determine_order_params(overnight)
    assert params["order_type"] == "market"
    assert params["extended_hours"] is False
    assert params["queued"] is True


def test_determine_order_params_market_close_boundary():
    """Exactly at 4:00 PM ET counts as out-of-session and queues."""
    from form4lab.services.alpaca_service import _determine_order_params

    close = datetime(2026, 2, 23, 16, 0)  # exact 4:00 PM ET
    params = _determine_order_params(close)
    assert params["queued"] is True
    open_edge = datetime(2026, 2, 23, 9, 30)  # exact 9:30 AM ET
    params = _determine_order_params(open_edge)
    assert params["queued"] is False


def test_calculate_exit_target_date():
    """Should add 60 trading days (skipping weekends)."""
    from form4lab.services.alpaca_service import _calculate_exit_target_date

    # Monday Feb 23, 2026 + 60 trading days
    entry = date(2026, 2, 23)
    target = _calculate_exit_target_date(entry, hold_days=60)
    assert target.weekday() < 5  # must be a weekday
    # 60 trading days from Feb 23 = roughly 12 weeks = ~May 18ish
    assert target.month in (5, 6)


def test_check_exits_finds_expired(db):
    """check_exits should find positions past their exit_target_date."""
    from form4lab.services.alpaca_service import get_positions_to_close

    order = BrokerOrder(
        alert_id=1, alpaca_order_id="old-1", symbol="AAPL",
        side="buy", qty=10, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()

    pos = BrokerPosition(
        alert_id=1, entry_order_id=order.id, symbol="AAPL",
        shares=10, entry_price=150.0, entry_date=date(2025, 12, 1),
        exit_target_date=date(2026, 2, 20), status="open",
        insider_name="Test", insider_role="CEO",
    )
    db.add(pos)
    db.commit()

    expired = get_positions_to_close(db, as_of=date(2026, 2, 21))
    assert len(expired) == 1
    assert expired[0].symbol == "AAPL"


def test_check_exits_skips_future(db):
    """check_exits should not close positions before their exit date."""
    from form4lab.services.alpaca_service import get_positions_to_close

    order = BrokerOrder(
        alert_id=1, alpaca_order_id="future-1", symbol="MSFT",
        side="buy", qty=5, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()

    pos = BrokerPosition(
        alert_id=1, entry_order_id=order.id, symbol="MSFT",
        shares=5, entry_price=400.0, entry_date=date(2026, 2, 1),
        exit_target_date=date(2026, 5, 15), status="open",
        insider_name="Test", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    expired = get_positions_to_close(db, as_of=date(2026, 2, 21))
    assert len(expired) == 0


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_execute_signal_uses_latest_bar_for_price(mock_cfg, mock_client_fn, db):
    """execute_signal should use StockLatestBarRequest (not snapshot) for last close."""
    alert = _make_alert(db, role_title="CEO")

    mock_cfg.enabled = True
    mock_cfg.max_positions_per_insider_ticker = 1
    mock_cfg.max_positions_per_ticker = 3
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.hold_days = 60
    mock_cfg.drawdown_threshold = None  # disable drawdown filter for this test
    mock_cfg.margin_multiplier = 1.0  # cash only for this test

    # Mock Alpaca trading client
    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "100000"
    mock_account.buying_power = "200000"
    mock_account.long_market_value = "0"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_order = MagicMock()
    mock_order.id = "test-order-id"
    mock_client.submit_order.return_value = mock_order
    mock_client_fn.return_value = mock_client

    # Mock the data client to verify correct request type
    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn:
        mock_data_client = MagicMock()
        mock_bar = MagicMock()
        mock_bar.close = 160.0
        mock_data_client.get_stock_latest_bar.return_value = {"AAPL": mock_bar}
        mock_data_fn.return_value = mock_data_client

        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

        # Verify get_stock_latest_bar was called (not get_stock_snapshot)
        mock_data_client.get_stock_latest_bar.assert_called_once()

        # Verify the request used StockLatestBarRequest
        call_args = mock_data_client.get_stock_latest_bar.call_args
        from alpaca.data.requests import StockLatestBarRequest
        assert isinstance(call_args[0][0], StockLatestBarRequest)


def test_get_52week_drawdown_computation():
    """Should compute drawdown from 52-week high correctly."""
    from form4lab.services.alpaca_service import _get_52week_drawdown

    mock_data_client = MagicMock()
    mock_bars = []
    for i in range(100):
        bar = MagicMock()
        bar.close = 200.0 if i == 50 else 150.0
        mock_bars.append(bar)

    mock_data_client.get_stock_bars.return_value = {"AAPL": mock_bars}

    # Last close is $160, highest close was $200 -> drawdown = (160-200)/200 = -0.20
    dd = _get_52week_drawdown(mock_data_client, "AAPL", 160.0)
    assert dd is not None
    assert abs(dd - (-0.20)) < 0.01


def test_get_52week_drawdown_api_error_no_db():
    """Without db (no fallback), API error returns None (fail-closed)."""
    from form4lab.services.alpaca_service import _get_52week_drawdown

    mock_data_client = MagicMock()
    mock_data_client.get_stock_bars.side_effect = Exception("API down")

    dd = _get_52week_drawdown(mock_data_client, "AAPL", 160.0)
    assert dd is None


def test_get_52week_drawdown_falls_back_to_yfinance():
    """When Alpaca returns < 20 bars, fall back to yfinance via the cached store."""
    import pandas as pd
    from form4lab.services.alpaca_service import _get_52week_drawdown

    mock_data_client = MagicMock()
    mock_data_client.get_stock_bars.return_value = {"AAPL": []}

    yf_df = pd.DataFrame({"close": [200.0] + [150.0] * 50})
    with patch("form4lab.data.price_fetcher.YFinanceProvider") as MockProvider:
        provider = MockProvider.return_value
        provider.get_daily_prices.return_value = yf_df
        dd = _get_52week_drawdown(mock_data_client, "AAPL", 160.0, db=MagicMock())

    assert dd is not None
    assert abs(dd - (-0.20)) < 0.01


def test_get_52week_drawdown_both_sources_fail():
    """When Alpaca + yfinance both come back empty, return None (fail-closed)."""
    import pandas as pd
    from form4lab.services.alpaca_service import _get_52week_drawdown

    mock_data_client = MagicMock()
    mock_data_client.get_stock_bars.side_effect = Exception("API down")

    with patch("form4lab.data.price_fetcher.YFinanceProvider") as MockProvider:
        provider = MockProvider.return_value
        provider.get_daily_prices.return_value = pd.DataFrame({"close": []})
        dd = _get_52week_drawdown(mock_data_client, "AAPL", 160.0, db=MagicMock())

    assert dd is None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_execute_signal_skips_near_high(mock_cfg, mock_client_fn, db):
    """execute_signal should skip when stock is near its 52-week high."""
    alert = _make_alert(db, role_title="CEO")

    mock_cfg.enabled = True
    mock_cfg.max_positions_per_insider_ticker = 1
    mock_cfg.max_positions_per_ticker = 3
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.hold_days = 60
    mock_cfg.drawdown_threshold = -0.30
    mock_cfg.margin_multiplier = 1.0

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "100000"
    mock_account.buying_power = "100000"
    mock_account.long_market_value = "0"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn, \
         patch("form4lab.services.alpaca_service._get_52week_drawdown") as mock_dd:
        mock_data_client = MagicMock()
        mock_bar = MagicMock()
        mock_bar.close = 195.0  # near the high
        mock_data_client.get_stock_latest_bar.return_value = {"AAPL": mock_bar}
        mock_data_fn.return_value = mock_data_client

        # Stock is only 2.5% below high — should be filtered out
        mock_dd.return_value = -0.025

        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

    assert result is None  # filtered by drawdown


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_execute_signal_skips_when_no_last_close(mock_cfg, mock_client_fn, db):
    """Fail-closed: skip trade when last close price is unavailable."""
    alert = _make_alert(db, role_title="CEO")

    mock_cfg.enabled = True
    mock_cfg.max_positions_per_insider_ticker = 1
    mock_cfg.max_positions_per_ticker = 3
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.hold_days = 60
    mock_cfg.drawdown_threshold = -0.30
    mock_cfg.margin_multiplier = 1.0

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "100000"
    mock_account.buying_power = "100000"
    mock_account.long_market_value = "0"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn:
        mock_data_client = MagicMock()
        # Simulate API failure — get_stock_latest_bar raises
        mock_data_client.get_stock_latest_bar.side_effect = Exception("API down")
        mock_data_fn.return_value = mock_data_client

        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

    assert result is None
    mock_client.submit_order.assert_not_called()


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_execute_signal_skips_when_drawdown_unavailable(mock_cfg, mock_client_fn, db):
    """Fail-closed: skip trade when 52-week drawdown cannot be computed."""
    alert = _make_alert(db, role_title="CEO")

    mock_cfg.enabled = True
    mock_cfg.max_positions_per_insider_ticker = 1
    mock_cfg.max_positions_per_ticker = 3
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.hold_days = 60
    mock_cfg.drawdown_threshold = -0.30
    mock_cfg.margin_multiplier = 1.0

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "100000"
    mock_account.buying_power = "100000"
    mock_account.long_market_value = "0"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn, \
         patch("form4lab.services.alpaca_service._get_52week_drawdown") as mock_dd, \
         patch("form4lab.services.alpaca_service._sell_spy_for_signal") as mock_spy_sell:
        mock_data_client = MagicMock()
        mock_bar = MagicMock()
        mock_bar.close = 150.0
        mock_data_client.get_stock_latest_bar.return_value = {"AAPL": mock_bar}
        mock_data_fn.return_value = mock_data_client

        # Drawdown computation fails — returns None (insufficient bars)
        mock_dd.return_value = None

        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

    assert result is None
    mock_client.submit_order.assert_not_called()
    # SPY should NOT be sold before drawdown rejection
    mock_spy_sell.assert_not_called()


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_execute_signal_margin_buying_power(mock_cfg, mock_client_fn, db):
    """With margin_multiplier > 1.0, should use margin-aware buying power."""
    alert = _make_alert(db, role_title="Director")

    mock_cfg.enabled = True
    mock_cfg.max_positions_per_insider_ticker = 1
    mock_cfg.max_positions_per_ticker = 3
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.hold_days = 60
    mock_cfg.drawdown_threshold = None  # disable drawdown filter
    mock_cfg.margin_multiplier = 1.5  # enable margin

    # Low cash but high equity via positions — margin should allow the trade
    mock_account = MagicMock()
    mock_account.equity = "50000"
    mock_account.cash = "1000"  # only $1K cash
    mock_account.buying_power = "40000"  # broker says $40K available
    mock_account.long_market_value = "49000"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_order = MagicMock()
    mock_order.id = "margin-order-id"
    mock_client.submit_order.return_value = mock_order
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn:
        mock_data_client = MagicMock()
        mock_bar = MagicMock()
        mock_bar.close = 150.0
        mock_data_client.get_stock_latest_bar.return_value = {"AAPL": mock_bar}
        mock_data_fn.return_value = mock_data_client

        from form4lab.services.alpaca_service import execute_signal
        result = execute_signal(alert, db)

    # Position size = 6% of $50K equity = $3K
    # Available = min(40000, 50000*1.5 - 49000) = min(40000, 26000) = 26000
    # $3K < $26K so trade should proceed
    assert result is not None
    assert result.symbol == "AAPL"


def test_execute_signal_skips_duplicate(db):
    """execute_signal should skip alerts that already have a BrokerPosition."""
    from form4lab.services.alpaca_service import execute_signal

    alert = _make_alert(db)

    # Create existing position for this alert (simulating prior execution)
    order = BrokerOrder(
        alert_id=alert.id, alpaca_order_id="dup-guard-1", symbol="AAPL",
        side="buy", qty=10, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=alert.id, entry_order_id=order.id, symbol="AAPL",
        shares=10, entry_price=150.0, entry_date=date(2026, 2, 1),
        exit_target_date=date(2026, 4, 25), status="open",
        insider_name="Jane Smith", insider_role="CEO",
    )
    db.add(pos)
    db.commit()

    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = True
        result = execute_signal(alert, db)

    # Should be skipped because position already exists for this alert
    assert result is None


@patch("form4lab.services.alpaca_service._get_trading_client")
def test_get_spy_position_returns_shares_and_value(mock_client_fn):
    """Should return (shares, market_value) for SPY position."""
    from form4lab.services.alpaca_service import _get_spy_position

    mock_position = MagicMock()
    mock_position.qty = "25.5"
    mock_position.market_value = "12750.00"
    mock_client = MagicMock()
    mock_client.get_open_position.return_value = mock_position
    mock_client_fn.return_value = mock_client

    shares, value = _get_spy_position(mock_client)
    assert shares == pytest.approx(25.5)
    assert value == pytest.approx(12750.0)


@patch("form4lab.services.alpaca_service._get_trading_client")
def test_get_spy_position_returns_zero_when_no_position(mock_client_fn):
    """Should return (0, 0) when no SPY position exists."""
    from form4lab.services.alpaca_service import _get_spy_position

    mock_client = MagicMock()
    mock_client.get_open_position.side_effect = Exception("position does not exist")
    mock_client_fn.return_value = mock_client

    shares, value = _get_spy_position(mock_client)
    assert shares == 0.0
    assert value == 0.0


def test_broker_position_alert_id_unique_constraint(db):
    """DB should reject two BrokerPositions with the same alert_id."""
    from sqlalchemy.exc import IntegrityError

    order = BrokerOrder(
        alert_id=1, alpaca_order_id="uniq-1", symbol="AAPL",
        side="buy", qty=10, order_type="market",
        extended_hours=False, status="filled",
    )
    db.add(order)
    db.flush()

    pos1 = BrokerPosition(
        alert_id=1, entry_order_id=order.id, symbol="AAPL",
        shares=10, entry_price=150.0, entry_date=date(2026, 2, 1),
        exit_target_date=date(2026, 4, 25), status="open",
        insider_name="Test", insider_role="CEO",
    )
    db.add(pos1)
    db.flush()

    pos2 = BrokerPosition(
        alert_id=1, entry_order_id=order.id, symbol="AAPL",
        shares=10, entry_price=150.0, entry_date=date(2026, 2, 1),
        exit_target_date=date(2026, 4, 25), status="open",
        insider_name="Test", insider_role="CEO",
    )
    db.add(pos2)
    with pytest.raises(IntegrityError):
        db.flush()


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_rebalance_spy_buys_when_excess_cash(mock_cfg, mock_client_fn, mock_spy_pos):
    """Should buy SPY when cash exceeds target buffer."""
    from form4lab.services.alpaca_service import rebalance_spy_parking

    mock_cfg.enabled = True
    mock_cfg.spy_parking_enabled = True
    mock_cfg.spy_parking_buffer = 0.20

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "80000"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (0.0, 0.0)

    rebalance_spy_parking()

    mock_client.submit_order.assert_called_once()
    order_arg = mock_client.submit_order.call_args[1]["order_data"]
    assert order_arg.notional == pytest.approx(60000.0)


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_rebalance_spy_sells_when_cash_below_buffer(mock_cfg, mock_client_fn, mock_spy_pos):
    """Should sell SPY when cash is below target buffer."""
    from form4lab.services.alpaca_service import rebalance_spy_parking

    mock_cfg.enabled = True
    mock_cfg.spy_parking_enabled = True
    mock_cfg.spy_parking_buffer = 0.20

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "5000"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (200.0, 95000.0)

    rebalance_spy_parking()

    mock_client.submit_order.assert_called_once()
    order_arg = mock_client.submit_order.call_args[1]["order_data"]
    assert order_arg.notional == pytest.approx(15000.0)


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_rebalance_spy_noop_when_near_buffer(mock_cfg, mock_client_fn, mock_spy_pos):
    """Should not trade when cash is within $100 of target."""
    from form4lab.services.alpaca_service import rebalance_spy_parking

    mock_cfg.enabled = True
    mock_cfg.spy_parking_enabled = True
    mock_cfg.spy_parking_buffer = 0.20

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "20050"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (160.0, 79950.0)

    rebalance_spy_parking()

    mock_client.submit_order.assert_not_called()


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_rebalance_spy_disabled(mock_cfg, mock_client_fn):
    """Should do nothing when spy_parking_enabled is False."""
    from form4lab.services.alpaca_service import rebalance_spy_parking

    mock_cfg.enabled = True
    mock_cfg.spy_parking_enabled = False

    rebalance_spy_parking()

    mock_client_fn.assert_not_called()


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
def test_sell_spy_for_signal_sells_shortfall(mock_client_fn, mock_spy_pos):
    """Should sell exactly the shortfall amount of SPY."""
    from form4lab.services.alpaca_service import _sell_spy_for_signal

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (100.0, 50000.0)

    result = _sell_spy_for_signal(mock_client, available=2000.0, position_size=8000.0)

    assert result is True
    mock_client.submit_order.assert_called_once()
    order_arg = mock_client.submit_order.call_args[1]["order_data"]
    assert order_arg.notional == pytest.approx(6000.0)


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
def test_sell_spy_for_signal_caps_at_holdings(mock_client_fn, mock_spy_pos):
    """Should sell at most the SPY market value."""
    from form4lab.services.alpaca_service import _sell_spy_for_signal

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (5.0, 2500.0)

    result = _sell_spy_for_signal(mock_client, available=1000.0, position_size=10000.0)

    assert result is True
    order_arg = mock_client.submit_order.call_args[1]["order_data"]
    assert order_arg.notional == pytest.approx(2500.0)


@patch("form4lab.services.alpaca_service._get_spy_position")
def test_sell_spy_for_signal_noop_when_no_spy(mock_spy_pos):
    """Should return False when no SPY is held."""
    from form4lab.services.alpaca_service import _sell_spy_for_signal

    mock_client = MagicMock()
    mock_spy_pos.return_value = (0.0, 0.0)

    result = _sell_spy_for_signal(mock_client, available=1000.0, position_size=5000.0)

    assert result is False
    mock_client.submit_order.assert_not_called()


@patch("form4lab.services.alpaca_service._get_spy_position")
@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_portfolio_summary_includes_spy_parking(mock_cfg, mock_client_fn, mock_spy_pos, db):
    """get_portfolio_summary should include spy_parking when enabled."""
    from form4lab.services.alpaca_service import get_portfolio_summary

    mock_cfg.enabled = True
    mock_cfg.api_key = "test"
    mock_cfg.spy_parking_enabled = True
    mock_cfg.spy_parking_buffer = 0.20

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "20000"
    mock_account.buying_power = "200000"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client
    mock_spy_pos.return_value = (160.0, 80000.0)

    result = get_portfolio_summary(db)

    assert result["account"] is not None
    assert "spy_parking" in result["account"]
    spy = result["account"]["spy_parking"]
    assert spy["shares"] == pytest.approx(160.0)
    assert spy["market_value"] == pytest.approx(80000.0)
    assert spy["buffer_pct"] == pytest.approx(0.20)


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_portfolio_summary_no_spy_when_disabled(mock_cfg, mock_client_fn, db):
    """get_portfolio_summary should omit spy_parking when disabled."""
    from form4lab.services.alpaca_service import get_portfolio_summary

    mock_cfg.enabled = True
    mock_cfg.api_key = "test"
    mock_cfg.spy_parking_enabled = False

    mock_account = MagicMock()
    mock_account.equity = "100000"
    mock_account.cash = "20000"
    mock_account.buying_power = "200000"
    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client_fn.return_value = mock_client

    result = get_portfolio_summary(db)

    assert result["account"] is not None
    assert "spy_parking" not in result["account"]


# ---------------------------------------------------------------------------
# reconcile_positions tests
# ---------------------------------------------------------------------------


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_closes_orphan(mock_cfg, mock_client_fn, db):
    """Reconcile should close DB positions when Alpaca position is gone."""
    order = BrokerOrder(
        alert_id=1, alpaca_order_id="test-order-recon-1",
        symbol="ZZG", side="buy", order_type="market",
        status="filled", filled_qty=78.0, filled_avg_price=73.87,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=1, entry_order_id=order.id, symbol="ZZG",
        shares=78.0, entry_price=73.87, entry_date=date(2026, 3, 12),
        exit_target_date=date(2026, 6, 5), status="open",
        insider_name="Doe Jane Q", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []
    mock_client.get_orders.return_value = []
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn, \
         patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="active"):
        mock_data_client = MagicMock()
        mock_bar = MagicMock()
        mock_bar.close = 72.50
        mock_data_client.get_stock_latest_bar.return_value = {"ZZG": mock_bar}
        mock_data_fn.return_value = mock_data_client

        from form4lab.services.alpaca_service import reconcile_positions
        count = reconcile_positions(db)

    assert count == 1
    db.refresh(pos)
    assert pos.status == "closed"
    assert pos.exit_price == 72.50
    assert pos.exit_date == date.today()
    assert pos.pnl is not None
    assert pos.pnl_pct is not None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_uses_alpaca_sell_price(mock_cfg, mock_client_fn, db):
    """Reconcile should use Alpaca sell order fill price when available."""
    order = BrokerOrder(
        alert_id=2, alpaca_order_id="test-order-recon-2",
        symbol="AAPL", side="buy", order_type="market",
        status="filled", filled_qty=10.0, filled_avg_price=150.0,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=2, entry_order_id=order.id, symbol="AAPL",
        shares=10.0, entry_price=150.0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 30), status="open",
        insider_name="Test Insider", insider_role="CEO",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []

    mock_sell_order = MagicMock()
    mock_sell_order.filled_avg_price = "155.50"
    mock_sell_order.filled_at = datetime(2026, 3, 13, 15, 30)
    mock_client.get_orders.return_value = [mock_sell_order]
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client"):
        from form4lab.services.alpaca_service import reconcile_positions
        count = reconcile_positions(db)

    assert count == 1
    db.refresh(pos)
    assert pos.status == "closed"
    assert pos.exit_price == 155.50
    assert pos.pnl == pytest.approx((155.50 - 150.0) * 10.0)


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_skips_spy(mock_cfg, mock_client_fn, db):
    """Reconcile should never touch SPY (parking position is untracked)."""
    order = BrokerOrder(
        alert_id=3, alpaca_order_id="test-order-recon-spy",
        symbol="SPY", side="buy", order_type="market",
        status="filled", filled_qty=5.0, filled_avg_price=500.0,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=3, entry_order_id=order.id, symbol="SPY",
        shares=5.0, entry_price=500.0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 30), status="open",
        insider_name="Test", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import reconcile_positions
    count = reconcile_positions(db)

    assert count == 0
    db.refresh(pos)
    assert pos.status == "open"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_no_orphans(mock_cfg, mock_client_fn, db):
    """Reconcile should do nothing when all DB positions exist in Alpaca."""
    order = BrokerOrder(
        alert_id=4, alpaca_order_id="test-order-recon-4",
        symbol="ZZH", side="buy", order_type="market",
        status="filled", filled_qty=50.0, filled_avg_price=92.0,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=4, entry_order_id=order.id, symbol="ZZH",
        shares=50.0, entry_price=92.0, entry_date=date(2026, 3, 12),
        exit_target_date=date(2026, 6, 5), status="open",
        insider_name="Test", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_alpaca_pos = MagicMock()
    mock_alpaca_pos.symbol = "ZZH"
    mock_alpaca_pos.qty = "50"
    mock_client.get_all_positions.return_value = [mock_alpaca_pos]
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import reconcile_positions
    count = reconcile_positions(db)

    assert count == 0
    db.refresh(pos)
    assert pos.status == "open"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_warns_on_partial_mismatch(mock_cfg, mock_client_fn, db):
    """Partial qty mismatch should log warning but not auto-close."""
    for aid in [5, 6]:
        order = BrokerOrder(
            alert_id=aid, alpaca_order_id=f"test-order-recon-{aid}",
            symbol="AAPL", side="buy", order_type="market",
            status="filled", filled_qty=50.0, filled_avg_price=150.0,
        )
        db.add(order)
        db.flush()
        db.add(BrokerPosition(
            alert_id=aid, entry_order_id=order.id, symbol="AAPL",
            shares=50.0, entry_price=150.0, entry_date=date(2026, 3, 1),
            exit_target_date=date(2026, 5, 30), status="open",
            insider_name="Test", insider_role="CEO",
        ))
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_alpaca_pos = MagicMock()
    mock_alpaca_pos.symbol = "AAPL"
    mock_alpaca_pos.qty = "50"
    mock_client.get_all_positions.return_value = [mock_alpaca_pos]
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import reconcile_positions
    count = reconcile_positions(db)

    assert count == 0
    positions = db.query(BrokerPosition).filter(
        BrokerPosition.symbol == "AAPL", BrokerPosition.status == "open"
    ).all()
    assert len(positions) == 2


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_no_exit_price(mock_cfg, mock_client_fn, db):
    """Position should still close even when exit price is unavailable."""
    order = BrokerOrder(
        alert_id=7, alpaca_order_id="test-order-recon-7",
        symbol="XYZ", side="buy", order_type="market",
        status="filled", filled_qty=20.0, filled_avg_price=50.0,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=7, entry_order_id=order.id, symbol="XYZ",
        shares=20.0, entry_price=50.0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 30), status="open",
        insider_name="Test", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []
    mock_client.get_orders.side_effect = Exception("API down")
    mock_client_fn.return_value = mock_client

    with patch("form4lab.services.alpaca_service._get_data_client") as mock_data_fn, \
         patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="active"):
        mock_data_client = MagicMock()
        mock_data_client.get_stock_latest_bar.side_effect = Exception("API down")
        mock_data_fn.return_value = mock_data_client

        from form4lab.services.alpaca_service import reconcile_positions
        count = reconcile_positions(db)

    assert count == 1
    db.refresh(pos)
    assert pos.status == "closed"
    assert pos.exit_price is None
    assert pos.pnl is None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_api_failure(mock_cfg, mock_client_fn, db):
    """Should return 0 and leave positions untouched on Alpaca API failure."""
    order = BrokerOrder(
        alert_id=8, alpaca_order_id="test-order-recon-8",
        symbol="FAIL", side="buy", order_type="market",
        status="filled", filled_qty=10.0, filled_avg_price=100.0,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=8, entry_order_id=order.id, symbol="FAIL",
        shares=10.0, entry_price=100.0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 30), status="open",
        insider_name="Test", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"

    mock_client = MagicMock()
    mock_client.get_all_positions.side_effect = Exception("Alpaca down")
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import reconcile_positions
    count = reconcile_positions(db)

    assert count == 0
    db.refresh(pos)
    assert pos.status == "open"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_positions_skips_unfilled_orders(mock_cfg, mock_client_fn, db):
    """A position with shares=0 is a pending fill, not an orphan.

    Regression guard: manually inserted BrokerPositions (created before
    market open so sync_orders could pick up the fills) were falsely closed
    by reconcile_positions because Alpaca showed 0 shares for the symbol
    while the buy was still queued at the broker.
    """
    order = BrokerOrder(
        alert_id=200, alpaca_order_id="pending-fill-1",
        symbol="ZZF", side="buy", notional=6240.0, order_type="market",
        extended_hours=False, status="accepted",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=200, entry_order_id=order.id, symbol="ZZF",
        shares=0, entry_price=0, entry_date=date(2026, 5, 20),
        exit_target_date=date(2026, 8, 12), status="open",
        insider_name="Test Insider", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True
    mock_cfg.api_key = "test-key"
    mock_cfg.secret_key = "test-secret"
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []  # not filled yet
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import reconcile_positions
    count = reconcile_positions(db)

    assert count == 0
    db.refresh(pos)
    assert pos.status == "open"
    assert pos.exit_price is None
    assert pos.exit_date is None


# ---------------------------------------------------------------------------
# sync_orders tests
# ---------------------------------------------------------------------------

@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_picks_up_new_status(mock_cfg, mock_client_fn, db):
    """sync_orders should track orders with 'new' status (not just 'submitted')."""
    order = BrokerOrder(
        alert_id=100, alpaca_order_id="test-sync-new-1",
        symbol="ZZI", side="buy", notional=5000.0, order_type="limit",
        limit_price=50.0, extended_hours=True, status="new",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=100, entry_order_id=order.id, symbol="ZZI",
        shares=0, entry_price=0, entry_date=date(2026, 3, 5),
        exit_target_date=date(2026, 5, 28), status="open",
        insider_name="Test Insider", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True

    # Alpaca says order is now expired
    mock_alpaca_order = MagicMock()
    mock_alpaca_order.status.value = "expired"
    mock_alpaca_order.filled_qty = None
    mock_alpaca_order.filled_avg_price = None
    mock_alpaca_order.filled_at = None

    mock_client = MagicMock()
    mock_client.get_order_by_id.return_value = mock_alpaca_order
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    updated = sync_orders(db)

    assert updated >= 1
    db.refresh(order)
    db.refresh(pos)
    assert order.status == "expired"
    assert pos.status == "closed"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_fills_update_position(mock_cfg, mock_client_fn, db):
    """When a 'new' order becomes 'filled', position should get shares and price."""
    order = BrokerOrder(
        alert_id=101, alpaca_order_id="test-sync-fill-1",
        symbol="ZZJ", side="buy", notional=5000.0, order_type="limit",
        limit_price=5.50, extended_hours=True, status="new",
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=101, entry_order_id=order.id, symbol="ZZJ",
        shares=0, entry_price=0, entry_date=date(2026, 3, 9),
        exit_target_date=date(2026, 6, 1), status="open",
        insider_name="Test Insider", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True

    mock_alpaca_order = MagicMock()
    mock_alpaca_order.status.value = "filled"
    mock_alpaca_order.filled_qty = 909.09
    mock_alpaca_order.filled_avg_price = 5.50
    mock_alpaca_order.filled_at = datetime(2026, 3, 10, 10, 30)

    mock_client = MagicMock()
    mock_client.get_order_by_id.return_value = mock_alpaca_order
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    updated = sync_orders(db)

    assert updated >= 1
    db.refresh(order)
    db.refresh(pos)
    assert order.status == "filled"
    assert order.filled_qty == 909.09
    assert order.filled_avg_price == 5.50
    assert pos.shares == 909.09
    assert pos.entry_price == 5.50
    assert pos.status == "open"  # still open, hasn't hit exit date


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_cancels_stale_orders(mock_cfg, mock_client_fn, db):
    """Stale unfilled buy orders (older than _STALE_ORDER_DAYS) get cancelled and phantom positions closed."""
    order = BrokerOrder(
        alert_id=102, alpaca_order_id="test-stale-1",
        symbol="ZZK", side="buy", notional=5000.0, order_type="limit",
        limit_price=43.0, extended_hours=True, status="new",
        created_at=datetime(2026, 3, 1, 12, 0, 0),  # >2 days ago
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=102, entry_order_id=order.id, symbol="ZZK",
        shares=0, entry_price=0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 25), status="open",
        insider_name="Test Insider", insider_role="Director",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True

    # First sync: Alpaca still says "new" (no status change)
    mock_alpaca_order = MagicMock()
    mock_alpaca_order.status.value = "new"
    mock_alpaca_order.filled_qty = None
    mock_alpaca_order.filled_avg_price = None
    mock_alpaca_order.filled_at = None

    mock_client = MagicMock()
    mock_client.get_order_by_id.return_value = mock_alpaca_order
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    updated = sync_orders(db)

    # Stale cleanup should have cancelled it
    mock_client.cancel_order_by_id.assert_called_once_with("test-stale-1")
    db.refresh(order)
    db.refresh(pos)
    assert order.status == "canceled"
    assert pos.status == "closed"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_skips_terminal_statuses(mock_cfg, mock_client_fn, db):
    """Orders already in terminal status should not be queried."""
    order = BrokerOrder(
        alert_id=103, alpaca_order_id="test-terminal-1",
        symbol="AAPL", side="buy", notional=5000.0, order_type="market",
        status="filled", filled_qty=33.0, filled_avg_price=150.0,
    )
    db.add(order)
    db.commit()

    mock_cfg.enabled = True
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    updated = sync_orders(db)

    assert updated == 0
    mock_client.get_order_by_id.assert_not_called()


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_stale_cleanup_only_buy_orders(mock_cfg, mock_client_fn, db):
    """Stale order cleanup should only cancel buy orders, not sell orders."""
    sell_order = BrokerOrder(
        alert_id=104, alpaca_order_id="test-stale-sell-1",
        symbol="AAPL", side="sell", qty=10.0, order_type="market",
        status="new",
        created_at=datetime(2026, 3, 1, 12, 0, 0),  # stale
    )
    db.add(sell_order)
    db.commit()

    mock_cfg.enabled = True

    mock_alpaca_order = MagicMock()
    mock_alpaca_order.status.value = "new"
    mock_client = MagicMock()
    mock_client.get_order_by_id.return_value = mock_alpaca_order
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    sync_orders(db)

    # Should NOT cancel stale sell orders
    mock_client.cancel_order_by_id.assert_not_called()


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_sync_orders_stale_cancel_failure_leaves_db_unchanged(mock_cfg, mock_client_fn, db):
    """If cancel fails AND Alpaca still reports a non-terminal status, do not corrupt the DB.

    Regression guard for the original bug where any cancel exception caused the order
    to be marked 'canceled' regardless — silently divesting us of live positions.
    """
    order = BrokerOrder(
        alert_id=105, alpaca_order_id="test-cancel-fail-1",
        symbol="ZZE", side="buy", notional=14000.0, order_type="limit",
        limit_price=6.50, extended_hours=True, status="new",
        created_at=datetime(2026, 3, 1, 12, 0, 0),  # stale
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=105, entry_order_id=order.id, symbol="ZZE",
        shares=0, entry_price=0, entry_date=date(2026, 3, 1),
        exit_target_date=date(2026, 5, 25), status="open",
        insider_name="Test Insider", insider_role="Chief Executive Officer",
    )
    db.add(pos)
    db.commit()

    mock_cfg.enabled = True

    # First-loop poll: no status change
    pending_alpaca = MagicMock()
    pending_alpaca.status.value = "new"
    pending_alpaca.filled_qty = None
    pending_alpaca.filled_avg_price = None
    pending_alpaca.filled_at = None

    # Stale-loop re-query after cancel failure: still "new" (non-terminal)
    requery_alpaca = MagicMock()
    requery_alpaca.status.value = "new"

    mock_client = MagicMock()
    mock_client.get_order_by_id.side_effect = [pending_alpaca, requery_alpaca]
    mock_client.cancel_order_by_id.side_effect = Exception("transient network blip")
    mock_client_fn.return_value = mock_client

    from form4lab.services.alpaca_service import sync_orders
    sync_orders(db)

    mock_client.cancel_order_by_id.assert_called_once_with("test-cancel-fail-1")
    # Order must NOT be marked canceled; position must NOT be closed.
    db.refresh(order)
    db.refresh(pos)
    assert order.status == "new"
    assert pos.status == "open"
