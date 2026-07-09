"""Tests for the volatility-targeted live-sizing MECHANISM in
form4lab.services.alpaca_service — the pieces that ship regardless of which
strategy is active.

The default Strategy.size() (form4lab/strategy/base.py) is a flat 5% that
never reads vol at all, and vol_target_k has no shipped default (None —
see form4lab/config.py's AlpacaConfig). What ships and is generic:
_realized_vol_live/_ticker_exposure_dollars (live data-access helpers) and
the gate in execute_signal that only computes vol/exposure when
vol_target_k is not None — the sizing DECISION itself is fully delegated to
whatever Strategy.size() is active. The tests below exercise exactly that
generic surface, supplying k explicitly per-test (since the shipped default
is None) via a capturing test-double strategy that records what
SizingContext it was given, rather than asserting a specific formula.
"""
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.alert import Alert
from form4lab.models.broker import BrokerOrder, BrokerPosition
from form4lab.models.price import PriceData
from form4lab.services.alpaca_service import (
    _realized_vol_live, _ticker_exposure_dollars, execute_signal,
)
from form4lab.strategy.base import SignalType, SizeDecision, Strategy
from form4lab.scoring.portfolio_simulator import realized_vol, build_price_index


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _make_alert(db, role_title="CEO", ticker="AAPL", alert_type="cap_buy"):
    insider = Insider(cik="t-1", name="Jane")
    company = Company(cik="t-2", ticker=ticker, name="Co")
    db.add_all([insider, company]); db.flush()
    db.add(InsiderRole(insider_id=insider.id, company_id=company.id, role_title=role_title,
                       is_officer=True, is_director=False, is_ten_percent_owner=False)); db.flush()
    txn = Transaction(insider_id=insider.id, company_id=company.id, accession_number="a-1",
                      filing_date=date(2026, 2, 21), transaction_date=date(2026, 2, 20),
                      transaction_code="P", shares=100, price_per_share=150.0, total_value=15000.0,
                      shares_owned_after=200, acquired_or_disposed="A", is_discretionary=True)
    db.add(txn); db.flush()
    alert = Alert(transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
                  alert_type=alert_type, conviction_score=2.5, insider_skill_score=1.2,
                  transaction_value=15000.0, summary="x", trade_date=date(2026, 2, 20))
    db.add(alert); db.commit()
    return alert


def _seed_prices(db, ticker, closes, end=date(2026, 2, 19)):
    """Insert `closes` as consecutive daily prices ending at `end` (oldest first)."""
    rows = []
    for i, c in enumerate(closes):
        d = end - timedelta(days=len(closes) - 1 - i)
        rows.append(PriceData(ticker=ticker, date=d, open=c, high=c, low=c, close=c,
                              adj_close=c, volume=1_000_000))
    db.add_all(rows); db.commit()


def _vol_series(n=30, daily=0.03, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily, n)
    return list(start * np.cumprod(1 + rets))


def _open_position(db, symbol, notional, n=0):
    """Create one OPEN broker position committing `notional` dollars to `symbol`."""
    ins = Insider(cik=f"o-{symbol}-{n}", name="x")
    co = Company(cik=f"c-{symbol}-{n}", ticker=symbol, name=symbol)
    db.add_all([ins, co]); db.flush()
    txn = Transaction(insider_id=ins.id, company_id=co.id, accession_number=f"acc-{symbol}-{n}",
                      filing_date=date(2026, 2, 1), transaction_date=date(2026, 1, 31), transaction_code="P",
                      shares=1, price_per_share=1.0, total_value=1.0, shares_owned_after=1,
                      acquired_or_disposed="A", is_discretionary=True)
    db.add(txn); db.flush()
    al = Alert(transaction_id=txn.id, insider_id=ins.id, company_id=co.id, alert_type="cap_buy",
               conviction_score=1.0, insider_skill_score=1.0, transaction_value=1.0, summary="x",
               trade_date=date(2026, 1, 31))
    db.add(al); db.flush()
    o = BrokerOrder(alert_id=al.id, alpaca_order_id=f"oid-{symbol}-{n}", symbol=symbol, side="buy",
                    notional=notional, order_type="market", status="submitted")
    db.add(o); db.flush()
    db.add(BrokerPosition(alert_id=al.id, entry_order_id=o.id, symbol=symbol, shares=0, entry_price=0,
                          entry_date=date(2026, 2, 1), exit_target_date=date(2026, 5, 1), status="open",
                          insider_name="x", insider_role="Director"))
    db.commit()


def _cfg(**over):
    base = dict(enabled=True, hold_days=60,
                max_positions_per_insider_ticker=9, max_positions_per_ticker=9,
                drawdown_threshold=None, margin_multiplier=1.0, spy_parking_enabled=False,
                stop_loss_pct=None, api_key="k", secret_key="s", paper=True,
                vol_targeting_enabled=False, vol_targeting_shadow=False,
                vol_target_k=None, vol_target_min_pct=0.03, vol_target_max_pct=0.20,
                vol_target_max_ticker_pct=None, vol_target_window=20)
    base.update(over)
    return SimpleNamespace(**base)


# --- 1. live-vol helper matches the backtest realized_vol -------------------
class TestRealizedVolLive:
    def test_matches_backtest_realized_vol(self, db):
        closes = _vol_series(30, daily=0.025, seed=1)
        _seed_prices(db, "AAPL", closes)
        live = _realized_vol_live("AAPL", db, window=20)
        # same data through the backtest helper
        dfrows = db.query(PriceData.date, PriceData.adj_close).filter(PriceData.ticker == "AAPL").all()
        df = pd.DataFrame([{"ticker": "AAPL", "date": d, "adj_close": c} for d, c in dfrows])
        ref = realized_vol(build_price_index(df), "AAPL", date.today(), window=20)
        assert live is not None and abs(live - ref) < 1e-12

    def test_insufficient_history_none(self, db):
        _seed_prices(db, "AAPL", _vol_series(10, seed=2))  # < window+1
        assert _realized_vol_live("AAPL", db, window=20) is None


# --- 2. ticker exposure aggregation ------------------------------------------
class TestTickerExposureDollars:
    def test_exposure_sums_open_positions(self, db):
        _open_position(db, "ZZD", 6500, n=1)
        _open_position(db, "ZZD", 6600, n=2)
        _open_position(db, "AAPL", 9000, n=3)
        assert _ticker_exposure_dollars("ZZD", db) == pytest.approx(13100.0)
        assert _ticker_exposure_dollars("AAPL", db) == pytest.approx(9000.0)
        assert _ticker_exposure_dollars("ZZZ", db) == 0.0


# --- 3. execute_signal's vol/exposure gate — mechanism, not formula ---------
#
# A capturing test-double strategy records the SizingContext execute_signal
# builds for it. This isolates the generic wiring (does the platform compute
# and pass through live vol/exposure under the right flag combination?) from
# any specific sizing formula, which is strategy territory.

class _CapturingSizeStrategy(Strategy):
    name = "capturing"
    captured_ctx = None
    decision = SizeDecision(dollars=12_345.0, method="captured", vol=None, pct=None)

    def signal_types(self):
        return [SignalType("cap_buy", tradeable=True, hold_days=60)]

    def classify(self, txn, f):
        return "cap_buy"

    def size(self, ctx):
        self.captured_ctx = ctx
        return self.decision


def _activate_capturing_strategy(monkeypatch):
    """Swap the active strategy for the capturing double and return the live
    (singleton-cached) instance so the test can inspect what it captured."""
    import form4lab.strategy.registry as reg
    monkeypatch.setattr(reg.settings, "strategy_path", "tests.test_vol_sizing:_CapturingSizeStrategy")
    strategy, _ = reg.get_active(refresh=True)
    return strategy


def _reset_active_strategy():
    import form4lab.strategy.registry as reg
    reg._active = None


def _run_execute(db, cfg, alert, last_close=150.0):
    with patch("form4lab.services.alpaca_service._alpaca_cfg", cfg), \
         patch("form4lab.services.alpaca_service._get_trading_client") as mtc, \
         patch("form4lab.services.alpaca_service._get_data_client") as mdc:
        acct = MagicMock(equity="100000", cash="100000", buying_power="100000", long_market_value="0")
        client = MagicMock(); client.get_account.return_value = acct
        client.submit_order.return_value = MagicMock(id="oid")
        mtc.return_value = client
        dclient = MagicMock()
        dclient.get_stock_latest_bar.return_value = {"AAPL": MagicMock(close=last_close)}
        mdc.return_value = dclient
        execute_signal(alert, db)
    return db.query(BrokerOrder).first()


def test_vol_target_k_none_disables_gate_even_when_enabled(db, monkeypatch):
    """Shipped default is vol_target_k=None: the vol/exposure-computation
    gate must stay off even with vol_targeting_enabled=True, so a strategy
    never sees live vol data it never asked for."""
    strategy = _activate_capturing_strategy(monkeypatch)
    try:
        alert = _make_alert(db)
        _seed_prices(db, "AAPL", _vol_series(30, seed=1))  # prices exist...
        cfg = _cfg(vol_targeting_enabled=True, vol_target_k=None)  # ...but k is None
        _run_execute(db, cfg, alert)
        assert strategy.captured_ctx is not None
        assert strategy.captured_ctx.vol is None
        assert strategy.captured_ctx.ticker_exposure_dollars is None
    finally:
        _reset_active_strategy()


def test_shadow_mode_also_gated_by_k(db, monkeypatch):
    """vol_targeting_shadow alone (enabled=False) must ALSO respect the
    k=None gate — shadow mode is not a backdoor around it."""
    strategy = _activate_capturing_strategy(monkeypatch)
    try:
        alert = _make_alert(db)
        _seed_prices(db, "AAPL", _vol_series(30, seed=1))
        cfg = _cfg(vol_targeting_enabled=False, vol_targeting_shadow=True, vol_target_k=None)
        _run_execute(db, cfg, alert)
        assert strategy.captured_ctx.vol is None
    finally:
        _reset_active_strategy()


def test_vol_target_k_set_computes_and_passes_through_live_vol(db, monkeypatch):
    """With k explicitly supplied and vol_targeting_enabled, execute_signal
    computes live realized vol and passes it through to strategy.size()."""
    strategy = _activate_capturing_strategy(monkeypatch)
    try:
        alert = _make_alert(db)
        closes = _vol_series(30, daily=0.03, seed=4)
        _seed_prices(db, "AAPL", closes)
        cfg = _cfg(vol_targeting_enabled=True, vol_target_k=0.005, vol_target_max_ticker_pct=None)
        expected_vol = _realized_vol_live("AAPL", db, 20)
        _run_execute(db, cfg, alert)
        assert strategy.captured_ctx.vol == pytest.approx(expected_vol)
        assert strategy.captured_ctx.ticker_exposure_dollars is None  # cap unset -> not computed
    finally:
        _reset_active_strategy()


def test_vol_target_max_ticker_pct_set_also_computes_exposure(db, monkeypatch):
    """max_ticker_pct set -> aggregate ticker exposure is computed too, on
    top of vol, and both flow through to the strategy."""
    strategy = _activate_capturing_strategy(monkeypatch)
    try:
        alert = _make_alert(db)
        _open_position(db, "AAPL", 9000, n=1)
        cfg = _cfg(vol_targeting_enabled=True, vol_target_k=0.005, vol_target_max_ticker_pct=0.20)
        _run_execute(db, cfg, alert)
        assert strategy.captured_ctx.ticker_exposure_dollars == pytest.approx(9000.0)
    finally:
        _reset_active_strategy()


def test_broker_order_persists_strategy_sizing_audit_trail(db, monkeypatch):
    """Whatever SizeDecision the active strategy returns is durably recorded
    on the BrokerOrder row, regardless of which strategy produced it."""
    strategy = _activate_capturing_strategy(monkeypatch)
    strategy.decision = SizeDecision(dollars=7_500.0, method="captured_test", vol=0.234, pct=0.075)
    try:
        alert = _make_alert(db)
        order = _run_execute(db, _cfg(), alert)
        assert order is not None
        assert order.notional == pytest.approx(7_500.0)
        assert order.sizing_method == "captured_test"
        assert order.sizing_vol == pytest.approx(0.234)
        assert order.sizing_pct == pytest.approx(0.075)
    finally:
        _reset_active_strategy()


def test_position_too_small_is_skipped(db, monkeypatch):
    """Sub-$100 sizing decisions must not place an order at all (existing
    guard in execute_signal, independent of which strategy produced $0)."""
    strategy = _activate_capturing_strategy(monkeypatch)
    strategy.decision = SizeDecision(dollars=0.0, method="captured_test", vol=None, pct=None)
    try:
        alert = _make_alert(db)
        order = _run_execute(db, _cfg(), alert)
        assert order is None
    finally:
        _reset_active_strategy()
