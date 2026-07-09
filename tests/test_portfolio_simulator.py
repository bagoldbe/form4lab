"""Portfolio simulator tests.

This file has two parts. First, a regression test covering the
non-preloaded run_simulation() flag-computation bug (see
form4lab/scoring/portfolio_simulator.py). Second (below the banner further
down), the mechanical test_portfolio_simulator.py subset — Portfolio/
Position math, price-index helpers, metrics, report/export, margin, SPY
parking, concentration limits, 52-week drawdown.
"""
from datetime import date, timedelta

from form4lab.database import SessionLocal
from form4lab.models.company import Company
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.price import PriceData
from form4lab.scoring.portfolio_simulator import run_simulation

TICKER = "ZZCB"


def _seed_cluster_buy_scenario(db):
    """Two distinct insiders buy the same ticker 3 days apart (inside the
    default 7-day causal cluster window), each >= the $25k cluster_buy value
    floor, plus daily price history covering entry through a full
    60-trading-day hold/exit. Minimal fixture shape: Company, 2 Insiders,
    2 InsiderRoles, 2 Transactions, 2 TradeOutcomes, ~105 days of PriceData
    (the same fixture shape backs the run_simulation regression test
    below)."""
    company = Company(cik="0001800001", ticker=TICKER, name="ZZ Cluster Buy Test Co")
    db.add(company)
    db.flush()

    insider_a = Insider(cik="0001800011", name="Seed Alpha Insider", is_institution=False)
    insider_b = Insider(cik="0001800012", name="Seed Beta Insider", is_institution=False)
    db.add_all([insider_a, insider_b])
    db.flush()

    db.add_all([
        InsiderRole(insider_id=insider_a.id, company_id=company.id,
                    role_title="Chief Executive Officer", is_officer=True,
                    is_director=False, is_ten_percent_owner=False,
                    first_filing_date=date(2020, 1, 1)),
        InsiderRole(insider_id=insider_b.id, company_id=company.id,
                    role_title="Director", is_officer=False, is_director=True,
                    is_ten_percent_owner=False, first_filing_date=date(2020, 1, 1)),
    ])

    txn_date_a = date(2024, 1, 8)   # Monday
    txn_date_b = date(2024, 1, 11)  # +3 days -- inside the 7-day causal cluster window

    txn_a = Transaction(
        insider_id=insider_a.id, company_id=company.id,
        accession_number="0001800001-24-000001",
        filing_date=txn_date_a, transaction_date=txn_date_a,
        transaction_code="P", shares=1_000.0, price_per_share=50.0,
        total_value=50_000.0, shares_owned_after=5_000.0,
        acquired_or_disposed="A", is_discretionary=True, is_common_stock=True,
    )
    txn_b = Transaction(
        insider_id=insider_b.id, company_id=company.id,
        accession_number="0001800001-24-000002",
        filing_date=txn_date_b, transaction_date=txn_date_b,
        transaction_code="P", shares=800.0, price_per_share=55.0,
        total_value=44_000.0, shares_owned_after=800.0,
        acquired_or_disposed="A", is_discretionary=True, is_common_stock=True,
    )
    db.add_all([txn_a, txn_b])
    db.flush()

    db.add_all([
        TradeOutcome(transaction_id=txn_a.id),
        TradeOutcome(transaction_id=txn_b.id),
    ])

    # Daily price history: 5 days before the first buy through 100 days
    # after, comfortably covering entry-price lookup plus a full
    # 60-trading-day hold/exit.
    start = txn_date_a - timedelta(days=5)
    end = txn_date_a + timedelta(days=100)
    prices = []
    d, px = start, 50.0
    while d <= end:
        prices.append(PriceData(
            ticker=TICKER, date=d, open=px, high=px + 1.0, low=px - 1.0,
            close=px, adj_close=px, volume=1_000_000,
        ))
        px += 0.05
        d += timedelta(days=1)
    db.add_all(prices)
    db.commit()

    return {"company": company, "insiders": (insider_a, insider_b),
            "transactions": (txn_a, txn_b)}


def _cleanup_seed(db, seeded):
    txn_ids = [t.id for t in seeded["transactions"]]
    db.query(TradeOutcome).filter(TradeOutcome.transaction_id.in_(txn_ids)).delete(synchronize_session=False)
    db.query(Transaction).filter(Transaction.id.in_(txn_ids)).delete(synchronize_session=False)
    insider_ids = [i.id for i in seeded["insiders"]]
    db.query(InsiderRole).filter(InsiderRole.insider_id.in_(insider_ids)).delete(synchronize_session=False)
    db.query(Insider).filter(Insider.id.in_(insider_ids)).delete(synchronize_session=False)
    db.query(PriceData).filter(PriceData.ticker == TICKER).delete(synchronize_session=False)
    db.query(Company).filter(Company.id == seeded["company"].id).delete(synchronize_session=False)
    db.commit()


def test_run_simulation_direct_call_opens_position_for_cluster_buy_scenario():
    """Regression: the non-preloaded run_simulation(db, ...) branch -- the
    exact path the CLI's `simulate-portfolio` command uses, since it never
    passes `preloaded=` -- must compute the same generic flags
    prepare_backtest_inputs() does, including cluster_size (which
    ClusterBuyStrategy.classify() needs via the cluster_unique_insiders
    alias). Before the fix, this branch only computed cluster/routine/
    filing-lag/liquidity flags when filter_routine/filter_filing_lag/
    liquidity_sizing/cluster_boost was truthy, and never computed
    first-time flags at all -- so a real cluster-buy scenario opened zero
    trades through this exact call shape."""
    db = SessionLocal()
    seeded = _seed_cluster_buy_scenario(db)
    try:
        portfolio, _price_index = run_simulation(db, shuffle_seed=1)
        assert len(portfolio.open_positions) + len(portfolio.closed_positions) >= 1
    finally:
        _cleanup_seed(db, seeded)
        db.close()


# ============================================================================
# Mechanical subset of test_portfolio_simulator.py: Portfolio/Position math,
# price-index helpers, metrics, report/CSV/JSON export, margin, SPY
# parking, concentration limits, 52-week drawdown.
#
# score_new_transaction delegates classification entirely to the active
# Strategy, so there is no tier-ladder-based TestSignalClassification here.
# Signal-type string labels used purely as arbitrary Position.signal_type
# values throughout (never validated against the registry) are
# "cluster_buy" (the shipped default's real name) or generic placeholders
# (test_signal_b/c) where a test needs several distinct labels.
# SIGNAL_HOLD_DAYS is narrowed to the one name the shipped registry
# actually knows.
# ============================================================================

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from form4lab.config import settings
from form4lab.strategy.registry import get_active
from form4lab.scoring.portfolio_simulator import (
    build_price_index,
    build_daily_equity_curve,
    get_price_on_or_after,
    get_price_n_td_later,
    get_last_price,
    get_52week_drawdown,
    _try_sell_spy_for_signal,
    Position,
    Portfolio,
    rebalance_spy,
    compute_metrics,
    format_report,
    export_trades_csv,
    export_trades_json,
    DEFAULT_HOLD_DAYS,
)

# Concentration limits now live on settings.alpaca (single-sourced from the
# strategy's allow_entry gate); hold-days now come from the signal registry.
# Same values as the deleted portfolio_simulator constants, new sources.
MAX_POSITIONS_PER_INSIDER_TICKER = settings.alpaca.max_positions_per_insider_ticker
MAX_POSITIONS_PER_TICKER = settings.alpaca.max_positions_per_ticker
SIGNAL_HOLD_DAYS = {
    "cluster_buy": get_active()[1].hold_days("cluster_buy", DEFAULT_HOLD_DAYS),
}


# ---------------------------------------------------------------------------
# Fixtures: synthetic data builders
# ---------------------------------------------------------------------------

def _make_price_df(ticker="TEST", start_date=date(2020, 1, 2), n_days=250,
                   base_price=100.0, daily_return=0.0005):
    """Create a synthetic price DataFrame with trading days only."""
    rows = []
    price = base_price
    d = start_date
    for _ in range(n_days):
        while d.weekday() >= 5:  # skip weekends
            d += timedelta(days=1)
        rows.append({"ticker": ticker, "date": d, "adj_close": price})
        price *= (1 + daily_return)
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _make_multi_ticker_prices():
    """Create price data for TEST, SPY, and CRASH tickers."""
    test_prices = _make_price_df("TEST", n_days=300, base_price=50.0, daily_return=0.001)
    spy_prices = _make_price_df("SPY", n_days=300, base_price=400.0, daily_return=0.0004)
    # CRASH: drops 30% around day 100
    crash_rows = []
    d = date(2020, 1, 2)
    price = 100.0
    for i in range(300):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        if 80 <= i <= 120:
            price = 100.0 * (1 - 0.30 * (i - 80) / 40)
        elif i > 120:
            price = 70.0 * (1 + 0.001 * (i - 120))
        crash_rows.append({"ticker": "CRASH", "date": d, "adj_close": price})
        d += timedelta(days=1)
    crash_prices = pd.DataFrame(crash_rows)
    return pd.concat([test_prices, spy_prices, crash_prices], ignore_index=True)


def _make_position(ticker="TEST", entry_date=date(2020, 3, 2), entry_price=50.0,
                   signal_type="cluster_buy", cost_basis=2000.0, insider_id=0):
    shares = cost_basis / entry_price
    return Position(
        txn_id=1,
        ticker=ticker,
        company_name="Test Corp",
        insider_name="John Smith",
        signal_type=signal_type,
        entry_date=entry_date,
        entry_price=entry_price,
        shares_held=shares,
        cost_basis=cost_basis,
        tier="Elite",
        skill_score=2.0,
        insider_id=insider_id,
    )


# ---------------------------------------------------------------------------
# Tests: Price index
# ---------------------------------------------------------------------------

class TestPriceIndex:
    def test_build_price_index(self):
        """Should create dict with sorted arrays per ticker."""
        df = _make_price_df("TEST", n_days=10)
        index = build_price_index(df)

        assert "TEST" in index
        dates, closes = index["TEST"]
        assert len(dates) == 10
        assert len(closes) == 10
        # Dates should be sorted
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_get_price_on_or_after_exact_date(self):
        """Should return exact price when date matches."""
        df = _make_price_df("TEST", start_date=date(2020, 1, 2), n_days=10)
        index = build_price_index(df)

        result = get_price_on_or_after(index, "TEST", date(2020, 1, 2))
        assert result is not None
        price, actual_date = result
        assert actual_date == date(2020, 1, 2)
        assert abs(price - 100.0) < 1.0

    def test_get_price_on_or_after_weekend(self):
        """Should return next trading day's price for weekend dates."""
        df = _make_price_df("TEST", start_date=date(2020, 1, 2), n_days=10)
        index = build_price_index(df)

        # Jan 4 2020 is a Saturday
        result = get_price_on_or_after(index, "TEST", date(2020, 1, 4))
        assert result is not None
        _, actual_date = result
        assert actual_date.weekday() < 5  # Should be a weekday

    def test_get_price_on_or_after_missing_ticker(self):
        """Should return None for unknown ticker."""
        index = build_price_index(_make_price_df("TEST"))
        assert get_price_on_or_after(index, "MISSING", date(2020, 1, 2)) is None

    def test_get_price_on_or_after_past_end(self):
        """Should return None for date past all data."""
        df = _make_price_df("TEST", n_days=5)
        index = build_price_index(df)
        assert get_price_on_or_after(index, "TEST", date(2025, 1, 1)) is None

    def test_get_price_n_td_later(self):
        """Should return price N trading days after entry."""
        df = _make_price_df("TEST", n_days=100)
        index = build_price_index(df)

        result = get_price_n_td_later(index, "TEST", date(2020, 1, 2), 60)
        assert result is not None
        price, exit_date = result
        assert exit_date > date(2020, 1, 2)
        assert price > 0

    def test_get_price_n_td_later_insufficient_data(self):
        """Should return None if not enough data after entry."""
        df = _make_price_df("TEST", n_days=10)
        index = build_price_index(df)
        assert get_price_n_td_later(index, "TEST", date(2020, 1, 2), 60) is None

    def test_get_last_price(self):
        """Should return the last available price."""
        df = _make_price_df("TEST", n_days=10, base_price=50.0)
        index = build_price_index(df)

        result = get_last_price(index, "TEST")
        assert result is not None
        price, last_date = result
        assert price > 0
        assert last_date > date(2020, 1, 2)

    def test_get_last_price_missing_ticker(self):
        """Should return None for unknown ticker."""
        index = build_price_index(_make_price_df("TEST"))
        assert get_last_price(index, "MISSING") is None


# ---------------------------------------------------------------------------
# Tests: Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_position_creation(self):
        pos = _make_position()
        assert pos.ticker == "TEST"
        assert pos.exit_date is None
        assert pos.pnl is None
        assert pos.force_closed is False

    def test_position_shares_from_cost_basis(self):
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        assert pos.shares_held == 40.0


# ---------------------------------------------------------------------------
# Tests: Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_initial_state(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        assert pf.cash == 10_000.0
        assert len(pf.open_positions) == 0
        assert len(pf.closed_positions) == 0

    def test_close_position_computes_pnl(self):
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        pf.close_position(pos, exit_price=55.0, exit_date=date(2020, 5, 1))

        assert len(pf.open_positions) == 0
        assert len(pf.closed_positions) == 1
        assert pos.exit_price == 55.0
        assert pos.exit_date == date(2020, 5, 1)
        assert abs(pos.pnl_pct - 0.10) < 0.001  # 55/50 - 1 = 10%
        assert abs(pos.pnl - (5.0 * 40.0)) < 0.01  # $5 * 40 shares = $200
        # Cash should increase by shares * exit_price
        assert abs(pf.cash - (8_000.0 + 40.0 * 55.0)) < 0.01

    def test_close_position_losing_trade(self):
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        pf.close_position(pos, exit_price=40.0, exit_date=date(2020, 5, 1))

        assert pos.pnl_pct < 0
        assert pos.pnl < 0
        assert abs(pos.pnl_pct - (-0.20)) < 0.001  # 40/50 - 1 = -20%

    def test_close_position_force_flag(self):
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position()
        pf.open_positions.append(pos)

        pf.close_position(pos, exit_price=50.0, exit_date=date(2020, 5, 1), force=True)

        assert pos.force_closed is True

    def test_total_value(self):
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        value = pf.total_value(index, date(2020, 3, 2))
        # 8000 cash + 40 shares * ~55 = 8000 + 2200 = ~10200
        assert value > 10_000.0

    def test_spy_shares_default_zero(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        assert pf.spy_shares == 0.0
        assert pf.spy_cost_basis == 0.0

    def test_total_value_includes_spy(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        value = pf.total_value(index, date(2020, 3, 2))
        assert abs(value - 10_000.0) < 1.0

    def test_gross_position_value_includes_spy(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        gross = pf.gross_position_value(index, date(2020, 3, 2))
        assert abs(gross - 8_000.0) < 1.0

    def test_buying_power_reduced_by_spy(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        bp = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=1.5)
        assert abs(bp - 7_000.0) < 1.0

    def test_record_snapshot(self):
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pf.record_snapshot(index, date(2020, 3, 2))

        assert len(pf.equity_curve) == 1
        assert pf.equity_curve[0][0] == date(2020, 3, 2)
        assert pf.equity_curve[0][1] == 10_000.0


class TestMetrics:
    def _make_portfolio_with_trades(self):
        """Create a portfolio with some closed trades for metrics testing."""
        prices = _make_multi_ticker_prices()
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)

        # Simulate 3 trades (2 winners, 1 loser) across 3 distinct signal
        # type labels, so the by-signal breakdown has something to group.
        trades = [
            ("TEST", date(2020, 2, 3), 50.0, 55.0, date(2020, 4, 20), "cluster_buy"),
            ("TEST", date(2020, 3, 2), 51.0, 56.0, date(2020, 5, 18), "test_signal_b"),
            ("CRASH", date(2020, 4, 1), 90.0, 72.0, date(2020, 6, 15), "test_signal_c"),
        ]

        for ticker, entry_d, entry_p, exit_p, exit_d, signal in trades:
            shares = 2000.0 / entry_p
            pos = Position(
                txn_id=len(pf.closed_positions) + 1,
                ticker=ticker,
                company_name=f"{ticker} Corp",
                insider_name="Test Insider",
                signal_type=signal,
                entry_date=entry_d,
                entry_price=entry_p,
                shares_held=shares,
                cost_basis=2000.0,
                tier="Elite",
                skill_score=2.0,
                exit_price=exit_p,
                exit_date=exit_d,
                pnl=(exit_p - entry_p) * shares,
                pnl_pct=(exit_p / entry_p) - 1.0,
            )
            pf.cash -= 2000.0
            pf.closed_positions.append(pos)
            pf.cash += shares * exit_p

        # Build equity curve
        pf.equity_curve = [
            (date(2020, 2, 3), 10_000.0),
            (date(2020, 4, 20), 10_200.0),
            (date(2020, 5, 18), 10_450.0),
            (date(2020, 6, 15), 10_050.0),
        ]

        return pf, index

    def test_compute_metrics_basic(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)

        assert metrics["total_trades"] == 3
        assert 0 < metrics["win_rate"] < 1  # 2/3 winners
        assert abs(metrics["win_rate"] - 2 / 3) < 0.01
        assert metrics["avg_win"] > 0
        assert metrics["avg_loss"] < 0
        assert metrics["profit_factor"] > 0

    def test_compute_metrics_returns(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)

        assert metrics["total_return"] > -1  # not total loss
        assert metrics["final_value"] > 0
        assert isinstance(metrics["cagr"], float)

    def test_compute_metrics_max_drawdown(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)

        assert metrics["max_drawdown"] <= 0  # drawdown is negative
        # From 10450 to 10050: dd = (10050-10450)/10450 = -3.8%
        assert metrics["max_drawdown"] < 0

    def test_compute_metrics_by_signal(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)

        assert "by_signal" in metrics
        assert "cluster_buy" in metrics["by_signal"]
        assert metrics["by_signal"]["cluster_buy"]["trades"] == 1

    def test_compute_metrics_spy_benchmark(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)

        assert metrics["spy_return"] is not None
        assert isinstance(metrics["spy_return"], float)

    def test_compute_metrics_empty_portfolio(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        index = build_price_index(_make_price_df())
        metrics = compute_metrics(pf, index)
        assert "error" in metrics

    def test_sharpe_ratio_is_finite(self):
        pf, index = self._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)
        assert np.isfinite(metrics["sharpe"])


# ---------------------------------------------------------------------------
# Tests: Report formatting
# ---------------------------------------------------------------------------

class TestReportFormatting:
    def test_format_report_produces_string(self):
        pf, index = TestMetrics()._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)
        report = format_report(metrics, pf)

        assert isinstance(report, str)
        assert "PORTFOLIO SIMULATION REPORT" in report
        assert "PERFORMANCE" in report
        assert "TRADE STATISTICS" in report
        assert "BY SIGNAL TYPE" in report

    def test_format_report_error(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        metrics = {"error": "No closed positions"}
        report = format_report(metrics, pf)
        assert "ERROR" in report

    def test_format_report_top_trades(self):
        pf, index = TestMetrics()._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)
        report = format_report(metrics, pf)

        assert "TOP 5 WINNERS" in report
        assert "TOP 5 LOSERS" in report


# ---------------------------------------------------------------------------
# Tests: CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_export_trades_csv(self, tmp_path):
        pf, _ = TestMetrics()._make_portfolio_with_trades()
        path = str(tmp_path / "trades.csv")
        n = export_trades_csv(pf, path)
        assert n == 3

        df = pd.read_csv(path)
        assert len(df) == 3
        assert "ticker" in df.columns
        assert "pnl" in df.columns
        assert "signal_type" in df.columns


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_multiple_positions_same_ticker(self):
        """Allow multiple positions in same stock."""
        pf = Portfolio(initial_cash=10_000.0, cash=6_000.0)
        pos1 = _make_position(ticker="TEST", entry_date=date(2020, 3, 2))
        pos2 = _make_position(ticker="TEST", entry_date=date(2020, 3, 9))
        pos2.txn_id = 2
        pf.open_positions.extend([pos1, pos2])
        assert len(pf.open_positions) == 2

    def test_insufficient_cash_prevents_trade(self):
        """Portfolio with no cash should not open new positions."""
        pf = Portfolio(initial_cash=10_000.0, cash=500.0)
        # Position size is $2000 but only $500 cash
        assert pf.cash < 2000.0

    def test_equity_curve_max_drawdown_no_loss(self):
        """If equity only goes up, max drawdown should be 0."""
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pos.exit_price = 60.0
        pos.exit_date = date(2020, 5, 1)
        pos.pnl = (60.0 - 50.0) * 40.0
        pos.pnl_pct = 0.20
        pf.closed_positions.append(pos)
        pf.equity_curve = [
            (date(2020, 3, 2), 10_000.0),
            (date(2020, 5, 1), 10_400.0),
        ]

        index = build_price_index(_make_multi_ticker_prices())
        metrics = compute_metrics(pf, index)
        assert metrics["max_drawdown"] == 0


# ---------------------------------------------------------------------------
# Tests: Per-signal hold days
# ---------------------------------------------------------------------------

class TestSignalHoldDays:
    def test_signal_hold_days_mapping(self):
        """SIGNAL_HOLD_DAYS should reflect the active strategy's registry
        (ClusterBuyStrategy declares cluster_buy at hold_days=60)."""
        assert SIGNAL_HOLD_DAYS["cluster_buy"] == 60
        assert DEFAULT_HOLD_DAYS == 60

    def test_position_default_hold_days(self):
        """Position should default to DEFAULT_HOLD_DAYS."""
        pos = _make_position()
        assert pos.hold_days == DEFAULT_HOLD_DAYS

    def test_position_custom_hold_days(self):
        """Position should accept custom hold_days."""
        pos = _make_position()
        pos.hold_days = 90
        assert pos.hold_days == 90

    def test_position_hold_days_in_constructor(self):
        """Position should accept hold_days in constructor."""
        pos = Position(
            txn_id=1, ticker="TEST", company_name="Test Corp",
            insider_name="John Smith",
            signal_type="cluster_buy",
            entry_date=date(2020, 3, 2), entry_price=50.0,
            shares_held=40.0, cost_basis=2000.0,
            tier="Elite", skill_score=2.0,
            hold_days=90,
        )
        assert pos.hold_days == 90

    def test_metrics_sharpe_uses_avg_hold_days(self):
        """Sharpe annualization should use average hold period."""
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)

        # Create two positions with different hold_days
        pos1 = Position(
            txn_id=1, ticker="TEST", company_name="Test Corp",
            insider_name="Test Insider",
            signal_type="cluster_buy",
            entry_date=date(2020, 2, 3), entry_price=50.0,
            shares_held=40.0, cost_basis=2000.0,
            tier="Elite", skill_score=2.0,
            hold_days=90,
            exit_price=55.0, exit_date=date(2020, 5, 20),
            pnl=200.0, pnl_pct=0.10,
        )
        pos2 = Position(
            txn_id=2, ticker="TEST", company_name="Test Corp",
            insider_name="Test Insider",
            signal_type="cluster_buy",
            entry_date=date(2020, 3, 2), entry_price=51.0,
            shares_held=39.2, cost_basis=2000.0,
            tier="Elite", skill_score=2.0,
            hold_days=60,
            exit_price=56.0, exit_date=date(2020, 5, 18),
            pnl=196.0, pnl_pct=0.098,
        )
        pf.closed_positions = [pos1, pos2]
        pf.equity_curve = [
            (date(2020, 2, 3), 10_000.0),
            (date(2020, 5, 20), 10_400.0),
        ]

        prices = _make_multi_ticker_prices()
        index = build_price_index(prices)
        metrics = compute_metrics(pf, index)

        assert np.isfinite(metrics["sharpe"])

    def test_csv_export_includes_hold_days(self, tmp_path):
        """CSV export should include hold_days column."""
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pos = Position(
            txn_id=1, ticker="TEST", company_name="Test Corp",
            insider_name="Test Insider",
            signal_type="cluster_buy",
            entry_date=date(2020, 3, 2), entry_price=50.0,
            shares_held=40.0, cost_basis=2000.0,
            tier="Elite", skill_score=2.0,
            hold_days=90,
            exit_price=55.0, exit_date=date(2020, 5, 20),
            pnl=200.0, pnl_pct=0.10,
        )
        pf.closed_positions = [pos]

        path = str(tmp_path / "trades.csv")
        n = export_trades_csv(pf, path)
        assert n == 1

        df = pd.read_csv(path)
        assert "hold_days" in df.columns
        assert df.iloc[0]["hold_days"] == 90


# ---------------------------------------------------------------------------
# Tests: Percentage-based position sizing
# ---------------------------------------------------------------------------

class TestPercentageSizing:
    def test_pct_sizing_scales_with_portfolio(self):
        """Position size should grow as portfolio value grows."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        # Small portfolio: 15% of $10K = $1,500
        pf_small = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        small_value = pf_small.total_value(index, date(2020, 3, 2))
        assert abs(small_value * 0.15 - 1_500.0) < 1.0

        # Large portfolio: 15% of $50K = $7,500
        pf_large = Portfolio(initial_cash=50_000.0, cash=50_000.0)
        large_value = pf_large.total_value(index, date(2020, 3, 2))
        assert abs(large_value * 0.15 - 7_500.0) < 1.0

    def test_pct_sizing_includes_open_positions(self):
        """Portfolio value should include market value of open positions."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        # 8000 cash + 40 shares * ~55 = ~10,200
        value = pf.total_value(index, date(2020, 3, 2))
        assert value > 10_000.0
        # 15% of ~10,200 = ~1,530
        pct_size = value * 0.15
        assert 1_400 < pct_size < 1_600


# ---------------------------------------------------------------------------
# Tests: JSON export
# ---------------------------------------------------------------------------

class TestJSONExport:
    def test_export_trades_json_returns_list(self):
        """Should return a list of trade dicts from closed positions."""
        pf, _ = TestMetrics()._make_portfolio_with_trades()
        trades = export_trades_json(pf)
        assert isinstance(trades, list)
        assert len(trades) == 3

    def test_export_trades_json_fields(self):
        """Each trade dict should have all required fields."""
        pf, _ = TestMetrics()._make_portfolio_with_trades()
        trades = export_trades_json(pf)
        required_fields = {
            "ticker", "company_name", "insider_name", "signal_type",
            "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "pnl", "pnl_pct", "hold_days", "force_closed",
        }
        for trade in trades:
            assert required_fields.issubset(trade.keys()), f"Missing: {required_fields - trade.keys()}"

    def test_export_trades_json_serializable(self):
        """All values should be JSON-serializable (no date objects, no numpy)."""
        import json
        pf, _ = TestMetrics()._make_portfolio_with_trades()
        trades = export_trades_json(pf)
        json.dumps(trades)

    def test_export_trades_json_sorted_by_exit_date_desc(self):
        """Trades should be sorted by exit_date descending (most recent first)."""
        pf, _ = TestMetrics()._make_portfolio_with_trades()
        trades = export_trades_json(pf)
        exit_dates = [t["exit_date"] for t in trades]
        assert exit_dates == sorted(exit_dates, reverse=True)


# ---------------------------------------------------------------------------
# Tests: Sell exit config
# ---------------------------------------------------------------------------

class TestSellExitConfig:
    def test_sell_exit_config_defaults(self):
        """SignalConfig should have sell exit parameters with defaults."""
        from form4lab.config import settings
        assert settings.signal.sell_exit_enabled is False
        assert settings.signal.sell_cluster_exit_delay == 0
        assert settings.signal.sell_large_exit_delay == 5


# ---------------------------------------------------------------------------
# Tests: Margin functionality
# ---------------------------------------------------------------------------

class TestMarginBuyingPower:
    def test_buying_power_no_margin(self):
        """With margin_multiplier=1.0, buying power = cash."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        bp = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=1.0)
        assert abs(bp - 10_000.0) < 0.01

    def test_buying_power_with_margin(self):
        """With margin_multiplier=2.0, buying power = equity*2 - gross_positions."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        # All cash, no positions: equity=10K, bp=10K*2 - 0 = 20K
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        bp = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=2.0)
        assert abs(bp - 20_000.0) < 0.01

    def test_buying_power_with_positions(self):
        """Buying power decreases as positions are held."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=5_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=5000.0)  # 100 shares * 55 = 5500 mkt value
        pf.open_positions.append(pos)

        # equity = 5000 cash + 100*55 = 10500
        # gross_positions = 100*55 = 5500
        # bp = 10500*2 - 5500 = 15500
        bp = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=2.0)
        equity = pf.total_value(index, date(2020, 3, 2))
        gross = pf.gross_position_value(index, date(2020, 3, 2))
        expected = equity * 2 - gross
        assert abs(bp - expected) < 0.01

    def test_buying_power_negative_cash_positive_equity(self):
        """With margin, cash can be negative but buying power uses equity."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        # Negative cash (borrowed $2K), but position is worth $5,500
        pf = Portfolio(initial_cash=10_000.0, cash=-2_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=5000.0)  # 100 shares * 55 = 5500
        pf.open_positions.append(pos)

        # No-margin buying power should be 0 (cash is negative)
        bp_no_margin = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=1.0)
        assert bp_no_margin == 0.0

        # With margin: equity = -2000 + 5500 = 3500, bp = 3500*2 - 5500 = 1500
        bp_margin = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=2.0)
        assert bp_margin > 0

    def test_buying_power_floors_at_zero(self):
        """Buying power never goes below 0."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=30.0)
        index = build_price_index(prices)

        # Underwater: cash=-8000, position cost=8000 but now worth 30*160=4800
        pf = Portfolio(initial_cash=10_000.0, cash=-8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=8000.0)  # 160 shares * 30 = 4800
        pf.open_positions.append(pos)

        bp = pf.buying_power(index, date(2020, 3, 2), margin_multiplier=2.0)
        assert bp >= 0.0


class TestMarginLoan:
    def test_no_loan_positive_cash(self):
        """No margin loan when cash is positive."""
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        assert pf.margin_loan() == 0.0

    def test_loan_negative_cash(self):
        """Margin loan equals abs(cash) when cash is negative."""
        pf = Portfolio(initial_cash=10_000.0, cash=-3_000.0)
        assert abs(pf.margin_loan() - 3_000.0) < 0.01


class TestGrossPositionValue:
    def test_gross_value_no_positions(self):
        """Empty portfolio has 0 gross position value."""
        index = build_price_index(_make_price_df("TEST", n_days=10))
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        assert pf.gross_position_value(index, date(2020, 1, 2)) == 0.0

    def test_gross_value_with_positions(self):
        """Gross position value should sum market values of open positions."""
        prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=55.0)
        index = build_price_index(prices)

        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)  # 40 shares
        pf.open_positions.append(pos)

        gross = pf.gross_position_value(index, date(2020, 3, 2))
        assert abs(gross - 40.0 * 55.0) < 1.0


class TestMarginInterest:
    def test_margin_stats_attached_to_portfolio(self):
        """run_simulation should attach _margin_stats to portfolio."""
        # We can't easily run the full simulation in unit tests, but we can
        # verify the Portfolio class supports the margin_stats attribute
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pf._margin_stats = {
            "margin_interest_paid": 100.0,
            "margin_calls": 0,
            "forced_liquidations": 0,
            "max_margin_loan": 5000.0,
            "avg_leverage": 1.5,
            "margin_multiplier": 2.0,
        }
        assert pf._margin_stats["margin_interest_paid"] == 100.0

    def test_metrics_include_margin_section(self):
        """compute_metrics should include margin section when margin used."""
        pf, index = TestMetrics()._make_portfolio_with_trades()
        pf._margin_stats = {
            "margin_interest_paid": 150.0,
            "margin_calls": 1,
            "forced_liquidations": 2,
            "max_margin_loan": 8000.0,
            "avg_leverage": 1.6,
            "margin_multiplier": 2.0,
        }
        metrics = compute_metrics(pf, index)
        assert "margin" in metrics
        assert metrics["margin"]["margin_interest_paid"] == 150.0
        assert metrics["margin"]["margin_calls"] == 1

    def test_metrics_no_margin_section_when_multiplier_1(self):
        """compute_metrics should not include margin when multiplier=1.0."""
        pf, index = TestMetrics()._make_portfolio_with_trades()
        pf._margin_stats = {
            "margin_interest_paid": 0.0,
            "margin_calls": 0,
            "forced_liquidations": 0,
            "max_margin_loan": 0.0,
            "avg_leverage": 1.0,
            "margin_multiplier": 1.0,
        }
        metrics = compute_metrics(pf, index)
        assert "margin" not in metrics


class TestMarginReport:
    def test_report_includes_margin_section(self):
        """format_report should include MARGIN section when margin used."""
        pf, index = TestMetrics()._make_portfolio_with_trades()
        pf._margin_stats = {
            "margin_interest_paid": 250.0,
            "margin_calls": 2,
            "forced_liquidations": 3,
            "max_margin_loan": 12000.0,
            "avg_leverage": 1.8,
            "margin_multiplier": 2.0,
        }
        metrics = compute_metrics(pf, index)
        report = format_report(metrics, pf)

        assert "MARGIN" in report
        assert "2.0x" in report
        assert "Margin calls:" in report
        assert "interest paid" in report.lower()

    def test_report_no_margin_section_when_no_margin(self):
        """format_report should not include MARGIN section for cash-only."""
        pf, index = TestMetrics()._make_portfolio_with_trades()
        metrics = compute_metrics(pf, index)
        report = format_report(metrics, pf)

        assert "MARGIN" not in report


class TestMarginMaintenanceCheck:
    def test_close_position_restores_margin(self):
        """Closing a position should free up margin (increase cash)."""
        pf = Portfolio(initial_cash=10_000.0, cash=-5_000.0)  # borrowed $5K
        pos = _make_position(entry_price=50.0, cost_basis=7000.0)  # 140 shares
        pf.open_positions.append(pos)

        # Close at $60 -> returns 140*60 = 8400 cash
        pf.close_position(pos, exit_price=60.0, exit_date=date(2020, 5, 1))

        assert pf.cash == -5_000.0 + 140.0 * 60.0  # 3400
        assert pf.cash > 0  # Margin loan repaid
        assert len(pf.open_positions) == 0


class TestConcentrationLimits:
    """Tests for per-ticker and per-insider-ticker concentration limits."""

    def test_constants_defined(self):
        """Concentration limit constants should have expected values."""
        assert MAX_POSITIONS_PER_INSIDER_TICKER == 1
        assert MAX_POSITIONS_PER_TICKER == 2

    def test_position_stores_insider_id(self):
        """Position should store the insider_id passed to constructor."""
        pos = _make_position(insider_id=5018)
        assert pos.insider_id == 5018

    def test_position_default_insider_id_is_zero(self):
        """Position without explicit insider_id defaults to 0."""
        pos = Position(
            txn_id=1, ticker="TEST", company_name="Test Corp",
            insider_name="John", signal_type="cluster_buy",
            entry_date=date(2020, 1, 2), entry_price=50.0,
            shares_held=10.0, cost_basis=500.0, tier="Elite",
            skill_score=2.0,
        )
        assert pos.insider_id == 0

    def test_per_insider_ticker_limit_blocks_same_insider(self):
        """Same insider with an open position in a ticker should be blocked.

        Simulates the check at step 4b of run_simulation: if an insider
        already has MAX_POSITIONS_PER_INSIDER_TICKER open positions in
        a ticker, additional signals from that insider are skipped.
        """
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        # Insider 5018 already has an open position in ZZL
        pf.open_positions.append(_make_position(ticker="ZZL", insider_id=5018))

        # Simulate the concentration check from run_simulation
        new_insider_id = 5018
        open_same_ticker = [p for p in pf.open_positions if p.ticker == "ZZL"]
        open_same_insider_ticker = [p for p in open_same_ticker if p.insider_id == new_insider_id]

        assert len(open_same_insider_ticker) >= MAX_POSITIONS_PER_INSIDER_TICKER
        # This trade should be skipped

    def test_per_insider_ticker_limit_allows_different_insider(self):
        """Different insider should be allowed even if ticker has a position."""
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pf.open_positions.append(_make_position(ticker="ZZL", insider_id=5018))

        new_insider_id = 9999  # different insider
        open_same_ticker = [p for p in pf.open_positions if p.ticker == "ZZL"]
        open_same_insider_ticker = [p for p in open_same_ticker if p.insider_id == new_insider_id]

        assert len(open_same_insider_ticker) < MAX_POSITIONS_PER_INSIDER_TICKER
        # This trade should be allowed

    def test_per_ticker_limit_blocks_at_max(self):
        """A ticker at MAX_POSITIONS_PER_TICKER open positions should block new entries."""
        pf = Portfolio(initial_cash=50_000.0, cash=50_000.0)
        # 2 different insiders each have a position in ZZL (at the limit)
        for iid in [5018, 5020]:
            pf.open_positions.append(_make_position(ticker="ZZL", insider_id=iid))

        open_same_ticker = [p for p in pf.open_positions if p.ticker == "ZZL"]
        assert len(open_same_ticker) >= MAX_POSITIONS_PER_TICKER
        # Any new ZZL trade should be skipped, regardless of insider

    def test_per_ticker_limit_allows_below_max(self):
        """A ticker below MAX_POSITIONS_PER_TICKER should allow new entries."""
        pf = Portfolio(initial_cash=50_000.0, cash=50_000.0)
        # Only 1 position in ZZL (below limit of 2)
        pf.open_positions.append(_make_position(ticker="ZZL", insider_id=5018))

        open_same_ticker = [p for p in pf.open_positions if p.ticker == "ZZL"]
        assert len(open_same_ticker) < MAX_POSITIONS_PER_TICKER
        # A new insider's ZZL trade should be allowed

    def test_insider_id_zero_matches_all_zero_positions(self):
        """If insider_id=0 (the old bug), all positions match each other.

        This verifies the bug scenario: when insider_id defaults to 0,
        the per-insider check would either block everything (if checking
        against 0) or nothing (if real IDs don't match 0).
        """
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        # Position with insider_id=0 (the bug)
        pf.open_positions.append(_make_position(ticker="ZZL", insider_id=0))

        # Real insider_id=5018 does NOT match insider_id=0
        real_insider_id = 5018
        open_same_ticker = [p for p in pf.open_positions if p.ticker == "ZZL"]
        open_same_insider_ticker = [p for p in open_same_ticker if p.insider_id == real_insider_id]
        assert len(open_same_insider_ticker) == 0  # bug: limit never fires


# ---------------------------------------------------------------------------
# Tests: 52-week drawdown helper
# ---------------------------------------------------------------------------

class TestDrawdownHelper:
    def test_52week_drawdown_at_high(self):
        """Stock at its 52-week high should return ~0."""
        # Price steadily rises to 100
        df = _make_price_df("TEST", n_days=260, base_price=50.0, daily_return=0.003)
        index = build_price_index(df)
        dates, closes = index["TEST"]
        last_date = pd.Timestamp(dates[-1]).date()
        last_price = float(closes[-1])

        dd = get_52week_drawdown(index, "TEST", last_price, last_date)
        assert dd is not None
        assert abs(dd) < 0.01  # at the high, drawdown ~= 0

    def test_52week_drawdown_deep_below(self):
        """Stock 30% below high should return ~-0.30."""
        # Create price data: rises to 100, then drops to 70
        rows = []
        d = date(2020, 1, 2)
        for i in range(260):
            while d.weekday() >= 5:
                d += timedelta(days=1)
            if i < 200:
                price = 70.0 + (30.0 * i / 200)  # rises to 100
            else:
                price = 100.0 - (30.0 * (i - 200) / 60)  # drops toward 70
            rows.append({"ticker": "DROP", "date": d, "adj_close": price})
            d += timedelta(days=1)
        df = pd.DataFrame(rows)
        index = build_price_index(df)
        dates, closes = index["DROP"]
        last_date = pd.Timestamp(dates[-1]).date()
        last_price = float(closes[-1])

        dd = get_52week_drawdown(index, "DROP", last_price, last_date)
        assert dd is not None
        assert dd < -0.20  # should be around -0.30

    def test_52week_drawdown_insufficient_data(self):
        """Should return None with fewer than 20 trading days."""
        df = _make_price_df("TINY", n_days=15, base_price=100.0)
        index = build_price_index(df)
        dates, closes = index["TINY"]
        last_date = pd.Timestamp(dates[-1]).date()

        dd = get_52week_drawdown(index, "TINY", 100.0, last_date)
        assert dd is None

    def test_52week_drawdown_missing_ticker(self):
        """Should return None for unknown ticker."""
        index = build_price_index(_make_price_df("TEST"))
        dd = get_52week_drawdown(index, "MISSING", 100.0, date(2020, 6, 1))
        assert dd is None


class TestSpyParking:
    def test_buy_spy(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pf.buy_spy(2_000.0, 400.0)
        assert pf.cash == 8_000.0
        assert abs(pf.spy_shares - 5.0) < 0.001
        assert abs(pf.spy_cost_basis - 2_000.0) < 0.01

    def test_buy_spy_adds_to_existing(self):
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pf.spy_shares = 5.0
        pf.spy_cost_basis = 2_000.0
        pf.buy_spy(1_000.0, 420.0)
        assert abs(pf.cash - 7_000.0) < 0.01
        assert abs(pf.spy_shares - (5.0 + 1_000.0 / 420.0)) < 0.001
        assert abs(pf.spy_cost_basis - 3_000.0) < 0.01

    def test_sell_spy(self):
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        pnl = pf.sell_spy(4_000.0, 420.0)
        shares_sold = 4_000.0 / 420.0
        assert abs(pf.spy_shares - (20.0 - shares_sold)) < 0.001
        assert abs(pf.cash - 6_000.0) < 0.01
        expected_pnl = shares_sold * (420.0 - 400.0)
        assert abs(pnl - expected_pnl) < 0.01

    def test_sell_spy_capped_at_holdings(self):
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 5.0
        pf.spy_cost_basis = 2_000.0
        pf.sell_spy(10_000.0, 400.0)
        assert pf.spy_shares == 0.0
        assert pf.spy_cost_basis == 0.0
        assert abs(pf.cash - 4_000.0) < 0.01

    def test_sell_spy_at_loss(self):
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        pnl = pf.sell_spy(3_500.0, 350.0)
        shares_sold = 3_500.0 / 350.0
        assert pnl < 0
        expected_pnl = shares_sold * (350.0 - 400.0)
        assert abs(pnl - expected_pnl) < 0.01

    def test_buy_spy_zero_amount(self):
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        pf.buy_spy(0.0, 400.0)
        assert pf.spy_shares == 0.0
        assert pf.cash == 10_000.0


class TestTrySellSpyForSignal:
    def test_no_sell_when_buying_power_sufficient(self):
        """Should not sell SPY if buying power already covers position size."""
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=5_000.0)
        pf.spy_shares = 10.0
        pf.spy_cost_basis = 4_000.0
        pnl, did_sell = _try_sell_spy_for_signal(pf, 3_000.0, index, date(2020, 3, 2), 1.0)
        assert not did_sell
        assert pnl == 0.0
        assert pf.spy_shares == 10.0  # unchanged

    def test_sells_correct_shortfall_with_margin(self):
        """With margin, shortfall should be position_size - buying_power, not position_size - cash."""
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        test_prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=50.0)
        index = build_price_index(pd.concat([spy_prices, test_prices]))

        # Scenario: negative cash but positive buying power via margin
        pf = Portfolio(initial_cash=10_000.0, cash=-1_000.0)
        pf.spy_shares = 25.0  # $10K in SPY
        pf.spy_cost_basis = 10_000.0
        # Add an open position so total_value includes it
        pos = _make_position(entry_price=50.0, cost_basis=5_000.0)
        pf.open_positions.append(pos)

        # With margin=1.5: equity = cash + positions + spy = -1000 + 5000 + 10000 = 14000
        # buying_power = equity * 1.5 - gross = 14000 * 1.5 - 15000 = 6000
        # position_size = 8000, shortfall should be 8000 - 6000 = 2000
        original_spy_shares = pf.spy_shares
        pnl, did_sell = _try_sell_spy_for_signal(pf, 8_000.0, index, date(2020, 3, 2), 1.5)
        assert did_sell
        # Should only sell ~$2000 worth of SPY (shortfall), NOT $9000 (position_size - cash)
        shares_sold = original_spy_shares - pf.spy_shares
        amount_sold = shares_sold * 400.0
        assert amount_sold < 3_000.0, f"Sold ${amount_sold:.0f} of SPY, expected ~$2000"

    def test_no_sell_when_no_spy_price(self):
        """Should return (0, False) if no SPY price data available."""
        test_prices = _make_price_df("TEST", start_date=date(2020, 3, 2), n_days=10, base_price=50.0)
        index = build_price_index(test_prices)  # no SPY
        pf = Portfolio(initial_cash=10_000.0, cash=1_000.0)
        pf.spy_shares = 10.0
        pf.spy_cost_basis = 4_000.0
        pnl, did_sell = _try_sell_spy_for_signal(pf, 5_000.0, index, date(2020, 3, 2), 1.0)
        assert not did_sell
        assert pnl == 0.0


class TestBuildDailyEquityCurve:
    def test_returns_pct_invested_column(self):
        """Equity curve should include pct_invested and num_positions columns."""
        prices_df = _make_multi_ticker_prices()
        index = build_price_index(prices_df)

        pos = _make_position(
            ticker="TEST", entry_date=date(2020, 3, 2),
            entry_price=50.0, cost_basis=2000.0,
        )
        pos.exit_date = date(2020, 6, 1)
        pos.exit_price = 55.0

        portfolio = Portfolio(initial_cash=10000.0, cash=10000.0)
        portfolio.closed_positions = [pos]

        df = build_daily_equity_curve(portfolio, index)

        assert "pct_invested" in df.columns
        assert "num_positions" in df.columns
        assert len(df) > 0

    def test_pct_invested_zero_when_no_positions(self):
        """Before any position opens and after all close, pct_invested should be 0."""
        prices_df = _make_multi_ticker_prices()
        index = build_price_index(prices_df)

        pos = _make_position(
            ticker="TEST", entry_date=date(2020, 3, 2),
            entry_price=50.0, cost_basis=2000.0,
        )
        pos.exit_date = date(2020, 6, 1)
        pos.exit_price = 55.0

        portfolio = Portfolio(initial_cash=10000.0, cash=10000.0)
        portfolio.closed_positions = [pos]

        df = build_daily_equity_curve(portfolio, index)

        # After exit, pct_invested should be 0
        post_exit = df[df["date"] >= date(2020, 6, 2)]
        if len(post_exit) > 0:
            assert all(post_exit["pct_invested"] == 0.0)
            assert all(post_exit["num_positions"] == 0)

    def test_pct_invested_nonzero_during_position(self):
        """While a position is open, pct_invested should be > 0."""
        prices_df = _make_multi_ticker_prices()
        index = build_price_index(prices_df)

        pos = _make_position(
            ticker="TEST", entry_date=date(2020, 3, 2),
            entry_price=50.0, cost_basis=2000.0,
        )
        pos.exit_date = date(2020, 6, 1)
        pos.exit_price = 55.0

        portfolio = Portfolio(initial_cash=10000.0, cash=10000.0)
        portfolio.closed_positions = [pos]

        df = build_daily_equity_curve(portfolio, index)

        # During the position, pct_invested should be > 0
        during = df[(df["date"] > date(2020, 3, 2)) & (df["date"] < date(2020, 6, 1))]
        assert len(during) > 0
        assert all(during["pct_invested"] > 0)
        assert all(during["num_positions"] == 1)

    def test_empty_portfolio_returns_empty_df(self):
        """Empty portfolio should return empty DataFrame with new columns."""
        prices_df = _make_multi_ticker_prices()
        index = build_price_index(prices_df)

        portfolio = Portfolio(initial_cash=10000.0, cash=10000.0)
        df = build_daily_equity_curve(portfolio, index)

        assert "pct_invested" in df.columns
        assert "num_positions" in df.columns
        assert len(df) == 0


class TestSpyRebalance:
    def test_rebalance_buys_spy_when_excess_cash(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        rebalance_spy(pf, index, date(2020, 3, 2), buffer_pct=0.20)
        assert pf.spy_shares > 0
        assert abs(pf.cash - 2_000.0) < 1.0
        assert abs(pf.spy_shares - 20.0) < 0.01

    def test_rebalance_sells_spy_when_cash_below_buffer(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=500.0)
        pf.spy_shares = 23.75
        pf.spy_cost_basis = 9_500.0
        rebalance_spy(pf, index, date(2020, 3, 2), buffer_pct=0.20)
        assert abs(pf.cash - 2_000.0) < 5.0

    def test_rebalance_noop_when_near_buffer(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=2_000.0)
        pf.spy_shares = 20.0
        pf.spy_cost_basis = 8_000.0
        original_shares = pf.spy_shares
        rebalance_spy(pf, index, date(2020, 3, 2), buffer_pct=0.20)
        assert pf.spy_shares == original_shares

    def test_rebalance_no_spy_without_price_data(self):
        index = build_price_index(_make_price_df("TEST", n_days=10))
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        rebalance_spy(pf, index, date(2020, 3, 2), buffer_pct=0.20)
        assert pf.spy_shares == 0.0
        assert pf.cash == 10_000.0

    def test_rebalance_zero_buffer_all_to_spy(self):
        spy_prices = _make_price_df("SPY", start_date=date(2020, 3, 2), n_days=10, base_price=400.0)
        index = build_price_index(spy_prices)
        pf = Portfolio(initial_cash=10_000.0, cash=10_000.0)
        rebalance_spy(pf, index, date(2020, 3, 2), buffer_pct=0.0)
        assert pf.cash < 1.0
        assert pf.spy_shares > 0


# ---------------------------------------------------------------------------
# Tests: build_daily_equity_curve with spy_parking_buffer
# ---------------------------------------------------------------------------

class TestEquityCurveSpyParking:
    def test_pct_invested_with_spy_parking_buffer(self):
        """With spy_parking_buffer > 0, pct_invested should include approximated SPY value."""
        prices = _make_multi_ticker_prices()
        index = build_price_index(prices)
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        # One open position worth $2,000
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        df = build_daily_equity_curve(pf, index, spy_parking_buffer=0.20)
        assert not df.empty
        # With 20% buffer and $8K cash / ~$10K total, buffer is ~$2K
        # SPY value ≈ $8K - $2K = $6K, so deployed ≈ $2K + $6K = $8K / $10K = 80%
        # pct_invested should be well above what it'd be without SPY parking
        first_pct = df["pct_invested"].iloc[0]
        assert first_pct >= 0.75, f"Expected ≥75% deployed with SPY parking, got {first_pct:.1%}"

    def test_pct_invested_without_spy_parking(self):
        """Without spy_parking_buffer, pct_invested is just market_value / total."""
        prices = _make_multi_ticker_prices()
        index = build_price_index(prices)
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        df = build_daily_equity_curve(pf, index, spy_parking_buffer=0.0)
        assert not df.empty
        # Without SPY parking: $2K market / $10K total ≈ 20%
        first_pct = df["pct_invested"].iloc[0]
        assert first_pct < 0.30, f"Expected <30% deployed without SPY parking, got {first_pct:.1%}"

    def test_spy_parking_buffer_increases_deployment(self):
        """spy_parking_buffer should always result in higher pct_invested than without."""
        prices = _make_multi_ticker_prices()
        index = build_price_index(prices)
        pf = Portfolio(initial_cash=10_000.0, cash=8_000.0)
        pos = _make_position(entry_price=50.0, cost_basis=2000.0)
        pf.open_positions.append(pos)

        df_no_spy = build_daily_equity_curve(pf, index, spy_parking_buffer=0.0)
        df_spy = build_daily_equity_curve(pf, index, spy_parking_buffer=0.20)
        # Every day's pct_invested should be >= with SPY parking
        for i in range(len(df_spy)):
            assert df_spy["pct_invested"].iloc[i] >= df_no_spy["pct_invested"].iloc[i]


