"""Tests for form4lab.scoring.signal_generator (score_new_transaction /
score_sell_transaction) and the live detector functions in
form4lab.strategy.features.

Tests for the platform's signal-generation orchestration.
score_new_transaction/score_sell_transaction
resolve entities, compute event value, dedup, and delegate the entire
classify/conviction decision to the active Strategy (see
form4lab/scoring/signal_generator.py). The shipped ClusterBuyStrategy
recognizes two rungs — cluster_buy / filtered_out — and declares zero sell
signal types:
  - Sell-scoring MECHANISM (dedup, value computation, delegation) is still
    generic platform code worth testing, so a minimal sell-capable test
    double strategy exercises it below — the default strategy alone would
    make a dedup test vacuous (both calls trivially return None).
  - The pure detector functions (role weight, insider median value,
    drawdown/short-momentum/cluster-sell/sell-pct) are unchanged generic
    logic, just relocated to form4lab.strategy.features — ported with
    updated imports.
"""
import pytest
from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.score import InsiderScore
from form4lab.models.alert import Alert
from form4lab.models.price import PriceData
from form4lab.scoring.signal_generator import score_new_transaction, score_sell_transaction
from form4lab.strategy.base import SellEvaluation, SignalType, Strategy
from form4lab.strategy.features import (
    get_role_weight, get_insider_median_value, detect_drawdown,
    detect_short_momentum, detect_cluster_sell, compute_sell_pct,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_role_weights():
    assert get_role_weight("CEO") == 1.0
    assert get_role_weight("Chief Financial Officer") == 0.95  # contains CFO
    assert get_role_weight("Director") == 0.6
    assert get_role_weight("Unknown Role") == 0.5


def test_role_weight_none():
    assert get_role_weight("") == 0.5
    assert get_role_weight(None) == 0.5


def test_insider_median_value_default(db):
    insider = Insider(cik="001", name="Test")
    db.add(insider)
    db.flush()
    assert get_insider_median_value(insider.id, db) == 50000.0


# ---------------------------------------------------------------------------
# score_new_transaction — rewritten against ClusterBuyStrategy's two rungs
# ---------------------------------------------------------------------------

def test_score_new_transaction_cluster_buy(db):
    """Two distinct insiders buying the same company within the cluster
    window, each transaction >= the $25k value floor -> cluster_buy."""
    company = Company(cik="002", ticker="TEST", name="Test Corp", sector="Technology")
    insider_a = Insider(cik="001", name="Jane Smith")
    insider_b = Insider(cik="003", name="Sam Jones")
    db.add_all([company, insider_a, insider_b])
    db.flush()

    db.add_all([
        InsiderRole(insider_id=insider_a.id, company_id=company.id, role_title="CEO",
                   is_officer=True, is_director=False, is_ten_percent_owner=False,
                   first_filing_date=date(2020, 1, 1)),
        InsiderRole(insider_id=insider_b.id, company_id=company.id, role_title="Director",
                   is_officer=False, is_director=True, is_ten_percent_owner=False,
                   first_filing_date=date(2020, 1, 1)),
    ])
    db.flush()

    txn_b = Transaction(
        insider_id=insider_b.id, company_id=company.id,
        accession_number="ACC-000", filing_date=date(2024, 5, 30),
        transaction_date=date(2024, 5, 30), transaction_code="P",
        shares=500, price_per_share=60.0, total_value=30000.0,
        acquired_or_disposed="A", is_discretionary=True,
    )
    txn_a = Transaction(
        insider_id=insider_a.id, company_id=company.id,
        accession_number="ACC-001", filing_date=date(2024, 6, 1),
        transaction_date=date(2024, 6, 1), transaction_code="P",
        shares=1000, price_per_share=100.0, total_value=100000.0,
        acquired_or_disposed="A", is_discretionary=True,
    )
    db.add_all([txn_b, txn_a])
    db.flush()

    alert = score_new_transaction(txn_a.id, db)
    assert alert is not None
    assert alert.alert_type == "cluster_buy"
    assert alert.conviction_score > 0
    assert "cluster_buy" in alert.summary  # ABC default evaluate_buy's summary format


def test_score_new_transaction_filtered_out_when_no_cluster(db):
    """A lone insider's buy (no other insider nearby) is filtered_out under
    ClusterBuyStrategy, regardless of size or tenure — cluster requires 2+
    distinct insiders."""
    insider = Insider(cik="001", name="Tom Doe")
    company = Company(cik="002", ticker="AVG", name="Average Corp")
    db.add_all([insider, company])
    db.flush()

    role = InsiderRole(
        insider_id=insider.id, company_id=company.id,
        role_title="Director", is_officer=False, is_director=True,
        is_ten_percent_owner=False,
    )
    db.add(role)
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-004", filing_date=date(2024, 6, 1),
        transaction_date=date(2024, 6, 1), transaction_code="P",
        shares=2000, price_per_share=100.0, total_value=200000.0,
        acquired_or_disposed="A", is_discretionary=True,
    )
    db.add(txn)
    db.flush()

    alert = score_new_transaction(txn.id, db)
    assert alert is not None
    assert alert.alert_type == "filtered_out"


def test_score_non_discretionary_returns_none(db):
    insider = Insider(cik="001", name="Jane")
    company = Company(cik="002", ticker="T", name="Test")
    db.add_all([insider, company])
    db.flush()
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-002", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="A",
        shares=100, acquired_or_disposed="A", is_discretionary=False
    )
    db.add(txn)
    db.flush()
    assert score_new_transaction(txn.id, db) is None


def test_unknown_plan_status_still_generates_alert(db):
    """Transactions with NULL plan status (pre-2023) should still generate
    an alert normally (unblocked by 10b5-1 status either way)."""
    insider = Insider(cik="001", name="Old Buyer")
    company = Company(cik="002", ticker="OLD", name="Old Corp")
    db.add_all([insider, company])
    db.flush()

    role = InsiderRole(
        insider_id=insider.id, company_id=company.id,
        role_title="CEO", is_officer=True, is_director=False,
        is_ten_percent_owner=False,
        first_filing_date=date(2020, 1, 1), last_filing_date=date(2022, 6, 1)
    )
    db.add(role)
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-OLD-001", filing_date=date(2022, 6, 1),
        transaction_date=date(2022, 6, 1), transaction_code="P",
        shares=1000, price_per_share=100.0, total_value=100000.0,
        acquired_or_disposed="A", is_discretionary=True,
        is_10b5_1_plan=None,
    )
    db.add(txn)
    db.flush()

    alert = score_new_transaction(txn.id, db)
    assert alert is not None  # NULL plan status should not block signal generation


# ---------------------------------------------------------------------------
# Live detector functions (form4lab.strategy.features) — generic detectors
# that live in the features module, exercised here via signal_generator.
# ---------------------------------------------------------------------------

def test_detect_drawdown_found(db):
    """Drawdown detected when stock fell > 15% over 60 trading days."""
    for i in range(70):
        price = 100.0 - (i * 20.0 / 69)  # linearly drop from 100 to ~80
        db.add(PriceData(
            ticker="DD", date=date(2024, 1, 1) + timedelta(days=i),
            open=price, high=price, low=price, close=price,
            adj_close=price, volume=100000,
        ))
    db.flush()

    result = detect_drawdown("DD", date(2024, 3, 11), db)  # day 70
    assert result is not None
    assert result["is_drawdown"] is True
    assert result["prior_return_60td"] < -0.15


def test_detect_drawdown_not_found(db):
    """No drawdown when stock is flat."""
    for i in range(70):
        db.add(PriceData(
            ticker="FLAT", date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0, high=100.0, low=100.0, close=100.0,
            adj_close=100.0, volume=100000,
        ))
    db.flush()

    result = detect_drawdown("FLAT", date(2024, 3, 11), db)
    assert result is None


def test_detect_drawdown_insufficient_data(db):
    """Returns None when not enough price data."""
    for i in range(30):  # only 30 days
        db.add(PriceData(
            ticker="SHORT", date=date(2024, 1, 1) + timedelta(days=i),
            open=50.0, high=50.0, low=50.0, close=50.0,
            adj_close=50.0, volume=100000,
        ))
    db.flush()

    result = detect_drawdown("SHORT", date(2024, 1, 31), db)
    assert result is None


def test_detect_drawdown_includes_still_falling(db):
    """Drawdown dict should include still_falling key when stock is falling."""
    for i in range(70):
        price = 100.0 - (i * 25.0 / 69)  # drop from 100 to ~75
        db.add(PriceData(
            ticker="SFDD", date=date(2024, 1, 1) + timedelta(days=i),
            open=price, high=price, low=price, close=price,
            adj_close=price, volume=100000,
        ))
    db.flush()

    result = detect_drawdown("SFDD", date(2024, 3, 11), db)
    assert result is not None
    assert "still_falling" in result
    assert "short_momentum_5d" in result
    assert result["still_falling"] is True
    assert result["short_momentum_5d"] < 0


def test_detect_drawdown_stabilizing(db):
    """Drawdown dict should have still_falling=False when stock stabilizes."""
    for i in range(70):
        if i < 50:
            price = 100.0 - (i * 30.0 / 49)  # drop from 100 to ~70
        else:
            price = 70.0  # stabilize at 70
        db.add(PriceData(
            ticker="STAB", date=date(2024, 1, 1) + timedelta(days=i),
            open=price, high=price, low=price, close=price,
            adj_close=price, volume=100000,
        ))
    db.flush()

    result = detect_drawdown("STAB", date(2024, 3, 11), db)
    assert result is not None
    assert result["still_falling"] is False
    assert result["short_momentum_5d"] is not None
    assert abs(result["short_momentum_5d"]) < 0.01


def test_detect_short_momentum_negative(db):
    """Stock falling over 5 trading days should return negative float."""
    for i in range(10):
        price = 100.0 - (i * 3.0)  # drops from 100 to ~73
        db.add(PriceData(
            ticker="FALL", date=date(2024, 1, 1) + timedelta(days=i),
            open=price, high=price, low=price, close=price,
            adj_close=price, volume=100000,
        ))
    db.flush()

    result = detect_short_momentum("FALL", date(2024, 1, 10), db)
    assert result is not None
    assert result < 0


def test_detect_short_momentum_positive(db):
    """Stock rising over 5 trading days should return positive float."""
    for i in range(10):
        price = 100.0 + (i * 2.0)  # rises from 100 to ~118
        db.add(PriceData(
            ticker="RISE", date=date(2024, 1, 1) + timedelta(days=i),
            open=price, high=price, low=price, close=price,
            adj_close=price, volume=100000,
        ))
    db.flush()

    result = detect_short_momentum("RISE", date(2024, 1, 10), db)
    assert result is not None
    assert result > 0


def test_detect_short_momentum_insufficient_data(db):
    """Should return None when fewer than 6 prices available."""
    for i in range(4):
        db.add(PriceData(
            ticker="FEW", date=date(2024, 1, 1) + timedelta(days=i),
            open=50.0, high=50.0, low=50.0, close=50.0,
            adj_close=50.0, volume=100000,
        ))
    db.flush()

    result = detect_short_momentum("FEW", date(2024, 1, 5), db)
    assert result is None


def test_detect_cluster_sell(db):
    """Should detect multiple insiders selling within 7 days."""
    insider1 = Insider(cik="CS1", name="Cluster Sell 1")
    insider2 = Insider(cik="CS2", name="Cluster Sell 2")
    company = Company(cik="CS3", ticker="CST", name="Cluster Sell Corp")
    db.add_all([insider1, insider2, company])
    db.flush()

    db.add(Transaction(
        insider_id=insider1.id, company_id=company.id,
        accession_number="CS-ACC-1", filing_date=date(2024, 6, 1),
        transaction_date=date(2024, 6, 1), transaction_code="S",
        shares=500, acquired_or_disposed="D", is_discretionary=False,
    ))
    db.add(Transaction(
        insider_id=insider2.id, company_id=company.id,
        accession_number="CS-ACC-2", filing_date=date(2024, 6, 3),
        transaction_date=date(2024, 6, 3), transaction_code="S",
        shares=300, acquired_or_disposed="D", is_discretionary=False,
    ))
    db.flush()

    result = detect_cluster_sell(company.id, date(2024, 6, 2), db)
    assert result["unique_insiders"] == 2


def test_compute_sell_pct(db):
    """Should compute fraction of holdings sold."""
    insider = Insider(cik="PCT1", name="Pct Test")
    company = Company(cik="PCT2", ticker="PCT", name="Pct Corp")
    db.add_all([insider, company])
    db.flush()

    # Sell 3000 of 10000 (3000 + 7000 remaining) = 30%
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="PCT-ACC-1", filing_date=date(2024, 6, 1),
        transaction_date=date(2024, 6, 1), transaction_code="S",
        shares=3000, price_per_share=50.0, total_value=150000.0,
        shares_owned_after=7000,
        acquired_or_disposed="D", is_discretionary=False,
    )
    db.add(txn)
    db.flush()

    pct = compute_sell_pct(insider.id, company.id, date(2024, 6, 1), db)
    assert pct is not None
    assert abs(pct - 0.30) < 0.01


# ---------------------------------------------------------------------------
# score_sell_transaction — mechanism (dedup, value computation, delegation)
# ---------------------------------------------------------------------------

class _SellCapableStrategy(Strategy):
    """Flags every S transaction as a sell signal, independent of tier or
    role — lets the generic sell-scoring plumbing be tested without any
    specific strategy's sell logic. The shipped ClusterBuyStrategy declares
    zero sell signal types (see test below), so it can't exercise this path
    at all."""
    name = "sell_capable"

    def signal_types(self):
        return [SignalType("sell_watch", direction="sell", tradeable=False, visible=True)]

    def classify(self, txn, f):
        return None  # buy side unused here

    def evaluate_sell(self, txn, f):
        return SellEvaluation(alert_type="sell_watch", conviction=1.0,
                              summary=f"SELL WATCH: {txn.ticker} on {txn.transaction_date}")


def _activate_sell_capable_strategy(monkeypatch):
    import form4lab.strategy.registry as reg
    monkeypatch.setattr(reg.settings, "strategy_path", "tests.test_signal_generator:_SellCapableStrategy")
    reg.get_active(refresh=True)


def _reset_active_strategy():
    import form4lab.strategy.registry as reg
    reg._active = None


def _make_sell_fixture(db):
    insider = Insider(cik="SELL1", name="Sell Insider")
    company = Company(cik="SELL2", ticker="SELL", name="Sell Corp", sector="Technology")
    db.add_all([insider, company])
    db.flush()
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="SELL-ACC-1", filing_date=date(2024, 6, 1),
        transaction_date=date(2024, 6, 1), transaction_code="S",
        shares=500, price_per_share=100.0, total_value=50000.0,
        shares_owned_after=10000,
        acquired_or_disposed="D", is_discretionary=False,
    )
    db.add(txn)
    db.flush()
    return insider, company, txn


def test_score_sell_transaction_delegates_to_strategy(db, monkeypatch):
    """With a strategy that DOES flag sells, score_sell_transaction resolves
    entities/value and creates the alert the strategy's evaluate_sell asked for."""
    _activate_sell_capable_strategy(monkeypatch)
    try:
        insider, company, txn = _make_sell_fixture(db)
        alert = score_sell_transaction(txn.id, db)
        assert alert is not None
        assert alert.alert_type == "sell_watch"
        assert alert.transaction_value == pytest.approx(50000.0)
        assert "SELL WATCH" in alert.summary
    finally:
        _reset_active_strategy()


def test_score_sell_dedup(db, monkeypatch):
    """Should not create duplicate sell alerts for the same event."""
    _activate_sell_capable_strategy(monkeypatch)
    try:
        insider, company, txn = _make_sell_fixture(db)
        alert1 = score_sell_transaction(txn.id, db)
        assert alert1 is not None

        alert2 = score_sell_transaction(txn.id, db)
        assert alert2 is None
    finally:
        _reset_active_strategy()


def test_score_sell_ignores_buy_transactions(db, monkeypatch):
    """score_sell_transaction should return None for P-buy transactions,
    even with a strategy that flags every sell."""
    _activate_sell_capable_strategy(monkeypatch)
    try:
        insider = Insider(cik="B1", name="Buyer")
        company = Company(cik="B2", ticker="BUYT", name="Buy Corp")
        db.add_all([insider, company])
        db.flush()
        txn = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number="SELL-BUY-1", filing_date=date(2024, 6, 1),
            transaction_date=date(2024, 6, 1), transaction_code="P",
            shares=100, price_per_share=100.0, total_value=10000.0,
            acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        db.flush()

        alert = score_sell_transaction(txn.id, db)
        assert alert is None
    finally:
        _reset_active_strategy()


def test_score_sell_returns_none_with_default_strategy(db):
    """The shipped ClusterBuyStrategy declares no sell signal types at all
    (evaluate_sell uses the ABC default, which always returns None) — sells
    never generate alerts under the default configuration."""
    insider, company, txn = _make_sell_fixture(db)
    assert score_sell_transaction(txn.id, db) is None
