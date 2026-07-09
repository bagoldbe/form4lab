from datetime import date

import pandas as pd

from form4lab.strategy import registry as reg
from form4lab.strategy.base import EntryContext, TxnView
from form4lab.strategy.registry import SignalRegistry
from form4lab.strategies.opportunistic_first_buy import OpportunisticFirstBuyStrategy


class DictFeatures:
    """FeatureView double — mirrors tests/test_cluster_buy_strategy.py."""

    def __init__(self, d):
        self._d = d

    def get(self, name, default=None):
        return self._d.get(name, default)

    def put(self, name, value):
        self._d[name] = value


def _txn(txn_value=40_000.0):
    return TxnView(insider_id=1, company_id=2, ticker="ABC",
                   transaction_date=date(2025, 1, 2), txn_value=txn_value,
                   role_title="Director")


# --- classify() ---

def test_classify_opportunistic_first_buy_when_qualifies():
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": False})
    assert s.classify(_txn(40_000.0), f) == "opportunistic_first_buy"


def test_classify_opportunistic_first_buy_at_exact_threshold():
    """Boundary: txn_value == 25_000 (inclusive) still qualifies."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": False})
    assert s.classify(_txn(25_000.0), f) == "opportunistic_first_buy"


def test_classify_filtered_out_when_is_first_buy_false():
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": False, "is_10b5_1_plan": False})
    assert s.classify(_txn(40_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_value_too_low():
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": False})
    assert s.classify(_txn(10_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_10b5_1_plan_true():
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": True})
    assert s.classify(_txn(40_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_10b5_1_plan_truthy_non_bool():
    """Gate correctness: classify() must gate 10b5-1 with TRUTHINESS
    (`not f.get(...)`), never identity (`f.get(...) is not True`). A non-bool
    truthy value must still block the signal -- an identity check would
    incorrectly let it through, since `1 is not True`."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": 1})
    assert s.classify(_txn(40_000.0), f) == "filtered_out"


def test_classify_opportunistic_first_buy_when_is_first_buy_truthy_non_bool():
    """Gate correctness (mirror image of the 10b5-1 pin above): classify()
    must gate is_first_buy with TRUTHINESS (`if f.get(...)`), never identity
    (`f.get(...) is True`). A non-bool truthy value must still let the
    signal through -- an identity check would incorrectly block it, since
    `1 is not True`."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": 1, "is_10b5_1_plan": False})
    assert s.classify(_txn(40_000.0), f) == "opportunistic_first_buy"


def test_classify_filtered_out_when_is_first_buy_missing():
    """f.get("is_first_buy") -- an absent key must not raise and must
    default to falsy, correctly BLOCKING the signal (missing/NULL is not a
    first buy -- the inverse default-direction from is_10b5_1_plan below)."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn(40_000.0), f) == "filtered_out"


def test_classify_opportunistic_first_buy_when_10b5_1_plan_missing_defaults_falsy():
    """f.get("is_10b5_1_plan") -- an absent key must not raise and must not
    block (missing/NULL is treated as "not a 10b5-1 plan", matching
    RowFeatureView's NULL->False parity semantics)."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True})
    assert s.classify(_txn(40_000.0), f) == "opportunistic_first_buy"


def test_classify_filtered_out_when_txn_value_is_none():
    """(txn.txn_value or 0.0) -- None must not raise a TypeError on comparison."""
    s = OpportunisticFirstBuyStrategy()
    f = DictFeatures({"is_first_buy": True, "is_10b5_1_plan": False})
    assert s.classify(_txn(None), f) == "filtered_out"


# --- classify_row() — backtest adapter (live/backtest feature-name divergence) ---
#
# Backtest rows carry total_value, is_first_buy, and is_10b5_1_plan
# (features resolved on both views) directly on the row. RowFeatureView
# normalizes both to a native Python bool (NULL/NaN -> False), matching
# LiveFeatureView -- this section exercises that path end to end.

def _row(is_first_buy=True, total_value=40_000.0, is_10b5_1_plan=False):
    return pd.Series({
        "is_first_buy": is_first_buy,
        "total_value": total_value,
        "is_10b5_1_plan": is_10b5_1_plan,
        "insider_id": 1,
        "ticker": "ABC",
        "transaction_date": date(2025, 1, 2),
    })


def test_classify_row_opportunistic_first_buy_when_thresholds_met():
    s = OpportunisticFirstBuyStrategy()
    assert s.classify_row(_row(), "Average", 0.0) == ("opportunistic_first_buy", "Average", 0.0)


def test_classify_row_filtered_out_when_10b5_1_plan_true():
    s = OpportunisticFirstBuyStrategy()
    assert s.classify_row(_row(is_10b5_1_plan=True), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_filtered_out_when_value_too_low():
    s = OpportunisticFirstBuyStrategy()
    assert s.classify_row(_row(total_value=10_000.0), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_filtered_out_when_is_first_buy_false():
    s = OpportunisticFirstBuyStrategy()
    assert s.classify_row(_row(is_first_buy=False), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_opportunistic_first_buy_when_10b5_1_plan_is_none():
    """NULL is_10b5_1_plan coalesces to False in RowFeatureView, so it does
    not block an otherwise-qualifying first buy."""
    s = OpportunisticFirstBuyStrategy()
    assert s.classify_row(_row(is_10b5_1_plan=None), "Average", 0.0) == ("opportunistic_first_buy", "Average", 0.0)


# --- registry wiring ---

def test_registry_tradeable_and_hidden_names():
    r = SignalRegistry(OpportunisticFirstBuyStrategy())
    assert r.tradeable_names() == frozenset({"opportunistic_first_buy"})
    assert r.hidden_names() == frozenset({"filtered_out"})


def test_registry_hold_days_and_priority():
    r = SignalRegistry(OpportunisticFirstBuyStrategy())
    assert r.hold_days("opportunistic_first_buy", default=99) == 60
    st = r.get("opportunistic_first_buy")
    assert st.priority == 50
    assert st.tradeable is True


# --- allow_entry() -- smoke coverage; the gate logic itself is the shared
# ClusterBuyStrategy pattern already fully parametrized in
# tests/test_cluster_buy_strategy.py, so this just confirms it is wired up
# on this class too. ---

def _ctx(open_positions_in_ticker, open_positions_for_insider_ticker):
    return EntryContext(ticker="ABC", role_title=None, insider_id=1,
                        open_positions_in_ticker=open_positions_in_ticker,
                        open_positions_for_insider_ticker=open_positions_for_insider_ticker)


def test_allow_entry_blocks_at_ticker_limit():
    """max_positions_per_ticker default = 2."""
    s = OpportunisticFirstBuyStrategy()
    assert s.allow_entry(_ctx(2, 0)) == "ticker_limit"


def test_allow_entry_allows_when_below_limits():
    s = OpportunisticFirstBuyStrategy()
    assert s.allow_entry(_ctx(0, 0)) is None


# --- resolve: strategy_path wiring (boot-style, mirrors
# tests/test_strategy_registry.py::test_get_active_singleton_and_refresh) ---

def test_get_active_resolves_opportunistic_first_buy(monkeypatch):
    monkeypatch.setattr(reg.settings, "strategy_path",
                        "form4lab.strategies.opportunistic_first_buy:OpportunisticFirstBuyStrategy")
    try:
        strategy, registry = reg.get_active(refresh=True)
        assert strategy.name == "opportunistic_first_buy"
        assert registry.tradeable_names() == frozenset({"opportunistic_first_buy"})
        assert registry.hidden_names() == frozenset({"filtered_out"})
    finally:
        reg._active = None  # next get_active() re-resolves from the real strategy_path after monkeypatch reverts
