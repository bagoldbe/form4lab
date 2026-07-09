from datetime import date

import pandas as pd

from form4lab.strategy import registry as reg
from form4lab.strategy.base import EntryContext, TxnView
from form4lab.strategy.registry import SignalRegistry
from form4lab.strategies.big_exec_buy import BigExecBuyStrategy


class DictFeatures:
    """FeatureView double — mirrors tests/test_cluster_buy_strategy.py."""

    def __init__(self, d):
        self._d = d

    def get(self, name, default=None):
        return self._d.get(name, default)

    def put(self, name, value):
        self._d[name] = value


def _txn(role_title="CEO", txn_value=150_000.0):
    return TxnView(insider_id=1, company_id=2, ticker="ABC",
                   transaction_date=date(2025, 1, 2), txn_value=txn_value,
                   role_title=role_title)


# --- classify() ---

def test_classify_big_exec_buy_when_ceo_meets_thresholds():
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("CEO", 150_000.0), f) == "big_exec_buy"


def test_classify_big_exec_buy_when_cfo_meets_thresholds():
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("CFO", 150_000.0), f) == "big_exec_buy"


def test_classify_big_exec_buy_at_exact_threshold():
    """Boundary: txn_value == 100_000 (inclusive) still qualifies."""
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("CEO", 100_000.0), f) == "big_exec_buy"


def test_classify_filtered_out_for_director_role():
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("Director", 150_000.0), f) == "filtered_out"


def test_classify_filtered_out_for_vp_role():
    """VP is deliberately excluded from is_ceo/is_cfo (VICE guard in form4lab/utils.py)."""
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("VP", 150_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_value_too_low():
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("CEO", 50_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_10b5_1_plan_true():
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": True})
    assert s.classify(_txn("CEO", 150_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_10b5_1_plan_truthy_non_bool():
    """Gate correctness: classify() must gate 10b5-1 with TRUTHINESS
    (`not f.get(...)`), never identity (`f.get(...) is not True`). A non-bool
    truthy value must still block the signal -- an identity check would
    incorrectly let it through, since `1 is not True`."""
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": 1})
    assert s.classify(_txn("CEO", 150_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_feature_missing_defaults_to_falsy():
    """f.get("is_10b5_1_plan") -- an absent key must not raise and must not
    block (missing/NULL is treated as "not a 10b5-1 plan", matching
    RowFeatureView's NULL->False parity semantics)."""
    s = BigExecBuyStrategy()
    assert s.classify(_txn("CEO", 150_000.0), DictFeatures({})) == "big_exec_buy"


def test_classify_filtered_out_when_txn_value_is_none():
    """(txn.txn_value or 0.0) -- None must not raise a TypeError on comparison."""
    s = BigExecBuyStrategy()
    f = DictFeatures({"is_10b5_1_plan": False})
    assert s.classify(_txn("CEO", None), f) == "filtered_out"


# --- classify_row() — backtest adapter (live/backtest feature-name divergence) ---
#
# Backtest rows carry role_title, total_value, and is_10b5_1_plan (a
# feature resolved on both views) directly on the row. RowFeatureView
# normalizes is_10b5_1_plan to a native Python bool (NULL/NaN -> False),
# matching LiveFeatureView -- this section exercises that path end to end.

def _row(role_title="CEO", total_value=150_000.0, is_10b5_1_plan=False):
    return pd.Series({
        "role_title": role_title,
        "total_value": total_value,
        "is_10b5_1_plan": is_10b5_1_plan,
        "insider_id": 1,
        "ticker": "ABC",
        "transaction_date": date(2025, 1, 2),
    })


def test_classify_row_big_exec_buy_when_thresholds_met():
    s = BigExecBuyStrategy()
    assert s.classify_row(_row(), "Average", 0.0) == ("big_exec_buy", "Average", 0.0)


def test_classify_row_filtered_out_when_10b5_1_plan_true():
    s = BigExecBuyStrategy()
    assert s.classify_row(_row(is_10b5_1_plan=True), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_filtered_out_when_value_too_low():
    s = BigExecBuyStrategy()
    assert s.classify_row(_row(total_value=50_000.0), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_filtered_out_when_role_is_director():
    s = BigExecBuyStrategy()
    assert s.classify_row(_row(role_title="Director"), "Average", 0.0) == ("filtered_out", "Average", 0.0)


def test_classify_row_big_exec_buy_when_10b5_1_plan_is_none():
    """NULL is_10b5_1_plan coalesces to False in RowFeatureView, so it does
    not block an otherwise-qualifying big-exec buy."""
    s = BigExecBuyStrategy()
    assert s.classify_row(_row(is_10b5_1_plan=None), "Average", 0.0) == ("big_exec_buy", "Average", 0.0)


# --- registry wiring ---

def test_registry_tradeable_and_hidden_names():
    r = SignalRegistry(BigExecBuyStrategy())
    assert r.tradeable_names() == frozenset({"big_exec_buy"})
    assert r.hidden_names() == frozenset({"filtered_out"})


def test_registry_hold_days_and_priority():
    r = SignalRegistry(BigExecBuyStrategy())
    assert r.hold_days("big_exec_buy", default=99) == 60
    st = r.get("big_exec_buy")
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
    s = BigExecBuyStrategy()
    assert s.allow_entry(_ctx(2, 0)) == "ticker_limit"


def test_allow_entry_allows_when_below_limits():
    s = BigExecBuyStrategy()
    assert s.allow_entry(_ctx(0, 0)) is None


# --- resolve: strategy_path wiring (boot-style, mirrors
# tests/test_strategy_registry.py::test_get_active_singleton_and_refresh) ---

def test_get_active_resolves_big_exec_buy(monkeypatch):
    monkeypatch.setattr(reg.settings, "strategy_path",
                        "form4lab.strategies.big_exec_buy:BigExecBuyStrategy")
    try:
        strategy, registry = reg.get_active(refresh=True)
        assert strategy.name == "big_exec_buy"
        assert registry.tradeable_names() == frozenset({"big_exec_buy"})
        assert registry.hidden_names() == frozenset({"filtered_out"})
    finally:
        reg._active = None  # next get_active() re-resolves from the real strategy_path after monkeypatch reverts
