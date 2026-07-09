from datetime import date

import pandas as pd

from form4lab.strategy.base import EntryContext, TxnView
from form4lab.strategy.features import RowFeatureView
from form4lab.strategy.registry import SignalRegistry
from form4lab.strategies.cluster_buy import ClusterBuyStrategy


class DictFeatures:
    """FeatureView double backed by a plain dict, for classify()-level unit tests."""

    def __init__(self, d):
        self._d = d

    def get(self, name, default=None):
        return self._d.get(name, default)

    def put(self, name, value):
        self._d[name] = value


def _txn(txn_value=30_000.0):
    return TxnView(insider_id=1, company_id=2, ticker="ABC",
                   transaction_date=date(2025, 1, 2), txn_value=txn_value)


# --- classify() ---

def test_classify_cluster_buy_when_thresholds_met():
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 2})
    assert s.classify(_txn(30_000.0), f) == "cluster_buy"


def test_classify_cluster_buy_at_exact_thresholds():
    """Boundary: >=2 insiders and >=25_000 value (both inclusive) still qualify."""
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 2})
    assert s.classify(_txn(25_000.0), f) == "cluster_buy"


def test_classify_filtered_out_when_insufficient_insiders():
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 1})
    assert s.classify(_txn(30_000.0), f) == "filtered_out"


def test_classify_filtered_out_when_value_too_low():
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 2})
    assert s.classify(_txn(24_999.99), f) == "filtered_out"


def test_classify_filtered_out_when_both_fail():
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 0})
    assert s.classify(_txn(0.0), f) == "filtered_out"


def test_classify_filtered_out_when_feature_missing_defaults_to_zero():
    """f.get("cluster_unique_insiders", 0) — an absent key must not raise."""
    s = ClusterBuyStrategy()
    assert s.classify(_txn(30_000.0), DictFeatures({})) == "filtered_out"


def test_classify_filtered_out_when_txn_value_is_none():
    """(txn.txn_value or 0.0) — None must not raise a TypeError on comparison."""
    s = ClusterBuyStrategy()
    f = DictFeatures({"cluster_unique_insiders": 2})
    assert s.classify(_txn(None), f) == "filtered_out"


# --- classify_row() — backtest adapter (live/backtest feature-name divergence) ---
#
# Backtest rows (from form4lab.scoring.flags.compute_cluster_flags) carry
# cluster_size, never the live-only cluster_unique_insiders name classify()
# reads — RowFeatureView must alias the two. Rows also carry total_value
# (not txn_value), which classify_row's TxnView construction must thread
# through as txn_value so the dollar-value gate isn't permanently starved.

def _row(cluster_size, total_value=50_000.0):
    return pd.Series({
        "cluster_size": cluster_size,
        "total_value": total_value,
        "insider_id": 1,
        "ticker": "ABC",
        "transaction_date": date(2025, 1, 2),
    })


def test_classify_row_cluster_buy_when_thresholds_met():
    s = ClusterBuyStrategy()
    assert s.classify_row(_row(cluster_size=3), "Average", 0.0) == ("cluster_buy", "Average", 0.0)


def test_classify_row_filtered_out_when_cluster_size_insufficient():
    s = ClusterBuyStrategy()
    assert s.classify_row(_row(cluster_size=1), "Average", 0.0) == ("filtered_out", "Average", 0.0)


# --- classify_row() threads company_id/role_title (backtest rows carry both:
# company_id from the SELECT, role_title from the roles merge) into the
# TxnView, mirroring the existing txn_value threading above. ---

class _CapturingStrategy(ClusterBuyStrategy):
    """classify() records the TxnView it receives instead of classifying, so
    the test can inspect exactly what classify_row() constructed."""

    def __init__(self):
        self.captured: TxnView | None = None

    def classify(self, txn, f):
        self.captured = txn
        return None


def test_classify_row_threads_company_id_and_role_title():
    s = _CapturingStrategy()
    row = pd.Series({
        "cluster_size": 3,
        "total_value": 1.0,
        "insider_id": 1,
        "company_id": 77,
        "role_title": "CEO",
        "ticker": "ABC",
        "transaction_date": date(2025, 1, 2),
    })
    s.classify_row(row, "Average", 0.0)
    assert s.captured.company_id == 77
    assert s.captured.role_title == "CEO"


# --- RowFeatureView.get() cluster_unique_insiders alias — None-coalescing
# must match the generic fall-through two lines below it: an explicit None
# stored under "cluster_size" degrades to `default`, not None. ---

def test_row_feature_view_cluster_alias_coalesces_none_to_default():
    f = RowFeatureView(pd.Series({"cluster_size": None}), "Average", 0.0)
    assert f.get("cluster_unique_insiders", 0) == 0


# --- registry wiring ---

def test_registry_tradeable_and_hidden_names():
    r = SignalRegistry(ClusterBuyStrategy())
    assert r.tradeable_names() == frozenset({"cluster_buy"})
    assert r.hidden_names() == frozenset({"filtered_out"})


def test_registry_hold_days_and_priority():
    r = SignalRegistry(ClusterBuyStrategy())
    assert r.hold_days("cluster_buy", default=99) == 60
    st = r.get("cluster_buy")
    assert st.priority == 50
    assert st.tradeable is True


# --- allow_entry() ---

def _ctx(open_positions_in_ticker, open_positions_for_insider_ticker):
    return EntryContext(ticker="ABC", role_title=None, insider_id=1,
                        open_positions_in_ticker=open_positions_in_ticker,
                        open_positions_for_insider_ticker=open_positions_for_insider_ticker)


def test_allow_entry_blocks_at_ticker_limit():
    """max_positions_per_ticker default = 2."""
    s = ClusterBuyStrategy()
    assert s.allow_entry(_ctx(2, 0)) == "ticker_limit"


def test_allow_entry_blocks_at_insider_ticker_limit():
    """max_positions_per_insider_ticker default = 1; checked before ticker_limit."""
    s = ClusterBuyStrategy()
    assert s.allow_entry(_ctx(0, 1)) == "insider_ticker_limit"


def test_allow_entry_allows_when_sentinel_unknown():
    """Both counts at -1 (pre-entry universe check) — no count-gate applies."""
    s = ClusterBuyStrategy()
    assert s.allow_entry(_ctx(-1, -1)) is None


def test_allow_entry_allows_when_below_limits():
    s = ClusterBuyStrategy()
    assert s.allow_entry(_ctx(0, 0)) is None
    assert s.allow_entry(_ctx(1, 0)) is None  # one under the ticker limit


def test_allow_entry_sentinel_is_per_field_independent():
    """Each count field is guarded individually (`>= 0`), not as an atomic pair —
    a real count on one field still gates even if the other is the -1 sentinel."""
    s = ClusterBuyStrategy()
    assert s.allow_entry(_ctx(-1, 1)) == "insider_ticker_limit"
    assert s.allow_entry(_ctx(2, -1)) == "ticker_limit"
