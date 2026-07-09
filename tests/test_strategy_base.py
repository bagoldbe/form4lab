from datetime import date

import pandas as pd
import pytest

from form4lab.strategy.base import (
    BuyEvaluation, EntryContext, FeatureView, SignalType, SizeDecision,
    SizingContext, Strategy, TxnView,
)


class DictFeatures:
    def __init__(self, d):
        self._d = d

    def get(self, name, default=None):
        return self._d.get(name, default)

    def put(self, name, value):
        self._d[name] = value


class ToyStrategy(Strategy):
    name = "toy"

    def signal_types(self):
        return [
            SignalType("toy_buy", tradeable=True, hold_days=20, priority=10),
            SignalType("toy_noise", visible=False),
        ]

    def classify(self, txn, f):
        return "toy_buy" if f.get("go") else None


def _txn():
    return TxnView(insider_id=1, company_id=2, ticker="ABC",
                   transaction_date=date(2025, 1, 2))


def test_signal_type_defaults_and_frozen():
    st = SignalType("x")
    assert (st.direction, st.tradeable, st.hold_days, st.visible) == ("buy", False, 60, True)
    with pytest.raises(Exception):
        st.name = "y"


def test_classify_and_default_evaluate_buy():
    s = ToyStrategy()
    ev = s.evaluate_buy(_txn(), DictFeatures({"go": True}))
    assert isinstance(ev, BuyEvaluation)
    assert ev.alert_type == "toy_buy"
    assert ev.conviction == 1.0
    assert ev.cluster_id is None
    assert s.evaluate_buy(_txn(), DictFeatures({})) is None


def test_default_evaluate_sell_is_none():
    assert ToyStrategy().evaluate_sell(_txn(), DictFeatures({})) is None


def test_default_size_is_five_pct():
    d = ToyStrategy().size(SizingContext(equity=10_000.0, ticker="ABC", role_title=None))
    assert isinstance(d, SizeDecision)
    assert d.dollars == 500.0
    assert d.method == "role"


def test_default_allow_entry_is_open():
    ctx = EntryContext(ticker="ABC", role_title=None, insider_id=1,
                       open_positions_in_ticker=0, open_positions_for_insider_ticker=0)
    assert ToyStrategy().allow_entry(ctx) is None


def test_default_classify_row_derives_from_classify():
    """ToyStrategy doesn't override classify_row, so the ABC default applies:
    it builds a TxnView + RowFeatureView from the row and delegates to
    classify(), translating None -> "skip"."""
    row_go = pd.Series({"insider_id": 1, "ticker": "ABC",
                        "transaction_date": pd.Timestamp("2025-01-02"), "go": True})
    assert ToyStrategy().classify_row(row_go, "Elite", 1.5) == ("toy_buy", "Elite", 1.5)

    row_no_go = pd.Series({"insider_id": 1, "ticker": "ABC",
                          "transaction_date": pd.Timestamp("2025-01-02"), "go": False})
    assert ToyStrategy().classify_row(row_no_go, "Elite", 1.5) == ("skip", "Elite", 1.5)
