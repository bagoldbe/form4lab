import pytest

from form4lab.strategy import registry as reg
from form4lab.strategy.base import SignalType, Strategy
from tests.test_strategy_base import ToyStrategy


def test_registry_lookups():
    r = reg.SignalRegistry(ToyStrategy())
    assert r.is_tradeable("toy_buy") is True
    assert r.is_tradeable("toy_noise") is False
    assert r.is_tradeable("unknown") is False
    assert r.tradeable_names() == frozenset({"toy_buy"})
    assert r.buy_names() == frozenset({"toy_buy"})
    assert r.hidden_names() == frozenset({"toy_noise"})
    assert r.hold_days("toy_buy", 99) == 20
    assert r.hold_days("unknown", 99) == 99


def test_registry_rejects_duplicate_names():
    class Dupe(ToyStrategy):
        def signal_types(self):
            return [SignalType("a"), SignalType("a")]
    with pytest.raises(ValueError):
        reg.SignalRegistry(Dupe())


def test_load_strategy_from_path():
    s = reg.load_strategy("tests.test_strategy_base:ToyStrategy")
    assert isinstance(s, Strategy) and s.name == "toy"


def test_load_strategy_bad_path_raises():
    with pytest.raises((ImportError, AttributeError)):
        reg.load_strategy("tests.test_strategy_base:Missing")


def test_get_active_singleton_and_refresh(monkeypatch):
    monkeypatch.setattr(reg.settings, "strategy_path", "tests.test_strategy_base:ToyStrategy")
    try:
        reg.get_active(refresh=True)
        s1, r1 = reg.get_active()
        s2, r2 = reg.get_active()
        assert s1 is s2 and r1 is r2
        s3, _ = reg.get_active(refresh=True)
        assert s3 is not s1
    finally:
        reg._active = None  # next get_active() re-resolves from the real strategy_path after monkeypatch reverts
