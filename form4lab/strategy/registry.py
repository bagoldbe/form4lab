"""Active-strategy resolution and the signal-type registry.

The registry is the single source of truth for which alert_type strings
exist, which are tradeable, and their hold periods. All read-side filtering
(dashboard, alert service, backtester defaults) derives from it.
"""
import importlib
import logging

from form4lab.config import settings
from form4lab.strategy.base import SignalType, Strategy

logger = logging.getLogger(__name__)


class SignalRegistry:
    def __init__(self, strategy: Strategy):
        self._types: dict[str, SignalType] = {}
        for st in strategy.signal_types():
            if st.name in self._types:
                raise ValueError(f"duplicate signal type name: {st.name!r}")
            self._types[st.name] = st

    def get(self, name: str) -> SignalType | None:
        return self._types.get(name)

    def is_tradeable(self, name: str) -> bool:
        st = self._types.get(name)
        return bool(st and st.tradeable)

    def tradeable_names(self) -> frozenset[str]:
        return frozenset(n for n, st in self._types.items() if st.tradeable)

    def buy_names(self) -> frozenset[str]:
        return frozenset(n for n, st in self._types.items()
                         if st.direction == "buy" and st.visible)

    def sell_names(self) -> frozenset[str]:
        return frozenset(n for n, st in self._types.items()
                         if st.direction == "sell" and st.visible)

    def hidden_names(self) -> frozenset[str]:
        return frozenset(n for n, st in self._types.items() if not st.visible)

    def hold_days(self, name: str, default: int) -> int:
        st = self._types.get(name)
        return st.hold_days if st else default


def load_strategy(path: str | None = None) -> Strategy:
    path = path or settings.strategy_path
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise ImportError(f"strategy path {path!r} must be 'pkg.module:ClassName'")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


_active: tuple[Strategy, SignalRegistry] | None = None


def get_active(refresh: bool = False) -> tuple[Strategy, SignalRegistry]:
    global _active
    if _active is None or refresh:
        strategy = load_strategy()
        _active = (strategy, SignalRegistry(strategy))
        logger.info("Active strategy: %s (%d signal types)",
                    strategy.name, len(strategy.signal_types()))
    return _active
