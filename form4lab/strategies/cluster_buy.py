"""Cluster-buy example strategy — a naive, well-documented illustration.

Multiple distinct insiders buying the same stock within a short window is one
of the most-replicated signals in the insider-trading literature. Lakonishok
& Lee (2001, "Are Insider Trades Informative?", Review of Financial Studies)
find that insider trades aggregated across multiple insiders carry more
information about future returns than isolated single-insider trades. The
numbers below are pedagogical defaults, NOT researched or validated for live
use.
"""
import logging

from form4lab.config import settings
from form4lab.strategy.base import (
    EntryContext, FeatureView, SignalType, Strategy, TxnView,
)

logger = logging.getLogger(__name__)
_alp = settings.alpaca

MIN_UNIQUE_INSIDERS = 2      # illustrative
MIN_TXN_VALUE = 25_000.0     # illustrative


class ClusterBuyStrategy(Strategy):
    name = "cluster_buy"

    def signal_types(self) -> list[SignalType]:
        return [
            SignalType("cluster_buy", tradeable=True, hold_days=60, priority=50,
                       description="2+ distinct insiders bought within the cluster window"),
            SignalType("filtered_out", priority=0, visible=False),
        ]

    def classify(self, txn: TxnView, f: FeatureView) -> str | None:
        if (f.get("cluster_unique_insiders", 0) >= MIN_UNIQUE_INSIDERS
                and (txn.txn_value or 0.0) >= MIN_TXN_VALUE):
            return "cluster_buy"
        return "filtered_out"

    def allow_entry(self, ctx: EntryContext) -> str | None:
        # counts are -1 at the pre-entry universe check (unknown) — apply
        # count-based gates only when real counts are supplied
        if ctx.open_positions_for_insider_ticker >= 0 and ctx.open_positions_for_insider_ticker >= _alp.max_positions_per_insider_ticker:
            return "insider_ticker_limit"
        if ctx.open_positions_in_ticker >= 0 and ctx.open_positions_in_ticker >= _alp.max_positions_per_ticker:
            return "ticker_limit"
        return None
