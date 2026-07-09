"""Opportunistic-first-buy example strategy — a naive, well-documented illustration.

An insider's first observed open-market purchase, excluding 10b5-1-planned
trades, over a small size floor. The opportunistic-vs-routine insider
literature (Cohen, Malloy, Pomorski 2012) finds that trades by insiders
without a predictable, repeating pattern -- "opportunistic" traders -- carry
more information than routine ones; a first-observed buy is a simple proxy
for "not yet part of a routine pattern." The threshold below is a
pedagogical default, NOT researched or validated for live use.

First-buy caveat (read before trusting a backtest run of this strategy):
`is_first_buy` (form4lab/strategy/features.py) means "first observed
discretionary buy by THIS INSIDER" -- insider-scoped, with no company filter
and no tenure gate -- not "first buy ever at this company." Live and
backtest derive it from the same predicate but over different windows: in
BACKTEST it is computed from a frame that INNER JOINs trade_outcomes
(load_backtest_data), so a transaction without a matured outcome yet is
invisible to it -- a genuinely-first buy can therefore leave a later, actual
repeat buy mislabeled "first" at the frame's recent edge. Live reasons over
the insider's complete transaction history with no outcome dependency, so it
has neither gap.
"""
import logging

from form4lab.config import settings
from form4lab.strategy.base import EntryContext, FeatureView, SignalType, Strategy, TxnView

logger = logging.getLogger(__name__)
_alp = settings.alpaca

MIN_TXN_VALUE = 25_000.0   # illustrative


class OpportunisticFirstBuyStrategy(Strategy):
    name = "opportunistic_first_buy"

    def signal_types(self) -> list[SignalType]:
        return [
            SignalType("opportunistic_first_buy", tradeable=True, hold_days=60, priority=50,
                       description="Insider's first observed open-market purchase over the size floor, ex-10b5-1"),
            SignalType("filtered_out", priority=0, visible=False),
        ]

    def classify(self, txn: TxnView, f: FeatureView) -> str | None:
        if (f.get("is_first_buy") and (txn.txn_value or 0.0) >= MIN_TXN_VALUE
                and not f.get("is_10b5_1_plan")):
            return "opportunistic_first_buy"
        return "filtered_out"

    def allow_entry(self, ctx: EntryContext) -> str | None:
        # counts are -1 at the pre-entry universe check (unknown) — apply
        # count-based gates only when real counts are supplied
        if ctx.open_positions_for_insider_ticker >= 0 and ctx.open_positions_for_insider_ticker >= _alp.max_positions_per_insider_ticker:
            return "insider_ticker_limit"
        if ctx.open_positions_in_ticker >= 0 and ctx.open_positions_in_ticker >= _alp.max_positions_per_ticker:
            return "ticker_limit"
        return None
