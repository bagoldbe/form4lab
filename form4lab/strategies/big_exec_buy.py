"""Big-executive-buy example strategy — a naive, well-documented illustration.

Open-market purchases by the CEO or CFO are among the most-studied insider
signals. Seyhun (1986, "Insiders' Profits, Costs of Trading, and Market
Efficiency", Journal of Financial Economics) finds that senior insiders'
trades are more informative than junior insiders' trades, consistent with
greater access to material information. The thresholds below are pedagogical
defaults, NOT researched or validated for live use.
"""
import logging

from form4lab.config import settings
from form4lab.strategy.base import EntryContext, FeatureView, SignalType, Strategy, TxnView
from form4lab.utils import is_ceo, is_cfo

logger = logging.getLogger(__name__)
_alp = settings.alpaca

MIN_TXN_VALUE = 100_000.0   # illustrative


class BigExecBuyStrategy(Strategy):
    name = "big_exec_buy"

    def signal_types(self) -> list[SignalType]:
        return [
            SignalType("big_exec_buy", tradeable=True, hold_days=60, priority=50,
                       description="CEO/CFO open-market purchase over the size threshold, ex-10b5-1"),
            SignalType("filtered_out", priority=0, visible=False),
        ]

    def classify(self, txn: TxnView, f: FeatureView) -> str | None:
        is_senior = is_ceo(txn.role_title) or is_cfo(txn.role_title)
        if (is_senior and (txn.txn_value or 0.0) >= MIN_TXN_VALUE
                and not f.get("is_10b5_1_plan")):
            return "big_exec_buy"
        return "filtered_out"

    def allow_entry(self, ctx: EntryContext) -> str | None:
        # counts are -1 at the pre-entry universe check (unknown) — apply
        # count-based gates only when real counts are supplied
        if ctx.open_positions_for_insider_ticker >= 0 and ctx.open_positions_for_insider_ticker >= _alp.max_positions_per_insider_ticker:
            return "insider_ticker_limit"
        if ctx.open_positions_in_ticker >= 0 and ctx.open_positions_in_ticker >= _alp.max_positions_per_ticker:
            return "ticker_limit"
        return None
