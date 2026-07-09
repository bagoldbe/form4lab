"""Strategy plugin interface.

A Strategy owns decisions (what counts as a signal, sizing, entry gates,
hold periods); the platform owns facts and mechanics (ingestion, features,
persistence, order routing, simulation). One strategy is active per process,
resolved from settings.strategy_path — see form4lab/strategy/registry.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class SignalType:
    """One alert type a strategy can emit, registered via signal_types().

    `direction` controls which registry lookup finds the type:
    SignalRegistry.sell_names() returns only types with direction="sell" AND
    visible=True; buy_names() is the direction="buy" mirror. A strategy whose
    evaluate_sell() emits an alert_type must declare it here with
    direction="sell", or it is invisible to the sell-alert dedup query
    (signal_generator.score_sell_transaction) and the dashboard's sell
    grouping (routes/summary._sig_sets) -- both resolve their sell set via
    sell_names(). Buy types default to direction="buy" and need no override.
    Sell alerts are informational only in this repo: nothing in the shipped
    code path closes a broker position because a sell alert fired -- live
    exits are hold_days/exit_target_date-driven (see
    form4lab/services/alpaca_service.py).
    """
    name: str
    direction: Literal["buy", "sell"] = "buy"
    tradeable: bool = False
    hold_days: int = 60
    priority: int = 0
    visible: bool = True
    description: str = ""


@dataclass(frozen=True)
class TxnView:
    insider_id: int
    company_id: int
    ticker: str
    transaction_date: date
    txn_value: float | None = None
    role_title: str | None = None


class FeatureView(Protocol):
    def get(self, name: str, default: Any = None) -> Any: ...

    def put(self, name: str, value: Any) -> None:
        """Install a strategy-computed value (e.g. "conviction", "cluster_id")
        mid-evaluation so classify() can read it back through the same view.

        Reserved names in the backtest adapter (RowFeatureView, form4lab/
        strategy/features.py): "conviction", "cluster_id", and "is_first_time"
        have no column or computation backing them there, so get() returns
        the caller-supplied `default` for them UNLESS this evaluation already
        called put() for that name first. A strategy that derives one of
        these mid-evaluate_buy (e.g. a real conviction score) and re-reads it
        in classify() must put() it, or every read falls through to
        `default`.
        """
        ...


@dataclass(frozen=True)
class SizeDecision:
    """Result of Strategy.size(): the notional to trade and how it was derived.

    Persisted verbatim to the broker_orders audit columns (sizing_method,
    sizing_vol, sizing_pct) by the live executor -- see
    form4lab/services/alpaca_service.py and form4lab/models/broker.py.

    `method` is a free-form label, but form4lab/models/broker.py documents
    five reserved values: "role" (role-tiered % of equity -- the Strategy ABC
    default, and what the shipped ClusterBuyStrategy example always
    returns), "voltarget" / "voltarget_capped" (vol-targeted sizing,
    uncapped vs. ticker-exposure-capped), "fallback" (vol-targeting wanted
    but `vol` was unavailable, so the role-tiered % was used instead), and
    "shadow" (a vol-targeted size was computed and logged for comparison,
    but the role-tiered dollars are what actually traded). Only "role" is
    emitted by any strategy shipped in this repo today -- the other four
    describe the contract for a strategy that implements real vol-targeted
    sizing off `SizingContext.vol` / `.ticker_exposure_dollars`, which the
    platform computes and supplies (see alpaca_service.py's live vol/exposure
    lookup) without computing a vol-targeted decision itself.
    """
    dollars: float
    method: str
    vol: float | None
    pct: float | None


@dataclass(frozen=True)
class SizingContext:
    equity: float
    ticker: str
    role_title: str | None
    vol: float | None = None
    ticker_exposure_dollars: float | None = None


@dataclass(frozen=True)
class EntryContext:
    ticker: str
    role_title: str | None
    insider_id: int
    open_positions_in_ticker: int
    open_positions_for_insider_ticker: int


@dataclass(frozen=True)
class BuyEvaluation:
    """Result of Strategy.evaluate_buy(): what to alert on and how strong it is.

    `conviction` is an unbounded, strategy-relative score with no fixed
    scale of its own -- it is persisted verbatim as Alert.conviction_score
    (form4lab/scoring/signal_generator.py). The dashboard
    (alert_service.normalize_conviction) buckets it into a 1-5 display
    rating by percentile rank within the current alert set being rendered,
    not against any absolute threshold. Only the relative ordering of a
    strategy's own conviction values matters; e.g. the Strategy ABC default
    evaluate_buy always returns conviction=1.0, so every alert from a
    strategy that never overrides it lands in the same display tier.
    """
    alert_type: str
    conviction: float
    cluster_id: str | None
    summary: str


@dataclass(frozen=True)
class SellEvaluation:
    alert_type: str
    conviction: float
    summary: str


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def signal_types(self) -> list[SignalType]: ...

    @abstractmethod
    def classify(self, txn: TxnView, f: FeatureView) -> str | None: ...

    def evaluate_buy(self, txn: TxnView, f: FeatureView) -> BuyEvaluation | None:
        name = self.classify(txn, f)
        if name is None:
            return None
        return BuyEvaluation(alert_type=name, conviction=1.0, cluster_id=None,
                             summary=f"{name}: {txn.ticker} {txn.transaction_date}")

    def evaluate_sell(self, txn: TxnView, f: FeatureView) -> SellEvaluation | None:
        return None

    def size(self, ctx: SizingContext) -> SizeDecision:
        return SizeDecision(dollars=ctx.equity * 0.05, method="role", vol=None, pct=0.05)

    def allow_entry(self, ctx: EntryContext) -> str | None:
        """Return a reason string to reject entry, or None to allow it.

        `ctx`'s count fields are -1 at the pre-entry universe check (a
        sentinel meaning "counts not yet known") — apply each count-based
        gate only when that count is >= 0. Prefix universe/eligibility
        reasons (ones that don't depend on open-position counts) with
        "universe:" so backtest consumers can distinguish a universe
        rejection from a concentration-limit rejection.
        """
        return None

    def classify_row(self, row, tier: str, skill_score: float) -> tuple[str, str, float]:
        """Backtest adapter: classify a precomputed flags row. Default derives the
        decision from classify(); strategies may override to customize backtest classification.

        Returns (signal_type, tier, skill_score). When classify() returns
        None (no signal), signal_type is the sentinel string "skip" -- not
        None -- so callers get a plain str with no Optional check needed.
        portfolio_simulator does not special-case "skip" by name: it is
        simply never a member of the backtest's target_signals set, so the
        default `signal_type not in target_signals` gate excludes it and no
        position is opened for that row.
        """
        from form4lab.strategy.features import RowFeatureView
        txn = TxnView(insider_id=int(row.get("insider_id", 0) or 0),
                      company_id=int(row.get("company_id", 0) or 0),
                      ticker=str(row.get("ticker", "")),
                      transaction_date=row.get("transaction_date"),
                      txn_value=row.get("total_value"),
                      role_title=row.get("role_title"))
        name = self.classify(txn, RowFeatureView(row, tier, skill_score))
        return (name or "skip", tier, skill_score)
