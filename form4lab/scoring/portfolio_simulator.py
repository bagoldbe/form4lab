"""Portfolio simulator — replays insider signals with capital constraints.

Uses temporal scoring (no look-ahead bias) for insider tier classification,
allocates capital to positions when signals fire, holds for N trading days,
and tracks portfolio value over time compared to SPY buy-and-hold.
"""
import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from form4lab.scoring.flags import (
    load_backtest_data,
    compute_drawdown_flags,
    compute_cluster_flags,
    compute_firsttime_flags,
    compute_routine_flags,
    compute_filing_lag_flags,
    compute_liquidity_flags,
    YF_SECTOR_ETF,
)
from form4lab.scoring.temporal_scorer import TemporalScoreCache
from form4lab.strategy.base import EntryContext, SizingContext, Strategy
from form4lab.utils import is_csuite, is_ceo, is_cfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price index — O(log n) lookups via np.searchsorted
# ---------------------------------------------------------------------------

def build_price_index(prices_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-ticker sorted (dates, closes) arrays for fast lookup."""
    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    index = {}
    for ticker, grp in prices.groupby("ticker"):
        index[ticker] = (grp["date"].values, grp["adj_close"].values)
    return index


def get_price_on_or_after(
    price_index: dict, ticker: str, target_date: date
) -> tuple[float, date] | None:
    """Find the closing price on or after target_date.

    Returns (price, actual_date) or None if not found.
    """
    if ticker not in price_index:
        return None
    dates, closes = price_index[ticker]
    dt64 = np.datetime64(pd.Timestamp(target_date))
    idx = np.searchsorted(dates, dt64, side="left")
    if idx >= len(dates):
        return None
    return float(closes[idx]), pd.Timestamp(dates[idx]).date()


def get_price_n_td_later(
    price_index: dict, ticker: str, entry_date: date, n: int
) -> tuple[float, date] | None:
    """Find price N trading days after entry_date.

    Returns (price, exit_date) or None if not enough data.
    """
    if ticker not in price_index:
        return None
    dates, closes = price_index[ticker]
    dt64 = np.datetime64(pd.Timestamp(entry_date))
    idx = np.searchsorted(dates, dt64, side="left")
    target_idx = idx + n
    if target_idx >= len(dates):
        return None
    return float(closes[target_idx]), pd.Timestamp(dates[target_idx]).date()


def get_last_price(price_index: dict, ticker: str) -> tuple[float, date] | None:
    """Get the last available price for a ticker."""
    if ticker not in price_index:
        return None
    dates, closes = price_index[ticker]
    return float(closes[-1]), pd.Timestamp(dates[-1]).date()


def _first_date_after(index: dict, ticker: str, after_date: date) -> date | None:
    """First indexed date strictly after `after_date` for a ticker, or None.

    Shared by the C-suite sell-reversal and earnings event exits.
    """
    arr = index.get(ticker) if index else None
    if arr is None or len(arr) == 0:
        return None
    dt64 = np.datetime64(pd.Timestamp(after_date))
    idx = np.searchsorted(arr, dt64, side="right")  # strictly after
    if idx >= len(arr):
        return None
    return pd.Timestamp(arr[idx]).date()


def _first_csuite_sell_after(
    csuite_sell_index: dict, ticker: str, after_date: date
) -> date | None:
    """First C-suite sell date strictly after `after_date` for a ticker, or None."""
    return _first_date_after(csuite_sell_index, ticker, after_date)


def _kth_date_after(index: dict, ticker: str, after_date: date, k: int) -> date | None:
    """The Kth indexed date strictly after `after_date` (k>=1), or None."""
    arr = index.get(ticker) if index else None
    if arr is None or len(arr) == 0:
        return None
    dt64 = np.datetime64(pd.Timestamp(after_date))
    idx = np.searchsorted(arr, dt64, side="right") + (k - 1)
    if idx >= len(arr):
        return None
    return pd.Timestamp(arr[idx]).date()


def realized_vol(
    price_index: dict, ticker: str, entry_date: date, window: int = 20,
) -> float | None:
    """Trailing realized volatility (stdev of daily log returns) before entry.

    Uses only prices strictly before entry_date (look-ahead-free). Returns the
    per-day stdev, or None if fewer than `window`+1 prior observations. Used by
    per-position vol-targeted sizing.
    """
    if ticker not in price_index:
        return None
    dates, closes = price_index[ticker]
    dt64 = np.datetime64(pd.Timestamp(entry_date))
    idx = np.searchsorted(dates, dt64, side="left")  # prices before entry_date
    if idx < window + 1:
        return None
    seg = closes[idx - (window + 1):idx]
    seg = seg[seg > 0]
    if len(seg) < window:
        return None
    rets = np.diff(np.log(seg))
    vol = float(np.std(rets, ddof=1))
    return vol if vol > 0 else None


def get_52week_drawdown(
    price_index: dict, ticker: str, entry_price: float, entry_date: date,
) -> float | None:
    """Compute drawdown from 52-week high at entry_date.

    Returns (entry_price - high_52w) / high_52w as a negative float (e.g. -0.30),
    or None if fewer than 20 trading days of history before entry_date.
    """
    if ticker not in price_index:
        return None
    dates, closes = price_index[ticker]
    dt64 = np.datetime64(pd.Timestamp(entry_date))
    idx = np.searchsorted(dates, dt64, side="right")  # entries before entry_date
    if idx < 20:
        return None
    lookback = closes[max(0, idx - 252):idx]
    high_52w = float(np.max(lookback))
    if high_52w <= 0:
        logger.warning("52-week high for %s is non-positive (%.2f) at %s — possible data corruption", ticker, high_52w, entry_date)
        return None
    return (entry_price - high_52w) / high_52w


# ---------------------------------------------------------------------------
# Per-signal-type hold periods (trading days)
# ---------------------------------------------------------------------------

from form4lab.config import settings as _settings
_sig_cfg = _settings.signal
DEFAULT_HOLD_DAYS = _sig_cfg.hold_days_default  # config, not strategy — Position dataclass default


# ---------------------------------------------------------------------------
# Position + Portfolio
# ---------------------------------------------------------------------------

@dataclass
class Position:
    txn_id: int
    ticker: str
    company_name: str
    insider_name: str
    signal_type: str
    entry_date: date
    entry_price: float
    shares_held: float
    cost_basis: float
    tier: str
    skill_score: float
    insider_id: int = 0
    role_title: str = ""
    position_pct: float = 0.0
    hold_days: int = DEFAULT_HOLD_DAYS
    exit_date: date | None = None
    exit_price: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    force_closed: bool = False
    extensions: int = 0  # times the hold has been extended (momentum / re-trigger)


@dataclass
class Portfolio:
    initial_cash: float
    cash: float
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    spy_shares: float = 0.0
    spy_cost_basis: float = 0.0
    sleeve_ticker: str = "SPY"  # idle-cash beta sleeve (SPY large-cap, or IJR/IWM smallcap)

    def _spy_market_value(self, price_index: dict, as_of: date) -> float:
        """Current market value of the idle-cash beta sleeve position."""
        if self.spy_shares <= 0:
            return 0.0
        result = get_price_on_or_after(price_index, self.sleeve_ticker, as_of)
        if result:
            return self.spy_shares * result[0]
        return self.spy_cost_basis  # fallback to cost basis

    def total_value(self, price_index: dict, as_of: date) -> float:
        """Compute total portfolio value (cash + market value of open positions + SPY)."""
        value = self.cash
        for pos in self.open_positions:
            result = get_price_on_or_after(price_index, pos.ticker, as_of)
            if result:
                value += pos.shares_held * result[0]
            else:
                value += pos.cost_basis  # fallback to cost basis
        value += self._spy_market_value(price_index, as_of)
        return value

    def gross_position_value(self, price_index: dict, as_of: date) -> float:
        """Sum of current market values of all open positions (including SPY)."""
        total = 0.0
        for pos in self.open_positions:
            result = get_price_on_or_after(price_index, pos.ticker, as_of)
            if result:
                total += pos.shares_held * result[0]
            else:
                total += pos.cost_basis
        total += self._spy_market_value(price_index, as_of)
        return total

    def margin_loan(self) -> float:
        """Outstanding margin loan (0 if cash >= 0)."""
        return max(0.0, -self.cash)

    def buying_power(self, price_index: dict, as_of: date,
                     margin_multiplier: float) -> float:
        """Available buying power under margin rules.

        With margin_multiplier=1.0 this equals cash (no margin).
        With margin_multiplier=2.0 (Reg T), buying power =
        equity * 2 - gross_positions, floored at 0.
        """
        if margin_multiplier <= 1.0:
            return max(0.0, self.cash)
        equity = self.total_value(price_index, as_of)
        gross_positions = self.gross_position_value(price_index, as_of)
        return max(0.0, equity * margin_multiplier - gross_positions)

    def buy_spy(self, amount: float, spy_price: float) -> None:
        """Buy SPY shares with idle cash."""
        if amount <= 0 or spy_price <= 0:
            return
        shares = amount / spy_price
        self.cash -= amount
        self.spy_shares += shares
        self.spy_cost_basis += amount

    def sell_spy(self, amount: float, spy_price: float) -> float:
        """Sell SPY shares to raise cash. Returns realized P&L.

        If amount exceeds current SPY holdings, sells everything.
        """
        if self.spy_shares <= 0 or spy_price <= 0 or amount <= 0:
            return 0.0
        shares_to_sell = min(amount / spy_price, self.spy_shares)
        proceeds = shares_to_sell * spy_price
        avg_cost = self.spy_cost_basis / self.spy_shares if self.spy_shares > 0 else 0
        realized_pnl = shares_to_sell * (spy_price - avg_cost)
        self.cash += proceeds
        self.spy_cost_basis -= shares_to_sell * avg_cost
        self.spy_shares -= shares_to_sell
        if self.spy_shares < 1e-9:
            self.spy_shares = 0.0
            self.spy_cost_basis = 0.0
        return realized_pnl

    def close_position(self, pos: Position, exit_price: float, exit_date: date,
                       force: bool = False) -> None:
        """Close a position: compute P&L, return cash, move to closed list."""
        pos.exit_price = exit_price
        pos.exit_date = exit_date
        pos.pnl = (exit_price - pos.entry_price) * pos.shares_held
        pos.pnl_pct = (exit_price / pos.entry_price) - 1.0
        pos.force_closed = force
        self.cash += pos.shares_held * exit_price
        self.open_positions.remove(pos)
        self.closed_positions.append(pos)

    def record_snapshot(self, price_index: dict, as_of: date) -> None:
        """Record an equity curve data point."""
        self.equity_curve.append((as_of, self.total_value(price_index, as_of)))


# ---------------------------------------------------------------------------
# SPY rebalance — maintain target cash buffer
# ---------------------------------------------------------------------------

def rebalance_spy(portfolio: Portfolio, price_index: dict, as_of: date,
                  buffer_pct: float) -> None:
    """Rebalance idle cash into/out of SPY to maintain target buffer.

    Args:
        portfolio: The portfolio to rebalance.
        price_index: Price lookup index.
        as_of: Current date for SPY price lookup.
        buffer_pct: Target cash as fraction of total portfolio value.
    """
    sleeve = portfolio.sleeve_ticker
    if sleeve not in price_index:
        return
    spy_result = get_price_on_or_after(price_index, sleeve, as_of)
    if spy_result is None:
        return
    spy_price = spy_result[0]
    if spy_price <= 0:
        return

    total_value = portfolio.total_value(price_index, as_of)
    target_cash = total_value * buffer_pct
    excess = portfolio.cash - target_cash

    if excess > 10.0:
        portfolio.buy_spy(excess, spy_price)
    elif excess < -10.0 and portfolio.spy_shares > 0:
        portfolio.sell_spy(abs(excess), spy_price)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def prepare_backtest_inputs(db: Session) -> dict:
    """Load data once and compute every generic flag, for reuse across many
    simulations.

    Returns a dict {"buys", "price_index", "temporal_cache", "csuite_sell_index",
    "earnings_index"} suitable for passing as run_simulation(preloaded=...). The
    buys frame carries the full generic flag set from form4lab.scoring.flags
    (drawdown, causal cluster, first-time, filing-lag, routine, liquidity), so
    any mode built on those flags works without recomputation.

    This exists so a multi-seed sweep pays the data load + temporal-cache
    preload ONCE instead of per simulation.
    """
    data = load_backtest_data(db)
    buys = data["buys"]
    logger.info(f"Loaded {len(buys):,} P-buys with outcomes")

    roles = data["roles"]
    if not roles.empty:
        buys = buys.merge(
            roles[["insider_id", "company_id", "role_title"]],
            on=["insider_id", "company_id"],
            how="left",
        )
    else:
        buys["role_title"] = None

    buys = compute_drawdown_flags(buys, data["prices"])
    buys = compute_cluster_flags(buys, data["roles"], causal=True)
    buys = compute_firsttime_flags(buys, data["roles"])
    buys = compute_routine_flags(buys)
    buys = compute_filing_lag_flags(buys)
    buys = compute_liquidity_flags(buys, data["prices"])

    price_index = build_price_index(data["prices"])

    temporal_cache = TemporalScoreCache()
    temporal_cache.preload(db)

    # Both index builders are defined locally in this module and have no
    # dependency on anything outside form4lab.scoring.flags, so they're cheap
    # to always include here; run_simulation only consumes them when
    # exit_on_csuite_sell / exit_after_earnings are requested.
    csuite_sell_index = build_csuite_sell_index(db)
    earnings_index = build_earnings_index(db)

    return {"buys": buys, "price_index": price_index, "temporal_cache": temporal_cache,
            "csuite_sell_index": csuite_sell_index, "earnings_index": earnings_index}


def build_earnings_index(db: Session) -> dict[str, np.ndarray]:
    """ticker -> sorted array of earnings report dates. Empty if table absent."""
    from sqlalchemy import text
    try:
        rows = pd.read_sql(text("""
            SELECT c.ticker, e.earnings_date
            FROM earnings_dates e JOIN companies c ON c.id = e.company_id
        """), db.connection())
    except Exception as e:
        logger.warning("earnings_dates not available (%s) — earnings flags disabled", e)
        return {}
    if rows.empty:
        return {}
    rows["earnings_date"] = pd.to_datetime(rows["earnings_date"])
    return {tk: np.sort(g["earnings_date"].values) for tk, g in rows.groupby("ticker")}


def build_csuite_sell_index(db: Session) -> dict[str, np.ndarray]:
    """ticker -> sorted array of dates on which a C-suite insider sold (code S)
    on the open market. Used by the sell-reversal early exit (D2)."""
    from sqlalchemy import text
    rows = pd.read_sql(text("""
        SELECT c.ticker, t.transaction_date, r.role_title
        FROM transactions t
        JOIN companies c ON c.id = t.company_id
        JOIN insider_roles r ON r.insider_id = t.insider_id AND r.company_id = t.company_id
        WHERE t.transaction_code = 'S'
    """), db.connection())
    if rows.empty:
        return {}
    rows = rows[rows["role_title"].map(is_csuite)]
    rows["transaction_date"] = pd.to_datetime(rows["transaction_date"])
    index = {}
    for ticker, grp in rows.groupby("ticker"):
        index[ticker] = np.sort(grp["transaction_date"].values)
    return index


def _try_sell_spy_for_signal(
    portfolio: Portfolio,
    position_size: float,
    price_index: dict,
    file_date: date,
    margin_multiplier: float,
) -> tuple[float, bool]:
    """Sell SPY if buying power is insufficient for a signal trade.

    Returns (realized_pnl, did_sell).
    """
    available = portfolio.buying_power(price_index, file_date, margin_multiplier)
    if available >= position_size:
        return 0.0, False
    shortfall = position_size - available
    if shortfall <= 0:
        return 0.0, False
    spy_result = get_price_on_or_after(price_index, portfolio.sleeve_ticker, file_date)
    if spy_result is None:
        return 0.0, False
    pnl = portfolio.sell_spy(shortfall, spy_result[0])
    return pnl, True


def _queue_bump_for_capital(
    portfolio: Portfolio, needed: float, price_index: dict,
    file_date: date, margin_multiplier: float,
) -> int:
    """Close oldest open positions until buying power >= needed (or none left).

    Capital routing: a new signal carries higher immediate edge than a position
    late in its hold, so retire the oldest to fund the fresher entry. Returns the
    count force-closed early.
    """
    closed = 0
    # oldest first by entry_date
    for pos in sorted(portfolio.open_positions, key=lambda p: p.entry_date):
        if portfolio.buying_power(price_index, file_date, margin_multiplier) >= needed:
            break
        px = get_price_on_or_after(price_index, pos.ticker, file_date)
        exit_price = px[0] if px else pos.entry_price
        exit_date = px[1] if px else file_date
        portfolio.close_position(pos, exit_price, exit_date, force=True)
        closed += 1
    return closed


def run_simulation(
    db: Session,
    initial_cash: float = 10_000.0,
    position_size: float | None = None,
    hold_days: int = 60,
    start_date: date | None = None,
    end_date: date | None = None,
    target_signals: set[str] | None = None,
    filter_routine: bool = False,
    filter_filing_lag: int | None = None,
    liquidity_sizing: bool = False,
    cluster_boost: bool = False,
    universe: set[str] | None = None,
    margin_multiplier: float = 1.0,
    margin_interest_rate: float = 0.06,
    maintenance_margin_pct: float = 0.25,
    drawdown_threshold: float | None = None,
    spy_parking: bool = False,
    spy_parking_buffer: float = 0.20,
    shuffle_seed: int | None = None,
    preloaded: dict | None = None,
    signal_predicate=None,
    size_fn=None,
    strategy: Strategy | None = None,
    exit_on_csuite_sell: bool = False,
    exit_after_earnings: int | None = None,
    earnings_exit_k: int = 1,
    beta_sleeve_ticker: str = "SPY",
    queue_bumping: bool = False,
    scale_out_mult: float | None = None,
    momentum_extend_days: int = 0,
    max_extensions: int = 2,
    extend_on_retrigger: bool = False,
    adv_cap_pct: float | None = None,
    slippage_bps_per_advpct: float = 0.0,
    vol_pctile_max: float | None = None,
    adv_pctile_min: float | None = None,
    sector_cap_pct: float | None = None,
    entry_delay_days: int = 0,
    ticker_exposure_cap: float | None = None,
) -> tuple["Portfolio", dict]:
    """Run the portfolio simulation.

    Args:
        db: Database session.
        initial_cash: Starting capital.
        position_size: Fixed dollar amount per position. If set, overrides
            the strategy's percentage-based (role-tiered) sizing.
        hold_days: Default trading days to hold. Overridden per signal type
            by the active strategy's registered hold_days (see SignalRegistry).
        start_date: Only open positions on/after this date.
        end_date: Only open positions on/before this date (for train/validate/test
            time splits). Positions opened in-window still close normally after it.
        target_signals: Signal types to trade. Defaults to the active
            strategy's tradeable signal names (registry.tradeable_names()).
        filter_routine: If True, skip trades from routine insiders (Cohen 2012).
        filter_filing_lag: If set, skip trades with filing lag > N days.
        liquidity_sizing: If True, 25% larger positions for low-liquidity stocks.
        cluster_boost: If True, 15% larger positions for role-diverse clusters.
        universe: If set, only trade tickers in this set.
        margin_multiplier: Leverage multiplier for buying power. 1.0 = cash
            only (default), 1.5 = 50% margin, 2.0 = Reg T margin.
        margin_interest_rate: Annual interest rate charged on margin loans
            (default 0.06 = 6%). Only applies when margin_multiplier > 1.0.
        maintenance_margin_pct: Minimum equity/gross_position ratio before
            margin call triggers forced liquidation (default 0.25 = 25%).
        drawdown_threshold: Minimum drawdown from 52-week high to enter a trade
            (e.g. -0.30 = stock must be 30%+ below its 52-week high). None to disable.
        spy_parking: If True, invest idle cash in the beta sleeve between signals.
        spy_parking_buffer: Cash buffer as fraction of portfolio when
            parking is enabled (default 0.20 = keep 20% in cash). Lower it to
            deploy more idle capital into the sleeve.
        beta_sleeve_ticker: Ticker for the idle-cash beta sleeve (default "SPY";
            use "IJR"/"IWM" for a smallcap beta sleeve).
        queue_bumping: If True, when a new signal can't be funded, close the
            OLDEST open position (drift mostly realized) to fund the fresher,
            higher-immediate-edge entry instead of skipping it.
        preloaded: Optional dict with keys {"buys", "price_index",
            "temporal_cache"} to reuse data across many simulations (e.g. a
            multi-seed sweep). When provided, the expensive load / flag-compute /
            price-index / temporal-cache-preload steps are skipped and the
            passed-in objects are used. `buys` must already carry every flag the
            requested mode needs (drawdown, causal cluster, etc.).
        signal_predicate: Optional callable (row, tier, skill_score) -> bool.
            When set, it is the SOLE entry gate — it overrides target_signals.
            Used by the research harness to express arbitrary boolean
            hypotheses over the flag columns. The strategy universe gate,
            filters, concentration and drawdown_threshold still apply on top.
        size_fn: Optional callable (row, tier, skill_score, portfolio_value)
            -> position-size fraction (e.g. 0.05). When set it overrides the
            strategy's default sizing. Used for vol targeting and stretch
            sizing. The liquidity/cluster multipliers are NOT applied on top of
            a custom size_fn.
        strategy: The Strategy implementation to source classification,
            sizing, entry gates, and hold periods from. Defaults to the
            process-wide active strategy (get_active()[0]) resolved from
            settings.strategy_path, so omitting it behaves exactly as before.
        exit_on_csuite_sell: If True, force-close a position early (instead of
            waiting hold_days) the first time a C-suite insider sells (code S)
            in that ticker after entry — a thesis-changed reversal exit. Uses
            the csuite_sell_index from `preloaded` (or built on demand).
        exit_after_earnings: If set to N, exit N trading days after the first
            earnings report following entry (event-driven exit capturing the
            catalyst), if that comes before hold_days. Uses earnings_index from
            `preloaded`.

    Returns:
        (Portfolio, price_index) tuple. Portfolio has closed/open positions and
        equity curve. price_index is reusable for metrics and chart generation.
    """
    from form4lab.strategy.registry import get_active, SignalRegistry
    if strategy is None:
        strategy = get_active()[0]
        registry = get_active()[1]
    else:
        registry = SignalRegistry(strategy)
    if target_signals is None:
        target_signals = registry.tradeable_names()

    use_pct_sizing = position_size is None

    if preloaded is None:
        # Load data + compute every generic flag exactly like a preloaded
        # multi-seed sweep would (see prepare_backtest_inputs). A direct call
        # -- e.g. from the CLI, which never passes `preloaded` -- must get the
        # same cluster_size / is_first_buy / etc. columns a strategy's
        # classify()/classify_row() may need, regardless of which filter_* /
        # *_sizing / *_boost kwargs below were requested. Previously this
        # branch only computed cluster/routine/filing-lag/liquidity flags
        # when one of those was truthy (and skipped first-time flags
        # entirely), so the default strategy -- which reads cluster_size via
        # cluster_unique_insiders -- saw zero trades through this exact path.
        # prepare_backtest_inputs is the single source of truth for "every
        # generic flag the platform knows how to compute"; reuse it wholesale
        # here instead of re-deriving a partial subset.
        logger.info("Loading backtest data...")
        preloaded = prepare_backtest_inputs(db)

    # Reuse data/flags/index/cache -- either passed in by the caller
    # (multi-seed sweeps) or just loaded above. Either way, the buys frame is
    # expected to already carry every flag the chosen mode needs (see
    # prepare_backtest_inputs).
    buys = preloaded["buys"].copy()
    price_index = preloaded["price_index"]
    temporal_cache = preloaded["temporal_cache"]
    csuite_sell_index = preloaded.get("csuite_sell_index", {})
    earnings_index = preloaded.get("earnings_index", {})

    # Sort buys by filing_date (when we'd discover the trade), not transaction_date
    buys["transaction_date"] = pd.to_datetime(buys["transaction_date"])
    buys["filing_date"] = pd.to_datetime(buys["filing_date"])
    # Fall back to transaction_date if filing_date is missing
    buys["filing_date"] = buys["filing_date"].fillna(buys["transaction_date"])
    buys = buys.sort_values("filing_date").reset_index(drop=True)

    # Shuffle same-day signals for order sensitivity testing. Skip on an empty
    # buys frame (e.g. a fresh/empty database) — a zero-group groupby leaves
    # shuffled_parts == [], and pd.concat([]) raises ValueError: No objects to
    # concatenate.
    if shuffle_seed is not None and not buys.empty:
        rng = np.random.default_rng(shuffle_seed)
        groups = buys.groupby(buys["filing_date"].dt.date)
        shuffled_parts = []
        for _, group in groups:
            shuffled = group.sample(frac=1.0, random_state=int(rng.integers(2**31)))
            shuffled_parts.append(shuffled)
        buys = pd.concat(shuffled_parts).reset_index(drop=True)

    if start_date:
        buys = buys[buys["filing_date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date:
        buys = buys[buys["filing_date"] <= pd.Timestamp(end_date)].reset_index(drop=True)

    if universe:
        buys = buys[buys["ticker"].isin(universe)].reset_index(drop=True)
        logger.info(f"Universe filter: {len(buys):,} buys in {len(universe):,} tickers")

    if use_pct_sizing:
        logger.info(f"Simulating {len(buys):,} buys with ${initial_cash:,.0f} capital, "
                    f"{_settings.alpaca.base_size_pct:.0%}/{_settings.alpaca.csuite_size_pct:.0%} "
                    f"tiered sizing, {hold_days}td default hold...")
    else:
        logger.info(f"Simulating {len(buys):,} buys with ${initial_cash:,.0f} capital, "
                    f"${position_size:,.0f}/position, {hold_days}td default hold...")

    portfolio = Portfolio(initial_cash=initial_cash, cash=initial_cash,
                          sleeve_ticker=beta_sleeve_ticker)
    trades_opened = 0
    trades_skipped_cash = 0
    queue_bumped_count = 0
    trades_skipped_signal = 0
    trades_skipped_dup = 0
    trades_skipped_routine = 0
    trades_skipped_lag = 0
    trades_skipped_concentration = 0
    trades_skipped_drawdown = 0
    trades_skipped_sector = 0
    trades_skipped_band = 0
    # Track opened positions to skip duplicates from batch filings
    opened_positions: set[tuple] = set()  # (insider_id, ticker, entry_date)
    # txn_id -> sector, for the sector concentration cap
    _pos_sector = dict(zip(buys["txn_id"], buys["sector"])) if "sector" in buys.columns else {}

    # Margin tracking
    use_margin = margin_multiplier > 1.0
    total_margin_interest = 0.0
    margin_call_count = 0
    forced_liquidation_count = 0
    max_margin_loan = 0.0
    leverage_samples: list[float] = []  # (gross_positions / equity) samples
    prev_file_date: date | None = None

    # SPY parking tracking
    spy_total_realized_pnl = 0.0
    spy_buy_count = 0
    spy_sell_count = 0
    spy_allocation_samples: list[float] = []
    # Capital deployment tracking (name positions only, excluding beta sleeve)
    invested_pct_samples: list[float] = []

    for _, row in buys.iterrows():
        # filing_date = when we discover the trade (execution date)
        # transaction_date = when the insider actually traded (classification date)
        file_date = row["filing_date"]
        if isinstance(file_date, pd.Timestamp):
            file_date = file_date.date()

        # 1. Close matured positions (using per-position hold days), or early on
        #    a C-suite sell-reversal if enabled — whichever exit comes first.
        to_close = []
        for pos in portfolio.open_positions:
            hold_exit = get_price_n_td_later(price_index, pos.ticker, pos.entry_date, pos.hold_days)
            exit_candidate = None  # (price, date)
            if hold_exit and hold_exit[1] <= file_date:
                # Momentum-conditional extension: let a winner run rather than exit.
                if (momentum_extend_days > 0 and pos.extensions < max_extensions
                        and hold_exit[0] > pos.entry_price):
                    pos.hold_days += momentum_extend_days
                    pos.extensions += 1
                    hold_exit = None  # re-evaluate next pass; don't close on the hold trigger
                else:
                    exit_candidate = hold_exit
            if exit_on_csuite_sell:
                sell_dt = _first_csuite_sell_after(csuite_sell_index, pos.ticker, pos.entry_date)
                if sell_dt is not None and sell_dt <= file_date:
                    if exit_candidate is None or sell_dt < exit_candidate[1]:
                        sell_px = get_price_on_or_after(price_index, pos.ticker, sell_dt)
                        if sell_px is not None:
                            exit_candidate = (sell_px[0], sell_px[1])
            if exit_after_earnings is not None:
                ed = _kth_date_after(earnings_index, pos.ticker, pos.entry_date, earnings_exit_k)
                if ed is not None:
                    ee = get_price_n_td_later(price_index, pos.ticker, ed, exit_after_earnings)
                    if ee is not None and ee[1] <= file_date:
                        if exit_candidate is None or ee[1] < exit_candidate[1]:
                            exit_candidate = ee
            if exit_candidate is not None:
                to_close.append((pos, exit_candidate[0], exit_candidate[1]))

        for pos, exit_price, exit_date in to_close:
            portfolio.close_position(pos, exit_price, exit_date)

        # --- Margin: interest accrual + maintenance check on date change ---
        if use_margin and prev_file_date is not None and file_date != prev_file_date:
            # Count trading days between prev_file_date and file_date
            if "SPY" in price_index:
                spy_dates = price_index["SPY"][0]
                dt_prev = np.datetime64(pd.Timestamp(prev_file_date))
                dt_cur = np.datetime64(pd.Timestamp(file_date))
                idx_prev = int(np.searchsorted(spy_dates, dt_prev, side="left"))
                idx_cur = int(np.searchsorted(spy_dates, dt_cur, side="left"))
                trading_days_elapsed = max(0, idx_cur - idx_prev)
            else:
                trading_days_elapsed = max(1, (file_date - prev_file_date).days * 5 // 7)

            # Accrue margin interest for each trading day elapsed
            if portfolio.cash < 0 and trading_days_elapsed > 0:
                daily_rate = margin_interest_rate / 252
                for _ in range(trading_days_elapsed):
                    interest = abs(portfolio.cash) * daily_rate
                    portfolio.cash -= interest
                    total_margin_interest += interest

            # Track max margin loan
            loan = portfolio.margin_loan()
            if loan > max_margin_loan:
                max_margin_loan = loan

            # Track leverage
            if portfolio.open_positions:
                equity = portfolio.total_value(price_index, file_date)
                gross_pos = portfolio.gross_position_value(price_index, file_date)
                if equity > 0:
                    leverage_samples.append(gross_pos / equity)

            # Maintenance margin check
            if portfolio.open_positions:
                gross_pos = portfolio.gross_position_value(price_index, file_date)
                equity = portfolio.total_value(price_index, file_date)
                if gross_pos > 0 and equity / gross_pos < maintenance_margin_pct:
                    margin_call_count += 1
                    # Sell the beta sleeve first if parking is enabled (most liquid)
                    if spy_parking and portfolio.spy_shares > 0:
                        spy_result_mc = get_price_on_or_after(price_index, portfolio.sleeve_ticker, file_date)
                        if spy_result_mc:
                            pnl_mc = portfolio.sell_spy(
                                portfolio.spy_shares * spy_result_mc[0],
                                spy_result_mc[0],
                            )
                            spy_total_realized_pnl += pnl_mc
                            spy_sell_count += 1
                    # Re-check margin (may have been restored by SPY sale)
                    gross_pos = portfolio.gross_position_value(price_index, file_date)
                    equity = portfolio.total_value(price_index, file_date)
                    if gross_pos > 0 and equity / gross_pos < maintenance_margin_pct:
                        # Still in margin call — force-liquidate insider positions
                        positions_by_pnl = sorted(
                            portfolio.open_positions,
                            key=lambda p: (
                                (get_price_on_or_after(price_index, p.ticker, file_date) or (p.entry_price, None))[0]
                                - p.entry_price
                            ) * p.shares_held,
                        )
                        for worst_pos in list(positions_by_pnl):
                            result = get_price_on_or_after(price_index, worst_pos.ticker, file_date)
                            if result:
                                portfolio.close_position(worst_pos, result[0], file_date, force=True)
                            else:
                                portfolio.close_position(worst_pos, worst_pos.entry_price, file_date, force=True)
                            forced_liquidation_count += 1
                            # Re-check maintenance margin
                            gross_pos = portfolio.gross_position_value(price_index, file_date)
                            equity = portfolio.total_value(price_index, file_date)
                            if gross_pos == 0 or equity / gross_pos >= maintenance_margin_pct:
                                break

        # --- SPY parking: rebalance on date change ---
        if spy_parking and prev_file_date is not None and file_date != prev_file_date:
            old_spy_shares = portfolio.spy_shares
            rebalance_spy(portfolio, price_index, file_date, spy_parking_buffer)
            if portfolio.spy_shares > old_spy_shares:
                spy_buy_count += 1
            elif portfolio.spy_shares < old_spy_shares:
                spy_sell_count += 1
            # Track beta-sleeve allocation
            total_val = portfolio.total_value(price_index, file_date)
            if total_val > 0 and portfolio.spy_shares > 0:
                spy_result = get_price_on_or_after(price_index, portfolio.sleeve_ticker, file_date)
                if spy_result:
                    spy_allocation_samples.append(
                        portfolio.spy_shares * spy_result[0] / total_val
                    )

        # Sample capital deployment on each date change (name positions only)
        if prev_file_date is None or file_date != prev_file_date:
            tv = portfolio.total_value(price_index, file_date)
            if tv > 0:
                name_mv = portfolio.gross_position_value(price_index, file_date) \
                    - portfolio._spy_market_value(price_index, file_date)
                invested_pct_samples.append(max(0.0, name_mv) / tv)

        prev_file_date = file_date

        # 2. Classify signal using temporal scoring (uses transaction_date,
        #    not filing_date — insider's skill and context at time of trade)
        insider_id = row["insider_id"]
        txn_date = row["transaction_date"]
        if isinstance(txn_date, pd.Timestamp):
            txn_date = txn_date.date()
        score = temporal_cache.get_score(insider_id, txn_date)
        signal_type, tier, skill_score = strategy.classify_row(row, score["tier"], score["skill_score"])
        role_title = row.get("role_title", None)

        # Signal gate: explicit predicate overrides the discrete target set.
        if signal_predicate is not None:
            if not signal_predicate(row, tier, skill_score):
                trades_skipped_signal += 1
                continue
            signal_type = "predicate_buy"
        else:
            if signal_type not in target_signals:
                trades_skipped_signal += 1
                continue

        # 2b. Strategy universe gate. Counts are -1 here (sentinel: unknown at
        # this pre-entry check) — allow_entry must not apply count-based gates
        # on a sentinel. A reason prefixed "universe:" signals a universe /
        # eligibility rejection, as opposed to the concentration rejection
        # handled at the later, real-counts site below (step 4b).
        entry_ctx = EntryContext(
            ticker=row["ticker"], role_title=role_title, insider_id=row["insider_id"],
            open_positions_in_ticker=-1, open_positions_for_insider_ticker=-1,
        )
        reason = strategy.allow_entry(entry_ctx)
        if reason is not None and str(reason).startswith("universe:"):
            trades_skipped_signal += 1
            continue

        # 2c. Research-backed filters
        if filter_routine and row.get("is_routine", False):
            trades_skipped_routine += 1
            continue
        if filter_filing_lag is not None:
            lag = row.get("filing_lag_days")
            if lag is not None and lag > filter_filing_lag:
                trades_skipped_lag += 1
                continue

        # 2d. Re-trigger as time-confirmation: a fresh qualifying signal in a name
        #     we already hold extends that position's hold (signal extension, not size).
        if extend_on_retrigger:
            for held in portfolio.open_positions:
                if held.ticker == row["ticker"] and held.extensions < max_extensions:
                    held.hold_days += hold_days
                    held.extensions += 1

        # 2e. Sector concentration cap — skip if the book is already at the cap for this sector
        if sector_cap_pct is not None and row.get("sector"):
            tv = portfolio.total_value(price_index, file_date)
            if tv > 0:
                sec = row["sector"]
                sec_mv = sum(
                    (get_price_on_or_after(price_index, p.ticker, file_date) or (0, None))[0] * p.shares_held
                    for p in portfolio.open_positions if _pos_sector.get(p.txn_id) == sec
                )
                if sec_mv / tv >= sector_cap_pct:
                    trades_skipped_sector += 1
                    continue

        # 2f. Volatility / liquidity exclusion bands
        if vol_pctile_max is not None:
            vp = row.get("vol_pctile")
            if vp is not None and not (isinstance(vp, float) and np.isnan(vp)) and vp > vol_pctile_max:
                trades_skipped_band += 1
                continue
        if adv_pctile_min is not None:
            ap = row.get("adv_pctile")
            if ap is not None and not (isinstance(ap, float) and np.isnan(ap)) and ap < adv_pctile_min:
                trades_skipped_band += 1
                continue

        # 3. Compute position size and check cash
        this_position_size_exact = None
        if size_fn is not None:
            # Custom sizing (vol targeting, stretch). Skip the built-in
            # liquidity/cluster multipliers — the size_fn owns sizing.
            current_value = portfolio.total_value(price_index, file_date)
            pct = float(size_fn(row, tier, skill_score, current_value))
        elif use_pct_sizing:
            # Strategy-owned sizing (role-tiered by default). ctx.vol is
            # always None here, so strategy.size() short-circuits to its
            # non-vol-targeted fallback regardless of env vol-targeting flags.
            current_value = portfolio.total_value(price_index, file_date)
            decision = strategy.size(SizingContext(
                equity=current_value, ticker=row["ticker"], role_title=role_title))
            this_position_size_exact = decision.dollars
            pct = decision.pct if decision.pct is not None else (
                (decision.dollars / current_value) if current_value > 0 else 0.0)
        else:
            # Fixed-dollar mode: `position_size` (used below) is the sole
            # sizing input. Skip strategy.size()/total_value() entirely here —
            # calling them would be dead work (the result is discarded below)
            # and can emit vol-targeting log noise for no benefit.
            pct = 0.0
        # Research-backed sizing modifiers (not applied when size_fn owns sizing)
        multiplier_fired = False
        if size_fn is None and liquidity_sizing and row.get("liquidity_quintile") is not None:
            if row["liquidity_quintile"] <= 2:  # least liquid quintiles
                pct *= 1.25
                multiplier_fired = True
        if size_fn is None and cluster_boost and row.get("cluster_has_mixed_roles", False):
            pct *= 1.15
            multiplier_fired = True
        if use_pct_sizing:
            # Tiered sizing: reuse the exact strategy-computed dollars when no
            # modifier touched pct (bit-exact with the equity*cfg_pct
            # expression); otherwise (a research modifier fired, or size_fn
            # owns the pct) fall back to current_value * pct.
            if this_position_size_exact is not None and not multiplier_fired:
                this_position_size = this_position_size_exact
            else:
                this_position_size = current_value * pct
            if adv_cap_pct is not None:
                adv = row.get("adv_dollar_20d")
                if adv is not None and not (isinstance(adv, float) and np.isnan(adv)) and adv > 0:
                    this_position_size = min(this_position_size, adv_cap_pct * adv)
            # Floor at $100 to avoid dust positions
            if spy_parking and portfolio.spy_shares > 0:
                pnl_sell, did_sell = _try_sell_spy_for_signal(
                    portfolio, this_position_size, price_index, file_date, margin_multiplier)
                if did_sell:
                    spy_total_realized_pnl += pnl_sell
                    spy_sell_count += 1
            available = portfolio.buying_power(price_index, file_date, margin_multiplier)
            if this_position_size >= 100 and available < this_position_size and queue_bumping:
                queue_bumped_count += _queue_bump_for_capital(
                    portfolio, this_position_size, price_index, file_date, margin_multiplier)
                available = portfolio.buying_power(price_index, file_date, margin_multiplier)
            if this_position_size < 100 or available < this_position_size:
                trades_skipped_cash += 1
                continue
        else:
            this_position_size = position_size
            if adv_cap_pct is not None:
                adv = row.get("adv_dollar_20d")
                if adv is not None and not (isinstance(adv, float) and np.isnan(adv)) and adv > 0:
                    this_position_size = min(this_position_size, adv_cap_pct * adv)
            if spy_parking and portfolio.spy_shares > 0:
                pnl_sell, did_sell = _try_sell_spy_for_signal(
                    portfolio, this_position_size, price_index, file_date, margin_multiplier)
                if did_sell:
                    spy_total_realized_pnl += pnl_sell
                    spy_sell_count += 1
            available = portfolio.buying_power(price_index, file_date, margin_multiplier)
            if available < this_position_size and queue_bumping:
                queue_bumped_count += _queue_bump_for_capital(
                    portfolio, this_position_size, price_index, file_date, margin_multiplier)
                available = portfolio.buying_power(price_index, file_date, margin_multiplier)
            if available < this_position_size:
                trades_skipped_cash += 1
                continue

        # 3b. Per-ticker aggregate exposure cap — bound single-name risk when
        #     multiple insiders stack into one (often low-vol) name.
        if ticker_exposure_cap is not None:
            tv = portfolio.total_value(price_index, file_date)
            tkr = row["ticker"]
            cur_tkr = sum(
                (get_price_on_or_after(price_index, p.ticker, file_date) or (0.0, None))[0] * p.shares_held
                for p in portfolio.open_positions if p.ticker == tkr
            )
            room = max(0.0, ticker_exposure_cap * tv - cur_tkr)
            if this_position_size > room:
                this_position_size = room
            if this_position_size < 100:
                trades_skipped_concentration += 1
                continue

        # 4. Get entry price (on filing_date, or N trading days later for the
        #    entry-delay / alpha-decay test)
        if entry_delay_days > 0:
            entry_result = get_price_n_td_later(price_index, row["ticker"], file_date, entry_delay_days)
        else:
            entry_result = get_price_on_or_after(price_index, row["ticker"], file_date)
        if entry_result is None:
            continue
        entry_price, actual_entry_date = entry_result

        if entry_price <= 0:
            continue

        # 4b. Concentration limits (single-sourced from the strategy). Counts
        # are real (>= 0) here, unlike the sentinel -1 at the earlier universe
        # gate (step 2b) — any non-None reason blocks entry.
        ticker = row["ticker"]
        insider_id = row["insider_id"]
        open_same_ticker = [p for p in portfolio.open_positions if p.ticker == ticker]
        open_same_insider_ticker = [p for p in open_same_ticker if p.insider_id == insider_id]
        entry_ctx = EntryContext(
            ticker=ticker, role_title=role_title, insider_id=insider_id,
            open_positions_in_ticker=len(open_same_ticker),
            open_positions_for_insider_ticker=len(open_same_insider_ticker),
        )
        if strategy.allow_entry(entry_ctx) is not None:
            trades_skipped_concentration += 1
            continue

        # 4b2. Drawdown entry filter — only trade when stock is sufficiently below 52wk high
        # Fail-closed: skip if drawdown can't be computed (insufficient price history)
        if drawdown_threshold is not None:
            dd = get_52week_drawdown(price_index, ticker, entry_price, actual_entry_date)
            if dd is None or dd > drawdown_threshold:
                trades_skipped_drawdown += 1
                continue

        # 4c. Skip if we already opened a position for same insider+ticker+date
        # (batch filings cause multiple buys to map to the same entry)
        pos_key = (insider_id, ticker, actual_entry_date)
        if pos_key in opened_positions:
            trades_skipped_dup += 1
            continue
        opened_positions.add(pos_key)

        # 4d. Slippage / market impact on entry (capacity realism). Round-trip
        #     impact ∝ position size as a % of dollar ADV — penalizes illiquid names.
        if slippage_bps_per_advpct > 0:
            adv = row.get("adv_dollar_20d")
            if adv is not None and not (isinstance(adv, float) and np.isnan(adv)) and adv > 0:
                adv_frac_pct = (this_position_size / adv) * 100.0
                impact = (slippage_bps_per_advpct / 10000.0) * adv_frac_pct
                entry_price = entry_price * (1 + 2 * impact)  # round-trip (buy + eventual sell)

        # 5. Open position(s) — single, or two half-legs for scale-out (tail-capture)
        portfolio.cash -= this_position_size
        pos_hold_days = registry.hold_days(signal_type, hold_days)
        legs = [(this_position_size, pos_hold_days)]
        if scale_out_mult:
            half = this_position_size / 2.0
            legs = [(half, pos_hold_days), (half, int(round(pos_hold_days * scale_out_mult)))]
        for leg_size, leg_hold in legs:
            portfolio.open_positions.append(Position(
                txn_id=row["txn_id"],
                ticker=row["ticker"],
                company_name=row["company_name"],
                insider_name=row["insider_name"],
                signal_type=signal_type,
                entry_date=actual_entry_date,
                entry_price=entry_price,
                shares_held=leg_size / entry_price,
                cost_basis=leg_size,
                tier=tier,
                skill_score=skill_score,
                insider_id=row["insider_id"],
                role_title=str(role_title or ""),
                position_pct=pct,
                hold_days=leg_hold,
            ))
        trades_opened += 1

        # 6. Record equity snapshot
        portfolio.record_snapshot(price_index, file_date)

    # Force-close remaining open positions at last available price
    force_closed = 0
    remaining = list(portfolio.open_positions)
    for pos in remaining:
        # Try to close at position's hold_days, fall back to last price
        result = get_price_n_td_later(price_index, pos.ticker, pos.entry_date, pos.hold_days)
        if result:
            portfolio.close_position(pos, result[0], result[1])
        else:
            last = get_last_price(price_index, pos.ticker)
            if last:
                portfolio.close_position(pos, last[0], last[1], force=True)
                force_closed += 1
            else:
                # No price data at all — close at cost basis
                portfolio.close_position(pos, pos.entry_price, pos.entry_date, force=True)
                force_closed += 1

    # Close the beta-sleeve position at last price
    if spy_parking and portfolio.spy_shares > 0:
        spy_last = get_last_price(price_index, portfolio.sleeve_ticker)
        if spy_last:
            pnl_final = portfolio.sell_spy(portfolio.spy_shares * spy_last[0], spy_last[0])
            spy_total_realized_pnl += pnl_final

    # Final equity snapshot
    if portfolio.closed_positions:
        last_exit = max(p.exit_date for p in portfolio.closed_positions)
        portfolio.record_snapshot(price_index, last_exit)

    skip_parts = [
        f"{trades_skipped_cash} no cash",
        f"{trades_skipped_signal} filtered",
        f"{trades_skipped_concentration} concentration",
        f"{trades_skipped_drawdown} drawdown",
        f"{trades_skipped_dup} batch dedup",
    ]
    if filter_routine:
        skip_parts.append(f"{trades_skipped_routine} routine")
    if filter_filing_lag is not None:
        skip_parts.append(f"{trades_skipped_lag} slow filing")
    logger.info(f"Simulation complete: {trades_opened} trades opened, "
                f"skipped ({', '.join(skip_parts)}), "
                f"{force_closed} force-closed")
    if use_margin:
        logger.info(f"Margin stats: {margin_call_count} margin calls, "
                    f"${total_margin_interest:,.2f} interest paid, "
                    f"${max_margin_loan:,.2f} max loan")

    # Attach margin stats to portfolio for metrics
    portfolio._margin_stats = {
        "margin_interest_paid": total_margin_interest,
        "margin_calls": margin_call_count,
        "forced_liquidations": forced_liquidation_count,
        "max_margin_loan": max_margin_loan,
        "avg_leverage": float(np.mean(leverage_samples)) if leverage_samples else 1.0,
        "margin_multiplier": margin_multiplier,
    }

    if spy_parking:
        portfolio._spy_parking_stats = {
            "spy_parking_buffer": spy_parking_buffer,
            "total_spy_pnl": spy_total_realized_pnl,
            "spy_buy_count": spy_buy_count,
            "spy_sell_count": spy_sell_count,
            "avg_spy_allocation_pct": float(np.mean(spy_allocation_samples)) if spy_allocation_samples else 0.0,
            "max_spy_allocation_pct": float(np.max(spy_allocation_samples)) if spy_allocation_samples else 0.0,
        }

    portfolio._avg_invested_pct = (
        float(np.mean(invested_pct_samples)) if invested_pct_samples else 0.0
    )

    return portfolio, price_index


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio: Portfolio, price_index: dict) -> dict:
    """Compute portfolio performance metrics."""
    closed = portfolio.closed_positions
    if not closed:
        return {"error": "No closed positions"}

    # Basic trade stats
    wins = [p for p in closed if p.pnl and p.pnl > 0]
    losses = [p for p in closed if p.pnl is not None and p.pnl <= 0]

    total_trades = len(closed)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = np.mean([p.pnl_pct for p in wins]) if wins else 0
    avg_loss = np.mean([p.pnl_pct for p in losses]) if losses else 0
    gross_profit = sum(p.pnl for p in wins) if wins else 0
    gross_loss = abs(sum(p.pnl for p in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total return from equity curve
    curve = portfolio.equity_curve
    if len(curve) >= 2:
        start_val = portfolio.initial_cash
        end_val = curve[-1][1]
        total_return = (end_val / start_val) - 1

        # CAGR
        first_date = curve[0][0]
        last_date = curve[-1][0]
        years = (last_date - first_date).days / 365.25
        if years > 0:
            cagr = (end_val / start_val) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        total_return = 0
        cagr = 0
        first_date = closed[0].entry_date
        last_date = closed[-1].exit_date

    # Max drawdown from equity curve
    if len(curve) >= 2:
        values = [v for _, v in curve]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    else:
        max_dd = 0

    # Sharpe ratio: annualized from per-trade returns
    trade_returns = [p.pnl_pct for p in closed if p.pnl_pct is not None]
    if len(trade_returns) > 1:
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns, ddof=1)
        if std_ret > 0:
            # Annualize using average hold period across positions
            avg_hold = np.mean([p.hold_days for p in closed])
            sharpe = (mean_ret / std_ret) * np.sqrt(252 / avg_hold)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # SPY benchmark (buy-and-hold over same period)
    spy_return = None
    if first_date and last_date:
        spy_start = get_price_on_or_after(price_index, "SPY", first_date)
        spy_end = get_price_on_or_after(price_index, "SPY", last_date)
        if spy_start and spy_end:
            spy_return = (spy_end[0] / spy_start[0]) - 1

    # By signal type breakdown
    by_signal = {}
    for signal_type in sorted(set(p.signal_type for p in closed)):
        subset = [p for p in closed if p.signal_type == signal_type]
        s_wins = [p for p in subset if p.pnl and p.pnl > 0]
        by_signal[signal_type] = {
            "trades": len(subset),
            "win_rate": len(s_wins) / len(subset) if subset else 0,
            "avg_return": float(np.mean([p.pnl_pct for p in subset if p.pnl_pct is not None])),
            "total_pnl": sum(p.pnl for p in subset if p.pnl is not None),
        }

    # Fragility: share of total positive PnL from the top-20 trades. High share
    # = returns concentrated in a few names (less robust).
    pnls = sorted((p.pnl for p in closed if p.pnl is not None), reverse=True)
    total_pos_pnl = sum(p for p in pnls if p > 0)
    top20_pnl_share = (
        sum(p for p in pnls[:20] if p > 0) / total_pos_pnl
        if total_pos_pnl > 0 else None
    )

    result = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "top20_pnl_share": float(top20_pnl_share) if top20_pnl_share is not None else None,
        "avg_invested_pct": float(getattr(portfolio, "_avg_invested_pct", 0.0)),
        "spy_return": float(spy_return) if spy_return is not None else None,
        "first_date": first_date,
        "last_date": last_date,
        "force_closed": sum(1 for p in closed if p.force_closed),
        "by_signal": by_signal,
        "initial_cash": portfolio.initial_cash,
        "final_value": float(portfolio.equity_curve[-1][1]) if portfolio.equity_curve else portfolio.cash,
    }

    # Include margin stats if present
    margin_stats = getattr(portfolio, "_margin_stats", None)
    if margin_stats and margin_stats.get("margin_multiplier", 1.0) > 1.0:
        result["margin"] = margin_stats

    spy_stats = getattr(portfolio, "_spy_parking_stats", None)
    if spy_stats:
        result["spy_parking"] = spy_stats

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(metrics: dict, portfolio: Portfolio) -> str:
    """Format simulation results as a CLI report."""
    lines = []
    lines.append("=" * 70)
    lines.append("PORTFOLIO SIMULATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    if "error" in metrics:
        lines.append(f"  ERROR: {metrics['error']}")
        return "\n".join(lines)

    # Summary
    lines.append("-" * 50)
    lines.append("CONFIGURATION")
    lines.append("-" * 50)
    lines.append(f"  Initial capital:     ${metrics['initial_cash']:>12,.2f}")
    lines.append(f"  Period:              {metrics['first_date']} to {metrics['last_date']}")
    lines.append(f"  Total trades:        {metrics['total_trades']:>12,}")
    if metrics["force_closed"] > 0:
        lines.append(f"  Force-closed:        {metrics['force_closed']:>12,}  (at last available price)")
    lines.append("")

    # Performance
    lines.append("-" * 50)
    lines.append("PERFORMANCE")
    lines.append("-" * 50)
    lines.append(f"  Final value:         ${metrics['final_value']:>12,.2f}")
    lines.append(f"  Total return:        {metrics['total_return']:>12.1%}")
    lines.append(f"  CAGR:                {metrics['cagr']:>12.1%}")
    lines.append(f"  Max drawdown:        {metrics['max_drawdown']:>12.1%}")
    lines.append(f"  Sharpe ratio:        {metrics['sharpe']:>12.2f}")
    if metrics["spy_return"] is not None:
        lines.append(f"  SPY buy-and-hold:    {metrics['spy_return']:>12.1%}")
        alpha = metrics["total_return"] - metrics["spy_return"]
        lines.append(f"  Alpha vs SPY:        {alpha:>+12.1%}")
    lines.append("")

    # Trade stats
    lines.append("-" * 50)
    lines.append("TRADE STATISTICS")
    lines.append("-" * 50)
    lines.append(f"  Win rate:            {metrics['win_rate']:>12.1%}")
    lines.append(f"  Avg win:             {metrics['avg_win']:>+12.1%}")
    lines.append(f"  Avg loss:            {metrics['avg_loss']:>+12.1%}")
    lines.append(f"  Profit factor:       {metrics['profit_factor']:>12.2f}")
    lines.append("")

    # By signal type
    lines.append("-" * 50)
    lines.append("BY SIGNAL TYPE")
    lines.append("-" * 50)
    lines.append(f"  {'Signal':<25s}  {'Trades':>6s}  {'Win%':>6s}  {'AvgRet':>8s}  {'TotalPnL':>10s}")
    lines.append(f"  {'------':<25s}  {'------':>6s}  {'----':>6s}  {'------':>8s}  {'--------':>10s}")
    for signal_type, stats in metrics["by_signal"].items():
        lines.append(
            f"  {signal_type:<25s}  {stats['trades']:>6d}  "
            f"{stats['win_rate']:>6.1%}  {stats['avg_return']:>+8.1%}  "
            f"${stats['total_pnl']:>+9,.0f}"
        )
    lines.append("")

    # Margin stats
    if "margin" in metrics:
        m = metrics["margin"]
        lines.append("-" * 50)
        lines.append("MARGIN")
        lines.append("-" * 50)
        lines.append(f"  Leverage:            {m['margin_multiplier']:>12.1f}x")
        lines.append(f"  Avg leverage ratio:  {m['avg_leverage']:>12.2f}x")
        lines.append(f"  Max margin loan:     ${m['max_margin_loan']:>11,.2f}")
        lines.append(f"  Total interest paid: ${m['margin_interest_paid']:>11,.2f}")
        lines.append(f"  Margin calls:        {m['margin_calls']:>12,}")
        lines.append(f"  Forced liquidations: {m['forced_liquidations']:>12,}")
        lines.append("")

    # SPY parking stats
    if "spy_parking" in metrics:
        sp = metrics["spy_parking"]
        lines.append("-" * 50)
        lines.append("SPY PARKING")
        lines.append("-" * 50)
        lines.append(f"  Cash buffer:         {sp['spy_parking_buffer']:>12.0%}")
        lines.append(f"  SPY total P&L:       ${sp['total_spy_pnl']:>+11,.2f}")
        lines.append(f"  SPY buys:            {sp['spy_buy_count']:>12,}")
        lines.append(f"  SPY sells:           {sp['spy_sell_count']:>12,}")
        lines.append(f"  Avg SPY allocation:  {sp['avg_spy_allocation_pct']:>12.1%}")
        lines.append(f"  Max SPY allocation:  {sp['max_spy_allocation_pct']:>12.1%}")
        lines.append("")

    # Top trades
    if portfolio.closed_positions:
        sorted_by_pnl = sorted(
            [p for p in portfolio.closed_positions if p.pnl is not None],
            key=lambda p: p.pnl,
            reverse=True,
        )

        lines.append("-" * 50)
        lines.append("TOP 5 WINNERS")
        lines.append("-" * 50)
        for p in sorted_by_pnl[:5]:
            lines.append(
                f"  {p.ticker:<6s}  {p.signal_type:<22s}  "
                f"{p.entry_date} -> {p.exit_date}  "
                f"{p.pnl_pct:>+7.1%}  ${p.pnl:>+8,.0f}"
            )
        lines.append("")

        lines.append("-" * 50)
        lines.append("TOP 5 LOSERS")
        lines.append("-" * 50)
        for p in sorted_by_pnl[-5:]:
            lines.append(
                f"  {p.ticker:<6s}  {p.signal_type:<22s}  "
                f"{p.entry_date} -> {p.exit_date}  "
                f"{p.pnl_pct:>+7.1%}  ${p.pnl:>+8,.0f}"
            )
        lines.append("")

    return "\n".join(lines)


def build_daily_equity_curve(
    portfolio: Portfolio, price_index: dict,
    spy_parking_buffer: float = 0.0,
) -> pd.DataFrame:
    """Build a daily equity curve from closed+open positions and price data.

    Returns DataFrame with columns: date, portfolio_value, spy_value,
    pct_invested, num_positions. Values are normalized to start at
    the portfolio's initial_cash.

    When spy_parking_buffer > 0, pct_invested includes an approximation of
    SPY parking value: excess cash above (total * buffer) is treated as
    deployed in SPY.
    """
    all_positions = portfolio.closed_positions + portfolio.open_positions

    if not all_positions:
        return pd.DataFrame(columns=["date", "portfolio_value", "spy_value", "pct_invested", "num_positions"])

    first_date = min(p.entry_date for p in all_positions)
    last_date = max(
        p.exit_date or p.entry_date for p in all_positions
    )

    # Get SPY trading day calendar as our date axis
    if "SPY" not in price_index:
        return pd.DataFrame(columns=["date", "portfolio_value", "spy_value", "pct_invested", "num_positions"])

    spy_dates, spy_closes = price_index["SPY"]
    start_idx = int(np.searchsorted(spy_dates, np.datetime64(pd.Timestamp(first_date)), side="left"))
    end_idx = int(np.searchsorted(spy_dates, np.datetime64(pd.Timestamp(last_date)), side="right"))
    if start_idx >= len(spy_dates):
        return pd.DataFrame(columns=["date", "portfolio_value", "spy_value", "pct_invested", "num_positions"])
    end_idx = min(end_idx, len(spy_dates))

    trading_dates = [pd.Timestamp(d).date() for d in spy_dates[start_idx:end_idx]]
    spy_prices_slice = spy_closes[start_idx:end_idx]

    # Normalize SPY to start at initial_cash
    spy_start = spy_prices_slice[0]
    spy_normalized = [portfolio.initial_cash * (p / spy_start) for p in spy_prices_slice]

    # Reconstruct portfolio value from position events + daily price lookups.
    # This gives smooth daily values for every trading day. Note: SPY parking
    # value is not included here (it's captured in the final metrics) because
    # reconstructing SPY cash flows day-by-day is complex; the chart shows
    # insider strategy performance which is the core signal.
    portfolio_values = []
    pct_invested_values = []
    num_positions_values = []
    cash = portfolio.initial_cash
    # Track cash changes: when a position opens, cash decreases; when it closes, cash increases
    events = []  # (date, delta_cash)
    for pos in all_positions:
        events.append((pos.entry_date, -pos.cost_basis))
        if pos.exit_date and pos.exit_price is not None:
            events.append((pos.exit_date, pos.shares_held * pos.exit_price))
    events.sort(key=lambda e: e[0])

    for td in trading_dates:
        # Update cash for all events up to this date
        while events and events[0][0] <= td:
            cash += events.pop(0)[1]

        # Sum market value of open positions on this date
        market_value = 0.0
        n_open = 0
        for pos in all_positions:
            if pos.entry_date <= td and (pos.exit_date is None or pos.exit_date > td):
                n_open += 1
                result = get_price_on_or_after(price_index, pos.ticker, td)
                if result:
                    market_value += pos.shares_held * result[0]
                else:
                    market_value += pos.cost_basis  # fallback

        total = cash + market_value
        portfolio_values.append(total)

        if spy_parking_buffer > 0 and total > 0:
            # Approximate SPY parking: excess cash above buffer is deployed in SPY
            spy_value = max(0.0, cash - total * spy_parking_buffer)
            deployed = market_value + spy_value
            pct_invested_values.append(deployed / total)
        else:
            pct_invested_values.append(market_value / total if total > 0 else 0.0)
        num_positions_values.append(n_open)

    return pd.DataFrame({
        "date": trading_dates,
        "portfolio_value": portfolio_values,
        "spy_value": spy_normalized,
        "pct_invested": pct_invested_values,
        "num_positions": num_positions_values,
    })


def generate_chart(equity_df: pd.DataFrame, output_path: str) -> None:
    """Generate a portfolio vs S&P 500 performance chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = pd.to_datetime(equity_df["date"])

    ax.plot(dates, equity_df["portfolio_value"], color="#2563eb", linewidth=1.8,
            label="Insider Signal Portfolio")
    ax.plot(dates, equity_df["spy_value"], color="#9ca3af", linewidth=1.5,
            linestyle="--", label="S&P 500 (Buy & Hold)")

    ax.fill_between(dates, equity_df["portfolio_value"], equity_df["spy_value"],
                    where=equity_df["portfolio_value"] >= equity_df["spy_value"],
                    alpha=0.15, color="#2563eb", interpolate=True)
    ax.fill_between(dates, equity_df["portfolio_value"], equity_df["spy_value"],
                    where=equity_df["portfolio_value"] < equity_df["spy_value"],
                    alpha=0.15, color="#ef4444", interpolate=True)

    # Format y-axis as dollars
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Final values annotation
    final_pf = equity_df["portfolio_value"].iloc[-1]
    final_spy = equity_df["spy_value"].iloc[-1]
    initial = equity_df["portfolio_value"].iloc[0]
    pf_ret = (final_pf / initial - 1) * 100
    spy_ret = (final_spy / initial - 1) * 100

    ax.annotate(f"${final_pf:,.0f} ({pf_ret:+.0f}%)",
                xy=(dates.iloc[-1], final_pf), fontsize=10, fontweight="bold",
                color="#2563eb", ha="right",
                xytext=(-10, 10), textcoords="offset points")
    ax.annotate(f"${final_spy:,.0f} ({spy_ret:+.0f}%)",
                xy=(dates.iloc[-1], final_spy), fontsize=10,
                color="#6b7280", ha="right",
                xytext=(-10, -15), textcoords="offset points")

    ax.set_title("Insider Signal Portfolio vs S&P 500", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_trades_csv(portfolio: Portfolio, path: str) -> int:
    """Export closed positions to CSV. Returns number of rows written."""
    rows = []
    for p in portfolio.closed_positions:
        rows.append({
            "txn_id": p.txn_id,
            "ticker": p.ticker,
            "company_name": p.company_name,
            "insider_name": p.insider_name,
            "signal_type": p.signal_type,
            "tier": p.tier,
            "skill_score": p.skill_score,
            "entry_date": p.entry_date,
            "entry_price": p.entry_price,
            "exit_date": p.exit_date,
            "exit_price": p.exit_price,
            "shares_held": p.shares_held,
            "cost_basis": p.cost_basis,
            "hold_days": p.hold_days,
            "pnl": p.pnl,
            "pnl_pct": p.pnl_pct,
            "force_closed": p.force_closed,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return len(df)


def export_trades_json(portfolio: Portfolio) -> list[dict]:
    """Export closed positions as a list of JSON-serializable dicts.

    Returns trades sorted by exit_date descending (most recent first).
    """
    trades = []
    for p in portfolio.closed_positions:
        trades.append({
            "ticker": p.ticker,
            "company_name": p.company_name,
            "insider_name": p.insider_name,
            "role_title": p.role_title or "",
            "position_pct": round(float(p.position_pct), 2) if p.position_pct else 0,
            "signal_type": p.signal_type,
            "entry_date": str(p.entry_date),
            "exit_date": str(p.exit_date) if p.exit_date else None,
            "entry_price": round(float(p.entry_price), 2),
            "exit_price": round(float(p.exit_price), 2) if p.exit_price else None,
            "shares": round(float(p.shares_held), 2),
            "pnl": round(float(p.pnl), 2) if p.pnl is not None else None,
            "pnl_pct": round(float(p.pnl_pct), 4) if p.pnl_pct is not None else None,
            "hold_days": p.hold_days,
            "force_closed": p.force_closed,
        })
    trades.sort(key=lambda t: t["exit_date"] or "", reverse=True)
    return trades
