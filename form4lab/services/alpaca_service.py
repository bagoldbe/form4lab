"""Alpaca paper trading execution service.

Handles the full signal-to-order lifecycle:
- execute_signal(): Place buy order when a tradeable alert fires
- get_positions_to_close(): Find positions past their hold period
- close_position(): Submit sell order to close a position
- sync_orders(): Update order statuses from Alpaca API
- reconcile_positions(): Classify & reconcile positions missing from Alpaca (sold/rename/delisted/orphan/hold)
- get_portfolio_summary(): Dashboard data
"""

import httpx
import logging
from datetime import date, datetime, timedelta
from typing import Literal, NamedTuple

from sqlalchemy.orm import Session

from form4lab.config import settings
from form4lab.models.alert import Alert
from form4lab.models.broker import BrokerOrder, BrokerPosition
from form4lab.models.insider import Insider, InsiderRole
from form4lab.strategy.base import EntryContext, SizingContext

logger = logging.getLogger(__name__)

_alpaca_cfg = settings.alpaca


def _realized_vol_live(symbol: str, db: Session, window: int = 20) -> float | None:
    """Trailing realized vol for a symbol from the price_data table, reusing the
    backtest's look-ahead-free `realized_vol` so live == research. Returns None on
    insufficient history (no external call — reads the table the price job keeps fresh)."""
    import pandas as pd
    from form4lab.models.price import PriceData
    from form4lab.scoring.portfolio_simulator import build_price_index, realized_vol

    rows = (
        db.query(PriceData.date, PriceData.adj_close)
        .filter(PriceData.ticker == symbol)
        .order_by(PriceData.date.desc())
        .limit(window + 15)
        .all()
    )
    if len(rows) < window + 1:
        return None
    df = pd.DataFrame([{"ticker": symbol, "date": r[0], "adj_close": r[1]} for r in rows])
    price_index = build_price_index(df)
    return realized_vol(price_index, symbol, date.today(), window=window)


def _ticker_exposure_dollars(symbol: str, db: Session) -> float:
    """Dollars already committed to a ticker across OPEN positions (for the
    per-ticker aggregate cap). Uses the order notional, falling back to
    shares*entry_price if notional is missing."""
    from form4lab.models.broker import BrokerOrder, BrokerPosition
    total = 0.0
    q = (
        db.query(BrokerPosition.shares, BrokerPosition.entry_price, BrokerOrder.notional)
        .join(BrokerOrder, BrokerOrder.id == BrokerPosition.entry_order_id)
        .filter(BrokerPosition.symbol == symbol, BrokerPosition.status == "open")
    )
    for shares, entry_price, notional in q.all():
        if notional is not None:
            total += float(notional)
        elif shares and entry_price:
            total += float(shares) * float(entry_price)
    return total


def _calculate_exit_target_date(entry_date: date, hold_days: int = 60) -> date:
    """Add N trading days (weekdays only) to entry date."""
    current = entry_date
    days_added = 0
    while days_added < hold_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 through Friday=4
            days_added += 1
    return current


def _determine_order_params(now: datetime) -> dict:
    """Routing for a buy order based on the current US/Eastern time.

    Every order is a market order with extended_hours=False. During regular
    hours (9:30-16:00 ET) Alpaca executes immediately at NBBO. Outside regular
    hours Alpaca queues the order and fills it at the next regular open.

    Extended-hours limit orders with a price buffer frequently expire
    unfilled in the thin after-hours
    session — a large after-hours alert can arrive with a limit price that
    never trades before the extended session closes, silently losing the
    signal. Queued market orders trade extended-hours discount potential
    for certainty of fill at next open.

    Args:
        now: Current time, US/Eastern, naive.

    Returns:
        dict with `order_type` ("market"), `extended_hours` (False), and
        `queued` (True iff outside regular hours).
    """
    time_minutes = now.hour * 60 + now.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60
    in_session = market_open <= time_minutes < market_close
    return {
        "order_type": "market",
        "extended_hours": False,
        "queued": not in_session,
    }


def _get_spy_position(client) -> tuple[float, float]:
    """Get current SPY position from Alpaca.

    Returns (shares, market_value). Returns (0, 0) if no position exists.
    Raises on unexpected API failures to prevent double-buying SPY.
    """
    try:
        position = client.get_open_position("SPY")
        return float(position.qty), float(position.market_value)
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
            return 0.0, 0.0
        logger.error("Unexpected error fetching SPY position: %s", e)
        raise


def _sell_spy_for_signal(client, available: float, position_size: float) -> bool:
    """Sell SPY to fund an insider signal. Returns True if order placed.

    Sells the minimum of (shortfall, spy_market_value). Fail-open: if the
    sell fails, execute_signal continues with whatever cash is available.

    Args:
        available: Current buying power (margin-aware).
        position_size: Dollar amount needed for the signal trade.
    """
    spy_shares, spy_value = _get_spy_position(client)
    if spy_shares <= 0 or spy_value <= 0:
        return False

    shortfall = position_size - available
    if shortfall <= 0:
        return False

    sell_amount = round(min(shortfall, spy_value), 2)
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_data = MarketOrderRequest(
            symbol="SPY",
            notional=sell_amount,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        client.submit_order(order_data=order_data)
        logger.info("SPY parking: sold $%.0f SPY to fund insider signal", sell_amount)
        return True
    except Exception as e:
        logger.error("SPY parking: failed to sell SPY for signal: %s", e)
        return False


def _get_data_client():
    """Create Alpaca StockHistoricalDataClient. Lazy import to avoid import errors when disabled."""
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        _alpaca_cfg.api_key, _alpaca_cfg.secret_key,
    )


CORP_ACTIONS_LOOKBACK_DAYS = 30


class CorporateAction(NamedTuple):
    kind: Literal["name_change", "other", "lookup_failed"]
    new_symbol: str | None
    process_date: str | None
    raw_type: str | None


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": _alpaca_cfg.api_key,
        "APCA-API-SECRET-KEY": _alpaca_cfg.secret_key,
    }


def _trading_base_url() -> str:
    return "https://paper-api.alpaca.markets" if _alpaca_cfg.paper else "https://api.alpaca.markets"


def _get_corporate_action(symbol: str, as_of: date | None = None) -> CorporateAction | None:
    """Look up a recent corporate action for `symbol` via Alpaca's data API.

    Returns a CorporateAction for a name/ticker change (kind="name_change"), a
    sentinel for any other CA category present (kind="other"), a lookup_failed
    sentinel (kind="lookup_failed") when the API call errors, or None ONLY on a
    successful-but-empty lookup (confirmed no CA). The lookup_failed sentinel is
    distinct from None so callers never mistake an outage for "no rename" and book
    a wrong delist.
    """
    as_of = as_of or date.today()
    start = (as_of - timedelta(days=CORP_ACTIONS_LOOKBACK_DAYS)).isoformat()
    try:
        resp = httpx.get(
            "https://data.alpaca.markets/v1/corporate-actions",
            headers=_alpaca_headers(),
            params={"symbols": symbol, "start": start, "end": as_of.isoformat(), "limit": 100},
            timeout=30,
        )
        resp.raise_for_status()
        # Parse inside the try so a malformed body (e.g. corporate_actions is a list,
        # not a dict) raises here and is caught below — consistent with _get_asset_status
        # returning "unknown" on any parse failure (never let a poisoned 200 propagate).
        actions = resp.json().get("corporate_actions", {})
        for nc in actions.get("name_changes") or []:
            if nc.get("old_symbol") == symbol and nc.get("new_symbol"):
                return CorporateAction(
                    kind="name_change", new_symbol=nc["new_symbol"],
                    process_date=nc.get("process_date"), raw_type="name_changes",
                )
        present = sorted(k for k, v in actions.items() if v)
        if present:
            return CorporateAction(kind="other", new_symbol=None, process_date=None,
                                   raw_type=",".join(present))
        return None
    except Exception as e:
        logger.warning("Corporate-action lookup failed for %s: %s", symbol, e)
        return CorporateAction(kind="lookup_failed", new_symbol=None,
                               process_date=None, raw_type=None)


def _get_asset_status(symbol: str) -> str:
    """Return 'active' | 'inactive' | 'not_found' | 'unknown' for an Alpaca asset."""
    try:
        resp = httpx.get(
            f"{_trading_base_url()}/v2/assets/{symbol}",
            headers=_alpaca_headers(), timeout=30,
        )
    except Exception as e:
        logger.warning("Asset status lookup failed for %s: %s", symbol, e)
        return "unknown"
    if resp.status_code == 404:
        return "not_found"
    if resp.status_code != 200:
        logger.warning("Asset status HTTP %s for %s", resp.status_code, symbol)
        return "unknown"
    try:
        data = resp.json()
        status = data.get("status")
        tradable = data.get("tradable", False)
    except Exception as e:
        # 200 with a non-JSON body or a non-dict shape (e.g. a JSON list) — do not
        # let a poisoned response propagate; treat as an uncertain "unknown".
        logger.warning("Asset status parse failed for %s: %s", symbol, e)
        return "unknown"
    if status == "active" and tradable:
        return "active"
    if status == "inactive":
        return "inactive"
    # Any other/unrecognized shape (including active-but-not-tradable) is ambiguous.
    return "unknown"


class ReconcileOutcome(NamedTuple):
    action: Literal["sold", "rename", "delisted", "orphan_close", "needs_review"]
    status: str | None           # new status to set; None = leave unchanged
    new_symbol: str | None
    exit_price: float | None
    close_reason: str
    last_market_price: float | None
    reconcile_hold: bool
    anomaly: Literal["warning", "error"] | None


def classify_disappeared_position(
    symbol: str,
    *,
    has_sell_order: bool,
    sell_price: float | None,
    corp_action: "CorporateAction | None",
    asset_status: str,
    last_bar_price: float | None,
    broker_holds_new_symbol: bool,
) -> ReconcileOutcome:
    """Decide how to book a DB-open position whose shares are gone from Alpaca.

    Pure function — all inputs are pre-fetched by the caller. Never books a loss
    on uncertain data: an unknown asset status or a non-rename corporate action
    routes to manual review (reconcile_hold) rather than an auto-close.

    Exception: when the asset is confirmed active, there is no sell order, and no
    corporate action, the position is treated as an orphan and auto-closed at the
    last bar price without a hold (orphan_close path).
    """
    if has_sell_order and sell_price is not None and sell_price > 0:
        return ReconcileOutcome("sold", "closed", None, sell_price, "sold",
                                sell_price, False, None)

    # Dispatch a known corporate action by kind BEFORE any asset-based branch.
    # A CA lookup failure or malformed rename must never fall through to a delist:
    # we cannot rule out a rename, so we route to manual review instead of booking
    # a loss. `delisted` is reachable ONLY once corp_action is confirmed None.
    if corp_action is not None:
        if corp_action.kind == "lookup_failed":
            # Could not confirm/deny a rename — fail safe, never book a loss.
            return ReconcileOutcome("needs_review", None, None, None,
                                    "ca_lookup_failed", last_bar_price, True, "error")

        if corp_action.kind == "name_change" and corp_action.new_symbol:
            if broker_holds_new_symbol:
                return ReconcileOutcome("rename", "open", corp_action.new_symbol, None,
                                        f"renamed_from:{symbol}", last_bar_price, False, None)
            return ReconcileOutcome("rename", "open", corp_action.new_symbol, None,
                                    f"renamed_from:{symbol};broker_missing", last_bar_price,
                                    True, "error")

        if corp_action.kind == "name_change":
            # name_change with a falsy new_symbol — malformed CA, do not auto-book.
            return ReconcileOutcome("needs_review", None, None, None,
                                    "malformed_name_change", last_bar_price, True, "error")

        if corp_action.kind == "other":
            return ReconcileOutcome("needs_review", None, None, None,
                                    f"ca_needs_review:{corp_action.raw_type}", last_bar_price,
                                    True, "error")

        # Catch-all: unrecognized CA kind added in the future — never fall through to
        # asset-based branches where an unknown CA could produce a spurious delist.
        return ReconcileOutcome("needs_review", None, None, None,
                                f"ca_unhandled:{corp_action.kind}", last_bar_price, True, "error")

    # corp_action is None (confirmed no CA) — safe to use asset-based branches.
    if asset_status in ("not_found", "inactive"):
        return ReconcileOutcome("delisted", "delisted", None, 0.0, "delisted",
                                last_bar_price, False, "error")

    if asset_status == "active":
        return ReconcileOutcome("orphan_close", "closed", None, last_bar_price,
                                "orphan_no_sell", last_bar_price, False, "warning")

    # asset_status == "unknown" (lookup failed) — do not auto-book on uncertain data
    return ReconcileOutcome("needs_review", None, None, None, "reconcile_uncertain",
                            last_bar_price, True, "error")


def _get_52week_drawdown(
    data_client, symbol: str, last_close: float, db: Session | None = None,
) -> float | None:
    """Compute drawdown from 52-week high.

    Tries Alpaca daily bars first; falls back to yfinance (via the cached
    PriceData store) when Alpaca returns fewer than 20 bars or errors.
    Returns (last_close - high_52w) / high_52w as a negative float,
    or None if both sources fail.

    Caller uses fail-closed semantics: returning None causes the trade to be skipped.
    Uses raw close (not adjusted) on both sources to keep the threshold consistent
    across paths; matches Alpaca's bar.close semantics.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    bar_list = []
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        bars = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
        )
        bar_list = bars[symbol] if symbol in bars else []
    except Exception as e:
        logger.warning("Alpaca bars query failed for %s — trying yfinance fallback: %s", symbol, e)

    if len(bar_list) >= 20:
        high_52w = max(b.close for b in bar_list)
        if high_52w <= 0:
            logger.warning("52-week high for %s is non-positive (%.2f) — possible data corruption", symbol, high_52w)
            return None
        return (last_close - high_52w) / high_52w

    if db is None:
        logger.info("Insufficient Alpaca history for %s (%d bars) and no DB for fallback", symbol, len(bar_list))
        return None

    logger.warning(
        "Alpaca returned %d bars for %s (need 20) — falling back to yfinance",
        len(bar_list), symbol,
    )
    return _yfinance_drawdown(symbol, last_close, db)


def _yfinance_drawdown(symbol: str, last_close: float, db: Session) -> float | None:
    """Compute 52-week drawdown from the cached yfinance price store.

    Used as a fallback when Alpaca's historical bars endpoint returns no data.
    Pulls ~400 calendar days to guarantee 252 trading days of coverage.
    """
    from form4lab.data.price_fetcher import YFinanceProvider

    try:
        provider = YFinanceProvider(db)
        end = date.today()
        start = end - timedelta(days=400)
        df = provider.get_daily_prices(symbol, start, end)
    except Exception as e:
        logger.error("yfinance drawdown fallback failed for %s: %s", symbol, e)
        return None

    if df.empty or len(df) < 20:
        logger.info(
            "yfinance fallback insufficient for %s: %d rows (need 20)",
            symbol, len(df),
        )
        return None

    high_52w = float(df["close"].iloc[-252:].max()) if len(df) > 252 else float(df["close"].max())
    if high_52w <= 0:
        logger.warning("yfinance 52w high for %s is non-positive (%.2f)", symbol, high_52w)
        return None
    dd = (last_close - high_52w) / high_52w
    logger.info("yfinance drawdown for %s: %.1f%% (high $%.2f, last $%.2f)", symbol, dd * 100, high_52w, last_close)
    return dd


def _get_trading_client():
    """Create Alpaca TradingClient. Lazy import to avoid import errors when disabled."""
    from alpaca.trading.client import TradingClient
    return TradingClient(
        _alpaca_cfg.api_key,
        _alpaca_cfg.secret_key,
        paper=_alpaca_cfg.paper,
    )


def _open_position_counts(alert: Alert, symbol: str, db: Session) -> tuple[int, int]:
    """Open BrokerPosition counts feeding the strategy's concentration gate:
    (open positions for this insider+ticker, open positions for the ticker
    alone). Pure DB access — the strategy's allow_entry() owns the thresholds.
    """
    insider_ticker_count = (
        db.query(BrokerPosition)
        .filter(
            BrokerPosition.status == "open",
            BrokerPosition.symbol == symbol,
        )
        .join(Alert, BrokerPosition.alert_id == Alert.id)
        .filter(Alert.insider_id == alert.insider_id)
        .count()
    )
    ticker_count = (
        db.query(BrokerPosition)
        .filter(
            BrokerPosition.status == "open",
            BrokerPosition.symbol == symbol,
        )
        .count()
    )
    return insider_ticker_count, ticker_count


def execute_signal(alert: Alert, db: Session) -> BrokerPosition | None:
    """Execute a tradeable buy signal on Alpaca.

    Returns the created BrokerPosition, or None if skipped.
    """
    if not _alpaca_cfg.enabled:
        return None

    from form4lab.strategy.registry import get_active
    strategy, registry = get_active()
    if not registry.is_tradeable(alert.alert_type):
        return None

    # Skip if this alert was already executed (belt-and-suspenders guard)
    existing = db.query(BrokerPosition).filter(
        BrokerPosition.alert_id == alert.id
    ).first()
    if existing:
        logger.info("Skipping signal %d: already has position %d", alert.id, existing.id)
        return None

    # Get company ticker + insider role (needed for the concentration/universe
    # gates below, and later for position sizing)
    from form4lab.models.company import Company
    company = db.get(Company, alert.company_id)
    if not company or not company.ticker:
        logger.info("Skipping signal %d: no ticker for company", alert.id)
        return None
    symbol = company.ticker

    role = (
        db.query(InsiderRole)
        .filter(
            InsiderRole.insider_id == alert.insider_id,
            InsiderRole.company_id == alert.company_id,
        )
        .first()
    )
    role_title = role.role_title if role else "Other"

    # Concentration + universe gates, delegated to the strategy
    n_insider_ticker, n_ticker = _open_position_counts(alert, symbol, db)
    skip_reason = strategy.allow_entry(EntryContext(
        ticker=symbol, role_title=role_title, insider_id=alert.insider_id,
        open_positions_in_ticker=n_ticker,
        open_positions_for_insider_ticker=n_insider_ticker,
    ))
    if skip_reason:
        logger.info("Skipping signal %d: %s (%s, role=%s)", alert.id, skip_reason, symbol, role_title)
        return None

    insider = db.get(Insider, alert.insider_id)
    insider_name = insider.name if insider else "Unknown"

    # Get account info from Alpaca
    try:
        client = _get_trading_client()
        account = client.get_account()
    except Exception as e:
        logger.error("Failed to connect to Alpaca: %s", e)
        return None

    equity = float(account.equity)
    cash = float(account.cash)

    # Calculate position size (vol-targeted when enabled and configured;
    # else falls back to the strategy's default role-tiered sizing)
    vol = None
    exposure = None
    if (
        (_alpaca_cfg.vol_targeting_enabled is True or _alpaca_cfg.vol_targeting_shadow is True)
        and _alpaca_cfg.vol_target_k is not None
    ):
        vol = _realized_vol_live(symbol, db, _alpaca_cfg.vol_target_window)
        if _alpaca_cfg.vol_target_max_ticker_pct is not None:
            exposure = _ticker_exposure_dollars(symbol, db)
    sizing = strategy.size(SizingContext(
        equity=equity, ticker=symbol, role_title=role_title,
        vol=vol, ticker_exposure_dollars=exposure,
    ))
    position_size = sizing.dollars
    if position_size < 100:
        logger.info("Position size too small ($%.0f), skipping", position_size)
        return None

    # Get last close price (needed for limit orders and drawdown check)
    from alpaca.data.requests import StockLatestBarRequest
    data_client = _get_data_client()
    last_close = None
    try:
        bars = data_client.get_stock_latest_bar(
            StockLatestBarRequest(symbol_or_symbols=symbol)
        )
        if bars and symbol in bars:
            last_close = float(bars[symbol].close)
    except Exception as e:
        logger.error("Failed to get last close for %s — trade will be skipped (fail-closed drawdown): %s", symbol, e)

    # Drawdown entry filter — only trade when stock is sufficiently below 52wk high
    # Fail-closed: if we can't compute drawdown, skip the trade (don't risk bad entries)
    # Runs BEFORE SPY sell to avoid liquidating SPY for a trade that gets rejected
    dd_threshold = _alpaca_cfg.drawdown_threshold
    if dd_threshold is not None:
        if last_close is None:
            logger.info("Skipping %s: no last close price available for drawdown check", symbol)
            return None
        dd = _get_52week_drawdown(data_client, symbol, last_close, db)
        if dd is None:
            logger.info("Skipping %s: unable to compute 52-week drawdown (insufficient data)", symbol)
            return None
        if dd > dd_threshold:
            logger.info(
                "Skipping %s: drawdown %.1f%% above threshold %.1f%%",
                symbol, dd * 100, dd_threshold * 100,
            )
            return None
        logger.info("Passed drawdown filter for %s: %.1f%% (threshold %.1f%%)", symbol, dd * 100, dd_threshold * 100)

    # Margin-aware buying power check
    margin_mult = _alpaca_cfg.margin_multiplier
    if margin_mult > 1.0:
        long_market_value = float(account.long_market_value)
        available = min(
            float(account.buying_power),
            equity * margin_mult - long_market_value,
        )
    else:
        available = cash

    # If SPY parking is on and we don't have enough, sell SPY first
    if available < position_size and _alpaca_cfg.spy_parking_enabled:
        sold = _sell_spy_for_signal(client, available, position_size)
        if sold:
            # Refresh account after SPY sell
            try:
                account = client.get_account()
                equity = float(account.equity)
                cash = float(account.cash)
                if margin_mult > 1.0:
                    long_market_value = float(account.long_market_value)
                    available = min(
                        float(account.buying_power),
                        equity * margin_mult - long_market_value,
                    )
                else:
                    available = cash
            except Exception as e:
                logger.warning("Failed to refresh account after SPY sell: %s — trade may be skipped due to stale buying power", e)

    if available < position_size:
        logger.warning(
            "Insufficient buying power ($%.0f) for $%.0f position in %s",
            available, position_size, symbol,
        )
        return None

    # Determine order type based on current time
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et).replace(tzinfo=None)

    order_params = _determine_order_params(now_et)

    if order_params["queued"]:
        logger.info(
            "Outside regular hours, %s buy for alert %d will queue for next open",
            symbol, alert.id,
        )

    # Submit order — always a DAY market order with extended_hours=False.
    # Alpaca executes immediately during RTH and queues otherwise.
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_data = MarketOrderRequest(
            symbol=symbol,
            notional=round(position_size, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        alpaca_order = client.submit_order(order_data=order_data)
    except Exception as e:
        logger.error("Failed to submit order for %s: %s", symbol, e)
        return None

    # Record in our DB — order is already placed with Alpaca at this point
    try:
        broker_order = BrokerOrder(
            alert_id=alert.id,
            alpaca_order_id=str(alpaca_order.id),
            symbol=symbol,
            side="buy",
            notional=round(position_size, 2),
            order_type="market",
            limit_price=None,
            extended_hours=False,
            status="submitted",
            sizing_method=sizing.method,
            sizing_vol=sizing.vol,
            sizing_pct=sizing.pct,
        )
        db.add(broker_order)
        db.flush()

        # Create position (will update entry_price when fill comes in)
        entry_date = date.today()
        position = BrokerPosition(
            alert_id=alert.id,
            entry_order_id=broker_order.id,
            symbol=symbol,
            shares=0,  # updated on fill
            entry_price=0,  # updated on fill
            entry_date=entry_date,
            exit_target_date=_calculate_exit_target_date(
                entry_date, _alpaca_cfg.hold_days,
            ),
            status="open",
            insider_name=insider_name,
            insider_role=role_title,
        )
        db.add(position)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(
            "Order placed for %s (alpaca_id=%s) but DB persist failed: %s — sync_orders will recover",
            symbol, alpaca_order.id, e,
        )
        return None

    logger.info(
        "Submitted market %s order for %s ($%.0f, %s, alert %d)%s",
        symbol, insider_name, position_size, role_title, alert.id,
        " (queued for next open)" if order_params["queued"] else "",
    )
    return position


def get_positions_to_close(db: Session, as_of: date | None = None) -> list[BrokerPosition]:
    """Find open positions past their exit target date."""
    if as_of is None:
        as_of = date.today()
    return (
        db.query(BrokerPosition)
        .filter(
            BrokerPosition.status == "open",
            BrokerPosition.exit_target_date <= as_of,
        )
        .all()
    )


def get_stop_loss_positions(db: Session) -> list[BrokerPosition]:
    """Find open positions that have breached the stop loss threshold.

    Fetches current quotes from Alpaca and compares to entry_price.
    Returns positions where current return <= stop_loss_pct.
    """
    if not _alpaca_cfg.enabled:
        return []

    stop_loss = _alpaca_cfg.stop_loss_pct
    if stop_loss is None or stop_loss >= 0:
        return []

    open_positions = (
        db.query(BrokerPosition)
        .filter(
            BrokerPosition.status == "open",
            BrokerPosition.entry_price > 0,
            BrokerPosition.shares > 0,
        )
        .all()
    )
    if not open_positions:
        return []

    symbols = list({p.symbol for p in open_positions})

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        data_client = StockHistoricalDataClient(
            _alpaca_cfg.api_key, _alpaca_cfg.secret_key,
        )
        quotes = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbols)
        )
    except Exception as e:
        logger.error("Failed to fetch quotes for stop loss check: %s", e)
        return []

    triggered = []
    for pos in open_positions:
        quote = quotes.get(pos.symbol)
        if not quote or not quote.ask_price:
            continue
        current_price = float(quote.ask_price)
        if current_price <= 0:
            continue
        ret = (current_price - pos.entry_price) / pos.entry_price
        if ret <= stop_loss:
            logger.warning(
                "Stop loss triggered for %s (position %d): entry $%.2f, "
                "current $%.2f, return %.1f%% <= %.1f%%",
                pos.symbol, pos.id, pos.entry_price,
                current_price, ret * 100, stop_loss * 100,
            )
            triggered.append(pos)

    return triggered


def close_position(position: BrokerPosition, db: Session) -> BrokerOrder | None:
    """Submit a sell order to close a position."""
    if not _alpaca_cfg.enabled:
        return None

    if position.shares <= 0:
        logger.warning("Skipping close for position %d: no shares (unfilled buy)", position.id)
        try:
            position.status = "closed"
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to mark zero-share position %d as closed: %s", position.id, e)
        return None

    try:
        client = _get_trading_client()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_data = MarketOrderRequest(
            symbol=position.symbol,
            qty=position.shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        alpaca_order = client.submit_order(order_data=order_data)
    except Exception as e:
        logger.error("Failed to close position %d (%s): %s", position.id, position.symbol, e)
        return None

    try:
        broker_order = BrokerOrder(
            alert_id=position.alert_id,
            entry_order_id=position.entry_order_id,
            alpaca_order_id=str(alpaca_order.id),
            symbol=position.symbol,
            side="sell",
            qty=position.shares,
            order_type="market",
            extended_hours=False,
            status="submitted",
        )
        db.add(broker_order)
        db.flush()

        position.status = "closing"
        position.exit_order_id = broker_order.id
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(
            "Sell order placed for %s (alpaca_id=%s) but DB persist failed: %s",
            position.symbol, alpaca_order.id, e,
        )
        return None

    logger.info("Submitted sell order for %s (position %d)", position.symbol, position.id)
    return broker_order


_TERMINAL_ORDER_STATUSES = frozenset({
    "filled", "canceled", "expired", "rejected", "suspended",
    "replaced", "done_for_day", "stopped",
})

_STALE_ORDER_DAYS = 7  # cancel unfilled buy orders older than this.
# Must comfortably exceed the longest weekend/holiday gap so a queued
# market order submitted after Friday's close (waiting for Monday's open,
# or Tuesday's after a Mon holiday) is never killed before its first
# chance to fill. 7d covers Wed-pre-Thanksgiving → next Monday with room.


def sync_orders(db: Session) -> int:
    """Sync pending order statuses from Alpaca. Returns count updated."""
    if not _alpaca_cfg.enabled:
        return 0

    # Include all non-terminal statuses so orders aren't lost after first sync
    # (e.g. Alpaca moves "submitted" → "new" which was previously untracked)
    pending = (
        db.query(BrokerOrder)
        .filter(~BrokerOrder.status.in_(_TERMINAL_ORDER_STATUSES))
        .all()
    )
    if not pending:
        return 0

    try:
        client = _get_trading_client()
    except Exception as e:
        logger.error("Failed to connect to Alpaca for order sync: %s", e)
        return 0

    updated = 0

    for order in pending:
        try:
            alpaca_order = client.get_order_by_id(order.alpaca_order_id)
        except Exception as e:
            logger.error("Failed to check order %s: %s", order.alpaca_order_id, e)
            continue

        new_status = str(alpaca_order.status.value)
        if new_status == order.status:
            continue

        try:
            order.status = new_status
            if new_status == "filled":
                order.filled_qty = float(alpaca_order.filled_qty) if alpaca_order.filled_qty else None
                order.filled_avg_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
                order.filled_at = alpaca_order.filled_at

                # Update linked position
                if order.side == "buy":
                    pos = db.query(BrokerPosition).filter(
                        BrokerPosition.entry_order_id == order.id
                    ).first()
                    if pos and order.filled_avg_price and order.filled_qty:
                        pos.shares = order.filled_qty
                        pos.entry_price = order.filled_avg_price
                    elif not pos:
                        logger.warning("Buy order %s filled but no linked position found", order.alpaca_order_id)
                elif order.side == "sell":
                    pos = db.query(BrokerPosition).filter(
                        BrokerPosition.exit_order_id == order.id
                    ).first()
                    if pos and order.filled_avg_price:
                        pos.exit_price = order.filled_avg_price
                        pos.exit_date = alpaca_order.filled_at.date() if alpaca_order.filled_at else date.today()
                        pos.status = "closed"
                        if pos.entry_price > 0:
                            pos.pnl = (order.filled_avg_price - pos.entry_price) * pos.shares
                            pos.pnl_pct = (order.filled_avg_price - pos.entry_price) / pos.entry_price
                    elif not pos:
                        logger.warning(
                            "Sell order %s filled but no linked position found (may have been reconciled)",
                            order.alpaca_order_id,
                        )

            elif new_status in ("canceled", "expired", "rejected"):
                # If a buy was canceled/rejected, close the position record
                if order.side == "buy":
                    pos = db.query(BrokerPosition).filter(
                        BrokerPosition.entry_order_id == order.id
                    ).first()
                    if pos and pos.status == "open" and pos.shares == 0:
                        pos.status = "closed"

            db.commit()
            updated += 1
        except Exception as e:
            db.rollback()
            logger.error("Failed to sync order %s: %s", order.alpaca_order_id, e)

    # Cancel stale unfilled buy orders (safety net for limit orders that never fill)
    stale_cutoff = datetime.now() - timedelta(days=_STALE_ORDER_DAYS)
    stale_orders = (
        db.query(BrokerOrder)
        .filter(
            BrokerOrder.side == "buy",
            ~BrokerOrder.status.in_(_TERMINAL_ORDER_STATUSES),
            BrokerOrder.created_at < stale_cutoff,
        )
        .all()
    )
    stale_cleaned = 0
    for order in stale_orders:
        # Decide what to record in the DB. If cancel succeeds, mark "canceled".
        # If cancel is rejected, re-query Alpaca: only mutate the DB when Alpaca
        # confirms the order is terminal. Leaving the row alone otherwise lets
        # the next sync_orders pass pick up the true state.
        new_status: str | None = None
        actual_order = None  # populated if we hit the race-fill path
        try:
            client.cancel_order_by_id(order.alpaca_order_id)
            new_status = "canceled"
            logger.info(
                "Cancelled stale %s buy order %s (%s, placed %s)",
                order.status, order.alpaca_order_id, order.symbol, order.created_at,
            )
        except Exception as cancel_err:
            try:
                actual_order = client.get_order_by_id(order.alpaca_order_id)
                actual_status = str(actual_order.status.value)
            except Exception as requery_err:
                logger.warning(
                    "Failed to cancel stale order %s and re-query also failed (%s) — leaving DB unchanged",
                    order.alpaca_order_id, requery_err,
                )
                continue
            if actual_status in _TERMINAL_ORDER_STATUSES:
                new_status = actual_status
                logger.info(
                    "Stale order %s already terminal on Alpaca (%s) — syncing DB",
                    order.alpaca_order_id, actual_status,
                )
            else:
                logger.warning(
                    "Failed to cancel stale order %s (Alpaca status=%s): %s — leaving DB unchanged",
                    order.alpaca_order_id, actual_status, cancel_err,
                )
                continue

        try:
            order.status = new_status
            # Race-fill: cancel was rejected because the order just filled.
            # Propagate fill data and update the position with real shares.
            pos = db.query(BrokerPosition).filter(
                BrokerPosition.entry_order_id == order.id
            ).first()
            if (
                actual_order is not None
                and new_status == "filled"
                and actual_order.filled_qty
                and actual_order.filled_avg_price
            ):
                order.filled_qty = float(actual_order.filled_qty)
                order.filled_avg_price = float(actual_order.filled_avg_price)
                order.filled_at = actual_order.filled_at
                if pos and pos.shares == 0:
                    pos.shares = order.filled_qty
                    pos.entry_price = order.filled_avg_price
                logger.warning(
                    "Race fill detected on stale order %s: shares=%s, price=%s",
                    order.alpaca_order_id, order.filled_qty, order.filled_avg_price,
                )
            elif pos and pos.status == "open" and pos.shares == 0:
                pos.status = "closed"
                logger.info("Closed phantom position %d (%s) from stale order", pos.id, pos.symbol)
            db.commit()
            stale_cleaned += 1
        except Exception as e:
            db.rollback()
            logger.error("Failed to clean up stale order %s: %s", order.alpaca_order_id, e)

    logger.info("Synced %d orders (%d stale cleaned)", updated + stale_cleaned, stale_cleaned)
    return updated + stale_cleaned


def _find_sell_fill(client, symbol: str) -> tuple[bool, float | None]:
    """Return (has_sell_order, fill_price) from Alpaca closed SELL orders for symbol.

    Invariant: when has_sell_order is True, fill_price is guaranteed > 0.
    A non-positive filled_avg_price is treated as no fill (returns False, None).
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        orders = client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, side=OrderSide.SELL,
                symbols=[symbol], limit=5,
            )
        )
        for order in orders:
            if order.filled_avg_price is not None and float(order.filled_avg_price) > 0:
                return True, float(order.filled_avg_price)
    except Exception as e:
        logger.error("Failed to fetch sell orders for %s: %s", symbol, e)
    return False, None


def _get_latest_bar_price(symbol: str) -> float | None:
    """Latest bar close from Alpaca data API (approximate); None on failure."""
    try:
        from alpaca.data.requests import StockLatestBarRequest
        data_client = _get_data_client()
        bars = data_client.get_stock_latest_bar(
            StockLatestBarRequest(symbol_or_symbols=symbol)
        )
        if bars and symbol in bars:
            return float(bars[symbol].close)
    except Exception as e:
        logger.warning("Failed to fetch latest bar for %s: %s", symbol, e)
    return None


def _apply_reconcile_outcome(db: Session, db_positions: list, symbol: str,
                             outcome: ReconcileOutcome) -> int:
    """Apply a ReconcileOutcome to each DB position. Returns the number mutated."""
    count = 0
    for pos in db_positions:
        try:
            if outcome.new_symbol:
                pos.symbol = outcome.new_symbol
            if outcome.status:
                pos.status = outcome.status
            pos.close_reason = outcome.close_reason
            pos.last_market_price = outcome.last_market_price
            pos.reconcile_hold = outcome.reconcile_hold
            if outcome.status in ("closed", "delisted"):
                pos.exit_date = date.today()
                pos.exit_price = outcome.exit_price
                if outcome.exit_price is not None and pos.entry_price > 0:
                    pos.pnl = (outcome.exit_price - pos.entry_price) * pos.shares
                    pos.pnl_pct = (outcome.exit_price - pos.entry_price) / pos.entry_price
                else:
                    logger.warning("Reconciled %s pos %d with no exit price — P&L unavailable",
                                   symbol, pos.id)
            db.commit()
            logger.info(
                "RECON_CLOSE symbol=%s pos=%d action=%s status=%s exit_price=%s pnl=%s",
                symbol, pos.id, outcome.action, pos.status, outcome.exit_price, pos.pnl,
            )
            count += 1
        except Exception as e:
            db.rollback()
            logger.error("Failed to apply reconcile outcome for %s pos %d: %s", symbol, pos.id, e)
            continue
    if outcome.anomaly == "error":
        logger.error("RECON_ANOMALY symbol=%s action=%s new_symbol=%s reason=%r — manual review",
                     symbol, outcome.action, outcome.new_symbol, outcome.close_reason)
    elif outcome.anomaly == "warning":
        logger.warning("RECON_ANOMALY symbol=%s action=%s reason=%r",
                       symbol, outcome.action, outcome.close_reason)
    return count


def reconcile_positions(db: Session) -> int:
    """Detect and reconcile BrokerPositions not backed by active Alpaca positions.

    Compares DB open/closing positions (excluding reconcile_hold=True rows) against
    live Alpaca positions. Before any per-symbol classification runs, a circuit
    breaker checks for mass disappearance: if MORE THAN
    `_alpaca_cfg.reconcile_mass_disappearance_limit` symbols with real shares are
    simultaneously missing from Alpaca, that is treated as a platform glitch rather
    than N independent orphans — a simultaneous multi-position disappearance is more
    likely a brokerage-platform data glitch (e.g. a stale or empty positions
    snapshot) than N real orphans occurring at once, so every affected position is
    held (reconcile_hold=True, close_reason="mass_disappearance_hold") for human
    review, nothing is classified, and the function returns early with the held
    count.

    Otherwise, when a symbol has zero Alpaca shares, classifies the disappearance
    and applies one of five outcomes:

    - sold: confirmed sell fill found → close at fill price.
    - rename: corporate-action name change detected → reopen under new symbol;
      no close is booked. Sets reconcile_hold=True when broker confirmation is absent.
    - delisted: asset confirmed inactive/not_found, no CA → close at 0.0.
    - orphan_close: asset active, no sell, no CA → close at last bar price.
    - needs_review: CA lookup failed, ambiguous CA, or unknown asset status →
      sets reconcile_hold=True, no close. Requires human intervention.

    Partial qty mismatches (Alpaca qty < DB qty but > 0) are logged as warnings
    and not auto-resolved. "closing" positions are included so margin-call orphans
    are also caught.

    Returns count of positions mutated.
    """
    if not _alpaca_cfg.enabled:
        return 0

    # 1. Get all open/closing DB positions grouped by symbol
    # Include "closing" to catch positions orphaned by margin calls during pending sells
    open_positions = (
        db.query(BrokerPosition)
        .filter(BrokerPosition.status.in_(["open", "closing"]))
        .filter(BrokerPosition.reconcile_hold.isnot(True))
        .all()
    )
    if not open_positions:
        return 0

    db_by_symbol: dict[str, list[BrokerPosition]] = {}
    for pos in open_positions:
        db_by_symbol.setdefault(pos.symbol, []).append(pos)

    # 2. Get all Alpaca positions
    try:
        client = _get_trading_client()
        alpaca_positions = client.get_all_positions()
    except Exception as e:
        logger.error("Failed to fetch Alpaca positions for reconciliation: %s", e)
        return 0

    alpaca_qty_by_symbol = {
        p.symbol: float(p.qty) for p in alpaca_positions
    }

    # 2b. Mass-disappearance circuit breaker (platform-glitch guard)
    missing_symbols = [
        sym for sym, positions in db_by_symbol.items()
        if sym != "SPY"
        and sum(p.shares for p in positions) >= 0.01
        and alpaca_qty_by_symbol.get(sym, 0.0) < 0.01
    ]
    limit = _alpaca_cfg.reconcile_mass_disappearance_limit
    if len(missing_symbols) > limit:
        held = 0
        try:
            for sym in missing_symbols:
                for pos in db_by_symbol[sym]:
                    pos.reconcile_hold = True
                    pos.close_reason = "mass_disappearance_hold"
                    held += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        logger.error(
            "MASS DISAPPEARANCE: %d symbols missing from Alpaca in one pass (limit %d) "
            "— holding %d position(s) for review, classifying nothing: %s",
            len(missing_symbols), limit, held, ", ".join(sorted(missing_symbols)),
        )
        return held

    # 3. Compare and reconcile
    reconciled = 0
    for symbol, db_positions in db_by_symbol.items():
        # Skip SPY — parking position is intentionally untracked
        if symbol == "SPY":
            continue

        # Isolate each symbol: a poisoned response or helper failure for one
        # symbol must not abort reconciliation for the rest of the portfolio.
        try:
            alpaca_qty = alpaca_qty_by_symbol.get(symbol, 0.0)
            db_total_shares = sum(p.shares for p in db_positions)

            # Skip positions that have not been filled yet (shares=0). These are
            # buy orders waiting at the broker; sync_orders will populate shares
            # when the fill arrives. Reconciling them here would race the fill
            # and falsely close real, pending positions with a stale exit price.
            if db_total_shares < 0.01:
                continue

            if alpaca_qty < 0.01:
                # Symbol gone from Alpaca — classify WHY before booking anything.
                has_sell, sell_price = _find_sell_fill(client, symbol)
                if has_sell:
                    corp = None
                    asset_status = "active"
                    last_bar = sell_price
                    broker_holds_new = False
                else:
                    corp = _get_corporate_action(symbol)
                    asset_status = _get_asset_status(symbol)
                    last_bar = _get_latest_bar_price(symbol)
                    broker_holds_new = (
                        corp is not None
                        and corp.kind == "name_change"
                        and corp.new_symbol is not None
                        and alpaca_qty_by_symbol.get(corp.new_symbol, 0.0) > 0.01
                    )

                outcome = classify_disappeared_position(
                    symbol,
                    has_sell_order=has_sell, sell_price=sell_price,
                    corp_action=corp, asset_status=asset_status,
                    last_bar_price=last_bar, broker_holds_new_symbol=broker_holds_new,
                )
                reconciled += _apply_reconcile_outcome(db, db_positions, symbol, outcome)
            elif alpaca_qty < db_total_shares:
                # Partial mismatch — log warning but don't auto-close
                logger.warning(
                    "Position qty mismatch for %s: DB has %.0f shares across %d positions, "
                    "Alpaca has %.0f — needs manual review",
                    symbol, db_total_shares, len(db_positions), alpaca_qty,
                )
        except Exception as e:
            logger.error(
                "Reconciliation failed for symbol %s: %s — skipping this symbol", symbol, e,
            )
            continue

    if reconciled:
        logger.info("Reconciled %d orphaned position(s)", reconciled)
    return reconciled


def correct_renamed_position(
    db: Session, old_symbol: str, new_symbol: str, new_price: float | None = None,
) -> int:
    """Re-book positions wrongly closed when their ticker was renamed.

    Re-points the record to the new symbol, re-opens it, clears the bogus exit/P&L,
    and flags it for manual review. Idempotent: re-running matches the already-
    corrected (new_symbol + reconcile_hold) row and leaves it in the same state.
    Returns the number of positions corrected.
    """
    candidates = (
        db.query(BrokerPosition)
        .filter(BrokerPosition.symbol.in_([old_symbol, new_symbol]))
        .filter(BrokerPosition.entry_price > 0)
        .filter(BrokerPosition.exit_order_id.is_(None))
        .all()
    )
    corrected = 0
    for pos in candidates:
        if pos.symbol != old_symbol and not (pos.symbol == new_symbol and pos.reconcile_hold):
            continue
        # Skip positions explicitly tagged as sold — a real sale or a confirmed
        # reconcile-as-sold; do not resurrect either.
        if pos.close_reason == "sold":
            continue
        try:
            pos.symbol = new_symbol
            pos.status = "open"
            pos.reconcile_hold = True
            pos.exit_price = None
            pos.exit_date = None
            pos.pnl = None
            pos.pnl_pct = None
            pos.last_market_price = new_price
            pos.close_reason = f"renamed_from:{old_symbol};manual_correction"
            db.commit()
            corrected += 1
        except Exception as e:
            db.rollback()
            logger.error("Failed to correct rename for pos %d (%s->%s): %s",
                         pos.id, old_symbol, new_symbol, e)
            raise
    logger.info("correct_renamed_position: %s->%s corrected %d position(s)",
                old_symbol, new_symbol, corrected)
    return corrected


def clear_reconcile_hold(db: Session, symbol: str) -> int:
    """Clear the reconcile_hold flag for all positions currently flagged for manual review.

    Sets reconcile_hold = False on every position matching ``symbol`` that currently
    has reconcile_hold = True. Commits in a single transaction; rolls back, logs, and
    re-raises on failure. Returns the count of positions cleared.

    Most close_reason values (e.g. "renamed_from:...") are left in place as an
    audit trail after the hold clears. The exception is
    "mass_disappearance_hold": that value carries no diagnostic content of its
    own (it just marks that the breaker fired), so it is reset to None here once
    an operator has manually confirmed the position's real status.
    """
    candidates = (
        db.query(BrokerPosition)
        .filter(
            BrokerPosition.symbol == symbol,
            BrokerPosition.reconcile_hold.is_(True),
        )
        .all()
    )
    if not candidates:
        return 0

    count = 0
    try:
        for pos in candidates:
            pos.reconcile_hold = False
            if pos.close_reason == "mass_disappearance_hold":
                pos.close_reason = None
            count += 1
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(
            "clear_reconcile_hold: failed to clear hold for symbol=%s: %s",
            symbol, e,
        )
        raise

    logger.info("clear_reconcile_hold: symbol=%s cleared %d position(s)", symbol, count)
    return count


def get_portfolio_summary(db: Session) -> dict:
    """Build portfolio summary data for dashboard display."""
    open_positions = (
        db.query(BrokerPosition)
        .filter(BrokerPosition.status == "open")
        .order_by(BrokerPosition.entry_date.desc())
        .all()
    )

    closed_positions = (
        db.query(BrokerPosition)
        .filter(BrokerPosition.status.in_(["closed", "delisted"]))
        .order_by(BrokerPosition.exit_date.desc())
        .limit(20)
        .all()
    )

    recent_orders = (
        db.query(BrokerOrder)
        .order_by(BrokerOrder.created_at.desc())
        .limit(20)
        .all()
    )

    # Get account info from Alpaca if enabled
    account_info = None
    if _alpaca_cfg.enabled and _alpaca_cfg.api_key:
        try:
            client = _get_trading_client()
            account = client.get_account()
            account_info = {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            }

            # Add SPY parking info if enabled
            if _alpaca_cfg.spy_parking_enabled:
                spy_shares, spy_value = _get_spy_position(client)
                account_info["spy_parking"] = {
                    "shares": spy_shares,
                    "market_value": spy_value,
                    "buffer_pct": _alpaca_cfg.spy_parking_buffer,
                }
        except Exception as e:
            logger.warning("Failed to fetch Alpaca account: %s", e)

    return {
        "account": account_info,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
        "recent_orders": recent_orders,
        "alpaca_enabled": _alpaca_cfg.enabled,
    }


def rebalance_spy_parking() -> None:
    """Rebalance idle cash into/out of SPY to maintain target buffer.

    Buys SPY when cash exceeds buffer, sells when cash is below.
    Only runs when both alpaca.enabled and alpaca.spy_parking_enabled are True.
    """
    if not _alpaca_cfg.enabled or not _alpaca_cfg.spy_parking_enabled:
        return

    try:
        client = _get_trading_client()
        account = client.get_account()
    except Exception as e:
        logger.error("SPY rebalance: failed to get account: %s", e)
        return

    equity = float(account.equity)
    cash = float(account.cash)
    target_cash = equity * _alpaca_cfg.spy_parking_buffer
    excess = cash - target_cash

    spy_shares, spy_value = _get_spy_position(client)

    if excess > 100:
        buy_amount = round(excess, 2)
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_data = MarketOrderRequest(
                symbol="SPY",
                notional=buy_amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order_data=order_data)
            logger.info("SPY parking: bought $%.0f of SPY (cash $%.0f -> target $%.0f)",
                        buy_amount, cash, target_cash)
        except Exception as e:
            logger.error("SPY parking: failed to buy SPY: %s", e)

    elif excess < -100 and spy_shares > 0:
        sell_amount = min(round(abs(excess), 2), round(spy_value, 2))
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_data = MarketOrderRequest(
                symbol="SPY",
                notional=sell_amount,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order_data=order_data)
            logger.info("SPY parking: sold $%.0f of SPY (cash $%.0f -> target $%.0f)",
                        sell_amount, cash, target_cash)
        except Exception as e:
            logger.error("SPY parking: failed to sell SPY: %s", e)
    else:
        logger.debug("SPY parking: no rebalance needed (cash $%.0f, target $%.0f)",
                      cash, target_cash)
