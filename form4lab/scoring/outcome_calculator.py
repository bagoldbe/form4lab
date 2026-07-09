import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.company import Company
from form4lab.data.price_fetcher import PriceProvider

logger = logging.getLogger(__name__)

TRADING_DAYS_MAP = {20: 30, 60: 90, 120: 180}  # trading days -> approximate calendar days


def compute_forward_return(price_start: float, price_end: float) -> float:
    """Compute simple return between two prices."""
    if price_start == 0:
        return 0.0
    return (price_end / price_start) - 1.0


def compute_excess_return(stock_return: float, benchmark_return: float) -> float:
    """Stock return minus benchmark return."""
    return stock_return - benchmark_return


def _get_price_at_date(prices_df, target_date: date, tolerance_days: int = 5) -> float | None:
    """Get the closing price on or near a target date.

    Looks for the closest available price within tolerance_days.
    """
    if prices_df.empty:
        return None

    # prices_df has columns including 'date' and 'adj_close' (split/dividend adjusted)
    for offset in range(tolerance_days + 1):
        for d in [target_date + timedelta(days=offset), target_date - timedelta(days=offset)]:
            match = prices_df[prices_df["date"] == d]
            if not match.empty:
                return float(match.iloc[0]["adj_close"])
    return None


def _get_trading_date_offset(prices_df, start_date: date, trading_days: int) -> date | None:
    """Get the date that is approximately N trading days after start_date."""
    if prices_df.empty:
        return None

    future_prices = prices_df[prices_df["date"] > start_date].sort_values("date")
    if len(future_prices) >= trading_days:
        return future_prices.iloc[trading_days - 1]["date"]
    return None


def compute_prior_momentum(ticker: str, transaction_date: date,
                           db: Session, price_provider: PriceProvider) -> float | None:
    """Compute stock return in the 20 trading days BEFORE the buy date."""
    start = transaction_date - timedelta(days=40)  # extra buffer for weekends/holidays
    end = transaction_date - timedelta(days=1)

    prices = price_provider.get_daily_prices(ticker, start, end)
    if prices.empty or len(prices) < 5:
        return None

    prices = prices.sort_values("date")
    # Get the price ~20 trading days before and the price right before the buy
    if len(prices) >= 20:
        price_start = float(prices.iloc[-20]["adj_close"])
    else:
        price_start = float(prices.iloc[0]["adj_close"])
    price_end = float(prices.iloc[-1]["adj_close"])

    return compute_forward_return(price_start, price_end)


def compute_trade_outcomes(transaction_id: int, db: Session,
                           price_provider: PriceProvider) -> TradeOutcome | None:
    """Compute forward returns for a discretionary buy transaction."""
    txn = db.get(Transaction, transaction_id)
    if not txn or not txn.is_discretionary:
        return None

    company = db.get(Company, txn.company_id)
    if not company or not company.ticker:
        return None

    ticker = company.ticker
    txn_date = txn.transaction_date

    # Fetch a wide price range to cover all windows
    start = txn_date - timedelta(days=40)   # for prior momentum
    end = txn_date + timedelta(days=200)    # for 120 trading day window

    stock_prices = price_provider.get_daily_prices(ticker, start, end)
    spy_prices = price_provider.get_daily_prices("SPY", start, end)

    # Get sector ETF prices
    sector_etf = price_provider.get_sector_etf(company.sector) if company.sector else None
    sector_prices = price_provider.get_daily_prices(sector_etf, start, end) if sector_etf else None

    # Get base price
    base_price = _get_price_at_date(stock_prices, txn_date)
    spy_base = _get_price_at_date(spy_prices, txn_date)
    sector_base = _get_price_at_date(sector_prices, txn_date) if sector_prices is not None else None

    if base_price is None or spy_base is None:
        return None

    outcome = TradeOutcome(transaction_id=transaction_id)

    # Compute prior momentum
    outcome.prior_momentum_20d = compute_prior_momentum(ticker, txn_date, db, price_provider)

    # Compute forward returns at each window
    for trading_days, label in [(20, "20d"), (60, "60d"), (120, "120d")]:
        forward_date = _get_trading_date_offset(
            stock_prices[stock_prices["date"] >= txn_date],
            txn_date, trading_days
        )
        if forward_date is None:
            continue

        stock_fwd = _get_price_at_date(stock_prices, forward_date)
        spy_fwd = _get_price_at_date(spy_prices, forward_date)

        if stock_fwd is None or spy_fwd is None:
            continue

        stock_ret = compute_forward_return(base_price, stock_fwd)
        bench_ret = compute_forward_return(spy_base, spy_fwd)
        excess_ret = compute_excess_return(stock_ret, bench_ret)

        setattr(outcome, f"stock_return_{label}", stock_ret)
        setattr(outcome, f"benchmark_return_{label}", bench_ret)
        setattr(outcome, f"excess_return_{label}", excess_ret)
        setattr(outcome, f"hit_{label}", excess_ret > 0)

        # Sector returns
        if sector_base is not None and sector_prices is not None:
            sector_fwd = _get_price_at_date(sector_prices, forward_date)
            if sector_fwd is not None:
                setattr(outcome, f"sector_return_{label}",
                        compute_forward_return(sector_base, sector_fwd))

    db.add(outcome)
    db.commit()
    return outcome


def batch_compute_outcomes(db: Session, price_provider: PriceProvider) -> int:
    """Compute outcomes for all discretionary buys missing outcomes."""
    from sqlalchemy import exists as sa_exists

    # Load just IDs upfront — avoids lazy-load crashes if connection drops mid-loop
    pending_ids = [
        row[0] for row in db.query(Transaction.id).filter(
            Transaction.is_discretionary == True,  # noqa: E712
            ~sa_exists().where(TradeOutcome.transaction_id == Transaction.id)
        ).all()
    ]

    count = 0
    for txn_id in pending_ids:
        try:
            outcome = compute_trade_outcomes(txn_id, db, price_provider)
            if outcome:
                count += 1
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to compute outcome for transaction {txn_id}: {e}")

    return count
