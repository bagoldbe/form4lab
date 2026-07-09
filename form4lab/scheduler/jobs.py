import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)


def ingest_daily_filings_job():
    """Check tracked companies for new Form 4 filings, generate alerts, and execute trades."""
    from form4lab.database import SessionLocal
    from form4lab.data.sec_fetcher import ingest_daily_filings
    from form4lab.services.alert_service import generate_and_execute_alerts

    logger.info("Running daily filing ingest...")
    with SessionLocal() as db:
        # 7-day lookback catches filings missed due to SEC /submissions/ cache lag
        # and weekend gaps (Friday-late filings fall outside a 2-day window by Monday).
        count = ingest_daily_filings(db, days_back=7)
        logger.info(f"Ingested {count} new transactions")

        generated, executed = generate_and_execute_alerts(db)
        logger.info(
            "Daily ingest: %d alerts generated, %d orders placed",
            generated, executed,
        )


def compute_outcomes_job():
    """Compute forward returns for pending trades."""
    from form4lab.database import SessionLocal
    from form4lab.data.price_fetcher import YFinanceProvider
    from form4lab.scoring.outcome_calculator import batch_compute_outcomes

    logger.info("Computing trade outcomes...")
    with SessionLocal() as db:
        provider = YFinanceProvider(db)
        count = batch_compute_outcomes(db, provider)
    logger.info(f"Computed outcomes for {count} trades")


def refresh_scores_job():
    """Recompute insider scores."""
    from form4lab.database import SessionLocal
    from form4lab.scoring.insider_scorer import refresh_all_scores

    logger.info("Refreshing insider scores...")
    with SessionLocal() as db:
        count = refresh_all_scores(db)
    logger.info(f"Refreshed scores for {count} insiders")


def backfill_prices_job():
    """Ensure price data is up to date for all tracked tickers."""
    from datetime import date, timedelta
    from form4lab.database import SessionLocal
    from form4lab.models.company import Company
    from form4lab.data.price_fetcher import YFinanceProvider, SECTOR_ETF_MAP

    logger.info("Backfilling prices...")
    with SessionLocal() as db:
        provider = YFinanceProvider(db)

        # Get all tickers in the database
        tickers = [
            r[0]
            for r in db.query(Company.ticker)
            .filter(Company.ticker.isnot(None))
            .distinct()
            .all()
        ]

        # Add SPY and sector ETFs
        tickers.append("SPY")
        tickers.extend(SECTOR_ETF_MAP.values())
        tickers = list(set(tickers))

        today = date.today()
        start = today - timedelta(days=5)  # Just get recent days

        for ticker in tickers:
            try:
                provider.get_daily_prices(ticker, start, today)
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")

    logger.info(f"Price backfill complete for {len(tickers)} tickers")


# ---------------------------------------------------------------------------
# Alpaca paper trading jobs
# ---------------------------------------------------------------------------

_last_poll_time: datetime | None = None


def continuous_ingestion_job():
    """Poll SEC EFTS for new Form 4 filings and execute tradeable signals.

    Only runs Mon-Fri between start_hour and end_hour (US/Eastern).
    SEC EDGAR doesn't receive filings outside business hours.
    """
    global _last_poll_time
    import pytz
    from form4lab.config import settings

    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)

    # Skip weekends (5=Saturday, 6=Sunday)
    if now_et.weekday() >= 5:
        return

    # Skip outside configured hours
    cfg = settings.scheduler
    if now_et.hour < cfg.continuous_ingestion_start_hour:
        return
    if now_et.hour >= cfg.continuous_ingestion_end_hour:
        return

    from form4lab.database import SessionLocal
    from form4lab.data.sec_fetcher import poll_recent_filings
    from form4lab.services.alert_service import generate_and_execute_alerts

    if _last_poll_time is None:
        _last_poll_time = datetime.now(timezone.utc) - timedelta(hours=2)

    with SessionLocal() as db:
        new_txns = poll_recent_filings(db, _last_poll_time)
        if new_txns > 0:
            generated, executed = generate_and_execute_alerts(db)
            logger.info(
                "Continuous ingest: %d txns, %d alerts, %d orders placed",
                new_txns, generated, executed,
            )
    _last_poll_time = datetime.now(timezone.utc)


def check_exits_job():
    """Close positions past their hold period or hitting stop loss."""
    from form4lab.database import SessionLocal
    from form4lab.services.alpaca_service import (
        get_positions_to_close, get_stop_loss_positions, close_position,
    )

    logger.info("Checking for positions to close...")
    with SessionLocal() as db:
        expired = get_positions_to_close(db)
        for pos in expired:
            try:
                close_position(pos, db)
            except Exception as e:
                db.rollback()
                logger.error("Failed to close expired position %d (%s): %s", pos.id, pos.symbol, e)

        stopped = get_stop_loss_positions(db)
        for pos in stopped:
            try:
                close_position(pos, db)
            except Exception as e:
                db.rollback()
                logger.error("Failed to close stop-loss position %d (%s): %s", pos.id, pos.symbol, e)

        logger.info(
            "Submitted close orders: %d expired, %d stop-loss",
            len(expired), len(stopped),
        )


def sync_orders_job():
    """Sync order statuses from Alpaca API, reconcile positions, and check stop losses."""
    from form4lab.database import SessionLocal
    from form4lab.services.alpaca_service import (
        sync_orders, reconcile_positions, get_stop_loss_positions, close_position,
    )

    with SessionLocal() as db:
        updated = sync_orders(db)
        if updated > 0:
            logger.info("Synced %d order statuses", updated)

        # Reconcile orphaned positions (manual closes, margin calls, etc.)
        try:
            reconciled = reconcile_positions(db)
            if reconciled > 0:
                logger.info("Reconciled %d orphaned positions", reconciled)
        except Exception as e:
            logger.error("Position reconciliation failed: %s", e)

        stopped = get_stop_loss_positions(db)
        for pos in stopped:
            try:
                close_position(pos, db)
            except Exception as e:
                db.rollback()
                logger.error("Failed to close stop-loss position %d (%s) in sync job: %s", pos.id, pos.symbol, e)
        if stopped:
            logger.info("Stop-loss closed %d position(s)", len(stopped))


def rebalance_spy_job():
    """Rebalance idle cash into SPY (parking).

    Runs after check_exits_job so proceeds from closed positions are parked.
    """
    from form4lab.services.alpaca_service import rebalance_spy_parking

    logger.info("Running SPY parking rebalance...")
    rebalance_spy_parking()
    logger.info("SPY parking rebalance complete")


def execution_health_job():
    """Daily check: did tradeable alerts produce broker positions?

    Logs a structured line prefixed with EXEC_HEALTH for log greppability.
    Emits at ERROR when alerts > 0 and positions == 0 (a silent execution
    freeze); INFO otherwise.
    """
    from form4lab.database import SessionLocal
    from form4lab.services.execution_health import check_execution_health, LOG_MARKER

    with SessionLocal() as db:
        report = check_execution_health(db)

    line = (
        f"{LOG_MARKER} window={report.window_days}d "
        f"alerts={report.tradeable_alerts} "
        f"positions={report.positions_opened} "
        f"healthy={report.healthy} reason={report.reason!r}"
    )
    if report.healthy:
        logger.info(line)
    else:
        logger.error(line)


def reconciliation_health_job():
    """Daily check: any delisted closures or positions held for manual review?

    Logs a structured line prefixed with RECON_HEALTH. ERROR when anomalies exist.
    """
    from form4lab.database import SessionLocal
    from form4lab.services.reconciliation_health import check_reconciliation_health, LOG_MARKER

    with SessionLocal() as db:
        report = check_reconciliation_health(db)

    line = (
        f"{LOG_MARKER} window={report.window_days}d "
        f"delisted={report.delisted_count} orphan={report.orphan_count} held={report.held_count} "
        f"healthy={report.healthy} reason={report.reason!r}"
    )
    if report.healthy:
        logger.info(line)
    else:
        logger.error(line)


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the APScheduler."""
    from form4lab.config import settings
    cfg = settings.scheduler

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        ingest_daily_filings_job, "cron",
        day_of_week="mon-fri", hour=cfg.ingest_hour, minute=cfg.ingest_minute,
        timezone=cfg.timezone, id="ingest_daily_filings",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    scheduler.add_job(
        compute_outcomes_job, "cron",
        day_of_week="mon-fri", hour=cfg.outcomes_hour, minute=cfg.outcomes_minute,
        timezone=cfg.timezone, id="compute_outcomes",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    scheduler.add_job(
        refresh_scores_job, "cron",
        day_of_week="mon-fri", hour=cfg.scores_hour, minute=cfg.scores_minute,
        timezone=cfg.timezone, id="refresh_scores",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    scheduler.add_job(
        backfill_prices_job, "cron",
        day_of_week="mon-fri", hour=cfg.prices_hour, minute=cfg.prices_minute,
        timezone=cfg.timezone, id="backfill_prices",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    # Continuous SEC ingestion — every 60 seconds
    # max_instances=1 prevents concurrent execution (protects _last_poll_time)
    scheduler.add_job(
        continuous_ingestion_job, "interval",
        seconds=cfg.continuous_ingestion_interval_seconds,
        id="continuous_ingestion",
        misfire_grace_time=cfg.misfire_grace_time,
        max_instances=1,
    )

    # Check exits — daily at 9:25 AM ET (before market open)
    scheduler.add_job(
        check_exits_job, "cron",
        day_of_week="mon-fri", hour=cfg.exits_hour, minute=cfg.exits_minute,
        timezone=cfg.timezone, id="check_exits",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    # Sync order statuses — every 5 minutes
    scheduler.add_job(
        sync_orders_job, "interval",
        minutes=cfg.sync_interval_minutes,
        id="sync_orders",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    # SPY parking rebalance — daily at 9:26 AM ET (right after exits close)
    scheduler.add_job(
        rebalance_spy_job, "cron",
        day_of_week="mon-fri", hour=cfg.spy_rebalance_hour, minute=cfg.spy_rebalance_minute,
        timezone=cfg.timezone, id="rebalance_spy",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    # Execution health — daily at 9:00 AM ET (after overnight ingestion, before exits)
    scheduler.add_job(
        execution_health_job, "cron",
        day_of_week="mon-fri", hour=9, minute=0,
        timezone=cfg.timezone, id="execution_health",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    # Reconciliation health — daily at 9:01 AM ET (right after execution health)
    scheduler.add_job(
        reconciliation_health_job, "cron",
        day_of_week="mon-fri", hour=9, minute=1,
        timezone=cfg.timezone, id="reconciliation_health",
        misfire_grace_time=cfg.misfire_grace_time,
    )

    return scheduler
