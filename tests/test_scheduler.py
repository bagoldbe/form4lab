import asyncio
from unittest.mock import patch, MagicMock

from form4lab.scheduler.jobs import create_scheduler, ingest_daily_filings_job


def test_create_scheduler():
    scheduler = create_scheduler()
    assert scheduler is not None
    jobs = scheduler.get_jobs()
    assert len(jobs) == 10
    job_ids = {j.id for j in jobs}
    assert "ingest_daily_filings" in job_ids
    assert "compute_outcomes" in job_ids
    assert "refresh_scores" in job_ids
    assert "backfill_prices" in job_ids
    assert "continuous_ingestion" in job_ids
    assert "check_exits" in job_ids
    assert "sync_orders" in job_ids
    assert "rebalance_spy" in job_ids
    assert "execution_health" in job_ids
    assert "reconciliation_health" in job_ids


def test_daily_ingest_job_executes_tradeable_alerts():
    """Daily ingest job must call generate_and_execute_alerts (not just generate_missing_alerts).

    Regression test: previously the daily job called generate_missing_alerts which
    only creates alerts without executing them on Alpaca. The continuous ingestion
    job then skipped these alerts because they already existed. This caused
    tradeable buy signals ingested at midnight to never be executed.
    """
    mock_db = MagicMock()
    mock_session = MagicMock()
    mock_db.__enter__ = MagicMock(return_value=mock_session)
    mock_db.__exit__ = MagicMock(return_value=False)

    with (
        patch("form4lab.database.SessionLocal", return_value=mock_db),
        patch("form4lab.data.sec_fetcher.ingest_daily_filings", return_value=3) as mock_ingest,
        patch(
            "form4lab.services.alert_service.generate_and_execute_alerts",
            return_value=(2, 1),
        ) as mock_gen_exec,
        patch(
            "form4lab.services.alert_service.generate_missing_alerts",
        ) as mock_gen_only,
    ):
        ingest_daily_filings_job()

        mock_ingest.assert_called_once_with(mock_session, days_back=7)
        mock_gen_exec.assert_called_once_with(mock_session)
        mock_gen_only.assert_not_called()


def test_daily_ingest_generates_alerts_even_with_zero_new_filings():
    """generate_and_execute_alerts must run even when no new filings are ingested.

    Previously-ingested transactions (from continuous ingestion or backfills)
    may still need alerts generated. Skipping on count==0 would lose those.
    """
    mock_db = MagicMock()
    mock_session = MagicMock()
    mock_db.__enter__ = MagicMock(return_value=mock_session)
    mock_db.__exit__ = MagicMock(return_value=False)

    with (
        patch("form4lab.database.SessionLocal", return_value=mock_db),
        patch("form4lab.data.sec_fetcher.ingest_daily_filings", return_value=0),
        patch(
            "form4lab.services.alert_service.generate_and_execute_alerts",
            return_value=(0, 0),
        ) as mock_gen_exec,
    ):
        ingest_daily_filings_job()
        mock_gen_exec.assert_called_once_with(mock_session)


def test_daily_ingest_runs_after_submissions_cache_refresh():
    """Daily ingest must run after SEC's /submissions/ cache rebuilds at 8pm ET.

    SEC's submissions.json endpoint is updated around 8pm ET; running the daily
    ingest at 7pm ET returns stale data, so same-day afternoon filings are
    silently missed. Shift to 9pm ET to read the refreshed cache.
    """
    scheduler = create_scheduler()
    job = scheduler.get_job("ingest_daily_filings")
    assert job is not None
    assert job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")].expressions[0].first == 21


# --- SCHEDULER_ENABLED lifespan gate ---
#
# TestClient(app) without an explicit `with` block never drives lifespan
# events in the installed starlette version (only __enter__/__exit__ send
# the lifespan.startup/shutdown ASGI messages — see tests/test_app.py, which
# uses a bare TestClient and therefore never touches the scheduler at all).
# Drive the @asynccontextmanager directly instead so these tests exercise
# exactly form4lab.main's startup/shutdown gating logic.

def _drive_lifespan():
    """Run form4lab.main's lifespan startup and shutdown once, synchronously."""
    from form4lab.main import app, lifespan

    async def _once():
        async with lifespan(app):
            pass

    asyncio.run(_once())


def test_lifespan_skips_scheduler_when_disabled(monkeypatch):
    """SCHEDULER_ENABLED=false: the web process must not build or start its own
    scheduler — a separate scheduler container/process owns the jobs instead."""
    from form4lab.config import settings
    monkeypatch.setattr(settings, "scheduler_enabled", False)

    with patch("form4lab.scheduler.jobs.create_scheduler") as mock_create:
        _drive_lifespan()
        mock_create.assert_not_called()


def test_lifespan_starts_scheduler_when_enabled(monkeypatch):
    """SCHEDULER_ENABLED=true (default): the web process still starts and later
    shuts down its own in-process scheduler, unchanged from before this task."""
    from form4lab.config import settings
    monkeypatch.setattr(settings, "scheduler_enabled", True)

    mock_scheduler = MagicMock()
    with patch("form4lab.scheduler.jobs.create_scheduler", return_value=mock_scheduler) as mock_create:
        _drive_lifespan()
        mock_create.assert_called_once()
        mock_scheduler.start.assert_called_once()
        mock_scheduler.shutdown.assert_called_once()
