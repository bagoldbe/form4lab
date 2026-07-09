"""Covers the shipped CLI subset (see form4lab/cli.py)."""
import re
import signal as signal_module
import sqlite3
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from form4lab.cli import cli
from form4lab.config import settings


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "init-db" in result.output
    assert "backfill" in result.output


def test_init_db_runs_migrations(tmp_path, monkeypatch):
    """`init-db` must go through Alembic (not a bare `create_all`) so it
    stamps `alembic_version` — otherwise a later `alembic upgrade` finds no
    version row and tries (and fails) to re-create every table `create_all`
    already made.

    Isolated on its own scratch sqlite file: `settings` is a module-level
    singleton constructed once (at conftest import time) from the process
    env, so it must be monkeypatched directly rather than relying on
    CliRunner's `env=` (which only patches `os.environ`, which `settings`
    has already finished reading).
    """
    scratch_db = tmp_path / "scratch_t3.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{scratch_db}")

    runner = CliRunner()
    result = runner.invoke(cli, ["init-db"])
    assert result.exit_code == 0, result.output
    assert "schema at head" in result.output.lower()

    conn = sqlite3.connect(str(scratch_db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        expected_tables = {
            "companies", "insiders", "price_data", "short_volume",
            "company_filing_events", "earnings_dates", "form4_filing_meta",
            "fundamentals", "insider_roles", "insider_scores", "transactions",
            "alerts", "backtest_results", "form4_deriv_txns", "form4_footnotes",
            "trade_outcomes", "broker_orders", "broker_positions",
        }
        assert expected_tables.issubset(tables), tables - expected_tables

        assert "alembic_version" in tables
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        assert row == ("0001",), row
    finally:
        conn.close()


def test_backfill_requires_ticker():
    """Backfill should fail without --ticker option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["backfill"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_compute_outcomes_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["compute-outcomes", "--help"])
    assert result.exit_code == 0
    assert "compute-outcomes" in result.output.lower() or "forward" in result.output.lower()


def test_refresh_scores_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["refresh-scores", "--help"])
    assert result.exit_code == 0
    assert "refresh" in result.output.lower() or "scores" in result.output.lower()


def test_score_insider_not_found():
    runner = CliRunner()
    result = runner.invoke(cli, ["score-insider", "--cik", "9999999"])
    assert "not found" in result.output.lower()


def test_simulate_portfolio_has_strategy_option():
    """--strategy is the pluggable-strategy entry point (module:Class path
    -> load_strategy()); there is no --composite mode."""
    runner = CliRunner()
    result = runner.invoke(cli, ["simulate-portfolio", "--help"])
    assert result.exit_code == 0
    assert "--strategy" in result.output
    assert "--composite" not in result.output


def test_alpaca_status_command():
    """alpaca-status should run without error (disabled by default)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["alpaca-status"])
    assert result.exit_code == 0
    assert "DISABLED" in result.output or "Account Status" in result.output


def test_sync_alpaca_command():
    """sync-alpaca should run without error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["sync-alpaca"])
    assert result.exit_code == 0
    assert "Synced" in result.output


def test_scheduler_dry_run_builds_and_exits():
    """`form4lab scheduler --dry-run` builds the real scheduler and reports its
    job count, without calling .start() or blocking on signal.pause()."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scheduler", "--dry-run"])
    assert result.exit_code == 0
    match = re.search(r"scheduler built: (\d+) jobs", result.output)
    assert match is not None, result.output
    assert int(match.group(1)) > 0


def test_scheduler_registers_sigterm_handler():
    """Rider R2a (Docker task): under `docker stop`, this process is
    container PID 1, which does not inherit the default SIGTERM disposition
    a normal process gets — without an explicit handler, `signal.pause()`
    would ignore SIGTERM and Docker would wait out the full stop timeout
    before SIGKILL. The blocking (non-dry-run) path must register a handler
    that shuts the scheduler down cleanly.

    Kept light per plan: mocks `create_scheduler` (no real APScheduler
    instance) and `signal.pause` (so the command returns instead of
    blocking forever waiting for a real signal) and only asserts that SOME
    non-default handler got installed — it does not send a real signal or
    exercise the handler's shutdown body. `signal.signal` mutates
    process-global state, so the original dispositions are restored in a
    `finally` to avoid leaking into other tests.
    """
    original_sigterm = signal_module.getsignal(signal_module.SIGTERM)
    original_sigint = signal_module.getsignal(signal_module.SIGINT)
    try:
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = []
        with patch("form4lab.scheduler.jobs.create_scheduler", return_value=mock_sched), \
             patch("signal.pause"):
            runner = CliRunner()
            result = runner.invoke(cli, ["scheduler"])

        assert result.exit_code == 0, result.output
        mock_sched.start.assert_called_once()

        sigterm_handler = signal_module.getsignal(signal_module.SIGTERM)
        sigint_handler = signal_module.getsignal(signal_module.SIGINT)
        assert callable(sigterm_handler)
        assert sigterm_handler not in (signal_module.SIG_DFL, signal_module.SIG_IGN)
        assert callable(sigint_handler)
        assert sigint_handler not in (signal_module.SIG_DFL, signal_module.SIG_IGN)
    finally:
        signal_module.signal(signal_module.SIGTERM, original_sigterm)
        signal_module.signal(signal_module.SIGINT, original_sigint)
