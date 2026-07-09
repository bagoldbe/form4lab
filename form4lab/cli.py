from datetime import date

import click
import yfinance as yf

from form4lab.models import *  # noqa: F401, F403 — ensure all models registered


@click.group()
def cli():
    """form4lab management commands."""
    pass


@cli.command()
def init_db():
    """Create the database schema by running Alembic migrations to head."""
    from alembic import command
    from alembic.config import Config as AlembicConfig
    from pathlib import Path
    ini = Path(__file__).resolve().parent.parent / "alembic.ini"
    cfg = AlembicConfig(str(ini))
    cfg.set_main_option("script_location", str(ini.parent / "alembic"))
    # Don't let Alembic's env.py hijack the host process's logging config
    # (logging.config.fileConfig defaults to disable_existing_loggers=True,
    # which would silently kill every form4lab.* logger for the rest of the
    # process — see alembic/env.py's matching guard).
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")
    click.echo("schema at head")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker to backfill")
@click.option("--years", default=10, help="Years of history to fetch")
def backfill(ticker, years):
    """Backfill Form 4 data for a company."""
    from form4lab.database import SessionLocal
    from form4lab.data.sec_fetcher import backfill_company_fast

    with SessionLocal() as db:
        count = backfill_company_fast(ticker, years, db)
    click.echo(f"Backfill complete for {ticker} ({count} transactions)")


@cli.command()
@click.option("--file", "ticker_file", required=True, type=click.Path(exists=True),
              help="Text file with one ticker per line")
@click.option("--years", default=10)
@click.option("--force", is_flag=True, default=False, help="Re-backfill even if data exists")
@click.option("--workers", default=4, help="Number of parallel workers (default 4)")
def backfill_tickers(ticker_file, years, force, workers):
    """Backfill Form 4 data from a file of tickers (one per line)."""
    import concurrent.futures
    from form4lab.database import SessionLocal
    from form4lab.data.sec_fetcher import backfill_company_fast
    from form4lab.data.utils import company_already_backfilled

    with open(ticker_file) as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    # Filter out already-backfilled tickers
    todo = []
    skipped = 0
    for ticker in tickers:
        with SessionLocal() as db:
            if not force and company_already_backfilled(ticker, db):
                skipped += 1
                continue
        todo.append(ticker)

    click.echo(f"Loaded {len(tickers)} tickers, {skipped} already done, {len(todo)} to backfill ({workers} workers)")

    done = 0
    errors = 0

    def _backfill_one(ticker):
        with SessionLocal() as db:
            return backfill_company_fast(ticker, years, db)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_backfill_one, t): t for t in todo}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
                done += 1
                click.echo(f"  [{done + errors}/{len(todo)}] Done: {ticker}")
            except Exception as e:
                errors += 1
                click.echo(f"  [{done + errors}/{len(todo)}] Error {ticker}: {e}")

    click.echo(f"\nFinished: {done} backfilled, {skipped} skipped, {errors} errors.")


@cli.command()
@click.option("--file", "ticker_file", required=True, type=click.Path(exists=True),
              help="Text file with one ticker per line")
@click.option("--years", default=10)
@click.option("--workers", default=1, help="Number of parallel workers (default 1; keep low to avoid SEC rate limits)")
def backfill_fast(ticker_file, years, workers):
    """Fast backfill using direct SEC APIs."""
    import concurrent.futures
    from form4lab.database import SessionLocal
    from form4lab.data.sec_fetcher import backfill_company_fast
    from form4lab.data.utils import company_already_backfilled

    with open(ticker_file) as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    todo = []
    skipped = 0
    for ticker in tickers:
        with SessionLocal() as db:
            if company_already_backfilled(ticker, db):
                skipped += 1
                continue
        todo.append(ticker)

    click.echo(f"Loaded {len(tickers)} tickers, {skipped} already done, {len(todo)} to backfill ({workers} workers)")

    done = 0
    errors = 0

    def _backfill_one(ticker):
        with SessionLocal() as db:
            return backfill_company_fast(ticker, years, db)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_backfill_one, t): t for t in todo}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                count = future.result()
                done += 1
                click.echo(f"  [{done + errors}/{len(todo)}] Done: {ticker} ({count} txns)")
            except Exception as e:
                errors += 1
                click.echo(f"  [{done + errors}/{len(todo)}] Error {ticker}: {e}")

    click.echo(f"\nFinished: {done} backfilled, {skipped} skipped, {errors} errors.")


@cli.command("backfill-bulk")
@click.option("--file", "ticker_file", required=True, type=click.Path(exists=True),
              help="Text file with one ticker per line")
@click.option("--years", default=10)
@click.option("--redownload", is_flag=True, default=False,
              help="Re-download ZIPs even if cached on disk")
def backfill_bulk(ticker_file, years, redownload):
    """Bulk backfill Form 4 data using SEC quarterly data sets.

    Downloads ~40 quarterly ZIP files instead of 13,000+ individual XML fetches.
    Completes in ~10 minutes vs ~1.5 hours for the per-filing approach.
    """
    from form4lab.database import SessionLocal
    from form4lab.data.bulk_fetcher import backfill_from_bulk

    with SessionLocal() as db:
        stats = backfill_from_bulk(ticker_file, years, db, redownload=redownload)

    click.echo(f"\nBulk backfill complete:")
    click.echo(f"  Quarters downloaded: {stats['quarters_downloaded']}")
    click.echo(f"  Quarters cached:     {stats['quarters_cached']}")
    click.echo(f"  Quarters missing:    {stats['quarters_missing']}")
    click.echo(f"  Transactions parsed: {stats['transactions_parsed']}")
    click.echo(f"  Transactions saved:  {stats['transactions_persisted']}")
    click.echo(f"  Errors:              {stats['errors']}")


@cli.command("backfill-10b51")
@click.option("--redownload", is_flag=True, default=False,
              help="Re-download ZIPs even if cached on disk")
def backfill_10b51(redownload):
    """Retroactively populate 10b5-1 plan flags on existing transactions.

    Downloads 2023 Q2+ quarterly ZIPs and updates existing transactions
    that have is_10b5_1_plan IS NULL with the flag from SUBMISSION.tsv.
    """
    from form4lab.database import SessionLocal
    from form4lab.data.bulk_fetcher import backfill_10b5_1_flags

    with SessionLocal() as db:
        stats = backfill_10b5_1_flags(db, redownload=redownload)

    click.echo(f"\n10b5-1 backfill complete:")
    click.echo(f"  Quarters processed:    {stats['quarters_processed']}")
    click.echo(f"  Quarters missing:      {stats['quarters_missing']}")
    click.echo(f"  Accessions with flag:  {stats['accessions_with_flag']}")
    click.echo(f"  Transactions updated:  {stats['transactions_updated']}")
    click.echo(f"  Errors:                {stats['errors']}")


@cli.command()
def compute_outcomes():
    """Compute forward returns for all pending trades."""
    from form4lab.database import SessionLocal
    from form4lab.data.price_fetcher import YFinanceProvider
    from form4lab.scoring.outcome_calculator import batch_compute_outcomes

    with SessionLocal() as db:
        provider = YFinanceProvider(db, db_only=True)
        count = batch_compute_outcomes(db, provider)
    click.echo(f"Computed outcomes for {count} trades.")


@cli.command()
def refresh_scores():
    """Recompute all insider scores."""
    from form4lab.database import SessionLocal
    from form4lab.scoring.insider_scorer import refresh_all_scores

    with SessionLocal() as db:
        count = refresh_all_scores(db)
    click.echo(f"Refreshed scores for {count} insiders.")


@cli.command()
@click.option("--cik", required=True)
def score_insider(cik):
    """Display score for a specific insider."""
    from form4lab.database import SessionLocal
    from form4lab.scoring.insider_scorer import compute_insider_score
    from form4lab.models.insider import Insider

    with SessionLocal() as db:
        insider = db.query(Insider).filter(Insider.cik == cik).first()
        if not insider:
            click.echo(f"Insider with CIK {cik} not found.")
            return
        score = compute_insider_score(insider.id, db)
        click.echo(f"Name: {insider.name}")
        click.echo(f"Tier: {score.credibility_tier}")
        click.echo(f"Skill Score: {score.skill_score:.3f}")
        if score.bayesian_hit_rate is not None:
            click.echo(f"Hit Rate ({score.horizon_days}d): {score.bayesian_hit_rate:.1%}")
        else:
            click.echo("Hit Rate: N/A")
        if score.shrunk_excess_return is not None:
            click.echo(f"Excess Return ({score.horizon_days}d): {score.shrunk_excess_return:.1%}")
        else:
            click.echo("Excess Return: N/A")
        click.echo(f"Buys: {score.num_discretionary_buys}")


@cli.command()
def generate_signals():
    """Generate alerts/signals for all discretionary buys missing alerts."""
    from form4lab.database import SessionLocal
    from form4lab.services.alert_service import generate_missing_alerts

    with SessionLocal() as db:
        generated = generate_missing_alerts(db)
        click.echo(f"Generated {generated} alerts.")


@cli.command()
@click.option("--file", "ticker_file", required=True, type=click.Path(exists=True),
              help="Text file with one ticker per line")
@click.option("--years", default=10, help="Years of price history to fetch")
def prefetch_prices(ticker_file, years):
    """Pre-fetch price data for all tickers from Yahoo Finance (bulk download)."""
    from datetime import timedelta
    from form4lab.database import SessionLocal
    from form4lab.data.price_fetcher import SECTOR_ETF_MAP
    from form4lab.models.price import PriceData

    with open(ticker_file) as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    # Add SPY benchmark + all sector ETFs
    extras = ["SPY"] + list(SECTOR_ETF_MAP.values())
    all_tickers = list(dict.fromkeys(tickers + extras))  # dedupe, preserve order

    click.echo(f"Fetching {years}y price data for {len(all_tickers)} tickers ({len(tickers)} stocks + {len(extras)} benchmarks)...")

    start = date.today() - timedelta(days=years * 365 + 200)
    end = date.today()

    # yfinance bulk download — much faster than one-at-a-time
    import os, sys
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        df = yf.download(
            all_tickers,
            start=str(start),
            end=str(end),
            progress=True,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    finally:
        sys.stderr = old_stderr
        devnull.close()

    if df.empty:
        click.echo("No data returned from Yahoo Finance.")
        return

    click.echo(f"Downloaded {len(df)} trading days. Saving to database...")

    saved = 0
    errors = 0
    with SessionLocal() as db:
        for ticker in all_tickers:
            try:
                if ticker not in df.columns.get_level_values(0):
                    continue
                ticker_df = df[ticker].dropna(subset=["Close"])
                if ticker_df.empty:
                    continue

                rows = [(idx.date() if hasattr(idx, "date") else idx, row)
                        for idx, row in ticker_df.iterrows()]

                # Portable upsert (sqlite by default, Postgres if configured):
                # batch-check which (ticker, date) pairs already exist, then
                # insert only the new ones. Mirrors the check-then-insert
                # pattern already used by
                # form4lab.data.price_fetcher.YFinanceProvider._save_to_db
                # rather than a Postgres-only ON CONFLICT upsert.
                existing_dates = {
                    r.date for r in db.query(PriceData.date)
                    .filter(PriceData.ticker == ticker, PriceData.date.in_([d for d, _ in rows]))
                    .all()
                }
                for d, row in rows:
                    if d in existing_dates:
                        continue
                    db.add(PriceData(
                        ticker=ticker,
                        date=d,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        adj_close=float(row.get("Adj Close", row["Close"])),
                        volume=int(row["Volume"]),
                    ))
                db.commit()
                saved += 1
                if saved % 25 == 0:
                    click.echo(f"  {saved}/{len(all_tickers)} tickers saved...")
            except Exception as e:
                db.rollback()
                errors += 1
                click.echo(f"  Error saving {ticker}: {e}")

    click.echo(f"Done: {saved} tickers saved, {errors} errors.")


@cli.command("simulate-portfolio")
@click.option("--cash", default=10_000.0, help="Starting capital")
@click.option("--position-size", default=None, type=float,
              help="Fixed dollar amount per position (overrides percentage-based role-tiered sizing)")
@click.option("--hold-days", default=60, help="Default trading days to hold (overridden per signal type)")
@click.option("--start-date", default=None, help="Only trade on/after this date (YYYY-MM-DD)")
@click.option("--export-csv", default=None, type=click.Path(), help="Export trades to CSV")
@click.option("--chart", default=None, type=click.Path(), help="Save performance chart to file (e.g. chart.png)")
@click.option("--universe", multiple=True, type=click.Path(exists=True),
              help="Ticker list file(s) to filter simulation (e.g. form4lab/data/universes/sp400_midcap.txt)")
@click.option("--margin", default=1.0, type=float,
              help="Margin multiplier (default 1.0 = cash only; e.g. 1.5 for 50% margin)")
@click.option("--margin-rate", default=0.06, type=float,
              help="Annual interest rate on margin loans (default 0.06 = 6%, only applies when --margin > 1.0)")
@click.option("--drawdown-threshold", default=0.0, type=float,
              help="Min drawdown from 52wk high to trade (default 0 = disabled; e.g. -0.30 for 30%+ below 52wk high)")
@click.option("--spy-parking", is_flag=True, default=False,
              help="Park idle cash in the beta sleeve (SPY) between signals")
@click.option("--spy-buffer", default=0.20, type=float,
              help="Cash buffer as fraction of portfolio when SPY parking (default 0.20 = 20%)")
@click.option("--shuffle-seed", default=None, type=int,
              help="Shuffle same-day signals with this seed (order-sensitivity testing)")
@click.option("--filter-routine", is_flag=True, default=False,
              help="Skip trades from routine insiders (Cohen 2012)")
@click.option("--filter-filing-lag", default=None, type=int,
              help="Skip trades with filing lag > N days")
@click.option("--strategy", default=None,
              help="Strategy as 'module.path:ClassName' (default: the active strategy from settings.strategy_path)")
def simulate_portfolio(cash, position_size, hold_days, start_date, export_csv, chart, universe,
                       margin, margin_rate, drawdown_threshold, spy_parking, spy_buffer,
                       shuffle_seed, filter_routine, filter_filing_lag, strategy):
    """Simulate a portfolio following insider signals with capital constraints."""
    from datetime import datetime
    from form4lab.database import SessionLocal
    from form4lab.scoring.portfolio_simulator import (
        run_simulation, compute_metrics, format_report,
        export_trades_csv, build_daily_equity_curve, generate_chart,
    )

    start = None
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()

    ticker_universe = None
    if universe:
        ticker_universe = set()
        for path in universe:
            with open(path) as f:
                ticker_universe |= {
                    line.strip().upper() for line in f
                    if line.strip() and not line.strip().startswith("#")
                }
        click.echo(f"Universe: {len(ticker_universe)} tickers from {len(universe)} file(s)")

    margin_multiplier = margin
    if margin_multiplier > 1.0:
        click.echo(f"Margin: {margin_multiplier:.1f}x buying power, {margin_rate:.1%} annual interest")

    dd_threshold = drawdown_threshold if drawdown_threshold != 0 else None
    if dd_threshold is not None:
        click.echo(f"Drawdown filter: only trade when stock is {abs(dd_threshold):.0%}+ below 52wk high")

    if spy_parking:
        click.echo(f"SPY parking: enabled, {spy_buffer:.0%} cash buffer")

    strategy_obj = None
    if strategy:
        from form4lab.strategy.registry import load_strategy
        strategy_obj = load_strategy(strategy)
        click.echo(f"Strategy: {strategy_obj.name} ({strategy})")

    with SessionLocal() as db:
        portfolio, price_index = run_simulation(
            db,
            initial_cash=cash,
            position_size=position_size,
            hold_days=hold_days,
            start_date=start,
            universe=ticker_universe,
            margin_multiplier=margin_multiplier,
            margin_interest_rate=margin_rate,
            drawdown_threshold=dd_threshold,
            spy_parking=spy_parking,
            spy_parking_buffer=spy_buffer,
            shuffle_seed=shuffle_seed,
            filter_routine=filter_routine,
            filter_filing_lag=filter_filing_lag,
            strategy=strategy_obj,
        )

    metrics = compute_metrics(portfolio, price_index)
    report = format_report(metrics, portfolio)
    click.echo(report)

    if export_csv:
        n = export_trades_csv(portfolio, export_csv)
        click.echo(f"Exported {n} trades to {export_csv}")

    if chart:
        click.echo("Building daily equity curve...")
        equity_df = build_daily_equity_curve(
            portfolio, price_index,
            spy_parking_buffer=spy_buffer if spy_parking else 0.0,
        )
        if equity_df.empty:
            click.echo("Error: Could not build equity curve (missing SPY price data?)")
        else:
            try:
                generate_chart(equity_df, chart)
            except ImportError:
                click.echo("Error: charting requires matplotlib — install form4lab[chart]")
            else:
                click.echo(f"Chart saved to {chart}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--reload/--no-reload", default=False)
def run(host, port, reload):
    """Start FastAPI server with scheduler."""
    import uvicorn
    uvicorn.run("form4lab.main:app", host=host, port=port, reload=reload)


@cli.command()
@click.option("--dry-run", is_flag=True, help="Build the scheduler and exit (for tests/CI).")
def scheduler(dry_run):
    """Run the background job scheduler standalone (for a separate scheduler container)."""
    from form4lab.scheduler.jobs import create_scheduler
    sched = create_scheduler()
    if dry_run:
        click.echo(f"scheduler built: {len(sched.get_jobs())} jobs")
        return

    # Under `docker stop` this process is container PID 1, which does not
    # get the default signal dispositions a normal process would — without
    # an explicit SIGTERM handler, `signal.pause()` below ignores SIGTERM
    # entirely and Docker waits out the full stop timeout before SIGKILL.
    # Registering handlers here lets the scheduler shut down its jobs
    # cleanly and exit promptly instead.
    import signal

    def _shutdown(signum, frame):
        sched.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    sched.start()
    click.echo("scheduler started; SIGTERM/Ctrl-C to stop")
    signal.pause()


@cli.command()
def alpaca_status():
    """Show Alpaca paper trading account status and open positions."""
    from form4lab.config import settings
    cfg = settings.alpaca

    if not cfg.enabled:
        click.echo("Alpaca paper trading is DISABLED (set ALPACA_ENABLED=true)")
        return

    if not cfg.api_key:
        click.echo("Error: ALPACA_API_KEY not set")
        return

    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        account = client.get_account()
    except Exception as e:
        click.echo(f"Failed to connect to Alpaca: {e}")
        return

    click.echo(f"Account Status: {account.status}")
    click.echo(f"Equity:         ${float(account.equity):,.2f}")
    click.echo(f"Cash:           ${float(account.cash):,.2f}")
    click.echo(f"Buying Power:   ${float(account.buying_power):,.2f}")
    click.echo(f"Paper:          {cfg.paper}")

    from form4lab.database import SessionLocal
    from form4lab.models.broker import BrokerPosition
    with SessionLocal() as db:
        open_positions = db.query(BrokerPosition).filter(
            BrokerPosition.status == "open"
        ).all()

    if open_positions:
        click.echo(f"\nOpen Positions ({len(open_positions)}):")
        for pos in open_positions:
            click.echo(
                f"  {pos.symbol:6s} {pos.shares:8.1f} shares @ ${pos.entry_price:8.2f}"
                f"  exit: {pos.exit_target_date}  ({pos.insider_name}, {pos.insider_role})"
            )
    else:
        click.echo("\nNo open positions tracked.")


@cli.command()
def sync_alpaca():
    """Sync order statuses from Alpaca and process exits."""
    from form4lab.database import SessionLocal
    from form4lab.services.alpaca_service import sync_orders, get_positions_to_close, close_position

    with SessionLocal() as db:
        updated = sync_orders(db)
        click.echo(f"Synced {updated} order statuses")

        expired = get_positions_to_close(db)
        for pos in expired:
            close_position(pos, db)
        click.echo(f"Closed {len(expired)} expired positions")


@cli.command("reconcile-positions")
def reconcile_positions_cmd():
    """Reconcile open positions against Alpaca: may close (sold/orphan/delisted),
    rename (corporate action), or hold for manual review (ambiguous CA or lookup failure)."""
    from form4lab.database import SessionLocal
    from form4lab.services.alpaca_service import reconcile_positions

    with SessionLocal() as db:
        count = reconcile_positions(db)
    click.echo(f"Reconciled {count} orphaned position(s)")


@cli.command("clear-hold")
@click.option("--symbol", required=True, help="Symbol whose reconcile_hold flag to clear")
def clear_hold_cmd(symbol):
    """Clear the reconcile_hold flag for all positions with the given symbol.

    Use after a human has reviewed and resolved a held position (e.g. after
    the broker credits shares following a corporate-action rename).
    """
    from form4lab.database import SessionLocal
    from form4lab.services.alpaca_service import clear_reconcile_hold

    with SessionLocal() as db:
        n = clear_reconcile_hold(db, symbol)
    click.echo(f"Cleared reconcile_hold for {n} position(s) with symbol {symbol}")


@cli.command("backfill-fundamentals")
@click.option("--universe", multiple=True, type=click.Path(exists=True),
              help="Ticker list file(s) to restrict (default: all companies with a CIK)")
def backfill_fundamentals(universe):
    """Backfill point-in-time fundamentals from SEC EDGAR company-facts."""
    from form4lab.database import SessionLocal
    from form4lab.data.fundamentals_fetcher import backfill_fundamentals as _backfill

    tickers = None
    if universe:
        tickers = set()
        for path in universe:
            with open(path) as f:
                tickers |= {ln.strip().upper() for ln in f
                            if ln.strip() and not ln.strip().startswith("#")}
        click.echo(f"Restricting to {len(tickers)} tickers from {len(universe)} file(s)")

    with SessionLocal() as db:
        stats = _backfill(db, tickers=sorted(tickers) if tickers else None)
    click.echo(f"Fundamentals backfill complete: {stats}")


@cli.command("backfill-sectors")
@click.option("--universe", multiple=True, type=click.Path(exists=True),
              help="Ticker list file(s) to restrict (default: all companies missing a sector)")
def backfill_sectors(universe):
    """Backfill companies.sector/industry via yfinance .info (GICS-ish sector)."""
    import contextlib, io
    import yfinance as yf
    from sqlalchemy import select
    from form4lab.database import SessionLocal
    from form4lab.models.company import Company

    restrict = None
    if universe:
        restrict = set()
        for path in universe:
            with open(path) as f:
                restrict |= {ln.strip().upper() for ln in f
                             if ln.strip() and not ln.strip().startswith("#")}

    updated = no_data = 0
    with SessionLocal() as db:
        companies = db.execute(select(Company).where(Company.ticker.isnot(None))).scalars().all()
        if restrict is not None:
            companies = [c for c in companies if (c.ticker or "").upper() in restrict]
        # only those missing a sector
        companies = [c for c in companies if not c.sector]
        click.echo(f"Backfilling sectors for {len(companies)} companies...")
        for i, c in enumerate(companies, 1):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    info = yf.Ticker(c.ticker).info
                sector = info.get("sector")
                industry = info.get("industry")
            except Exception:
                sector = industry = None
            if sector:
                c.sector = sector
                c.industry = industry
                updated += 1
            else:
                no_data += 1
            if i % 50 == 0:
                db.commit()
                click.echo(f"  {i}/{len(companies)} — {updated} updated")
        db.commit()
    click.echo(f"Sector backfill complete: {updated} updated, {no_data} no_data")


@cli.command("backfill-earnings")
@click.option("--universe", multiple=True, type=click.Path(exists=True),
              help="Ticker list file(s) to restrict backfill (default: all companies)")
@click.option("--limit", default=80, type=int, help="Max earnings dates per ticker to request")
def backfill_earnings(universe, limit):
    """Backfill historical earnings report dates from yfinance into earnings_dates."""
    from form4lab.database import SessionLocal
    from form4lab.data.earnings_fetcher import backfill_earnings as _backfill

    tickers = None
    if universe:
        tickers = set()
        for path in universe:
            with open(path) as f:
                tickers |= {ln.strip().upper() for ln in f
                            if ln.strip() and not ln.strip().startswith("#")}
        click.echo(f"Restricting to {len(tickers)} tickers from {len(universe)} file(s)")

    with SessionLocal() as db:
        stats = _backfill(db, tickers=sorted(tickers) if tickers else None, limit=limit)
    click.echo(f"Earnings backfill complete: {stats}")


@cli.command("backfill-form4-details")
@click.option("--redownload", is_flag=True, default=False,
              help="Re-download quarterly ZIPs even if cached")
def backfill_form4_details_cmd(redownload):
    """Re-parse SEC bulk ZIPs into form4 detail tables (derivatives + footnotes).

    Parses derivative Table II rows (option expiration dates, strikes) and
    footnote text for every filing already present in transactions. Not
    consumed by the default strategy or generic flags — available for your
    own feature engineering. Idempotent per filing.
    """
    from form4lab.database import SessionLocal
    from form4lab.data.form4_details_fetcher import backfill_form4_details as _backfill

    with SessionLocal() as db:
        stats = _backfill(db, redownload=redownload)
    click.echo(f"Form4 details backfill complete: {stats}")


@cli.command("harvest-filing-events")
def harvest_filing_events_cmd():
    """Harvest 8-K + SC 13D/13G filing events per company.

    8-K Item 2.02 filing dates are a proxy for earnings-release timing; SC
    13D filings indicate activist-investor context. Not consumed by the
    default strategy or generic flags — available for your own feature
    engineering. Idempotent per accession; ~2-3k SEC requests.
    """
    from form4lab.database import SessionLocal
    from form4lab.data.filing_events_fetcher import harvest_filing_events as _harvest

    with SessionLocal() as db:
        stats = _harvest(db)
    click.echo(f"Filing-events harvest complete: {stats}")


@cli.command("backfill-short-volume")
@click.option("--start", default="2018-08-01",
              help="Start date (YYYY-MM-DD); FINRA's CDN has no files before ~2018-09")
def backfill_short_volume_cmd(start):
    """Backfill FINRA Reg SHO daily short-sale volume.

    Not consumed by the default strategy or generic flags — available for
    your own feature engineering. ~2,700 daily files at a polite non-SEC
    rate limiter; idempotent per day.
    """
    from form4lab.database import SessionLocal
    from form4lab.data.short_volume_fetcher import backfill_short_volume as _backfill

    with SessionLocal() as db:
        stats = _backfill(db, start=date.fromisoformat(start))
    click.echo(f"Short-volume backfill complete: {stats}")


if __name__ == "__main__":
    cli()
