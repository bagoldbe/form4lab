# form4lab

**form4lab helps you research one question: when the executives and directors who run a public company trade its own stock, does that tell you anything worth acting on?**

U.S. law requires those "insiders" to report their trades to the SEC within a few business days, on a public filing called a [**Form 4**](https://www.sec.gov/about/forms/form4.pdf). This is the *legal, disclosed* kind of insider trading — not the illegal kind you hear about in the news. form4lab collects those filings, organizes them into a database, scores the insiders behind them, and gives you the tools to test trading ideas against history and — if you choose — run them on a simulated brokerage account. You bring the trading idea; form4lab handles the data, the backtesting, and the plumbing.

It comes with a few simple, textbook example strategies to learn from, but it deliberately **does not ship a proven money-making strategy** — that part is up to you.

Under the hood it's a self-hostable app: rate-limit-compliant SEC EDGAR ingestion, a backtester that avoids look-ahead bias and models realistic trading costs, a pluggable strategy interface, optional paper-trading execution, and a research workflow built to stop you from fooling yourself with results that only look good in hindsight.

> **Not investment advice.** form4lab is educational research software. The example strategies it ships are illustrative implementations of published academic heuristics — unvalidated, untuned, and expected to underperform after costs. Read [DISCLAIMER.md](DISCLAIMER.md) before pointing any of this at real money.

This is pre-release software; commands and configuration may still change before a tagged release.

## Quickstart (Docker)

Requires Docker and Docker Compose.

```bash
export SEC_IDENTITY="Your Name you@example.com"   # required — see "SEC EDGAR access" below
docker compose up -d
```

This starts three containers: a Postgres `db`, the `web` dashboard/API (which also applies database migrations on startup), and a standalone `scheduler` that runs ingestion/scoring/exit jobs in the background. Wait for `docker compose ps` to show `web` as healthy, then pull some data in and generate a signal:

```bash
docker compose run --rm web backfill --ticker AAPL
docker compose run --rm web compute-outcomes
docker compose run --rm web refresh-scores
docker compose run --rm web generate-signals
```

Open http://localhost:8000 for the dashboard, or `GET /api/v1/alerts` for JSON.

The word after `run --rm web` is passed straight to `docker/entrypoint.sh`, which forwards anything other than the reserved words `web`/`scheduler` to the `form4lab` CLI — don't repeat `form4lab` in the override (`docker compose run --rm web backfill --ticker AAPL`, not `... web form4lab backfill ...`).

To run your own strategy, drop a module into `./strategies/` on the host (bind-mounted into both the `web` and `scheduler` containers as `/app/strategies`) and point `STRATEGY_PATH` at it before `docker compose up`, e.g. `export STRATEGY_PATH=strategies.my_strategy:MyStrategy`. See [docs/strategy-authoring.md](docs/strategy-authoring.md).

## Quickstart (local)

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[trade,dev]"

export SEC_IDENTITY="Your Name you@example.com"   # required
form4lab init-db                                   # creates form4lab.db (sqlite) via Alembic
form4lab backfill --ticker AAPL                     # pull one ticker's Form 4 history
form4lab compute-outcomes                           # forward returns for each transaction
form4lab refresh-scores                             # insider credibility tiers / skill scores
form4lab generate-signals                           # alerts for the active strategy

form4lab run                                        # dashboard + API on :8000, scheduler in-process
```

`form4lab run` also starts the background job scheduler in-process by default (`SCHEDULER_ENABLED=true`). Set it `false` and run `form4lab scheduler` as a separate process if you want ingestion/scoring split from the web process — that's what `docker-compose.yml` does.

## Configuration

form4lab reads configuration from environment variables (or a `.env` file — see `.env.example`). The variables most people need:

| Variable | Default | Meaning |
|---|---|---|
| `SEC_IDENTITY` | *(none — required)* | Your name/org + contact email. Sent verbatim as the `User-Agent` on every SEC EDGAR request. `form4lab run` refuses to start without it. |
| `DATABASE_URL` | `sqlite:///form4lab.db` | SQLAlchemy connection string. `docker compose up` points this at the bundled Postgres for you. |
| `STRATEGY_PATH` | `form4lab.strategies.cluster_buy:ClusterBuyStrategy` | `module:ClassName` of the active `Strategy`. See [docs/strategy-authoring.md](docs/strategy-authoring.md). |
| `SCHEDULER_ENABLED` | `true` | Run the background job scheduler (ingestion, scoring, exit checks, order sync) in-process. Set `false` when a separate `form4lab scheduler` process/container owns the jobs. |
| `ALPACA_ENABLED` | `false` | Turn on paper/live trade execution via Alpaca. Off by default — form4lab never places an order unless you opt in. |
| `ALPACA_PAPER` | `true` | Paper-trade against Alpaca's paper endpoint. Only relevant once `ALPACA_ENABLED=true`; setting this `false` means real orders. |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | *(empty)* | Alpaca credentials, only read when `ALPACA_ENABLED=true`. Requires `pip install form4lab[trade]`. |

`form4lab/config.py` also exposes nested `Sec` / `Scoring` / `Signal` / `Scheduler` / `Alpaca` settings groups for deeper tuning, each with its own env-var prefix (`SEC_`, `SCORING_`, `SIGNAL_`, `SCHEDULER_`, `ALPACA_` — e.g. `SCORING_ELITE_SKILL_MIN`) and safe, documented defaults. You shouldn't need to touch them to get started.

## Trade execution (optional)

form4lab is first and foremost a research tool — you get the data, signals, dashboard, and backtests **without ever connecting a brokerage.** Live/paper execution is off by default (`ALPACA_ENABLED=false`), and form4lab never places an order unless you explicitly opt in.

When you *do* want to act on signals automatically, the one built-in integration is **[Alpaca](https://alpaca.markets/)**, a US brokerage with a developer API. It's the default for a practical reason: Alpaca offers free **paper trading** — a simulated account funded with fake money — so you can run the entire signal-to-order pipeline and watch how it behaves at zero financial risk. Paper mode stays on by default whenever execution is enabled; placing real orders takes a deliberate `ALPACA_PAPER=false`.

Alpaca is *not* a pluggable interface the way strategies are: form4lab ships a single Alpaca adapter (`form4lab/services/alpaca_service.py`), not a general broker abstraction. Supporting a different broker (Interactive Brokers and the like) means writing your own adapter against that broker's API. If you'd rather not wire up a brokerage at all, just leave execution off and lean on the backtester and research harness — that's the main workflow regardless. (form4lab is an independent project, unaffiliated with Alpaca; see [DISCLAIMER.md](DISCLAIMER.md).)

## Running it continuously (hosting)

The quickstarts above run form4lab on your own machine, which is ideal for trying it out and for research. But the background scheduler only pulls new filings and refreshes signals *while it's actually running* — so to keep form4lab up to date day-to-day, it needs to live somewhere that stays on.

Because it's a standard Docker Compose app with no dependency on any particular provider, you can run it anywhere that runs containers:

- **An always-on machine you control** — a home server or a small Linux VPS. Clone the repo, set your environment variables, `docker compose up -d`, and leave it.
- **A managed container platform** — for example [Railway](https://railway.app/), [Render](https://render.com/), or [Fly.io](https://fly.io/). These host the container (and a Postgres database) for you; you set the same environment variables (`SEC_IDENTITY`, `STRATEGY_PATH`, and any `ALPACA_*` keys) in the platform's dashboard instead of your shell.

Whichever you choose, make sure the scheduler runs on **exactly one** process — either a single service with `SCHEDULER_ENABLED=true`, or a dedicated `form4lab scheduler` container — so ingestion and scoring jobs fire once instead of being duplicated.

## Architecture

```
SEC EDGAR / Yahoo Finance
        │   ingestion — form4lab/data/*
        ▼
  normalized database    SQLAlchemy models + Alembic migrations
        │                 form4lab/models/, alembic/
        ▼
  scoring & outcomes      credibility tiers, skill scores, forward returns
        │                 form4lab/scoring/
        ▼
  strategy (pluggable)    classify / size / allow_entry
        │                 form4lab/strategy/, form4lab/strategies/
        ▼
  signals & alerts        persisted, surfaced on the dashboard + JSON API
        │                 form4lab/services/alert_service.py, form4lab/routes/
        │
        ├──▶ backtest              form4lab/scoring/portfolio_simulator.py
        ├──▶ paper / live trade    form4lab/services/alpaca_service.py (optional — Alpaca, paper by default)
        └──▶ research harness      form4lab/research/ (train / validate / locked-test)
```

## Strategies

form4lab ships three example strategies — naive, well-documented illustrations of published insider-trading heuristics, not tuned or validated for live use:

- **`cluster_buy`** (default) — 2+ distinct insiders buying the same company within a short window.
- **`big_exec_buy`** — a CEO or CFO open-market purchase over a size threshold, excluding pre-arranged 10b5-1 plan trades.
- **`opportunistic_first_buy`** — an insider's first observed open-market purchase over a size floor, excluding 10b5-1 plan trades.

Only one strategy is active per process, chosen by `STRATEGY_PATH`. Writing your own is the intended way to use form4lab — see [docs/strategy-authoring.md](docs/strategy-authoring.md) for the plugin interface, the feature-parity rule between live and backtest evaluation, and a worked example.

## Research harness

`form4lab/research/` is an anti-overfitting loop for turning strategy ideas into backtest specs: screen on a training window, auto-confirm on a validation window, score survivors with a Deflated Sharpe Ratio, and take exactly one look at a locked test window when you're genuinely done iterating. It ships empty — a handful of public-literature building blocks and no banned regions — you build your own search space on top of it. See [docs/research-harness.md](docs/research-harness.md).

## SEC EDGAR access

SEC EDGAR requires every automated requester to identify itself with a descriptive `User-Agent` (name/org + contact email) and enforces a rate limit. form4lab sends whatever you set `SEC_IDENTITY` to, verbatim, on every request — set it to your own identity, not a placeholder or someone else's. form4lab already rate-limits its own requests conservatively by default (see `SecConfig` in `form4lab/config.py`); you shouldn't need to change that to get started. See [DISCLAIMER.md](DISCLAIMER.md) for the full fair-access note.

## License

Apache-2.0 — see [LICENSE](LICENSE). Not investment advice — see [DISCLAIMER.md](DISCLAIMER.md).
