# form4lab

Self-hostable platform for researching SEC Form 4 insider-trading signals: compliant EDGAR ingestion, a look-ahead-free backtester with realistic frictions, pluggable strategy modules, paper-trading execution, and an anti-overfitting research harness. Ships with naive textbook example strategies. It does not ship alpha, and nothing in it is investment advice — see DISCLAIMER.md.

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
        ├──▶ paper / live trade    form4lab/services/alpaca_service.py (Alpaca, paper by default)
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
