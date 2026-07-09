# Writing a strategy

A `Strategy` owns decisions — what counts as a signal, position sizing, entry gates, hold periods. The platform owns facts and mechanics — ingestion, features, persistence, order routing, simulation. Exactly one strategy is active per process, resolved from `STRATEGY_PATH` by `registry.get_active()` (`form4lab/strategy/registry.py`) — a module-level singleton populated **lazily, on first use**, not at container boot. Neither `main.py`'s FastAPI `lifespan` nor `cli.py`'s `run`/`scheduler` commands call it; only route, service, and job code does. A broken `STRATEGY_PATH` therefore won't crash startup — it surfaces as an `ImportError`/`AttributeError` on the first request or job that resolves the strategy, possibly after the container already reports healthy. Validate `STRATEGY_PATH` before deploying.

The ABC lives in `form4lab/strategy/base.py`; its docstrings are the canonical reference for every contract described here. This guide is the narrative walk-through — read both.

## The ABC surface

```python
class Strategy(ABC):
    name: str = "strategy"

    def signal_types(self) -> list[SignalType]: ...      # required
    def classify(self, txn: TxnView, f: FeatureView) -> str | None: ...  # required

    def evaluate_buy(self, txn, f) -> BuyEvaluation | None: ...  # optional override
    def evaluate_sell(self, txn, f) -> SellEvaluation | None: ...  # optional override
    def size(self, ctx: SizingContext) -> SizeDecision: ...       # optional override
    def allow_entry(self, ctx: EntryContext) -> str | None: ...   # optional override
    def classify_row(self, row, tier, skill_score) -> tuple[str, str, float]: ...  # rarely overridden
```

- **`signal_types()`** — declare every `SignalType` your strategy can emit. Each has a `name`, `direction` (`"buy"` default or `"sell"`), `tradeable` (does it open a position), `hold_days`, `priority`, `visible`, and a human `description`. `SignalRegistry` (`form4lab/strategy/registry.py`) is built from this list and is the single source of truth for which alert types exist and which are tradeable — the dashboard, alert service, and backtester defaults all derive from it.
- **`classify(txn, f)`** — the core decision: given one transaction and a feature view, return the `name` of a declared `SignalType`, or `None` for no signal. All three shipped strategies (`form4lab/strategies/{cluster_buy,big_exec_buy,opportunistic_first_buy}.py`) return a real signal name or `"filtered_out"` — a second, `visible=False` `SignalType` used as an explicit "we looked at this and rejected it" record rather than silence.
- **`evaluate_buy(txn, f)`** — the ABC default calls `classify()` and wraps a match in `BuyEvaluation(alert_type=name, conviction=1.0, cluster_id=None, summary=...)`. Override it if you want a real, strategy-relative conviction score instead of the flat default. `conviction` is unbounded and has no fixed scale of its own — the dashboard buckets it into a 1–5 display rating by percentile rank within the alert set being rendered, so only the relative ordering of your own strategy's values matters.
- **`evaluate_sell(txn, f)`** — the ABC default returns `None` (no sell alerts). See "Sell alerts and position exits" below before implementing this.
- **`size(ctx)`** — the ABC default returns a flat 5%-of-equity `SizeDecision(dollars=ctx.equity * 0.05, method="role", vol=None, pct=0.05)`. See "Sizing" below.
- **`allow_entry(ctx)`** — the ABC default allows everything (`None`). See "The universe prefix and the -1 sentinel" below.
- **`classify_row(row, tier, skill_score)`** — the backtest adapter. The default builds a `TxnView` + `RowFeatureView` from a precomputed flags row and calls your `classify()`; you don't need to override it, and none of the shipped strategies do.

### The supporting dataclasses

- **`TxnView`** — a minimal, read-only view of one transaction: `insider_id`, `company_id`, `ticker`, `transaction_date`, `txn_value`, `role_title`. This is the same shape whether you're being called live or from a backtest row.
- **`FeatureView`** (a `Protocol`) — `.get(name, default)` / `.put(name, value)`. Two implementations exist (`form4lab/strategy/features.py`): `LiveFeatureView` (per-transaction, DB-backed, used when evaluating real filings) and `RowFeatureView` (backtest adapter over a precomputed pandas row). **Both implementations must resolve the same feature name to the same value for the same underlying transaction, or your strategy behaves differently live than it did in your backtest.** This is the single most important rule in this document — see the next section.
- **`SizingContext`** — `equity`, `ticker`, `role_title`, and (when available) `vol` / `ticker_exposure_dollars`, supplied by the platform for strategies that implement vol-targeted sizing.
- **`EntryContext`** — `ticker`, `role_title`, `insider_id`, `open_positions_in_ticker`, `open_positions_for_insider_ticker`.
- **`BuyEvaluation`** / **`SellEvaluation`** — the result types for `evaluate_buy`/`evaluate_sell`.

## The feature-parity rule

`classify()` is called from two different code paths: live, against a `LiveFeatureView` built from one real transaction and a live DB session; and backtest, against a `RowFeatureView` built from one row of a pandas DataFrame that `form4lab/scoring/flags.py`'s vectorized `compute_*_flags` builders produced ahead of time. **Every feature your `classify()` reads must resolve to the same value in both**, or a strategy that looks profitable in `simulate-portfolio` can silently do something else once it's live — the exact failure class this project's own feature-parity test suite (`tests/test_feature_parity.py`) exists to pin down.

### How the two views actually resolve a name

- **`LiveFeatureView._compute`** recognizes a fixed, explicitly-enumerated set of names — `is_10b5_1_plan`, `is_first_buy`, `is_drawdown`/`drawdown`/`still_falling`, `cluster_unique_insiders`, `cluster_member_skill_scores`, `insider_median_value`, `sell_pct`, `cluster_sell_unique`, `insider_name`, `company_name`, and a few more — plus `tier`, `skill_score`, `role_title` seeded at construction. **Anything else silently returns your `default`.**
- **`RowFeatureView.get`** special-cases a smaller set (`tier`, `skill_score`, `is_drawdown`, `cluster_unique_insiders`, `is_first_buy`, `is_10b5_1_plan`), reserves three names to always return `default` (see below), and then **falls back to reading the backtest row by that literal column name** for everything else.

That asymmetry cuts both ways:

- A name recognized live via a bespoke DB query (`insider_median_value`, `sell_pct`, `cluster_sell_unique`, `cluster_member_skill_scores`, the raw `drawdown`/`cluster` dicts) has no backtest counterpart — `f.get(...)` on those returns `default` in a backtest, silently.
- A name that happens to be a real column in the backtest DataFrame (e.g. `total_value`, `is_routine`, `filing_lag_days`, `avg_volume_20d` — anything a `compute_*_flags` builder in `form4lab/scoring/flags.py` produces) resolves to a real value in backtest but returns `default` live, because `LiveFeatureView._compute` never learned that name.

A key "working" when you run `simulate-portfolio` is not proof it works live, and vice versa.

### Feature keys confirmed identical on both sides

Pinned by a parity test in `tests/test_feature_parity.py` — extend that file the same way when you add a shared feature of your own (build a DB fixture, run `LiveFeatureView`, build the matching row via the paired `compute_*_flags` builder, assert equal):

| Key | Type | Meaning |
|---|---|---|
| `is_10b5_1_plan` | `bool` | Transaction filed under a pre-arranged Rule 10b5-1 plan. `NULL`/missing → `False` on both sides. |
| `is_first_buy` | `bool` | This insider's first observed discretionary buy — scoped to the insider (any company), no tenure gate. See the worked example below. |
| `is_drawdown` | `bool` | Stock is ≤ −15% over the prior 60 trading days. |
| `cluster_unique_insiders` | `int` | Count of distinct insiders (including this one) buying the same company within the cluster window. |
| `tier` | `str` | The insider's credibility tier — passed in at evaluation time, not computed by either view. |
| `skill_score` | `float` | The insider's skill score — likewise passed in. |

Also resolve to matching values on both sides, sourced from the same underlying data, though not covered by a dedicated parity test:

| Key | Meaning |
|---|---|
| `still_falling` | Drawdown is deepening (negative short-term momentum on top of `is_drawdown`). |
| `insider_name` / `company_name` | Display names. |

### Reserved keys — need `put()` first

`conviction`, `cluster_id`, and `is_first_time` have no computation behind them in `RowFeatureView` — `.get()` returns your `default` for them **unless this evaluation already called `.put()` for that name**. If you derive one of these mid-`evaluate_buy` (a real conviction score, say) and read it again in `classify()`, you must `put()` it — live and backtest alike. (`is_first_time` is additionally a *different predicate* from `is_first_buy` — company-scoped with a 2-year tenure gate, kept for strategies that need it; new strategies should read `is_first_buy` instead.)

### Worked example: "backtest realism ≠ live" (`is_first_buy`)

`is_first_buy` means "this insider's first observed discretionary buy," scoped to the insider (not the company), with no tenure gate. Both views compute that identical predicate — but over different windows, and it's easy to miss why. The shipped `opportunistic_first_buy` strategy's docstring (`form4lab/strategies/opportunistic_first_buy.py`) spells it out because the strategy depends on it directly:

- **In backtest**, `is_first_buy` comes from `compute_firsttime_flags` run over the `buys` frame that `load_backtest_data` builds — and that frame INNER JOINs `trade_outcomes`, so a transaction without a matured outcome yet (too recent to have a computed forward return) is invisible to it. A genuinely-first buy that hasn't matured an outcome can leave a *later*, actual repeat buy mislabeled `is_first_buy=True` at the recent edge of a backtest run.
- **Live**, `LiveFeatureView._is_first_observed_buy` has no such gap — it reasons over the insider's complete transaction history directly, with no outcome dependency (a live signal can't wait for a future return to materialize before firing).

Neither side is "wrong" — a backtest is bounded by the window and maturity of the data you loaded it over; live always sees full history. The general lesson: before trusting a backtested number for any feature that depends on "have I seen this before," check whether the backtest frame's join conditions could be hiding recent history that live would see. `LiveFeatureView._is_first_observed_buy`'s own docstring documents this plus one more edge case (a same-day, cross-company tiebreak) in full.

## `allow_entry`: the universe prefix and the `-1` sentinel

`allow_entry(ctx: EntryContext) -> str | None` returns `None` to allow entry, or a reason string to reject it. It's called at two different points:

1. **Pre-entry universe check** (once per candidate signal, before concentration is known) — `ctx.open_positions_in_ticker` and `ctx.open_positions_for_insider_ticker` are both `-1` here, a sentinel meaning "not yet computed." Only a reason prefixed `"universe:"` is honored at this stage, so a naive count-based gate can't accidentally reject every candidate against the sentinel.
2. **Real entry gate**, with real position counts, where any non-`None` reason rejects the trade.

Write count-based gates so they only fire on real counts:

```python
def allow_entry(self, ctx: EntryContext) -> str | None:
    if ctx.open_positions_for_insider_ticker >= 0 and ctx.open_positions_for_insider_ticker >= MAX_PER_INSIDER_TICKER:
        return "insider_ticker_limit"
    if ctx.open_positions_in_ticker >= 0 and ctx.open_positions_in_ticker >= MAX_PER_TICKER:
        return "ticker_limit"
    return None
```

All three shipped strategies use exactly this pattern, gated against `settings.alpaca.max_positions_per_insider_ticker` / `.max_positions_per_ticker`. If you need a genuine universe/eligibility gate (e.g. "only trade tickers in my index-membership file"), prefix its reason with `"universe:"` so it's honored at the pre-entry check too.

## Sizing

The ABC default `size()` returns flat 5%-of-equity, `method="role"`. All three shipped strategies use this default — none implement vol-targeted sizing. `SizeDecision.method` is a free-form label persisted verbatim to the `broker_orders` audit columns, but five values are reserved:

- `"role"` — role-tiered percent of equity (the ABC default, and what every shipped strategy returns).
- `"voltarget"` / `"voltarget_capped"` — vol-targeted sizing, uncapped vs. ticker-exposure-capped.
- `"fallback"` — vol-targeting was requested but `SizingContext.vol` was unavailable, so a role-tiered percent was used instead.
- `"shadow"` — a vol-targeted size was computed and logged for comparison, but role-tiered dollars are what actually traded.

The platform supplies `SizingContext.vol` / `.ticker_exposure_dollars` when available; a real vol-targeted `size()` override is on you.

## Sell alerts and position exits

Two things worth being explicit about before you implement `evaluate_sell`:

1. **Declare sell types with `direction="sell"`.** `SignalRegistry.sell_names()` — used by the sell-alert dedup query and the dashboard's sell grouping — only returns types with `direction="sell"`. Buy types default to `direction="buy"` and need no override. A sell `alert_type` your strategy emits that isn't declared this way in `signal_types()` is invisible to both.
2. **Sell alerts are informational only.** Nothing in the shipped code path closes a broker position because a sell alert fired. Positions close on a purely time-based schedule — `hold_days`/`exit_target_date`, set at entry from the firing signal type's `hold_days` — checked by `get_positions_to_close` (`form4lab/services/alpaca_service.py`) live and its backtest equivalent in `portfolio_simulator.py`. There's also an optional price-based stop-loss (`ALPACA_STOP_LOSS_PCT` / `AlpacaConfig.stop_loss_pct`, `None`/off by default), checked independently by `get_stop_loss_positions`. If you want a sell alert to actually close a position early, you have to wire that yourself — it does not happen automatically.

## `classify_row` and the `"skip"` sentinel

The default `classify_row` wraps `classify()`; you don't need to override it. Its return is `(signal_type, tier, skill_score)`, where `signal_type` is the literal string `"skip"` — not `None` — when `classify()` returns `None`, so callers always get a plain `str`. The backtester doesn't special-case `"skip"` by name: it's simply never a member of `target_signals`, so the default "signal type not in target set" gate excludes it like any other non-tradeable type.

## Registering your strategy

Two ways to point `STRATEGY_PATH` at your code:

1. **The `STRATEGY_PATH` env var** (simplest, no packaging): `STRATEGY_PATH=mymodule:MyStrategy`, where `mymodule` is importable — on `PYTHONPATH`, installed, or otherwise reachable. `registry.load_strategy` does `importlib.import_module(module_name)` then `getattr(module, class_name)`; nothing more.
2. **A `pyproject.toml` entry point**, if you're packaging your strategy properly: add a line under `[project.entry-points."form4lab.strategies"]` (see how the three shipped strategies register themselves in this repo's own `pyproject.toml`) for discoverability. Note that `load_strategy` does **not** read entry points itself — you still set `STRATEGY_PATH=module:ClassName` to activate one; the entry point group is metadata, not an alternate resolution path.

**In Docker:** drop your module into `./strategies/` on the host. `docker-compose.yml` bind-mounts it to `/app/strategies` in both the `web` and `scheduler` containers (the image installs form4lab in editable mode, so `/app` is already on `sys.path` — no `PYTHONPATH` wrangling needed). Set `STRATEGY_PATH=strategies.my_strategy:MyStrategy` before `docker compose up`. Both containers need the mount: `web` imports it for dashboard reads, `scheduler` is what actually executes it against live signals.

## A worked minimal example

`form4lab/strategies/big_exec_buy.py`, in full, as a template — a CEO/CFO open-market buy over a size floor, excluding 10b5-1-planned trades:

```python
"""Big-executive-buy example strategy — a naive, well-documented illustration.

Open-market purchases by the CEO or CFO are among the most-studied insider
signals. Seyhun (1986, "Insiders' Profits, Costs of Trading, and Market
Efficiency", Journal of Financial Economics) finds that senior insiders'
trades are more informative than junior insiders' trades, consistent with
greater access to material information. The thresholds below are pedagogical
defaults, NOT researched or validated for live use.
"""
import logging

from form4lab.config import settings
from form4lab.strategy.base import EntryContext, FeatureView, SignalType, Strategy, TxnView
from form4lab.utils import is_ceo, is_cfo

logger = logging.getLogger(__name__)
_alp = settings.alpaca

MIN_TXN_VALUE = 100_000.0   # illustrative


class BigExecBuyStrategy(Strategy):
    name = "big_exec_buy"

    def signal_types(self) -> list[SignalType]:
        return [
            SignalType("big_exec_buy", tradeable=True, hold_days=60, priority=50,
                       description="CEO/CFO open-market purchase over the size threshold, ex-10b5-1"),
            SignalType("filtered_out", priority=0, visible=False),
        ]

    def classify(self, txn: TxnView, f: FeatureView) -> str | None:
        is_senior = is_ceo(txn.role_title) or is_cfo(txn.role_title)
        if (is_senior and (txn.txn_value or 0.0) >= MIN_TXN_VALUE
                and not f.get("is_10b5_1_plan")):
            return "big_exec_buy"
        return "filtered_out"

    def allow_entry(self, ctx: EntryContext) -> str | None:
        # counts are -1 at the pre-entry universe check (unknown) — apply
        # count-based gates only when real counts are supplied
        if ctx.open_positions_for_insider_ticker >= 0 and ctx.open_positions_for_insider_ticker >= _alp.max_positions_per_insider_ticker:
            return "insider_ticker_limit"
        if ctx.open_positions_in_ticker >= 0 and ctx.open_positions_in_ticker >= _alp.max_positions_per_ticker:
            return "ticker_limit"
        return None
```

Notes on the pattern, in order:

- **Module docstring** states the literature basis and, explicitly, that the thresholds are pedagogical and unvalidated. Every example strategy in this repo does this — do it in yours too.
- **`MIN_TXN_VALUE`** is a module-level constant, not buried in `classify()` — makes the threshold easy to find and tune.
- **`classify`** reads `txn.role_title` directly from `TxnView` (no feature lookup needed for a raw transaction field), and `f.get("is_10b5_1_plan")` for the one feature it needs — a both-view-safe key from the table above. It returns the real signal name or falls through to `"filtered_out"`, never `None` with no record.
- **`allow_entry`** is the standard concentration-limit pattern from "The universe prefix and the -1 sentinel" above, verbatim.
- **No `size()` override** — this strategy is happy with the ABC's flat role-tiered default.

`cluster_buy.py` and `opportunistic_first_buy.py` follow the identical shape; read all three side by side before writing your own.
