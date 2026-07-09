# Research harness

`form4lab/research/` is a small, opinionated framework for turning strategy ideas into backtest configurations and screening them without fooling yourself. It doesn't discover alpha for you — it makes it harder to accidentally p-hack a train/validate split into something that looks good and isn't.

## What it does

- **Declarative specs.** A *spec* is a JSON-serializable dict describing one backtest configuration: which base strategy signal to restrict to (`base`), which extra row-predicate `atoms` to AND together on top of it, a `hold_days`, and a `window` name. `form4lab/research/space.py`'s `resolve_spec()` turns a spec into kwargs for `form4lab.scoring.portfolio_simulator.run_simulation` — reusing the real simulator engine, so a spec is directly comparable to running the CLI's `simulate-portfolio` by hand.
- **Atoms and banned regions.** `ATOMS` (in `space.py`) is a dict of named row-predicates a spec can AND together: `cluster_2plus`, `senior_role`, `value_100k`, `not_10b5_1`, `opportunistic`, `first_buy` — six entries drawn from public literature, operating on whatever columns the `compute_*_flags` pipeline (`form4lab/scoring/flags.py`) produced. `is_banned(spec)` is a guard checked before a spec burns a training-window trial; it ships returning `(False, "")` for everything — no banned regions yet, because a fresh research program hasn't falsified anything. As your own research rules out a region (a lever family that consistently underperforms in your own train-window screens), add it to `is_banned` so the loop skips it for free instead of re-spending a trial (and a DSR penalty point) on it.
- **Train → validate → locked test.** `form4lab.research.loop` (the evaluator; it is read-only against the database) screens every spec in a batch on the `train` window; anything clearing `--train-screen-sharpe` (default: 0.85 per-observation Sharpe) is auto-confirmed on `validate`. The `test` window is refused entirely unless you pass `--allow-test` — a config gets exactly one look at the test window, ever, and that's a human decision, not something the loop grants itself.
- **Deflated Sharpe Ratio.** `form4lab/research/stats.py` implements the Bailey & López de Prado Deflated Sharpe Ratio: the probability a candidate's true Sharpe exceeds the *inflated* benchmark you'd expect from the best of N independent trials under the null, penalized for the number of configs you've tried so far (`n_trials_cumulative`, tracked in the ledger) and for non-normal return shape (skew/kurtosis of the trade-return series). A candidate only becomes a nominee if its DSR clears `--dsr-pass` (default `0.95`).
- **The ledger.** `--ledger` (default `research/ledger.json`, created fresh on first run — not shipped) stores every trial's spec, per-window metrics, and DSR, plus running metadata (`n_trials_cumulative`, `sr_variance`, a stall counter). Re-running a batch upserts by spec `id` instead of duplicating. A nominee must: sit on the validate-window Pareto frontier (Sharpe, return, max drawdown, top-20-trade PnL concentration, and coefficient of variation across shuffle seeds, all "higher/better-is-better" after sign-flipping); clear the DSR bar; have enough validate trades to not be a small-sample fluke (`--min-validate-trades`, default `100`); be a genuinely new "core" — base signal + sizing + exits, not a cosmetic atom swap of something you've already tried; and beat the best incumbent validate Sharpe by a real margin (`--nominee-margin`, default `0.10`), not just barely.

## Running it

```bash
python -m form4lab.research.loop \
  --batch research/batches/b1.json \
  --train-end 2015-12-31 \
  --validate-start 2016-01-01 --validate-end 2018-12-31 \
  --test-start 2019-01-01 \
  --seeds 20 \
  --ledger research/ledger.json
```

(The dates above are illustrative, not a recommendation — pick your own train/validate/test boundaries for your own data history. `space.py` never hardcodes a date range; `--train-end`/`--validate-start`/`--validate-end`/`--test-start` are required CLI args precisely so the split lives with you, not the library.)

`--batch` is a JSON file shaped `{"batch": N, "specs": [...]}`, where each spec looks like:

```json
{"id": "b1_0", "base": "cluster_buy", "atoms": ["senior_role", "value_100k"], "hold_days": 60, "window": "train"}
```

`base` must name a tradeable signal in `BASE_SIGNALS` (ships with just `"cluster_buy"`, the only tradeable signal the default `ClusterBuyStrategy` registers — add your own strategy's signal names as you register them). `window` is resolved against the `--train-end`/`--validate-start`/... args (there's also a `"full"` window, meaning the entire dataset, and specs without a `window` key run over everything). `--universe-file` (repeatable) restricts to a ticker list, the same convention as `simulate-portfolio --universe`. `research/` — batches and the ledger — is your own working state; this repo ships none of it.

To use your own atoms, base signals, or banned regions without touching this repo's code, point `FORM4LAB_RESEARCH_SPACE` at a module of your own with the same shape as `form4lab.research.space` (module-level `ATOMS`, `BASE_SIGNALS`, `is_banned`, `resolve_spec`, `spec_complexity`):

```bash
export FORM4LAB_RESEARCH_SPACE=myproject.research.space
```

Other useful flags, all with the defaults shown above unless overridden: `--seeds` (shuffle-seed repeats per window evaluation, default `20`), `--stall-threshold` (consecutive no-nominee batches before the loop flags a stall in its own ledger metadata, default `5`).

## Honest framing

This harness makes overfitting *visible*, not impossible. As shipped, it has:

- Six atoms and one base signal — all drawn from public literature, none of it tuned to any dataset.
- An **empty** banned-list (`is_banned` always returns `False`). A fresh research program has no falsified regions yet — you populate `is_banned` as your own train/validate screens rule things out, which is also how you keep the loop from re-spending trials (and DSR penalty) on ideas you've already killed.
- No results, ledger, or batches of any kind — `research/ledger.json` is created on first run, not shipped, and contains nothing until you run something.

The DSR penalty and the train/validate/locked-test split reduce — they don't eliminate — the odds that whatever clears your bar is noise dressed up as a strategy. The one-shot test-window gate (`--allow-test`) only works if you actually treat it as one-shot: spend it on a spec you're not genuinely done iterating on, and every subsequent validate-window nominee is implicitly contaminated by having been selected with knowledge of how that test read.
