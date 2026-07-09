"""Anti-overfit research loop driver (the compute arm). READ-ONLY on the DB.

Given a batch of specs (JSON), it: loads the dataset once, screens each on
TRAIN (N seeds), auto-confirms promising ones on VALIDATE, scores them with
the Deflated Sharpe Ratio (penalized by cumulative trials, tracked in the
ledger), updates the ledger + Pareto frontier, and emits a batch report +
nominee list. The TEST window is refused unless --allow-test (the human
one-shot gate) — a config gets exactly one look at the test window, ever.

The *researcher* reads the report + ledger and writes the next batch JSON —
that generate/learn step is where the loop "thinks." This script is only the
evaluator. Point FORM4LAB_RESEARCH_SPACE at your own module (same shape as
form4lab.research.space: ATOMS, BASE_SIGNALS, is_banned, resolve_spec,
spec_complexity) once you have atoms/signals/banned-regions of your own.

Usage:
  python -m form4lab.research.loop --batch research/batches/b1.json \\
      --train-end 2015-12-31 --validate-start 2016-01-01 \\
      --validate-end 2018-12-31 --test-start 2019-01-01
"""
import argparse
import importlib
import json
import logging
import os
import statistics as st
import sys
from datetime import date, datetime

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("form4lab.research.loop")
log.setLevel(logging.INFO)

from form4lab.database import get_db
from form4lab.scoring.portfolio_simulator import prepare_backtest_inputs, run_simulation, compute_metrics
from form4lab.research.stats import deflated_sharpe, per_trade_sharpe_stats

# The search space: swap in your own module (same shape) without touching
# this file, e.g. FORM4LAB_RESEARCH_SPACE=myproject.research.space
ss = importlib.import_module(os.environ.get("FORM4LAB_RESEARCH_SPACE", "form4lab.research.space"))

# Thresholds — CLI-overridable (see main()); these are just the defaults.
DSR_PASS = float(os.environ.get("FORM4LAB_DSR_PASS", "0.95"))
TRAIN_SCREEN_SHARPE = float(os.environ.get("FORM4LAB_TRAIN_SCREEN_SHARPE", "0.85"))
MIN_VALIDATE_TRADES = int(os.environ.get("FORM4LAB_MIN_VALIDATE_TRADES", "100"))
NOMINEE_MARGIN = float(os.environ.get("FORM4LAB_NOMINEE_MARGIN", "0.10"))
STALL_THRESHOLD = int(os.environ.get("FORM4LAB_STALL_THRESHOLD", "5"))


def core_key(spec: dict) -> tuple:
    """The 'core' of a config: base signal + sizing + exits. Ignores cosmetic
    atoms so the loop doesn't keep nominating the same core in disguise.

    form4lab.research.space doesn't interpret "sizing"/"exits" spec keys yet
    (LEVERS ships empty) — this tuple already accounts for them so extending
    the space's schema later doesn't require revisiting nominee dedup.
    """
    sz = spec.get("sizing", {})
    return (spec.get("base", "cluster_buy"), sz.get("type"),
            round(sz.get("k", 0.0) or 0.0, 4), tuple(sorted(spec.get("exits", []))))


def load_universe(files: list[str] | None) -> set[str] | None:
    """Union of tickers across `files` (one per line, '#' comments allowed,
    case-insensitive). None/empty means no universe restriction — matches
    run_simulation's own universe=None semantics."""
    if not files:
        return None
    tickers: set[str] = set()
    for path in files:
        with open(path) as f:
            tickers |= {ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")}
    return tickers


def _objs(m):
    # objectives, all "higher is better": sharpe, ret, maxdd(less negative), -top20, -cv
    return (m["sharpe"], m["ret"], m["maxdd"], -m.get("top20", 1.0), -m.get("cv", 1.0))


def _dominates(a, b):
    oa, ob = _objs(a), _objs(b)
    return all(x >= y for x, y in zip(oa, ob)) and any(x > y for x, y in zip(oa, ob))


def frontier_for_window(records, window):
    pts = [(r["id"], r["metrics"][window]) for r in records
           if window in r.get("metrics", {}) and r["metrics"][window].get("trades", 0) > 0]
    nd = []
    for i, (idi, mi) in enumerate(pts):
        if not any(_dominates(mj, mi) for j, (idj, mj) in enumerate(pts) if j != i):
            nd.append(idi)
    return nd


def eval_spec(inputs, spec, window, universe, seeds, windows):
    kwargs = ss.resolve_spec({**spec, "window": window}, windows=windows)
    sharpes, rets, dds, top20s, trades = [], [], [], [], []
    rep_pnls = None
    for seed in range(seeds):
        pf, pi = run_simulation(None, preloaded=inputs, shuffle_seed=seed,
                                universe=universe, **kwargs)
        m = compute_metrics(pf, pi)
        if "error" in m:
            continue
        sharpes.append(m["sharpe"]); rets.append(m["total_return"]); dds.append(m["max_drawdown"])
        top20s.append(m.get("top20_pnl_share") or np.nan); trades.append(m["total_trades"])
        if seed == 0:
            rep_pnls = [p.pnl_pct for p in pf.closed_positions if p.pnl_pct is not None]
    if not sharpes:
        return None
    mret = st.mean(rets)
    return {
        "sharpe": round(st.mean(sharpes), 4),
        "ret": round(mret, 4),
        "maxdd": round(st.mean(dds), 4),
        "top20": round(float(np.nanmean(top20s)), 4),
        "cv": round((st.pstdev(rets) / mret) if mret else 0.0, 4),
        "trades": round(st.mean(trades), 1),
        "_pnls": rep_pnls or [],
    }


def _load_ledger(path: str, stall_threshold: int) -> dict:
    if os.path.exists(path):
        return json.load(open(path))
    return {"meta": {"n_trials_cumulative": 0, "sr_variance": 0.0,
                     "stall_counter": 0, "stall_threshold": stall_threshold},
            "records": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help='JSON file: {"batch": N, "specs": [...]}')
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--allow-test", action="store_true", help="permit window=test (one-shot human gate)")
    ap.add_argument("--ledger", default=os.path.join("research", "ledger.json"),
                    help="Path to the JSON ledger of prior trials (created fresh if absent).")
    ap.add_argument("--universe-file", action="append", default=None,
                    help="Ticker universe file (one ticker per line, '#' comments allowed). "
                         "Repeatable. Default: no universe restriction.")
    ap.add_argument("--train-end", required=True, type=date.fromisoformat,
                    help="Train window: dataset start .. this date (inclusive), ISO YYYY-MM-DD.")
    ap.add_argument("--validate-start", required=True, type=date.fromisoformat)
    ap.add_argument("--validate-end", required=True, type=date.fromisoformat)
    ap.add_argument("--test-start", required=True, type=date.fromisoformat,
                    help="Test window: this date .. dataset end. Held out — see --allow-test.")
    ap.add_argument("--dsr-pass", type=float, default=DSR_PASS)
    ap.add_argument("--train-screen-sharpe", type=float, default=TRAIN_SCREEN_SHARPE)
    ap.add_argument("--min-validate-trades", type=int, default=MIN_VALIDATE_TRADES)
    ap.add_argument("--nominee-margin", type=float, default=NOMINEE_MARGIN)
    ap.add_argument("--stall-threshold", type=int, default=STALL_THRESHOLD,
                    help="Consecutive no-nominee batches before the loop flags a stall.")
    args = ap.parse_args()

    windows = {
        "train": (None, args.train_end),
        "validate": (args.validate_start, args.validate_end),
        "test": (args.test_start, None),
        "full": (None, None),
    }

    batch = json.load(open(args.batch))
    batch_no = batch.get("batch", "?")
    specs = batch["specs"]

    ledger_dir = os.path.dirname(args.ledger)
    if ledger_dir:
        os.makedirs(ledger_dir, exist_ok=True)
    ledger = _load_ledger(args.ledger, args.stall_threshold)
    meta = ledger["meta"]
    meta.setdefault("stall_threshold", args.stall_threshold)

    # Snapshot incumbents BEFORE this batch (for the near-duplicate + margin guards)
    incumbents = list(ledger["records"])
    incumbent_cores = {core_key(r["spec"]) for r in incumbents}
    incumbent_best_val = max(
        [r["metrics"]["validate"]["sharpe"] for r in incumbents
         if "validate" in r.get("metrics", {})] + [0.0])

    # holdout guard
    for s in specs:
        if s.get("window") == "test" and not args.allow_test:
            log.error("Spec requests window=test but --allow-test not set. Refusing (holdout guard).")
            sys.exit(2)

    universe = load_universe(args.universe_file)
    db = next(get_db())
    log.info("Loading inputs once...")
    t0 = datetime.now()
    inputs = prepare_backtest_inputs(db)
    log.info("Inputs ready in %.0fs (%d buys)", (datetime.now() - t0).total_seconds(), len(inputs["buys"]))

    n_trials = meta["n_trials_cumulative"]
    sr_var = meta["sr_variance"]
    new_records, nominees, skipped = [], [], []

    for i, spec in enumerate(specs):
        sid = spec.get("id") or f"b{batch_no}_{i}"
        banned, why = ss.is_banned(spec)
        if banned:
            skipped.append((sid, why)); log.info("SKIP %s (banned: %s)", sid, why); continue

        train_m = eval_spec(inputs, spec, "train", universe, args.seeds, windows)
        if train_m is None:
            skipped.append((sid, "no closed positions on train")); continue
        rec = {"id": sid, "batch": batch_no, "spec": spec,
               "metrics": {"train": {k: v for k, v in train_m.items() if k != "_pnls"}},
               "verdict": "open", "notes": ""}

        line = f"  {sid:28} train Sh={train_m['sharpe']:.2f} ret={train_m['ret']*100:+.0f}% dd={train_m['maxdd']*100:.0f}% top20={train_m['top20']:.2f}"
        if train_m["sharpe"] >= args.train_screen_sharpe:
            val_m = eval_spec(inputs, spec, "validate", universe, args.seeds, windows)
            if val_m:
                rec["metrics"]["validate"] = {k: v for k, v in val_m.items() if k != "_pnls"}
                stats = per_trade_sharpe_stats(val_m["_pnls"])
                dsr = deflated_sharpe(stats["sr_hat"], stats["n_obs"], stats["skew"],
                                      stats["kurtosis"], n_trials, sr_var)
                rec["deflated_sharpe"] = round(dsr, 4)
                line += f" | VAL Sh={val_m['sharpe']:.2f} ret={val_m['ret']*100:+.0f}% DSR={dsr:.3f}"
                rec["_val_sr_hat"] = stats["sr_hat"]
        log.info(line)
        new_records.append(rec)
        n_trials += 1

    # update ledger (upsert by id so re-running a batch doesn't duplicate)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in new_records]
    new_ids = {r["id"] for r in clean}
    ledger["records"] = [r for r in ledger["records"] if r["id"] not in new_ids] + clean
    # update sr_variance from new validate per-obs Sharpes (running)
    new_srs = [r["_val_sr_hat"] for r in new_records if "_val_sr_hat" in r]
    if len(new_srs) >= 2:
        sr_var = round(0.5 * sr_var + 0.5 * float(np.var(new_srs)), 6)
    meta["n_trials_cumulative"] = n_trials
    meta["sr_variance"] = sr_var

    # frontier + nominees (validate window). Hardened gate: a nominee must be a
    # GENUINELY NEW core (not a cosmetic variant of an incumbent), clear DSR, sit
    # on the validate frontier, have enough validate trades (no small-subsample
    # flukes), and beat the best incumbent validate Sharpe by a real margin.
    after = frontier_for_window(ledger["records"], "validate")
    variants_rejected = []
    for r in new_records:
        m = r.get("metrics", {}).get("validate")
        if not m or r["id"] not in after or r.get("deflated_sharpe", 0) < args.dsr_pass:
            continue
        if m.get("trades", 0) < args.min_validate_trades:
            variants_rejected.append((r["id"], f"only {m.get('trades')} validate trades"))
            continue
        if core_key(r["spec"]) in incumbent_cores:
            variants_rejected.append((r["id"], "cosmetic variant of an existing core"))
            continue
        if m["sharpe"] < incumbent_best_val + args.nominee_margin:
            variants_rejected.append((r["id"], f"doesn't beat incumbent by {args.nominee_margin}"))
            continue
        nominees.append(r["id"])

    if nominees:
        meta["stall_counter"] = 0
    else:
        meta["stall_counter"] = meta.get("stall_counter", 0) + 1

    json.dump(ledger, open(args.ledger, "w"), indent=2)
    db.close()

    # report
    log.info("=" * 80)
    log.info("BATCH %s: %d evaluated, %d skipped(banned), n_trials_cumulative=%d",
             batch_no, len(new_records), len(skipped), n_trials)
    log.info("validate frontier: %s", after)
    if nominees:
        log.info(">>> NOMINEES (Pareto-improve validate AND DSR>=%.2f): %s", args.dsr_pass, nominees)
        log.info(">>> HUMAN GATE: approve before spending the one-shot TEST-window read.")
    if variants_rejected:
        log.info("rejected (frontier+DSR but not a real new unlock): %s", variants_rejected)
    if not nominees:
        log.info("No nominees this batch (stall_counter=%d/%d).", meta["stall_counter"], meta["stall_threshold"])
    if skipped:
        log.info("skipped: %s", skipped)


if __name__ == "__main__":
    main()
