"""Golden-baseline harness for refactors of the simulation engine.

Runs two pinned simulations and dumps every position (closed + open) plus
metrics. `--record` writes the goldens; `--compare` re-runs and byte-diffs
against them. A refactor's acceptance gate is an empty diff.

No golden files ship with this repo — record your own against your own
database (`--record`) before starting a refactor, then `--compare` after.
"""
import argparse
import csv
import filecmp
import json
import shutil
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))  # repo root on sys.path so `form4lab.*` imports resolve
OUT = BASE / "analysis" / "baselines"

# Pinned DB source for baseline runs (override to point at your own data).
DB_URL = "sqlite:///form4lab.db"

CONFIGS = {
    "plain": dict(shuffle_seed=42),
    "margin": dict(
        shuffle_seed=42,
        margin_multiplier=1.5,
        margin_interest_rate=0.06,
        spy_parking=True,
        spy_parking_buffer=0.20,
    ),
}

FIELDS = [
    "txn_id", "ticker", "insider_id", "signal_type", "entry_date", "entry_price",
    "shares_held", "cost_basis", "position_pct", "hold_days", "exit_date",
    "exit_price", "pnl", "pnl_pct", "force_closed", "tier", "skill_score",
    "role_title", "extensions",
]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def dump(portfolio, path: Path):
    rows = list(portfolio.closed_positions) + list(portfolio.open_positions)
    rows.sort(key=lambda p: (p.txn_id, str(p.entry_date), p.ticker, f"{p.cost_basis:.10g}"))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for p in rows:
            writer.writerow([_fmt(getattr(p, k)) for k in FIELDS])
    return len(rows)


def run(name: str, out_dir: Path, extra_kwargs: dict | None = None):
    from form4lab.scoring.portfolio_simulator import run_simulation, compute_metrics
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    kwargs = dict(CONFIGS[name])
    kwargs.update(extra_kwargs or {})
    with Session() as db:
        portfolio, price_index = run_simulation(db, **kwargs)
    n = dump(portfolio, out_dir / f"{name}_trades.csv")
    metrics = compute_metrics(portfolio, price_index)
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump({k: _fmt(v) for k, v in sorted(metrics.items())}, f, indent=1, sort_keys=True)
    print(f"{name}: {n} positions dumped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--strategy-kwarg", action="store_true",
                    help="pass strategy=load_strategy() explicitly")
    args = ap.parse_args()

    extra = {}
    if args.strategy_kwarg:
        from form4lab.strategy.registry import load_strategy
        extra = {"strategy": load_strategy()}

    if args.record:
        OUT.mkdir(parents=True, exist_ok=True)
        for name in CONFIGS:
            run(name, OUT, extra)
        return

    if args.compare:
        tmp = OUT / "_compare"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        ok = True
        for name in CONFIGS:
            run(name, tmp, extra)
            for suffix in ("trades.csv", "metrics.json"):
                a, b = OUT / f"{name}_{suffix}", tmp / f"{name}_{suffix}"
                if not filecmp.cmp(a, b, shallow=False):
                    print(f"MISMATCH: {b} != {a}")
                    ok = False
        print("BASELINE OK" if ok else "BASELINE FAILED")
        sys.exit(0 if ok else 1)

    ap.error("pass --record or --compare")


if __name__ == "__main__":
    main()
