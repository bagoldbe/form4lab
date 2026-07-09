"""Declarative search space for form4lab.research.loop.

A *spec* is a JSON-serializable dict describing one backtest configuration:

    {
      "base":       "cluster_buy",   # a name in BASE_SIGNALS — which active-
                                      # strategy tradeable signal this spec's
                                      # trades restrict to (default: "cluster_buy")
      "atoms":      ["value_100k"],  # extra row-predicate atoms, AND-composed
      "hold_days":  60,
      "window":     "train",         # a key into the `windows` dict you pass
                                      # to resolve_spec() — see WINDOWS note below
    }

`resolve_spec()` turns a spec into form4lab.scoring.portfolio_simulator.
run_simulation kwargs, reusing the simulator engine so hypotheses become
storable, comparable, and machine-generable — see form4lab.research.loop for
the train -> validate -> (human-gated) test loop that drives specs through it.

This module ships EMPTY of research conclusions: six public-literature atoms,
one base signal (the shipped ClusterBuyStrategy's only tradeable signal), no
banned regions, no additional levers. As your own research program falsifies
regions of the space or earns new atoms, add them here (or in a module of
your own — see form4lab.research.loop's FORM4LAB_RESEARCH_SPACE env var).

WINDOWS is deliberately NOT a module constant: date ranges belong to your own
train/validate/test split, not to this library. Build your own
{"train": (start, end), "validate": (start, end), ...} mapping (dates or
None) and pass it to resolve_spec() as `windows` whenever a spec names a
"window" — form4lab.research.loop builds this from its own required CLI args.
"""
import math
from typing import Callable

Row = object  # a mapping-like row (pandas Series or dict) exposing .get(key, default)


def _nn(x) -> bool:
    """True if x is present and not NaN (numeric null-check for backtest rows)."""
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# Row-predicate atoms — operate on a row of the `buys` DataFrame that
# form4lab.scoring.flags / portfolio_simulator.prepare_backtest_inputs builds
# (accessed via .get(), so a missing column degrades safely instead of
# raising). Six public-literature entries; extend with your own findings.
# ---------------------------------------------------------------------------

def _cluster_2plus(r: Row) -> bool:
    """2+ distinct insiders bought within the cluster window (Bettis/Vickrey)."""
    v = r.get("cluster_size")
    return _nn(v) and v >= 2


def _senior_role(r: Row) -> bool:
    """Buyer holds a C-suite title (CEO/CFO/COO/CTO/President/Chairman)."""
    from form4lab.utils import is_csuite
    return is_csuite(r.get("role_title"))


def _value_100k(r: Row) -> bool:
    """Transaction dollar value >= $100k."""
    v = r.get("total_value")
    return _nn(v) and v >= 100_000


def _not_10b5_1(r: Row) -> bool:
    """Not filed under a pre-arranged Rule 10b5-1 trading plan.

    form4lab.scoring.flags.load_backtest_data SELECTs Transaction.is_10b5_1_plan
    into the buys frame (and carries it through the same-day multi-lot dedup
    groupby/agg), so this atom is live. A missing or null value degrades to
    True (treated as "not a 10b5-1 plan"). Uses _nn + bool() rather than an
    `is not True` identity check: SQLite has no native boolean type, so this
    column round-trips as numpy.int64(0/1) (or float64 with NaN when nulls
    are mixed in), and `np.int64(1) is True` is False — an identity check
    would silently leave the atom a no-op on every SQLite-backed row.
    """
    v = r.get("is_10b5_1_plan")
    return not (_nn(v) and bool(v))


def _opportunistic(r: Row) -> bool:
    """NOT a routine trader (Cohen, Malloy & Pomorski 2012) — all predictive
    power in the literature sits in the non-routine group."""
    return not bool(r.get("is_routine", False))


def _first_buy(r: Row) -> bool:
    """This insider's first observed open-market buy in this company (bounded
    by the backfill window — see compute_firsttime_flags's docstring)."""
    return bool(r.get("is_first_buy", False))


ATOMS: dict[str, Callable[[Row], bool]] = {
    "cluster_2plus": _cluster_2plus,
    "senior_role": _senior_role,
    "value_100k": _value_100k,
    "not_10b5_1": _not_10b5_1,
    "opportunistic": _opportunistic,
    "first_buy": _first_buy,
}

# Valid values for a spec's "base" key: the active strategy's tradeable
# signal name(s) this space knows about. The shipped default strategy
# (form4lab.strategies.cluster_buy.ClusterBuyStrategy) registers exactly one
# tradeable signal, "cluster_buy". Add your own strategy's signal names here
# as you register them (see form4lab.strategy.registry.SignalRegistry).
BASE_SIGNALS: set[str] = {"cluster_buy"}

# Named OR-unions across atom groups (a row enters if ANY leg's atoms ALL
# hold) — none shipped. To add one: a dict of name -> list[list[str]], each
# inner list AND-ed and the outer list OR-ed, consulted from resolve_spec()
# the same way BASE_SIGNALS is.

# The legal space a generator may sample from — empty. Extend as your program
# grows, e.g. LEVERS = {"atoms": list(ATOMS), "hold_days": [10, 30, 90]}, and
# have your spec generator sample only from these keys/values.
LEVERS: dict[str, list] = {}


def is_banned(spec: dict) -> tuple[bool, str]:
    """Banned-region guard: reject a spec before it burns a training-window trial.

    Ships empty — a fresh research program has no falsified regions yet.
    Populate with regions your own research has falsified (e.g. a lever
    family that consistently underperforms in your train-window screens),
    returning (True, "short reason") for each. The loop logs the reason and
    skips the spec without spending a trial.
    """
    return False, ""


def _build_predicate(spec: dict) -> Callable[[Row, str, float], bool]:
    atom_names = spec.get("atoms", [])
    unknown = [a for a in atom_names if a not in ATOMS]
    if unknown:
        raise ValueError(f"unknown atom(s): {unknown}")
    funcs = [ATOMS[a] for a in atom_names]

    def predicate(row: Row, tier: str, skill_score: float) -> bool:
        return all(f(row) for f in funcs)

    return predicate


def resolve_spec(spec: dict, windows: dict[str, tuple] | None = None) -> dict:
    """Turn a spec into form4lab.scoring.portfolio_simulator.run_simulation kwargs.

    NEUTRAL defaults — nothing here engages margin, a drawdown entry
    filter, or idle-cash sleeve parking, and sizing is left to the
    strategy's own role-tiered `size()` (no size_fn override):
        margin_multiplier=1.0, drawdown_threshold=None, spy_parking=False

    Args:
        spec: see the module docstring for shape. `base` (default
            "cluster_buy") must be a name in BASE_SIGNALS. `atoms` (default
            []) are AND-composed into the returned `signal_predicate`.
        windows: a {name: (start_date, end_date)} mapping (dates or None) for
            resolving spec["window"], e.g. {"train": (None, date(2015, 12, 31))}.
            Required only when the spec names a "window" — this module never
            hardcodes date ranges itself; see the module docstring.

    Returns:
        kwargs suitable for run_simulation(**kwargs). `signal_predicate` is
        present whenever `atoms` is non-empty. NOTE: run_simulation treats
        signal_predicate as the SOLE entry gate when set — it supersedes
        `target_signals` entirely. A spec that combines `base` with `atoms`
        therefore only restricts on `base` via `target_signals` when `atoms`
        is empty; fold a base-signal check into an atom yourself if you need
        both enforced together once `atoms` is non-empty.

    Raises:
        ValueError: unknown `base`, unknown atom name, or an unresolvable
            `window` name (not found in `windows`).
    """
    banned, why = is_banned(spec)
    if banned:
        raise ValueError(f"banned spec: {why}")

    base = spec.get("base", "cluster_buy")
    if base not in BASE_SIGNALS:
        raise ValueError(f"unknown base signal {base!r} (not in BASE_SIGNALS)")

    kwargs: dict = dict(
        target_signals={base},
        hold_days=spec.get("hold_days", 60),
        margin_multiplier=1.0,     # neutral: no margin
        drawdown_threshold=None,   # neutral: no drawdown entry filter
        spy_parking=False,         # neutral: no idle-cash sleeve parking
        # sizing intentionally omitted: no size_fn means run_simulation falls
        # back to the active strategy's role-tiered size() (see
        # form4lab.strategy.base.Strategy.size).
    )

    window_name = spec.get("window")
    if window_name is not None:
        if not windows or window_name not in windows:
            raise ValueError(
                f"window {window_name!r} not found — pass a `windows` mapping "
                "with that key to resolve_spec()"
            )
        kwargs["start_date"], kwargs["end_date"] = windows[window_name]

    atom_names = spec.get("atoms", [])
    if atom_names:
        kwargs["signal_predicate"] = _build_predicate(spec)

    return kwargs


def spec_complexity(spec: dict) -> int:
    """Number of active levers beyond the base signal — a parsimony tie-break
    for comparing two candidates that score similarly (prefer the simpler)."""
    n = len(spec.get("atoms", []))
    if spec.get("base", "cluster_buy") != "cluster_buy":
        n += 1
    return n
