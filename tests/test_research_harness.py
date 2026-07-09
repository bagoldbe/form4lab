"""Tests for the anti-overfitting research harness skeleton
(form4lab.research.stats / space / loop)."""
import json
import os
from datetime import date

import pytest

from form4lab.research import loop as rl
from form4lab.research import space as ss
from form4lab.research.stats import deflated_sharpe

# ---------------------------------------------------------------------------
# stats.py — Deflated Sharpe Ratio (pure Bailey & Lopez de Prado 2014 math,
# ported verbatim). This numeric case is a hand-checkable math fixture, not
# a strategy result.
# ---------------------------------------------------------------------------

def test_deflated_sharpe_strong_candidate_with_few_trials_passes():
    """A strong, well-sampled candidate clears the DSR bar even after several
    prior trials have deflated the null benchmark."""
    assert deflated_sharpe(0.25, 300, 0.2, 4.0, 5, 0.005) > 0.95


# ---------------------------------------------------------------------------
# space.py — is_banned ships empty
# ---------------------------------------------------------------------------

def test_is_banned_always_false():
    """Fresh research program: no region of the space has been falsified yet."""
    banned, why = ss.is_banned({"base": "cluster_buy", "atoms": ["value_100k"]})
    assert banned is False
    assert why == ""


# ---------------------------------------------------------------------------
# space.py — resolve_spec / signal_predicate composition
# ---------------------------------------------------------------------------

def test_resolve_spec_returns_callable_signal_predicate():
    kwargs = ss.resolve_spec({"base": "cluster_buy", "atoms": ["value_100k"]})
    assert callable(kwargs["signal_predicate"])
    assert kwargs["target_signals"] == {"cluster_buy"}
    # neutral defaults — no margin, no drawdown filter, no idle-cash parking
    assert kwargs["margin_multiplier"] == 1.0
    assert kwargs["drawdown_threshold"] is None
    assert kwargs["spy_parking"] is False


def test_signal_predicate_behaves_on_synthetic_rows():
    kwargs = ss.resolve_spec({"base": "cluster_buy", "atoms": ["value_100k"]})
    predicate = kwargs["signal_predicate"]
    above_threshold = {"total_value": 150_000.0}
    below_threshold = {"total_value": 50_000.0}
    assert predicate(above_threshold, "Insufficient", 0.0) is True
    assert predicate(below_threshold, "Insufficient", 0.0) is False


def test_not_10b5_1_predicate_excludes_10b5_1_plan_rows():
    """form4lab.scoring.flags.load_backtest_data now SELECTs is_10b5_1_plan
    into the buys frame, so this atom is no longer a no-op (see its docstring
    history) — a plan-filed buy is excluded, everything else (including a
    row missing the column entirely) passes through."""
    kwargs = ss.resolve_spec({"base": "cluster_buy", "atoms": ["not_10b5_1"]})
    predicate = kwargs["signal_predicate"]
    assert predicate({"is_10b5_1_plan": True}, "Insufficient", 0.0) is False
    assert predicate({"is_10b5_1_plan": False}, "Insufficient", 0.0) is True
    assert predicate({}, "Insufficient", 0.0) is True  # missing column degrades safely


def test_resolve_spec_without_atoms_has_no_predicate():
    """No atoms -> no signal_predicate; target_signals alone does the work
    (see resolve_spec's docstring on signal_predicate/target_signals precedence)."""
    kwargs = ss.resolve_spec({"base": "cluster_buy"})
    assert "signal_predicate" not in kwargs
    assert kwargs["target_signals"] == {"cluster_buy"}


def test_resolve_spec_unknown_base_raises():
    with pytest.raises(ValueError):
        ss.resolve_spec({"base": "not_a_real_signal"})


def test_resolve_spec_unknown_atom_raises():
    with pytest.raises(ValueError):
        ss.resolve_spec({"base": "cluster_buy", "atoms": ["not_a_real_atom"]})


def test_resolve_spec_resolves_named_window():
    windows = {"train": (None, date(2020, 12, 31))}
    kwargs = ss.resolve_spec({"base": "cluster_buy", "window": "train"}, windows=windows)
    assert kwargs["start_date"] is None
    assert kwargs["end_date"] == date(2020, 12, 31)


def test_resolve_spec_named_window_without_windows_mapping_raises():
    with pytest.raises(ValueError):
        ss.resolve_spec({"base": "cluster_buy", "window": "train"})


# ---------------------------------------------------------------------------
# space.py — spec_complexity (parsimony tie-break)
# ---------------------------------------------------------------------------

def test_spec_complexity_trivial_spec_is_zero():
    assert ss.spec_complexity({"base": "cluster_buy"}) == 0


def test_spec_complexity_counts_atoms():
    assert ss.spec_complexity(
        {"base": "cluster_buy", "atoms": ["value_100k", "senior_role"]}) == 2


# ---------------------------------------------------------------------------
# All six shipped atoms must degrade safely (no KeyError/TypeError) on a row
# missing every column — real backtest rows always have these columns, but a
# defensive .get()-based atom should never crash on an incomplete one.
# ---------------------------------------------------------------------------

def test_atoms_is_exactly_the_six_shipped_entries():
    assert set(ss.ATOMS) == {
        "cluster_2plus", "senior_role", "value_100k",
        "not_10b5_1", "opportunistic", "first_buy",
    }


@pytest.mark.parametrize("atom_name", list(ss.ATOMS))
def test_every_atom_handles_a_row_with_no_columns(atom_name):
    assert ss.ATOMS[atom_name]({}) in (True, False)


# ---------------------------------------------------------------------------
# loop.py — pure-Python helpers (no DB required)
# ---------------------------------------------------------------------------

def test_core_key_ignores_atoms():
    """Atoms are cosmetic for nominee dedup — only base/sizing/exits define a core."""
    base = {"base": "cluster_buy"}
    with_atoms = {"base": "cluster_buy", "atoms": ["value_100k"]}
    assert rl.core_key(base) == rl.core_key(with_atoms)


def test_core_key_differs_on_base():
    assert rl.core_key({"base": "cluster_buy"}) != rl.core_key({"base": "other_signal"})


def test_load_universe_none_when_no_files():
    assert rl.load_universe(None) is None
    assert rl.load_universe([]) is None


def test_load_universe_unions_files_and_skips_comments(tmp_path):
    f1 = tmp_path / "u1.txt"
    f1.write_text("aapl\n# a comment\nmsft\n\n")
    f2 = tmp_path / "u2.txt"
    f2.write_text("GOOG\n")
    assert rl.load_universe([str(f1), str(f2)]) == {"AAPL", "MSFT", "GOOG"}


def test_load_ledger_fresh_when_missing(tmp_path):
    path = str(tmp_path / "ledger.json")
    ledger = rl._load_ledger(path, stall_threshold=7)
    assert ledger["records"] == []
    assert ledger["meta"] == {
        "n_trials_cumulative": 0, "sr_variance": 0.0,
        "stall_counter": 0, "stall_threshold": 7,
    }


def test_load_ledger_reads_existing_file(tmp_path):
    path = str(tmp_path / "ledger.json")
    seeded = {"meta": {"n_trials_cumulative": 3, "sr_variance": 0.01,
                       "stall_counter": 1, "stall_threshold": 5},
              "records": [{"id": "x"}]}
    with open(path, "w") as f:
        json.dump(seeded, f)
    assert rl._load_ledger(path, stall_threshold=99) == seeded
