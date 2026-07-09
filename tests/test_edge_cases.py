"""Edge case and negative tests for scoring, signals, and outcomes.

Tests behavior with NULL values, zero transactions, missing data,
and boundary conditions that could cause silent failures.
"""
import uuid
from datetime import date

import numpy as np

from form4lab.database import SessionLocal, Base, engine
from form4lab.models.alert import Alert
from form4lab.models.company import Company
from form4lab.models.insider import Insider
from form4lab.models.outcome import TradeOutcome
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.scoring.insider_scorer import (
    bayesian_hit_rate,
    shrinkage_excess_return,
    momentum_adjustment,
    compute_credibility,
    compute_skill_score,
    assign_tier,
)
from form4lab.scoring.outcome_calculator import (
    compute_forward_return,
    compute_trade_outcomes,
    _get_price_at_date,
)
from form4lab.utils import to_python_float
from form4lab.data.utils import persist_transaction


def _uid():
    return str(uuid.uuid4().int)[:10]


# --- to_python_float ---

def test_to_python_float_none():
    assert to_python_float(None) is None


def test_to_python_float_numpy():
    val = np.float64(3.14)
    result = to_python_float(val)
    assert isinstance(result, float)
    assert abs(result - 3.14) < 1e-10


def test_to_python_float_int():
    assert to_python_float(42) == 42.0
    assert isinstance(to_python_float(42), float)


def test_to_python_float_zero():
    assert to_python_float(0) == 0.0


# --- Bayesian scoring edge cases ---

def test_bayesian_hit_rate_zero_total():
    """Zero total trades should produce prior-dominated result."""
    result = bayesian_hit_rate(0, 0)
    assert abs(result["bayesian_hit_rate"] - 0.55) < 0.01


def test_bayesian_hit_rate_all_hits():
    """All hits with small sample should still be pulled toward prior."""
    result = bayesian_hit_rate(3, 3)
    # With 3 hits out of 3, posterior should be > baseline but < 1.0
    assert result["bayesian_hit_rate"] > 0.55
    assert result["bayesian_hit_rate"] < 1.0


def test_shrinkage_empty_list():
    """Empty excess returns list should still produce valid result."""
    result = shrinkage_excess_return([])
    assert "shrunk_excess" in result
    # With 0 observations, should return prior
    assert abs(result["shrunk_excess"] - 0.01) < 0.01


def test_shrinkage_single_value():
    """Single observation should be heavily shrunk toward prior."""
    result = shrinkage_excess_return([0.50])
    assert result["shrunk_excess"] < 0.50  # pulled toward 0.01 prior


def test_shrinkage_all_zeros():
    """All zero returns should produce near-prior result."""
    result = shrinkage_excess_return([0.0, 0.0, 0.0, 0.0])
    # Heavily shrunk toward 0.01
    assert abs(result["shrunk_excess"]) < 0.05


def test_momentum_adjustment_identical_values():
    """All identical momentum values should return None (zero variance)."""
    result = momentum_adjustment([0.05] * 10, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert result is None


def test_momentum_adjustment_too_few_pairs():
    """Below minimum pairs should return None."""
    result = momentum_adjustment([0.1, 0.2], [0.3, 0.4])
    assert result is None


def test_credibility_zero():
    assert compute_credibility(0) == 0.0


def test_credibility_one():
    assert compute_credibility(1) > 0.0
    assert compute_credibility(1) < 1.0


def test_credibility_large_n():
    """Large n should approach 1.0."""
    assert compute_credibility(100) == 1.0


def test_skill_score_zero_credibility():
    """Zero credibility should produce zero skill score."""
    assert compute_skill_score(0.55, 0.01, 0.0) == 0.0


def test_assign_tier_boundary():
    """Test exact boundary values for tier assignment."""
    assert assign_tier(1.5, 0.80, 6) == "Elite"
    assert assign_tier(1.49, 0.80, 6) != "Elite"  # Just below
    assert assign_tier(1.5, 0.79, 6) != "Elite"   # Confidence too low
    assert assign_tier(1.5, 0.80, 5) != "Elite"   # Too few trades


# --- Outcome calculator edge cases ---

def test_forward_return_zero_price():
    assert compute_forward_return(0, 100) == 0.0


def test_forward_return_same_price():
    assert compute_forward_return(50, 50) == 0.0


def test_get_price_at_date_empty_df():
    import pandas as pd
    df = pd.DataFrame(columns=["date", "close", "adj_close"])
    assert _get_price_at_date(df, date(2026, 1, 15)) is None


def test_compute_trade_outcomes_nonexistent_transaction():
    """Should return None for a transaction ID that doesn't exist."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        from form4lab.data.price_fetcher import YFinanceProvider
        provider = YFinanceProvider(db, db_only=True)
        result = compute_trade_outcomes(999999999, db, provider)
        assert result is None


# --- persist_transaction edge cases ---

def test_persist_transaction_dedup():
    """Second insert with same accession_number should return None."""
    Base.metadata.create_all(engine)
    acc = f"dedup-test-{uuid.uuid4().hex[:8]}_P_2026-01-20_100.0"
    cik1, cik2 = _uid(), _uid()
    txn_data = {
        "insider_cik": cik1,
        "insider_name": "Dedup Test",
        "is_officer": False,
        "is_director": False,
        "is_ten_pct_owner": False,
        "officer_title": "",
        "company_cik": cik2,
        "company_name": "Dedup Corp",
        "company_ticker": "DDP",
        "accession_number": acc,
        "filing_date": date(2026, 1, 20),
        "transaction_date": date(2026, 1, 20),
        "transaction_code": "P",
        "shares": 100,
        "price_per_share": 50.0,
        "total_value": 5000.0,
        "shares_owned_after": 500,
        "acquired_or_disposed": "A",
        "is_discretionary": True,
        "security_title": "Common Stock",
        "is_common_stock": True,
    }
    with SessionLocal() as db:
        result1 = persist_transaction(txn_data, db)
        db.commit()
        assert result1 is not None

        result2 = persist_transaction(txn_data, db)
        assert result2 is None  # duplicate


def test_persist_transaction_null_price():
    """Transaction with None price should still persist."""
    Base.metadata.create_all(engine)
    cik1, cik2 = _uid(), _uid()
    txn_data = {
        "insider_cik": cik1,
        "insider_name": "Null Price Test",
        "is_officer": False,
        "is_director": False,
        "is_ten_pct_owner": False,
        "officer_title": "",
        "company_cik": cik2,
        "company_name": "NullPrice Corp",
        "company_ticker": None,
        "accession_number": f"null-price-{uuid.uuid4().hex[:8]}_P_2026-01-21_50.0",
        "filing_date": date(2026, 1, 21),
        "transaction_date": date(2026, 1, 21),
        "transaction_code": "P",
        "shares": 50,
        "price_per_share": None,
        "total_value": None,
        "shares_owned_after": None,
        "acquired_or_disposed": "A",
        "is_discretionary": True,
        "security_title": "Common Stock",
        "is_common_stock": True,
    }
    with SessionLocal() as db:
        result = persist_transaction(txn_data, db)
        db.commit()
        assert result is not None
        assert result.price_per_share is None
