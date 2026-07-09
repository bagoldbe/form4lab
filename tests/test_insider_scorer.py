import pytest
import numpy as np
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.scoring.insider_scorer import (
    bayesian_hit_rate,
    shrinkage_excess_return,
    momentum_adjustment,
    compute_credibility,
    compute_skill_score,
    assign_tier,
    compute_insider_score,
    refresh_all_scores,
)


# ---------------------------------------------------------------------------
# bayesian_hit_rate tests
# ---------------------------------------------------------------------------

def test_bayesian_hit_rate_prior_dominates_small_sample():
    """With 1 trade (1 hit), posterior should be close to prior (55%)."""
    result = bayesian_hit_rate(hits=1, total=1)
    assert 0.50 < result["bayesian_hit_rate"] < 0.65


def test_bayesian_hit_rate_data_dominates_large_sample():
    """With 100 trades (90 hits), posterior should be close to 90%."""
    result = bayesian_hit_rate(hits=90, total=100)
    assert result["bayesian_hit_rate"] > 0.85


def test_bayesian_hit_rate_confidence_above_baseline():
    """Elite insider (90% over 20 trades) should have high confidence."""
    result = bayesian_hit_rate(hits=18, total=20)
    assert result["confidence_above_baseline"] > 0.90


def test_bayesian_hit_rate_average_insider():
    """55% over 10 trades should have ~50% confidence above baseline."""
    result = bayesian_hit_rate(hits=6, total=10)
    assert 0.30 < result["confidence_above_baseline"] < 0.70


def test_bayesian_hit_rate_zero_hits():
    """0 hits should produce low hit rate and low confidence."""
    result = bayesian_hit_rate(hits=0, total=10)
    assert result["bayesian_hit_rate"] < 0.40
    assert result["confidence_above_baseline"] < 0.05


def test_bayesian_hit_rate_posterior_params():
    """Verify alpha_post and beta_post are correctly computed."""
    result = bayesian_hit_rate(hits=3, total=5, alpha_0=5.5, beta_0=4.5)
    assert result["alpha_post"] == 8.5
    assert result["beta_post"] == 6.5


# ---------------------------------------------------------------------------
# shrinkage_excess_return tests
# ---------------------------------------------------------------------------

def test_shrinkage_small_sample():
    """With few observations, should shrink toward prior mean (1%)."""
    result = shrinkage_excess_return([0.20], prior_mu=0.01, k=5)
    assert 0.01 < result["shrunk_excess"] < 0.20


def test_shrinkage_large_sample():
    """With many observations, should be close to sample mean."""
    returns = [0.10] * 50
    result = shrinkage_excess_return(returns, prior_mu=0.01, k=5)
    assert result["shrunk_excess"] > 0.08


def test_shrinkage_prob_positive():
    """Strongly positive returns should have high probability positive."""
    returns = [0.10] * 20
    result = shrinkage_excess_return(returns, prior_mu=0.01, k=5)
    assert result["prob_positive"] > 0.90


def test_shrinkage_negative_returns():
    """Negative returns should shrink toward prior but remain negative-ish."""
    returns = [-0.10] * 20
    result = shrinkage_excess_return(returns, prior_mu=0.01, k=5)
    assert result["shrunk_excess"] < 0.0


# ---------------------------------------------------------------------------
# momentum_adjustment tests
# ---------------------------------------------------------------------------

def test_momentum_adjustment_insufficient_data():
    """With fewer than 5 data points, should return None."""
    result = momentum_adjustment([0.1, 0.2], [0.05, 0.10])
    assert result is None


def test_momentum_adjustment_with_data():
    """With enough data, should return the intercept."""
    np.random.seed(42)
    momentums = list(np.random.randn(20) * 0.1)
    returns = [0.05 + 0.3 * m + np.random.randn() * 0.02 for m in momentums]
    result = momentum_adjustment(momentums, returns)
    assert result is not None
    assert abs(result - 0.05) < 0.03  # close to true intercept of 0.05


def test_momentum_adjustment_filters_none():
    """None values in momentums/returns should be filtered out."""
    momentums = [0.1, None, 0.2, 0.3, 0.4, 0.5, None]
    returns = [0.05, 0.10, None, 0.15, 0.20, 0.25, 0.30]
    result = momentum_adjustment(momentums, returns)
    # Only 3 valid pairs after filtering, so should be None (< 5)
    assert result is None


# ---------------------------------------------------------------------------
# compute_credibility tests
# ---------------------------------------------------------------------------

def test_credibility_weight():
    assert compute_credibility(0) == 0.0
    assert compute_credibility(8) == 1.0
    assert 0.5 < compute_credibility(4) < 0.8
    assert compute_credibility(20) == 1.0


def test_credibility_negative_n():
    """Negative n should return 0."""
    assert compute_credibility(-1) == 0.0


# ---------------------------------------------------------------------------
# compute_skill_score tests
# ---------------------------------------------------------------------------

def test_skill_score_elite():
    """High hit rate + high excess return + enough trades = high score."""
    score = compute_skill_score(
        bayesian_hit_rate=0.85, momentum_adjusted_excess=0.12, credibility=1.0
    )
    assert score > 1.0


def test_skill_score_weak():
    """Low hit rate + negative excess = negative score."""
    score = compute_skill_score(
        bayesian_hit_rate=0.35, momentum_adjusted_excess=-0.05, credibility=1.0
    )
    assert score < 0.0


def test_skill_score_zero_credibility():
    """Zero credibility should zero out the score."""
    score = compute_skill_score(
        bayesian_hit_rate=0.85, momentum_adjusted_excess=0.12, credibility=0.0
    )
    assert score == 0.0


def test_logit_clamp_all_hits():
    """100% hit rate should not cause infinity."""
    score = compute_skill_score(bayesian_hit_rate=0.99, momentum_adjusted_excess=0.1, credibility=1.0)
    assert np.isfinite(score)


def test_logit_clamp_zero_hits():
    """0% hit rate should not cause negative infinity."""
    score = compute_skill_score(bayesian_hit_rate=0.01, momentum_adjusted_excess=-0.1, credibility=1.0)
    assert np.isfinite(score)


def test_skill_score_symmetric():
    """Scores should be roughly symmetric around 0.5 hit rate and 0 excess."""
    score_high = compute_skill_score(
        bayesian_hit_rate=0.70, momentum_adjusted_excess=0.05, credibility=1.0
    )
    score_low = compute_skill_score(
        bayesian_hit_rate=0.30, momentum_adjusted_excess=-0.05, credibility=1.0
    )
    # Both should be on opposite sides of zero
    assert score_high > 0
    assert score_low < 0


# ---------------------------------------------------------------------------
# assign_tier tests
# ---------------------------------------------------------------------------

def test_tier_elite():
    assert assign_tier(skill_score=2.0, confidence=0.85, n=10) == "Elite"


def test_tier_strong():
    assert assign_tier(skill_score=1.0, confidence=0.70, n=8) == "Strong"


def test_tier_average():
    assert assign_tier(skill_score=0.3, confidence=0.55, n=5) == "Average"


def test_tier_weak():
    assert assign_tier(skill_score=-0.5, confidence=0.30, n=5) == "Weak"


def test_tier_insufficient():
    assert assign_tier(skill_score=0.5, confidence=0.60, n=2) == "Insufficient"


def test_tier_elite_requires_min_trades():
    """Elite requires n >= 6. With n=5, should downgrade to Strong or Average."""
    tier = assign_tier(skill_score=2.0, confidence=0.85, n=5)
    assert tier != "Elite"
    assert tier == "Strong"


# ---------------------------------------------------------------------------
# Integration: compute_insider_score with database
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_compute_insider_score_insufficient(db):
    """Insider with fewer than 3 discretionary buys should be Insufficient."""
    insider = Insider(cik="100", name="Few Trades")
    company = Company(cik="200", ticker="FEW", name="Few Inc")
    db.add_all([insider, company])
    db.flush()

    # Add 2 discretionary buys with outcomes on different dates
    for i in range(2):
        txn = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number=f"ACC-FEW-{i}", filing_date=date(2024, 1, 1 + i),
            transaction_date=date(2024, 1, 1 + i), transaction_code="P",
            shares=100, acquired_or_disposed="A", is_discretionary=True
        )
        db.add(txn)
        db.flush()
        outcome = TradeOutcome(
            transaction_id=txn.id, excess_return_60d=0.05, hit_60d=True
        )
        db.add(outcome)
    db.flush()

    score = compute_insider_score(insider.id, db)
    assert score.credibility_tier == "Insufficient"
    assert score.num_discretionary_buys == 2
    assert score.skill_score == 0.0


def test_compute_insider_score_with_enough_trades(db):
    """Insider with 10+ discretionary buys (8 hits) should get a real score."""
    insider = Insider(cik="300", name="Good Trader")
    company = Company(cik="400", ticker="GOOD", name="Good Inc")
    db.add_all([insider, company])
    db.flush()

    # Add 10 discretionary buys, 8 hits with varied momentum values
    momentum_values = [0.01, -0.02, 0.03, 0.05, -0.01, 0.02, -0.03, 0.04, 0.00, -0.04]
    for i in range(10):
        txn = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number=f"ACC-GOOD-{i}", filing_date=date(2024, 1, 1 + i),
            transaction_date=date(2024, 1, 1 + i), transaction_code="P",
            shares=100, acquired_or_disposed="A", is_discretionary=True
        )
        db.add(txn)
        db.flush()
        is_hit = i < 8  # first 8 are hits
        outcome = TradeOutcome(
            transaction_id=txn.id,
            excess_return_60d=0.05 if is_hit else -0.03,
            hit_60d=is_hit,
            prior_momentum_20d=momentum_values[i]
        )
        db.add(outcome)
    db.flush()

    score = compute_insider_score(insider.id, db)
    assert score.credibility_tier != "Insufficient"
    assert score.num_discretionary_buys == 10
    assert score.bayesian_hit_rate is not None
    assert score.bayesian_hit_rate > 0.55  # should be above baseline with 80% raw
    assert score.skill_score != 0.0
    assert score.credibility_weight > 0.0


def test_refresh_all_scores(db):
    """refresh_all_scores should process all insiders with discretionary buys."""
    # Create two insiders
    ins1 = Insider(cik="500", name="Insider A")
    ins2 = Insider(cik="600", name="Insider B")
    company = Company(cik="700", ticker="REF", name="Refresh Inc")
    db.add_all([ins1, ins2, company])
    db.flush()

    # Give each 1 discretionary buy (will be insufficient but should still process)
    for ins in [ins1, ins2]:
        txn = Transaction(
            insider_id=ins.id, company_id=company.id,
            accession_number=f"ACC-REF-{ins.cik}", filing_date=date(2024, 1, 1),
            transaction_date=date(2024, 1, 1), transaction_code="P",
            shares=100, acquired_or_disposed="A", is_discretionary=True
        )
        db.add(txn)
        db.flush()

    count = refresh_all_scores(db)
    assert count == 2


def test_compute_insider_score_updates_existing(db):
    """If score already exists, it should be updated, not duplicated."""
    insider = Insider(cik="800", name="Updater")
    company = Company(cik="900", ticker="UPD", name="Update Inc")
    db.add_all([insider, company])
    db.flush()

    # First computation (insufficient)
    score1 = compute_insider_score(insider.id, db)
    score1_id = score1.id

    # Second computation should update the same record
    score2 = compute_insider_score(insider.id, db)
    assert score2.id == score1_id
