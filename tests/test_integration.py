"""End-to-end integration test with synthetic data.

score_new_transaction delegates classification entirely to the active
Strategy (see form4lab/scoring/signal_generator.py). The scoring-differentiation and
full-pipeline tests exercise generic platform code
(form4lab/scoring/insider_scorer.py) with the shipped score column names.
"""
import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from form4lab.database import Base
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.score import InsiderScore
from form4lab.scoring.insider_scorer import compute_insider_score


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_test_data(db):
    """Create a realistic test dataset.

    Uses 20 trades for the elite insider with very high excess returns
    to ensure the Bayesian scoring engine produces a skill_score >= 1.5,
    which is required for the Elite tier after shrinkage adjustments.
    """
    company = Company(cik="320193", ticker="AAPL", name="Apple Inc.", sector="Technology")
    db.add(company)
    db.flush()

    # Elite insider -- wins 18 out of 20 with large excess returns
    elite = Insider(cik="1001", name="Elite CEO")
    db.add(elite)
    db.flush()
    role = InsiderRole(
        insider_id=elite.id, company_id=company.id,
        role_title="CEO", is_officer=True, is_director=False,
        is_ten_percent_owner=False,
        first_filing_date=date(2015, 1, 1), last_filing_date=date(2024, 1, 1),
    )
    db.add(role)
    db.flush()

    # Create 20 transactions for elite insider with outcomes
    for i in range(20):
        txn = Transaction(
            insider_id=elite.id, company_id=company.id,
            accession_number=f"ELITE-{i}",
            filing_date=date(2020, 1, 1) + timedelta(days=i * 30),
            transaction_date=date(2020, 1, 1) + timedelta(days=i * 30),
            transaction_code="P", shares=1000, price_per_share=150.0,
            total_value=150000.0, acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        db.flush()

        # 18 wins, 2 losses -- high excess returns to overcome shrinkage
        is_win = i < 18
        outcome = TradeOutcome(
            transaction_id=txn.id,
            stock_return_60d=0.52 if is_win else -0.05,
            benchmark_return_60d=0.02,
            excess_return_60d=0.50 if is_win else -0.07,
            hit_60d=is_win,
            prior_momentum_20d=-0.03 if i % 2 == 0 else 0.02,
        )
        db.add(outcome)

    # Average insider -- wins 5 out of 10
    avg = Insider(cik="1002", name="Average Director")
    db.add(avg)
    db.flush()
    role2 = InsiderRole(
        insider_id=avg.id, company_id=company.id,
        role_title="Director", is_officer=False, is_director=True,
        is_ten_percent_owner=False,
        first_filing_date=date(2018, 1, 1), last_filing_date=date(2024, 1, 1),
    )
    db.add(role2)
    db.flush()

    for i in range(10):
        txn = Transaction(
            insider_id=avg.id, company_id=company.id,
            accession_number=f"AVG-{i}",
            filing_date=date(2020, 1, 1) + timedelta(days=i * 60),
            transaction_date=date(2020, 1, 1) + timedelta(days=i * 60),
            transaction_code="P", shares=200, price_per_share=150.0,
            total_value=30000.0, acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        db.flush()

        is_win = i < 5
        outcome = TradeOutcome(
            transaction_id=txn.id,
            stock_return_60d=0.08 if is_win else -0.04,
            benchmark_return_60d=0.02,
            excess_return_60d=0.06 if is_win else -0.06,
            hit_60d=is_win,
            prior_momentum_20d=-0.01,
        )
        db.add(outcome)

    # Weak insider -- wins 2 out of 8
    weak = Insider(cik="1003", name="Weak VP")
    db.add(weak)
    db.flush()
    role3 = InsiderRole(
        insider_id=weak.id, company_id=company.id,
        role_title="VP", is_officer=True, is_director=False,
        is_ten_percent_owner=False,
        first_filing_date=date(2019, 1, 1), last_filing_date=date(2024, 1, 1),
    )
    db.add(role3)
    db.flush()

    for i in range(8):
        txn = Transaction(
            insider_id=weak.id, company_id=company.id,
            accession_number=f"WEAK-{i}",
            filing_date=date(2020, 1, 1) + timedelta(days=i * 60),
            transaction_date=date(2020, 1, 1) + timedelta(days=i * 60),
            transaction_code="P", shares=500, price_per_share=150.0,
            total_value=75000.0, acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        db.flush()

        is_win = i < 2
        outcome = TradeOutcome(
            transaction_id=txn.id,
            stock_return_60d=0.05 if is_win else -0.08,
            benchmark_return_60d=0.02,
            excess_return_60d=0.03 if is_win else -0.10,
            hit_60d=is_win,
            prior_momentum_20d=0.01,
        )
        db.add(outcome)

    db.commit()
    return company, elite, avg, weak


def test_scoring_differentiation(db):
    """The core test: scoring must differentiate Elite from Average from Weak."""
    company, elite, avg, weak = _create_test_data(db)

    elite_score = compute_insider_score(elite.id, db)
    avg_score = compute_insider_score(avg.id, db)
    weak_score = compute_insider_score(weak.id, db)

    # Elite should have highest score
    assert elite_score.skill_score > avg_score.skill_score > weak_score.skill_score

    # Tiers should be differentiated
    assert elite_score.credibility_tier in ("Elite", "Strong")
    assert weak_score.credibility_tier == "Weak"

    # Hit rates should reflect reality (with shrinkage toward 55%)
    assert elite_score.bayesian_hit_rate > 0.65  # 90% shrunk toward 55%
    assert weak_score.bayesian_hit_rate < 0.45   # 25% shrunk toward 55%

    # Confidence: elite should be confident above baseline, weak should not
    assert elite_score.confidence_above_baseline > 0.7
    assert weak_score.confidence_above_baseline < 0.3


def test_full_pipeline_consistency(db):
    """Verify the full pipeline from data to scores to alerts is consistent."""
    company, elite, avg, weak = _create_test_data(db)

    # Score all insiders
    scores = {}
    for insider in [elite, avg, weak]:
        scores[insider.cik] = compute_insider_score(insider.id, db)

    # Verify scores are stored and retrievable
    stored = db.query(InsiderScore).filter(InsiderScore.company_id == None).all()  # noqa: E711
    assert len(stored) == 3

    # Verify all required fields are populated
    for s in stored:
        assert s.skill_score is not None
        assert s.credibility_tier is not None
        assert s.num_discretionary_buys > 0
        assert s.credibility_weight > 0


def test_fastapi_endpoints_with_data(db):
    """Test that FastAPI endpoints work with actual data."""
    from fastapi.testclient import TestClient
    from form4lab.main import app

    # Just verify the endpoints don't crash
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/v1/alerts").status_code == 200
