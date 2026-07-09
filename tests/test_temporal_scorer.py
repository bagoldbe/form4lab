import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.scoring.temporal_scorer import compute_temporal_score


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_insider_with_outcomes(db, num_trades=10, num_hits=8):
    """Create an insider with trades spread over time, each 30 days apart.

    Trade 0 is on 2020-01-01, trade 1 on 2020-01-31, etc.
    """
    insider = Insider(cik="100", name="Temporal Test")
    company = Company(cik="200", ticker="TMP", name="Temporal Corp")
    db.add_all([insider, company])
    db.flush()

    base = date(2020, 1, 1)
    for i in range(num_trades):
        txn_date = base + timedelta(days=i * 30)
        txn = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number=f"TMP-{i}", filing_date=txn_date,
            transaction_date=txn_date, transaction_code="P",
            shares=100, price_per_share=50.0, total_value=5000.0,
            acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        db.flush()
        is_hit = i < num_hits
        outcome = TradeOutcome(
            transaction_id=txn.id,
            excess_return_60d=0.08 if is_hit else -0.05,
            hit_60d=is_hit,
            prior_momentum_20d=0.01 * (i - 5),
        )
        db.add(outcome)
    db.commit()
    return insider, company, base


def test_temporal_score_uses_only_prior_data(db):
    """Score at midpoint should only use first half of trades."""
    insider, company, base = _seed_insider_with_outcomes(db, num_trades=10, num_hits=8)

    # Score as of trade #5 (5 months in). Should only see trades 0-4.
    midpoint = base + timedelta(days=5 * 30)
    score = compute_temporal_score(insider.id, db, as_of_date=midpoint)

    assert score["num_prior_buys"] == 5
    # First 5 trades are all hits (indices 0-4), so hit rate should be high
    assert score["bayesian_hit_rate"] > 0.55


def test_temporal_score_no_prior_data(db):
    """Score before any trades should return Insufficient."""
    insider, company, base = _seed_insider_with_outcomes(db, num_trades=10, num_hits=8)

    score = compute_temporal_score(insider.id, db, as_of_date=date(2019, 1, 1))

    assert score["num_prior_buys"] == 0
    assert score["tier"] == "Insufficient"
    assert score["skill_score"] == 0.0


def test_temporal_score_all_data(db):
    """Score after all trades should use everything."""
    insider, company, base = _seed_insider_with_outcomes(db, num_trades=10, num_hits=8)

    score = compute_temporal_score(insider.id, db, as_of_date=date(2025, 1, 1))

    assert score["num_prior_buys"] == 10


def test_temporal_score_grows_over_time(db):
    """Score credibility should increase as more data becomes available."""
    insider, company, base = _seed_insider_with_outcomes(db, num_trades=10, num_hits=8)

    early = compute_temporal_score(insider.id, db, as_of_date=base + timedelta(days=90))
    late = compute_temporal_score(insider.id, db, as_of_date=date(2025, 1, 1))

    assert late["credibility_weight"] >= early["credibility_weight"]


def test_temporal_score_insufficient_with_few_trades(db):
    """With fewer than 3 prior trades, tier should be Insufficient."""
    insider, company, base = _seed_insider_with_outcomes(db, num_trades=10, num_hits=8)

    # Only 2 trades before this date (trades 0 and 1)
    score = compute_temporal_score(insider.id, db, as_of_date=base + timedelta(days=55))

    assert score["num_prior_buys"] == 2
    assert score["tier"] == "Insufficient"
