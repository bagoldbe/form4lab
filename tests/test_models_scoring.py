import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.score import InsiderScore
from form4lab.models.alert import Alert


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def seed(db):
    insider = Insider(cik="001", name="Jane")
    company = Company(cik="002", ticker="TEST", name="Test Inc")
    db.add_all([insider, company])
    db.flush()
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-001", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="P",
        shares=100, acquired_or_disposed="A", is_discretionary=True
    )
    db.add(txn)
    db.flush()
    return insider, company, txn


def test_create_trade_outcome(db, seed):
    _, _, txn = seed
    outcome = TradeOutcome(transaction_id=txn.id, stock_return_60d=0.05, hit_60d=True)
    db.add(outcome)
    db.commit()
    assert outcome.id is not None
    assert txn.outcome.hit_60d is True


def test_trade_outcome_unique_transaction(db, seed):
    _, _, txn = seed
    db.add(TradeOutcome(transaction_id=txn.id))
    db.commit()
    db.add(TradeOutcome(transaction_id=txn.id))
    with pytest.raises(Exception):
        db.commit()


def test_create_insider_score(db, seed):
    insider, company, _ = seed
    score = InsiderScore(
        insider_id=insider.id, company_id=company.id,
        num_discretionary_buys=5, skill_score=1.5, credibility_tier="Elite"
    )
    db.add(score)
    db.commit()
    assert score.id is not None
    assert insider.scores[0].credibility_tier == "Elite"


def test_insider_score_unique_constraint(db, seed):
    insider, company, _ = seed
    db.add(InsiderScore(insider_id=insider.id, company_id=company.id))
    db.commit()
    db.add(InsiderScore(insider_id=insider.id, company_id=company.id))
    with pytest.raises(Exception):
        db.commit()


def test_create_alert(db, seed):
    insider, company, txn = seed
    alert = Alert(
        transaction_id=txn.id, insider_id=insider.id, company_id=company.id,
        alert_type="elite_buy", conviction_score=2.5, insider_skill_score=1.8,
        transaction_value=150000.0, summary="ELITE: Jane bought $150K"
    )
    db.add(alert)
    db.commit()
    assert alert.id is not None
