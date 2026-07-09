import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.price import PriceData


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
    return insider, company


def test_create_transaction(db, seed):
    insider, company = seed
    txn = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number="0001-24-000001",
        filing_date=date(2024, 1, 15),
        transaction_date=date(2024, 1, 14),
        transaction_code="P",
        shares=1000.0,
        price_per_share=50.0,
        total_value=50000.0,
        acquired_or_disposed="A",
        is_discretionary=True,
    )
    db.add(txn)
    db.commit()
    assert txn.id is not None
    assert insider.transactions[0].shares == 1000.0


def test_accession_unique(db, seed):
    insider, company = seed
    txn1 = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number="ACC-UNIQUE",
        filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1),
        transaction_code="P",
        shares=100,
        acquired_or_disposed="A",
    )
    db.add(txn1)
    db.commit()
    txn2 = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number="ACC-UNIQUE",
        filing_date=date(2024, 1, 2),
        transaction_date=date(2024, 1, 2),
        transaction_code="P",
        shares=200,
        acquired_or_disposed="A",
    )
    db.add(txn2)
    with pytest.raises(Exception):
        db.commit()


def test_price_data(db):
    pd = PriceData(
        ticker="AAPL",
        date=date(2024, 1, 15),
        open=185.0,
        high=186.0,
        low=184.0,
        close=185.5,
        adj_close=185.5,
        volume=50000000,
    )
    db.add(pd)
    db.commit()
    assert pd.id is not None


def test_price_data_unique_ticker_date(db):
    pd1 = PriceData(
        ticker="AAPL",
        date=date(2024, 1, 15),
        open=185.0,
        high=186.0,
        low=184.0,
        close=185.5,
        adj_close=185.5,
        volume=50000000,
    )
    db.add(pd1)
    db.commit()
    pd2 = PriceData(
        ticker="AAPL",
        date=date(2024, 1, 15),
        open=186.0,
        high=187.0,
        low=185.0,
        close=186.5,
        adj_close=186.5,
        volume=60000000,
    )
    db.add(pd2)
    with pytest.raises(Exception):
        db.commit()
