import pytest
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.scoring.outcome_calculator import (
    compute_forward_return,
    compute_excess_return,
    _get_price_at_date,
    batch_compute_outcomes,
    compute_trade_outcomes,
)


def test_forward_return_positive():
    result = compute_forward_return(100.0, 110.0)
    assert abs(result - 0.10) < 0.001


def test_forward_return_negative():
    result = compute_forward_return(100.0, 90.0)
    assert abs(result - (-0.10)) < 0.001


def test_forward_return_zero_start():
    result = compute_forward_return(0.0, 100.0)
    assert result == 0.0


def test_excess_return():
    assert abs(compute_excess_return(0.10, 0.05) - 0.05) < 0.001


def test_excess_return_negative():
    """Excess return should be negative when stock underperforms benchmark."""
    assert abs(compute_excess_return(0.03, 0.08) - (-0.05)) < 0.001


def test_get_price_at_date():
    df = pd.DataFrame([
        {"date": date(2024, 1, 2), "close": 100.0, "adj_close": 100.0},
        {"date": date(2024, 1, 3), "close": 101.0, "adj_close": 101.0},
    ])
    assert _get_price_at_date(df, date(2024, 1, 2)) == 100.0
    assert _get_price_at_date(df, date(2024, 1, 1)) == 100.0  # finds next available
    assert _get_price_at_date(df, date(2024, 2, 1)) is None  # too far away


def test_get_price_at_date_empty():
    df = pd.DataFrame(columns=["date", "close", "adj_close"])
    assert _get_price_at_date(df, date(2024, 1, 1)) is None


def test_get_price_at_date_exact_match():
    """When the exact date exists, it should be returned."""
    df = pd.DataFrame([
        {"date": date(2024, 3, 15), "close": 150.0, "adj_close": 150.0},
    ])
    assert _get_price_at_date(df, date(2024, 3, 15)) == 150.0


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_compute_trade_outcomes_with_mock_provider(db):
    """Test full outcome computation with mock price data."""
    # Create entities
    insider = Insider(cik="001", name="Test")
    company = Company(cik="002", ticker="TEST", name="Test Inc", sector="Technology")
    db.add_all([insider, company])
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-001", filing_date=date(2024, 1, 15),
        transaction_date=date(2024, 1, 15), transaction_code="P",
        shares=100, price_per_share=100.0, total_value=10000.0,
        acquired_or_disposed="A", is_discretionary=True
    )
    db.add(txn)
    db.flush()

    # Create mock price provider with known prices
    mock_provider = MagicMock()

    # Generate price data covering the full range
    dates = pd.date_range(start="2023-12-01", end="2024-09-01", freq="B")
    base_prices = pd.DataFrame([
        {"date": d.date(), "open": 100, "high": 102, "low": 99,
         "close": 100 + (i * 0.1), "adj_close": 100 + (i * 0.1), "volume": 1000000}
        for i, d in enumerate(dates)
    ])

    spy_prices = pd.DataFrame([
        {"date": d.date(), "open": 400, "high": 402, "low": 399,
         "close": 400 + (i * 0.05), "adj_close": 400 + (i * 0.05), "volume": 5000000}
        for i, d in enumerate(dates)
    ])

    def get_prices(ticker, start, end):
        if ticker == "SPY":
            return spy_prices[(spy_prices["date"] >= start) & (spy_prices["date"] <= end)]
        elif ticker == "XLK":
            return spy_prices[(spy_prices["date"] >= start) & (spy_prices["date"] <= end)]  # reuse
        return base_prices[(base_prices["date"] >= start) & (base_prices["date"] <= end)]

    mock_provider.get_daily_prices.side_effect = get_prices
    mock_provider.get_sector_etf.return_value = "XLK"

    outcome = compute_trade_outcomes(txn.id, db, mock_provider)
    assert outcome is not None
    assert outcome.stock_return_20d is not None
    assert outcome.benchmark_return_20d is not None
    assert outcome.excess_return_20d is not None
    assert outcome.hit_20d is not None


def test_compute_trade_outcomes_non_discretionary(db):
    """Non-discretionary transactions should return None."""
    insider = Insider(cik="010", name="NonDisc")
    company = Company(cik="020", ticker="ND", name="NonDisc Inc")
    db.add_all([insider, company])
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-ND-001", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="A",
        shares=100, acquired_or_disposed="A", is_discretionary=False
    )
    db.add(txn)
    db.flush()

    mock_provider = MagicMock()
    result = compute_trade_outcomes(txn.id, db, mock_provider)
    assert result is None


def test_compute_trade_outcomes_no_ticker(db):
    """Company with no ticker should return None."""
    insider = Insider(cik="030", name="NoTicker")
    company = Company(cik="040", ticker=None, name="NoTicker Inc")
    db.add_all([insider, company])
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-NT-001", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="P",
        shares=100, acquired_or_disposed="A", is_discretionary=True
    )
    db.add(txn)
    db.flush()

    mock_provider = MagicMock()
    result = compute_trade_outcomes(txn.id, db, mock_provider)
    assert result is None


def test_batch_compute_outcomes(db):
    """Test that batch processing finds pending transactions."""
    insider = Insider(cik="001", name="Test")
    company = Company(cik="002", ticker="TEST", name="Test Inc")
    db.add_all([insider, company])
    db.flush()

    # Non-discretionary -- should be skipped
    txn1 = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-001", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="A",
        shares=100, acquired_or_disposed="A", is_discretionary=False
    )
    db.add(txn1)
    db.flush()

    mock_provider = MagicMock()
    mock_provider.get_daily_prices.return_value = pd.DataFrame(columns=["date", "close", "adj_close"])

    count = batch_compute_outcomes(db, mock_provider)
    # Non-discretionary should not be processed
    assert count == 0


def test_batch_compute_skips_already_computed(db):
    """Transactions that already have outcomes should be skipped."""
    insider = Insider(cik="050", name="Already")
    company = Company(cik="060", ticker="DONE", name="Done Inc")
    db.add_all([insider, company])
    db.flush()

    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-DONE-001", filing_date=date(2024, 1, 1),
        transaction_date=date(2024, 1, 1), transaction_code="P",
        shares=100, acquired_or_disposed="A", is_discretionary=True
    )
    db.add(txn)
    db.flush()

    # Pre-create an outcome for this transaction
    outcome = TradeOutcome(transaction_id=txn.id, stock_return_60d=0.05)
    db.add(outcome)
    db.flush()

    mock_provider = MagicMock()
    count = batch_compute_outcomes(db, mock_provider)
    assert count == 0
    # The mock provider should not have been called for this transaction
    mock_provider.get_daily_prices.assert_not_called()
