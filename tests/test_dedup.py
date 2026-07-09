"""Tests for transaction deduplication utilities."""
import pytest
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.insider import Insider
from form4lab.models.company import Company
from form4lab.models.transaction import Transaction
from form4lab.scoring.dedup import dedup_transactions, dedup_outcome_tuples


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_insider(db, name="Test Insider"):
    insider = Insider(cik=f"CIK-{name}", name=name)
    db.add(insider)
    db.flush()
    return insider


def _make_company(db, ticker="TST", name="Test Corp"):
    company = Company(cik=f"CIK-{ticker}", ticker=ticker, name=name)
    db.add(company)
    db.flush()
    return company


def _create_txns(db, insider, company, dates_and_values, acc_prefix="ACC"):
    """Helper to create Transaction objects in db."""
    txns = []
    for i, (txn_date, total_value) in enumerate(dates_and_values):
        txn = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number=f"{acc_prefix}-{i}",
            filing_date=txn_date, transaction_date=txn_date,
            transaction_code="P", shares=100,
            price_per_share=total_value / 100,
            total_value=total_value,
            acquired_or_disposed="A", is_discretionary=True,
        )
        db.add(txn)
        txns.append(txn)
    db.flush()
    return txns


class TestDedupTransactions:
    def test_no_duplicates(self, db):
        """Transactions on different dates should all be kept."""
        insider = _make_insider(db)
        company = _make_company(db)
        txns = _create_txns(db, insider, company, [
            (date(2024, 1, 1), 10000),
            (date(2024, 2, 1), 20000),
            (date(2024, 3, 1), 30000),
        ])
        result = dedup_transactions(txns)
        assert len(result) == 3

    def test_same_day_lots_collapsed(self, db):
        """Multiple transactions on the same day should collapse to one."""
        insider = _make_insider(db)
        company = _make_company(db)
        txns = _create_txns(db, insider, company, [
            (date(2024, 1, 1), 10000),
            (date(2024, 1, 1), 20000),
            (date(2024, 1, 1), 5000),
        ])
        result = dedup_transactions(txns)
        assert len(result) == 1
        assert result[0].total_value == 20000
        assert result[0].event_total_value == 35000

    def test_mixed_dates(self, db):
        """Mix of same-day and different-day transactions."""
        insider = _make_insider(db)
        company = _make_company(db)
        txns = _create_txns(db, insider, company, [
            (date(2024, 1, 1), 10000),
            (date(2024, 1, 1), 20000),
            (date(2024, 2, 1), 15000),
            (date(2024, 3, 1), 5000),
            (date(2024, 3, 1), 8000),
        ])
        result = dedup_transactions(txns)
        assert len(result) == 3

    def test_different_insiders_same_date(self, db):
        """Same-day txns from different insiders should NOT be collapsed."""
        ins1 = _make_insider(db, "Insider A")
        ins2 = _make_insider(db, "Insider B")
        company = _make_company(db)
        txns1 = _create_txns(db, ins1, company, [(date(2024, 1, 1), 10000)], "A")
        txns2 = _create_txns(db, ins2, company, [(date(2024, 1, 1), 20000)], "B")
        result = dedup_transactions(txns1 + txns2)
        assert len(result) == 2

    def test_different_companies_same_date(self, db):
        """Same-day txns at different companies should NOT be collapsed."""
        insider = _make_insider(db)
        co1 = _make_company(db, "AAA", "Company A")
        co2 = _make_company(db, "BBB", "Company B")
        txns1 = _create_txns(db, insider, co1, [(date(2024, 1, 1), 10000)], "C1")
        txns2 = _create_txns(db, insider, co2, [(date(2024, 1, 1), 20000)], "C2")
        result = dedup_transactions(txns1 + txns2)
        assert len(result) == 2

    def test_empty_list(self, db):
        result = dedup_transactions([])
        assert result == []

    def test_event_total_value_single_lot(self, db):
        """Single lot should have event_total_value == total_value."""
        insider = _make_insider(db)
        company = _make_company(db)
        txns = _create_txns(db, insider, company, [(date(2024, 1, 1), 10000)])
        result = dedup_transactions(txns)
        assert len(result) == 1
        assert result[0].event_total_value == 10000

    def test_cohen_style_12_lots(self, db):
        """Simulate Cohen's 12 lots on the same day."""
        insider = _make_insider(db, "Ryan Cohen")
        company = _make_company(db, "GME", "GameStop")
        lots = [(date(2022, 3, 22), 1_000_000 + i * 100_000) for i in range(12)]
        txns = _create_txns(db, insider, company, lots)
        result = dedup_transactions(txns)
        assert len(result) == 1
        expected_sum = sum(v for _, v in lots)
        assert result[0].event_total_value == expected_sum


class TestDedupOutcomeTuples:
    def test_no_duplicates(self):
        tuples = [
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 2, 1), False, -0.03, -0.02),
        ]
        result = dedup_outcome_tuples(tuples)
        assert len(result) == 2

    def test_same_date_collapsed(self):
        tuples = [
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 1, 1), True, 0.05, 0.01),
        ]
        result = dedup_outcome_tuples(tuples)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        tuples = [
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 1, 1), False, -0.03, -0.02),
        ]
        result = dedup_outcome_tuples(tuples)
        assert len(result) == 1
        assert result[0] == (date(2024, 1, 1), True, 0.05, 0.01)

    def test_empty_list(self):
        assert dedup_outcome_tuples([]) == []

    def test_mixed(self):
        tuples = [
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 1, 1), True, 0.05, 0.01),
            (date(2024, 2, 1), False, -0.03, -0.02),
            (date(2024, 3, 1), True, 0.08, 0.03),
            (date(2024, 3, 1), True, 0.08, 0.03),
        ]
        result = dedup_outcome_tuples(tuples)
        assert len(result) == 3
