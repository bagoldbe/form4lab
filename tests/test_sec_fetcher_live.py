"""Live integration tests against SEC EDGAR APIs.

Run with: pytest tests/test_sec_fetcher_live.py -v -m live
These tests hit real SEC endpoints and are slow.
"""
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

pytestmark = pytest.mark.live


class TestLiveSecApi:
    def test_resolve_cik_apple(self):
        from form4lab.data.sec_fetcher import resolve_cik, _cik_cache
        _cik_cache.clear()
        cik = resolve_cik("AAPL")
        assert cik is not None
        assert cik == "0000320193"

    def test_fetch_submissions_apple(self):
        from form4lab.data.sec_fetcher import fetch_submissions
        filings = fetch_submissions("0000320193", cutoff_date=date(2024, 1, 1))
        assert len(filings) > 0
        assert all(f["form"] == "4" for f in filings)
        assert all("accession_number" in f for f in filings)

    def test_fetch_and_parse_form4_produces_valid_output(self):
        from form4lab.data.sec_fetcher import fetch_submissions, fetch_and_parse_form4
        # Get a real Form 4 filing for Apple
        filings = fetch_submissions("0000320193", cutoff_date=date(2024, 6, 1))
        assert len(filings) > 0

        filing = filings[0]
        txns = fetch_and_parse_form4(
            "0000320193",
            filing["accession_number"],
            filing["primary_document"],
            filing["filing_date"],
        )

        # Verify output matches expected dict format
        if txns:  # some filings may have no non-derivative transactions
            txn = txns[0]
            assert "insider_cik" in txn
            assert "company_cik" in txn
            assert "transaction_code" in txn
            assert "shares" in txn
            assert "is_discretionary" in txn
            assert "accession_number" in txn

    def test_end_to_end_backfill_one_ticker(self):
        """Backfill a single ticker (1 year only to keep it fast) and verify data integrity."""
        from form4lab.data.sec_fetcher import backfill_company_fast, _cik_cache
        from form4lab.database import Base
        from form4lab.models.transaction import Transaction
        from form4lab.models.insider import Insider
        from form4lab.models.company import Company

        # Use a fresh in-memory SQLite database to avoid conflicts with
        # existing data in the default form4lab.db file.
        test_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)

        _cik_cache.clear()
        with TestSession() as db:
            count = backfill_company_fast("AAPL", years=1, db=db)
            assert count > 0

            # Verify company was created
            company = db.query(Company).filter(Company.ticker == "AAPL").first()
            assert company is not None

            # Verify insiders were created
            insider_count = db.query(Insider).count()
            assert insider_count > 0

            # Verify transactions exist
            txn_count = db.query(Transaction).filter(
                Transaction.company_id == company.id
            ).count()
            assert txn_count > 0
