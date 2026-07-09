"""Tests for form4lab.data.sec_fetcher: RateLimiter, resolve_cik, fetch_submissions, parse_form4_xml, and backfill_company_fast."""

import time
import threading
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from form4lab.data.sec_fetcher import (
    RateLimiter,
    resolve_cik,
    _cik_cache,
    fetch_submissions,
    _extract_form4_filings,
    parse_form4_xml,
    backfill_company_fast,
    ingest_daily_filings,
)
from form4lab.database import Base
from form4lab.models.transaction import Transaction
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company

SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": "789019", "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc."},
}


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_enforces_minimum_interval(self):
        """5 calls at 10/s should take at least 0.35s (4 intervals of 0.1s each)."""
        limiter = RateLimiter(max_per_second=10.0)
        start = time.monotonic()
        for _ in range(5):
            limiter.wait()
        elapsed = time.monotonic() - start
        # 4 gaps of 0.1s = 0.4s minimum, but allow small tolerance
        assert elapsed >= 0.35, f"Expected >= 0.35s, got {elapsed:.3f}s"

    def test_thread_safety(self):
        """5 threaded calls should maintain minimum gaps of >= 0.08s between each."""
        limiter = RateLimiter(max_per_second=10.0)
        timestamps = []
        lock = threading.Lock()

        def record_time():
            limiter.wait()
            t = time.monotonic()
            with lock:
                timestamps.append(t)

        threads = [threading.Thread(target=record_time) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        timestamps.sort()
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for i, gap in enumerate(gaps):
            assert gap >= 0.08, f"Gap {i} was {gap:.4f}s, expected >= 0.08s"


class TestResolveCik:
    """Tests for the resolve_cik function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the CIK cache before each test."""
        _cik_cache.clear()
        yield
        _cik_cache.clear()

    @patch("form4lab.data.sec_fetcher._fetch_company_tickers")
    def test_resolves_known_ticker(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_COMPANY_TICKERS
        result = resolve_cik("AAPL")
        assert result == "0000320193"

    @patch("form4lab.data.sec_fetcher._fetch_company_tickers")
    def test_resolves_case_insensitive(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_COMPANY_TICKERS
        result = resolve_cik("aapl")
        assert result == "0000320193"

    @patch("form4lab.data.sec_fetcher._fetch_company_tickers")
    def test_returns_none_for_unknown(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_COMPANY_TICKERS
        result = resolve_cik("ZZZNOTREAL")
        assert result is None

    @patch("form4lab.data.sec_fetcher._fetch_company_tickers")
    def test_caches_after_first_call(self, mock_fetch):
        """_fetch_company_tickers should only be called once even with multiple resolve_cik calls."""
        mock_fetch.return_value = SAMPLE_COMPANY_TICKERS

        resolve_cik("AAPL")
        resolve_cik("MSFT")
        resolve_cik("GOOGL")

        mock_fetch.assert_called_once()

    @patch("form4lab.data.sec_fetcher._fetch_company_tickers")
    def test_zero_pads_cik_to_10_digits(self, mock_fetch):
        """CIK strings should be zero-padded to 10 digits."""
        mock_fetch.return_value = SAMPLE_COMPANY_TICKERS

        assert resolve_cik("AAPL") == "0000320193"
        assert resolve_cik("MSFT") == "0000789019"
        assert resolve_cik("GOOGL") == "0001652044"


SAMPLE_SUBMISSIONS = {
    "cik": "320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000100",
                "0000320193-23-000050",
            ],
            "filingDate": ["2024-06-15", "2024-03-10", "2023-01-05"],
            "form": ["4", "10-K", "4"],
            "primaryDocument": [
                "xslForm4X01/primary_doc.xml",
                "aapl-20240310.htm",
                "xslForm4X01/primary_doc.xml",
            ],
        },
        "files": [],
    },
}


class TestFetchSubmissions:
    """Tests for fetch_submissions and _extract_form4_filings."""

    @patch("form4lab.data.sec_fetcher._sec_get")
    def test_returns_only_form4_filings(self, mock_sec_get):
        """Should return only Form 4 filings, filtering out 10-K and others."""
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SUBMISSIONS
        mock_sec_get.return_value = mock_response

        results = fetch_submissions("0000320193")

        assert len(results) == 2
        for filing in results:
            assert filing["form"] == "4"

    @patch("form4lab.data.sec_fetcher._sec_get")
    def test_filters_by_cutoff_date(self, mock_sec_get):
        """With a cutoff of Jan 1 2024, only the mid-2024 filing should be returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SUBMISSIONS
        mock_sec_get.return_value = mock_response

        results = fetch_submissions("0000320193", cutoff_date=date(2024, 1, 1))

        assert len(results) == 1
        assert results[0]["filing_date"] == "2024-06-15"

    @patch("form4lab.data.sec_fetcher._sec_get")
    def test_returns_correct_fields(self, mock_sec_get):
        """Each returned dict should have the expected keys."""
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SUBMISSIONS
        mock_sec_get.return_value = mock_response

        results = fetch_submissions("0000320193")

        expected_keys = {"accession_number", "filing_date", "primary_document", "form"}
        for filing in results:
            assert set(filing.keys()) == expected_keys

    @patch("form4lab.data.sec_fetcher._sec_get")
    def test_handles_pagination_files(self, mock_sec_get):
        """Should fetch paginated filing files and combine results."""
        # Main submission response with a pagination file reference
        main_submissions = {
            "cik": "320193",
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-24-000123"],
                    "filingDate": ["2024-06-15"],
                    "form": ["4"],
                    "primaryDocument": ["xslForm4X01/primary_doc.xml"],
                },
                "files": [{"name": "CIK0000320193-submissions-001.json"}],
            },
        }

        # Paginated file response with one additional Form 4
        paginated_filings = {
            "accessionNumber": ["0000320193-22-000999"],
            "filingDate": ["2022-11-20"],
            "form": ["4"],
            "primaryDocument": ["xslForm4X01/old_doc.xml"],
        }

        mock_main_response = MagicMock()
        mock_main_response.json.return_value = main_submissions

        mock_paginated_response = MagicMock()
        mock_paginated_response.json.return_value = paginated_filings

        mock_sec_get.side_effect = [mock_main_response, mock_paginated_response]

        results = fetch_submissions("0000320193")

        assert len(results) == 2
        assert mock_sec_get.call_count == 2
        # Verify the second call fetched the paginated file
        second_call_url = mock_sec_get.call_args_list[1][0][0]
        assert "CIK0000320193-submissions-001.json" in second_call_url


class TestParseForm4Xml:
    """Tests for the parse_form4_xml function."""

    @pytest.fixture()
    def form4_xml(self) -> str:
        return Path("tests/fixtures/sample_form4.xml").read_text()

    @pytest.fixture()
    def parsed(self, form4_xml) -> list[dict]:
        return parse_form4_xml(form4_xml, "0000320193-24-000123", "2024-01-16")

    def test_extracts_issuer_info(self, parsed):
        """Should extract company CIK (leading zeros stripped), name, and ticker."""
        txn = parsed[0]
        assert txn["company_cik"] == "320193"
        assert txn["company_name"] == "Apple Inc."
        assert txn["company_ticker"] == "AAPL"

    def test_extracts_owner_info(self, parsed):
        """Should extract insider CIK (leading zeros stripped), name, and role flags."""
        txn = parsed[0]
        assert txn["insider_cik"] == "1234567"
        assert txn["insider_name"] == "DOE JOHN"
        assert txn["is_officer"] is True
        assert txn["is_director"] is False
        assert txn["officer_title"] == "SVP Engineering"

    def test_extracts_transaction_fields(self, parsed):
        """P transaction should have correct shares, price, total_value, shares_owned_after, and acq/disp."""
        p_txns = [t for t in parsed if t["transaction_code"] == "P"]
        assert len(p_txns) == 1
        txn = p_txns[0]
        assert txn["shares"] == 1000.0
        assert txn["price_per_share"] == 185.50
        assert txn["total_value"] == 185500.0
        assert txn["shares_owned_after"] == 5000.0
        assert txn["acquired_or_disposed"] == "A"

    def test_discretionary_flag(self, parsed):
        """P is discretionary, M is not."""
        p_txns = [t for t in parsed if t["transaction_code"] == "P"]
        m_txns = [t for t in parsed if t["transaction_code"] == "M"]
        assert p_txns[0]["is_discretionary"] is True
        assert m_txns[0]["is_discretionary"] is False

    def test_accession_number_format(self, parsed):
        """P transaction accession_number should be '{accession}_{code}_{date}_{shares}'."""
        p_txns = [t for t in parsed if t["transaction_code"] == "P"]
        assert p_txns[0]["accession_number"] == "0000320193-24-000123_P_2024-01-15_1000.0"

    def test_filing_date_parsed(self, parsed):
        """filing_date should be parsed as a date object."""
        txn = parsed[0]
        assert txn["filing_date"] == date(2024, 1, 16)


class TestBackfillCompanyFast:
    """Tests for the backfill_company_fast function with an in-memory SQLite database."""

    @pytest.fixture()
    def db_session(self):
        """Create an in-memory SQLite database and return a session."""
        test_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)
        session = TestSession()
        yield session
        session.close()

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    @patch("form4lab.data.sec_fetcher.resolve_cik")
    def test_inserts_transaction(self, mock_resolve, mock_submissions, mock_parse, db_session):
        """backfill_company_fast should persist a transaction to the database."""
        mock_resolve.return_value = "0000320193"
        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        count = backfill_company_fast("AAPL", 10, db_session)

        assert count == 1
        txn = db_session.query(Transaction).first()
        assert txn is not None
        assert txn.accession_number == "0000320193-24-000123_P_2024-01-15_1000.0"
        assert txn.shares == 1000.0
        assert txn.transaction_code == "P"
        assert txn.is_discretionary is True

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    @patch("form4lab.data.sec_fetcher.resolve_cik")
    def test_skips_duplicate_accession_numbers(self, mock_resolve, mock_submissions, mock_parse, db_session):
        """Calling backfill_company_fast twice should not duplicate transactions."""
        mock_resolve.return_value = "0000320193"
        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        count1 = backfill_company_fast("AAPL", 10, db_session)
        count2 = backfill_company_fast("AAPL", 10, db_session)

        assert count1 == 1
        assert count2 == 0
        total = db_session.query(Transaction).count()
        assert total == 1

    @patch("form4lab.data.sec_fetcher.resolve_cik")
    def test_returns_zero_for_unknown_ticker(self, mock_resolve, db_session):
        """Should return 0 when ticker cannot be resolved to a CIK."""
        mock_resolve.return_value = None

        count = backfill_company_fast("ZZZNOTREAL", 10, db_session)

        assert count == 0

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    @patch("form4lab.data.sec_fetcher.resolve_cik")
    def test_creates_insider_and_company(self, mock_resolve, mock_submissions, mock_parse, db_session):
        """Should create Insider, Company, and InsiderRole records."""
        mock_resolve.return_value = "0000320193"
        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        backfill_company_fast("AAPL", 10, db_session)

        insider = db_session.query(Insider).filter(Insider.cik == "1234567").first()
        assert insider is not None
        assert insider.name == "DOE JOHN"

        company = db_session.query(Company).filter(Company.cik == "320193").first()
        assert company is not None
        assert company.ticker == "AAPL"

        role = db_session.query(InsiderRole).filter(
            InsiderRole.insider_id == insider.id,
            InsiderRole.company_id == company.id,
        ).first()
        assert role is not None
        assert role.is_officer is True


class TestIngestDailyFilings:
    """Tests for the ingest_daily_filings function."""

    @pytest.fixture()
    def db_session(self):
        """Create an in-memory SQLite database and return a session."""
        test_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)
        session = TestSession()
        yield session
        session.close()

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    def test_ingests_filings_for_tracked_companies(self, mock_submissions, mock_parse, db_session):
        """Should find tracked companies, fetch filings, and persist transactions."""
        # Pre-populate a Company record
        company = Company(cik="320193", ticker="AAPL", name="Apple Inc.")
        db_session.add(company)
        db_session.commit()

        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        count = ingest_daily_filings(db_session, days_back=1)

        assert count == 1
        txn = db_session.query(Transaction).first()
        assert txn is not None
        assert txn.accession_number == "0000320193-24-000123_P_2024-01-15_1000.0"
        assert txn.shares == 1000.0
        assert txn.transaction_code == "P"

        # Verify fetch_submissions was called with the zero-padded CIK
        mock_submissions.assert_called_once()
        call_args = mock_submissions.call_args
        assert call_args[0][0] == "0000320193"

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    def test_skips_existing_transactions(self, mock_submissions, mock_parse, db_session):
        """Should not duplicate transactions that already exist."""
        company = Company(cik="320193", ticker="AAPL", name="Apple Inc.")
        db_session.add(company)
        db_session.commit()

        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        count1 = ingest_daily_filings(db_session, days_back=1)
        count2 = ingest_daily_filings(db_session, days_back=1)

        assert count1 == 1
        assert count2 == 0
        total = db_session.query(Transaction).count()
        assert total == 1

    def test_returns_zero_when_no_companies(self, db_session):
        """Should return 0 when no companies are tracked."""
        count = ingest_daily_filings(db_session, days_back=1)
        assert count == 0

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    def test_skips_implausible_ciks(self, mock_submissions, mock_parse, db_session):
        """Fixture-like CIKs (> 10M, e.g. uuid-generated test rows) must not
        trigger SEC requests — they 404 and waste the 10 req/s budget."""
        db_session.add_all([
            Company(cik="320193", ticker="AAPL", name="Apple Inc."),
            Company(cik="2875515598", ticker="TST253", name="Alert Svc Test Corp"),
        ])
        db_session.commit()
        mock_submissions.return_value = []

        ingest_daily_filings(db_session, days_back=1)

        mock_submissions.assert_called_once()
        assert mock_submissions.call_args[0][0] == "0000320193"

    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    def test_continues_on_submission_failure(self, mock_submissions, db_session):
        """Should continue processing other companies when one fails."""
        company1 = Company(cik="320193", ticker="AAPL", name="Apple Inc.")
        company2 = Company(cik="789019", ticker="MSFT", name="Microsoft Corp")
        db_session.add_all([company1, company2])
        db_session.commit()

        # First CIK raises, second returns empty
        mock_submissions.side_effect = [
            Exception("Network error"),
            [],
        ]

        count = ingest_daily_filings(db_session, days_back=1)

        assert count == 0
        assert mock_submissions.call_count == 2

    @patch("form4lab.data.sec_fetcher.fetch_and_parse_form4")
    @patch("form4lab.data.sec_fetcher.fetch_submissions")
    def test_creates_insider_and_company_records(self, mock_submissions, mock_parse, db_session):
        """Should create Insider, Company, and InsiderRole records for new transactions."""
        company = Company(cik="320193", ticker="AAPL", name="Apple Inc.")
        db_session.add(company)
        db_session.commit()

        mock_submissions.return_value = [
            {
                "accession_number": "0000320193-24-000123",
                "filing_date": "2024-01-16",
                "primary_document": "xslForm4X01/primary_doc.xml",
                "form": "4",
            }
        ]
        mock_parse.return_value = [
            {
                "insider_cik": "1234567",
                "insider_name": "DOE JOHN",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "officer_title": "SVP Engineering",
                "company_cik": "320193",
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "accession_number": "0000320193-24-000123_P_2024-01-15_1000.0",
                "filing_date": date(2024, 1, 16),
                "transaction_date": date(2024, 1, 15),
                "transaction_code": "P",
                "shares": 1000.0,
                "price_per_share": 185.50,
                "total_value": 185500.0,
                "shares_owned_after": 5000.0,
                "acquired_or_disposed": "A",
                "is_discretionary": True,
            }
        ]

        ingest_daily_filings(db_session, days_back=1)

        insider = db_session.query(Insider).filter(Insider.cik == "1234567").first()
        assert insider is not None
        assert insider.name == "DOE JOHN"

        role = db_session.query(InsiderRole).filter(
            InsiderRole.insider_id == insider.id,
        ).first()
        assert role is not None
        assert role.is_officer is True


# ---------------------------------------------------------------------------
# CIK plausibility guard
# ---------------------------------------------------------------------------

class TestIsPlausibleCik:
    """is_plausible_cik keeps fixture rows (should any ever leak into the DB
    again) from triggering doomed SEC requests."""

    def test_real_ciks_pass(self):
        from form4lab.data.utils import is_plausible_cik
        for cik in ("320193", "0000320193", "1318605", "2200000", "9876543", "10000000"):
            assert is_plausible_cik(cik) is True, cik

    def test_fixture_ciks_fail(self):
        from form4lab.data.utils import is_plausible_cik
        # uuid-generated fixture CIKs are numerically > 10,000,000
        for cik in ("10000001", "2875515598", "9999999999"):
            assert is_plausible_cik(cik) is False, cik

    def test_non_numeric_fail(self):
        from form4lab.data.utils import is_plausible_cik
        for cik in (None, "", "   ", "abc", "12a4", "12.3", "-5", "²34"):
            assert is_plausible_cik(cik) is False, repr(cik)


def test_poll_recent_filings_skips_implausible_tracked_ciks():
    """The EFTS poll loop must also skip fixture-like CIKs before fetching."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch
    from form4lab.data.sec_fetcher import poll_recent_filings

    hits = [{"_source": {"ciks": ["0000320193", "2875515598"]}}]
    response = MagicMock(json=lambda: {"hits": {"hits": hits}})

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [
        ("320193",), ("2875515598",),
    ]

    with patch("form4lab.data.sec_fetcher._sec_get", return_value=response), \
         patch("form4lab.data.sec_fetcher.fetch_submissions", return_value=[]) as mock_subs:
        poll_recent_filings(mock_db, datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc))

    mock_subs.assert_called_once()
    assert mock_subs.call_args[0][0] == "0000320193"


# ---------------------------------------------------------------------------
# EFTS polling
# ---------------------------------------------------------------------------

def test_build_efts_url():
    """Verify the EFTS URL is constructed correctly."""
    from form4lab.data.sec_fetcher import _build_efts_url
    from datetime import datetime

    since = datetime(2026, 2, 21, 14, 30, 0)
    url = _build_efts_url(since)
    assert "efts.sec.gov" in url
    assert "forms=4" in url
    assert "2026-02-21" in url


def test_build_efts_url_includes_pagination_offset():
    """URL builder must accept a `from_offset` so pages after the first can be requested.

    SEC EFTS caps each response at 100 hits. On busy days (>100 Form 4s) we must
    paginate via the `from` query parameter, or filings past rank 100 are lost.
    """
    from form4lab.data.sec_fetcher import _build_efts_url
    from datetime import datetime

    since = datetime(2026, 3, 13, 14, 30, 0)
    url = _build_efts_url(since, from_offset=200)
    assert "from=200" in url


def test_poll_recent_filings_paginates_all_hits():
    """poll_recent_filings must iterate EFTS pages, not stop at the first 100.

    Regression test: a busy filing day can exceed 800 Form 4 filings.
    Without pagination we only saw the top 100, losing the rest.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch
    from form4lab.data.sec_fetcher import poll_recent_filings

    def hit(cik):
        return {"_source": {"ciks": [cik]}}

    # Simulate 3 pages of EFTS results: full page, full page, partial (terminator)
    page1_hits = [hit(f"{i:010d}") for i in range(1, 101)]
    page2_hits = [hit(f"{i:010d}") for i in range(101, 201)]
    page3_hits = [hit(f"{i:010d}") for i in range(201, 251)]

    page_responses = [
        MagicMock(json=lambda: {"hits": {"hits": page1_hits}}),
        MagicMock(json=lambda: {"hits": {"hits": page2_hits}}),
        MagicMock(json=lambda: {"hits": {"hits": page3_hits}}),
    ]

    call_urls: list[str] = []

    def fake_sec_get(url):
        call_urls.append(url)
        return page_responses[len(call_urls) - 1]

    mock_db = MagicMock()
    # No tracked CIKs → no fetches, but the paginate loop must still run
    mock_db.query.return_value.filter.return_value.all.return_value = []

    with patch("form4lab.data.sec_fetcher._sec_get", side_effect=fake_sec_get):
        poll_recent_filings(mock_db, datetime(2026, 3, 13, 14, 30, tzinfo=timezone.utc))

    # Must have paginated: 3 calls with from=0, from=100, from=200
    assert len(call_urls) == 3, f"Expected 3 paginated calls, got {len(call_urls)}: {call_urls}"
    assert "from=100" in call_urls[1]
    assert "from=200" in call_urls[2]


def test_poll_recent_filings_terminates_on_exact_page_size():
    """Pagination must terminate when a page returns exactly EFTS_PAGE_SIZE hits
    and the next page is empty. Guards against infinite-loop off-by-one.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch
    from form4lab.data.sec_fetcher import poll_recent_filings, EFTS_PAGE_SIZE

    page1 = [{"_source": {"ciks": [f"{i:010d}"]}} for i in range(EFTS_PAGE_SIZE)]
    page2: list = []  # empty — acts as terminator
    responses = [
        MagicMock(json=lambda: {"hits": {"hits": page1}}),
        MagicMock(json=lambda: {"hits": {"hits": page2}}),
    ]
    calls: list[str] = []

    def fake(url):
        calls.append(url)
        return responses[len(calls) - 1]

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    with patch("form4lab.data.sec_fetcher._sec_get", side_effect=fake):
        poll_recent_filings(mock_db, datetime(2026, 3, 13, tzinfo=timezone.utc))

    assert len(calls) == 2, f"Expected 2 calls (full page + empty terminator), got {len(calls)}"


def test_poll_recent_filings_stops_at_max_offset(caplog, monkeypatch):
    """Pagination must stop at EFTS_MAX_OFFSET (SEC's documented 10K deep-paging
    limit) and log at ERROR so monitoring catches the truncation.
    """
    import logging
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch
    from form4lab.data import sec_fetcher
    from form4lab.data.sec_fetcher import poll_recent_filings, EFTS_PAGE_SIZE

    # Tight cap so the test runs fast: 2 full pages, then the loop must stop.
    monkeypatch.setattr(sec_fetcher, "EFTS_MAX_OFFSET", 2 * EFTS_PAGE_SIZE)

    full_page = [{"_source": {"ciks": [f"{i:010d}"]}} for i in range(EFTS_PAGE_SIZE)]
    calls: list[str] = []

    def fake(url):
        calls.append(url)
        return MagicMock(json=lambda: {"hits": {"hits": full_page}})

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    with patch("form4lab.data.sec_fetcher._sec_get", side_effect=fake):
        with caplog.at_level(logging.ERROR, logger="form4lab.data.sec_fetcher"):
            poll_recent_filings(mock_db, datetime(2026, 3, 13, tzinfo=timezone.utc))

    # Two full-page fetches (offset 0 and offset PAGE_SIZE), then cap-reached stop.
    assert len(calls) == 2, f"Expected 2 calls before hitting cap, got {len(calls)}"
    assert any(
        "EFTS poll hit max offset" in rec.message
        for rec in caplog.records
    ), "Max-offset termination must log at ERROR"


def test_poll_recent_filings_breaks_on_mid_pagination_error(caplog):
    """If a mid-pagination request fails, the loop must stop cleanly and log
    at ERROR level so monitoring catches the partial-fetch condition.
    """
    import logging
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch
    from form4lab.data.sec_fetcher import poll_recent_filings, EFTS_PAGE_SIZE

    page1 = [{"_source": {"ciks": [f"{i:010d}"]}} for i in range(EFTS_PAGE_SIZE)]
    calls: list[str] = []

    def fake(url):
        calls.append(url)
        if len(calls) == 1:
            return MagicMock(json=lambda: {"hits": {"hits": page1}})
        raise RuntimeError("simulated network blip")

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    with patch("form4lab.data.sec_fetcher._sec_get", side_effect=fake):
        with caplog.at_level(logging.ERROR, logger="form4lab.data.sec_fetcher"):
            poll_recent_filings(mock_db, datetime(2026, 3, 13, tzinfo=timezone.utc))

    assert len(calls) == 2, "Loop must attempt the second page then stop"
    assert any(
        "EFTS poll failed at offset 100" in rec.message
        for rec in caplog.records
    ), "Mid-pagination failure must be logged at ERROR with the offset"


class TestParse10b51Plan:
    """Tests for 10b5-1 plan indicator parsing from Form 4 XML."""

    FORM4_XML_WITH_PLAN = """<?xml version="1.0"?>
    <ownershipDocument>
        <schemaVersion>X0508</schemaVersion>
        <documentType>4</documentType>
        <periodOfReport>2024-06-15</periodOfReport>
        <issuer>
            <issuerCik>0000001234</issuerCik>
            <issuerName>Test Corp</issuerName>
            <issuerTradingSymbol>TST</issuerTradingSymbol>
        </issuer>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerCik>0000005678</rptOwnerCik>
                <rptOwnerName>Jane Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isDirector>0</isDirector>
                <isOfficer>1</isOfficer>
                <officerTitle>CEO</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <aff10b5One>1</aff10b5One>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <securityTitle><value>Common Stock</value></securityTitle>
                <transactionDate><value>2024-06-15</value></transactionDate>
                <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>500</value></transactionShares>
                    <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
                    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>"""

    FORM4_XML_NO_PLAN = """<?xml version="1.0"?>
    <ownershipDocument>
        <schemaVersion>X0508</schemaVersion>
        <documentType>4</documentType>
        <periodOfReport>2024-06-15</periodOfReport>
        <issuer>
            <issuerCik>0000001234</issuerCik>
            <issuerName>Test Corp</issuerName>
            <issuerTradingSymbol>TST</issuerTradingSymbol>
        </issuer>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerCik>0000005678</rptOwnerCik>
                <rptOwnerName>Jane Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isDirector>0</isDirector>
                <isOfficer>1</isOfficer>
                <officerTitle>CEO</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <aff10b5One>0</aff10b5One>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <securityTitle><value>Common Stock</value></securityTitle>
                <transactionDate><value>2024-06-15</value></transactionDate>
                <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>500</value></transactionShares>
                    <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
                    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>"""

    FORM4_XML_MISSING_PLAN = """<?xml version="1.0"?>
    <ownershipDocument>
        <schemaVersion>X0306</schemaVersion>
        <documentType>4</documentType>
        <periodOfReport>2022-01-15</periodOfReport>
        <issuer>
            <issuerCik>0000001234</issuerCik>
            <issuerName>Test Corp</issuerName>
            <issuerTradingSymbol>TST</issuerTradingSymbol>
        </issuer>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerCik>0000005678</rptOwnerCik>
                <rptOwnerName>Jane Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isDirector>0</isDirector>
                <isOfficer>1</isOfficer>
                <officerTitle>CEO</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <securityTitle><value>Common Stock</value></securityTitle>
                <transactionDate><value>2022-01-15</value></transactionDate>
                <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>500</value></transactionShares>
                    <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
                    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>"""

    def test_plan_trade_flagged_true(self):
        results = parse_form4_xml(self.FORM4_XML_WITH_PLAN, "0001234-24-000001", "2024-06-15")
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is True

    def test_non_plan_trade_flagged_false(self):
        results = parse_form4_xml(self.FORM4_XML_NO_PLAN, "0001234-24-000002", "2024-06-15")
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is False

    def test_missing_plan_element_is_none(self):
        results = parse_form4_xml(self.FORM4_XML_MISSING_PLAN, "0001234-22-000003", "2022-01-15")
        assert len(results) == 1
        assert results[0]["is_10b5_1_plan"] is None
