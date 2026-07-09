"""Tests for form4lab.services.alert_service."""
import uuid
from datetime import date, timedelta
from unittest.mock import patch

from form4lab.database import SessionLocal, Base, engine
from form4lab.models.alert import Alert
from form4lab.models.company import Company
from form4lab.models.insider import Insider
from form4lab.models.score import InsiderScore
from form4lab.models.transaction import Transaction
from form4lab.services.alert_service import (
    build_person_alerts_query, enrich_alerts, normalize_conviction,
    generate_missing_alerts, generate_and_execute_alerts,
)


def _unique_cik():
    """Generate a unique CIK to avoid collisions with existing data."""
    return str(uuid.uuid4().int)[:10]


def _setup_test_data(db):
    """Insert minimal test data for alert service tests."""
    cik1 = _unique_cik()
    cik2 = _unique_cik()
    insider = Insider(cik=cik1, name="Alert Svc Test Insider", is_institution=False)
    db.add(insider)
    db.flush()

    company = Company(cik=cik2, ticker=f"TST{cik1[:3]}", name="Alert Svc Test Corp")
    db.add(company)
    db.flush()

    txn = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number=f"test-{uuid.uuid4().hex[:12]}_P_2026-01-15_100.0",
        filing_date=date(2026, 1, 15),
        transaction_date=date(2026, 1, 15),
        transaction_code="P",
        shares=100,
        price_per_share=50.0,
        total_value=5000.0,
        acquired_or_disposed="A",
        is_discretionary=True,
        is_common_stock=True,
    )
    db.add(txn)
    db.flush()

    alert = Alert(
        transaction_id=txn.id,
        insider_id=insider.id,
        company_id=company.id,
        alert_type="elite_buy",
        conviction_score=0.85,
        insider_skill_score=1.2,
        transaction_value=5000.0,
        summary="Test alert",
        trade_date=date(2026, 1, 15),
    )
    db.add(alert)

    score = InsiderScore(
        insider_id=insider.id,
        company_id=None,
        num_discretionary_buys=5,
        skill_score=1.2,
        credibility_tier="Elite",
        credibility_weight=0.8,
        confidence_above_baseline=0.9,
    )
    db.add(score)
    db.commit()
    return insider, company, txn, alert, score


def test_build_person_alerts_query_returns_results():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        insider, company, txn, alert, score = _setup_test_data(db)
        cutoff = date(2026, 1, 1)
        alerts = build_person_alerts_query(db, cutoff).filter(
            Alert.insider_id == insider.id
        ).all()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "elite_buy"


def test_build_person_alerts_query_excludes_filtered():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        insider, company, txn, alert, score = _setup_test_data(db)

        # Add a filtered_out alert
        txn2 = Transaction(
            insider_id=insider.id, company_id=company.id,
            accession_number=f"test-{uuid.uuid4().hex[:12]}_P_2026-01-16_50.0",
            filing_date=date(2026, 1, 16), transaction_date=date(2026, 1, 16),
            transaction_code="P", shares=50, acquired_or_disposed="A",
            is_discretionary=True, is_common_stock=True,
        )
        db.add(txn2)
        db.flush()
        db.add(Alert(
            transaction_id=txn2.id, insider_id=insider.id, company_id=company.id,
            alert_type="filtered_out", conviction_score=0.1,
            insider_skill_score=0.0, transaction_value=2500.0,
            summary="Filtered", trade_date=date(2026, 1, 16),
        ))
        db.commit()

        # Default excludes filtered_out
        alerts = build_person_alerts_query(db, date(2026, 1, 1)).filter(
            Alert.insider_id == insider.id
        ).all()
        assert len(alerts) == 1

        # Include filtered_out
        alerts_all = build_person_alerts_query(db, date(2026, 1, 1), exclude_filtered=False).filter(
            Alert.insider_id == insider.id
        ).all()
        assert len(alerts_all) == 2


def test_enrich_alerts_batch_loads():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        insider, company, txn, alert, score = _setup_test_data(db)
        cutoff = date(2026, 1, 1)
        alerts = build_person_alerts_query(db, cutoff).filter(
            Alert.insider_id == insider.id
        ).all()
        enriched = enrich_alerts(alerts, db)

        assert len(enriched) == 1
        assert enriched[0]["insider"].cik == insider.cik
        assert enriched[0]["company"].ticker == company.ticker
        assert enriched[0]["score"].credibility_tier == "Elite"


def test_enrich_alerts_empty():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        enriched = enrich_alerts([], db)
        assert enriched == []


# --- normalize_conviction tests ---


def test_normalize_conviction_assigns_1_to_5():
    """Percentile-based normalization maps raw scores to 1-5 scale."""
    alerts = [{"alert": type("A", (), {"conviction_score": i})(), "conviction_display": None} for i in range(1, 21)]
    result = normalize_conviction(alerts)
    assert result[0]["conviction_display"] == 1
    assert result[-1]["conviction_display"] == 5
    for r in result:
        assert 1 <= r["conviction_display"] <= 5


def test_normalize_conviction_empty():
    assert normalize_conviction([]) == []


def test_normalize_conviction_single():
    alerts = [{"alert": type("A", (), {"conviction_score": 5.0})()}]
    result = normalize_conviction(alerts)
    assert result[0]["conviction_display"] == 5
    assert result[0]["conviction_label"] == "Very Strong"


def test_normalize_conviction_labels():
    """Each score maps to a human-readable label."""
    alerts = [{"alert": type("A", (), {"conviction_score": i})()} for i in range(1, 21)]
    result = normalize_conviction(alerts)
    labels = {r["conviction_display"]: r["conviction_label"] for r in result}
    assert labels[5] == "Very Strong"
    assert labels[4] == "Strong"
    assert labels[3] == "Moderate"
    assert labels[2] == "Weak"
    assert labels[1] == "Noise"


# --- generate_missing_alerts date filter tests ---


def _setup_old_and_new_transactions(db):
    """Insert one old transaction (200 days ago) and one recent (5 days ago), both without alerts."""
    cik1 = _unique_cik()
    cik2 = _unique_cik()
    insider = Insider(cik=cik1, name="Date Filter Test Insider", is_institution=False)
    db.add(insider)
    db.flush()

    company = Company(cik=cik2, ticker=f"DFT{cik1[:3]}", name="Date Filter Test Corp")
    db.add(company)
    db.flush()

    old_date = date.today() - timedelta(days=200)
    old_txn = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number=f"test-{uuid.uuid4().hex[:12]}_P_{old_date}_100.0",
        filing_date=old_date,
        transaction_date=old_date,
        transaction_code="P",
        shares=100,
        price_per_share=50.0,
        total_value=5000.0,
        acquired_or_disposed="A",
        is_discretionary=True,
        is_common_stock=True,
    )
    db.add(old_txn)
    db.flush()

    recent_date = date.today() - timedelta(days=5)
    recent_txn = Transaction(
        insider_id=insider.id,
        company_id=company.id,
        accession_number=f"test-{uuid.uuid4().hex[:12]}_P_{recent_date}_200.0",
        filing_date=recent_date,
        transaction_date=recent_date,
        transaction_code="P",
        shares=200,
        price_per_share=50.0,
        total_value=10000.0,
        acquired_or_disposed="A",
        is_discretionary=True,
        is_common_stock=True,
    )
    db.add(recent_txn)
    db.commit()
    return insider, company, old_txn, recent_txn


def test_generate_missing_alerts_skips_old_transactions():
    """generate_missing_alerts should skip transactions older than max_age_days."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        insider, company, old_txn, recent_txn = _setup_old_and_new_transactions(db)

        # Mock score_new_transaction to track which txn_ids are processed
        processed_ids = []

        def mock_score(txn_id, session):
            processed_ids.append(txn_id)
            return None  # don't actually create alerts

        with patch("form4lab.scoring.signal_generator.score_new_transaction", mock_score):
            generate_missing_alerts(db, max_age_days=90)

        # Old transaction (200 days ago) should NOT be in processed list
        assert old_txn.id not in processed_ids
        # Recent transaction (5 days ago) should be processed
        assert recent_txn.id in processed_ids


def test_generate_and_execute_alerts_skips_old_transactions():
    """generate_and_execute_alerts should skip transactions older than max_age_days."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        insider, company, old_txn, recent_txn = _setup_old_and_new_transactions(db)

        processed_ids = []

        def mock_score(txn_id, session):
            processed_ids.append(txn_id)
            return None

        with patch("form4lab.scoring.signal_generator.score_new_transaction", mock_score):
            generated, executed = generate_and_execute_alerts(db, max_age_days=90)

        assert old_txn.id not in processed_ids
        assert recent_txn.id in processed_ids
