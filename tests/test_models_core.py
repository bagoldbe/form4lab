import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from form4lab.database import Base
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.company import Company


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_insider(db):
    insider = Insider(cik="0001234567", name="Jane Smith")
    db.add(insider)
    db.commit()
    assert insider.id is not None
    assert insider.cik == "0001234567"


def test_create_company(db):
    company = Company(cik="0000320193", ticker="AAPL", name="Apple Inc.", sector="Technology")
    db.add(company)
    db.commit()
    assert company.id is not None


def test_insider_role_links(db):
    insider = Insider(cik="0001234567", name="Jane Smith")
    company = Company(cik="0000320193", ticker="AAPL", name="Apple Inc.")
    db.add_all([insider, company])
    db.flush()
    role = InsiderRole(
        insider_id=insider.id,
        company_id=company.id,
        role_title="CEO",
        is_officer=True,
        is_director=False,
        is_ten_percent_owner=False,
        first_filing_date=date(2020, 1, 1),
        last_filing_date=date(2024, 1, 1),
    )
    db.add(role)
    db.commit()
    assert role.insider_id == insider.id
    assert insider.roles[0].role_title == "CEO"


def test_unique_cik_constraint(db):
    db.add(Insider(cik="0001234567", name="Jane Smith"))
    db.commit()
    db.add(Insider(cik="0001234567", name="Jane Smith Duplicate"))
    with pytest.raises(Exception):
        db.commit()
