import pytest
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.price import PriceData
from form4lab.data.price_fetcher import PriceProvider, YFinanceProvider, SECTOR_ETF_MAP


def test_price_provider_is_abstract():
    with pytest.raises(TypeError):
        PriceProvider()


def test_sector_etf_mapping():
    assert SECTOR_ETF_MAP["Technology"] == "XLK"
    assert SECTOR_ETF_MAP["Healthcare"] == "XLV"
    assert SECTOR_ETF_MAP.get("NonExistent") is None


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_cached_to_df_empty(db):
    provider = YFinanceProvider(db)
    df = provider._cached_to_df([])
    assert len(df) == 0
    assert "close" in df.columns


def test_cached_to_df_with_data(db):
    p = PriceData(
        ticker="SPY",
        date=date(2024, 12, 2),
        open=100,
        high=102,
        low=99,
        close=101,
        adj_close=101,
        volume=50000000,
    )
    db.add(p)
    db.commit()
    records = db.query(PriceData).all()
    provider = YFinanceProvider(db)
    df = provider._cached_to_df(records)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 101


def test_sector_etf_returns_none_for_unknown(db):
    provider = YFinanceProvider(db)
    assert provider.get_sector_etf("NonExistent") is None


def test_sector_etf_returns_correct_etf(db):
    provider = YFinanceProvider(db)
    assert provider.get_sector_etf("Technology") == "XLK"
    assert provider.get_sector_etf("Energy") == "XLE"
    assert provider.get_sector_etf("Financials") == "XLF"


def test_cached_to_df_preserves_all_columns(db):
    p = PriceData(
        ticker="AAPL",
        date=date(2024, 1, 15),
        open=150.0,
        high=155.0,
        low=149.0,
        close=153.0,
        adj_close=152.5,
        volume=80000000,
    )
    db.add(p)
    db.commit()
    records = db.query(PriceData).all()
    provider = YFinanceProvider(db)
    df = provider._cached_to_df(records)
    expected_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    assert set(df.columns) == expected_cols
    assert df.iloc[0]["open"] == 150.0
    assert df.iloc[0]["adj_close"] == 152.5
    assert df.iloc[0]["volume"] == 80000000
