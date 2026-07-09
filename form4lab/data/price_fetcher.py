import contextlib
import io
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from form4lab.models.price import PriceData

logger = logging.getLogger(__name__)

# Suppress noisy yfinance/urllib3 warnings for delisted or missing tickers
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


class PriceProvider(ABC):
    @abstractmethod
    def get_daily_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Returns DataFrame with columns: date, open, high, low, close, adj_close, volume"""
        pass

    @abstractmethod
    def get_sector_etf(self, sector: str) -> str | None:
        pass


class YFinanceProvider(PriceProvider):
    def __init__(self, db: Session, db_only: bool = False):
        self.db = db
        self.db_only = db_only

    def get_sector_etf(self, sector: str) -> str | None:
        return SECTOR_ETF_MAP.get(sector)

    def get_daily_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        # Check cache first
        cached = (
            self.db.query(PriceData)
            .filter(
                PriceData.ticker == ticker,
                PriceData.date >= start_date,
                PriceData.date <= end_date,
            )
            .order_by(PriceData.date)
            .all()
        )

        if not self.db_only:
            cached_dates = {p.date for p in cached}
            # Fetch from yfinance if we appear to be missing data
            # Estimate expected trading days (rough: 5/7 of calendar days)
            expected_days = max(1, int((end_date - start_date).days * 5 / 7))
            if len(cached_dates) < expected_days - 2:
                try:
                    # Suppress yfinance stderr noise for delisted/missing tickers
                    with contextlib.redirect_stderr(io.StringIO()):
                        df = yf.download(
                            ticker,
                            start=str(start_date),
                            end=str(end_date),
                            progress=False,
                            auto_adjust=False,
                        )
                    if not df.empty:
                        self._save_to_db(ticker, df)
                        # Re-query after saving new data
                        cached = (
                            self.db.query(PriceData)
                            .filter(
                                PriceData.ticker == ticker,
                                PriceData.date >= start_date,
                                PriceData.date <= end_date,
                            )
                            .order_by(PriceData.date)
                            .all()
                        )
                except Exception as e:
                    logger.warning("Failed to fetch prices from yfinance for %s: %s", ticker, e)
                    try:
                        self.db.rollback()
                    except Exception:
                        pass

        return self._cached_to_df(cached)

    def _save_to_db(self, ticker: str, df: pd.DataFrame):
        # yfinance may return MultiIndex columns like (Close, SPY) -- flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Batch-check existing dates to avoid per-row queries
        all_dates = [
            idx.date() if hasattr(idx, "date") else idx for idx in df.index
        ]
        existing_dates = {
            r.date for r in self.db.query(PriceData.date)
            .filter(PriceData.ticker == ticker, PriceData.date.in_(all_dates))
            .all()
        }

        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            if d not in existing_dates:
                p = PriceData(
                    ticker=ticker,
                    date=d,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    adj_close=float(row.get("Adj Close", row["Close"])),
                    volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                )
                self.db.add(p)
        self.db.commit()

    def _cached_to_df(self, records) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )
        return pd.DataFrame(
            [
                {
                    "date": r.date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "adj_close": r.adj_close,
                    "volume": r.volume,
                }
                for r in records
            ]
        )

    def get_stock_sector(self, ticker: str) -> str | None:
        try:
            info = yf.Ticker(ticker).info
            return info.get("sector")
        except Exception as e:
            logger.warning("Failed to get sector for %s: %s", ticker, e)
            return None
