from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from form4lab.database import Base


class ShortVolume(Base):
    """Daily off-exchange short-sale volume from FINRA's Reg SHO consolidated
    (CNMS) files — the FREE cousin of short interest (which has no free
    historical source). short_volume/total_volume per day is the pressure
    proxy; FINRA-facility volume only, so a biased-but-consistent measure.
    FINRA publishes each file the same evening — usable from the next day
    (look-ahead guard lives in the flag computation, not here).
    """
    __tablename__ = "short_volume"
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_short_volume_day"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    short_volume: Mapped[int] = mapped_column(BigInteger)
    short_exempt_volume: Mapped[int] = mapped_column(BigInteger)
    total_volume: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
