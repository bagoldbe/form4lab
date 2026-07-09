from datetime import date, datetime
from sqlalchemy import Date, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from form4lab.database import Base


class EarningsDate(Base):
    """Historical (and scheduled) earnings report dates for a company.

    Used for event-driven exits. Sourced from
    yfinance get_earnings_dates. `report_time` is BMO/AMC/None when known.
    """
    __tablename__ = "earnings_dates"
    __table_args__ = (UniqueConstraint("company_id", "earnings_date", name="uq_company_earnings_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    earnings_date: Mapped[date] = mapped_column(Date, index=True)
    report_time: Mapped[str | None] = mapped_column(String, nullable=True)  # 'BMO' | 'AMC' | None
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
