from datetime import date, datetime
from sqlalchemy import Date, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from form4lab.database import Base


class Fundamental(Base):
    """A single point-in-time financial fact from SEC EDGAR XBRL company-facts.

    Stored raw (one row per concept × reporting period × filing) so features can
    be computed look-ahead-free: at any date, use only rows whose `filed_date`
    is on/before that date. Derived metrics (EV/EBITDA, leverage, dilution) are
    computed downstream, not stored here.
    """
    __tablename__ = "fundamentals"
    __table_args__ = (
        UniqueConstraint("company_id", "concept", "period_end", "filed_date",
                         name="uq_fundamental_fact"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    concept: Mapped[str] = mapped_column(String, index=True)   # e.g. us-gaap:NetIncomeLoss
    period_end: Mapped[date] = mapped_column(Date)            # reporting period end ('end')
    filed_date: Mapped[date] = mapped_column(Date, index=True)  # when it became public ('filed')
    fiscal_period: Mapped[str | None] = mapped_column(String, nullable=True)  # FY/Q1.. ('fp')
    form: Mapped[str | None] = mapped_column(String, nullable=True)           # 10-K/10-Q
    value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
