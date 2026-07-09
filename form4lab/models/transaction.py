from datetime import date, datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, Boolean, Date, DateTime, BigInteger, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from form4lab.database import Base

if TYPE_CHECKING:
    from form4lab.models.insider import Insider
    from form4lab.models.company import Company
    from form4lab.models.outcome import TradeOutcome


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    insider_id: Mapped[int] = mapped_column(ForeignKey("insiders.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    accession_number: Mapped[str] = mapped_column(String, unique=True)
    filing_date: Mapped[date] = mapped_column(Date, index=True)
    transaction_date: Mapped[date] = mapped_column(Date, index=True)
    transaction_code: Mapped[str] = mapped_column(String)
    shares: Mapped[float] = mapped_column(Float)
    price_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares_owned_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    acquired_or_disposed: Mapped[str] = mapped_column(String)
    is_discretionary: Mapped[bool] = mapped_column(Boolean, default=False)
    security_title: Mapped[str | None] = mapped_column(String, nullable=True)
    is_common_stock: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_10b5_1_plan: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    insider: Mapped["Insider"] = relationship(back_populates="transactions")
    company: Mapped["Company"] = relationship(back_populates="transactions")
    outcome: Mapped["TradeOutcome"] = relationship(back_populates="transaction", uselist=False)
