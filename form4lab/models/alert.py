from datetime import date, datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, Text, DateTime, Date, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from form4lab.database import Base

if TYPE_CHECKING:
    from form4lab.models.transaction import Transaction
    from form4lab.models.insider import Insider
    from form4lab.models.company import Company


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    insider_id: Mapped[int] = mapped_column(ForeignKey("insiders.id"))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    alert_type: Mapped[str] = mapped_column(String)
    conviction_score: Mapped[float] = mapped_column(Float)
    insider_skill_score: Mapped[float] = mapped_column(Float)
    transaction_value: Mapped[float] = mapped_column(Float)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    trade_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    transaction: Mapped["Transaction"] = relationship()
    insider: Mapped["Insider"] = relationship()
    company: Mapped["Company"] = relationship()
