from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import Float, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from form4lab.database import Base

if TYPE_CHECKING:
    from form4lab.models.transaction import Transaction


class TradeOutcome(Base):
    __tablename__ = "trade_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"), unique=True)
    stock_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    stock_return_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    stock_return_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_return_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_return_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    excess_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    excess_return_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    excess_return_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    prior_momentum_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_20d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_60d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_120d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    transaction: Mapped["Transaction"] = relationship(back_populates="outcome")
