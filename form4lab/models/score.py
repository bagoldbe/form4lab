from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from form4lab.database import Base

if TYPE_CHECKING:
    from form4lab.models.insider import Insider


class InsiderScore(Base):
    __tablename__ = "insider_scores"
    __table_args__ = (UniqueConstraint("insider_id", "company_id", name="uq_insider_company_score"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    insider_id: Mapped[int] = mapped_column(ForeignKey("insiders.id"), index=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), nullable=True)
    num_discretionary_buys: Mapped[int] = mapped_column(Integer, default=0)
    raw_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    bayesian_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_above_baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_excess_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    shrunk_excess_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_adjusted_excess: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Horizon (trading days) the above fields were computed against — 60 maps
    # to the 60d outcome columns on TradeOutcome; 20/120 selectable.
    horizon_days: Mapped[int] = mapped_column(Integer, default=60)
    credibility_weight: Mapped[float] = mapped_column(Float, default=0.0)
    skill_score: Mapped[float] = mapped_column(Float, default=0.0)
    credibility_tier: Mapped[str] = mapped_column(String, default="Insufficient")
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    insider: Mapped["Insider"] = relationship(back_populates="scores")
