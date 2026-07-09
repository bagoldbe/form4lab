from datetime import date, datetime
from sqlalchemy import String, Float, Boolean, Integer, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from form4lab.database import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    run_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    insider_id: Mapped[int] = mapped_column(ForeignKey("insiders.id"))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    transaction_date: Mapped[date] = mapped_column(Date, index=True)
    simulated_alert_type: Mapped[str] = mapped_column(String)
    simulated_conviction: Mapped[float] = mapped_column(Float)
    simulated_skill_score: Mapped[float] = mapped_column(Float)
    simulated_tier: Mapped[str] = mapped_column(String)
    num_prior_buys: Mapped[int] = mapped_column(Integer)
    actual_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_return_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_return_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_excess_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_excess_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_excess_120d: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_hit_20d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    actual_hit_60d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    actual_hit_120d: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
