from datetime import date, datetime
from sqlalchemy import String, Float, Boolean, Date, DateTime, ForeignKey, func, false
from sqlalchemy.orm import Mapped, mapped_column
from form4lab.database import Base


class BrokerOrder(Base):
    __tablename__ = "broker_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), index=True)
    entry_order_id: Mapped[int | None] = mapped_column(
        ForeignKey("broker_orders.id"), nullable=True
    )
    alpaca_order_id: Mapped[str] = mapped_column(String, unique=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)  # "buy" or "sell"
    qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_type: Mapped[str] = mapped_column(String)  # "market" or "limit"
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    extended_hours: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String, index=True)
    filled_qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Sizing audit: how the notional was decided (form4lab.strategy.base.SizeDecision,
    # returned by the active Strategy's size()).
    # method: "role" | "voltarget" | "voltarget_capped" | "fallback" | "shadow";
    # vol/pct are the realized vol and target fraction when vol-targeting ran.
    sizing_method: Mapped[str | None] = mapped_column(String, nullable=True)
    sizing_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    sizing_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class BrokerPosition(Base):
    __tablename__ = "broker_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), unique=True)
    entry_order_id: Mapped[int] = mapped_column(ForeignKey("broker_orders.id"))
    exit_order_id: Mapped[int | None] = mapped_column(
        ForeignKey("broker_orders.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String, index=True)
    shares: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_date: Mapped[date] = mapped_column(Date)
    exit_target_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String, index=True)  # open, closing, closed, delisted
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    last_market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reconcile_hold: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    insider_name: Mapped[str] = mapped_column(String)
    insider_role: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
