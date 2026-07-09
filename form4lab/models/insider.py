from datetime import date, datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, Boolean, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from form4lab.database import Base

if TYPE_CHECKING:
    from form4lab.models.company import Company
    from form4lab.models.transaction import Transaction
    from form4lab.models.score import InsiderScore


import re

# Patterns that indicate an institution (matched as whole words)
_INSTITUTION_PATTERNS = re.compile(
    r"\b("
    r"LLC|L\.?L\.?C\.?|INC\.?|CORP\.?|CORPORATION|COMPANY"
    r"|L\.?P\.?|LIMITED PARTNERSHIP"
    r"|TRUST|FUND|CAPITAL|MANAGEMENT|PARTNERS|HOLDINGS"
    r"|& CO\.?|GROUP|ADVISORS|ASSOCIATES|ENTERPRISES"
    r"|INVESTMENTS|ASSET|PENSION|FOUNDATION"
    r")\b",
    re.IGNORECASE,
)


def detect_is_institution(name: str) -> bool:
    """Detect if an insider name is an institution rather than a person."""
    if not name:
        return False
    return bool(_INSTITUTION_PATTERNS.search(name))


class Insider(Base):
    __tablename__ = "insiders"

    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    is_institution: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    roles: Mapped[list["InsiderRole"]] = relationship(back_populates="insider")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="insider")
    scores: Mapped[list["InsiderScore"]] = relationship(back_populates="insider")


class InsiderRole(Base):
    __tablename__ = "insider_roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    insider_id: Mapped[int] = mapped_column(ForeignKey("insiders.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    role_title: Mapped[str] = mapped_column(String)
    is_officer: Mapped[bool] = mapped_column(Boolean, default=False)
    is_director: Mapped[bool] = mapped_column(Boolean, default=False)
    is_ten_percent_owner: Mapped[bool] = mapped_column(Boolean, default=False)
    first_filing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_filing_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    insider: Mapped["Insider"] = relationship(back_populates="roles")
    company: Mapped["Company"] = relationship(back_populates="insider_roles")
