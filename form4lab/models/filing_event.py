from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from form4lab.database import Base


class CompanyFilingEvent(Base):
    """Non-Form-4 filing events harvested from the SEC submissions API
    (research-only): 8-Ks (with their item codes — Item 2.02 = results
    announcement, the free full-history earnings-release proxy) and SC 13D/13G
    beneficial-ownership filings (activist-confluence context). filing_date is
    the public-knowledge date; report_date is the underlying event date when
    the SEC provides one (8-Ks).
    """
    __tablename__ = "company_filing_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    form_type: Mapped[str] = mapped_column(String, index=True)
    filing_date: Mapped[date] = mapped_column(Date, index=True)
    report_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    items: Mapped[str | None] = mapped_column(String, nullable=True)
    accession_number: Mapped[str] = mapped_column(String, unique=True)
    primary_document: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
