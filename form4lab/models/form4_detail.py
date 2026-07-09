from datetime import date, datetime

from sqlalchemy import (
    BigInteger, Date, DateTime, Float, ForeignKey, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from form4lab.database import Base


class Form4FilingMeta(Base):
    """Filing-level Form 4 detail from the SEC bulk insider datasets.

    Keyed by the RAW SEC accession number (transactions.accession_number is a
    synthetic composite `{accession}_{code}_{date}_{shares}` — join via its
    prefix). Research-only tables: the live transactions pipeline never reads
    these. `source` records provenance ('bulk_tsv' today; 'xml' if a direct
    re-fetch path is ever added, which would also populate raw_xml).
    """
    __tablename__ = "form4_filing_meta"

    accession_number: Mapped[str] = mapped_column(String, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    period_of_report: Mapped[date | None] = mapped_column(Date, nullable=True)
    remarks: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_xml: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String, default="bulk_tsv")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Form4DerivTxn(Base):
    """One derivative-table (Table II) transaction row from a Form 4.

    Carries what the live parser drops on the floor: option expiration dates
    (→ early-exercise / forgone-time-value research), conversion/exercise
    prices (a cleaner strike than the Table I price), exercisable dates,
    underlying share counts, and the SEC timeliness code ('E' early/'L' late).
    `deriv_trans_sk` is the SEC's own row key, kept for idempotent re-ingest.
    """
    __tablename__ = "form4_deriv_txns"
    __table_args__ = (
        UniqueConstraint("accession_number", "deriv_trans_sk",
                         name="uq_form4_deriv_row"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    accession_number: Mapped[str] = mapped_column(
        ForeignKey("form4_filing_meta.accession_number"), index=True)
    deriv_trans_sk: Mapped[int] = mapped_column(BigInteger)
    security_title: Mapped[str | None] = mapped_column(String, nullable=True)
    trans_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    trans_code: Mapped[str | None] = mapped_column(String, nullable=True)
    conv_exercise_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trans_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    trans_price_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    acquired_disposed: Mapped[str | None] = mapped_column(String(1), nullable=True)
    exercisable_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    expiration_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    underlying_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares_owned_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeliness: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Form4Footnote(Base):
    """One footnote from a Form 4 — the only pre-April-2023 source of 10b5-1
    plan status (the <aff10b5One> checkbox didn't exist before then) and of
    mechanical-buy context (DRIP / 401(k) / ESPP / ownership-guideline buys).
    Stored raw; classification happens downstream so regexes can evolve.
    """
    __tablename__ = "form4_footnotes"
    __table_args__ = (
        UniqueConstraint("accession_number", "footnote_id",
                         name="uq_form4_footnote"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    accession_number: Mapped[str] = mapped_column(
        ForeignKey("form4_filing_meta.accession_number"), index=True)
    footnote_id: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
