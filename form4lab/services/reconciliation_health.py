"""Reconciliation health check.

Surfaces positions that reconciliation flagged for manual review (renamed-but-
broker-missing, ambiguous corporate actions) or booked as delisted losses.
Daily scheduler job logs a structured line prefixed with RECON_HEALTH, mirroring
EXEC_HEALTH so it is trivially greppable in deploy logs.
"""
from datetime import date, timedelta
from typing import NamedTuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from form4lab.models.broker import BrokerPosition

WINDOW_DAYS = 14
LOG_MARKER = "RECON_HEALTH"


class ReconciliationHealth(NamedTuple):
    window_days: int
    delisted_count: int
    orphan_count: int
    held_count: int
    healthy: bool
    reason: str


def check_reconciliation_health(
    db: Session, window_days: int = WINDOW_DAYS, as_of: date | None = None,
) -> ReconciliationHealth:
    """Count delisted/orphan closures in the window + positions held for manual review.

    A non-zero held_count means reconciliation could not safely auto-resolve a
    disappeared position (CA ambiguity, asset-status/CA lookup failure, or an
    unconfirmed rename) and the position needs human review.

    A non-zero delisted_count means positions were auto-booked as zero-value
    delist closes (confirmed inactive asset, no CA) in the reporting window.

    A non-zero orphan_count means positions were auto-closed as orphans (asset
    active, no sell order, no CA) in the reporting window. These can represent
    missed renames and warrant human review.

    Unhealthy when any count is non-zero.
    """
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=window_days)

    delisted = (
        db.query(func.count(BrokerPosition.id))
        .filter(BrokerPosition.status == "delisted", BrokerPosition.exit_date >= cutoff)
        .scalar()
    ) or 0
    orphan = (
        db.query(func.count(BrokerPosition.id))
        .filter(
            BrokerPosition.close_reason == "orphan_no_sell",
            BrokerPosition.exit_date >= cutoff,
        )
        .scalar()
    ) or 0
    held = (
        db.query(func.count(BrokerPosition.id))
        .filter(
            BrokerPosition.reconcile_hold.is_(True),
            BrokerPosition.status.in_(["open", "closing"]),
        )
        .scalar()
    ) or 0

    if delisted == 0 and orphan == 0 and held == 0:
        return ReconciliationHealth(window_days, 0, 0, 0, True,
                                    "no reconciliation anomalies")
    return ReconciliationHealth(
        window_days, delisted, orphan, held, False,
        f"{delisted} delisted close(s) in window, {orphan} orphan close(s) in window, "
        f"{held} position(s) held for manual review",
    )
