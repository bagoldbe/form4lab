"""Execution health check.

Surfaces the silent-execution-freeze failure mode: the scheduler keeps
generating alerts while an entry gate silently skips every one. Without this
check, a bug that zeroes out execution can sit undetected for months.

Daily scheduler job calls check_execution_health() and logs a structured line
prefixed with "EXEC_HEALTH" so it is trivially greppable in deploy logs.
"""
from datetime import date, timedelta
from typing import NamedTuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from form4lab.models.alert import Alert
from form4lab.models.broker import BrokerPosition

WINDOW_DAYS = 14
LOG_MARKER = "EXEC_HEALTH"


class ExecutionHealth(NamedTuple):
    window_days: int
    tradeable_alerts: int
    positions_opened: int
    healthy: bool
    reason: str


def check_execution_health(
    db: Session,
    window_days: int = WINDOW_DAYS,
    as_of: date | None = None,
) -> ExecutionHealth:
    """Compare tradeable alerts vs broker positions opened in the window.

    Unhealthy when at least one tradeable alert was generated but zero
    broker_positions were opened. This is the exact pattern of the
    silent-execution-freeze pattern; every other state (no alerts, or alerts +
    positions, or alerts + skips that opened other positions) is healthy.
    """
    from form4lab.strategy.registry import get_active

    as_of = as_of or date.today()
    cutoff_date = as_of - timedelta(days=window_days)

    # Derive the tradeable set from the active strategy so this check can
    # never silently drift from what actually executes — a strategy change
    # that isn't mirrored here is exactly the class of bug this guards against.
    tradeable_alert_types = tuple(sorted(get_active()[1].tradeable_names()))

    alerts_count = (
        db.query(func.count(Alert.id))
        .filter(
            Alert.alert_type.in_(tradeable_alert_types),
            func.date(Alert.created_at) >= cutoff_date,
        )
        .scalar()
    ) or 0

    positions_count = (
        db.query(func.count(BrokerPosition.id))
        .filter(BrokerPosition.entry_date >= cutoff_date)
        .scalar()
    ) or 0

    if alerts_count == 0:
        return ExecutionHealth(
            window_days=window_days,
            tradeable_alerts=0,
            positions_opened=positions_count,
            healthy=True,
            reason="no tradeable alerts in window — nothing to evaluate",
        )

    if positions_count == 0:
        return ExecutionHealth(
            window_days=window_days,
            tradeable_alerts=alerts_count,
            positions_opened=0,
            healthy=False,
            reason=(
                f"{alerts_count} tradeable alert(s) generated but 0 positions opened "
                "— execution may be silently frozen (gate fires for every signal)"
            ),
        )

    return ExecutionHealth(
        window_days=window_days,
        tradeable_alerts=alerts_count,
        positions_opened=positions_count,
        healthy=True,
        reason=f"{positions_count} of {alerts_count} alert(s) traded",
    )
