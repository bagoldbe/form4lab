"""Transaction deduplication utilities.

Same-day lot splits (e.g., an insider filing 12 separate lots on the same date)
should count as one buy *event* for scoring, backtesting, and alerting.

A buy event is defined as a unique (insider_id, company_id, transaction_date) triple.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from form4lab.models.transaction import Transaction


def dedup_transactions(transactions: list[Transaction]) -> list[Transaction]:
    """Collapse same-day lots into one Transaction per (insider_id, company_id, transaction_date).

    Keeps the row with the largest total_value from each group.
    Sets a transient ``event_total_value`` attribute on each kept row
    with the *sum* of all lots' values for that event — callers that need the
    true economic magnitude of the buy (e.g. size_score) should use this.
    """
    groups: dict[tuple, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        key = (txn.insider_id, txn.company_id, txn.transaction_date)
        groups[key].append(txn)

    deduped: list[Transaction] = []
    for key, lots in groups.items():
        # Pick the lot with the largest total_value as the representative
        best = max(lots, key=lambda t: (t.total_value or 0))
        # Sum all lot values for the event
        summed = sum((t.total_value or 0) for t in lots)
        best.event_total_value = summed  # type: ignore[attr-defined]
        deduped.append(best)

    return deduped


def dedup_outcome_tuples(
    tuples: list[tuple],
) -> list[tuple]:
    """Dedup (txn_date, hit, excess, momentum) tuples by txn_date.

    For TemporalScoreCache which works with raw tuples, not ORM objects.
    When multiple tuples share a txn_date, keeps the first one encountered
    (they have identical outcomes since same ticker/date).
    """
    seen: set = set()
    deduped: list[tuple] = []
    for t in tuples:
        txn_date = t[0]
        if txn_date not in seen:
            seen.add(txn_date)
            deduped.append(t)
    return deduped
