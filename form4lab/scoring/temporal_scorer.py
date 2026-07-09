"""Temporal insider scoring -- computes scores using only data available before a cutoff date.

Used by the backtester to avoid look-ahead bias. Imports all math functions
from insider_scorer.py and adds a transaction_date filter.
"""
import logging
from collections import defaultdict
from datetime import date

from sqlalchemy.orm import Session

from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.scoring.dedup import dedup_transactions, dedup_outcome_tuples
from form4lab.scoring.insider_scorer import (
    bayesian_hit_rate,
    shrinkage_excess_return,
    momentum_adjustment,
    compute_credibility,
    compute_skill_score,
    assign_tier,
)

logger = logging.getLogger(__name__)


def compute_temporal_score(insider_id: int, db: Session, as_of_date: date) -> dict:
    """Compute insider score using only outcomes from before as_of_date.

    Returns a dict with: skill_score, tier, bayesian_hit_rate, shrunk_excess,
    momentum_adjusted_excess, credibility_weight, confidence, num_prior_buys.
    """
    transactions = dedup_transactions(
        db.query(Transaction)
        .join(TradeOutcome)
        .filter(
            Transaction.insider_id == insider_id,
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.transaction_date < as_of_date,
            TradeOutcome.excess_return_60d.isnot(None),
        )
        .all()
    )

    N = len(transactions)

    if N < 3:
        return {
            "skill_score": 0.0,
            "tier": "Insufficient",
            "bayesian_hit_rate": None,
            "shrunk_excess": None,
            "momentum_adjusted_excess": None,
            "credibility_weight": compute_credibility(N),
            "confidence": None,
            "num_prior_buys": N,
        }

    outcomes = [t.outcome for t in transactions]

    # Bayesian hit rate
    hits = sum(1 for o in outcomes if o.hit_60d)
    hr_result = bayesian_hit_rate(hits, N)

    # Shrinkage excess return
    excess_returns = [o.excess_return_60d for o in outcomes if o.excess_return_60d is not None]
    if excess_returns:
        sr_result = shrinkage_excess_return(excess_returns)
        shrunk_excess = sr_result["shrunk_excess"]
    else:
        shrunk_excess = 0.0

    # Momentum adjustment
    momentums = [o.prior_momentum_20d for o in outcomes]
    returns = [o.excess_return_60d for o in outcomes]
    adj = momentum_adjustment(momentums, returns)
    momentum_adjusted = adj if adj is not None else shrunk_excess

    # Skill score
    cred = compute_credibility(N)
    skill = compute_skill_score(hr_result["bayesian_hit_rate"], momentum_adjusted, cred)

    # Tier
    tier = assign_tier(skill, hr_result["confidence_above_baseline"], N)

    return {
        "skill_score": skill,
        "tier": tier,
        "bayesian_hit_rate": hr_result["bayesian_hit_rate"],
        "shrunk_excess": shrunk_excess,
        "momentum_adjusted_excess": momentum_adjusted,
        "credibility_weight": cred,
        "confidence": hr_result["confidence_above_baseline"],
        "num_prior_buys": N,
    }


class TemporalScoreCache:
    """Preloads all outcomes into memory, computes temporal scores without DB queries.

    Reuses the same math functions as compute_temporal_score() but operates
    entirely from in-memory data, eliminating per-transaction DB round-trips.
    """

    def __init__(self):
        # insider_id -> sorted list of (transaction_date, hit_60d, excess_return_60d, prior_momentum_20d)
        self._outcomes: dict[int, list[tuple]] = defaultdict(list)

    def preload(self, db: Session) -> None:
        """Bulk-load all discretionary transaction outcomes with date filtering support."""
        rows = (
            db.query(
                Transaction.insider_id,
                Transaction.transaction_date,
                TradeOutcome.hit_60d,
                TradeOutcome.excess_return_60d,
                TradeOutcome.prior_momentum_20d,
            )
            .join(TradeOutcome, TradeOutcome.transaction_id == Transaction.id)
            .filter(
                Transaction.is_discretionary == True,  # noqa: E712
                TradeOutcome.excess_return_60d.isnot(None),
            )
            .all()
        )

        self._outcomes.clear()
        for insider_id, txn_date, hit_60d, excess_60d, momentum_20d in rows:
            self._outcomes[insider_id].append(
                (txn_date, bool(hit_60d) if hit_60d is not None else False,
                 float(excess_60d), float(momentum_20d) if momentum_20d is not None else None)
            )

        # Dedup and sort each insider's outcomes by date for efficient filtering
        total_deduped = 0
        for insider_id in self._outcomes:
            self._outcomes[insider_id].sort(key=lambda x: x[0])
            self._outcomes[insider_id] = dedup_outcome_tuples(self._outcomes[insider_id])
            total_deduped += len(self._outcomes[insider_id])

        logger.info(
            f"TemporalScoreCache: preloaded {len(rows)} rows -> {total_deduped} unique events "
            f"for {len(self._outcomes)} insiders"
        )

    def get_score(self, insider_id: int, as_of_date: date) -> dict:
        """Compute temporal score using only preloaded outcomes before as_of_date.

        Produces identical results to compute_temporal_score() but with zero DB queries.
        """
        all_outcomes = self._outcomes.get(insider_id, [])

        # Filter to outcomes before as_of_date
        prior = [o for o in all_outcomes if o[0] < as_of_date]
        N = len(prior)

        if N < 3:
            return {
                "skill_score": 0.0,
                "tier": "Insufficient",
                "bayesian_hit_rate": None,
                "shrunk_excess": None,
                "momentum_adjusted_excess": None,
                "credibility_weight": compute_credibility(N),
                "confidence": None,
                "num_prior_buys": N,
            }

        # Extract outcome components
        hits = sum(1 for _, hit, _, _ in prior if hit)
        excess_returns = [excess for _, _, excess, _ in prior if excess is not None]
        momentums = [mom for _, _, _, mom in prior]
        returns = [excess for _, _, excess, _ in prior]

        # Bayesian hit rate
        hr_result = bayesian_hit_rate(hits, N)

        # Shrinkage excess return
        if excess_returns:
            sr_result = shrinkage_excess_return(excess_returns)
            shrunk_excess = sr_result["shrunk_excess"]
        else:
            shrunk_excess = 0.0

        # Momentum adjustment
        adj = momentum_adjustment(momentums, returns)
        momentum_adjusted = adj if adj is not None else shrunk_excess

        # Skill score
        cred = compute_credibility(N)
        skill = compute_skill_score(hr_result["bayesian_hit_rate"], momentum_adjusted, cred)

        # Tier
        tier = assign_tier(skill, hr_result["confidence_above_baseline"], N)

        return {
            "skill_score": skill,
            "tier": tier,
            "bayesian_hit_rate": hr_result["bayesian_hit_rate"],
            "shrunk_excess": shrunk_excess,
            "momentum_adjusted_excess": momentum_adjusted,
            "credibility_weight": cred,
            "confidence": hr_result["confidence_above_baseline"],
            "num_prior_buys": N,
        }
