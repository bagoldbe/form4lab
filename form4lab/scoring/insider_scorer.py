import logging

import numpy as np
from scipy.stats import beta as beta_dist, norm, linregress
from scipy.special import logit as scipy_logit
from sqlalchemy import or_
from sqlalchemy.orm import Session

from form4lab.models.insider import Insider
from form4lab.models.transaction import Transaction
from form4lab.models.outcome import TradeOutcome
from form4lab.models.score import InsiderScore
from form4lab.scoring.dedup import dedup_transactions
from form4lab.config import settings
from form4lab.utils import to_python_float as _f

logger = logging.getLogger(__name__)

_scoring = settings.scoring


def bayesian_hit_rate(hits: int, total: int,
                      alpha_0: float | None = None,
                      beta_0: float | None = None) -> dict:
    """Beta-Binomial posterior for insider hit rate."""
    alpha_0 = alpha_0 if alpha_0 is not None else _scoring.hit_rate_alpha_0
    beta_0 = beta_0 if beta_0 is not None else _scoring.hit_rate_beta_0

    alpha_post = alpha_0 + hits
    beta_post = beta_0 + (total - hits)

    bayesian_hr = alpha_post / (alpha_post + beta_post)

    confidence = 1 - beta_dist.cdf(_scoring.hit_rate_baseline, alpha_post, beta_post)

    return {
        "bayesian_hit_rate": bayesian_hr,
        "confidence_above_baseline": confidence,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
    }


def shrinkage_excess_return(excess_returns: list[float],
                            prior_mu: float | None = None,
                            k: float | None = None) -> dict:
    """Normal-Normal shrinkage for mean excess return."""
    prior_mu = prior_mu if prior_mu is not None else _scoring.prior_excess_return
    k = k if k is not None else _scoring.shrinkage_k

    N = len(excess_returns)
    sum_AR = sum(excess_returns)

    shrunk_excess = (k * prior_mu + sum_AR) / (k + N)

    if N >= 3:
        sigma = np.std(excess_returns, ddof=1)
    else:
        sigma = _scoring.default_volatility

    if sigma == 0:
        sigma = _scoring.default_volatility

    posterior_var = 1 / (k / sigma**2 + N / sigma**2)
    prob_positive = 1 - norm.cdf(0, loc=shrunk_excess, scale=np.sqrt(posterior_var))

    return {
        "shrunk_excess": shrunk_excess,
        "prob_positive": prob_positive,
        "sigma": sigma,
    }


def momentum_adjustment(momentums: list[float], returns: list[float]) -> float | None:
    """Regress excess returns on prior momentum. Return intercept (alpha after removing momentum)."""
    min_pairs = _scoring.momentum_min_pairs
    if len(momentums) < min_pairs or len(returns) < min_pairs:
        return None

    pairs = [(m, r) for m, r in zip(momentums, returns) if m is not None and r is not None]
    if len(pairs) < min_pairs:
        return None

    ms, rs = zip(*pairs)

    # linregress raises ValueError if all x values are identical
    if np.ptp(ms) == 0:
        return None

    slope, intercept, r_value, p_value, std_err = linregress(ms, rs)
    return intercept


def compute_credibility(n: int, n_0: int | None = None) -> float:
    """Credibility weight. Increases with sample size, caps at 1.0."""
    n_0 = n_0 if n_0 is not None else _scoring.credibility_n0
    if n <= 0:
        return 0.0
    return float(min(1.0, np.sqrt(n / n_0)))


def compute_skill_score(bayesian_hit_rate: float, momentum_adjusted_excess: float,
                        credibility: float) -> float:
    """Combined skill score (weighted hit-rate and return-magnitude)."""
    # Clamp hit rate to avoid logit(0) or logit(1)
    clamped_hr = max(0.01, min(0.99, bayesian_hit_rate))
    hit_rate_component = scipy_logit(clamped_hr)

    magnitude_component = momentum_adjusted_excess / _scoring.sigma_typical

    w1 = _scoring.skill_w_hit_rate
    w2 = _scoring.skill_w_magnitude
    return credibility * (w1 * hit_rate_component + w2 * magnitude_component)


def assign_tier(skill_score: float, confidence: float, n: int) -> str:
    """Assign credibility tier based on score, confidence, and sample size."""
    if n < _scoring.insufficient_sample_min:
        return "Insufficient"
    elif (skill_score >= _scoring.elite_skill_min
          and confidence >= _scoring.elite_confidence_min
          and n >= _scoring.elite_sample_min):
        return "Elite"
    elif (skill_score >= _scoring.strong_skill_min
          and confidence >= _scoring.strong_confidence_min):
        return "Strong"
    elif skill_score >= 0.0:
        return "Average"
    else:
        return "Weak"


def compute_insider_score(insider_id: int, db: Session, company_id: int | None = None) -> InsiderScore:
    """Full scoring pipeline for one insider."""
    # Step 1: Gather track record
    query = db.query(Transaction).join(TradeOutcome).filter(
        Transaction.insider_id == insider_id,
        Transaction.is_discretionary == True,  # noqa: E712
        or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
        TradeOutcome.excess_return_60d.isnot(None),
    )
    if company_id:
        query = query.filter(Transaction.company_id == company_id)

    transactions = dedup_transactions(query.all())
    N = len(transactions)

    # Get or create score record
    score = db.query(InsiderScore).filter(
        InsiderScore.insider_id == insider_id,
        InsiderScore.company_id == company_id,
    ).first()
    if not score:
        score = InsiderScore(insider_id=insider_id, company_id=company_id)
        db.add(score)

    score.num_discretionary_buys = N
    # horizon_days=60 maps to the 60d outcome columns read below; 20/120 selectable.
    score.horizon_days = _scoring.horizon_days

    if N < 3:
        score.credibility_tier = "Insufficient"
        score.skill_score = 0.0
        score.credibility_weight = compute_credibility(N)
        db.commit()
        return score

    # Step 2: Bayesian hit rate
    outcomes = [t.outcome for t in transactions]
    hits = sum(1 for o in outcomes if o.hit_60d)
    hr_result = bayesian_hit_rate(hits, N)

    score.raw_hit_rate = _f(hits / N) if N > 0 else None
    score.bayesian_hit_rate = _f(hr_result["bayesian_hit_rate"])
    score.confidence_above_baseline = _f(hr_result["confidence_above_baseline"])

    # Step 3: Shrinkage excess return
    excess_returns = [o.excess_return_60d for o in outcomes if o.excess_return_60d is not None]
    if excess_returns:
        sr_result = shrinkage_excess_return(excess_returns)
        score.avg_excess_return = _f(np.mean(excess_returns))
        score.shrunk_excess_return = _f(sr_result["shrunk_excess"])
    else:
        score.avg_excess_return = 0.0
        score.shrunk_excess_return = 0.0

    # Step 4: Momentum adjustment
    momentums = [o.prior_momentum_20d for o in outcomes]
    returns = [o.excess_return_60d for o in outcomes]
    adj = momentum_adjustment(momentums, returns)
    score.momentum_adjusted_excess = _f(adj) if adj is not None else score.shrunk_excess_return

    # Step 5: Skill score
    cred = compute_credibility(N)
    score.credibility_weight = cred
    score.skill_score = _f(compute_skill_score(
        score.bayesian_hit_rate,
        score.momentum_adjusted_excess,
        cred
    ))

    # Step 6: Tier
    score.credibility_tier = assign_tier(
        score.skill_score, score.confidence_above_baseline, N
    )

    db.commit()
    return score


def refresh_all_scores(db: Session) -> int:
    """Recompute scores for all non-institutional insiders with discretionary buys."""
    # Find all unique insider IDs that have discretionary transactions with outcomes
    # Skip institutions -- they're not individual insiders with informational edge
    insider_ids = (
        db.query(Transaction.insider_id)
        .join(Insider, Transaction.insider_id == Insider.id)
        .filter(
            Transaction.is_discretionary == True,  # noqa: E712
            Insider.is_institution == False,  # noqa: E712
        )
        .distinct()
        .all()
    )

    count = 0
    for (insider_id,) in insider_ids:
        try:
            compute_insider_score(insider_id, db, company_id=None)
            count += 1
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to score insider {insider_id}: {e}")

    return count
