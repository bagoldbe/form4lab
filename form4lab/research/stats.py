"""Anti-overfitting statistics for the research loop.

Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the probability that a
candidate's true Sharpe exceeds the *inflated* benchmark you'd expect from the
maximum of N independent trials under the null. It penalizes (a) the number of
configs tried so far, (b) non-normal return shape (skew/kurtosis). A candidate
"passes" only if DSR clears a high bar (e.g. 0.95) — this is the central guard
against p-hacking the frontier.
"""
import math

from scipy.stats import norm

GAMMA = 0.5772156649015329  # Euler-Mascheroni


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum (per-observation) Sharpe under `n_trials` null trials.

    sr_variance = variance of the per-observation Sharpe estimates across trials.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - GAMMA) * z1 + GAMMA * z2)


def deflated_sharpe(sr_hat: float, n_obs: int, skew: float, kurtosis: float,
                    n_trials: int, sr_variance: float) -> float:
    """Deflated Sharpe Ratio in [0,1].

    sr_hat   : per-observation (NOT annualized) Sharpe of the candidate.
    n_obs    : number of return observations (trades).
    skew     : skewness of the return series.
    kurtosis : Pearson kurtosis (normal = 3).
    n_trials : cumulative number of independent configs tried (from the ledger).
    sr_variance : variance of per-observation Sharpes across those trials.
    """
    if n_obs < 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    denom = 1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat ** 2
    denom = math.sqrt(max(1e-12, denom))
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def per_trade_sharpe_stats(pnl_pcts: list[float]) -> dict:
    """(per-observation) Sharpe, n, skew, kurtosis of a list of trade returns."""
    import numpy as np
    from scipy.stats import skew as _skew, kurtosis as _kurt
    x = np.asarray([p for p in pnl_pcts if p is not None], dtype=float)
    n = len(x)
    if n < 2:
        return {"sr_hat": 0.0, "n_obs": n, "skew": 0.0, "kurtosis": 3.0}
    sd = x.std(ddof=1)
    sr = float(x.mean() / sd) if sd > 0 else 0.0
    return {"sr_hat": sr, "n_obs": n,
            "skew": float(_skew(x)), "kurtosis": float(_kurt(x, fisher=False))}
