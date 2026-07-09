"""Vectorized, look-ahead-free feature builders for backtests.

Each has a live counterpart in form4lab/strategy/features.py.
"""
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_backtest_data(db: Session, horizon: str = "60d") -> dict[str, pd.DataFrame]:
    """Load all data needed for signal backtest in 5 queries.

    Returns dict with keys: buys, sells, scores, prices, roles.
    """
    conn = db.connection()

    # 1. P-buys joined with outcomes + companies + insiders
    buys = pd.read_sql(text("""
        SELECT t.id as txn_id, t.insider_id, t.company_id, t.transaction_date,
               t.filing_date, t.accession_number,
               t.shares, t.price_per_share, t.total_value,
               t.shares_owned_after, t.is_10b5_1_plan,
               c.ticker, c.name as company_name,
               i.name as insider_name, i.is_institution,
               o.excess_return_20d, o.excess_return_60d, o.excess_return_120d,
               o.hit_20d, o.hit_60d, o.hit_120d,
               o.prior_momentum_20d,
               o.stock_return_20d, o.stock_return_60d, o.stock_return_120d
        FROM transactions t
        JOIN trade_outcomes o ON o.transaction_id = t.id
        JOIN companies c ON c.id = t.company_id
        JOIN insiders i ON i.id = t.insider_id
        WHERE t.is_discretionary = true
          AND i.is_institution = false
          AND (t.is_common_stock IS NULL OR t.is_common_stock = true)
    """), conn)

    # Deduplicate multi-lot same-day filings: an insider buying the same stock
    # in multiple lots on one day is ONE event. Sum shares/value, keep first outcome.
    if not buys.empty:
        buys = buys.sort_values("txn_id")  # deterministic: keep first lot's outcome
        agg_cols = {
            "txn_id": "first",
            "filing_date": "first",
            "accession_number": "first",
            "shares": "sum",
            "price_per_share": "first",
            "total_value": "sum",
            "shares_owned_after": "last",
            "is_10b5_1_plan": "first",
            "ticker": "first",
            "company_name": "first",
            "insider_name": "first",
            "is_institution": "first",
            "excess_return_20d": "first",
            "excess_return_60d": "first",
            "excess_return_120d": "first",
            "hit_20d": "first",
            "hit_60d": "first",
            "hit_120d": "first",
            "prior_momentum_20d": "first",
            "stock_return_20d": "first",
            "stock_return_60d": "first",
            "stock_return_120d": "first",
        }
        buys = (
            buys.groupby(["insider_id", "company_id", "transaction_date"], as_index=False)
            .agg(agg_cols)
        )

    # 2. S/F transactions for same-day sell classification
    sells = pd.read_sql(text("""
        SELECT DISTINCT insider_id, company_id, transaction_date
        FROM transactions
        WHERE transaction_code IN ('S', 'F')
    """), conn)

    # 3. Insider scores with credibility_tier
    scores = pd.read_sql(text("""
        SELECT insider_id, skill_score, credibility_tier,
               bayesian_hit_rate, num_discretionary_buys,
               credibility_weight
        FROM insider_scores
        WHERE company_id IS NULL
    """), conn)

    # 4. Price data for drawdown computation + liquidity analysis.
    prices = pd.read_sql(text("""
        SELECT ticker, date, adj_close, volume
        FROM price_data
        ORDER BY ticker, date
    """), conn)

    # 5. InsiderRole for role-based analysis
    roles = pd.read_sql(text("""
        SELECT insider_id, company_id, role_title, is_officer, is_director,
               is_ten_percent_owner, first_filing_date
        FROM insider_roles
    """), conn)

    return {
        "buys": buys,
        "sells": sells,
        "scores": scores,
        "prices": prices,
        "roles": roles,
    }


# ---------------------------------------------------------------------------
# Signal computation — all vectorized pandas
# ---------------------------------------------------------------------------

def compute_filing_lag_flags(buys: pd.DataFrame) -> pd.DataFrame:
    """Compute filing lag (days between transaction and filing) and bucket.

    Research basis: Ozlen & Batumoglu (2025) — 70-80% of alpha dissipates
    between transaction and filing date. Shorter gaps may indicate urgency.

    Returns buys with added columns: filing_lag_days, filing_lag_bucket.
    """
    buys = buys.copy()
    txn_dates = pd.to_datetime(buys["transaction_date"])
    file_dates = pd.to_datetime(buys["filing_date"])
    buys["filing_lag_days"] = (file_dates - txn_dates).dt.days
    # Clamp negative lags (data errors) to 0
    buys["filing_lag_days"] = buys["filing_lag_days"].clip(lower=0)
    buys["filing_lag_bucket"] = pd.cut(
        buys["filing_lag_days"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=["same_day", "fast", "normal", "slow"],
    )
    return buys


def compute_routine_flags(buys: pd.DataFrame) -> pd.DataFrame:
    """Classify insiders as routine vs opportunistic traders.

    Research basis: Cohen, Malloy & Pomorski (2012) — insiders who trade in the
    same calendar month for 3+ consecutive years are "routine". All predictive
    power is in the opportunistic (non-routine) group.

    An insider is routine if ANY of their trading months shows 3+ consecutive
    years of activity.

    Returns buys with added column: is_routine.
    """
    buys = buys.copy()
    txn_dates = pd.to_datetime(buys["transaction_date"])
    buys["_txn_year"] = txn_dates.dt.year
    buys["_txn_month"] = txn_dates.dt.month

    # For each insider, find unique (year, month) pairs they traded in
    insider_months = (
        buys.groupby(["insider_id", "_txn_year", "_txn_month"])
        .size()
        .reset_index(name="_count")
    )

    routine_insiders = set()
    for insider_id, grp in insider_months.groupby("insider_id"):
        # For each calendar month this insider has traded in
        for month, month_grp in grp.groupby("_txn_month"):
            years = sorted(month_grp["_txn_year"].unique())
            if len(years) < 3:
                continue
            # Check for 3+ consecutive years
            consecutive = 1
            for i in range(1, len(years)):
                if years[i] == years[i - 1] + 1:
                    consecutive += 1
                    if consecutive >= 3:
                        routine_insiders.add(insider_id)
                        break
                else:
                    consecutive = 1
            if insider_id in routine_insiders:
                break

    buys["is_routine"] = buys["insider_id"].isin(routine_insiders)
    buys.drop(columns=["_txn_year", "_txn_month"], inplace=True)
    return buys


def compute_liquidity_flags(buys: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 20-day average volume and liquidity quintile.

    Research basis: 2024 ScienceDirect study — insider trading returns are
    negatively correlated with stock liquidity. Signal is stronger in less
    liquid names.

    Returns buys with added columns: avg_volume_20d, liquidity_quintile.
    """
    buys = buys.copy()

    if "volume" not in prices.columns:
        buys["avg_volume_20d"] = np.nan
        buys["liquidity_quintile"] = np.nan
        return buys

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    # Build per-ticker volume arrays
    vol_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, grp in prices.groupby("ticker"):
        vol_index[ticker] = (grp["date"].values, grp["volume"].values.astype(float))

    avg_vols = []
    for _, row in buys.iterrows():
        ticker = row["ticker"]
        txn_date = pd.Timestamp(row["transaction_date"])
        if ticker not in vol_index:
            avg_vols.append(np.nan)
            continue
        dates, volumes = vol_index[ticker]
        txn_dt64 = np.datetime64(txn_date)
        idx = np.searchsorted(dates, txn_dt64, side="right") - 1
        if idx < 19:  # need at least 20 days
            avg_vols.append(np.nan)
            continue
        avg_vol = float(np.mean(volumes[idx - 19 : idx + 1]))
        avg_vols.append(avg_vol)

    buys["avg_volume_20d"] = avg_vols

    # Compute quintiles across all buys with valid volume
    valid_mask = buys["avg_volume_20d"].notna()
    buys["liquidity_quintile"] = np.nan
    if valid_mask.sum() >= 5:
        buys.loc[valid_mask, "liquidity_quintile"] = pd.qcut(
            buys.loc[valid_mask, "avg_volume_20d"],
            5, labels=[1, 2, 3, 4, 5], duplicates="drop",
        ).astype(float)

    return buys


def compute_cluster_flags(buys: pd.DataFrame, roles: pd.DataFrame,
                          causal: bool = False) -> pd.DataFrame:
    """Compute cluster buying and role diversity flags.

    Research basis: Bettis/Vickrey — clustered buying by multiple insiders is
    more predictive. Role diversity in a cluster (CEO + CFO + Director) signals
    broader internal consensus.

    Args:
        causal: If True, only count OTHER buys on or before this buy's
            transaction date (within the window) — no forward look-ahead. Use
            this when the cluster flag feeds a tradeable signal. The default
            (False) uses the symmetric ±window and is for descriptive analysis.

    Returns buys with added columns: cluster_size, is_cluster,
    cluster_role_diversity, cluster_has_mixed_roles, cluster_has_ceo_and_cfo.
    """
    from form4lab.config import settings as _settings
    from form4lab.utils import is_ceo, is_cfo
    window_days = _settings.signal.cluster_window_days

    buys = buys.copy()
    txn_dates = pd.to_datetime(buys["transaction_date"])
    buys["_txn_dt"] = txn_dates

    # Build role lookups: (insider_id, company_id) -> category and -> title
    role_map: dict[tuple, str] = {}
    title_map: dict[tuple, str] = {}
    if not roles.empty:
        for _, r in roles.iterrows():
            key = (r["insider_id"], r["company_id"])
            if r.get("is_officer"):
                role_map[key] = "officer"
            elif r.get("is_director"):
                role_map[key] = "director"
            else:
                role_map[key] = "10pct_owner"
            title_map[key] = r.get("role_title") or ""

    cluster_sizes = []
    role_diversities = []
    has_ceo_and_cfo = []

    for idx, row in buys.iterrows():
        company_id = row["company_id"]
        insider_id = row["insider_id"]
        txn_dt = row["_txn_dt"]

        # Find other buys at the same company within the window.
        # Causal mode counts only buys on/before this one (no look-ahead).
        delta_days = (txn_dt - buys["_txn_dt"]).dt.days
        if causal:
            window_mask = (delta_days >= 0) & (delta_days <= window_days)
        else:
            window_mask = delta_days.abs() <= window_days
        same_company = buys[
            (buys["company_id"] == company_id) &
            (buys["insider_id"] != insider_id) &
            window_mask
        ]
        unique_insiders = same_company["insider_id"].nunique()
        cluster_size = unique_insiders + 1  # include self

        # Role diversity + CEO/CFO co-presence for the cluster
        if cluster_size >= 2:
            cluster_insider_ids = list(same_company["insider_id"].unique()) + [insider_id]
            roles_in_cluster = set()
            titles_in_cluster = []
            for iid in cluster_insider_ids:
                role = role_map.get((iid, company_id))
                if role:
                    roles_in_cluster.add(role)
                titles_in_cluster.append(title_map.get((iid, company_id), ""))
            role_diversity = len(roles_in_cluster)
            ceo_and_cfo = (
                any(is_ceo(t) for t in titles_in_cluster) and
                any(is_cfo(t) for t in titles_in_cluster)
            )
        else:
            role_diversity = 0
            ceo_and_cfo = False

        cluster_sizes.append(cluster_size)
        role_diversities.append(role_diversity)
        has_ceo_and_cfo.append(ceo_and_cfo)

    buys["cluster_size"] = cluster_sizes
    buys["is_cluster"] = buys["cluster_size"] >= 2
    buys["cluster_role_diversity"] = role_diversities
    buys["cluster_has_mixed_roles"] = buys["cluster_role_diversity"] >= 2
    buys["cluster_has_ceo_and_cfo"] = has_ceo_and_cfo
    buys.drop(columns=["_txn_dt"], inplace=True)
    return buys


def compute_drawdown_flags(buys: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Flag buys where the stock had a <=−15% return over the prior 60 trading days.

    Returns buys with added columns: prior_return_60td, drawdown_flag, drawdown_severity.
    """
    buys = buys.copy()

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    # Build per-ticker price arrays
    prior_returns = {}
    for ticker, grp in prices.groupby("ticker"):
        dates = grp["date"].values
        closes = grp["adj_close"].values
        prior_returns[ticker] = (dates, closes)

    results_60td = []
    results_5td = []
    for _, row in buys.iterrows():
        ticker = row["ticker"]
        txn_date = pd.Timestamp(row["transaction_date"])

        if ticker not in prior_returns:
            results_60td.append(np.nan)
            results_5td.append(np.nan)
            continue

        dates, closes = prior_returns[ticker]
        txn_dt64 = np.datetime64(txn_date)

        # Find T+0 index (nearest trading day on or before transaction)
        idx0 = np.searchsorted(dates, txn_dt64, side="right") - 1
        if idx0 < 0:
            results_60td.append(np.nan)
            results_5td.append(np.nan)
            continue

        # Look back 60 trading days
        idx_back = idx0 - 60
        if idx_back < 0:
            results_60td.append(np.nan)
        else:
            ret = (closes[idx0] / closes[idx_back]) - 1
            results_60td.append(float(ret))

        # 5-trading-day momentum
        idx_5 = idx0 - 5
        if idx_5 < 0:
            results_5td.append(np.nan)
        else:
            ret_5 = (closes[idx0] / closes[idx_5]) - 1
            results_5td.append(float(ret_5))

    buys["prior_return_60td"] = results_60td
    buys["short_momentum_5d"] = results_5td
    buys["drawdown_flag"] = buys["prior_return_60td"] <= -0.15
    buys["still_falling"] = buys["drawdown_flag"] & (buys["short_momentum_5d"] < 0)
    buys["drawdown_severity"] = pd.cut(
        buys["prior_return_60td"],
        bins=[-np.inf, -0.40, -0.25, -0.15, -0.05, 0.10, np.inf],
        labels=["crash", "severe", "moderate_dd", "moderate_dip", "neutral", "strength"],
    )
    return buys


def compute_firsttime_flags(buys: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Flag an insider's first observed open-market buy, and tenure context.

    Behavioral thesis: a long-tenured insider who has *never* bought on the open
    market, suddenly buying, is a regime-change signal — distinct from a routine
    adder. We approximate "first ever" by the first P-buy we observe for that
    insider in the backfill window (look-ahead-free: ordered by transaction_date,
    each buy only knows about earlier buys). Tenure comes from the earliest
    InsiderRole.first_filing_date for that insider.

    Returns buys with added columns: is_first_buy, tenure_days,
    first_time_conviction (is_first_buy AND tenure >= 2 years).

    NOTE: "first observed" is bounded by the backfill window, so a buy at the
    very start of the data may be mislabeled first. Footnote this in any report.

    NOTE: the buys frame comes from an INNER JOIN against trade_outcomes (see
    load_backtest_data), so a genuinely-first buy without a matured outcome
    yet is invisible here — a later repeat buy can then be mislabeled
    is_first_buy=True at the frame's recent edge (live has full history and
    no such gap).
    """
    buys = buys.copy()
    txn = pd.to_datetime(buys["transaction_date"])

    # First observed buy per insider (causal: earlier buys only).
    order = txn.argsort(kind="mergesort")
    cum = (
        pd.Series(1, index=buys.index)
        .iloc[order]
        .groupby(buys["insider_id"].iloc[order].values)
        .cumcount()
    )
    is_first = pd.Series(False, index=buys.index)
    is_first.iloc[order] = (cum.values == 0)
    buys["is_first_buy"] = is_first.values

    # Tenure from earliest role first_filing_date per insider.
    if not roles.empty and "first_filing_date" in roles.columns:
        ff = roles.dropna(subset=["first_filing_date"]).copy()
        ff["first_filing_date"] = pd.to_datetime(ff["first_filing_date"])
        earliest = ff.groupby("insider_id")["first_filing_date"].min()
        tenure_start = buys["insider_id"].map(earliest)
        buys["tenure_days"] = (txn - tenure_start).dt.days
    else:
        buys["tenure_days"] = np.nan

    buys["first_time_conviction"] = (
        buys["is_first_buy"] & (buys["tenure_days"].fillna(0) >= 730)
    )
    return buys


# yfinance sector names (as stored on companies.sector) -> SPDR sector ETFs.
# NOTE: keys here are the yfinance taxonomy ("Financial Services", "Consumer
# Cyclical", ...), NOT the GICS-ish names used by price_fetcher.SECTOR_ETF_MAP.
YF_SECTOR_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
