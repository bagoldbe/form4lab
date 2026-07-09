"""Live (per-transaction, DB-backed) feature functions and the FeatureView
implementations consumed by Strategy.classify().

Each live feature has a vectorized counterpart in form4lab/scoring/flags.py
(compute_*_flags); tests/test_feature_parity.py pins the pairs to each other.
"""
from datetime import date, timedelta

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from form4lab.models.transaction import Transaction
from form4lab.models.insider import Insider
from form4lab.models.price import PriceData
from form4lab.config import settings as _settings

_sig_cfg = _settings.signal

ROLE_WEIGHTS = {
    "CEO": 1.0, "CFO": 0.95, "COO": 0.9, "CTO": 0.85,
    "President": 0.9, "EVP": 0.8, "SVP": 0.75, "VP": 0.7,
    "Director": 0.6, "10% Owner": 0.5, "Other Officer": 0.65,
    "Other": 0.5,
}

# Longer-form title patterns mapped to their weights, checked via substring
_TITLE_PATTERNS = [
    ("Chief Executive Officer", 1.0),
    ("Chief Financial Officer", 0.95),
    ("Chief Operating Officer", 0.9),
    ("Chief Technology Officer", 0.85),
    ("Director", 0.6),
    ("President", 0.9),
    ("Executive Vice President", 0.8),
    ("Senior Vice President", 0.75),
    ("Vice President", 0.7),
    ("10% Owner", 0.5),
    ("Other Officer", 0.65),
    ("CEO", 1.0),
    ("CFO", 0.95),
    ("COO", 0.9),
    ("CTO", 0.85),
    ("EVP", 0.8),
    ("SVP", 0.75),
    ("VP", 0.7),
]


def get_role_weight(role_title: str) -> float:
    """Get weight for an insider's role. Matches partial strings.

    Checks longer-form titles first to avoid false positives from
    short abbreviations (e.g., 'CTO' matching inside 'Director').
    """
    if not role_title:
        return _sig_cfg.default_role_weight
    role_upper = role_title.upper()
    for pattern, weight in _TITLE_PATTERNS:
        if pattern.upper() in role_upper:
            return weight
    return _sig_cfg.default_role_weight


def get_insider_median_value(insider_id: int, db: Session) -> float:
    """Get median transaction value for an insider's history.

    Deduplicates same-day lots by summing their values per date, so that
    multi-lot filings count as one event with the combined dollar amount.
    """
    from collections import defaultdict

    txns = db.query(Transaction.transaction_date, Transaction.total_value).filter(
        Transaction.insider_id == insider_id,
        Transaction.is_discretionary == True,  # noqa: E712
        Transaction.total_value.isnot(None),
        Transaction.total_value > 0,
    ).all()
    if not txns:
        return 50000.0  # default
    daily_sums: dict = defaultdict(float)
    for txn_date, val in txns:
        daily_sums[txn_date] += val
    values = sorted(daily_sums.values())
    mid = len(values) // 2
    return values[mid]


def detect_cluster(company_id: int, transaction_date: date, db: Session) -> dict:
    """Find other unique person insider buys at the same company within 7 days.

    Returns dict with 'unique_insiders' count and 'transactions' list.
    """
    window_start = transaction_date - timedelta(days=_sig_cfg.cluster_window_days)
    window_end = transaction_date + timedelta(days=_sig_cfg.cluster_window_days)

    cluster_txns = (
        db.query(Transaction)
        .join(Insider, Transaction.insider_id == Insider.id)
        .filter(
            Transaction.company_id == company_id,
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.transaction_date >= window_start,
            Transaction.transaction_date <= window_end,
            Insider.is_institution == False,  # noqa: E712
        )
        .all()
    )

    unique_insider_ids = set(t.insider_id for t in cluster_txns)

    return {
        "unique_insiders": len(unique_insider_ids),
        "transactions": cluster_txns,
    }


def detect_short_momentum(ticker: str, transaction_date: date, db: Session) -> float | None:
    """Compute 5-trading-day return ending on transaction_date.

    Returns the return as a float, or None if insufficient data (< 6 prices).
    """
    prices = (
        db.query(PriceData.adj_close)
        .filter(PriceData.ticker == ticker, PriceData.date <= transaction_date)
        .order_by(PriceData.date.desc())
        .limit(10)
        .all()
    )
    if len(prices) < 6:
        return None
    return (prices[0].adj_close / prices[5].adj_close) - 1


def detect_drawdown(ticker: str, transaction_date: date, db: Session) -> dict | None:
    """Check if the stock is in a >= 15% drawdown over prior 60 trading days.

    Returns dict with prior_return_60td, is_drawdown, still_falling, and
    short_momentum_5d if drawdown detected, else None.
    """
    prices = (
        db.query(PriceData.date, PriceData.adj_close)
        .filter(
            PriceData.ticker == ticker,
            PriceData.date <= transaction_date,
        )
        .order_by(PriceData.date.desc())
        .limit(80)
        .all()
    )

    if len(prices) < 61:
        return None

    price_now = prices[0].adj_close
    price_60td = prices[60].adj_close
    prior_return = (price_now / price_60td) - 1

    if prior_return <= -0.15:
        short_mom = detect_short_momentum(ticker, transaction_date, db)
        return {
            "prior_return_60td": float(prior_return),
            "is_drawdown": True,
            "still_falling": short_mom is not None and short_mom < 0,
            "short_momentum_5d": float(short_mom) if short_mom is not None else None,
        }
    return None


def detect_cluster_sell(company_id: int, transaction_date: date, db: Session) -> dict:
    """Find other unique-person insider sells at the same company within 7 days.

    Returns dict with 'unique_insiders' count and 'transactions' list.
    """
    window_start = transaction_date - timedelta(days=_sig_cfg.cluster_window_days)
    window_end = transaction_date + timedelta(days=_sig_cfg.cluster_window_days)

    cluster_txns = (
        db.query(Transaction)
        .join(Insider, Transaction.insider_id == Insider.id)
        .filter(
            Transaction.company_id == company_id,
            Transaction.transaction_code == "S",
            Transaction.transaction_date >= window_start,
            Transaction.transaction_date <= window_end,
            Insider.is_institution == False,  # noqa: E712
        )
        .all()
    )

    unique_insider_ids = set(t.insider_id for t in cluster_txns)

    return {
        "unique_insiders": len(unique_insider_ids),
        "transactions": cluster_txns,
    }


def compute_sell_pct(
    insider_id: int, company_id: int, transaction_date: date, db: Session
) -> float | None:
    """Compute total shares sold as a fraction of pre-sale holdings.

    Sums all same-day S-transactions for this insider+company, then divides
    by (shares_owned_after + shares_sold) to get the percentage disposed.
    Returns None if we can't determine the ratio.
    """
    same_day_sells = (
        db.query(Transaction)
        .filter(
            Transaction.insider_id == insider_id,
            Transaction.company_id == company_id,
            Transaction.transaction_code == "S",
            Transaction.transaction_date == transaction_date,
        )
        .all()
    )
    if not same_day_sells:
        return None

    total_shares_sold = sum(t.shares for t in same_day_sells if t.shares)
    # Use shares_owned_after from the last lot to infer pre-sale holdings
    last_txn = same_day_sells[-1]
    if last_txn.shares_owned_after is not None and total_shares_sold > 0:
        pre_sale = last_txn.shares_owned_after + total_shares_sold
        if pre_sale > 0:
            return total_shares_sold / pre_sale
    return None


class LiveFeatureView:
    """Lazily computes and caches live features for one transaction."""

    def __init__(self, db, txn, tier: str, skill_score: float, role_title: str | None,
                 score=None):
        self._db = db
        self._txn = txn
        self._cache: dict = {"tier": tier, "skill_score": skill_score,
                             "role_title": role_title, "score_row": score}

    def get(self, name, default=None):
        if name in self._cache:
            return self._cache[name]
        val = self._compute(name, default)
        self._cache[name] = val
        return val

    def put(self, name, value):
        self._cache[name] = value

    def _compute(self, name, default):
        t = self._txn
        if name in ("drawdown", "is_drawdown", "still_falling"):
            dd = self.get("drawdown_raw")
            if name == "drawdown":
                return dd
            if name == "is_drawdown":
                return dd is not None
            return bool(dd and dd.get("still_falling"))
        if name == "drawdown_raw":
            from form4lab.models.company import Company
            company = self._db.get(Company, t.company_id)
            return detect_drawdown(company.ticker, t.transaction_date, self._db)
        if name == "cluster":
            return detect_cluster(t.company_id, t.transaction_date, self._db)
        if name == "cluster_unique_insiders":
            return self.get("cluster")["unique_insiders"]
        if name == "insider_median_value":
            return get_insider_median_value(t.insider_id, self._db)
        if name == "sell_pct":
            return compute_sell_pct(t.insider_id, t.company_id, t.transaction_date, self._db)
        if name == "cluster_sell_unique":
            return detect_cluster_sell(t.company_id, t.transaction_date, self._db)["unique_insiders"]
        if name == "is_10b5_1_plan":
            return bool(t.is_10b5_1_plan) if t.is_10b5_1_plan is not None else False
        if name == "is_first_buy":
            return self._is_first_observed_buy()
        if name == "is_first_time":
            return self._is_first_buy()
        if name == "cluster_member_skill_scores":
            from form4lab.models.score import InsiderScore
            cluster_scores = []
            seen_insider_ids = set()
            for ct in self.get("cluster")["transactions"]:
                if ct.insider_id in seen_insider_ids:
                    continue
                seen_insider_ids.add(ct.insider_id)
                cs = self._db.query(InsiderScore).filter(
                    InsiderScore.insider_id == ct.insider_id,
                    InsiderScore.company_id == None,  # noqa: E711
                ).first()
                if cs:
                    cluster_scores.append(cs.skill_score)
            return cluster_scores
        if name == "insider_name":
            insider = self._db.get(Insider, t.insider_id)
            return insider.name if insider else ""
        if name == "company_name":
            from form4lab.models.company import Company
            company = self._db.get(Company, t.company_id)
            return company.name if company else ""
        return default

    def _is_first_buy(self) -> bool:
        """True if this is the insider's first-ever discretionary buy at this
        company AND they've been an insider (per InsiderRole.first_filing_date)
        for at least 2 years. Shared by the "is_first_buy" and legacy
        "is_first_time" branches of _compute so the two stay identical.
        """
        from form4lab.models.insider import InsiderRole
        t = self._txn
        prior_buys = self._db.query(Transaction).filter(
            Transaction.insider_id == t.insider_id,
            Transaction.company_id == t.company_id,
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.id != t.id,
        ).count()
        if prior_buys == 0:
            role = self._db.query(InsiderRole).filter(
                InsiderRole.insider_id == t.insider_id,
                InsiderRole.company_id == t.company_id,
            ).first()
            if role and role.first_filing_date:
                years_as_insider = (t.transaction_date - role.first_filing_date).days / 365.25
                if years_as_insider >= 2:
                    return True
        return False

    def _is_first_observed_buy(self) -> bool:
        """True if this insider has zero prior discretionary buys, at ANY
        company, ordered before this transaction.

        This is the live equivalent of the bare `is_first_buy` column built by
        compute_firsttime_flags in form4lab/scoring/flags.py: insider-scoped
        (no company filter) and with NO tenure gate. compute_firsttime_flags
        establishes "prior" causally by argsorting transaction_date with a
        stable (mergesort) sort, then taking each insider's first occurrence
        in that order; ties on transaction_date fall back to whatever row
        order the DataFrame already had. We mirror that with an explicit
        (transaction_date, id) ordering: `id` is assigned in insertion order,
        the closest live analogue to "earlier in the frame" for a same-day
        tie. Feeds ONLY the "is_first_buy" branch of _compute -- "is_first_time"
        keeps calling the company+tenure-gated _is_first_buy() above, unchanged.
        The prior-buys query also mirrors load_backtest_data's `buys` SQL
        filter (is_common_stock IS NULL OR is_common_stock = true, form4lab/
        scoring/flags.py), the same idiom used in alert_service.py and the
        route modules: a prior P-coded buy of NON-common stock (preferred/
        warrant/unit) never reaches compute_firsttime_flags' input frame, so
        it must not count as a "prior buy" here either.

        Two documented, non-silent residual edges (not fixed here because
        they're structural, not bugs):

        1. Outcome-availability window: compute_firsttime_flags only ever
           sees the subset of transactions that made it into
           load_backtest_data's `buys` frame, which INNER JOINs
           trade_outcomes -- a transaction without a computed outcome yet
           (too recent) is invisible to it. So a genuinely-first buy that
           hasn't matured an outcome can leave a later, actual repeat buy
           mislabeled is_first_buy=True on the backtest side, while this
           live method (which has no outcome dependency -- a live signal
           can't wait on a future return) correctly sees the true earlier
           buy and returns False. This is intentional, not a bug to match:
           live must reason over the complete transaction history, and
           compute_firsttime_flags' own docstring already flags "first
           observed" as bounded by whatever window it's loaded over.
        2. Same-day, cross-company tiebreak: in the real load_backtest_data
           pipeline (not these hand-built test fixtures), buys is already
           sorted by (insider_id, company_id, transaction_date) ascending
           before compute_firsttime_flags runs (an artifact of its dedup
           groupby, not a deliberate ordering choice), so a same-insider,
           same-date tie across two different companies breaks on
           company_id there, not on id/arrival order as this method assumes.
           This is a rarer edge than (1) and does not affect any case where
           the two buys fall on different dates.
        """
        t = self._txn
        prior_buys = self._db.query(Transaction).filter(
            Transaction.insider_id == t.insider_id,
            Transaction.is_discretionary == True,  # noqa: E712
            Transaction.id != t.id,
            or_(Transaction.is_common_stock.is_(None), Transaction.is_common_stock == True),  # noqa: E712
            or_(
                Transaction.transaction_date < t.transaction_date,
                and_(
                    Transaction.transaction_date == t.transaction_date,
                    Transaction.id < t.id,
                ),
            ),
        ).count()
        return prior_buys == 0


class RowFeatureView:
    """Backtest FeatureView over a precomputed flags row (pd.Series)."""

    def __init__(self, row, tier: str, skill_score: float):
        self._row = row
        self._tier = tier
        self._skill = skill_score
        self._overlay: dict = {}

    def put(self, name, value):
        self._overlay[name] = value

    def get(self, name, default=None):
        if name in self._overlay:
            return self._overlay[name]
        if name == "tier":
            return self._tier
        if name == "skill_score":
            return self._skill
        if name == "is_drawdown":
            return bool(self._row.get("drawdown_flag", False))
        if name in ("conviction", "cluster_id", "is_first_time") and name not in self._overlay:
            return default
        if name == "cluster_unique_insiders":
            val = self._row.get("cluster_size")
            return default if val is None else val
        if name in ("is_first_buy", "is_10b5_1_plan"):
            val = self._row.get(name)
            # Normalize to native Python bool, matching LiveFeatureView:
            # - is_first_buy's column is always a real True/False, but as a
            #   whole-column numpy bool array, so per-row access yields
            #   numpy.bool_ (or np.int64 once the row itself is a mixed-dtype
            #   Series) instead of Python bool.
            # - is_10b5_1_plan is a raw nullable-boolean passthrough column:
            #   NULL surfaces as Python None (single-value/object-dtype
            #   columns) or float NaN (once the column upcasts to float64
            #   alongside True/False rows) -- `val != val` is True only for
            #   NaN, so this catches both without needing pandas/numpy here.
            #   Live already coalesces None -> False; mirror that.
            if val is None or val != val:
                return False
            return bool(val)
        val = self._row.get(name, default)
        return default if val is None else val
