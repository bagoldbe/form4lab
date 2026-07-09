"""Parity tests pinning the live (per-transaction, DB-backed) detectors in
form4lab/strategy/features.py against their vectorized counterparts in
form4lab/scoring/flags.py.

Only six detectors are paired here (see flags.py's six-function allowlist) — a cluster-parity case covers
detect_cluster vs compute_cluster_flags(causal=True) alongside the
drawdown parity cases.

Two more parity pairs are pinned: is_10b5_1_plan (LiveFeatureView._compute
vs load_backtest_data's raw passthrough column) and is_first_buy
(LiveFeatureView._is_first_buy vs compute_firsttime_flags's is_first_buy
column). Both were FeatureView asymmetries: is_10b5_1_plan already
resolved in the backtest view but not live; is_first_buy resolved in
neither (the live view only had is_first_time, which RowFeatureView
reserves and never answers).

is_first_buy's live and backtest sides can disagree on two axes
(tenure gate, company- vs insider-scope) that the original fixtures never
exercised, and both is_first_buy/is_10b5_1_plan leaked numpy scalars out
of RowFeatureView instead of native bool. Fixed by: a new
LiveFeatureView._is_first_observed_buy() (insider-scoped, no tenure gate
-- see its docstring for the exact mirror of compute_firsttime_flags'
causal ordering and a documented outcome-availability caveat) now backing
the "is_first_buy" branch, with "is_first_time" left pointed at the
untouched, company+tenure-gated _is_first_buy(); and explicit
bool()/NaN-safe normalization in RowFeatureView.get() for both keys. The
tests below include fixtures inside the two divergence regions
(no-tenure-gate, multi-company) plus isinstance(..., bool) / `is True`/`is
False` checks on both sides, so they fail on either axis reopening.
"""
from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.company import Company
from form4lab.models.insider import Insider, InsiderRole
from form4lab.models.outcome import TradeOutcome
from form4lab.models.price import PriceData
from form4lab.models.transaction import Transaction
from form4lab.scoring.flags import (
    compute_cluster_flags,
    compute_drawdown_flags,
    compute_firsttime_flags,
    load_backtest_data,
)
from form4lab.strategy.features import (
    LiveFeatureView,
    RowFeatureView,
    detect_cluster,
    detect_drawdown,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_insider_company(db, ticker):
    insider = Insider(cik=f"CIK-{ticker}", name=f"{ticker} Insider", is_institution=False)
    company = Company(cik=f"CO-{ticker}", ticker=ticker, name=f"{ticker} Corp")
    db.add_all([insider, company])
    db.flush()
    return insider, company


def _add_buy(db, insider, company, txn_date, accession, is_10b5_1_plan=None):
    """A P (discretionary open-market buy) transaction."""
    txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number=accession, filing_date=txn_date,
        transaction_date=txn_date, transaction_code="P",
        shares=500, acquired_or_disposed="A", is_discretionary=True,
        is_10b5_1_plan=is_10b5_1_plan,
    )
    db.add(txn)
    db.flush()
    return txn


def _add_prices(db, ticker, rows):
    """rows: list of (date, price) tuples -> flat-OHLC PriceData rows."""
    for d, price in rows:
        db.add(PriceData(
            ticker=ticker, date=d, open=price, high=price, low=price,
            close=price, adj_close=price, volume=100_000,
        ))
    db.flush()


def _price_series(n, flat_days, start_price, end_price, start=date(2024, 1, 1)):
    """n rows of consecutive calendar days: flat at start_price through index
    `flat_days`, then a linear decline over the remaining
    (n - 1 - flat_days) steps down to end_price.

    Choosing flat_days = n - 61 puts exactly 60 rows in the decline leg, so
    prior_return_60td (and the live 60-trading-day lookback) both compute the
    same start_price -> end_price ratio.
    """
    decline_span = n - 1 - flat_days
    rows = []
    for i in range(n):
        if i <= flat_days:
            price = start_price
        else:
            frac = (i - flat_days) / decline_span
            price = start_price + (end_price - start_price) * frac
        rows.append((start + timedelta(days=i), price))
    return rows


# ---------------------------------------------------------------------------
# Cluster parity (detect_cluster vs compute_cluster_flags(causal=True))
# ---------------------------------------------------------------------------
#
# detect_cluster's live query uses a *symmetric* +/- window (it has no
# look-ahead guard — see its docstring), while compute_cluster_flags(causal=
# True) only counts peers on/before the evaluated buy. The two are only
# guaranteed to agree when the fixture itself is causal-consistent: every
# cluster peer transaction falls on or before the buy being evaluated, so
# the forward half of the live detector's window is empty and the symmetric
# vs. backward-only distinction never bites.

def test_cluster_parity_causal_consistent(db):
    """Insider A buys on day T; B and C (same company) bought on T-2 and T-5
    (both inside the default 7-day window, both on/before T -> causal-
    consistent). detect_cluster's unique-insider count (which includes A's
    own transaction, since its query has no self-exclusion) must equal
    compute_cluster_flags(causal=True)'s cluster_size (unique *other*
    insiders + 1 for self) for A's row.
    """
    insider_a, company = _make_insider_company(db, "CLPAR")
    insider_b = Insider(cik="CIK-CLPAR-B", name="CLPAR Insider B", is_institution=False)
    insider_c = Insider(cik="CIK-CLPAR-C", name="CLPAR Insider C", is_institution=False)
    db.add_all([insider_b, insider_c])
    db.flush()

    buy_date = date(2024, 6, 8)
    date_b = buy_date - timedelta(days=2)
    date_c = buy_date - timedelta(days=5)
    _add_buy(db, insider_a, company, buy_date, "ACC-CLPAR-A")
    _add_buy(db, insider_b, company, date_b, "ACC-CLPAR-B")
    _add_buy(db, insider_c, company, date_c, "ACC-CLPAR-C")

    live = detect_cluster(company.id, buy_date, db)
    assert live["unique_insiders"] == 3  # A (self, same-day) + B + C

    buys = pd.DataFrame([
        {"insider_id": insider_a.id, "company_id": company.id, "transaction_date": buy_date},
        {"insider_id": insider_b.id, "company_id": company.id, "transaction_date": date_b},
        {"insider_id": insider_c.id, "company_id": company.id, "transaction_date": date_c},
    ])
    roles = pd.DataFrame()  # empty: role diversity isn't under test here
    result = compute_cluster_flags(buys, roles, causal=True)
    a_row = result[result["insider_id"] == insider_a.id].iloc[0]

    assert int(a_row["cluster_size"]) == live["unique_insiders"]


def test_cluster_parity_no_peers_is_singleton(db):
    """No other insiders buying nearby -> both report a cluster of 1."""
    insider, company = _make_insider_company(db, "CLSOLO")
    buy_date = date(2024, 6, 8)
    _add_buy(db, insider, company, buy_date, "ACC-CLSOLO-A")

    live = detect_cluster(company.id, buy_date, db)
    assert live["unique_insiders"] == 1

    buys = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "transaction_date": buy_date},
    ])
    result = compute_cluster_flags(buys, pd.DataFrame(), causal=True)
    assert int(result.iloc[0]["cluster_size"]) == live["unique_insiders"] == 1


# ---------------------------------------------------------------------------
# Drawdown parity (detect_drawdown vs compute_drawdown_flags)
# ---------------------------------------------------------------------------

def test_drawdown_parity_deep(db):
    """80 prices, flat then a -20% decline over the final 60 rows -> both True."""
    ticker = "DDDEEP"
    rows = _price_series(n=80, flat_days=19, start_price=100.0, end_price=80.0)
    _add_prices(db, ticker, rows)
    txn_date = rows[-1][0]

    live = detect_drawdown(ticker, txn_date, db)
    assert live is not None
    assert live["is_drawdown"] is True

    buys = pd.DataFrame([{"ticker": ticker, "transaction_date": txn_date}])
    prices = pd.DataFrame([{"ticker": ticker, "date": d, "adj_close": p} for d, p in rows])
    result = compute_drawdown_flags(buys, prices)
    assert bool(result.iloc[0]["drawdown_flag"]) is True


def test_drawdown_parity_shallow(db):
    """Same shape but only a -10% decline (above the -15% threshold) -> both falsy."""
    ticker = "DDSHAL"
    rows = _price_series(n=80, flat_days=19, start_price=100.0, end_price=90.0)
    _add_prices(db, ticker, rows)
    txn_date = rows[-1][0]

    live = detect_drawdown(ticker, txn_date, db)
    assert live is None

    buys = pd.DataFrame([{"ticker": ticker, "transaction_date": txn_date}])
    prices = pd.DataFrame([{"ticker": ticker, "date": d, "adj_close": p} for d, p in rows])
    result = compute_drawdown_flags(buys, prices)
    assert bool(result.iloc[0]["drawdown_flag"]) is False


def test_drawdown_parity_insufficient_history(db):
    """Fewer than 61 prices -> live returns None; vectorized flag is falsy."""
    ticker = "DDSHORT"
    rows = [(date(2024, 1, 1) + timedelta(days=i), 100.0) for i in range(40)]
    _add_prices(db, ticker, rows)
    txn_date = rows[-1][0]

    live = detect_drawdown(ticker, txn_date, db)
    assert live is None

    buys = pd.DataFrame([{"ticker": ticker, "transaction_date": txn_date}])
    prices = pd.DataFrame([{"ticker": ticker, "date": d, "adj_close": p} for d, p in rows])
    result = compute_drawdown_flags(buys, prices)
    assert not bool(result.iloc[0]["drawdown_flag"])


# ---------------------------------------------------------------------------
# is_10b5_1_plan parity (LiveFeatureView._compute vs load_backtest_data's
# raw passthrough SELECT + same-day-lot dedup aggregation)
# ---------------------------------------------------------------------------
#
# Unlike drawdown/cluster, there's no compute_*_flags builder for this flag:
# load_backtest_data's SQL SELECT pulls t.is_10b5_1_plan straight through,
# and the same-day-lot dedup groupby keeps it via agg_cols={"is_10b5_1_plan":
# "first"}. So the realistic backtest-side check is load_backtest_data
# itself -- that's the one place a SELECT or dedup regression could silently
# drop the column -- not a hand-built frame.

def test_is_10b5_1_plan_parity_true(db):
    """A transaction filed under a 10b5-1 plan resolves True in both views,
    as a native Python bool (not a numpy/float passthrough) on both sides.
    """
    insider, company = _make_insider_company(db, "PLANT")
    txn_date = date(2024, 6, 8)
    txn = _add_buy(db, insider, company, txn_date, "ACC-PLANT-A", is_10b5_1_plan=True)
    db.add(TradeOutcome(transaction_id=txn.id))
    db.flush()

    live = LiveFeatureView(db, txn, tier="Insufficient", skill_score=0.0, role_title=None)
    live_val = live.get("is_10b5_1_plan")
    assert live_val is True
    assert live_val == txn.is_10b5_1_plan
    assert isinstance(live_val, bool)

    data = load_backtest_data(db)
    row = data["buys"][data["buys"]["txn_id"] == txn.id].iloc[0]
    assert bool(row["is_10b5_1_plan"]) is True
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_10b5_1_plan")
    assert bt_val == live_val
    assert bt_val is True
    assert isinstance(bt_val, bool)


def test_is_10b5_1_plan_parity_false(db):
    """A transaction with no 10b5-1 plan resolves False in both views, as a
    native Python bool on both sides.
    """
    insider, company = _make_insider_company(db, "NOPLAN")
    txn_date = date(2024, 6, 8)
    txn = _add_buy(db, insider, company, txn_date, "ACC-NOPLAN-A", is_10b5_1_plan=False)
    db.add(TradeOutcome(transaction_id=txn.id))
    db.flush()

    live = LiveFeatureView(db, txn, tier="Insufficient", skill_score=0.0, role_title=None)
    live_val = live.get("is_10b5_1_plan")
    assert live_val is False
    assert live_val == txn.is_10b5_1_plan
    assert isinstance(live_val, bool)

    data = load_backtest_data(db)
    row = data["buys"][data["buys"]["txn_id"] == txn.id].iloc[0]
    assert bool(row["is_10b5_1_plan"]) is False
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_10b5_1_plan")
    assert bt_val == live_val
    assert bt_val is False
    assert isinstance(bt_val, bool)


def test_is_10b5_1_plan_parity_null(db):
    """A transaction where the filer left the 10b5-1 flag NULL (~75% of real
    buys) resolves False in BOTH views,
    as a native Python bool. Live already coalesced None -> False;
    RowFeatureView previously returned the raw NULL straight through (as
    Python None for a single-row column, or NaN once the column upcasts to
    float64 alongside True/False rows) instead of matching live's False.
    """
    insider, company = _make_insider_company(db, "PLANNULL")
    txn_date = date(2024, 6, 8)
    txn = _add_buy(db, insider, company, txn_date, "ACC-PLANNULL-A", is_10b5_1_plan=None)
    db.add(TradeOutcome(transaction_id=txn.id))
    db.flush()
    assert txn.is_10b5_1_plan is None  # confirms the fixture is genuinely NULL

    live = LiveFeatureView(db, txn, tier="Insufficient", skill_score=0.0, role_title=None)
    live_val = live.get("is_10b5_1_plan")
    assert live_val is False
    assert isinstance(live_val, bool)

    data = load_backtest_data(db)
    row = data["buys"][data["buys"]["txn_id"] == txn.id].iloc[0]
    raw = row["is_10b5_1_plan"]
    assert raw is None or raw != raw  # None (object dtype) or NaN (float64 dtype)
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_10b5_1_plan")
    assert bt_val == live_val
    assert bt_val is False
    assert isinstance(bt_val, bool)


# ---------------------------------------------------------------------------
# is_first_buy parity (LiveFeatureView._is_first_observed_buy vs
# compute_firsttime_flags's is_first_buy column)
# ---------------------------------------------------------------------------
#
# compute_firsttime_flags IS a compute_*_flags builder (like the drawdown/
# cluster ones above), so these hand-build minimal buys/roles frames the same
# way the drawdown/cluster tests hand-build prices/buys.
#
# Genuine live/backtest parity: is_first_buy is now
# insider-scoped with NO tenure gate on both sides, matching
# compute_firsttime_flags's bare column exactly -- see
# LiveFeatureView._is_first_observed_buy's docstring for the exact mirror of
# its causal (transaction_date, id) ordering and a documented,
# outcome-availability backtest-realism caveat.
#
# The first two fixtures below (_true, _false) sit in the region where the
# OLD company+tenure-gated _is_first_buy() and the new insider-scoped method
# happened to agree -- the region where the two definitions coincide.
# The next two (_no_tenure_gate, _insider_scoped_not_company_scoped) are
# built specifically inside the two axes where they used to diverge, so they
# fail if either axis regresses. "is_first_time" is intentionally left on the
# OLD company+tenure logic (a separate, reserved feature name --
# RowFeatureView never answers it) and is asserted to diverge from
# is_first_buy in exactly those two new fixtures, proving the branches are
# now decoupled rather than coincidentally aliased.

def test_is_first_buy_parity_true(db):
    """An insider's first-ever (and only) discretionary buy at a company,
    filed >=2 years after their first_filing_date, resolves True in both
    views.
    """
    insider, company = _make_insider_company(db, "FBFIRST")
    first_filing = date(2020, 1, 1)
    db.add(InsiderRole(
        insider_id=insider.id, company_id=company.id, role_title="Director",
        first_filing_date=first_filing,
    ))
    db.flush()

    txn_date = date(2023, 1, 1)  # 3 years after first_filing_date
    years_as_insider = (txn_date - first_filing).days / 365.25
    assert years_as_insider >= 2  # confirms the fixture exercises the tenure gate

    txn = _add_buy(db, insider, company, txn_date, "ACC-FBFIRST-A")

    live = LiveFeatureView(db, txn, tier="Insufficient", skill_score=0.0, role_title="Director")
    live_val = live.get("is_first_buy")
    assert live_val is True
    assert isinstance(live_val, bool)
    # Both fixtures here sit in a single-company, single-buy region where the
    # OLD company+tenure-gated is_first_time logic happens to agree with the
    # new insider-scoped is_first_buy -- see test_is_first_buy_parity_no_tenure_gate
    # and _insider_scoped_not_company_scoped below for the regions where they
    # (correctly) diverge.
    assert live.get("is_first_time") == live_val

    buys = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "transaction_date": txn_date},
    ])
    roles = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "first_filing_date": first_filing},
    ])
    result = compute_firsttime_flags(buys, roles)
    row = result.iloc[0]
    assert bool(row["is_first_buy"]) is True
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_first_buy")
    assert bt_val == live_val
    assert bt_val is True
    assert isinstance(bt_val, bool)


def test_is_first_buy_parity_false(db):
    """A second (repeat) buy for the same insider+company resolves False in
    both views: live because prior_buys != 0 short-circuits before the
    tenure check runs; backtest because compute_firsttime_flags's causal
    first-buy-per-insider assignment marks only the earlier row True.
    """
    insider, company = _make_insider_company(db, "FBREPEAT")
    first_filing = date(2020, 1, 1)
    db.add(InsiderRole(
        insider_id=insider.id, company_id=company.id, role_title="Director",
        first_filing_date=first_filing,
    ))
    db.flush()

    first_buy_date = date(2023, 1, 1)
    repeat_buy_date = date(2023, 6, 1)
    _add_buy(db, insider, company, first_buy_date, "ACC-FBREPEAT-A")
    repeat_txn = _add_buy(db, insider, company, repeat_buy_date, "ACC-FBREPEAT-B")

    live = LiveFeatureView(db, repeat_txn, tier="Insufficient", skill_score=0.0,
                            role_title="Director")
    live_val = live.get("is_first_buy")
    assert live_val is False
    assert isinstance(live_val, bool)
    assert live.get("is_first_time") == live_val

    buys = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "transaction_date": first_buy_date},
        {"insider_id": insider.id, "company_id": company.id, "transaction_date": repeat_buy_date},
    ])
    roles = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "first_filing_date": first_filing},
    ])
    result = compute_firsttime_flags(buys, roles)
    repeat_row = result.iloc[1]  # inserted second -> the later, repeat buy
    assert bool(repeat_row["is_first_buy"]) is False
    backtest = RowFeatureView(repeat_row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_first_buy")
    assert bt_val == live_val
    assert bt_val is False
    assert isinstance(bt_val, bool)


def test_is_first_buy_parity_no_tenure_gate(db):
    """An insider's first-ever (and only) discretionary buy, filed LESS than
    2 years after their first_filing_date, resolves True in BOTH views --
    proving is_first_buy has NO tenure gate (unlike the reserved
    is_first_time branch, which stays tenure-gated and correctly diverges
    here: False).
    """
    insider, company = _make_insider_company(db, "FBNOTEN")
    first_filing = date(2023, 6, 1)
    db.add(InsiderRole(
        insider_id=insider.id, company_id=company.id, role_title="Director",
        first_filing_date=first_filing,
    ))
    db.flush()

    txn_date = date(2024, 1, 1)  # ~7 months of tenure, well under 2 years
    years_as_insider = (txn_date - first_filing).days / 365.25
    assert years_as_insider < 2  # confirms the fixture is inside the tenure-gate axis

    txn = _add_buy(db, insider, company, txn_date, "ACC-FBNOTEN-A")

    live = LiveFeatureView(db, txn, tier="Insufficient", skill_score=0.0, role_title="Director")
    live_val = live.get("is_first_buy")
    assert live_val is True
    assert isinstance(live_val, bool)
    # is_first_time keeps the OLD tenure-gated logic: <2yr tenure -> False.
    # This is the fix's intended decoupling, not a regression.
    assert live.get("is_first_time") is False

    buys = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "transaction_date": txn_date},
    ])
    roles = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company.id, "first_filing_date": first_filing},
    ])
    result = compute_firsttime_flags(buys, roles)
    row = result.iloc[0]
    assert bool(row["is_first_buy"]) is True
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_first_buy")
    assert bt_val == live_val
    assert bt_val is True
    assert isinstance(bt_val, bool)


def test_is_first_buy_parity_insider_scoped_not_company_scoped(db):
    """An insider buys at company A, then later makes their first-ever buy at
    company B (with >=2yr tenure at B by that point). Both views must say
    False for the B buy -- proving is_first_buy is scoped to the INSIDER
    (any company), not the company. (The reserved is_first_time branch stays
    company-scoped and correctly diverges here: True, since it's the
    insider's first buy *at company B specifically*, past the tenure gate.)
    """
    insider = Insider(cik="CIK-FBSCOPE", name="FBSCOPE Insider", is_institution=False)
    company_a = Company(cik="CO-FBSCOPE-A", ticker="FBSCA", name="FBSCOPE A Corp")
    company_b = Company(cik="CO-FBSCOPE-B", ticker="FBSCB", name="FBSCOPE B Corp")
    db.add_all([insider, company_a, company_b])
    db.flush()

    first_filing_b = date(2020, 1, 1)
    db.add(InsiderRole(
        insider_id=insider.id, company_id=company_b.id, role_title="Director",
        first_filing_date=first_filing_b,
    ))
    db.flush()

    date_a = date(2022, 1, 1)  # earlier buy, at company A
    date_b = date(2023, 1, 1)  # insider's first buy AT company B, but not first overall
    years_as_insider_b = (date_b - first_filing_b).days / 365.25
    assert years_as_insider_b >= 2  # isolates company-scope, not tenure, as the cause

    _add_buy(db, insider, company_a, date_a, "ACC-FBSCOPE-A")
    txn_b = _add_buy(db, insider, company_b, date_b, "ACC-FBSCOPE-B")

    live = LiveFeatureView(db, txn_b, tier="Insufficient", skill_score=0.0, role_title="Director")
    live_val = live.get("is_first_buy")
    assert live_val is False
    assert isinstance(live_val, bool)
    # is_first_time keeps the OLD company-scoped logic: at company B alone,
    # this IS the insider's first buy with >=2yr tenure -- True, correctly
    # diverging from the insider-scoped is_first_buy.
    assert live.get("is_first_time") is True

    buys = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company_a.id, "transaction_date": date_a},
        {"insider_id": insider.id, "company_id": company_b.id, "transaction_date": date_b},
    ])
    roles = pd.DataFrame([
        {"insider_id": insider.id, "company_id": company_b.id, "first_filing_date": first_filing_b},
    ])
    result = compute_firsttime_flags(buys, roles)
    row_b = result[result["company_id"] == company_b.id].iloc[0]
    assert bool(row_b["is_first_buy"]) is False
    backtest = RowFeatureView(row_b, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_first_buy")
    assert bt_val == live_val
    assert bt_val is False
    assert isinstance(bt_val, bool)


def test_is_first_buy_parity_excludes_noncommon_prior_buy(db):
    """A third live/backtest divergence axis: an insider's
    earlier discretionary buy of NON-common stock (preferred/warrant/unit --
    is_discretionary=True, is_common_stock=False is a real, reachable
    combination) must not count as a "prior buy" on either side.

    load_backtest_data's buys SQL restricts to `(is_common_stock IS NULL OR
    is_common_stock = true)` BEFORE compute_firsttime_flags ever runs (see
    flags.py), so the preferred-stock buy never reaches the backtest frame at
    all -- it isn't merely down-weighted, it's invisible. The live query in
    _is_first_observed_buy must mirror that same filter, or a later, first
    COMMON-stock buy would resolve is_first_buy=False live (counting the
    preferred buy as "prior") while resolving True in the backtest (which
    never saw it) -- a permanent divergence, since a past preferred-stock buy
    never ages out or self-heals.
    """
    insider, company = _make_insider_company(db, "FBNONCOMM")

    preferred_date = date(2022, 1, 1)  # earlier: a P-code buy of NON-common stock
    preferred_txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-FBNONCOMM-PREF", filing_date=preferred_date,
        transaction_date=preferred_date, transaction_code="P",
        shares=200, acquired_or_disposed="A", is_discretionary=True,
        is_common_stock=False, security_title="Series A Preferred Stock",
    )
    db.add(preferred_txn)
    db.flush()
    db.add(TradeOutcome(transaction_id=preferred_txn.id))

    common_date = date(2023, 1, 1)  # later: the insider's first COMMON-stock buy
    common_txn = Transaction(
        insider_id=insider.id, company_id=company.id,
        accession_number="ACC-FBNONCOMM-COMMON", filing_date=common_date,
        transaction_date=common_date, transaction_code="P",
        shares=500, acquired_or_disposed="A", is_discretionary=True,
        is_common_stock=True,
    )
    db.add(common_txn)
    db.flush()
    db.add(TradeOutcome(transaction_id=common_txn.id))
    db.flush()

    live = LiveFeatureView(db, common_txn, tier="Insufficient", skill_score=0.0,
                            role_title=None)
    live_val = live.get("is_first_buy")
    assert live_val is True
    assert isinstance(live_val, bool)

    # Real load_backtest_data path (not a hand-built frame): this is the one
    # place the SQL is_common_stock filter actually runs, so it's the only
    # way to prove the live side now mirrors it rather than being
    # fixture-lucky.
    data = load_backtest_data(db)
    assert preferred_txn.id not in set(data["buys"]["txn_id"])  # SQL-excluded
    result = compute_firsttime_flags(data["buys"], data["roles"])
    row = result[result["txn_id"] == common_txn.id].iloc[0]
    assert bool(row["is_first_buy"]) is True
    backtest = RowFeatureView(row, tier="Insufficient", skill_score=0.0)
    bt_val = backtest.get("is_first_buy")
    assert bt_val == live_val
    assert bt_val is True
    assert isinstance(bt_val, bool)
