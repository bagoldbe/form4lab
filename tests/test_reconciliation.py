import itertools
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from form4lab.database import Base
from form4lab.models.broker import BrokerOrder, BrokerPosition

_alert_id_seq = itertools.count(1)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_position(db, symbol="ZZA", shares=100.0, entry_price=25.0, status="open"):
    alert_id = next(_alert_id_seq)
    order = BrokerOrder(
        alert_id=alert_id, alpaca_order_id=f"order-{symbol}-{alert_id}", symbol=symbol, side="buy",
        order_type="market", status="filled", filled_qty=shares, filled_avg_price=entry_price,
    )
    db.add(order)
    db.flush()
    pos = BrokerPosition(
        alert_id=alert_id, entry_order_id=order.id, symbol=symbol, shares=shares,
        entry_price=entry_price, entry_date=date(2026, 1, 15),
        exit_target_date=date(2026, 8, 19), status=status,
        insider_name="Test Insider", insider_role="Director",
    )
    db.add(pos)
    db.commit()
    return pos


def test_broker_position_has_reconcile_fields_with_defaults(db):
    pos = _make_position(db)
    db.refresh(pos)
    assert pos.reconcile_hold is False
    assert pos.close_reason is None
    assert pos.last_market_price is None

    pos.reconcile_hold = True
    pos.close_reason = "renamed_from:ZZA"
    pos.last_market_price = 24.50
    db.commit()
    db.refresh(pos)
    assert pos.reconcile_hold is True
    assert pos.close_reason == "renamed_from:ZZA"
    assert pos.last_market_price == 24.50


# --- network helpers ---

def _fake_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        from httpx import HTTPStatusError, Request, Response
        resp.raise_for_status.side_effect = HTTPStatusError(
            "err", request=MagicMock(), response=MagicMock()
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_get_corporate_action_parses_name_change(db):
    body = {"corporate_actions": {"name_changes": [
        {"old_symbol": "ZZA", "new_symbol": "ZZB", "process_date": "2026-06-29"}
    ]}}
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=_fake_response(200, body)):
        from form4lab.services.alpaca_service import _get_corporate_action
        ca = _get_corporate_action("ZZA")
    assert ca is not None
    assert ca.kind == "name_change"
    assert ca.new_symbol == "ZZB"


def test_get_corporate_action_none_when_empty(db):
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=_fake_response(200, {"corporate_actions": {}})):
        from form4lab.services.alpaca_service import _get_corporate_action
        assert _get_corporate_action("AAPL") is None


def test_get_corporate_action_lookup_failed_on_error(db):
    """On ANY exception the lookup returns a lookup_failed sentinel (never None),
    so the classifier can fail safe rather than mistakenly book a delist."""
    with patch("form4lab.services.alpaca_service.httpx.get", side_effect=Exception("network down")):
        from form4lab.services.alpaca_service import _get_corporate_action
        ca = _get_corporate_action("AAPL")
    assert ca is not None
    assert ca.kind == "lookup_failed"
    assert ca.new_symbol is None


def test_get_corporate_action_flags_other_ca_type(db):
    body = {"corporate_actions": {"cash_mergers": [{"symbol": "XYZ"}]}}
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=_fake_response(200, body)):
        from form4lab.services.alpaca_service import _get_corporate_action
        ca = _get_corporate_action("XYZ")
    assert ca is not None and ca.kind == "other" and ca.new_symbol is None


def test_get_asset_status_active(db):
    body = {"status": "active", "tradable": True}
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=_fake_response(200, body)):
        from form4lab.services.alpaca_service import _get_asset_status
        assert _get_asset_status("ZZC") == "active"


def test_get_asset_status_not_found(db):
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=_fake_response(404)):
        from form4lab.services.alpaca_service import _get_asset_status
        assert _get_asset_status("ZZA") == "not_found"


def test_get_asset_status_unknown_on_error(db):
    with patch("form4lab.services.alpaca_service.httpx.get", side_effect=Exception("boom")):
        from form4lab.services.alpaca_service import _get_asset_status
        assert _get_asset_status("ZZA") == "unknown"


# --- pure classifier ---

def _classify(**kw):
    from form4lab.services.alpaca_service import classify_disappeared_position
    defaults = dict(
        symbol="ZZA", has_sell_order=False, sell_price=None, corp_action=None,
        asset_status="active", last_bar_price=24.97, broker_holds_new_symbol=False,
    )
    defaults.update(kw)
    sym = defaults.pop("symbol")
    return classify_disappeared_position(sym, **defaults)


def test_classify_sold():
    out = _classify(has_sell_order=True, sell_price=155.5)
    assert out.action == "sold" and out.status == "closed" and out.exit_price == 155.5


def test_classify_rename_broker_missing():
    from form4lab.services.alpaca_service import CorporateAction
    ca = CorporateAction("name_change", "ZZB", "2026-06-29", "name_changes")
    out = _classify(corp_action=ca, broker_holds_new_symbol=False)
    assert out.action == "rename" and out.new_symbol == "ZZB"
    assert out.status == "open" and out.reconcile_hold is True
    assert out.anomaly == "error" and out.exit_price is None
    assert out.close_reason == "renamed_from:ZZA;broker_missing"


def test_classify_rename_broker_has_new():
    from form4lab.services.alpaca_service import CorporateAction
    ca = CorporateAction("name_change", "ZZB", "2026-06-29", "name_changes")
    out = _classify(corp_action=ca, broker_holds_new_symbol=True)
    assert out.action == "rename" and out.status == "open"
    assert out.reconcile_hold is False and out.anomaly is None


def test_classify_delisted():
    out = _classify(corp_action=None, asset_status="not_found")
    assert out.action == "delisted" and out.status == "delisted"
    assert out.exit_price == 0.0 and out.anomaly == "error"


def test_classify_orphan_close():
    out = _classify(corp_action=None, asset_status="active", last_bar_price=72.5)
    assert out.action == "orphan_close" and out.status == "closed"
    assert out.exit_price == 72.5 and out.anomaly == "warning"


def test_classify_other_ca_needs_review():
    from form4lab.services.alpaca_service import CorporateAction
    ca = CorporateAction("other", None, None, "cash_mergers")
    out = _classify(corp_action=ca)
    assert out.action == "needs_review" and out.reconcile_hold is True
    assert out.status is None and out.anomaly == "error"


def test_classify_uncertain_never_books_loss():
    out = _classify(corp_action=None, asset_status="unknown")
    assert out.action == "needs_review" and out.reconcile_hold is True
    assert out.status is None and out.exit_price is None


# --- reconcile_positions integration ---

def _mock_cfg(mock_cfg):
    mock_cfg.enabled = True
    mock_cfg.api_key = "k"
    mock_cfg.secret_key = "s"
    mock_cfg.paper = True
    mock_cfg.spy_parking_enabled = False
    mock_cfg.reconcile_mass_disappearance_limit = 2


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_renamed_broker_missing_remaps_no_loss(mock_cfg, mock_client_fn, db):
    pos = _make_position(db, symbol="ZZA", shares=100.0, entry_price=25.0)
    _mock_cfg(mock_cfg)

    client = MagicMock()
    client.get_all_positions.return_value = []   # neither ZZA nor ZZB held
    client.get_orders.return_value = []           # no sell order
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import CorporateAction, reconcile_positions
    ca = CorporateAction("name_change", "ZZB", "2026-06-29", "name_changes")
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=ca), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=24.50):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.symbol == "ZZB"
    assert pos.status == "open"           # not closed, not a loss
    assert pos.reconcile_hold is True
    assert pos.pnl is None
    assert pos.last_market_price == 24.50
    assert pos.close_reason == "renamed_from:ZZA;broker_missing"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_held_position_skipped_next_pass(mock_cfg, mock_client_fn, db):
    pos = _make_position(db, symbol="ZZB")
    pos.reconcile_hold = True
    db.commit()
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import reconcile_positions
    with patch("form4lab.services.alpaca_service._get_corporate_action") as ca_fn:
        reconcile_positions(db)
        ca_fn.assert_not_called()   # held position must not be reprocessed
    db.refresh(pos)
    assert pos.status == "open"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_delisted_books_full_loss(mock_cfg, mock_client_fn, db):
    pos = _make_position(db, symbol="ZZZZ", shares=100.0, entry_price=10.0)
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import reconcile_positions
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=0.5):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.status == "delisted"
    assert pos.exit_price == 0.0
    assert pos.pnl == pytest.approx(-1000.0)        # -(100 * 10.0)
    assert pos.pnl_pct == pytest.approx(-1.0)       # (0 - 10)/10
    assert pos.last_market_price == 0.5


# --- reconciliation health ---

def test_reconciliation_health_healthy_when_no_anomalies(db):
    _make_position(db, symbol="ZZC", status="open")
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is True
    assert report.delisted_count == 0 and report.held_count == 0


def test_reconciliation_health_flags_delisted_and_held(db):
    delisted = _make_position(db, symbol="ZZZZ", status="delisted")
    delisted.exit_date = date(2026, 6, 28)
    held = _make_position(db, symbol="ZZB", status="open")
    held.reconcile_hold = True
    db.commit()
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is False
    assert report.delisted_count == 1 and report.held_count == 1


# --- one-off rename correction ---

def test_correct_renamed_position_rebooks_without_loss(db):
    # Simulate the erroneous close reconcile produced for ZZA today.
    pos = _make_position(db, symbol="ZZA", shares=100.0, entry_price=25.0, status="closed")
    pos.exit_price = 24.97
    pos.exit_date = date(2026, 6, 29)
    pos.pnl = (24.97 - 25.0) * 100.0
    pos.pnl_pct = (24.97 - 25.0) / 25.0
    db.commit()

    from form4lab.services.alpaca_service import correct_renamed_position
    n = correct_renamed_position(db, "ZZA", "ZZB", 24.50)
    assert n == 1
    db.refresh(pos)
    assert pos.symbol == "ZZB"
    assert pos.status == "open"
    assert pos.reconcile_hold is True
    assert pos.pnl is None and pos.exit_price is None and pos.exit_date is None
    assert pos.last_market_price == 24.50
    assert pos.close_reason == "renamed_from:ZZA;manual_correction"


def test_correct_renamed_position_idempotent(db):
    pos = _make_position(db, symbol="ZZA", status="closed")
    db.commit()
    from form4lab.services.alpaca_service import correct_renamed_position
    correct_renamed_position(db, "ZZA", "ZZB", 24.50)
    n2 = correct_renamed_position(db, "ZZA", "ZZB", 24.50)   # re-run
    assert n2 == 1                                            # still finds the ZZB+hold row
    db.refresh(pos)
    assert pos.symbol == "ZZB" and pos.status == "open"


# --- Fix 2: per-position RECON_CLOSE log ---

def test_apply_reconcile_outcome_logs_recon_close(db, caplog):
    """_apply_reconcile_outcome emits a RECON_CLOSE INFO line for every committed position."""
    import logging
    from form4lab.services.alpaca_service import ReconcileOutcome, _apply_reconcile_outcome

    pos = _make_position(db, symbol="SOLD", shares=100.0, entry_price=10.0)
    outcome = ReconcileOutcome(
        action="sold", status="closed", new_symbol=None, exit_price=12.0,
        close_reason="sold", last_market_price=12.0, reconcile_hold=False, anomaly=None,
    )
    with caplog.at_level(logging.INFO, logger="form4lab.services.alpaca_service"):
        _apply_reconcile_outcome(db, [pos], "SOLD", outcome)

    recon_lines = [r.message for r in caplog.records if "RECON_CLOSE" in r.message]
    assert len(recon_lines) == 1
    assert "SOLD" in recon_lines[0]
    assert "sold" in recon_lines[0]


# --- Fix 1: held count scoped to live positions ---

def test_recon_health_closed_held_not_counted(db):
    """A reconcile_hold=True position that is closed must NOT count as held."""
    pos = _make_position(db, symbol="OLDCO", status="closed")
    pos.reconcile_hold = True
    db.commit()
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 30))
    assert report.held_count == 0
    assert report.healthy is True


def test_recon_health_open_held_is_counted(db):
    """A reconcile_hold=True position that is open MUST count as held."""
    pos = _make_position(db, symbol="HOLDCO", status="open")
    pos.reconcile_hold = True
    db.commit()
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 30))
    assert report.held_count == 1
    assert report.healthy is False


# --- Fix 3: clear_reconcile_hold ---

def test_clear_reconcile_hold_clears_held_position(db):
    """clear_reconcile_hold returns 1 and sets reconcile_hold=False for a held open position."""
    pos = _make_position(db, symbol="CLR", status="open")
    pos.reconcile_hold = True
    db.commit()
    from form4lab.services.alpaca_service import clear_reconcile_hold
    n = clear_reconcile_hold(db, "CLR")
    assert n == 1
    db.refresh(pos)
    assert pos.reconcile_hold is False


def test_clear_reconcile_hold_unaffected_without_flag(db):
    """clear_reconcile_hold returns 0 and does not touch a position without the flag."""
    pos = _make_position(db, symbol="SAFE", status="open")
    # reconcile_hold defaults to False — do not set it
    db.commit()
    from form4lab.services.alpaca_service import clear_reconcile_hold
    n = clear_reconcile_hold(db, "SAFE")
    assert n == 0
    db.refresh(pos)
    assert pos.reconcile_hold is False


# =====================================================================
# Money-safety core behavior
# =====================================================================

# --- Successful-but-empty CA lookup still returns None ---

def test_get_corporate_action_none_on_success_empty(db):
    """A successful lookup with no CA present returns None (confirmed no CA),
    distinct from the lookup_failed sentinel returned on error."""
    from form4lab.services.alpaca_service import _get_corporate_action
    with patch("form4lab.services.alpaca_service.httpx.get",
               return_value=_fake_response(200, {"corporate_actions": {}})):
        assert _get_corporate_action("AAPL") is None


# --- Classifier fails safe on lookup failure / malformed CA ---

def test_classify_lookup_failed_never_delisted():
    """MONEY SAFETY: a CA lookup failure routes to needs_review even when the asset
    lookup says not_found — we cannot rule out a rename, so never book a loss."""
    from form4lab.services.alpaca_service import CorporateAction
    ca = CorporateAction("lookup_failed", None, None, None)
    out = _classify(corp_action=ca, asset_status="not_found")
    assert out.action == "needs_review"
    assert out.status is None
    assert out.reconcile_hold is True
    assert out.anomaly == "error"
    assert out.exit_price is None
    assert out.close_reason == "ca_lookup_failed"


def test_classify_malformed_name_change_needs_review():
    """A name_change CA with no new_symbol is malformed → needs_review, never a close."""
    from form4lab.services.alpaca_service import CorporateAction
    ca = CorporateAction("name_change", None, "2026-06-29", "name_changes")
    out = _classify(corp_action=ca, asset_status="not_found")
    assert out.action == "needs_review"
    assert out.status is None
    assert out.reconcile_hold is True
    assert out.anomaly == "error"
    assert out.exit_price is None
    assert out.close_reason == "malformed_name_change"


def test_classify_inactive_delisted():
    """With a confirmed-absent CA (None) and an inactive asset, book the delist."""
    out = _classify(corp_action=None, asset_status="inactive")
    assert out.action == "delisted"
    assert out.status == "delisted"
    assert out.exit_price == 0.0
    assert out.anomaly == "error"


# --- _get_asset_status parsing + inactive-only ---

def test_get_asset_status_inactive(db):
    from form4lab.services.alpaca_service import _get_asset_status
    with patch("form4lab.services.alpaca_service.httpx.get",
               return_value=_fake_response(200, {"status": "inactive"})):
        assert _get_asset_status("ZZA") == "inactive"


def test_get_asset_status_unknown_on_list_body(db):
    """A 200 whose JSON body is a list (not a dict) must not crash — return unknown."""
    from form4lab.services.alpaca_service import _get_asset_status
    with patch("form4lab.services.alpaca_service.httpx.get",
               return_value=_fake_response(200, [{"status": "active"}])):
        assert _get_asset_status("ZZA") == "unknown"


def test_get_asset_status_unknown_when_json_raises(db):
    """A 200 whose .json() raises must not propagate — return unknown."""
    from form4lab.services.alpaca_service import _get_asset_status
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    with patch("form4lab.services.alpaca_service.httpx.get", return_value=resp):
        assert _get_asset_status("ZZA") == "unknown"


def test_get_asset_status_active_not_tradable_is_unknown(db):
    """status active but tradable false is an ambiguous shape → unknown (not inactive)."""
    from form4lab.services.alpaca_service import _get_asset_status
    with patch("form4lab.services.alpaca_service.httpx.get",
               return_value=_fake_response(200, {"status": "active", "tradable": False})):
        assert _get_asset_status("ZZA") == "unknown"


# --- reconcile_positions fails safe & isolates symbols ---

@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_migrated_rename_remaps_to_new_symbol(mock_cfg, mock_client_fn, db):
    """When the broker already holds the NEW symbol, remap the DB position to it,
    keep it open, no hold, and never book a P&L."""
    pos = _make_position(db, symbol="ZZA", shares=100.0, entry_price=25.0)
    _mock_cfg(mock_cfg)

    new_symbol_pos = MagicMock()
    new_symbol_pos.symbol = "ZZB"
    new_symbol_pos.qty = "100"
    client = MagicMock()
    client.get_all_positions.return_value = [new_symbol_pos]   # broker holds the new symbol
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import CorporateAction, reconcile_positions
    ca = CorporateAction("name_change", "ZZB", "2026-06-29", "name_changes")
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=ca), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=24.50):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.symbol == "ZZB"
    assert pos.status == "open"
    assert pos.reconcile_hold is False
    assert pos.pnl is None
    assert pos.close_reason == "renamed_from:ZZA"


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_other_ca_stays_open_on_hold(mock_cfg, mock_client_fn, db):
    """A non-rename corporate action keeps the position open under a reconcile hold."""
    pos = _make_position(db, symbol="XYZC", shares=100.0, entry_price=10.0)
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import CorporateAction, reconcile_positions
    ca = CorporateAction("other", None, None, "cash_mergers")
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=ca), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=9.0):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.status == "open"
    assert pos.reconcile_hold is True
    assert pos.pnl is None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_unknown_asset_stays_open_on_hold(mock_cfg, mock_client_fn, db):
    """An unknown asset status (lookup uncertain) keeps the position open on hold."""
    pos = _make_position(db, symbol="QQQZ", shares=100.0, entry_price=10.0)
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import reconcile_positions
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="unknown"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=9.0):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.status == "open"
    assert pos.reconcile_hold is True
    assert pos.pnl is None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_lookup_failed_stays_open_not_delisted(mock_cfg, mock_client_fn, db):
    """MONEY SAFETY integration: a CA lookup failure never books a delist even when
    the asset lookup says not_found."""
    pos = _make_position(db, symbol="LFAIL", shares=100.0, entry_price=10.0)
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    from form4lab.services.alpaca_service import CorporateAction, reconcile_positions
    ca = CorporateAction("lookup_failed", None, None, None)
    with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=ca), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=9.0):
        reconcile_positions(db)

    db.refresh(pos)
    assert pos.status == "open"
    assert pos.status != "delisted"
    assert pos.reconcile_hold is True
    assert pos.pnl is None


@patch("form4lab.services.alpaca_service._get_trading_client")
@patch("form4lab.services.alpaca_service._alpaca_cfg")
def test_reconcile_per_symbol_isolation(mock_cfg, mock_client_fn, db):
    """A poisoned symbol (helper raises) must not abort the whole loop — the second
    vanished symbol is still reconciled."""
    aaa = _make_position(db, symbol="AAA", shares=100.0, entry_price=10.0)
    bbb = _make_position(db, symbol="BBB", shares=50.0, entry_price=20.0)
    _mock_cfg(mock_cfg)
    client = MagicMock()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    mock_client_fn.return_value = client

    def poisoned_find_sell(_client, symbol):
        if symbol == "AAA":
            raise RuntimeError("poison pill")
        return (False, None)

    from form4lab.services.alpaca_service import reconcile_positions
    with patch("form4lab.services.alpaca_service._find_sell_fill", side_effect=poisoned_find_sell), \
         patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
         patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
         patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=0.5):
        n = reconcile_positions(db)

    db.refresh(aaa)
    db.refresh(bbb)
    assert aaa.status == "open"        # poisoned symbol left untouched
    assert bbb.status == "delisted"    # second symbol still processed
    assert n == 1


# =====================================================================
# Operator-safety, dashboard, and money-safety behavior
# =====================================================================

# --- FIX B0: $0 sell fill must not book a total loss ---

def test_find_sell_fill_zero_price_not_a_fill():
    """A filled sell order with filled_avg_price=0.0 is NOT a valid fill."""
    from form4lab.services.alpaca_service import _find_sell_fill
    client = MagicMock()
    order = MagicMock()
    order.filled_avg_price = 0.0
    client.get_orders.return_value = [order]
    has_fill, price = _find_sell_fill(client, "AAPL")
    assert has_fill is False
    assert price is None


def test_find_sell_fill_zero_string_not_a_fill():
    """A filled sell order with filled_avg_price='0' (string zero) is NOT a valid fill."""
    from form4lab.services.alpaca_service import _find_sell_fill
    client = MagicMock()
    order = MagicMock()
    order.filled_avg_price = "0"
    client.get_orders.return_value = [order]
    has_fill, price = _find_sell_fill(client, "AAPL")
    assert has_fill is False
    assert price is None


def test_find_sell_fill_positive_price_is_a_fill():
    """A filled sell order with a positive price is still a valid fill (regression guard)."""
    from form4lab.services.alpaca_service import _find_sell_fill
    client = MagicMock()
    order = MagicMock()
    order.filled_avg_price = 155.5
    client.get_orders.return_value = [order]
    has_fill, price = _find_sell_fill(client, "AAPL")
    assert has_fill is True
    assert price == pytest.approx(155.5)


def test_classify_sold_at_zero_not_sold():
    """classify_disappeared_position with has_sell_order=True, sell_price=0.0
    must NOT route to the 'sold' outcome — a $0 fill is not a valid fill."""
    out = _classify(has_sell_order=True, sell_price=0.0)
    assert out.action != "sold"


# --- correct_renamed_position must not clobber legitimately-sold trades ---

def _make_legitimately_sold_position(db, symbol="ZZA"):
    """Create a position closed by a real sell order (exit_order_id set)."""
    alert_id = next(_alert_id_seq)
    entry_order = BrokerOrder(
        alert_id=alert_id,
        alpaca_order_id=f"entry-real-sell-{alert_id}",
        symbol=symbol,
        side="buy",
        order_type="market",
        status="filled",
        filled_qty=100.0,
        filled_avg_price=30.0,
    )
    db.add(entry_order)
    db.flush()
    exit_order = BrokerOrder(
        alert_id=alert_id,
        alpaca_order_id=f"exit-real-sell-{alert_id}",
        symbol=symbol,
        side="sell",
        order_type="market",
        status="filled",
        filled_qty=100.0,
        filled_avg_price=35.0,
        entry_order_id=entry_order.id,
    )
    db.add(exit_order)
    db.flush()
    pos = BrokerPosition(
        alert_id=alert_id,
        entry_order_id=entry_order.id,
        exit_order_id=exit_order.id,      # real sell order linked
        symbol=symbol,
        shares=100.0,
        entry_price=30.0,
        entry_date=date(2026, 6, 1),
        exit_target_date=date(2026, 8, 1),
        status="closed",
        exit_price=35.0,
        exit_date=date(2026, 6, 29),
        pnl=500.0,
        pnl_pct=0.1667,
        insider_name="Test Insider",
        insider_role="CEO",
    )
    db.add(pos)
    db.commit()
    return pos


def test_correct_renamed_position_skips_position_with_exit_order_id(db):
    """A legitimately-sold position (exit_order_id set) must NOT be re-opened
    or have its P&L wiped by correct_renamed_position."""
    pos = _make_legitimately_sold_position(db, symbol="SOLD1")
    saved_pnl = pos.pnl

    from form4lab.services.alpaca_service import correct_renamed_position
    n = correct_renamed_position(db, "SOLD1", "NEW1")
    assert n == 0
    db.refresh(pos)
    assert pos.symbol == "SOLD1"       # not renamed
    assert pos.status == "closed"      # not re-opened
    assert pos.exit_price == pytest.approx(35.0)   # P&L preserved
    assert pos.pnl == pytest.approx(saved_pnl)


def test_correct_renamed_position_skips_close_reason_sold(db):
    """A reconcile-closed position tagged close_reason='sold' (no exit_order_id)
    must NOT be re-opened — 'sold' is the explicit sentinel for a confirmed sale."""
    pos = _make_position(db, symbol="SOLD2", status="closed")
    pos.close_reason = "sold"
    pos.exit_price = 28.5
    pos.exit_date = date(2026, 6, 20)
    db.commit()

    from form4lab.services.alpaca_service import correct_renamed_position
    n = correct_renamed_position(db, "SOLD2", "NEW2")
    assert n == 0
    db.refresh(pos)
    assert pos.symbol == "SOLD2"
    assert pos.status == "closed"
    assert pos.exit_price == pytest.approx(28.5)


def test_correct_renamed_position_still_fixes_reconcile_misclosed(db):
    """A position wrongly closed by reconcile (no exit_order_id, close_reason != 'sold')
    IS corrected — the primary fix target is unchanged."""
    pos = _make_position(db, symbol="ZZAFIX", status="closed")
    pos.exit_price = 24.97
    pos.exit_date = date(2026, 6, 29)
    pos.pnl = (24.97 - 25.0) * 100.0
    pos.pnl_pct = (24.97 - 25.0) / 25.0
    # exit_order_id is None (not set by _make_position), close_reason is None
    db.commit()

    from form4lab.services.alpaca_service import correct_renamed_position
    n = correct_renamed_position(db, "ZZAFIX", "ZZBFIX", 24.50)
    assert n == 1
    db.refresh(pos)
    assert pos.symbol == "ZZBFIX"
    assert pos.status == "open"
    assert pos.pnl is None and pos.exit_price is None


# --- Commit failures must surface (re-raise) ---

def test_clear_reconcile_hold_raises_on_commit_failure(db):
    """clear_reconcile_hold must re-raise after rollback on commit failure,
    not silently return 0 (which is indistinguishable from 'nothing matched')."""
    pos = _make_position(db, symbol="CLRF")
    pos.reconcile_hold = True
    db.commit()

    from form4lab.services.alpaca_service import clear_reconcile_hold
    with patch.object(db, "commit", side_effect=RuntimeError("DB write failed")):
        with pytest.raises(RuntimeError, match="DB write failed"):
            clear_reconcile_hold(db, "CLRF")

    # Confirm rollback happened — position is still held
    db.rollback()
    db.refresh(pos)
    assert pos.reconcile_hold is True


def test_correct_renamed_position_raises_on_commit_failure(db):
    """correct_renamed_position must re-raise after rollback on commit failure."""
    pos = _make_position(db, symbol="ZZAR", status="closed")
    pos.exit_price = 29.0
    pos.exit_date = date(2026, 6, 20)
    db.commit()

    from form4lab.services.alpaca_service import correct_renamed_position
    with patch.object(db, "commit", side_effect=RuntimeError("DB write failed")):
        with pytest.raises(RuntimeError, match="DB write failed"):
            correct_renamed_position(db, "ZZAR", "ZZBR")

    # Confirm rollback happened — position is still in original state
    db.rollback()
    db.refresh(pos)
    assert pos.symbol == "ZZAR"


# --- Minor: delisted positions appear on the dashboard ---

def test_get_portfolio_summary_includes_delisted(db):
    """A delisted position must appear in closed_positions on the dashboard,
    not silently vanish (it represents a real -100% loss)."""
    pos = _make_position(db, symbol="DELIST", status="delisted")
    pos.exit_price = 0.0
    pos.exit_date = date(2026, 6, 25)
    pos.pnl = -1234.5
    db.commit()

    from form4lab.services.alpaca_service import get_portfolio_summary
    with patch("form4lab.services.alpaca_service._alpaca_cfg") as mock_cfg:
        mock_cfg.enabled = False
        summary = get_portfolio_summary(db)

    symbols = [p.symbol for p in summary["closed_positions"]]
    assert "DELIST" in symbols


# =====================================================================
# Catch-all handling, parse-inside-try, and orphan-health behavior
# =====================================================================

# --- FIX C2: catch-all for unrecognized CA kinds ---

def test_classify_unknown_ca_kind_routes_to_needs_review_not_delisted():
    """MONEY SAFETY: an unrecognized future CorporateAction kind must route to
    needs_review with reconcile_hold=True, never fall through to the delisted
    asset-based branch."""
    from form4lab.services.alpaca_service import CorporateAction, classify_disappeared_position
    # Construct a CA with a kind not in the current Literal set.
    # Literal typing is unenforced at runtime, so this is valid for testing.
    ca = CorporateAction(kind="some_future_kind", new_symbol=None,
                         process_date=None, raw_type=None)
    out = classify_disappeared_position(
        "FUTR",
        has_sell_order=False, sell_price=None,
        corp_action=ca, asset_status="not_found",
        last_bar_price=10.0, broker_holds_new_symbol=False,
    )
    assert out.action == "needs_review"
    assert out.reconcile_hold is True
    assert out.status is None   # NOT "delisted"
    assert out.anomaly == "error"
    assert "ca_unhandled:some_future_kind" in out.close_reason


# --- FIX C3: parse inside try — malformed 200 body returns lookup_failed ---

def test_get_corporate_action_lookup_failed_on_malformed_body(db):
    """A 200 whose corporate_actions value is a list (not a dict) must return a
    lookup_failed sentinel, not raise an AttributeError to the caller."""
    from form4lab.services.alpaca_service import _get_corporate_action
    malformed_body = {"corporate_actions": ["not", "a", "dict"]}
    with patch("form4lab.services.alpaca_service.httpx.get",
               return_value=_fake_response(200, malformed_body)):
        ca = _get_corporate_action("MALF")
    assert ca is not None
    assert ca.kind == "lookup_failed"   # not a raised exception
    assert ca.new_symbol is None


# --- FIX C4: orphan_close counted in reconciliation health ---

def test_reconciliation_health_counts_orphan_close_as_unhealthy(db):
    """A windowed orphan_no_sell close makes health unhealthy and orphan_count == 1."""
    pos = _make_position(db, symbol="ORPH", status="closed")
    pos.close_reason = "orphan_no_sell"
    pos.exit_date = date(2026, 6, 28)   # within 14-day window of 2026-06-29
    db.commit()

    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is False
    assert report.orphan_count == 1
    assert report.delisted_count == 0
    assert report.held_count == 0
    assert "1 orphan close(s) in window" in report.reason


def test_reconciliation_health_orphan_outside_window_not_counted(db):
    """An orphan_no_sell close older than window_days is NOT counted."""
    pos = _make_position(db, symbol="OLDORPH", status="closed")
    pos.close_reason = "orphan_no_sell"
    pos.exit_date = date(2026, 6, 1)    # > 14 days before 2026-06-29
    db.commit()

    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is True
    assert report.orphan_count == 0


def test_reconciliation_health_healthy_includes_orphan_count_zero(db):
    """Healthy report has orphan_count == 0 (field exists and is accessible)."""
    _make_position(db, symbol="HLTHY", status="open")
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is True
    assert report.orphan_count == 0
    assert report.delisted_count == 0
    assert report.held_count == 0


def test_reconciliation_health_flags_delisted_and_held_with_orphan_field(db):
    """Existing delisted+held test updated to confirm orphan_count field is present."""
    delisted = _make_position(db, symbol="ZZZZ2", status="delisted")
    delisted.exit_date = date(2026, 6, 28)
    held = _make_position(db, symbol="ZZB2", status="open")
    held.reconcile_hold = True
    db.commit()
    from form4lab.services.reconciliation_health import check_reconciliation_health
    report = check_reconciliation_health(db, as_of=date(2026, 6, 29))
    assert report.healthy is False
    assert report.delisted_count == 1
    assert report.orphan_count == 0
    assert report.held_count == 1


# =====================================================================
# Mass-disappearance circuit breaker (platform-glitch guard)
# =====================================================================
# See the mass-disappearance rationale in reconcile_positions' docstring:
# Alpaca's paper platform administratively wiped ALL positions overnight and
# restored them midday with zero fills either way. A simultaneous
# multi-position disappearance is a platform glitch until proven otherwise —
# this breaker holds everything for review instead of classifying/
# orphan-closing it.

class TestMassDisappearanceBreaker:
    def _three_missing(self, db):
        """3 open DB positions with real shares; Alpaca returns none of them."""
        for sym in ("AAA", "BBB", "CCC"):
            _make_position(db, symbol=sym, shares=10.0)

    @patch("form4lab.services.alpaca_service._get_trading_client")
    @patch("form4lab.services.alpaca_service._alpaca_cfg")
    def test_mass_disappearance_holds_all_and_classifies_none(self, mock_cfg, mock_client_fn, db):
        self._three_missing(db)
        _mock_cfg(mock_cfg)
        client = MagicMock()
        client.get_all_positions.return_value = []   # everything vanished
        mock_client_fn.return_value = client

        from form4lab.services.alpaca_service import reconcile_positions
        with patch("form4lab.services.alpaca_service._find_sell_fill") as mock_find_sell, \
             patch("form4lab.services.alpaca_service._get_corporate_action") as mock_get_ca:
            count = reconcile_positions(db)
            mock_find_sell.assert_not_called()   # no per-symbol classification ran
            mock_get_ca.assert_not_called()

        held = db.query(BrokerPosition).filter(BrokerPosition.reconcile_hold.is_(True)).all()
        assert {p.symbol for p in held} == {"AAA", "BBB", "CCC"}
        assert all(p.close_reason == "mass_disappearance_hold" for p in held)
        assert all(p.status == "open" for p in held)          # nothing closed
        assert count == 3

    @patch("form4lab.services.alpaca_service._get_trading_client")
    @patch("form4lab.services.alpaca_service._alpaca_cfg")
    def test_at_limit_classifies_normally(self, mock_cfg, mock_client_fn, db):
        """Exactly `limit` missing symbols (default 2) -> normal per-symbol path, no mass hold."""
        for sym in ("AAA", "BBB"):
            _make_position(db, symbol=sym, shares=10.0)
        _mock_cfg(mock_cfg)
        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.return_value = []
        mock_client_fn.return_value = client

        from form4lab.services.alpaca_service import reconcile_positions
        with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
             patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
             patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=1.0):
            reconcile_positions(db)

        assert db.query(BrokerPosition).filter(
            BrokerPosition.close_reason == "mass_disappearance_hold").count() == 0

    @patch("form4lab.services.alpaca_service._get_trading_client")
    @patch("form4lab.services.alpaca_service._alpaca_cfg")
    def test_unfilled_and_spy_do_not_count(self, mock_cfg, mock_client_fn, db):
        """A 0-share (unfilled) position and SPY (parking) never count toward the limit."""
        _make_position(db, symbol="AAA", shares=10.0)
        _make_position(db, symbol="BBB", shares=10.0)
        _make_position(db, symbol="ZRO", shares=0.0)      # unfilled — excluded
        _make_position(db, symbol="SPY", shares=5.0)      # parking — excluded
        _mock_cfg(mock_cfg)
        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.return_value = []
        mock_client_fn.return_value = client

        from form4lab.services.alpaca_service import reconcile_positions
        with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
             patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
             patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=1.0):
            reconcile_positions(db)

        assert db.query(BrokerPosition).filter(
            BrokerPosition.close_reason == "mass_disappearance_hold").count() == 0

    @patch("form4lab.services.alpaca_service._get_trading_client")
    @patch("form4lab.services.alpaca_service._alpaca_cfg")
    def test_mass_disappearance_logs_error(self, mock_cfg, mock_client_fn, db, caplog):
        """The breaker firing must emit a greppable ERROR line, not just hold
        positions silently — operators tail deploy logs for exactly this."""
        import logging

        self._three_missing(db)
        _mock_cfg(mock_cfg)
        client = MagicMock()
        client.get_all_positions.return_value = []
        mock_client_fn.return_value = client

        from form4lab.services.alpaca_service import reconcile_positions
        with caplog.at_level(logging.ERROR, logger="form4lab.services.alpaca_service"):
            reconcile_positions(db)

        error_lines = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        mass_lines = [m for m in error_lines if "MASS DISAPPEARANCE" in m]
        assert len(mass_lines) == 1
        assert "AAA" in mass_lines[0] and "BBB" in mass_lines[0] and "CCC" in mass_lines[0]

    @patch("form4lab.services.alpaca_service._get_trading_client")
    @patch("form4lab.services.alpaca_service._alpaca_cfg")
    def test_custom_limit_threads_through_config(self, mock_cfg, mock_client_fn, db):
        """A non-default limit (5) is honored: 4 missing symbols would trip the
        hardcoded original default of 2, but must classify normally under a
        configured limit of 5 — proving the threshold is read from config,
        not a hardcoded constant."""
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            _make_position(db, symbol=sym, shares=10.0)
        _mock_cfg(mock_cfg)
        mock_cfg.reconcile_mass_disappearance_limit = 5
        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.return_value = []
        mock_client_fn.return_value = client

        from form4lab.services.alpaca_service import reconcile_positions
        with patch("form4lab.services.alpaca_service._get_corporate_action", return_value=None), \
             patch("form4lab.services.alpaca_service._get_asset_status", return_value="not_found"), \
             patch("form4lab.services.alpaca_service._get_latest_bar_price", return_value=1.0):
            reconcile_positions(db)

        # No mass hold at 4 missing under limit=5 — every symbol went through
        # normal per-symbol classification (not_found + no CA -> delisted).
        assert db.query(BrokerPosition).filter(
            BrokerPosition.close_reason == "mass_disappearance_hold").count() == 0
        delisted = db.query(BrokerPosition).filter(BrokerPosition.status == "delisted").all()
        assert {p.symbol for p in delisted} == {"AAA", "BBB", "CCC", "DDD"}
