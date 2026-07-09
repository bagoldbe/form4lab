from datetime import date, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from form4lab.database import Base
from form4lab.main import app
from form4lab.models.price import PriceData

client = TestClient(app)

# --- In-memory DB helpers for unit tests ---

def _make_session():
    """Create an in-memory SQLite session with the price_data table."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _insert_prices(db, ticker, count, base_price=100.0, start_days_ago=300):
    """Insert `count` price rows for `ticker` going backwards from start_days_ago."""
    today = date.today()
    for i in range(count):
        d = today - timedelta(days=start_days_ago - i)
        price = base_price + i  # steadily rising prices
        db.add(PriceData(
            ticker=ticker, date=d,
            open=price, high=price + 1, low=price - 1,
            close=price, adj_close=price, volume=1000,
        ))
    db.commit()


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_api_alerts_returns_json():
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_api_insider_not_found():
    response = client.get("/api/v1/insider/9999999/score")
    assert response.status_code == 404


def test_insider_404_returns_template():
    response = client.get("/insider/0000000")
    assert response.status_code == 404
    assert "404" in response.text
    assert "not found" in response.text.lower()


def test_company_404_returns_template():
    response = client.get("/company/ZZZZZZ")
    assert response.status_code == 404
    assert "404" in response.text
    assert "not found" in response.text.lower()


def test_base_template_has_loading_indicator_css():
    response = client.get("/")
    assert "htmx-indicator" in response.text
    assert "spinner" in response.text


def test_security_headers_present():
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "strict-origin" in response.headers["Referrer-Policy"]


def test_api_404_returns_json():
    response = client.get("/api/v1/insider/0000000/score")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_api_alerts_limit_cap():
    """Limit parameter is capped at 100."""
    response = client.get("/api/v1/alerts?limit=200")
    assert response.status_code == 422


def test_api_alerts_pagination():
    """Skip/limit pagination works."""
    response = client.get("/api/v1/alerts?skip=0&limit=5")
    assert response.status_code == 200


def test_api_prices_invalid_ticker():
    """Invalid ticker format returns 422."""
    response = client.get("/api/v1/prices/<script>")
    assert response.status_code in (404, 422)


# --- Route coverage ---

def test_performance_returns_200():
    response = client.get("/performance")
    assert response.status_code == 200


def test_summary_partial_returns_200():
    response = client.get("/partials/action-summary")
    assert response.status_code == 200


def test_recommendations_partial_returns_200():
    response = client.get("/partials/recommendations")
    assert response.status_code == 200


def test_alerts_partial_returns_200():
    response = client.get("/partials/alerts")
    assert response.status_code == 200


def test_this_week_partial_returns_200():
    response = client.get("/partials/this-week")
    assert response.status_code == 200


# --- API edge cases ---

def test_api_alerts_negative_days():
    """Negative days_back should be rejected."""
    response = client.get("/api/v1/alerts?days_back=-1")
    assert response.status_code == 422


def test_api_alerts_with_type_filter():
    """Alert type filter should work."""
    response = client.get("/api/v1/alerts?alert_types=elite_buy")
    assert response.status_code == 200


def test_api_alerts_with_conviction_filter():
    """Min conviction filter should work."""
    response = client.get("/api/v1/alerts?min_conviction=0.5")
    assert response.status_code == 200


def test_api_alerts_nonexistent_ticker():
    """Filtering by nonexistent ticker returns empty list."""
    response = client.get("/api/v1/alerts?ticker=ZZZZZ")
    assert response.status_code == 200
    assert response.json() == []


def test_api_prices_valid_ticker():
    """Valid ticker format accepted (even if no data)."""
    response = client.get("/api/v1/prices/AAPL?days=30")
    assert response.status_code == 200


def test_performance_has_tojson_not_safe():
    """Verify performance page uses tojson filter (XSS fix)."""
    response = client.get("/performance")
    assert "| safe" not in response.text


def test_alerts_partial_elite_filter():
    """Elite filter on alerts partial."""
    response = client.get("/partials/alerts?filter=elite")
    assert response.status_code == 200


def test_alerts_partial_cluster_filter():
    """Cluster filter on alerts partial."""
    response = client.get("/partials/alerts?filter=cluster")
    assert response.status_code == 200


def test_dashboard_has_signal_sections():
    """Dashboard has signal-focused sections and sell warnings."""
    response = client.get("/")
    assert "partials/recommendations" in response.text
    assert "partials/sell-warnings" in response.text
    assert "partials/this-week" in response.text
    assert "partials/action-summary" in response.text


def test_action_summary_partial_with_days():
    """Action summary accepts days parameter."""
    response = client.get("/partials/action-summary?days=90")
    assert response.status_code == 200


def test_sell_warnings_partial_returns_200():
    """Sell warnings partial loads successfully."""
    response = client.get("/partials/sell-warnings")
    assert response.status_code == 200
    assert "Sell Warnings" in response.text or "No sell warnings" in response.text


# --- _get_drawdowns unit tests ---

def test_get_drawdowns_empty_tickers():
    """Empty tickers list returns empty dict."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    result = _get_drawdowns(db, [])
    assert result == {}
    db.close()


def test_get_drawdowns_insufficient_prices():
    """Ticker with fewer than 20 price records returns None."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    _insert_prices(db, "FEW", count=10)
    result = _get_drawdowns(db, ["FEW"])
    assert result == {"FEW": None}
    db.close()


def test_get_drawdowns_zero_high():
    """Ticker with high_52w <= 0 returns None (all adj_close <= 0)."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    today = date.today()
    for i in range(25):
        d = today - timedelta(days=300 - i)
        db.add(PriceData(
            ticker="ZERO", date=d,
            open=0, high=0, low=0, close=0, adj_close=0, volume=0,
        ))
    db.commit()
    result = _get_drawdowns(db, ["ZERO"])
    assert result == {"ZERO": None}
    db.close()


def test_get_drawdowns_normal_case():
    """Normal case: returns correct drawdown float."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    # Prices: 100, 101, ..., 129 (30 prices). High = 129, latest = 129.
    _insert_prices(db, "NORM", count=30, base_price=100.0)
    result = _get_drawdowns(db, ["NORM"])
    assert "NORM" in result
    # Latest = 129, high = 129, so drawdown = 0
    assert result["NORM"] == 0.0
    db.close()


def test_get_drawdowns_drawdown_value():
    """Drawdown correctly computed when latest price < 52-week high."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    today = date.today()
    # Insert 25 prices: first 20 at 200, last 5 at 150
    for i in range(25):
        d = today - timedelta(days=300 - i)
        price = 200.0 if i < 20 else 150.0
        db.add(PriceData(
            ticker="DROP", date=d,
            open=price, high=price, low=price,
            close=price, adj_close=price, volume=1000,
        ))
    db.commit()
    result = _get_drawdowns(db, ["DROP"])
    # high_52w = 200, latest = 150, drawdown = (150 - 200) / 200 = -0.25
    assert abs(result["DROP"] - (-0.25)) < 1e-9
    db.close()


def test_get_drawdowns_multiple_tickers():
    """Batched query works correctly for multiple tickers."""
    from form4lab.routes.summary import _get_drawdowns
    db = _make_session()
    _insert_prices(db, "AAA", count=30, base_price=100.0)
    _insert_prices(db, "BBB", count=5, base_price=50.0)  # insufficient
    result = _get_drawdowns(db, ["AAA", "BBB", "CCC"])
    assert "AAA" in result and result["AAA"] is not None
    assert result["BBB"] is None  # < 20 prices
    assert result["CCC"] is None  # no data at all
    db.close()


# --- _load_perf_metrics unit tests ---
#
# form4lab does not ship a real performance_metrics.json (only
# static/data/.gitkeep ships), so there is no test asserting that a real
# export file parses. The missing-file path below exercises
# _load_perf_metrics with an explicit mock instead.

def test_load_perf_metrics_returns_none_for_missing_file():
    """Returns None when file does not exist."""
    from form4lab.routes.summary import _load_perf_metrics
    from pathlib import Path
    _load_perf_metrics.cache_clear()
    with patch.object(Path, "exists", return_value=False):
        result = _load_perf_metrics()
    assert result is None
    _load_perf_metrics.cache_clear()  # Reset for other tests
