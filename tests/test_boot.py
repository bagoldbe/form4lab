"""First end-to-end boot smoke test: the default strategy resolves and the
FastAPI app serves a real route against an empty (schema-only) database.
"""
from fastapi.testclient import TestClient

from form4lab.database import get_db
from form4lab.main import app
from form4lab.scoring.portfolio_simulator import run_simulation
from form4lab.strategy.registry import get_active

client = TestClient(app)


def test_get_active_resolves_default_strategy():
    strategy, registry = get_active(refresh=True)
    assert strategy.name == "cluster_buy"
    assert registry.tradeable_names() == frozenset({"cluster_buy"})


def test_recommendations_partial_returns_200_on_empty_db():
    response = client.get("/partials/recommendations")
    assert response.status_code == 200


def test_run_simulation_with_shuffle_seed_on_empty_db_returns_empty_portfolio():
    """Regression: the same-day shuffle block did pd.concat([]) on a zero-group
    groupby when the buys frame was empty (a fresh/empty database), raising
    ValueError: No objects to concatenate whenever shuffle_seed was set."""
    db = next(get_db())
    portfolio, _price_index = run_simulation(db, shuffle_seed=42)
    assert portfolio.open_positions == []
    assert portfolio.closed_positions == []
