import pytest
from sqlalchemy import text
from form4lab.database import engine, SessionLocal, Base, _refuse_remote_db_under_pytest


def test_engine_connects():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


def test_session_works():
    with SessionLocal() as session:
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1


def test_base_exists():
    assert Base is not None
    assert hasattr(Base, "metadata")


# --- Test isolation: the suite must never touch a remote database ---

def test_suite_runs_on_sqlite():
    """Canary: the root conftest must have routed this run to local SQLite."""
    assert engine.url.drivername.startswith("sqlite")
    assert ".pytest_form4lab" in (engine.url.database or "")


def test_guard_blocks_remote_url_when_pytest_loaded():
    # pytest is in sys.modules right now, so only the URL decides the outcome
    with pytest.raises(RuntimeError, match="Refusing to create"):
        _refuse_remote_db_under_pytest(
            "postgresql://postgres:secret@remote-db.example.com:5432/prod"
        )


def test_guard_error_hides_password():
    with pytest.raises(RuntimeError) as excinfo:
        _refuse_remote_db_under_pytest(
            "postgresql://postgres:supersecret@db.example.com/prod"
        )
    assert "supersecret" not in str(excinfo.value)


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///./.pytest_form4lab.db",
        "sqlite:///:memory:",
        "sqlite://",
        "postgresql://localhost/insider_local",
        "postgresql://user:pw@127.0.0.1:5432/insider_local",
    ],
)
def test_guard_allows_sqlite_and_localhost(url):
    _refuse_remote_db_under_pytest(url)  # must not raise
