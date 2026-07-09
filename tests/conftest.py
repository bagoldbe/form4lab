"""Minimal test-suite bootstrap: route the DB to a throwaway SQLite file and
ensure SEC_IDENTITY is set before any ``form4lab.*`` import.

pytest imports conftest.py before any test module, so the environment
overrides below run before ``form4lab.config`` is ever imported. A real
environment variable takes precedence over both ``load_dotenv()`` and
pydantic-settings' ``env_file`` — see ``form4lab/database.py``'s
``_refuse_remote_db_under_pytest`` guard, which independently refuses to
create a non-local engine while pytest is loaded.

This intentionally covers only what tests/test_cluster_buy_strategy.py and
tests/test_boot.py need: a sqlite engine/session with schema created, plus
SEC_IDENTITY for the form4lab.main import-time guard. Per-test transactional
fixtures, mock clients, and other broader conftest patterns can be added
here as the suite grows — merge carefully rather than assuming this file is
the final word.
"""
import os
from pathlib import Path

_TEST_DB_PATH = Path(__file__).resolve().parent / ".pytest_form4lab.db"

os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ.setdefault("SEC_IDENTITY", "test test@example.com")

# Start from an empty database every run (including sqlite journal leftovers).
for _suffix in ("", "-journal", "-wal", "-shm"):
    _stale = Path(f"{_TEST_DB_PATH}{_suffix}")
    if _stale.exists():
        _stale.unlink()

import form4lab.models  # noqa: E402,F401 — registers every table on Base.metadata
from form4lab.database import Base, engine  # noqa: E402

Base.metadata.create_all(engine)
