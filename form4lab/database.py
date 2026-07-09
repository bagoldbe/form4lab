import logging
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from form4lab.config import settings

logger = logging.getLogger(__name__)


def _refuse_remote_db_under_pytest(url: str) -> None:
    """Refuse to create an engine to a remote database while tests run.

    Engines are created at import time — before ``PYTEST_CURRENT_TEST``
    exists — so pytest is detected via ``sys.modules`` instead; nothing
    under ``form4lab/`` imports pytest, so this guard never fires in the
    deployed process.
    """
    if "pytest" not in sys.modules:
        return
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite"):
        return
    if parsed.host in ("localhost", "127.0.0.1", "::1"):
        return
    raise RuntimeError(
        "Refusing to create a database engine for "
        f"{parsed.render_as_string(hide_password=True)} while pytest is loaded; "
        "tests must use SQLite or a localhost database."
    )


_refuse_remote_db_under_pytest(settings.database_url)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
    pool_pre_ping=True,       # check connection liveness before using it
    pool_recycle=120,          # recycle connections every 2 minutes (hosted-proxy idle timeouts)
)

SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection():
    """Test database connectivity. Raises OperationalError if unreachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified: %s", engine.url.render_as_string(hide_password=True))
    except OperationalError:
        logger.error("Cannot connect to database: %s", engine.url.render_as_string(hide_password=True))
        raise
