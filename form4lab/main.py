import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware

from form4lab.config import settings

if not settings.sec_identity:
    raise RuntimeError(
        "Set SEC_IDENTITY (name + contact email) — required by SEC EDGAR's "
        "fair-access policy"
    )

# Uvicorn only configures its own "uvicorn.*" loggers; the "form4lab.*" tree has
# no handler by default, so INFO lines (startup flag dump, sizing decisions, job
# summaries) would be silently dropped from deploy logs. Attach a handler to the
# "form4lab" tree only — root stays untouched so httpx/apscheduler INFO noise
# (one line per SEC request) is not enabled.
_app_logger = logging.getLogger("form4lab")
if not _app_logger.handlers:  # idempotent under re-imports (tests, --reload)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


@asynccontextmanager
async def lifespan(app):
    from form4lab.config import settings
    from form4lab.database import check_db_connection
    from form4lab.scheduler.jobs import create_scheduler
    check_db_connection()
    # Make the effective trading/sizing flags verifiable straight from deploy
    # logs ("is the flag on in the running process?"). Never log credentials.
    a = settings.alpaca
    logger.info(
        "Alpaca config: enabled=%s paper=%s | vol_targeting_enabled=%s "
        "vol_targeting_shadow=%s vol_target_k=%s vol_target_min_pct=%s "
        "vol_target_max_pct=%s vol_target_max_ticker_pct=%s vol_target_window=%s",
        a.enabled, a.paper, a.vol_targeting_enabled, a.vol_targeting_shadow,
        a.vol_target_k, a.vol_target_min_pct, a.vol_target_max_pct,
        a.vol_target_max_ticker_pct, a.vol_target_window,
    )
    # SCHEDULER_ENABLED lets docker-compose run the web server and the
    # background scheduler as separate services: set it false in the web
    # container when a standalone `form4lab scheduler` process (see cli.py)
    # owns the jobs instead. Track what we actually started in a local so
    # shutdown never calls .shutdown() on a scheduler this process never
    # started (double-shutdown against the *other* service's scheduler).
    scheduler = None
    if settings.scheduler_enabled:
        scheduler = create_scheduler()
        scheduler.start()
        logger.info("Scheduler started in-process (SCHEDULER_ENABLED=true)")
    else:
        logger.info(
            "Scheduler disabled in this process (SCHEDULER_ENABLED=false); "
            "expecting a separate `form4lab scheduler` service to run the jobs"
        )
    yield
    if scheduler is not None:
        scheduler.shutdown()
    # Clean up persistent HTTP connections
    from form4lab.data.sec_fetcher import close_client
    close_client()


app = FastAPI(title="form4lab", lifespan=lifespan)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
from form4lab.templating import templates  # shared instance with asset_version global


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return JSON errors for API routes, re-raise for HTML routes."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    # Let FastAPI's default handler render HTML for non-API routes
    raise exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return JSON validation errors for API routes."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=422, content={"error": "Validation error", "details": exc.errors()})
    raise exc

from form4lab.routes.dashboard import router as dashboard_router
from form4lab.routes.api import router as api_router
from form4lab.routes.insider import router as insider_router
from form4lab.routes.company import router as company_router
from form4lab.routes.partials import router as partials_router
from form4lab.routes.summary import router as summary_router
from form4lab.routes.performance import router as performance_router

app.include_router(dashboard_router)
app.include_router(insider_router)
app.include_router(company_router)
app.include_router(partials_router)
app.include_router(summary_router)
app.include_router(performance_router)
app.include_router(api_router, prefix="/api/v1")
