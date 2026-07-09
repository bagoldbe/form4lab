import json
from pathlib import Path

from fastapi import APIRouter, Request

from form4lab.templating import templates

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "data"


@router.get("/performance")
async def performance_page(request: Request):
    metrics = None
    equity_curve = None
    trade_log = None

    metrics_path = DATA_DIR / "performance_metrics.json"
    curve_path = DATA_DIR / "equity_curve.json"
    trade_log_path = DATA_DIR / "trade_log.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    if curve_path.exists():
        with open(curve_path) as f:
            equity_curve = json.load(f)

    if trade_log_path.exists():
        with open(trade_log_path) as f:
            trade_log = json.load(f)

    return templates.TemplateResponse(
        request,
        "performance.html",
        {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trade_log": trade_log,
        },
    )
