"""Shared Jinja2Templates instance.

Use this everywhere instead of constructing a per-route Jinja2Templates,
so that `env.globals` (e.g. asset_version for cache-busting) actually
applies to every rendered template.
"""
import os
import time
from pathlib import Path

from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Cache-bust static assets on each deploy. Prefer a per-deploy commit SHA
# (set GIT_COMMIT_SHA in your deploy environment); fall back to a per-startup
# timestamp so a redeploy of the same SHA still invalidates browser caches.
ASSET_VERSION = (
    os.environ.get("GIT_COMMIT_SHA")
    or str(int(time.time()))
)[:12]

templates = Jinja2Templates(directory=_TEMPLATES_DIR)
templates.env.globals["asset_version"] = ASSET_VERSION
