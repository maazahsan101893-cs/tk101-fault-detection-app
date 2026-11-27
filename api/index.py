"""
Vercel serverless entrypoint. Vercel's Python runtime serves the ASGI `app`
exported here; every /api/* request is rewritten to this function (see vercel.json).
"""
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("ARTIFACTS_DIR", str(BACKEND / "artifacts"))
os.environ.setdefault("MAX_UPLOAD_MB", "4.4")  # Vercel caps request bodies at 4.5 MB

from app.main import app  # noqa: E402,F401
