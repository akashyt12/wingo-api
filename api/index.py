"""Vercel Python serverless entrypoint (ASGI FastAPI app)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("WINGO_FORCE_NUMPY", "1")

from app import app  # noqa: E402,F401
