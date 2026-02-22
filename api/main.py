"""
TSR API shim — Re-exports the FastAPI app from backend.main
so that `python -m uvicorn api.main:app` works the same as
`python -m uvicorn backend.main:app`.
"""

from backend.main import app  # noqa: F401
