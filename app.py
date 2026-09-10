"""Entrypoint for the AlphaAgent HTTP API server.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000

Or with reload during development:
    uvicorn app:app --reload
"""

from __future__ import annotations

import os
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

from alpha_agent.api import app  # noqa: E402

__all__ = ["app"]
