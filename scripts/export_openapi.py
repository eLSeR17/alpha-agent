#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to docs/openapi.json.

Usage:
    PYTHONPATH=src python scripts/export_openapi.py

The Swagger UI at /docs renders from the live schema; this export lets
reviewers and CI diff the API contract over time.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha_agent.api import app


def main() -> int:
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "openapi.json")
    schema = app.openapi()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")
    print(f"OpenAPI schema written to {os.path.abspath(out)}")
    print(f"  title={schema['info']['title']} version={schema['info']['version']}")
    print(f"  paths={sorted(schema['paths'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
