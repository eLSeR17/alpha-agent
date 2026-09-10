#!/usr/bin/env python3
"""Create an AlphaAgent API key.

Usage:
    PYTHONPATH=src python scripts/create_api_key.py --name alice
    PYTHONPATH=src python scripts/create_api_key.py --name alice --rate-limit 30

The plaintext key is printed exactly once.  Store it somewhere safe —
it cannot be recovered later (only its hash is persisted).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha_agent.auth import APIKeyStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an AlphaAgent API key")
    parser.add_argument("--name", required=True, help="Human-readable key name (e.g. 'alice-dev')")
    parser.add_argument("--rate-limit", type=int, default=60, help="Requests per minute (default: 60)")
    parser.add_argument(
        "--db",
        default=os.getenv("ALPHA_AGENT_AUTH_DB", "auth.sqlite3"),
        help="Path to the auth database (default: auth.sqlite3)",
    )
    args = parser.parse_args()

    store = APIKeyStore(db_path=args.db)
    key_id, plaintext = store.create_key(args.name, rate_limit_per_min=args.rate_limit)

    print("=" * 60)
    print("API key created successfully")
    print("=" * 60)
    print(f"  Key ID      : {key_id}")
    print(f"  Name        : {args.name}")
    print(f"  Rate limit  : {args.rate_limit} req/min")
    print()
    print("  Use it in requests as:")
    print(f'    curl -H "X-API-Key: {plaintext}" http://localhost:8000/ask \\')
    print('         -d \'{"question": "What is the price of AAPL?"}\'')
    print()
    print("  !! The plaintext key is shown only once. Store it securely.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
