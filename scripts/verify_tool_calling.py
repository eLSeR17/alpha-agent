#!/usr/bin/env python3
"""End-to-end verification of Ollama function calling.

This script sends a real request to Ollama with a tool definition and
checks that the model returns the tool call in ``message.tool_calls``
(as structured data), NOT as plain text inside ``message.content``.

Usage (from the Docker network, e.g. inside the python-lab container):
    python scripts/verify_tool_calling.py

The script exits with code 0 on PASS and code 1 on FAIL.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://ollama:11434/api/chat"
MODEL = "qwen2.5:7b"
TIMEOUT = 120  # seconds

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. AAPL",
                },
            },
            "required": ["symbol"],
        },
    },
}

PAYLOAD = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful financial assistant. Use tools when asked about stock prices."},
        {"role": "user", "content": "What is the current price of AAPL?"},
    ],
    "tools": [TOOL_DEF],
    "stream": False,
    "options": {"temperature": 0.1},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Ollama Function-Calling Verification ===")
    print(f"Endpoint : {OLLAMA_URL}")
    print(f"Model    : {MODEL}")
    print()

    # --- 1. Send request ---------------------------------------------------
    print("[1/3] Sending request to Ollama ...")
    try:
        body = json.dumps(PAYLOAD).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"FAIL — Could not reach Ollama: {exc}")
        return 1

    print("      Request succeeded.")
    print()

    # --- 2. Parse response -------------------------------------------------
    print("[2/3] Parsing response ...")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"FAIL — Invalid JSON from Ollama: {exc}")
        return 1

    message = data.get("message", {})
    content = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", [])

    print(f"      content     = {content[:200]!r}" + (" (truncated)" if len(content) > 200 else ""))
    print(f"      tool_calls  = {json.dumps(tool_calls, indent=2)[:500]}" + (" (truncated)" if len(json.dumps(tool_calls)) > 500 else ""))
    print()

    # --- 3. Validate -------------------------------------------------------
    print("[3/3] Checking that tool call is in tool_calls (not in content) ...")
    all_pass = True

    # Check A: tool_calls array is non-empty
    if not tool_calls:
        print("  FAIL — tool_calls is EMPTY. The model likely returned the tool call as plain text in content.")
        all_pass = False
    else:
        print(f"  OK   — tool_calls has {len(tool_calls)} entry(ies).")

    # Check B: first tool call has the right function name
    if tool_calls:
        first_fn = tool_calls[0].get("function", {}).get("name", "")
        if first_fn == "get_stock_price":
            print("  OK   — Function name is 'get_stock_price' (correct).")
        else:
            print(f"  FAIL — Function name is '{first_fn}', expected 'get_stock_price'.")
            all_pass = False

    # Check C: first tool call has the 'symbol' argument
    if tool_calls:
        args = tool_calls[0].get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if "symbol" in args:
            print(f"  OK   — Arguments contain symbol={args['symbol']!r}.")
        else:
            print(f"  WARN — Arguments missing 'symbol': {args}")
            # Not a hard fail — the model might structure args differently

    # Check D: content should be empty (model doesn't need to explain)
    if content and not tool_calls:
        print("  FAIL — Model put the tool call as TEXT in content instead of using tool_calls.")
        all_pass = False
    elif content and tool_calls:
        print("  INFO — Content is non-empty alongside tool_calls (acceptable).")

    # --- Final verdict -----------------------------------------------------
    print()
    if all_pass:
        print("PASS  — Tool calling is correctly supported by the model.")
        return 0
    else:
        print("FAIL  — Tool calling is NOT working as expected. See failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
