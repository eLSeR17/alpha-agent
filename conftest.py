"""Root conftest – makes ``alpha_agent`` importable from ``staging/src``."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src/`` is on the import path so ``import alpha_agent`` works.
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
