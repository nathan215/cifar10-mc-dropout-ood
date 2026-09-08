"""
Ensure the repo root (which holds metrics.py, models.py, etc. as top-level
modules, not a package) is importable regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
