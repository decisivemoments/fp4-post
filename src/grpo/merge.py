#!/usr/bin/env python3
"""Compatibility wrapper for the unified BitLinear merge tool."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from convert_bitlinear_to_standard import main  # noqa: E402


if __name__ == "__main__":
    main()
