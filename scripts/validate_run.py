#!/usr/bin/env python3
"""Standalone entry for independent data-root checks. Prefer `mlsys validate`."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.validate_run import main

if __name__ == "__main__":
    sys.exit(main())
