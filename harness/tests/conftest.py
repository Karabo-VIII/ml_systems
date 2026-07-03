"""pytest bootstrap: make `metaop` importable without an install, and isolate all harness bookkeeping under a
throwaway workspace so the suite never touches the user's real ./.harness_runs. No emoji (Windows cp1252)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# harness/ is the parent of tests/ -> put it on sys.path so `import metaop` resolves in-place (no pip install needed).
_HARNESS = Path(__file__).resolve().parents[1]
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

# Force a fresh temp workspace for the whole session (never inherit a real one) so default-path tests are hermetic.
os.environ["HARNESS_WORKSPACE"] = tempfile.mkdtemp(prefix="harness_test_ws_")
