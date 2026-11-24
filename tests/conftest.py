import os
import sys
import tempfile
from pathlib import Path

# Run logs from tests go to a temp directory, never the repository's runs/.
os.environ.setdefault("MLSYS_RUNS_DIR", tempfile.mkdtemp(prefix="mlsys_test_runs_"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
