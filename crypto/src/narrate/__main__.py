"""src/narrate/__main__.py -- package entry point so `python -m narrate` works.

When invoked as `python -m narrate` from the repo root with `src/` on PYTHONPATH
(e.g. PYTHONPATH=src python -m narrate ...) the import resolves normally.

When invoked directly as `python src/narrate/__main__.py` the sys.path shim below
adds src/ so `import narrate` and `import pipeline` both resolve.
"""
import os
import sys

# Ensure src/ is on the path -- needed when run without PYTHONPATH=src
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from narrate.cli import main  # noqa: E402

sys.exit(main())
