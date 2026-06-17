#!/usr/bin/env python3
"""No-PYTHONPATH launcher for the metaop manager -- shell-agnostic (works in PowerShell, cmd, bash).

  python scripts/autonomy/run_metaop.py launch  --backend cli --objective "..." --durable --thread t1
  python scripts/autonomy/run_metaop.py status  --thread t1
  python scripts/autonomy/run_metaop.py resume  --thread t1 --budget 16
  python scripts/autonomy/run_metaop.py approve  --thread t1 --node <id>

(Adds scripts/autonomy to sys.path so `metaop` imports without PYTHONPATH.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metaop.manager import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
