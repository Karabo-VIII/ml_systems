# Pipeline v1 — reference only

This tree is the previous data layer (`crypto/src/pipeline` before 2026-11-20). It is kept so the history and the old formulas stay readable. Do not import it from production code and do not run it to build new gold.

The replacement is `crypto/src/pipeline_v2`. The contract is `crypto/docs/DATA_CONTRACT.md`.

## Why it was archived

Three orchestrators (`run_pipeline`, `refresh`, `refresh_full`), a large legacy/archived subtree, and several silent data bugs:

- Binance funding was fetched from the empty daily Vision prefix, and the parser expected `calcTime` / `fundingRate` while the monthly files use `calc_time` / `last_funding_rate`.
- aggTrades were deduped with `unique()` and no sort, which shuffled the parquet.
- `long_short_ratio` was written as the constant `1.0` when Vision did not ship the column.
- The v51 "cadence views" took the last dollar bar in a clock bucket instead of aggregating OHLCV.
- The v51 frontier joined daily features on calendar date plus one day, not on `knowable_at <= ts_ms`.
- Bar filenames used the `--start` year. A 2026 build could be named `*_2023.parquet`.
- Cross-exchange fetch aborted the job on an empty venue (`KeyError: 'date'`).
- `token_unlocks` pointed at a file that does not exist.
- Freshness checks were directory-level. The runner had no `--start` / `--end`.

## If an old script still needs this tree

The package is no longer named `pipeline` on `PYTHONPATH`. Point the script at this directory explicitly:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("crypto/archive/pipeline_v1")))
from chimera_loader import ChimeraLoader
```

That reads old gold files only. New bars and features come from `python -m pipeline_v2.cli`.
