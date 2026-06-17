# Pipeline Framework (2026-05-01)

> **TL;DR:** Pipeline producers used to copy-paste argparse, atomic writes, ProcessPool dispatch, and skip-fresh logic in every script. Three new modules — `parquet_io`, `dispatch`, `cli` — consolidate these into a single source of truth. 10 producers have been migrated; future producers should use the framework by default.

## Why this exists

Across 26 pipeline stages, eight cross-cutting concerns drifted into per-script implementations:

1. **Parallelism** — ThreadPool / ProcessPool / hybrid / serial, each with its own worker-cap math
2. **Force / skip semantics** — file-existence vs mtime-vs-inputs vs manifest vs DAG-hash
3. **Error handling** — silent `return None`, print-and-continue, sys.exit(2), raise+caller-catch
4. **Output paths** — flat per-file vs per-asset dirs vs panels/daily; `_layout.*` only sometimes
5. **Atomic writes** — the G-AUDIT-020 contract (tmp + col-verify + rename) reimplemented in 7+ files
6. **Universe handling** — `--universe u10/u50/u100` vs hardcoded asset lists vs hand-tuned tables
7. **Logging / heartbeat** — `flush=True` per author preference; refresh.py's heartbeat goes silent if the producer doesn't print between heartbeats (the chimera_legacy Phase-2 stall surfaced this)
8. **Contract metadata** — `__contract__` dicts only on recently-touched files

All eight were enforced by *convention + audit reading*, not by *abstraction*. CLAUDE.md and the @browser directive list the rules; each producer re-implements them in its own style. New rules required N edits.

## The three primitives

### `src/pipeline/parquet_io.py` — atomic writes + skip-fresh

```python
from parquet_io import atomic_write_parquet, is_fresh, safe_unlink

# Replaces the 10-LoC G-AUDIT-020 inline pattern.
# tmp + write + col-verify + (unlink + rename). Cleanup on failure.
atomic_write_parquet(df, out_path, required_cols={"date", "asset"})

# Replaces ad-hoc out.exists() + max(input mtimes) comparisons.
# force=True always returns False (caller rebuilds).
if is_fresh(out, input_paths, force=args.force):
    print(f"[stage] SKIP fresh: {out.name}")
    return

# Force-rebuild helper: deletes if present, returns True iff something was removed.
if args.force and safe_unlink(out):
    print(f"[stage] --force: cleared stale {out.name}")
```

### `src/pipeline/dispatch.py` — parallel task dispatch

```python
from dispatch import run_per_task

def _build_one_asset(symbol: str, threshold: float, ...) -> dict:
    """Worker MUST return a dict with 'status' field. 'ok' = success.
    Other fields surface in per-task progress logs."""
    ...
    return {"status": "ok", "symbol": symbol, "n_bars": n, "elapsed_s": t}

# Replaces the 30+ LoC per-script ProcessPool + futures + result handling.
# Captures worker exceptions per-task. Auto-caps workers to min(workers, len(tasks), MAX).
# Loud per-task progress lines (heartbeat-friendly for refresh.py).
# sys.exit(2) on full-fail (no-silent-failure invariant).
run_per_task(
    tasks,                        # list of arg-tuples; tasks[i][0] is task id
    _build_one_asset,             # worker function
    workers=args.workers,
    mode="process",               # serial | thread | process
    stage_name="dib",
    progress_summary_keys=["n_bars", "elapsed_s"],
)
```

### `src/pipeline/cli.py` — standard CLI surface

```python
from cli import add_standard_args, resolve_assets

ap = argparse.ArgumentParser()
add_standard_args(ap, default_workers=8, date_window=True)
# Adds: --workers --force --universe --assets --dry-run [--start --end]
args = ap.parse_args()

# Replaces ad-hoc universe-resolution blocks (15+ copies).
# Priority: --assets > --universe (UniverseLoader) > given default.
# Always announces the resolution per @browser B5/B3 (LOUD).
symbols = resolve_assets(args, default=["BTCUSDT", "ETHUSDT"], stage_name="dib")
```

## Producer template (copy this for new stages)

```python
"""<stage description>"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import polars as pl

# Framework primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parquet_io import atomic_write_parquet, is_fresh
from dispatch import run_per_task
from cli import add_standard_args, resolve_assets

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "..."
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_one_asset(symbol: str, fps: list[str], out_path: str) -> dict:
    """ProcessPool worker: build artifacts for one asset's full window."""
    # ... business logic, write via atomic_write_parquet ...
    atomic_write_parquet(df, out_path, required_cols={"date", "asset"})
    return {"status": "ok", "symbol": symbol, "n_rows": len(df)}


def main():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=1)
    args = ap.parse_args()

    symbols = resolve_assets(args, stage_name="<stage>")

    tasks = []
    n_skipped = 0
    for symbol in symbols:
        fps = sorted(...)  # input files
        out = OUT_DIR / f"{symbol}.parquet"
        if not fps:
            print(f"[<stage>] {symbol} no inputs; skip", flush=True)
            continue
        if is_fresh(out, fps, force=args.force):
            print(f"[<stage>] {symbol} SKIP (fresh: {out.name})", flush=True)
            n_skipped += 1
            continue
        tasks.append((symbol, [str(p) for p in fps], str(out)))

    if args.dry_run:
        print(f"[<stage>] dry-run: {len(tasks)} tasks queued, {n_skipped} fresh-skipped")
        return

    if not tasks:
        print(f"[<stage>] nothing to build ({n_skipped} skipped)", flush=True)
        return

    run_per_task(tasks, _build_one_asset,
                  workers=args.workers, mode="process",
                  stage_name="<stage>",
                  progress_summary_keys=["n_rows"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
```

## Migration status (2026-05-01)

### Migrated (10 of 25 active producers)

| Stage | Path | Notes |
|---|---|---|
| bar_dib | `bars/dib_bars_fast.py` | DIB bars, ProcessPool |
| bar_range | `bars/range_bars_fast.py` | Range bars, ProcessPool |
| bar_runs_tick | `bars/runs_bars.py --modes tick` | Tick imbalance |
| bar_runs_volume | `bars/runs_bars.py --modes vol` | Volume imbalance |
| bar_adaptive_vol | `bars/adaptive_vol_bars.py` | Adaptive-vol bars |
| whale_activity | `ingest/whale_activity.py` | --workers honored from CLI |
| liq_daily_approx | `ingest/liquidations_approx.py` | Same |
| te_panel | `features/transfer_entropy_panel.py` | --workers via _TE_N_WORKERS knob |
| hawkes_branching | `features/hawkes_branching_ratio.py` | Atomic write via framework |
| s3_metrics_panel | `ingest/binance_s3_metrics.py` | Per-asset cache + panel |
| frontier_consolidator | `frontier_consolidator.py` | Per-asset slim + atomic |

### Not migrated yet (15 producers; opportunistic)

These still use inline patterns. They work; they're candidates for migration when next touched for any reason:

- `make_dataset.py` / `make_dataset_legacy.py` (gold layer; complex Phase 1+2)
- `fetch_all.py` (bronze; complex retry / manifest logic; mature)
- `ingest/binance_spot_klines.py` / `ingest/binance_vision_depth.py` / `ingest/etf_flows.py` / `ingest/multi_venue_listings.py` / `ingest/cross_exchange_spreads.py`
- `features/basis_signals.py` / `features/liq_features.py` / `features/lob_imbalance_panel.py` / `features/lob_proxy_daily.py` / `features/lob_proxy_panel.py` / `features/multi_venue_features.py` / `features/realized_volatility.py` / `features/top_trader_signals.py`

Single-pass aggregation panels in particular get less benefit (no per-asset parallelism to standardize); leaving them on the inline pattern is a reasonable default.

## Cross-cutting payoff

| Concern | Before | After |
|---|---|---|
| Atomic writes | 7+ inline copies of tmp+verify+rename | 1 helper, 1 site of change |
| Worker dispatch | 5 different ProcessPool/ThreadPool blocks | 1 helper with serial/thread/process modes |
| CLI flags | 15 ad-hoc argparses | 1 `add_standard_args` |
| Universe resolution | 4 different styles | 1 `resolve_assets` |
| Force semantics | 5 different conventions | 1 `is_fresh(force=...)` |
| Silent-failure invariant | Per-author `sys.exit(2)` | Built into `run_per_task` |
| Heartbeat-friendly logs | Per-author `flush=True` | Built into `run_per_task` per-task progress |

## CDAP enforcement

`config/_invariants.yaml` `atomic_write` section now matches both inline AND framework patterns:

```yaml
atomic_write:
  required_patterns:
    - "atomic_write_parquet|\\.tmp\\.parquet"
    - "atomic_write_parquet|rename|replace"
    - "atomic_write_parquet|read_parquet_schema|verify_cols"
```

Future drift back to copy-paste gets caught by the existing `python src/audit/check_invariants.py` pre-commit hook.

A new `framework_usage` section lists migrated stages so a CDAP regression flags any rollback.

## refresh.py orchestrator integration

`refresh.py --workers N` patches the `--workers` value in every stage's `producer_args` (only stages that *declare* `--workers` in `config/asset_dag.yaml` are affected — others are unaffected). 18 of 26 stages declare `--workers` and accept the override. The 8 that don't are intrinsically serial single-pass aggregations (etf_flows: 2 assets, multi_venue_listings: 3 venues, pre_train_gate: composite gate, etc.).

`refresh.py --all` walks every stage in topological order. `refresh.py --exclude STAGE [...]` drops specific stages from the walk (downstream stages read the excluded stage's existing on-disk output).

`refresh.py --live` is now a continuously-polling NRT viewer with:
- Configurable poll interval (`--live-interval`, default 2s)
- Stale-detection (`--live-stale`, default 300s) — exits if state file mtime gaps without `summary`
- De-duplication (only redraws when state actually changes)
- Auto-clear-screen (`--no-clear` for log-friendly output)
- Auto-exit on `summary` field appearing
- `--live-once` for the legacy one-shot snapshot

## Adding a new framework primitive

When a 4th cross-cutting concern emerges (e.g., uniform retry policy, uniform progress-bar protocol):

1. Add the helper to `src/pipeline/<name>.py` with `__contract__` dict
2. Smoke test with synthetic data
3. Migrate 1-2 high-traffic producers as proof-of-pattern
4. Add a CDAP rule to `config/_invariants.yaml::framework_usage`
5. Update this doc with the new section

Don't migrate every producer in one wave. Opportunistic migration (whenever a stage is touched for a real reason) is sufficient because the framework is *additive* — old code keeps working. Big-bang rewrites burn time and risk regressions.

## Delta rebuild (post-2026-05-01)

A third state between fresh-skip and full-rebuild: append only the missing
date(s). Lives as `parquet_io.delta_state()` for per-asset producers and
`parquet_io.panel_delta_state()` for multi-asset panels.

### Three semantic modes

```python
delta_state(out_path, input_paths,
            force=False,
            date_from_filename=...,    # callable Path -> date
            burn_from_first_gap=False, # see (2)
            window_days=0)             # see (3)
```

1. **Pure set-difference** (default; per-day independent producers): fills
   ONLY the specific missing dates. Day X's output doesn't depend on day Y.
   Safe for: bars/dib, bars/range, bars/runs, hawkes, whale, liq, lob_proxy_bars.

2. **Burn-from-first-gap** (`burn_from_first_gap=True`): on detecting a
   mid-stream gap, rebuild EVERY input from the first gap date forward.
   Use when Binance just backfilled a previously-missing date and you want
   the chain reconstructed deterministically from there.

3. **Windowed delta** (`window_days=W`): extend the rebuild backwards by W
   days from the existing max so rolling-window features get their tail
   recomputed. Required for: bars/adaptive_vol (W=30, sigma_30d threshold),
   te_panel (W=90), rv_jump_panel (W=20).

### Corruption guards (the load-bearing edge case)

`delta_state` accepts two optional kwargs that automatically fall through to
**full rebuild** when the existing output is corrupt:

```python
delta_state(...,
    required_cols={"date", "asset", "feature_x"},   # schema check
    max_null_rate={"feature_x": 0.05},              # null-rate sentinel
)
```

- **Schema mismatch** — if any `required_cols` is missing from the existing
  parquet (e.g., a new feature was added between builds), rebuild.
- **Null corruption** — if any `max_null_rate` column has a higher null rate
  than the budget (e.g., a prior build left feature_x null for 80% of rows),
  rebuild. Prevents the prior failure mode where corrupted features
  propagated forever via append-only delta.

Both checks are performed via `validate_existing(path, required_cols, max_null_rate)`
which returns `(ok, reason)`. The reason flows into `delta['reason']` so the
log says exactly why a rebuild was triggered.

### Recommended rebuild on suspected corruption

If you suspect the existing output is corrupt for any reason not caught by
the automated guards (e.g., off-by-one bug in past date range, wrong
threshold value, etc.):

```bash
# Force rebuild for a single stage
python src/pipeline/refresh.py --target whale_activity_daily --scope u50 --force

# Force rebuild across the pipeline
python src/pipeline/refresh.py --scope u50 --workers 12 --force --all

# Per-producer flag for surgical rebuild on suspected corruption
python src/pipeline/bars/dib_bars_fast.py --force --assets BTCUSDT
```

For producers that have `--burn-from-first-gap`: pass the flag so the
rebuild is contiguous from the first detected gap, not just patches the
specific missing date(s).

### Panel-mode delta (multi-asset producers)

`whale_activity` and `liquidations_approx` write a single panel parquet
where rows are `(asset, date, ...features)`. Use `panel_delta_state` instead:

```python
delta = panel_delta_state(
    OUT_PATH, per_asset_dates,         # dict {asset_root: [date, ...]}
    force=args.force,
    required_cols={"date", "asset", "key_feature"},
    max_null_rate={"key_feature": 0.05},
)
# delta["mode"]: 'fresh' | 'append' | 'rebuild'
# delta["per_asset_new_dates"]: dict {asset_root: [date, ...]} -- subset to build
```

Then `append_panel_parquet(out, new_rows_df)` drops existing `(asset, date)`
rows that overlap the new ones, concats, sorts, atomic-writes. Idempotent
under repeated calls with the same input.

### Per-stage delta-safety table (post-2026-05-01 round 3)

| Stage | Mode | Window | Notes |
|---|---|---|---|
| bars/dib | pure set-diff | 0 | per-day independent |
| bars/range | pure set-diff | 0 | per-day independent |
| bars/runs (tick / vol) | pure set-diff | 0 | per-day independent |
| bars/adaptive_vol | windowed | 30 | sigma_30d threshold needs context |
| whale_activity | panel pure | 0 | (asset, date) keyed |
| liquidations_approx | panel pure | 0 | (asset, date) keyed |
| hawkes_branching | panel pure | 0 | manual `_load_existing_keys` + framework corruption guard |
| te_panel | windowed | 90 | per-window TE; min_anchor_date filter + delta append |
| rv_jump_panel | windowed (panel) | 30 | EMA-seeded delta append; rebuilds last 30d for context |
| frontier_consolidator | NOT delta-able | — | per-asset silver join+ffill; ffill makes mid-update wrong; full rebuild is cheap (<5s/asset) |
| chimera_legacy / chimera_v51 | NOT delta-able | — | Phase 2 cross-asset depends on full panel |
| etf_flows / multi_venue_listings | NOT delta-able | — | full HTML scrape, no per-day source |
| Single-pass aggregations | NOT delta-able | — | basis_features / multi_venue_features / s3_features / liq_features (cheap full rebuild from upstream) |

### Why frontier_consolidator is NOT delta-able

The consolidator joins multiple silver source panels (hawkes, te, rv,
whale, liq, basis, ...) onto a contiguous per-asset date index, then
forward-fills NULL across sources. The forward-fill creates a hidden
dependency: an existing row's value at date D may be the ffill of a
source's last-observation as of D-K. If a NEW silver-source row arrives
for date D-K+1, it should propagate into D's ffilled value — but a pure
delta-append wouldn't recompute D.

So either:
- Full rebuild every consolidation (current behaviour; ~3s/asset, cheap)
- Detect which silver sources have new data, find earliest such date,
  rebuild from that date forward (windowed delta with cross-source-aware
  cutoff). Substantial complexity for marginal speedup.

The pragmatic choice: full rebuild. Marked NOT delta-able in the table.

## Provenance

- 2026-05-01 morning — initial framework + 10-producer migration
- 2026-05-01 afternoon — launch-date harvester (Binance exchangeInfo) →
  98K HTTP requests saved at u50; smart-skip for unlisted-pre-listing dates
- 2026-05-01 evening — delta rebuild (gap-aware + burn-from-first-gap +
  windowed + corruption guards); 4 bar builders + whale + liq migrated to
  delta semantics
- 2026-05-01 night — round 3: hawkes harmonized to corruption guard;
  te_panel windowed delta (W=90, min_anchor_date filter); rv_jump_panel
  windowed delta (W=30, EMA seeded); frontier_consolidator marked
  NOT delta-able with explicit rationale
- Predecessor: `docs/PIPELINE_HARDENING_2026_04_25.md` (the SOTA layout that this builds on)
- Driver issue: chimera_legacy Phase-2 O(N²) stall (2026-04-29 → 2026-05-01) exposed how fragmented the parallel-dispatch story was; user-mandated pipeline harmonization
