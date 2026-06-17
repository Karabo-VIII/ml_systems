> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments. See [docs/SPLIT_DISCIPLINE.md](../../docs/SPLIT_DISCIPLINE.md).

# Pipeline Bug: Binance aggTrades scale + sort regressions (2026-05-12 / 13)

**Severity**: CRITICAL — silent corruption of all 2025-Q4+ alt-bar output, with potential downstream impact on chimera_v51 2025-Q4+ features.

## Summary

Two distinct regressions in Binance aggTrades raw input that the bar builders did not handle. Both surfaced during F2 alt-bar pilot rebuild (BTC+ETH 2020-2026). Both fixed via single shared utility `src/pipeline/bars/_aggtrades_utils.py`.

## Bug 1: Timestamp scale switch (us, 16-digit) starting ~2025-Q3

**Empirical signature**:
- 2024 BTC aggTrades: `timestamp[0] = 1705276800002` (13-digit ms)
- 2026 BTC aggTrades: `timestamp[0] = 1768435200261788` (16-digit us)
- AltBarLoader schema validator caught it at first load attempt:
  `RuntimeError: timestamp_ms out of 13-digit ms range. min=1577836800594 max=1778283284707756`

**Root cause**: Binance changed aggTrades archive precision from ms to us between 2025-Q2 and 2025-Q3. The 4 bar builders (`dib_bars_fast.py`, `range_bars_fast.py`, `runs_bars.py`, `adaptive_vol_bars.py`) read `pl.read_parquet(fp)` and pass `timestamp` through unmodified → output inherits us-scale → 13-digit ms project invariant violated → downstream joins with `move_events.parquet` (13-digit ms) silently broken.

**Fix**: `normalize_ts_to_ms(df, ts_col)` in `src/pipeline/bars/_aggtrades_utils.py`. Auto-detects scale (ms < 2e12, us < 2e15, ns >= 2e15), downscales to ms. Idempotent.

## Bug 2: aggTrades arrive unsorted starting ~2026-03

**Empirical signature**:
- 2026-03-04 BTC aggTrades: **1,135,215 of 2,525,468 rows have negative ts-diff to prior row** (45% unsorted)
- Adjacent rows after normalize: mean tick-to-tick price move = **2.92%** (vs 0.0003% on 2024 sorted data)
- Range builder: BTC 2026 produced **31.9M bars in 128 days** (vs ~30/day expected at 0.5% threshold). Median tick_count = 2 (vs 16,290 in 2024). Every bar closed after 1-2 ticks because consecutive trades came from different times.

**Root cause**: Binance aggTrades 2026-03+ arrive interleaved from multiple shards/threads; the per-day file is no longer pre-sorted by `transact_time`. Bar builders assumed sorted input.

**Fix**: `prepare_aggtrades(df, ts_col)` in `src/pipeline/bars/_aggtrades_utils.py` does normalize + sort. Used by all 4 builders.

## Files touched

```
src/pipeline/bars/_aggtrades_utils.py         (new)
src/pipeline/bars/dib_bars_fast.py             (import + 1 call site)
src/pipeline/bars/range_bars_fast.py           (import + 1 call site, polars-then-pandas path)
src/pipeline/bars/runs_bars.py                 (import + 1 call site)
src/pipeline/bars/adaptive_vol_bars.py         (import + 1 call site)
```

## Detection mechanism

- Bug 1 caught by **AltBarLoader.\_validate()** at first load attempt (`src/oracle/alt_bar_loader.py`)
- Bug 2 caught by **F3 audit script** density check (`scripts/oracle/audit_alt_bar_pilot.py`); 6,644 bars/day BTC > 500 sane upper bound
- Both fixes verified post-rebuild: F2_v3 → F3 re-audit clean (10/10 OK)

## Spillover risk: OTHER aggTrades consumers

Inventory of `src/pipeline/` aggTrades consumers, per-file state (2026-05-13):

| File | Scale normalize | Sort by ts | Action needed | Status 2026-05-13 |
|------|-----------------|------------|---------------|-------------------|
| `features/transfer_entropy_panel.py` | NO | NO | both (sort matters for first/last price daily-ret semantics) | **PATCHED** ✓ |
| `features/hawkes_branching_ratio.py` | YES (`per_row_to_seconds`) | NO | sort + canonicalize to "timestamp" | **PATCHED** ✓ |
| `features/realized_volatility.py` | YES (`_ts_to_ms`) | NO | replaced ad-hoc with prepare_aggtrades | **PATCHED** ✓ |
| `features/lob_proxy_panel.py` | YES (own logic) | NO | canonicalize ts col + prepare_aggtrades | **PATCHED** ✓ |
| `ingest/whale_activity.py` | NO | NO | **SKIP — output is pure aggregation (sum/count by mask), no ts usage** | not needed |
| `ingest/liquidations_approx.py` | NO | NO | **SKIP — same; pure aggregation** | not needed |
| `build_panels.py` | NO | NO | **SKIP — no aggTrades read; consumes upstream panels** | not needed |

**Net: 4 of 4 consumers that ACTUALLY use timestamps in output are patched. The 3 audited-but-skipped do pure aggregation and are unaffected by the bugs.**

These feed `chimera_v51`. Impact:
- TRAIN period (2020-07 → 2023-07): **unaffected** (pre-bug era)
- VAL / OOS / UNSEEN (2024-Q2 → 2026-Q2): chimera features for the 4 patched consumers were potentially corrupted; **NOW FIXED at builder level**.
- **Rebuilds NOT yet run** — code is fixed; existing chimera artifacts for VAL/OOS/UNSEEN still reflect pre-fix builds. Specific chimera shards (15m / 1h / 4h / 1d / dollar) for 2025-Q3+ would need rebuild to pick up the patched consumer outputs. Compute cost: large (chimera rebuild is multi-day across u59).
- **Deploy guidance**: TRAIN-period work (MA specialist, F4-F8) is unaffected. F8 OOS+UNSEEN validation will use stale VAL/OOS chimera unless a targeted rebuild is run; flag this in F8 caveats rather than block on full rebuild.

**Pre-action debate verdict (2026-05-13)**: YELLOW. 4-file patch all idempotent; py_compile clean. Cross-file drift mitigated by per-file verification. Compute cost of rebuild is large; rebuild deferred with explicit F8 caveat.

## Prevention

Adopt as project invariant (CLAUDE.md candidate):

> **Any reader of `data/raw/<sym>/aggTrades/*.parquet` MUST call `prepare_aggtrades(df, "timestamp")` immediately after `pl.read_parquet(fp)`. Validators (AltBarLoader / ChimeraLoader) MUST reject ts outside [1.5e12, 2.0e12].**

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
