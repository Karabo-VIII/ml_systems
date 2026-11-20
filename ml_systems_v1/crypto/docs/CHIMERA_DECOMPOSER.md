# Chimera Decomposer / Viewer

**Purpose (user intent, 2026-06-09):** *select a time period + a timeframe (+ an asset or the whole universe) and VIEW
how all the chimera features behaved over that slice.* A DESCRIPTIVE inspection tool — not signal-mining.

> **THE LENSES (2026-06-09).** The decomposition harness reads a slice through complementary lenses:
> - **chimera lens** (this page): per-window, exact, narrative — `decompose` (the full feature table) + `narrate`
>   (the prose story). Reads features that ALREADY EXIST in chimera.
> - **math/econ lens**: [`ECONOMETRIC_SIGNATURE.md`](ECONOMETRIC_SIGNATURE.md) (`python -m mining.econometric_signature`)
>   — agnostic, whole-series ("what *kind of process* is this?": GARCH / tails / Hurst / stationarity / jumps).
>   COMPUTES canonical estimators (no config choice).
> - **configured-TI lens**: [`TI_HARNESS.md`](TI_HARNESS.md) (`python -m mining.ti_harness`) — rolling-window
>   technical indicators (RSI/MA/MACD/Bollinger/ADX/ATR). Fundamentally different: a TI does NOT exist until you
>   CONFIGURE it (family + params + thresholds); the SAME window reads differently under a different config, so the
>   config is a first-class, echoed input.
>
> **Pivot:** agnostic "what process is this" → econometric signature; what a *configured* indicator set says over the
> window → ti_harness; exact feature behaviour / the story at a period → `decompose` + `narrate` below. Every lens is
> tracked in the framework store (01_mining tools + 00_research docs).

Two complementary tools (the chimera lens):

## Setup (one-time, makes `python -m ...` work for any `src/` package)
The repo keeps source under `src/`, so `python -m mining.decompose` needs `src` on the path. Either run the **script
form** (works in any shell, zero setup) or add `src` as a venv source-root once so `-m` works from the repo root.
```
# always-works, no setup (PowerShell/bash):
python src/mining/decompose.py --asset BTC --cadence 4h --start 2025-01-01 --end 2025-02-01 --plots

# OR one-time: register src as a source root in the active venv (then `python -m mining.decompose ...` just works):
python -c "import sysconfig,os; open(os.path.join(sysconfig.get_paths()['purelib'],'v4_src_root.pth'),'w').write(os.path.join(os.getcwd(),'src'))"
```
(The `.pth` lives in the venv, not git — re-run the one-liner after recreating the venv. It also makes
`python -m narrate ...` work without `PYTHONPATH`.)

## 1. `src/mining/decompose.py` — the complete feature-behaviour table
Every chimera feature for a chosen `(asset | universe, period, timeframe)`, grouped by family, with: window-mean,
**percentile vs full history** (is this feature unusually high/low this period?), in-window trend, min/max, 3-sigma
spike count, and an `UNUSUAL` flag. Reuses the narrate `feature_map` taxonomy (100% column coverage) + ChimeraLoader.

```
# ASSET mode -- every feature's behaviour for one asset over a window
python -m mining.decompose --asset BTC --cadence 4h --start 2025-01-01 --end 2025-02-01

# UNIVERSE mode -- top performers in the window + each feature's cross-asset behaviour (median + extreme assets)
python -m mining.decompose --universe u100 --cadence 1d --start 2025-10-01 --end 2025-11-01 --top-n 25

# JSON out (for programmatic use); also always written to runs/mining/decompose_<tag>.json
python -m mining.decompose --asset ETH --cadence 1h --start 2025-03-01 --end 2025-03-15 --json
```
Cadences: `1d 4h 1h 30m 15m`. Asset accepts `BTC` or `BTCUSDT`. Window flags `--start/--end` are ISO dates (optional;
omit for full history).

### Visual mode (`--plots`)
Add `--plots` to render PNGs to `plots/<date>/` (same convention + style as `src/pipeline/inspect_dataset.py`):
- **asset**: a `feature x time` z-score **heatmap** (every live feature's trajectory over the window, grouped + labelled
  by family) with a price + regime-ribbon panel on top; plus a "most UNUSUAL features this window" percentile bar.
- **universe**: a top-performers bar + a `feature x asset` window-mean **heatmap** (z across assets, ranked to the most
  cross-asset-varying features so the grid is dense; globally-constant/sparse features are dropped).
```
python -m mining.decompose --asset BTC --cadence 4h --start 2025-01-01 --end 2025-02-01 --plots
python -m mining.decompose --universe u100 --cadence 1d --start 2025-10-01 --end 2025-11-01 --plots
```

## 2. `src/narrate/` — the curated PROSE read of the same slice (pre-existing)
A human-readable narration of an `(asset, cadence, period)`: per-family intensity + the most decision-relevant
features, regime, notable events, crypto-context caveats. Use when you want the *story*; use `decompose` when you want
the *full table*.
```
python -m narrate --asset BTC --cadence 4h --start 2025-01-01 --end 2025-02-01
```

## Supporting corpus (whole-history descriptive aggregates)
`src/mining/chimera_mine.py` builds `runs/mining/{corpus_<cad>.parquet, feature_catalog_<cad>.csv}` — per-asset
structure metrics + per-column health across u100 x {1d,4h,1h,30m,15m}. `analyze.py / deep_mine.py / conditional.py`
add regime/cluster/trend/seasonality lenses; see `docs/CHIMERA_MINING_FINDINGS_2026_06_08.md` (note: that report was
written under a *signal* lens — kept as a descriptive characterization, but the decomposer above is the intended tool).

No emoji (cp1252).
