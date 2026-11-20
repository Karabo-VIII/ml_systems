# Strategy Layer Usage Guide

End-to-end operational guide for the V4 Crypto System strategy layer:
data pipeline, strategy engines, paper trading, and live operations.

---

## System overview

```
Raw Binance data                     Strategy layer                  Output
─────────────────────────────────    ──────────────────────────      ───────────
data.binance.vision daily zips       16 engines + DNA routing        signals
  (aggTrades, funding, metrics)      HoldingPeriodController         trade log
         │                           Realistic cost model            equity tape
Binance REST API (intraday)          Macro gate + regime gates       rolling metrics
         │                           ContextualMetaController
         ▼
data/raw/{ASSET}/                    30% swing + 70% short-term
         │                                  │
make_dataset.py                         ▼
         │                           paper_trader_v2.py
         ▼                           live_orchestrator.py
data/processed/*_chimera.parquet     signals_live.py
                                     live_loop.py
```

---

## 1. Data pipeline

### 1.1 Fetch raw data from Binance

```powershell
# Full historical fetch (or catch-up on gaps)
python src/pipeline/fetch_all.py

# Fetch from a specific start date
python src/pipeline/fetch_all.py --start-date 2026-04-01

# Fetch specific assets only
python src/pipeline/fetch_all.py --assets BTC/USDT ETH/USDT SOL/USDT

# Retry previously confirmed-missing dates (Binance backfills)
python src/pipeline/fetch_all.py --recheck-missing
```

**What it does**: downloads daily zip files from `data.binance.vision`:
- `data/raw/{ASSET}/aggTrades/{DATE}.parquet` — tick-level trade data
- `data/raw/{ASSET}/funding/{DATE}.parquet` — funding rates (perp)
- `data/raw/{ASSET}/metrics/{DATE}.parquet` — open interest snapshots

**Smart skipping**: maintains per-asset manifest tracking which dates have been fetched, which are confirmed missing (Binance didn't publish), and which need retry.

**Rate limits**: ~3-4s per asset per date. For 50 assets × 30 missing days = ~1 hour. Use `--reverse` in a second terminal to fetch from both ends.

### 1.2 Build chimera features (dollar bars + 41 features)

```powershell
python src/pipeline/make_dataset.py
```

**What it does**:
1. Per-asset: raw aggTrades → dollar bars (volume-sized, ~5-min resolution)
2. Compute 34 base features per dollar bar (SOTA microstructure features)
3. Cross-asset enrichment: +7 cross-sectional features (BTC lead, dispersion, etc.)
4. Compute 10 multi-horizon return targets + regime_label
5. Write to `data/processed/{ASSET}_v50_chimera.parquet`

**Output schema**: 41 features + 10 targets + regime_label per dollar bar. See `CLAUDE.md` for full feature table.

**When to run**: after any `fetch_all.py` run that added new raw data. Idempotent — safe to rerun.

### 1.3 Inspect data health

```powershell
# Quick per-asset check
python src/pipeline/inspect_dataset.py --asset btcusdt --quick

# Full pipeline health
python src/pipeline/inspect_pipeline.py
```

---

## 2. Data split boundaries (CRITICAL — prevents leakage)

### The problem

The system uses ratio-based splits (50/20/20/10) by default. When new data arrives and chimera is rebuilt, these ratios **shift** — bars that were "unseen" become "OOS", and former "OOS" bars leak into "val". This contaminates training data and invalidates model integrity.

### The fix: date-based frozen boundaries

`config/data_config.yaml` now contains explicit date boundaries:

```yaml
splits:
  train_end:    "2023-07-01"    # Historical training data (frozen)
  val_end:      "2024-05-15"    # Validation (frozen)
  oos_end:      "2025-03-15"    # Out-of-sample evaluation (frozen)
  unseen_start: "2025-03-15"    # Backtesting + paper trading
  live_start:   "2025-09-03"    # Paper-trading equity baseline
  purge_bars:   400              # Gap between segments
```

### Segment usage rules

| Segment | Date range | Who uses it | Rules |
|---|---|---|---|
| **TRAIN** | 2020-01-01 → 2023-07-01 | Model training ONLY | NEVER touch during evaluation. Frozen. |
| **VAL** | 2023-07-01 → 2024-05-15 | Hyperparameter selection, early stopping | Used during training loop. |
| **OOS** | 2024-05-15 → 2025-03-15 | Strategy evaluation, walk-forward testing | Bull/bear regime performance. |
| **UNSEEN** | 2025-03-15 → 2025-09-03 | Final backtesting (the "held-out" set) | Touched ONCE for weekly replay validation. |
| **LIVE** | 2025-09-03 → ongoing | Paper trading + new live data | Equity tracked from this date forward. |

### Handling new data

When you run `fetch_all.py` + `make_dataset.py`, new bars are appended to chimera. These new bars fall into the **LIVE** zone (after `live_start`). The TRAIN/VAL/OOS/UNSEEN boundaries do NOT shift.

```
 2020         2023-07      2024-05      2025-03      2025-09     TODAY
  │─── TRAIN ──│─── VAL ───│─── OOS ───│─ UNSEEN ──│── LIVE ──→ [growing]
  │            │            │            │            │
  │  FROZEN    │  FROZEN    │  FROZEN    │  FROZEN    │ NEW BARS LAND HERE
```

### For model retraining

When retraining WM or any model in the future:
1. Read `splits` from `config/data_config.yaml`
2. Discard all bars after `train_end` before training
3. Validate on `val_end` window
4. Report OOS metrics on `oos_end` window
5. Never peek at UNSEEN or LIVE data during training

---

## 3. Strategy engines (the decision-making layer)

### 3.1 Default stack (16 engines, post-audit)

Built by `build_extended_engines()` in `src/analysis/integrated_walk_forward.py`:

**Momentum / breakout**: `mom_30d`, `donchian_20`, `dow`, `cascade`, `discovered`
**Volume / microstructure**: `vol_break_20`, `vpin_trigger_20`, `activity_spike`, `activity_breakout_20`, `volofvol_20_60`
**Structural**: `efficiency`, `funding_carry`
**WM-based**: `wm_ensemble_h4` (momentum proxy until U50 retrain)
**Combinatorial**: `pairs_meanrev`, `and_vpin_trigger_20_efficiency`

### 3.2 DNA routing

Per-engine bucket filtering (`src/strategy/engine_dna_filter.py`):
- `sharpe_90d`, `efficiency`, `pairs_meanrev` → BLUE/STEADY only
- `vol_break_20`, `activity_spike`, `funding_carry` → VOLATILE/DEGEN only
- `wm_ensemble_h4` → excludes DEGEN (U10 training scope)
- Others → all buckets

### 3.3 Two deployment books

| Book | Purpose | Capital | Max hold | Engines |
|---|---|---|---|---|
| **Swing (Config B)** | Multi-day trend-follow | 30% | 30 days | All 16, macro-gate ON |
| **Short-term (v2 short_3d)** | Event-driven speculation | 70% | 3 days | 4 (and_gate, efficiency, funding_carry, discovered) |

---

## 4. Paper trading — daily workflow

### 4.1 First-time initialization

```powershell
# Initialize paper trader with $10K fresh capital
python src/analysis/paper_trader_v2.py --init --capital 10000
```

This runs a full warmup replay (engine state built from history) and sets equity to $10K at the current last bar. All prior backtest PnL is discarded; equity starts fresh.

### 4.2 Daily data refresh + paper-trade advance

**Option A — One-command orchestrator (recommended)**

```powershell
python src/analysis/live_orchestrator.py --run
```

This chains:
1. Check chimera panel's last date
2. If gap ≥ 1 day → `fetch_all.py --start-date <last_date>`
3. `make_dataset.py` to rebuild chimera with new raw data
4. `paper_trader_v2.py --update` to advance paper-trade equity
5. `rolling_metrics.py --summary` to update 7d/14d per-asset metrics

**Option B — Manual step-by-step**

```powershell
# 1. Fetch new raw data
python src/pipeline/fetch_all.py --start-date 2026-04-15

# 2. Rebuild chimera features
python src/pipeline/make_dataset.py

# 3. Advance paper trader state
python src/analysis/paper_trader_v2.py --update

# 4. Update rolling metrics
python src/analysis/rolling_metrics.py --summary
```

**Schedule**: run once per UTC day, after ~01:00 UTC (when Binance publishes yesterday's daily zip on `data.binance.vision`).

### 4.3 Intraday live signals (sub-1-day)

```powershell
# One-shot live signals (fetches from Binance REST, merges into panel)
python src/analysis/signals_live.py

# Continuous polling every 15 minutes
python src/analysis/live_loop.py --interval-min 15 --quiet
```

**What happens**: fetches today's partial 1-day kline (OHLCV) + funding rate from Binance REST API per asset. Merges into the historical chimera panel in-memory. Runs the full engine stack. Emits live signals + marks open positions to current price.

**Data persistence**: each intraday poll saves to `data/raw/{ASSET}/intraday_snapshots/{YYYY-MM-DD}.parquet` (normal raw data folder, appends one row per poll).

**Does NOT advance equity or persist state** — only the daily orchestrator does that.

### 4.4 Status checks

```powershell
# Paper trader equity + trade count (no panel load)
python src/analysis/paper_trader_v2.py --status

# Orchestrator state (gap detection, internal gaps, recent runs)
python src/analysis/live_orchestrator.py --status

# One-shot signal report (uses only historical chimera, no live fetch)
python src/analysis/signals_today.py --capital 10000

# Rolling 7d/14d per-asset metrics
python src/analysis/rolling_metrics.py --summary
```

### 4.5 Reset / start over

```powershell
python src/analysis/paper_trader_v2.py --reset
python src/analysis/paper_trader_v2.py --init --capital 10000
```

---

## 5. Data flow — where everything lives

### Raw data (fetched from Binance)

```
data/raw/{ASSET}/
├── aggTrades/                    ← Daily zip downloads (tick-level)
│   ├── BTCUSDT-aggTrades-2026-04-15.parquet
│   └── ...
├── funding/                      ← Daily funding rate history
├── metrics/                      ← Daily OI + market data
└── intraday_snapshots/           ← Live REST polls (1 parquet/day, growing)
    └── 2026-04-16.parquet
```

### Processed data (chimera features)

```
data/processed/
├── BTCUSDT_v50_chimera.parquet   ← 41 features + 10 targets + regime
├── ETHUSDT_v50_chimera.parquet
├── ...
└── ZECUSDT_v50_chimera.parquet
```

### Paper trading state + logs

```
logs/paper_trader_v2/
├── state/                         ← Persisted engine state + equity
│   ├── swing_meta.json
│   ├── swing_engines.pkl
│   ├── short_meta.json
│   └── short_engines.pkl
├── trade_log.csv                  ← Every closed trade (book, asset, engine, PnL)
├── daily_snapshot.csv             ← Daily equity tape
├── rolling_metrics_7d.csv         ← Per-asset 7-day rolling stats
├── rolling_metrics_14d.csv        ← Per-asset 14-day rolling stats
├── live_actions.csv               ← New signals detected per intraday poll
├── live_mark.csv                  ← Open-position mark-to-market per poll
├── live_latest.json               ← Most recent structured signal payload
└── live_loop.log                  ← Run-level summary (one line per poll)

logs/live_orchestrator/
└── orchestrator_runs.csv          ← Audit trail of every daily run

logs/live_data/
└── snapshot_latest.json           ← Most recent Binance REST snapshot
```

---

## 6. Two fetching processes: pipeline vs paper trading

The system fetches data in two contexts. They are complementary, not conflicting:

### Pipeline fetch (daily, authoritative)

| Step | Script | What it fetches | Where it saves |
|---|---|---|---|
| Raw pull | `fetch_all.py` | Tick-level aggTrades + funding + metrics from `data.binance.vision` | `data/raw/{ASSET}/{aggTrades,funding,metrics}/` |
| Feature build | `make_dataset.py` | Aggregates raw → dollar bars → 41 features | `data/processed/*_chimera.parquet` |
| Equity advance | `paper_trader_v2.py --update` | Reads chimera, runs engines, advances state | `logs/paper_trader_v2/` |

**Cadence**: once per UTC day (after Binance publishes yesterday's zip).
**Authority**: this is the SOURCE OF TRUTH for features, equity, and trade log.

### Intraday fetch (sub-daily, advisory)

| Step | Script | What it fetches | Where it saves |
|---|---|---|---|
| Live poll | `fetch_binance_live.py` | Today's partial 1d kline + funding rate via REST API | `data/raw/{ASSET}/intraday_snapshots/` + `logs/live_data/` |
| Signal | `signals_live.py` | Merges live data into chimera IN MEMORY, runs engines | `logs/paper_trader_v2/live_*.csv` |
| Monitor | `live_loop.py` | Repeats poll+signal every N minutes, detects new signals | `logs/paper_trader_v2/live_*.csv` |

**Cadence**: every 5/15/60 minutes.
**Authority**: advisory only — emits signals but does NOT advance equity or persist state. Equity only advances on the daily pipeline run.

### Why two processes don't conflict

1. **Pipeline writes to `data/raw/` and `data/processed/` (chimera).**
2. **Intraday writes to `data/raw/{ASSET}/intraday_snapshots/` (separate subfolder)** + `logs/`.
3. **Neither writes to the other's files.**
4. **Paper trader v2 only reads chimera and persists to `logs/paper_trader_v2/state/`.**
5. **Intraday signal runner reads chimera + live snapshot IN MEMORY, writes only to logs.**

No conflict. No data race. No leakage.

### How date-based splits prevent contamination

```yaml
# config/data_config.yaml
splits:
  train_end:    "2023-07-01"
  val_end:      "2024-05-15"
  oos_end:      "2025-03-15"
  unseen_start: "2025-03-15"
  live_start:   "2025-09-03"
  purge_bars:   400
```

When `make_dataset.py` re-runs after new data arrives:
- **New bars** (after today's date) are appended to chimera. They fall into the **LIVE** zone.
- **TRAIN/VAL/OOS/UNSEEN boundaries do NOT move** — they're dates, not ratios.
- Model retraining scripts MUST read these dates and filter data accordingly.
- Paper trader equity starts at `live_start` and only tracks LIVE-zone PnL.

**Future model retraining**: read `splits.train_end` from config, discard everything after that date before training. This guarantees the training set is FROZEN regardless of how much new data has been fetched.

---

## 7. Decision engine — how strategies actually decide

At each bar, the system makes decisions through this pipeline:

```
Bar t arrives (daily or partial-day)
    │
    ├─ [SWING BOOK]
    │   1. Macro gate check: BTC > SMA-200 AND slope rising?
    │      NO → hold cash, skip entries, continue held exits
    │      YES ↓
    │   2. All 16 engines compute_signals(asset_data, t)
    │   3. DNA routing filters: each engine only emits on allowed buckets
    │   4. ContextualMetaController weights engines per (regime × vol) cell
    │   5. Combined score = sum(w[engine] × signal[engine][asset])
    │   6. Conviction gate: |combined| >= 0.15
    │   7. Bucket cap: max 2 new entries per DNA bucket
    │   8. Top-K=5 selection from FLAT assets only
    │   9. Open positions via HoldingPeriodController
    │  10. HPC ticks all held positions (trailing stop, stop-loss, max-hold)
    │
    ├─ [SHORT-TERM BOOK]
    │   1. Per-engine regime gate: BTC 30d < -5% → block non-funding engines
    │   2. 4 engines compute_signals(asset_data, t)
    │   3. Signal threshold: AND-gate >= 0.15, singles >= 0.30
    │   4. Cooldown check: asset not in post-exit cooldown
    │   5. Rank by signal strength, take top max_concurrent=12
    │   6. Open with DNA-bucket-scaled position size
    │   7. Exit: stop-loss 3%, trailing 5%, max-hold 3 bars
    │
    └─ [COMBINED OUTPUT]
        ├─ trade_log.csv: every close (asset, engine, ret%, pnl%, reason)
        ├─ daily_snapshot.csv: equity tape (swing$, short$, total$)
        └─ live signals: (if intraday) what would fire at the NEXT bar
```

### Exit authority: HoldingPeriodController

Positions exit via whichever trigger fires first:
- **Trailing stop**: position drops N% from peak since entry
- **Stop-loss**: position drops N% from entry price
- **Max hold**: position held for N bars
- **Profit target**: (disabled in default configs — let winners run)

Swing: trail=10%, stop=5%, max_hold=30 bars.
Short-term: trail=5%, stop=3%, max_hold=3 bars.

### Warmup requirements

Engines need historical data to produce valid signals:

| Engine | Min bars |
|---|---|
| DayOfWeekEngine (lookback_fit=900) | **900** |
| CascadeEngine (fit_window=720) | **720** |
| Macro gate (SMA-200 + slope) | **220** |
| sharpe_90d | 90 |
| volofvol_20_60 | 60 |
| Other engines | 20-60 |

System uses `WARMUP_BARS=1000` to have all engines fully warm. First 1000 bars of chimera history are always available (earliest data is 2020-01-01). No paper-trade signals emit during warmup.

---

## 8. Monitoring and graduation checklist

### Weekly review (during paper trading)

Check these files every week:

| File | What to look for |
|---|---|
| `daily_snapshot.csv` | equity_total trending up; DD < -15% |
| `rolling_metrics_7d.csv` | per-asset hit rates above 30%; profit_factor > 1.0 |
| `live_actions.csv` | new signals making sense for current regime |
| `live_mark.csv` | open positions' unrealized PnL (no large uncontrolled losses) |

### Graduation criteria (to live capital)

After 30-90 days of paper data:
- [ ] Realized Sharpe > +2.0
- [ ] Realized max DD < -15%
- [ ] Realized slippage within 50bps of modeled (`trade_log.csv ret_pct` vs `cost_pct`)
- [ ] No schema-version invalidations in the last 7 days
- [ ] You can explain what each engine is firing on and why
- [ ] `daily_snapshot.csv macro_gate_on` column behaves correctly (0 during confirmed bear)

---

## 9. Quick reference

### Scripts

| Script | Purpose | When |
|---|---|---|
| `fetch_all.py` | Download raw Binance data | Daily or when catching up |
| `make_dataset.py` | Build chimera features from raw | After any fetch |
| `live_orchestrator.py --run` | Full daily pipeline (fetch→features→paper→metrics) | Once/day at 01:15 UTC |
| `paper_trader_v2.py --init` | Initialize paper trader ($capital) | Once (first time) |
| `paper_trader_v2.py --update` | Advance paper-trade state | Via orchestrator |
| `paper_trader_v2.py --status` | Quick equity summary | Anytime |
| `signals_today.py` | One-shot signals from historical chimera | Ad-hoc check |
| `signals_live.py` | One-shot signals with live Binance data | Intraday check |
| `live_loop.py --interval-min 15` | Continuous signal polling | Run alongside orchestrator |
| `rolling_metrics.py --summary` | 7d/14d per-asset stats | Via orchestrator or ad-hoc |

### Key config files

| File | What it controls |
|---|---|
| `config/data_config.yaml` | Assets, bar sizes, date-based splits, purge gaps |
| `src/strategy/universe.py` | U10/U24/U50 definitions, DNA_BUCKET map |
| `src/strategy/engine_dna_filter.py` | Per-engine allowed DNA buckets |
| `src/strategy/realistic_cost_model.py` | Per-DNA cost + stress multipliers |

### Cron / scheduled task template

```
# Run at 01:15 UTC every day
15 1 * * *  cd /path/to/v4_crypto_stystem && python src/analysis/live_orchestrator.py --run >> logs/live_orchestrator/cron.log 2>&1

# Live loop (run as a background service or screen/tmux session)
python src/analysis/live_loop.py --interval-min 15 --quiet
```

---

## 10. Docs reference

| Doc | Covers |
|---|---|
| `CLAUDE.md` | Architecture, feature tables, invariants, validation gates |
| `docs/regime_aware_swing_capture.md` | Frozen swing engine v1.0 reference |
| `docs/short_term_speculation_research.md` | Short-term research findings |
| `docs/deployment_config_2026_04_16.md` | 8-layer deployment config comparison |
| `docs/paper_trading_deployment.md` | Paper trader operational guide |
| `docs/live_deployment_guide.md` | Live orchestrator + daily pipeline |
| `docs/intraday_live_guide.md` | Intraday REST signals + polling loop |
| `docs/unseen_replay_2026_04_16.md` | Weekly replay validation |
| `docs/strategy_layer_audit_2026_04_16.md` | Initial engine audit |
| `docs/tier123_implementation_findings.md` | Tier 1/2/3 implementation results |
| `docs/short_term_addendum_2026_04_16.md` | Multi-bear + combined capital + v3 |
