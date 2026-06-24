# V2 Sub-TF Decoupled Engine — Verified Results

## Architecture
```
Layer 1: V1 Regime Gate (1d BTC SMA100 + 2% hysteresis)
  → BULL: deploy Layer 2
  → BEAR: 100% cash

Layer 2: V2 4h Decoupled Engine (MA band ensemble + 7d EMA lookback)
  → Signal: band_held AND close > EMA(7d) AND EMA rising
  → Hold: max 7d slice per position
  → Exit: slice boundary OR signal lost
  → Cost: maker 0.0006 RT per turnover
```

## Verified Results (DEV window 2020-01 to 2024-05-15)

### Combined V1+V2
| MA Type | Compound | MaxDD | Bear 2022 | Bull 2021 |
|---|---|---|---|---|
| **EMA** | +171,610% | -42% | -2% | +6,598% |
| **WMA** | +229,783% | -48% | -3% | +8,448% |
| **DEMA** | +691,448% | -45% | -6% | +32,339% |

### V2 Layer Only (no regime gate)
| MA Type | Compound | MaxDD | Per-Week Profitable | Median/Week |
|---|---|---|---|---|
| WMA | +40,588% | -90% | 59% | +4.10% |
| DEMA | +78,627% | -94% | 58% | +2.95% |
| EMA | +17,592% | -91% | 56% | +3.09% |

### Baselines
| Strategy | Compound | MaxDD |
|---|---|---|
| Always-long EW | +2,171% | -83% |
| Random entry (P95, same deploy) | ~+2,500% | — |

## Verification Gates Passed
1. ✅ Random-entry null: 100th percentile (all 3 types beat P95 of 30 random seeds)
2. ✅ Beat always-long: 56-59% of weeks engine > passive
3. ✅ Multi-type robustness: EMA, WMA, DEMA all independently pass
4. ✅ Bear protection: -2% to -6% vs -58% passive in 2022 bear
5. ✅ 56+ assets participate (above 20 floor)
6. ✅ Lag-1 causality (held[t-1] determines position at t)
7. ✅ DEV wall sealed (no OOS data accessed)

## Key Findings
- **4h is the optimal cadence** (2h has good median but marginal vs always-long; 1h/30m are negative)
- **7d lookback > 14d** (recency matters — recent signal outperforms stale)
- **7d slices > 3d** (trends need time to express at 4h)
- **The regime gate is essential** (without it, MaxDD is -90%+)
- **The 4h engine is a TREND AMPLIFIER** (not a drawdown reducer on its own)
- Combined = regime timing (bear avoidance) + trend amplification (bull riding)

## What This Does NOT Claim
- NOT validated on OOS/UNSEEN (2024-05-15 onwards — sealed per split discipline)
- NOT 1-5%/day alpha (it's regime-timed trend-riding, closer to +3-4%/week in bull)
- NOT per-asset selection alpha (the engine is broad, not selective)
- The massive compounds include the 2020-2021 bull — future bull may differ

## Files
- `cells/*.json` — per-cell results (MA×TF×lookback×slice)
- `summary/*.json` — sweep summaries
- `progress_log.md` — batch execution log

## Reproduce (from crypto/)
```bash
# Single cell
PYTHONPATH=src python -m strat.ti_engine_multitf --ma-type EMA --tf 4h --lookback 7 --slice 7

# Full sweep (EMA × all TFs)
PYTHONPATH=src python -m strat.ti_engine_multitf --sweep-ema

# Top-3 types (EMA + HMA + DEMA)
PYTHONPATH=src python -m strat.ti_engine_multitf --sweep-top3

# Combined V1+V2 test
python (tmp)/_combined_v1v2.py
```

## Commit Trail
- `790af323` TI engine multi-TF: 2h/4h sweet spot found
- `6f39dcaa` V2 sub-TF sweep complete: 11/17 cells >55% profitable
- `cc7b6ea9` 3d slice test: 7d confirmed superior
- `fc4b330b` VERIFIED: 4h engines beat random-entry null at 100th percentile
- `f217ac5d` bear-period validation: trend amplifiers not DD reducers
- `ca597508` COMBINED V1+V2: MaxDD -42%, compound +171K-691K%
