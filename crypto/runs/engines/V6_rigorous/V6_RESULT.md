# V6 RIGOROUS ENGINE BUILD — RESULT (2026-06-25)

Built EVERY non-MA family (12) at the EXACT same rigor as the MA V1.1/V2 family, per
CHARTER §ENGINE BUILD RIGOR. ALL 5 timeframes {1d, 4h, 2h, 1h, 30m}. 57 cells total.

## Architecture (identical to MA V1.1)
1. Systematic config grid: 50-100+ per family (not hand-picked)
2. TRAIN-positive selection: keep only configs with TRAIN compound > 0 (→ curated sub-band)
3. Decoupled 7d lookback window (signal uses trailing 7d only)
4. Multi-layer confirmation: raw signal AND close > EMA(lookback) AND EMA rising
5. Regime gate: V1 BTC SMA100 (causal, hysteresis)
6. Maker cost 0.0006 RT on transitions
7. Lag-1 causal replay (audited: cheat inflates +65→+4217%)
8. Random-null n=50 at engine deploy rate

## RESULT: 1 of 57 cells is selective. ADX|1d only.

| Family | TF | sel% | deploy | edge/wk | null | Sharpe | comp | verdict |
|--------|----:|---:|------:|-------:|----:|------:|-----:|---------|
| **ADX** | **1d** | **100%** | **67%** | **+2.26pp** | **100** | **3.38** | **+551%** | **SELECTIVE** |
| PSAR | 1d | 100% | 100% | +0.33 | 100 | 2.24 | +184% | tilt |
| OBV | 1d | 100% | 100% | +0.36 | 100 | 2.28 | +189% | tilt |
| DONCHIAN | 1d | 100% | 100% | +0.28 | 100 | 2.20 | +177% | tilt |
| RSI | 1d | 100% | 98% | +0.27 | 54 | 2.18 | +176% | tilt (weak null) |
| SUPERTREND | 1d | 100% | 100% | +0.16 | 0 | 2.13 | +164% | tilt |
| MFI | 1d | 100% | 100% | +0.17 | 0 | 2.09 | +164% | tilt |
| VWAP | 1d | 100% | 99% | +0.15 | 0 | 2.14 | +163% | tilt |
| all@4h | — | 100% | 98-100% | -0.10 to +0.20 | 0 | ~1.8 | +136-155% | neutral/tilt |
| all@2h | — | 100% | 99-100% | -0.19 to +0.15 | 0 | ~1.9 | +131-164% | neutral |
| all@1h | — | 100% | 100% | -0.17 to -0.04 | 0 | ~1.8 | +134-149% | bleed |
| all@30m | — | 100% | 100% | -0.65 to -0.79 | 0 | ~1.3 | +80-92% | bleed |

## Why only ADX is selective

The TRAIN-positive selection step keeps ALL configs (selection rate = 100%) for every family
except ADX. This happens because in a 2020-2023 bull market under a bull regime gate, signals
like "close > Bollinger middle" or "RSI > 50" or "VWAP above" are TRUE most of the time —
so every config is individually positive on TRAIN. The TRAIN filter can't reduce the band
because there's nothing to filter out.

ADX is structurally different: its efficiency-ratio signal measures TREND QUALITY, which is
NOT always present even in a bull market. Choppy uptrends (which crypto has frequently)
produce low efficiency-ratios → ADX says "no" even when price is up. This is why it can be
selective.

The MA family's selectivity comes from a different mechanism: its BAND configs create
multi-period crossovers that disagree during whipsaw periods, and the ensemble of a CURATED
sub-band (many configs filtered down to 10-30) requires near-unanimity of different MA
timescales. Non-MA indicators don't have this multi-timescale-disagreement property.

## Sub-daily TFs: universally worse

1h edges: -0.04 to -0.17pp (mildly bleed from cost + noise)
30m edges: -0.65 to -0.79pp (severely bleed — high churn + noise destroys any signal)

At no timeframe below 1d does ANY family become selective or produce a positive edge.
The cost of turnover at higher frequency + signal noise overwhelms any timing value.

## Final engine roster (V6 rigorous, all timeframes)

SELECTIVE (deploy < 70%, edge > 0, null ≥ 95):
  - **MA V2 BAND (WMA|4h)** — the original validated engine from ma_wealth_driver
  - **ADX|1d (lb7d, thr mix, 67% deploy)** — the V6 discovery

NON-SELECTIVE TILTS (deploy 98-100%, small positive edge at 1d only):
  - PSAR|1d, OBV|1d, DONCHIAN|1d, RSI|1d (tilts, not discrete engines)

CONFIRMED NULL at all TFs (including 1h, 30m):
  - BOLLINGER, WILLR, CCI, KELTNER, SUPERTREND, VWAP, MFI at sub-daily
  - ALL families at 30m (universally bleed)

## Honest conclusion

Under full V1.1 rigor, the non-MA TI universe produces ONE genuine discrete selective
engine (ADX|1d) and several non-selective long-tilts. The earlier V3/V4 "robust tilt"
findings were artifacts of a less-rigorous architecture (no TRAIN selection, no decoupled
confirmation). When the same confirmation layer is applied, most of that "edge" disappears
because it was really just "being long under the gate" — and the confirmation layer ALSO
says long most of the time. The regime gate remains the dominant value lever.

This is the complete, honest, end-to-end static engine build at full rigor.
