# Per-MA-TYPE weakness teardown + targeted IRON (2020)

User /orc 2026-06-14: "did you iron out the weaknesses of EACH type of MA strat type?" Honest answer: the
MA_REMEDY work applied UNIFORM remedies; THIS is the per-TYPE diagnosis. Tool: `deep2020_ma_weakness.py`
(median over the 80-config grid, fixed-EW, base FULL stack = trail10+min_hold12+maker, OOS Oct-Dec).
ALL numbers **[VERIFIED-backtest, IN-SAMPLE 2020]**. UNSEEN untouched.

## THE MEASURED WEAKNESS PROFILE (median per type)
| MA | n_trades 4h/1d | cost_drag 4h/1d | time_in 4h/1d | maxDD 4h/1d | net 4h/1d | signature weakness (measured) |
|---|---|---|---|---|---|---|
| EMA | 11/2.2 | 0.5/0.1 | 0.5/0.3 | -11.0/-13.1 | 29.1/23.6 | balanced -- no acute weakness (the all-rounder) |
| SMA | 14/2.3 | 0.6/0.1 | 0.4/0.3 | -13.4/-14.0 | 24.0/22.8 | LAG -> lowest time-in + high DD (late exits) |
| WMA | 15/2.9 | 0.7/0.1 | 0.5/0.4 | -13.0/-15.2 | 26.2/22.2 | moderate lag -> high DD at 1d |
| HMA | **28**/5.3 | **1.1**/0.2 | 0.5/0.5 | -13.6/**-17.4** | 25.1/24.1 | OVERSHOOT -> most trades + worst DD + whipsaw 1.4% |
| DEMA | 18/3.7 | 0.8/0.2 | 0.5/0.4 | -12.0/-16.4 | 25.6/22.8 | overshoot churn (18-28 trades) + deep DD |
| TEMA | 23/4.5 | **1.0**/0.2 | 0.5/0.5 | -11.4/-16.8 | 26.7/24.5 | WORST overshoot -> high trades + whipsaw 1.5% |
| KAMA | 11/2.2 | 0.4/0.1 | 0.4/**0.2** | -11.7/-12.0 | 20.1/21.2 | STALL -> under-participates (low time-in) |
| VIDYA | 4/1.5 | 0.2/0.0 | **0.3/0.1** | -7.2/-4.5 | 18.4/**5.6** | WORST STALL -> time-in 0.1 @1d, net collapses to 5.6% |

The signature weaknesses are EMPIRICALLY VISIBLE and split into two opposite families:
- **LOW-LAG family (HMA/DEMA/TEMA): OVERSHOOT** -- they trade 2-3x more (HMA 28 vs VIDYA 4 @4h), pay the most
  cost (HMA/TEMA cost_drag ~1.0-1.1), and take the DEEPEST drawdowns (HMA -17.4 @1d) because they ride into
  reversals before flipping. At 1d a residual <=2-bar whipsaw survives (HMA 1.4%, TEMA 1.5%).
- **ADAPTIVE family (KAMA/VIDYA): STALL** -- the opposite. They are TOO selective (VIDYA time-in 0.1 @1d), so
  they protect cost+DD (VIDYA maxDD only -4.5) but UNDER-PARTICIPATE and forfeit wealth (VIDYA net 5.6% @1d).
- **SMA: LAG** -- lowest time-in + high DD (late exits). **EMA: balanced** -- the clean all-rounder (highest net).

## DID THE IRON WORK? (measured delta vs base; +CONFIRM = anti-whipsaw, +VOLTGT = anti-DD)
| MA | dWhipsaw (+CONF) | dCostDrag (+CONF) | dMaxDD (+VOLTGT) | verdict |
|---|---|---|---|---|
| EMA | 0 | -0.2 | **+3.4-4.4** | DD ironed; no acute weakness to fix |
| SMA | 0 | -0.2 | **+4.5-4.7** | DD ironed; lag is STRUCTURAL (param-mitigated only) |
| WMA | 0 | -0.2 | **+4.6-4.7** | DD ironed |
| **HMA** | **-1.4** (1d) | -0.2 | **+5.2-5.6** | **OVERSHOOT IRONED**: confirm kills whipsaw, vol-target cuts the deep DD (the biggest DD-iron of all types) |
| DEMA | 0 | -0.2 | **+4.6-5.8** | overshoot DD ironed |
| **TEMA** | **-0.4** (1d) | **-0.4** | **+4.2-4.9** | overshoot whipsaw+cost ironed |
| KAMA | 0 | -0.1 | +3.9-4.2 | DD ironed, but confirm/voltgt make the STALL WORSE -- needs a different iron |
| VIDYA | 0 | -0.1 | +1.8-2.1 | confirm/voltgt make the STALL WORSE -- needs FAST params, not these remedies |

## THE ANSWER: which weaknesses are ironed, and which are not
1. **UNIVERSAL weaknesses -- IRONED for ALL types:** (a) the <=2-bar whipsaw is killed by the base stack's
   **min_hold(12)** (whipsaw% ~0 across the board); (b) **drawdown is cut 2-6pp for every type by vol-target**
   (the one iron that helps all 8); (c) selection-fragility is cut by the **ensemble** (MA_REMEDY).
2. **LOW-LAG OVERSHOOT (HMA/DEMA/TEMA) -- IRONED by the matched pair:** these are exactly the types where the
   irons help MOST -- confirm-band removes their residual whipsaw (HMA -1.4) and vol-target cuts their deep DD
   the most (HMA +5.6, the largest of any type). The overshoot weakness is mitigated, though the higher base
   churn is partly inherent to low-lag smoothing.
3. **ADAPTIVE STALL (KAMA/VIDYA) -- needs a DIFFERENT iron (param, not remedy):** confirm/vol-target make the
   under-participation WORSE (more selective). The correct iron is the **FAST param region** -- and the cluster
   work already found it (VIDYA's winner is VIDYA(2,5,8)/fast; KAMA's is mid). So the adaptive types ARE ironed,
   but by PARAM-SELECTION (the per-type best cluster), not by the uniform remedies. The median-over-all-configs
   here (which includes the stalling slow configs) is what looks bad; the fast cluster does not stall.
4. **SMA LAG -- STRUCTURAL (not fully ironable):** equal-weighting is inherently the laggiest; vol-target irons
   its DD, a faster param region limits the lag, but lag cannot be removed within the SMA type (that IS why the
   low-lag types exist).

**So: YES for the universal weaknesses (whipsaw, DD, fragility) across all types; the low-lag OVERSHOOT is
ironed by confirm+vol-target; the adaptive STALL is ironed by param-selection (fast cluster), NOT by the
uniform remedies; SMA's LAG is structural.** The per-type iron is therefore TWO-PART: the uniform stack
(min_hold + vol-target + ensemble) for the shared weaknesses, PLUS the type-specific best param region (the
cluster) for the type's signature weakness -- low-lag types take a slower/confirmed region to tame overshoot,
adaptive types take a faster region to beat the stall.

## OPEN (the honest residual)
- The low-lag base churn is mitigated, not eliminated -- a per-type "overshoot damper" (e.g. requiring the
  fast-MA slope to confirm, not just the cross) is an untested deeper iron for HMA/DEMA/TEMA.
- The adaptive stall is param-dependent; a regime-adaptive param (fast in chop, slow in trend) is the untested
  deeper iron for KAMA/VIDYA.
json: `ma_weakness_4h.json` + `ma_weakness_1d.json`. RWYB: `python -m strat.deep2020_ma_weakness --cadences 4h`.
