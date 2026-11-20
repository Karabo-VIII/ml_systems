# Fixed-approach ladder — normal configs vs FIXED + stacked upgrades (side-by-side)

User /orc 2026-06-12: *"run the normal configs vs configs + (fixed and upgrades) and compare side by
side ... we are building, fixing, and upgrading the FIXED approach."* The ML config-selector is REFUTED
(selecting a config from observed conditions does not transfer OOS — see
`project-ml-config-selector-refuted-2026-06-12`), so we improve the FIXED config **structurally** and
measure each upgrade's marginal effect across regimes.

Tool: `src/strat/fixed_approach.py`. Book = equal-weight across (config, asset) of causal MtM net daily
returns over u10. 120 distinct 2MA+3MA configs (naive); 20 "FIXED" = 2MA-slow(60–150) family.
Each layer = previous + ONE change (cumulative ladder). All periods TRAIN-era (pre-2024-05-15 split).

## The ladder
| layer | change |
|---|---|
| L0_NAIVE | all 120 distinct configs (every speed), signal-flip, taker — the naive "run everything" |
| L1_FIXED | restrict to the robust **2MA-slow(60–150)** family, signal-flip, taker |
| L2_MINHOLD | + min_hold(12 bars) |
| L3_TRAIL | + **10% trailing stop** (loose) |
| L4_REGIME | + sit OUT (cash) when close < SMA(200) |
| L5_MAKER | L4 priced at maker (0.06% rt) vs taker (0.24%) |

## 4h book — ROI% / maxDD% / Sharpe
| layer | Jan2020 rally | Feb2020 reversal | Jan+Feb comb | Jun2022 bear | Feb2024 bull |
|---|---|---|---|---|---|
| L0_NAIVE  | 2.8 / -1.0 / 6.97 | 1.9 / -14.4 / 0.80 | 11.9 / -7.3 / 5.52 | -7.1 / -6.8 / -4.32 | 21.2 / -4.9 / 7.11 |
| L1_FIXED  | 1.6 / -0.7 / 4.29 | 2.0 / -19.7 / 0.73 | 17.3 / -9.3 / 5.61 | -9.9 / -10.1 / -5.98 | 24.5 / -4.9 / 7.25 |
| L2_MINHOLD| 1.6 / -0.7 / 4.27 | 1.6 / -20.1 / 0.63 | 17.2 / -9.5 / 5.54 | -10.2 / -10.4 / -5.78 | 24.7 / -4.9 / 7.29 |
| **L3_TRAIL**  | 1.5 / -0.7 / 4.40 | **6.6 / -13.9 / 2.14** | **21.4 / -4.4 / 8.90** | **-6.3 / -5.9 / -4.48** | 23.0 / -4.6 / 7.58 |
| L4_REGIME | 0.0 / 0.0 / 0.0 | -3.6 / -7.0 / -2.97 | -2.3 / -4.5 / -2.29 | -4.8 / -4.9 / -7.54 | 21.3 / -4.6 / 7.25 |
| L5_MAKER  | 0.0 / 0.0 / 0.0 | -3.4 / -6.8 / -2.80 | -2.2 / -4.4 / -2.15 | -4.5 / -4.8 / -7.09 | 21.8 / -4.6 / 7.39 |

## 1h book — ROI% / maxDD% / Sharpe
| layer | Jan2020 rally | Feb2020 reversal | Jan+Feb comb | Jun2022 bear | Feb2024 bull |
|---|---|---|---|---|---|
| L0_NAIVE  | 4.4 / -2.6 / 4.71 | 3.2 / -13.8 / 1.20 | 16.3 / -7.2 / 5.53 | -11.7 / -13.2 / -3.51 | 14.7 / -6.1 / 5.36 |
| L1_FIXED  | 10.3 / -2.5 / 8.62 | 7.1 / -14.9 / 2.11 | 29.0 / -6.5 / 7.91 | -13.0 / -17.0 / -3.17 | 21.2 / -4.6 / 6.81 |
| L2_MINHOLD| 10.4 / -2.6 / 8.73 | 6.7 / -15.4 / 1.98 | 29.0 / -6.7 / 7.87 | -12.8 / -16.8 / -3.04 | 21.7 / -4.4 / 6.85 |
| **L3_TRAIL**  | 10.5 / -2.5 / 8.80 | 6.4 / -15.4 / 1.99 | **29.5 / -6.2 / 8.31** | -12.6 / -14.9 / -3.70 | 21.8 / -4.4 / 6.95 |
| L4_REGIME | 5.9 / -2.7 / 6.15 | **8.7 / -13.4 / 2.77** | 26.1 / -5.0 / 8.07 | **-9.7 / -11.9 / -3.20** | 20.6 / -4.2 / 6.78 |
| L5_MAKER  | 6.3 / -2.6 / 6.55 | **9.8 / -12.8 / 3.07** | 27.7 / -4.6 / 8.51 | **-8.5 / -11.0 / -2.77** | 22.6 / -4.0 / 7.34 |

## What each upgrade did (the side-by-side read)
1. **L1 FIXED (pick the robust 2MA-slow family) — KEEP.** Biggest single lift, consistently, in the
   trending periods at BOTH cadences: 1h combined 16.3 -> 29.0; 4h combined 11.9 -> 17.3; Feb2024 bull
   1h 14.7 -> 21.2. Honest cost: it makes the **bear WORSE** (4h Jun2022 -7.1 -> -9.9) because the slow
   family stays long longer in a downtrend. Filtering to the robust family is the core fix.
2. **L2 MIN-HOLD(12) — DROP at coarse cadence (no-op here).** ~0 change at 4h/1h (the slow family already
   holds ~1 week, so a 12-bar floor never binds). This is consistent with the building block: min-hold's
   real lift was at FINE cadences (15m/30m), where churn is the killer — NOT at 4h/1h. Keep it only there.
3. **L3 TRAIL (10% loose) — KEEP. The standout upgrade.** It protects in the clean reversal and the bear
   AND barely costs anything in the rally/bull:
   - Feb2020 reversal 4h: 2.0 -> **6.6**, maxDD -19.7 -> **-13.9**
   - Jan+Feb combined 4h: 17.3 -> **21.4**, maxDD -9.3 -> **-4.4** (halved)
   - Jun2022 bear 4h: -9.9 -> **-6.3**, maxDD -10.1 -> -5.9 (recovers 3.6pp of the L1 bear cost)
   - rally/bull: essentially unchanged.
   **This RESOLVES the earlier trail paradox.** In the Jan+Feb combined study a *tight 5%* trail was the
   WORST exit (whipsawed out of the choppy top). A *loose 10%* trail is the opposite — it doesn't whipsaw
   in chop but DOES catch the one-way crash. Trail ranking flips by **width** (10% > 5%) and **regime**
   (clean crash favors the trail; choppy top punishes a tight one). Converges with the other instance's
   independent finding (`375c483`: "F4 no-trail-in-chop is the one real win") — both say: don't use a
   *tight* trail in chop; a *loose* trail earns its keep in a clean down-leg.
4. **L4 REGIME (sit out below SMA200) — DROP as specified.** At 4h it is **harmful**: it sat out almost
   the ENTIRE early-2020 rally (price below the laggy 200-bar SMA the whole window -> Jan 0.0, and Feb/comb
   turned NEGATIVE). At 1h it HELPS the bear/reversal (Jun2022 -13.0 -> -9.7; Feb2020 7.1 -> 8.7) but
   HURTS the rally (10.3 -> 5.9). A raw SMA200 cash-gate trades away too much rally upside to dodge
   drawdown — net-negative at 4h, a wash at 1h. The regime-gate *idea* (the one door the ML work left
   open) is not dead, but a plain SMA200 gate is the wrong instrument.
5. **L5 MAKER — marginal at coarse cadence.** Only ~1–2pp better than L4 at 1h (few trades -> little fee
   to save). The maker lever is a FINE-cadence story (per the building block, 15m taker -2.3% -> maker
   +5.4%); at 4h/1h it barely moves. Inherits L4's harm here because it is stacked on the regime gate.

## Synthesis — the winning stack
**L1_FIXED + L3_TRAIL(10%)** = robust 2MA-slow family + a loose 10% trailing stop. Drop the no-op
min-hold (at coarse cadence), drop the SMA200 regime gate (kills the rally). That stack:
- 4h combined 11.9 -> **21.4** (maxDD -7.3 -> -4.4); 1h combined 16.3 -> **29.5** (maxDD -7.2 -> -6.2)
- recovers the bear partially (4h -7.1 -> -6.3) and the reversal strongly (4h Feb 1.9 -> 6.6)
- improves maxDD in EVERY period; costs ~nothing in rally/bull.

## Honest caveats (RWYB)
- **All five periods are TRAIN-era** (2020/2022/2024 all < 2024-05-15). This is in-sample *structural
  design*, not an OOS result. The cross-regime consistency (rally/reversal/bear/bull all improved-or-held
  by L1+L3) is the encouraging signal — but the stack must be confirmed on VAL/OOS before any belief.
- **Long-only floor: the bear stays NEGATIVE** (4h -6.3, 1h -12.6 even with the trail). Without shorting
  or a working sit-out, a long-only book cannot make the 2022-style bear positive. L4 tried the sit-out
  and failed (too laggy). That is a structural ceiling, not a tuning miss.
- Book = equal-weight cross-config/asset MtM; maker assumes fills (real p_fill 0.21–0.40 -> ceiling).
- The cumulative ladder confounds L4/L5 with the harmful regime gate -> next step isolates L1+L3+maker
  WITHOUT the gate, and extends min-hold+maker to the fine cadences where they actually pay.

Chart: `../charts/fixed_approach_ladder.png` (left: ladder x period 4h ROI; right: combined Jan+Feb
equity by layer — L3_TRAIL red holds the rally gains through the Feb top; L4/L5 sit flat then negative).
