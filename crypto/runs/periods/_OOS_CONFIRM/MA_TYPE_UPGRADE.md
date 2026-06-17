# Upgrading the 2MA/3MA: the MA TYPE axis (same cross, smarter MA)

User /orc 2026-06-12 (halt+correct): stay on the 2MA/3MA we have been dealing with -- upgrade THEM, do not
jump to other indicators/families (breakout/MR parked). The core MA weakness the building block exposed is
the FIXED-EMA tradeoff (fast EMA whipsaws in chop, slow EMA lags in trend). The genuine on-family upgrade
is the MA ITSELF. Tool: `src/strat/ma_type_upgrade.py`. Same 2-cross structure on the slow family; swap
the MA type. FIXED level (pure MA-type effect, no overlay confound), 4h, u10 book, taker, UNSEEN sealed.

## 2MA-slow, FIXED level -- ROI%/maxDD% + Jan-2020 whipsaw% + OOS scorecard
| MA type | Jan20 whip | bear (Jun22) | VAL | OOS | OOS Sharpe | OOS-heldout p05 |
|---|---|---|---|---|---|---|
| **EMA (baseline)** | 3.3% | -9.9 | 26.2 | -0.0 | 0.17 | -39.1 |
| SMA | 14.1% | -12.0 | 28.8 | -8.0 | -0.13 | -45.3 |
| WMA | 22.1% | -13.0 | 33.2 | -9.3 | -0.18 | -46.4 |
| HMA (Hull) | 14.7% | -9.9 | 34.7 | -8.6 | -0.14 | -46.7 |
| DEMA | 5.2% | -20.0 | 37.3 | -9.4 | -0.15 | -48.2 |
| TEMA | 11.0% | -12.4 | 39.1 | -11.3 | -0.24 | -48.1 |
| KAMA (adaptive, efficiency) | 22.3% | -13.6 | 13.0 | -10.8 | -0.29 | -44.8 |
| **VIDYA (adaptive, momentum)** | **0.0%** | **-0.8** | 20.8 | **+4.7** | **0.34** | **-38.0** |

## Finding: VIDYA is a genuine UPGRADE; the low-lag and efficiency-adaptive types are NOT
- **VIDYA (Variable Index Dynamic Average -- a volatility/momentum-adaptive MA) is the only MA type that
  beats EMA on OOS compound (+4.7 vs -0.0), Sharpe (0.34 vs 0.17), AND p05 (-38 vs -39).** It does exactly
  what an adaptive MA should: FAST in trend, SLOW in chop. Concretely:
  - Jan-2020 whipsaw 0.0% (vs EMA 3.3%) -- it stops whipsawing in chop.
  - **Bear (Jun2022) -0.8% (vs EMA -9.9%)** -- the standout: the adaptive MA sits still when there is no
    clean trend, so the long-only bear barely bleeds. This is the long-only-bear problem NOTHING else in
    the whole arc touched (regime gates refuted; trail only softened) -- VIDYA's adaptiveness sidesteps it.
  - OOS +4.7% (vs EMA flat) with 2x the Sharpe -- positive on the hard held-out tape.
  - Cost: VAL 20.8 vs EMA 26.2 (the adaptive lag gives up some clean-bull upside -- the expected trade).
- **The LOW-LAG types (Hull/DEMA/TEMA) make it WORSE** -- they react faster -> trade more (higher whipsaw
  in VAL, deeper maxDD) -> worse OOS. Low-lag is the WRONG direction (more reactive = more cost+chop loss).
- **KAMA (efficiency-ratio adaptive) FAILED** (whipsaw 22%, OOS -10.8) -- so it is NOT "adaptive" generically
  that helps; it is specifically VIDYA's MOMENTUM/volatility adaptation. On crypto, KAMA's trend-efficiency
  measure mis-fires; VIDYA's CMO-based one works.

## Honest caveats
- VIDYA OOS p05 is still -38 (< 0) -- BETTER than EMA (-39) and the best MA type, but NOT absolute-robust.
  The MA-type axis lifts the level (Sharpe, bear, OOS sign) but does not by itself clear the robustness bar.
- FIXED level only (no overlays). NEXT (on-scope): VIDYA + the FULL stack (trail/minhold/maker) and on 3MA.
- One OOS window; the bear -0.8% is one bear month -- confirm VIDYA's bear-sidestep across more bear spans.

## Verdict
The user's correction was right: staying on the 2MA/3MA surfaced a REAL upgrade (VIDYA) that the breakout
detour would have skipped. **VIDYA upgrades the EMA 2MA/3MA cross** -- same structure, adaptive smoothing,
materially better held-out behaviour (positive OOS, 2x Sharpe, and a near-flat bear). It is the new MA-type
baseline to carry forward. json: `ma_type_upgrade.json`. RWYB: `python -m strat.ma_type_upgrade`.
