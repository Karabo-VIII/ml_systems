# MA Drawdown-Insurance Sleeve — 1st VERIFIED SUCCESS (2026-06-20)

**Status: the first verified, deployable result of the move-capture campaign.** Not alpha — *capital-preservation
within long-only*, the charter's honest fallback, now empirically EARNED (genuine same-exposure timing skill) rather
than asserted. Aggressively red-teamed on 3 axes; GO. Incubation (DEV-window), not live-validated.

## The defensible claim (commit this verbatim)
> At coarse TFs (**1d / 4h**), an equal-weight u100 sleeve that goes **long while the MA band holds and sits in cash
> otherwise** roughly **HALVES portfolio max-drawdown (~−35% vs buy-hold ~−82%)** by de-risking into every major crypto
> selloff (COVID-2020, May-2021, 2022 bear, 2023 chop — robust across all 8 MA types), at the cost of giving up part of
> the upside. This is **genuine same-exposure timing skill** (+6 to +15.5pp of drawdown avoided beyond a φ-matched dumb
> book at 1d/4h), NOT a low-exposure artifact. It is **capital-preservation / drawdown-insurance within long-only spot —
> NOT alpha, NOT a money-maker.**

## What it is + how built
8 engines, one per MA TYPE (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA). Built **per (type × timeframe)**: each TF trades the
**fixed-EW ENSEMBLE** of that TF's canonical working band (configs net-positive across 2020 TRAIN∧VAL in
`config_leaderboard.json`; OOS field never read), long when the majority of the band's `held_cross` is aligned, cash
otherwise. MA lengths differ by TF (slow-floor rises at finer cadence). Universe = 56 of 100 point-in-time u100 names
(≥20 floor met). Accounting (`src/strat/ma_wealth_driver.py`): MtM-no-double-count (reconciles 2e-16), lag-1 causal,
maker 0.0006 (survives taker 0.0024), TRUE buy@listing HODL baseline, DEV wall < 2024-05-15 (OOS/UNSEEN sealed).

## Verification (aggressive red-team — GO on all 3)
1. **Timing skill, not low-exposure (the make-or-break).** Engine avg exposure φ≈0.40 (in cash ~60%). Against
   exposure-matched dumb controls — CFAL (constant-fraction always-long) and **REHTM** (random-entry hold-to-maturity,
   24 seeds, churn-immune; NOT the broken A9 shuffle, hold-length matched to the engine's run length) — the engine
   avoids **+6 to +15.5pp extra drawdown** at the same exposure (EMA 1d +6.0 / SMA 1d +8.1 / DEMA 1d +5.7 / EMA 4h
   +14.7 z=3.53 / SMA 4h +13.5), and preserves the 2022 bear +12 to +24pp beyond the dumb book. It is in cash at the
   *right* times. (`_skeptic1_exposure_*.json`, `src/strat/ma_exposure_control.py`)
2. **Leak-free + survivorship-honest + cost-robust.** lag-0 cheat explodes compound to 2.6M% and shrinks DD +26.6pp
   (test has power; shipped numbers are causal). DEV wall holds; selection reads only TRAIN/VAL net. Survives taker
   cost. Survivorship: injecting dead names makes HODL *worse* (−82.3→−83.1), WIDENING the gap (engine holds cash, not
   the zeros) — the *relative* halving is fair. (`_skeptic2_results.json`)
3. **Robust, not 2022-curve-fit.** Drawdown-cut in ALL FOUR stress events — largest in COVID/May-2021, not 2022:

| Event | index maxDD | engine dd-cut (range over engines) |
|---|---|---|
| COVID 2020 | −61.7% | **+42 to +53pp** |
| May 2021 | −65.0% | **+34 to +51pp** |
| 2022 bear | −77.3% | **+34 to +56pp** |
| 2023 chop | −43.8% | **+15 to +16pp** |

   In the 2023 recovery the engine re-risks (gross 0.32–0.46, captures 55–100% of the rally) — not a permanent-cash
   dodger. Consistent across all 8 engines × {1d,4h} (16/16 cells, beta-gate passes). (`_skeptic3_per_event_dd.json`)

## Per-timeframe (median across 8 engines) — the cadence gradient
| TF | up-mkt α | maxDD eng / HODL | bear-presv | capt-A | beta-gate |
|---|---|---|---|---|---|
| **1d** | −5111 | **−35% / −82%** | +56.7 | 4.5 | 16/16 |
| **4h** | −5548 | **−34% / −83%** | +59.1 | 3.1 | 16/16 |
| 2h | −6015 | −38% / −83% | +49.9 | 2.6 | 15/16 |
| 1h | −3648 | −56% / −83% | +37.6 | 2.0 | 12/14 |
| 30m | −425 | −77% / −83% | +26.0 | 1.8 | 11/14 |
| **15m** | +18 | **−98% / −83% (BROKEN)** | +15.2 | 0.7 | 5/11 |

Best preservers at 1d/4h: **HMA, TEMA** (maxDD ~−28%). **Scope is binding: 1d/4h only — fine TFs are broken
(churn-dominated), not merely "degraded."**

## Mandatory caveats (ship attached)
- **ZERO up-market timing alpha** (−2851 to −7307pp vs index at 1d/4h) — underperforms HODL in a pure bull.
- **Do NOT cite the full-window compound** (e.g. DEMA +4581% vs HODL +465%) as a headline — it is the compounded
  bear-dodge, not skill. That is THE overclaim to kill.
- **DEV-window only** (band selected 2020 TRAIN+VAL; OOS/UNSEEN sealed) — incubation, not live-validated.
- Panel = 56 of 100 point-in-time u100 names; absolute HODL −82% is survivor-flattered (true ≈−83%, strengthens the
  relative case).

## Provenance
Drivers: `src/strat/ma_wealth_driver.py` (wealth, verified), `src/strat/ma_exposure_control.py` (skill control),
`src/strat/ma_movecatch_engine.py` (band selection + holds; `ma_wealth_driver` is authoritative for wealth — the old
`run_engine` wealth path is superseded). Capture: `src/strat/capture_lab.py` (STEP 1, committed 19e291a2). 2h TF fix:
34be2be5. Results: `runs/periods/TRAIN/2020/DEEP_DIVE/MA_MOVECATCH/engine_*_clean.json` + `_skeptic{1,2,3}_*.json`.
Open (un-verified, on disk): `val_forward_*.json` (held-out forward windows), `decoupled_7slice_14lb_*.json`
(7d-slice/14d-lookback) — pending adversarial verification before use.
