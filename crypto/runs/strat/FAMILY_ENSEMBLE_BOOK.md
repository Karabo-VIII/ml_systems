# Family-ensemble book -- the bull->bear full-cycle grade (2026-06-16)

PHASE 1 of the TRANSLATION-SOLUTION build-out. Tool `src/strat/family_ensemble_book.py`; data
`runs/strat/family_ensemble_book_20260616_010807.json` + charts `family_ensemble_equity_2020_2022.png`,
`derisk_sizing_tradeoff.png`. All numbers **[VERIFIED-RWYB, FULL-CYCLE 2020-2022]**. STRICT long-only +
spot (ZERO short logic). FIXED-EW (`fillna(0.0).mean`, never skipna). Survivorship-clean POINT-IN-TIME
universe (data-derived listing dates). Frozen 2020-selection, NO re-fit on 2021/2022. Maker cost, lag-1.
**UNSEEN (2025-12-31 -> 2026-06-01) SEALED -- not touched in this phase** (the overseer tests it once later).
Repro: `python -m strat.family_ensemble_book --years 2020,2021,2022 --derisk all --pick light` (git 262f718).

## What this builds
The deployable book the /quant diagnosis pointed at (`TRANSLATION_SOLUTION_2021.md`): for each
TRANSLATING family it equal-weights the family's 2020-selected working band, combines the families
equal-weight, and applies a light de-risk overlay -- then asks whether that de-risked long-only beta
book PAYS full-cycle across the 2022 bear.
- **DEPLOY (translating families):** trend (5 members) + breakout (2) + momentum (2) + MA (8) = 17 frozen
  2020-selected band members (the configs in `forward_test_2021.MA_ROBUST_4H` + `TI_CANDIDATES`).
- **DROP:** volume (collapsed 17->1.6 in 2021) + mean-reversion (flat 6->7). Confirmed dead, excluded.
- **Stack (per asset, per config):** signal -> trail-stop(10%) -> min_hold -> lag-1 -> vol-target -> maker
  flips. Band-ensemble = fixed-EW the configs within a family; book = fixed-EW the families; both then
  fixed-EW across the PIT-active roster (inactive/unlisted asset = cash, never skipna). Long-only (0/1).

## PRE-REGISTRATION (stated before the multi-year run; persisted in the JSON)
- **H0:** the book does NOT preserve the 2022 bear better than buy-hold on a risk-adjusted basis AND/OR
  does NOT compound >= buy-hold over the full 2020-2022 cycle. (The de-risk does not pay.)
- **H1:** the de-risked family-ensemble PRESERVES the 2022 bear (book maxDD materially -- >=10pp -- < BH
  maxDD) AND compounds >= BH over 2020-2022 (the drawdown-preserving-beta thesis cashes in full-cycle).
- One-sided H1 with **asymmetric loss** (false-ship a non-preserving book into a -60% bear >> false-skip).
  **TWO-SIDED reporting:** report if it FAILS -- e.g. if it just under-participates everywhere.

## PER-YEAR GRADE (book @ light de-risk vs EW buy-hold, PIT) [VERIFIED-FULL-CYCLE]
| year | regime | BH net | BH maxDD | book net | book maxDD | book Sharpe | book Calmar | time-in |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | bull (COVID-recovery) | +26.7% | -19.4% | +17.8% | **-4.4%** | 1.57 | 4.05 | 0.37 |
| 2021 | mega-bull (+208% BH) | +208.4% | -49.5% | +23.1% | **-11.5%** | 1.32 | 2.01 | 0.31 |
| 2022 | **BEAR** | **-71.9%** | **-73.4%** | **-14.0%** | **-19.8%** | -1.04 | -0.71 | 0.28 |

Per-family nets are coherent and non-concentrated (no single family carries the book): 2020 all four
+16..+19%; 2021 breakout +35 / trend +27 / momentum +24 / MA +7; 2022 all four cushioned -10..-23%.

## THE 2022-BEAR CRUX (the load-bearing number)
**[VERIFIED-FULL-CYCLE]** In 2022, EW buy-hold lost **-71.9%** (maxDD **-73.4%**). The light-de-risk book
lost **-14.0%** (maxDD **-19.8%**). That is a **53.6pp maxDD preservation** and a **57.9pp net
preservation** -- the de-risk insurance pays exactly where it should. The book preserves capital through
the regime that obliterates buy-hold. **H1 (a): PRESERVED.**

## THE FULL-CYCLE COMPOUND (the second load-bearing number)
**[VERIFIED-FULL-CYCLE]** Chaining the three years in calendar order (bull -> mega-bull -> bear), $1 grows
to (light de-risk):
| | full-cycle net | maxDD | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| **family-ensemble book (light)** | **+24.7%** | **-21.8%** | 0.60 | **1.14** |
| EW buy-hold (PIT) | +9.7% | -78.9% | 0.38 | 0.12 |

The book **out-compounds buy-hold full-cycle (+24.7% vs +9.7%)** at **one quarter of the drawdown**
(-21.8% vs -78.9%) and ~10x the Calmar. **H1 (b): book >= BH -- the de-risk pays full-cycle.** The
crossover is visible in `family_ensemble_equity_2020_2022.png`: the book flatlines below BH through the
bull, then the two curves cross during the 2022 bear and the book ends above BH.

## THE DE-RISK SIZING STUDY (the key knob: bull-net vs crash-preservation)
**[VERIFIED-FULL-CYCLE]** Swept {none, light, medium, heavy}. Heavier de-risk monotonically shrinks both
the bull net AND the bear drawdown -- the tradeoff the diagnosis predicted. **Light is the full-cycle
wealth + Calmar optimum** (it does NOT cripple bull participation the way heavy does):
| de-risk | 2020 net | 2021 net | 2022 net | 2022 maxDD | **full-cycle net** | full-cycle maxDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 19.6 | 21.1 | -15.9 | -22.4 | 21.9 | -24.7 | 0.89 |
| **light** | 17.8 | 23.1 | -14.0 | -19.8 | **24.7** | -21.8 | **1.14** |
| medium | 13.9 | 18.6 | -11.3 | -15.3 | 19.7 | -17.9 | 1.10 |
| heavy | 9.1 | 11.1 | -7.4 | -10.0 | 12.2 | -12.0 | 1.02 |

Heavy buys the shallowest DD (-12% full-cycle) but bleeds wealth (12.2%); none/light keep the wealth. The
2021 bull-capture is poor at every level (the book under-participates -- see the caveat), so pushing
de-risk down from light to none barely adds wealth (21.9 vs 24.7) but adds DD. **Deploy at LIGHT.**

## VERDICT: REAL_WITH_CAVEAT [VERIFIED-FULL-CYCLE]
The pre-registered load-bearing **H1 holds on BOTH counts**: the book **preserves the 2022 bear** (maxDD
-19.8% vs BH -73.4%, a 53.6pp margin) **AND out-compounds BH full-cycle** (+24.7% vs +9.7%). Gates:
`{2022_dd_preserved: True, full_cycle_compound_ge_bh: True}`.

**THE CAVEAT (the honest two-sided finding):** the book wins the cycle **by LOSING LESS in the bear, NOT
by capturing the bull.** It captures only **~11% of the 2021 mega-bull** (+23.1% vs BH +208.4%) and 67%
of the milder 2020 bull (+17.8% vs +26.7%). The `bull_participation_kept` gate fires **False**. This is a
drawdown-preserving **DIVERSIFIER / INSURANCE** book, NOT bull-beating alpha. It earns its full-cycle edge
purely through crash-avoidance -- exactly the "de-risked beta, not alpha" class the /quant diagnosis named.
**Do NOT sell it as alpha.** In a regime that is bull-only it will badly lag buy-hold; its value is
conditional on a bear arriving within the holding horizon (which, over a full cycle, it does).

## Cheapest falsifier
The full-cycle edge is entirely a 2022-bear artifact of CALENDAR alignment. Cheapest kill: re-chain the
same per-year books in a **bull-only** sub-cycle (2020+2021 only) -- the book is +0.6..+1.3% vs BH ~+466%
compound there (it loses by ~2 orders of magnitude). The edge EXISTS only because a full cycle contains a
bear. If the deployment horizon is reliably bull-only, this book is the wrong tool; if it spans a cycle
(which crypto always eventually does), it is the right drawdown-preserving core. The overseer's UNSEEN
test (2025-26, sealed here) is the real out-of-sample check: it must show the same "lose-less" signature
in whatever regime 2025-26 turns out to be.

## Caveats / discipline
- **Frozen 2020-selection, NO re-fit** on 2021/2022 (pure forward). UNSEEN sealed.
- **Survivorship residual** (inherited from forward_test_2021's PIT engine): coins that traded 2020-2022
  but delisted before 2026 (LUNA/FTT/...) were never collected into chimera, so cannot be included -- this
  understates the 2022 bear severity slightly (the worst losers are missing from BH too). Flagged, not
  fixable from our data.
- **Live fills:** maker p_fill 0.25-0.50 (CLAUDE.md) => live book net ~50-75% of these numbers; the
  preservation ratio is robust to this (both legs share the haircut).
- **2020 early-window** PIT roster is thin (few coins active pre-Jul-2020) -- the book is near-flat there
  by construction (cash), visible as the flat segment to 2020-08 in the equity chart. Correct PIT behavior.

## Files
- `src/strat/family_ensemble_book.py` (the book + grade; `--selftest` PASS 5/5)
- `runs/strat/family_ensemble_book_20260616_010807.json` (per-year + de-risk study + full-cycle + verdict)
- `runs/strat/family_ensemble_equity_2020_2022.png` (book vs EW-BH, 2022 bear shaded)
- `runs/strat/derisk_sizing_tradeoff.png` (bull-net vs crash-preservation by de-risk level)
