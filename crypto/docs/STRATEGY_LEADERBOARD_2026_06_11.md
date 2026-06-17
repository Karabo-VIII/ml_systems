# Strategy Leaderboard — honest re-grade of every strategy (2026-06-11)

> User mandate (/orc): *"get all the strats we've ever generated... revalidate and regrade them
> properly using the correct methodology... honest and proper candidates... 1d/3d numbers are a
> soft-benchmark (not to be used to eliminate), but I want honesty across the board."*
>
> Method: every candidate flows through **`src/strat/scorecard.py`** (the canonical evaluator) —
> splits **SEL** (pre-OOS, selection) / **OOS** (2025 bear-onset, validate) / **UNSEEN**
> (2026, test-once); book = compound/ann/maxDD + block-bootstrap p05; trades = per-trade mean±SE +
> **fraction-matched drop-top-5% jackknife** + n_eff + breadth; 1d/3d/7d rolling ROI as a
> **reported soft-benchmark, never an eliminator**. This fixes the session's confirmed failure modes
> (OOS-selection, fixed-count jackknife, sum/breadth bias). Books re-graded **RWYB** via
> `regrade_leaderboard.py`; full data in `runs/strat/REGRADE_LEADERBOARD_u50_*.json`.

## A. BOOK candidates — the ship hunt (RWYB, u50 1d taker)
| Book | SEL ann% (DD) | OOS ann% | UNSEEN comp% | full p05 | held-out p05 | 3d med/+% | verdict |
|---|---|---|---|---|---|---|---|
| **regime_beta** | **+73.3** (−55) | −14.6 | −3.5 | **+106** | −45.6 | 0.0 / 0.45 | ROBUST full-cycle; preserves bear; not UNSEEN-pos |
| BLEND_75r | +68.7 (−52) | −11.6 | −3.1 | +101 | −41.7 | 0.0 / 0.50 | same family |
| **BLEND_50r** | +63.9 (−49) | −8.5 | −2.8 | +88 | −38.0 | 0.0 / 0.50 | charter CORE; preserves bear |
| BLEND_25r | +59.0 (−49) | −5.5 | −2.4 | +73 | −34.9 | +0.03 / 0.51 | most defensive blend |
| TSMOM_breadth | +53.8 (−49) | −2.5 | −2.1 | +59 | −31.7 | 0.0 / 0.49 | least drawdown of trend books |
| buy_hold | +70.1 (−82) | −23.4 | −18.3 | **−69** | −80 | +0.37 / 0.54 | benchmark; full p05 NEGATIVE |
| low_vol_tilt | +51.3 (−79) | −23.4 | −20.1 | −74 | −76 | −0.18 / 0.50 | NULL (no low-vol anomaly in crypto LO) |
| RANDOM_null | +31.9 (−60) | −18.4 | −6.4 | −43 | −46 | −0.08 / 0.46 | the null floor |

**Read:** the regime-gated trend family (regime_beta / BLEND / TSMOM) has a **strongly positive
full-cycle block-bootstrap p05 (+59 to +106%)** and **decisively beats both buy&hold (p05 −69) and
the random null (p05 −43)** — it is a *robust full-cycle wealth + capital-preservation* book (UNSEEN
−2 to −3.5% vs buy&hold −18.3%). It is **NOT UNSEEN-positive** in the 2026 bear and its held-out p05
is negative — i.e. it preserves, it does not earn, in a sustained long-only bear. That is the honest
beta+regime-gate ceiling, re-confirmed cleanly under correct methodology.

## B. TRADE candidates — per-asset / regime (regime_dna r2, fair metrics)
| System | OOS mean±SE | UNSEEN mean±SE | UNSEEN fair-jk | UNSEEN breadth | verdict |
|---|---|---|---|---|---|
| SYS_A regime_sw [u50] | +1.9±1.4% | −1.2±0.8% | −3.4% | 11/34 | best-decay but UNSEEN-neg |
| SYS_B per_asset [u50] | +4.8±2.9% | −4.9±1.0% | −7.1% | 5/22 | per-asset DNA = noise |
| SYS_C pooled [u50] | +8.3±6.5% | −2.8±8.7% | −15.2% | 3/9 | concentration-fragile |

**Read:** per-asset config DNA is **not real** (systems statistically identical, per-cell survival
below a random-config null); the lever is regime-**gating** (UP-only long-trend), not switching. All
UNSEEN-negative — same ceiling as §A. (Full: `REGIME_DNA_FINDINGS_2026_06_11.md`.)

## C. LAB candidates (single-strategy, ingested honest verdicts)
| Lab | Honest held-out | verdict |
|---|---|---|
| Family2 mover-rotation | re-graded fair: OOS +8.2%/trade (firewall 100% > random), UNSEEN +3.9%/trade but **firewall only 73% > random** (random-3 of MA-pass = +2.5%), n=22/3 assets, p05 −75% | NOT SHIP — OOS selection real, UNSEEN within random (= D68); +172% headline was concentration + MA-filter beta |
| trend_book_lab | UNSEEN 0 trades (flat in bear); OOS −7.5%/yr | = regime-gate (folds into §A) |
| symmetric_trend L/S perp | UNSEEN +13.8%/yr on **4 short trades** | tiny-n; needs perp sign-off |
| setup_chaser_book | UNSEEN −33%/yr; battery FAIL, PBO 0.79 | DEAD (clean null) |
| alt_bar trend (Renko/Range/HA) | UNSEEN −20 to −53%/yr; 0/10 seeds | DEAD (chart-type axis null) |

## D. Dead-list (the honesty ledger) — D01–D73 + A1–A8
73 refuted theories + 8 measurement artifacts, each with its falsifying number + HARD/SCOPED scope
(full: `docs/MARKET_FRAMEWORK/01_DEAD_LIST.md`). **HARD kills are not re-graded** (re-running a
mechanism-level kill is pointless — re-confirmed by the inventory). The **A1–A8 artifacts are
re-grade preconditions**: any candidate citing a pre-fix number (MtM double-count, voladj-IC,
same-bar-close fill, flat-30bps null, daily-feature-on-4h leak) is invalid until re-graded on the
current harness — which is exactly what §A/§B do.

## E. Verdicts
- **SURVIVOR (the honest candidate):** the **regime-gated trend book** (regime_beta / BLEND_50r) —
  robust full-cycle wealth (p05 +88 to +106), preserves capital in bears, decisively beats buy&hold
  and the random null. Honest caveat: preserves-not-earns in a sustained bear; not UNSEEN-positive.
- **REFINE (now graded):** Family2 mover-rotation — firewall RUN (RWYB): OOS momentum selection
  beats random 100% (real edge) but UNSEEN beats random only 73% (within the random-3-of-trend-pass
  band) → NOT a ship, confirms D68 (selection real OOS, not UNSEEN-robust); the +172% was
  concentration. Remaining REFINE: D54 funding-extreme **regime-filtered** reversion (port + fair
  test); D61 exit sub-axes (regime-conditioned / take-profit, with the no-skill hold-length control).
- **DEAD (confirmed, not re-graded):** setup_chaser, alt-bar trend, low_vol_tilt, and all HARD
  D-list mechanisms (per-asset config DNA, naive MA, buy-the-extreme, sub-bar liq-signature,
  mover-riding on internal data, factor→config, 1h MR cost-wall, …).
- **BUILD (gaps):** a **bear-regime return engine** that is robust (gold sleeve FAILED robustness —
  one gold-bull episode, hurts when gold+crypto fall together; bear-short needs perp sign-off);
  and the **external-data discriminator** (D71/D72 spec: trigger-time OOS AUC ≥0.58 — needs Coinglass/
  on-chain, parked by user until returns shown).

## F. The honest bottom line
**Yes — we have one honest, proper candidate:** the regime-gated trend book, a robust full-cycle
~25–73%/yr wealth book (depending on blend) that preserves capital in bears (UNSEEN −2 to −3.5% vs
market −18%). It is the Fork-A product the charter describes. What it is **not** is a strategy that
earns positive returns in a sustained long-only bear, or that captures the daily movers — those
require either shorts/perps (sign-off) or external leading data (parked). The 1d/3d soft-benchmark
(median ≈ 0%, ~50% of windows positive) confirms per-window returns are a coin-flip; the wealth comes
from the regime-gated trend tail over the full cycle, not from per-window positivity. **No inflated
numbers; every figure is claim-tagged and reproducible** via the cited harnesses and JSONs.

## Family2 firewall addendum (2026-06-11 RWYB)
momentum_rotation_lab best config (N10_K3_R10_MA200_ATR3) re-graded with the fair scorecard +
a random-selection firewall (random-3 of the MA200-passing set, same scaffold, 200 seeds):
- OOS: momentum +8.21%/trade beats random-3 (mean -0.62%, p95 +1.52) in 100% of draws -> selection edge REAL OOS.
- UNSEEN: momentum +3.93%/trade vs random-3 +2.51% (p95 +6.22); beats only 73% of draws (< 95% bar)
  -> selection edge NOT established on the held-out bear; the MA200/trend filter does the work.
- n=22 UNSEEN trades, 3 assets, bootstrap p05 -75% -> underpowered + concentration. The prior +172%
  compound headline = a few surviving alts x the trend filter, not a robust momentum-selection edge.
VERDICT: confirms D68 (cross-sec selection real OOS, not UNSEEN-robust). NOT a ship.
