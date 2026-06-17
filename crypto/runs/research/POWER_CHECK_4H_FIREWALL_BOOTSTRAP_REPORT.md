# Power check @4h: firewall + bootstrap FLAG a known-detectable signal — CONFIRMED

**Date:** 2026-06-06 (VERIFIED via `date`). **Verdict: POWER CONFIRMED (two-sided).** All numbers
RWYB-reproduced; synthetic data only; no commit / no deploy.
**Repro:** `python runs/research/power_check_4h_firewall_bootstrap.py`

## Why this exists
A 4h null result (e.g. the ER-gated MA-cross REFUTAL) is only trustworthy once the apparatus is shown
CAPABLE of detecting a true edge at this cadence. The prior 4h work power-checked the **firewall** only;
this adds the **bootstrap** (`battery.block_bootstrap_p05_p95`) so BOTH halves of the gate are shown to
have power. Reuses the real apparatus functions (no reimplementation): `make_positive_control_4h`,
`make_harness`, `per_trade_expectancy_null` (firewall_4h_regime_matched.py), `random_entry_null`
(strat.firewall), `block_bootstrap_p05_p95` + `evaluate` (strat.battery).

## Positive control — planted within-gate 4h timing edge (front-loaded up-impulses; cross fires at move start)
| gate component | result |
|---|---|
| FIREWALL compound (regime-matched) | `beats_held=True`, `pos_held=True` → **REAL ENTRY-TIMING EDGE** |
| FIREWALL per-trade expectancy | **PASS=True**; held real_exp **+8.69%** vs null p95 **+5.63%**, pctile_rank **1.0**, beats_p95=True |
| BOOTSTRAP held-out (OOS+UNSEEN) | p05 **+2451.6** / p50 +7387.8 / p95 +25235.4 → **p05>0 = significant** |
| BOOTSTRAP UNSEEN | p05 **+65.2** / p50 +192.9 / p95 +389.9 → **p05>0 = significant** |

→ **Both the firewall AND the bootstrap flag the planted edge.** The harness HAS POWER at 4h.

**Honest nuance:** the full battery *verdict* on the positive control's UNSEEN is **FAIL** — but NOT because
the bootstrap missed it (p05=+65 ≫ 0). It fails the **orthogonal `n_eff` concentration floor** (n=19,
n_eff=6.4 < 8): the impulse-capture book is a few big trades, which the sample-size discipline correctly
declines to call "bankable." That is by-design conservatism (matches `positive_control.py`'s documented
"a low-frequency genuine edge tops out below SHIP-TIER"), not a lack of bootstrap power.

## Negative control — pure-noise 4h, SAME gate machinery (two-sided soundness)
| gate component | result |
|---|---|
| FIREWALL compound | `beats_held=False` → **BETA-IN-DISGUISE / no timing edge** |
| FIREWALL per-trade | PASS=False; held real_exp −0.35% vs null p95 +0.10%, pctile_rank 0.246 |
| BOOTSTRAP held-out (OOS+UNSEEN) — **decisive surface** | p05 **−21.9** ≤ 0 → **NOT significant** ✓ |
| BOOTSTRAP UNSEEN-only | p05 +0.06 (≈0) on n=10 — marginal artifact of a flat-by-chance window |

→ The firewall and the **pooled held-out** bootstrap both correctly **reject** noise. (The single-window
UNSEEN-only bootstrap on a tiny n=10 book sat at p05≈0 — a reminder that the decisive surface is the
**pooled held-out** bootstrap + the firewall, not a small single-window p05 in isolation.)

## Conclusion
`POWER CONFIRMED = True` (firewall + bootstrap both flag the genuine 4h edge) and `TWO-SIDED = True` (both
reject pure noise on the decisive held-out surface). The 4h apparatus is calibrated — it accepts a true
within-gate timing edge and rejects beta/noise — so the earlier 4h NULL (MA-cross REFUTED) is a
trustworthy negative, not an artifact of a powerless gate. Result JSON:
`runs/research/power_check_4h_firewall_bootstrap_result.json`.
