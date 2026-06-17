# ER-gated fixed-MA @ 4h vs REGIME-MATCHED null — BLOCK-BOOTSTRAP + JACKKNIFE FALSIFIER

**RED-team task (the statistical-rigour layer):** build a regime-matched firewall (random entries restricted
to the SAME high-ER, ER>0.4 windows the gated strategy trades in) and **prove signal > null with
block-bootstrap p05>0 + jackknife.** If the ER-gated MA does not beat this regime-matched null, the *gate*
(not the cross) is doing the work and the "signal" is regime-beta.

**Verdict: REFUTED.** The ER-gated fixed-MA @ 4h is **not** a signal above the regime-matched null. On
held-out (OOS+UNSEEN) it clears the contract on **0 of 77** u100 assets. The block-bootstrap and the
jackknife each independently kill it, and a regime-matched random-entry null **robustly out-earns** it
(pooled diff p05 < 0, only 2.7% of bootstrap draws favour the strategy). The return that exists inside the
ER>0.4 regime is **beta** — random entries capture it *better* than the MA cross.

## What this adds over the existing percentile firewall
`ER_GATE_4H_FALSIFIER_REPORT.md` used a Monte-Carlo **percentile** null (`null_p95`). This layer adds the two
tests the brief names by hand, using the audited battery primitives
(`strat.battery.block_bootstrap_p05_p95` [F6/F7-hardened] + `jackknife`), with confidence intervals instead
of point estimates. Script: `er_gate_4h_bootstrap.py` (reuses the EXACT apparatus: `er_gate_4h.build_cols`,
8/21 EMA, Kaufman ER(20)>0.4 gate, ATR-trail×3.0 + 42-bar cap, taker 0.0024, next-open fill).

## The contract (a signal is GENUINE only if ALL three hold on held-out)
1. **Robustly positive** — block-bootstrap p05 of the real strategy's held-out trade returns **> 0**.
2. **Not one-trade luck** — jackknife **jk2 > 0 AND jk3 > 0** (compound survives dropping the 2/3 biggest trades).
3. **Signal > null** — real held-out compound **> regime-matched null p95** AND block-bootstrap p05 of the
   paired **(real − null) difference > 0**.

Null = regime-matched: random entries drawn ONLY from held-out bars where ER>0.4 (the SAME regime the gate
selects), durations sampled from the strategy's OWN held-out hold distribution, same taker cost.

## Results (RWYB-reproduced 2026-06-05, real chimera 4h, u100)

**Per-asset (77 evaluated):**

| Contract condition | assets passing |
|---|:--:|
| cond1 — block-bootstrap p05 of real held-out **> 0** | **1 / 77** |
| cond2 — jackknife jk3 **> 0** (drop top-3 trades, stays positive) | **0 / 77** |
| cond3 — beats regime-null p95 **AND** (real−null) diff p05 > 0 | **0 / 77** |
| **GENUINE SIGNAL (all three)** | **0 / 77** |

- real held-out block-bootstrap **p05: median −52.89%**, frac>0 = **0.014**
- jackknife **jk3 (drop top-3): median −39.97%, frac>0 = 0.000** — not ONE asset's held-out compound
  survives removing its three biggest trades.
- real held-out compound: median **−14.04%**, frac>0 = 0.325.

**The single cond1-passer is the proof both tests are needed:** `POLUSDT` had block-bootstrap p05 = **+9.85**
(robustly positive *looking*) and real held-out compound +75.29% — but **jk3 = −8.17** (its three biggest
trades carry it; drop them and it flips negative) AND the regime-matched null p95 = **+108.01% > 75.29%**
(random ER>0.4 entries on POL clear an even higher bar). The jackknife and the null each catch what p05 alone
missed. GENUINE = False.

**Pooled (universe-level, 2,202 real vs 411,800 regime-matched-null held-out trades):**
- real per-trade expectancy = **+0.0338%**  [bootstrap p05 −0.337, p95 +0.436] — ~zero, CI straddles 0.
- null per-trade expectancy = **+0.495%**   [bootstrap p05 +0.474, p95 +0.516] — **robustly positive** (this
  is the regime/beta: simply being long inside ER>0.4 windows earns ~0.5%/trade).
- **(real − null) expectancy bootstrap p05 = −0.832%**, frac_pos = **0.027** — in **97.3%** of resamples the
  random regime-matched null OUT-earns the MA-timed strategy. The difference is robustly **negative**.

## Interpretation — the gate, not the cross
Inside the ER>0.4 regime there IS a positive return (~0.5%/trade, robustly bootstrapped). But it is **regime
beta**: random entries drawn from those same bars capture it. The 8/21-EMA cross **subtracts** that beta
(real ≈ 0.03%/trade vs null ≈ 0.50%/trade) — it enters late / into the back of the move. So the gate
(ER>0.4 regime selection) does 100% of the work; the MA timing is value-destructive. This is the falsifier's
hypothesis confirmed with confidence intervals.

Two-sidedness (this is a real refutation, not a dead test): `positive_control_4h.py` runs the EXACT same
firewall on a synthetic 4h price with a GENUINE within-ER>0.4 long-only timing edge and the firewall DETECTS
it (OOS real +431% vs null_p95 +27; UNSEEN +123% vs +8; beats_held=True). The apparatus has power at 4h.

## RWYB reproduction
```
python src/strat/setup_harness.py --selftest                                   # apparatus PASS (accepts genuine, rejects random, catches leak)
python src/strat/battery.py                                                    # block-bootstrap p05 + jackknife gate logic OK
python experiments/adaptive_ma/expert/positive_control_4h.py                   # firewall HAS POWER @4h (PASS)
python experiments/adaptive_ma/expert/er_gate_4h_bootstrap.py --probe BTCUSDT  # single-asset detail (p05<0, jk<0, loses to null)
python experiments/adaptive_ma/expert/er_gate_4h_bootstrap.py                  # full u100: 0/77 GENUINE; pooled diff p05 -0.83%, frac_pos 0.027
```
Artifacts: `er_gate_4h_bootstrap.py`, `er_gate_4h_bootstrap_u100.json`, `er_gate_4h_bootstrap_quick.json`.

SAFETY: no commit / push / deploy / capital. Analysis + JSON only, under `experiments/`.
```
```
