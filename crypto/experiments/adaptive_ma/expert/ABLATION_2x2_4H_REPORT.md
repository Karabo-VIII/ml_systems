# 2x2 ablation @ 4h — ENTRY-gate (ER) vs EXIT-policy (ATR-trail) — where is the (non-)edge?

**Auditor RED-team task:** isolate WHERE the ER-gate-fixed-MA-4h edge comes from. Ablate the ER entry-gate
against the ATR-trail+time-stop exit in a 2x2 `{ER-gate on/off} x {ATR-trail vs opposite-cross exit}` on the
same 4h data, all else locked, minimal added DOF (just on/off switches). **Decisive hypothesis:** *if
ER-gate-OFF + ATR-trail already beats the null, the conditioning premise is wrong and credit belongs to the
exit policy.*

**Verdict: REFUTED.** ER-gate-OFF + ATR-trail (cell C) does **NOT** beat the null — it beats it on **0/77**
assets and is the **WORST** of the four cells (median held-out OOS+UNSEEN **−54.98%**). Credit does **not**
belong to the exit policy. **No cell beats the null on any held-out window (0/77 in all four).** The only
component with a measurable effect is the **ER gate as a REGIME SELECTOR (beta), not timing** — and even
that loses to random entries drawn from inside the same regime. Both the entry-timing premise and the
exit-policy premise are refuted.

## Design (all locked to the expert rig; only two binary switches move)
- Data: real chimera **4h**, u100 (77 evaluable, ≥1000 bars). Cost taker **0.0024** round-trip.
- Entry STATE: `fast>slow` **[AND er>0.4 when gate ON]** — the *only* entry switch is the `& er>0.4` term.
  MA = **8/21 EMA**, Kaufman **ER(20)**, gate **0.40**. Fill = next-bar open (Pattern T banned).
- Exit switch: **ATR-trail ×3.0 (ATR14) + 42-bar cap** vs **opposite-cross (`fast<slow`) + 42-bar cap**.
  Both keep the identical 42-bar (7d) time backstop; only the price-exit mechanism changes. ATR read as
  `atr[j-1]` (leak-safe), exit-signal filled next-open → all past-only.
- Null = the project firewall (`strat.firewall.random_entry_null`): cost-matched RANDOM-ENTRY held for the
  cell's OWN hold-duration distribution. ER-on cells (A,B) use the **regime-matched** null (random entries
  only from `er>0.4` bars → isolates within-regime entry TIMING); ER-off cells (C,D) use the **plain** null.

|              | EXIT = ATR-trail ×3 + 42cap | EXIT = opposite-cross + 42cap |
|--------------|:---------------------------:|:-----------------------------:|
| ER-gate ON   | **A** (= the redirect strat)| **B**                         |
| ER-gate OFF  | **C**                       | **D**                         |

## Results (RWYB-reproduced 2026-06-05, real chimera 4h, 77 assets)

| cell | n | OOS real med | UNSEEN real med | OOS null p50 med | UNS null p50 med | beat-null & pos | pos-held |
|------|--:|--:|--:|--:|--:|:--:|:--:|
| A ER-on  + ATR-trail | 77 | −12.99% | −0.70%  | +8.36%  | −2.79%  | **0/77** | 13/77 |
| B ER-on  + opp-cross | 77 | −21.19% | −6.64%  | +2.35%  | −6.94%  | **0/77** | 7/77  |
| C ER-off + ATR-trail | 77 | −34.61% | −20.37% | −34.96% | −21.97% | **0/77** | 2/77  |
| D ER-off + opp-cross | 77 | −32.35% | −17.84% | −28.05% | −18.20% | **0/77** | 4/77  |

Median held-out (OOS+UNSEEN) compound: **A=−13.69  B=−27.83  C=−54.98  D=−50.19**.

**Provenance check:** cell A reproduces the prior `ER_GATE_4H_FALSIFIER_REPORT.md` numbers **exactly**
(OOS median −12.99%, 0/77 beat the regime-matched null) → the 2x2 apparatus is consistent with the prior run.

### Attribution (real-vs-real deltas — independent of the null definition)
- **ENTRY-GATE effect (ER on − off):** ATR-trail **A−C = +41.29pp**, opp-cross **B−D = +22.36pp**.
  Removing the ER gate makes held-out **dramatically worse**. So the gate *does* something — but…
- **…it is REGIME SELECTION (beta), not timing.** The regime-matched null *beats the real entry* inside the
  gate: cell A OOS real **−12.99%** vs regime-null p50 **+8.36%**. Random entries drawn from `er>0.4` bars
  out-earn the MA-timed entries. The MA cross *inside* the regime **subtracts** value; the gate's only worth
  is keeping you long less often, in trending windows — and that selection still doesn't clear the null.
- **EXIT-POLICY effect (ATR-trail − opp-cross):** ER-on **A−B = +14.14pp**, ER-off **C−D = −4.79pp**. Small,
  sign-flips, and **never beats the null**. Decisively: **cell C's real ≈ its plain null** (OOS −34.61 vs
  −34.96; UNSEEN −20.37 vs −21.97) → `fast>slow` entry + ATR-trail exit is statistically indistinguishable
  from random-entry-and-hold. The ATR-trail exit adds **nothing detectable**.

### Single-asset detail (BTC, n_books=300)
A: OOS +1.31 / UNSEEN −3.29 · B: +1.09 / −7.71 · C: −5.67 / −30.15 · D: −9.71 / −13.95 — every cell
below its null; ER-off cells far worse. (A OOS +1.31% is below its regime-null p50 +4.43% — matches prior.)

## Honest caveat (does not rescue the strategy)
Everything is absolute-negative partly because the held-out window (OOS 2025-H2 + UNSEEN 2026-H1) was a
period where buy-random-and-hold on the median altcoin *also* lost ~20–35% (see the plain-null p50 column).
But the firewall is **relative** — "do you beat random entry under the SAME conditions?" — and the answer is
**0/77 in every cell**. A bad long-only regime explains the absolute level; it does not create a missing edge.

## Conclusion
The 2x2 refutes the decisive hypothesis: **ER-gate-OFF + ATR-trail does not beat the null**, so credit does
**not** belong to the exit policy. Neither component is a source of held-out alpha:
1. **Entry timing** (the MA cross) adds **negative** value within the regime (regime-null > real entry).
2. **The ER gate** contributes only **regime/beta selection** (removing it craters held-out by +41pp), and
   even that loses to random selection inside the regime → consistent with the prior REFUTED finding.
3. **The exit policy** (ATR-trail) carries **no** edge — cell C is the worst cell, C-real ≈ C-null, and
   ATR-vs-opp-cross is a wash that never clears the null.

The redirect's "edge" is neither in the entry conditioning nor in the exit policy — it is residual
regime/beta, and it does not survive a cost-matched random-entry null on any of 77 assets.

## RWYB reproduction
```
python experiments/adaptive_ma/expert/ablation_2x2_4h.py --probe BTCUSDT   # single-asset 2x2 detail
python experiments/adaptive_ma/expert/ablation_2x2_4h.py --quick           # 25 assets
python experiments/adaptive_ma/expert/ablation_2x2_4h.py                   # full u100 (77 evaluable)
```
Artifacts: `ablation_2x2_4h.py`, `ablation_2x2_4h_u100.json`, `ablation_2x2_4h_quick.json`.
```
SAFETY: no commit/push/deploy/capital — analysis + JSON only, under experiments/.
```
