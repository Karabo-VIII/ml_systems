# Minimal 3-DOF 4h Breakout (ER-gate + ATR-trail) — Honest Report

**Status: REFUTED** as a held-out entry-timing edge. The researcher prescription
(experiments/adaptive_ma/OVERSEER_LOG.md check-in 6: ER HARD GATE + 4h + breakout-confirm + ATR-trail,
minimal DOF) was built exactly to spec, verified causal (no look-ahead), and scored on the kept apparatus.
After cost, on held-out data, it does **not** beat a cost-matched random-entry null on essentially any
asset or any parameter config. A refutation is a valid, valuable outcome; logged here.

## Candidate (PRE-REGISTERED before seeing any held-out number)
- cadence **4h**; single fixed MA = **EMA 8/21**; HARD GATE **ER > 0.40** (Kaufman, win 20, skip when below)
- entry (confirmed at close): `close > prior-20-bar HIGH  AND  fast>slow  AND  ER>0.40`
- exit: **ATR-trailing stop** (3.0 x ATR, win 14) **+ time-stop 42 bars (= 7 days)**. No take-profit.
- cost = taker **0.0024** round-trip; LONG-ONLY, SPOT, lev=1; UNSEEN = 2025-12-31 -> 2026-05-22.
- Apparatus: `src/strat/setup_harness.py` (SETUP->MOVE, IC-independent, next-bar-open fill, intra-bar
  breach via highs/lows) + `src/strat/firewall.py` (cost-matched random-entry null) + `battery.py`.

## Held-out result — full u100 (77 assets, pre-registered config)
| window | pool exp% | pool wr | pool n | mean comp | **median comp** | pos/assets |
|--------|----------:|--------:|-------:|----------:|----------------:|:----------:|
| TRAIN  | +0.528 | .334 | 2420 | +33.46% |  0.00% | 21/77 |
| VAL    | +0.060 | .334 |  809 |  −0.58% | −0.14% | 27/77 |
| OOS    | +0.272 | .333 |  828 |  +1.64% | −2.61% | 29/77 |
| **UNSEEN** | **+1.269** | **.377** | **358** | **+5.38%** | **−2.10%** | **32/77** |

- **UNSEEN per-trade expectancy = +1.27% (pooled MEAN)** but **median per-asset compound = −2.10%**, and
  only **32/77 (42%) of assets are positive** — i.e. the majority lose. The positive pooled mean is
  **outlier/beta-skewed**, not a broad edge.

## Decisive test — cost-matched random-ENTRY firewall (held-out OOS+UNSEEN)
- **regime-matched null (within setup-ON timing): 0/77 assets beat the null.**
- **plain null (selection vs random-anywhere): 1/77 assets beat the null** — *below* the ~3.85/77 you'd
  expect from chance alone at the p95 bar. => the entry timing / setup selection **adds nothing** over
  random entries at the same count, holding durations, and cost. **Beta-in-disguise.**

## Robustness — the refutation is NOT a knife-edge (27-config sweep, 25 assets)
ER-gate {0.30,0.40,0.50} x breakout-N {10,20,30} x ATR-mult {2,3,4}, all 27 combos:
- **regime-matched firewall = 0/25 for EVERY config**; **plain firewall <= 1/25 for every config**.
- UNSEEN **median compound is negative** in nearly every config (a few exactly 0.0); positive assets
  always a minority (8-13 / 25).
- Higher ATR-mult *raises* pooled mean expectancy (wider trail lets the regime's up-moves run) while the
  firewall stays 0/25 — direct proof the expectancy is **exit-policy + regime beta, not entry timing**.

## −k falsifiers (soundness, two-sided)
- **Apparatus has power**: `setup_harness.py --selftest` PASS — the gate accepts a genuine synthetic
  move-capture, rejects random, and fires on a deliberate future-leak. The 0/77 is a real negative, not a
  dead gate.
- **No look-ahead**: entry[t] re-derived from the truncated prefix `df[:t+1]` EXACTLY matches the
  full-series entry[t] — **0 mismatches / 51 sampled bars** on DOGE, ZEC, BNB. `causal_ok=True`. Entry is
  strictly past-only by construction (every feature `.shift(1)` / `rolling().shift(1)`, fill = next open).
- **leak_guard caveat (honest)**: on 3/4 positive-base assets the SetupHarness `leak_guard` returned
  LEAK_SUSPECT. This is a **false-positive of the relative lead/lag ratio on CLUSTERED breakout entries**
  (a ±1-bar shift is nearly a no-op when setups fire on consecutive bars, so the ratio is noise — the
  limitation its own docstring flags). The prefix re-derivation above is the authoritative causal proof
  and is clean. Moreover the firewall is leak-robust: any residual entry leak would only HELP the real arm
  beat the null, and it still does not.

## Mechanism diagnosis (why it fails)
Gating on ER>0.4 + a 20-bar breakout selects bars inside clean up-trends, and the ATR trail rides those
moves — so the *average* trade is mildly positive (you're long during up-regimes). But **random entries
into those same setup-ON / regime bars, held for the same durations, do equally well or better** (0/77).
There is no information in *which* breakout bar you pick beyond being in the regime — i.e. no held-out
entry-timing alpha. Consistent with the prior 1d adaptive-MA refutation and the standing premise that
daily/4h long-only MA-style timing has no robust held-out edge; the breakout-confirm + ER-gate + ATR-trail
prescription does not change it.

## RWYB — exact reproduction
```
python src/strat/setup_harness.py --selftest                          # apparatus has power: PASS
python runs/research/minimal_3dof_4h_breakout.py                      # full u100: pooled+firewall (0/77, 1/77)
python runs/research/minimal_3dof_4h_breakout.py --assets BTCUSDT,ETHUSDT,SOLUSDT
python runs/research/minimal_3dof_4h_sweep.py                         # 27-config robustness + causal/leak checks
```
Outputs: `minimal_3dof_4h_result.json`, `..._result_quick.json`, `..._sweep_result.json`, `..._OUT.txt`.
NOT committed (research artifact).
