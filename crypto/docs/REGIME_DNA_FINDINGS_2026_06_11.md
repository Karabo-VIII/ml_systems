# Regime-DNA reality check — FINDINGS (2026-06-11, audit-corrected)

Answers the architecture fork (1 system or 50? does regime-reactive switching help? is per-asset
DNA real?) the user posed. `src/strat/regime_dna_lab.py` r2 — built, RED-teamed (1 CRITICAL + 4 HIGH
defects found and fixed: cross-split hold leak, fixed-count jackknife, agnostic-vs-filtered Q3 null,
sum/breadth favoring high-n, firewall overclaim), re-run clean. The preliminary "coverage wins" read
was an artifact; this is the corrected verdict. **Scope: 6 predetermined MA-family configs (EMA
crosses / Donchian / ROC / RSI / Boll, ±ATR trail), 1d, long-only spot, u10 + u50.**

## The three systems (TRAIN-selected, OOS-validated, UNSEEN-tested-once)
- SYS_C pooled: one config (TRAIN-best across all assets) applied to every asset.
- SYS_B per_asset: per asset, its single TRAIN-best config (asset DNA).
- SYS_A regime_sw: per (asset×regime), the regime's TRAIN-best config; reactive switching.

## Results (per-trade expectancy ± se; fair drop-top-5% jackknife)
| | u10 OOS | u10 UNSEEN | u50 OOS | u50 UNSEEN |
|---|---|---|---|---|
| SYS_C pooled | +5.89±7.43% | −6.38±3.31% | +8.31±6.48% | −2.76±8.69% |
| SYS_B per_asset | +6.22±5.76% | −3.32±1.76% | +4.82±2.92% | −4.89±1.04% |
| SYS_A regime_sw | +6.00±3.67% | −2.08±1.14% | +1.87±1.36% | −1.25±0.82% |

## Verdict (answers the fork)
1. **Per-asset config DNA is NOT real.** The three systems' OOS means are statistically
   indistinguishable (every se ≥ mean/1.6), and per-cell config survival is BELOW a random-config
   null (u10 0.58 vs 0.69; u50 0.46 vs 0.56). Building 50 bespoke configs would be fitting noise.
   → **Build ONE robust config, not 50.**
2. **Regime-config SWITCHING adds no value.** Under the fair jackknife every system's OOS edge
   collapses to ≤0 (concentration in all three); SYS_A's prior "win" was a trade-count artifact.
3. **Regime GATING is the real lever — and it's confirmed.** SYS_A decomposed by regime:
   - u10: OOS UP **+12.01%** (n57) vs DOWN −1.97% (n43); UNSEEN UP −0.91% vs DOWN −2.74%.
   - u50: OOS UP **+6.41%** (n225) vs DOWN −2.00% (n263); UNSEEN UP **+1.18%** (n105) vs DOWN −2.69% (n177).
   ALL the edge is UP-regime trend-continuation (beta); the DOWN-regime cells LOSE on both OOS and
   UNSEEN — confirming **D58** (long-only cannot harvest non-trend regimes; the "coverage" half is
   the dead bear-bounce vein). **At u50 the UP-regime long-trend is POSITIVE even on the UNSEEN
   2025 bear (+1.18%/trade)** — the one thing that survives the clean held-out.
4. **The architecture is therefore: ONE robust long-trend config, regime-GATED to UP (cash in
   DOWN, never try to harvest it), broad across u50.** That is exactly the Wave-1 / thread-22
   regime-managed-beta book. The per-asset + regime-switch elaboration is noise to be avoided.

## Honest scope / what this does NOT say
- It tests MA-FAMILY config space at 1d only. It does not prove "no per-asset edge exists anywhere"
  — a different signal family, or non-config differentiation, is untested.
- All blended systems are net-negative on the UNSEEN bear (long-only ceiling); only the UP-gated
  component is positive. This is the beta+regime-gate result the project already established —
  re-confirmed cleanly, not exceeded.
- The user's "movers span all regimes / need coverage" intuition is correct as a MARKET FACT, but a
  long-only SPOT book structurally cannot convert DOWN-regime moves to long profit (D58). Coverage
  would require shorts/perps (a separate LO-exception decision) or non-price leading data.

Repro: `python -m strat.regime_dna_lab --universe {u10,u50} --cadence 1d --regime trend` (seed 7;
lineage in the run JSONs).
