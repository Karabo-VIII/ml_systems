# The Strategy That Works + the Honest Target Verdict (2026-06-10)

**The decision-grade capstone of the 2026-06-09/10 /orc run.** After exhaustively testing entry signals, mover-capture
(D67/D68, signal-limited null), and the user's reframes, this is *the one thing that works*, framed against the user's
actual targets (2X/yr; 1d/3d/7d bands; max-DD <30%; LO+spot+lev=1). All figures **VERIFIED (RWYB** from
`src/strat/entry_signal_lab.py`).

## The strategy: a REGIME-MANAGED-BETA BOOK (the only validated edge)
- **Universe:** equal-weight u50/u100 (broad — the edge widens with breadth).
- **Per asset:** LONG when `close > SMA(fast, ~100-150d)`, FLAT when below. (A *fast* MA; the classic 200-day is too
  laggy for crypto and loses — D-context.) Use *a* fast MA; the exact length is not tunable (PBO 0.64).
- **Exit:** regime-exit (cross below the MA) — the simple always-exit beats the DNA-conditional at the book level
  (diversification subsumes the per-asset whipsaw).
- **Sizing:** equal-weight, flat=cash. (Per-trade sizing is NOT the lever — see below.)
- **Why it works:** it is **capital-PRESERVATION**, not alpha. Entry timing is fungible (membership-matched firewall);
  the value is regime-participation + diversification — being out of the alt crashes. It **survives the
  exposure-matched phase-shift null** (real regime-timing, not a de-risking artifact).

## What it delivers (VERIFIED, full cycle 2020-2026, u50)
| | regime book | buy&hold basket |
|---|---|---|
| geometric annual | **~48%/yr** | ~43%/yr |
| full-cycle compound | ~12× (1087%) | ~10× (901%) |
| max drawdown | **−58%** | −82% |
| Calmar | **18.7** | 11.0 |
| 2X/yr (+100%) hit | 1/7 years (bull 2021 only) | 1/7 |

Per-year (u50): 2020 +49%, **2021 +519%**, 2022 −31%, 2023 +64%, 2024 +67%, 2025 −24%, 2026 −4%. It **beats buy&hold on
both return AND drawdown over the full cycle**, and preserves capital in every bear year (2022 −31% vs −75%; 2025 −24%
vs −56%).

## The honest target verdict
- **2X/yr robustly: UNREACHABLE within LO+spot+lev=1.** It is hit only in *bull* years (2021), where beta alone gives
  it. There is no daily/4h causal alpha to lift the non-bull years (proven exhaustively: D14/D17/D44/D63/D67 + this
  run). Robust 2X/yr requires **leverage** (excluded by lev=1) or **alpha** (null).
- **The <30% max-DD constraint is VIOLATED at full deployment (−58%).** Drawdown is a *fixed tradeoff* with return:
  sizing is NOT a free lever — deploying more raises DD toward the basket's −82%; deploying less lowers return. To meet
  <30% DD you size down to ~half exposure → **~25%/yr @ ~30% DD** (the deployable, constraint-respecting point).
- **So the achievable frontier is ~25%/yr (@ <30% DD) to ~48%/yr (@ −58% DD)** — a genuinely strong risk-managed-beta
  product (beats buy&hold on both axes), but **not the 2X/yr target.**

## Why this is the ceiling (not a failure of effort)
The entire 2026-06 investigation converges on one structural truth: **within LO+spot+lev=1 at daily/4h, upside
prediction/capture is null; the only edge is downside-avoidance.** Mover-capture is signal-limited (per-move negative
even frictionless — D67). The dead-list (D01-D68) is the evidence. This is the *constraint's* ceiling, not a global one.

## The two ways past it — both now tested + weaker than hoped (the fork, honestly priced)
1. **Relax lev=1** — but leverage only helps as **DUMB leverage-on-beta** (raise return AND DD proportionally → 2X/yr
   needs ~2× DD = >100% = ruin). **SMART conviction-leverage is NULL** (VERIFIED 2026-06-10): sizing the book up on
   macro-conviction (ETF+stablecoin risk-on) gave FULL 1007% vs 1167% fixed-size, Calmar 18.7 vs 21.2 — *worse*,
   because the macro signal does not predict *when* to deploy more. So leverage is not a clean path to 2X.
2. **New LEADING data (Fork B)** — the FREE crypto-macro half is now tested and is **preservation, not alpha**: a
   macro-liquidity risk-on/off GATE (ETF outflow / stablecoin-crash / cross-vol-spike) on the book cuts held-out DD
   (OOS −17% vs −24%, UNSEEN −7% vs −11%) but *costs* return (48.7→40.3%/yr) → same frontier, lower full-cycle Calmar.
   Within FIXED-SIZE, leading data can only gate (remove risk), not amplify. The remaining (paid) leading data — CFTC
   COT, CryptoQuant netflow, Coinglass liq-heatmap — is now a **low-prior** bet (every leading-data-like signal tested
   — structural, learned, macro — is coincident/non-forward-predictive). Honest EV: low.

**The hard truth (now exhaustively earned):** NO signal tested — price, structural microstructure, learned multi-feature,
OR macro/cross-asset flows — is forward-predictive enough to add alpha within LO+spot. The only edge is
capital-preservation. **2X/yr robust is structurally unreachable here**; the achievable, genuinely-good product is the
~25-48%/yr risk-managed-beta book. More requires either accepting ruinous leverage-DD, or a *yet-unfound* predictive
signal (the paid-data bet, low-prior). This is the constraint's ceiling — escaping it is a data/instrument decision, not
a research-effort one.

**Bottom line:** there *is* a profitable, robust strategy — the regime-managed-beta book at ~25-48%/yr beating
buy&hold on return and drawdown. There is *not* a robust 2X/yr within the current constraints. Hitting 2X needs a
constraint relaxation (leverage) or new leading data (the fork) — both your call.

*Provenance: /orc autonomous run, 2026-06-09 21:44 → 2026-06-10. VERIFIED from entry_signal_lab.py.*
