# Find a LO/no-lev winner — exhaustive open-axis search verdict (2026-06-10)

User mandate: "find me a long-only, no-leverage strat that hits the 1d/3d/7d target (relaxed 2x/yr), use everything."
5-family open-axis Workflow, each built + grid-searched + run through candidate_gate + PBO + UNSEEN-once on real
chimera, then Opus synthesis + adversarial-skeptic verify (numbers re-checked vs artifacts + dead-list). No silent
target reframing.

## Verdict: NO robust LO/no-lev winner. 2x/yr is NOT reachable at bar-level under LO+spot+lev=1.

UNSEEN = Jan–May 2026, a deep bear (B&H −26% to −55%/yr). Every family "beats beta" ONLY via **bear-abstention**
(regime gate → cash in the bear); 4 of 5 **lose money absolutely** while losing less than B&H. That is the known
SCOPED result, not alpha.

| Rank | Family (axis tested) | UNSEEN CAGR | gate | read |
|---|---|---|---|---|
| 1 | **F2 mover-rotation (CSMOM harvest)** | **+1080%/yr** | **INCOMPLETE→FAIL** | the flicker — see below |
| 2 | F3 Renko (chart-type breadth) | −20%/yr | FAIL (0/10 seeds) | "least-bad" loss-mgmt, not alpha |
| 3 | F1 unclamped Chandelier exit | −24%/yr | FAIL (p05 −100) | exit adds no compound; stop-whipsawed on 4h |
| 4 | F4 setup-chaser book | −33%/yr | FAIL (**PBO 0.79** severe overfit) | diversification ≠ orthogonal returns (all trend, one regime) |
| 5 | F5 non-MA entries / F3 HA/Range | ~0 / −44%/yr | FAIL (0 UNSEEN trades / firewall fail) | regime gate blocked all bear-window entries |

## The one flicker — F2 — is NOT a winner (it's a known dead vein)
F2 (top-K by trailing return + MA200-rising filter + ATR trail) clears all bands *arithmetically* on UNSEEN
(+172% / +1080%/yr, DD −10.9%, battery LENS_A). But the adversarial pass kills it:
- **Gate incomplete = failed gate:** firewall (random-entry null) NOT RUN; PBO NOT RUN (it used a custom simulator).
- **Concentration-fragile (kill-shot):** drop top-10 days → +172% collapses to +32%; the conservative ATR8 variant goes
  **negative (−9.5%)**. It's event-capture of ONE alt-season (late-Apr/May 2026 NEAR/DEX/INJ/JST spike), not a compounder.
- **Thin:** n=22 trades, n_eff=4.1 months, win-rate 0.50.
- **Already dead-listed:** this IS **D68** ("finer-cadence x-sec SELECTION… edge does NOT hold UNSEEN + beats_rnd ~0.5,
  a few big days → regime-dependent") + **D67 HARD** ("causal mover-capture is SIGNAL-limited even frictionless"). The
  dead-list *predicted* the exact concentration-fragility the jackknife exposed. Prior: it fails a clean second window.

## Axes now REFUTED (this sweep, UNSEEN, LO+spot+no-lev)
- **Exit-mechanism** (unclamped Chandelier/ATR trail on 4h — no compound, stop-whipsawed). F1.
- **Chart-type / bar-construction** (Renko/Range/Heikin-Ashi do NOT rescue a LO trend book; regime, not bar, is the driver). F3.
- **Within-1d single-asset entry-signal** (MA, Donchian, vol-squeeze, RSI-pullback, 52w-new-high — all ≤+1.3%/yr OOS,
  none beats the random-entry null). Confirms D67/D68. F5.
- **Setup-family diversification** (214 slots × 3 cadences — PBO 0.79, all trend-following sharing one regime). F4.

## The only un-refuted path (a CONSTRAINT RELAXATION, out of scope for THIS ask)
**Sub-bar / leading-data entry** — sub-hour event-clock liquidation-cascade entries with **leading data** (on-chain /
OI-delta / funding pre-event). D67 explicitly names this as the single fork that *could* break the bar-level
signal-limit (execution alone can't). It requires new instrumentation and is **Fork B** — not within the LO+bar-level
frame. The perp-short leg (closes the bear-participation gap) is explicitly excluded by LO+no-lev.

## Bottom line
On the evidence in hand: **2x/yr robust is NOT reachable with bar-level (1d/4h/1h) entry signals under LO+spot+lev=1 —
that surface is now exhaustively refuted.** The honest robust deliverable is a controlled-loss bear-abstention book
(loses ~20%/yr in this bear vs B&H −55%). Do NOT deploy anything from this sweep. The only un-refuted path to 2x is the
sub-bar/leading-data frontier (Fork B), a data/instrument decision, not a research result.

Artifacts: `runs/strat/{family1_chandelier_trail,momentum_rotation_lab,alt_bar_trend_lab,setup_chaser_book,nonma_entry_lab}_2026-06-10.json`.
