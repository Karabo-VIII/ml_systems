# Adaptive-MA META Loop Log

The META loop's synthesis ledger: each entry reads both rigs' output, audits overfit/look-ahead,
synthesizes the next highest-EV adjacent objective + an explicit anti-overfit BOUND, and writes the same
[META] guidance to BOTH learnings lanes (channel=expert, channel=plain) + here. RWYB: every number below
was produced/recomputed from a command, not quoted from a rig's printed headline.

---

## [META] #1 -- 2026-06-05 23:49:53 SAST (VERIFIED `date`) -- cycle-2 synthesis: the 4h ER-gated redirect is REFUTED

### Inputs synthesized (m1-m4 = the OVERSEER redirect memos, now in both lanes)
- **m1**: 1d MA-cross REFUTED (0/69 beat the random-entry firewall on held-out; do not re-mine the 1d cross).
- **m2**: redirect lever #1 -- use Efficiency-Ratio as a HARD GATE (trade only when ER>~0.4), not a switcher.
- **m3**: cadence=4h; exit=ATR-trail + time-stop (<7d), not opposite-cross; add breakout-confirm.
- **m4**: anti-overfit -- MINIMAL 3-DOF (ER thr + 1 MA + 1 exit); prove ER-gated FIXED-MA beats the
  regime-matched null BEFORE adding map cells.

### New cycle-2 empirics (BOTH rigs built the redirect; RWYB-recomputed by META from per-asset arrays)
- **PLAIN rig** -- `plain/ergated_fixed_ma_4h.py` (the exact 3-DOF config m2-m4 prescribed: ER>0.4 hard-gate |
  SMA10/30 | ATR-trail 3xATR14 + 42-bar(7d) time-stop | entry = ER>0.4 AND close>prior-20-high AND fast>slow):
  - Pooled **held-out per-trade NET = -2.2224%** (n=1098, winrate 0.270, p05 -12.78%).
  - Pooled **UNSEEN per-trade NET = -1.8059%** (n=347, winrate 0.305).
  - **11/77** assets positive held-out (median asset -2.08%).
  - => The minimal honest "null to beat" does **not clear zero**, let alone the 2-5%/move floor.
- **EXPERT rig** -- `expert/er_gate_4h.py` (the decisive regime-matched FALSIFIER: does ER-gated *timing*
  beat random entries drawn from inside the SAME ER>0.4 windows, same hold dist, same cost?):
  - **0/77** assets beat the regime-matched (ER>0.4) random-entry null; **0/77** beat-null-AND-pos-held.
  - 13/77 positive-held = regime/beta, NOT timing (none survive the within-regime firewall).
  - **`positive_control_4h.json`: has_power=True** -- the firewall ACCEPTS a planted real edge
    (verdict "REAL ENTRY-TIMING EDGE"), so the 0/77 is a **credible refutation, not a dead gate**
    (two-sided soundness satisfied).
  - Live META re-run (`--probe ETHUSDT`): 1627 ER-gated entries -> "BETA-IN-DISGUISE / no timing edge".

### Synthesis / verdict
The redirect (m2-m4) was the highest-EV repair of m1 -- it kept the MA-cross TRIGGER and changed only the
conditioning (ER gate), cadence (4h), and exit (ATR-trail). Both rigs, independently, with a
power-confirmed firewall, show that was not enough: **the MA cross has no held-out entry-timing edge --
1d AND 4h, raw AND ER-gated, EMA AND SMA, state AND cross-event.** The failure is the **TRIGGER**, not the
gate / cadence / exit. An ER gate cannot manufacture timing edge from a trigger that has none; it only
reshapes when an edge-less signal fires.

**2-5%/move held-out null-beat status: REFUTED for the entire MA-cross trigger family. STILL OPEN for
non-MA (exogenous / event-driven) trigger families, which are untested.** The objective itself is not
globally refuted; the MA avenue within it is now closed (consistent with the market-research premise that
naive harvest is a coin-flip and a real *signal* is required -- MA-cross is demonstrably not that signal).

### NEXT highest-EV adjacent objective (falsifiable): CHANGE THE TRIGGER, not the conditioning
Keep the now-proven-sound apparatus (4h cadence; `src/strat/setup_harness.py` SetupHarness + ATR-trail
exit; the ER/trend-regime conditioning *concept*; the regime-matched random-entry firewall WITH its
positive-control power-check; WindowSpec TRAIN/VAL/OOS/UNSEEN splits; pooled-NET + per-asset-count + p05
verdict). The single new variable = the **entry trigger**. Per the market research (#1 durable avenue),
test ONE **exogenous / structural** setup that is independent of the MA-cross family:
**liquidation-cascade / capitulation-reversal LONG** -- a large down-flush + volume/range spike + reclaim
of a reference level (proxy if no liquidation feed: oversold flush + reclaim of prior swing/VWAP). LONG-ONLY.

### Anti-overfit BOUND (explicit, pre-registered)
1. **<=3 DOF** for the new trigger (threshold + 1 exit + optional 1 conditioner). ONE trigger, ONE config.
2. **Pre-register ALL constants on TRAIN/VAL only**; UNSEEN (>=2025-12-31) is the verdict surface, touched once.
3. **Mandatory regime-matched firewall + positive-control power-check** (prove the gate accepts a real
   signal AND would reject this one if it's null) -- two-sided soundness, every run.
4. **DSR / Holm correction at family_n** -- this is the 3rd trigger family tested (1d MA, 4h ER-MA, new);
   apply multiple-comparisons discipline.
5. **Falsifier (the bar):** the new setup must beat the regime-matched random null AND stay positive on
   held-out for a pre-registered MEANINGFUL count -- **>=10/77 assets, pooled block-bootstrap p05 > 0, AND
   pooled UNSEEN per-trade NET >= 0** -- as the FLOOR *before* the 2-5%/move target is even discussed.
   If it 0/77's like the MA family with the power-confirmed firewall, the "price/structure-only 4h
   long-only setup has held-out timing edge" hypothesis is REFUTED -> escalate scope (instrument:
   perp funding / basis / cross-asset) OR surface Fork A (accept the beta-only ceiling) to the user.

### What to NOT do (explicit)
- **Do NOT sweep** ER thresholds, MA pairs (8/21, 10/30, 15/40...), ATR multipliers, breakout-N, or
  entry_style (state vs cross). The MA family is refuted with a **power-confirmed** firewall; a positive
  config found by sweeping = a multiple-comparisons FALSE POSITIVE, not an edge.
- **Do NOT re-test** any MA-cross trigger variant (1d/4h, EMA/SMA, raw/gated/adaptive-map).
- **Do NOT add adaptive-map cells / regimes** to switch among MA configs -- there is no per-config timing
  edge to switch among (F-B: all fixed beat the switcher; firewall 0/77).
- **Do NOT re-prove the 3-DOF baseline on UNSEEN** -- it is DONE (UNSEEN -1.81%, 11/77). The earlier
  instruction "prove the 3-DOF baseline on UNSEEN first" is now SATISFIED; the result is the refutation.
- **Do NOT chase the outlier-skewed mean** (expert UNSEEN mean comp +6.85% is a ZEC-type tail; median
  -0.7%). Decide on median / per-asset count / firewall, never the pooled compound mean.

### RWYB reproduction (commands META ran)
```
# recompute plain pooled held-out + expert firewall counts from saved per-asset arrays:
python - <<'PY'  # (see META session) -> plain -2.2224% held / -1.8059% UNSEEN / 11-77 ; expert 0/77 beats-null
PY
python experiments/adaptive_ma/plain/run_u100_4h.py            # plain 3-DOF, full u100
python experiments/adaptive_ma/expert/er_gate_4h.py            # expert regime-matched firewall, full u100
python experiments/adaptive_ma/expert/er_gate_4h.py --probe ETHUSDT   # live: BETA-IN-DISGUISE
```
