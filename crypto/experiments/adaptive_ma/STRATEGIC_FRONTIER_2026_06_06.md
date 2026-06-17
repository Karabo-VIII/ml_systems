# Strategic Frontier — what tonight closed, what is genuinely open (2026-06-06, ~02:45 SAST)

Honest report for the user's return. Every result below is held-out (OOS+UNSEEN), net of 0.0024 round-trip cost,
on a confirmed-sound apparatus (shuffled-label control + positive control + regime-matched firewall + seed-robustness
+ OOS→UNSEEN persistence). No headline survived without surviving all controls. The diagnosis, not a decision —
strategic choice deferred to you (per your report-first preference).

## What was DEFINITIVELY CLOSED tonight (bar-level entry-timing)
The mandate was "MA on all instruments, decompose the oracle DNA." Completed comprehensively, then extended to the
strongest fair forms of the idea. Every one is a controlled NEGATIVE:

1. **Scalp oracle (the original framing was mis-specified).** `oracle_high_capture` maximized compound with no
   per-move floor → it decomposed into ~2-bar wiggles at every cadence. Asking a lagging MA to time 2-bar reversals
   was unfair. Broad causal DNA (40 features: MA+momentum+vol+orderflow/micro): **0/14 genuine**, capture −16%→−99%.

2. **Swing oracle (the CORRECT framing — fixed tonight).** Added a per-move net floor (`min_move_net`); at 3-5% the
   oracle becomes multi-day moves (1-4 day holds, 5-15% mean net/move) — the project's actual target unit.
   - Linear DNA: ICs flip positive on 1d (the frame is far less hostile) but **0/8 genuine**.
   - Nonlinear DNA (GBM = strongest fair form of "2-MA/3-MA adaptive"): one apparent BTC-1d hit, **stress-refuted**
     (seed-dependent: 1/3 seeds; OOS capture +58% → UNSEEN +0.1% = OOS-overfit).
   - + Liquidation/book/positioning features (82 total, the market-research top avenue): **0/6 genuine**.

3. **Liquidation cascades as an EVENT study (the proper framing for a rare signal).** Forward swing return after
   liq_short_spike / liq_long_spike / liq_capitulation, held-out: **no event clears significance** (best p_boot=0.178),
   sign-inconsistent across assets, most lifts negative. Closed in both framings (feature AND event).

**Bottom line:** no bar-level feature family (MA + orderflow + momentum + micro + liquidation + book + positioning),
linear or nonlinear, robustly times even multi-day swing entries out-of-sample at daily/4h resolution. This
reproduces the project's deepest standing finding (no active daily/4h long-only entry-timing alpha; what is robust is
beta) — now at the mechanism level, on a sound apparatus, having tested the idea's strongest fair forms and
stress-tested every apparent hit to destruction.

## What is GENUINELY OPEN (unrefuted; each different in KIND — a real fork for you)
Ranked by my EV estimate. None is "the proven path" — everything active is refuted; the proven path remains
beta+yield. These are research programs, not a continuation of the closed avenue.

1. **Sub-bar / tick / LOB representational data (highest prior — THREE avenues now converge here).** All three active-
   alpha avenues probed tonight (bar-level entry features, liquidation cascades, cross-asset lead-lag) are null at
   daily/4h AND each independently points to finer resolution: the 2-bar scalp oracle, the AUC-rises-as-bars-get-finer
   pattern, the liquidation event-decay, and lead-lag's known intraday strength all say the entry information lives
   *below* the daily/4h bar. Data-gated: needs tick/L2 ingestion (partial `bd_*` book-depth exists but ~53% coverage,
   recent-only). Biggest lift, biggest cost, and the single most-supported hypothesis from tonight's evidence.

2. **Cross-asset / relative-value / lead-lag — daily single-config now PROBED = null; HF untested.** Tested the most
   cited version (BTC up over 3d + alt lagging → alt forward 4d swing catch-up) across 10 alts, held-out, net cost:
   **0/10 clear significance** (p<0.1), aggregate lift −0.21%, only 3/10 positive. The lagging-alt condition selects
   weak alts that keep underperforming (momentum, not catch-up). The daily single-config is null — BUT lead-lag is
   known to be strongest at INTRADAY/minute resolution (BTC moves propagate to alts in minutes-to-hours), which again
   points to sub-bar/HF data (→ folds into frontier 1). Cross-sectional rank / pair relative-value at daily remain
   untested but now a lower prior.

3. **Exit/sizing/risk layer on a beta entry.** Entry-timing is refuted, but the swing oracle's value is partly in
   *where it sells*. A realizable exit policy (trailing/time/target) on a dumb trend entry is untested — though my
   prior is it mostly reconfirms beta (lower EV).

4. **Higher-frequency event study (4h/1h liquidation events).** More events than daily, but liq *features* already
   failed at 4h, so a lower prior than (1)-(2).

## The durable assets built tonight (reusable regardless of fork)
- **Swing oracle** (`runs/research/oracle_ceiling_builder.py` `min_move_net=`) — state the per-move floor + hold band
  so the oracle matches the target unit-of-trading. A no-floor max-capture DP silently becomes a scalper.
- **Two-sided falsifier** (`experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py`) — shuffled + positive +
  regime firewall, now with `--model {logistic,gbm}` and `--min-move-net`. Soundness gate fixed (mean-based, not
  p95-tail). The mandatory bar before believing any "genuine": seed-robustness + OOS→UNSEEN persistence.
- Both fold into `src/strat`'s candidate_gate in the 03:00 evolution cycle.
