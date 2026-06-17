# The Oracle Objective is a Design Variable — a per-objective taxonomy (2026-06-06)

> **Companion to [`docs/ORACLE_DECOMPOSITION_2026_06_06.md`](ORACLE_DECOMPOSITION_2026_06_06.md).** That note
> says *how* to mine an edge (construct the oracle → decompose its DNA → diffuse the noise → build a proxy). This
> note fixes a prior defect in **step 1 (CONSTRUCT THE ORACLE)**: the oracle's *objective* is not a constant — it
> is a **design variable** that silently determines the unit-of-trading you end up decomposing. Choose it wrong
> and you run a structurally unfair test (a lagging indicator vs a 2-bar scalp) and conclude "no signal" when the
> real conclusion is "wrong oracle". **The fix: any future indicator search FIRST selects the oracle objective
> whose hold-time distribution matches the indicator's lag, THEN decomposes.**

---

## 1. The defect this fixes (provenance — verified)

The 2026-06-06 adaptive-MA run built a perfect-foresight long-only max-capture oracle (`oracle_high_capture` in
[`runs/research/oracle_ceiling_builder.py`](../runs/research/oracle_ceiling_builder.py)) with **no per-move
floor**, decomposed its DNA, and found **0/14 genuine** at every cadence — capture −16% → −99%. The headline
"oracle holds ~2 bars at every cadence" was read as a structural property of the *market*.

The orchestrator correction (recorded in
[`experiments/adaptive_ma/MA_ORACLE_CONCLUSION.md`](../experiments/adaptive_ma/MA_ORACLE_CONCLUSION.md) §"SWING-ORACLE
EXTENSION") identified the real cause: **the ~2-bar hold was an artifact of the objective, not the market.** A
clairvoyant max-*compound* trader with no per-move floor greedily chops every up-leg into the smallest profitable
wiggles — a **scalping oracle**. Asking a lagging MA (which cannot confirm a move until several bars in) to time
2-bar reversals is an unfair test by construction. The fix was a `min_move_net` per-move floor that converts the
same DP into a **swing oracle** (multi-day moves). The swing reframe was the right unit — but the lesson was
recorded as a one-off patch, not as the general principle. This note generalises it.

## 2. Why a no-floor oracle becomes a scalper (the mechanism)

The DP maximises the product of per-move multipliers `∏ (high[j]/open[i] − cost)` over non-overlapping long
trades. With perfect foresight and **no floor**, splitting one big up-leg into two smaller ones is *always*
weakly better whenever both sub-legs individually clear the round-trip cost:

```
(1 + a)(1 + b)  ≥  (1 + a + b)      for a, b ≥ 0          # compounding rewards re-entry
```

So the unconstrained optimum decomposes the price path into the **finest cost-clearing wiggles** — i.e. the
shortest holds the cost floor allows. The floor that actually shapes this is **not** a time floor. Verified: the
prior `oracle_holdtime_prefilter.py` used `min_hold_hours=1.0`, which at 4h/1d is a **no-op** (1 h < 1 bar), so it
still reported ~1–2-bar holds and mislabelled 4h/1d as "too sharp". The load-bearing knob is the **per-move net
return floor `min_move_net`**: it forces each eligible move to clear a *size* threshold, which (since larger moves
take longer) lengthens the hold distribution.

`oracle_high_capture(..., min_move_net=f)` raises the per-move eligibility bound to `1 + f`
(`oracle_ceiling_builder.py:118` / DP at `:64`). `f = 0` ⇒ scalp; `f = 0.05` ⇒ swing.

## 3. The empirical floor → hold-time mapping (VERIFIED — sweep run 2026-06-06)

`experiments/adaptive_ma/_oracle_objective_taxonomy_sweep.py` (BTC/ETH/SOL, 4h & 1d, cost 0.0024). The floor —
**not the cadence** — sets the unit. Cross-asset aggregate (median of per-asset medians):

| cadence | min_move_net | median hold (bars) | median hold (days) | mean net / move |
|---|---|---|---|---|
| 4h | 0% (scalp) | 2 | 0.33 | 2.4% |
| 4h | 3% | 5 | 0.83 | 6.1% |
| 4h | 5% | 9 | 1.50 | 8.7% |
| 4h | 8% | 15 | 2.50 | 12.5% |
| 4h | 10% | 19 | 3.17 | 15.2% |
| 4h | 15% | 29 | 4.83 | 21.6% |
| 4h | 25% | 35 | 5.83 | 32.2% |
| 1d | 0% (scalp) | 2 | 2.00 | 6.0% |
| 1d | 3% | 2.5 | 2.50 | 9.2% |
| 1d | 5% | 3 | 3.01 | 11.5% |
| 1d | 8% | 4.5 | 4.50 | 14.8% |
| 1d | 10% | 5 | 5.00 | 17.0% |
| 1d | 15% | 5 | 5.00 | 22.0% |
| 1d | 25% | 6 | 6.00 | 33.3% |

Two facts fall straight out and both are load-bearing for the taxonomy:

1. **The floor is the unit-of-trading dial.** At 4h, sweeping the floor moves the median hold from 2 bars (0.33 d)
   to 35 bars (5.8 d) and the per-move size from 2.4% to 32% — a continuum from scalp to multi-day swing, on the
   *same* price series, *same* cost, *same* cadence.
2. **The 7-day hold cap saturates the high tiers.** The oracle DP caps hold at `MAX_HOLD_MS = 7 days`
   (`oracle_ceiling_builder.py:51`). At 1d the median hold flattens at 5–6 bars from the 10% floor up because the
   cap binds. **The current apparatus therefore cannot express a true "position" oracle** (multi-week / month
   holds). The highest realisable tier today is a ~1-week swing. This is a real limitation, not a tuning choice —
   see §6.

## 4. The taxonomy

Three canonical oracle objectives, keyed on `min_move_net`, each defining a unit-of-trading and therefore the
class of indicator that can fairly time it. Hold figures are VERIFIED at 4h from the §3 sweep; the position tier is
**aspirational** (blocked by the 7-day cap).

| objective | `min_move_net` | mean net / move | typical hold (4h) | unit-of-trading | indicator class that fits |
|---|---|---|---|---|---|
| **Scalp** | 0% | ~2–3% | ~2 bars (hours) | single-candle reversal | breakout / near-zero-lag only |
| **Swing** | 3–8% | ~6–13% | 5–15 bars (~0.8–2.5 d @4h; 2.5–4.5 d @1d) | a SETUP across a multi-candle move | fast MA / EMA / RSI / MACD |
| **Position** | 15–30%+ | ~20–35%+ | 30+ bars (1 wk, **cap-bound**) | a macro trend leg | slow MA cross / regime trend |

> The **project's stated unit-of-trading is "a SETUP across a MOVE"** (MEMORY.md founding framing) ⇒ the **swing
> oracle is the default**, and the scalp oracle (no floor) is an *invalid* objective for any lagging-indicator
> search. The scalp tier is retained only for genuinely low-lag signals (breakout, microstructure).

## 5. The selection protocol — match the oracle to the indicator's lag, THEN decompose

This is the operational rule. **Before** spending DNA-fit compute, select the oracle objective whose hold
distribution gives the indicator room to (a) confirm the move has started and (b) still have move left to ride.

**Step A — estimate the indicator's lag in bars.** For a moving average of window `W`, effective lag ≈ `(W−1)/2`
bars; the binding lag of a cross is the *slow* leg's lag plus any confirm bar. Breakout/Donchian ≈ 1 bar
(near-immediate). RSI(N) ≈ `N/2`. MACD(12,26,9) ≈ slow-EMA lag (~12.5) + signal confirm (~4) ≈ 16.5.

**Step B — require `median_hold_bars ≥ 2 × lag`.** The factor 2 is the minimum for "confirm + ride": one lag's
worth of bars to recognise the move, one lag's worth still remaining to capture. (This is the generalisation of the
prior `bars_per_MA_lag ≥ 2 ⇒ "MA_can_time: YES"` heuristic in `_audit_holdtime_driver.py`, now applied to any
indicator and any floor.)

**Step C — pick the lowest floor that satisfies Step B at the chosen cadence.** Lowest floor ⇒ most positive
labels ⇒ most statistical power for the DNA fit (see §7), subject to the lag constraint.

**Step D — decompose THAT oracle** (hand off to `oracle_decomposition` step 2: the shuffled + positive +
regime-firewall falsifier, then the mandatory seed-robustness + OOS→UNSEEN persistence bar).

### Empirical selection table (VERIFIED — computed from the §3 sweep)

Minimum `min_move_net` floor that satisfies `median_hold_bars ≥ 2 × lag`:

| indicator | lag (bars) | need hold (bars) | 4h | 1d |
|---|---|---|---|---|
| Donchian / breakout(20) | 1.0 | 2 | **floor ≥ 0%** (2 b / 0.3 d) | **floor ≥ 0%** (2 b / 2 d) |
| EMA(10) fast | 4.5 | 9 | **floor ≥ 5%** (9 b / 1.5 d) | **UNREACHABLE** within 7-d cap |
| RSI(14) | 7.0 | 14 | **floor ≥ 8%** (15 b / 2.5 d) | **UNREACHABLE** |
| MA(10/20) cross | 9.5 | 19 | **floor ≥ 10%** (19 b / 3.2 d) | **UNREACHABLE** |
| MACD(12,26,9) | 16.5 | 33 | **floor ≥ 25%** (35 b / 5.8 d) | **UNREACHABLE** |
| MA(20/50) cross | 24.5 | 49 | **UNREACHABLE** | **UNREACHABLE** |

**Reading this table (the whole point):**

- The MA-search's *correct* fair objective at 4h is the **5–10% floor swing oracle**, exactly what the corrected
  run used. The scalp oracle (0% floor, 2-bar hold) was structurally unfair to *every* indicator except a breakout
  — its 0/14 was preordained by the objective, not the market.
- **At 1d, every indicator slower than a breakout is UNREACHABLE** within the 7-day cap (MA(10) needs 9-bar = 9-day
  holds; the oracle never holds that long at 1d). This sharpens the prior conclusion: even the corrected 1d "swing"
  test (5% floor → only ~4-bar holds, VERIFIED) was *still under-matched* to MA(10)'s 9-bar lag — which is exactly
  the cadence where the lone (later stress-refuted) false-positive appeared. **An under-matched objective is where
  noise masquerades as signal.** To fairly test an MA at daily resolution you must either drop to a faster
  indicator or **lift the hold cap** (the position tier, §6) — *not* re-tune the model.
- **A genuine "no signal" verdict is only valid on a lag-matched objective.** Run the indicator only at the floor
  its lag earns; a refutation at the wrong floor is uninterpretable.

## 6. The missing tier: position oracle (apparatus gap, honest)

The taxonomy's position tier is **not currently buildable** — `MAX_HOLD_MS = 7 days` caps every hold (§3 fact 2).
To test slow trend indicators (MA(20/50), regime followers), the oracle needs a `max_hold_days` parameter so the DP
can express multi-week legs. This is a small, backward-compatible change (mirror the `min_move_net` addition:
thread a parameter into `oracle_high_capture` and widen the `jhi` searchsorted window). **Until then, any "slow MA
is refuted" claim at 1d is unproven, because the fair oracle for it cannot be constructed.** Flagged, not fixed
(no commit per task safety).

## 7. The power tradeoff the taxonomy must respect

Raising the floor buys a fair lag-match but **costs positive labels** (VERIFIED, 4h BTC): floor 0% → 2984 moves;
5% → 377; 15% → 65; 25% → 15. Fewer oracle-entry labels ⇒ a noisier DNA classifier and weaker held-out power.
So Step C's "lowest floor that satisfies the lag constraint" is not just convention — it is the **max-power choice
within the fairness constraint**. When the lag-matched floor leaves `< ~30` held-out positive labels, the DNA fit
is under-powered: tighten the firewall thresholds (sample-size discipline) and treat any "genuine" with extra
suspicion (this is the regime where the 2026-06-06 1d false-positive arose).

## 8. Integration (no skill edits — this is a methodology note)

- **`oracle_decomposition` step 1** gains a mandatory pre-step: *select the oracle objective via §5 before
  constructing it.* The objective (`min_move_net`, `max_hold_days`, cadence) is now an explicit, logged input to
  every decomposition, never a silent default.
- **`oracle_holdtime_prefilter.py`** should rank by `min_move_net`-induced hold (not the no-op `min_hold_hours`),
  and its verdict column should read against the *specific* indicator's lag, not a fixed 4.5-bar MA yardstick.
- **`src/strat` candidate_gate**: record the (objective, lag, floor) triple in every candidate's provenance; reject
  any decomposition whose floor was not lag-matched to its indicator (an unfair-test guard).
- The 7-day-cap → `max_hold_days` parameter (§6) is the one code change this implies; queued, not made.

## 9. What this does NOT change

The north star (robust, held-out, after-cost compound return; the "setup across a move" unit). The anti-overfit
battery. The honesty discipline — a lag-matched, well-powered refutation is still a valid, valuable result. This
note only ensures we refute the **right** oracle: it removes "we asked the wrong question" from the list of reasons
a search can fail, leaving only the real one — "the market does / does not offer this edge".

---

### Reproduce
```
.venv/Scripts/python.exe runs/research/oracle_ceiling_builder.py --selftest        # DP correctness
.venv/Scripts/python.exe experiments/adaptive_ma/_oracle_objective_taxonomy_sweep.py   # §3 floor→hold sweep
# §5 selection table is computed inline from _oracle_objective_taxonomy_sweep_result.json
```
All numbers in §3 / §5 / §7 are VERIFIED (produced by the sweep run on 2026-06-06, cost_rt=0.0024, BTC/ETH/SOL).
**Independently re-run on 2026-06-06 (overseer RWYB pass) and reproduced bit-exact:** DP selftest PASS; §3
cross-asset aggregate matched every cell; §5 selection table matched (EMA10→5% / RSI14→8% / MA10·20→10% /
MACD→25% @4h, all UNREACHABLE @1d); §7 power-tradeoff move counts matched (4h BTC: 2984 / 377 / 65 / 15 at
floors 0 / 5 / 15 / 25 %). The §4 position-tier holds and §6 are aspirational/blocked by the 7-day cap — tagged as such.
