# Phase 2 — Dynamic/Adaptive Engines: RIGOROUS DESIGN (2026-06-25)

**Status: DESIGN (user approved; build next). Supersedes PHASE2_DYNAMIC_ENGINES_PLAN.md
which was written before V6's findings.**

---

## 0. What V6 proved (the honest starting position)

The V6 rigorous static build (12 families × 5 TFs, full V1.1 architecture) established:

1. **Only 2 engines in the entire static TI space are genuinely selective:**
   - MA V2 BAND (WMA|4h): 35% deploy, +4.11%/wk OOS, 100th-pct null — the gold standard
   - ADX|1d: 67% deploy, +2.26pp/wk VAL, Sharpe 3.38 — the V6 discovery

2. **Every other TI indicator deploys 98-100%** under full rigor (TRAIN selection +
   decoupled confirmation + regime gate). Their signal logic is structurally "long when
   trending" which is always true under a bull gate. The TRAIN filter can't reduce them.

3. **Sub-daily TFs (1h/30m) universally bleed** from cost + noise. 4h is marginally positive
   for a few families; 1d carries the most signal.

4. **The regime gate is the dominant value lever** (D78 CASH THEOREM) — being long in bulls
   and cash in bears captures most of the opportunity. TI timing adds marginal value.

**Implication for Phase 2:** the static bottleneck is NOT "wrong indicator" — it's that
static parameters can't make indicators selective in a structurally always-on environment.
Dynamic adaptation has a concrete, testable hypothesis: if adaptation can make the signal
say "no" more often (increase selectivity from 98% to ~40-70%), THAT is the mechanism by
which it could add value. This is mechanically different from static (CHARTER §ENGINE BUILD
RIGOR doesn't fix it — it's not a rigor problem, it's a design-space exhaustion).

---

## 1. The precise hypothesis (what must be proven, not assumed)

**H: endogenous adaptation of an indicator's threshold can increase SELECTIVITY (from
~100% to 40-70% deploy) while maintaining positive edge and beating null.**

The mechanism: in periods where the adaptation state-variable says "regime is uncertain /
choppy / high-vol-of-vol," the adapted threshold RISES (demands stronger signal to enter)
→ engine goes to cash in uncertain periods → deploy drops → if those periods genuinely
have lower expected return than selected periods, edge appears.

This is NOT the same as the V1 regime gate (which uses BTC SMA100 and goes cash in BEARS).
This is a WITHIN-BULL-REGIME selectivity layer: even when the gate says "bull, be long,"
the dynamic engine can say "not now — uncertainty within the bull is too high."

**Kill criterion (decide now, before building):** if the dynamic engine at its optimal
adaptation parameter does NOT reduce deploy below 80% AND beat its static twin on VAL,
it adds nothing beyond what the static already does. Document the null; don't iterate.

---

## 2. Adaptation state variables (causal, endogenous)

Computed from the price/vol data itself within the trailing lookback window [t-L, t]:

| State var | Formula | Intuition | D73 safe? |
|---|---|---|---|
| `rv` realized vol | `std(returns[t-L:t])` | wide threshold in high-vol (avoid noise) | YES (own price) |
| `eff` efficiency ratio | `|close[t]-close[t-L]| / Σ|diff[t-L:t]|` | short MA in trends, long in chop | YES (own price) |
| `vov` vol-of-vol | `std(rolling_rv[t-2L:t])` | uncertainty → demand more agreement | YES (own price) |
| `ac1` lag-1 autocorr | `autocorr(returns[t-L:t], lag=1)` | trend vs mean-revert switch | YES (own price) |
| `maxdd` recent drawdown | `max(close[t-L:t])-close[t]) / max(...)` | de-risk after drawdown | YES (own price) |

ALL endogenous. None uses external factors, calendar, macro, or cross-asset data. D73 safe.

---

## 3. Dynamic engine candidates (ordered by prior probability of success)

### DYN-3: Efficiency-Adaptive MA Period ("MA-EA") — augment the WINNER

**Rationale:** MA V2 BAND is the program's strongest engine. Its weakness: fixed MA
periods lag in fast trends and whipsaw in chop. KAMA already does something similar
(efficiency-weighted smoothing), but in a fixed/opaque way. MA-EA makes it explicit.

**Mechanism:**
```
eff = |close[t] - close[t-L]| / sum(|diff|[t-L:t])
adapted_period = round(base_period * (1 - m * eff))
                 clamp [min_period, max_period]
```
When `eff` is high (strong trend): period SHORTENS → faster MA → catches moves earlier.
When `eff` is low (chop): period LENGTHENS → slower MA → fewer false crosses.

**Free params:** `m` (adaptation strength), `base_period`, `min_period`, `max_period`.
But `base_period` IS the static engine's period (frozen from V2), and min/max are
structural bounds (3 bars, 80 bars). So effectively ONE free param: `m`.

**Static twin (the control):** WMA|4h with `m=0` (= the frozen V2 band engine). The
dynamic engine must beat THIS on VAL. If it can't beat WMA|4h, it adds nothing.

**Validation target:** dynamic deploy < 80% AND edge > static-twin AND null ≥ 95th.

---

### DYN-4: Efficiency-Adaptive ADX Threshold ("ADX-EA") — augment the V6 DISCOVERY

**Rationale:** ADX|1d is already selective (67%) with a fixed threshold. Can adaptation
make it BETTER (higher edge or tighter selectivity) by raising the threshold when
efficiency is already high (demand stronger trend to enter) vs lowering when efficiency
just turned up from a chop period (early entry)?

**Mechanism:**
```
eff_long = efficiency_ratio(close, lookback=28)
eff_recent = efficiency_ratio(close, lookback=7)
delta_eff = eff_recent - eff_long
adapted_thr = base_thr + a * delta_eff
```
When efficiency is ACCELERATING (delta_eff > 0): lower threshold → enter earlier.
When efficiency is DECELERATING: raise threshold → demand more confirmation.

**Free params:** `a` (adaptation strength). Base is the frozen V6 ADX|1d (thr=0.45, lb=28).

**Static twin:** ADX|1d lb28 thr0.45 with `a=0`.

---

### DYN-1: Vol-Adaptive RSI Threshold ("RSI-VA")

**Rationale:** RSI|1d had a small positive edge (+0.27pp) but at 98% deploy (not selective).
If the RSI threshold RISES in high-vol (demand stronger RSI reading to confirm), the engine
should deploy less and avoid the noise-heavy periods.

**Mechanism:**
```
rv = std(returns[t-L:t])
rv_z = (rv - rolling_mean_rv) / rolling_std_rv   # z-score
adapted_long_thr = base_thr + k * rv_z
```
High vol → higher threshold → fewer entries → selective.

**Free param:** `k`. Static twin: RSI|1d with `k=0` (all configs, no adaptation).

**Expected outcome (honest):** Lower-prior than DYN-3/DYN-4. RSI under V6 rigor was
already non-selective at static — adaptation may not rescue it because the SIGNAL ITSELF
(RSI > threshold) is structurally too easy to satisfy. But worth the test for completeness.

---

### DYN-2: Bandwidth-Adaptive Bollinger ("BB-VA")

**Rationale:** Bollinger at static = 100% deploy (close > middle band is always true in
a bull). Adaptation: widen the UPPER band in high vov → demand a stronger breakout to enter.

**Mechanism:**
```
vov = std(rolling_rv[t-2L:t])
adapted_mult = base_mult * (1 + g * vov_z)
```
High vov → wider bands → price must break FURTHER above to trigger → selective.

**Free param:** `g`. Static twin: Bollinger with `g=0`.

**Expected outcome:** Similar to RSI-VA — the fundamental issue is that "close > middle
band" is always true. Only the UPPER-band breakout variant has a chance at selectivity.
I'll use `entry_type="breakout"` (close > upper band) as the base signal, which is already
more selective than middle-band. Then adapt the multiplier.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────┐
│  LAYER 0: Regime Gate (V1 BTC SMA100)               │  unchanged from static
├─────────────────────────────────────────────────────┤
│  LAYER 1: Adaptive Engine                           │
│  ┌───────────────────┐  ┌────────────────────────┐  │
│  │ adapt_fn(window)  │→ │ signal(adapted_params) │  │
│  │ computes state    │  │ majority vote over     │  │
│  │ vars from [t-L,t] │  │ configs → confirmed   │  │
│  └───────────────────┘  └────────────────────────┘  │
│            ↓                        ↓                │
│  adapted_threshold / period  → held state → cost    │
├─────────────────────────────────────────────────────┤
│  OUTPUT: held[t] = 0 or 1 per asset, lag-1 causal   │
└─────────────────────────────────────────────────────┘
```

Implementation: extend `ti_engine_rigorous.py` with an `adapt_fn` parameter:
```python
def make_dynamic_hold_provider(family, configs, lookback_bars, slice_bars,
                                vote_threshold, adapt_fn):
    """adapt_fn(close_window) -> adapted_params (dict)."""
    ...
```
The static twin is `adapt_fn = lambda w: {}` (no adaptation, base params used).

---

## 5. Validation gauntlet (HIGHER bar than static — more free params = more overfit risk)

| Step | Description | Kill if... |
|---|---|---|
| 1. DEV build | TRAIN ≤ 2023-07-01, optimise `m`/`k`/`g`/`a` on TRAIN only | — |
| 2. Static-twin A/B | dynamic VAL edge > static VAL edge (both on VAL 2023-07→2024-05-15) | dynamic ≤ static |
| 3. Selectivity check | dynamic VAL deploy < 80% (down from 98-100% static) | deploy ≥ 90% (not selective) |
| 4. Random-null | engine vs random-adaptation (random `m`/`k` drawn uniformly) at same deploy | < 95th pct |
| 5. Adaptation-null | engine vs random SHUFFLE of the adaptation state (breaks temporal structure) | < 95th pct |
| 6. OOS-2024 | 2024-05-15 → 2024-12-31 (FROZEN). Must still be positive. | OOS edge ≤ 0 |
| 7. Overfit-probe | sweep free param `m`/`k`; OOS surface must be a plateau | spike (single value works) |
| 8. UNSEEN SEALED | ≥ 2025. NOT touched until deploy decision. | — |

Step 5 (**adaptation-null**) is NEW and Phase-2-specific: it shuffles the time-ordering
of the adaptation state variable so the adaptation has no temporal structure but the same
distribution. If the engine still "works" with shuffled adaptation → the adaptation adds
nothing; it's the base signal doing the work. This isolates adaptation-VALUE from
base-signal-value.

---

## 6. Build order (DYN-3 first, then DYN-4, then optional DYN-1/2)

| Priority | Engine | Why first |
|---|---|---|
| 1 | **DYN-3 MA-EA** | Augments the WINNER (WMA|4h). Highest prior. |
| 2 | **DYN-4 ADX-EA** | Augments the V6 DISCOVERY (ADX|1d). Already selective; can it improve? |
| 3 | DYN-1 RSI-VA | Lower prior (RSI was non-selective). Test for completeness. |
| 4 | DYN-2 BB-VA (upper) | Lowest prior (Bollinger breakout may gain selectivity with adapted width). |

Build protocol (per engine, same as V1→V1.1):
1. Define signal + adapt_fn + static-twin
2. TRAIN sweep of the free param (10-20 values, systematic)
3. VAL evaluation of the winning TRAIN param
4. Static-twin A/B on VAL → pass/kill
5. If pass: random-null + adaptation-null
6. If pass: OOS-2024 (frozen, one shot)
7. If pass: overfit-probe (sweep param on OOS surface)
8. Commit + document in V7_dynamic/ runs

---

## 7. Constraints & dead-path guards

- **D73:** NO external/calendar/macro factors. ALL adaptation uses ENDOGENOUS state vars
  (price returns, vol, efficiency). If a PR or build accidentally introduces a cross-asset
  or calendar feature → reject on sight + cite D73.
- **CHARTER §ENGINE BUILD RIGOR:** still applies. Systematic grids, decoupled lookback,
  confirmation, maker cost, lag-1, null. The dynamic layer sits ABOVE the rigorous base.
- **One free param per engine.** Two or more → combinatorial overfit risk. If one param
  can't make it work, the mechanism is too weak — don't add a second.
- **Report BOTH linear/wk and compound.** Single-asset BTC baseline (no rebal premium).
- **2025 SEALED.** Not touched until a deploy decision.

---

## 8. What success looks like (define BEFORE building)

**A dynamic engine is VALIDATED if:**
- VAL edge > its own static twin's VAL edge (adaptation adds value)
- VAL deploy < 80% (selectivity achieved)
- VAL random-null ≥ 95th AND adaptation-null ≥ 95th
- OOS-2024 edge > 0 (survives held-out)
- OOS overfit-probe = smooth plateau (robust to param choice)

**If NO dynamic engine clears this:** the honest conclusion is that endogenous adaptation
at the TI-indicator level does not add edge beyond what the regime gate + static MA/ADX
already capture. The system deploys the static base (MA V2 + ADX|1d). This is a perfectly
valid outcome and will be documented as such.

---

## 9. Deliverables

- `crypto/src/strat/ti_engine_dynamic.py` — adapt_fn + 4 dynamic engines
- `crypto/src/strat/dynamic_gauntlet.py` — the §5 validation (static-twin, null, adapt-null)
- `crypto/runs/engines/V7_dynamic/` — results per engine
- Updated CURRENT.md + KNOWLEDGE_MAP
- Final verdict doc: `V7_dynamic/DYNAMIC_ENGINES_RESULT.md`

---

## 10. Timeline estimate

DYN-3 (MA-EA): ~2-3 hours (fast TF, existing infra)
DYN-4 (ADX-EA): ~1-2 hours (1d panel is small)
DYN-1/2 (RSI/BB): ~1-2 hours each
Gauntlet (per engine): ~30 min each
Total: ~6-10 hours of agent execution time.

---

**This design is BINDING for the Phase 2 build. Do not deviate without user approval.**
