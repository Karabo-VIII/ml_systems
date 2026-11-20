# Oracle Decomposition — the methodology (2026-06-06)

> **The north star (1d/3d/7d ROI + per-trade targets) is what an ORACLE attains from the market. We don't chase
> the number narrowly — we CONSTRUCT the oracle, DECOMPOSE its DNA (what causal conditions precede its moves),
> DIFFUSE the noise (separate the realizable signal from the perfect-foresight luck), and BUILD a causal proxy
> that gets *as close as the DNA allows*.** This replaces bottom-up trigger-guessing with top-down, data-driven
> signal discovery, and it makes the target honest (grounded in what the market actually offers).

## Why (provenance)
Bottom-up trigger search (MA-cross → ER-gate → liquidation → …) tests one human hypothesis at a time and is slow
+ POV-limited. The MA-cross family was fully refuted that way. The oracle frame inverts it: ask the *perfect*
trader what it exploits, then copy the part that is causally predictable. The targets become the oracle's
attainment; "closeness to the oracle" is the real objective. (User directive 2026-06-06: *"adopt them as what an
oracle can attain... match and decompose the oracle... diffuse the noise and get the DNA so we can copy it."*)

## The four steps (each with its rigor gate)

### 1. CONSTRUCT THE ORACLE (the ceiling)
For each asset at the target cadence (4h primary), compute the perfect-foresight LONG-only trade set that
maximizes net per-move capture under the constraints: hold ≤ 7 days (42×4h bars), net of taker 0.0024, non-
overlapping. This is a DP over the price path (max-profit constrained long trades) — clearly uses the future, but
ONLY to DEFINE the target; it is never a feature. Output: oracle equity, oracle trades (entry/exit bars), oracle
per-move expectancy = the **attainable ceiling** (a grounded, per-asset reading of "what 2-5%/move means here").

### 2. DECOMPOSE THE DNA (what does the oracle exploit?)
Label each bar: is it an oracle-ENTRY bar? Then learn `P(oracle-entry at t | CAUSAL features at t)` using only
past-only features (the 215 chimera features, all lookahead_safe). The features that predict oracle entries AND
generalize on UNSEEN = the **oracle's DNA** (the realizable signal). Rigor: held-out (DNA must predict on UNSEEN),
regularized (guard overfit — the feature space is large), causal-only (tradeable), permutation/importance with a
shuffled-label control (a feature that "predicts" shuffled oracle-entries is noise).

### 3. DIFFUSE THE NOISE (the realizable ceiling)
The fraction of oracle capture that the causal DNA can predict = the **realizable ceiling** (our honest best). The
unpredictable remainder is perfect-foresight luck — unattainable. Report it: "the oracle makes X%/move; ~Y% of
that is causally predictable → the realistic target is Y, not X." This kills false hope and false despair.

### 4. BUILD + VALIDATE THE PROXY (capture the DNA)
Build a causal strategy that enters on the DNA signal. KPI = **CAPTURE RATE = realized / oracle-max** (the
project's L2 capture KPI — cost-free, capital-free). Validate against the FULL battery: regime-matched random-
entry firewall, positive_control (two-sided soundness), block-bootstrap p05, jackknife, multi-seed, walk-forward
UNSEEN split, beta-honest. Iterate to raise capture rate toward the realizable ceiling.

## How candidate ideas plug in (mine, the user's, the literature's)
Any trigger hypothesis (liquidation-cascade, momentum-accel, breakout, orderflow, a user idea, a new idea I
generate) is no longer tested in isolation — it is checked against the oracle's DNA: *does this condition actually
coincide with the oracle's entries, on held-out?* The oracle is the ground truth that ranks every avenue
empirically. Generation stays broad; the oracle + the battery do the narrowing.

## What this does NOT change
The north star: robust, held-out, after-cost ROI/per-trade targets (1d/3d/7d + 2-5%/move). The anti-overfit
discipline. The "setup across a multi-candle move" unit. The honesty (a low realizable ceiling is a valid, valuable
finding — it tells us the market's actual offer).
