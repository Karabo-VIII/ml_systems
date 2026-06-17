# The General Problem-Solving Harness — can the engine solve an arbitrary unbounded problem?

> 2026-06-09. Written after the user corrected a scope-narrowing: the "engine" is **not** AlphaZero self-play; that is
> ONE branch. The real question — *"if I hand you an unbounded problem (a market, or DeepMind-frontier science like
> protein folding), can the engine handle it?"* — demands a **multi-method harness with a router**, not one algorithm.
> Grounded in an inspection of `src/wm` (the WM *harness*, not "good world models" — those we don't have yet).

## The correction (what I got wrong)

I framed "engine-agnostic" as the AlphaZero self-play pipeline + a `GameAdapter`. But:
- **chess / Go / Atari = the SOLVED category** ("what DeepMind already did") — self-play RL. Table stakes, not the goal.
- **AlphaFold (protein folding) = the FRONTIER category** — and it uses a COMPLETELY DIFFERENT method: supervised,
  geometry-aware deep learning on structural data. **No self-play, no MCTS, no game.** Self-play can't reach it.
- **Markets (crypto, any market) = forecasting / sequential decision** — the `src/wm` world-model branch.

So "the engine" must be a **harness that routes a problem to the right METHOD**, learns, and validates — not a single
solver. DeepMind isn't one algorithm; it's an apparatus that picks the method per problem. That's the right altitude.

## The harness = 5 steps (the engine, properly scoped)

```
GIVEN an unbounded problem:
 1. REPRESENT   -> a ProblemAdapter: observation/state, action-or-target, objective, data interface, constraints
 2. ROUTE       -> pick the METHOD by problem properties (the router below)
 3. LEARN       -> the architecture zoo + training loop for that method
 4. VALIDATE    -> the rigorous robustness/anti-self-deception spine (DOMAIN-GENERAL — our real edge)
 5. HONEST CEILING -> report the best model + its VALIDATED performance + the gap. Never false victory.
```

### Step 2 — the METHOD ROUTER (the branch I'd omitted)

| Problem shape | Method | DeepMind exemplar | OURS today |
|---|---|---|---|
| sequential **decision**, exact simulator, discrete | **self-play RL + search** (AlphaZero) | AlphaGo/Zero, AlphaStar, AlphaTensor, AlphaDev | `projects/chess_zero` (proven on toy games) |
| sequential **decision**, NO simulator / pixels | **model-based RL** (MuZero/DreamerV3) | MuZero, DreamerV3, plasma control | partial — would use a WM as dynamics |
| **forecasting / world-model** of a system | **supervised/self-sup sequence model** | GraphCast (weather) | **`src/wm` V0–V25 zoo** (modest: V1.x ShIC≈0.033) |
| **structured prediction** (geometry/graph) | **supervised geometric DL** | **AlphaFold**, GNoME, AlphaMissense | **NONE** (no equivariant/geometric family) |
| **discovery / proof** (math, algorithms) | **search + neural** | AlphaProof, AlphaGeometry | partial (the self-play branch is adjacent) |

A market is the *forecasting* + *model-based-decision* rows; a protein is the *structured-prediction* row — a method we
have **zero** of. That is the honest scope gap.

### Step 4 — the SHARED SPINE (this is the actual differentiator)

Across EVERY branch, the same apparatus catches self-deception, and it is **domain-general** (verified by inspection):
anti-fragile training (walk-forward + purge gaps), **shuffled-IC / no-temporal-memorization**, DSR/PBO (deflated
Sharpe, probability-of-backtest-overfit), block-bootstrap p05/jackknife battery (`src/strat/battery.py`), CDAP
invariants, the no-look-ahead rule. DeepMind's frontier wins are ~90% "learn a model + validate honestly vs ground
truth"; **the validation spine is the part that generalizes, and we have a real one.** (Caveat: shuffled-IC's semantics
are time-series-specific — it would *run* on a non-sequential task but need re-interpretation.)

## What we HAVE vs the gap (grounded in `src/wm`)

- **REUSABLE / domain-general (the ~5–8k-line spine):** the architecture zoo (parameterized by `input_dim`, no crypto
  in the forward passes), the anti-fragile training loop (`src/anti_fragile.py`, consumes generic `segments:
  List[dict]` of `features`/`target_*`), TwoHot/symlog loss, DSR/PBO, the battery, CDAP.
- **MARKET-COUPLED (the per-problem adapter, ~400–600 lines):** the chimera feature pipeline + `*_v51_chimera*.parquet`
  loader, the crypto asset-embedding (`ASSET_LIST=["BTCUSDT",...]`), `CryptoPeriodEmbedding` (funding/daily/weekly bar
  cycles), the bear/neutral/bull regime head, `target_return_[1,4,16,64]`, the Binance cost model.
- **MISSING for frontier-class:** a structured-prediction architecture family (equivariant/geometric nets) — AlphaFold
  is in a method bucket we don't implement; plus its data + compute.

## The honest answer to "will the engine handle an arbitrary unbounded problem?"

- **Architecturally, YES it can be made to** — *handle* = ingest (ProblemAdapter) → route (the method router) → learn
  (the zoo) → validate honestly (the spine) → report a model + its validated ceiling. The gap to general is a
  **ProblemAdapter contract + a full method router** (the spine + most of the zoo already carry over). This is a worthy,
  SOTA goal and most of the load-bearing parts exist.
- **But "handle" ≠ "solve to frontier SOTA."** Cracking a *specific* frontier problem (protein folding) to AlphaFold
  level is **data + compute + architecture-family + team bound** — not a laptop weekend, and not what a 4060 does. The
  harness's job is to return an **HONEST model + ceiling** for the problem you hand it, not to pretend it solved it.
  (Same lesson as the chess ceiling and the crypto north-star: the apparatus + honesty is the deliverable; raw SOTA per
  problem is resource-bound.)
- **For markets specifically (the real target):** this is the harness's home turf — the WM forecasting branch + a
  model-based-decision policy over it, A/B'd against predict-then-rule on held-out **compound** return, with the spine
  preventing the model-exploitation trap. The bottleneck is **good models** (the WMs are modest), not the harness shape.

## The path (un-limited, ranked)

1. **Generalize the adapter:** lift the crypto coupling into a `ProblemAdapter` (loader→`segments`, target, objective,
   constraints) so the spine + zoo accept any dataset — markets first, then a non-market time-series as the proof.
2. **Build the full method ROUTER** (not just self-play): forecasting / model-based-decision / supervised — pick by
   problem properties. (The chess `GameAdapter` decision tree is one leaf of this.)
3. **Keep the spine as the keystone** — it's the part that makes any answer trustworthy and is already real.
4. **Frontier science (AlphaFold-class) is a deliberate, separate investment** (a new architecture family + data +
   compute) — name it honestly as out-of-current-reach, not implied-for-free.

> Bottom line: the engine is best understood as a **mini problem-solving apparatus** — represent, route, learn,
> validate, state the honest ceiling. We have a strong validation spine and two market/game-coupled method branches;
> "general" is a ProblemAdapter + a method router away. Frontier-SOTA per problem stays resource-bound — the harness's
> promise is an honest answer, not a miracle.
