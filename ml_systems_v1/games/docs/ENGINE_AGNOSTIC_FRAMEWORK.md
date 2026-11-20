# Engine-Agnostic Self-Play Framework — solving ANY game, not just chess

> The generalization layer. Companions: [BUILD_PLAYBOOK.md](BUILD_PLAYBOOK.md) (the lifecycle),
> [SELFPLAY_SOTA_FEATURES.md](SELFPLAY_SOTA_FEATURES.md) (the feature canon),
> [PRE_RESTART_AUDIT_2026_06_09.md](PRE_RESTART_AUDIT_2026_06_09.md) (the applied audit). **This doc answers: "if you
> give me a DIFFERENT engine to solve, how do I do it?"** — with a decision tree (which method), a code contract (what
> to implement), and a *working non-chess proof*.
>
> Research grounding: the architecture decision tree is from a web-grounded survey (AlphaZero→MuZero→EfficientZero→
> Gumbel→Sampled→Stochastic + OpenSpiel/alpha-zero-general); the adapter contract is distilled from OpenSpiel's
> Game/State API + the alpha-zero-general `Game` interface. Sources at the bottom.

## The core insight (why this is even possible)

The AlphaZero-lineage pipeline — search + net + self-play + replay + training + the **entire** champion-gate /
eval-trust / numerical-safety / invariant-gate / checkpoint / watchdog stack we built in
[`az/train_robust.py`](../az/train_robust.py) — is **engine-agnostic**. Only three things are per-engine:

| Engine-specific (the ~10%) | Engine-agnostic (the ~90%, reuse verbatim) |
|---|---|
| the **rules** (legal moves, transitions, terminal/reward) | the MCTS search kernel (PUCT/UCT, backup, the value-sign convention) |
| the **state/observation encoding** (→ net input planes) | the net *skeleton* (resnet/transformer trunk + policy/value heads) |
| the **action space** (size + index↔move mapping) | the self-play→replay→train loop, champion gate, curriculum, anchor-KL |
| | eval-trust (Wilson CI, forgetting axis, fixed baseline), safety (NaN guard, invariant gate, heartbeat/watchdog), infra (atomic ckpt, resume, pruning) |

So **"solve a new engine" = "implement ~7 methods"** (the `GameAdapter`) + run the decision tree to pick the variant.

---

## 1. The `GameAdapter` contract ([`az/game_adapter.py`](../az/game_adapter.py) — real code, not prose)

The minimal interface a perfect-information game implements. `state` is any object you choose (a tuple, a board, a
FEN); the pipeline treats it opaquely and only calls these:

| Method | Purpose |
|---|---|
| `num_actions` | size of the flat action space → the **policy head width** (chess 4672, TicTacToe 9) |
| `initial_state()` | the start state (for opening diversity, sample from a *set* — `openings.py` generalized) |
| `current_player(state)` | 0/1 — drives the **per-ply value-sign** convention (the I1 lesson, generalized) |
| `legal_actions(state)` | legal action indices in `[0, num_actions)` — the legal-move **mask** |
| `apply(state, action)` | the **next** state — **pure** (never mutates input), so search can branch |
| `is_terminal(state)` | game over? |
| `returns(state)` | terminal value in `[-1,1]` from **player-0's** perspective (the pipeline negates per ply) |
| `encode(state)` *(optional)* | → net input planes — only the **neural** pipeline needs it (UCT does not) |
| `symmetries(state, π)` *(optional)* | data-aug under board symmetries (Go = 8-fold; **chess = none** — castling/EP break it, the C11 lesson) |

**Imperfect-info / stochastic / continuous engines add a little** (see the decision tree): imperfect-info →
`information_state()` (public+private factorization); stochastic → an **afterstate** + chance-node distribution;
continuous actions → a **sampleable** policy instead of `legal_actions` enumeration.

---

## 2. The architecture decision tree — which method for a new engine

```
Given a NEW engine:

Q1. Is a fast exact simulator (rules engine) available?
    NO  (black-box env, pixels, learned world) ─────────► MuZero family (learn the dynamics)
        ├─ data budget tiny (<=100k steps)?  ─► EfficientZero V2  (discrete OR continuous, data-scarce)
        └─ else                               ─► MuZero  (muzero-general for single-GPU)
    YES (rules known: chess, Go, TicTacToe) ────────────► AlphaZero family (no dynamics learning)

Q2. Action space shape?
    CONTINUOUS (robot joints, trade sizes) ─► Sampled MuZero  /  EfficientZero V2
    HUGE DISCRETE (>~10k legal moves)      ─► Gumbel AlphaZero/MuZero (Sequential Halving on top-m)
    SMALL-MEDIUM DISCRETE (chess ~35)      ─► standard AlphaZero

Q3. Transitions stochastic (dice, cards, market noise)?  YES ─► Stochastic MuZero (afterstate factorization)

Q4. Simulation budget tiny (<=32 sims/move, real-time)?  YES ─► wrap in Gumbel (policy-improvement guarantee at low n)

Q5. Imperfect information (hidden cards/units)?          YES ─► Student of Games (sound, CFR+search) or AlphaZe** (quick belief-state baseline)
                                                              (naive AZ here CONVERGES TO AN EXPLOITABLE policy — don't)

Q6. Benchmarking across many game types?                YES ─► wrap everything in OpenSpiel's Game API
```

**Worked examples:** chess → AlphaZero (+Gumbel if inference is tight) · Atari/pixels, scarce data → EfficientZero V2 ·
robot arm (continuous) → Sampled MuZero · backgammon (dice) → Stochastic MuZero · poker (hidden) → Student of Games ·
TicTacToe/Connect-4 → AlphaZero (our `game_adapter.py` proof).

### Method comparison (condensed)

| Method | Selected when | Mechanism | Tier (single-GPU) |
|---|---|---|---|
| **AlphaZero** | rules known, perfect-info, discrete, deterministic | MCTS+PUCT over a policy/value net, self-play | MUST |
| **MuZero** | no simulator / pixels | learns representation+dynamics+reward, plans in latent | MUST (community impl) |
| **EfficientZero (V1/V2)** | low data; V2 adds continuous | + self-supervised consistency, value-prefix, aug | MUST (data-scarce) |
| **Gumbel AZ/MuZero** | low sim budget OR huge action space | Gumbel-top-k + Sequential Halving at root | MUST (default when sims scarce) |
| **Sampled MuZero** | continuous / huge-discrete actions | sample a subset of actions per node | MUST (continuous) |
| **Stochastic MuZero** | stochastic transitions | afterstate (deterministic) → chance node (stochastic) | situational |
| **Student of Games / ReBeL** | imperfect information | guided self-play + CFR/subgame solving | datacenter/research |
| **OpenSpiel / alpha-zero-general** | framework/harness | the Game API + algorithm library / a minimal scaffold | MUST (as library) |

---

## 3. Beyond board games — the breadth (so "an engine" isn't only board games)

The same recipe, with the right variant, reaches far past chess — each is "a different engine":

| System | The "engine" | What it shows |
|---|---|---|
| **MuZero** | Atari (pixels, no rules) | learn the dynamics → plan without a simulator |
| **DreamerV3** | 150+ tasks incl. Minecraft (collect a diamond from scratch) | ONE world-model agent, fixed hyperparams across domains |
| **Sampled MuZero** | continuous control (DM Control) | search over a *sampled* continuous action space |
| **AlphaTensor / AlphaDev** | matrix-mult / sorting+hashing as a single-player game | *algorithm discovery* framed as a game the same search solves |
| **AlphaProof / AlphaGeometry** | math theorem-proving as search | formal reasoning as a "game" with a search+net loop |
| **ReBeL / Student of Games** | poker, Scotland Yard (hidden info) | self-play + CFR generalizes the recipe to imperfect info |

The takeaway for *generalizing*: **the search/train/eval/safety machinery is the reusable engine; the per-game work
is the adapter + (if no simulator) a learned dynamics model.** That's exactly the split the `GameAdapter` encodes.

---

## 4. The recipe — add a NEW engine in 6 steps

1. **Run the decision tree** (§2) → pick AlphaZero / MuZero / Gumbel / Sampled / Stochastic / imperfect-info.
2. **Implement the `GameAdapter`** (§1) — the ~7 methods. (For no-simulator engines, also a learned dynamics head;
   for imperfect-info, the information-state factorization.)
3. **Plug into the generic search.** [`game_adapter.py`](../az/game_adapter.py)'s `uct_search` runs over *any*
   adapter today (proof below); the neural pipeline swaps the random rollout for the net's value + policy prior
   (PUCT) — the adapter calls are identical.
4. **Bootstrap → refine → gate** (Playbook Phase 5). If a strong baseline exists (a classical engine, a solver),
   imitation-bootstrap, then dual-refine with `mix` + anchor-KL behind the champion gate.
5. **Stand up the trust + safety stack** (reuse verbatim): the invariant gate (re-point its checks at the new
   adapter's encode/terminal), Wilson-CI eval + a fixed external baseline + the forgetting axis, NaN-guard/grad-clip,
   heartbeat+watchdog, atomic/pruned checkpoints.
6. **RWYB + run the Lessons catalog** (Playbook appendix) — opening diversity, no-bloat checkpoints, eval-noise CIs,
   etc. They're engine-agnostic; they'll bite the new engine too.

---

## 5. Proof: a non-chess engine solved through the same pipeline (RWYB)

[`az/game_adapter.py`](../az/game_adapter.py) ships the contract + a generic UCT search + a **TicTacToe** adapter +
a `__main__`/[`_test_game_adapter.py`](../az/_test_game_adapter.py) smoke. The generic search (only ever calling the
adapter's methods — zero chess-specific code) plays TicTacToe vs random:

```
[game_adapter] GENERIC UCT over the 'tictactoe' adapter vs random, 40 games (300 sims): UCT W37 D3 L0
[game_adapter] PROOF PASS: generic search never loses -> the engine-agnostic pipeline works
```

**L0 — never loses** (optimal TicTacToe never loses). The contract is sufficient: a new engine is plug-in. The next
adapter (Connect-4, then a neural pipeline over it) follows the same 6 steps.

---

## 6. Unbounded problems — time-series & crypto (the WM *is* the learned dynamics)

Board games are the **bounded** case (exact simulator, terminal, 2-player zero-sum). The project's real targets —
**time-series and the crypto book** — are the **unbounded** case, and this framework reaches them through the *same*
decision tree, routed to the **model-based branch**. The hard part is already built: **the WM (`src/wm/*`) is the
learned dynamics model that branch requires.** This is the `DecisionProblemAdapter` contract in
[`az/game_adapter.py`](../az/game_adapter.py).

### Crypto as a decision problem (the honest framing)

| Game concept | Crypto instance |
|---|---|
| state / observation | **partially observed**: chimera features (past-only — the look-ahead lesson is load-bearing) |
| action | **continuous** position in `[-1,1]` under LO+spot+lev=1 (or discrete `{flat, long, …}`) |
| reward | per-step P&L net of cost → **robust held-out COMPOUND return** (per-bar IC is *banned* as the objective) |
| dynamics | **UNKNOWN** — no market simulator → **the WM** (learned) |
| horizon | (near-)infinite → discounted / episodic |
| players | **single agent** vs a stochastic environment (not 2-player zero-sum) |

→ no-simulator + stochastic + continuous + partial-obs ⇒ the decision tree (§2) routes to **MuZero / Sampled MuZero /
DreamerV3** (model-based RL), with the WM as the dynamics function.

### Two "different angles" vs the current predict-then-rule

The current WM use: **WM predicts → a hand-coded rule sizes the position.** The new angles instead *learn the policy*
by using the WM as a model:
- **A. MuZero-style — plan over the WM.** MCTS/planning over the WM's latent dynamics to choose the trade action;
  value/policy heads trained on the planned targets. Search = imagining market trajectories.
- **B. DreamerV3-style — imagine-train.** Train an actor-critic *purely in the WM's latent "imagination"* (rollouts),
  then act. DreamerV3 is the "one algorithm, many domains" generalist — the strongest fit for *unbounded problems
  generally* (time-series beyond crypto).

Both optimize the **actual objective (compound return) end-to-end** and can discover policies a hand-coded rule
wouldn't — that's the "different angle" you're after.

### The central risk (why the robustness stack is load-bearing here, not optional)

A policy that plans/imagines over an **imperfect** WM learns to **exploit the WM's errors** — superb in imagination,
worthless live. This is the classic model-based-RL failure and *exactly* the project's "plausible-but-wrong" trap. So
the entire eval-trust/safety stack is **mandatory**: robust held-out compound (10/10 seeds, block-bootstrap p05>0,
maxDD<30%), the no-look-ahead invariant, firewall + PBO/CSCV, the forgetting/CI gates. The framework's value here is
as much the **scaffold-that-catches-self-deception** as the agent itself.

### Complementarity, not replacement

This does **not** discard the WMs or the predict-then-rule work — it is a **second consumer of the same WM**.
Predict-then-rule and plan/imagine-over-WM become two policies you **A/B on one held-out compound metric**; keep both,
let the trustworthy eval decide. Any sequential-decision time-series problem maps the same way (obs = past-only
features, action = the decision, reward = the objective, dynamics = a learned model); DreamerV3 is the generalist
default when the domain is unfamiliar.

> **Status (honest):** the contract + the routing are in place; the chess `GameAdapter` is the *proven* instance
> (§5). A crypto/time-series `DecisionProblemAdapter` + the plan-over-WM / imagine-train pipeline is the **next build**
> — and the WM you already have is its keystone.

---

## Sources

- [AlphaZero 1712.01815](https://arxiv.org/abs/1712.01815) · [MuZero 1911.08265](https://arxiv.org/abs/1911.08265) ·
  [EfficientZero 2111.00210](https://arxiv.org/abs/2111.00210) · [EfficientZero V2 2403.00564](https://arxiv.org/abs/2403.00564)
- [Gumbel MuZero/AZ (ICLR 2022)](https://openreview.net/forum?id=bERaNdoegnO) ·
  [Sampled MuZero 2104.06303](https://arxiv.org/abs/2104.06303) ·
  [Stochastic MuZero (ICLR 2022)](https://openreview.net/forum?id=X6D9bAHhBQ1)
- [Student of Games 2112.03178](https://arxiv.org/abs/2112.03178) · [ReBeL 2007.13544](https://arxiv.org/abs/2007.13544)
- [DreamerV3 2301.04104](https://arxiv.org/abs/2301.04104) · AlphaTensor (Nature 2022) · AlphaDev (Nature 2023)
- [OpenSpiel 1908.09453](https://arxiv.org/abs/1908.09453) ·
  [alpha-zero-general (GitHub)](https://github.com/suragnair/alpha-zero-general) ·
  [muzero-general (GitHub)](https://github.com/werner-duvaud/muzero-general)
