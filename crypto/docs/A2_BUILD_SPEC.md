# A2 BUILD SPEC -- raw-data self-evolving agent (Decision-Transformer inner + evolutionary outer)

> Status: PLANNING / SPEC ONLY. **DO NOT BUILD.** This is the code-level plan for
> the harder, higher-ceiling agent track (the one that sidesteps forecaster-GIGO).
> It does not authorize a build; it does not assume Gate 0 / Gate A passed. The
> A2 keystone gate (the shuffled-market policy-overfit control) is ALREADY BUILT
> (`src/agents/_shared/shuffled_market_control.py`); everything else here is plan.
>
> Provenance: docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S4 (the A2 build) + S2
> (the 2-ceiling gating) + S1.6 (the A2 gate) + the empty scaffold at
> `src/agents/a2_raw_data/` + the robustness apparatus it must reuse
> (`src/strat/{battery,pbo_cscv,firewall,positive_control}.py`,
> `src/wealth_bot/{harness,leak_probe,framework/walk_forward}.py`).
> Mirrors the structure of docs/A1_BUILD_SPEC.md. ASCII only (Windows cp1252).

---

## 0. ONE-LINE FRAME

Learn a trading POLICY **directly from raw price/bars/TIs** -- NO frozen forecaster
in the loop, so NO forecaster-GIGO (Ceiling 1 is sidestepped). The inner learner is
a **Decision Transformer / offline-RL** model (return-to-go conditioning -- it never
bootstraps a value, so it never extrapolates into a blowup), wrapped in an
**evolutionary / population OUTER loop** that **selects survivors on HELD-OUT
compound** -- selection-on-held-out is the built-in overfit firewall (it maps to the
chess champion-gate). KPI = held-out **compound** on the policy's realized trade-return
stream. The episode is a **SETUP-to-EXIT move** (multi-candle), reward = realized
compound net of cost; **NO per-bar IC shaping** (banned).

A2 sidesteps Ceiling-1/GIGO but is **NOT** rescued from Ceiling-2: it removes the
forecaster amplifier, it cannot manufacture signal that is not in the bars. The
**first deliverable is a VERDICT, not a bot** -- does the policy beat the
cost-matched random-entry firewall on UNSEEN at the tested resolution? The most
likely first result at 4h is **NO**, and that null bounds Ceiling 2 from a new
direction, cheaply (the gates already exist).

Honest prior: **A2 is STRICTLY HARDER than A1 and HIGHER-CEILING.** Treat
"**A2 finds alpha**" as **[PROJECTED]**, never measured.

---

## PART A -- BASELINE (what exists today; cite file:line)

These are the load-bearing facts the spec builds on. Verified this session.
Claim tags: **[MEASURED]** verified in tree this session; **[APPARATUS]** existing
harness/gate this reuses; **[PROJECTED]** a design hypothesis, unproven.

### A.1 The A2 tree is an EMPTY SCAFFOLD [MEASURED]
- `src/agents/a2_raw_data/` contains ONLY `README.md` + `__init__.py`. **No agent
  code is moved or written.** The README states the class contract: ingests raw
  price/bars/TIs directly, learns rep+policy end-to-end, is its own implicit world
  model, does NOT consume a `ForecastBundle`; KPI = held-out compound; gate =
  A1's gate MINUS the source-forecaster clause PLUS the shuffled-market control;
  GIGO exposure on the WM axis = NONE.
- `__init__.py` records the rule: any module added here declares
  `__class_tag__ = "A2"` (CDAP `agent_class_declared`).

### A.2 The A2 KEYSTONE GATE is ALREADY BUILT + two-sided-validated [MEASURED]
- `src/agents/_shared/shuffled_market_control.py` -- the policy-side ShIC analog,
  the single most important A2-specific gate (the doc S4.2 H3 "does NOT yet exist
  for policies" item, now built).
- It operates on an **abstract trade-return stream**: a callable
  `policy(returns: np.ndarray) -> per-trade NET return stream` -- agnostic to whether
  the policy is a DT, PPO, SAC, or an evolutionary champion. So it is ready for the
  A2 build with zero rework.
- **The verdict is driven ONLY by PREDICTABILITY-DESTROYING surrogates** (`perm` =
  full index permute, `block` = block-shuffle) -- they scramble ALL temporal
  predictability (linear AND nonlinear) while preserving the marginal EXACTLY
  (`shuffled_market_control(...)`, L273; `verdict_kinds=("perm","block")`, L277).
- **`phase`/`iaaft` are SECONDARY mechanism diagnostics, NOT verdict drivers**
  (finding #9, L335-341): they PRESERVE linear autocorrelation, so an
  autocorr-exploiting GENUINE policy would survive them and be FALSE-FLAGGED
  OVERFIT. A hard `ValueError` fires if a caller tries to put them in `verdict_kinds`.
- Two-sided validated (`run_two_sided_demo`, L546): POSITIVE-A (nonlinear
  setup->payoff) -> GENUINE, POSITIVE-B (linear autocorr, the finding-#9 case) ->
  GENUINE via perm/block while iaaft would have false-flagged it, NEGATIVE
  (beta/marginal harvester) -> OVERFIT. RWYB:
  `python src/agents/_shared/shuffled_market_control.py` (exit 0 == all hold).
- Cost honesty baked in: `TAKER_COST_RT = 0.0024` charged INSIDE each trade (L92);
  no look-ahead (re-scores an already-produced held-out stream, L64).
- **`__class_tag__ = None` BY DESIGN** (L90) -- pure shared apparatus, not an agent;
  a tag here would be a category error. The `agent_class_declared` invariant scopes
  tags to agent-logic entry points (`a2_raw_data/*_agent.py`), not `_shared/` utils.

### A.3 The robustness battery exists and is policy-agnostic-ready [APPARATUS]
- `src/strat/battery.py` -- Lens A (strict), Lens B (pragmatic money-in-bank),
  Lens C (temporal barometer). Entry: `evaluate(unseen_returns, comps, ...)` (L102).
  Operates on a returns array -> directly consumable by a policy's trade-return stream.
  Gates: all-4-windows-positive, n_eff floor, jackknife jk2/jk3>0, bootstrap p05>0,
  maxDD<30% (loosest ceiling; the project binding floor is 20%, applied downstream).
- `src/strat/pbo_cscv.py` -- PBO via CSCV (Bailey/Lopez de Prado). **Ship rule
  PBO < 0.10.** Answers the orthogonal-to-DSR question: "does our SELECTION PROCESS
  produce OOS under-performers?" -- exactly the risk of an evolutionary/population
  search at scale. `__contract__` present; two-sided self-test
  (`python -m strat.pbo_cscv --selftest`).
- `src/strat/firewall.py` -- LD-4 the cost-matched random-ENTRY null (the PRIMARY
  gate). A candidate's per-window compound must beat a null of the SAME trade count
  entered at RANDOM bars, held for durations from the candidate's OWN holding
  distribution, at the SAME cost. If it does not beat random entries on held-out ->
  BETA-IN-DISGUISE. **This is the firewall the A2 first-deliverable VERDICT is
  measured against.**
- `src/strat/positive_control.py` -- the STATISTICAL-POWER half: a synthetic series
  with a genuine past-only timing edge must SHIP through the full chain (proves the
  gate ACCEPTS a real edge, not just rejects ghosts -- the two-sided-soundness rule).

### A.4 The leak + seed + walk-forward apparatus exists [APPARATUS]
- `src/wealth_bot/harness.py` -- the canonical simulator: taker `cost_rt = 0.0024`,
  MtM-no-double-count (`net = exit_p/entry_p - 1 - cost_rt`), G1 gate =
  compound_pct>0 in ALL FOUR windows. **Reuse; do NOT hand-roll a sim.**
- `src/wealth_bot/leak_probe.py` -- LD-2 shift-sensitivity look-ahead probe;
  **the cadence-robust verdict is `relative_leak_test(candidate_harness,
  reference_harness, ...)` (L168)** -- the ABSOLUTE pp verdict over-triggers on
  coarse bars (a genuine daily WMA swung 33pp from a 1-bar shift = false positive),
  so use the RELATIVE test against a same-cadence past-only baseline.
- `src/wealth_bot/framework/walk_forward.py` -- N-seed audit + bootstrap CIs.
  **`PER_SEED_OOS_GATE_PCT = 70.0` (L151)** -- >=70% of seeds must be OOS-positive.
  This module carries the canonical A2 cautionary tale verbatim in its docstring:
  the single-seed +44%/+40% LSTM/DQN claims debunked to median -7%/-34% at 10-seed
  audit -- **the project's worst over-fit incident, recorded before A2 was named.**

### A.5 The reference inner/outer machinery exists to PORT (not rebuild) [MEASURED]
- `projects/chess_zero/az/train_robust.py` -- the **champion-gate** pattern (the
  monotonic promotion gate the MEMORY lesson calls "what makes experimenting safe").
  This is the template for the A2 OUTER evolutionary loop's select-on-held-out.
- `scripts/autonomy/skill_library.py` -- [APPARATUS] the validated-sub-policy /
  curriculum store for the SECONDARY skill-library mechanism (only after the
  population loop is proven).
- There is **NO existing offline-RL / Decision-Transformer inner learner** in tree
  for crypto. That is the net-new build. The half-built A1 in `src/agent(s)/` is
  entirely WM-coupled and is NOT reusable for A2 (re-using it would re-acquire
  Ceiling 1).

---

## PART B -- THE BUILD SPEC

### B.1 The recommended class (doc S4.1) -- and what is EXCLUDED

| Ranked | Role | Why |
|---|---|---|
| **(D) Offline RL / Decision Transformer** | **the SUBSTRATE / inner learner** | The market has no exact simulator and you cannot re-run 2021. Offline RL is designed for "learn from a fixed dataset you cannot extend." DT (return-to-go conditioning) makes policy learning **supervised sequence modeling** -- the regime our infra validates -- and **never bootstraps a value, so no extrapolation blowup** (the central offline-RL pathology). "Self-evolving" maps onto re-conditioning on a higher return-to-go. |
| **(C) Evolutionary / population** | **the OUTER loop** (NOT the inner learner) | Population + **selection-on-held-out** is the cleanest "self-evolving" mechanism AND a built-in overfit guard. Maps to the chess `train_robust.py` champion-gate. Evolves hyperparameters / reward-shaping / diversity, with DT as the gradient inner learner. |
| **(A) Recurrent PPO / SAC** | **BASELINE TO BEAT, NOT the ship candidate** | Recurrent is mandatory for the multi-bar MOVE, but: one history => memorizes the path on repeated passes; market exploration is *adverse* (random trades pay real cost, teach nothing). **It produced the project's worst over-fit incident** (A.4: +44%/+40% -> median -7%/-34% at 10-seed audit). Keep it ONLY as the number A2 must beat. |
| **(B) Model-based-from-raw** | **EXCLUDED by definition** | Learning dynamics from raw and planning over them **re-acquires Ceiling 1** -- it is A1 in disguise. Out of A2 by definition. |

**Decision: DT inner + evolutionary outer. Recurrent-PPO is the baseline-to-beat.
Model-based-from-raw is forbidden.**

### B.2 The hard problems + concrete mitigations (doc S4.2)

**H1 -- Representation learning from low-SNR raw data.**
End-to-end from raw will fit the **backtest path**, not market structure.
- **Mitigation (PRIMARY): warm-start the encoder from one of the 7 anchored
  forecasters** (V1.1, V12, V3, V4, V6, V8, V13). Import the genuinely-LEARNT
  REPRESENTATION (the `feat = cat([h_seq, z_post])` encoder weights) **WITHOUT
  consuming its PREDICTION** -- this is the GIGO sidestep: we take the encoder that
  the ShIC>0 anti-memorization anchor certified as having-learnt-structure, and we
  do NOT take its return head. Encoder frozen or low-LR; policy head fresh.
- **Mitigation (SECONDARY): self-supervised aux losses** (contrastive / recon) as an
  information-bottleneck regularizer -- the same logic that produces ShIC>0 in F.
- NOTE: warm-starting from F is a REPRESENTATION transfer, not a prediction
  consumption, so it does NOT make A2 an A1 and does NOT trip
  `no_predicted_return_as_realized_reward`. Document this carefully in the module
  header so the next instance does not mistake it for a GIGO path.

**H2 -- Credit assignment over multi-bar MOVES with sparse compound reward.**
- **Mitigation: DT return-to-go** is a direct INPUT, not a noisy bootstrap -- DT
  shines exactly here; PPO struggles (GAE with a market value-function is noisy).
- **Episode = SETUP-to-EXIT** (the founding-framing unit of trading). Reward =
  realized compound net of cost. **NO per-bar IC shaping** (banned -- it is the
  per-candle lens the whole project rejected).

**H3 -- THE OVERFITTING TRAP = the RL analog of ShIC=0, and it is INVISIBLE.**
A memorizing policy's ONLY symptom is a profitable backtest curve --
indistinguishable by inspection from a real edge. **The detection apparatus, IN
PRIORITY ORDER (this ordering is load-bearing):**
1. **The shuffled-market control [BUILT: `src/agents/_shared/shuffled_market_control.py`].**
   Re-score the SAME policy on surrogate markets whose marginal is preserved but
   temporal structure destroyed (perm + block, the predictability-destroying
   primaries). Genuine policy -> ~0 on the surrogate; memorizer -> still "profits"
   -> that residual IS the policy ShIC=0 signature. **This is the single most
   important A2-specific gate and it already exists, two-sided validated.** Run it
   FIRST -- it is necessary-not-sufficient (it refutes path-memorization; it does
   NOT by itself prove OOS generalization), so compose it with 2 and 3.
2. **PBO via CSCV [`src/strat/pbo_cscv.py`], ship < 0.10.** Run on the policy
   POPULATION (the evolutionary candidates). The in-sample-best of 10^3-10^5
   candidates is BY CONSTRUCTION the most path-fit -- PBO answers whether the
   SELECTION PROCESS itself produces OOS under-performers. This is the gate the
   evolutionary OUTER loop specifically requires (a population search without PBO is
   the canonical selection-bias trap).
3. **N-seed + per-seed-OOS gate [`src/wealth_bot/framework/walk_forward.py`,
   `PER_SEED_OOS_GATE_PCT=70`].** LIVE deploys ONE seed; RL policies are MORE
   seed-sensitive than supervised (env interaction amplifies variance), so this gate
   is MORE binding for A2 than for F. >=70% of seeds OOS-positive.

**H4 -- Non-stationarity / regime shift.**
- **Mitigation:** walk-forward rolling re-fit (**50/20/20/10 + 400-bar purge** --
  the project split invariant), **per-regime evaluation** (trending /
  mean-reverting / high-vol separately, not pooled), **`worst_month_pct > -10%`**
  (battery Lens C). **Report per-regime compound, NEVER just aggregate** -- an
  aggregate +X% that is +40% bull / -30% bear is a regime BET, not an edge.

**H5 -- The simulator becomes the ADVERSARY.**
Because `reward()` is now INSIDE the training loop, any leak / MtM-double-count /
optimistic-fill is **actively found and amplified** -- a policy HUNTS a leak; a
forecaster only passively benefits.
- **Mitigations [all APPARATUS]:** `leak_probe.relative_leak_test` (the
  cadence-robust verdict -- the absolute/shift verdicts FAILED on coarse bars);
  the MtM-no-double-count invariant in `harness._simulate`; a **point-in-time
  universe** (no survivorship -- the policy must not be allowed to see a coin that
  only exists because it survived); **calibrated cost `p_fill 0.21-0.40`** (NOT the
  0.80 default -- a policy trained on optimistic fills learns trades that won't fill
  live). The simulator A2 trains against MUST be the same hardened `harness.py` the
  battery audits, not a fast surrogate sim.

### B.3 What "SELF-EVOLVING" concretely means (doc S4.3) -- NOT unconstrained online adaptation

1. **Outer evolutionary / population loop (PRIMARY).** Each generation: train the
   inner DT, evaluate on HELD-OUT, SELECT survivors by val-compound + diversity,
   mutate (hyperparameters / reward-shaping / architecture). **Selection on
   held-out = the built-in overfit firewall.** Maps to the chess champion-gate (the
   MEMORY lesson: "the monotonic promotion gate is what makes experimenting safe").
2. **Skill-library / curriculum (SECONDARY)** [APPARATUS
   `scripts/autonomy/skill_library.py`]. Accumulate VALIDATED sub-policies; easy
   regimes -> hard. **Only after the population loop is proven.**
3. **Online / continual learning (LAST, BOUNDED).** ONLY as walk-forward
   fine-tuning behind a gate: re-fit on the newest CLOSED window, but the update
   **must re-pass N-seed + PBO + shuffled-market against the prior champion before
   promotion.** **Unconstrained continual learning = recency-overfit +
   catastrophic forgetting + the self-approval bypass** (the memory lesson: a
   sub-agent self-approving a gated step). Treat every online update as a potential
   data leak with a logged pre/post snapshot; never let the policy self-approve its
   own promotion.

### B.4 Exploration in a market (doc S4.4) -- the A2-specific trap

**Action-level exploration is HARMFUL in a low-SNR market.** A random exploratory
trade pays real spread+fee+slippage and the reward is dominated by noise -- it costs
real money and teaches nothing.
- **The correct action-level exploration budget is near-ZERO.**
- **Spend the exploration budget at the POPULATION / ARCHITECTURE level instead**
  (mutate hyperparameters / reward-shaping / encoder variants across the population,
  not random trades inside an episode).
- **Curiosity / intrinsic-reward is a TRAP** -- a non-stationary market is endlessly
  novel, so the agent chases noise forever. Do not add it.

### B.5 The robustness apparatus applied to the POLICY's trade-return stream

The whole battery is repointed from a forecaster's prediction at the bar to the
**policy's realized per-trade / per-setup NET return stream** -- the load-bearing
"repoint" effort. The exact gates, by file:

| Gate | File | Rule |
|---|---|---|
| Shuffled-market overfit control | `src/agents/_shared/shuffled_market_control.py` | verdict GENUINE (overfit_fraction <= 0.20, beats perm/block p95); OVERFIT (overfit_fraction >= 0.35) => KILL |
| Random-entry firewall (PRIMARY) | `src/strat/firewall.py` | beat the cost-matched random-entry null on ALL held-out windows, else BETA-IN-DISGUISE |
| Battery Lens A/B/C | `src/strat/battery.py` `evaluate(...)` | all-4-positive, n_eff floor, jk2/jk3>0, p05>0, maxDD<30% (20% downstream) |
| PBO via CSCV | `src/strat/pbo_cscv.py` | **PBO < 0.10** on the candidate population |
| N-seed + per-seed-OOS | `src/wealth_bot/framework/walk_forward.py` | **>=70% seeds OOS-positive** (`PER_SEED_OOS_GATE_PCT`) |
| Leak probe (relative) | `src/wealth_bot/leak_probe.py` | `relative_leak_test` vs a same-cadence past-only reference |
| Positive-control (gate power) | `src/strat/positive_control.py` | confirm the chain SHIPs a known genuine synthetic edge (two-sided soundness) |
| Simulator | `src/wealth_bot/harness.py` | taker 0.0024, MtM-no-double-count, calibrated p_fill 0.21-0.40 |

### B.6 The A2 GATE (vs A1) -- doc S1.6

> **A2 gate == A1's gate MINUS the source-forecaster clause PLUS the
> shuffled-market overfit control.**

Concretely, relative to A1's gate (A1_BUILD_SPEC B.5):
- **REMOVE** the `source-F genuine_learning.passed` clause -- there is NO forecaster
  in the A2 loop, so there is no forecaster-quality precondition to assert. (The
  registry write uses `requires_genuine_forecaster=False`.)
- **KEEP** everything else: held-out compound beats the champion AND beats the
  random-entry firewall, >=8/10 seeds positive on UNSEEN, bootstrap p05>0, maxDD<30%,
  per-regime, taker cost.
- **ADD** the shuffled-market overfit control (the policy-side ShIC analog) as a
  mandatory pass before any compound number is trusted.

**The first deliverable is a VERDICT, NOT a bot:** *does the A2 policy beat the
cost-matched random-entry firewall on UNSEEN at the tested resolution?* A clean
"NO at 4h" is a high-value Ceiling-2 bound, not a failure. Ship a bot ONLY if the
full gate passes; tie or marginal => there is no edge to ship.

### B.7 Action space + reward (staged, mirrors A1)

- **v1: discrete {flat, long}** -- 2-state, matches harness long-only `_simulate`,
  zero sizing-overfit surface. Build this first.
- **v2 fast-follow: position buckets {0, 0.25, 0.5, 0.75, 1.0}** -- sizing as a
  legitimate value-add only after discrete wins.
- **continuous [0,1]: ONLY after discrete wins** (biggest silent-overfit surface).
- **Reward:** episode = SETUP-to-EXIT; per-trade realized compound NET of taker
  cost 0.0024 charged INSIDE the trade (never bolted on after -- the +501%->+94%
  MtM/cost history). gamma ~ 1 (episodic compound). **NO per-bar IC reward.**
  Asymmetric loss (a false LONG costs more than a missed winner) pre-registered,
  never tuned on UNSEEN.

---

## PART C -- COMPONENTS TO BUILD (under `src/agents/a2_raw_data/`)

> All NEW; nothing here is reused from the A1 tree. Each agent-logic entry module
> declares `__class_tag__ = "A2"` (CDAP `agent_class_declared`); pure utilities
> under a future `a2_raw_data/_util/` may set `__class_tag__ = None` with the same
> rationale as `shuffled_market_control.py` (A.2).

### C.1 The inner learner -- `decision_transformer_agent.py`
- Decision-Transformer offline-RL policy over raw f41 bars + TIs. Inputs: a context
  window of (state, action, return-to-go) tokens; output: next action.
- **Encoder warm-start hook** (H1): load the `feat` encoder weights from a Gate-A
  forecaster's checkpoint (representation only, NOT the return head); freeze or
  low-LR. A `--no-warm-start` path for the ablation (does warm-start actually help?).
- `__class_tag__ = "A2"`. No `ForecastBundle` import (that would be A1). No decoded
  return used as the realized reward (CDAP).

### C.2 The recurrent-PPO BASELINE -- `recurrent_ppo_baseline.py`
- The number A2 must BEAT, NOT a ship candidate. Recurrent (LSTM/GRU) PPO over the
  same raw inputs. Carries a banner pointing at A.4 (the +44%/-7% incident) so no
  future instance mistakes it for the ship path. `__class_tag__ = "A2"`.

### C.3 The evolutionary OUTER loop -- `evolutionary_outer.py`
- Population of inner learners; each generation trains, evaluates on HELD-OUT,
  selects survivors by **val-compound + diversity**, mutates hyperparameters /
  reward-shaping / encoder-variant. **Select-on-held-out is the firewall.** Port the
  champion-gate pattern from `projects/chess_zero/az/train_robust.py` (monotonic
  promotion only). Wire PBO (`pbo_cscv.py`) on the candidate population every
  generation. `__class_tag__ = "A2"`.

### C.4 The policy-stream gate runner -- `a2_gate.py`
- Chains B.5 in order: shuffled-market control FIRST (kill OVERFIT early) -> firewall
  -> battery Lens A/B/C -> PBO -> N-seed/walk-forward -> relative leak probe. Emits a
  single VERDICT (B.6) + a `layer_kpis` decomposition (entry vs exit vs sizing). This
  is the module that produces the **first deliverable** (the verdict). Reuses every
  gate from `src/strat/` + `src/wealth_bot/` -- builds NO new statistics.

### C.5 The sim coupling -- `a2_environment.py`
- Wraps the hardened `wealth_bot/harness.py` (NOT a fast surrogate sim) as the
  training environment: point-in-time universe, calibrated p_fill 0.21-0.40, taker
  0.0024, MtM-no-double-count. Exposes `(state, action, realized_reward, next_state)`
  tuples for the DT replay buffer. `__class_tag__ = "A2"`.

### C.6 Reuse (do NOT rebuild)
- Overfit control: `src/agents/_shared/shuffled_market_control.py` (BUILT).
- Robustness: `src/strat/{battery.py, pbo_cscv.py, firewall.py, positive_control.py}`.
- Sim / cost / seed / leak: `src/wealth_bot/{harness.py, leak_probe.py,
  framework/walk_forward.py}`.
- Champion-gate + curriculum: `projects/chess_zero/az/train_robust.py`,
  `scripts/autonomy/skill_library.py`.

### C.7 CDAP -- must stay green
- `agent_class_declared`: every `a2_raw_data/*_agent.py` (and outer-loop / env /
  baseline entry modules) declares `__class_tag__ = "A2"`; the registry rejects
  untagged writes.
- `no_predicted_return_as_realized_reward`: A2 has no forecaster, so the realized
  reward traces to `target_return_*` (REAL) trivially -- but the warm-start MUST be
  representation-only (no decoded return entering the reward). Assert this at
  train-time.
- The A2 registry write keys on held-out **compound** into `runs/registry/agents.json`
  with a `class: "A2"` field (NEVER onto the IC-keyed `forecasters.json` -- that
  back-imports the banned IC objective).
- Run `python src/audit/check_invariants.py` after each step (exit 0/1, never 2).

---

## PART D -- EFFORT + HONEST PRIOR

### D.1 Effort estimate (honest; A2 is STRICTLY HARDER than A1)
- **Pre-req (NOT counted here):** Gate 0 (does exploitable structure exist at the
  tested resolution?) -- model-free, can FAIL and STOP the whole program. If Gate 0
  fails (the likely 4h outcome given D17/D44/D45), this spec is shelved and the
  deliverable is the null. A2 does NOT need Gate A (no forecaster), but warm-starting
  the encoder (H1) wants at least one Gate-A-passing F.
- **C.1 DT inner learner (+ warm-start hook + aux losses):** ~3-4 days. The
  representation warm-start + the offline-RL token plumbing is the load-bearing,
  error-prone part.
- **C.2 recurrent-PPO baseline:** ~1-1.5 days (it is the number to beat, build it
  cheaply).
- **C.3 evolutionary outer loop (+ PBO-per-generation + champion-gate port):**
  ~2-3 days.
- **C.5 sim coupling to the hardened harness (point-in-time universe + calibrated
  fills):** ~2 days -- this is where the simulator-as-adversary (H5) risk lives;
  budget for finding a leak the policy hunts.
- **C.4 gate runner + the first-deliverable VERDICT (mostly reuse):** ~1-1.5 days.
- **Tuning + the per-regime + N-seed + shuffled-market re-gate loop:** ~2-3 days.
- **Total: ~11-15 focused days** for the v1 discrete {flat,long} single-resolution
  verdict -- noticeably MORE than A1's ~6-9 days, because A2 builds the agent AND the
  sim-in-the-loop AND repoints the whole battery at a policy stream, with no
  half-built code to inherit.

### D.2 The honest prior
- **A2 is STRICTLY HARDER and HIGHER-CEILING.** Harder: build the agent + the
  simulator-in-the-loop + repoint the entire robustness battery at a policy's
  trade-return stream (the keystone overfit control is the only piece already built).
  Higher-ceiling: **it sidesteps Ceiling-1/GIGO**, and representation-learning-from-raw
  is the single class whose advantage *might* surface structure that hand-built
  features miss -- the most interesting upside test in the taxonomy.
- **But its MOST LIKELY first result at 4h is that it CANNOT beat random-entry on
  UNSEEN.** That is not a failure -- **it bounds Ceiling-2 from a new direction,
  cheaply** (the gates already exist; the verdict is the product). Treat
  "**A2 finds alpha**" as **[PROJECTED]**, never measured.
- **Ceiling 2 caps everything.** A2 removes the GIGO amplifier, NOT the constraint:
  if daily/4h LO crypto genuinely has weak signal (the project's repeated history),
  F / A1 / A2 / A1H all converge to the same low/flat number and the real lever is
  **resolution (HF / microstructure)** -- a DATA decision orthogonal to this
  taxonomy. The 1m+liquidation dataset is already built (D71); D72 located the move
  at 1m for movers but says continuation needs an EXTERNAL signal at AUC>0.58.
- **A2 relocates GIGO into an INVISIBLE policy** -- so the shuffled-market control
  is non-negotiable and runs FIRST. The whole value of A2's first phase is a clean,
  cheap Ceiling-2 verdict, not a shipped bot.
