# A1 BUILD SPEC -- DreamerV3 imagination actor-critic over a Gate-A-passing forecaster

> Status: READY-TO-EXECUTE PREP. **DO NOT BUILD until a forecaster PASSES Gate A**
> (docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4). This spec is the code-level
> plan for the moment that gate goes green. It does not authorize a build.
>
> Provenance: audit of the half-built A1 (src/agents/a1_wm_consuming/) +
> docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S3 (S3.2 portfolio-state
> factorization, S3.3 action/reward, S3.4 guardrails, S3.5 A/B vs B0).
> ASCII only (Windows cp1252).

---

## 0. ONE-LINE FRAME

Wrap a stochastic actor + pessimistic critic around a FROZEN Gate-A forecaster.
Keep the market latent transition **action-free** (we are a price-taker:
LO + spot + lev=1 -- our order does not move BTC, so no WM retrain). The action
conditions only the **reward** and the **portfolio/inventory state**. Reward on
**REALIZED** next-bar returns (REPLAY), the next-bar return drawn from the WM's
**full twohot DISTRIBUTION** as a belief feature -- never from the WM's point
estimate, and the realized reward is **real** returns, never the WM's prediction
(the GIGO firewall). Ship A1 ONLY IF it beats the ~1-parameter predict-then-rule
baseline B0 on UNSEEN compound by a pre-registered margin.

Honest prior: **P(A1 > B0 at daily/4h) is LOW.** A1's defensible role is
calibrated sizing / abstention / exit-timing over a WM that already proved (via
Gate A) there is signal to time -- not a signal manufacturer.

---

## PART A -- AUDIT BASELINE (what exists today; cite file:line)

These are the load-bearing facts the spec builds on. Verified this session.

### A.1 The DreamerV3 agent is a SMOKE-TEST SHELL, not a trained agent
- `src/agents/a1_wm_consuming/dreamer_v3_agent.py` defines `Actor` (L39),
  `Critic` (L69), `lambda_return` (L89), `DreamerV3Agent` (L111) and a
  `smoke_test()` (L216). **There is NO training loop, NO data, NO checkpointing,
  NO env coupling** -- `__main__` calls only `smoke_test()` (L246-247).
- It imports the WM **directly**, NOT via the frozen contract:
  `sys.path.insert(0, .../v16_dreamerv3/v16_training)` then
  `from dreamer_v3 import DreamerV3WorldModel ...` (L34-36). It never touches
  `src/wm/forecast_bundle.py`. **This is the #1 contract gap.**
- WM **is** frozen in this module: `for p in self.wm.parameters(): p.requires_grad
  = False` (L118-119); `smoke_test` asserts `wm_grad == 0` (L240). Good -- but it
  freezes by sub-moduling the WM, not by consuming a detached bundle (so the
  freeze is not the CDAP-enforced structural one).

### A.2 dream_step / imagination IS action-free (CORRECT for a price-taker -- but currently vacuous)
- The agent's imagination is action-CONDITIONED today and that is the DEFECT, not
  a feature: `imagine_with_actor` (L149-170) calls
  `self.wm.rssm.imagine_step(state, action)` (L160) -- it feeds the actor's action
  into the **market latent** transition. The backbone RSSM
  (`backbones/v16_dreamerv3/v16_training/dreamer_v3.py`) `imagine_step(prev_state,
  prev_action)` concatenates the action into the GRU input (L158-166). So the
  current code makes the actor's trade move the market -- WRONG for spot LO.
- The anchored forecaster V1.1 is the opposite and CORRECT:
  `src/wm/v1/v1_1_training/world_model.py` `dream_step(h_prev, z_prev, gru_hidden)`
  (L772) takes **NO action** ("One-step imagination using dream GRU (no
  observation)", L779; samples `z_next` from the prior at L802). Per doc S3.2 this
  is exactly right for a price-taker. **The build must adopt V1.1's action-free
  market latent and move the action OUT of the transition into the reward +
  portfolio state.**

### A.3 The env REPLAY/DREAM seam exists and REPLAY is GIGO-safe
- `src/agents/a1_wm_consuming/environment.py` documents REPLAY (real returns) vs
  DREAM (predicted returns) at L7-13.
- REPLAY rewards on **REAL** returns: `_compute_step_pnl` uses
  `data["target_return_1"][bar_idx]` (L523) and step PnL flows from real returns
  (L625-634). The WM is used only as a **feature source**
  (`_precompute_episode_features`, L331; `encode_sequence` at L362). This is the
  correct GIGO-safe default per doc S3 / S1.3.
- DREAM (predicted-return reward) is the dangerous mode -- it is NOT wired into
  the DreamerV3 agent yet (the agent's `imagine_with_actor` uses
  `self.wm.reward_head`, a predicted reward, but it is never trained against real
  data -- it is pure smoke). The build keeps DREAM gated behind
  `genuine_learning.passed` and entered LAST.

### A.4 The actor sees portfolio state in the ENV path, but NOT in the Dreamer path
- ENV path (PPO): `_build_observation` (L428) writes per-asset position at
  `obs[offset + PER_ASSET_OBS_DIM - 2]` and an unrealized-PnL proxy at
  `obs[offset + PER_ASSET_OBS_DIM - 1]` (L450-451); global cash fraction at
  `obs[-1]` (L454). So the **PPO** policy (`policy.py`) sees position + pnl + cash.
- Dreamer path: `DreamerV3Agent.actor` consumes only `feat = cat([h, z])`
  (feat_dim at L121); it sees **NO portfolio state at all**. The actor cannot know
  if it is already long, how many bars it has held, or its return-so-far. **This
  is the S3.2 gap to close on the Dreamer side.**

### A.5 Reward source today
- ENV `RewardCalculator` (`rewards.py`) computes from **realized** net PnL
  (`net_pnl = pnl - transaction_cost - funding_cost`, L72) -- realized, not
  predicted. Cost subtracted once (no double-count, L69-71). Good.
- Dreamer `actor_critic_loss` (L172) regresses the critic to a `lambda_return`
  (L186) built from `self.wm.reward_head` predictions (L161) -- i.e. **PREDICTED
  reward inside pure imagination**. This is fine ONLY because it never trains;
  the real build must ground the critic target in REALIZED returns (REPLAY) per
  the GIGO firewall.

### A.6 Contract + CDAP status
- `src/wm/forecast_bundle.py` EXISTS, frozen+detached, self-tests (L143).
  `genuine_learning` carries `{shic, held_out_ic, passed}` (L67, L89). **Not yet
  imported by any agent** (grep: agent modules do not reference it).
- CDAP `agent_taxonomy` invariants are LIVE: `check_agent_taxonomy` in
  `src/audit/check_invariants.py` (L770) enforces `forecaster_frozen_in_agents`
  (L784), `no_predicted_return_as_realized_reward` (L805), `agent_class_declared`
  (L831), `v16_v17_not_in_wm` (L854). Spec at `config/_invariants.yaml` L2103+.
  Every agent module already carries `__class_tag__ = "A1"` (e.g.
  dreamer_v3_agent.py:21). **The build must not trip (a) or (b).**

### A.7 Cost + gate surface to reuse (do NOT hand-roll)
- `src/wealth_bot/harness.py`: taker `cost_rt = 0.0024` default (L55, L178, L450;
  F9 fix); `_simulate` charges round-trip on net (`net = exit_p/entry_p - 1 -
  cost_rt`, L662); G1 gate = `compound_pct > 0` in ALL FOUR windows (L246, L800).
- `config.py`: SPOT round-trip = `SPOT_FEE_BPS(10) + SPOT_SLIPPAGE_BPS(2)` per
  side = 12 bps/side = **24 bps round-trip = 0.0024** (L83-90). Matches the
  harness taker. Use this, charged INSIDE imagination on `|delta_position|`.

---

## PART B -- THE BUILD SPEC

### B.1 The S3.2 portfolio-state factorization (the core design)

**Keep the market latent transition action-free. No WM retrain.**

The WM (frozen Gate-A forecaster, e.g. V1.1) rolls `dream_step(h, z, gru_hidden)`
(action-free, V1.1 world_model.py:772) OR is consumed in REPLAY as a feature
extractor. The action NEVER enters the market latent.

**Augment the actor's observation with PORTFOLIO STATE** (3 scalars, per traded
instrument in the v1 single-asset framing):

| field          | meaning                                  | range / encoding             |
|----------------|------------------------------------------|------------------------------|
| `position`     | current position in {0, 1} (v1 discrete) | {0.0, 1.0}                   |
| `bars_in_trade`| bars held since entry (0 if flat)        | `min(bars, CAP)/CAP`, CAP=64 |
| `return_so_far`| compounded net return since entry (0 if flat) | raw log-return, clip [-0.5,0.5] |

Actor observation = `cat([feat, regime_feat, belief_feat, portfolio_state])`
where:
- `feat = cat([h, z])` -- the WM latent (the Dreamer/MuZero state), from the
  bundle.
- `regime_feat = softmax(regime_logits)` -- the cheap robust gate channel (3 dims).
- `belief_feat` -- moments of the decoded twohot **distribution** at the queried
  horizons: `[E[r], Std[r], P(r<0)]` per horizon (NOT just the mean; doc S1.3 +
  FM "return-distribution collapsed to a point").
- `portfolio_state` -- the 3 scalars above.

**Reward (REPLAY, realized):**
```
r_t = action_t * realized_next_bar_return_{t+1} - cost_rt * |action_t - action_{t-1}|
```
- `realized_next_bar_return_{t+1}` = REAL `target_return_1` (env._compute_step_pnl
  path), NEVER the WM's predicted return. (CDAP `no_predicted_return_as_realized_reward`.)
- `cost_rt = 0.0024` (taker), charged on `|delta_position|`, **inside imagination**
  (not bolted on after -- FM4 / the +501%->+94% MtM history).
- The WM's twohot distribution is used ONLY as a belief FEATURE the actor
  observes (for sizing/abstention) and, in the explicitly-flagged DREAM variant,
  as the imagined-reward source -- **sampled from the distribution**, never the
  point estimate.
- Episode = the SETUP (entry to policy-exit, multi-candle). gamma ~ 1 (episodic
  compound), small gamma<1 allowed for critic variance control only.
- Asymmetric loss: a false LONG costs more than a missed winner. Pre-register the
  coefficient (`REWARD_ASYMMETRY` already exists, rewards.py:84-87); NEVER tune on
  UNSEEN.

### B.2 Action space (staged; continuous only after discrete wins)

- **v1: discrete {flat, long}** -- 2-state machine, matches harness long-only
  `_simulate`, zero sizing-overfit surface. Build this first.
- **v2 (fast-follow): position buckets {0, 0.25, 0.5, 0.75, 1.0}** -- sizing as
  the legitimate A1 value-add (size down when predictive variance high).
- **continuous [0,1]: ONLY after discrete wins** (biggest silent-overfit surface;
  it can fit the validation equity curve's exact shape).

### B.3 The S3.4 guardrails (each a concrete component, all ON before trusting a number)

1. **WM ENSEMBLE + disagreement penalty (PRIMARY).** We have 7 anchored
   forecasters (V1.1, V12, V3, V4, V6, V8, V13). Roll imagination through K
   bundles; penalize reward by cross-model disagreement of decoded `E[r]` (std
   across models), OR refuse to act where disagreement > tau. Component:
   `_shared/ensemble_disagreement.py` -> `disagreement(bundles, h) -> [B,T]`.
   Requires >=3 Gate-A-passing F (doc Phase-2 gate); a single-WM A1 is REJECTED.
2. **Value pessimism / LCB critic (MANDATORY).** Ensemble of `Critic` (>=2); act
   on `min` / low-quantile of V. Counters imagined-value overestimation (CQL /
   SAC-N lesson). Component: `EnsembleCritic` wrapping the existing `Critic` (L69).
3. **Short imagination horizon H = 4-8 bars** (~ one setup), NOT 15. Change
   `self.imagination_horizon = 15` (dreamer_v3_agent.py:130) -> a config
   `IMAGINATION_HORIZON = 6` (sweepable 4-8). Tune H DOWN if imagined return >>
   real-data return.
4. **Real-data Dyna grounding + divergence alarm.** Interleave imagination
   batches with REAL (obs, action, realized-reward, next-obs) replay from the env
   REPLAY path. Track `gap = mean(imagined_reward) - mean(realized_reward)` every
   epoch; **widening gap => HALT (do not tune through it)** -- it is the leading
   indicator the actor left the data manifold. Component:
   `_shared/dyna_grounding.py` with a `DivergenceAlarm(threshold, window)`.
5. **Distributional planning.** Use the full twohot distribution (variance for
   sizing, tail for risk), never the mean. (Already structured: bundle exposes
   `return_logits` + `return_bins`; decode to moments in `belief_feat`.)
6. **KL-to-B0 behavior regularization.** Add `+ beta * KL(pi || pi_B0)` to the
   actor loss, where pi_B0 is the predict-then-rule policy (B.5). Caps how weird
   the agent gets; degrades gracefully to B0. Component: a `b0_policy(obs) ->
   action_logits` reference and a KL term in `actor_critic_loss`.
7. **OOD-latent detector.** Flag `(h,z)` far from the training manifold (recon
   error from the WM decoder OR prior-posterior KL spike) and force `action=flat`
   there. Component: `_shared/ood_detector.py` -> `is_ood(h,z) -> bool mask`.

### B.4 GIGO firewall (mechanical, must hold)
- REPLAY is the default; reward target traces to `target_return_*` (realized).
- DREAM (predicted-return reward) is a SECOND, explicitly-flagged variant in a
  `dream_*` module, entered LAST, and only if the source bundle's
  `genuine_learning.passed == True` (asserted at A1 train-time). This is exactly
  what CDAP `no_predicted_return_as_realized_reward` allows (the `dream_*` +
  assert carve-out, _invariants.yaml L2127+).
- The WM is consumed ONLY through `ForecastBundle.from_forecaster(...)` (detached,
  eval). No `sys.path.insert` direct import; no F params in the actor/critic
  optimizer (CDAP `forecaster_frozen_in_agents`).

### B.5 The A/B protocol vs B0 (the ship decision)
- **B0 = predict-then-rule, ~1 free parameter.** Decode the WM return head; enter
  long when `E[r] - cost > threshold`; exit by policy; size 1. Reuse
  `src/strat/wm_value_probe.py` if present, else a 20-line `b0_predict_then_rule`.
- **Discipline:** WM frozen; actor+critic fit on TRAIN+VAL imagination/replay
  ONLY; UNSEEN touched ONCE (50/20/20/10, 400-bar purge).
- **Ship A1 ONLY IF** on UNSEEN it beats B0 on **compound** by the pre-registered
  `wm_promotion_gate.should_promote` margin **AND** >=8/10 seeds positive **AND**
  block-bootstrap p05 > 0 **AND** maxDD < 30%. Gate via
  `src/wealth_bot/harness.py` (G1 = all-4-windows-positive) + `src/strat/battery.py`.
- **Tie or marginal win => ship B0** (simpler, fewer failure modes).
- **Decompose any win** via the harness `layer_kpis` (entry-timing vs exit-timing
  vs sizing/abstention). An A1 that "wins" only by going flat in bear windows is a
  **regime filter, not alpha** -- name it honestly (the exit-axis-NULL lesson).

---

## PART C -- EXACT CODE CHANGES (file:line level)

### C.1 `dreamer_v3_agent.py` -- consume the bundle, augment the actor, fix imagination

1. **Replace the direct WM import (L34-36)** with the contract:
   - DELETE `sys.path.insert(...)` + `from dreamer_v3 import DreamerV3WorldModel...`.
   - ADD `from wm.forecast_bundle import ForecastBundle`.
   - `DreamerV3Agent.__init__` (L114) takes a **frozen bundle provider** (a callable
     `make_bundle(batch) -> ForecastBundle`) and a list of K such providers for the
     ensemble, NOT a raw `wm`. Assert `bundle.is_frozen` and
     `bundle.genuine_learning["passed"]` at construction (REPLAY allows passed
     unknown; DREAM requires True).
2. **Actor input dim (L121-122)** -- change `feat_dim = wm.rssm.hidden_dim +
   wm.rssm.stoch_dim` to `obs_dim = feat_dim + 3 (regime) + 3*len(H_QUERIED)
   (belief moments) + 3 (portfolio_state)`. Construct `Actor(obs_dim, ...)`.
3. **Portfolio-state plumbing (NEW)** -- add a `PortfolioState` carrier
   (position, bars_in_trade, return_so_far) updated each imagined/real step; build
   the actor observation via a new `_actor_obs(feat, regime_logits, return_logits,
   portfolio_state)` helper.
4. **`imagine_with_actor` (L149-170)** -- the LOAD-BEARING fix:
   - REMOVE the action from the market transition: replace
     `state = self.wm.rssm.imagine_step(state, action)` (L160) with the
     **action-free** roll `state = wm_dream_step(state)` (V1.1
     dream_step-style; no action arg).
   - The action now updates only `portfolio_state` and the **reward**:
     `reward_t = action_t * realized_or_sampled_return - cost_rt *
     abs(action_t - prev_action)`.
   - In REPLAY: `realized_return` comes from the env's real `target_return_1`
     stream (B.4); in DREAM: SAMPLE from the twohot distribution (not the mean).
5. **`actor_critic_loss` (L172-208)** -- critic target must trace to REALIZED
   returns (B.4). Add: LCB over the critic ensemble (guardrail 2),
   `+ beta * KL(pi || pi_B0)` (guardrail 6), OOD mask -> flat (guardrail 7),
   ensemble-disagreement penalty on the reward (guardrail 1), Dyna divergence
   alarm hook (guardrail 4).
6. **`self.imagination_horizon = 15` (L130)** -> `IMAGINATION_HORIZON` config,
   default 6 (guardrail 3).
7. **Keep `__class_tag__ = "A1"` (L21)** (CDAP `agent_class_declared`).

### C.2 `environment.py` -- expose the realized-reward + portfolio channels to the Dreamer path

1. The REPLAY real-return path already exists (`_compute_step_pnl` L512, real
   `target_return_1` L523). **Add a public accessor** the Dyna grounding loop can
   pull `(obs_latent, action, realized_reward, next_obs_latent)` tuples from for
   the replay buffer (guardrail 4). No change to the cost/MtM accounting (it is
   already single-charge, L69-72 rewards.py / L662 harness).
2. **Portfolio-state already in the ENV obs** (position L450, pnl L451, cash
   L454). For the Dreamer path, surface `position`, `bars_in_trade`,
   `return_so_far` as an explicit `portfolio_state()` method returning the 3
   scalars in B.1 (bars_in_trade + return_so_far are NEW -- track entry bar +
   compounded return in `step`).
3. Keep DREAM mode (predicted returns) gated -- do not route it into the critic
   target outside a `dream_*` module (CDAP (b)).

### C.3 NEW shared components (under `src/agents/_shared/`)
- `ensemble_disagreement.py` (guardrail 1), `dyna_grounding.py` + `DivergenceAlarm`
  (guardrail 4), `ood_detector.py` (guardrail 7), `b0_predict_then_rule.py` (B.5),
  `bundle_provider.py` (wraps each Gate-A F into `make_bundle`). Each carries
  `__class_tag__` only if it is agent-logic; pure utilities can omit but must not
  trip CDAP (b) (no decoded-return-as-reward outside dream_*).

### C.4 Reuse (do NOT rebuild)
- Reward/cost/gate: `src/wealth_bot/harness.py` (cost_rt 0.0024, G1 gate).
- Robustness: `src/strat/battery.py` (Lens A/B/C), `pbo_cscv.py`, `walk_forward.py`
  (>=8/10 seeds, p05>0, maxDD<30%).
- Contract: `src/wm/forecast_bundle.py`.
- The env REPLAY/DREAM seam: `src/agents/a1_wm_consuming/environment.py`.

### C.5 CDAP -- must stay green
- `forecaster_frozen_in_agents`: no `F.train()`, no F params in the actor/critic
  optimizer, all bundle tensors detached (structural via `from_forecaster`).
- `no_predicted_return_as_realized_reward`: REPLAY critic target = realized
  `target_return_*`; predicted-return reward only inside a `dream_*` module with
  the `genuine_learning.passed` assert.
- `agent_class_declared`: every agent module keeps `__class_tag__ in {A1,A2,A1H}`.
- Run `python src/audit/check_invariants.py` after each step (must be exit 0/1,
  never 2).

---

## PART D -- EFFORT + HONEST PRIOR

### D.1 Effort estimate (honest, gated on Gate A passing first)
- **Pre-req (NOT counted here):** Gate 0 (signal exists at resolution) + Gate A
  (>=3 forecasters pass) -- both can FAIL and stop the build. If they fail, this
  spec is shelved; the deliverable is the null.
- **C.1 + C.2 (bundle wiring + action-free imagination + portfolio state):**
  ~1.5-2 days. The action-free refactor of `imagine_with_actor` is the
  load-bearing, error-prone part.
- **C.3 guardrails (all 7):** ~2-3 days (ensemble + LCB + Dyna alarm + OOD + KL-B0
  are each a small but real component).
- **B.5 A/B harness wiring + B0 + seed/bootstrap/DD battery:** ~1-1.5 days (mostly
  reuse).
- **Tuning + the H-sweep + divergence-gap calibration:** ~1-2 days.
- **Total: ~6-9 focused days** AFTER Gate A, single-asset v1 discrete {flat,long},
  REPLAY-only. v2 buckets + DREAM variant are separate fast-follows.

### D.2 The honest prior
- **P(A1 beats B0 on UNSEEN compound at daily/4h, by a robust margin) is LOW.**
  B0 has ~1 free parameter; A1 has ~1000x over ~thousands of effectively-
  independent bars -- the bias-variance math favors B0 unless the WM signal is
  strong AND the sizing edge is real.
- A1's **defensible role** is a calibrated **sizing / abstention / exit-timing**
  layer over a WM that ALREADY passed Gate A (proving there is signal to
  time/size) -- NOT a hail-mary to conjure signal.
- **Ceiling 2 caps everything:** if daily/4h LO crypto has weak fundamental signal
  (the project's repeated history), F / A1 / A2 / A1H all converge to the same
  low/flat number and the lever is RESOLUTION (HF/microstructure), a DATA decision
  orthogonal to this taxonomy. A clean A1-ties-B0 result is itself a high-value
  Ceiling-2 bound, cheap because the gates already exist.
- **Tie => ship B0.** Build A1 to get an honest answer, not to force a win.
