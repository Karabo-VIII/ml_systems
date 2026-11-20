# Agent-Layer Architecture & Roadmap (2026-06-11)

> From a 5-lens design workflow (taxonomy/interfaces, WM-agent build, raw-data-agent build, GIGO/ceiling gating, literature). The user-established framing: 3 model classes (Forecasters / WM-consuming agents / raw-data self-evolving agents) + the 2-ceiling gating (forecaster-GIGO vs fundamental-signal).

---

# SYNTHESIS (the decision-grade roadmap)

All claims confirmed against the actual tree:
- `dream_step(h_prev, z_prev, gru_hidden)` is **action-free** ("no observation" / no action arg) — the load-bearing A1 defect is real.
- V16/V17 are **orphaned** — 0 hits in `wm_tournament.py`.
- `src/agent` importers are all internal (no live external coupling except a `decision_spine.py` comment reference, not a hard import) — migration is low-risk.

I have enough verified ground truth. Producing the document.

---

# Building BOTH the WM-Consuming Agent (A1) and the Raw-Data Self-Evolving Agent (A2) to World-Class, Without GIGO

**A decision-grade architecture + sequenced roadmap. 2026-06-11.**

Every claim below is tagged **[MEASURED]** (verified in tree/code this session), **[APPARATUS]** (existing harness/gate this reuses), or **[PROJECTED]** (a design hypothesis, unproven). The honest one-line verdict up front: **this taxonomy is necessary plumbing and a GIGO-isolation instrument — it is not a signal-finder. Its highest-value early output is a clean null that bounds Ceiling 2 from a new direction, obtained cheaply because the gates already exist.**

---

## 1. THE 3-CLASS TAXONOMY (F / A1 / A2)

### 1.1 The separation, and why it is the right axis

The defining axis is **NOT "does it use a neural net" — it is the SOURCE OF THE WORLD MODEL the policy plans against.**

| | **F — Forecaster** | **A1 — WM-Consuming Agent** | **A2 — Raw-Data Agent** | **A1H — Hybrid (4th, real)** |
|---|---|---|---|---|
| **Consumes** | raw 41-dim chimera features | a FROZEN F's `ForecastBundle` | raw price/bars/TIs directly | frozen F's bundle **AND** raw bars |
| **Builds the world model?** | yes — it IS the WM | no — plans over one someone else froze | it is its own implicit model | no — plus a raw side-channel |
| **Emits** | beliefs (latent, return-dist, regime) — **never acts** | actions + value | actions + value + own learned rep | actions + value |
| **Trains F?** | yes | **NO — F frozen (mechanical invariant)** | N/A | **NO — F frozen** |
| **Primary KPI** | genuine learning (ShIC>0.015 + held-out IC>0 as **diagnostic**) | held-out **compound** | held-out **compound** | held-out compound **AND beats A2-ablation** |
| **GIGO exposure** | n/a (it's the source) | **HIGH** — inherits F's defects | **none on the WM axis** | MEDIUM |
| **Example** | V1.1, V12, V3, V13 | DreamerV3-over-V1.1, MuZero-over-V1.1 | DT/offline-RL on raw f41 | Dreamer over (V1.1-latent ⊕ raw f41) |

The asymmetry that matters: **F's gate is about TRUTH (did it learn or memorize?). A1/A2's gate is about MONEY (does it compound, robustly, out-of-sample?).** Never let an A1's compound number launder a memorized F into "good."

### 1.2 What is on disk RIGHT NOW [MEASURED this session]

- **F** lives in `src/wm/v*/`. The anchored 7-set (V1.1, V12, V3, V4, V6, V8, V13). V1.1 `world_model.py` is a genuine RSSM with `forward_train` (L268), `encode_sequence` (L758), `dream_step` (L772).
- **A1 already exists, half-built and UNREGISTERED**, in `src/agent/` (singular): `dreamer_v3_agent.py` already imports a **frozen V16** (`sys.path.insert ... v16_training`, L29) and trains **actor+critic only** ("Backward only on actor + critic (WM frozen)", L228). The GIGO seam is already coded: `environment.py:7-13` documents **REPLAY** (real returns, ground truth) vs **DREAM** (predicted returns, for exploration), and L476/L623 confirm REPLAY uses `target_return_1` / "REAL returns".
- **The mis-file is real**: V16/V17 sit in the forecaster zoo but are A1 backbones; **`wm_tournament.py` references them 0 times** — orphaned, neither registered as F nor as A1.
- **A2 does NOT exist.** `src/agent/` is entirely WM-coupled.
- **The reference planning machinery exists**: `projects/chess_zero/az/{mcts.py, game_adapter.py, train_robust.py}` — and `game_adapter.py` already encodes the contract and the warning ("a policy that PLANS over an imperfect WM learns to exploit the WM's ERRORS").

### 1.3 The layer contract — `ForecastBundle` (the single most important deliverable)

One frozen, versioned dataclass is the **ONLY** thing A1 may import from F. Owned by F (`src/wm/forecast_bundle.py`), all tensors `.detach()`'d, source forecaster in `eval()`.

```python
@dataclass(frozen=True)
class ForecastBundle:
    # --- LATENT (the 'dynamics' channel -> A1's observation/state) ---
    feat:          Tensor              # [B,T,d_model+flat] = cat([h_seq, z_post])
    h_seq:         Tensor; z_post: Tensor
    # --- BELIEF (DISTRIBUTIONS, not points -> A1's belief feature, NEVER its reward) ---
    return_logits: dict[int, Tensor]   # {h: [B,T,255]} TwoHot logits per horizon
    return_bins:   Tensor              # [255] bin centers for decode
    # --- REGIME (the cheap, robust gate channel) ---
    regime_logits: Tensor              # [B,T,3] bear/range/trend
    # --- PROVENANCE (the GIGO firewall, machine-checked) ---
    forecaster_id:    str
    genuine_learning: dict             # {"shic":..., "held_out_ic":..., "passed": bool}
    is_frozen:        bool             # asserted True at A1 train-time
```

**Where the boundary sits — all three channels, strict roles:**
1. **`feat` → A1's observation** (the Dreamer/MuZero state).
2. **`return_logits[h]` → A1's belief INPUT, optionally imagined-reward shaping — but NEVER the realized reward.** The critic regresses to **real forward return** (`target_return_*`), not F's prediction. This is the GIGO firewall.
3. **`regime_logits` → A1's conditioning/gate.** Cheapest, slowest-moving, most robust channel.

**Default for the first A1:** consume `feat` (state) + `regime_logits` (gate), decode `return_logits` as a belief feature, **reward on realized returns only** (REPLAY mode). Reserve latent imagination (DREAM, predicted-return reward) for an explicitly-flagged second variant — that is where GIGO bites hardest.

### 1.4 V16/V17 reclassification (directory surgery, each step a testable commit)

```
src/wm/                       # CLASS F ONLY
  v0..v14, v21..v25/          # stays
  forecast_bundle.py          # NEW — the F->A1 contract
  wm_promotion_gate.py        # stays (compound, IC-excluded)
src/agents/                   # NEW TREE — all ACTING agents
  registry.py                 # class-aware (A1/A2/A1H), tagged
  a1_wm_consuming/
    backbones/{dreamerv3.py <-v16, tdmpc2.py <-v17}
    dreamer_v3_agent.py environment.py   # <- MOVED from src/agent/
    muzero_agent.py           # NEW — port chess mcts.py to ForecastBundle
  a2_raw_data/{self_evolving_agent.py NEW, ppo.py sac.py <-moved}
  a1h_hybrid/hybrid_agent.py  # NEW
  _shared/{policy.py rewards.py}
```

Migration discipline (CLAUDE.md per-file rule): `git mv` V16→backbones, fix the `sys.path.insert` at `dreamer_v3_agent.py:29`; `git mv src/agent/*`; leave a `MOVED.md` **tombstone** (not a dead dir); grep importers first (verified this session: all `src/agent` importers are internal; `decision_spine.py` only *mentions* it in a comment — no hard coupling, so migration is low-risk).

### 1.5 Registry — two physically separate leaderboards (critical anti-pattern guard)

```
runs/registry/
  forecasters.json   # F champions, keyed by held-out IC + ShIC (TRUTH)
  agents.json        # A1/A2/A1H champions, keyed by held-out COMPOUND + 'class' field (MONEY)
  champion.json      # the single deployable: reads ONLY from agents.json
```

Putting F (ranked by IC) and agents (ranked by compound) on one leaderboard re-imports the **banned IC-as-objective** through the back door. Keep them physically separate.

### 1.6 Per-class KPI + gate (reusing existing machinery [APPARATUS])

| Class | Primary gate | Mechanism |
|---|---|---|
| **F** | `ShIC(h=1) > 0.015` AND `ShIC/contig-IC > 0.3` AND held-out IC>0 (diagnostic, **not** objective) | `SHUFFLED_IC_PATIENCE` early-stop + `wm_promotion_gate` |
| **A1** | held-out compound > champion **AND** source-F `genuine_learning.passed` **AND** 8–10/10 seeds positive on UNSEEN + bootstrap p05>0 + maxDD<30% | `wm_promotion_gate.should_promote` + `src/strat/battery.py` |
| **A2** | identical to A1 **MINUS** the source-F clause (no F) + a policy-overfit control (the shuffled-market test, §4) | same gate, `requires_genuine_forecaster=False` |
| **A1H** | A1's full gate **PLUS** beats its own A2-ablation by a pre-registered margin | new `hybrid_value_gate` |

### 1.7 The 4th class A1H is a TRAP-DETECTOR, not just a class

A1H (A1 that also peeks at raw data) exists because **A1H gated against an A2-ablation is the cleanest empirical test of whether the forecaster adds ANY value.** If A1H ≤ A2-ablation, the WM is proven useless on this task — itself a high-value finding given the project's honest history. **An un-ablated A1H is the most seductive overfit in the whole taxonomy** (it can always fall back to raw data and credit the WM). Non-shippable without the ablation, by rule.

### 1.8 CDAP invariants to add [APPARATUS — make the boundary unbreakable]

Add to `config/_invariants.yaml` + `src/audit/check_invariants.py`:
1. **`forecaster_frozen_in_agents`** — any `F.train()`, F-params-in-optimizer, or missing `.detach()` on a bundle tensor in `src/agents/**` = exit 2.
2. **`no_predicted_return_as_realized_reward`** — critic target must trace to `target_return_*`, not decoded `return_logits`, unless in an explicit `dream_*` module with the `genuine_learning.passed` assert present.
3. **`agent_class_declared`** — every agent declares `__class_tag__ in {A1,A2,A1H}`; registry rejects untagged writes (stops the V16/V17 mis-file recurring).
4. **`v16_v17_not_in_wm`** — hard-assert the dirs no longer exist post-migration (locks the reclassification).

---

## 2. THE 2-CEILING GATING

Two **independent, multiplicative** ceilings. No architecture escapes both.

```
realized_edge  ≤  Ceiling2_signal  ×  Ceiling1_fidelity   (A1, A1H)
realized_edge  ≤  Ceiling2_signal  ×  policy_extraction    (A2)
```

A perfect planner over a perfect WM yields **zero** if Ceiling 2 is zero.

### 2.1 Ceiling 1 — Forecaster-quality / GIGO (binds F, A1, A1H; A2 sidesteps)

A planner does not consume the WM's *average* prediction — it computes **argmax over the WM's value surface**, which is an **adversary against the WM's error surface**:
- **MCTS/MuZero**: PUCT `argmax_a[Q + c·P·√ΣN/(1+N)]` preferentially expands toward high-predicted-value states — i.e., it *seeks* the WM's hallucinations.
- **DreamerV3**: the actor is trained inside imagined rollouts; one-step error compounds geometrically over horizon H, worst exactly where the WM is least data-constrained (tails, regime transitions).
- **In-repo precedent [MEASURED]**: pure self-play DEGRADED the chess imitation net (its games were below the teacher it copied). Substitute "WM-predicted return" for "self-play value" and "held-out compound" for "real strength" — that is the crypto GIGO risk verbatim.

**Consequence: a weak WM + a planner is strictly WORSE than a weak WM + a simple threshold policy** — the planner is a high-variance error-seeker. This is why GATE A clause 1 is non-negotiable for A1.

### 2.2 Ceiling 2 — Fundamental signal at our resolution (binds ALL classes)

The existence question. The HARD dead-list entries answer "no" at daily/4h/dollar LO spot for tested avenues: D13 (IC-as-primary archived), D17 (cross-sectional IC≈0 across 6 architectures, 1d/4h/3d/weekly), D44 (1-5%/day via daily-bar prediction needs IC~0.6; measured ~0 six ways), D45 (bar-level entry-timing info "lives BELOW the bar"). **A2 does NOT rescue this — A2 only removes the GIGO amplifier; it cannot manufacture signal that isn't in the bars.** If Ceiling 2 is binding, the entire F→A1 stack is rearranging deck chairs and the real lever is **resolution (HF/microstructure) — a DATA decision, orthogonal to this taxonomy.**

### 2.3 GATE 0 — does exploitable structure exist at our resolution? (precedes ALL agent-building)

Model-FREE on purpose, so a failed agent can never be blamed on the agent.
- **Oracle-gap / harvestability** [APPARATUS `src/oracle/`]: build the hindsight oracle on the multi-candle move; is there a **past-only conditioner** that raises realized capture above a **regime-matched, cost-matched random-entry null**, held-out, ≥8/10 seeds, **OOS→UNSEEN persistent**? Oracle ceiling high but no conditioner narrows the gap ⇒ not extractable at this resolution (the D45 finding).
- **IC-at-the-move-scale, one-sided diagnostic only**: IC is banned as a primary metric, but as a *negative* existence test it is decisive — shuffled-controlled predictive correlation at the move scale indistinguishable from 0 across the universe/cadences ⇒ existence REFUTED at this resolution.

**Gate-0 FAIL ⇒ STOP. Do not build A1 or A2. Escalate the DATA decision** (tick/LOB/liquidation microstructure — note D71/D72: the "meat" exists at 1m for movers but continuation-given-onset has no internal info → needs an EXTERNAL signal, AUC>0.58). A fancier agent is the wrong lever and the single most expensive mistake the project can make.

### 2.4 GATE A — the exact bar a forecaster clears before ANY A1 is wired onto it

Hard, pre-registered, mechanically checked. Fail ANY clause ⇒ **A1 forbidden** (the WM may still serve as a lower-stakes sizer/filter). All numbers measured on OOS/UNSEEN, never train.

1. **Genuine learning (anti-memorization), MANDATORY** — `ShIC(h=1) > 0.015` AND `ShIC/contig-IC > 0.3`. **Tighten for A1: ShIC must hold at the horizons the planner queries** (h ∈ {1,4,16}), not just h=1 — a WM genuine at h=1 but memorized at h=16 produces garbage 16-step imagined trajectories, exactly where Dreamer/MuZero spend their search. Rejects the V22/V25 `ic1=+0.21 / ShIC=0.000` memorizers and the D02 voladj `IC=0.10 / raw=0.017` shortcut.
2. **Held-out compound over the right null, MANDATORY** — via `src/strat/wm_value_probe.py`: on UNSEEN the WM-driven predict-then-rule policy beats BOTH buy-and-hold AND regime-matched cost-matched random entry across ≥10 seeds, positive margin. (`wm_promotion_gate` already excludes IC — `no_ic_gate`.)
3. **Seed-robustness, MANDATORY** — ≥8/10 seeds clear clauses 1–2.
4. **Regime-coverage, MANDATORY** — clauses 1–2 hold separately in trending / mean-reverting / high-vol, not just pooled (a bull-only-genuine WM is a beta proxy a planner will over-trade).
5. **Cost honesty, MANDATORY** — TAKER 0.0024 round-trip is the gate value [APPARATUS, F9 fix]; maker 0.0010 is a labeled sensitivity, never the headline (D43: real p_fill 0.21–0.40).

**Verdict rule:** all 5 PASS ⇒ eligible A1 substrate (one-way ratchet, monotonic floor). Any FAIL ⇒ A1 forbidden.

---

## 3. WHAT IT TAKES — CLASS A1 (WM-consuming agent)

### 3.1 Recommendation: DreamerV3 imagination actor-critic over the V1.1 RSSM

Not MuZero-MCTS, not TD-MPC2-MPPI. Three reasons, decisive in order:
1. **Architectural fit.** Our WMs ARE RSSMs; Dreamer's actor-critic consumes `(h,z)` natively. MuZero requires *value-equivalent* dynamics (trained to predict reward/value, not reconstruction) ⇒ **retraining the WM, discarding the ShIC>0 anti-memorization anchor that is the only thing certifying it learned signal.** TD-MPC2 wants its own latent-dynamics + Q-ensemble end-to-end. The Dreamer path is **already half-built** [MEASURED: `dreamer_v3_agent.py` imports frozen V16, trains actor+critic only].
2. **Amortized policy > per-decision search** at 4h/daily — we trade one bar at a time; a reactive π(a|s) avoids MCTS compounding WM error over a tree.
3. **Continuation/horizon heads** already exist in the RSSM (V16's head set IS the Dreamer set).

Keep MuZero/TD-MPC2 as **named falsifiers**: if Dreamer's actor underperforms a short-horizon MPPI planner over the *same* WM, that's evidence the actor (not the model) is the bottleneck — a useful A/B, not a dead end.

### 3.2 The LOAD-BEARING gap — action-conditioned dynamics [MEASURED: this is the real blocker]

V1.1 `dream_step(h_prev, z_prev, gru_hidden)` takes **no action** (verified L772–783, "no observation") — it rolls the market's **unconditional** next-state prior. Without fixing this, "imagination" is just unrolling the market prior and the actor has nothing to control.

**The correct factorization (and it's where finance is EASIER than control):** at LO+spot+lev=1 we are a **price-taker** — our position does not move BTC. So the market latent transition is **legitimately action-free** (keep `dream_step` as-is — a major de-risk, no WM retrain). The action conditions the **reward and the portfolio/inventory state**, not the market latent:
- **Market latent** evolves action-free (keep `dream_step`).
- **Agent state augmented** with portfolio state (position, bars-in-trade, return-so-far). Actor sees `(h, z, portfolio_state)`.
- **Reward** = action × decoded next-bar return − cost·|Δposition|, the return sampled from the WM's **full predictive distribution** (we have twohot logits — use the distribution, not the point estimate).

(For perps/large size on thin alts the price-taker assumption breaks and you'd need the conditioned transition — out of scope for spot LO. Measure on the actual universe; clearly holds for BTC/ETH.)

### 3.3 Action space, reward

- **v1: discrete {flat, long}** — smallest action space, matches the harness long-only `_simulate`, no sizing-overfit surface.
- **v2 fast-follow: position buckets {0, 0.25, 0.5, 0.75, 1.0}** — sizing as the legitimate A1 value-add (calibrated uncertainty ⇒ size down when predictive variance is high).
- **continuous [0,1]** only under MPPI / squashed-Gaussian Dreamer, only after discrete wins (continuous sizing is the biggest silent-overfit surface — it can fit the validation equity curve's exact shape).
- **Reward** [APPARATUS `harness.py`]: per-step `r_t = a_t·r^log_{t+1} − cost_rt·|a_t−a_{t-1}|`, taker 0.0024. Episode = the SETUP (entry to policy-exit, multi-candle), γ≈1 episodic compound. **Cost charged INSIDE imagination**, not bolted on after (else the actor learns a high-turnover policy the harness then taxes to death — the +501%→+94% MtM/cost history). **No per-bar IC reward** (banned). **Asymmetric loss**: a false LONG costs more than a missed winner (drawdown is path-dependent; the gate is all-window-positive) — pre-register the coefficient, never tune on UNSEEN.

### 3.4 Model-exploitation guardrails (the heart of A1 — defense-in-depth)

1. **WM ensemble + disagreement penalty (PRIMARY).** We have **7 anchored forecasters** — a genuine asset most projects lack. Roll imagination through K of them; penalize reward by cross-model disagreement (or refuse to act where disagreement > τ). The actor cannot exploit an error only one WM makes.
2. **Value pessimism / LCB critic (MANDATORY).** Ensemble of critics, act on min/low-quantile of V. The offline-RL CQL/SAC-N lesson — directly counters imagined-value overestimation.
3. **Short imagination horizon H = 4–8 bars** (≈ one setup), NOT 15–50. Literature + theory (loss-compounding result) say ≤3–5 in fat-tailed domains. Tune H DOWN if imagined return >> real-data return (a divergence alarm).
4. **Real-data grounding (Dyna ratio).** Interleave imagination with real-transition replay; track the imagined-vs-realized reward gap every epoch — **widening gap ⇒ HALT, don't tune through it** (it's the leading indicator the actor is leaving the data manifold).
5. **Distributional planning** — use the full twohot distribution (variance for sizing, tail for risk), not the mean.
6. **Behavior regularization** — KL(π ‖ predict-then-rule) so A1 degrades gracefully to the baseline.
7. **OOD-latent detector** — flag `(h,z)` far from the training manifold (recon error / prior-posterior KL spike) and force flat there.

### 3.5 The A/B vs predict-then-rule (the ship decision)

Baseline **B0**: take the WM return head, enter long when `E[r]−cost > threshold`, exit by policy, size 1 — ~1 free parameter. A1 has ~1000×. **Bias-variance math favors B0 unless the WM signal is strong AND the sizing edge is real.** Ship A1 ONLY IF it beats B0 on UNSEEN compound by a pre-registered margin (same `should_promote` gate) AND passes ≥8/10 seeds + bootstrap p05>0 + maxDD<30%. **A tie or marginal win ⇒ ship B0.** Decompose any win via the harness `layer_kpis` (entry-timing vs exit-timing vs sizing/abstention) — an A1 that "wins" only by going flat in bear windows is a **regime filter, not alpha** (the exit-axis-NULL lesson; name it honestly).

### 3.6 Effort + honest prior

Dreamer is the most plumbing-heavy of the three but reuses the most existing code (the agent is half-built), so **net effort is the lowest**. Honest prior: **P(A1 > B0 on UNSEEN, daily/4h, by a robust margin) is LOW.** A1's defensible role is a **calibrated sizing/abstention/exit-timing layer over a WM that already passed Gate A** — not a signal-manufacturing machine. Build A1 on the WM that already *proves* there's signal to time/size, not as a hail-mary to conjure it.

---

## 4. WHAT IT TAKES — CLASS A2 (raw-data self-evolving agent) — harder, higher-ceiling

### 4.1 Recommendation: Decision-Transformer / offline-RL inner learner + evolutionary outer loop

Ranked for low-SNR non-stationary markets:
- **(D) Offline RL / Decision Transformer — the SUBSTRATE.** The market has no exact simulator and you cannot re-run 2021; offline RL is *designed* for "learn from a fixed dataset you cannot extend." DT (return-to-go conditioning) turns policy learning into supervised sequence modeling — the regime our infra already validates — and **never bootstraps a value, so no extrapolation blowup** (the central offline-RL pathology). "Self-evolving" maps cleanly onto re-conditioning on a higher return-to-go.
- **(C) Evolutionary / population — the OUTER loop, not the inner learner.** A population + selection-on-held-out is the cleanest "self-evolving" mechanism AND a built-in overfit guard (select on validation, not train). It maps onto the chess `train_robust.py` **champion-gate** [MEASURED exists]. Use it to evolve hyperparameters/reward-shaping/diversity, with DT as the gradient inner learner.
- **(A) Model-free PPO/SAC — BASELINE TO BEAT, not the ship candidate.** Recurrent (mandatory for the multi-bar MOVE) but: one history ⇒ memorizes the path on repeated passes; exploration in a market is *adverse* (random trades pay real costs, teach nothing); it produced the project's **worst over-fit incident** [MEASURED `walk_forward.py:9-12`: single-seed +44%/+40% debunked to median −7%/−34% at 10-seed audit — this is the canonical A2 incident, recorded before A2 was named].
- **(B) Model-based-from-raw = A1 in disguise** — learning dynamics from raw and planning over it re-acquires Ceiling 1. Excluded from A2 by definition.

### 4.2 Hard problems + mitigations

- **H1 — Representation learning from low-SNR raw data.** End-to-end will fit the backtest path, not market structure. **Mitigation: warm-start the encoder from one of the 7 anchored forecasters** (import the genuinely-learnt REPRESENTATION without consuming its PREDICTION — sidesteps A1 GIGO). Encoder frozen/low-LR; policy head fresh. Add self-supervised aux losses (contrastive/recon) as an information-bottleneck regularizer (the same logic that gives ShIC>0).
- **H2 — Credit assignment over multi-bar MOVES with sparse compound reward.** DT shines here (return-to-go is a direct input, not a noisy bootstrap); PPO struggles (GAE with a market value-function is noisy). Episode = SETUP-to-EXIT; reward = realized compound net cost; **no per-bar IC shaping** (banned).
- **H3 — The OVERFitting trap = the RL analog of ShIC=0, and it is INVISIBLE.** A memorizing policy's only symptom is a profitable backtest curve — indistinguishable by inspection from a real edge. **The detection apparatus (the policy-side ShIC analog), in priority order:**
  - **The shuffled/surrogate-market control** — re-train/re-evaluate on phase-randomized or block-shuffled returns (marginal preserved, temporal structure destroyed). A genuine policy → ≈0; a memorizer still "profits" → that residual is the ShIC=0 signature for a policy. **This is the single most important gate to build and does NOT yet exist for policies.**
  - **PBO via CSCV** [APPARATUS `src/strat/pbo_cscv.py`] on the policy POPULATION — the in-sample-best of 10³–10⁵ candidates is by construction the most path-fit. Ship rule PBO < 0.10 (already two-sided validated).
  - **N-seed + per-seed-OOS gate** [APPARATUS `walk_forward.py`, `PER_SEED_OOS_GATE_PCT=70`] — LIVE deploys ONE seed; RL policies are *more* seed-sensitive than supervised (env interaction amplifies variance), so this gate is MORE binding for A2.
- **H4 — Non-stationarity / regime shift.** Walk-forward rolling re-fit (50/20/20/10 + 400-bar purge), per-regime evaluation, `worst_month_pct > −10%` (battery Lens C). **Report per-regime compound, never just aggregate** (an aggregate +X% that is +40% bull / −30% bear is a regime bet, not an edge).
- **H5 — The simulator becomes the adversary.** Because `reward()` is now in the training loop, any leak/MtM-double-count/optimistic-fill is **actively found and amplified** (a policy *hunts* a leak; a forecaster only passively benefits). Mitigations [APPARATUS]: `leak_probe.relative_leak_test` (the cadence-robust verdict — the absolute/shift-spectrum verdicts FAILED on coarse bars), MtM-no-double-count invariant, point-in-time universe (no survivorship), **calibrated cost** (p_fill 0.21–0.40, not the 0.80 default — a policy trained on optimistic fills learns trades that won't fill live).

### 4.3 What "self-evolving" concretely means (NOT unconstrained online adaptation)

1. **Outer evolutionary/population loop (PRIMARY)** — each generation: train inner DT, evaluate on held-out, SELECT survivors by val-compound + diversity, mutate. Selection on held-out = built-in overfit firewall. Maps to the chess champion-gate (the MEMORY lesson: "the monotonic promotion gate is what makes experimenting safe").
2. **Skill-library / curriculum (SECONDARY)** — [APPARATUS `scripts/autonomy/skill_library.py`] accumulate validated sub-policies; easy regimes → hard. Only after the population loop is proven.
3. **Online/continual learning (LAST, bounded)** — ONLY as walk-forward fine-tuning behind a gate: re-fit on the newest closed window, but the update must re-pass N-seed + PBO + shuffled-market against the prior champion before promotion. **Unconstrained continual learning = recency-overfit + catastrophic forgetting + the self-approval bypass** (treat every online update as a potential data leak with a logged pre/post snapshot).

### 4.4 Exploration in a market (the A2-specific trap)

Standard exploration is **actively harmful**: a random exploratory trade pays real spread+fee+slippage and the reward is dominated by noise. **In a low-SNR market the correct action-level exploration budget is near-zero; spend it at the population/architecture level instead.** Curiosity/intrinsic-reward is a trap (a non-stationary market is endlessly novel → the agent chases noise).

### 4.5 Effort + honesty

A2 is **strictly harder** (build the agent, the simulator-in-the-loop, AND the new shuffled-market control + repoint the entire robustness battery at a policy's trade-return stream) and **higher-ceiling** (sidesteps Ceiling 1). But its first deliverable is **not a profitable bot — it is an HONEST verdict** on whether an end-to-end policy beats the cost-matched random-entry firewall on UNSEEN at a given resolution. **If it cannot beat random-entry on UNSEEN at 4h (the most likely outcome given history), that bounds Ceiling 2 from a new direction — and it's cheap, because the gates already exist.**

---

## 5. LITERATURE REALITY — adoptable vs hype/contaminated

### 5.1 Genuinely adoptable (cite-grounded, reproducible mechanism)

1. **HiP-POMDP / context-conditional world model** (arXiv:2411.01342) — condition the RSSM on an inferred regime latent; structurally addresses non-stationarity. **Tested only on proprioceptive control — untested on finance.** Adoption path: add a regime context vector to V1.1/V12 conditioning. A genuine next-step, but **[PROJECTED]** for crypto.
2. **Contrastive encoder pretrain, FROZEN before RL** (arXiv:2407.18645 + 2403.09809) — decouple representation from policy optimization (encoder doesn't backprop through the RL signal ⇒ less WM exploitation). This is exactly the A2 §4.2/H1 warm-start.
3. **Regime-gate / policy-vector library** (ReCAP, arXiv:2606.00143) — CUSUM regimes + composable policy vectors + small auditable gate net; cleaner eval than most (17yr, strict no-look-ahead normalization). The most directly adoptable continual-learning technique. Caveat: continuously trained during test vs static baselines (asymmetric); short 5yr live window inflates long-biased strategies.
4. **Pessimistic offline RL via synthetic OOD augmentation** (MetaTrader, arXiv:2505.12759) — F₁/F₂/F₃ transforms + min-Q conservatism. Sound recipe, still early/volatile (±0.10 on metrics), 2019–2022 test avoids a genuine bear.
5. **Short-rollout discipline** (arXiv:2404.09946) — ≤3–5 steps in noisy/fat-tailed domains. Directly sets A1's H.

### 5.2 Honest negative results (where RL-for-trading reliably FAILS)

- **No reproducible MBRL-beats-model-free-after-cost finding exists.** Zero peer-reviewed DreamerV3/MuZero/TD-MPC2 finance studies that are look-ahead-clean, multi-regime, OOS. The project's treatment of V16/V17 as **stubs is correct** — adopting them is a research bet, not an established technique.
- **MuZero-style value-equivalence loss formally FAILS in stochastic environments** (arXiv:2404.09946) and incurs exponential sample complexity with coverage gaps — both endemic to markets. This is a *theorem-level* reason to prefer reconstruction-augmented (Dreamer-RSSM) objectives over pure value-equivalence.
- **Every reviewed method degrades under genuine regime change.** ReCAP mitigates, does not solve. No method survives a held-out test spanning a regime it never trained on.
- **Offline RL universally overfits training-regime distributions** (arXiv:2505.12759, 2411.12746) — and pessimism is in *structural tension* with the task (the optimal action — e.g., regime-appropriate short — is often OOD relative to a long-biased behavior policy).
- **The "competitive performance" rhetorical trap** — many papers' "competitive with baselines" means underperforming risk-matched buy-and-hold after costs (FinRL BTC: 0.66% vs B&H 0.74%). Always use a **volatility-targeted B&H** denominator.
- **Only honest live result in the broad literature is ~5 days.** No peer-reviewed sustained live RL trading over a meaningful horizon.

**Net:** the architecture backbones are real and function elsewhere; the *finance-specific evidence is absent or contaminated*. Take the **techniques** (regime-conditioning, frozen contrastive encoder, short rollouts, pessimism, PBO/seed gates); reject the **claims**.

---

## 6. THE SEQUENCED ROADMAP + RECOMMENDATION

Dependency order is **foundation-first**. Each gate is a measured-true precondition, not a projection. Skipping Gate 0 = GPU-days planning over noise. Skipping Gate A clause 1 = a confidently-wrong planner on real capital.

### Phase 0 — RECLASSIFY + CONTRACT (1–2 days, zero research risk, pure plumbing)
- `git mv` V16/V17 → `src/agents/a1_wm_consuming/backbones/`; move `src/agent/*` per §1.4; `MOVED.md` tombstones; fix the `dreamer_v3_agent.py:29` path.
- Write `src/wm/forecast_bundle.py` (the frozen contract).
- Two registry files (`forecasters.json`, `agents.json`); add the 4 CDAP invariants (§1.8).
- **Gate:** `py_compile` all moved files; `check_invariants.py` green; `v16_v17_not_in_wm` passes. **Deliverable:** the 3-class separation is *real and machine-enforced*, not prose.

### Phase 1 — GATE 0 (does signal exist at our resolution?) — **THE deciding experiment** (model-free, days)
- Run the oracle-gap / harvestability test [APPARATUS `src/oracle/`] on the multi-candle move across cadences; shuffled-controlled IC-at-move-scale as the one-sided negative diagnostic.
- **Gate-0 PASS** (a past-only conditioner beats the regime/cost-matched null, ≥8/10 seeds, OOS→UNSEEN persistent) ⇒ proceed to Phase 2.
- **Gate-0 FAIL** (the likely outcome given D17/D44/D45) ⇒ **STOP the agent program. Escalate the DATA/resolution decision** (1m+liq / tick / LOB; D72 says the move exists at 1m for movers but needs an EXTERNAL continuation signal, AUC>0.58). **This null is itself a high-value deliverable** — it bounds Ceiling 2 and redirects compute to the only lever that can lift it.

### Phase 2 — GATE A on the 7 forecasters (which F is a legal A1 substrate?) (days)
- Run Gate A (§2.4) on V1.1, V12, V3, V4, V6, V8, V13: ShIC>0.015 + ratio>0.3 **at h∈{1,4,16}**, held-out compound beats BOTH nulls via `wm_value_probe.py`, ≥8/10 seeds, per-regime, taker cost.
- **Gate:** ≥3 forecasters PASS (so the ensemble disagreement penalty has real diversity; a single-WM A1 is REJECTED — no epistemic anchor).
- If **0–2 pass** ⇒ A1 is premature; A2 (which sidesteps Ceiling 1) becomes the better bet, and a calibration probe (is the V1.1 predictive distribution even calibrated on held-out?) is the prerequisite before any uncertainty-aware planning.

### Phase 3 — A1 as the FASTER BRIDGE (agent half-built; reuses the most code)
- Implement the §3.2 portfolio-state factorization (keep `dream_step` action-free — correct for a price-taker, no WM retrain).
- v1 discrete {flat, long}, REPLAY-only (realized reward), all §3.4 guardrails ON, H=4–8.
- **A/B vs B0** (predict-then-rule). **Ship A1 ONLY IF** it beats B0 on UNSEEN compound by the pre-registered `should_promote` margin + ≥8/10 seeds + p05>0 + maxDD<30%. Tie ⇒ ship B0. Decompose any win via `layer_kpis`.
- DREAM mode (predicted-return reward) is a *second, explicitly-flagged* variant, gated on `genuine_learning.passed` — the GIGO-exposed mode, entered last.

### Phase 4 — A2 as the HARDER / HIGHER-CEILING PARALLEL TRACK (start in parallel with Phase 3 if compute allows)
- Build the **shuffled-market control** first (it does not exist; validate it two-sided — positive control + negative control — before trusting any "passed the shuffle" claim).
- DT/offline-RL inner + evolutionary outer + recurrent-PPO baseline; warm-start the encoder from a Gate-A-passing forecaster (representation transfer, not prediction).
- Repoint the full battery (`battery.py` Lens A/B/C, `pbo_cscv.py` PBO<0.10, `walk_forward.py` 70%-seed) at the **policy's** trade-return stream.
- **First deliverable is the verdict, not a bot:** does the policy beat the cost-matched random-entry firewall on UNSEEN at 4h? Most-likely NO → bounds Ceiling 2 from the policy direction (cheap, gates exist) → the expensive HF build is justified only AFTER this coarse null.

### THE RECOMMENDATION

**Build in this order: Phase 0 (plumbing) → Phase 1 (Gate 0) → Phase 2 (Gate A) → Phase 3 (A1 bridge) ∥ Phase 4 (A2 track).**

- **A1 first as the bridge** because it is *half-built and reuses the 7 anchored forecasters + the chess planning machinery* — net lowest effort to a first honest answer. Its defensible role is a calibrated **sizing/abstention/exit-timing layer over a Gate-A-passing WM**, NOT a signal manufacturer. My honest prior on A1 > B0 at daily/4h is **LOW**.
- **A2 in parallel as the higher-ceiling, harder bet** — it sidesteps Ceiling 1 and is the only class whose representation-learning-from-raw advantage *might* surface signal at an intermediate resolution where hand-built features could not (the single most interesting upside test). But it relocates GIGO into an invisible policy, so the shuffled-market control is non-negotiable.

### THE HONEST RISK — BOTH may be capped by Ceiling 2

**If daily/4h LO crypto genuinely has weak signal (the project's repeatedly-confirmed history — the dead-list, the 30m/15m cost-cliff, the "active alpha unproven" consensus), then F, A1, A2, and A1H all converge to the same low/flat number and this entire taxonomy is organizational + GIGO-isolation, not alpha.** A2 does NOT rescue this — it removes the amplifier, not the constraint. **No architecture escapes Ceiling 2.**

**What we do then (named in advance, so a Ceiling-2 null is a decision, not a surprise):** the lever is **resolution/data, not architecture**. Route compute to **HF/microstructure** — 1m+liquidation (D71 dataset already built), tick, LOB — where samples are abundant (favoring A2's data-hungry end-to-end learning) and where D71/D72 located the *move* (though continuation needs an EXTERNAL signal at AUC>0.58, thread 24). The taxonomy then runs unchanged at the new resolution; only the data layer moves.

**Treat "this plumbing finds alpha" as UNPROVEN/[PROJECTED], not measured.** The schema's guaranteed value is that it makes Ceiling 1 *measurable* (via A1H-vs-A2) and isolatable, and makes a Ceiling-2 null *cheap and clean* — which, given the honest history, is the most probable and still genuinely valuable outcome.

**File paths referenced (all absolute under `c:\Users\karab\Documents\coding\v4_crypto_stystem\`):** `src/wm/v1/v1_1_training/world_model.py` (RSSM, action-free `dream_step` L772), `src/agent/{dreamer_v3_agent.py, environment.py}` (half-built A1 + REPLAY/DREAM seam — to move to `src/agents/`), `src/wm/v16,v17` (mis-filed A1 backbones — to move), `src/wm/wm_promotion_gate.py`, `src/strat/{wm_value_probe.py, battery.py, pbo_cscv.py, positive_control.py, firewall.py}`, `src/wealth_bot/{harness.py, leak_probe.py, framework/walk_forward.py}`, `src/oracle/` (Gate 0), `projects/chess_zero/az/{mcts.py, game_adapter.py, train_robust.py}` (planning machinery + champion-gate pattern), `src/audit/check_invariants.py` + `config/_invariants.yaml` (new CDAP invariants). New: `src/wm/forecast_bundle.py`, `src/agents/` tree, `runs/registry/{forecasters,agents,champion}.json`.

---

# APPENDIX -- the 5 expert lenses (full)

## LENS: TAXONOMY + INTERFACES — the 3-class separation (Forecasters / WM-Consuming Agents / Raw-Data Agents), the layer contract, and how to make it real in the tree

**Summary:** The 3-class separation already exists physically but is undeclared and mis-wired: forecasters live in `src/wm/v*/`, a real (but unregistered) agent tree lives in `src/agent/`, and V16/V17 are stubs mis-filed in the forecaster zoo. Formalize it as F (forecasters: emit beliefs, never act; KPI = genuine learning) → A1 (WM-consuming agents: plan/act over a FROZEN forecaster's outputs; KPI = held-out compound) and A2 (raw-data agents: ingest raw bars/TIs, learn rep+policy end-to-end; KPI = held-out compound). The F→A1 boundary is the load-bearing decision: A1 must consume the forecaster's FROZEN, DETACHED outputs through a versioned `ForecastBundle` (latent feat = `[h_seq, z_post]`, the per-horizon return DISTRIBUTIONS `return_logits[h]`, and `regime_logits`) — NOT gradients, and critically NOT the forecaster's PREDICTED returns as reward (that path is the GIGO trap: it learns to trade the WM's hallucination). A 4th hybrid class (A1H) is real and needed — an A1 that also peeks at raw data — but it MUST be gated against an A2-baseline to prove the WM adds value, else it silently collapses into A2-with-overhead. Concretely: keep F in `src/wm/`, create `src/agents/{a1,a2,a1h}/`, move V16/V17 there as backbones, register all three classes in one `runs/registry/` with class-specific gates, and enforce the boundary with a `ForecastBundle` frozen-contract + a CDAP invariant that fails the build if an A1 trains the forecaster or rewards on predicted (not realized) returns.

# The 3-Class (+1 hybrid) Taxonomy — grounded in the actual tree

## 0. What's actually on disk RIGHT NOW (measured, not projected)

- **F (forecasters)**: `src/wm/v*/` — V0–V25. The anchored 7-set (V1.1, V12, V3, V4, V6, V8, V13). Each `world_model.py` exposes `forward_train(...) -> dict`. **Measured emit surface** (V1.1 `world_model.py:353-362`): `{recon, return_logits[h] for h in {1,4,16,64}, regime_logits, prior_logits, post_logits, h_seq, z_post, ret_trunk}` plus `feat = cat([h_seq, z_post])` at L328. `return_logits[h]` are **TwoHot DISTRIBUTIONS** (255 bins, BIN_MIN/MAX=-1/1), not point estimates — this matters for the A1 contract.
- **A1 (WM-consuming), already half-built but UNREGISTERED**: `src/agent/` (singular) — `dreamer_v3_agent.py`, `sac_agent.py`, `ppo.py`, `policy.py`, `environment.py`, `train_agent.py`. `dreamer_v3_agent.py:28-30` hard-imports a **frozen** V16 and consumes `imagine_rollout`. `environment.py` has the exact boundary question baked in as REPLAY mode (WM = features, returns REAL) vs DREAM mode (returns PREDICTED) — see §3.
- **The mis-file**: V16 (`dreamerv3_backbone.py` + `v16_training/dreamer_v3.py`) and V17 (`tdmpc2_backbone.py`) sit in `src/wm/` (forecaster zoo) but are **backbones for A1 planners**, not forecasters. `wm_tournament.py` does NOT reference them (grep = 0 hits) — so they're orphaned, neither registered as F nor as A1.
- **A2 (raw-data / self-evolving)**: does NOT exist yet. `src/agent/` is all WM-coupled. The chess `projects/chess_zero/az/mcts.py` `MCTS(net, ...)` with `net._evaluate(state)->(policy,value)` is the **reference planning machinery** to port for A1, but it is not A2.

## 1. CLASS DEFINITIONS (the contract each class signs)

| | **F — Forecaster** | **A1 — WM-Consuming Agent** | **A2 — Raw-Data Agent** | **A1H — Hybrid (4th class, real)** |
|---|---|---|---|---|
| **Consumes** | raw features (41-dim chimera) | a FROZEN F's `ForecastBundle` (latent + return-dist + regime) | raw price/bars/TIs directly | a frozen F's bundle **AND** raw bars |
| **Emits** | beliefs: latent, return-dist, regime — **no action** | actions (position/size) + value | actions + value + its own learned rep | actions + value |
| **Learns** | representation of market dynamics | a policy/plan over F's dynamics | representation **and** policy end-to-end | policy over (F-latent ⊕ raw) |
| **Trains F?** | yes (it IS F) | **NO — F is frozen** (mechanical invariant) | N/A (no F) | **NO — F is frozen** |
| **Primary KPI** | genuine learning (ShIC>0.015 + held-out IC>0 as DIAGNOSTIC) | **held-out compound return** | **held-out compound return** | **held-out compound, AND must beat A2 baseline** |
| **GIGO exposure** | n/a (it's the source) | **HIGH** — inherits F's defects | **none** (sidesteps F) | MEDIUM — partial raw fallback |
| **Example** | V1.1, V12, V3, V13 | DreamerV3-over-V16, MuZero-over-V1.1, TD-MPC2-over-V17 | self-eval policy on raw f41 + PPO/GRPO | Dreamer over (V1.1-latent ⊕ raw f41) |

**The defining axis is NOT "does it use a neural net" — it is the SOURCE OF THE WORLD MODEL the policy plans against:**
- F: builds the model, never plans.
- A1: plans against a model **someone else built and froze**.
- A2: plans against the **raw environment** (it is its own implicit model).
- A1H: plans against a frozen model **plus** a raw side-channel.

## 2. THE LAYER CONTRACT — `ForecastBundle` (the F→A1 interface, made concrete)

This is the single most important deliverable. Define one frozen, versioned dataclass that is the ONLY thing A1 may import from F. Put it at `src/wm/forecast_bundle.py` (owned by F, imported by A1).

```python
# src/wm/forecast_bundle.py
@dataclass(frozen=True)
class ForecastBundle:
    """The ONLY surface an A1 agent may consume from a forecaster.
    All tensors are .detach()'d and the source forecaster is in eval() mode.
    A1 NEVER receives gradients into F. CDAP-enforced (see §6)."""
    # --- LATENT (the 'dynamics' channel) ---
    feat:        Tensor   # [B,T,d_model+flat_dim] = cat([h_seq, z_post]); the consumable state
    h_seq:       Tensor   # [B,T,d_model] recurrent/deterministic part
    z_post:      Tensor   # [B,T,flat_dim] stochastic posterior latent
    # --- BELIEF (the 'return' channel) — DISTRIBUTIONS, not points ---
    return_logits: dict[int, Tensor]   # {h: [B,T,255]} TwoHot logits per horizon
    return_bins:   Tensor              # [255] bin centers (BIN_MIN..BIN_MAX) for decode
    # --- REGIME (the 'context' channel) ---
    regime_logits: Tensor   # [B,T,3]  bear / range / trend
    # --- PROVENANCE (GIGO firewall) ---
    forecaster_id:    str    # e.g. "v1.1-f34-ep120"
    genuine_learning: dict   # {"shic": 0.033, "held_out_ic": 0.067, "passed": True}
    is_frozen:        bool   # must be True; A1 trainer asserts this
```

**WHERE THE BOUNDARY SITS — the precise answer to "latent vs return-dist vs regime":**

The boundary is **all three channels, but with strict role separation, and the return channel is a BELIEF feature, never a reward**:

1. **Latent (`feat`) → A1's observation.** This is the Dreamer/MuZero path: the policy/critic see `feat` as the state. This is the richest channel and the one that makes A1 "WM-consuming" in the strong sense (planning in latent space, `imagine_rollout`).
2. **Return distribution (`return_logits[h]`) → A1's belief input AND, optionally, its imagined-reward shaping — but NEVER the realized reward.** Decode to a distribution (mean, variance, tail) and feed as features OR as the imagined-reward in DREAM rollouts. **The realized reward an A1 critic regresses to MUST be the REAL forward return, not F's predicted return.** (See §3 — this is the GIGO firewall.)
3. **Regime (`regime_logits`) → A1's conditioning / gating signal.** Cheapest, most robust channel (regime is slow-moving). Good for a regime-gated A1 that switches sub-policies.

**Recommended default for the first A1**: consume `feat` (state) + `regime_logits` (gate), decode `return_logits` as a belief feature, and **reward on realized returns only**. Reserve full latent-imagination (DREAM, predicted-return reward) for a second, explicitly-flagged variant — because that variant is where GIGO bites hardest.

## 3. THE GIGO FIREWALL (the single most dangerous edge, already half-present in code)

`src/agent/environment.py` already encodes the trap and the safe path:
- **REPLAY mode (SAFE)**: WM gives features/latent; **returns are REAL**. The critic learns from ground-truth forward returns. GIGO-immune on the reward axis (still inherits feature-quality risk, see §7).
- **DREAM mode (DANGEROUS)**: WM imagines forward; **returns are PREDICTED**. If the frozen F has `ShIC≈0` (memorized, e.g. the V22/V25 `ic1=+0.21 ShIC=0.000` failure), the A1 learns to exploit a HALLUCINATED dynamics — backtest looks great, live is garbage. This is the GIGO ceiling in its purest form.

**Contract rule (mechanical):** an A1 may only enter DREAM mode if its frozen F carries `genuine_learning.passed == True` (ShIC > 0.015 AND held-out IC > 0). REPLAY mode is always allowed. **A2 has no DREAM mode** — it cannot GIGO on a WM because it has no WM (this is exactly why A2 sidesteps ceiling #1).

## 4. RECLASSIFY V16/V17 — directory + registry surgery (implementable)

**Target tree:**
```
src/wm/                      # CLASS F ONLY — forecasters. Emit beliefs, never act.
  v0..v14, v21..v25/         # stays
  v15/ (PatchTST backbone)   # stays IF used as a forecaster encoder; else -> agents
  forecast_bundle.py         # NEW — the F->A1 contract (frozen dataclass)
  wm_promotion_gate.py       # stays — F's genuine-learning + compound gate
  wm_tournament.py           # stays — F leaderboard
src/agents/                  # NEW TREE — all ACTING agents
  __init__.py
  registry.py                # NEW — class-aware registry (A1/A2/A1H champions)
  a1_wm_consuming/
    backbones/
      dreamerv3.py           # <- MOVED from src/wm/v16/  (it's a planner backbone)
      tdmpc2.py              # <- MOVED from src/wm/v17/
    dreamer_v3_agent.py      # <- MOVED from src/agent/   (actor-critic over frozen F)
    muzero_agent.py          # NEW — port projects/chess_zero/az/mcts.py to ForecastBundle
    environment.py           # <- MOVED from src/agent/  (REPLAY/DREAM, the GIGO seam)
  a2_raw_data/
    self_evolving_agent.py   # NEW — raw f41 -> own rep -> policy (PPO/GRPO/SAC)
    ppo.py sac.py            # <- MOVED from src/agent/
  a1h_hybrid/
    hybrid_agent.py          # NEW — frozen-F-latent (+) raw-bars
  _shared/
    policy.py rewards.py     # <- MOVED from src/agent/ (class-agnostic heads)
```

**Migration steps (each is a discrete, testable commit):**
1. `git mv src/wm/v16 src/agents/a1_wm_consuming/backbones/dreamerv3` (and v17→tdmpc2). Fix the `sys.path.insert` in `dreamer_v3_agent.py:29` to the new backbone path.
2. `git mv src/agent/* src/agents/...` per the map above; update imports; `py_compile` each (the per-file thoroughness rule in CLAUDE.md applies — `src/firm/decision_spine.py` imports `src/agent`, so update that caller too — grep found it).
3. Add a TOMBSTONE `src/wm/v16/MOVED.md` and `src/agent/MOVED.md` pointing to the new homes (don't leave dead paths; CLAUDE.md's "stale imports after file moves" anti-pattern).
4. Register: add V16/V17 backbones + the DreamerV3 agent to `src/agents/registry.py` under class `A1`, NOT to `wm_tournament.py`.

**Registry structure (`runs/registry/`):** one store, class-tagged, so cross-class comparison is possible but gates are class-specific:
```
runs/registry/
  forecasters.json   # F champions: keyed by held-out IC + ShIC (genuine-learning)
  agents.json        # A1/A2/A1H champions: keyed by held-out COMPOUND, with a 'class' field
  champion.json      # the single deployable: whichever ACTING agent (A1/A2/A1H) compounds best on UNSEEN
```
`forecasters.json` answers "which F is the best belief-source"; `agents.json` answers "which acting agent makes the most money". They are DIFFERENT objectives and must not share a leaderboard.

## 5. PER-CLASS KPI + GATE (exact thresholds, reusing existing machinery)

| Class | Primary gate | Mechanism (exists / to-build) |
|---|---|---|
| **F** | `ShIC > 0.015` (genuine learning, anti-memorization) **AND** held-out IC > 0 as diagnostic. IC is NOT the objective — it's the GIGO-prevention proof. | `SHUFFLED_IC_PATIENCE=5` early-stop (settings.py:316) + `wm_promotion_gate` (compound, for ranking deployability of F-as-sizer). |
| **A1** | held-out **compound > champion** (strict monotonic floor) **AND** source-F `genuine_learning.passed` **AND** 10/10 seeds positive on UNSEEN + block-bootstrap p05>0 + maxDD<30% (PROJECT_NORTH_STAR robustness battery). | `wm_promotion_gate.should_promote` (already pure/compound-based) + the `src/strat/battery.py` robustness spine. |
| **A2** | identical compound + robustness gate to A1, **MINUS** the source-F clause (no F to gate). | same gate, `requires_genuine_forecaster=False`. |
| **A1H** | A1's full gate **PLUS** must beat its own A2-ablation (drop the F channel, retrain) by a margin. If it can't, it IS A2 and the F is dead weight. | new `hybrid_value_gate`: `compound(A1H) > compound(A2_ablation) + margin`. |

**The asymmetry that matters**: F's gate is about TRUTH (did it learn, or memorize?). A1/A2's gate is about MONEY (does it compound, robustly, out-of-sample?). Never let an A1's compound number launder a memorized F into "good" — the `genuine_learning.passed` clause is the firewall, and it must be a HARD precondition checked at A1 train-time, not a soft score.

## 6. CDAP INVARIANTS to add (make the boundary unbreakable, per CLAUDE.md's enforcement model)

Add to `config/_invariants.yaml` + `src/audit/check_invariants.py`:
1. **`forecaster_frozen_in_agents`**: grep `src/agents/**` — any `forecaster.train()`, any optimizer that includes forecaster params, or any missing `.detach()` on a `ForecastBundle` tensor = **exit 2**. (Prevents A1 from co-training F.)
2. **`no_predicted_return_as_realized_reward`**: in `src/agents/**`, the critic's regression target must trace to `target_return_*` (realized), not `return_logits`/decoded prediction, unless the file is explicitly in a `dream_*` module with the `genuine_learning.passed` assert present. (The GIGO firewall, mechanized.)
3. **`agent_class_declared`**: every agent module declares `__class_tag__ in {"A1","A2","A1H"}` in its `__contract__`; registry writes are rejected if the tag is missing. (Prevents the V16/V17 mis-file from recurring.)
4. **`v16_v17_not_in_wm`**: hard assert `src/wm/v16` and `src/wm/v17` no longer exist (post-migration); they live under `src/agents/`. (Locks the reclassification.)

## 7. IS THERE A 4TH CLASS? — Yes, A1H, and it's a TRAP-DETECTOR not just a class

A1H (A1 that also peeks at raw data) is a legitimate, distinct class because it has a different failure mode than either parent. But its REASON TO EXIST is diagnostic: **A1H gated against an A2-ablation is the cleanest empirical test of whether the forecaster adds ANY value at all.** If A1H ≤ A2-ablation, you've proven the WM is GIGO/useless on this task — which is itself a high-value finding given the project's honest history. So A1H is both a candidate architecture AND the instrument that measures ceiling #1 (forecaster-quality). Do not ship it without the ablation; an un-ablated A1H is the most seductive overfit in the whole taxonomy (it can always fall back to raw data and credit the WM).

## 8. THE TWO CEILINGS, mapped onto the taxonomy (honest)

- **Ceiling #1 (forecaster-quality / GIGO)**: bites F and A1 and A1H. A2 is immune. Measured, not projected: V22/V25 memorization (`ic1=0.21, ShIC=0.000`) is the documented instance of this ceiling killing an F before it ever reached an A1.
- **Ceiling #2 (fundamental signal at our resolution)**: bites ALL FOUR classes. No architecture escapes it. Honest history says daily/4h long-only is weak; A2 does NOT rescue this — A2 only removes the GIGO *amplifier*, it cannot manufacture signal that isn't in the bars. If ceiling #2 is the binding one (plausible per the dead-list), then the entire F→A1 stack is rearranging deck chairs and the real move is resolution (HF/microstructure), which is a DATA-LAYER change orthogonal to this taxonomy. **This taxonomy makes ceiling #1 measurable (via A1H-vs-A2) and isolatable; it does NOT and cannot resolve ceiling #2.** Flag this loudly to the user: the schema is necessary plumbing, not a signal-finder.

**Failure modes:** **1. GIGO laundering (THE big one).** An A1's good backtest compound silently legitimizes a memorized forecaster (ShIC≈0). The A1 learned to trade the WM's hallucination (DREAM mode), not the market. *Mitigation*: `genuine_learning.passed` is a HARD precondition asserted at A1 train-time (not a score); CDAP invariant #2 forbids regressing the critic on predicted (vs realized) returns outside explicitly-flagged dream modules; default first A1 runs REPLAY-only.

**2. The reclassification half-done — dead imports.** `src/firm/decision_spine.py` and others import `src/agent`; moving the tree without updating callers = silent ImportError or, worse, a shadow `src/agent` left behind that newer code still imports. *Mitigation*: grep ALL importers before `git mv` (already found: decision_spine, firm); leave a `MOVED.md` tombstone, not a dead dir; CDAP invariant #4 hard-fails if `src/wm/v16` still exists.

**3. A1H crediting the WM for A2's work.** A hybrid that can fall back to raw bars will always be ≥ its WM-only variant and looks like "WM helps" — when the WM contributes nothing. *Mitigation*: A1H gate REQUIRES beating an A2-ablation (drop the F channel, retrain) by a margin; an un-ablated A1H is non-shippable by rule.

**4. Forecaster co-training leak.** An A1 optimizer that accidentally includes frozen-F params (easy if F is a submodule) un-freezes F and turns A1 into an end-to-end thing that overfits the train window and breaks the class boundary. *Mitigation*: `ForecastBundle` tensors are `.detach()`'d at the boundary; CDAP invariant #1 greps for F.train()/F-params-in-optimizer in `src/agents/**`.

**5. Leaderboard conflation.** Putting F (ranked by IC/ShIC) and A1/A2 (ranked by compound) on one leaderboard re-imports the banned IC-as-objective through the back door, or lets a high-compound-but-memorized F win. *Mitigation*: two physically separate registry files (`forecasters.json` vs `agents.json`) with different keys; `champion.json` (the deployable) reads ONLY from agents.json.

**6. Return-distribution collapsed to a point.** A1 decoding `return_logits` to a scalar mean throws away the tail/variance that is the whole reason to consume a DISTRIBUTION — and then A1 sizes on a point estimate, defeating the purpose of a probabilistic forecaster. *Mitigation*: bundle exposes `return_logits` (full 255-bin) + `return_bins`; the contract documents that A1 SHOULD use distributional moments (variance for sizing, tail for risk), and a lint warns if only `.argmax()`/mean is used.

**7. Boundary-creep over time.** Six months in, someone adds a raw-data feature to an A1 'just this once' and it silently becomes A1H without the ablation gate; or an A2 quietly imports a frozen F 'for a regime hint'. *Mitigation*: `__class_tag__` is mandatory in every agent's `__contract__` and the registry rejects untagged writes (CDAP invariant #3); changing class requires changing the tag, which trips review.

**Open questions:** **1. Which channel actually carries the A1 value — measured, not assumed?** The contract exposes latent (`feat`), return-dist, and regime. Which one an A1 should primarily consume is UNKNOWN until ablated. Plausible (per honest history) that `feat` adds nothing over raw bars and only `regime_logits` (slow, robust) helps — or that none do. Must be measured via A1H-vs-A2 ablation per channel; do not assume the rich latent is best.

**2. Is REPLAY-mode A1 even distinguishable from a fancy F-as-sizer?** A REPLAY A1 (WM=features, returns=real, reward=realized) is arguably just 'use F's latent as input features to an RL policy' — very close to the existing `wm_value_probe`/`WMEntryProducer` sizer path. The genuine novelty (latent imagination, planning) only appears in DREAM mode, which is the GIGO-exposed mode. So: does A1 buy anything over 'F-as-feature-extractor + simple policy' WITHOUT entering the dangerous DREAM regime? Genuinely unknown; needs a head-to-head.

**3. Does MuZero-style learned-dynamics (plan WITHOUT a separate F) blur A1/A2?** MuZero learns its own latent dynamics from raw observations end-to-end — it has a 'world model' but builds it itself, like A2, yet plans over it, like A1. Where does it sit? Tentatively A2 (it builds its own model) but it's a genuine boundary case the axis 'source of the world model' must adjudicate; the chess `mcts.py` port will force this decision.

**4. What's the right A2-ablation margin for A1H?** 'Beat A2 by a margin' needs a number. Too tight = noise promotes a useless WM; too loose = kills genuinely-helpful hybrids. Must be set from the block-bootstrap p05 spread on UNSEEN, not picked a priori — and it interacts with sample size (n_unseen often <30 per the wealth-bot trust stack).

**5. Does ANY of this move ceiling #2?** Open and arguably the only question that matters: if the binding constraint is fundamental signal at daily/4h resolution (the dead-list's repeated verdict), then F/A1/A2/A1H are all capped at the same low number and the schema's value is purely organizational + GIGO-isolation, not alpha. The taxonomy cannot answer this; only an A2 (or A1H-ablation) hitting a hard compound ceiling regardless of architecture would confirm it. Treat 'this plumbing finds alpha' as UNPROVEN/projected, not measured.

## LENS: The WM-consuming agent build (Class A1: DreamerV3 / MuZero / TD-MPC2 over the anchored RSSM forecasters)

**Summary:** Build the A1 agent as a **DreamerV3-style imagination actor-critic over the existing V1.1 RSSM**, not MuZero-MCTS and not TD-MPC2-MPPI — because our WMs ARE RSSMs (V1.1's `world_model.py` already has prior/posterior heads, `encode_sequence`, and a `dream_step`/`dream_gru` imagination path), so Dreamer is a drop-in consumer while MuZero/TD-MPC2 would require re-architecting or retraining the dynamics. The single load-bearing blocker is that the WM was trained as a PASSIVE FORECASTER: its `dream_step` is action-FREE (the latent transition does not condition on enter/exit/size), so before any actor can plan over it you must add an action-conditioned dynamics head and fine-tune it — without this, "imagination" is just unrolling the unconditional market prior and the actor has nothing to control. The reward is per-step net-of-cost log-return laddering to held-out COMPOUND over the setup/move (harness-grounded: taker 0.0024 default, maker 0.0010 sensitivity), action space is discrete {flat, long} first (LO+spot+lev=1) with continuous sizing as a fast-follow. The GIGO gate is a hard pre-condition: do NOT wire an actor onto any WM that hasn't cleared ShIC>0.015 AND positive held-out compound as a predict-then-rule baseline — and even then, model-based RL for finance has a graveyard, so the entire build is gated A/B against the predict-then-rule policy and ships only if it beats it on UNSEEN compound with seed-robustness. Honest stance: I rate P(A1 beats predict-then-rule on daily/4h crypto) as LOW; A1's defensible value is as a principled sizing/exit-timing layer over a WM that already has a real signal, not as a signal-manufacturing machine.

## 0. What already exists (grounded, measured) — so we build the delta, not a rewrite

- **`projects/chess_zero/az/game_adapter.py`** already defines `DecisionProblemAdapter` (lines 100-134): the single-agent / partial-obs / no-exact-simulator / learned-dynamics contract, with `observation()`, `action_spec()`, `reward()`, `discount()`, `dynamics()`. Its own docstring names our exact risk: *"a policy that PLANS over an imperfect WM learns to exploit the WM's ERRORS (great in imagination, fails live)."* **Reuse this interface verbatim** — the A1 agent is a concrete subclass, not a new contract.
- **`projects/chess_zero/az/mcts.py`** is a clean PUCT MCTS (181 lines) — reusable for the MuZero branch ONLY, but it assumes an exact `apply()`. Over a learned WM, depth-d search compounds per-step model error; this is why MCTS is NOT my recommendation (see §2).
- **`src/wm/v1/v1_1_training/world_model.py`**: V1.1 is a genuine RSSM. It has `prior_head`/`posterior_head` (Gumbel-softmax categorical latent 24×24=576), `encode_sequence()` → (h_seq, z_post, return_preds), and `dream_step(h_prev, z_prev, gru_hidden)` → (h_next, z_next, gru_hidden, return_preds) via `dream_proj`+`dream_gru`+`prior_head`. **This is the latent dynamics Dreamer needs.** CRITICAL DEFECT for A1: `dream_step` takes **no action** — it rolls the *unconditional* market prior. (world_model.py:772-821, and the train-time roll at :712-723.)
- **`src/wealth_bot/harness.py`**: the canonical reward/cost/gate surface. `cost_rt=0.0024` taker default (F9 fix 2026-06-05; maker 0.0010 is an explicit sensitivity, NOT default — MakerCostModel p_fill is 0.21-0.40 empirically, so maker is optimistic). `_simulate` charges round-trip cost on net return (`net = exit_p/entry_p - 1 - cost_rt`). G1 gate = compound_pct>0 in ALL FOUR windows. **This is the reward function and the ship gate — do not hand-roll a second one.**
- **`src/wm/wm_promotion_gate.py`**: held-out COMPOUND only; IC explicitly demoted to within-WM diagnostic. The A1 champion competes here on the same metric — no special-casing.
- **V16 (`dreamerv3_backbone.py`)/V17 (`tdmpc2_backbone.py`)**: BACKBONE STUBS, no trainer, mis-filed in the forecaster zoo. V16's docstring is explicit: *"The actor/critic loop is omitted -- this is the WORLD-MODEL component only."* So nothing here is an actor yet; V16 is the cleanest *future* home but V1.1 is the cheaper start because it's trained+anchored.

## 1. RECOMMENDATION: DreamerV3 imagination actor-critic over V1.1 RSSM

**Choose Dreamer. Concretely:**
- Freeze (or LoRA-fine-tune) the V1.1 RSSM as the world model.
- Add an **action-conditioned transition** (the missing piece — §3).
- Train a stochastic **actor** π(a | h,z) and a **critic** V(h,z) purely on **imagined rollouts** of length H (short — see guardrails), with the actor maximizing λ-returns of the cost-aware reward and the critic regressing them (symlog twohot, as DreamerV3).
- Ground every K imagined-training-batches against a replay buffer of REAL (obs, action, reward, next-obs) transitions; the WM and critic see real data, the actor never trades a real bar it wasn't validated on.

**Why Dreamer over the alternatives (decision, not survey):**
1. **Architectural fit is decisive.** Our WMs are RSSMs. Dreamer's actor-critic consumes (h,z) latents natively. MuZero would require a *value-equivalent* dynamics (a different training objective — dynamics trained to predict reward/value, not reconstruction), i.e. retraining the WM, throwing away the ShIC>0 anti-memorization anchor that is the ONLY thing certifying the WM learned signal. TD-MPC2 similarly wants its own latent-dynamics + Q-ensemble trained end-to-end for control.
2. **Amortized policy > per-decision search at our cadence.** Trading 1 bar at a time on 4h/daily, we don't need MCTS's per-move deliberation; a reactive π(a|s) is sufficient and avoids search compounding WM error over a tree.
3. **Continuation/horizon machinery already in the RSSM.** V16's heads (reward/value/continuation/decoder) are literally the Dreamer head set — the design intent was already Dreamer.

**Honest cost:** Dreamer is the most plumbing-heavy of the three (replay, imagination loop, λ-returns, actor entropy schedule). But it reuses the most existing code, so net effort is lowest.

## 2. Why NOT MuZero / TD-MPC2 (keep them as named falsifiers, not dead ends)
- **MuZero-MCTS:** value-equivalent model + tree search. Two killers for us: (a) requires retraining dynamics → loses the ShIC anchor; (b) MCTS over a *learned* model compounds error with depth — exactly the GIGO trap, amplified by search. Keep the chess MCTS for a *future* exact-simulator sub-problem (e.g., an order-book microstructure sim), not the market.
- **TD-MPC2-MPPI:** strong in continuous control with a *good* learned model and dense reward; trajectory optimization over latent dynamics. It's the right tool IF we move to continuous sizing AND the WM is high-fidelity. Park it as the +k upgrade: if Dreamer's actor underperforms a short-horizon MPPI planner over the same WM, that's evidence the actor (not the model) is the bottleneck — a useful A/B.

## 3. The LOAD-BEARING gap: action-conditioned dynamics (without this, A1 is vacuous)
The V1.1 `dream_step` rolls p(z_{t+1} | h_t) — the market's unconditional next-state. For trading at LO+spot+lev=1 this is *almost* fine because **our action barely affects the market** (we're a price-taker; our position doesn't move BTC). This is the ONE place finance is EASIER than control: the environment dynamics are action-independent. **Implication:** the action conditions the **reward and the portfolio/inventory state**, NOT the market latent transition.

So the correct factorization:
- **Market latent** evolves action-free: keep `dream_step` as-is (this is legitimate, and it means we do NOT need to retrain the WM dynamics — a major de-risk).
- **Augment the agent state** with portfolio state (current position, bars-in-trade, entry price/return-so-far). The actor sees (h, z, portfolio_state).
- **Reward** = action × decoded next-bar return − cost·|Δposition|, where the return is sampled from the WM's return head (or, better, the full predictive distribution — we have twohot logits, use the distribution not the point estimate).
This is a clean, honest design: it sidesteps retraining the WM AND it's *correct* for a price-taker. (For perps/large size, action→market feedback returns and you'd need the conditioned transition; out of scope for spot LO.)

## 4. Action space
- **v1: discrete {flat, long}** (LO+spot+lev=1). Smallest action space, matches the harness `_simulate` long-only logic, easiest to validate, no sizing-overfit surface. Enter/exit/hold = transitions in this 2-state machine; "hold" is implicit (stay in current state).
- **v2 fast-follow: discrete position buckets {0, 0.25, 0.5, 0.75, 1.0}** — sizing as a *capture-rate/risk* layer, the legitimate A1 value-add (a WM with calibrated uncertainty SHOULD size down when its predictive variance is high).
- **continuous [0,1] sizing** only under TD-MPC2/MPPI or a squashed-Gaussian Dreamer actor, and only after discrete wins. Continuous sizing is the biggest silent-overfit risk (it can fit the validation equity curve's exact shape).

## 5. Reward (transaction-cost-aware, laddering to compound)
- Per-step: r_t = a_t · r^{market}_{t+1} − cost_rt · |a_t − a_{t-1}| where r^{market} is **log**-return (so the episodic sum = log compound, the additive form the critic needs), cost via the harness (`cost_rt=0.0024` taker default; report maker 0.0010 as sensitivity, never as headline).
- **Setup/move framing:** the episode is the SETUP — entry to policy-exit, multi-candle. Discount γ near 1 (undiscounted episodic compound, per `DecisionProblemAdapter.discount()` default 1.0) but with a small γ<1 for variance control in the critic.
- **DO NOT reward per-bar IC** (banned). The critic's target is the λ-return of net log-return — i.e., compound, by construction.
- **Asymmetric loss:** a false LONG (enter a loser) costs more than a missed winner (false flat), because drawdown is path-dependent and the harness gate is all-window-positive. Bake this into the reward via a drawdown/CVaR penalty term OR into the critic via a quantile/pessimistic value (§6). Pre-register the asymmetry coefficient; do not tune it on UNSEEN.

## 6. MODEL-EXPLOITATION GUARDRAILS (the heart of A1 — in detail)
The actor's job in imagination is to find the action sequence with the highest imagined return. If the WM is wrong anywhere, the actor will find and exploit that wrongness. Layered defense (defense-in-depth, matching this project's stack):
1. **WM ensemble + disagreement penalty (PRIMARY).** We have 7 anchored forecasters (V1.1, V12, V3, V4, V6, V8, V13). Roll imagination through K of them; penalize the reward by the *cross-model disagreement* of the predicted return (or refuse to act where disagreement > τ). This is the single most effective guardrail: the actor cannot exploit an error that only one WM makes. (PETS/Plan2Explore-style epistemic penalty, but using our REAL diverse ensemble — a genuine asset most projects lack.)
2. **Value pessimism / conservatism.** Critic = lower-confidence-bound: train an ensemble of critics, act on min (or a low quantile) of V. This is the offline-RL CQL/SAC-N lesson — directly counters imagined-value overestimation. Mandatory, not optional, for finance.
3. **Short imagination horizon H.** Model error compounds geometrically in rollout length. Start H=4-8 bars (≈ one setup), NOT 50. Long horizons are where the actor "discovers" free money. Tune H DOWN if imagined return >> real-data return (a divergence alarm).
4. **Real-data grounding (Dyna ratio).** Every imagination batch is interleaved with real-transition replay; the critic is anchored to realized returns. Track imagined-vs-realized reward gap per epoch — if it widens, halt (the actor is leaving the data manifold).
5. **Uncertainty-aware / distributional planning.** Use the WM's full predictive distribution (we have twohot return logits), not the point estimate. Reward = E over sampled futures, with variance penalty. An actor that plans on the mean ignores tail risk; an actor that plans on samples respects it.
6. **Stay-near-data / behavior regularization.** Penalize actions far from a baseline behavior policy (the predict-then-rule policy) — KL(π ‖ π_baseline). Caps how weird the agent can get; degrades gracefully to predict-then-rule.
7. **Out-of-distribution latent detector.** Flag (h,z) far from the training manifold (reconstruction error / posterior-prior KL spike) and force action=flat there. The agent should not trade states the WM never saw.

## 7. THE GIGO GATE (exact pre-condition before wiring an actor onto a WM)
A WM is eligible to be an A1 dynamics model ONLY if it passes ALL of:
- **G-GIGO-1 (genuine learning):** ShIC(h=1) > 0.015 AND ShIC/contiguous-IC ratio > 0.3 (the anti-memorization anchor; rejects the V22/V25 ic1=+0.21/ShIC=0.000 memorizers). Measured, not projected.
- **G-GIGO-2 (held-out signal exists):** the WM's own predict-then-rule policy (threshold the return head, enter long, exit by policy) clears the harness G1 gate — **compound_pct>0 in all 4 windows on held-out** — at taker cost. If the WM can't make a dumb rule positive, an actor over it is GIGO by definition.
- **G-GIGO-3 (calibration):** the WM's predictive distribution is calibrated on held-out (reliability/PIT) — required because guardrails 5-7 trust the WM's uncertainty. An overconfident WM defeats pessimism.
- **G-GIGO-4 (ensemble breadth):** ≥3 WMs independently pass G-GIGO-1/2 so the disagreement penalty (guardrail 1) has real diversity. A single-WM A1 has no epistemic anchor — REJECT.
If a WM fails any of these, **do not wire an actor**. The actor cannot manufacture signal the WM doesn't have; it can only re-weight/time existing signal.

## 8. A/B PROTOCOL vs predict-then-rule (the ship decision)
Predict-then-rule baseline B0: take the WM return head, enter long when E[r_{t+1}] − cost > 0 (or > a pre-registered threshold), exit by fixed policy, size = 1. This is the incumbent and the thing A1 must beat.
- **Train/val/test discipline:** WM frozen; actor+critic fit on TRAIN+VAL imagination/replay only; **UNSEEN segment touched once.** (Project rule: 50/20/20/10, 400-bar purge.)
- **Metric:** held-out UNSEEN **compound** (the only ship metric), with seed-robustness (≥8/10 seeds positive), block-bootstrap p05>0, maxDD<30% — the existing robustness battery.
- **Ship A1 ONLY IF** it beats B0 on UNSEEN compound by a pre-registered margin (use `wm_promotion_gate.should_promote`'s min_improvement, same gate) AND passes seed/bootstrap/DD. A tie or marginal win = ship B0 (simpler, fewer failure modes).
- **Decompose the win (layer KPIs):** if A1 wins, attribute it — is it better *entry timing*, better *exit timing*, or *sizing/abstention*? (The harness emits `layer_kpis`.) An A1 that wins only via abstention in bear regimes is a regime filter, not an alpha — name it honestly.

## 9. Where model-based RL for FINANCE has FAILED (brutal, so we don't repeat it)
- **Imagined-equity ≠ live-equity, every time.** The literature is littered with Dreamer/MuZero-for-trading papers reporting Sharpe 3+ in backtest that are (a) look-ahead-contaminated (G-AUDIT-011 class — the WM or features peek), (b) cost-free or maker-optimistic, or (c) single-seed. Treat any such paper as refuted-until-reproduced under our harness.
- **The actor games the WM** — documented above; the #1 real failure. Most published agents have NO disagreement penalty and NO value pessimism; they overfit the WM's errors and die live.
- **Non-stationarity breaks the frozen WM.** A WM trained on 2021-2024 has a stale dynamics prior in a new regime; the actor confidently trades a model of a market that no longer exists. Mitigation: walk-forward WM refresh + the OOD detector (guardrail 7), and treat the WM champion as perishable.
- **Reward mis-specification → degenerate policies.** Reward per-bar IC or un-costed return and the agent overtrades; reward Sharpe and it learns to do nothing. Our objective (cost-aware episodic compound) is the right one BUT must include the cost term inside imagination, not just in the final backtest, or the actor learns a high-turnover policy that the harness then taxes to death.
- **Tiny effective sample size.** Daily crypto over a few years is ~thousands of bars = a handful of independent regimes. An actor-critic with thousands of params over that is trivially overfit. This is the deepest reason A1 may not beat B0: B0 has ~1 free parameter (the threshold); A1 has thousands. **The bias-variance math favors the simple policy unless the WM signal is strong AND the sizing edge is real.**

## 10. CEILING HONESTY (both ceilings, named)
- **Ceiling 1 (forecaster-quality/GIGO):** the GIGO gate (§7) is the firewall. A1 inherits the WM's signal ceiling exactly — it cannot exceed it, only re-weight it. A2 (raw-data agent) sidesteps THIS ceiling (no intermediate WM to be garbage), which is the real argument for the parallel A2 track.
- **Ceiling 2 (fundamental signal at our resolution):** if daily/4h LO crypto genuinely has weak signal (the honest project history), then BOTH B0 and A1 are near-flat and A1's marginal win is noise. **No architecture escapes Ceiling 2.** A1 is worth building IFF a WM clears G-GIGO-2 with real held-out compound — i.e., build A1 *on the WM that already proves there's signal to time/size*, not as a hail-mary to conjure signal. My honest prior: P(A1 > B0 on UNSEEN, daily/4h, by a robust margin) is LOW; A1's defensible role is a **calibrated sizing/abstention/exit-timing layer** over a WM that already passed the gate.

**Failure modes:** **FM1 — Vacuous imagination (action-free dynamics).** V1.1's `dream_step` rolls the unconditional market prior; if you wrap an actor around it without the §3 factorization (action conditions reward+portfolio-state, NOT the price-taker market latent), the actor has nothing to control and "learns" noise. MITIGATION: implement the portfolio-state-augmented agent state; keep market latent action-free (correct for spot LO price-taker); verify the actor's action measurably changes imagined reward before any training run.

**FM2 — Actor games the WM (the canonical MBRL-finance death).** A frozen imperfect WM has exploitable error pockets; the actor finds them, posts spectacular imagined equity, dies live. MITIGATION (must ALL be on before trusting a number): ensemble disagreement penalty (guardrail 1, using our 7 real WMs), value pessimism/LCB critic (guardrail 2), short H=4-8 (guardrail 3), imagined-vs-realized reward-gap alarm (guardrail 4). Track the gap every epoch; a widening gap is the leading indicator — halt, don't tune through it.

**FM3 — GIGO bypass (wiring an actor onto a memorizer).** Wrapping A1 around a high-contiguous-IC/ShIC-0 WM (the V22/V25 trap) gives a confident agent over a memorized model = catastrophic live. MITIGATION: G-GIGO-1..4 are HARD gates checked mechanically before the actor sees the WM; a single-WM A1 is rejected outright (no epistemic anchor for guardrail 1).

**FM4 — Cost-model bypass inside imagination.** If the cost term is only applied in the final harness backtest and not inside the imagined reward, the actor learns a high-turnover policy that imagination loves and the real cost destroys (this codebase's G-AUDIT-010 + the +501%→+94% MtM/cost history). MITIGATION: cost_rt charged on |Δposition| INSIDE the imagined reward, taker 0.0024 default; the actor must internalize cost, not have it bolted on after.

**FM5 — Look-ahead via the WM (G-AUDIT-011 class).** The WM's encode/normalization or any feature peeking forward silently makes A1's imagined and even held-out numbers fraudulent. MITIGATION: the obs encode is past-only by the existing pipeline contract; re-verify with the leak_probe (`src/wealth_bot/leak_probe.py`) on the A1 obs path specifically — do not assume the forecaster's clean bill transfers to the agent's observation function.

**FM6 — Overfit by parameter count.** A1 has 1000× B0's free parameters over ~thousands of effectively-independent bars; it WILL overfit absent discipline. MITIGATION: WM frozen (no joint training), actor/critic fit on TRAIN+VAL only, UNSEEN touched once, ship only on ≥8/10 seed-positive + block-bootstrap p05>0; behavior regularization (guardrail 6) toward B0 so it degrades gracefully. If A1 only ties B0, ship B0.

**FM7 — Maker-optimism / single-seed self-deception.** Reporting maker 0.0010 or a lucky seed as the headline (the project's recurring inflation pattern). MITIGATION: taker 0.0024 is the headline cost; maker is a labeled sensitivity; every claim carries the seed-robustness and bootstrap CI; tag every number measured-vs-projected.

**FM8 — Regime-abstention masquerading as alpha.** A1 "wins" by going flat in bear windows (a slow regime filter), not by superior timing. MITIGATION: decompose the win via the harness `layer_kpis` (entry-timing vs exit-timing vs sizing/abstention); if the edge is pure bear-abstention, name it as a regime filter (a no-skill control, per the project's exit-axis-NULL lesson), not as agent alpha.

**Open questions:** **Q1 (must measure, not assume):** Does ANY of the 7 anchored WMs clear G-GIGO-2 — predict-then-rule positive compound in all 4 windows on UNSEEN at taker cost? If NONE do, A1 is premature regardless of architecture (Ceiling 2 dominates). This is the first experiment to run; everything downstream is conditional on it. The project's honest history says daily/4h LO is weak — so this may fail, and that result is itself the deliverable.

**Q2:** Is the V1.1 predictive distribution actually CALIBRATED on held-out (PIT/reliability)? Guardrails 5-7 trust the WM's uncertainty; if it's overconfident, pessimism is defeated. Unmeasured — needs a calibration probe before trusting any uncertainty-aware planning.

**Q3:** What is the minimum imagination horizon H at which the imagined-vs-realized reward gap stays bounded? This sets the whole regime — measure by sweeping H and plotting the divergence, don't pick H=15 by analogy to Dreamer-on-Atari (different dynamics, different SNR).

**Q4:** Does the action-free-market-latent factorization (§3) actually hold for our instruments, or does size→slippage feedback matter even at spot LO=1 on thin alts? For BTC/ETH it clearly holds; for low-cap alts the price-taker assumption may break and you'd need the action-conditioned transition (the harder retrain). Measure on the actual universe.

**Q5:** Given ~thousands of effectively-independent bars, is the sample size sufficient for ANY actor-critic to generalize, or is B0's ~1-parameter policy simply the bias-variance-optimal choice at this resolution? This is the deepest open question — it may be that A1 is only worth it at HF/microstructure resolution (where samples are abundant), aligning with the project's standing hypothesis that the real edge needs finer resolution. A1 at daily/4h may be architecturally elegant and empirically pointless; honest measurement, not enthusiasm, decides.

**Q6 (cross-class):** Should compute go to A1-over-WM at all, or to the A2 raw-data agent that sidesteps Ceiling 1? A1 is the right build IF a WM passes G-GIGO; if no WM does, A2 (or finer resolution) is the better bet. This is a portfolio-of-bets allocation question, not an A1-internal one — flag to the user.

## LENS: The Raw-Data Self-Evolving Agent (Class A2): ingest raw price/bars/TIs directly, no forecaster, learn representation + policy end-to-end

**Summary:** A2 (raw-data end-to-end RL/evolutionary agent) sidesteps forecaster-GIGO — there is no WM to memorize the backtest into — but it does NOT escape the two ceilings; it merely relocates the GIGO failure mode from a measurable forecaster diagnostic (ShIC=0) into the policy itself, where it is HARDER to detect because the policy's only output is a P&L curve and a profitable backtest curve is the exact thing a memorizing policy produces. My recommendation for low-SNR, non-stationary markets is NOT model-free online RL (recurrent PPO/SAC will overfit the simulator and chase exploration noise) but OFFLINE RL / Decision-Transformer-on-historical-trajectories as the substrate, wrapped in an OUTER evolutionary/population loop for the "self-evolving" mechanism, with online learning used ONLY as bounded, gated, walk-forward fine-tuning — never as unconstrained continual adaptation. The single most important deliverable is reframing the project's existing, hard-won robustness apparatus (battery Lens A/B/C at `src/strat/battery.py`, PBO/CSCV at `src/strat/pbo_cscv.py`, the N-seed walk-forward + per-seed-OOS gate at `src/wealth_bot/framework/walk_forward.py`, the positive-control and leak-probe) so it gates a POLICY rather than a forecaster — because the project's own history (single-seed LSTM/DQN claims of +44%/+40% debunked to median -7%/-34% at 10-seed audit, documented in `walk_forward.py:9-12`) is already the canonical A2 over-fitting incident, recorded before A2 was named. The honest verdict: A2 has a higher ceiling and is the correct long-horizon bet, but at daily/4h resolution it inherits the SAME fundamental-signal ceiling as A1 (Ceiling 2 is architecture-invariant), so its first job is not to "win" but to PROVE it can beat a no-skill control on UNSEEN compound — and the existing apparatus already shows that bar is rarely cleared at coarse resolution.

# Class A2: the Raw-Data Self-Evolving Agent — what it takes to be world-class

## 0. Where this plugs into what already exists (do not rebuild)

The contract for A2 is ALREADY written in the codebase: `projects/chess_zero/az/game_adapter.py:100-135` defines `DecisionProblemAdapter` — the "RL/POMDP superset for UNBOUNDED problems," single-agent vs stochastic env, partial observability (`observation()` = the chimera feature vector, past-only), continuous action (`action_spec()` = position in [-1,1] under LO+spot+lev=1), per-step reward (`reward()` = realized step P&L net cost). It explicitly carries the key disclaimer: `has_exact_simulator = False`. That single flag is the entire fork:
- **A1 (WM-consuming)** sets `dynamics()` = the WM (`src/wm/*`) and plans/imagines over it → DreamerV3 (`src/wm/v16/dreamerv3_backbone.py`) / TD-MPC2 (`src/wm/v17/tdmpc2_backbone.py`). Those two stubs are A1 by construction (RSSM / latent dynamics + value head). They are NOT A2 material.
- **A2 (raw-data)** does NOT call `dynamics()` at all. It learns representation + policy directly from the observation stream. No learned forward model → **no model to GIGO into.**

So A2's build is: implement `DecisionProblemAdapter` for crypto + the simulator backing `reward()` + the policy/learner + reuse the existing robustness stack as the policy gate. The AlphaZero machinery in `projects/chess_zero/az/` (mcts.py, net.py, selfplay.py, **train_robust.py** with its champion-gate + curriculum + OOM guards) is the *infrastructure pattern* A2's outer loop copies — NOT the planner (chess has an exact simulator; the market does not).

## 1. The class choice — recommendation + why

The prompt's four candidates, ranked for low-SNR non-stationary markets:

**(D) Offline RL / Decision-Transformer on historical trajectories — RECOMMEND as the substrate.**
- Why it wins here: the market env has NO exact simulator and online interaction is impossible at research time (you cannot re-run 2021). Offline RL is *designed* for "learn a policy from a fixed dataset of logged trajectories you cannot extend." A Decision Transformer (return-conditioned sequence model: condition on a target return-to-go, autoregress actions) turns policy learning into a SUPERVISED sequence-modeling problem — which is exactly the regime our infrastructure already validates (the `signal_picker`/LGBM path in `wealth_bot/framework` is the degenerate 1-step version of this).
- Why it sidesteps a specific A2 trap: model-free online RL (PPO/SAC) needs an environment to step. Our "environment" is a SIMULATOR over historical bars; stepping it repeatedly = the policy gets thousands of passes over the SAME finite history → it memorizes the path. Offline RL with a held-out trajectory split makes the train/test boundary a first-class object, the same way the forecaster's 50/20/20/10 split does.
- The honest caveat: offline RL has its own pathology — *distributional shift / extrapolation error* (the policy queries actions the dataset never logged; the value estimate is hallucinated). Conservative offline-RL (CQL / IQL) or the simpler return-conditioned DT (which never bootstraps a value, so no extrapolation blowup) are the safe choices. DT is my specific pick: no value-function extrapolation, and "self-evolving" maps cleanly onto re-conditioning on a higher return-to-go.

**(A) Model-free deep RL (recurrent PPO/SAC) — NOT recommended as the primary.** Recurrent (the LSTM/GRU is mandatory for partial observability — a feedforward policy on a single bar is blind to the multi-bar MOVE) but: (i) on-policy PPO is sample-inefficient and you only have one history; (ii) exploration in a market is *adverse* — random exploratory trades pay real costs and teach nothing (see §4 exploration); (iii) it is the architecture that produced the project's worst over-fit incident (the debunked DQN). Keep it as a BASELINE to beat, not the ship candidate.

**(C) Evolutionary / population — RECOMMEND as the OUTER loop, not the inner learner.** A population of policies + selection-on-held-out is the cleanest "self-evolving" mechanism AND a built-in overfit guard (you select survivors on a validation window, not the training window). But evolution as the *sole* representation learner is sample-hungry and gradient-free — wasteful for the rich gradient signal a DT gives you. Use it to evolve *hyperparameters / reward-shaping / population diversity*, with DT/offline-RL as the inner gradient learner. This is the NEAT-over-deep-learner hybrid.

**(B) Model-based-from-raw — this is A1 in disguise.** If you learn dynamics from raw data and plan over it, you have built a WM and re-acquired Ceiling 1. Excluded from A2 by definition.

**Net recommendation: Decision-Transformer / offline-RL inner learner + evolutionary/population outer loop + recurrent-PPO as the no-skill-ceiling baseline. Online learning permitted ONLY as gated walk-forward fine-tuning (§3).**

## 2. The HARD problems specific to A2 (and the concrete mitigation for each)

**(H1) Representation learning from low-SNR raw data.** The agent must learn its own features from f34 chimera + raw bars — the thing the forecaster spent V1.1/V12/etc. building. With SNR this low, an end-to-end policy will find the representation that best fits the *backtest path*, not the *market structure*. Mitigation: (a) do NOT learn the representation from scratch — *warm-start the encoder* from one of the 7 anchored forecasters' encoders (they have a PROVEN anti-memorization anchor: ShIC>0 via the information bottleneck + reconstruction). This is subtle: it imports the forecaster's REPRESENTATION (which is genuinely-learnt) without consuming its PREDICTION (which would re-introduce A1 GIGO). The encoder is frozen or low-LR; the policy head is fresh. (b) Auxiliary self-supervised losses (reconstruction, contrastive next-bar) as a representation regularizer — the same information-bottleneck logic that gives ShIC>0 applies to a policy's representation.

**(H2) Credit assignment over multi-bar MOVES with sparse/delayed compound reward.** The unit of trading is the SETUP across a MOVE (`MEMORY.md` founding framing). A reward only at trade-close is sparse and the gradient to the entry decision is long-range. This is precisely where DT shines (return-to-go conditioning makes the long-horizon return a direct input, not a discounted bootstrap) and where PPO struggles (GAE with a market value-function is noisy). Mitigation: episode = one SETUP-to-EXIT, reward = realized compound net cost (laddering up to the project objective, per `DecisionProblemAdapter.reward()` docstring: "reward must ladder up to compound"). Do NOT shape with per-bar IC — that is the banned objective and re-introduces the per-candle fallacy.

**(H3) The OVERFitting trap = the RL analog of ShIC=0 — THE central deliverable.** A forecaster that memorizes shows contiguous-IC high / ShIC≈0 (the V22/V25 incident: ic1=+0.21, ShIC=0.000). A policy that memorizes shows **high backtest compound / collapses on UNSEEN or on a shuffled/surrogate market**. The detection apparatus — the A2 analog of the ShIC test:
  - **The shuffled/surrogate-market control (the ShIC analog).** Re-train/re-evaluate the policy on phase-randomized or block-shuffled returns that preserve marginal distribution but destroy temporal structure. A genuinely-skilled policy degrades to ≈0; a memorizing policy still "profits" because it has memorized the path → that residual profit is the ShIC=0 signature for a policy. **This is the single most important gate to build and does not yet exist for policies.**
  - **PBO via CSCV (`src/strat/pbo_cscv.py`) applied to the POLICY POPULATION.** When the evolutionary loop produces 10^3–10^5 candidate policies, the in-sample-best is by construction the most path-fit. PBO answers "does our SELECTION PROCESS produce OOS under-performers?" — exactly the population-selection risk. Ship rule: PBO < 0.10. This is already two-sided-validated (`pbo_cscv.py:128-161`: rejects a noise family, accepts a genuine edge).
  - **The N-seed audit + per-seed-OOS gate (`src/wealth_bot/framework/walk_forward.py`).** This module's own provenance IS the A2 over-fit incident: "single-seed claims of +44%/+40% were debunked to median -7%/-34% at 10-seed audit" (`walk_forward.py:9-12`). A policy must be ≥70% seeds OOS-positive (`PER_SEED_OOS_GATE_PCT = 70.0`) because LIVE deploys ONE seed; the ensemble headline masks per-seed instability. RL policies are *more* seed-sensitive than supervised models (the env interaction amplifies seed variance), so this gate is MORE binding for A2, not less.

**(H4) Non-stationarity / regime shift.** A policy fit on 2021 bull does not transfer to 2022 bear. Mitigation: walk-forward training with rolling re-fit (the existing 50/20/20/10 split + 400-bar purge generalizes), regime-conditioned evaluation (the firm's `regime_playbook.py` / `market_state.py` give the regime labels), and the `worst_month_pct > -10%` + monthly-positive ≥0.60 temporal barometer (battery Lens C) which directly tests regime robustness. Critically: report per-regime compound, never just aggregate — an aggregate +X% that is +40% in one bull and -30% across two bears is a regime bet, not an edge.

**(H5) Survivorship / look-ahead leaking through the SIMULATOR.** A2's biggest *infrastructure* risk: the simulator backing `reward()` is now in the training loop, so any leak there is amplified by the optimizer (the policy will *find and exploit* any forward-peek). The existing leak machinery transfers: `leak_probe.py` (the cadence-robust `relative_leak_test` is the verdict to use — the absolute and shift-spectrum verdicts BOTH FAILED on coarse bars, documented at `leak_probe.py:122-131`), and the MtM-no-double-count simulator invariant (CLAUDE.md "Backtest Simulator Invariants"). Survivorship: the universe must be point-in-time (no using today's top-50 as 2021's universe). The simulator must charge the calibrated cost model — `MakerCostModel` p_fill is 0.21–0.40 empirically, NOT the 0.80 default (CLAUDE.md), and a policy trained against an optimistic fill model learns trades that won't fill live.

## 3. What "self-evolving" should concretely mean here

NOT "an unsupervised agent that adapts online forever" — that is the fastest route to silently over-fitting the most recent (noisy) window. Three concrete mechanisms, in priority order:

1. **Outer evolutionary/population loop (PRIMARY).** A population of policies; each generation: train inner DT, evaluate on a held-out validation window, SELECT survivors by val-compound + diversity, mutate hyperparameters/reward-shaping/architecture. This is "self-evolving" with a built-in overfit firewall (selection is on held-out, not train). It maps directly onto the chess `train_robust.py` **champion-gate** pattern: a new policy only replaces the champion if it beats it on a held-out gate — the exact monotonic-promotion discipline the chess engine already uses, and the `MEMORY.md` chess→crypto lesson ("the monotonic promotion gate is what makes experimenting safe").

2. **Skill-library / curriculum (SECONDARY).** A Voyager-style library (the repo already has `scripts/autonomy/skill_library.py`) where validated sub-policies ("breakout-rider," "mean-revert-in-range") accumulate and compose. Curriculum = train on easy regimes first (clear trends) then hard (chop). Concrete, but only after the population loop is proven.

3. **Online/continual learning (LAST, and bounded).** Permitted ONLY as walk-forward fine-tuning behind a gate: re-fit on the newest closed window, but the updated policy must PASS the same N-seed + PBO + shuffled-market gate against the prior champion before it goes live. Unconstrained continual learning = the catastrophic-forgetting + recency-over-fit trap. The `DecisionProblemAdapter.dynamics()` docstring already warns of the analog: "a policy that PLANS over an imperfect WM learns to exploit the WM's ERRORS... the robustness/eval-trust stack is MANDATORY here, not optional."

## 4. Exploration in a market env (the A2-specific exploration problem)

Standard RL exploration (ε-greedy, entropy bonus) is *actively harmful* here: a random exploratory trade pays real spread+fee+slippage and the resulting reward is dominated by market noise, not by the information value of the action. Mitigation: (a) prefer offline RL / DT where "exploration" = conditioning on diverse return-to-go targets over the FIXED logged dataset (no live cost); (b) if using PPO, the exploration is over POLICY SPACE via the population/evolutionary loop, not over per-trade randomization; (c) curiosity/intrinsic-reward methods are a trap in low-SNR data — they reward novelty, and a non-stationary market is endlessly novel, so the agent chases noise. State this explicitly: **in a low-SNR market, the correct exploration budget is near-zero at the action level and is instead spent at the population/architecture level.**

## 5. The robustness apparatus applied to the POLICY (reuse, don't rebuild)

Everything below already exists and was hardened against a real red-audit; the work is *re-pointing it at a policy's trade-return stream* instead of a forecaster's signal:
- **Walk-forward, held-out compound:** `walk_forward.py` N-seed + bootstrap CIs (`bootstrap_trade_returns`) + per-seed-OOS gate + the ML-n_eff≥30 floor (which generalizes: a policy with <30 trades across all segments has no statistical power, exactly as LGBM does).
- **The battery (Lens A/B/C):** `src/strat/battery.py` — feed it the policy's UNSEEN per-trade returns. Lens A needs all-4-windows-positive AND n≥15 AND n_eff≥15 AND jackknife-2/3 > 0 AND bootstrap-p05 > 0 AND maxDD<30%. The n_eff/jackknife guards catch a policy whose backtest is carried by 1-2 lucky trades (the concentration ghost in `battery.py:163-166`).
- **Anti-overfit (selection-bias):** `pbo_cscv.py` on the policy population.
- **The no-skill control (THE control):** `src/strat/positive_control.py` proves the gate HAS POWER (ships a genuine edge), and the firewall (`src/strat/firewall.py`) is the cost-matched RANDOM-ENTRY null the policy MUST beat on held-out. For A2 add the shuffled-market control (§H3) as the policy-specific no-skill baseline. A policy that does not beat random-entry on UNSEEN is rejected REGARDLESS of its backtest compound.
- **Leak verdict:** `leak_probe.relative_leak_test` (cadence-robust).

## 6. The honest two-ceiling verdict for A2

- **Ceiling 1 (forecaster-quality / GIGO):** A2 SIDESTEPS this — no WM to be garbage. This is its real advantage and the reason it has a higher ceiling. BUT it relocates the failure mode: instead of a measurable ShIC=0 on a forecaster, you get a memorizing policy whose only symptom is a good backtest curve. That is HARDER to catch, which is why the shuffled-market control (§H3) is non-negotiable.
- **Ceiling 2 (fundamental signal at our resolution):** A2 does NOT escape this. No architecture does. If daily/4h LO crypto has weak signal (the project's repeatedly-confirmed honest history — the dead-list, the 30m/15m cost-cliff, the "active alpha unproven" consensus in `MEMORY.md`), then a perfectly-trained A2 policy converges to the same weak edge, possibly to flat-after-costs. A2's higher ceiling is only REALIZABLE if the signal exists at the resolution it operates on — which is itself the open empirical question the project keeps hitting.

**Therefore A2's first deliverable is not a profitable bot. It is: an A2 agent + the shuffled-market control + the existing gates, run on UNSEEN, producing an HONEST verdict on whether an end-to-end policy beats the no-skill control at a given resolution. If it cannot beat random-entry on UNSEEN at 4h (the most likely outcome given history), that is a real finding that bounds Ceiling 2 from a new direction — and it is cheap to obtain because the gates already exist.** The expensive, high-ceiling bet (HF/microstructure resolution where Ceiling 2 may lift) is only worth A2's full build cost AFTER the cheap 4h null result confirms the coarse-resolution ceiling — otherwise you have built a sophisticated agent to rediscover a known wall.

**Failure modes:** **(F1) The policy memorizes the backtest path = the RL ShIC=0 — and it is INVISIBLE.** Unlike a forecaster (where ShIC=0 is a single measurable number), a memorizing policy's only output is a profitable backtest curve, which is indistinguishable by inspection from a real edge. THIS IS ALREADY THE PROJECT'S WORST INCIDENT: `walk_forward.py:9-12` records single-seed DQN/LSTM claims of +44%/+40% debunked to median -7%/-34% at 10-seed audit. MITIGATION: the shuffled/surrogate-market control (re-train on phase-randomized returns — a genuine policy →≈0, a memorizer still "profits") + mandatory N-seed≥10 with the 70%-OOS-positive gate + PBO<0.10 on the population. Never report a single-seed backtest number.

**(F2) Online/continual self-evolution silently over-fits the most recent noisy window + catastrophically forgets.** "Self-evolving" interpreted as unconstrained online adaptation is the fastest route to ruin. MITIGATION: online learning ONLY as gated walk-forward fine-tuning behind the champion-gate; the updated policy must re-pass N-seed + PBO + shuffled-market against the prior champion before promotion. The chess `train_robust.py` champion-gate is the exact pattern.

**(F3) The simulator becomes the adversary the optimizer exploits.** Because `reward()` is now inside the training loop, ANY look-ahead, MtM-double-count, or optimistic-fill in the simulator is actively found and amplified by the policy (gradient pressure toward the leak). A forecaster only passively benefits from a leak; a policy hunts it. MITIGATION: `leak_probe.relative_leak_test` (the cadence-robust verdict — the absolute/shift-spectrum verdicts FAILED on coarse bars per `leak_probe.py:122-131`), the MtM-no-double-count invariant, point-in-time universe (no survivorship), and the CALIBRATED cost model (p_fill 0.21-0.40 not the 0.80 default — a policy trained on optimistic fills learns trades that won't fill).

**(F4) Exploration chases noise.** ε-greedy / entropy bonus / curiosity in low-SNR non-stationary data rewards endless novelty (the market is always novel) and pays real costs for nothing. MITIGATION: prefer DT/offline-RL (exploration = return-to-go conditioning over the fixed dataset, zero live cost); spend the exploration budget at the population/architecture level, near-zero at the action level. State this explicitly so the next builder does not reach for a standard entropy coefficient.

**(F5) Mistaking A2-sidesteps-Ceiling-1 for A2-escapes-both-ceilings.** A2's higher ceiling is real but only realizable if signal exists at the operating resolution. At 4h/daily LO crypto the project has repeatedly confirmed the signal is weak; a perfect A2 policy converges to the same weak/flat edge. MITIGATION: gate the full A2 build behind a cheap 4h null result — run the agent + shuffled-control + existing gates first; if it cannot beat random-entry on UNSEEN at 4h, that bounds Ceiling 2 and says the expensive HF-resolution build is where the ceiling might lift, NOT another coarse-resolution agent.

**(F6) Reward-shaping re-introduces the banned per-bar IC objective.** Tempting to densify the sparse compound reward with a per-bar predictive term. That is the banned per-candle metric and re-imports the per-candle fallacy. MITIGATION: episode = SETUP-to-EXIT, reward = realized compound net cost, full stop — per `DecisionProblemAdapter.reward()`: "reward must ladder up to compound; per-bar IC is BANNED as the objective."

**(F7) Sample-size illusion.** RL eats data; a policy with <30 trades across all segments has no statistical power (the `ML_PATH_N_EFF_MIN = 30` floor in `walk_forward.py:45`). A coarse-bar policy naturally produces few trades → tops out below ship-tier by sample size alone (exactly as `positive_control.py:33-38` documents for the crossover edge). MITIGATION: enforce the n_eff floor; do not chase ship-tier by over-trading a coarse cadence (that whipsaws and the gate correctly rejects it).

**Open questions:** **(Q1) Does an end-to-end A2 policy beat the cost-matched random-entry firewall on UNSEEN at 4h?** This is THE measurement and it is unknown — must be run, not assumed. Given the project's "active alpha unproven" consensus and the 30m/15m cost-cliff, the prior is NO. A clean null here is itself a valuable finding (bounds Ceiling 2 from the policy direction). Cheap to obtain (gates exist).

**(Q2) Does the shuffled/surrogate-market control actually discriminate a memorizing policy from a skilled one in practice?** The logic is sound (it is the ShIC analog) but it has NOT been built or validated two-sided for a POLICY (only for forecasters). It needs the same positive-control + negative-control proof the forecaster ShIC test has. Until validated, "the policy passed the shuffle test" is an unproven claim.

**(Q3) How much seed-variance does an RL policy actually have on this data vs a supervised model?** Hypothesis: more (env interaction amplifies it), which would make the 70%-OOS-seed gate more binding. Must be MEASURED — if 10-seed variance is enormous, even a real edge may fail the gate, and the gate threshold may need recalibration FOR POLICIES specifically (don't assume the forecaster-calibrated 70% transfers).

**(Q4) Does warm-starting the encoder from an anchored forecaster (ShIC>0) genuinely transfer a non-memorizing representation, or does the fresh policy head re-memorize through a frozen good encoder?** Unknown. The clean experiment: frozen-good-encoder + fresh-head vs scratch-encoder, both through the shuffled-market control — does the warm-start lower the shuffle-residual profit?

**(Q5) At what resolution does Ceiling 2 actually lift for a POLICY?** The project hypothesis is HF/microstructure. But A2's representation-learning-from-raw advantage might surface signal at an intermediate resolution (dollar bars, 1m+liq — note D71/D72 in MEMORY.md found *information* at 1m on movers but no continuation edge). Whether A2 can convert 1m information into a held-out-compound POLICY edge where hand-built features could not is genuinely open and is A2's single most interesting upside test.

**(Q6) Evolutionary outer loop vs single-DT: does population selection on held-out actually reduce PBO vs picking the single best-on-train policy?** Assumed yes (selection on val, not train) but the population itself is a multiple-comparisons machine — PBO could stay high if the val window is too short. Must measure PBO as a function of population size and val-window length.

## LENS: GIGO + the 2-ceiling gating (the project's conscience / skeptic)

**Summary:** There are two independent, multiplicative ceilings and no agent architecture escapes both. Ceiling 1 (forecaster-quality / GIGO) bounds any WM-CONSUMING agent (Class A1: a Dreamer/MuZero/MCTS planner over a forecaster) — a planner is a model-EXPLOITER, so it does not average out a weak WM, it AMPLIFIES its errors into confidently-wrong, concentrated bets; A1's realizable edge is upper-bounded by how genuinely the WM learned, which the project already measures with ShIC and a held-out compound probe. A2 (raw-data self-evolving agent) sidesteps Ceiling 1 by never trusting a learned dynamics model, but pays it back as sample-inefficiency and offline-RL extrapolation risk. Ceiling 2 (fundamental signal at our resolution) gates BOTH classes equally and is, per the HARD entries in the dead-list (D13/D17/D44/D45), the binding constraint at daily/4h/dollar long-only: if exploitable multi-candle structure does not EXIST at our bar resolution, no agent — A1 or A2 — manufactures it, and the honest answer is HF/microstructure DATA, not a fancier agent. The make-or-break sequencing is therefore: (0) prove structure EXISTS at this resolution before building any agent, (1) prove the specific WM genuinely learned before wiring ANY A1 planner onto it, (2) only then spend GPU-days on the agent. Skipping (0) or (1) is how you burn GPU-days planning over noise and ship a confidently-wrong bot onto real capital.

## The two ceilings, stated precisely

**Ceiling 1 — Forecaster-quality / GIGO (binds A1 only; A2 sidesteps it).**
A Class-A1 agent (DreamerV3-style imagination rollouts, or a MuZero/MCTS planner — the exact PUCT machinery that already exists and is load-bearing at `projects/chess_zero/az/mcts.py`) acts by *querying a learned model of the world*. Its policy quality is bounded above by the fidelity of that model on the states it actually visits. If the WM has high contiguous IC but ShIC≈0 (memorization — recorded historically as V22/V25 ic1=+0.21 ShIC=0.000; canonical GIGO precedent D02 = voladj WM "IC=0.10" that was predicting vol not returns, raw IC=0.017), then the model is a lookup table of the train set. A planner over a lookup table produces plans that are *internally optimal and externally meaningless*. A2 (raw price/bars/TIs → act end-to-end) never instantiates a learned dynamics model it can be fooled by, so Ceiling 1 does not apply to it — at the cost of paying Ceiling-1's GIGO back in a different currency (below).

**Ceiling 2 — Fundamental signal at our resolution (binds A1 AND A2; no architecture escapes).**
This is the existence question: at daily/4h/dollar-bar long-only spot, is there exploitable multi-candle structure AT ALL? The dead-list answers, at the mechanism level (HARD scope = survives apparatus/resolution-preserving changes): D13 (IC-as-primary HARD-archived), D17 (cross-sectional prediction IC≈0 across 6 architectures at 1d/4h/3d/weekly), D44 (1-5%/day via daily-bar prediction needs IC~0.6; measured ~0 six ways → math-infeasible without ruinous leverage), D45 (bar-level entry-timing info "lives BELOW the bar" — 0/14 scalp, 0/8 swing genuine). If the mutual information between (our features at time t) and (the multi-candle move after t) is near zero, then BOTH an A1 planner and an A2 RL agent are optimizing over noise. The agent is irrelevant; the answer is a resolution change (tick/LOB/microstructure data), which is a DATA decision, not an architecture decision.

The two are **multiplicative**: realized_edge ≤ Ceiling2_signal × Ceiling1_fidelity (A1) or ≤ Ceiling2_signal × policy_extraction_quality (A2). A perfect planner over a perfect WM still yields zero if Ceiling 2 is zero.

---

## GATE A — The exact bar a FORECASTER must clear before ANY A1 agent is wired onto it

This is a hard, pre-registered, mechanically-checked gate. A forecaster that fails ANY clause is **forbidden as an A1 substrate** — you may still ship it as a position-SIZING input (lower-stakes), but a planner is denied. Concrete numbers (grounded in existing apparatus):

1. **Genuine-learning (anti-memorization), MANDATORY.** ShIC(h=1) > 0.015 AND ShIC/contiguous-IC ratio > 0.3 (the existing `GATE_IC_MIN=0.015` + the CLAUDE.md Shuffled/Contiguous > 0.3 gate). This is the GIGO firewall — it is the ONE clause A2 does not need. Rationale: ShIC≈0 = lookup table = a planner amplifies it. Measured on the OOS segment, not train.
   - *Tighten for A1 specifically:* because a planner compounds model error over a multi-step rollout horizon H, require ShIC to hold at the rollout horizons the planner will actually query (h ∈ {1,4,16} per ACTIVE_HORIZONS), not just h=1. A WM genuine at h=1 but memorized at h=16 produces good 1-step priors and garbage 16-step imagined trajectories — exactly where Dreamer/MuZero spend their search.
2. **Held-out compound value over the right null, MANDATORY.** Using the EXISTING `src/strat/wm_value_probe.py` protocol: on the UNSEEN segment the WM-driven policy must beat BOTH (a) buy-and-hold AND (b) regime-MATCHED, cost-matched random entry across ≥10 seeds (mean + spread), with positive margin. This is already the `should_promote` compound-only gate (`src/wm/wm_promotion_gate.py`, which *deliberately excludes IC* — `no_ic_gate` invariant). The probe already runs the BOTH-beat test; do not weaken it.
3. **Seed-robustness, MANDATORY.** ≥8/10 training seeds clear clauses 1–2. A single lucky seed that passes is the multiple-comparisons trap; the dead-list is full of seed-flattered findings (D21 continuous TSMOM, D45's "1 BTC hit was seed-dependent"). For an A1 substrate the WM itself must be seed-stable, not just the downstream policy.
4. **Regime-coverage, MANDATORY.** Clauses 1–2 must hold (not collapse) across trending / mean-reverting / high-vol regimes separately — not just pooled. A WM genuine only in the bull regime that dominates the sample is a beta proxy (the D04/D05 failure mode); a planner will confidently over-trade the regime it was fit to.
5. **Cost honesty, MANDATORY.** The compound number uses the TAKER default (0.0024 round-trip per the F9 apparatus fix in `harness.py`/`wm_value_probe.py`), not optimistic maker (0.0010). Maker is a sensitivity, never the gate value (D43: real p_fill 0.21–0.40, not 0.8).

**Gate-A verdict rule:** all 5 clauses PASS → eligible A1 substrate. Any FAIL → A1 forbidden; the WM is at best a sizer/filter. This is a one-way ratchet (monotonic floor, as `wm_promotion_gate` already enforces).

---

## GATE 0 — The TEST for whether exploitable structure EXISTS at our resolution at all (the foundational question)

This precedes Gate A and precedes ALL agent-building. It is a model-FREE existence test, because the whole point is to answer the question *before* committing to an architecture (so a failed agent can never be blamed on the agent).

- **Oracle-gap / harvestability test (model-free).** Construct the hindsight oracle on the multi-candle move (the `src/oracle/` decomposition apparatus already does this), then ask: is there a PAST-ONLY conditioner whose presence raises realized capture above a regime-matched random-entry null, held-out, ≥8/10 seeds, OOS→UNSEEN persistent? If the oracle ceiling is high but NO past-only conditioner narrows the gap, the information is not extractable at this resolution (this is exactly the D45 finding).
- **Mutual-information / IC-at-the-move-scale floor (diagnostic only, not objective).** Per founding framing IC is banned as a PRIMARY metric, but as a *one-sided existence diagnostic* it is decisive in the negative: if even shuffled-controlled predictive correlation at the move scale is indistinguishable from zero across the universe and across cadences (D17: IC≈0 six ways), the existence hypothesis is REFUTED at this resolution.
- **The verdict that matters:** if Gate 0 FAILS (no exploitable structure at daily/4h/dollar), **STOP — do not build A1 or A2.** The answer is a DATA/resolution change (tick-level, LOB, liquidation microstructure — note D71/D72 show the "meat" exists at 1m for movers but continuation-given-onset has no internal info → needs an EXTERNAL signal, AUC>0.58). A fancier agent is the wrong lever; this is the single most expensive mistake the project can make and the dead-list shows it has been made before.

---

## How MODEL-EXPLOITATION is the GIGO-AMPLIFIER (the mechanism, not a slogan)

A planner does not consume the WM's average prediction — it *searches for the action sequence that maximizes the WM's predicted value*. That argmax is an adversary against the WM's error surface:
- **MCTS/MuZero (the chess machinery, `mcts.py`):** PUCT selects `argmax_a [Q + c·P·√ΣN/(1+N)]`. If the WM's value head is wrong-but-confident on a rare state, the search will *preferentially expand toward that state* (high predicted value attracts visits). The planner systematically finds and exploits the WM's hallucinations — the opposite of averaging them out.
- **DreamerV3 imagination:** the actor is trained on-policy inside the WM's imagined rollouts. Compounding one-step model error over an H-step rollout means the actor optimizes a trajectory distribution that diverges from reality as H grows — and it does so *most* where the WM is least constrained by data (the tails, the regime transitions — exactly where ShIC-at-horizon matters).
- **The chess precedent is the in-repo microcosm:** pure self-play DEGRADED an imitation net because its self-generated games were below the teacher it copied (memory: chess learning-engine). Optimizing a PROXY (self-play value) degraded the REAL objective (held-out strength). Substitute "WM-predicted return" for "self-play value" and "held-out compound" for "real strength" and you have the crypto GIGO risk verbatim.

Consequence: a weak WM + a planner is strictly WORSE than a weak WM + a simple threshold policy, because the planner is a high-variance error-seeker. This is why Gate A clause 1 is non-negotiable for A1 and why A1's stakes are higher than a sizer's.

---

## A2 does NOT get a free pass — where its ceiling re-enters

A2 sidesteps Ceiling 1 but inherits two failure modes the skeptic must name:
- **Offline-RL extrapolation (the GIGO-in-disguise):** a self-evolving agent trained on logged/backtested data extrapolates its Q-function off-distribution and prefers actions it has never seen costed — the same confidently-wrong pathology, now in the policy/value net instead of a dynamics model. Mitigation: A2 must clear the SAME Gate 0 (existence) and the SAME held-out compound + regime + seed + cost clauses (Gate-A clauses 2–5); only clause 1 (ShIC of a WM) is replaced by an equivalent policy-overfitting control (e.g., behavior-cloning-anchored / conservative offline-RL + the regime-matched random null).
- **It still hits Ceiling 2 dead-on.** A2 over noise is just a slower, more expensive way to discover there is no signal.

---

## The honest sequencing (what must be measured-TRUE before each build step)

1. **Gate 0 (existence, model-free).** If FAIL → STOP, escalate the DATA decision (HF/microstructure). Do not proceed to any agent. *Cost of skipping: GPU-days planning over noise.*
2. **Forecaster training + Gate A (genuine-learning + held-out compound + seed + regime + cost).** If the WM fails clause 1 → it is NEVER an A1 substrate (sizer at most). *Cost of skipping: a confidently-wrong planner on real capital.*
3. **Only now: build the A1 agent over the gated WM** (or the A2 agent under its equivalent controls). Re-run the held-out compound + seed + regime battery on the AGENT (the WM passing is necessary, not sufficient — the agent can still overfit the search).
4. **Champion gate (already exists):** promote on UNSEEN compound only, monotonic floor, IC excluded — `wm_promotion_gate.should_promote`.

This sequence guarantees we never spend GPU-days on an agent over noise (Gate 0) and never wire a planner onto a memorizer (Gate A clause 1). Each gate is a measured-true precondition, not a projection.

---

## MIS-FILING flag (correctness of the current zoo)

V16=DreamerV3 and V17=TD-MPC2 are ACTING-AGENT backbones (A1 class — they consume a WM and plan), not forecasters. They are currently mis-filed in the forecaster zoo with no trainer. Under this lens that is not cosmetic: filing a planner as a forecaster invites someone to score it with forecaster gates (ShIC/IC) when it must be scored with Gate-A-substrate + agent-level held-out compound. Re-file V16/V17 into an explicit A1 agent layer, and make the WM-substrate-Gate-A check a hard precondition in their (future) trainer's preflight.

**Failure modes:** **(1) "The planner will average out WM noise."** FALSE and the most dangerous belief in this lens. A planner is an argmax over the WM's value surface — it SEEKS the WM's errors (high-predicted-value hallucinations attract MCTS visits / Dreamer rollouts). Mitigation: Gate-A clause 1 (ShIC>0.015 + ratio>0.3 AT the rollout horizons, not just h=1) is a HARD precondition; a sub-gate WM is forbidden as A1 substrate, period.

**(2) Memorization masquerading as skill (the V22/V25/D02 trap).** High contiguous IC with ShIC≈0 looks like a great forecaster and produces a planner that is internally optimal, externally random. Mitigation: ShIC is the gate, never contiguous IC; measure on OOS; voladj/any-shortcut targets banned (target_prefix="target_return").

**(3) Blaming the agent for a Ceiling-2 (existence) failure.** Spending GPU-days iterating Dreamer/MuZero variants when the real problem is no signal at this resolution. Mitigation: Gate 0 (model-free oracle-gap + ShIC-controlled existence test) runs BEFORE any agent; a FAIL routes to a DATA/resolution decision (HF/LOB), never to a fancier architecture. Dead-list D44/D45/D17 are the standing evidence this resolution is signal-thin.

**(4) Seed-flattered / single-window A1 'wins'.** A planner has more knobs and more variance than a threshold rule, so it overfits the search to one seed/window more easily (D21, D45 precedents). Mitigation: ≥8/10 seed-robustness AND OOS→UNSEEN persistence AND per-regime (not pooled) at the AGENT level, not just the WM level.

**(5) Optimistic cost erasing a fragile A1 edge.** A planner that trades more reacts more to the maker-vs-taker assumption (D43: real p_fill 0.21–0.40). Mitigation: TAKER 0.0024 is the gate value (F9 fix already in harness); maker is a one-sided sensitivity only.

**(6) A2 treated as GIGO-immune.** It is immune to WM-GIGO but not to offline-RL extrapolation (confidently-wrong Q off-distribution) nor to Ceiling 2. Mitigation: A2 clears the same Gate 0 + held-out/seed/regime/cost battery; clause 1 is replaced by a policy-overfitting control (conservative offline-RL / BC-anchor + regime-matched random null), not waived.

**(7) Projected-vs-measured confusion.** Calling a WM "agent-ready" because its IC looks good (projection) instead of because it cleared the held-out compound + ShIC + seed + regime battery (measured). Mitigation: every Gate-A/Gate-0 verdict must cite the UNSEEN-segment number and the seed/regime breakdown; no promotion on a projected or train/OOS number (wm_promotion_gate already enforces UNSEEN-only, IC-excluded).

**(8) Mis-filed agents scored as forecasters.** V16/V17 in the forecaster zoo invite ShIC-scoring of a planner. Mitigation: re-file into an explicit A1 layer; their trainer preflight must assert Gate-A PASS on the substrate WM before training the planner.

**Open questions:** - **Does exploitable multi-candle structure exist at our resolution AT ALL?** This is the one genuinely-unknown, must-be-MEASURED question (Gate 0). The HARD dead-list entries (D17/D44/D45) say "no" at daily/4h/dollar long-only spot for the avenues tested — but the framing caveat is that those are SCOPED to tested cadences/chart-types; finer chart-types and the external-signal-for-mover-continuation avenue (D72, needs AUC>0.58) are NOT exhausted. Until Gate 0 is run on a NEW avenue, assume null.
- **At what rollout horizon H does the WM stop being genuine?** Gate-A clause 1 demands ShIC at the horizons the planner queries; we have ShIC at h=1 but have NOT measured whether it holds at h=16/h=64 for the candidate WMs. Must be measured before any Dreamer/MuZero rollout-depth is chosen — unknown today.
- **Is the A1 advantage over a simple threshold policy positive at all, GIVEN a gated WM?** Even with a genuine WM, whether planning beats a 1-line threshold on held-out compound is unmeasured for crypto. The chess precedent warns the proxy (search value) can diverge from the real objective (held-out wealth). Needs a head-to-head probe.
- **What is the minimum data resolution at which Gate 0 flips to PASS?** If daily/4h is null, is 1m enough, or is true tick/LOB required? D71/D72 suggest 1m has the move but lacks internal continuation info → external signal needed. The resolution↔existence curve is unmapped.
- **For A2: does conservative offline-RL actually control the extrapolation pathology on logged crypto backtests, or does it just hide it?** Unmeasured; A2 has not been built, so its GIGO-in-disguise risk is theoretical-but-named, not yet quantified.

## LENS: Honest Literature Reality-Check: Model-Based RL, Self-Evolving Agents, and Offline RL for Financial Trading (2024-2026 SOTA)

**Summary:** The honest state of the literature as of mid-2026 is that no MBRL architecture — DreamerV3, MuZero, TD-MPC2, or any variant — has been genuinely evaluated on financial trading in a peer-reviewed, look-ahead-clean, multi-regime, out-of-sample study. The backbone architectures exist and function in robotics/games; there is zero reproducible finance-specific evidence that they beat a well-tuned model-free baseline after costs. Self-evolving / continual-learning agents show real methodological progress (ReCAP, MetaTrader) but share a common structural flaw: the continually-learning agent is evaluated against static baselines, which is not a fair comparison for a non-stationary environment. Offline RL for finance is theoretically attractive and actively researched but is universally demonstrated to overfit to training-regime distributions, with the "distribution shift = catastrophic" finding appearing repeatedly (arXiv:2505.12759, arXiv:2411.12746). Representation learning (contrastive / JEPA / RSSM) shows genuine promise as a pre-training step for financial agents, but the downstream RL performance gains are small and rarely hold past regime changes. The single most consistent empirical finding is negative: RL-for-trading reliably degrades under regime change, and the backtest-to-live gap is large enough that even the FinRL benchmark paper implicitly acknowledges it (the "why don't they trade their own money" section).

# Dimension-by-Dimension Analysis

---

## (a) Model-Based RL for Financial Trading: DreamerV3 / MuZero / TD-MPC2

### What exists
- **DreamerV3** (arXiv:2301.04104, Hafner et al. 2023): proven across 150+ tasks (Atari, Minecraft, DMC, BSuite). Open-source. The architecture — RSSM + categorical latent + two-hot return prediction — is directly relevant to this project's V1/V12 family. However, **no peer-reviewed paper applies DreamerV3 to financial trading**. Searches across arXiv 2023-2026 found zero finance-specific DreamerV3 papers.
- **TD-MPC2** (arXiv:2310.16828, ICLR 2024): MCTS-style latent planning, strong on continuous control, single hyperparameter set across 104 tasks. **No finance application found.** The architecture has a documented flaw in stochastic environments (arXiv:2404.09946): the MuZero-style loss fails in stochastic settings and incurs exponential sample complexity even in deterministic settings with coverage gaps — two properties endemic to financial markets.
- **MuZero / EfficientZero V2**: MuZero's discrete-action MCTS is structurally hard to apply to continuous portfolio weight spaces without discretization, and "MuZero's learned model is generally not accurate enough for policy evaluation" (arXiv:2306.00840). EfficientZero V2 claims +45% over DreamerV3 on Vision Control but is also non-financial.

### The GIGO reality (your framing is exactly right)
The entire class of WM-consuming agents (A1) rests on the quality of the world model. The literature on WM quality for financial data is thin but consistent: (1) financial time series are highly non-stationary — the world model trained on one regime produces biased synthetic rollouts in another; (2) the RSSM/latent-imagination paradigm was designed for MDPs with stable transition dynamics; (3) in finance, even the transition function (volatility regime, liquidity) changes on monthly timescales, well within a training run. The paper "Adaptive World Models" (arXiv:2411.01342) directly addresses non-stationarity via HiP-POMDP but evaluates **only on proprioceptive continuous control**, not finance, and explicitly scopes out high-dimensional inputs as future work.

### Honest verdict
**No reproducible evidence that MBRL (DreamerV3 / MuZero / TD-MPC2) works on financial trading better than a model-free baseline after transaction costs.** These are promising architecture BACKBONES — this project's decision to treat V16/V17 as stubs is correct. Adopting them is a research bet, not an established technique.

---

## (b) Self-Evolving / Continual / Open-Ended RL Agents for Trading

### What is genuinely new
- **ReCAP** (arXiv:2606.00143, Jun 2026): Regime-adaptive continual learning — CUSUM-based regime detection + policy vector library + regime-gate neural network. Evaluated on 17 years (DOW30, NAS100, SP500, NIKKEI30, COMMODITY_ETF), 5-year online hold-out. Strict time-series preprocessing (no look-ahead in normalization). Results: NAS100 164.89% vs 125.37% best baseline over 5 years, SR 1.14 vs 0.91. **Positive: this is one of the cleaner evaluations in the field.** Critical caveat: ReCAP is continuously trained during the test period while most baselines are static — the comparison is inherently asymmetric. The appendix does include matched rolling-window retraining comparisons where ReCAP still wins, which partially mitigates this.
- **QuantEvolve** (arXiv:2510.18569): LLM-driven evolutionary alpha mining via multi-agent framework. Targets factor discovery, not RL per se. No robust OOS validation found in the abstract.
- **RLMF (Reinforcement Learning from Market Feedback)**: LLM fine-tuning with market-derived reward signals — active in FinRL Contest 2025. Conceptually interesting; no held-out performance numbers from independent replication.

### Structural weakness of the whole class
Self-evolving agents for trading face a meta-overfitting problem: the evolution mechanism itself can be fit to the training distribution. A system that "adapts" its policy continuously on the same data it is being evaluated on is doing online overfitting, not genuine generalization. The only clean test is strict regime-gated hold-out: train regime A, test regime B with no leakage. ReCAP does this more carefully than most, but the short 5-year live window (2020-2025) includes the post-COVID bull run, which inflates all long-biased strategies.

---

## (c) The Model-Exploitation / Compounding-Error Problem

### The theoretical picture
arXiv:2404.09946 ("A Note on Loss Functions and Error Compounding in Model-Based RL") is the most directly relevant 2024 paper for this project. Key findings:
1. The MuZero loss (value-equivalence without observation reconstruction) **fails in stochastic environments** — it cannot distinguish lucky outcomes from good models, so the learned dynamics model diverges from the true environment.
2. In deterministic settings with coverage gaps, MuZero incurs **exponential sample complexity** — the model exploits the gaps by hallucinating high-reward trajectories.
3. The simulation lemma bounds performance degradation as a function of one-step model error, but error compounds multiplicatively over rollout length. In practice, horizon > 5 steps causes substantial degradation in noisy domains.

### Financial-specific aggravation
- Financial returns are approximately i.i.d. at the bar level (per ShIC=0 findings in this project) — this means the world model **has minimal signal to latch onto** and its "planning" may be pure noise exploitation.
- Volatility clustering (GARCH effects) means the variance of one-step prediction errors is itself non-stationary — the simulation lemma's bound is loose and degrades unpredictably.
- **Mitigations from the literature**: (1) Short rollout horizons (≤3 steps for financial applications); (2) Model-ensemble uncertainty penalties (MOPO, COMBO style); (3) Pessimistic value estimates (MetaTrader's min-over-transforms trick); (4) Decoupling representation learning from policy optimization (pretrain RSSM/JEPA on reconstruction, freeze, then train actor-critic). These mitigations reduce but do not eliminate the problem.

---

## (d) Representation Learning from Raw Financial Data for RL

### What works (with caveats)
- **Contrastive learning of asset embeddings** (arXiv:2407.18645): Properly temporally separated (train 2000-2012, test 2013-2018). 69% sector classification accuracy, 19.1% vs 23.8% portfolio volatility. **Survivorship bias is unaddressed** (611-stock universe over 18 years). This is still downstream prediction/hedging, not RL policy learning.
- **CNN/autoencoder encoders from OHLC**: Learn candlestick patterns in lower layers, short-term trends in middle layers, complex multi-bar patterns in upper layers. Functionally equivalent to learned TI features. Genuinely useful as RL state encoder, but evidence is in supervised forecasting, not trading RL.
- **Self-supervised (contrastive vs generative)** (arXiv:2403.09809): Comparative study shows contrastive wins on classification tasks; generative wins on reconstruction-requiring tasks. For RL state encoding, the contrastive pretext task is likely more appropriate — it aligns with what the RL agent needs (discriminative, not generative).
- **LLM-NAS for RL state encoding** (arXiv:2512.06982): LLM-driven NAS for selecting multi-source state encoders. Novel but not yet validated at production scale.

### The representation-quality bottleneck
The fundamental problem is that raw financial data at daily/4h resolution has very low information content per bar. Even the best representation learning cannot manufacture signal that isn't there. The project's own ShIC>0 diagnostic is the right gate: if a representation-based world model's ShIC ≈ 0, the downstream RL policy is learning to exploit the model's errors, not the market's structure.

---

## (e) Offline RL / Decision Transformers on Historical Market Trajectories

### The distribution-shift trap is universal and documented
- **arXiv:2505.12759 (MetaTrader)**: The definitive 2025 statement: naive offline RL "memorizes optimal trading patterns within historical data rather than learning generalizable policies." Even with bilevel optimization and synthetic data augmentation, results show high variance (±0.10 on some metrics) and the method is an "early study." The 3-year test window (2019-2022) avoids a genuine bear regime.
- **arXiv:2411.12746 (Survey)**: "All surveyed works use online RL algorithms" — offline RL remains largely unexplored in peer-reviewed finance RL as of late 2024. The handful of offline approaches that exist don't formally address the non-stationarity that makes distribution shift in finance especially severe.
- **Decision Transformer (arXiv:2411.17900)**: GPT-2 + LoRA as offline DT for trading on DJIA-29 stocks. "Performs competitively with established baseline methods" — notably NOT claimed to outperform all baselines. The expert trajectories it trains on are themselves backtest-generated, compounding the distribution mismatch.

### The pessimism / conservatism gap
Offline RL theory recommends pessimism (penalize out-of-distribution actions) as the principled fix. MetaTrader operationalizes this via min-Q over transformed states. However, in financial markets, the optimal policy IS often out-of-distribution relative to the historical behavior policy (e.g., going short during bear regimes when the training dataset had predominantly long positions). Conservative offline RL is therefore in structural tension with the task: being too conservative means never taking the trades that would beat the market.

---

## Cross-Cutting Contamination Issues

The following contamination failure modes appear repeatedly in the surveyed literature and are directly relevant to this project:

| Failure Mode | Frequency in Literature | This Project's Guard |
|---|---|---|
| Look-ahead in normalization (G-AUDIT-011 class) | Very common — wavelet/BOCPD papers affected | CDAP atomic-write + split gate |
| Survivorship bias (contrastive paper 2407.18645) | Pervasive in equity studies | Crypto perpetuals avoid delisting |
| Train-on-test (FinRL contest Rachev ratio error) | Documented | 50/20/20/10 split + unseen-sealed |
| Static baseline vs adaptive agent comparison | ReCAP + all continual-learning papers | Must mandate iso-computational-budget baselines |
| Regime cherry-picking (MetaTrader: 2019-2022 only) | Common | Multi-regime gate in harness |
| No transaction cost realism | FinRL ensemble BTC: 0.66% vs 0.74% B&H after costs | CostModel + p_fill calibration |
| Policy instability from seed variance | FinRL: "sensitive to hyperparameters" | 10/10 seed gate required |

---

## What Is Genuinely SOTA and Reproducible (Adoptable Techniques)

1. **HiP-POMDP / context-conditional world models** (arXiv:2411.01342): Condition the RSSM on an inferred regime latent. The mechanism is sound and addresses the non-stationarity problem structurally. Tested only on proprioceptive control — would require validation on financial data. Adoption path: add regime context vector to the V1.1/V12 RSSM conditioning. This is a genuine next-step.

2. **Contrastive pretraining of asset encoder, frozen before RL** (arXiv:2407.18645 + 2403.09809): Decouple representation learning from policy optimization. Train the encoder on a self-supervised objective (contrastive or masked reconstruction), freeze it, then train the actor-critic on the frozen latent. Reduces WM exploitation risk because the encoder doesn't backprop through the RL signal.

3. **Regime-gate / policy vector library** (ReCAP, arXiv:2606.00143): The architecture — CUSUM-detected regimes + composable policy vectors + gate network — is both principled and implementable. The gate network itself is small (few parameters) and auditable. This is the most directly adoptable continual-learning technique for the setup-chaser context.

4. **Pessimistic offline RL via synthetic OOD augmentation** (MetaTrader, arXiv:2505.12759): The three data transformations (short-term noise F₁, long-term trend F₂, multi-scale F₃) + min-Q conservatism is a workable recipe for training offline without distributional collapse. Still early, still volatile, but the idea is sound.

5. **Short rollout horizon discipline**: Literature consensus and theory (arXiv:2404.09946) say ≤3-5 step rollouts in noisy domains. Financial bars have fat-tailed noise. Any MBRL deployment should cap imagination rollout at 3-5 bars, not the 15-50 used in DreamerV3's game benchmarks.

---

## Honest Negative Results (Where RL-for-Trading Reliably Fails)

1. **Under regime change**: All reviewed methods degrade. ReCAP mitigates but does not solve. No method survives a held-out test spanning a genuine regime it was never trained on.

2. **Model-free vs model-based gap**: The FinRL survey finds "no significant differences between Policy Gradient and DQN" across a broad literature sweep. Adding a world model does not consistently beat model-free baselines in financial domains.

3. **Long rollout MBRL in stochastic environments**: arXiv:2404.09946 formally proves MuZero-style losses fail here. This includes every implementation this project is considering as A1 agents.

4. **Live deployment vs backtest**: The only honest live result in the broad literature is Wei et al.'s 5-day profitable deployment — a sample of 5 days. The FinRL benchmark paper implicitly acknowledges skepticism ("why don't model producers trade their own funds"). No peer-reviewed paper demonstrates sustained live RL trading performance over a meaningful horizon.

5. **LLM-injected signals without risk-aware framework**: "Simply injecting strong LLM-based biases into a plain PPO agent actually hurt performance" — only combined with risk-aware frameworks did signals add value.

6. **Self-play / pure exploration in financial RL**: By analogy with the chess experiment in this project, pure self-play degrades a policy that was initialized from a good prior (the chess net that copied a teacher). The financial analog: a WM-agent that "explores" in imagined rollouts will generate adversarial trajectories that look like high-reward states but are exploitation artifacts.


**Failure modes:** **FM-1: The GIGO Trap (most dangerous for this project)**
A WM-consuming agent (A1 class: DreamerV3/MuZero over V1.1/V12) will plan in latent space. If the WM's ShIC ≈ 0 (the V22/V25 memorization failure pattern this project has already seen), the "imagined rollouts" are pure model-exploitation — the agent learns to find states the model assigns high reward to, not states the actual market delivers high reward to. Mitigation: mandatory ShIC > 0.015 gate on the WM BEFORE any A1 agent is trained on it; rollout horizon capped at 3-5 bars; model-ensemble uncertainty penalty on imagined transitions.

**FM-2: MuZero-Style Loss Failure in Stochastic Environments**
(arXiv:2404.09946, formally proven) The value-equivalence loss used in MuZero and TD-MPC2 fails in environments with stochastic transitions and incurs exponential sample complexity in deterministic environments with coverage gaps. Financial markets are stochastic by definition. Mitigation: use reconstruction-augmented WM objectives (DreamerV3-style RSSM with observation reconstruction loss) rather than pure value-equivalence; this gives the WM an independent grounding signal beyond the policy gradient.

**FM-3: Continual-Learning Self-Approval (the agent-delegation bypass pattern)**
A self-evolving agent that updates its own policy during evaluation can construct a feedback loop where the agent effectively "approves" its own strategy changes. This is the financial analog of the agent-delegation bypass gate memory entry. Mitigation: every policy update during the evaluation window must be logged as a parameter-change event with a pre-update vs post-update performance snapshot; the "update" itself must be treated as a potential data leak.

**FM-4: Non-Stationary World Model Staleness**
The RSSM/JEPA WM is trained on a fixed historical window. As the market regime shifts, the WM becomes stale but continues generating confident rollouts — the RL agent has no way to know the WM is wrong. This is the financial equivalent of the chess net "playing the same" (the monotonic gate held, but it was measuring the wrong thing). Mitigation: HiP-POMDP regime context conditioning (arXiv:2411.01342); explicit WM drift monitoring (compare WM prediction error on a rolling recent window to training error; if the gap exceeds a threshold, freeze the RL policy and retrain the WM).

**FM-5: Offline RL Distribution Shift → Catastrophic Generalization Failure**
(arXiv:2505.12759, documented formally) An offline RL policy trained on 2018-2022 data is likely a long-biased equity momentum policy. A 2022-2024 bear/sideways market is out-of-distribution in exactly the direction where the policy should behave differently. Pessimistic offline RL is structurally unable to take "regime-appropriate short" actions it has never seen in training data. Mitigation: generate synthetic OOD trajectories (MetaTrader's F₁/F₂/F₃ transforms); evaluate on a genuine held-out bear-regime window before any live deployment; include a "no-trade" action as a valid policy output.

**FM-6: Backtest-Contaminated Expert Trajectories for Decision Transformers**
The DT-LoRA paper (arXiv:2411.17900) trains on trajectories generated by RL agents trained on historical data — which are themselves backtest-optimized. The DT thus learns a meta-policy that is doubly contaminated. Mitigation: if using DT-style imitation of historical strategies, the "expert" trajectories must come from a provably causal, cost-aware strategy with confirmed OOS performance, not from a WM-agent that may itself have been contaminated.

**FM-7: The "Competitive Performance" Rhetorical Trap**
Multiple papers (DT-LoRA, FinRL ensemble BTC) claim "competitive with baselines" while the baseline is buy-and-hold on BTC during 2020-2022 (a +300% period). "Competitive" meaning SR 0.28 on 0.66% cumulative return vs B&H 0.74% — i.e., the agent underperformed B&H after costs. Always inspect the baseline denominator; in crypto, passive exposure in a bull regime beats almost any active strategy. Mitigation: always include a risk-matched passive baseline (volatility-targeted B&H) as the denominator, not raw B&H.


**Open questions:** 1. **Does MBRL actually help on financial data vs model-free PPO/SAC at the same compute budget?** Not a single clean ablation exists in the 2024-2026 literature. The control experiment — DreamerV3 vs PPO on the same financial environment, same hyperparameter search budget, same OOS split — has not been published. This must be measured before committing GPU budget to A1 agents.

2. **What is the ShIC signature of a context-conditional RSSM (HiP-POMDP variant) trained on crypto bars?** The adaptive world model architecture (arXiv:2411.01342) is theoretically motivated but untested on financial data. Specifically: does conditioning on a regime context variable meaningfully improve ShIC, or does the context encoder itself memorize the temporal structure?

3. **What rollout horizon preserves WM signal vs degrades to exploitation?** Theory says ≤5 steps in noisy domains, but the crossover point for 15m/4h crypto bars is unknown. This requires empirical measurement: plot value-function bias as a function of rollout depth on held-out data for the current V1.1/V12 WMs.

4. **Is contrastive pre-training of the encoder (decoupled from RL) a genuine improvement over end-to-end joint training for setup detection?** The arXiv:2407.18645 paper shows benefit for sector classification but not for RL policy quality. The downstream question for this project is whether a frozen contrastive encoder produces better RL state representations than the RSSM encoder trained jointly with the actor-critic.

5. **Can the regime-gate architecture (ReCAP's policy vector library) be applied to multi-bar setup detection — not just portfolio allocation?** ReCAP is evaluated on continuous portfolio weight optimization. Whether the same CUSUM-gate + composable-policy mechanism transfers to the discrete setup-entry/exit space is unmeasured.

6. **What is the minimum look-back needed for a regime context encoder to infer the current market state without look-ahead contamination?** All regime-detection methods (CUSUM, HMM, SMA-based) have a detection lag. The lag vs accuracy tradeoff for financial regimes at 15m/4h resolution has no published benchmark.

7. **Does the MetaTrader bilevel optimization genuinely generalize, or does the synthetic augmentation just inflate the effective training set size?** The ablation shows the transformations hurt in-domain performance, which is expected, but whether they help on regimes truly outside 2019-2022 (e.g., a 2022-style deleveraging) remains untested. This is specifically the scenario this project must stress-test.


