# WM World-Class Architecture — Design Doc (2026-06-10)

> **Status**: DESIGN ONLY. No code changed. This is the load-bearing design for
> whether to kick off training. Author: Architecture Expert (architect skill).
> Repo-grounded with file:line. Brutally honest about the ceiling.
>
> **Scope decision up front**: the "world-class WM" we design for is the WM whose
> signal, used by the trading AGENT as a regime/setup FILTER, maximally improves
> robust held-out COMPOUND return on UNSEEN — *not* the banned IC>0.10-per-bar
> target (CLAUDE.md "ARCHIVED 2026-06-04" tombstone; MEMORY.md founding framing).
> IC h=1 survives only as a within-WM convergence diagnostic (>0.015).

---

## (A) The reframe + decompose-the-ideal

### A.1 The reframe (what "world-class" means for OUR problem)

The scoresheet's "Headline / agent-teaching" bar (IC>0.10 / ShIC>0.05,
`docs/WM_SCORESHEET_MERGED_2026_04_29.md:78`) is **archived as a primary objective**.
We do not design toward it. We design toward the objective the promotion gate already
encodes (`src/wm/wm_promotion_gate.py:8-20`):

> A candidate WM is world-class iff its signal, fed to the agent, produces a strictly
> higher **robust held-out compound return on UNSEEN** than the current champion —
> verified by `src/strat/wm_value_probe.py` against TWO controls (buy-and-hold and a
> regime-matched random-entry null).

This is a different optimization target than per-bar IC, and the proof-of-value already
told us they DIVERGE: V1.1 (IC 0.067) as a long-only per-bar input **LOSES to
buy-and-hold** on an UNSEEN bear, **BEATS random entry +20pp**, but its regime gate is
~coin-flip (49%). The value is in *detection* (stand aside in bears), not in per-candle
return prediction.

### A.2 Decompose-the-ideal — construct the best-achievable compound-filter WM

The IDEAL filter is the oracle `keep[t] ∈ {0,1}` that, applied to a long-only base
policy, maximizes compound return on UNSEEN. Decompose the oracle's information:

| Oracle sub-signal | What it answers | Realizable head (scaffolded) |
|---|---|---|
| **bear-onset** | "will the next K bars draw down > θ?" → stand aside | `forward_bear_label` (`regime_targets.py:97`) → `bear_head` |
| **move-onset** | "is a net-of-cost up-MOVE available in [t+a,t+b]?" → enter | `move_onset_label` (`regime_targets.py:179`) → `move_head` |
| **trend regime** | "down / neutral / up over next K bars?" → size/gate | `forward_trend_label` (`regime_targets.py:136`) → `trend_head` |
| **magnitude / dispersion** | "how big, and how uncertain?" → size, risk | `QuantileHeads` (`headline_components.py:525`) |

The ideal's information is **forward, multi-candle, and direction-of-regime**, NOT
per-bar return. Three of the four sub-signals are already built as no-look-ahead label
builders + OFF-by-default heads (`forward_regime_head.py`), validated mechanically
(`regime_targets.py:_selftest`). **The architecture's job is to learn a latent that
makes these forward heads accurate on UNSEEN.** That is the whole design.

### A.3 Adversarial paragraph (de-biased self-critique — architect SOTA #1)

*Why would this design fail on UNSEEN compound?* The forward labels are derived from
future bars; a high-capacity encoder with no bottleneck will MEMORIZE the
training-period mapping from a 96-bar window to its forward label, and ShIC will collapse
to 0 — exactly the V25 failure (README:53, IC≈+0.21/ShIC=0.000). The forward-onset
target does not change this; it makes it WORSE, because the bear/move labels are
auto-correlated in time (clustered), giving the memorizer more temporal handholds. The
design is only correct if **every variant keeps the RSSM categorical bottleneck +
reconstruction anchor that demonstrably produces ShIC>0** (V1.1) and the forward heads
are ADDITIVE aux losses on that anchored latent, never the primary path. A forward head
on an un-anchored encoder (V22/V25 as-is) is a confident way to ship a memorizer. This
survives the adversarial paragraph **only** under the bottleneck constraint in (C.1).

---

## (B) The target design (loss + heads, precisely)

### B.1 Primary target: keep per-bar TwoHot as a DIAGNOSTIC ANCHOR, add forward heads as the OBJECTIVE-aligned signal

Do **not** delete the per-bar TwoHot return head. It is the convergence diagnostic
(the >0.015 IC gate) AND, empirically, the reconstruction/return supervision is part of
what anchors the RSSM latent (V1.1 ShIC=0.033 with it). Instead, ADD the forward heads
and make THEM the promotion signal.

**Loss (V1.1-anchored, forward-augmented):**

```
L_total =  L_recon            (MSE on base features; decoder anchor)   # KEEP — load-bearing
         + L_kl               (free-nats RSSM categorical KL)          # KEEP
         + w_ret * L_twohot   (per-bar TwoHot, 4 horizons; DIAGNOSTIC) # KEEP, weight unchanged
         + w_dir * L_huber    (DIRECT_RETURN_WEIGHT=3.0)               # KEEP (invariant)
         + λ_fwd * L_forward                                           # ADD (the new objective)

L_forward = w_bear  * CE(bear_logits,  forward_bear_label)            # binary, masked
          + w_move  * CE(move_logits,  move_onset_label)              # binary, masked
          + w_trend * CE(trend_logits, forward_trend_label)           # 3-class, masked
          + w_qtl   * pinball(quantile_logits, fwd_h16_return)        # optional magnitude
```

- `L_forward` is the EXACT additive masked-CE already implemented:
  `forward_regime_aux_loss` (`forward_regime_head.py:132`), NaN rows masked
  (`_masked_ce:118`). It reads the SAME fused `feat = cat(h_seq, z_post)` the existing
  heads read (`world_model.py:281`) — **no new input plumbing**.
- **λ_fwd small and annealed** (start 0.0, ramp to ~0.3 over 10 epochs) so the latent is
  anchored by recon+KL+TwoHot FIRST, then the forward heads shape it. Rationale: a large
  forward weight from step 0 invites the memorization in (A.3).
- **Class imbalance**: bear-onset and move-onset are rare-positive. Use `pos_weight` in
  the CE (NOT focal — focal is banned, `TWOHOT_FOCAL_GAMMA=0.0`, and the ban exists
  because focal upweights temporally-clustered tails = memorization; the same logic
  applies to the forward binary heads). A static inverse-frequency `pos_weight` is safe;
  a learned/temporal reweighting is not.

### B.2 Why this target (and not a from-scratch move-onset-only model)

A move-onset-ONLY model throws away the recon+TwoHot anchor that is the *only* mechanism
we have empirical proof produces ShIC>0. We are NOT confident a fresh forward-only
objective anchors the latent — the V25 evidence says un-anchored forward supervision
memorizes. So the design is **anchor-preserving augmentation**, not replacement. The
forward heads are where the compound value comes from; the recon/RSSM/TwoHot stack is the
anti-memorization skeleton they ride on.

### B.3 Promotion is on COMPOUND, IC is a gate only

The candidate is scored by `wm_value_probe` → `wm_promotion_gate.should_promote`
(`wm_promotion_gate.py:91`, strict monotonic compound floor). New entry mode needed in
`src/strat/wm_entry_producer.py`: threshold `bear_logits`/`move_logits` instead of
rolling the per-bar h16 return (the producer's current `h4_roll_rgm16` signal,
`wm_value_probe.py:11`). This is the single most important wiring change and is
**already specified** in `forward_regime_head.py:31-34`. IC h=1 stays a within-training
print, never a gate.

---

## (C) The backbone + gap-closure design

### C.1 Backbone verdict: V1.1 RSSM is the spine. V22–V25 are NOT drop-in world-class — they are BROKEN as shipped.

This is the brutal-honesty core of the doc. The modern cohort is built but its current
state is a **memorizer**, not a candidate:

- **V25** (frontier synthesis): README:52-58 documents **IC ≈ +0.21 / ShIC = 0.000** —
  the textbook memorization signature. Root cause: `recon = torch.zeros(B,T,1)`
  (`world_model.py:446` in V22; same in V25) — **there is no reconstruction anchor and
  no categorical RSSM bottleneck.** It has a RateBudgetVIB, but the VIB alone does not
  anchor (KL-rate budget ≠ a faithful-latent constraint).
- **V22** (iTransformer): identical `recon=torch.zeros` stub (`v22/.../world_model.py:446`),
  `USE_FORECAST_HEAD=False` (settings.py:234, reverted because turning it on regressed).
  It now reaches Epoch-1 after the host-RAM OOM fix, but it will train to the same
  ShIC=0 wall unless the anchor is added.
- **Canonical precedent**: `memory/fix_logs/INDEX.md:22-24` — "Clean refactor stripped
  the V1.0 RSSM categorical latent (the load-bearing anti-memorization mechanism)...
  IC1=0.2729, ShIC=0.0002". V22/V25 are this exact failure mode at the architecture
  level. **Pattern I (architect skill gotcha): any non-RSSM variant MUST have an explicit
  bottleneck (VIB/InfoNCE) AND a reconstruction/forecast anchor. A return head reading
  directly from an un-anchored encoder = catastrophic memorization.**

**Decision: the world-class backbone is V1.1's anchored RSSM-Transformer**, optionally
with a stronger *cross-feature/cross-asset* representation grafted on **behind** the
bottleneck — never a backbone that removes the bottleneck.

### C.2 Which modern idea is worth grafting (and how), in priority

| Idea | Source | Verdict for low-SNR crypto | How to make it world-class-safe |
|---|---|---|---|
| **iTransformer cross-feature attention** (V22) | ICLR24 | **Useful** — attends ACROSS the 18/29/41 features, no timestamp sync needed (V22 README:9). Cross-asset becomes cross-feature. | Graft as an ENCODER stage that feeds the RSSM (`h_seq` producer), keep recon+KL+categorical z_post downstream. Add `USE_FORECAST_HEAD=True` + a real feature-recon decoder (V22 settings.py:230 "Path B") so the encoder is anchored. |
| **Cross-asset attention** (V12) | in-repo | **Highest-EV gap** — the proof-of-value lever is bear/regime detection, and regime is a CROSS-ASSET phenomenon (BTC leads alts). | V12 is the ONLY version with this AND it keeps a loss path. `get_multi_loss` is wired+masked (`v12/.../world_model.py:579`), `MultiAssetDataset` is COMPLETE (`multi_asset_dataset.py`, "2026-06-10 COMPLETE"). Verify V12 retains an anchor before trusting it (audit item). |
| **xLSTM matrix memory** (V23) | NeurIPS24 | **Marginal** — linear-in-T recurrence is nice but our seq is 96 and the bottleneck is SNR, not context length. No anchor in scaffold (README says interface only). | DEFER. Only revisit if seq_len needs >256 (it doesn't at daily/dollar bar). |
| **TimesNet 2D freq blocks** (V24) | ICLR23 | **Speculative** — multi-period FFT structure could expose regime cycles, but crypto periodicity is weak/non-stationary. | DEFER behind V12/V22-anchored. Frequency multi-scale better served by `MultiResolutionEncoder` (below). |
| **Multi-resolution encoder** | `headline_components.py:224` | **Useful** — 3 encoders at 3 strides capture move-onset at multiple horizons (the multi-candle unit). | Graft into V1.1 as the `h_seq` producer (CC-H1). Cheap, anchor-preserving. |
| **Linear attention** | `headline_components.py:361` | Only relevant if seq>96 (CC-H4). | DEFER — seq stays 96. |

### C.3 The three named gaps and their closure

1. **Cross-asset (the regime lever)** — close via **V12** (`HEADLINE_MODE`,
   `get_multi_loss`), the only anchored cross-asset path. This is the single most
   on-objective gap: regime/bear detection is cross-asset, and the proof-of-value says
   regime detection is where the compound value is.
2. **Memorization (the V22/V25 disease)** — close via the HEADLINE_MODE anti-mem stack
   that V1.6 already validates: ATME p=0.15 per-sample (`world_model.py:404` V22 has it),
   KL anneal 0→1/20ep, Gumbel τ 1.0→0.5/50ep, XD-dropout (`XD_DROPOUT_RATE`), **plus the
   non-negotiable recon+categorical anchor.** The fix for V22/V25 is structural (add the
   anchor), not a knob.
3. **Sequence length** — NON-issue at daily/dollar bar. V4/Mamba's 512+ capacity is
   wasted here (SNR-bound, not context-bound). Keep seq_len=96 (`WM_SEQ_LEN=96`). Do not
   spend the GPU budget on long-context.

---

## (D) Ensemble / regime-conditioning

### D.1 Does a diverse ensemble beat the best single? — design the test, don't assume

The scoresheet flags this is UNMEASURED: V10 has "no recorded per-pair lift number"
(CC5, `WM_SCORESHEET_MERGED:170`) and pairwise V1.x correlation was never computed (CC1,
:166). The ensemble bet is only worth GPU IF the members are DECORRELATED on UNSEEN.

**Design**: a 2-member meta-ensemble (V10 ensembler exists, scoresheet row :200) of
**(i) V1.1-forward-augmented (single-asset, anchored)** and **(ii) V12-cross-asset
(forward-augmented)**. These two differ on a FUNDAMENTAL axis (single- vs cross-asset
representation), so they are the most likely pair to be decorrelated — unlike the V1.x
siblings (V1.0/1.1/1.4/1.6) which are near-duplicates (scoresheet D4 "sibling
redundancy", :185-187). **Gate the ensemble on a measured residual-corr < 0.9 on OOS**
before spending UNSEEN; if ρ>0.95 ship the single best and skip the ensemble.

### D.2 Regime-conditioned routing — yes, but as a HEAD, not a backbone

The world-class move is NOT a separate routed backbone per regime (V25's `regime_ffn` is
12M DEAD params, README:72 — a cautionary tale of over-engineering routing). The
world-class move is **RegimeFiLM conditioning of the shared latent**
(`headline_components.py:768`, `RegimeFiLM`) + the forward trend head as the router. One
shared anchored latent, FiLM-modulated by the predicted forward regime, with the
bear/move heads reading the modulated feature. This gives regime-conditioning's benefit
(different behavior in bull/bear/neutral — exactly the proof-of-value lever) at ~0 extra
backbone cost and no dead paths.

---

## (E) The 4060-trainability analysis (HARD constraint — user flagged slow trainings)

Constraint: ONE RTX 4060 (8GB), AMP, fit ~4GB train, batch ≤32, seq 96, all 4 horizons,
bounded wall-clock (hours-to-a-day, NOT weeks). **Self-consistency on the VRAM number
(architect SOTA #3): two independent derivations must agree within 10%.**

| Candidate | d_model | layers | params (layer-count) | params (sanity: ~scale of V1.1 5–8M) | seq | batch | est. train VRAM (AMP) | est. wall-clock @ 2000 steps/ep × ~80 ep | TRAINABLE? |
|---|---|---|---|---|---|---|---|---|---|
| **C1 V1.1-forward-aug** | 256 | 3 | ~6M base + ~0.3M fwd heads | ✓ V1.1 is 5–8M, heads add <5% | 96 | 32 | ~3.5–4 GB | ~8–14 GPU-h (V1.1-class run) | **YES** |
| **C2 V12 cross-asset-fwd** | 256 | 3 + cross-attn | ~7M + A×attn | ✓ same base, cross-attn small | 96 | 32 / **A=4 assets** → effective batch 128 in encoder | ~5–6 GB (A-fold flatten) | ~14–24 GPU-h | **YES, if A≤4** |
| **C3 V22-iTransformer + anchor** | 320 | 4 | ~5M | ✓ above 4M floor (README:40) | 96 | 32 | ~4–5 GB | ~12–20 GPU-h | **YES** |
| **C4 V25 frontier (as-is)** | 320 | 6 | **17.76M, ~12M DEAD** (README:72) | ✗ over budget, mostly wasted | 96 | 32 | ~6.5–7.5 GB (tight) | ~30–50 GPU-h | **MARGINAL — must shrink first** |
| **C5 2-member ensemble (C1+C2)** | — | — | sum | — | 96 | 32 | trained SEQUENTIALLY, not jointly | C1 + C2 cost | **YES (sequential)** |

**Self-consistency check on the tightest box (C4 V25)**:
- Derivation 1 (params): 17.76M params × 4 bytes × 4 (param+grad+2 Adam moments) ≈ 284 MB
  weights/optim; activations at d=320, seq=96, batch=32, 6 layers dominate → peak ~6–7 GB.
- Derivation 2 (peak-activation): attention is over FEATURES (F≈29 tokens), not seq, so
  attn is cheap; the cost is the 6× d=320 FFN activations × batch 32 × (F+1 tokens) +
  patch-embed → ~5.5–6.5 GB. **Agreement within ~10%** → C4 is on the edge of 8GB and the
  12M dead `regime_ffn` is pure waste. **Shrink first** (refactor out `regime_ffn`,
  README gap #2) → drops to ~5.5M live params, comfortable.

**Shrink levers, ranked** (if any candidate is too heavy): (1) delete dead `regime_ffn`
(C4: −12M params, −150MB, README:72); (2) cap cross-asset A=4 not 10 (C2 VRAM); (3)
d_model 320→256 (C3/C4); (4) layers 6→4 (C4); (5) keep seq 96 (never raise — no SNR
benefit). **AMP + per-epoch `torch.cuda.empty_cache()` are already in the trainers**
(`v22/.../train_world_model.py:415` OOM fix). The host-RAM OOM in AntifragileDataset that
blocked the whole cohort was fixed THIS session (numpy index) — that was the binding
blocker, now cleared.

**Wall-clock honesty**: the user flagged trainings "taking too long". C1 (V1.1-class) is
the FASTEST path to a testable compound number (~8–14 GPU-h = overnight). C4 (V25 as-is)
is the slowest AND most likely to ship a memorizer — it is explicitly DE-prioritized.

---

## (F) RANKED CANDIDATE LIST (what to TRAIN, in priority order)

> Ranking criterion: **expected lift in robust held-out COMPOUND per GPU-hour**, given
> the proof-of-value (the lever is bear/regime detection), the anchor constraint (C.1),
> and the 4060 budget (E). Each carries a FALSIFIER (architect SOTA #2).

### Rank 1 — **C1: V1.1 + forward-regime heads (anchor-preserving augmentation)** ⭐ TRAIN FIRST
- **What**: V1.1 backbone UNCHANGED + attach `ForwardRegimeHead` (`forward_regime_head.py`),
  add `λ_fwd · forward_regime_aux_loss` (annealed), new bear/move-threshold entry mode in
  `wm_entry_producer.py`, score via `wm_value_probe` → `wm_promotion_gate`.
- **Why first**: directly attacks the PROVEN lever (regime/bear detection) on the ONLY
  backbone with proven ShIC>0; smallest change; fastest to a compound number; all scaffolding
  exists and is OFF-by-default (zero risk to the base model).
- **Expected deliverable**: a WM whose bear/move filter beats V1.1's coin-flip regime gate
  (49%) → the first candidate that can BEAT buy-and-hold on an UNSEEN bear (the proof-of-value's
  open failure). Target: `should_promote` PROMOTE vs the current champion on compound.
- **Cost**: ~8–14 GPU-h (one overnight run).
- **FALSIFIER**: if the forward heads train but compound on UNSEEN does NOT beat the random-entry
  null margin V1.1 already has (+20pp), the forward target adds nothing over the coincident head →
  the regime signal is not learnable from daily/dollar bars (→ representation ceiling, see G).

### Rank 2 — **C2: V12 cross-asset + forward heads** ⭐ TRAIN SECOND
- **What**: V12 `HEADLINE_MODE=1`, `MultiAssetDataset` (A=4: BTC + 3 alts), `get_multi_loss`
  (`v12/.../world_model.py:579`) + the same forward heads. **Audit first**: confirm V12 retains a
  recon/KL anchor (it inherits V1.x RSSM — verify it wasn't stripped) BEFORE trusting any ShIC.
- **Why second**: regime/bear is a CROSS-ASSET phenomenon (BTC leads); this is the highest-ceiling
  on-objective gap. But it is heavier and has a data-plumbing dependency, so it follows C1.
- **Expected deliverable**: a cross-asset regime filter that detects bear-onset earlier/more
  reliably than single-asset C1 → higher compound on UNSEEN, especially in the bear segment.
- **Cost**: ~14–24 GPU-h.
- **FALSIFIER**: if C2's compound does not exceed C1's by the gate margin, cross-asset attention is
  not adding orthogonal regime info at daily/dollar resolution → ship C1, drop the cross-asset path.

### Rank 3 — **C3: V22-iTransformer + ANCHOR ADDED (Path B)** — TRAIN ONLY AFTER C1/C2
- **What**: V22 + `USE_FORECAST_HEAD=True` AND a real feature-reconstruction decoder replacing
  `recon=torch.zeros` (`world_model.py:446`; settings.py:230 "Path B"). Cross-feature attention as an
  anchored encoder.
- **Why third**: cross-feature attention is a genuinely different representation (could decorrelate
  from C1/C2 for the ensemble), BUT it requires fixing the anchor bug first and has a track record of
  ShIC=0. Higher risk, real upside only if the anchor fix takes.
- **Expected deliverable**: a decorrelated member for the Rank-5 ensemble; standalone only if it
  clears ShIC>0.015 AND compound beats C1.
- **Cost**: ~12–20 GPU-h (+ anchor-fix engineering).
- **FALSIFIER**: train with the anchor added; if ShIC is STILL ≈0, the iTransformer inversion is
  intrinsically a memorizer for this data → KILL the V22/V25 line (this also answers the V25 README's
  open question, README:138).

### Rank 4 — **C5: 2-member ensemble (C1 + C2), regime-FiLM routed** — TRAIN AFTER C1+C2 EXIST
- **What**: V10 ensembler over C1 and C2; gate on measured OOS residual-corr < 0.9 (CC1) BEFORE
  spending UNSEEN.
- **Why fourth**: an ensemble can only help if members are decorrelated; we can't know that until C1
  and C2 exist. Cheap to run once they do (no new training of members).
- **Expected deliverable**: a small compound lift over the best single IF ρ<0.9; otherwise a measured
  "ship the single" decision (also valuable — kills the V1.x sibling-ship question).
- **Cost**: ~2–4 GPU-h (inference + meta-head only).
- **FALSIFIER**: residual-corr ≥ 0.95 → ensemble is redundant, ship the best single.

### DE-PRIORITIZED (do NOT train until the above resolve)
- **C4 V25 frontier as-is** — 17.76M params, 12M DEAD (README:72), same ShIC=0 anchor gap as V22,
  slowest run. Only revisit AFTER shrinking (delete `regime_ffn`) AND after C3 proves the
  iTransformer line can be anchored. If C3's FALSIFIER fires, V25 is KILL.
- **V23 xLSTM / V24 TimesNet** — DEFER. Solve a context-length / periodicity problem we don't have at
  daily/dollar bar. No anchor in scaffold. Revisit only on a representation change (G).

**Total budget for Rank 1–4**: ~36–62 GPU-h ≈ 2–3 overnight runs + ensemble. Fits the
"hours-to-a-day, not weeks" constraint when run as a sequential pipeline. Rank 1 alone
(one overnight run) produces the first go/no-go compound number.

---

## (G) The honest ceiling (do NOT pretend a daily-bar tweak reaches IC>0.10)

**What architecture CAN close at daily/dollar bar:**
- The gap from V1.1's coin-flip regime gate (49%) toward a USEFUL forward bear/move filter.
  The coincident regime label was the wrong target; the forward labels are a genuinely better
  supervision signal. This is a real, architecture-closeable gap — and it is the gap that matters for
  COMPOUND (the proof-of-value lever). C1/C2 are the play.
- Per-bar IC: the literature-grounded estimate is the daily/dollar-bar ceiling is ~**0.08–0.087**
  (CLAUDE.md archived ladder context; consistent with V1.1's 0.067 + a forward-target/cross-asset
  bump). **An anchored forward-augmented V1.1/V12 might reach ~0.08 IC — it will NOT reach 0.10.**

**What architecture genuinely CANNOT close (needs a REPRESENTATION change):**
- **Literal IC > 0.10 per-bar at daily/dollar resolution is not architecture-closeable.** The
  proof-of-value measured daily/dollar-bar direction AUC ≈ 0.51 (near random) — that is a SIGNAL
  ceiling in the data, not a modeling failure. No backbone (iTransformer, xLSTM, TimesNet, Mamba,
  RSSM) extracts >0.10 per-bar IC from a 0.51-AUC representation. Reaching IC>0.10 requires
  **sub-bar / HF / tick / LOB features** (V20-class representation change), which is OUT of this
  cohort's scope and budget.
- **Crucially, we are NOT chasing IC>0.10.** Per the reframe (A), the objective is COMPOUND via a
  regime/setup FILTER. The honest claim is: **architecture (C1/C2) can plausibly turn V1.1 from a
  random-beating-but-B&H-losing input into a B&H-beating filter on UNSEEN bears — without ever
  touching IC>0.10.** That is the world-class deliverable for OUR problem. If C1 AND C2 both fire
  their FALSIFIERs (compound does not beat the null/champion), the conclusion is that the regime
  signal is not learnable at daily/dollar resolution and the next move is a DATA/representation change
  (sub-bar), not another architecture — and we will have proven it cheaply (≤1 overnight run for the
  first signal).

**One-line ceiling**: *Architecture can make a world-class COMPOUND filter at daily/dollar bar; it
cannot make a world-class per-bar IC. We design for the former and refuse to claim the latter.*

---

## Reflexion forward-note (architect SOTA #4)

If C3 (V22 anchored) still shows ShIC≈0, write to `memory/fix_logs/INDEX.md` Cross-Cutting:
`[V22 2026-06-1x] iTransformer inversion memorizes even WITH recon anchor → cross-feature
tokenization lacks the temporal bottleneck the RSSM categorical provides; KILL the V22/V25 line for
daily-bar.` A failure not written forward is re-paid next version.

## Audit hooks before training (pre-first-training, CLAUDE.md #11)
- Verify V12 retains recon+KL anchor (grep `recon = self.decoder` in
  `v12/.../world_model.py`) — do NOT trust a cross-asset ShIC from an un-anchored V12.
- Grep each candidate's `settings.py` against the Cross-Version Training Invariants table
  (BIN_MIN/MAX=[-1,1], NUM_BINS=255, ACTIVE_HORIZONS=[1,4,16,64], TWOHOT_FOCAL_GAMMA=0.0,
  DIRECT_RETURN_WEIGHT=3.0, WM_STEPS_PER_EPOCH=2000).
- Confirm `strict=False` on both `model` and `ema_model` `load_state_dict` (all ckpts
  INCOMPATIBLE with V51 — fresh train, but the guard must be present).
- Run the two scaffold self-tests green before wiring:
  `python src/wm/_shared/regime_targets.py` and
  `python src/wm/_shared/forward_regime_head.py`.
```
