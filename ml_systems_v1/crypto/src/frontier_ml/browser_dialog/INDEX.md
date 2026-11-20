# Browser Dialog — Prompts & Responses Index

> Single source of truth for all `@browser`-routed research dialogues.
> Each prompt the main session sends to a browser-tool-equipped session
> gets an entry here. Status is one of `OPEN` (pending response) or
> `CLOSED` (response received and acted on).
>
> Two-pass operating model (per user 2026-05-02):
>   1. **Pre-information layer**: browser does first-pass research; returns a
>      synthesis with citations.
>   2. **Decision layer**: main session deliberates on the response, files
>      action items, and may issue follow-up prompts that depend on the result.
>
> All dialogue artifacts live in `src/frontier_ml/browser_dialog/` from
> 2026-05-02 forward. The 2026-05-02 originals (`RESEARCH_BRIEF_*` and
> `FRONTIER_RESEARCH_RESPONSE_*`) sit in the parent `src/frontier_ml/` for
> commit-history continuity; from B003 onward, both prompt and response
> live inside `browser_dialog/`.

## Status table

| ID | Status | Title | Prompt path | Response path | Sent | Closed |
|---|---|---|---|---|---|---|
| B001 | CLOSED | Frontier-tier WM + agent for crypto on 4060/8GB (initial scope) | [src/frontier_ml/RESEARCH_BRIEF_2026_05_02.md](../RESEARCH_BRIEF_2026_05_02.md) | [src/frontier_ml/FRONTIER_RESEARCH_RESPONSE_2026_05_02.md](../FRONTIER_RESEARCH_RESPONSE_2026_05_02.md) | 2026-05-02 | 2026-05-02 |
| B002 | CLOSED | Frontier LLM lab approaches (Gemini / Claude / DeepSeek / ChatGPT) overlay onto crypto WM | [PROMPT_B002_frontier_lab_overlay.md](PROMPT_B002_frontier_lab_overlay.md) | [RESPONSE_B002_frontier_lab_overlay.md](RESPONSE_B002_frontier_lab_overlay.md) | 2026-05-02 | 2026-05-02 |
| B003 | CLOSED | V0+ envelope push: literature + first-principles novel ideas to lift the V1.x family beyond IC 0.067 / ShIC 0.037 | [PROMPT_B003_v0plus_envelope_push.md](PROMPT_B003_v0plus_envelope_push.md) | [RESPONSE_B003_v0plus_envelope_push.md](RESPONSE_B003_v0plus_envelope_push.md) | 2026-05-02 | 2026-05-02 |
| B004 | CLOSED | V2-V5 model upgrade review (active V3/V4 + archived V2/V5 revival decision) | [PROMPT_B004_v2_v5_models_upgrade.md](PROMPT_B004_v2_v5_models_upgrade.md) | [RESPONSE_B004_v2_v5_models_upgrade.md](RESPONSE_B004_v2_v5_models_upgrade.md) | 2026-05-02 | 2026-05-02 |
| B005 | CLOSED | V5+ model upgrade review (V6/V8/V9/V10/V11/V12/V13/V14/V15-V19) — isolated round | [PROMPT_B005_v5plus_models_upgrade.md](PROMPT_B005_v5plus_models_upgrade.md) | [RESPONSE_B005_v5plus_models_upgrade.md](RESPONSE_B005_v5plus_models_upgrade.md) | 2026-05-02 | 2026-05-02 |
| B006 | CLOSED | New frontiers probe (TTT / modern optimizers / Liquid-NN / EBM / self-distillation / causal-discovery / hyperbolic) | [PROMPT_B006_new_frontiers.md](PROMPT_B006_new_frontiers.md) | [RESPONSE_B006_new_frontiers.md](RESPONSE_B006_new_frontiers.md) | 2026-05-02 | 2026-05-02 |
| B007 | CLOSED | Complementary frontier — microstructure + calibration + multi-asset + continual + score-based heads + finance-specialist OW | [PROMPT_B007_complementary_frontier.md](PROMPT_B007_complementary_frontier.md) | [RESPONSE_B007_complementary_frontier.md](RESPONSE_B007_complementary_frontier.md) | 2026-05-02 | 2026-05-02 |

## Action items by closed prompt

### B007 — closed 2026-05-02

Complementary frontier round, isolated from B001-B006 (no cross-citation). 5 of 14 paper-quantitative claims raw-fetched; decision-gating numbers VERIFIED, transfer-to-IC deltas all INFERRED.

**Modules built and smoke-passed** (commit pending):
- **E1 ACI online-tuned** [VERIFIED arXiv 2208.08401] → `src/frontier_ml/v1_upgrades/adaptive_conformal.py`. Inference wrapper, no retrain. Tracks 0.86 vs 0.90 target on synthetic 1.0→1.4 sigma shift.
- **E2 calibrated label-noise** [REPORTED arXiv 2510.17526] → `src/frontier_ml/v1_upgrades/label_noise.py`. ~0 marginal compute. Trainer flag `--label-noise --label-noise-ratio` + regime-aware extension stub.
- **§3.2 isotonic post-hoc TwoHot calibration** [VERIFIED arXiv 2311.12436] → `src/frontier_ml/v1_upgrades/isotonic_calibrator.py`. Smoke ECE 0.237 → 0.0002 on synthetic miscalibrated TwoHot.
- **§5.2 LogitClip** [REPORTED arXiv 2212.04055] → `src/frontier_ml/v1_upgrades/logit_clip.py`. Trainer flag `--logit-clip --logit-clip-tau`. Identity at eval.
- **§7.3 IQN head** [VERIFIED arXiv 1806.06923] → `src/frontier_ml/v1_upgrades/iqn_head.py`. Continuous quantile alternative to TwoHot 255-bin. Smoke fits N(0,1) median.

**Trainer flags wired in `trainer_helpers.py`**: `--label-noise`, `--label-noise-ratio`, `--label-noise-sigma-residual`, `--logit-clip`, `--logit-clip-tau`. Helper functions `maybe_label_noise(ctx, target, regime_label)` + `maybe_logit_clip(ctx, logits)`. Identity when flag OFF.

**Concedes (drop from cohort plan)**:
1. Liquidation-cascade-features-as-directional-alpha — credible academic test (SSRN 5611392) returned null on OOS. Tigro Blanc Medium "+299%/Sharpe 3.58" claim is UNTRUSTED blog. Liquidation-as-stress-gate may still have value, not as a directional return predictor.
2. MEV-as-directional-feature — cost-side phenomenon. Reclassify as execution-cost modeling input only.
3. Hyperbolic embeddings + Score-based regression + EBM — no surfaced 2024-2026 crypto evidence.
4. Born-Again iterations past Generation 2 — covered by B006 verdict; deltas under +0.003 ShIC unworth.
5. Per-asset training of V12 IF asset-as-token reframe is adopted (saves 10× backbone training).

**Queued for next-round build** (not yet shipped, sized for budget):
- CGFM residual flow wrapper (5-7 GPU-h, B007 §3.4 / E3) — distributional add. Highest-leverage Foundational item from B007.
- CDSeer drift detector (1 GPU-h wire, B007 §6.1) — orchestrator-level evidence-based retrain trigger.
- Asset-as-token reframe of V12 (8-12 GPU-h retrain, B007 §4.1).
- LoRA-per-asset adapter on shared backbone (6-10 GPU-h, B007 §4.2).
- Volatility-gated MoE-over-assets at V10 (4-6 GPU-h, B007 E4).
- FinCast zero-shot baseline (2-4 GPU-h, conditional on weights release; E5).

**Probe sequencing per RESPONSE §10** (16-25 GPU-h total for E1-E4):
1. **E1 ACI wrapper on V1.1 inference** — DONE (V1.1 BTC VAL, 535K windows). **CONCEDE on default-deployment for single-asset.**
   - Upstream gates: per-regime coverage PASS (bear 0.853 / chop 0.861 / bull 0.876, all in 0.85-0.95); aggregate coverage 0.864 (undershoots target 0.90 by 3.6pp); Spearman(width, |y|) = 0.022 (sig p=7.5e-58 but practically tiny — much smaller than the 0.239 from a 5K-window OOS smoke).
   - Downstream gate: 4 paradigm classes (continuous c/width / binary regime-gate / quantile-bin / multiplicative-modulation) all FAIL the +0.3 Sortino-lift gate. Best is width_mod with dSortino -0.035 / dSharpe +0.08.
   - **Verdict**: ACI module stays shipped (zero-cost wrapper); NOT promoted to default V*.x inference layer. Useful as risk-management input (position cap on width spikes), not as alpha-sizing.
   - **Cheap follow-ups queued**: u10 multi-asset width-aggregation test; the per-asset width may aggregate to a stronger portfolio signal. Defer until E2/E3 close out.
2. **E2 label-noise on V1.0 retrain** (4-6 GPU-h) — decision: ShIC delta ≥ +0.005 with IC stable → cohort-wide.
3. **E3 CGFM residual flow on V1.1** — DONE. Module built (synthetic smoke PASS: CRPS lift 5%, het. correlation 0.96). V1.1 BTC VAL test (20K capture, 3K iters, 256-hidden, fair bin-distribution baseline):
   - h=1: IC delta **-0.032**; CRPS vs bin-baseline **-21x worse**.
   - h=64: IC delta **-0.224** (V1.1 baseline IC +0.211 destroyed).
   - **CONCEDE on V1.1 single-asset.** V1.1's 255-bin symlog distribution is already a strong distributional baseline; CGFM underfits the narrow conditional density relative to the discrete bin head. Mean-of-samples adds noise σ/√M that washes out the directional signal.
   - **Verdict**: cgfm_residual.py module stays shipped (zero-cost utility for any future predictor whose bin head IS bottlenecked); NOT promoted to V1.x distributional add.
   - **Cross-finding with E1**: distributional inference-time wrappers on V1.x are not the bottleneck. Pivot to E2/training-paradigm probes.
4. **E4 vol-gated MoE at V10** (4-6 GPU-h) — decision: Sharpe ≥+0.3 vs flat V10.
5. **E5 FinCast zero-shot** — conditional on open weights.

⚠ Reliability budget: 36% VERIFIED, below 80% target. Mechanism-existence VERIFIED at 36%; transfer-to-our-cohort claims all INFERRED. **E1 first because the underlying paper is fully VERIFIED, the wrapper is purely additive, and failure costs nothing.**

### B001 — closed 2026-05-02

Adopted:
- **R1 → tested → REJECTED.** Kronos-small zero-shot pooled IC = +0.0292 across 1000 OOS windows; below all thresholds. Stay on scratch-pretrain plan. (See `LITERATURE.md` 2026-05-02 update.)
- **R2 — no agent prong.** Confirmed; defer to 2027.
- **R3 — adaptive log-spaced bins.** Shipped at `foundation/adaptive_bins.py`.
- **R4 — keep cross-asset lead-lag JEPA.** Cited TS-JEPA + CHARM in LITERATURE.md.
- **R5 — CPCV gate for V19+ Headline candidates.** Recorded as future gate.

Open caveat from B001 — closed 2026-05-02 via E1c:
- Time-bar A/B test ran. Pooled IC +0.0135 (vs dollar +0.0292), p=0.67, sign-flips across half the universe = pure noise pattern. Dollar-vs-time was NOT the issue. Kronos doesn't transfer to our crypto distribution regardless of bar type. Dollar bars empirically validated. (See `LITERATURE.md` Hole 9 closure + `logs/frontier_ml/kronos_baseline/time1h_e1c.log`.)

### B002 — closed 2026-05-02

Top three frontier-lab techniques transferable to a 31.7M time-series WM on 4060/8GB:
- **R1: Multi-Token Prediction (DeepSeek-V3)** [VERIFIED arXiv 2412.19437] — sequential causal-chain across horizons. +0.005-0.015 IC [INFERRED]. ~0 marginal training cost.
- **R2: Hybrid Mamba+Attention (Jamba 1:7)** [VERIFIED weights public; ratio REPORTED] — swap 1 Mamba layer for attention. +0.005-0.020 IC [INFERRED]. ~5 GPU-h.
- **R3: Native multi-modal pretrain (Llama 4 / Gemini)** [VERIFIED Llama 4 blog] — refit Prong 3 from frozen-adapter to joint pretrain. +0.010-0.025 IC [INFERRED]. ~35 GPU-h.

Scaling-law re-estimate: at 31.7M params and 50M-bar corpus we have ~1.6 tokens/param vs modern small-model norm of 1875-80,000. **Recommend shrinking foundation to 15M params** OR expanding corpus 6×.

Agent-tier verdict (R2 from B001): NO CHANGE. CryptoBench Prediction accuracy still ~6.25% [REPORTED]; multi-agent crypto Sharpe 1.54 [REPORTED] vs our XGB xsec 3.36. Defer to 2027.

⚠ Same reliability caveat as B003: 0 of 15 IC/ShIC deltas are VERIFIED (paper bodies not all fetched). Recommend E1 (V1.1+MTP A/B, 2 GPU-h) as ground-truth probe BEFORE committing the 35-GPU-h native multi-modal pretrain.

### B006 — closed 2026-05-02

New frontiers probe (isolated round per protocol §9):
- **Top FOUNDATIONAL pick: Koopman Neural Forecaster (KNF / Koopa / SKOLR family)** [VERIFIED arXiv 2210.03675: "explicitly addresses temporal distributional shifts"]. Mechanism-to-problem fit for crypto non-stationarity. ~5-7 GPU-h probe at 2M params.
- **Top training-paradigm pick: Test-Time Training (TTT)** [REPORTED — Christou et al. 2024 robustness improvements on time-series under nonstationarity]. **EXTRA at architecture level / FOUNDATIONAL at training paradigm**. Direct fit for crypto. ~2-3 GPU-h.
- **Top safe-EXTRA pick: Born-Again Self-Distillation iterated 2-3 generations** [VERIFIED arXiv 1805.04770]: 3 V1.1 generations ~9-12 GPU-h, expected per-gen +0.005-0.010 IC.
- **Optimizer probes**: Lion-SAM combo + Sophia A/B [REPORTED]; modest expectations vs SAM alone.
- **Hyena Hierarchy — SKIP at our scale.** Crossover with attention is ~6K seq length [REPORTED]; we're at seq=96.
- **CfC liquid networks** [VERIFIED 1-5 orders-of-magnitude speedup vs NODE]: dominated by Mamba in our cohort; skip unless continuous-time formulation specifically wins.
- **Causal discovery (PCMCI)**: low-priority for V1.x (Pattern P already addressed dead features); higher value at foundation prong on f121.
- **Hyperbolic embeddings**: speculative; no published crypto application.

Top experiment recommendation: **E2 TTT wrapper (2-3 GPU-h)** as cheapest ground-truth probe. If TTT lifts OOS IC ≥ +0.005, entire "non-stationarity adaptation" frontier (including E1 Koopa) gets validated by analogy.

⚠ Reliability: 31% VERIFIED on paper claims; 0% on IC/ShIC deltas.

### B005 — closed 2026-05-02

V5+ cohort review (isolated per protocol §9 — no cross-citation of B001-B004):
- **V6 (JEPA + Discriminator) — APPLY VICReg integration (C-JEPA fix)** [VERIFIED arXiv 2410.19560: VICReg "addresses the inefficacy of EMA from I-JEPA in preventing entire collapse"]. **Single highest-leverage action across V5+. 5 GPU-h A/B**.
- **V8 (Neural ODE) — STAY ARCHIVED-IN-PLACE.** Pure NODE at 4× compute is dominated by Mamba SSD on financial regime per 2024-2026 lit.
- **V9 — FORMALLY KILL.** Archive directory + remove from run_all_training.py.
- **V11 — STAY FROZEN.** 2.9M params is below MoE-sparse threshold; revisit only at >50M.
- **V13 — STAY FROZEN; TFT-GNN hybrid is V20+ candidate** (NEW architecture, not retrofit).
- **V14 (Diffusion) — REVIVE WITH CAUTION.** Conditional on R3 quantile-vector consumption probe (engineering, no GPU). If meta-learner ingests quantile vectors → V14 retrain becomes worthwhile.
- **V15 / V16 / V17 — DEFER as-is.** No published 2024-2026 financial deployment of DreamerV3 or TD-MPC2.
- **V18 — KILL CONFIRMED.** Chronos-2 is 120M apache-2.0 [VERIFIED HF card] but no published crypto IC > 0.05 with finetune; paradigm closed for our regime.
- **V19 — DEFER.** When v51 lands, do "V1.x at f29 vs f121-with-VSN" — VSN-style feature selection, not raw input-dim scaling.

⚠ Reliability budget: 31% VERIFIED on paper claims; 0% on V5+ IC/ShIC deltas. R1 (V6+VICReg, 5 GPU-h) is the ground-truth probe.

### B004 — closed 2026-05-02

V3/V4 active cohort + V2/V5 archived revival decision:
- **V3 (WaveNet) — KEEP AS-IS.** No 2024-2026 dilated-causal-conv upgrade measured to lift IC ≥ +0.01 on <2M-param crypto. Re-train at f29 with current invariants.
- **V4 (Mamba-3 + RSSM) — TWO LOAD-BEARING UPGRADES**:
  - **R1: QKNorm/BCNorm** [VERIFIED via together.ai Mamba-3 blog: "empirically stabilizes the training of Mamba-3 models"] — direct fix for V4's ShIC-decline failure mode. **2 GPU-h probe**.
  - **R2: FSQ alternative bottleneck** [VERIFIED arXiv 2309.15505: "does not suffer from codebook collapse"] — replaces 24×24 categorical RSSM. ~5 GPU-h.
- **V2 — STAY ARCHIVED.** Transformer-hybrid covered by V1.x.
- **V5 — STAY ARCHIVED.** SSM covered by V4 (Mamba-3 strictly dominates).

xLSTM and RWKV-TS surfaced as **NEW architecture classes** worth filing as V20+ proposals (NOT V2/V5 revivals).

Inter-architecture ρ estimates [INFERRED]: V1.x↔V3 ~0.80, V1.x↔V4 ~0.75, V3↔V4 ~0.70 → V10 ensemble lift modest (~10% over best single member at K=3, ρ=0.75). Recommend extending CC1 probe to measure these.

⚠ Reliability budget: 33% VERIFIED on paper claims; 0% on V3/V4 IC/ShIC deltas. R1 (V4+QKNorm, 2 GPU-h) is the ground-truth probe.

### B003 — closed 2026-05-02

Top three V1.x upgrades ranked by (IC + ShIC) lift per GPU-h, all from VERIFIED-via-raw-arxiv papers:
- **R1: SAM optimizer drop-in** (SAMformer arXiv 2402.10198, [VERIFIED]) — sharpness-aware minimization on V1.x Transformer encoder. Expected +0.005-0.015 IC AND +0.005-0.010 ShIC [INFERRED]. ~6 GPU-h cohort retrain.
- **R2: FrAug frequency-domain augmentation** (arXiv 2302.09292, [VERIFIED]) — FFT/IFFT mask augmentation. Expected +0.003-0.010 IC, +0.005-0.015 ShIC [INFERRED]. ~0 marginal training cost.
- **R3: TwoHot → Skewed-Student-t MDN head** (LSTM-MDN arXiv 2508.18921, [VERIFIED]). Expected +0.005-0.020 IC at h=1 [INFERRED]. ~5 GPU-h cohort retrain.

⚠ Response published with REPORTED-grade decision gates and 0% VERIFIED on the IC/ShIC deltas (none of the cited papers measure IC or ShIC directly). Recommendation: run R1 (V1.0+SAM) as ground-truth probe FIRST before committing the stack — see RESPONSE §8 Caveats + §9 Reliability ledger.

Compound projection if all three ship: V1.x record envelope shifts from IC=0.067/ShIC=0.037 to IC ≈ 0.080-0.095 / ShIC ≈ 0.050-0.060 [INFERRED]. Within striking distance of Headline (IC > 0.10) without foundation-prong compute.

## Conventions

- Each prompt file ends with an `## Output format` and a `## Stop conditions`
  section so the browser session can size its response.
- Each response file should ground every claim in either a paper / repo /
  benchmark URL or an explicit "expert opinion, not literature" annotation.
- The main session may issue follow-up prompts that DEPEND on a prior
  response. Mark these with `depends_on: BNNN` in the prompt frontmatter.
- Token + tone budget per prompt: 8-15 WebSearch + 3-6 WebFetch, output
  2000-4000 words, /un tone (no polite hedging). Each prompt restates this.
