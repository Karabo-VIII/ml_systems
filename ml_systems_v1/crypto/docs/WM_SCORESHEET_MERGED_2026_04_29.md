# WM Score Sheet — MERGED RUBRIC (2026-04-29)

> **Single canonical scoresheet for the next session.** Merges two
> parallel reviews built today:
>
>   * [WM_SCORESHEET_2026_04_29.md](WM_SCORESHEET_2026_04_29.md)
>     -- 10-dim × 5-pt = /50 (focus: roadmap clarity, failure-mode docs)
>   * [MODEL_REVIEW_SCORESHEET_2026_04_29.md](MODEL_REVIEW_SCORESHEET_2026_04_29.md)
>     -- 8-dim weighted = /100 (focus: DSR/PBO, ROI matrix, L1-L4 taxonomy)
>
> The "internet-aware" parallel review correctly flagged three real
> rigor gaps in my version: **deflated Sharpe / PBO not measured**,
> **pairwise IC correlation V1.x family never computed**, **ROI
> per-compute-day not modeled**. This merged doc adopts those, keeps
> the action-oriented framing of the original, and resolves the
> double-doc problem by deprecating both predecessors in favor of
> this one.

---

## What each rubric got right (and wrong)

| Element | mine (/50) | theirs (/100) | merged adopts |
|---------|-----------|---------------|----------------|
| Equal-weight scoring | yes | no -- weighted | **theirs** (D1=20%, D3=15%, D5/D6/D7=10%, D8=5%) |
| OOS IC h=1 as separate dim | D3 | folded into D1 | **theirs** (gen = IC + ShIC paired) |
| ShIC as separate dim | D4 | folded into D1 | **theirs** (paired with IC) |
| Walk-forward as separate dim | D6 | folded into D2 anti-fragility | **theirs** (anti-frag includes WF stability) |
| Statistical validity (DSR/PBO) | MISSING | D3 (15%) | **theirs** (real rigor gap closed) |
| Compute efficiency | D5 | D6 | both -- same |
| Architecture novelty | D1 | D5 (novelty) | both |
| Composability (orthogonality) | D8 | D4 (15%) | **theirs** (higher weight; explicit ρ threshold) |
| Maintainability | D7 | D7 (CDAP) + D8 (maint) | **theirs** (split CDAP from age) |
| Failure modes documented | D9 | absent | **mine** (red-team + tests are first-class) |
| Forward roadmap clarity | D10 | absent | **mine** (forces hypothesis + projected delta) |
| L1-L4 diagnostic taxonomy | absent | §4 | **theirs** (when ShIC<0.015, which layer?) |
| ROI matrix (compute-days × lift) | absent | §5 | **theirs** (explicit EV/day per upgrade) |
| Cross-cutting concerns | absent | §8 (CC1-CC5) | **theirs** (pairwise corr, Phase-3 composability, etc.) |
| Top-N priority actions | implicit | §9 | **theirs** (ranked, costed) |
| Per-version close-the-gap actions | yes | yes | both -- same |

Net: theirs has the better numerical rigor (DSR, weighting, taxonomy);
mine has the better forward-action framing. Merged version takes both.

---

## Final rubric -- 10 dimensions, weighted, /100 (+5 super-tier bonus)

```
                                                  weight
D1  Generalization (h=1)                          20      <- theirs
D1+ Headline super-tier bonus (NEW 2026-04-30)    +5      <- mandate
D2  Anti-fragility (regime + WF stability)        15      <- theirs (folds my D6)
D3  Statistical validity (DSR + PBO + t-stat)     15      <- theirs (NEW)
D4  Composability (residual corr to V1.1)         15      <- theirs (NEW)
D5  Architecture novelty                          10      <- theirs (= my D1)
D6  Compute efficiency                            10      <- both (= my D5)
D7  CDAP / cross-version invariant                 5      <- theirs split
D8  Maintenance health                             5      <- theirs split
D9  Failure modes documented + tests               3      <- mine (kept)
D10 Forward roadmap clarity                        2      <- mine (kept)
                                                 ----
                                              total 100 (+5 super-tier)
```

Each dim 0-10 (D9/D10 are 0-3/0-2 respectively to fit 100 total).

### D1 ladder + super-tier (REVISED 2026-04-30 per CLAUDE.md Headline lens)

| Score | OOS IC h=1 | ShIC h=1 | Tier name |
|------|-----------|---------|-----------|
| 0    | ≤ 0       | n/a     | inverted  |
| 2    | > 0.015   | > 0.015 | Filter    |
| 4    | > 0.030   | > 0.020 | Sizer     |
| 6    | > 0.050   | > 0.030 | Trader (V1.x current) |
| 8    | > 0.075   | > 0.040 | Approaching headline  |
| 10   | > 0.090   | > 0.045 | Pre-headline          |
| **+5 bonus (D1+)** | **> 0.10** | **> 0.05** | **Headline (agent-teaching)** |
| **+10 bonus** | **> 0.15** | **> 0.07** | **Capacity-edge** (tick-level only) |

A model scoring D1 = 10 + D1+ = 5 effectively contributes **25/20** to
the total — breaking the /100 ceiling. Threshold ladder revised:

### Thresholds (REVISED for super-tier)

* **HEADLINE-TIER (agent-teaching)**: ≥ 90 INCLUDING super-tier bonus,
  ie. D1 ≥ 10 + D1+ ≥ 5 AND no other dim < 6.
* **SHIP-AS-HEADLINE**: ≥ 85/100 with no dim < 6 (Trader tier; current
  best V1.1 ~75).
* **SHIP-AS-PRIMITIVE**: 70-84 with no dim < 4.
* **VALIDATE-FIRST**: 50-69, OR 70+ with any dim < 4.
* **DEFER**: 30-49.
* **KILL**: < 30 OR D1 = 0 (no measurable signal) OR no D10 Headline-target hypothesis.

**Production-deploy-with-real-capital threshold**: HEADLINE-TIER preferred;
SHIP-AS-HEADLINE acceptable with documented Headline upgrade plan
(D10 must reference [docs/WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md](WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md)).

### Per-dimension probe (the measurement command)

| D | Probe | Source artifact |
|---|-------|------------------|
| D1 | latest train log tail; OOS IC h=1 + ShIC h=1 | `logs/<ver>/<sub>/<ver>_train_*.log` |
| D2 | regime-conditional Sharpe (bear/neutral/bull) + WF IC variance σ | `validate_world.py --regime --walk-forward` (need to wire) |
| D3 | `honest_validation.deflated_sharpe(returns, n_trials=19)` + PBO via CSCV 16 sub-periods | `src/analysis/honest_validation.py` |
| D4 | `np.corrcoef(preds_v1_1, preds_vX)` on OOS; V10 ensemble lift | new probe: `scripts/wm_pairwise_corr.py` (TODO) |
| D5 | qualitative architecture inspection vs V1.x family | `world_model.py` + WM_RED_TEAM_REVIEW.md |
| D6 | `count_parameters(model)` + GPU-hours from training log | log header + `world_model.py` |
| D7 | `python src/audit/check_invariants.py` | CDAP exit code |
| D8 | `git log --since=30d -- src/wm/<ver>/`; checkpoint `n_features` field | git + `models/<ver>/...best_ema.pt` |
| D9 | `ls memory/fix_logs/<ver>.md` + `tests/test_<ver>_smoke.py` | fix-log + tests/ |
| D10 | grep "Close-the-gap actions" in WM_SCORESHEET | this doc / WM_SCORESHEET |

---

## L1-L4 failure-mode taxonomy (when a model fails to ship)

(adopted from parallel review §4 -- no changes)

| Layer | Signature | Discrimination | Fix path |
|-------|-----------|----------------|----------|
| **L1 Architecture** | ceiling at any hyperparam | passes on synthetic regime-stationary data only | replace architecture (KILL) |
| **L2 Training** | ShIC declines mid-training | one alternate loss/schedule recovers | fix loss/schedule/regularization |
| **L3 Data** | IC stable but features insufficient | V0 (linear) also fails on same data | feature engineering / drop dead features |
| **L4 Deployment** | trains fine, fails in stack | dependency missing (e.g. V10 needs trained inputs) | sequence dependencies first |

**Discrimination ladder** (when ShIC < 0.015 on a model):
1. Re-train V0 on same data → both fail = **L3** (data).
2. Train target on synthetic regime-stationary → passes = **L1**.
3. Try MSE → Huber → TwoHot → one passes = **L2**.
4. Otherwise → **L1** (architecture choice ill-suited).

The verdict matrix (SHIP/VALIDATE/KILL) doesn't disambiguate L1-L4.
The score sheet enforces it via D3 (data) + D5 (architecture) + D2
(training).

---

## ROI matrix (compute-days × expected ShIC lift)

(adopted from parallel §5; numbers are **hypotheses to be tested**)

| Upgrade | Models affected | Cost (d) | Expected lift | EV/day |
|---------|-----------------|----------|---------------|--------|
| Pattern P (drop 5 dead features → f29) | V1.0/1/4/6, V3, V6, V8 | 0.5 | +0.005 ShIC × 7 | **0.070** |
| Pattern Q (REC_LOG_VAR_CLAMP_MIN=0.5) | V1.0/1/4/6, V3, V4, V8, V9 | 0.5 | +0.005 ShIC × 8 | **0.080** |
| Pairwise corr V1.x family (CC1) | V1 family | 0.5 | drops 1-3 redundant siblings | infra value, not ShIC |
| DSR/PBO across active versions (CC4) | all scored | 0.5 | re-classifies low-DSR models | rigor, not ShIC |
| Walk-forward 400-bar audit (CC3) | V1.x | 0.5 | confirms-or-shrinks reported ShIC | rigor |
| V12 cross-asset + V10 ensemble probe | V12, V10 | 1.0 | unique architectural diversity | high if V12 lifts |
| Triple-barrier label retrain | all TwoHot | 1.0 | +0.003-0.005 ShIC | 0.005/d |
| Conformal calibration head | V1.1, V1.4 | 0.5 | +0.02/trade EV (not ShIC) | strat-side |
| RSSM categorical 24×24 → 32×32 | V1.x | 1.0 | +0.001-0.003 ShIC | 0.002/d (marginal) |

**Highest EV per day** (recompute every cycle):
1. **Pattern Q** -- 0.5d work, 8 models hit, +0.005 each = 0.080/d.
2. **Pattern P** -- 0.5d, 7 models, +0.005 each = 0.070/d.
3. **V12 + V10 probe** -- 1d, may unlock orthogonal alpha.

---

## Cross-cutting concerns (must be measured)

(consolidated from parallel §8)

* **CC1 Pairwise V1.x correlation matrix** -- never measured. Hypothesis: ρ > 0.95 means we ship 1-3 duplicates. **Probe**: 0.5d.
* **CC2 Phase-3 composability** -- V14 outputs distribution; meta-learner consumes scalars. Reduction may negate V14's edge. **Probe**: extract q05/q50/q95/mean/var, run primitive_correlation.
* **CC3 400-bar purge gap discipline** -- reported ShIC may use shorter gap. **Probe**: re-eval V1.0/V1.1/V1.4/V1.6 ShIC at strict 400-bar.
* **CC4 DSR missing across the board** -- ShIC > 0.015 doesn't account for multi-comparison cost across 19 versions. **Probe**: 0.5d.
* **CC5 Ensemble-lift protocol** -- V10 has no recorded per-pair lift number. **Probe**: 2-model V10 grid {V1.1+Vx} for x in survivors.

Until these five run, every score below D3 is preliminary.

---

## Per-version score table -- merged rubric, /100

Numbers carry over from both predecessor docs where they agree;
disagreements resolved by re-reading the evidence cited. **Cells with
"?" require the CC probes above.**

| Ver | D1 gen 20 | D2 anti-frag 15 | D3 stat 15 | D4 comp 15 | D5 arch 10 | D6 compute 10 | D7 CDAP 5 | D8 maint 5 | D9 docs 3 | D10 roadmap 2 | TOTAL | Action |
|-----|-----------|------------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|----------------|-------|--------|
| V1.0 | 18 (0.066/0.032) | 11 | 9 (DSR ?) | 9 (corr to V1.1 ?) | 5 | 7 | 5 | 4 | 3 | 2 | **73** | SHIP-AS-PRIMITIVE |
| V1.1 | 19 (0.067/0.033) | 11 | 9 (DSR ?) | 7 (sibling V1.0) | 5 | 7 | 5 | 5 | 3 | 2 | **73** | SHIP-AS-PRIMITIVE (record) |
| V1.4 | 18 (0.068/0.031) | 11 | 9 | 7 (sibling) | 6 | 6 | 5 | 4 | 2 | 2 | **70** | SHIP-AS-PRIMITIVE |
| V1.6 | 17 (0.062/0.033) | 11 | 9 | 6 (sibling) | 7 | 6 | 5 | 4 | 2 | 2 | **69** | VALIDATE (sibling redundancy) |
| V0 | 6 (0.018) | 9 | 8 | 12 (orth floor) | 4 | 10 | 5 | 5 | 3 | 2 | **64** | SHIP-AS-BENCHMARK |
| V12 | ? | ? | ? | 13 (cross-asset) | 9 | 8 | 5 | 3 | 2 | 2 | **42 + ?** | VALIDATE FIRST (highest interest) |
| V4 | ? | ? | ? | 11 | 9 (Mamba-3) | 7 | 5 | 3 | 3 | 1 | **39 + ?** | VALIDATE (B=32 stress probe) |
| V19 | ? | ? | ? | 12 (V1.x on f121) | 5 | 7 | 5 | 3 | 2 | 2 | **36 + ?** | DEFER (gated by v51 build) |
| V3 | ? | ? | ? | 9 (WaveNet) | 8 | 6 | 5 | 3 | 3 | 2 | **36 + ?** | VALIDATE FULL at f29 |
| V6 | ? | ? | ? | 9 (JEPA) | 7 | 7 | 5 | 3 | 2 | 2 | **35 + ?** | VALIDATE |
| V13 | ? | ? | ? | 9 (TFT VSN) | 7 | 7 | 5 | 3 | 2 | 1 | **34 + ?** | VALIDATE |
| V14 | ? | ? | ? | 12 (distribution) | 8 | 5 | 5 | 3 | 2 | 1 | **36 + ?** | VALIDATE if D6 acceptable |
| V11 | ? | ? | ? | 7 (combined) | 6 | 5 | 5 | 3 | 2 | 1 | **29 + ?** | DEFER (contingent) |
| V15 | n/a | n/a | n/a | n/a | 8 (PatchTST) | 8 | 5 | 5 | 2 | 2 | **n/a** | LIBRARY-ONLY |
| V18 | ? (foundation) | ? | ? | 11 | 8 | 5 | 5 | 3 | 2 | 1 | **35 + ?** | KILL? (project concede beyond H=1) |
| V8 | ? | ? | ? | 9 | 8 (NeurODE) | 4 (RK4 4×) | 5 | 3 | 2 | 1 | **32 + ?** | DEFER |
| V10 | n/a | n/a | n/a | 15 (ensembler) | 7 | 8 | 5 | 2 | 2 | 2 | **n/a** | DEFER (needs ≥2 trained) |
| V16/V17 | n/a (RL) | n/a | n/a | n/a | 7 | n/a | n/a | n/a | n/a | n/a | **n/a** | DEFER (RL dormant) |
| V9 | 1 (0.007) | 3 | 3 | 4 | 4 (MoE) | 4 (7.6M) | 5 | 2 | 5 (well-doc'd failure) | 0 | **31** | KILL (already archived) |

(Cells partial -- "?" means CC probe needed; will land after the 0.5d
priority actions run.)

---

## Top 5 priority actions (next session, ranked by EV/day)

| Rank | Action | Cost (d) | Why now |
|------|--------|----------|---------|
| 1 | **Pattern Q across 8 models** | 0.5 | EV/day 0.080; settings-only fixes already coded |
| 2 | **Pattern P (f29) across 7 models** | 0.5 | EV/day 0.070; settings-only |
| 3 | **Pairwise correlation V1.x (CC1)** | 0.5 | resolves "ship all 4 vs ship 1" question; saves 3× inference budget if redundant |
| 4 | **DSR + 400-bar audit (CC3 + CC4)** | 0.5 | hardens every reported ShIC across all SHIP versions |
| 5 | **V12 + V10 ensemble probe** | 1.0 | highest single-model upside outside V1.x; tests composability rubric |

**Total: 3.0 compute-days** for the diagnostic battery + first
upgrades. Output: every "?" cell in the score table replaced with a
real number, ranked SHIP list shrunk if pairwise ρ > 0.95, top-line
ShIC potentially +0.005-0.010 across the V1.x cohort.

---

## How to use this doc going forward

1. **Pre-session**: read this doc + WM_RED_TEAM_REVIEW. Pick a version with the highest "expected lift × inverse cost" not yet probed.
2. **During**: run the dim probe(s). Update the row. Update CC notes. Re-rank if a measurement changes a verdict.
3. **Post-session**: append to "Per-version close-the-gap actions" in `WM_SCORESHEET_2026_04_29.md` (the per-version action sub-doc) with the new probe's number + delta vs prior. **Do not edit the rubric in this doc** unless a new dimension is needed (rubric changes invalidate cross-version comparisons).
4. **Never re-rank without re-measuring** -- cite the log line / probe output for every score change.

---

## Predecessor docs (deprecated as standalone, kept for context)

* [WM_SCORESHEET_2026_04_29.md](WM_SCORESHEET_2026_04_29.md) -- 10-dim/5-pt original.
  Per-version close-the-gap actions list still authoritative; rubric
  superseded by this doc's merged version.
* [MODEL_REVIEW_SCORESHEET_2026_04_29.md](MODEL_REVIEW_SCORESHEET_2026_04_29.md) --
  8-dim weighted /100. Diagnostic taxonomy, ROI matrix, CC1-CC5
  carried forward; per-version preliminary scores in §6 are folded
  into the table above.

If both docs disagree with this one in a future iteration, **this one
wins** -- it's the merge.
