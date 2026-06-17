# Wealth-Bot Framework v8.5 Amendments (2026-05-26)

> Author: META (Opus 4.7) via MAXX autonomous session, INST-MAXX-2026-05-26-NIGHT.
> Trigger: 2026-05-26 audit found ALL 5 published PEPE × MA/EMA candidates fail
> Phase 1 n_eff gate (n_eff < 15) AND fail Deflated Sharpe at family N=1000 (all p > 0.91).
> Provenance: `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md` §1 F1-F2.
> Status: BINDING from 2026-05-27 00:00 SAST forward.

## Why v8.5 exists

v8.4 closed entry-vs-exit separation (SM11) and exit-mechanism exploration (Phase 2.5). But the post-v8.4 audit revealed structural deficiencies that no entry/exit/cadence work could fix:

1. **n_eff was advisory, not gating** — Phase 1 candidates with n_unseen ≥ 15 but Herfindahl-weighted n_eff < 15 were promoted. ALL 5 leaderboard candidates fail this in retrofit (R12 n_eff=7.07, R23a n_eff=8.81, R23c n_eff=8.81, P4 n_eff=4.55, R23h n_eff=4.43).
2. **DSR was never enforced as a ship gate** — F56 finding (2026-05-23) showed N=937 family kills all baskets at p>0.34; the leaderboard never re-priced for family size. Retrofit: ALL 5 candidates fail DSR p<0.05 at N=1000.
3. **Repro contract non-compliant** — 0 of 5 sampled audit JSONs include `canonical_seeds` key per architect 2026-05-25 mandate (bag/feat/rng triple).
4. **12 active Pattern T scripts** lurk in `scripts/wealth_bot/` — silent same-bar close fill (look-ahead) inflation surface.
5. **Indicator subtypes were conflated** — (PEPE, MA/EMA) treated SMA and EMA as one dossier. They are different signals. Per v8.5, separate dossiers.
6. **Multi-timeframe was aspirational, not gated** — v8 mandated cadence × regime × TI × approach grid, but candidates shipped on single-cadence (4h) without explicit multi-timeframe coverage required for "quality signals" per user mandate.

## v8.5 amendments (binding)

### §SM12 — n_eff GATE BAKED INTO PHASE 1 (replaces advisory)

**Rule**: Phase 1 ship requires BOTH `n_unseen ≥ 15` AND `n_eff ≥ 15` where `n_eff = 1 / Herfindahl(|per_trade_returns|)`.

**Mechanism**: candidates with high `n_unseen` but high top-trade concentration have effective sample size << n. A candidate with n_unseen=20 but H=0.5 has n_eff=2.0 — two trades carry the entire signal. Block-bootstrap p05 and jackknife K-removal are unreliable below n_eff < 15.

**Falsifier**: if a candidate passes all other Phase 1 gates but n_eff < 15, status = `REFUTED_AT_PHASE1_NEFF_GATE`.

**Override**: candidates with n_unseen < 30 may use stress-bootstrap (1000 resamples) AND combined K2+S9 jackknife stress test — if stressed compound passes G1 floor (i.e., not just compound_raw), can proceed under PARTIALLY_PASS tag.

**Enforcement**: `claim_contract.py` `PHASE1_N_EFF_MIN = 15`; CDAP `check_wealth_bot_claims.py` exit 2 on `phase1_n_eff_gate.passes == False AND verdict ∈ {SHIP, PARTIALLY_PASS}`.

### §SM13 — DSR PRE-SHIP GATE (canonical script enforced)

**Rule**: Every SHIP-tier or PARTIALLY_PASS candidate MUST report DSR at family-N where N = total candidate count in the project's audit JSON corpus.

**Mechanism**: López de Prado 2014 — observed Sharpe must exceed the null's expected-maximum-Sharpe under the test family. With small n and large family, the expected maximum is high, making naive Sharpes uninformative.

**Falsifier**: DSR p-value > 0.05 at N=actual-family-size → `REFUTED_AT_PHASE1_DSR_GATE`.

**Caveat for n < 30**: canonical `scripts/oracle/deflated_sharpe.py` hard-gates at n<30 (returns `status: thin`). For candidates below this floor, BOTH conditions must hold:
- Pre-registered hypothesis (Phase 2 imagine-frame applies retroactively if not already pre-registered)
- ≥30 forward-frozen paper-trade trades accumulated before strict-ship promotion

**Enforcement**: `claim_contract.py` field `dsr_at_family_N_p_value` REQUIRED; CDAP exit 2 if missing on ship candidates.

### §SM14 — INDICATOR SUBTYPE SEPARATION (binding)

**Rule**: Each (TI_subtype, ASSET) is a SEPARATE dossier. SMA / EMA / WMA / Bollinger / etc. are NOT collapsed into "MA family."

**Rationale**: SMA(N) and EMA(N) have materially different lag and noise characteristics; their winners under any given filter library differ. Treating them as one dossier hides which sub-family carries the signal.

**Migration**: existing (PEPE, MA/EMA) dossier formally split into:
- `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` (NEW)
- `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` (NEW)
- Old `docs/dossiers/PEPE_MA_EMA_*` files archived to `docs/dossiers/archive_pre_v8_5/` (do not delete — historical reference)

**WMA**: prior R12 used WMA(10,30); WMA receives its own dossier when work begins. For now, the R12 WMA candidate is REFUTED-UNDER-STRESS along with all other (PEPE, MA/EMA) candidates and is NOT carried into any v8.5 dossier without re-mining.

### §SM15 — MULTI-TIMEFRAME COVERAGE MANDATE (binding)

**Rule**: Every Phase 1 mining sweep MUST cover ALL five canonical cadences {15m, 30m, 1h, 4h, 1d}. Phase 1 ship requires ≥1 SHIP at any single cadence, but DSR family-N includes results from ALL cadences (not just the winner's).

**Rationale**: prior dossiers shipped on 4h alone (R57b1/b1b/b1c refuted 1h, R57b2 refuted 15m+1d, R57c refuted 30m). The DSR family count under-stated the search space when single-cadence framing was used.

**NULL cells valid**: a cadence may legitimately fail (per v8 r8 stratified mining mandate) — the requirement is COVERAGE, not coverage-and-pass.

**Enforcement**: dossier §1A grid must enumerate cadence dimension with status per cadence: SHIPPED / REFUTED / INCONCLUSIVE / PARTIAL / NOT EXPLORED.

### §SM16 — QUALITY-SIGNAL GATES (BEFORE compound gates)

**Rule**: Every Phase 1 candidate must clear quality gates BEFORE any compound-based decision is made:

| Gate | Spec | Threshold |
|---|---|---|
| QS1 — n_eff | Herfindahl-weighted effective sample size | ≥ 15 (or stress-override per §SM12) |
| QS2 — DSR feasibility | Per-trade returns can be DSR-tested | n_eff ≥ 30 for direct DSR, else paper-trade path |
| QS3 — L2 capture | mean realized/available within signal-valid window | ≥ +0.40 (BROKEN if < 0) |
| QS4 — Pattern Q | per_trade_returns_sorted_desc + top_3_pct + jackknife K=0..K=5 + combined_K2+S9 + mechanism_falsifier all populated | all present |
| QS5 — Look-ahead clean | Pattern S/T/U grep on producing script returns 0 hits | 0 |
| QS6 — Repro block | git_sha + chimera_mtime + canonical_seeds (bag/feat/rng) + schema_version present in audit JSON | all present |

**Rationale**: prior framework gated on compound (G1-G6) without first proving the candidate is statistically real. Compound on a top-3-trade-driven candidate is illusory.

**Enforcement**: `claim_contract.py` quality_signals_gates dict REQUIRED; CDAP exit 2 if any QS fails.

### §SM17 — CANONICAL_SEEDS BAG/FEAT/RNG TRIPLE (binding for ML+random)

**Rule**: Every audit JSON that involves randomness (LGBM seeds, bootstrap RNG, RNG decorrelation in signal_picker, dropout, etc.) MUST include:

```json
"canonical_seeds": {
  "bag_seed": <seed>,
  "feat_seed": <seed + 1000>,
  "rng_seed": <seed + 7919>
}
```

Per architect 2026-05-25 RNG-decorrelation recommendation. Plain `seeds: [array]` no longer satisfies the contract.

**Enforcement**: `claim_contract.py` `canonical_seeds` field REQUIRED when `uses_randomness == True`; CDAP exit 2 on missing.

### §SM18 — POST-v8.3 HARNESS REQUIRED FOR PHASE 1+

**Rule**: All Phase 1 mining scripts MUST import from `framework.data_loader` (next_bar_open / next_bar_close fill modes only) OR `wealth_bot.harness.CanonicalHarness`. No inline `simulate()` functions with `entry_p = closes[i]` (Pattern T) accepted.

**Migration**: 10 of the 12 active Pattern T scripts identified 2026-05-26 are added to `harness.py::MIGRATION_BACKLOG` with FREEZE flag — results from these scripts are advisory-only, not deploy-eligible.

**Enforcement**: pre-commit hook greps for Pattern T patterns in new scripts; CDAP fails on imports from FREEZE-flagged scripts in ship-candidate provenance chain.

### §SM19 — WEALTH-FIRST RANKING (user mandate, formalized)

**Rule**: Cross-bot leaderboard ranks by compound_discounted (= compound_raw × min(1, n_eff/12)) per `claim_contract.py SR1_4_FULL_CREDIT_N=12`. Sharpe is tiebreak only. Robustness (10/10 seeds, p05>0, max DD<30%) is the CONSTRAINT, not the RANKING.

**Carry-forward**: this matches the 2026-05-24 user mandate at `feedback_wealth_not_sharpe.md`; v8.5 formalizes it as the leaderboard sort key.

## What v8.5 RETIRES from prior frameworks

- **v8.2 SM9.1 conditional max-hold extension**: retained but with stricter §SM18 enforcement
- **Combined (PEPE, MA/EMA) dossier**: archived; replaced by separate (PEPE, SMA) and (PEPE, EMA) per §SM14
- **Advisory n_eff** (v8.4): promoted to PHASE 1 GATE per §SM12
- **Single-cadence Phase 1 ships**: REJECTED — must cover all 5 cadences per §SM15

## v8.5 First-Use Dossiers

- `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md`
- `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md`

Both start at 0% exhaustion under v8.5 gates. Prior (PEPE, MA/EMA) work is REFERENCE ONLY, not credit.

## Provenance

- 2026-05-26 ~22:00 SAST: MAXX gap-closure session identified F1 (n_eff fail), F2 (DSR fail), F3 (L2 broken on R23a), F4 (R62/R23a fork), F5 (12 active Pattern T), F6 (R12 ship_claim missing), F7 (repro non-compliant) → necessitated framework upgrade.
- 2026-05-26 23:30 SAST: user mandate "go back to the beginning with upgraded framework; expect real and better results; wealth building, quality signals, multi-timeframe; autonomous 8h" → v8.5 spec drafted by META.
- 2026-05-27 00:00 SAST: v8.5 BINDING.
