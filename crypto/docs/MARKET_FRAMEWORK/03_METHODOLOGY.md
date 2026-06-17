# 03 — Methodology & Validation Discipline (how we decide something is REAL)

The project's hard-won validation method. Every gate here was born from a specific failure (the "why" column).
**STATUS:** BINDING (current) · DEMOTED (kept as diagnostic, not primary verdict) · SUPERSEDED. The current gate chain
lives in `src/strat/` (candidate_gate → leak-probe → firewall → benchmark → battery) + CDAP pre-commit.

### Arc (5 lines)
The project began trusting numbers that looked good. Four interlocking failures fired: (1) simulator MtM double-count
inflated every curve 5×; (2) K-selection on future returns inflated a headline 14×; (3) tautological cell conditioning
fabricated 100%-hit alpha; (4) single-seed ML locked in init-luck. Each produced a binding gate. By v8.5 the gates
hardened (n_eff, DSR at true family-N, mechanism falsifiers, canonical seeds). At the 2026-06-04 reset the IC-primary
paradigm was banned and the alpha-frame replaced by returns-not-alpha; the apparatus was rebuilt as a sequenced chain
with **two-sided soundness** (a positive control must ACCEPT a genuine signal, not just reject ghosts) + unskippable CDAP.

## The register

| Principle / gate | What it enforces | Why (the failure that birthed it) | Status | Where |
|---|---|---|---|---|
| **MtM-no-double-count** | per-bar PnL = MtM only; no cumulative ret_from_entry; skip entry-bar | 2026-04-22: (2N−1)/N inflation; champion +501%→+94% | BINDING | sim guards; check_invariants |
| **Maker p_fill 0.21–0.40** (not 0.80) | budget p_fill∈[0.25,0.50]; live = 50–75% of fixed-backtest | OHLC replay: real p_fill 0.21–0.40 vs assumed 0.80 | BINDING | maker_cost_calibration.yaml |
| **Taker baseline 0.24% RT** | every candidate at taker; below-taker triggers warning | maker-only survival rejected post-calibration | BINDING | candidate_gate (TAKER_COST_RT) |
| **Walk-forward purge gap (400 bars)** | purge between train/val/OOS/UNSEEN; no normalization leak | xsec ranker had purge=0 (G-AUDIT-002) | BINDING | split_four_way; CDAP |
| **UNSEEN touched once** | dark holdout; never in any search; one verdict | repeated OOS-overfit (OOS +58% → UNSEEN +0.1%) | BINDING | strat/README; gate chain |
| **Look-ahead in selection (4-bounds)** | K-selection past-only; report BEST/SIGNAL/RANDOM/WORST-K | V2 sorted by future return → +468% (honest +33%) | BINDING | honesty_no_inflation; Layer-2 |
| **No full-history normalization** | rolling-only z/quantile/BOCPD | BOCPD warm-up used full-history std (G-AUDIT-011) | BINDING | CDAP leakage cat |
| **Tautology / outcome-conditioning guard** | no conditioning on outcome; 100% hit @n<50 = bug; symmetry check | DNA cells keyed by (def,side) → 100% tautological hit; 0/50 real Sh≥1 | BINDING | decompose.py diagnostics |
| **Multi-seed ML gate (N≥10, p05/p95)** | ≥10-seed median + p05/p95 before promotion | LSTM +44.6%/DQN +40.9% → 10-seed medians −6.7%/−34.5% = init-luck | BINDING | feedback_multi_seed |
| **Mechanism falsifier** | per-trade returns + top-3% + jk K0..5 + filter keeps/drops; CDAP exit2 | INST-A "filter strips top-tail" was FALSE — it kept the top 3 | BINDING | claim_contract; check_wealth_bot_claims |
| **n_eff (Herfindahl effective-N)** | ≥15 strict (Lens A) / ≥8 pragmatic (Lens B); 1–2-trade book not bankable | all 5 PEPE candidates n_eff 4.4–8.8 | DEMOTED to diagnostic (setup-chaser) / BINDING strict | battery.py |
| **DSR / Holm at true family-N** | deflated Sharpe at total-candidate-N; p>0.05 = refuted | 5 PEPE candidates DSR p 0.34–0.91 at N=1000 | DEMOTED from primary verdict (returns-not-alpha) | check_dsr_holm |
| **Block-bootstrap p05>0** | 2000-resample block=5; p05>0 (seed-robustness for rules) | F6 off-by-one biased all percentiles | BINDING | battery.py |
| **Jackknife K=2,K=3** | compound + after dropping top-2/3; K=3 = cap | PEPE 0/81 survived jk K=2 (2 trades = everything) | DEMOTED to diagnostic (few-big-trades IS CTA design) | battery.py |
| **Random-entry firewall null** | beat cost-matched random-entry (same count/durations) | a random-firing strat in a bull looks real = beta in disguise | BINDING (PRIMARY) | firewall.py |
| **Regime-matched null** | gated strats draw null from gate-ON bars only | a bull-only gate inflates beat-rate via null bear entries | BINDING (gated) | firewall(regime_matched) |
| **Membership-matched null** | draw null from within the same multi-candle move | a move-selector beats a whole-window null by selection alone | BINDING (setups) | firewall(membership_matched) |
| **Benchmark-excess / beat-passive / bear_preserved** | beat beta-matched costless passive at own time-in-market f; lose less in bears | times-bull-exposure but holds beta looks good vs firewall | BINDING | benchmark.py |
| **Leak probe (relative_leak_test)** | Δpp vs known-clean twin at same noise floor | shift_sensitivity over-triggered on coarse bars (false alarms) | BINDING | leak_probe |
| **Selectivity (beats flat/passive exposure)** | E[ret\|signalled] > E[ret\|flat] | a setup that always fires has good bull expectancy = buy-hold | BINDING (primary precision) | battery.evaluate_setup_chaser |
| **Positive control (two-sided soundness)** | the gate must ACCEPT a synthetic genuine signal, not only reject | a reject-everything sieve is uncalibrated | BINDING | positive_control; synthetic_positive_control |
| **UNSEEN-selection guard** | no selection loop may read UNSEEN returns to pick configs | per-asset chaser +36.8% was UNSEEN-selection-inflated | BINDING | discover.py F5 |
| **Wealth-not-Sharpe** | rank by held-out compound; robust=gate, Sharpe=tiebreak | over-recommended Sh1.92/+50% vs Sh1.45/+70% | BINDING (PERMANENT) | NORTH_STAR §3.1 |
| **Returns-not-alpha** | optimize robust compound; beta/regime/carry are GOOD; alpha-tests = honesty checks only | dismissed a 12–45% CAGR book as "not alpha" | BINDING (supersedes alpha-frame) | feedback_returns_not_alpha |
| **IC banned as primary metric** | unit = SETUP across a MOVE; IC only a >0.015 WM diagnostic | instances anchored on IC, concluded "no signal", stopped | BINDING (post-reset) | CLAUDE.md ARCHIVED |
| **2-phase wealth-bot + imagine-frame** | P1 robust discovery; P2 oracle-refine under "imagine another instance did P1" + pre-reg + asymmetric loss | oracle F1 "+28pp" collapsed under honest TRAIN+VAL-only fit | BINDING (wealth-bot ctx) | WEALTH_BOT_FRAMEWORK |
| **Sample-size discipline (stressed gate)** | n<20 → +25pp threshold on min(baseline, K2+S9), not baseline alone | INST-A +2.25% stressed-pass but K=3 → −5% | BINDING | claim_contract |
| **PBO/CSCV (backtest-overfitting probability)** | a discovery-SEARCH winner needs PBO < 0.10 via combinatorially-symmetric CV; orthogonal to DSR (deflates the selection PROCESS over a candidate family, not one Sharpe) | battery carried DSR as a caller-NOTE only (line 133), computed no PBO — a generator at scale (10³–10⁵ candidates) would smuggle in OOS-losers | BINDING (strat-discovery layer; BUILT 2026-06-09) | strat/pbo_cscv.py |
| **QS1–QS6 + canonical seeds** | pre-compound gates (n_eff, DSR-feasible, L2-capture≥0.4, mechanism fields, leak-grep=0, repro block) + {bag,feat,rng} seeds | gated on compound before proving statistically real; 0/5 had canonical_seeds | BINDING (wealth-bot) / partially superseded by gate-chain | WEALTH_BOT v8.5 §SM16-17 |
| **Pattern T / CanonicalHarness** | next-bar-open/close fills; reject inline same-bar-close fill | 12 Pattern-T scripts had silent same-bar-close inflation | BINDING | harness.py MIGRATION_BACKLOG |
| **RWYB (run-what-you-build)** | every change runs on real data pre-commit; cmd+result in commit body | Phase-2 silent exit0 @17% coverage; 25GB to wrong dir | BINDING (LAYER-1) | feedback_run_what_you_build |
| **CDAP (contract-driven audit)** | pre-commit check_invariants; exit2 HALT; unskippable | 27-file commit hid broken strat imports | BINDING | check_invariants; mandatory_gates.yaml |
| **Double-audit (two-stage RED TEAM)** | build-time caller-search/py_compile + pre-commit re-import + Sonnet red-team; verify agent claims vs code | CRITICAL unimportable strat layer landed (no e2e audit) | BINDING | DOUBLE_AUDIT_PROTOCOL |
| **Pre-delivery self-audit (two-layer)** | every agent self-audits before ANY deliverable; coordinator per-commit gate | META reported wrong numbers before auditor caught (R46/51/54) | BINDING | STANDARDS §6 |
| **Setup-chaser book vs single-edge** | a BOOK over a coverage lattice (TF×regime×cluster); per-asset configs; robustness = book + portfolio corr | jk-as-rejector killed legit trend-following (CTA = 90% on 10% of trades) | BINDING (current modus) | battery.evaluate_portfolio; SETUP_CHASER_METHODOLOGY |
| **Oracle decomposition** | treat ROI as oracle-attainment; decompose DNA; fit P(oracle-entry\|past-only) w/ shuffle+positive control; min-move-net = design var | bar-level entry-timing kept finding null because the oracle objective was a scalp (unfair for trend) | BINDING (prior question) | ORACLE_DECOMPOSITION; oracle_ceiling_builder |
| **Retail pragmatic robustness (K=3 cap)** | jk2 AND jk3 +; ≥60% months +; 80/20 OK; K=5+ = overfit-on-rejector | n_eff≥15 + DSR family-N=240 returned 0 survivors | BINDING (user mandate, overrides institutional when endorsed) | feedback_retail_pragmatic |
| **Empirical probe for numerical issues** | 200–300 real-data steps under AMP at B=32; track h_seq.max(); commit only after probe | V3 had 3 failed NaN fixes (6 GPU-hrs) without probing | BINDING | CLAUDE.md §12 |
| **Self-improving loop + failure catalog** | re-rank EV frontier on new learnings; design from prior fixes; log dead-ends (this dir) | autonomous instances re-mined refuted veins (no persistent dead-list) | BINDING | feedback_self_improving_loop |

**The one-line method:** *a finding is real only if it beats a cost-matched (taker) random-entry null AND a
beta-matched passive on UNSEEN-once data, with the win not carried by 2–3 trades (jk/n_eff as diagnostics), the
mechanism empirically falsified, and the whole thing reproducible — and the gate itself proven to accept a genuine
synthetic signal.*
