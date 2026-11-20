# DATA BUG: chimera_v51 `target_return_1_raw` and `returns_clean` columns are CORRUPTED

**Discovered**: 2026-05-23 10:35 SAST by Oracle mining audit
**Reproduced**: 6 assets (BTC, ETH, SOL, FET, SHIB, AAVE) — bug is systematic
**Severity**: HIGH (affects all downstream consumers using these columns directly)

## The bug

The `target_return_1_raw` column in `data/processed/chimera/1d/*_chimera_1d_*.parquet` is supposed to be `(close.shift(-1) - close) / close` per the producer formula at `src/pipeline/make_dataset.py:107-117`. The STORED values are:

| asset | actual fwd_1d abs_mean | stored target_return_1_raw abs_mean | ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| BTC | 0.02137 | 0.00071 | 30x smaller | +0.037 | +0.044 |
| ETH | 0.02923 | 0.00080 | 37x | +0.064 | +0.044 |
| SOL | 0.04144 | 0.00208 | 20x | +0.135 | +0.095 |
| FET | 0.05078 | 0.00530 | 10x | +0.098 | +0.087 |
| SHIB | 0.03651 | 0.00225 | 16x | +0.104 | +0.103 |
| AAVE | 0.04159 | 0.00201 | 21x | +0.056 | (TBD) |

**Both Pearson and Spearman are ≤+0.14 across all 6 assets** — the stored column is essentially noise relative to actual returns.

`returns_clean` has the same problem (correlation +0.09 with `close.pct_change()`).

## Evidence — single example (BTC 2023-10-22 to 23)

- close on 2023-10-22: $29,966.19
- close on 2023-10-23: $33,046.87
- COMPUTED `(close.shift(-1) - close) / close` at 2023-10-22 row: +0.10283 (= +10.28%)
- STORED `target_return_1_raw` at 2023-10-22 row: **+0.000355** (= +0.0355%)

That's a real 10% close-to-close move, stored as 0.0355%.

## Impact on Mining Framework 2026-05-23 outputs

| Finding | Column used | Status |
|---|---|---|
| F1 32-engine basket +0.71%/d | event_eval_rows.pnl_post_cost_pct | ✅ SAFE — RECONFIRMED at +0.583-0.698%/d on FIXED close-derived returns |
| F2 922/937 engines beat random | catalog top25_catch metrics | ✅ SAFE |
| F3 xrel discriminator finding | catalog metrics | ✅ SAFE |
| F4-F8, F9, F10 | catalog + event_eval | ✅ SAFE |
| F11 lifecycle decay (17/234 stable) | event_eval pnl_post_cost_pct | ✅ SAFE |
| F11.b stable basket -0.13%/d | target_return_1_raw | 🔴 RETRACTED (was data-bug artifact) |
| F12 listwise stbl 6.32x lift | target_return_1 (rank) | ✅ ROBUST (cross-sectional ranking survives the bug; reconfirmed at 5.86x on FIXED) |
| F13 anti-fragility | event_eval pnl_post_cost_pct | ✅ SAFE |
| F14 consensus NOT monotonic | target_return_1_raw | 🔴 REVERSED — corrected sim shows consensus IS informative (3-4 engines = +1.025%/d / 5.00x baseline) |
| F15 cluster-residual 70 BETA_DISGUISED | target_return_1_raw | 🟡 RE-VALIDATION IN FLIGHT (POV-7 re-running with fix) |
| F16 temporal seasonality | event_eval pnl_post_cost_pct | ✅ SAFE |
| F17 triple-filter ALL NEGATIVE | target_return_1_raw | 🔴 RETRACTED (data-bug artifact; FIXED variants now show +0.45-0.70%/d positive) |
| F18 critical-phenomena ARKM 4.51x | returns_clean (autocorr) | 🟡 AT RISK — AC(1)/var-susc are scale-invariant in formula but on noisy data may produce spurious lifts |

## Other project code that uses these corrupted columns

`grep -r "target_return_1_raw" --include="*.py"` finds 23+ files. Notable consumers:

- `scripts/oracle/build_setup_panel.py` — feeds the setup-filter
- `scripts/research/subday_realistic_capture*.py` — sub-day analysis
- `src/oracle/setup_filter.py` — strategy filter logic
- `scripts/oracle/mine_listwise_topk.py` (corrupted — fixed at `mine_listwise_topk_fixed.py`)
- `scripts/oracle/mine_engine_consensus.py` (corrupted — fixed at `mine_engine_consensus_fixed.py`)
- `scripts/oracle/sim_stable_basket.py` (corrupted)
- `scripts/oracle/sim_triple_filter_basket.py` (corrupted)
- `scripts/oracle/sim_decoupling_basket_audit.py` (corrupted)
- `scripts/oracle/mine_cluster_residual_audit.py` (corrupted)
- `scripts/oracle/mine_critical_phenomena_engines.py` (uses returns_clean — at risk)

**Recommendation**: any consumer of target_return_1_raw / returns_clean should be audited. Replace with `close.pct_change().shift(-h)` directly.

## Catalog rebuild status

The Wave 5 catalog rebuild (started 2026-05-22T23:24 UTC) currently shows 3 FAILED stages:
- `chimera_legacy` (rc=3221225477 = Windows access violation)
- `s3_metrics_panel` (rc=1)
- `s3_features_long` (rc=1)

Whether the rebuild produces FIXED chimera v51 data is unclear given the upstream chimera_legacy failure. This rebuild's chimera output should be SPOT-CHECKED before any downstream use.

## Workaround for downstream consumers

```python
import pandas as pd
d = pd.read_parquet('chimera_v51_<ASSET>_1d_*.parquet')
d = d.sort_values('date').reset_index(drop=True)
d['actual_fwd_ret_1d'] = d['close'].shift(-1) / d['close'] - 1   # USE THIS
d['actual_ret_clean'] = d['close'].pct_change()                   # USE THIS
# do NOT use target_return_1_raw or returns_clean from the parquet directly
```

## Hypothesis on root cause (UNVERIFIED)

The chimera v51 builder in `src/pipeline/make_dataset.py` has the correct formula for these columns. So the bug is likely:
1. A producer SHIM (post-make_dataset) that overwrites these columns with a transformed quantity (e.g., normalized or smoothed)
2. A column-collision in the parquet schema (a feature column ends up sharing the name)
3. An old refresh that wrote bad data and was never re-built

Investigation deferred — surface to pipeline owner. For immediate analysis use the workaround above.

## Action items

- [x] Re-run sims that used target_return_1_raw using close-derived returns
- [x] Document affected findings + status in EMERGENT_STORY_FINAL F-section retractions/reversals
- [x] Update HEADLINE_1PAGER with the bug + reconfirmations
- [ ] POV-7 re-validates cluster-residual on FIXED data (in flight)
- [ ] Future: re-validate F18 critical-phenomena on FIXED returns_clean (if catalog rebuild produces fixed data, use it; else compute manually)
- [ ] Pipeline team: investigate root cause of chimera v51 target_return_1_raw corruption
