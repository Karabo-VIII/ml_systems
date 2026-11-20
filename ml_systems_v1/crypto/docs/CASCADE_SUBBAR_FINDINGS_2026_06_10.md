# Sub-bar liquidation-cascade study — FINDINGS (2026-06-10)

**The first empirical test of the only un-refuted Fork-B avenue.** D67 closed bar-level
mover-capture with: *"the ONLY fork avenue that could help is LEADING DATA (on-chain/cascade
pre-event), NOT execution."* The sub-4h liquidation cell had never been tested — not because it
failed, but because intraday liquidation data did not exist in the repo (artifact A6: the daily
liq features broadcast-leak intraday). This study built the missing instrumentation from data
already on disk and ran the test.

## What was built (durable, reusable)
1. **`src/mining/liq_subbar.py`** — 1m liquidation-proxy builder from local tick aggTrades
   (132 GB, 104 assets available). Method-identical to the canonical daily approximation
   (`liquidations_approx.py`); **day-sums reconcile to the dollar** against
   `liq_daily_approx.parquet`. Deterministic (stable sort on tied timestamps). Output:
   `data/processed/liq_subbar/<SYM>_1m.parquet` — u10 built, 2021-01-01→2026-05-28,
   1974 days/asset, 0 failures, ~28M minute-bars. **This closes artifact A6's gap**: the liq
   family is now usable sub-daily, leak-free, for any future study.
2. **`src/mining/cascade_oracle.py` (r2)** — event-clock cascade study harness with matched
   nulls, RED-teamed before any pooled result was read (1 CRITICAL + 7 HIGH findings fixed;
   workflow-verified). Key discipline the audit forced:
   - **Vol-robust PASS estimands**: a suffix-max "oracle ceiling" mechanically scales with
     post-event vol (zero-edge Monte Carlo: 1.5–3× post-vol ⇒ +52–104% fake delta), so PASS
     gates on fixed-horizon DRIFT (r60/r240/r1440) and LONG-MINUS-SHORT oracle ASYMMETRY;
     the ceiling contrast is kill-only.
   - Per-split null pools + vol bins (no UNSEEN minute ever priced into TRAIN/OOS stats).
   - NULL-A = vol-matched random minutes ("is there an opportunity at all?");
     NULL-B = same 60m drop, NO abnormal liq flow ("does the liq signature add anything?").
   - 48h-cluster bootstrap (serial + cross-asset simultaneity), disjoint event windows,
     split-boundary purge, real-data coverage gates, causal-only DNA (incl. the BTC-coincidence
     fix), UNSEEN events stored as count-stubs only.

## The run (u10, 2021→2026, seed 7, params pre-registered)
231 events detected: TRAIN 139 (104 clusters) / OOS 49 (36) / UNSEEN 43 (**sealed, unspent**).
Artifact: `runs/mining/cascade_oracle_u10_20260610_184843.json`.

| Estimand (paired delta, pp) | TRAIN | OOS | jackknife (TRAIN K=1/3; median) | breadth |
|---|---|---|---|---|
| r1440 vs NULL-A (vol-matched) | +5.41 [90% CI +0.35,+12.62] | +2.65 [+0.74,+4.60] | +1.90 / +0.88; median **+0.13** | 4/9, 4/6 |
| r1440 vs NULL-B (drop, no-liq) | +2.59 [−2.65,+9.99] | **+0.02** [−1.75,+1.97] | **−1.08 / −2.12**; median −1.22 | 3/9, 3/6 |
| asym vs NULL-A | +7.47 [+0.35,+17.76] | +1.74 [−0.87,+4.37] | tail-driven | 4/9, 4/6 |
| asym vs NULL-B | +5.63 [−1.74,+16.24] | **−0.50** [−3.06,+2.20] | negative at K=1 | 3/9, 3/6 |

Pre-registered gates: PASS requires drift/asym > 0 on TRAIN **and** OOS **with breadth ≥6/10**.

## VERDICT
1. **The Fork-B liquidation premise is REFUTED at this scope (KILL).** Conditioning on the
   liquidation *signature* adds **nothing** beyond the price drop itself held-out: NULL-B drift
   delta = +0.02pp OOS (asym −0.50pp), negative under drop-top-1 jackknife, breadth 3/9. The
   information is in the DROP, not in the forced-flow signature — at 1m event-clock resolution,
   u10, 24h horizons, with the OI/funding/LSR leading features we have. The dead-list's
   concentration prediction (lesson #6) fired AGAIN: DOGE alone (+27pp/event TRAIN) carries the
   pooled mean; 3 of 139 events carry the TRAIN contrast.
2. **A real but unusable post-drop phenomenon exists vs ordinary minutes**: r1440 vs NULL-A is
   positive on both splits (OOS CI excludes 0) — sub-bar sharp drops do drift up over the next
   24h relative to vol-matched controls. But it is concentration-fragile (median +0.13pp TRAIN /
   +0.55pp OOS ≈ sub-cost), breadth-poor (4/9 assets; BTC negative), and NOT liq-specific.
   This is consistent with D47's "forced-short-cover beta" reading, now measured at 1m.
3. **Leading-feature DNA decays exactly like the bar-level version**: funding rho +0.24 TRAIN →
   +0.04 OOS (non-stationary, = mover_capture's finding). The ONE marginally stable thread:
   **oi_d24h rho −0.22 TRAIN / −0.28 OOS (p≈0.05 both)** — deeper 24h OI contraction before the
   trigger ⇒ better forward drift. Registered as an open thread, not evidence.
4. **Phase C was correctly NOT run.** The candidate failed TRAIN/OOS pre-registered gates, so no
   causal rule was built and **UNSEEN remains unspent** (43 events, count only). A future
   instance with genuinely *pre-event* external data (Coinglass heatmap proximity) inherits a
   sealed UNSEEN set and a ready harness.

## What this changes
- **Dead-list**: new entry D71 (SCOPED — see 01_DEAD_LIST.md) closing the *internal-data* half
  of the Fork-B cascade avenue. The refutation does NOT cover: external pre-event leading data
  (Coinglass liquidation-heatmap proximity — we lack the ingest), sub-24h exhaustion-confirmed
  horizons with a liq-channel that would first need to show incremental info (NULL-B disputes
  it), the short side (LO excludes), or maker-execution variants of the generic drop-drift.
- **The A/B/C fork narrows materially**: Fork B's top-ranked avenue, tested with the best
  internal data available, fails its premise. What remains of Fork B is (a) EXTERNAL leading
  data (Coinglass ~$29/mo + ingest), (b) sub-15m lead-lag (D46's untested cell), (c) info-bar
  chimera population. The generic post-drop drift (NULL-A) is real but thin/concentrated — it
  is a *bear-abstention/beta-class* observation, not an alpha channel.
- **Apparatus lesson (reusable)**: a hindsight max-statistic ("oracle ceiling") contrast is
  structurally biased toward events whenever events elevate post-window vol. Any future
  oracle-decomposition study MUST gate on mean-class estimands (drift) or sign-symmetric
  statistics (long-minus-short), never the ceiling. This is now encoded in the harness.

Repro: `python -m mining.liq_subbar --universe u10 --start-date 2021-01-01` then
`python -m mining.cascade_oracle --universe u10` (seed 7 default; git lineage in the JSON).
