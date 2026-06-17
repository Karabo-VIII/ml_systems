# Per-MA-type per-timeframe strategy CLUSTERS + TOP-K (the ML-harness target) -- 2020 OOS

User /orc 2026-06-13: "what works in CLUSTER/statistical form (a family of clusters of strats that work), per
MA type SEPARATELY, top-K per MA per timeframe, weaknesses solved (min-hold = FULL stack); then an ML harness
replicates the best-in-class of each." Tools: `deep2020_clusters.py` + `deep2020_clusters_export.py`. Full
distinct config grid (all speeds), FULL stack (trail10 + min_hold12 + maker), OOS = Oct-Dec 2020, Sharpe is
daily-resampled (sqrt365, comparable across TF). (30m/15m appended when their run completes.)

## WINNING CLUSTER per MA type x timeframe (struct-speed)
| MA | 1d | 4h | 2h | 1h |
|---|---|---|---|---|
| EMA | 3MA-vslow | 3MA-vslow | 2MA-vslow | 2MA-vslow |
| SMA | 3MA-vslow | 3MA-vslow | 3MA-vslow | 2MA-slow |
| WMA | 2MA-mid | 2MA-vslow | 2MA-slow | 2MA-vslow |
| HMA | 3MA-slow | 2MA-slow | 3MA-slow | 2MA-vslow |
| DEMA | 2MA-mid | 2MA-slow | 3MA-mid | 2MA-vslow |
| TEMA | 2MA-mid | 2MA-mid | 2MA-mid | 2MA-vslow |
| KAMA | 2MA-mid | 2MA-vslow | 2MA-vslow | 2MA-vslow |
| VIDYA | 3MA-fast | 3MA-slow | 3MA-vslow | 3MA-vslow |

**The cluster structure (the family of clusters that works):**
- **The winning cluster shifts SLOWER as the timeframe finens** (1d mid/slow -> 1h almost all vslow). At
  finer cadence you need slower MAs to survive the cost/whipsaw (the building-block law, confirmed at the
  cluster level). At 1h, the winner is overwhelmingly the vslow cluster across MA types.
- **The winning cluster VARIES by MA type at a given TF** -- e.g. at 1d: EMA/SMA -> 3MA-vslow, WMA/DEMA/TEMA/
  KAMA -> 2MA-mid, HMA -> 3MA-slow, VIDYA -> 3MA-fast (the adaptive MA handles noise so it can run FAST
  params). This is the "one config won't suffice" -- each MA type has its OWN best param region.
- **Within a winning cluster the configs are TIGHT (corr 0.74-0.95) and interchangeable** -- the cluster is
  the unit, not the single config (eff-N~1.2). The top-K are a STATISTICAL family (mean+-std), not a point.

## TOP-3 ML-TARGET configs per (MA, TF) -- the best-in-class to replicate (full list in clusters_ML_TARGET.json)
- 1d: EMA(12,84,148)/EMA(73,145)/EMA(19,34,48); SMA(15,27,48)/SMA(186,208,233)/SMA(12,14,30);
  WMA(22,33)/WMA(15,33)/WMA(19,34,48); HMA(48,67,94)/HMA(24,53,148)/HMA(22,128); DEMA(18,33)/DEMA(15,33)/
  DEMA(22,33); TEMA(22,33)/TEMA(15,67,233)/TEMA(12,84,148); KAMA(5,37)/KAMA(3,19,84)/KAMA(3,48,166);
  VIDYA(2,5,8)/VIDYA(2,3,4)/VIDYA(4,5).
- 4h: EMA(2,148,208)/EMA(8,132,233); HMA(31,75)/HMA(48,67,94); TEMA(102,237)/TEMA(26,45); VIDYA(4,48,84).
- 1h: EMA(37,124)/EMA(6,210); SMA(8,108)/SMA(10,248); TEMA(15,67,233); VIDYA(15,67,233)/VIDYA(8,132,233).
(Full top-5 with Sharpe/net per (MA, TF) in clusters_ML_TARGET.json.)

## THE ML-HARNESS HANDOFF
`clusters_ML_TARGET.json` is the structured target: per (MA, TF) -> {best_cluster, cluster_stats (mean+-std
net/Sharpe, within-corr, n), top5 configs, ml_target_config}. The ML harness (the other agent) consumes it to:
1. **REPLICATE** the best-in-class signal: each ml_target_config + the FULL stack regenerates its causal
   long/flat positions (no look-ahead) -> the ML learns to reproduce that position series from market features.
2. **SELECT within the winning cluster** from market state (which of the tight cluster to run now).
Because the cluster is tight (corr ~0.9), the ML target is well-posed: replicating ANY member ~= the cluster.

## PITFALLS (for the ML harness + the user)
1. **The cluster is the unit, not the config.** Within-cluster corr ~0.9 -> the "rank" among top-5 is noise;
   the ML should target the CLUSTER (or its medoid), not chase the #1 (which won't be #1 on UNSEEN).
2. **In-sample / same-regime.** These are 2020-OOS (adjacent same-bull); the cluster identities + ranks are
   NOT UNSEEN-validated. The ML must be trained/validated with a real held-out (and ideally cross-year).
3. **Sharpe rewards under-participation** (deep-dive Block leaderboard): a high-Sharpe cluster can be the one
   barely in the market (low net). The ML target should be net-aware, not Sharpe-only.
4. **All ~0.85 correlated (one beta).** Replicating the best MA cluster reproduces a de-risked drift-beta, NOT
   orthogonal alpha. The ceiling on what the ML can replicate is the beta + the (thin) calendar structure.
5. **VSLOW clusters at fine TF can be near-buy-hold** (high time-in) -- check the ML isn't just learning to
   hold (the leaderboard showed buy-hold/vol-target win on net at fine TF).

json: clusters_ML_TARGET.json + clusters_*.json. RWYB: python -m strat.deep2020_clusters --cadences <tf>;
python -m strat.deep2020_clusters_export.
