# ML Config-Recommender V2 -- config-granular learned selection (2026-06-15)

> User mandate (2h autonomous): *"upgrade it (DON'T touch the original) to be one with the ability to MATCH, BEAT,
> or be the BEST of the variation of models ... the model should match/beat our best candidates OR fall within the
> TOP-10 of the static models."* Re [`src/strat/ml_config_recommender.py`](../src/strat/ml_config_recommender.py)
> (v1). The original is untouched; v2 is a NEW file: [`src/strat/ml_config_recommender_v2.py`](../src/strat/ml_config_recommender_v2.py).

## The diagnosed gap in v1 (why it could never top-10)

v1's candidate set was the **8 MA-type FAMILY books** (each an equal-weight of one type's slow configs) + BUYHOLD +
VOLTGT_BH. A family book is a **diluted average** -- it throws away exactly the per-config selection that makes a
config a leader. So v1's honest verdict was "converges to vol-target buy-hold" (correct) but it was structurally
**incapable** of landing in the top-10 of the *individual* static configs (e.g. the 4h robust deployables
DEMA(18,33) ~40% / HMA(18,128) ~38% OOS net). v1 was never given the granularity to compete with the variation of
models the user benchmarks against.

## The v2 upgrades (each closes that gap; all keep v1's honest apparatus)

1. **Config-granular candidates** -- the candidate set is now **individual MA configs** (the top-k by VAL net per
   MA-type from the static leaderboard `ma_top10_*.json`, **no OOS peek**) + BUYHOLD + VOLTGT_BH. The ML now selects
   a *specific* config per window, so its realized book CAN be a top-config book.
2. **Stacking ensemble** -- blend Tier-A (James-Stein ridge) + Tier-B (gradient-boosted) per-row z-scored scores; the
   tier is chosen among `{A, B, STACK}` on VAL (never OOS). Ensemble-of-learners, not either/or.
3. **Conformal abstention** -- when the top-candidate score *margin* is below a VAL-calibrated threshold, the model
   **abstains to VOLTGT_BH** (the safe default). A calibrated "only act when confident" gate.
4. **Honest switch cost** -- a config change between windows charges a maker round-trip on the first bar of the new
   segment (v1 stitched bar-returns for free; v2 makes the ML *pay* for churn). The timing-skill shuffle uses the
   **same** switch-cost accounting (consistent, not apples-to-oranges).
5. **The reframed benchmark (the user's actual bar)** -- rank the ML's realized OOS book *within* the static config
   leaderboard at the cadence. Bar **MET** if the ML OOS net lands in the **top-10** of the static models OR
   matches/beats the best **robust** deployable config (min |drift|, positive VAL).

Kept verbatim from v1: rank by NET (wealth) not Sharpe; SELECT on VAL never OOS; past-only causal features
standardized on TRAIN; the hardened timing-skill test (ML must beat the MEDIAN of N>=100 block-shuffle re-timings of
its OWN picks with one-sided p<0.10); VOLTGT_BH / BUYHOLD / ORACLE / RANDOM / STATIC-pick controls; a two-sided
SELFTEST that PASSES; fixed-EW (unlisted=cash, cadence-invariant); long-only spot lev=1; maker cost; UNSEEN untouched.

## RESULT (2020 runway, TRAIN Jan-Aug / VAL Sep / OOS Oct-Dec; [VERIFIED-2020-OOS])

| TF | ML OOS net | rank in static (of 120) | top-10? | matches robust? | beats VOLTGT_BH? | timing skill | abstain |
|---|---|---|---|---|---|---|---|
| 1d | 43.2% | **8 / 120** | YES | YES (vs TEMA 30.9%) | YES (42.6%) | no (p=0.17) | 0.58 |
| 4h | 62.9% | **1 / 120** | YES | YES (vs SMA 28.6%) | YES (50.0%) | no (p=0.12) | 0.25 |
| 2h | 63.6% | **1 / 120** | YES | YES (vs KAMA 34.0%) | YES (54.6%) | no (p=0.16) | 0.58 |
| 1h | 47.6% | 35 / 120 | no | YES (vs TEMA 29.2%) | tie (47.6%) | no (p=0.0) | 1.0 |

**Robustness (k=5 candidate pool, vs k=3 above):** 4h rank **2/120**, 2h rank **2/120** -- both still TOP-10 +
MATCHES-ROBUST. The result is **not** a candidate-pool-size artifact.

### The bar is MET at 4/4 cadences -- and the ADVERSARIAL ABLATION decomposes exactly *why* (cadence-dependent)

- **MET (realistic deployment, beta-holds allowed):** the v2 book lands in the **top-10 of the static models at 3/4
  cadences** (rank 1-8 / 120) and **matches/beats the best robust deployable config at 4/4**, and **beats VOLTGT_BH
  on NET at 3/4**. A genuine, robust (k=3 vs k=5 -> rank 1-2/120) improvement over v1, which converged to VOLTGT_BH
  and was structurally unable to top-10. No leak: ORACLE (89-149%) sits far above the ML (43-63%), as a causal model must.

- **The `--no-universals` ablation (forbid BUYHOLD/VOLTGT_BH -> MA-config SELECTION only, apples-to-apples vs the
  single-MA-config static models)** answers the red-team question "is the win just being allowed to hold beta?":

  | TF | full v2 rank in static | MA-config-only rank | config-only timing skill |
  |---|---|---|---|
  | 1d | 8 / 120 | 29 / 120 (not top-10) | no (p=0.21) |
  | 4h | 1 / 120 | 74 / 120 (not top-10) | no (p=0.72) |
  | 2h | 1 / 120 | 17 / 120 (not top-10) | **YES (p=0.07)** |
  | 1h | 35 / 120 | **8 / 120 (TOP-10)** | no (p=0.15) |

- **The honest, cadence-dependent decomposition:**
  - **Coarse TF (1d / 4h):** the full-v2 top-10 came from **being allowed to hold the beta** (vol-target/buy-hold).
    Forbidding it drops the book to rank 29-74/120. At these cadences holding the beta IS the better policy and the
    model correctly learns to lean on it (no config-selection timing edge).
  - **Fine TF (1h / 2h):** genuine **config SELECTION** carries value. At **1h**, the MA-config-only book lands
    **rank 8/120 (TOP-10) on configs alone** -- better than letting it abstain to beta (rank 35/120 in full v2).
    At **2h**, the MA-config-only book shows the session's **first significant timing skill (p=0.07)** -- forced to
    select, it beats the median of re-timings of its own picks. This is exactly where v1's family-dilution hurt
    most, so config-granularity helps most here.
  - **Caveat on the 1h config top-10:** the static fine-TF configs are themselves OOS-lucky (the MA_TOP10 doc warns
    1h configs win OOS Sharpe but are high-+drift / OOS-lucky); the 1h book's timing is NOT significant (p=0.15), so
    its top-10 is selection-of-which-(OOS-lucky)-config-region, not a clean timing edge. The 2h significant timing
    (p=0.07) is the cleaner genuine signal.

### What this means (sharper than the first-pass read, consistent with the drift-beta verdict)

v2 answers the user's bar: the upgraded ML **matches/beats the best candidates and lands in the top-10 of the static
models** (v1 could not). The ablation shows the edge is **asymmetric across cadence**: at coarse TF it is the
**drift-beta verdict expressed as a selection policy** ("hold the beta, tilt only when confident"); at **fine TF
(1h/2h) it is a genuine config-granular SELECTION edge** (1h tops-10 on configs alone; 2h has significant timing).
The improvement over v1 is real and is largest exactly where v1 was weakest (fine TF, family-dilution). It is honest
deployment-quality + a small fine-TF selection edge -- not a break of the internal-data ceiling (no clean timing
alpha at coarse TF; the bull beta remains the dominant return source).

## RWYB

```
python -m strat.ml_config_recommender_v2 --selftest                 # two-sided (PASS): skill->timing skill+low abstain; noise->abstain+no skill
python -m strat.ml_config_recommender_v2                            # 2020 runway, MA family, {1d,4h,2h,1h}
python -m strat.ml_config_recommender_v2 --cadences 4h --k-per-type 5
python -m strat.ml_config_recommender_v2 --no-universals            # ADVERSARIAL ablation: MA-config selection ONLY (no beta holds)
```

Persists `runs/strat/ml_config_recommender_v2_*.json` (repro: git_sha + cost + splits + k + n_shuffle + the
per-TF static-benchmark verdict). The original v1 (`ml_config_recommender.py`) is untouched and still runnable.
