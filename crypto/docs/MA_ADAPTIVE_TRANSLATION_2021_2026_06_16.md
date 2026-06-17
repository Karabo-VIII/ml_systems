# Is time-adaptation the missing x-factor for MA 2020->2021 translation? (quant referee, 2026-06-16)

> User /quant 2026-06-16: *"solve for MA. The missing x-factor is probably HOW DO WE ADAPT ACROSS TIME. MAs are
> supposed to be adaptive (other than whipsaws); if they are not adaptive on their own, we have to MAKE them so.
> Across timeframes I expect positive performance."* Tool:
> [`src/strat/ma_adaptive_translation_2021.py`](../src/strat/ma_adaptive_translation_2021.py).

## Pre-registration (stated before running, persisted in the JSON)
- **H0:** time-adaptation does NOT translate 2020->2021 better than a STATIC EMA of the same cross-structure,
  beyond a **planted-null** ("fake adaptation" = random/scrambled-ER KAMA: real adaptation magnitude, scrambled timing).
- **H1:** genuine adaptation (real efficiency-ratio/CMO timing) translates better AND the advantage **survives the
  null** (it is the TIMING of adaptation, not just extra smoothing).
- **One-sided; asymmetric loss** (false-ship a non-adaptive "win" >> false-skip). **Decision rule:** adaptive must
  beat static **AND** the null **AND** have block-bootstrap p05(adaptive-static per-TF net) > 0.
- **Isolation:** all contestants share the SAME fast/slow grid + stack (trail10+min_hold+lag+maker) + band-ensemble
  (EW 1/N). The ONLY thing that varies is the MA smoothing mechanism. STRICT long-only; PIT-core; UNSEEN sealed.

Contestants: STATIC_EMA · ADAPT_KAMA (efficiency-ratio adaptive) · ADAPT_VIDYA (CMO adaptive) · ADAPT_LOOKBACK
(EMA span self-adjusts to vol) · NULL_RANDOM_ER (KAMA with the real per-window ER **phase-shuffled** -> trades the
same, scrambles only the regime-timing).

## Result [VERIFIED-2021-forward]

| contestant | worst-TF 2021 net | frac-TF positive | sign-agree 2020->2021 | \|drift\| | worst crash | p05(adaptive-static) |
|---|---|---|---|---|---|---|
| STATIC_EMA (baseline) | **20.9%** | 1.0 | 1.0 | 12.7 | -13.8% | -- |
| ADAPT_KAMA | 14.8% | 1.0 | 1.0 | 13.0 | -10.5% | **-8.6** |
| ADAPT_VIDYA | 17.6% | 1.0 | 1.0 | **8.4** | **-5.3%** | **-10.3** |
| ADAPT_LOOKBACK | 15.9% | 1.0 | 1.0 | 13.6 | -12.7% | **-2.9** |
| **NULL_RANDOM_ER** (scrambled timing) | 23.0% | 1.0 | 1.0 | 13.1 | -- | **-1.4** |

## Verdict: time-adaptation is NOT the missing x-factor. H0 is NOT rejected.

**1. Adaptation does not translate better on net.** All three adaptive variants have **block-bootstrap p05 < 0**
vs static EMA (KAMA -8.6, VIDYA -10.3, lookback -2.9) -- robustly *lower* 2021 net, not higher. None beats static.

**2. The "adaptation" is MECHANICAL, not genuine regime-timing -- the decisive control.** The NULL_RANDOM_ER
(real adaptation magnitude, **timing scrambled**) performs **as well as static** (p05 -1.4) and **better than the
real adaptive MAs** (-8.6/-10.3/-2.9). If genuine adaptation timing carried out-of-sample information, scrambling
it would HURT -- it doesn't. So KAMA/VIDYA's edge over a plain EMA is their average *amount* of smoothing (the
fast/slow ER-window weighting), **not** when they adapt. The efficiency-ratio/CMO "adaptation signal" is
**uninformative out-of-sample**. (This is the prior /quant lesson applied: a test the null also passes isolates
nothing -- here the null passes, so the adaptation-timing hypothesis is refuted.)

**3. The cross-TF positive translation the user expects IS real -- but it comes from the BAND-ENSEMBLE, not
adaptation.** Every contestant -- static, adaptive, even the scrambled null -- is **positive across all 6 TFs in
both years** (frac-TF-positive 1.0, sign-agree 1.0). The 1/N band-ensemble + min_hold + trail already neutralize
the whipsaw the user worries about. Adaptation adds nothing on top of what the ensemble already delivers.

**4. What "adaptation" actually does is move the risk/return knob -- and you can get it from a slower static MA.**
VIDYA (the smoothest) has the lowest net but the best crash-preservation (-5.3% vs -13.8%) and lowest drift (8.4
vs 12.7). But the scrambled-timing null shows this is *smoothness*, not adaptive timing -- a slower static MA buys
the same de-risking without the "adaptive" label. Adaptation is a slower MA in disguise, not a translation unlock.

## The honest answer to "how do we get 2020 to translate into 2021 for MA"

The translation that EXISTS -- cross-TF positive in both years -- is **the de-risked-beta class translating, captured
by the BAND-ENSEMBLE (1/N over the robust band)** ([forward_ensemble_2021](ENSEMBLE_TRANSFER_2020_TO_2021_2026_06_16.md),
cf486ca). **Time-adaptation is not the missing x-factor** -- it does not beat static, its timing is uninformative
(scrambled-timing null does as well), and it does not improve the cross-TF consistency the ensemble already
saturates. The deployable MA translation remains: **the band-ensemble + de-risk sizing** -- a drawdown-preserving
de-risked beta, still < buy-hold on net (the established long-only ceiling, 0/21 both years).

**Caveats / honest bounds:** both 2020 and 2021 are BULL years -- the cross-TF positivity is bull-conditional (the
2022 bear is the real translation stress test, flagged next). The VIDYA-specific null (scramble CMO, not ER) is an
unbuilt refinement, but the cross-contestant result (all adaptives < static on net; the scrambled null ties static)
is decisive without it. n = 6 TFs (small); the p05 is a block-bootstrap over the per-TF vector. Maker cost; PIT-core
u10; UNSEEN 2025-26 sealed.

## RWYB
```
python -m strat.ma_adaptive_translation_2021 --selftest          # contestants distinct + null decoupled (PASS)
python -m strat.ma_adaptive_translation_2021                      # all 6 TFs, all contestants, the referee verdict
```
Persists `runs/strat/ma_adaptive_translation_2021_*.json` (pre-registration + per-TF 2020/2021 + the verdict).
