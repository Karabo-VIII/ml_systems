# PER-ASSET COMPLEMENTARITY -- the final fresh-axis stage (2020 band, u10, maker)

**Axis tested (NEW):** the prior 8 deep-dive stages tailored the sleeve by TF / MA-type but ran a **UNIFORM
sleeve across all 10 assets**. This stage asks: does tailoring the **SLEEVE FAMILY per asset** (trend-adaptive-MA
vs mean-reversion-oscillator vs 50/50 blend) beat one-size-fits-all? And is "across 10 it hits somewhere"
(cross-asset breadth) the actual diversifier -- or is it the per-asset tailoring?

**Discipline:** 2020 band only (`2020-07-01 .. 2021-01-01`). SELECT the per-asset family on TRAIN+VAL
(`Jul-Sep 2020`), CONFIRM on OOS (`Oct-Dec 2020`); selection never sees OOS. Maker cost, causal, min-hold.
**MANDATORY control:** RANDOM per-asset assignment of the **same family composition** (preserves the count of
each family, shuffles which asset gets which) -- if random does as well, the per-asset choice is noise.

**HONEST PRIOR:** per-asset *config DNA* was already NOISE OOS (dead-list D62 + regime_dna). This stage tests a
strictly **coarser** cut (the sleeve FAMILY, not the fine config) + an asset-ARCHETYPE cut -- the genuinely
untested question of whether a coarse per-asset choice transfers.

Apparatus: `src/strat/per_asset_complementarity.py` (reuses the `deep2020_complementarity` / `deep2020_osc` /
`ma_2020_breakdown` mechanics). JSON: `per_asset_complementarity.json`. Charts: `charts/per_asset_archetype.png`,
`charts/per_asset_vs_uniform_vs_random.png`.

---

## VERDICT (two-sided)

**[CLAIM, OOS-confirmed, 2020-bull-only] Per-asset sleeve-family tailoring is NULL -- it does NOT beat the
uniform book OOS, and on 1d it is WORSE than random assignment.** The per-asset CONFIG-DNA NULL (D62) extends
to the **coarser sleeve-FAMILY cut**: choosing trend-vs-MR per asset on TRAIN+VAL does not transfer.

**[CLAIM, OOS-confirmed] The cross-asset BREADTH is the real diversifier, not the per-asset selection.** On 4h
the 10-asset book Sharpe (2.42) beats the mean single-asset Sharpe (1.32) by **+1.10** at avg pairwise corr 0.35
(n_eff ~2.4). "Across 10 it hits somewhere" is empirically TRUE (on book-down days >=1 asset is up 53-69% of
the time) -- but that is a property of holding 10 low-correlation assets, achieved by ANY composition (including
uniform and random), NOT by tailoring which family each asset runs.

| metric (OOS Oct-Dec 2020) | 1d | 4h |
|---|---|---|
| per-asset SELECTED net% | **7.4** | **23.3** |
| uniform-trend net% | 20.6 | 29.2 |
| uniform-MR net% | 11.9 | 8.9 |
| uniform-blend net% | 16.6 | 19.3 |
| selection pctile in random-assignment | **1.5th** | **34.5th** |
| best uniform beats selection? | YES (trend +13.2) | YES (trend +5.9) |
| avg pairwise corr / n_eff | 0.24 / 3.19 | 0.35 / 2.41 |
| breadth Sharpe-gain (book vs mean single) | -0.38 | **+1.10** |
| on book-DOWN days, >=1 asset up | 53.2% | 69.2% |

`selection pctile in random < 90` on BOTH TFs => the selection adds **no skill**. On 1d the selection is at the
**1.5th percentile** -- i.e. nearly the worst possible assignment -- because the selection criterion (TRAIN+VAL
Sharpe) picked MR for 6/10 assets (MR shone in the Jul-Sep chop), but OOS Oct-Dec 2020 was a strong TREND leg,
so the selection systematically chose the LOSING family. This is the textbook over-fit-to-the-selection-window
failure: per-asset family preference is **regime-transient, not an asset property**.

---

## 1. PER-ASSET CHARACTERIZATION -- do assets separate into archetypes?

**[CLAIM, descriptive] No clean trend-vs-MR archetype separation; the assets are one loosely-correlated cluster,
and the family that "wins" on TRAIN+VAL is unstable across TF.** Hurst(VR) spread across assets is tiny (0.165 on
1d, **0.076 on 4h**) -- all assets sit near the 0.5 random-walk line, none is a clean trender or clean reverter.

The selection FLIPS by TF, proving it is not a stable asset trait:

| asset | 1d pick | 4h pick |
|---|---|---|
| BTC | MR | (trend-leaning) |
| ETH | MR | -- |
| XRP | MR | **TREND** |
| LINK | TREND | TREND |
| LTC | MR | TREND |
| family counts | trend 4 / MR 6 | **trend 8 / MR 2** |

XRP/LTC flip MR(1d)->TREND(4h); the 1d book leans MR (6), the 4h book leans trend (8). A genuine asset
archetype would be TF-stable. It is not -- it tracks whichever family happened to fit the recent chop/trend mix
at that resolution. (Caveat: SOL and AVAX have `None` Hurst -- they started mid-band [SOL/AVAX < full 2020-H2
history], short sample; their selections are the least reliable.)

See `charts/per_asset_archetype.png` (Hurst x ER scatter colored by winning sleeve + per-asset trend-vs-MR
Sharpe bars).

## 2. SELECTION vs UNIFORM vs RANDOM (the noise test)

Covered in the verdict table. The decisive number is the **random-assignment percentile**: a real per-asset
signal would place the selection in the top decile (>=90th) of the same-composition random shuffles. It lands at
1.5th (1d) and 34.5th (4h) -- **below the random median on both**. The selection is not merely un-skilled; on the
daily it is anti-skilled because the TRAIN+VAL Sharpe criterion is negatively predictive of OOS family across
this regime boundary. See `charts/per_asset_vs_uniform_vs_random.png`.

## 3. CROSS-ASSET COMPLEMENTARITY -- breadth IS the work

`avg pairwise corr` 0.24-0.35, `n_eff` 2.4-3.2, `participation ratio` 4.1-5.3 -- the 10-asset book genuinely
diversifies (effective 2.4-3.2 independent bets from 10 assets). On book-DOWN days at least one asset is up
53-69% of the time, so "across 10 it hits somewhere" is REAL. But the diversification gain (4h: +1.10 book
Sharpe over mean single asset) comes from **holding 10 low-correlation assets at all**, which the uniform and
random books get for free. The per-asset family choice contributes nothing on top -- uniform-trend captures the
same breadth and beats the tailored book.

> The 1d row shows a *negative* breadth Sharpe-gain (-0.38) only because the per-asset SELECTED composition
> (6 MR) is a poor composition that OOS-regime punishes; with uniform-trend the breadth gain is positive. The
> diversification mechanism is sound; the per-asset selection degrades it.

---

## Where this lands

- **Converges D62 at the coarser grain.** Per-asset tailoring is noise not just at the fine-config level but at
  the sleeve-FAMILY level. The actionable cut for a per-asset book is therefore: **run the SAME (best uniform)
  sleeve on all assets and let cross-asset breadth do the diversifying** -- do not spend a per-asset selection
  budget that random beats.
- **The breadth result is a positive, reusable finding** (not the per-asset axis, but adjacent): a u10 book at
  n_eff ~2.4-3.2 / avg-corr ~0.24-0.35 is a legitimate diversification engine; the lever is *number of
  low-correlation assets*, not per-asset specialization.
- **2020-bull-only caveat (binding):** OOS here is the Q4-2020 trend leg, which is precisely why trend-uniform
  dominated and the MR-leaning selection failed. The NULL on "selection beats random" is robust regardless of
  regime (random has the same composition), but the *magnitude* of which uniform wins is regime-specific. A
  bear/chop OOS would likely flip uniform-trend<->uniform-MR while leaving the per-asset NULL intact.

**Claim tags:** all numbers are in-sample-2020 / OOS-confirmed within 2020 / bull-leg-only; maker cost; n=10
assets, ~92 OOS days (1d). NOT validated on 2021+ / UNSEEN -- this is a 2020-band deep-dive finding, not a ship
claim.
