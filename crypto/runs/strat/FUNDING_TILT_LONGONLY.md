# Long-Only Funding-Tilt -- the one untested long-only WEALTH angle

**Verdict: ARTIFACT.** A strict long-only spot book that tilts its weights toward low/negative-funding
assets does **NOT** harvest a wealth edge. The tilt-alpha (tilt minus EW) is **negative in every cell of
the entire sweep** -- in-sample and held-out, gross and net. The market-neutral funding-dispersion edge does
**not** survive amputation of the short leg: the long leg alone is wrong-signed for wealth in spot.

- Script: `src/strat/funding_tilt_longonly.py` (selftest 6/6 PASS)
- JSON: `runs/strat/funding_tilt_longonly_20260616_020859.json`
- Charts: `runs/strat/funding_tilt_vs_ew.png`, `runs/strat/tilt_alpha_by_regime.png`
- Run: `python -m strat.funding_tilt_longonly --universe u50` (panel: 2338 days x 48 assets, 2020-01-03..2026-05-28)

---

## The claim under test

Use the funding signal to **TILT a STRICT long-only spot book's weights** -- overweight LOW/negative-funding
assets (cheap / paid to hold), underweight HIGH-funding (expensive). ALL weights >= 0, sum to 1, net-long,
ZERO short logic. This is distinct from the ruled-out market-neutral carry: it captures only the long leg via
reweight, never shorts. Does this long-only funding-tilt translate as a genuine, robust held-out wealth edge?

## Pre-registration (stated before the run)

- **H0**: the long-only funding-tilt does NOT beat the EW-long-only baseline on held-out wealth (tilt-alpha p05 <= 0; book is just net-long beta).
- **H1**: positive, robust tilt-alpha on the held-out path (OOS AND UNSEEN), block-bootstrap p05 > 0, surviving max-stat deflation.
- One-sided (ship only if tilt beats EW). Asymmetric loss: false-ship >> false-skip.
- **Decisive statistic**: the **tilt-alpha** daily series = (tilt net) - (EW net) on the *identical* PIT roster
  with *identical* cost. Both books are net-long the same names, so **beta cancels in the difference** -- the
  tilt-alpha isolates the funding signal's pure long-only cross-sectional contribution.
- Deflate: 12 variants swept (4 tilt strengths x 3 lookbacks) -> max-stat / PBO on the grid.

## Discipline (mechanically verified, selftest 6/6)

| Invariant | Test | Result |
|---|---|---|
| STRICT long-only (w>=0, sum=1, net-long, no short) | random signals x strengths | PASS |
| strength=0 == EW exactly (shared code path) | 1/n identity | PASS |
| monotone tilt (low funding overweighted > 1/n > high funding) | ordered signal | PASS |
| PIT survivorship (NaN-next-bar name excluded from pool, weights renormalize) | dead-asset probe | PASS |
| **Leak guard** (future funding spike does NOT alter past signal; lag=1) | spike injection | PASS |
| tilt-alpha cancels beta (flat funding -> tilt==EW -> alpha==0) | max|alpha|=0.00e+00 | PASS |

Leak guard is doubly conservative: the chimera daily-funding column is itself the *prior* day's 8h sum (a known
1-day label offset caught in the 2026-06 dispersion run), and lag=1 ranks on funding strictly before the held
bar. We rank on STALE funding, never future.

---

## The decisive result

### Per-year grade (selected variant st0.25_lb30, the BEST-on-SEL; UNSEEN read-once)

| period | tilt net% | EW net% | BH net% | **tilt-alpha%** |
|---|---|---|---|---|
| 2020 | 3.34 | 3.28 | 3.09 | **+0.07** |
| 2021 | 1.02 | 1.30 | 1.32 | **-0.27** |
| 2022 | -0.05 | 0.48 | 0.55 | **-0.53** |
| 2023 | 7.54 | 7.62 | 7.58 | **-0.08** |
| 2024 | 0.76 | 1.01 | 0.98 | **-0.25** |
| 2025 (OOS) | 1.41 | 1.85 | 1.84 | **-0.43** |
| UNSEEN | -1.35 | -1.09 | -1.08 | **-0.27** |

Tilt-alpha is negative in **6 of 7** periods (only 2020 marginally +0.07%). The absolute net tracks buy-hold
almost exactly -- the book is **pure beta**, and the tilt subtracts from it.

### Full sweep -- every variant, held-out tilt-alpha + block-bootstrap p05

All 12 variants. **SEL, OOS, UNSEEN, and held-out tilt-alpha are negative in EVERY cell.** Block-bootstrap p05
is below 0 for all 12 (held-p95 is *also* below 0 for all -- the entire bootstrap distribution sits left of zero).

```
       variant   SEL-a%   OOS-a%  UNSEEN-a%   held-a%  held-p05  held-p95
   st0.25_lb14    -1.20    -0.35      -0.22     -0.56     -0.83     -0.33
   st0.25_lb30    -1.10    -0.39      -0.27     -0.65     -0.96     -0.39   <- best-on-SEL
   st0.90_lb7     -6.97    -1.94      -0.80     -2.73     -3.74     -1.78   <- worst
```

- **Variants with held-out tilt-alpha p05 > 0: 0 of 12** (under H0, ~0.6 expected by chance). Not even one false positive.
- **PBO** (prob. of backtest overfit on the tilt-alpha grid) = 0.0004 -- but this is the wrong-direction reassurance:
  PBO is low because *no* config is good in-sample, so there is nothing to overfit toward.
- Tilt-alpha gets **monotonically worse with tilt strength** (st0.25 -> st0.90 worsens the loss at every lookback).
  A signal that is more harmful the harder you lean on it is a wrong-signed signal, not noise.

---

## Decomposition: is the kill cost or signal? (the load-bearing adversarial check)

Re-ran the st0.50_lb14 variant gross (zero cost) vs net (maker 6bps):

| | tilt-alpha SEL | OOS | UNSEEN | FULL |
|---|---|---|---|---|
| **GROSS (zero cost)** | -0.67% | -0.36% | -0.28% | **-1.31%** |
| net (maker 6bps) | -2.40% | -0.70% | -0.43% | -3.50% |

The **gross price-tilt itself is negative** everywhere. Cost adds ~2.2pp of drag (the tilt runs ~11x the EW
turnover: 0.035 vs 0.0032 daily), making net worse -- but cost is **not** the cause. **Even a free-rebalancing
version loses.** The signal has the wrong sign for spot wealth before a single basis point of friction.

## The mechanism (why the long leg alone fails)

Cross-sectional rank-IC of `corr(trailing funding level, next-day return)` = **+0.0060** (median +0.0049, t=+1.47,
N=2332 days) -- weakly **positive**. HIGH-funding names tend to have slightly HIGHER forward returns: funding is a
mild **momentum proxy** in crypto spot (you pay to hold what is pumping). The long-only tilt overweights LOW-funding
names, i.e. it tilts toward laggards -- a weak **anti-momentum** tilt, which loses in a momentum-driven market.

This is exactly why the market-neutral dispersion book worked and this one does not: the dispersion edge lived in
**SHORTING the expensive high-funding names** (and earning their funding cash-flow while they mean-reverted). Amputate
the short leg and you are left with a long leg that, on price, is the wrong cross-sectional bet. The funding cash-flow
(the carry) is a perp-settlement flow that a **spot** long-only book never collects -- so the tilt keeps the harmful
price selection and loses the only beneficial component.

---

## Verdict

**ARTIFACT.** A long-only funding-tilt does **not** translate as a wealth edge. It collapses to beta like everything
directional, and the tilt itself is value-destructive (wrong-signed for spot price, gross of all cost). H0 is not
rejected; it is confirmed with margin -- 0 of 12 swept variants clear p05>0 held-out, the entire bootstrap distribution
of the best variant sits below zero (held p05 -0.96%, p95 -0.39%), and the gross (cost-free) signal already loses.

- **Decisive statistic**: held-out tilt-alpha for the best-on-SEL variant = **-0.65%** (block-bootstrap [p05 -0.96%, p95 -0.39%], i.e. the entire 90% interval is negative). 0 of 12 variants positive-p05.
- **Cheapest falsifier (already run, and it fails)**: the cross-sectional rank-IC `corr(funding, fwd return)` is **+0.0060**, not negative. For a low-funding overweight to add wealth this IC would need to be materially *negative*. It is the wrong sign. Any future re-test should compute this single number first -- if it is non-negative, the long-only low-funding tilt cannot pay, and no parameter sweep will rescue it.

The "internal-data ceiling" holds for this avenue. The funding signal's wealth content lives in the short leg, which
is off-constraint; the long leg in spot is anti-momentum and loses. This door is closed.
