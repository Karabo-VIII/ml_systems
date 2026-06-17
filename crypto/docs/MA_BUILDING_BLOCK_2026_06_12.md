# MA building block — full-funnel analysis (oldest month, 2026-06-12)

**The exercise (user-directed):** decompose the MA strategy space into its DISTINCT configs
(306 2MA + 1,466 3MA, near-dup-eliminated), funnel **all of them** through **each u10 asset** ×
**4 cadences** (4h/1h/30m/15m) × **3 exits** (signal-flip / trail-5% / trail-10%) over the **oldest
month we have** (2020-01-07 → 02-07), then explain what worked and what didn't. **137,766 cells.**
This is a *starting-point* building block — one rally regime, descriptive — not a validated edge.

Tools: `ma_per_instrument.py` (funnel), `ma_mechanics.py` (economics), `ma_analysis_plots.py`,
`ma_move_grid.py`, `ma_equity_grid.py` (charts). All reproducible; cells in
`runs/strat/ma_per_instrument_u10_*.json` + `runs/strat/ma_mechanics*.json`.

## 1. The headline mechanism: gross is flat, cost is not
Per-cadence means over all 137,766 cells (taker 0.24% round-trip):

| cadence | **gross** | **net** | **cost drag** | trades | trades/day | whipsaw | win% | % of configs positive |
|---|---|---|---|---|---|---|---|---|
| **4h** | +12.9% | **+12.1%** | 0.8% | 3 | 0.1 | 10% | 72% | **91%** |
| **1h** | +13.1% | +9.9% | 3.2% | 13 | 0.4 | 12% | 51% | 84% |
| **30m** | +11.3% | +5.0% | 6.3% | 26 | 0.8 | 14% | 38% | 64% |
| **15m** | +8.2% | **−2.3%** | **10.5%** | 47 | 1.5 | 15% | 30% | **42%** |

The **gross** edge is ~8–13% per month at *every* cadence — the trend is equally catchable. What
changes is the **cost of harvesting it**. That single fact explains the whole block.

## 2. What WORKED
- **4h, slow MAs, ~1-week holds, few trades.** 91% of all 4h configs are net-positive; mean +12.1%;
  only ~3 trades; 0.8% cost drag. This is the regime where the gross edge survives contact with fees.
- **The best config at every timeframe is 7/7-positive across assets** and converges to a
  ~1-week WALL-CLOCK hold (the MA length scales to the bar duration):
  - 4h `ema_8_28` +23.9% (160h) · 1h `ema_22_108` +24.7% (141h) · 30m `ema_37_203` +24.5% (132h) ·
    15m `ema_186_208_233` +21.7% (137h). All ≈ 5.5–6.7 days.
- **Per-asset peaks** (full funnel found them): LTC `ema_6_248`/15m **+63%**, ADA `ema_12_14_24`/4h
  +36%, LINK `ema_26_53`/30m +31%, XRP `ema_5_22`/4h +27%.
- **Winner anatomy:** 6 trades, ~87-bar (~1wk) hold, 6% whipsaw, 67% win, **1.7% cost drag**. Enters
  the Jan trend once or twice and *holds*.

## 3. What did NOT work
- **Fine cadences at taker.** 15m nets **−2.3%** despite +8.2% gross — the 10.5% cost drag eats it.
  Only 42% of 15m configs are positive (a coin-flip).
- **Overtrading.** trades/day go 0.1 → 1.5 (15×); each round-trip pays ~0.24%. The net cloud descends
  straight down as trade-count rises (chart A).
- **Whipsaw.** 10–15% of trades are held ≤2 bars (enter on a cross, exit on the immediate re-cross +
  two commissions). Worse at fine cadences.
- **Fast MAs everywhere.** The worst cells are tiny MAs on 15m: BTC `ema_2_3`/15m **−65%**, 377 trades,
  1-bar holds, 15% win.
- **Loser anatomy:** 59 trades, 16-bar hold, 23% whipsaw, 27% win, **13.0% cost drag** — gross can be
  positive; fees obliterate it.
- **The naive pooled BOOK at 15m** (all configs equal-weighted) is −42% with 21%/bar turnover — pooling
  *without selecting out the fast configs* is itself a way to lose. Selection is the whole game at fine
  cadence: best 15m config +21% vs naive book −42%.

## 4. The exit mechanism is SECOND-ORDER
Signal-flip ≈ trail-10% ≫ tight trail-5% only matters at fine cadence. At 4h the trail never triggers
(a real trend doesn't retrace 10%), so flip ≈ trail. A *tight* 5% trail stops you out into noise and
tends to **underperform** the flip. The exit shapes drawdown, not the core return. The first-order
levers are **cadence** and **MA length** (they set trades/day → cost).

## 5. The lever (maker vs taker) — IT FLIPS THE VERDICT
All of the above is at **taker** (0.24% rt). Re-running the full mechanics at **maker** (0.06% rt):

| cadence | taker net | **maker net** | taker drag | maker drag |
|---|---|---|---|---|
| 4h | +12.1% | +12.7% | 0.8% | 0.2% |
| 1h | +9.9% | +12.2% | 3.2% | 0.8% |
| 30m | +5.0% | **+9.7%** | 6.3% | 1.6% |
| 15m | **−2.3%** | **+5.4%** | 10.5% | 2.8% |

**At maker, all four cadences are net-positive** — 15m flips −2.3% → **+5.4%**, 30m nearly doubles, and
the cadence spread compresses (12.7/12.2/9.7/5.4 vs taker 12.1/9.9/5.0/−2.3). Cost drag collapses ~4×.
So **"fine cadences don't work" = "don't work AT TAKER."** The cost model is the binding verdict, not
the signal.

**Honest caveat (the open question):** this assumes you actually GET the maker fills. Per the project's
MakerCostModel invariant, real maker **p_fill = 0.21–0.40**, not 1.0 — so the realized maker advantage
is *partial* and probabilistic. The clean conclusion: the MA trend edge is **cost-bound**, and maker
*can* unlock fine cadences IF maker fills are achievable — which is itself the unproven p_fill question
(engine gap-audit / Coinglass thread). Don't bank the maker numbers; they're the ceiling, not the floor.

## 6. Methodology note (the full funnel vs the sample)
The earlier 8-config-per-family sample faithfully captured the *shape* (cadence decay, cost bleed,
hold discriminator — the per-instrument means barely moved). But the full funnel was needed to find the
*true best* config per asset/timeframe — and they're meaningfully better (+2–6pp). Decompose → funnel
all → explain → then move to the next block.
