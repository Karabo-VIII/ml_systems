# Perfect-Foresight LONG-ONLY Realizable-Ceiling Map (oracle) — 2026-06-06

**Built/verified this run (RWYB).** The per-(cadence × asset) ceiling the MA-DNA must approach.
Method = **exact** perfect-foresight high-capture DP (entry = `open[k]`, exit = `high[j]`, single
non-overlapping position, `1h ≤ hold < 7d` expressed per cadence in bar units via real timestamps,
net taker round-trip = **0.0024**, objective = max compound). DP correctness: `--selftest` **PASS**
(single-move, flat-no-trade, and pullback-two-move cases all exact). Numbers reproduce the prior
`oracle_ceiling_map.json` bit-for-bit (cross-check MATCH=True).

Builder: `runs/research/oracle_ceiling_map_v2.py` · data: `oracle_ceiling_map_v2.json` · assets =
SOL, BTC, ETH, BNB, AVAX · span ≈ 5.7 yr (2020→2026-04).

## Construction verdict (which oracle)
- **`capturable_4h_catalog` REFUTED as "the 4h oracle."** It is a **binary win-LABEL** catalog
  (`win_3pct_12bar`, `win_5pct_18bar`; base rates 0.41 / 0.33 over 87 assets; values ∈ {0,1}) — a
  classifier *target*, not a max-per-move-capture oracle. Not reused.
- **setup_harness fixed TP/SL would UNDER-state the ceiling** (a fixed take-profit caps capture below
  the true intrabar high). The **DP high-capture (exit at the exact future high)** is the *tight*
  perfect-foresight ceiling and matches the task spec verbatim ("max per-move capture"). Used.
- **dollar bars excluded** — perfect foresight on ~4M ultra-fine bars is granularity-degenerate
  (close-to-close oracle → `Infinity` for SOL/AVAX). Not comparable; reported only as a caveat.

## THE MAP (full-span | UNSEEN held-out slice)

| cad | asset | n_bars | **moves** | **mean%** | **med%** | sum% | medHold_h | UNSEEN moves | UNSEEN mean% | UNSEEN med% | UNSEEN sum% |
|-----|-------|-------:|----------:|----------:|---------:|-----:|----------:|-------------:|-------------:|------------:|------------:|
| 1d  | SOL  |   2083 |   520 | **9.33** | 5.71 | 4852 | 24.70 |  34 | 4.96 | 3.65 |  169 |
| 1d  | BTC  |   2334 |   590 | 4.40 | 2.99 | 2594 | 47.95 |  33 | 3.65 | 3.42 |  120 |
| 1d  | ETH  |   2326 |   602 | 6.02 | 4.02 | 3624 | 47.91 |  33 | 4.79 | 4.16 |  158 |
| 1d  | BNB  |   2325 |   595 | 5.82 | 3.51 | 3463 | 47.97 |  30 | 3.39 | 3.19 |  102 |
| 1d  | AVAX |   2075 |   528 | **9.09** | 5.71 | 4798 | 47.61 |  33 | 5.20 | 4.20 |  171 |
| 4h  | SOL  |  12482 |  3049 | 3.50 | 2.10 | 10683 | 7.82 | 185 | 2.00 | 1.26 |  370 |
| 4h  | BTC  |  13996 |  2984 | 1.76 | 1.01 | 5267 | 7.99 | 177 | 1.40 | 0.84 |  249 |
| 4h  | ETH  |  13951 |  3105 | 2.36 | 1.41 | 7328 | 7.99 | 182 | 1.79 | 0.97 |  325 |
| 4h  | BNB  |  13947 |  3175 | 2.26 | 1.33 | 7182 | 7.98 | 186 | 1.29 | 0.83 |  239 |
| 4h  | AVAX |  12444 |  3102 | 3.40 | 2.00 | 10542 | 5.37 | 206 | 1.93 | 1.20 |  398 |
| 1h  | SOL  |  48926 |  9520 | 1.99 | 1.25 | 18931 | 2.01 | 595 | 1.07 | 0.65 |  638 |
| 1h  | BTC  |  55956 |  8613 | 1.05 | 0.63 | 9010 | 2.98 | 513 | 0.84 | 0.54 |  431 |
| 1h  | ETH  |  55771 |  9534 | 1.32 | 0.84 | 12622 | 2.07 | 568 | 1.02 | 0.65 |  579 |
| 1h  | BNB  |  55750 |  9674 | 1.31 | 0.79 | 12668 | 2.10 | 551 | 0.79 | 0.56 |  434 |
| 1h  | AVAX |  49096 |  9839 | 1.96 | 1.23 | 19285 | 2.01 | 689 | 1.11 | 0.75 |  767 |
| 30m | SOL  |  96379 | 13858 | 1.71 | 1.10 | 23762 | 1.52 | 870 | 0.95 | 0.59 |  825 |
| 30m | BTC  | 111900 | 12458 | 0.91 | 0.58 | 11352 | 1.99 | 753 | 0.72 | 0.48 |  542 |
| 30m | ETH  | 111443 | 13824 | 1.15 | 0.75 | 15923 | 1.96 | 801 | 0.91 | 0.56 |  732 |
| 30m | BNB  | 111160 | 14131 | 1.14 | 0.73 | 16083 | 1.65 | 807 | 0.69 | 0.49 |  559 |
| 30m | AVAX |  96540 | 14232 | 1.72 | 1.11 | 24506 | 1.51 | 973 | 1.01 | 0.72 |  981 |
| 15m | SOL  | 188060 | 17608 | 1.60 | 1.07 | 28203 | 1.46 |1140 | 0.89 | 0.59 | 1011 |
| 15m | BTC  | 223098 | 16061 | 0.85 | 0.57 | 13688 | 1.51 | 950 | 0.69 | 0.48 |  658 |
| 15m | ETH  | 221277 | 17924 | 1.07 | 0.72 | 19161 | 1.50 |1067 | 0.84 | 0.54 |  899 |
| 15m | BNB  | 217213 | 17958 | 1.07 | 0.72 | 19248 | 1.49 | 995 | 0.67 | 0.50 |  667 |
| 15m | AVAX | 185865 | 17967 | 1.63 | 1.09 | 29284 | 1.31 |1191 | 0.98 | 0.74 | 1169 |

`mean%`/`med%`/`sum%` = net-of-cost per-move %. `sum%` = additive (fixed-stake) capture over the
window — the honest aggregate. medHold_h = median hold in hours.

## "Total capturable compound" — emitted but DEGENERATE (read this)
The requested *compound* metric explodes under perfect-foresight reinvestment of hundreds–thousands
of perfectly-timed moves and is **not a target anything can approach**:
`full-span SOL 1d = 1.6e21%`, `4h = 3.2e46%`, `15m = 1.9e122%`. Even the held-out **UNSEEN** compound
runs `1d BTC 2.2e2%` → `15m SOL 2.2e6%`. Use it only as an order-of-magnitude "edge exists" sniff.
**The actionable ceiling is the per-move triplet `{n_moves, mean/median net%, sum net%}`.**

## What the map says (Oracle reading)
1. **The per-move quality ceiling DEGRADES monotonically as the cadence gets finer** — and this is a
   pure **cost-to-capture** effect of the fixed 0.24% taker. Median net per move: 1d ≈ 3.5–5.7% →
   4h ≈ 1.0–2.1% → 1h ≈ 0.6–1.25% → 15m ≈ 0.5–1.1%. On **15m BTC the median net move is 0.57%**, so the
   0.24% round-trip is ~30% of the gross — the cadence is *structurally cost-bled* even for a clairvoyant.
   On **1d SOL (9.33% mean)** cost is a rounding error. → **A long-only MA-DNA has its highest realizable
   headroom at the COARSEST cadences (1d ≥ 4h), and the thinnest at 15m/30m.**
2. **Finer cadence buys move-COUNT, not quality:** moves scale 1d≈550 → 15m≈17,500 (~32×), but mean net
   per move shrinks ~6×. The oracle's *opportunity surface* is large at every cadence; the binding
   constraint a real strategy hits is per-move quality net of cost, which is set by cadence.
3. **Cross-asset:** SOL & AVAX carry the richest ceiling at every cadence (mean ~1.6–9.3%); BTC the
   leanest (~0.85–4.4%) — higher-vol alts have more capturable per-move amplitude.
4. **The held-out target to beat:** on UNSEEN, the per-move ceiling is ~**2.0%/move on 4h SOL** (185
   moves) and ~**5.0%/move on 1d SOL** (34 moves). The MA-DNA's score is its **capture rate** =
   (MA realized mean net per move ÷ oracle UNSEEN mean) × (MA move-count ÷ oracle move-count). Both
   factors < 1; a credible MA target is single-digit-to-~30% of these per-move ceilings.

## Caveats
- Ceiling is a *strict upper bound*: clairvoyant on the entry bar AND exits at the exact intrabar high
  — unreachable by construction. The MA-DNA captures a fraction; that fraction is the whole game.
- Survivorship: 5 still-listed majors. UNSEEN = post-2025-12-31 only (33–34 moves at 1d → wide CIs on 1d).
- 30m is partial in the wider universe (77/104 assets) but present for all 5 here.
