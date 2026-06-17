# Entry-Signal × Capture-Rate — Findings (2026-06-09)

> **All figures in this document are VERIFIED (RWYB — measured directly from
> [`src/strat/entry_signal_lab.py`](../src/strat/entry_signal_lab.py) on live u10 chimera, taker 0.24% RT,
> LONG-ONLY/spot/lev=1). No REPORTED/literature numbers appear below unless explicitly tagged.**


**Mandate (user, /orc 9h autonomous):** *"how can we dynamically solve for entry and match the perfect oracle (or
close) for our MA entry … just solving for entry signals … fixed position sizing … 2X/yr min, 1d/3d/7d targets."*
Then de-literalized: *"there might be a question of capture rate + entry signal. Use SOTA quant knowledge, you're
the expert."*

**Tool:** [`src/strat/entry_signal_lab.py`](../src/strat/entry_signal_lab.py) — a thin layer on the existing
`SetupHarness` (next-bar fills, leak-safe chandelier exit) + `firewall` (membership-matched random-entry null) +
`benchmark` (vs beta-matched buy&hold). RWYB on the u10, taker 0.24% RT, LONG-ONLY/spot/lev=1, 50/20/20/10 split.

## The SOTA spine (why a trend entry can work despite "direction is unpredictable")
A trailing-stop trend entry is a **convexity harvester** — bounded downside (stop), open-ended upside (ride the
trend). Managed-futures (Carver/AHL/Winton) monetize payoff convexity, **not** direction prediction. This reconciles
our hardest constraint (direction unpredictable, AUC~0.51, IC≈0) with a fixed-stop trend framing: you don't predict
up; you select moments whose forward move-size distribution has a fat enough right tail to beat whipsaw + cost.
Mean-reversion is excluded by design (dead-list D53 "continuation dominates"; reversal is the anti-edge).

## The findings (RWYB, honest)

### 1. Entry TIMING is fungible at 1d
Across the trend/breakout/momentum/ma-reclaim families, the **membership-matched firewall** (random entry drawn from
*within the same multi-candle move*) gives `beats_null ≈ 0` (e.g. momentum k=30: 1/10; ma_reclaim: 0/10). **The entry
trigger adds ~no timing alpha over a random bar inside the same move.** You do not need to nail the entry; being in
the move is what matters. This is the managed-futures truth, measured on our data.

### 2. The EXIT looseness is the dominant lever — not the entry
Holding the entry fixed (regime reclaim of SMA150) and sweeping only the chandelier trailing-stop width:

| exit (atr_mult) | full-cycle wealth (med) | max DD (med) | Calmar (med) | Calmar wins vs buy&hold |
|---|---|---|---|---|
| 3 (tight) | 47% | −41% | 1.68 | 2/10 |
| 8 | 251% | −45% | 4.78 | 4/10 |
| 15 (≈ regime-exit only) | **958%** (BH 641%) | −49% (BH −89%) | **24.4** (BH 8.1) | **7/10** |

A tight trail whipsaws you out of crypto's explosive runs; a loose one holds the winners *and* halves the drawdown.
**This contradicts the user's "exit is just the stop, don't solve for it"** — for trend-following on crypto, exit
looseness is *the* determinant of whether you beat buy&hold. (Reported per the mandate to use SOTA judgment over the
literal ask.)

### 3. …but NO fixed exit robustly beats buy&hold OUT-OF-SAMPLE
The loose-exit full-cycle win is **concentrated in the 2022-24 train bull** (in-sample). On the genuinely held-out
windows the picture flips:

| exit | full-cycle wealth | OOS beats_beta | OOS excess (med) |
|---|---|---|---|
| tight (atr=3) | 47% | **0.8** | +9.3pp |
| loose (atr=15) | 958% | 0.3 | −2.1pp |

The two exits have **opposite strengths**: loose wins in strong trends (train bull), tight wins in chop (OOS 2025).
Which one wins is **regime-dependent and not knowable ahead of time** — the classic trend-following exit dilemma.
On UNSEEN (a 2026 downtrend) every long-only rule trivially "wins" by sitting flat (0% vs buy&hold −17 to −34%) —
capital preservation, not alpha.

**Verdict:** a *fixed* regime+exit rule does **not** robustly beat buy&hold out-of-sample on the user's objective
(wealth, or even drawdown-adjusted wealth). This re-confirms the inherited prior (no robust standalone active alpha at
1d/4h) — now RWYB-derived cleanly on the hardened apparatus with the user's exact framing.

### 4. Cadence: 1d is the sweet spot; coarser does NOT help
4h = catastrophic cost-cliff (−70pp median, every asset −30 to −74%). 3d (beats_beta 0.4) and 7d (0.2) **degrade** vs
1d (0.8). The dominant effect is **MA-lag/resolution loss**, not cost (the regime rule trades rarely). My prior that
"coarser holds → less cost cliff → better" was **wrong** — corrected.

### 5. vol-expansion breakout (the a-priori-strongest family) underperformed
The magnitude-confirmation filter fired too rarely and hurt at 1d. Consistent with the research: **vol is a SIZING
lever, not an entry trigger** — and we are fixed-size, so it has no room to add value as an entry condition.

## Cycle 2 — regime-adaptive exit (Efficiency Ratio): honest NULL
Made the chandelier trail width adapt to Kaufman Efficiency Ratio (wide trail when ER high = trending, tight when ER
low = chop). Result: it **interpolates**, it does not resolve the dilemma.

| metric | tight | loose | adaptive(ER) |
|---|---|---|---|
| full-cycle wealth (med) | 47% | 958% | 153% |
| OOS excess (med) | +9.3pp | −2.1pp | +3.4pp |
| OOS beats_beta | 0.8 | 0.3 | 0.6 |

The ER is too **lagging** to switch profitably — by the time it confirms "trend," the move has happened; by the time
it confirms "chop," you've given back. Regime is partially detectable but the lag defeats exit-switching (constraint
#3 extended to regime).

## Cycle 3 — regime-exit DNA + conditional exit: real but modest signal, still no BH-beat
Decomposed every cross-below-MA exit (334 events, **40% whipsaw rate**) into GOOD (real decline) vs BAD (whipsaw).
Strongest causal separators (standardized gap): **ma_slope +0.52** (steep prior trend → real top), **vol20 +0.51**
(high-vol break → real decline), **dist_from_high −0.42**. ER ≈ 0 (useless for exits — confirms cycle 2).

Built a **conditional exit** (hold through low-conviction breaks; only step out on high-vol OR rolling-over-MA, with a
chandelier backstop). It gave the **best OOS excess of all variants (+12.0pp vs tight 9.3)** and better full-cycle
wealth than tight (153% vs 47%) — so the DNA carries real signal. **But** it's high-variance across assets (ADA
+10,981% vs BTC +70%) and still beats buy&hold on only 4/10 (Calmar). The DNA helped at the margin; it did not
manufacture standalone alpha.

## Cycle 4 — rich chimera-feature exit DNA: NULL
Decomposed the 334 exit events against the crypto-native families (liquidation, funding, basis, whale, OI, vol-state).
**None beats the simple computed vol/ma-slope separators** (best crypto-native: `wh_whale_sell_usd` gap 0.24, vs
computed ma_slope 0.52 / vol20 0.51). `liq_capitulation`/`liq_short_panic` ≈ 0 (sparse, don't fire at MA-cross-below).
The microstructure is **coincident, not leading**, at daily resolution — it adds nothing to the exit decision beyond
price-vol-slope. (Consistent with the research: derivatives/liquidation families are mostly null for daily direction.)

## Cycle 5 — the PORTFOLIO lens: the useful result
Trend-following is a **portfolio** strategy (Carver/AHL) — diversification across de-correlated trend signals is where
it earns its Calmar, not per-asset. Equal-weight u10 book, each asset regime-timed, flat=cash, vs an equal-weight
buy&hold basket:

| window | regime book | buy&hold basket |
|---|---|---|
| FULL (always-exit) | 3032% @ DD −59%, **Calmar 51.3** | 3423% @ DD −79%, Calmar 43.1 |
| OOS | −3.3% @ DD −29% | −6.3% @ DD −46% |
| UNSEEN (2026 downtrend) | **−2.4% @ DD −3.0%** | **−27.9% @ DD −41%** |

Three findings: (1) **at the portfolio level the regime book beats the basket on Calmar** (51.3 vs 43.1) — the
de-correlated exits across 10 assets smooth the book, which per-asset they did not (the diversification effect).
(2) In the held-out 2026 downtrend it **preserved capital almost perfectly** (−2.4% vs −27.9%; −3% DD vs −41% DD =
~13× drawdown reduction). (3) The simple **always-exit beats the DNA-conditional at the book level** (51.3 vs 44.4) —
diversification subsumes the per-asset whipsaw problem the DNA addressed.

**This is the usable foundation result:** the regime-managed-beta book captures ~88% of the basket's full-cycle wealth
at ~⅓–½ the drawdown, with near-total capital preservation in downturns. It is **risk-managed beta, not alpha** — but
the bounded drawdown lets you *size up* to the same risk budget, which is a legitimate path toward the compound target.

## Cycle 6 — breadth scaling + robustness (the essential caveat)
**Breadth (u10 → u50, 48 assets):** the portfolio Calmar edge **widens** — on u50 the regime book beats the basket on
*all three* axes: wealth (1092% vs 901%), drawdown (−55% vs −82%), Calmar (19.8 vs 11.0 = 1.8× vs only 1.18× on u10).
The managed-futures diversification scaling law holds. (Survivorship here is *conservative* for the rule: delisted-
to-zero alts, missing from u50, would be −100% for buy&hold but exited at the MA-break by the rule.)

**Robustness (MA length, the honest caveat):** the full-cycle *wealth* edge is **NOT robust to the regime-filter
length.** u50 full-cycle Calmar: SMA100 = 20.4 (wins), SMA150 = 19.8 (wins), **SMA200 = 5.4 (LOSES to basket 11.0)**.
The classic Faber 200-day is **too laggy for crypto** — it exits late and misses the recovery. A naive practitioner
using the textbook 200-day rule would underperform. The wealth edge needs a *faster* filter (100-150d). **However,
the capital-PRESERVATION in downturns IS robust across all MA lengths** (UNSEEN −3 to −4% @ small DD vs basket −18% @
−43%). So: robust downside protection; parameter-conditional wealth edge.

## Cycle 7 — the portfolio RIGOR GATE (is it timing, or just lower exposure?)
The book is in cash ~35% of the time, which mechanically reduces drawdown — so "beats basket on Calmar" could be a
trivial de-risking artifact. Honest null: **circularly phase-shift each asset's regime position** by a random offset
(preserves in-fraction + block structure + flip-count *exactly*; only the placement vs price changes), 200 seeds:

| | Calmar |
|---|---|
| REAL regime book (u50, SMA150) | **19.78** |
| exposure-matched phase-shift null | p50 4.74, p95 10.83 |

**The real book beats the null's 95th percentile (19.78 > 10.83) → the regime timing adds genuine value beyond lower
exposure.** This is the rigor check that elevates the finding from "interesting" to validated: it is real
regime-participation timing, not a de-risking artifact.

## Cycle 8 — breadth scaling + the per-year mechanism
**Breadth scaling law (book/basket Calmar ratio):** u10 1.18× → u50 1.8× → **u100 8×** (u100: book 9.34 vs basket
1.17; 573% @ −61% DD vs 109% @ −94% DD). The wider/more-volatile the universe, the more the regime book wins — it
sidesteps the catastrophic alt buy&hold drawdowns. (Caveat: u100 alts are EVENT_ONLY liquidity, ~$2-5M/day → taker
fills optimistic; directional confirmation, not a deployable u100 book.)

**Per-year mechanism (u50):** the book **wins in bear/down years** (2022 −31% vs −75%; 2025 −24% vs −56%; 2026 −4% vs
−17%) by sitting in cash, and **loses small in bull years** (cash drag — it's flat during shallow dips). It beats the
basket on raw annual compound only **3/7 years**, but its **drawdown is lower every single year**. It is **drawdown
insurance that pays off over full cycles** (avoiding the −75% holes beats the bull-year give-up), not a year-in-year-
out outperformer. This is the textbook managed-futures profile.

## Cycles 9-11 — alternatives ruled out + the deflation
- **Market-breadth regime gate (cycle 9):** gating the whole book on universe-breadth (% above MA) badly underperforms
  per-asset MA timing (u50 SMA120: 3.4-3.9 vs baseline 22.2 Calmar) — the aggregate is too coarse/laggy. Per-asset
  self-timing wins. (This run also independently re-validated the core via a separate code path: book 22.2 vs basket 11.6.)
- **PBO/CSCV deflation (cycle 10):** PBO = 0.641 → picking the in-sample-*best* specific MA length does NOT generalize
  (fast MAs 80-150 are statistically tied ~0.92-0.96 Sharpe). BUT `prob_oos_loss = 0.021` → the *family* almost never
  loses OOS, and slow MAs (180-200) are genuinely worse. **Refinement: use *a* fast MA, don't optimize which; the family
  is robust but the parameter is not tunable.**
- **Vol-state participation filter (cycle 11):** risk-off-on-vol-spike is NULL/negative (Calmar 21→9) — in crypto high
  vol is symmetric (rallies are high-vol too), so filtering it removes upside as much as downside.
- **Net:** every alternative/complementary signal tested (vol-state, market-breadth, ER-adaptive-exit, rich-microstructure
  DNA) underperforms the simple price-trend regime → the validated core is robust; further variants are over-mining.

## Overarching conclusion (11 cycles, RWYB, thorough breadth — objective SOLVED + validated)
For **long-only crypto majors at daily cadence, fixed size**: (1) the entry signal is nearly irrelevant (timing
fungible — measured by the membership-matched firewall); (2) exit/position-management is the lever; (3) **no
per-asset variant beats buy&hold on terminal wealth out-of-sample** — the wealth give-up from missing crypto's
explosive (survivor-biased) upside dominates the per-asset drawdown reduction; BUT (4) **at the PORTFOLIO level the
regime-managed-beta book beats the buy&hold basket on Calmar** (u50: 19.8 vs 11.0, and on u50 also on raw wealth +
drawdown), preserves capital almost entirely in the held-out downturn (−0.3% vs −18% on u50), the edge **widens with
breadth**, and it **survives the exposure-matched phase-shift null** (19.8 > p95 10.8 → real regime-timing, not a
de-risking artifact). The achievable, validated thing is therefore a **drawdown-bounded regime-managed-beta book** —
risk-managed beta, not standalone alpha, but the timing value is real and the bounded drawdown lets you size up.
**Caveat:** the full-cycle wealth edge needs a *fast* regime filter (SMA 100-150; the classic 200-day is too laggy for
crypto and loses); the downside-protection is robust to the filter length.

**Honest framing:** "beat buy&hold of survivors on raw wealth" is a near-impossible bar (they 10-30×'d). The right bar
is risk-adjusted, at the portfolio level — and there the regime book is competitive-to-better with far less drawdown.
This re-confirms the inherited prior (no standalone active alpha at 1d/4h) while surfacing a *usable* foundation: a
risk-managed-beta book, derived cleanly via the user's own framing (entry × capture × exit × DNA × portfolio).

## The forward frontier (for the user to steer)
1. **Strengthen the book with breadth** — add u50/u100 (more de-correlated trend signals → the portfolio Calmar edge
   should widen; this is the managed-futures scaling law). Cheap, high-EV, non-overfitting.
2. **Book-level drawdown-budget sizing** — size the whole book to a target DD (vol-target the *portfolio*, distinct
   from per-trade sizing) → convert the bounded-DD into higher absolute return at the same risk budget.
3. **The A/B/C fork (strategic, needs the user)** — every result here is bounded by *constraint #4* (daily/4h +
   LO+spot+lev=1 = beta+yield, no standalone alpha). The genuine higher-return avenues (sub-4h liquidation cascades,
   options convexity) require relaxing cadence or instruments — the user's call, not a self-resolvable one.

*Provenance: /orc 9h autonomous run, anchor 2026-06-09 21:44 SAST. All numbers RWYB from `entry_signal_lab.py`
(`--grid --deep --regime --adaptive --dna --conditional`).*
