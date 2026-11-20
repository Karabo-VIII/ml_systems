# Market Research — the case for (and against) crypto opportunity, BEFORE any strategy (2026-06-05)

> **Mandate:** characterize the OPPORTUNITY SURFACE, make the case, **do not mine an edge**. Harvestability is
> held OPEN. The premise is re-derived FRESH from our data; the archived "20 movers/day" is NOT inherited.
> Every number below is reproducible from a tracked script on the real 104-asset chimera (2020-01 → 2026-05-28,
> ~6.4y). Scripts: `scripts/research/{move_distribution,beta_confound,regime_variation}.py`.

## 0. The verdict (the case, in five sentences)
1. **Raw opportunity is abundant and persistent:** ~**15 assets/day** move ≥5% (|daily return|), **97.8% of days**
   have ≥1 such mover, present *every year* 2020-2026 and in *both* bull and bear regimes. **Long-only correction
   (load-bearing):** only ~half are UP-moves — the long-only-harvestable count is **~7.6 up-movers/day ≥5%**
   (~5/day after stripping beta). The headline |move| count is ~2× the number a LO+spot bot could actually act on.
2. **It is mostly idiosyncratic, not just BTC-beta:** BTC explains only ~30% of a typical alt's daily variance;
   after stripping beta, ~**10 idiosyncratic movers/day ≥5%** remain (65% of the raw count).
3. **But raw moves ≠ an edge:** entering at the close at *random*, the next day's high beats cost only **~47% of
   the time** (≈ coin-flip; median next-day up-excursion ≈ 0%). The moves are real; capturing them **requires a
   signal**, which this research deliberately does not build.
4. **The literature agrees it is real but fragile:** out-of-sample predictability exists, concentrated in
   **small-caps** (momentum / reversal / illiquidity) — yet the headline momentum edge is **largely a
   survivorship artifact** and capacity-limited.
5. **Net:** there is a genuine, durable pool of raw material here, but the naive edges that "explain" it are mostly
   illusions (survivorship, beta, cost, capacity). **The case to proceed is real; the bar is to find an edge that
   survives exactly those biases** — which is what our apparatus (taker cost, random-entry null, two-sided
   positive-control, survivorship-aware) is built to test.

## 1. The opportunity surface (r1/r2 — `move_distribution.py --cadence 1d --horizon 1 --cost 0.0024`)
For each bar, entering at `close[t]`, MFE = best next-bar up-excursion `(high-close)/close`; net = MFE − taker RT (0.0024).

| Metric (universe-avg over 104 assets, 2325 days) | Value |
|---|---|
| asset-days with a ≥2% up-move available (net of cost) | **29.4%** |
| asset-days with a ≥5% up-move available (net) | **13.7%** |
| asset-days with a ≥10% up-move available (net) | **4.7%** |
| asset-days with *any* net-positive up-move (random entry) | **46.8%** ← ≈ coin-flip |
| median next-day up-excursion | **~0.0%** |

**Movers/day premise, re-derived fresh** (|daily return| across the universe):

| Threshold | avg movers/day | median | % of days with ≥1 mover |
|---|---|---|---|
| ≥2% | 35.3 | 31 | 100% |
| ≥5% | **15.5** | 10 | **97.8%** |
| ≥10% | 4.4 | 2 | 78.0% |

The archived premise ("≥1 asset ≥5% on 95%+ of days, ~20 movers/day") **re-confirms** — slightly stronger for ≥5%.

## 2. Beta-confound (r4 — `beta_confound.py --cadence 1d`)
Most crypto co-moves with BTC; a per-asset edge needs *idiosyncratic* movement. Per-asset OLS of daily returns on BTC:

| Metric | Value |
|---|---|
| median beta to BTC | 1.22 |
| median R² (BTC-explained variance) | **0.31** (mean 0.30) |
| assets with R² ≥ 0.3 | 55.6% |

Movers/day **raw → idiosyncratic** (residual after removing beta):

| Threshold | raw | idiosyncratic | idiosyncratic fraction |
|---|---|---|---|
| ≥2% | 34.9 | 28.5 | 82% |
| ≥5% | 15.4 | **10.0** | **65%** |
| ≥10% | 4.4 | 2.6 | 58% |

**Read:** ~35% of ≥5% movers are just BTC; ~65% survive as independent per-asset moves. The confound is real but
does not erase the opportunity. (Caveat: residual uses a *full-sample* beta — mildly optimistic / in-sample.)

**But "idiosyncratic" is not fully independent** (`idiosyncratic_clustering.py`): after stripping BTC-beta, the
residual daily returns still co-move at **mean pairwise corr 0.27** (vs 0.46 raw). There are common factors
*beyond* BTC (ETH-beta / alt-season / sector rotation) that a single-factor strip doesn't remove. So the
diversifiable, truly-independent opportunity is **smaller** than "65% survives" suggests — a portfolio of
"idiosyncratic" movers still carries ~0.27 cross-correlation (higher concentration risk than naive). A
multi-factor (BTC+ETH+alt-market) residual would shrink this further; single-factor here is conservative-honest.

## 3. Durability across time + regime (r3 — `regime_variation.py`)
Movers/day ≥5% (raw |return|):

| By year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| avg movers/day | 6.5 | 16.2 | 13.2 | 9.8 | 20.8 | 24.5 | 20.2 |

| By BTC regime (trailing-30d) | bull (1271d) | bear (1033d) |
|---|---|---|
| avg movers/day ≥5% | 15.0 | **16.4** |
| % days with ≥1 mover | 98.1% | 98.2% |

**Read:** present every year (never near zero; rising into 2024-26) and **regime-independent** — bear markets have
*slightly more* ≥5% movers. The pool is durable, not a bull-market artifact.
**Honest caveat (long-only):** this counts |moves|; in bear markets many are *down*-moves a long-only spot bot
(our LO+spot+lev=1 constraint) cannot capture — quantified next.

## 3b. Two refining cuts (r3b `long_only_skew.py` + cadence sweep `move_distribution.py --cadence ...`)
**Long-only UP vs DOWN movers/day ≥5%** (UP = harvestable for LO+spot):

| Regime/year | UP /day | DOWN /day | UP share |
|---|---|---|---|
| ALL | 7.6 | 7.9 | 49% |
| BULL | 8.5 | 6.5 | 57% |
| BEAR | 6.7 | 9.8 | 41% |
| 2022 (bear) | 5.9 | 7.3 | 45% | 
| 2025 | 11.1 | 13.4 | 45% |

**Read:** the long-only surface is **~7.6 up-movers/day** (≈ half the |move| headline), **bull-tilted but never
zero in bear** (6.7/day) — durable, modestly bull-weighted. The honest harvestable raw number is ~7.6/day (~5/day
idiosyncratic), not 15.

**Cadence — where the cost-clearing moves live** (next-bar net MFE, taker 0.0024):

| Cadence | net-positive (random) | net ≥2% per bar | net ≥5% per bar |
|---|---|---|---|
| 1d | 46.8% | 29.4% | 13.7% |
| 4h | 47.9% | 13.7% | 2.9% |
| 1h | 47.5% | 5.0% | 0.6% |
| 15m | 43.7% | 1.6% | 0.2% |

**Read:** with a *fixed* 0.24% round-trip, cost-clearing single-bar moves concentrate at **coarser cadences**
(daily/4h). Intraday has many small moves but few clear cost *per bar* → intraday harvesting needs **multi-bar
capture** (ride a move across many bars) or **maker fees**, not single-bar taker. (Caveat: H=1 understates
intraday *multi-bar* moves; a multi-bar-horizon intraday cut is the proper test of intraday opportunity.)

**Concentration — broad, not a few names** (`concentration.py`): the top-10 assets hold only **20%** of all
≥5% up-mover-days; it takes **29 assets to reach 50%** and **57 to reach 80%** (Gini 0.36, moderate). The
opportunity is spread across most of the universe, not a bet on 5 volatile names — which *partially* mitigates
single-name survivorship/capacity risk (the universe as a whole is still survivors — see §6).

**Timing + asymmetry** (`timing_asymmetry.py`): opportunity is **bursty** — the daily mover-count has lag-1
autocorr **0.26**, and a high-opportunity day follows a high one **66%** of the time (base 52%); it clusters in
waves (deploy capital regime-aware, not uniformly). **Long-only asymmetry:** at the daily horizon, big UP-moves
are *larger* than big DOWN-moves (median P95 up **13.5%** vs down **11.3%**, ratio 0.84) — the long-only right
tail (pumps) is fat, *favorable* for a setup-chaser. **Survivorship caveat:** this is the surviving universe, so
the worst left-tail (death-spiral delistings) is truncated — the true down-tail is fatter than measured.

**Bar-type + multi-bar (adjacent cuts a2/a3) — corrects the single-bar cadence read above:**
- **Multi-bar beats single-bar at finer cadences.** Over a *full-day* horizon at 4h (H=6), the harvestable
  ceiling is **larger** than daily: net≥5% **20.1%** (vs 13.7% daily), net-positive **74%** (vs 47%) — multi-bar
  MFE captures the intraday high *and* gives ~6 entries/day. So the §3b "intraday is cost-dominated" finding was a
  *single-bar* artifact; **with multi-bar capture, finer time cadences offer MORE opportunity** (conditional on
  timing the exit — still a perfect-exit ceiling, not an edge).
- **Dollar bars are not for single-bar harvesting** (`--cadence dollar`): equal-activity sampling → tiny per-bar
  moves (net-positive 22.5%, net≥2% ~0.4%) *by design*. Their value is statistical (more-stationary returns for
  ML), not move-density. **Range bars** exist on only 3 assets — data-limited. → **Time bars (daily/4h) remain the
  frame** for the cost-clearing-move surface; dollar bars are a modeling tool, not an opportunity source.

## 4. External context (r5 — literature, cited)
- Predictability is **real and out-of-sample**, strongest in **small-caps**; dominant predictors are past alpha,
  illiquidity, momentum, size, reversal ([Cakici et al. 2024](https://ideas.repec.org/a/eee/finana/v94y2024ics1057521924001765.html),
  [Liu et al., ML high-dimensional factors](https://www.sciencedirect.com/science/article/abs/pii/S0927538X25003701),
  [intraday momentum+reversal](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)).
- **The skeptic's half (load-bearing for honesty):** the headline momentum edge is **largely a survivorship
  artifact** — "survivor cryptocurrency momentum portfolios do not generate significant payoffs"
  ([On survivor cryptocurrency momentum](https://www.sciencedirect.com/science/article/pii/S1544612326001339);
  [Grobys, "Cryptocurrency Momentum: Is It an Illusion?"](https://onlinelibrary.wiley.com/doi/abs/10.1002/ijfe.70036)),
  and net-of-cost profitability is mixed ([momentum has (not) its moments](https://link.springer.com/article/10.1007/s11408-025-00474-9),
  [risk-managed momentum](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377)).
- **Synthesis:** the edges that exist live exactly where our biases bite hardest (small-cap = survivorship +
  liquidity + capacity). The literature is a *warning*, not a green light: an edge must be proven survivorship-
  and cost-honest or it is one of these illusions.

## 5. The avenue map — where to look (g1; named, NOT mined)
- **By size:** opportunity (raw + idiosyncratic move density) concentrates in **small-cap alts** — the top
  net-positive movers are SPK, PROM, UTK, DEXE, CFG (small caps), consistent with the small-cap predictability
  literature. This is also the highest-survivorship-risk, lowest-capacity corner — opportunity and hazard coincide.
- **By cadence:** daily shows a rich surface; intraday (15m/30m/1h/4h) + dollar/range bars are available and
  unexamined here — a cadence sweep is a clean next cut.
- **Structural-event avenues — MEASURED density** (g1 — `structural_events.py`; |z|≥2 events, daily). How often each
  avenue fires + whether same-day |move| is elevated (CONTEMPORANEOUS coincidence, **NOT** forward prediction):

  | Avenue (feature) | available on | fires (% of days) | same-day move vs quiet |
  |---|---|---|---|
  | **Liquidation spike** (`liq_short_z30`/`liq_long_z30`) | 75-77 assets | ~5-6% | **1.24-1.25×** (elevated) |
  | Basis dislocation (`bs_basis_z30`) | 77 | ~7.3% | 1.06× |
  | Tape imbalance (`norm_flow_imbalance`) | 100 | ~1.8% | 0.95× |
  | Whale flow (`norm_whale`) | 100 | ~1.5% | 0.94× |
  | Funding extreme (`norm_funding`) | 96 | ~0.8% | **0.69×** (LOWER — a calm-market carry signal, not a vol trigger) |

  **Read:** liquidation cascades are the standout volatility-coincident avenue (material frequency + 25% bigger
  moves) — the highest-priority structural marker to investigate. Funding is a *different kind* of signal (carry/
  positioning, fires on quiet days). **Hard caveat:** contemporaneous ≠ predictive — a liq spike may be *caused by*
  the move (reverse causality); whether any of these *predict* a tradeable forward move is the held-open strategy
  question. Also: derivatives avenues (funding/liq/basis) are thinner — present on only 75-96 of 104 assets.

## 6. The −k falsifier ledger (what would make this case WRONG)
| Risk | Status in this research |
|---|---|
| **opportunity ≠ harvestable** | ACKNOWLEDGED + quantified: random entry ≈ 47% net-positive (no edge). A signal is required and untested. |
| **survivorship** | REAL + literature-confirmed: the 104 are survivors; the move surface (and academic momentum) is survivor-optimistic. **Partial read (a4 `survivorship_texture.py`):** *within* survivors the case is NOT a recent-listing artifact — move-density is similar across age cohorts (long-history 0.135, recent 0.139; corr(history,freq)=−0.12). BUT this cannot see DELISTED coins (the true bias the literature calls the killer) — that drag remains unquantifiable with this data. Listing-recency: 45 long(≥4y) / 25 mid / 34 recent(<2y). |
| **beta-confound** | QUANTIFIED: ~35% of ≥5% movers are BTC; 65% idiosyncratic remain. |
| **cost** | NETTED throughout at taker RT 0.0024 (honest default). |
| **look-ahead** | Characterization is descriptive (no forward leak); the only in-sample item is the beta residual (flagged). |
| **long-only skew** | FLAGGED: |move| count over-states the long-only-capturable (up-only) surface, esp. in bear. |
| **capacity** | FLAGGED (literature): small-cap edges face liquidity/capacity limits at size. |

## 7. What this means for the (held-open) strategy phase
**Proceed — with eyes open.** There is a genuine, durable, mostly-idiosyncratic pool of raw moves (~7.6 long-only
up-movers/day ≥5%; ~5/day after beta), every year, both regimes, broad across the universe. That is necessary,
not sufficient: the same data + the literature show the *naive* ways to harvest it (buy-the-mover, plain
momentum) are coin-flips or survivorship illusions. The disciplined path is a **conditioned setup** (gated on the
structural avenues in §5) that beats a cost-matched random-entry null on UNSEEN data, survivorship- and
beta-honest — the exact test our apparatus runs (now **regression-protected**: `check_strat_apparatus` re-proves
its two-sided power on every commit, 2026-06-05).

**Operational refinements from the adjacent cuts (a1-a4 + clustering):**
- **Cadence:** prefer daily/4h for cost-clearing; intraday is viable only with *multi-bar* capture (4h-over-a-day
  beat daily: net≥5% 20% vs 14%) — not single-bar taker. Dollar bars = a modeling tool, not an opportunity source.
- **First avenue to test:** liquidation cascades (fire ~5-6% of days, +25% contemporaneous move) — but verify
  forward (not contemporaneous) and reverse-causality-clean.
- **Sizing/diversification:** "idiosyncratic" movers still co-move at ~0.27 — a basket carries real concentration
  risk; size for it. Deploy regime-aware (opportunity is bursty).
- **Honest ceiling:** every "available move" number here is a perfect-exit ceiling; the realized fraction depends
  entirely on the (untested) signal. The case justifies the search; it does not pre-judge its success.

*Reproduce:* `python scripts/research/move_distribution.py --cadence 1d --horizon 1` ·
`python scripts/research/beta_confound.py --cadence 1d` · `python scripts/research/regime_variation.py`.
