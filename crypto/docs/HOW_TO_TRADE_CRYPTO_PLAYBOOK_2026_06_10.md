# How crypto is actually traded profitably — the evidence-grounded playbook (2026-06-10)

> Commissioned by the user: *"research and pull all the information about how to trade crypto, we will
> apply that."* Six parallel research lenses (trend/CTA practice, verified swing practice, risk/sizing,
> market structure, regime detection, psychology/operations), every claim labeled VERIFIED / MODERATE /
> LORE, reconciled against our own 73-entry dead-list. Sources inline. This is the practitioner
> knowledge base the strategy build applies.

## 0. The base rates (the strongest evidence in this document)
- Persistent retail trading profitability is a **~1–3% tail outcome**: Taiwan full-population study
  (<1% of day traders durably profitable net of fees), Brazil futures (97% of 300-day persisters lost;
  0.4% earned more than a bank teller), BIS Bulletin 69 (73–81% of retail crypto-app investors lost
  money 2015–2022), EU regulators (74–89% of CFD accounts lose).
- The best AUDITED multi-decade swing record found: **Peter Brandt, ~58%/yr over ~27 years — at 42%
  win rate and max 1% risk per trade**. That is ~0.15%/day equivalent: two orders of magnitude below
  "1–5%/day", produced by the best verified operator in the genre.
- The survivors are not the predictors. They are the **mechanizers**: sizing discipline + positive
  skew + survival envelopes, executed without override.

## 1. The structure survivors use (verified tier)
**Trend/CTA systems** (Clenow, Carver/AHL, Turtles; crypto adaptation Zarattini 2025 SSRN):
- Entries: Donchian breakouts (20/55d) or EMA crossovers — but as **ENSEMBLES of lookbacks**, not one
  config (the 2025 survivorship-free crypto study: ensemble of Donchian lookbacks on top-20 coins,
  Sharpe >1.5 net, +10.8%/yr alpha vs BTC — single backtest, not a track record).
- Exits: ATR trails 2.0–3.5x, or opposite-channel breaks, or Carver's continuous forecast (position
  shrinks as trend fades). No profit targets. Modern systems don't pyramid discretely — they scale
  continuously.
- **Expected anatomy: win rate 30–45%, payoff 2–5x, profit factor 1.3–2.0, profits concentrated in a
  minority of trades; multi-quarter flat/losing stretches are NORMAL operation** (SG Trend index: two
  >15% 12-month losses since 2000; 2011–2019 dull stretch inside a century of positive decades).
**Verified swing practice** (Brandt audited; Minervini competition-verified; Qullamaggie self-reported
but precise): trade few setups (consolidation-tightening breakout after a strong move; episodic
pivots), HTF bias + LTF trigger so stops sit tight, risk 0.25–1%/trade, win rate 25–45% by design,
partials at 3–5 days, trail the rest with the 10/20d MA, holds days-to-months, low frequency.

## 2. The risk envelope (where the real "edge" of survivors lives)
- **0.5–1% equity risk per trade** (distance-to-stop, not notional); crypto-specific guides converge
  BELOW the classic 2%.
- **Portfolio heat ≤5–6%** total open risk (Elder); in crypto treat the whole alt book as ONE
  ~BTC-beta bet (correlations 0.5–0.8, →1 in crashes) — Turtle-style correlated-unit caps.
- **Vol targeting**: the single best-evidenced practice for crypto specifically — Man Group on BTC
  2012–2024: constant-30%-vol scaling added ~0.4 Sharpe (Moreira-Muir JF 2017 generalizes). Note: a
  20–30% vol-target book runs ~0.4–0.75x notional in calm regimes → it WILL lag buy-and-hold in bull
  legs. That is the design, not a bug.
- **Kelly: quarter-Kelly or less** (estimation error beyond full Kelly = ruin; Thorp).
- **Tiered drawdown ladder** (prop/pod convention): ~5% daily kill-switch, de-risk ~50% at −5%
  campaign, hard review at −10%; crypto prop firms that pay real withdrawals enforce 3–4% daily /
  6–10% max — their rulebooks are revealed preferences about what survives crypto.
- **Counterparty discipline**: per-exchange caps, sweep to custody, expect liquidation engines to run
  on MARK price and ADL to exist (Oct-2025 cascade: book depth −98%, spreads ×30 average during event).

## 3. Market-structure knowledge: risk context, mostly NOT alpha
- Funding/OI = **fragility gauges** (BIS): elevated funding + high OI = crowded longs = cascade fuel.
  Per-bar contrarian thresholds are lore. (Matches our D71: the liq signature adds ~0 timing info.)
- Liquidation cascades are sub-minute and mechanical — value is in leverage/stop/venue DESIGN, not
  real-time trading (matches D47/D71).
- **The two documented calendar edges**: token unlocks (Keyrock, 16k events: ~90% decline base rate,
  weakness starts T-30d, sized by %-of-supply) and exchange listings (Empirica 2024: sell-the-news;
  98% of Binance listings dump from listing price; −37.6% median at 6 months). Low frequency,
  single-digit magnitudes, borrow access often binding — but these are REAL base rates from large n.
- Session effects: return seasonality unstable (don't bet it); LIQUIDITY seasonality persistent (time
  your EXECUTION: depth peaks in LSE/NYSE overlap, thins on weekends).
- All-in costs run **0.15–0.30% above headline fees** once spread/slippage/funding included; maker
  p_fill reality 0.21–0.40 (our own calibration, independently consistent).

## 4. Regime detection: drawdown technology, not return technology
- The 200d-MA family (Faber): on 111 years of equities, +0.9pp CAGR but Sharpe 0.32→0.55 and max DD
  83.7%→42.2%; **underperforms buy-and-hold in roughly half of all years**. Zakamulin: even the return
  edge may be data-mining; the DD reduction is the durable part. Mechanism (Gayed): below-200d = the
  high-vol regime. (Matches our thread-22 book + M2 + the bear-abstention findings precisely.)
- ADX/ER/breadth/HMM filters: mechanically sensible, publicly UNVERIFIED as return-improvers; hard
  thresholds cut trade counts 60–80% without improving expectancy; prefer continuous scaling (Carver)
  over binary gates. HMM regime IDENTIFICATION is academically real; TRADING it profitably OOS is not
  established (and smoothed-probability lookahead is the standard trap — our G-AUDIT-011 class).
- n_effective of regime evidence is ~3–5 historical episodes, not thousands of bars. Expect filters to
  look useless for years.

## 5. Psychology/operations — where the 1–3% is actually decided
- **Drawdowns are guaranteed**: even a PERFECT-foresight portfolio drew down 76% ("Even God Would Get
  Fired", Alpha Architect). At Sharpe 0.5/10% vol, ≥20% DD is ~80% probable over 25y. A crypto book at
  30–60% vol should EXPECT 40–70% drawdowns. Size for that, not for the backtest's DD.
- **The #1 destroyer of positive-skew returns is abandonment at the trough**: Goyal-Wahal (J. Finance,
  3,400 plan sponsors): fired managers subsequently outperform; Man Group: a 20%-DD exit trigger costs
  ~2.2%/yr; post->10%-DD windows average +9.8% forward 12m. Write kill criteria BEFORE launch;
  "drawdown alone" is an invalid kill reason.
- **Win-rate psychology**: at 35% win rate, 8–12 consecutive losses are routine binomial outcomes.
  Pre-compute the streak math and publish it to yourself before going live.
- **"I can see trends" — the eye is real but miscalibrated**: humans CAN distinguish real markets from
  noise (Lo's Financial Turing Test) — but systematically extrapolate trends (De Bondt), underestimate
  vol via hindsight bias (Biais-Weber: more-biased bankers performed worse), and see clusters in noise.
  Rule: a visually-spotted pattern earns capital only after surviving a mechanical, costed, held-out
  test. Eyeballed conviction gets zero sizing weight.
- **Algorithm aversion**: people abandon systems they watch err faster than humans who err identically
  (Dietvorst). The Turtle cohort's differentiator was who TOOK every signal. Automate flows; the
  measured behavior gap is ~−1.1%/yr and shrinks toward zero with automation (Morningstar).
- Review cadence: daily = operations only (fills/slippage/feeds); quarterly = signal bands; multi-year
  = keep/kill verdicts. P&L feelings are not a review input.

## 6. Reconciliation with OUR evidence (what we've independently confirmed or refuted)
| Playbook claim | Our independent verdict |
|---|---|
| Trend anatomy: win 30–45%, skew pays | CONFIRMED per-trade (family_regime_map OOS: win 33–37%, PF 1.66–1.93, 8–10/10 assets positive sum) |
| Regime filter = DD technology | CONFIRMED (thread-22 book: Calmar 19.8 vs 11; bear-abstention value; M2) |
| Per-bar funding/OI/liq timing = lore | CONFIRMED-REFUTED at 1m resolution (D71, D72: AUC 0.52) |
| Config choice doesn't persist | CONFIRMED (D73: persistence anti-predictive; weekly oracle DNA churn) |
| Vol targeting helps | UNTESTED in-house at book level — adopt from evidence (Man/JF) into the build |
| Unlock/listing calendar edges | UNTESTED in-house — the one NEW exogenous avenue this research surfaced (needs unlock-calendar data; cf. thread 24's external-data theme) |
| Maker p_fill optimism | CONFIRMED long ago (D43: 0.21–0.40) |
| 1–5%/day targets | INCONSISTENT with the entire published record incl. the best audited operators and the perfect-foresight benchmark |

## 7. What we apply (the build contract this playbook implies)
1. ONE portfolio trend book: ensemble lookbacks (not one config), u10→u50, LONG-only spot per
   constraints, 200d-class regime gate, ATR trails, **vol-targeted sizing, 0.5–1% risk/trade, 5–6%
   heat cap treating alts as one beta bet**, daily kill-switch + tiered DD ladder, maker-first
   execution in the LSE/NYSE overlap, costed at all-in (not headline) rates.
2. Expectations published BEFORE launch: win rate 30–45%; 8–12-loss streaks routine; DD up to
   ~half the vol budget; flat years possible; judged on multi-year compound vs pre-registered bands,
   never on weekly P&L or vs B&H in bull legs.
3. The calendar sleeve (unlocks/listings) enters research as the one new exogenous edge candidate —
   through candidate_gate like everything else.
4. The eye's role: idea GENERATION only. Every visual hypothesis becomes a mechanical rule and faces
   the battery. (This is also the only honest answer to hindsight bias.)
