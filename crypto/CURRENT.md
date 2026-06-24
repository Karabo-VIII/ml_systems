# CURRENT — live state (Kiro-written each cycle; committed; <2KB)

PHASE: BUILD (TI Engine Rebuild — sub-day decoupled 7d-slice / 14d-lookback)
UPDATED: 2026-06-24

## ACTIVE CAMPAIGN
Rebuild TI-based trading engines for max profitability within 7d windows.
Target: per-slice profit at {30m, 1h, 2h, 4h, 1d} × {EMA, HMA, DEMA} × {7d, 14d lookback}.
The decoupled architecture (14d signal, 7d trade) is UNTESTED at sub-day cadences.

## WHAT EXISTS (verified, committed)
- V1 MA drawdown-insurance: 8 types × {1d, 4h} = VERIFIED (halves DD, -35% vs -82%)
- V1.1 fine-TF exits: REFUTED (look-ahead leak caught, de-leaked = null, D81)
- V1.2 combinatrix: 30 engines × 275 combos = disciplined NULL (D82)
- Decoupled 7d/14d architecture: BUILT (uncommitted) but NEVER RUN at 30m/1h/2h
- Chimera data: available at ALL cadences (15m/30m/1h/4h/1d), 104 assets, 243 features

## THIS SESSION (2026-06-24, 6h autonomous)
- [ ] Create Kiro spec for the TI engine rebuild
- [ ] Run decoupled engine at 30m/1h/2h/4h with per-slice metrics
- [ ] Measure: % slices profitable, median slice ROI, vs always-long
- [ ] Test recency-decay: does the last 7d signal outperform 14d?
- [ ] If positive: iterate on best cells. If null: document honestly.

## STANDING FINDINGS (from prior session)
- No daily-bar SELECTION alpha (D17/D55/D77 + 6 new experiments)
- Vol IS predictable (top feature: vol-rank, rv_rv_5m)
- Continuation > reversion at daily. REVERSAL at sub-daily (multi-TF analysis)
- Regime timing is the only verified value lever (V1, V3 selection)
- Trailing stops create illusion of alpha (vol-breakout audit)
- Oracle ceiling: 10.4%/day available (empirically real opportunity)

## SPLIT CONVENTIONS
TI/MA: 2020=DEV, 2021=IRON, 2022=no-touch BEAR.
Convention B: TRAIN<=2024-05-15 / VAL / OOS / UNSEEN (>=2024-05-15 sealed).
