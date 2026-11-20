# TI CANONICAL STORE -- index (2020 technical-indicator discovery)

The canonical, navigable entry point to ALL technical-indicator (non-MA) discoveries on the 2020 band. 18
indicators / 6 families, every config x timeframe, run through RAW -> IRONED -> RECOVERED (mechanical fighter),
wealth-ranked, robust-split, firewall-checked. STRICT long-only spot, fixed-EW (cadence-invariant), VAL Jul-Sep
/ OOS Oct-Dec, maker cost. ALL numbers **[VERIFIED-backtest, IN-SAMPLE 2020 OOS]**. (The 8 base MA types are the
other lane: `CONFIG_LEADERBOARD.md` / `MA_*`.)

## THE INDICATOR UNIVERSE (18 / 6 families)
- **trend:** MACD, SUPERTREND, PSAR, VORTEX, ADX
- **momentum:** ROC, TSI
- **breakout:** DONCHIAN, KELTNER
- **mean-reversion:** RSI, STOCH, BBPCT (Bollinger %b), CCI, WILLR (Williams %R)
- **volume/order-flow:** OBV, MFI, VOLIMB (taker buy/sell imbalance), CMF

## THE LAYERS (what was run)
1. **RAW** -- the bare indicator signal + FULL stack (trail10 + min_hold + maker), per config per TF.
2. **IRONED** -- raw + the family iron (trend = zero-line/slow-trend confirm; mean-reversion = buy-dip-in-uptrend;
   breakout = ATR-confirm; momentum/volume = uptrend-confirm) + vol-target.
3. **RECOVERED** -- the MECHANICAL FIGHTER at fine TFs (15m/30m): entry-CONFIRM + longer MIN-HOLD + COOLDOWN to
   cut the over-trading that cost-destroys fine TFs. Salvages 15m/30m from negative to positive de-risked beta.

## THE CANONICAL DOCS (start here)
| doc | what it answers |
|---|---|
| **`TI_REGISTRY.md`** + `ti_registry.json` | **PRIMARY.** Per-TI performance profile: RAW/IRONED/RECOVERED best + TOP-10 per TF + the deployable profile (best config, ceiling-xBH, deployable band). One card per indicator. |
| `TI_BEST.md` + `ti_top10.json` | The 1118 ROBUST ironed configs ranked by wealth -- the single best deployable picks (overall / per TF / per family). |
| `TI_MASTER.md` | All 92 (indicator x TF) cells, base vs ironed, + per-family iron-effectiveness. |
| `TI_FAMILY_TF_SUMMARY.md` | The 4 family x TF matrices: capture (xBH), robust-fraction, maxDD, Sharpe. |
| `TI_FAMILIES.md` | Per-family base-vs-ironed deep read + the deep-research irons. |
| `TI_BREADTH.md` | Per-INSTRUMENT firewall: are the deployable picks broad across coins (yes, 69% mean breadth) or concentrated (no). |
| `TI_FIGHTER_15m.md` / `TI_FIGHTER_30m.md` | The turnover-fighter ladder -- proof the fine-TF deterioration is mechanical (cost) and recoverable. |
| `LITERATURE_CROSSCHECK.md` | The deep-research validation of the de-risked-beta thesis + the irons. |

## THE TOOLS (regenerate everything)
`src/strat/deep2020_ti_pipeline.py` (raw+ironed, `--indicator X --cadences ...`) -> `_render.py` (cross-indicator)
-> `_top10.py` (per-TI top-10) -> `_master.py` (TI_MASTER) -> `_best.py` (TI_BEST) -> `_breadth.py` (firewall)
-> `_15m_fighter.py --cadence {15m,30m}` (fighter ladder) -> `_recover.py --cadences 15m,30m` (full-grid fighter)
-> `_registry.py` (the canonical TI_REGISTRY). All read/write under this folder.

## THE HEADLINE VERDICT (uniform across all 18 TIs / 6 families)
**The iron buys RISK-REDUCTION + ROBUSTNESS, NOT return; no internal-data indicator family beats long-only
buy-hold on net in the 2020 bull.** [VERIFIED-2020-OOS]
- Trend/momentum/breakout/volume = de-risked betas: best ironed ~0.5-0.85x buy-hold; iron cuts maxDD ~40% +
  robustifies at modest net cost. Deployable band 1d-1h (4h sweet spot); single best = ADX(14,20)@4h (0.77xBH,
  Sh 3.49, maxDD -6.9).
- Mean-reversion = weak-but-most-robust (~0.5x BH, lowest maxDD -2..-7, 73-100% robust) = a defensive,
  low-return diversifier; it does relatively BEST at fine TF with the fighter (STOCH 15m recovered 0.56xBH Sh 5.39).
- FINE TF (15m/30m) is NOT to be discarded: the mechanical fighter recovers ALL 18 TIs from cost-destroyed to
  positive de-risked beta (monotone in turnover-cut) -- but recovers them TO the drift-beta ceiling, not past it.
- Firewall PASS: picks are broad cross-asset drift, not single-coin concentration.

## METHODOLOGY LOCKS (binding for any extension)
- Rank by WEALTH (OOS compound), NOT Sharpe. Robust := |drift|=|OOS-VAL| <= 10 (delivers in BOTH windows).
- Fixed-EW alignment (unlisted=cash), NOT mean(skipna) -- cadence-invariant (buy-hold ~47-55% across all TFs).
- SELECT on VAL, REPORT on OOS. Maker cost. UNSEEN (2025-2026) untouched.

## OPEN (next levers -- both gated by user-set constraints)
- Full-cycle (2021->2022) validation of the deployable picks -- needs lifting "stay in 2020".
- Combining the orthogonal mean-reversion + trend families -- needs lifting "not solving for correlation".
