# CLAUDE_CURRENT — Claude's live state (Claude-owned; Kiro does NOT touch). Synced 2026-06-28.

> Project-spec live state = Kiro's `CURRENT.md` (re-read for Kiro's latest; it updates as Kiro builds). THIS file =
> my synced view + my operating-model state. Co-existing: Kiro builds the project-spec; I review/verify/build + own
> the anti-fabrication discipline. Shared truth (dead-list, registry, docs/engines) co-maintained.

PHASE: AUDIT+ORACLE+UPGRADE campaign DONE (runs/engines/AUDIT_ORACLE_UPGRADE/). Audited Kiro V2-V15 (honest, one
real leak found+fixed), built the long-only oracle, delivered the de-leaked capture upgrade. Leak-fix to source in
progress (delegated). Next genuine frontier = crypto-native (user-gated).
UPDATED: 2026-06-28

## THE AUDIT VERDICT (Kiro V2-V15, adversarial)
Kiro's body of work is HONEST + reproducible — no fabrication, every headline reproduces. ONE systemic defect: a
**same-day 1d-regime LOOK-AHEAD LEAK** (the exact V1.1 `align_htf_on` bug class) in the `btc_regime` broadcast,
replicated across ~9 gating consumers. MODEST: touches 1.7% of OOS bars, edge survives, ~15% haircut (NOT the V1.1
catastrophe). Fixed via `regime.shift(1)` (day-D reads the last CLOSED 1d bar). All numbers below are DE-LEAKED.
Nulls D83/D84/D85 confirmed (D83 narrative refined: "selection-by-recent-P&L=no-skill" holds, but a random-10-config
book matched frozen-114 in 6/12 cells => the 114-config band is over-diluted, a minor live lever). Pipeline indicator
path clean; the regime path inherited the same leak (now fixed). ADX|1d genuine-selective but VAL-fit => PROVISIONAL.

## THE SYSTEM MAKES MONEY OOS — momentum mover-selection +57%/2yr (2026-06-28; full_track_momentum)
CLEAN OOS (2024-05-15..2026-05, ~2yr, $10k, prod-grade, NO in-sample): mom top8 inv-vol +57% ($15.7k) > mom EW +47%
> inv-vol de-risk (what I shipped) +43% > BH BTC +11% ($11.1k) > BH alt-basket -55% ($4.5k). maxDD: mom -36 vs BTC
-50 vs basket -81. So the deployable (momentum) BEATS buy-hold BTC ~5x at lower drawdown -- it is NOT a net-loser.
I had understated it twice: (a) anchored on the single WORST 14-mo sub-slice (2025-Q2..2026 net-down = breakeven),
(b) shipped the weaker de-risked config. Full-track incl in-sample = +679% but that is IN-SAMPLE-band-inflated (2023->
2024-05 band fit) -- do NOT headline; the clean OOS +57% is the honest number. Per-year: catches bull movers (2023/24
big), small loss in down years (2025/26). Cost: DD-36 (momentum concentration). DEPLOYABLE = momentum top-8 mover-selection.

## MOMENTUM UN-SHELVED — the deployable SELECTION is mover-selection, NOT de-risked EW (2026-06-28; forward_mover_diagnosis)
User critique (valid): "can't miss the movers all year and be net-negative." CONFIRMED: over 2025-Q2..2026-05 the
ORACLE (perfect top-8 mover-selection per 7d slice) = +604.6% (Q3-25 +220%); my deployed inv-vol engine captured ~none
(-12.2%). That is a SELECTION FAILURE, not de-risk. FIX: momentum mover-selection (mom top-8 EW -0.5% / vol-weighted
-0.6%, Q3-25 +14..25%) recovers ~12pp -> breakeven + positive Sharpe. I had SHELVED momentum on ONE quarter (Q1-25, a
momentum CRASH = worst case) -- WRONG; cross-window it WINS 3 of 4 (VAL/OOS24/forward; only Q1-25 bad). UN-SHELVED:
**deployable SELECTION = momentum top-8** (recent STRENGTH; NOT raw vol -- vol-tilt = -43% disaster). CEILING HONESTY:
even momentum ~breakeven on this NET-DOWN cycle (basket -37%); the gap to +604% = perfect which-names-run foresight
(un-capturable); momentum is positive in net-up legs. LESSON: never decide a lever on ONE window (a crash quarter is
the worst case, not the verdict). [[feedback-no-single-window-verdict]]

## THE DEPLOYABLE = THE MA FAMILY OF ENGINES on the FULL u100 (NOT BTC). PROD-GRADE (2026-06-28; prod_ma_family_q1_2025.py)
CORRECTION (user /orc 2026-06-28): I drifted to a "BTC-core" book off the Q1 "BTC fell less" observation -- that
COLLAPSES the breadth surface (u100->BTC) + is OFF-SPEC. The spec/product = the MA FAMILY OF ENGINES on the FULL
u100 alt universe, canonical **14d lookback / 7d trading window, ROLLING, regime-gated, long-only spot**. BTC is a
context benchmark ONLY, never the product. A regime observation is NOT a mandate to change the product. [[feedback-no-u100-to-btc-collapse]]
PROD-GRADE backtest (u100 77 names, 14d/7d rolling, de-leaked, next-open fills, prod cost, PIT, turnover):
- **VAL: MA family +118-130%** (SMA|1d +130, CONV|1d +122) ~= BH alt-basket +138 (gate costs a little in the rip).
- **OOS24: MA family +54-87%** (KAMA|4h +87, EMA|1d +74, CONV +65-72) -- BEATS BH alt-basket +33 AND BH BTC +41 (gate adds value in chop).
- **Q1-2025 (crash): MA family PRESERVED CAPITAL** -- CONV|1d breadth cap33 -7.3% (next-open) / -14.6% (close-close);
  KAMA|1d -5.1/-15.3 -- vs **BH alt-basket -50.5%** and BH BTC -12.9%. ROBUST claim = preserved capital vs the basket
  (regardless of fills); whether it beat BTC is EXECUTION-SENSITIVE (o2o yes / c2c ~tie) in this small gappy quarter -- do NOT claim beat-BTC.
DEPLOYABLE config: the cross-type CONVERGENCE book (or per-type ensemble) @ **1d, breadth gate, cap33, 7d/14d, EW
inv-vol (NO momentum -- refuted on Q1 unseal)**. 1d is the resolution (4h bled in the Q1 chop). The MA family is an
all-weather capital-preservation product on the FULL universe: captures most of the bull, beats the basket in chop,
preserves capital in the crash. No BTC core needed.

## 2025-Q1 UNSEAL VERDICT (the final OOS gate, 2026-06-28; src/strat/deployable_v3.py + wave4_2025q1_unseal)
Ran ALL MA-family candidates FROZEN on Jan-Mar 2025 (= all the 2025 OOS data; ends 2025-03-31). Q1-2025 = a brutal
alt quarter (BH EW-u100 -48.6%, BH BTC -11.8%). VERDICTS:
- **BREADTH GATE = CONFIRMED.** Best engine: conv breadth+EW -14.6% (deploy 19%) vs old btc-gate cap33 -24.5%
  (deploy 51%) vs alt-basket -48.6%. The de-risk core held on a real stress quarter across 3 windows. **This is the
  validated deployable: breadth gate + conv band + EW (NO momentum).**
- **MOMENTUM = REFUTED on fresh OOS.** Adding it made everything WORSE (breadth+EW -14.6 -> +mom8 -21.7; btc+cap33
  -24.5 -> +mom8 -34.8) = textbook MOMENTUM CRASH (concentrated into Jan winners as they reversed Feb-Mar). The
  OOS-2024 "+170%/2.5x" was REGIME-SPECIFIC (2024-H2 melt-up), exactly as the 94th-pctile borderline warned.
  **V3-as-locked (with momentum) MUST NOT be deployed. Momentum SHELVED** (regime-conditional, crashes in reversals).
- **SOBERING:** nothing beat BH BTC (-11.8%) -- the alt apparatus underperformed BTC in this risk-off quarter; the
  gate->cash is the right floor but still lagged BTC. Retail-wealth implication: a BTC core hold matters in BTC-dominance.

## THE DEPLOYABLE VERDICT (both agents converge here; DE-LEAKED)
**Capital-preservation within long-only = the deployable answer.** MA band engine (V1 1d-BTC-SMA100 regime gate +
V2 4h decoupled MA band) = drawdown-insurance: bear-2022 ≈ flat (94% cash), ~65% deployed in bull, Sharpe ~1.4-1.9.
**Honest de-leaked deployable rate (prod cost, OOS-2024 / cross-checked on VAL):**
- cap15 (old headline): de-leaks to **~+1.4-1.6%/wk** (OOS +1.56 / VAL +1.84) — was leak-inflated ~15% AND cap-suppressed.
- **cap33 = the robust UPGRADE operating point: ~+1.8-2.0%/wk** (OOS +1.82 Sh1.39 DD-33; VAL +1.96 Sh1.86 DD-43),
  cross-window-robust, no single-name dependence, above the 1.5%/wk floor. THIS is the delivered capture upgrade.
- cap50/EW (+2.2/+3.5/wk on OOS) = REJECTED: window-luck. The EW lift is 69% ONE name (ZENUSDT) + reverses on VAL.
The per-name cap is single-name TAIL INSURANCE, NOT a drawdown dial — maxDD ~30% is FLAT across caps; the REGIME GATE
is the only drawdown control. **DO NOT headline the +171K..+691K% DEV compounds** (~9.5x rebal-premium; OOS only ~1.1x).

## ROSTER (audited, Kiro V5/V6 + my V1.x)
- SELECTIVE: **MA band V2 (WMA|4h)** = the canonical selective winner; **ADX|1d (PROVISIONAL)** = the only non-MA
  selective (40-67% deploy, +0.6-1.1pp OOS) — but VAL-fit (VAL +2.7-3.8pp >> OOS), lookback-sensitive (dies <lb22),
  re-confirm on 2025 unseal. Treat 2025 as pass/fail, not a formality.
- ROBUST NON-SELECTIVE tilts (100th-pct null, VAL/OOS-consistent, deploy 96-100%): HMA|1d, WILLR|1d, Bollinger|1d,
  TEMA|1d, VIDYA|4h, SMA|4h, VWAP|1d, MFI|1d, Supertrend|4h, WMA|4h, PSAR|4h (+~10 @ score 4). Refine WHICH longs,
  not WHETHER. Bollinger|1d 0.96 corr w/ MA = redundant. RSI|4h is OOS-only (VAL -0.64) = weaker than V3 claimed.

## THE NORTH-STAR RECKONING (the key strategic truth, 2026-06-27)
Mandate "profitable in ANY 7d window" → measured **34%** of 7d windows profitable (3d=39%, 1d=32%); median +0.00%.
Trend is STRUCTURALLY low-hit-rate (lose small often, win big rarely). **This mandate is unreachable long-only** —
you can't make a market-wide down-week green without shorting (banned) or perfect cash-timing (= the de-risk ceiling
the engine already approximates). Kiro's "needs anti-correlated complements (mean-reversion / market-neutral)" path
brushes the banned shortcut — STRIKE the market-neutral framing. Honest reframe (user, 2026-06-27): the objective is
de-risk the down-weeks + minimize exposure, NOT profit every week. That is already solved (V1).

## SETTLED NULLS (don't re-mine; fold to dead-list)
- Config-rotation by recent P&L: REFUTED — frozen band wins 11/12 (D73 5th confirmation, → D83).
- Within-type top-5 convergence: loses to full band (near-clones; only ACROSS-type adds marginal lift) (→ D84).
- Dynamic/adaptive (V7 VoV-threshold): VAL-fit, null OOS (→ D85). Phase 2 (DYN-1..4) re-tests with adaptation-null.
- Sub-daily (1h/30m): universally bleeds (cost+noise). My V1.1 fine-TF null + V1.2 combinatrix null agree.
- Combinatorial TI mining (a/ab/abc): disciplined null, PBO 0.71 (D82).

## THE LONG-ONLY ORACLE + GAP DNA (DONE — the proposed item, runs/engines/AUDIT_ORACLE_UPGRADE/oracle_gap.json)
3 levels (%/wk, slice-prof; DE-LEAKED): L1 perfect-cash-timing per-slice **+12.0%/wk @94%** = absolute long-only
ceiling; L2 best-realizable causal (hindsight MA type, EW) +3.5; L3 V2 frozen-EW +3.29 (capped deployable ~+1.4-2%).
GAP SPLIT: recoverable (L3→L2) = the concentration/cap dial; un-capturable (L2→L1) = directional prediction (refuted
vein) = the structural ceiling (= North-Star 34% hit-rate). **REFINEMENT (Wave 2):** the oracle's "+10.5pp/wk
recoverable via concentration" OVER-states it — most of that headroom is single-name window-luck (the ZEN/EW mirage),
NOT robust. True robust recoverable within the MA mechanism ≈ the cap15→cap33 step (~+0.3pp/wk). Internal price-TI
ceiling is REACHED; higher robust capture needs a new de-correlated mechanism or crypto-native data.

## WAVE 3 — MOVER-SELECTION (user 2026-06-28: extract more from MA DEPLOYMENT, internal-data only)
Oracle says recoverable headroom = name-SELECTION (perfect top-3 = +14/wk @ -24%DD, 4x return AND shallower DD), NOT
type/config/lookback (refuted/thin). Tested 5 causal price scores (mom/accel/brk/dist/volx) as top-K tilts vs
random-K null + EW. **Cross-sectional MOMENTUM is the only one with cross-window skill** (beats random-p90 + EW on
BOTH windows; the documented crypto factor). BUT hardening (200-draw null + deployable soft-tilt): mom top-5 = 100th
pctile VAL / **94th OOS** (just shy of 0.05); the de-risked soft-tilt WINS on VAL (Sh1.39->1.92) but REVERTS to ~EW on
OOS. You are paid ~fairly for the concentration risk (= "magnitude is priced"). => momentum-tilt = PROVISIONAL/regime-
conditional (pays in high cross-sectional DISPERSION), NOT a free upgrade. Internal price-TI ceiling effectively REACHED.

## OPEN / NEXT
1. **Momentum-tilt = the one live lift candidate** (PROVISIONAL): gated on 2025 unseal; pays in high-dispersion regimes.
   Possible refinement = dispersion-conditioned tilt, but risks overfit on only 2 windows. Treat 2025 as pass/fail.
2. **ADX|1d** — PROVISIONAL/gated (same bucket, less principled): re-confirm on 2025. Wave-2B: diversification VAL-fit.
3. Confirmed deployable = **cap33 EW ~+1.8-2%/wk**, drawdown-insured by the regime gate. Bank this.
4. Crypto-native (funding/OI/on-chain) = the only NEW-edge frontier, but user (2026-06-28) said NO new data sources.
5. **Phase 2 dynamic** (Kiro-designed, user-gated) — expect mostly null; run as falsification.

## COORDINATION + HOUSEKEEPING STATE
- CHARTER/CURRENT/KNOWLEDGE_MAP = Kiro-owned. CLAUDE_CHARTER/CLAUDE_CURRENT = mine. Router injects mine for Claude
  sessions (+ Kiro's CURRENT pointer). Give-up gate untouched (reads Kiro's CHARTER constraints — fine; never loosen).
- Dead-list synced to D82+Kiro nulls (D83-D85). Auto-memory has a Kiro-era shard. Knowledge map fresh (Kiro, 06-26).
