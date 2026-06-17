# WAVE CAMPAIGN — 8h autonomous run state (rolling; update every cycle)

**Mandate (user, 2026-06-10 ~23:38 SAST):** *"exhaust every nook and cranny of the project, work
autonomously for the next 8 hours... result north star is still the same [CAMPAIGN_CHARTER]. Note
the per-asset angle and DNA... what works on one asset may not work on another. Build things end to
end until I return."* Infra-only (Coinglass deferred until returns are shown).

**VERIFIED start:** 2026-06-10 23:40:30 SAST (`date`) → window ends ~07:40 SAST 2026-06-11.
**Operating rules:** commit per finished item (explicit paths, NEVER `git add -A` — concurrent
instance active, commit 552c828 is theirs); UNSEEN discipline per component; honest verdicts only;
update the CHECKPOINT below each cycle so compaction survives; charter + register are the law:
[CAMPAIGN_CHARTER_2026_06_10.md](../../docs/CAMPAIGN_CHARTER_2026_06_10.md),
[CANDIDATE_REGISTER_2026_06_10.md](../../docs/CANDIDATE_REGISTER_2026_06_10.md).

## Build queue (EV-ordered; per-asset DNA woven in per the user's principle)
1. **XAUT gold bear-sleeve** on wave1_book (LO-legal, data in-repo): idle-cash → XAUT when
   XAUT>SMA120, variants {50%, 100%} of idle cash (pre-registered, 2 trials) → re-run gates.
2. **Param-perturbation battery + firewall** on the composed Wave-1 book (LB_SETS × SMA × overlay
   robustness; ≥80% configs positive full-cycle = gate).
3. **PER-ASSET DNA workstream** (the user's named angle): extend family_regime_map to u50; test
   per-asset family assignment vs pooled per-class (PBO-disciplined; D62 caution: capture may be
   regime- not asset-driven — the test decides); wire asset_dna (BLUE/STEADY/VOLATILE) as sizing
   tiers (pos_cap/kelly_frac already in universe yaml).
4. **Breadth-bounce 4h RSI satellite** book assembly (VERIFIED +4.93%/trade OOS PF 13 at f35c7d4;
   rebuild from archive on current apparatus, regime-gated, κ trailing exit).
5. Family-map trend satellites u50 + battery (per-class, under heat cap).
6. Sector-stratified rotation (needs config/sectors yaml first, frozen before backtest).
7. Breadth-thrust gate variant on the CORE; D61 exit sub-axes on the bounce satellite.
8. Token-unlock ingest (DefiLlama free) + exclusion-overlay event study; listing-seasoning
   event study (descriptive artifact).
9. MR chop cells (RSI/BOLL 1d from family_regime_map) as small chop satellites.
10. Compose ALL surviving sleeves under the §3 envelope → stage-03 SHIP run → stage-04 paper wiring
    (the end-to-end the user asked for).

## CHECKPOINT (update each cycle)
- [23:40] Wave-1 v1 RUN + committed: BLEND core + seasoning + vol-target. Full-cycle +23-47%/yr
  p05>0; UNSEEN bear: preserved (−6.6% vs B&H −39%) but ann<0 → NOT SHIP. Mechanism = LO bear
  ceiling → bear sleeve needed.
- [23:48] GOLD sleeve attempt: code wired into wave1_book.py (pre-registered {50,100}% idle-cash,
  XAUT>SMA120 gate) but **XAUT/PAXG in-repo history = only 48-51 days (2026 listings)** → gold is
  NEEDS_DATA. ACTION: PAXG spot klines 2019→ fetch launched in background (fetch_all.py, free
  Binance vision = our infra). When it lands: resample 1m→1d, plug into the gold block (load path
  swap), re-run gates. ALSO launched: family_regime_map --universe u50 (per-asset DNA angle,
  queue item 3) in background.
- [23:50] u50 PER-ASSET DNA MAP DONE: runs/mining/family_regime_map_u50_20260610_234730.json.
  Map pooled OOS: n=2118 trades, win 35%, PF 1.27, 29/~48 assets positive (display says "/10" --
  cosmetic hardcode bug in family_regime_map print, fix when next touched). TREND class OOS PF 1.23
  (n=24k); MR class OOS PF 0.91 = LOSES at u50 breadth (was ~1.0 at u10) -> MR chop satellites
  DEPRIORITIZED. Trend premium persists at breadth, thinner per-trade than u10.
- [23:48] PAXG fetch DONE but only 440 days (futures-era 2025-03->2026-05; spot 2019+ did not
  resolve -- listing-date resolution used fapi). Usable for HELD-OUT-window gold-sleeve read only;
  full-cycle gold backtest needs spot-zip fetch with explicit start date (retry with
  --start-date 2019-09-27 forced, or accept held-out-only evidence).
- wave1_book.py now has --battery mode (perturbation with overlays held) -- NOT YET RUN.
- NEXT (in order): (a) run `python -m strat.wave1_book --universe u50 --battery`; (b) per-asset
  DNA analysis: TRAIN-select family per (asset,regime) at u50 -> OOS map economics vs pooled
  per-class (PBO-disciplined; D62 caution); (c) gold sleeve held-out read w/ 440d PAXG (swap
  load path from XAUT to PAXG 1m-klines resampled 1d); (d) breadth-bounce 4h satellite rebuild
  (queue 4, archive code reference only); (e) compose surviving sleeves -> stage-03 run.
- Artifacts: runs/strat/wave1_book_u50_20260610_{233717,234231}.json; src/strat/wave1_book.py;
  runs/mining/family_regime_map_u50_20260610_234730.json; data/raw/PAXGUSDT/ (440d).

## /orc RE-GRADE CAMPAIGN COMPLETE (2026-06-11 ~12:22 SAST)
Objective VERIFIED SOLVED. Delivered end-to-end + committed:
- wave1 OOS-selection bug FIXED (select pre-OOS; OOS+UNSEEN untouched). Honest: NOT-SHIP, gold non-robust.
- src/strat/scorecard.py -- canonical honest evaluator (selftest PASS). Use for ALL future grading.
- src/strat/regrade_leaderboard.py + docs/STRATEGY_LEADERBOARD_2026_06_11.md -- every strat re-graded.
- Family2 firewall RWYB: OOS selection real (100%>rand), UNSEEN within random (73%) = D68; +172% was concentration.
- docs/DEPLOY_DECISION_2026_06_11.md -- survivor (regime-gated trend book) + risk-point frontier + honest expectations.
SURVIVOR: regime-gated trend book (full p05 +59..+106, beats buyhold+null, preserves bear -2 to -3.5%, NOT UNSEEN-positive).
Internal LO+spot data EXHAUSTED-CONFIRMED across the whole inventory. BUILD gaps user-gated (perp short / external data).
NEXT (user decision): risk point (BLEND_25/50/75 / regime_beta) + paper-trade greenlight.

## UNSEEN PAPER-TRADE (2026-06-11 ~12:35) -- the final litmus, all candidates
src/strat/unseen_paper_trade.py: every candidate frozen, run forward on sealed UNSEEN (5mo bear).
RANKED final%: TSMOM -2.1, BLEND_25 -2.4, BLEND_50 -2.8, BLEND_75 -3.1, regime_beta -3.5, RANDOM -6.4,
Family2 -9.3, buy_hold -18.3, low_vol -20.1. Regime-gated family (top5) preserved robustly + beat null;
NONE forward-positive (long-only can't earn in a bear); only positive = short leg (perp, user-gated).
ANSWER to 'only one survived?': NO -- a regime-gated FAMILY preserves robustly; none earns long-only
in a bear; the short side is the only forward-positive (needs sign-off). Plot: runs/strat/plots/.
