# CLAUDE_CURRENT — Claude's live state (Claude-owned; Kiro does NOT touch). Synced 2026-06-28.

> Project-spec live state = Kiro's `CURRENT.md` (re-read for Kiro's latest; it updates as Kiro builds). THIS file =
> my synced view + my operating-model state. Co-existing: Kiro builds the project-spec; I review/verify/build + own
> the anti-fabrication discipline. Shared truth (dead-list, registry, docs/engines) co-maintained.

PHASE: SYNCED + caught up on the Kiro-era (V2-V15). Awaiting the user's next build direction.
UPDATED: 2026-06-28

## THE DEPLOYABLE VERDICT (both agents converge here)
**Capital-preservation within long-only = the deployable answer.** The MA band engine (V1 1d-BTC-SMA100 regime gate
+ V2 4h decoupled MA band, 7d lookback) is drawdown-insurance: bear-2022 ≈ flat (94% cash), ~35% deployed, Sharpe
~1.5-2. **Deployable rate (prod cost, 15% per-name cap, OOS-2024): ~+1.4-1.6%/wk** (SMA|1d +1.67 Sh1.57 DD-26%,
EMA|1d +1.68, SMA|4h +1.60). The per-name cap is the return DIAL: 15%→+1.4, 33%→+1.6, 50%→+2.1, EW(no cap)→+3.4 raw.
**DO NOT headline the +171K..+691K% DEV compounds** — ~9.5x rebalancing-premium + bull-window inflated (Kiro's own
A5 audit; OOS-2024 premium only ~1.1x). Use the capped OOS %/wk.

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

## OPEN / NEXT (the genuinely-untested or user-gated)
1. **The long-only ORACLE + ceiling gap** (the resourceful default, my offer) — best-achievable cash-vs-deploy
   timing vs the realizable engine → quantifies whether any headroom exists WITHIN the rules. Highest-value analysis.
2. **Crypto-native frontier** (funding/OI/on-chain) — the catalog's flagged data-advantage surface; USER-GATED.
3. **Phase 2 dynamic** (Kiro-designed, user-gated) — expect mostly null (more params); run as falsification, the
   adaptation-null + static-twin A/B are make-or-break.
4. Apparatus checks I offered: (a) HTF-alignment leak-check on the V1→V2 gate (the exact bug I found in V1.1 — does
   the 4h engine read the CLOSED 1d regime bar?); (b) demote the compound/cap headline in Kiro's CURRENT + registry.

## COORDINATION + HOUSEKEEPING STATE
- CHARTER/CURRENT/KNOWLEDGE_MAP = Kiro-owned. CLAUDE_CHARTER/CLAUDE_CURRENT = mine. Router injects mine for Claude
  sessions (+ Kiro's CURRENT pointer). Give-up gate untouched (reads Kiro's CHARTER constraints — fine; never loosen).
- Dead-list synced to D82+Kiro nulls (D83-D85). Auto-memory has a Kiro-era shard. Knowledge map fresh (Kiro, 06-26).
