---
session_id: 2026-04-24-frontier-hunt
turn: 6
from: Bravo
to: Alpha
parent_turn: 5
sub_protocol: REPORT
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/2a5c1f59-e323-48e9-9e9e-7d4170a43eaa.jsonl"
reply_marker: "2026-04-24T18:30:00Z"
artifacts_touched:
  - scripts/bravo_supply_flow_meta_probe.py            # NEW
  - logs/frontier/supply_flow_meta/turn006_mvp_probe.log  # NEW
  - comms/SESSIONS/2026-04-24-frontier-hunt/session.yaml
  - comms/SESSIONS/2026-04-24-frontier-hunt/turns/006_bravo_REPORT.md
verifications_run:
  - "read PROTOCOL.md v2.1 (new side-channel rule + human-summary rule)"
  - "read turn marker 005 v2 + Alpha JSONL slice 09:34Z-10:15Z (2 human side-channel msgs captured: infra preference + trading constraints + airdrop automation gate)"
  - "independent re-verification of 4-sleeve blend: CAGR 74.20%, Sharpe 6.16 @ 365-day ann, Sortino 18.91, DD -2.01% -- matches summary JSON within rounding"
  - "discovered + documented in-code: portfolio_ret_pct column in blend CSV is CUMULATIVE from inception, NOT daily. Daily returns correctly re-derived from portfolio_equity.pct_change()"
  - "ran supply-flow meta-multiplier MVP probe: 7 variants (4 discrete + 3 continuous). ALL underperform flat on Sharpe."
  - "computed signal-to-return correlations: all |corr| < 0.10, mixed signs. stable_z anti-correlates at 1d/3d."
human_directives_received:
  - "via side-channel from Alpha JSONL (not direct to Bravo): human summaries in every turn, infra = minimize non-CEX-Binance, trading = spot-only/no-lev/no-futures-directional/futures-data-exploitable, airdrops OPEN if automated"
external_context_seen:
  - "no third-party JSONLs active"
expects_next: |
  Alpha reports in turn 007: (a) cycle-gate historical replay outcome,
  (b) yield-idle infra audit (project on-chain infra present? -> Simple Earn
  vs Pendle decision). Bravo executes in parallel: (a) R1 DIB-rescue probe
  (2-3h) on the failed supply-flow meta-multiplier, (b) 30-min BTC.D sizing
  probe. If both rescues fail, supply-flow meta CONCEDES and weekly-xsec
  probe elevates to Tier 1.

  Sub-protocol for turn 007: REPORT (Alpha on cycle-gate + yield-idle).
  Turn 008: REPORT from Bravo on R1 + BTC.D.
---

## Summary

**REPORT**: Supply-flow meta-multiplier MVP (2-signal: stable + ETF) FAILED.

**Findings** (full content in Bravo's JSONL around reply_marker 2026-04-24T18:30:00Z):
1. Baseline re-verified: CAGR 74.20%, Sharpe 6.16 @ 365d annualization (matches summary 6.17)
2. ALL 7 sizing variants (4 discrete + 3 continuous) UNDERPERFORM flat on Sharpe. Best continuous adds +3% CAGR but widens DD 53% -- pure leverage-in-disguise, no alpha lift.
3. Per-regime dispersion is U-shaped, not monotone. risk_on=0 days have HIGHEST hit-rate (58.5%) and 2nd-highest mean -- the sizing premise is violated.
4. Signal-to-forward-return correlations tiny (all |corr| < 0.10) and mixed sign. stable_z anti-correlates to blend at 1d/3d horizons.

**Interpretation**: 4-sleeve blend is already regime-orthogonal across these 2 signals. This is actually a positive finding about blend construction -- the 74% CAGR is spread across macro regimes, not concentrated in "risk-on" periods.

**4 rescue paths defined**, ranked by probability of lift:
- R1 (35%): Add DIB signal -> 3-signal stack, 2-3h cost -- NEXT STEP
- R2 (40%): Per-sleeve conditional sizing -- needs per-sleeve daily returns, 3-4h
- R3 (25%): Kill-switch only in extreme adverse regime, 1h
- R4 (25%): Signal-of-signal momentum, 2-3h

**Turn-006 artifacts**: scripts/bravo_supply_flow_meta_probe.py, logs/frontier/supply_flow_meta/turn006_mvp_probe.log

**Status change**: Supply-flow meta-multiplier (Tier-1 item #1) is WIP-FAILED at MVP, awaiting R1 rescue. If R1 also fails -> CONCEDE + elevate weekly-xsec probe or BTC.D to replace it.

**Protocol compliance**: v2.1 side-channel rule applied (read Alpha's JSONL slice for 2 human directives; registered in this turn marker). Human-summary-in-every-turn applied in main reply.
