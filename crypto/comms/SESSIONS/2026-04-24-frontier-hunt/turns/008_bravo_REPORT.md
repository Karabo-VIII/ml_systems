---
session_id: 2026-04-24-frontier-hunt
turn: 8
from: Bravo
to: Alpha
parent_turn: 7
sub_protocol: REPORT
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/2a5c1f59-e323-48e9-9e9e-7d4170a43eaa.jsonl"
reply_marker: "2026-04-24T19:30:00Z"
artifacts_touched:
  - scripts/bravo_turn008_multiprobe.py                                     # NEW
  - logs/frontier/orthogonality_ship/turn008_multiprobe.log                 # NEW (gitignored)
  - comms/SESSIONS/2026-04-24-frontier-hunt/session.yaml
  - comms/SESSIONS/2026-04-24-frontier-hunt/turns/008_bravo_REPORT.md
verifications_run:
  - "v2.1 side-channel: read Alpha JSONL slice 10:11Z-12:58Z. 2 human directives found: (1) 10:17:13Z codifies v2.1 rule itself, (2) 10:49:27Z authorizes 'build in parallel, validate/add-on, pivot fast'"
  - "independent recompute of A11 funding-gate: -7.54pp CAGR, -0.12 Sharpe. Matches Alpha claim -7.6/-0.12 within rounding. CONFIRMED."
  - "4 probes run: V1 verify A11, P1 R2 rescue per-sleeve, P2 30-min BTC.D, P3 orthogonality regression"
  - "P3 multivariate R^2 = 0.0041 on blend daily return regressed on [stable_z, etf_z, fund_z, btc_d_chg_30d, cycle_regime_OH, btcd_regime_OH]. Forward-1d R^2 = 0.0112. Both << 0.05 threshold."
human_directives_received:
  - "via Alpha side-channel turn 007: 'build in parallel, partner validates/adds-on, pivot fast' -- this turn honors that by running add-on validations rather than duplicating work"
external_context_seen:
  - "none beyond session scope"
expects_next: |
  Alpha to REVIEW orthogonality-ship + PROPOSE Tier-2 additive-sleeve pivot.
  User-answer to Q-user-3 (Simple Earn deploy), Q-user-4 (airdrop commit),
  Q-user-5 (accept orthogonality + pivot) may land in either JSONL side-channel.
  Both agents must read.

  Bravo's recommendation for Tier 2 starting order (additive, CEX-native):
    1. Launchpool operationalization (p6 already partially built, 2-3d)
    2. Pre-listing front-run scraper MVP (2w)
    3. Insider-unlock MR event study (1-2w, token-unlock calendar needed)
    4. Futures-DATA as FEATURE (not sizing gate) -- retrain xsec with funding/OI/liq panel as inputs

  Sub-protocol for turn 009: REVIEW + PROPOSE (Alpha) or ANSWER (user inputs)
---

## Summary

**REPORT + validation**: 4 probes ran in one pass using Alpha's pre-built
artifacts. Canonical orthogonality finding CONFIRMED and SHIPPED.

**Key results** (full detail in Bravo's JSONL around reply_marker 2026-04-24T19:30Z):
1. **V1 (verify Alpha A11)**: -7.54pp CAGR, -0.12 Sharpe. CONFIRMED within rounding of Alpha claim -7.6/-0.12.
2. **P1 (R2 per-sleeve sizing)**: ONE near-win: all-sleeves-by-etf_z alpha=0.25 adds +4.8pp CAGR +0.05 Sharpe but DD widens 24%. Under no-leverage, mult>1.0 is accounting-ambiguous. Not worth shipping alone.
3. **P2 (30-min BTC.D probe)**: Sleeve-level regime structure is real (asym_vol_exp LOSES money in SIDEWAYS) but blend-level gates are weak. Best no-leverage variant (favor ALT_SEASON 1.25) trades -8pp CAGR for +0.28 Sharpe.
4. **P3 (orthogonality regression)**: Multivariate R^2=0.0041, forward-1d R^2=0.0112. Both 4-12x below 0.05 threshold. **CANONICAL FINDING SHIPPED: blend_is_regime_orthogonal.**

**Recommendation on Q-user-3/4/5**:
- Q3 (Simple Earn): YES with $500 test first, then scale.
- Q4 (airdrop commit): CONDITIONAL on 10X aspiration. 3X/yr CEX ceiling without it.
- Q5 (accept orthogonality + pivot): YES -- formally confirmed.

**Proposed Tier-2 pivot order** (CEX-native additive sleeves, respecting infra
and trading constraints):
  1. Launchpool operationalization (fastest path)
  2. Pre-listing front-run scraper MVP
  3. Insider-unlock MR event study
  4. Futures-DATA as feature input to xsec ranker retrain (A11 reframed)

**Artifacts shipped**: scripts/bravo_turn008_multiprobe.py (4 probes in one
script), logs/frontier/orthogonality_ship/turn008_multiprobe.log.

**Protocol v2.1 compliance**: side-channel read (2 human directives
registered), human summary in main reply, paranoid cross-verification of
Alpha's A11 claim completed.
