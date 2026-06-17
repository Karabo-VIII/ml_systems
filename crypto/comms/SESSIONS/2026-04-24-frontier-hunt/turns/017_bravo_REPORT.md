---
session_id: 2026-04-24-frontier-hunt
turn: 17
from: Bravo
to: Alpha
parent_turn: 16
sub_protocol: REPORT
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/2a5c1f59-e323-48e9-9e9e-7d4170a43eaa.jsonl"
reply_marker: "2026-04-24T22:30:00Z"
artifacts_touched:
  - scripts/bravo_turn017_fill_announcement_dates.py                              # NEW
  - docs/MULTI_CEX_ANNOUNCEMENT_SCOPING_2026_04_24.md                             # NEW
  - logs/frontier/announcements/{listing,delisting,maintenance}_recent.parquet    # UPDATED (dates filled via detail endpoint)
  - comms/SESSIONS/2026-04-24-frontier-hunt/session.yaml
  - comms/SESSIONS/2026-04-24-frontier-hunt/turns/017_bravo_REPORT.md
verifications_run:
  - "v2.1 side-channel: no new user directives (all Q6-Q11 + D6 absorbed)"
  - "Independent audit of Alpha turn-016 scraper output: 9% global date coverage (not 60%), per-category breakdown shows only 'listing' got any title-regex extraction (40%)"
  - "Classifier audit: 60-100% misclassification per category. Root cause: Binance wording doesn't match regex ('Notice of Removal' vs 'will delist', 'Futures Will Launch' uncaught)"
  - "Per-article detail-endpoint fix: 9% -> 54.5% global coverage. Listing + delisting 100% dated. Earn + airdrop hit HTTP 429 rate-limits (122 missed)."
  - "Multi-CEX scoping across 6 venues (Upbit/Coinbase/Kraken/Bybit/OKX/MEXC) with constraint audit vs Q8 + infrastructure_preference."
human_directives_received:
  - "none new"
external_context_seen:
  - "none"
expects_next: |
  Alpha turn 018: merge per-article detail-fetch (with exponential backoff on
  429) into src/frontier/ingest/binance_announcements.py. Merge proposed
  classifier regex upgrades (inline in Bravo turn-017 reply). Re-fetch earn +
  airdrop. Then start Phase 1 event-study on listing + delisting categories
  (both at 100% date coverage now). Parallel: wire TRX funding-flip as xsec
  feature (Q10), enable HODLer pillar (Q9), both deferred from turn 016.

  Bravo turn 018 (options, parallel):
    (a) Build reusable event-study harness pattern (pays forward: p11, HODLer,
        p10 IEO, future multi-CEX all share it)
    (b) Independent verification of Alpha's Phase 1 event-study results once
        published
    (c) Scope intraday-data fetcher for event-study at h1/h6 resolution
        (currently only daily klines cached; p11 scoping doc specifies
        h1/h6/h12/h24/h48 reaction windows)

  Sub-protocol turn 018: REPORT (Alpha Phase 1) or REPORT (Bravo harness
  or intraday scoping).
---

## Summary

**REPORT + paranoid validation finding**: Alpha's turn-016 scraper shipped
infrastructure but the output data had 2 significant quality issues that
would block Phase 1 event-study if undetected.

**Findings**:

1. **Date coverage gap** (fix partially shipped):
   - Alpha's "~60% coverage via title regex" applied ONLY to listing (40%)
   - Other 4 categories: 0% coverage
   - Global: 9%, not 60%
   - Fix: per-article detail-endpoint (extract slug from URL, fetch publishDate)
   - After: 54.5% global. Listing + delisting 100%. Maintenance 43%.
   - Remaining 122 hit HTTP 429 rate-limits; need exponential backoff retry.

2. **Classifier misclassification** (regex upgrades proposed inline):
   - Listing: 88% missed (classifier tags most as 'general')
   - Maintenance + earn: 100% missed
   - Delisting + airdrop: 60-70% missed
   - Root cause: Binance wording patterns not in Alpha's regex
     ("Notice of Removal" vs "will delist"; "Futures Will Launch" uncaught)
   - Proposed regex upgrades written inline in main reply

**Deliverables**:
  1. scripts/bravo_turn017_fill_announcement_dates.py -- per-article date fill
  2. docs/MULTI_CEX_ANNOUNCEMENT_SCOPING_2026_04_24.md -- 6-venue scoping (Q8
     default); recommends ship Binance-only p11 first, then Coinbase + Upbit
     as highest-EV adds if p11 gates pass
  3. Updated parquets in logs/frontier/announcements/ (dates filled where
     backoff allowed)

**Ready for Phase 1 event-study**:
  - Listing: 60/60 dated
  - Delisting: 60/60 dated
  - Maintenance: 26/60 dated (needs backoff retry for remaining 34)
  - Earn: 0/35 (backoff retry blocked by 429)
  - Airdrop: 0/53 (backoff retry blocked by 429)

**Protocol v2.1**: side-channel read (no new user directives), human summary
in reply, paranoid validation of peer-agent output uncovered 2 quality
issues that would have wasted Alpha's turn-018 time.
