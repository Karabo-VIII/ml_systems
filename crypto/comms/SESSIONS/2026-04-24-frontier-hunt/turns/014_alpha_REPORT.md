---
session_id: 2026-04-24-frontier-hunt
turn: 14
from: Alpha
to: Bravo
parent_turn: 13
sub_protocol: REPORT + ANSWER_ACK
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/ad6bf44a-83df-464a-a6c8-f4f453ea23ed.jsonl"
reply_marker: "2026-04-24T15:15:00Z"
artifacts_touched:
  - scripts/alpha_subday_dipbuy_U50.py                        # NEW
  - scripts/alpha_subday_3d_rotation_U50.py                   # NEW
  - logs/frontier/subday_dipbuy/dipbuy_3to7d_U50.json         # NEW
  - logs/frontier/subday_3d_mom/mom3d_top5_U50.json           # NEW
  - comms/SESSIONS/2026-04-24-frontier-hunt/session.yaml
verifications_run:
  - "user answered Q6-Q11: defaults accepted on all 6"
  - "user added D6 directive: shift focus or add dimension of sub-day strategies (small bursts 1-3%+, AND/or holding < 3-7d)"
  - "2 D6-compliant probes executed at U50 balanced per strat_test_min_universe rule, each with TRAIN/VAL/OOS chronological split"
human_directives_received:
  - "defaults Q6-Q11 all accepted (30d max, Track A first, multi-CEX scrapers, HODLer yes / Megadrop no, TRX as xsec feature, p11 Announcement-Volatility as next build)"
  - "D6: 'shift focus or add the dimension of sub-day strategies: in holding time (small bursts capture: 1 - 3%+), AND/or something with holding period (still on smaller timer frames) < 3/7 days'"
external_context_seen:
  - "none beyond session scope"
expects_next: |
  Bravo turn 015:
    (a) Formalize orthogonality finding as docs/ORTHOGONALITY_FINDING_2026_04_24.md
        (now 6 independent evidences: prior 5 + turn-014 3d-mom OOS-kill)
    (b) Scoping doc for p11 Announcement-Volatility (Q11 default=yes)
    (c) Consider probing cross-asset U50 breakout-density as one more D6 angle
        (novel: uses U50 synchronously rather than per-asset; untested)

  Alpha turn 015:
    (a) p10 IEO live-mode robustify
    (b) Scope HODLer pillar (Q9=default yes, needs D1-exception operationalization)
    (c) TRX funding-flip as xsec feature (Q10=default, schedule for next xsec retrain)
---

## Summary

**ANSWER_ACK + REPORT** on D6 sub-day exploration.

### User answers locked in (Q6-Q11 defaults)

- Q6: 30d max hold cap, explicit exceptions ask
- Q7: Track A first, Track B decision at 1-2w mark
- Q8: Multi-CEX scrapers OK (execution stays Binance)
- Q9: HODLer yes (10-14d brief window counts as short-term), Megadrop no
- Q10: TRX funding-flip -> xsec feature input (not dedicated sleeve)
- Q11: p11 Announcement-Volatility approved as next build after p10 IEO robust

### D6 sub-day probes (2 executed this turn)

**Probe 1 -- Dip-buy 3-7d hold (U50, paranoid OOS split)**:
- Pullback: 7d ret < -15%, trend: 30d ret > 0
- Target +5%, stop -7.5%, 7d time stop
- **Result: 1/50 OOS survivor (LDO n=11 t=2.86 hit 82%)** -- narrow, not shippable as sleeve
- Notable inversions OOS: FIL t=-4.78 hit 8%, ICP t=-3.63 hit 21%, ZEC t=-3.29 hit 20%

**Probe 2 -- 3-day momentum rotation (U50 top-5 basket)**:
- Rank U50 by 3d return daily, buy top-5 EW, hold 3d, rebalance
- **Result: period-concentrated artifact, OOS-dead**
- TRAIN 2020-23: n=485 mean +0.89% t=+2.17 cagr +195%
- VAL 2024: n=122 mean +1.87% t=+1.94 cagr +854% (!)
- **OOS 2025-26: n=159 mean -0.02% t=-0.03 hit 47% cagr -2.5%** -- dead
- Identical failure pattern to A7: 2020-24 works, 2025-26 regime dies

### Pattern emerging across D6 + prior sub-day probes

6 sub-day concedes now:
  1. Scalp-squeeze breakout (BTC) -- dead
  2. Scalp vol-spike U50 -- 0/50 candidates
  3. Scalp flow-imbalance U50 -- 0/47 candidates
  4. A7 liq-cascade 3-7d -- paranoid OOS 0/10
  5. Dip-buy 3-7d -- 1/50 narrow (unshippable)
  6. 3-day momentum rotation -- OOS-dead

**Unified interpretation**: 2025-2026 regime is actively hostile to bar-based
short-term alpha. Momentum and mean-reversion both fail at sub-daily AND
3-7d horizons. Event-triggered (P8 listing h1, p10 IEO) remains the only
viable sub-day path in OUR cost model.

This is a REGIME observation, not a permanent death. If 2025-2026
compression ends (big bull or bear), sub-day bar-based alpha may revive.
Cycle-gate monitor (shipped turn 010) will flag regime transitions.

### What this means for D6

Honest report: D6 cannot be fully honored with NEW sub-day sleeves right
now without either:
  (a) A truly novel trigger framework (e.g., cross-asset sync breakout
      density; Bravo may probe next turn)
  (b) Different feature class (e.g., social sentiment, on-chain velocity;
      new infra required under D2's "faster" not sufficient)
  (c) Event-triggered expansion (p10 IEO launch + p11 Announcement-Vol
      build -- both in queue)

Queued for turn 015:
  - p10 IEO live-mode robustification (holding 48-72h, D6-compliant)
  - p11 Announcement-Volatility build (1h-48h event-triggered, D6-compliant)
  - HODLer pillar enablement (Q9 default yes)
  - TRX funding-flip as xsec feature (Q10)

## Human Summary

**All 6 user defaults accepted and locked in**. Q6-Q11 answers written to
turn marker.

**D6 (sub-day dimension) executed -- 2 probes both fail paranoid OOS**:

1. Dip-buy (down 15% in 7d, uptrend 30d, hold 7d) across U50 -> only LDO
   ships OOS with n=11. Too narrow to be a sleeve.
2. Simple 3-day momentum top-5 rotation U50 -> CAGR 195%/854%/−2.5% across
   TRAIN/VAL/OOS. Works 2020-2024, **dies 2025-2026**. Same regime-death
   pattern as A7 that Bravo killed.

**6 sub-day concedes now across the full session** (3 scalp triggers + A7
+ dip-buy + 3d momentum). All bar-based short-term alpha fails paranoid
OOS in 2025-2026 regime. **Only event-triggered sub-day works** right now
(P8 listing h1 live, p10 IEO ready, p11 Announcement-Vol next).

**Honest framing on D6**: the sub-day dimension is genuinely hard in the
current regime. I can't manufacture a bar-based sub-day sleeve that passes
paranoid validation. What I CAN do:
  - Ship p10 IEO live-mode (event-triggered, 48-72h holds, ~10-20 IEOs/yr)
  - Build p11 Announcement-Volatility (event-triggered, 1-48h holds)
  - Add future-DATA features to xsec (TRX funding-flip already in Q10)
  - Cycle-gate monitor will signal when regime unlocks bar-based sub-day

If D6 is load-bearing for a specific growth target, the realistic path is
stacking 3-4 event-triggered engines (P8 + p10 + p11 + potentially more)
vs trying to force a bar-pattern sleeve that 6 probes say is regime-dead.

Next turn: p10 IEO robust + HODLer scaffold enable + start p11 scoping.
