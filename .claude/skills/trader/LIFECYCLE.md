# Sleeve Lifecycle Protocol

> Every deployed sleeve traverses 5 stages. Each stage has explicit promote / demote / retire gates. Stage transitions are **the most failure-prone moment** in this project — H18 went paper -> ship in one step (commit fd0a870 / 4ba027b) without an explicit small-size buffer stage. This protocol closes that hole.
>
> **CDAP enforcement**: every sleeve carries a `lifecycle.yaml` at `config/sleeves/<sleeve_id>/lifecycle.yaml` with current_stage + stage_entered_at + min_time_in_stage + min_n_observations. Transitions write to `runs/lifecycle/<sleeve_id>_<from>_to_<to>_<utc>.json` with the gate evidence. The deploy_claim from `PRE_DEPLOY_CHECKLIST.md` is the transition record.

## Stage diagram

```
[1] INCUBATION    --(IC>0.015, p05>0, 10/10 seeds)-->  [2] PAPER
       ^                                                   |
       |                                                   |
       +<--------- demote on any item failure -----+       v
                                                   +--(claim_contract v1.2, 30+ paper days, decay_monitor green)-->  [3] LIVE_SMALL
                                                                                                                          |
                                                                                                                          |
                                                   +<--------- demote on DD>10% or IC decay -----+                        v
                                                                                                                    +--(60 live days, IC stable, capacity headroom verified)--> [4] LIVE_SCALE
                                                                                                                                                                                    |
                                                                                                                                                                                    |
                                                                                                                    +<--------- demote on DD>15% or 30d underperform vs paper -----+    v
                                                                                                                                                                                       +--(decay, regime shift, or capacity ceiling hit)--> [5] RETIRED
```

---

## Stage 1: INCUBATION

**What it is**: backtest-only exploration on TRAIN/VAL/OOS. UNSEEN reserved.

**Time-in-stage minimum**: none. Time-in-stage maximum: 2 weeks of active exploration before a refute-or-promote decision.

**Promote to PAPER requires** (ALL must hold):
- Backtest passes 4-window positivity (TRAIN/VAL/OOS/UNSEEN — `all_4_positive == True`).
- Block-bootstrap p05 > 0 on UNSEEN.
- DSR > 0.95 AND CSCV PBO < 0.50 if sweep > 20 configs.
- 10/10 seeds positive on UNSEEN (for ML components).
- `claim_contract v1.2` strict_gate passes.
- All 16 items of `PRE_DEPLOY_CHECKLIST.md` walked; items 1-4, 6, 9-11 VERIFIED.

**Demote triggers**:
- Any look-ahead or stride-staleness discovered.
- Mechanism falsifier check rejects the stated mechanism.
- p05 turns negative on a re-run with different seeds.

**Failure-catalog pattern**: 49 of 324 PEPE MA/EMA tuples positive in TRAIN/VAL/OOS/UNSEEN — only those 49 may enter Stage 2. The 275 others stay in INCUBATION as REFUTED entries in `WEALTH_BOT_FAILURE_CATALOG.md`.

---

## Stage 2: PAPER

**What it is**: simulated live trading on REAL-TIME bars, fixed cost model, no real capital. Writes a `runs/paper_trade/<sleeve>_signals.jsonl` and decay status JSON every bar.

**Time-in-stage minimum**: 30 calendar days OR 20 trades, whichever is greater.
**Time-in-stage maximum**: 90 days (after which sleeve must promote or be re-evaluated for INCUBATION return).

**Promote to LIVE_SMALL requires**:
- Paper-trade IC remains within 1 standard error of the backtest IC.
- No `lifecycle.decay_status == TRIPPED` events during paper window.
- Bar-level fills match expected p_fill range [0.25, 0.50] (when applicable).
- Decay monitor confirms IC half-life > 200 bars.
- Deploy claim emits with `stage_transition: "paper -> live_small"` and passes CDAP.

**Demote to INCUBATION triggers**:
- Paper-trade compound diverges by > 50% from backtest expectation.
- IC drops > 30% relative to backtest IC.
- Mechanism behavior in paper contradicts the falsifier check.

**Stop-and-fix triggers** (don't demote, don't promote — pause):
- Exchange API issue / data feed gap.
- Sleeve code edited (re-enter paper start).

**Current example**: H18 is in PAPER (commit fd0a870 2026-05-27). 4 phases REFUTED so far — the paper trade is doing its job by finding what backtest missed.

---

## Stage 3: LIVE_SMALL

**What it is**: real capital at $100-1000 per trade (tier-1 assets) or $20-200 (tier-3 assets). 1/8-Kelly to 1/16-Kelly sizing (defensive).

**Time-in-stage minimum**: 60 calendar days AND 50 round-trip trades.
**Time-in-stage maximum**: 6 months (then promote-or-retire decision).

**Promote to LIVE_SCALE requires**:
- Live PnL distribution KS-test against paper-trade distribution: `p > 0.10` (consistent).
- Live IC within 1 standard error of paper IC.
- Capacity headroom: tested fills at 2x current notional show no degradation.
- No DD > 10% during the window.
- 60+ days of clean operations (no API outage, no exchange-side fault, no manual intervention).

**Demote to PAPER triggers**:
- Live DD > 10% (drawdown is doing its job).
- Live IC drops > 50% from paper-IC.
- Mechanism stops working (e.g., regime shift, signal decay).
- 30 days underperforming paper expectation.

**Retire to STAGE 5 triggers**:
- DD > 20% (catastrophic, not recoverable at current sizing).
- Exchange-side trust violation (counterparty risk realized).
- Sleeve cannot be re-validated after a code change (treat as new sleeve).

---

## Stage 4: LIVE_SCALE

**What it is**: capital sized per `RiskController.size_trade()` recommendation, vol-targeted to portfolio risk budget. May reach 1/4-Kelly to 1/2-Kelly (still defensive; never full-Kelly).

**Promote conditions to add notional**:
- 90+ days at current notional with clean ops.
- Capacity test (place 2x next-step orders, measure slippage).
- Cross-sleeve correlation check: HRP doesn't drop sleeve weight below 0.05.

**Demote to LIVE_SMALL triggers**:
- DD > 12% from any peak.
- 30 days underperforming LIVE_SMALL expectation.
- Capacity ceiling hit (slippage > 2x cost model).
- Regime shift detected in REGIME_ROUTER.

**Hard size cap**: per North Star LO+SPOT+LEV=1 — sleeve notional cannot exceed `portfolio_equity * sleeve_weight_in_HRP`. There is no upgrade path beyond LEV=1.

---

## Stage 5: RETIRED

**What it is**: sleeve has either decayed past usefulness or hit operational/risk limit. Stops trading. Audit JSON archives state.

**Reasons to retire**:
- Permanent regime shift (e.g., feature is no longer predictive).
- DD > 20% in LIVE_SMALL or > 15% in LIVE_SCALE.
- Counterparty risk realized.
- Sleeve replaced by a strictly-better successor.

**Cannot un-retire**. A reincarnated sleeve enters Stage 1 (INCUBATION) as a NEW sleeve and walks the full path.

**Retire artifact**: `WEALTH_BOT_FAILURE_CATALOG.md` entry + `runs/retired/<sleeve_id>_retirement.json` with final stats, cause, and "do not retry without [list of changes]".

---

## Stage transition record (canonical fields)

Every transition writes `runs/lifecycle/<sleeve_id>_<from>_to_<to>_<utc>.json`:

```jsonc
{
  "sleeve_id": "h18_v2",
  "from_stage": "paper",
  "to_stage": "live_small",
  "transition_utc": "2026-06-15T14:00:00Z",
  "min_time_in_prev_stage_met": true,
  "prev_stage_days": 31,
  "prev_stage_trades": 24,
  "gate_evidence": {
    "paper_vs_backtest_compound_divergence_pct": 12.3,
    "paper_IC_vs_backtest_IC_se_ratio": 0.4,
    "decay_status": "GREEN",
    "deploy_claim_json": "runs/deploy/h18_v2/deploy_claim.json"
  },
  "authorizer": "trader-skill + auditor-skill consensus",
  "demote_triggers_for_next_stage": ["DD>10%", "IC_drop>50%", "30d_underperform_paper"]
}
```

## CDAP wiring

Added to `config/_invariants.yaml`:

| Rule | Severity | What it checks |
|---|---|---|
| `trader_lifecycle_min_time_in_stage` | critical | Stage transition JSON declares prev_stage_days >= stage minimum |
| `trader_lifecycle_min_n_trades_paper_to_live` | critical | >= 20 paper trades before paper -> live_small |
| `trader_lifecycle_stage_yaml_present` | critical | Every `config/sleeves/<id>/` has a `lifecycle.yaml` with current_stage |
| `trader_lifecycle_transition_record_emitted` | warn | Recent stage transitions have a record JSON |

## Lifecycle YAML (one per sleeve)

```yaml
# config/sleeves/h18_v2/lifecycle.yaml
sleeve_id: h18_v2
current_stage: paper
stage_entered_at_utc: "2026-05-27T22:30:00Z"
min_time_in_stage_days: 30
min_n_observations: 20
max_time_in_stage_days: 90
authorized_demote_target: incubation
authorized_promote_target: live_small
last_transition_record: runs/lifecycle/h18_v2_incubation_to_paper_2026_05_27.json
```
