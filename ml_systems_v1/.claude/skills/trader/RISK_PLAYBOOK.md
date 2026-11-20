# Risk Playbook (drawdown + regime + decision-asymmetry response)

> **Purpose**: when a sleeve hits a drawdown threshold OR a regime indicator flips OR a sizing decision is asymmetric, the trader skill must know what to DO — not just what to detect. Numbers in CLAUDE.md set thresholds; this playbook sets responses.
>
> **Authority**: project empirics first. Every threshold has provenance from a refuted hypothesis or a working safeguard already in code.

## Decision-asymmetry first principle

A trader's job is not to maximize return — it is to **minimize the variance of catastrophic outcomes** while keeping expected compound positive. The asymmetry:

- **Sizing UP** (adding notional) is bounded by Kelly + capacity + correlation. Mistakes here cost 25-50% growth (defensible).
- **Sizing DOWN** (removing notional) is bounded by nothing. Mistakes here cost everything (un-defensible).

Rule: **size down faster than you size up**. Default sizing change is asymmetric: 50% down in one step on a trigger, 10-25% up per stage transition.

Provenance: H23 trail-stop refutation (-35.5%) showed that no working downside-protection layer exists beyond max-hold. Until one exists, asymmetric sizing IS the downside-protection layer.

---

## Drawdown response decision tree

Drawdown = peak-to-trough on sleeve equity. Measured per-sleeve AND portfolio-level. Both must be checked every bar by `RiskController`.

### -3% sleeve DD (yellow)

- No action.
- Log to `crypto/runs/risk/<sleeve_id>_dd_log.jsonl`.
- Continue normal sizing.

### -5% sleeve DD (orange)

- Verify decay_monitor status. If GREEN, continue.
- Verify mechanism still firing as expected: spot-check last 5 trades against signal output.
- If both clean: continue. If either anomalous: drop to -10% protocol.

### -10% sleeve DD (red)

- **Halve sizing** in next trade. 1/8-Kelly -> 1/16-Kelly. Vol-target -> 0.5x vol-target.
- Pause new sleeve additions (do not start a new sleeve while existing one is in red).
- Run `decay_monitor` analysis: was the DD foreshadowed by IC decay?
- Wait minimum 20 trades OR 30 days at reduced size before re-evaluating.
- **Do NOT add a trail-stop on H6-class entries** — H23 refutation showed this destroys edge without protecting downside.

### -15% sleeve DD (critical)

- **Pause sleeve**. No new entries. Existing positions ride to natural exit OR max-hold.
- Trigger the audit skill on sleeve code + claim JSON.
- Walk every PRE_DEPLOY_CHECKLIST item — if any FAILED, demote to PAPER immediately.
- If all VERIFIED clean, the regime shifted: demote to PAPER for 30-day re-evaluation.

### -20% sleeve DD (catastrophic)

- **Retire sleeve immediately**. No path back to LIVE.
- Close all positions at market (accept worst execution).
- Trigger audit + decide on what failed.
- Write `crypto/runs/retired/<sleeve_id>_retirement.json` with cause analysis.
- A new sleeve with the same mechanism must enter Stage 1 (INCUBATION) under a new sleeve_id with explicit "what's different this time" documentation.

### Portfolio-level -10% DD

- Halve sizing on ALL sleeves (cross-sleeve halving, not per-sleeve).
- Trigger decide on whether portfolio-level correlation hit 1 (which Kelly + HRP underestimate).

### Portfolio-level -15% DD

- Halt all sleeves. No new trades.
- Wait 7 days, re-evaluate at portfolio level.
- Resume only with explicit user authorization.

---

## Regime-shift response

Regime detection signals (from `src/strategy/meta_allocator.py` + REGIME_ROUTER):

| Signal | What it means | Action |
|---|---|---|
| `IC_decay_60d > 30%` | Feature predictiveness fading | Halve sizing, paper-trade in parallel for re-validation |
| `corr_cross_sleeve_60d > 0.7` | Sleeves no longer independent | Down-weight all sleeves via HRP recalculation |
| `vol_regime_z > 2.5` | Vol spike | Halve sizing on all vol-sensitive sleeves; Donchian baseline unaffected |
| `funding_sign_flipped_7d` | PERP funding flip (if any perp sleeve live) | Halt PERP sleeves; SPOT unaffected |
| `liquidations_cascade_event` | Whale liquidation series | Pause all sleeves for 24h; do not chase |
| `stablecoin_depeg > 0.5%` | USDT/USDC depeg signal | Halt all sleeves; assess custody risk |
| `REGIME_ROUTER class change` | Trend / range / volatile flip | Re-evaluate sleeve gating; some sleeves may be regime-incompatible |

### When IC decays mid-life

Example: H23 trail-stop -35.5% — REFUTED. The expected mechanism (trail-stop catches downside on H6-class entries) didn't fire because PEPE catalyst plays have no downside protection within the hold window.

Response template:
1. Confirm the decay is signal (not noise) — block-bootstrap p05 on last 30 days vs baseline 90 days.
2. If p05 turns negative, demote one stage. If still p05 > 0, sample-size discipline check (n_eff at last 30 days >= 12?).
3. If sample-size insufficient, hold position size, gather more data.
4. If sample-size sufficient AND signal degraded, demote per LIFECYCLE.md triggers.

---

## Sizing decision asymmetry table

| Situation | Direction | Speed | Magnitude |
|---|---|---|---|
| New sleeve in INCUBATION -> PAPER | Up | One step, no graduation period | Match Kelly recommendation × 0.5 (fractional) |
| PAPER -> LIVE_SMALL | Up | One step, requires deploy claim | $100-1000 tier 1, $20-200 tier 3 |
| LIVE_SMALL -> LIVE_SCALE | Up | Stage gates, 60+ days | +25% notional max per stage |
| LIVE_SCALE adding notional | Up | Quarter-by-quarter | +10-25% per 90 days |
| Any sleeve hits -10% DD | Down | Same bar | -50% notional (halve) |
| Any sleeve hits -15% DD | Down | Same bar | Pause entirely (-100%) |
| Portfolio hits -10% DD | Down | Same bar | -50% across all sleeves |
| IC decay 30% in 60d | Down | Next bar | -50% notional |
| Vol regime z > 2.5 | Down | Same bar | -50% notional, vol-sensitive sleeves |
| Stablecoin depeg | Down | Same bar | Halt all |

**Symmetry violations are the trader-skill failure mode**: e.g. responding to a -10% DD with a -10% sizing change ("symmetric") is wrong. The DD is a stochastic outcome; the sizing change is a deterministic choice. Match the sizing to the *posterior on edge*, which has shifted downward by more than 10% after a DD.

---

## Fat-tail sizing override

When the asset is in the **fat-tail regime** (PEPE-class memecoins, post-listing assets, regulatory-event windows), Kelly's σ² assumption fails. σ² understates tail risk and Kelly oversizes.

Override rules:
- For PEPE-class: cap sizing at 1/16-Kelly regardless of vol-target. Empirically, this is the only safe regime per gold-standard dossier.
- For post-listing < 60 days: cap at 1/32-Kelly. New asset = new distribution.
- During regulatory event windows (announcements, exchange listings/delistings): halt sleeves on affected asset until 24h after event.

Provenance: gap T5 in trader-skill audit 2026-05-28. Kelly + HRP both assume well-behaved second moments — both fail in tail events.

---

## Kill switches (hard rules)

These trigger automatically via `RiskController`, bypassing any sleeve logic:

1. **Portfolio DD > 20%**: hard halt, manual intervention required.
2. **Single-trade loss > 5%**: pause sleeve for audit, do not auto-resume.
3. **Cross-sleeve correlation hits 0.9+ rolling 7d**: down-weight all sleeves to HRP minimum (0.05 each).
4. **Exchange API error rate > 5% in 1h**: pause all order entry, existing positions ride.
5. **Data feed gap > 5 minutes**: halt all sleeves until backfilled and reconciled.
6. **Mechanism falsifier check turns FALSE post-deploy**: halt sleeve, run the audit skill.

---

## What this playbook intentionally does NOT do

- It does not promise to avoid drawdowns. Drawdowns are how the system tells you it doesn't fully understand the market. The job is to size down faster than the drawdown grows.
- It does not specify exact target portfolio Sharpe. Per WEALTH-not-SHARPE mandate, compound is the ranking metric; Sharpe is tiebreak.
- It does not enable leverage > 1 under any circumstance. Hard North Star bound.

## CDAP wiring

| Rule | Severity | Checked file |
|---|---|---|
| `trader_risk_controller_has_kill_switches` | critical | `src/strategy/risk_controller.py` must implement all 6 kill switches |
| `trader_dd_thresholds_match_playbook` | warn | Risk controller thresholds match this playbook |
| `trader_fat_tail_assets_capped` | warn | Sleeve YAML for PEPE-class declares `max_kelly_fraction <= 0.0625` |

## Cross-references

- LIFECYCLE.md — stage transitions trigger sizing changes per this playbook.
- PRE_DEPLOY_CHECKLIST.md item 15 — every deploy claim cites a row from this playbook.
- SIZING_THEORY.md — when Kelly fails, which alternative to use.
- DAILY_OPS.md — daily live monitoring loop reads risk thresholds.
