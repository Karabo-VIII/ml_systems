# Pre-Deploy Checklist (canonical, CDAP-enforced)

> **Status**: BINDING for any commit that flips a sleeve from paper to live (or scales an already-live sleeve). Items 1-16 must each be VERIFIED before any real-capital exposure.
>
> **Authority**: Project empirics first. Every item below has provenance from a real failure already shipped in this codebase.
>
> **Enforcement**: items marked `[CDAP]` are checked by `crypto/src/audit/check_deploy_gates.py` against `crypto/config/_invariants.yaml::trader_deploy_gates`. Commit exits 2 on any CRITICAL violation. Items marked `[CLAIM]` are required fields in the deploy claim block emitted to `crypto/runs/deploy/<sleeve_id>/deploy_claim.json`.

## How to use

1. Copy the checklist below into the sleeve's deploy PR description.
2. Walk every item top-to-bottom. Fill VERIFIED / FAILED / N/A with one-line evidence (file:line, audit JSON path, run command + output).
3. Any FAILED on items 1-12 = HALT deploy. Items 13-16 may be deferred with explicit acceptance from a non-trader skill (auditor or oracle).
4. The completed checklist is the audit record — commit it alongside the deploy.

---

## Item 1 — Cost source is canonical [CDAP]

- Strategy reads SPOT_COST / PERP_COST from `src/strategy/cost_model.py`.
- No local re-definition of fee, slippage, or funding constants.
- Verify: `grep -n "0.0010\|0.0024\|fee_bps\s*=\|slippage_bps\s*=" <sleeve_file>` returns 0 hits.
- Provenance: pre-2026-04 several sleeves silently hardcoded fees, escaping cost-model recalibration (G-AUDIT-010).

## Item 2 — MtM reconciliation gate passes [CDAP]

- Simulator emits a probe block: `abs(sum(pnl_stream) - sum(trade_log.pnl)) / abs(sum(trade_log.pnl)) < 0.001`.
- Provenance: 2026-04-22 MtM double-count = 5-7x equity inflation. Pre-fix headline +501% became truth +94%. CLAUDE.md Backtest Simulator Invariants is binding.

## Item 3 — Look-ahead audit clean [CDAP]

- Feature columns referenced at decision time `t` are constructed only from data `<= t-1` (or `<= t` with explicit close-bar-execution acknowledgment).
- Targets are NEVER used to pick K, gate entries, or filter trades. Run `grep -rn "target_return\|future_return\|forward_return" <sleeve_file>` — any hit on a SELECTION path = HALT.
- Provenance: 2026-05-25 R51 E51_1 gap-down fallback miss + R54 A54_2 PSEUDO-VB forward-close leak. CLAUDE.md §UNIVERSAL PRE-DELIVERY SELF-AUDIT cites three gap-window incidents.

## Item 4 — Stride-1 prediction freshness [CDAP]

- WM predictions emitted via `generate_wm_predictions.py` with `stride=1` (verify in command line) OR via `gen_preds.py` post-2026-04-14 stride-1 fix.
- Provenance: Pattern N — pre-2026-04-14 all backtests stale up to 95 bars. Any sleeve using pre-fix predictions is invalid.

## Item 5 — p_fill realism budget [CLAIM]

- Deploy claim declares `p_fill_live_budget ∈ [0.25, 0.50]` and confirms sizing math used the lower bound.
- Expected live equity is documented as 50-75% of fixed-backtest equity.
- Provenance: empirical OHLC replay 2026-04-22 (`crypto/src/analysis/execution_sim.py`) — actual p_fill 0.21-0.40 across buckets. MakerCostModel default 0.80 is optimistic.

## Item 6 — Claim contract v1.2 passes [CDAP]

- Deploy claim emits `ship_claim` block via `build_ship_claim_block(...)` in `crypto/src/wealth_bot/framework/claim_contract.py`.
- All v1.2 required fields present, `passes_strict_gate == True`, `phase1_n_eff_gate.passes == True`.
- `python crypto/src/audit/check_wealth_bot_claims.py <claim_json>` exits 0.
- Provenance: 2026-05-25 INST-A P4_route_basis_pos_only ship-claimed with FALSE mechanism. Contract is the trust-stack layer.

## Item 7 — Mechanism falsifier verified [CLAIM]

- If `concentration_metrics.herfindahl_on_abs_contribution > 0.50` OR `top_3_pct_of_compound > 70%` at `n_unseen < 30`, `mechanism_falsifier_check.verified_by` MUST name an auditor + timestamp.
- The trade-level diff (what filter KEPT vs DROPPED) is attached. Hand-verify the claimed mechanism matches the data.
- Provenance: 2026-05-25 P4_route_basis_pos_only — filter kept top-3 and dropped diversifying, opposite of stated mechanism.

## Item 8 — Multi-seed robustness floor [CLAIM]

- For ML-component sleeves (PPO / LSTM / DQN / any randomized trainer): N >= 10 seeds, **all 10 positive on UNSEEN**, p05 block-bootstrap > 0, maxDD < 30%.
- Single-seed numbers are unverified. See `feedback_multi_seed_ml_audit_gate.md` — 2026-05-24 LSTM +44.6% / DQN +40.9% turned out to be init-luck.

## Item 9 — Jackknife K=2 + combined K2+S9 > 0 [CDAP]

- `jackknife.K=2 > 0` AND `combined_K2_plus_S9_pct >= sample_size_discipline.ship_threshold_compound_required`.
- Without these, top-K trade concentration is hiding the result. Enforced by claim_contract v1.2.

## Item 10 — Walk-forward purge gap [CDAP]

- Walk-forward harness uses `purge_gap_bars >= 400`.
- For dollar-bar sleeves on Tier 3 assets, document whether 400 is sufficient given the smaller bar volume.
- Provenance: G-AUDIT-002. CLAUDE.md cross-version invariants require purge gap presence.

## Item 11 — DSR + CSCV multi-test gate [CDAP]

- If the sleeve was selected from a sweep of N > 20 configurations, `deflated_sharpe.DSR_p_value > 0.95` AND `cscv_pbo_pct < 0.50`.
- Deploy claim cites the sweep size N and reports both numbers.
- Provenance: post-2026-06-04-reset, DSR/Holm is a **caller-supplied contract** in `crypto/src/strat/battery.py` (no standalone `deflated_sharpe.py`; the archived `src/strategy/` path is dead). Math per Bailey-Lopez de Prado 2014, Bailey-Borwein-Lopez de Prado 2017.

## Item 12 — Universe survivorship declared [CLAIM]

- Deploy claim declares the asset list and acknowledges that `master_top_assets.csv` carries delisting survivorship bias.
- For sleeves restricted to Tier 1/2 (BTC, ETH, SOL, BNB, XRP) bias is low; Tier 3 + memecoins requires explicit caveat.

## Item 13 — Capacity / scaling estimate

- Deploy claim includes an estimated capacity curve: at what notional $ does the sleeve degrade by 10% / 25% / 50%?
- For sleeves without empirical capacity data: cite the asset's typical dollar-bar size from CLAUDE.md asset universe table and note "capacity unknown — start at <1x bar size".
- Provenance: gap G5 in trader-skill audit 2026-05-28.

## Item 14 — Decay monitor configured

- `RiskController.decay_monitor` is wired and writes to `crypto/runs/paper_trade/<sleeve_id>_decay_status.json` every bar.
- Halt thresholds declared: `IC_halflife_bars`, `consecutive_DD_days`.
- Provenance: H18 paper-trade is the first live decay-monitor ground-truth. Deploy commit fd0a870 + 4ba027b.

## Item 15 — DD response plan attached

- Deploy claim names a row in `RISK_PLAYBOOK.md` for the expected DD response (size-halving thresholds, kill-switch trigger).
- For new sleeve archetypes without a matching row, add one BEFORE deploy.

## Item 16 — Exchange-side verification

- Test-fill confirmed on a $20-100 notional order at the actual venue.
- API rate-limit budget understood: max orders/sec, max API weight/min, weight-per-order.
- Withdrawal limits and 2FA confirmed.
- For multi-venue: hedge venue identified and tested (default: none — Binance single venue).

---

## Deploy claim JSON shape

```jsonc
{
  "sleeve_id": "h18_v2",
  "stage_transition": "paper -> live_small",
  "ship_claim": { /* from build_ship_claim_block(...) */ },
  "deploy_gates": {
    "item_01_cost_source_canonical": "VERIFIED at scripts/wealth_bot/<sleeve>.py:42 imports SPOT_COST",
    "item_02_mtm_reconciliation":   "VERIFIED probe_simulator_fix.py exit 0, ratio=0.0003",
    "item_03_look_ahead":           "VERIFIED grep clean",
    "item_04_stride_1":             "VERIFIED gen_preds --stride 1 in commit fd0a870",
    "item_05_p_fill_budget":        { "low": 0.25, "high": 0.50, "expected_live_equity_pct_of_backtest": [50, 75] },
    "item_06_claim_contract":       "VERIFIED check_wealth_bot_claims.py exit 0",
    "item_07_mechanism_falsifier":  { "concentrated_flag": false, "verified_by": "N/A (low concentration)" },
    "item_08_multi_seed":           { "n_seeds": 10, "all_positive_on_unseen": true, "p05_bootstrap": 0.012 },
    "item_09_jackknife":            { "K=2": 0.34, "combined_K2_plus_S9_pct": 41.2 },
    "item_10_purge_gap":            { "bars": 400 },
    "item_11_dsr_cscv":             { "sweep_N": 324, "DSR_p_value": 0.97, "cscv_pbo_pct": 0.32 },
    "item_12_survivorship":         "DECLARED tier 1 only, survivorship bias low",
    "item_13_capacity_estimate":    { "tier": 1, "starting_notional_usd": 500, "degrades_at_pct": { "10": 5000, "25": 25000, "50": 75000 } },
    "item_14_decay_monitor":        { "wired": true, "halt_IC_halflife_bars": 200, "halt_consecutive_DD_days": 7 },
    "item_15_dd_response_plan":     "RISK_PLAYBOOK.md::row_long_only_spot_tier1_kelly_fractional",
    "item_16_exchange_verification": { "test_fill_usd": 50, "venue": "binance_spot", "rate_limit_budget_orders_per_sec": 4 }
  }
}
```

## CDAP wiring

Items 1, 2, 3, 4, 6, 9, 10, 11 = `[CDAP]` — added to `crypto/config/_invariants.yaml::trader_deploy_gates` (rule names below) and validated by `crypto/src/audit/check_deploy_gates.py`. The deploy commit is identified by file-path pattern `crypto/runs/deploy/*/deploy_claim.json` OR a `crypto/config/sleeves/*.yaml` flag change from paper -> live_*.

| Item | Rule name in _invariants.yaml |
|---|---|
| 1 | `trader_cost_source_canonical` |
| 2 | `trader_mtm_reconciliation_probe` |
| 3 | `trader_no_lookahead_in_selection` |
| 4 | `trader_stride_1_predictions` |
| 6 | `trader_claim_contract_v12_passes` |
| 9 | `trader_jackknife_k2_positive` |
| 10 | `trader_walkforward_purge_gap` |
| 11 | `trader_dsr_cscv_when_sweep_gt_20` |

Items 5, 7, 8, 12, 13, 14, 15, 16 = `[CLAIM]` — required fields in the deploy_claim JSON; absence = HALT.
