# Parallel Researcher Report #2 (scout-strat) — 2026-06-05 ~23:58 — NEXT-TRIGGER RANKING

Context: the MA-cross trigger is refuted (1d+4h, raw+ER-gated, raw+beta-residualized — 0/77 beat the
regime-matched null). The MA/ER survives only as a regime FILTER. This ranks the candidate NON-MA entry
TRIGGERS to test within the ER>0.4 regime. All chimera features below are lookahead_safe + pre-computed
(config feature_catalog; 215 features). Apparatus: src/strat/setup_harness.py + regime-matched firewall +
positive_control (proven two-sided sound).

## TOP-3 (EV-ranked) — test each vs the regime-matched null on held-out, 3-DOF max, beat null BEFORE complexity

### 1. LIQUIDATION-CASCADE reversal  (HIGHEST EV)
- Signal: LONG after a short-flush + reclaim. Gate on `liq_short_z30.shift(1)` / `liq_short_panic.shift(1)`
  (STRICT prior-bar — avoids reverse causality), enter on reclaim of a reference level.
- Mechanism (why it beats random): forced short-covers = genuine NON-discretionary buy demand — structural,
  not a price-pattern coincidence. Market-research: liq_short_z30 fires ~5-6% of days, +25% contemporaneous move.
- KILL CONDITION = **beta inflation** (Tigro 2026: post-cascade returns ~54% beta, α non-significant). → test on
  **beta-RESIDUALIZED** returns (reuse rig-E2 `beta_residualize_4h`) + the regime-matched firewall. Both already built.
- Overfit bound: ≤3 DOF; <6% event freq → small UNSEEN, pre-register thresholds; .shift(1) mandatory.

### 2. MOMENTUM-ACCELERATION
- Signal: `norm_momentum_accel` (pre-computed, 100% non-null, lookahead_safe — NO lookback window to overfit).
  Enter when acceleration positive within ER>0.4 (impulse phase).
- Mechanism: forward derivative of momentum (distinct from the lagging MA cross); narrows to the impulse-only subset.
- Overfit risk: crypto momentum survivorship bias; but the feature is window-locked → low DOF. ~0.5 dev-day.

### 3. ISOLATED BREAKOUT / range-expansion
- Signal: `close[t-1] > rolling_max(high, N=10 or 20)` + `norm_log_volume` confirmation; filter fake-outs with
  `bd_thin_book_frac`. Pre-register N (no grid search).
- Note: breakout was PARTIALLY tested co-polluted WITH the dead MA trigger (plain rig, refuted). The ISOLATED
  breakout (no MA) is the genuinely untested clean step. Cheapest to build (pure OHLCV).

## Lower priority: pullback-in-uptrend (weak evidence base), VPIN/orderflow (`norm_vpin`, `norm_hawkes_buy_intensity`
— theoretically grounded but BTC-specific + decaying in the literature).

## Verdict for the rigs
Test trigger #1 (liquidation-cascade) FIRST — strongest mechanism, kill-condition (beta) already instrumented via
beta_residualize + the firewall, ~1 dev-day. Then #2 (momentum-accel, ready feature). Each: prove it beats the
regime-matched random-entry null on held-out at ≤3 DOF before adding anything.

Sources: Tigro Blanc liq-cascade + VPIN (Medium 2026), arXiv 2602.11708 (adaptive trend), Springer 2025 (crypto
momentum moments), ScienceDirect 2025/2026 (order-flow toxicity, survivor momentum), market-research §5.
