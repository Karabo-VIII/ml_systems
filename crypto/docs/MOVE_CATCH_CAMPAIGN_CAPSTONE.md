# Move-CATCH campaign — CAPSTONE (DEV-walled, 2026-06-20)

The honest, turnkey result of the 7h autonomous DEV-walled move-catch campaign. **DEV = TRAIN+VAL only (≤ 2024-05-15).
OOS/UNSEEN never touched** — validation of the unseen window is the user's one-command next step (see §5).

## 1. The question + the honest answer

**Ask:** "build engines that trade any 7-day slice profitably (14-day lookback, multi-timeframe, u50/u100)."

**Answer (proven, not asserted):**
- **Structurally IMPOSSIBLE:** a long-only-spot engine reliably positive on *any* 7d slice. **Cash theorem:** all-slice
  profit-rate ≤ `in_market_rate (0.538) × in_market_up_rate (0.84) ≈ 0.34`. Cash earns 0; bear participation earns
  negative. No LO-spot configuration escapes this. Any claim of a >~0.34 slice-rate is an artifact.
- **ACHIEVABLE (validated DEV-clean):** a **regime-aware participate-preserve book** — full bull/chop participation
  (50–58% slice profit-rate) + cash-in-bear preservation. It out-compounds buy-hold by **losing less in bear**, not by a
  per-slice edge. Honestly: drawdown-insurance with up-regime beta, not regime-conditional alpha.

## 2. What the campaign proved — the de-risked-beta WALL is invariant

Five DEV-walled cycles, each with an adversarial referee that re-derived independently, all hit the same wall. **No
internal feature produces a positive bear edge whose block-bootstrap p05 clears zero:**

| Axis | bull/chop | BEAR (the wall) |
|---|---|---|
| price selection (mom14/brk14) | chop p_le0 0.018/0.014 REAL | mom14 p05 −1.94 (p_le0 0.66); brk14 p05 −4.00 (0.94) |
| move-capture 4h/1h | — | TF-invariant negative |
| capture-rate (churn-immune null) | up-regime continuation | not regime-conditional (regime-shuffle fails) |
| v51 exogenous (funding/liq/basis/whale/…) | liq flush p_le0 0.017–0.034 REAL | 0/12 bear-positive; 0 Holm survivors |
| combined book | bull p_le0 0.0085, chop 0.021 | combined bear p05 −1.80 (p_le0 0.67) |

**The methodology is the durable win.** The broken same-exposure-shuffle null (per-bar reselection churn-penalizes the
control, deflating SE 2–3.6×) was replaced by a **churn-immune random-ENTRY null** + a **date-block moving-block
bootstrap**. The **regime-shuffle is an artifact detector, not an edge detector** — it fires on cross-regime *sign
asymmetry*, so its p must be paired with the regime's own `block_p_le0 > 0` to mean anything. Use this machinery for any
future cycle.

## 3. The deployable — `src/strat/move_catch_book.py` (FROZEN 2026-06-20)

- **Selection (component A):** top-K=5 by EW z-composite of {mom14, brk14}, hold 7d, time-exit, 1d, rebalance every hold.
- **Amplifier (component B):** {liq_capitulation OR liq_short_panic} → size-up 1.5×, **bull/chop only**. *Real but ~93%
  redundant with A (flush fires inside the mom/brk pool 93% of the time); a within-continuation amplifier, NOT an
  independent onset detector. Premium ≈ +2.5pp/slice. Treat as OPTIONAL.*
- **Gate (component C):** causal `regime_series(w_days=50)` — bull/chop participate; **bear → cash (0% exposure)**.
  Thresholds structural (breadth 0.5, trend 0.0), not fitted. **This is an EXPOSURE CONTROL, not a signal — its value is
  reducing exposure in bad periods, and it is the decay vector (if bear rallies emerge, the gate stays out of them too).**

### Honest DEV numbers (report THESE — not the compound artifact)

> The book's compound (+16,926%) is a frictionless EW-of-selected-pool concurrency artifact — a same-exposure RANDOM
> pick compounds to +1,226,164% in the identical steps. The real selection edge over same-exposure random is **+0.87pp/step
> (block p05 +0.12, p_le0 0.030 = barely real)**. The deliverable metrics are slice-rate + maxDD-saved.

| metric | frozen K=5 book | buy-hold | gate-alone EW (capacity-free) |
|---|---|---|---|
| bull slice profit-rate / mean | 50.7% / +6.16% | — | — |
| chop slice profit-rate / mean | 58.5% / +5.13% | — | — |
| bear slice (by construction) | cash: 0 / 0 | — | — |
| **full-period maxDD** | **−56.5%** | −83.4% | **−42.3%** |
| **maxDD saved vs BH** | **+26.8pp** | — | **+41pp** |
| bear-only maxDD | 0% (cash) | −92.2% | — (gate) |

**Reconciled maxDD (the earlier −27.5/−42.3/−56.5 spread = three different portfolios, not a bug):** the frozen K=5 book
is **−56.5%**; the gate-alone-EW preservation mechanism is **−42.3%**. The −14pp gap is the **concentration cost** — K=5 +
the 1.5× amplifier buy higher participation at deeper drawdown. **Deployment tradeoff:** a higher-K or gate-on-EW variant
trades upside for a shallower drawdown (−42% vs −56%); pick per risk tolerance.

## 4. Deployment readiness — CONDITIONAL_GO

OOS-handoff is mechanically sound (5/5 wall checks pass: `load_wide` hard-asserts on end ≥ 2024-05-15; `oos_validate`
rejects pre-DEV starts; FROZEN_CONFIG is a constant; no refit path). Residual risks:
- **HIGH:** ~23.5-bar (≈3.3 week) mean bear-ENTRY lag — holds losing positions before the gate flips (v2: faster bear
  detection / an intra-hold stop). Data 20d stale (refresh before any live use). No DSR/IC-decay monitor; no
  consecutive-DD halt wired.
- **MEDIUM:** bear whipsaw (55.6% of bear episodes < 10 bars → add a 3–5 bar dwell filter); the gate-is-exposure-control
  decay vector.

## 5. The user's decisive next step + the external frontier

1. **Run the OOS falsifier ONCE** (cheap, decisive — preservation should persist; the thin +0.87pp/step selection edge is
   the part most likely to decay):
   ```python
   from strat.move_catch_book import oos_validate
   results = oos_validate(oos_start="2024-05-15")   # frozen params, no refit; I never ran this
   ```
2. **External data = the only axis with a bear-specific / exogenous-onset prior internal data structurally cannot carry**
   (deferred per the charter; needs the user). Ranked: **Coinbase/Upbit listing announcements** (cleanest exogenous-onset),
   spot-ETF creation/redemption flow, on-chain exchange-INFLOW capitulation, absolute-threshold stablecoin mint/burn
   (note: `stbl_*` is a single global metric broadcast cross-sectionally — needs an absolute-threshold trigger, not the
   degenerate XS tercile).
   **PRE-REGISTERED GATE (bind BEFORE any OOS):** an external feed ships only if its bear edge clears `block_p_le0 < 0.05`
   under Holm correction across the regime family, on the same churn-immune random-ENTRY + date-block bootstrap machinery.
3. **Cheapest deployable falsifier (pre-deploy):** recompute `liq_short_panic` vs the mom/brk pool with **listing-age
   stratification** — if the +2.72pp premium collapses, it was micro-cap listing-age beta proxying as a flush signal (drop
   the amplifier); if it survives, the amplifier is locked.

## Artifacts
- Deployable + handoff: `src/strat/move_catch_book.py`
- Labs (DEV-walled): `src/strat/fleet_lab.py`, `src/strat/capture_lab.py` (`evaluate_ti(block=True)`), `src/strat/v51_feature_lab.py`
- Referee/adversarial: `src/strat/quant_book_adversarial.py`, `runs/strat/capture_adversarial_20260620_024550.json`,
  `runs/strat/v51_exo_decisive_battery.py`, `runs/strat/referee_chop_rederive.py`
- Cumulative ledger: `runs/strat/_meta_fold_ledger.md` · Framework: `docs/META_FOLD_FRAMEWORK.md`
