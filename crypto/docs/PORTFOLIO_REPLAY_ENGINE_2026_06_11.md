# Portfolio Replay Engine — fit TIs, replay a portfolio (paper trading) (2026-06-11)

> User /orc: *"a strat replay engine where we can fit TI(s), then have them replay a portfolio, like
> paper trading."* Tool: [`src/strat/portfolio_replay.py`](../src/strat/portfolio_replay.py). This is
> the missing wire between the firm harness pieces that already existed
> (`src/firm/decision_spine.py` sizing, `src/firm/portfolio.py` allocation) and a sequential replay
> loop. **Generalized paper-trading: a declarative TI spec → one risk-budgeted portfolio equity curve
> over any window.**

## What it does (the pipeline, wired end-to-end)
```
TI library (declarative)          -> per-(asset, strategy) HOLDING state (long while in-position; causal)
firm/decision_spine.decide()      -> per-bet sizing+gating (Kelly + confidence floor + uncertainty
   [optional, --spine]               haircut + regime posterior); default = inverse-vol risk parity
firm/portfolio.allocate()         -> portfolio weights: vol-targeted, gross-cap, per-name-cap, corr-aware
MtM-correct replay                -> lagged weights x next-bar return - turnover cost -> equity curve
```
- **Declarative** — a SPEC is just `--strategies "ema_50_100,donch20,rsi_30_50"` (any subset of the TI
  library) + universe + cadence + risk policy + window. Swap TIs freely.
- **Any window = paper-trade** — `--window {TRAIN|VAL|OOS|UNSEEN|ALL}`. UNSEEN is forward paper-trading
  on data sealed during dev. (Live-forward is the same call on a future window.)
- **Honest** — strictly causal (signals/vol trailing; weights lagged 1 bar), MtM no-double-count (the
  CLAUDE.md invariant), taker/maker cost on turnover, date-aligned panels (floored+deduped).
- **Output** — `$1 → $X` equity, ann %, maxDD, Sharpe, win-bar rate, the charter **3d-ROI** soft-
  benchmark (positive-rate + median), avg gross/turnover, and **per-strategy attribution** (hold-bars).

## The TI library (extend `STRATS` to add more)
`ema_50_100 · ema_50_200 · ema_20_100 · sma_50_200 · donch20 · roc20 · rsi_30_50 · boll20 · trendgate`
(2-MA crosses, Donchian channel, ROC momentum, RSI bounce, Bollinger mean-reversion, SMA100 trend-gate.)

## Demonstration (u10, 3 TIs, RWYB)
| Window | Sizing | $1 → | ann | maxDD | Sharpe | note |
|---|---|---|---|---|---|---|
| ALL (2020–25) | inverse-vol | $35.8 | +74.9% | −70% | 1.19 | unhedged trend book (bull-loaded) |
| ALL | **firm-spine** | **$48.0** | **+83.2%** | **−40%** | **1.67** | decision_spine sizing **improves risk-adjusted** |
| UNSEEN (2026 bear) | inverse-vol | $0.75 | −51% | −32% | −1.25 | unhedged loses the bear (no regime gate — expected) |

The firm-spine result is the engine's point: the same TIs, sized by the decision spine (Kelly +
confidence floor + uncertainty), lift Sharpe 1.19→1.67 and halve the drawdown — the risk layer does
real work. (A regime gate on top is what the deploy-decision book adds to survive the bear; that is a
strategy choice the engine can replay, not a property of the engine.)

## How to use it
```
# paper-trade a 3-TI book forward on sealed UNSEEN, firm-spine sized:
python -m strat.portfolio_replay --universe u50 --window UNSEEN --strategies "ema_50_100,donch20" --spine

# full-cycle replay, taker cost, tighter per-name cap:
python -m strat.portfolio_replay --universe u10 --window ALL --strategies "trendgate,rsi_30_50" \
    --vol-target 0.015 --max-per-name 0.10
```
Every run persists a JSON (spec + git SHA + result) under `runs/strat/portfolio_replay_*.json`.

## Scope / honest notes
- It replays whatever TIs you give it — it does NOT discover edges or assert a TI is good; the equity
  is only as good as the spec (and our research says daily-regime-gated-trend is the robust spec).
- **Correctness RED-team passed 2026-06-11: 0 CRITICAL / 0 HIGH** — MtM no-double-count, weights
  lagged 1 bar, signals/vol strictly causal, UNSEEN does not leak, date-alignment verified. Hardened
  with a floor-invariant assertion (a dropped date-floor would otherwise silently produce a −99% curve).
- **Honest caveats the audit surfaced (read before trusting the risk knobs):**
  - `--vol-target` is **weak**: `allocate(corr=None)` assumes independence, so on a basket of
    high-correlation crypto longs the 2% target rarely binds — the book is governed by the **gross**
    (1.0) and **per-name** caps, and realized daily vol ran ~3.5% vs the 2% nominal. Treat it as
    cap-governed until a real correlation matrix is wired in.
  - `--spine` acts as a **confidence/momentum GATE on top of inverse-vol** (trailing-edge>0 AND
    P(profit)≥floor), not as a fine Kelly sizer — the continuous-Kelly pins to the per-name cap, and
    no `edge_se` is fed so the uncertainty haircut is inert. Attribute the Sharpe uplift to the gate.
  - WIN split dates are **bespoke to this engine** (TRAIN<2024-05-15 / OOS / UNSEEN 2025-12-31→) and
    purge-gap-free; not directly comparable to family_regime_map's split. No leakage (nothing is
    fitted; params are fixed textbook values) — but cross-engine UNSEEN numbers aren't apples-to-apples.
  - Windowed runs treat a position already open at the window's left edge as cost-free at entry
    (correct paper-trade "already holding when the window opens" semantic; slightly optimistic on entry cost).
- Extensions (open): per-strategy P&L attribution (not just hold-bars), a regime-gate overlay flag,
  live-data window, real correlation matrix into `allocate()` (makes vol-target bind), `edge_se` into
  the spine (engages the Kelly haircut).
