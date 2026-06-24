# Paper Trader / Portfolio Replay Wiring

## Engine Compatibility: CONFIRMED

The V2 engines are **already compatible** with the canonical paper trading infrastructure
(`portfolio_replay.run()`). The `hold_provider` interface is the connection point:

```python
# The hold_provider callable is what the engine produces
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider

sub_band, _ = select_band_trainval("WMA", "4h")  # FROZEN band from TRAIN+VAL
provider = make_hold_provider(sub_band, "WMA")    # returns callable(sym, o, h, l, c) -> int8[]
```

## Usage with portfolio_replay (production backtest)

```python
from strat.portfolio_replay import run

# DEV window (in-sample)
result = run(universe="u100", cadence="4h", strat_names=["wma_4h"],
             window="TRAIN", cost_rt=0.0006, use_spine=False,
             vol_target=0.02, max_per_name=0.15,
             hold_provider=provider, wall_end="2024-05-15")

# OOS window (held-out validation)
result_oos = run(universe="u100", cadence="4h", strat_names=["wma_4h"],
                 window="TRAIN", cost_rt=0.0006, use_spine=False,
                 vol_target=0.02, max_per_name=0.15,
                 hold_provider=provider, wall_end="2025-01-01")
```

## For the decoupled (7d-lookback) version

The decoupled provider adds the lookback filter on top:
```python
from strat.ma_wealth_driver_decoupled import make_decoupled_hold_provider

provider = make_decoupled_hold_provider(sub_band, "WMA",
                                         lookback_bars=42,  # 7d * 6 bars/day
                                         slice_bars=42)     # 7d trade horizon
```

## For the combined V1+V2 (regime-gated)

Apply the BTC regime gate BEFORE feeding to portfolio_replay:
the regime gate zeros out the hold_provider's output during bear periods.
See `(tmp)/_combined_v1v2.py` for the reference implementation.

## What portfolio_replay provides

- Lag-1 causal accounting (holds at t-1 determine position at t)
- Maker cost on turnover (0.0006 RT)
- Vol-targeting (optional, per-asset inverse-vol sizing)
- Max per-name concentration cap
- 2h cadence synthesis from 1h (built into the function)
- DEV wall enforcement (physically cannot peek OOS)
- Output: final_pct, maxdd_pct, sharpe, n_bars, n_assets

## Status

No additional wiring needed — the engines slot directly into the existing infra.
The `oos_engine_test.py` module provides a standalone OOS testing framework that
reimplements the logic independently (for verification/audit purposes).
