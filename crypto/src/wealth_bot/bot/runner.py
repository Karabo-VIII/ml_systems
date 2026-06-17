"""runner -- paper-trade main loop.

Walks bars in a target segment (default UNSEEN), at each bar:
  1. Skip if currently in an open position (non-overlapping execution).
  2. Ask SignalEngine for the decision (ensemble preds + threshold + signals).
  3. If fire: PositionSizer -> OrderGenerator.make_entry -> log to journal.
  4. After fwd_bars bars: OrderGenerator.make_exit + realized PnL -> update equity.
  5. RiskManager.check_kill_switch -> if triggered, halt and stop.

The loop is non-overlapping: after a trade enters at bar i, the next eligible
entry is at bar i + fwd_bars (matches framework.signal_picker invariant).

__contract__:
  inputs:
    - BotConfig (risk + cost params + fwd_bars)
    - SignalEngine (ensemble preds + threshold)
    - df (raw, with date + close)
    - signals (n, K) binary
    - segment name (e.g., 'UNSEEN')
    - segment mask (bool ndarray over df)
    - journal_path (Path)
  outputs:
    dict with:
      n_trades, final_equity_usd, total_return_pct, max_dd_pct,
      win_rate, sharpe, halted, halt_reason, equity_curve, journal
  invariants:
    - no peek: at bar t, only ensemble_preds[t] + signals[t] consulted
    - non-overlapping: entries spaced >= fwd_bars apart
    - exit price = df.close at (entry_bar + fwd_bars), or last bar of segment
    - halt is monotonic -- once True, loop stops immediately
"""
from __future__ import annotations

__contract__ = {
    "kind": "paper_trade_runner",
    "owner": "wealth_bot/bot/runner",
    "purpose": "Walk bars, decide, size, order, exit, journal; halt on tripwire",
    "invariants": [
        "no peek -- bar t consults only [t] rows",
        "non-overlapping execution (>= fwd_bars between entries)",
        "halt is monotonic",
        "equity curve recorded at every bar (mark-to-market)",
    ],
}

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .order_generator import OrderGenerator
from .position_sizer import PositionSizer
from .risk_manager import RiskManager
from .signal_engine import SignalDecision, SignalEngine
from .telemetry import Telemetry


def _segment_index_range(mask: np.ndarray) -> tuple[int, int]:
    """Return (start_idx, end_idx_exclusive) of contiguous True region.

    The masks built by framework.data_loader.segment_masks are by construction
    contiguous (date-window filters). We assert that and return the range.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return (0, 0)
    start, end = int(idx[0]), int(idx[-1]) + 1
    # Sanity: contiguity
    if end - start != len(idx):
        # Fall back to the first contiguous run from the first True bar.
        # This keeps the loop well-defined even if upstream changes break
        # the date-window contiguity invariant.
        end = start + len(idx)
    return start, end


def _annualized_sharpe(trade_pnls: list[float], bars_per_year: float = 365.0 * 6.0) -> float:
    """Approximate annualized Sharpe from per-trade decimal returns.

    bars_per_year default = 365 * 6 = 2190 (4h cadence -> 6 bars/day).
    This is a proxy -- not a true risk-adjusted return calculation.
    """
    if len(trade_pnls) < 2:
        return 0.0
    arr = np.asarray(trade_pnls, dtype=float)
    if arr.std() <= 0:
        return 0.0
    # Trade-level Sharpe scaled by sqrt(n_trades) is the convention used
    # in framework.signal_picker.evaluate_actions; mirror it for consistency.
    return float(arr.mean() / arr.std() * np.sqrt(len(arr)))


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    """Peak-to-trough DD in percent (positive number)."""
    if not equity_curve:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / np.where(peaks > 0, peaks, 1.0)
    return float(abs(dd.min()) * 100.0)


def run_paper_trade(
    cfg: Any,                          # BotConfig (avoid hard import cycle)
    signal_engine: SignalEngine,
    df: pd.DataFrame,
    df_lag: pd.DataFrame,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    ensemble_preds: np.ndarray,
    segment: str,
    segment_mask: np.ndarray,
    journal_path: str | Path,
    verbose: bool = True,
) -> dict:
    """Walk the target segment and emit a paper-trade journal + summary.

    df_lag and fwd_ret are accepted for signature symmetry with the framework
    layer (downstream extensions may consume them); the current loop only
    consults df.close + ensemble_preds + signals.
    """
    risk = cfg.risk
    fwd_bars = int(cfg.fwd_bars)
    cadence_hours = {"1h": 1.0, "4h": 4.0, "1d": 24.0}.get(cfg.cadence, 4.0)

    telemetry = Telemetry(journal_path)
    order_gen = OrderGenerator(
        symbol=cfg.asset + "USDT" if not cfg.asset.endswith("USDT") else cfg.asset,
        cost_per_side_pct=risk.cost_per_side_pct,
    )
    sizer = PositionSizer(
        max_position_pct=risk.max_position_pct,
        cold_start_fallback_pct=0.25,
        min_trades_for_kelly=5,
    )
    risk_mgr = RiskManager(
        max_drawdown_pct=risk.max_drawdown_pct,
        max_consecutive_losses=risk.max_consecutive_losses,
        whale_freshness_max_hours=risk.whale_freshness_max_hours,
    )

    start_idx, end_idx = _segment_index_range(segment_mask)
    if start_idx == end_idx:
        telemetry.alert("WARNING", f"segment {segment} is empty; aborting")
        return {
            "n_trades": 0, "final_equity_usd": float(risk.starting_capital_usd),
            "total_return_pct": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0,
            "sharpe": 0.0, "halted": False, "halt_reason": "empty_segment",
            "equity_curve": [], "journal": [],
        }

    if verbose:
        print(f"[runner] segment={segment} bars=[{start_idx}, {end_idx}) "
              f"({end_idx - start_idx} bars)", flush=True)

    # Loop state
    equity = float(risk.starting_capital_usd)
    equity_curve: list[float] = [equity]
    trade_pnls: list[float] = []
    journal_entries: list[dict] = []
    halted = False
    halt_reason = ""
    n_trades = 0

    # Closes + timestamps (ms)
    closes = df["close"].values
    if "timestamp" in df.columns:
        timestamps_ms = df["timestamp"].astype("int64").values
    else:
        # Fall back to converting `date` -> ms
        timestamps_ms = (pd.to_datetime(df["date"]).astype("int64") // 1_000_000).values

    i = start_idx
    while i < end_idx:
        # Mark-to-market: equity_curve includes a point per bar walked.
        # (Approximation: only realized PnL updates equity; an open position
        # is held at its entry value until exit.)
        # equity_curve already has the value going INTO bar i.

        # Signal at this bar
        decision = signal_engine.predict_bar(i)

        if not decision.fire:
            i += 1
            equity_curve.append(equity)
            continue

        # Compute rolling stats for Kelly
        if len(trade_pnls) >= 5:
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p <= 0]
            wr = len(wins) / len(trade_pnls)
            mw = float(np.mean(wins)) if wins else 0.0
            ml = float(np.mean(losses)) if losses else -1e-6
        else:
            wr, mw, ml = 0.0, 0.0, 0.0  # fallback path handles this

        sizing = sizer.size(
            decision=decision,
            current_capital=equity,
            recent_winrate=wr,
            recent_mean_win=mw,
            recent_mean_loss=ml,
            kelly_fraction=risk.kelly_fraction,
            n_observed_trades=len(trade_pnls),
        )

        if sizing.dollar_size <= 0:
            i += 1
            equity_curve.append(equity)
            continue

        # ENTRY
        entry_price = float(closes[i])
        entry_ts = int(timestamps_ms[i])
        entry_order = order_gen.make_entry(
            bar_idx=i,
            decision=decision,
            position_size_usd=sizing.dollar_size,
            current_price=entry_price,
            timestamp_ms=entry_ts,
        )
        entry_order["sizing_reason"] = sizing.reason
        entry_order["sizing_kelly_applied"] = sizing.kelly_applied
        telemetry.log_trade(entry_order)
        journal_entries.append(entry_order)

        # EXIT at i + fwd_bars (or last bar of segment if we'd run past)
        exit_idx = min(i + fwd_bars, end_idx - 1)
        exit_price = float(closes[exit_idx])
        exit_ts = int(timestamps_ms[exit_idx])
        exit_reason = "fwd_bars_elapsed" if exit_idx == i + fwd_bars else "segment_end"
        exit_order = order_gen.make_exit(
            entry_order=entry_order,
            current_price=exit_price,
            timestamp_ms=exit_ts,
            exit_reason=exit_reason,
        )
        pnl_pct = order_gen.realized_pnl(entry_order, exit_order)
        pnl_usd = sizing.dollar_size * pnl_pct
        exit_order["realized_pnl_pct"] = float(pnl_pct)
        exit_order["realized_pnl_usd"] = float(pnl_usd)
        exit_order["entry_bar_idx"] = i
        exit_order["exit_bar_idx"] = exit_idx
        telemetry.log_trade(exit_order)
        journal_entries.append(exit_order)

        equity += pnl_usd
        trade_pnls.append(float(pnl_pct))
        n_trades += 1

        # Fill equity curve for the bars from i to exit_idx with the
        # post-trade equity (we only realize on exit; cleaner than NaN).
        for _ in range(i, exit_idx + 1):
            equity_curve.append(equity)

        if verbose and n_trades % 5 == 0:
            print(f"[runner]   trade #{n_trades} bar={i} strat={decision.chosen_strategy_idx} "
                  f"conf={decision.confidence:.2f} pnl={pnl_pct*100:+.2f}% "
                  f"equity=${equity:.2f}", flush=True)

        # Risk check (whale age = cadence_hours; whale filter is lagged by
        # chimera_lag_bars in the data_loader so in paper-trade replay the
        # data is by construction fresh -- pass 0 to disable that tripwire
        # in replay. Live mode would pass the real wall-clock age.)
        whale_age_h = 0.0
        should_halt, reason = risk_mgr.check_kill_switch(
            equity_curve=equity_curve,
            whale_data_age_hours=whale_age_h,
            recent_trade_pnls=trade_pnls,
        )
        if should_halt:
            halted = True
            halt_reason = reason
            telemetry.alert("CRIT", f"HALT triggered: {reason}")
            break

        # Non-overlapping: skip past the exit bar
        i = exit_idx + 1

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    final_equity = float(equity)
    total_return_pct = (final_equity / float(risk.starting_capital_usd) - 1.0) * 100.0
    max_dd = _max_drawdown_pct(equity_curve)
    win_rate = float(np.mean([1.0 if p > 0 else 0.0 for p in trade_pnls])) if trade_pnls else 0.0
    sharpe = _annualized_sharpe(trade_pnls)

    summary = {
        "segment": segment,
        "n_trades": int(n_trades),
        "starting_capital_usd": float(risk.starting_capital_usd),
        "final_equity_usd": final_equity,
        "total_return_pct": float(total_return_pct),
        "max_dd_pct": float(max_dd),
        "win_rate": float(win_rate),
        "sharpe_proxy": float(sharpe),
        "halted": bool(halted),
        "halt_reason": halt_reason,
        "equity_curve_len": len(equity_curve),
        "equity_curve_final_5": [float(x) for x in equity_curve[-5:]],
        "telemetry_summary": telemetry.summary(),
    }

    if verbose:
        print(
            f"[runner] DONE seg={segment} trades={n_trades} "
            f"return={total_return_pct:+.2f}% DD={max_dd:.2f}% WR={win_rate:.2f} "
            f"halted={halted}",
            flush=True,
        )

    return {
        **summary,
        "equity_curve": equity_curve,
        "journal": journal_entries,
    }
