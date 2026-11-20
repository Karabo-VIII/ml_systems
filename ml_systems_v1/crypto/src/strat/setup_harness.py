"""src/strat/setup_harness.py -- the SETUP-LEVEL (multi-candle MOVE) evaluation harness.

WHY THIS EXISTS (the gap it fills)
----------------------------------
The kept `wealth_bot.harness.CanonicalHarness` scores a CROSSOVER signal (fast_col > slow_col) with
signal-flip / max-hold exits. It cannot express the project's *founding unit of trading* (MEMORY.md,
2026-06-04 reset):

    "The unit of trading is the SETUP across a MOVE (multiple candles) -- wait for a confirmed setup,
     enter, exit by policy. We capture *moves*, not single candles. IC / per-bar predictability is the
     WRONG lens and is BANNED as a primary metric."

This module is that missing primitive. It scores an ARBITRARY past-only boolean ENTRY ("the setup is
confirmed at the close of bar t") chased to a declarative POLICY EXIT (take-profit / stop-loss /
trailing-stop / time-stop / dedicated exit-signal) and reports the COMPOUND return per window, with the
held-out (UNSEEN) window as the verdict surface. It NEVER computes IC / ShIC / per-bar predictability;
the score is the compound return of entry->move->exit, full stop. (IC-INDEPENDENCE is a design
invariant, not an omission -- see __contract__.)

It is deliberately APPARATUS-COMPATIBLE: `SetupHarness` exposes the same duck-typed surface the existing
gate reads (`.run()` -> results with `.trades` / `.window_stats` / `.all_4_positive`, plus `.WINDOWS`,
`._window_label`, `.spec.cost_rt`, `.spec.filter_col`, `.df`). So an evaluated setup plugs straight into
`strat.firewall.random_entry_null` (cost-matched random-entry null), `strat.battery.evaluate` (Lens
A/B/C robustness) and `strat.benchmark.benchmark_excess` with ZERO new glue.

LOOK-AHEAD / LEAKAGE GUARDS (built in, defence in depth)
--------------------------------------------------------
1. STRUCTURAL (by construction -- a leak is not expressible through the API):
     - entry fill  = opens[i+1]              (NEXT-BAR-OPEN; same-bar closes[i] fill is Pattern T, banned)
     - TP/SL/trail breach via highs[j]/lows[j] only  (intra-bar; Pattern S `max(low,trail)` impossible)
     - trailing stop uses the PRIOR-bar high-water-mark for the breach check, then ratchets -- so the
       current bar's high can never loosen the stop that the current bar's low must clear (pessimistic)
     - within a single bar, an adverse stop is assumed to fire BEFORE a favourable target (pessimistic
       intra-bar ordering -- we cannot know which the bar hit first, so we take the worse one)
     - stop/target fills honour GAP-THROUGH: a long stop fills at min(open, stop_level); a long target
       fills at max(open, tp_level) -- you get the worse-for-a-stop / available-for-a-limit price, never
       a fictitious better-than-level fill
2. MEASURED (`leak_guard()`): a self-contained RELATIVE lead/lag test. Re-run the setup with the entry
   column shifted +1 (one bar MORE lag, still past-only) and -1 (one bar of FUTURE injected). A
   genuinely past-only entry has large headroom -- injecting the future (-1) changes held-out compound
   far more than adding lag (+1). If injecting the future barely changes the result, the entry ALREADY
   encodes the future -> LEAK_SUSPECT. The ratio compares two arms of the SAME setup on the SAME
   cadence, so the (cadence-dependent) shift-sensitivity noise floor cancels -- the fix the absolute-pp
   leak_probe could not achieve on coarse bars (see wealth_bot/leak_probe.py CALIBRATION FINDING).

RWYB self-tests (no market data needed):  `python src/strat/setup_harness.py --selftest`
RWYB on real chimera (BTC 1d breakout):   `python src/strat/setup_harness.py`

HARD CONSTRAINTS (inherited, non-negotiable): LONG-ONLY, SPOT, LEVERAGE=1, TAKER 0.24% round-trip
honest cost, UNSEEN touched once, objective = WEALTH (compound %) under the robustness battery.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path for wealth_bot.harness
from wealth_bot.harness import WindowStats  # exact-parity window aggregation container

__contract__ = {
    "kind": "setup_level_move_harness",
    "version": "1.0",
    "inputs": ["df(date,open,high,low,close[,funding])", "entry_col(past-only boolean)", "ExitPolicy", "WindowSpec"],
    "outputs": ["SetupResults(per-trade log, per-window compound/DD, all_4_positive); leak_guard() verdict"],
    "invariants": [
        "IC-INDEPENDENT: never computes IC/ShIC/per-bar predictability; score is compound of entry->exit",
        "entry_p = opens[i+1] -- NEVER closes[i] (Pattern T banned by API)",
        "TP/SL/trail breach via highs[j]/lows[j] only (Pattern S banned by API)",
        "trailing stop uses PRIOR-bar hwm for breach then ratchets (no same-bar loosening)",
        "ATR trail width uses PRIOR-bar ATR (atr[j-1]) -- same leak-safe convention as the prior-bar hwm",
        "adverse stop assumed to fire before favourable target within a bar (pessimistic ordering)",
        "stop fills at min(open,level); target fills at max(open,level) -- honest gap-through",
        "entry column is the caller's past-only contract; leak_guard() measures residual leak",
        "apparatus-compatible: duck-types CanonicalHarness so firewall/battery/benchmark reuse unchanged",
    ],
}


# ---------------------------------------------------------------------------
@dataclass
class ExitPolicy:
    """Declarative multi-candle MOVE exit. At least one mechanism MUST be set (no open-ended hold).

    tp_pct       : take-profit, +fraction from entry (0.10 = +10%). None = no target.
    sl_pct       : stop-loss, fraction BELOW entry as a POSITIVE number (0.05 = -5%). None = no stop.
    trail_pct    : trailing stop, fraction below the running high-water-mark (0.06 = 6% trail). None = off.
    atr_trail_mult / atr_col: ATR trailing stop -- stop = high-water-mark - atr_trail_mult * ATR, where
                    ATR is read from the past-only `atr_col`. The breach width on bar j uses the PRIOR-bar
                    ATR (atr[j-1]) -- same leak-safe convention as the prior-bar high-water-mark (the
                    current bar's range can never set the stop the current bar's low must clear). Both
                    must be set together. None = off. This is the volatility-adaptive trail (vs the fixed
                    `trail_pct`); set both and the tighter (higher, for a long) stop wins.
    max_hold_bars: time stop in bars. None = no time cap (relies on price/signal exits + tail flush).
    exit_signal_col: optional past-only boolean column; True = "exit at next open" (enter on setup, exit on B).
    """
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    trail_pct: Optional[float] = None
    atr_trail_mult: Optional[float] = None
    atr_col: Optional[str] = None
    max_hold_bars: Optional[int] = None
    exit_signal_col: Optional[str] = None

    def __post_init__(self):
        if not any([self.tp_pct, self.sl_pct, self.trail_pct, self.atr_trail_mult,
                    self.max_hold_bars, self.exit_signal_col]):
            raise ValueError("[ExitPolicy] no exit mechanism set -- a setup with no exit policy is not a move.")
        for nm, v in (("tp_pct", self.tp_pct), ("sl_pct", self.sl_pct), ("trail_pct", self.trail_pct)):
            if v is not None and not (0.0 < v < 5.0):
                raise ValueError(f"[ExitPolicy] {nm}={v} out of sane (0,5) fractional range.")
        if (self.atr_trail_mult is not None) != (self.atr_col is not None):
            raise ValueError("[ExitPolicy] atr_trail_mult and atr_col must be set together (ATR trail "
                             "needs both the multiplier and the past-only ATR column).")
        if self.atr_trail_mult is not None and not (0.0 < self.atr_trail_mult < 50.0):
            raise ValueError(f"[ExitPolicy] atr_trail_mult={self.atr_trail_mult} out of sane (0,50) range.")


@dataclass
class _SetupSpec:
    """Minimal spec shim so SetupHarness duck-types CanonicalHarness for firewall/battery reuse.
    filter_col defaults to the ENTRY column so firewall(regime_matched=True) draws random entries from
    setup-ON bars only -- isolating MOVE-CAPTURE (the exit policy) from setup SELECTION."""
    cost_rt: float
    filter_col: Optional[str]
    filter_op: str = "gt"
    filter_val: float = 0.5  # boolean True (1) > 0.5
    use_funding: bool = False
    funding_col: str = "fund_rate_mean"
    funding_scale: float = 1.0


@dataclass
class SetupResults:
    trades: list
    window_stats: dict
    all_4_positive: bool

    def summary(self) -> str:
        lines = ["SetupHarness Results (entry->policy-exit compound, IC-independent):"]
        for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
            s = self.window_stats.get(w)
            if s:
                lines.append(f"  {w:6} compound={s.compound_pct:+8.2f}%  n={s.n_trades:<4} "
                             f"DD={s.max_dd_pct:7.2f}%  win={s.win_rate:5.1%}")
        lines.append(f"  all_4_positive: {self.all_4_positive}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
class SetupHarness:
    """Score an arbitrary past-only boolean ENTRY chased to a POLICY EXIT, per window, on held-out data.

    df            : DataFrame with date, open, high, low, close (+ optional funding col). NO indicator
                    pre-population required -- the ENTRY is whatever boolean column you precompute.
    entry_col     : str -- boolean column; True = "setup confirmed at CLOSE of this bar". The harness
                    fills at opens[t+1], so a close-of-bar setup is structurally past-only vs the fill.
    policy        : ExitPolicy.
    windows       : wealth_bot.harness.WindowSpec.
    cost_rt       : round-trip cost fraction (TAKER 0.0024 = the honest baseline).
    funding_col   : if use_funding, subtract this per held bar * funding_scale.
    """

    WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]

    def __init__(self, df: pd.DataFrame, entry_col: str, policy: ExitPolicy, windows,
                 cost_rt: float = 0.0024, use_funding: bool = False,
                 funding_col: str = "fund_rate_mean", funding_scale: float = 1.0,
                 regime_match_on_entry: bool = True):
        self.df = df.reset_index(drop=True).copy()
        self.entry_col = entry_col
        self.policy = policy
        self.windows = windows
        self.use_funding = use_funding
        self.funding_col = funding_col
        self.funding_scale = funding_scale
        # duck-typed spec for apparatus reuse
        self.spec = _SetupSpec(cost_rt=cost_rt,
                               filter_col=entry_col if regime_match_on_entry else None,
                               use_funding=use_funding, funding_col=funding_col, funding_scale=funding_scale)
        self._train_e = pd.Timestamp(windows.train_end)
        self._val_e = pd.Timestamp(windows.val_end)
        self._oos_e = pd.Timestamp(windows.oos_end)
        self._validate()

    def _validate(self):
        need = {"date", "open", "high", "low", "close"}
        miss = need - set(self.df.columns)
        if miss:
            raise ValueError(f"[SetupHarness] df missing required columns: {miss}")
        if self.entry_col not in self.df.columns:
            raise ValueError(f"[SetupHarness] entry_col '{self.entry_col}' not in df. Precompute it (past-only) first.")
        ec = self.df[self.entry_col]
        # accept bool / 0-1 numeric; reject continuous (a setup is a CONFIRMED state, not a score)
        uniq = set(pd.unique(ec.dropna()))
        if not uniq.issubset({0, 1, True, False, 0.0, 1.0}):
            raise ValueError(f"[SetupHarness] entry_col '{self.entry_col}' must be boolean/0-1 (a confirmed "
                             f"setup state), got values {sorted(uniq)[:6]}... Threshold a score into a boolean.")
        if self.policy.exit_signal_col and self.policy.exit_signal_col not in self.df.columns:
            raise ValueError(f"[SetupHarness] exit_signal_col '{self.policy.exit_signal_col}' not in df.")
        if self.policy.atr_col and self.policy.atr_col not in self.df.columns:
            raise ValueError(f"[SetupHarness] atr_col '{self.policy.atr_col}' not in df. Precompute a "
                             f"past-only ATR column (rolling-mean true range) first.")

    def _window_label(self, ts: pd.Timestamp) -> str:
        if ts < self._train_e:
            return "TRAIN"
        if ts < self._val_e:
            return "VAL"
        if ts < self._oos_e:
            return "OOS"
        return "UNSEEN"

    # ------------------------------------------------------------------
    def _simulate(self, entry: np.ndarray) -> list:
        """Single-position event simulator: setup-bar i -> fill opens[i+1] -> walk to policy exit."""
        df = self.df
        opens = df["open"].to_numpy(float)
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        dates = df["date"]
        fund = (df[self.funding_col].to_numpy(float)
                if (self.use_funding and self.funding_col in df.columns) else np.zeros(len(opens)))
        exit_sig = (df[self.policy.exit_signal_col].to_numpy() > 0
                    if self.policy.exit_signal_col else None)
        atr = (df[self.policy.atr_col].to_numpy(float)
               if (self.policy.atr_col and self.policy.atr_col in df.columns) else None)

        n = len(opens)
        p = self.policy
        cost = float(self.spec.cost_rt)
        trades = []

        i = 0
        while i < n - 2:
            if not entry[i]:
                i += 1
                continue
            entry_i = i
            entry_fill = i + 1                  # FILL CONTRACT: next-bar open (Pattern T banned)
            entry_p = opens[entry_fill]
            tp_level = entry_p * (1.0 + p.tp_pct) if p.tp_pct else None
            sl_level = entry_p * (1.0 - p.sl_pct) if p.sl_pct else None
            hwm = max(entry_p, highs[entry_fill])   # high-water-mark seeded incl. the fill bar
            exit_fill = None
            exit_p = None
            reason = "tail_flush"

            j = entry_fill + 1
            while j < n:
                duration = j - entry_fill
                # ---- 1. PESSIMISTIC intra-bar STOP check (uses PRIOR-bar hwm for trail) ----
                stop_level = None
                if sl_level is not None:
                    stop_level = sl_level
                if p.trail_pct is not None:
                    trail_level = hwm * (1.0 - p.trail_pct)   # hwm excludes bar j's high -> no same-bar loosen
                    stop_level = trail_level if stop_level is None else max(stop_level, trail_level)
                if p.atr_trail_mult is not None and atr is not None:
                    atr_ref = atr[j - 1]                       # PRIOR-bar ATR: known before bar j (leak-safe)
                    if np.isfinite(atr_ref):
                        atr_trail = hwm - p.atr_trail_mult * atr_ref   # hwm excludes bar j's high too
                        stop_level = atr_trail if stop_level is None else max(stop_level, atr_trail)
                if stop_level is not None and lows[j] <= stop_level:
                    exit_fill = j
                    exit_p = min(opens[j], stop_level)        # gap-through -> fill at the worse open
                    reason = "stop"
                    break
                # ---- 2. favourable TARGET (only if no adverse stop fired this bar) ----
                if tp_level is not None and highs[j] >= tp_level:
                    exit_fill = j
                    exit_p = max(opens[j], tp_level)          # gap-up -> limit fills at the available open
                    reason = "target"
                    break
                # ---- 3. TIME stop -> exit at next open (parity with canonical max_hold) ----
                if p.max_hold_bars is not None and duration >= p.max_hold_bars:
                    if j + 1 < n:
                        exit_fill, exit_p, reason = j + 1, opens[j + 1], "time"
                    else:
                        exit_fill, exit_p, reason = n - 1, closes[n - 1], "time_tail"
                    break
                # ---- 4. dedicated EXIT SIGNAL (close-of-bar) -> next open ----
                if exit_sig is not None and exit_sig[j]:
                    if j + 1 < n:
                        exit_fill, exit_p, reason = j + 1, opens[j + 1], "exit_signal"
                    else:
                        exit_fill, exit_p, reason = n - 1, closes[n - 1], "exit_signal_tail"
                    break
                # ---- no exit: ratchet hwm with this bar's high, advance ----
                hwm = max(hwm, highs[j])
                j += 1

            if exit_fill is None:                              # ran off the end -> tail flush at last close
                exit_fill, exit_p, reason = n - 1, closes[n - 1], "tail_flush"

            net = exit_p / entry_p - 1.0 - cost
            fund_net = 0.0
            if self.use_funding and exit_fill > entry_fill:
                fund_net = float(np.sum(fund[entry_fill:exit_fill]) * self.funding_scale)
                net -= fund_net

            ts = dates.iloc[entry_i]
            trades.append({
                "window": self._window_label(pd.Timestamp(ts)),
                "entry_idx": int(entry_i), "entry_fill_idx": int(entry_fill), "exit_idx": int(exit_fill),
                "entry_ts": str(ts), "entry_p": float(entry_p), "exit_p": float(exit_p),
                "net_pnl": float(net), "duration_bars": int(exit_fill - entry_fill),
                "fund_net": float(fund_net), "exit_reason": reason,
            })
            i = max(exit_fill, entry_i + 1)                    # single-position: no overlap
        return trades

    @staticmethod
    def _window_stats(trades: list, window: str) -> WindowStats:
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            return WindowStats(window=window, compound_pct=0.0, n_trades=0, win_rate=0.0, max_dd_pct=0.0)
        rets = np.array([t["net_pnl"] for t in sub])
        eq = np.cumprod(1.0 + rets)
        comp = float((eq[-1] - 1.0) * 100)
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100)
        return WindowStats(window=window, compound_pct=comp, n_trades=len(sub),
                           win_rate=float((rets > 0).mean()), max_dd_pct=dd,
                           fund_net_sum=float(sum(t["fund_net"] for t in sub)))

    def run(self) -> SetupResults:
        # robust bool coercion (bool / 0-1 / NaN) without the object-dtype fillna downcast warning
        entry = pd.to_numeric(self.df[self.entry_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
        trades = self._simulate(entry)
        ws = {w: self._window_stats(trades, w) for w in self.WINDOWS}
        all4 = all(ws[w].compound_pct > 0 for w in self.WINDOWS)
        return SetupResults(trades=trades, window_stats=ws, all_4_positive=all4)

    # ------------------------------------------------------------------
    # MEASURED look-ahead guard: self-contained relative lead/lag test
    # ------------------------------------------------------------------
    def _compound_with_shifted_entry(self, shift_bars: int) -> dict:
        df2 = self.df.copy()
        df2[self.entry_col] = df2[self.entry_col].shift(shift_bars)
        h2 = SetupHarness(df2, self.entry_col, self.policy, self.windows,
                          cost_rt=self.spec.cost_rt, use_funding=self.use_funding,
                          funding_col=self.funding_col, funding_scale=self.funding_scale)
        res = h2.run()
        return {w: float(res.window_stats[w].compound_pct) for w in self.WINDOWS}

    def leak_guard(self, ratio_threshold: float = 1.5, held=("OOS", "UNSEEN"),
                   min_edge_pp: float = 2.0) -> dict:
        """Self-contained RELATIVE lead/lag leak verdict.

        base   = compound(entry as given)
        lag1   = compound(entry shifted +1)  -- one bar MORE lag (still strictly past-only)
        lead1  = compound(entry shifted -1)  -- one bar of FUTURE injected into the entry

        For a GENUINELY past-only entry WITH a real edge, injecting the future (lead1) hands the strategy
        a peek it does not currently have -> held-out compound moves a LOT, while adding lag (lag1) moves
        it only a little. ratio = |lead1-base| / max(|lag1-base|, eps). Large ratio = the future is NOT
        already baked in = PAST_ONLY_OK. Ratio ~1 (future barely changes it) = the entry is ALREADY as
        good as the future-peek version = it likely encodes the future = LEAK_SUSPECT. The two arms share
        the cadence's shift-sensitivity noise floor, so it cancels (the cadence-robust fix the absolute-pp
        leak_probe could not achieve -- see wealth_bot/leak_probe.py).

        SCOPE (RWYB 2026-06-05, real BTC 1d): the ratio is only diagnostic when there is a POSITIVE
        held-out edge to defend. For a no-edge / negative-base config the lead & lag deltas are both just
        noise and the ratio is meaningless (a strictly past-only 20-bar breakout that simply has no edge
        gave a spurious low ratio). So the verdict is rendered ONLY when held_base >= min_edge_pp;
        otherwise it returns INSUFFICIENT_EDGE and DEFERS to the structural guarantee (which is
        leak-proof by construction regardless). This keeps the guard from crying leak on honest no-edge
        configs while still firing on a genuinely leaked POSITIVE result.
        """
        base = {w: float(self.run().window_stats[w].compound_pct) for w in self.WINDOWS}
        lag1 = self._compound_with_shifted_entry(+1)
        lead1 = self._compound_with_shifted_entry(-1)
        held_base = sum(base[w] for w in held)
        held_lag = sum(lag1[w] for w in held)
        held_lead = sum(lead1[w] for w in held)
        lag_delta = abs(held_base - held_lag)
        lead_delta = abs(held_lead - held_base)
        eps = 1e-9
        ratio = lead_delta / max(lag_delta, eps)
        if held_base < min_edge_pp:
            verdict = ("INSUFFICIENT_EDGE (no positive held-out edge to leak-test; "
                       "ratio is noise here -- rely on the structural guarantee)")
        elif ratio < ratio_threshold:
            verdict = "LEAK_SUSPECT (future injection barely changes result -> entry already encodes future)"
        else:
            verdict = "PAST_ONLY_OK (future injection moves result far more than added lag)"
        return {
            "verdict": verdict,
            "held_compound_base_pp": round(held_base, 2),
            "held_compound_lag+1_pp": round(held_lag, 2),
            "held_compound_lead-1_pp": round(held_lead, 2),
            "lag_delta_pp": round(lag_delta, 2),
            "lead_injection_delta_pp": round(lead_delta, 2),
            "ratio_lead_over_lag": (round(ratio, 3) if ratio != float("inf") else "inf"),
            "ratio_threshold": ratio_threshold,
            "structural_guarantee": "entry fill=opens[i+1]; breach via highs/lows; pessimistic stop-first & prior-hwm trail",
        }


# ===========================================================================
# RWYB self-test (synthetic; validates the harness + the leak guard, NO market data)
# ===========================================================================
def _make_setup_frame(seed=3, start="2022-01-01", end="2026-05-22"):
    """Daily OHLC with a GENUINE, past-only, multi-candle setup edge.

    Construction: occasional 'dip' bars (a >4% down close). After a dip, the next ~6 bars drift UP
    (mean-reversion bounce). A DIP-BUY setup = 'this bar closed >4% down' is confirmed at close, filled
    next open, and captures the bounce MOVE. The signal is strictly past-only (uses only this bar's
    close vs the prior close). Random entries catch dips and non-dips alike -> no bounce -> the setup's
    MOVE-CAPTURE genuinely beats random. Repeats across all 4 windows.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)          # mild positive baseline drift + noise
    # inject dips followed by bounces
    bounce_left = 0
    for t in range(1, n):
        if bounce_left > 0:
            rets[t] += 0.010                      # bounce drift while active
            bounce_left -= 1
        elif rng.random() < 0.04:                 # ~4% of bars start a dip->bounce
            rets[t] -= 0.060                       # the dip bar itself (a sharp down close)
            bounce_left = 6                        # next 6 bars drift up
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    # past-only dip setup: this bar's close is >4% below the prior close (confirmed at close)
    prev_close = df["close"].shift(1)
    df["dip_setup"] = (df["close"] / prev_close - 1.0 < -0.04).fillna(False)
    return df


def _selftest():
    from wealth_bot.harness import WindowSpec
    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    df = _make_setup_frame()
    policy = ExitPolicy(tp_pct=0.08, sl_pct=0.04, max_hold_bars=8)
    print("=" * 78)
    print("[setup_harness selftest]  (synthetic; no market data; validates harness + leak guard)")
    print("=" * 78)

    # (a) GENUINE dip-buy setup -> positive held-out compound + leak-clean
    h = SetupHarness(df, "dip_setup", policy, win, cost_rt=0.0024)
    res = h.run()
    print("\n(a) GENUINE past-only dip->bounce setup:")
    print(res.summary())
    lg = h.leak_guard()
    print(f"    leak_guard: {lg['verdict']}")
    print(f"      base_held={lg['held_compound_base_pp']}pp  lag+1={lg['held_compound_lag+1_pp']}pp  "
          f"lead-1={lg['held_compound_lead-1_pp']}pp  ratio={lg['ratio_lead_over_lag']}")
    genuine_positive = res.window_stats["UNSEEN"].compound_pct > 0 and res.window_stats["OOS"].compound_pct > 0
    genuine_leakclean = lg["verdict"].startswith("PAST_ONLY_OK")

    # (b) NO-EDGE random entry (same base rate, but RANDOM bars) -> ~0 / not systematically positive
    rng = np.random.default_rng(99)
    df_rand = df.copy()
    rate = float(df["dip_setup"].mean())
    df_rand["rand_setup"] = (rng.random(len(df)) < rate)
    hr = SetupHarness(df_rand, "rand_setup", policy, win, cost_rt=0.0024)
    rr = hr.run()
    print("\n(b) NO-EDGE random entry (same base rate, random bars):")
    print(rr.summary())
    random_not_all4 = not rr.all_4_positive
    # the genuine setup should beat random on held-out compound
    genuine_beats_random = (res.window_stats["UNSEEN"].compound_pct + res.window_stats["OOS"].compound_pct) \
        > (rr.window_stats["UNSEEN"].compound_pct + rr.window_stats["OOS"].compound_pct)

    # (c) DELIBERATELY LEAKED entry: a setup that peeks one bar into the FUTURE -> guard must FIRE
    df_leak = df.copy()
    # "enter when the NEXT bar will close up >1%" -- pure look-ahead, built by shifting a future condition back
    future_up = (df["close"].shift(-1) / df["close"] - 1.0 > 0.01)
    df_leak["leaked_setup"] = future_up.fillna(False)
    hl = SetupHarness(df_leak, "leaked_setup", policy, win, cost_rt=0.0024)
    rl = hl.run()
    lgl = hl.leak_guard()
    print("\n(c) DELIBERATELY LEAKED entry (peeks 1 bar into the future):")
    print(rl.summary())
    print(f"    leak_guard: {lgl['verdict']}")
    print(f"      base_held={lgl['held_compound_base_pp']}pp  lag+1={lgl['held_compound_lag+1_pp']}pp  "
          f"lead-1={lgl['held_compound_lead-1_pp']}pp  ratio={lgl['ratio_lead_over_lag']}")
    leak_fires = lgl["verdict"].startswith("LEAK_SUSPECT")

    print("\n" + "-" * 78)
    print("SOUNDNESS (two-sided):")
    print(f"  (a) genuine setup positive on held-out (OOS&UNSEEN) : {genuine_positive}")
    print(f"  (a) genuine setup leak-clean (PAST_ONLY_OK)         : {genuine_leakclean}")
    print(f"  (b) random entry NOT all-4-positive                 : {random_not_all4}")
    print(f"  (b) genuine setup BEATS random on held-out compound : {genuine_beats_random}")
    print(f"  (c) leaked entry FIRES the leak guard               : {leak_fires}")
    ok = genuine_positive and genuine_leakclean and genuine_beats_random and leak_fires
    print(f"\n[setup_harness selftest] {'PASS' if ok else 'CHECK'} -- "
          f"{'gate accepts a genuine move-capture, rejects random, and catches a future leak.' if ok else 'see flags above.'}")
    return ok


# ===========================================================================
# RWYB on REAL chimera data (BTC 1d N-bar breakout setup -> TP/SL/time policy)
# ===========================================================================
def _rwyb():
    import json
    from pipeline.chimera_loader import ChimeraLoader
    from wealth_bot.harness import WindowSpec
    try:
        from .firewall import random_entry_null
        from .battery import evaluate
    except ImportError:
        from strat.firewall import random_entry_null
        from strat.battery import evaluate

    print("=" * 78)
    print("[setup_harness RWYB] BTC 1d -- 20-bar BREAKOUT setup -> (TP +12% / SL -6% / 20-bar time) policy")
    print("=" * 78)
    g = ChimeraLoader().load("BTCUSDT", cadence="1d")
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    df = pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float), "high": np.asarray(d["high"], float),
                       "low": np.asarray(d["low"], float), "close": np.asarray(d["close"], float)})

    # PAST-ONLY breakout setup: close exceeds the max CLOSE of the prior 20 bars (confirmed at close).
    # rolling(20).max().shift(1) uses bars [t-20 .. t-1] -> strictly prior; compared to close[t]. Fill opens[t+1].
    prior_max = df["close"].rolling(20).max().shift(1)
    df["breakout"] = (df["close"] > prior_max).fillna(False)
    n_setups = int(df["breakout"].sum())
    print(f"  setups (breakout bars): {n_setups} / {len(df)} bars\n")

    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    policy = ExitPolicy(tp_pct=0.12, sl_pct=0.06, max_hold_bars=20)
    h = SetupHarness(df, "breakout", policy, win, cost_rt=0.0024)
    res = h.run()
    print(res.summary())

    print("\n  -- built-in leak guard (self-contained lead/lag relative test) --")
    lg = h.leak_guard()
    print(json.dumps(lg, indent=2, default=str))

    print("\n  -- apparatus reuse: cost-matched random-ENTRY firewall (regime-matched on setup-ON bars) --")
    fw = random_entry_null(h, n_books=300, seed=7, regime_matched=True)
    for w, r in fw["per_window"].items():
        print(f"     {w:6} real={r['real']:>+8}%  null_p50={r['null_p50']}  null_p95={r['null_p95']}  "
              f"beats_null={r['beats_null']}  n={r['n_trades']}")
    print(f"     beats_held={fw['beats_held']}  pos_held={fw['pos_held']}  [{fw['regime_mode']}]")
    print(f"     VERDICT: {fw['verdict']}")

    print("\n  -- apparatus reuse: robustness battery on UNSEEN trades (Lens A/B/C) --")
    uns = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
    comps = {w: res.window_stats[w].compound_pct for w in h.WINDOWS}
    uns_dd = res.window_stats["UNSEEN"].max_dd_pct
    bat = evaluate(uns, comps, uns_dd, family_n=1)
    print(f"     battery verdict={bat['verdict']}  n={bat['n']} n_eff={bat['n_eff']} jk3={bat['jk3']} p05={bat['p05']}")

    # LEAK-GUARD POWER on REAL data (two-sided): inject a deliberate 1-bar future peek into a real BTC
    # setup -> positive base + the guard MUST fire LEAK_SUSPECT. Pairs with the honest no-edge breakout
    # above (INSUFFICIENT_EDGE -> defers). Proves the measured guard discriminates on real bars too.
    print("\n  -- leak-guard POWER check on REAL data: a deliberately future-peeking setup must FIRE --")
    df_leak = df.copy()
    # peek ALIGNED to the policy: enter now iff price will be >15% higher in 10 bars -> the +12% TP is
    # near-certain to fill within the 20-bar hold -> strongly POSITIVE base (the regime the guard scores).
    df_leak["peek"] = (df["close"].shift(-10) / df["close"] - 1.0 > 0.15).fillna(False)
    hlk = SetupHarness(df_leak, "peek", ExitPolicy(tp_pct=0.12, sl_pct=0.06, max_hold_bars=20), win, cost_rt=0.0024)
    lgk = hlk.leak_guard()
    print(f"     leaked-setup leak_guard: {lgk['verdict']}")
    print(f"       base_held={lgk['held_compound_base_pp']}pp  lag+1={lgk['held_compound_lag+1_pp']}pp  "
          f"lead-1={lgk['held_compound_lead-1_pp']}pp  ratio={lgk['ratio_lead_over_lag']}")
    real_leak_fires = lgk["verdict"].startswith("LEAK_SUSPECT")
    print(f"     -> guard fires on real future-leak: {real_leak_fires}")

    print("\n[setup_harness RWYB] DONE. The harness scored a multi-candle SETUP->MOVE on REAL held-out "
          "data WITHOUT touching IC, ran its built-in leak guard, and plugged unchanged into the existing "
          "firewall + battery. (This is the MEASUREMENT apparatus -- it endorses no strategy; the breakout "
          "config is a demonstration substrate, read the per-window numbers, not a ship claim.)")
    return {"window_stats": {w: res.window_stats[w].compound_pct for w in h.WINDOWS},
            "leak_guard": lg["verdict"], "firewall": fw["verdict"], "battery": bat["verdict"],
            "real_leak_power_fires": real_leak_fires}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = _selftest()
        sys.exit(0 if ok else 1)
    else:
        _selftest()
        print()
        _rwyb()
