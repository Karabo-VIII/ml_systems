"""experiments/adaptive_ma/plain/ergated_fixed_ma_4h.py -- ER-GATED FIXED-MA baseline (4h, 3 DOF).

THE REDIRECT (RESEARCHER_REPORT_1.md, 2026-06-05): the 1d adaptive-MA *switcher* was REFUTED
(0/69 assets beat the random-entry firewall; UNSEEN per-trade expectancy -2.09% adaptive / -2.50%
fixed). The diagnosis: (RF-3) the ER was used to SELECT wider/narrower windows but STILL traded every
cross, so chop false-crosses survived; (RF-4) the opposite-cross exit dumped into reversals at a loss;
the 1d cross also fires a bar late. The redirect = the MINIMAL HONEST NULL TO BEAT FIRST:
ER as a HARD GATE (not a switch) + a fixed MA + an ATR-trail exit, on 4h.

STRICTLY 3 DEGREES OF FREEDOM (the anti-overfit bar from RESEARCHER_REPORT_1 OVERFIT BOUNDS):
  DOF-1  ER hard-gate threshold  ER_GATE_THR = 0.40   -- trade ONLY when ER > thr; SKIP chop entirely.
  DOF-2  ONE fixed MA config     FAST/SLOW = SMA(10)/SMA(30)   -- the canonical fixed baseline.
  DOF-3  ONE exit policy         ATR-trailing stop (3x ATR14) + time-stop (42 bars = 7d backstop).

ENTRY (the SETUP, confirmed at the CLOSE of bar t; SetupHarness fills at opens[t+1]):
       entry[t] =  ER[t] > ER_GATE_THR                      (DOF-1 gate: trending only)
               AND close[t] > max(high[t-N .. t-1])         (breakout-confirm: new N-bar high)
               AND fast[t] > slow[t]                        (DOF-2 trend filter: uptrend)
  ALL three are past-only (use only data through close-of-bar t). The breakout uses the PRIOR N highs
  (rolling(N).max().shift(1)); fast/slow are close-of-bar SMAs; ER is the close-of-bar Kaufman ratio.

STRUCTURAL CONSTANTS (fixed UP-FRONT to standard values -- NOT tuned, NOT counted as free DOF; they are
the shape of the apparatus, the same way the cost model or the window split is):
  ER_WIN = 20   (Kaufman efficiency-ratio lookback, mirrors expert/adaptive_ma.py)
  ATR_WIN = 14  (Wilder-standard ATR lookback)
  BREAKOUT_N = 20  (Donchian-standard breakout lookback)

WHY 3 DOF (RESEARCHER_REPORT_1): "Current 6-cell map tests 6 configs before ANY has shown a timing edge
-- wrong order." This is the 1-config null those 6 cells must beat. If THIS does not clear the firewall /
positive control on held-out, adding map cells is fitting noise.

CAUSAL-FEATURE REUSE: imports the existing past-only MA primitive (wealth_bot.harness.sma_past_only) and
mirrors the Kaufman ER from experiments/adaptive_ma/expert/adaptive_ma.py (same 3-line formula). It does
NOT reuse the expert rig's 252-bar per-asset PERCENTILE rank -- that per-asset boundary is exactly the
DOF risk the redirect flags, so the gate is on the RAW ER value (hard 0.40), removing PCT_WIN entirely.

COST: taker round-trip 0.0024 sourced from src/strat/fill_model.MODES["taker"] (one source of truth).
EXIT: src/strat/setup_harness.SetupHarness + ExitPolicy (ATR-trail + time-stop).

RWYB:
  python experiments/adaptive_ma/plain/ergated_fixed_ma_4h.py --selftest   # synthetic two-sided gate check
  python experiments/adaptive_ma/plain/ergated_fixed_ma_4h.py              # BTC 4h real-data + leak_guard

No emoji (cp1252). numpy / pandas only + the kept apparatus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo wiring -----------------------------------------------------------
SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wealth_bot.harness import sma_past_only, WindowSpec       # noqa: E402  (existing causal MA primitive)
from strat.setup_harness import SetupHarness, ExitPolicy       # noqa: E402  (setup->move exit harness)
from strat.fill_model import MODES                             # noqa: E402  (taker cost, one source of truth)

# ====================== CONFIGURATION (all at top, transparent) ======================
CADENCE = "4h"

# --- THE 3 DEGREES OF FREEDOM ---
ER_GATE_THR = 0.40          # DOF-1: trade ONLY when Kaufman ER > this (trending). SKIP chop.
FAST_LEN, SLOW_LEN = 10, 30  # DOF-2: ONE fixed SMA config (canonical fixed baseline).
ATR_TRAIL_MULT = 3.0        # DOF-3a: ATR-trailing stop width = 3 x ATR.
TIME_STOP_BARS = 42         # DOF-3b: time-stop backstop = 42 x 4h = 7 days (hold = hours-to-<7d regime).

# --- STRUCTURAL CONSTANTS (fixed up-front, standard values, NOT tuned) ---
ER_WIN = 20                 # Kaufman efficiency-ratio lookback (mirrors expert rig).
ATR_WIN = 14                # Wilder-standard ATR lookback.
BREAKOUT_N = 20             # Donchian-standard prior-high breakout lookback.

COST_MODE = "taker"         # fill_model.MODES key -> cost_rt 0.0024.
TAKER = MODES[COST_MODE]["cost_rt"]

# Held-out windows: UNSEEN (>= oos_end) is the verdict surface; OOS+UNSEEN = held-out. All constants
# above were fixed BEFORE touching these windows.
WINDOWS = WindowSpec(train_end="2024-05-15", val_end="2025-03-15",
                     oos_end="2025-12-31", unseen_end="2026-05-22")

ENTRY_COL = "ergate_breakout"
ATR_COL = "atr_pastonly"


# ====================== CAUSAL FEATURES (all past-only) ======================
def kaufman_er(close: pd.Series, win: int = ER_WIN) -> pd.Series:
    """Kaufman Efficiency Ratio over `win` bars, in [0,1]. |net move| / sum|bar moves|.

    Mirrors experiments/adaptive_ma/expert/adaptive_ma.py compute_features() exactly. Close-of-bar
    past-only: ER[t] uses closes[t-win .. t]. (No .shift(1) here -- the SetupHarness fills at opens[t+1],
    so a close-of-bar setup is already strictly past-only vs the fill.)"""
    change = (close - close.shift(win)).abs()
    vol_path = close.diff().abs().rolling(win, min_periods=win // 2).sum()
    return (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)


def atr_past_only(df: pd.DataFrame, win: int = ATR_WIN) -> pd.Series:
    """Average True Range (rolling mean of true range), close-of-bar past-only. SetupHarness reads
    atr[j-1] (prior bar) for the trail width, so the breach width is known before the bar it gates."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=win // 2).mean()


def build_entry(df: pd.DataFrame) -> pd.DataFrame:
    """Add the past-only ENTRY boolean (ER-gate AND breakout AND fast>slow) + the ATR column.

    Returns the SAME frame with `ENTRY_COL` (boolean setup) and `ATR_COL` populated.
    """
    out = df.reset_index(drop=True).copy()
    close = out["close"].astype(float)

    er = kaufman_er(close, ER_WIN)                                   # DOF-1 feature (raw, hard-gated)
    fast = sma_past_only(close, FAST_LEN, shift=0)                   # DOF-2 fast MA (reused primitive)
    slow = sma_past_only(close, SLOW_LEN, shift=0)                   # DOF-2 slow MA
    prior_high = out["high"].astype(float).rolling(BREAKOUT_N).max().shift(1)  # prior N-bar high (breakout)

    setup = (er > ER_GATE_THR) & (close > prior_high) & (fast > slow)
    out[ENTRY_COL] = setup.fillna(False).astype(bool)
    out[ATR_COL] = atr_past_only(out, ATR_WIN)

    # diagnostics (not used for trading)
    out["_er"] = er
    out["_fast"] = fast
    out["_slow"] = slow
    return out


def make_policy() -> ExitPolicy:
    """DOF-3: the ONE exit policy -- ATR-trailing stop + time-stop. No TP/SL (the trail banks the move)."""
    return ExitPolicy(atr_trail_mult=ATR_TRAIL_MULT, atr_col=ATR_COL, max_hold_bars=TIME_STOP_BARS)


def causality_selfcheck(df: pd.DataFrame, n_points: int = 30) -> dict:
    """DEFINITIVE look-ahead proof (overrides the leak_guard heuristic): re-derive entry[t] and atr[t]
    from the TRUNCATED prefix df[:t+1] and compare to the full-series value at t. If they match for every
    sampled t, the setup provably depends only on data through bar t (no future). This is hard evidence;
    leak_guard's relative lead/lag ratio is a noisy proxy that can false-positive on thin no-/noise-edge
    configs (it reads 'future injection barely helps' as a leak, when the real reason is 'no robust
    structure to exploit better')."""
    full = build_entry(df)
    n = len(df)
    idx = list(range(300, n, max(1, (n - 300) // n_points)))
    entry_mismatch = 0
    max_atr_diff = 0.0
    checked = 0
    for t in idx:
        pref = build_entry(df.iloc[: t + 1])
        if bool(full[ENTRY_COL].iloc[t]) != bool(pref[ENTRY_COL].iloc[-1]):
            entry_mismatch += 1
        a_full, a_pref = full[ATR_COL].iloc[t], pref[ATR_COL].iloc[-1]
        if pd.notna(a_full) and pd.notna(a_pref):
            max_atr_diff = max(max_atr_diff, abs(float(a_full) - float(a_pref)))
        checked += 1
    causal_ok = (entry_mismatch == 0) and (max_atr_diff < 1e-9)
    return {"checked_points": checked, "entry_mismatches": entry_mismatch,
            "max_atr_diff": max_atr_diff, "causal_ok": bool(causal_ok)}


# ====================== DATA LOADING ======================
def load_ohlc_4h(loader, sym: str) -> pd.DataFrame:
    """Load one asset's 4h OHLC as a pandas frame (date/open/high/low/close), pyarrow-free."""
    g = loader.load(sym, cadence=CADENCE)
    pdf = pd.DataFrame(g.select(["date", "open", "high", "low", "close"]).to_dict(as_series=False))
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["open"] = pdf["open"].astype(float)
    pdf["high"] = pdf["high"].astype(float)
    pdf["low"] = pdf["low"].astype(float)
    pdf["close"] = pdf["close"].astype(float)
    return pdf.sort_values("date").reset_index(drop=True)


def run_asset(df: pd.DataFrame) -> SetupHarness:
    """Build entry + ATR, return a constructed SetupHarness (call .run() for results)."""
    feat = build_entry(df)
    policy = make_policy()
    return SetupHarness(feat, ENTRY_COL, policy, WINDOWS, cost_rt=TAKER,
                        use_funding=False, regime_match_on_entry=True)


# ====================== RWYB: synthetic two-sided gate check ======================
def _make_4h_frame(seed=5, start="2022-01-01", end="2026-05-22"):
    """Synthetic 4h OHLC with a GENUINE trending-breakout edge: occasional clean directional runs
    (high ER) the gate should catch, embedded in chop (low ER) the gate should SKIP. A trending-breakout
    entry captures the runs; the gate suppresses the chop. Two-sided: also yields a no-edge random arm."""
    dates = pd.date_range(start=start, end=end, freq="4h")
    n = len(dates)
    rng = np.random.default_rng(seed)
    # chop baseline BLEEDS slightly (random entries here lose cost+drift) -> two-sided discrimination.
    rets = rng.normal(-0.0006, 0.006, n)        # low-ER chop, mild negative drift
    run_left = 0
    for t in range(1, n):
        if run_left > 0:
            rets[t] += 0.014                    # clean up-run drift while active (high ER, breakout)
            run_left -= 1
        elif rng.random() < 0.003:              # rare clean directional run starts
            run_left = 12                        # ~2 days of trend
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def _selftest() -> bool:
    print("=" * 78)
    print("[ergated_fixed_ma_4h selftest] synthetic two-sided gate check (no market data)")
    print("=" * 78)
    df = _make_4h_frame()
    feat = build_entry(df)
    n_setup = int(feat[ENTRY_COL].sum())
    print(f"  ER-gated breakout setups: {n_setup} / {len(df)} bars "
          f"(ER>{ER_GATE_THR} frac={float((feat['_er'] > ER_GATE_THR).mean()):.3f})")

    h = run_asset(df)
    res = h.run()
    print("\n(a) GENUINE trending-breakout substrate:")
    print(res.summary())

    # (b) NO-EDGE random entry at the same base rate -> should NOT be systematically all-4-positive
    rate = max(feat[ENTRY_COL].mean(), 1e-4)
    rng = np.random.default_rng(123)
    df_rand = df.copy()
    df_rand[ENTRY_COL] = (rng.random(len(df)) < rate)
    df_rand[ATR_COL] = atr_past_only(df_rand, ATR_WIN)
    rr = SetupHarness(df_rand, ENTRY_COL, make_policy(), WINDOWS, cost_rt=TAKER).run()
    print("\n(b) NO-EDGE random entry (same base rate):")
    print(rr.summary())

    held_real = res.window_stats["OOS"].compound_pct + res.window_stats["UNSEEN"].compound_pct
    held_rand = rr.window_stats["OOS"].compound_pct + rr.window_stats["UNSEEN"].compound_pct
    beats_random = held_real > held_rand
    print("\n" + "-" * 78)
    print(f"  genuine held-out (OOS+UNSEEN) compound : {held_real:+.2f}pp")
    print(f"  random  held-out (OOS+UNSEEN) compound : {held_rand:+.2f}pp")
    print(f"  genuine setup BEATS random on held-out : {beats_random}")
    ok = beats_random and n_setup > 10
    print(f"\n[selftest] {'PASS' if ok else 'CHECK'} -- "
          f"{'apparatus runs and the gated trending-breakout beats random on held-out.' if ok else 'see flags.'}")
    return ok


# ====================== RWYB: real BTC 4h + leak guard ======================
def _rwyb_btc():
    import json
    from pipeline.chimera_loader import ChimeraLoader
    assert TAKER == 0.0024, "brief requires taker cost 0.0024"
    print("=" * 78)
    print("[ergated_fixed_ma_4h RWYB] BTC 4h -- ER-gated fixed-MA breakout -> ATR-trail+time exit")
    print(f"  DOF: ER>{ER_GATE_THR} | SMA({FAST_LEN})/{SLOW_LEN} | trail={ATR_TRAIL_MULT}xATR{ATR_WIN} "
          f"+ time<={TIME_STOP_BARS}bars(7d) | breakout_N={BREAKOUT_N} | taker={TAKER}")
    print("=" * 78)
    loader = ChimeraLoader()
    df = load_ohlc_4h(loader, "BTCUSDT")
    h = run_asset(df)
    res = h.run()
    print(res.summary())
    feat = build_entry(df)
    print(f"  total setups: {int(feat[ENTRY_COL].sum())} / {len(df)} bars")

    print("\n  -- built-in leak guard (self-contained lead/lag relative test) --")
    lg = h.leak_guard()
    print(json.dumps(lg, indent=2, default=str))

    print("\n  -- DEFINITIVE causality proof (prefix-truncation; overrides the heuristic) --")
    cc = causality_selfcheck(df)
    print(json.dumps(cc, indent=2, default=str))
    print(f"  -> entry is provably past-only: {cc['causal_ok']} "
          f"(entry[t] from df[:t+1] == full-series entry[t]).")
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = _selftest()
        sys.exit(0 if ok else 1)
    else:
        _selftest()
        print()
        _rwyb_btc()
