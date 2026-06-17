"""experiments/adaptive_ma/plain/adaptive_ma_plain.py -- PLAIN-RIG adaptive-MA scaffold.

Brief: docs/ADAPTIVE_MA_BRIEF_2026_06_05.md

WHAT THIS DOES (end-to-end, RWYB):
  1. Load each u100 asset's 1d bars via ChimeraLoader.load(sym, '1d').
  2. Compute per-asset CAUSAL rolling features (past-only, NO look-ahead):
       - realized_vol  : trailing std of log-returns (FEAT_VOL_WIN bars)
       - trend_slope   : trailing OLS slope of close, normalized by price (FEAT_TREND_WIN)
       - dispersion    : trailing mean of intrabar range (high-low)/close (FEAT_DISP_WIN)
  3. Map features -> an adaptive MA config (fast/slow SMA lengths) AT EACH BAR via a
     causal percentile blend (vol + dispersion -> faster; strong trend -> slower).
  4. Build the adapted fast/slow MA lines by selecting, per bar, the precomputed SMA
     of the chosen length (all SMAs are close-of-bar past-only, shift=0).
  5. Entries: adapted MA crossover (ama_fast > ama_slow) via the CanonicalHarness.
  6. Exit: a SINGLE UNIFORM policy for every asset/config -- opposite-cross
     (signal_flip) with a uniform max-hold time-stop backstop.
  7. Route fills through src/strat/fill_model.py (taker, cost_rt=0.0024).
  8. Emit per-trade AFTER-COST returns on the TRAIN split ONLY; print n_trades + mean/median.

PAST-ONLY GUARANTEE:
  - Every feature at bar t uses only data through close-of-bar t (returns[t] uses close[t]).
  - The adaptive length at bar t is a trailing-window percentile (rolling rank) -- no full-sample stats.
  - Each candidate SMA is close-of-bar past-only; the CanonicalHarness fills at opens[t+1].
  So the length CHOICE and the MA VALUE at bar t are both available before the t+1 fill. No leakage.

SCOPE: this is the PLAIN rig's scaffold. It reports the TRAIN-only per-trade edge so the
adaptation logic can be iterated. It deliberately does NOT make ship claims, does NOT touch
VAL/OOS/UNSEEN, and does NOT commit anything. Held-out evaluation + null/baseline comparison
(firewall, benchmark, positive_control) are downstream steps per the brief.
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

from pipeline.chimera_loader import ChimeraLoader            # noqa: E402
from wealth_bot.harness import (                              # noqa: E402
    CanonicalHarness, StrategySpec, WindowSpec, sma_past_only,
)
from strat.fill_model import MODES                            # noqa: E402  (taker cost lives here)

# --- configuration (transparent, all at top) -------------------------------
UNIVERSE = "u100"
CADENCE = "1d"

# Feature windows (bars). Past-only trailing windows.
FEAT_VOL_WIN = 20
FEAT_TREND_WIN = 20
FEAT_DISP_WIN = 20
RANK_WIN = 252            # trailing window for the causal percentile rank of each feature

# Adaptive length ladder. Index 0 = SLOWEST config, last index = FASTEST.
# Each bucket's fast length is strictly < its slow length (valid crossover).
FAST_CANDIDATES = [20, 15, 10, 8, 5]
SLOW_CANDIDATES = [60, 45, 35, 25, 20]
N_BUCKETS = len(FAST_CANDIDATES)

# Feature blend -> speed score in [0,1]. Higher score = faster MAs (shorter lengths).
#   high realized-vol  -> faster ; high dispersion -> faster ; strong trend -> SLOWER (ride it).
W_VOL, W_DISP, W_TREND = 0.5, 0.3, 0.2

# Single UNIFORM exit policy for every asset/config.
EXIT_POLICY = "signal_flip"      # opposite-cross of the adapted MAs
UNIFORM_MAX_HOLD_BARS = 30       # uniform time-stop backstop (days)

COST_MODE = "taker"              # fill_model.MODES key -> cost_rt 0.0024, p_fill 1.0, adverse 0.0

WINDOWS = WindowSpec(
    train_end="2024-05-15",
    val_end="2025-03-15",
    oos_end="2025-12-31",
    unseen_end="2026-05-22",
)


# --- data loading ----------------------------------------------------------
def load_asset_1d(loader: ChimeraLoader, sym: str) -> pd.DataFrame:
    """Load one asset's 1d bars as a pandas frame with date/open/high/low/close.

    Uses to_dict(as_series=False) (pyarrow-free) per the fill_model._rwyb convention.
    """
    pl_df = loader.load(sym, cadence=CADENCE)
    pdf = pd.DataFrame(
        pl_df.select(["date", "open", "high", "low", "close"]).to_dict(as_series=False)
    )
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("date").reset_index(drop=True)
    return pdf


# --- causal features -------------------------------------------------------
def _causal_rank(series: pd.Series, win: int) -> pd.Series:
    """Trailing-window percentile rank (pct) of the current value. Past-only by construction:
    the value at t is its rank within [t-win+1 .. t]."""
    return series.rolling(win, min_periods=max(5, win // 4)).rank(pct=True)


def compute_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add realized_vol, trend_slope, dispersion (all past-only). Returns the same frame."""
    close = df["close"].astype(float)
    logret = np.log(close).diff()

    # realized volatility: trailing std of log-returns
    df["realized_vol"] = logret.rolling(FEAT_VOL_WIN, min_periods=FEAT_VOL_WIN // 2).std()

    # trend slope: OLS slope of close over a trailing window, normalized by price level.
    # rolling.apply with a fixed x-grid keeps it past-only (uses only the window's closes).
    x = np.arange(FEAT_TREND_WIN, dtype=float)
    x_c = x - x.mean()
    denom = float((x_c ** 2).sum())

    def _slope(window_vals: np.ndarray) -> float:
        y = window_vals - window_vals.mean()
        return float((x_c * y).sum() / denom)

    raw_slope = close.rolling(FEAT_TREND_WIN, min_periods=FEAT_TREND_WIN).apply(_slope, raw=True)
    df["trend_slope"] = raw_slope / close            # normalize by price -> comparable across assets

    # dispersion: trailing mean of intrabar range fraction
    rng_frac = (df["high"].astype(float) - df["low"].astype(float)) / close
    df["dispersion"] = rng_frac.rolling(FEAT_DISP_WIN, min_periods=FEAT_DISP_WIN // 2).mean()
    return df


# --- feature -> adaptive length mapping ------------------------------------
def map_features_to_lengths(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map causal features -> per-bar (fast_len, slow_len).

    speed_score = W_VOL*vol_pct + W_DISP*disp_pct + W_TREND*(1 - |trend|_pct)
       -> high vol / high dispersion / weak trend = faster (shorter) MAs.
    Percentiles are causal trailing-window ranks (no full-sample standardization).
    Bars with undefined features default to the middle bucket (no edge claimed there;
    the MA itself is still NaN during warmup so no trade fires).
    """
    vol_pct = _causal_rank(df["realized_vol"], RANK_WIN)
    disp_pct = _causal_rank(df["dispersion"], RANK_WIN)
    trend_strength = df["trend_slope"].abs()
    trend_pct = _causal_rank(trend_strength, RANK_WIN)

    speed = (W_VOL * vol_pct + W_DISP * disp_pct + W_TREND * (1.0 - trend_pct))
    speed = speed.fillna(0.5).clip(0.0, 1.0).to_numpy()

    bucket = np.floor(speed * N_BUCKETS).astype(int)
    bucket = np.clip(bucket, 0, N_BUCKETS - 1)

    fast_arr = np.array(FAST_CANDIDATES)[bucket]
    slow_arr = np.array(SLOW_CANDIDATES)[bucket]
    return fast_arr, slow_arr


def build_adaptive_mas(df: pd.DataFrame, fast_len: np.ndarray, slow_len: np.ndarray) -> pd.DataFrame:
    """Select, per bar, the precomputed SMA of the chosen length -> ama_fast / ama_slow columns.

    All candidate SMAs are close-of-bar past-only (shift=0). Selecting by per-bar length keeps
    causality: at bar t the chosen length is known from past-only features, and the SMA value
    uses only closes through t.
    """
    close = df["close"]
    lengths = sorted(set(FAST_CANDIDATES) | set(SLOW_CANDIDATES))
    sma_cols = {L: sma_past_only(close, length=L, shift=0).to_numpy() for L in lengths}
    n = len(df)
    ama_fast = np.full(n, np.nan)
    ama_slow = np.full(n, np.nan)
    for L in lengths:
        col = sma_cols[L]
        fmask = fast_len == L
        smask = slow_len == L
        ama_fast[fmask] = col[fmask]
        ama_slow[smask] = col[smask]
    df["ama_fast"] = ama_fast
    df["ama_slow"] = ama_slow
    return df


# --- per-asset run ---------------------------------------------------------
def run_asset(df: pd.DataFrame, sym: str):
    """Build the adaptive frame, run the harness, return (harness, results)."""
    df = compute_causal_features(df)
    fast_len, slow_len = map_features_to_lengths(df)
    df = build_adaptive_mas(df, fast_len, slow_len)

    spec = StrategySpec(
        fast_col="ama_fast",
        slow_col="ama_slow",
        signal="crossover",
        filter_col=None,                 # plain rig: no conditioning filter
        exit_policy=EXIT_POLICY,         # single uniform exit policy
        cost_rt=MODES[COST_MODE]["cost_rt"],   # taker 0.0024 sourced from fill_model
        use_funding=False,               # spot long-only -> no funding
        funding_scale=0.0,
        max_hold_bars=UNIFORM_MAX_HOLD_BARS,
        max_hold_ext_bars=None,          # uniform: no conditional extension
    )
    harness = CanonicalHarness(df, spec, WINDOWS, chimera_path=f"chimera_v51_1d::{sym}",
                               command_line=f"adaptive_ma_plain.run_asset({sym})")
    results = harness.run()
    return harness, results


def taker_after_cost(trade: dict, spec_cost: float) -> float:
    """Route ONE trade's fill through fill_model's taker mode, return its after-cost return.

    Mirrors apply_fill_model's per-trade transform exactly (taker = deterministic, p_fill 1.0,
    adverse 0.0): recover gross = net + spec_cost + fund, then new_net = gross - adverse*|gross|
    - cost_rt - fund. The cost constant is sourced from fill_model.MODES so there is one source
    of truth. For taker (adverse 0, fund 0 in spot) this equals gross - 0.0024.
    """
    m = MODES[COST_MODE]
    gross = trade["net_pnl"] + spec_cost + trade["fund_net"]
    return gross - m["adverse"] * abs(gross) - m["cost_rt"] - trade["fund_net"]


# --- main ------------------------------------------------------------------
def main() -> int:
    assert MODES[COST_MODE]["cost_rt"] == 0.0024, "brief requires taker cost 0.0024"
    loader = ChimeraLoader()
    syms = loader.universes.list(UNIVERSE)
    print(f"[adaptive_ma_plain] universe={UNIVERSE} cadence={CADENCE} assets={len(syms)}")
    print(f"[adaptive_ma_plain] exit_policy={EXIT_POLICY} max_hold={UNIFORM_MAX_HOLD_BARS} "
          f"cost_mode={COST_MODE} (cost_rt={MODES[COST_MODE]['cost_rt']})")
    print(f"[adaptive_ma_plain] length ladder fast={FAST_CANDIDATES} slow={SLOW_CANDIDATES}")

    train_returns: list[float] = []
    n_assets_ok = 0
    n_assets_skipped = 0
    per_asset_train_n: list[tuple[str, int]] = []

    for sym in syms:
        try:
            df = load_asset_1d(loader, sym)
        except Exception as e:                        # missing/broken data -> skip honestly
            n_assets_skipped += 1
            print(f"  SKIP {sym}: {type(e).__name__}: {e}")
            continue
        if len(df) < RANK_WIN + max(SLOW_CANDIDATES) + 5:
            n_assets_skipped += 1
            print(f"  SKIP {sym}: too few bars ({len(df)})")
            continue
        harness, results = run_asset(df, sym)
        spec_cost = float(harness.spec.cost_rt)
        train_trades = [t for t in results.trades if t["window"] == "TRAIN"]
        rets = [taker_after_cost(t, spec_cost) for t in train_trades]
        train_returns.extend(rets)
        per_asset_train_n.append((sym, len(rets)))
        n_assets_ok += 1

    print("-" * 72)
    print(f"[adaptive_ma_plain] assets_run={n_assets_ok} assets_skipped={n_assets_skipped}")
    arr = np.array(train_returns, dtype=float)
    n_trades = arr.size
    if n_trades == 0:
        print("[adaptive_ma_plain] TRAIN per-trade after-cost returns: NO TRADES")
        return 0

    mean_r = float(arr.mean())
    median_r = float(np.median(arr))
    win_rate = float((arr > 0).mean())
    print("[adaptive_ma_plain] === TRAIN-split per-trade AFTER-COST returns (taker 0.0024) ===")
    print(f"  n_trades                = {n_trades}")
    print(f"  mean   per-trade return = {mean_r:+.6f}  ({mean_r*100:+.4f}%)")
    print(f"  median per-trade return = {median_r:+.6f}  ({median_r*100:+.4f}%)")
    print(f"  win_rate                = {win_rate:.4f}")
    print(f"  std / p05 / p95         = {arr.std():.4f} / {np.percentile(arr,5):+.4f} / {np.percentile(arr,95):+.4f}")
    # top contributors by trade count (transparency)
    per_asset_train_n.sort(key=lambda kv: kv[1], reverse=True)
    top = ", ".join(f"{s}:{n}" for s, n in per_asset_train_n[:8])
    print(f"  top assets by TRAIN trade count: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
