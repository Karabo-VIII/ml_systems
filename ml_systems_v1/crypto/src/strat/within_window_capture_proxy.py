"""src/strat/within_window_capture_proxy.py -- the WITHIN-WINDOW CAPTURE-RATE PROXY on REAL data.

WHAT THIS IS (and how it differs from within_window_capture_gate.py)
-------------------------------------------------------------------
`within_window_capture_gate.py` proves, on SYNTHETIC moves, that the best-vs-worst-spread capture
statistic is sound (two-sided + calibrated detection floor). THIS module is the REAL-DATA proxy: given
an instrument/timeframe and a past-only TRIGGER (the setup firing), it measures -- for every move the
trigger fires in -- where the trigger's realized return landed inside the [worst, best] spread of all
candidate entry timings for that same move. The headline artifact is a per-window table of capture
values, each in [0,1].

THE UNIT (MEMORY.md founding framing): the unit of trading is a SETUP across a MOVE (multiple candles).
Per-bar IC is the WRONG lens. The RIGHT lens for entry-TIMING skill is: GIVEN a move you are present in,
did your TRIGGER land closer to the BEST-possible entry than random timing inside the same move would?

THE PROXY (best-vs-worst-spread capture, per move-window)
---------------------------------------------------------
For each move the trigger fires at bar t, define the move-WINDOW as the candidate entry bars
[t - win_radius, t + win_radius] (clamped to data; the trigger bar t is always included). Fix ONE
common exit bar  exit = (window_end) + horizon  so that EVERY candidate entry is chased to the SAME
exit -- only the ENTRY TIMING varies. With next-bar-open fills:
    ret(s) = open[exit] / open[s+1] - 1 - cost     for each candidate entry bar s in the window
    best  = max_s ret(s)      worst = min_s ret(s)      spread = best - worst
    capture = (ret(t) - worst) / spread    in [0,1]      (t = the trigger's actual entry bar)
capture = 1.0 -> the trigger timed the BEST candidate entry; 0.0 -> the worst; 0.5 -> the middle of the
available timing range. Degenerate windows (spread < min_spread) carry no timing decision and are dropped.

THE TRIPLE CONFOUND CONTROL (baked in)
--------------------------------------
1. HORIZON.  Every candidate entry inside a window is chased to ONE common exit bar (window_end+horizon).
   So differences in capture are NOT a horizon artifact (a late entry is not secretly given a longer or
   shorter hold than an early one beyond the timing it chose). Capture is computed at a FIXED horizon, and
   the proxy is run/reported per-horizon -- horizon is held constant within every comparison.
2. REGIME.   Each window is labelled by a PAST-ONLY regime bin (volatility tercile by default). The
   candidate pool for a window is restricted to bars sharing the trigger bar's regime bin (the trigger
   bar itself is always kept), and the random-timing NULL draws only from that same-regime pool. Capture
   is additionally reported stratified BY regime bin. So a trigger cannot be credited for capture that is
   really just "this move happened in an easy regime".
3. WINDOW-MEMBERSHIP.  Capture is defined ONLY within the move-window the trigger is actually present in,
   and the random-timing null re-draws the entry uniformly from that SAME window's candidate bars. The
   proxy therefore scores WHERE-inside-the-move timing, never WHICH-move selection (mirrors
   firewall.random_entry_null(membership_matched=True), but as a clean bounded [0,1] statistic).

ARTIFACT: a CSV (one row per move-window) + a JSON summary are written under runs/strat/. Every `capture`
value is bounded in [0,1] by construction (the trigger bar is always in its own candidate pool, and a
defensive clip is applied). The verify path asserts the artifact exists, all captures are in [0,1], and
exits 0.

RWYB (dry-run on one instrument/timeframe + self-verify):
    python src/strat/within_window_capture_proxy.py            # BTC 1d dry-run -> writes + verifies artifact
    python src/strat/within_window_capture_proxy.py --selftest # synthetic two-sided soundness (no market data)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
ARTIFACT_DIR = ROOT / "runs" / "strat"


# ---------------------------------------------------------------------------
# Window labelling (date-based, matches the rest of the apparatus)
# ---------------------------------------------------------------------------
@dataclass
class Windows:
    train_end: str = "2024-05-15"
    val_end: str = "2025-03-15"
    oos_end: str = "2025-12-31"

    def label(self, ts: pd.Timestamp) -> str:
        ts = pd.Timestamp(ts)
        if ts < pd.Timestamp(self.train_end):
            return "TRAIN"
        if ts < pd.Timestamp(self.val_end):
            return "VAL"
        if ts < pd.Timestamp(self.oos_end):
            return "OOS"
        return "UNSEEN"


# ---------------------------------------------------------------------------
# Regime labelling -- PAST-ONLY volatility tercile (confound control #2)
# ---------------------------------------------------------------------------
def past_only_regime_bins(df: pd.DataFrame, n_bins: int = 3, vol_window: int = 20) -> np.ndarray:
    """Assign each bar a regime bin in {0..n_bins-1} from a PAST-ONLY rolling volatility.

    vol[t] = std of the prior `vol_window` log-returns, available at close of bar t-1 (shift(1) makes the
    label strictly prior). Terciles are cut on the TRAIN-era distribution only would be ideal, but for a
    leak-safe *labelling* (not a fitted decision) we cut on global quantiles of the past-only series; the
    cut is a fixed monotone transform of a past-only quantity, so it injects no future return info into
    the per-window timing comparison (the comparison is WITHIN a bin, and the bin id at bar t uses only
    closes up to t-1). Bars before the window is defined get bin 0.
    """
    close = df["close"].to_numpy(float)
    logret = np.zeros(len(close))
    logret[1:] = np.diff(np.log(close))
    s = pd.Series(logret)
    vol = s.rolling(vol_window).std().shift(1)  # strictly prior -> past-only label
    v = vol.to_numpy(float)
    finite = np.isfinite(v)
    bins = np.zeros(len(v), dtype=int)
    if finite.sum() >= n_bins:
        qs = np.quantile(v[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
        bins[finite] = np.digitize(v[finite], qs)
    return bins


# ---------------------------------------------------------------------------
# THE PROXY
# ---------------------------------------------------------------------------
@dataclass
class CaptureProxyResult:
    rows: list                       # per-window dicts
    per_window_summary: dict         # {WINDOW: {...}}
    per_regime_summary: dict         # {bin: {...}}
    held_summary: dict
    config: dict
    rng_seed: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def capture_proxy(df: pd.DataFrame, trigger_col: str, *, horizon: int = 5, win_radius: int = 3,
                  cost: float = TAKER, min_spread: float = 0.003, windows: Windows | None = None,
                  regime_bins: np.ndarray | None = None, n_regime_bins: int = 3,
                  seed: int = 11) -> CaptureProxyResult:
    """Compute the within-window capture-rate proxy for every move the trigger fires in.

    df         : DataFrame with date, open, high, low, close (past-only by construction of trigger_col).
    trigger_col: boolean/0-1 column; True = the setup is CONFIRMED at the close of this bar (the trigger).
    horizon    : bars from the window END to the single common exit bar (held constant -> confound #1).
    win_radius : candidate entry bars on EACH side of the trigger bar form the move-window (membership).
    """
    windows = windows or Windows()
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    if regime_bins is None:
        regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)

    trig = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    trig_idx = np.flatnonzero(trig)

    rows = []
    for t in trig_idx:
        win_lo = max(0, t - win_radius)
        win_hi = min(n - 1, t + win_radius)
        window_end = win_hi
        exit_bar = window_end + horizon
        # need room for the last candidate's fill (s+1) and the common exit
        if exit_bar >= n:
            continue
        reg_t = int(regime_bins[t])
        # candidate entry bars: in-window AND same regime as the trigger; trigger bar always included
        cand = [s for s in range(win_lo, win_hi + 1)
                if s + 1 < n and (int(regime_bins[s]) == reg_t or s == t)]
        cand = np.array(sorted(set(cand)), dtype=int)
        if cand.size < 2:
            continue                                        # no timing decision (need >=2 candidates)
        rets = opens[exit_bar] / opens[cand + 1] - 1.0 - cost
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        if spread < min_spread:
            continue                                        # degenerate -> no timing decision -> drop
        # trigger's realized return = the candidate row where s == t
        ti = int(np.flatnonzero(cand == t)[0])
        trig_ret = float(rets[ti])
        capture = (trig_ret - worst) / spread
        capture = float(min(1.0, max(0.0, capture)))        # defensive clip -> guaranteed [0,1]
        # window-membership + regime matched random-timing null (analytic mean over same candidate pool)
        null_mean_capture = float(((rets - worst) / spread).mean())
        rows.append({
            "trigger_idx": int(t),
            "trigger_ts": str(pd.Timestamp(dates.iloc[t])),
            "window": windows.label(dates.iloc[t]),
            "regime_bin": reg_t,
            "horizon": int(horizon),
            "n_candidates": int(cand.size),
            "best_ret": round(best, 6),
            "worst_ret": round(worst, 6),
            "spread": round(spread, 6),
            "trigger_ret": round(trig_ret, 6),
            "capture": round(capture, 6),
            "null_mean_capture": round(null_mean_capture, 6),
        })

    # ---- Monte-Carlo random-timing null per window (membership+regime matched) ----
    def _mc_null_mean(sub_rows, n_books=2000):
        if not sub_rows:
            return None, None
        # reconstruct candidate ret-curves to redraw timing; store curves cheaply via stored stats only:
        # we use the analytic per-window null mean already stored, MC over windows to get the mean's band.
        caps = np.array([r["null_mean_capture"] for r in sub_rows])
        # bootstrap the mean of per-window analytic null means -> null band for the pooled mean capture
        idx = np.arange(len(caps))
        means = np.empty(n_books)
        for b in range(n_books):
            take = rng.choice(idx, size=len(idx), replace=True)
            means[b] = caps[take].mean()
        return float(np.percentile(means, 50)), float(np.percentile(means, 95))

    def _summ(sub_rows):
        if not sub_rows:
            return {"n_windows": 0, "mean_capture": None, "median_capture": None,
                    "null_p50": None, "null_p95": None, "beats_null": None}
        caps = np.array([r["capture"] for r in sub_rows])
        null_mean = float(np.mean([r["null_mean_capture"] for r in sub_rows]))
        p50, p95 = _mc_null_mean(sub_rows)
        mean_cap = float(caps.mean())
        return {
            "n_windows": len(sub_rows),
            "mean_capture": round(mean_cap, 4),
            "median_capture": round(float(np.median(caps)), 4),
            "null_analytic_mean": round(null_mean, 4),
            "null_p50": round(p50, 4) if p50 is not None else None,
            "null_p95": round(p95, 4) if p95 is not None else None,
            "beats_null": bool(mean_cap > p95) if p95 is not None else None,
        }

    per_window = {w: _summ([r for r in rows if r["window"] == w]) for w in WINDOWS}
    bins_present = sorted({r["regime_bin"] for r in rows})
    per_regime = {int(b): _summ([r for r in rows if r["regime_bin"] == b]) for b in bins_present}
    held = _summ([r for r in rows if r["window"] in HELD])

    config = {"trigger_col": trigger_col, "horizon": horizon, "win_radius": win_radius,
              "cost": cost, "min_spread": min_spread, "n_regime_bins": n_regime_bins,
              "confound_control": {"horizon": "single common exit per window (held constant)",
                                   "regime": "candidate pool + null restricted to trigger's past-only vol-tercile bin",
                                   "window_membership": "capture + null defined only within the move-window"}}
    return CaptureProxyResult(rows=rows, per_window_summary=per_window, per_regime_summary=per_regime,
                              held_summary=held, config=config, rng_seed=seed)


# ---------------------------------------------------------------------------
# Artifact emit + verify
# ---------------------------------------------------------------------------
def write_artifacts(res: CaptureProxyResult, tag: str, out_dir: Path = ARTIFACT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"capture_proxy_{tag}.csv"
    json_path = out_dir / f"capture_proxy_{tag}.json"
    df = res.to_frame()
    df.to_csv(csv_path, index=False)
    summary = {
        "tag": tag,
        "n_windows": len(res.rows),
        "config": res.config,
        "rng_seed": res.rng_seed,
        "per_window_summary": res.per_window_summary,
        "per_regime_summary": res.per_regime_summary,
        "held_summary": res.held_summary,
        "capture_min": (round(float(df["capture"].min()), 6) if len(df) else None),
        "capture_max": (round(float(df["capture"].max()), 6) if len(df) else None),
    }
    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.replace(json_path)
    return {"csv": str(csv_path), "json": str(json_path)}


def verify_artifact(csv_path: str | Path) -> tuple[bool, str]:
    """Assert the artifact exists, is non-empty, and every capture value is bounded in [0,1]."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, f"artifact missing: {csv_path}"
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return False, "artifact has zero rows (no move-windows emitted)"
    if "capture" not in df.columns:
        return False, "artifact has no 'capture' column"
    cap = df["capture"].to_numpy(float)
    if not np.all(np.isfinite(cap)):
        return False, "capture contains non-finite values"
    lo, hi = float(cap.min()), float(cap.max())
    if lo < 0.0 or hi > 1.0:
        return False, f"capture out of [0,1]: min={lo}, max={hi}"
    return True, f"OK: {len(df)} windows, capture in [{lo:.4f}, {hi:.4f}] (all bounded in [0,1])"


# ===========================================================================
# Dry-run on ONE instrument/timeframe (BTC 1d), write + verify the artifact
# ===========================================================================
def _load_btc_1d() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load("BTCUSDT", cadence="1d")
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                         "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                         "close": np.asarray(d["close"], float)})


def _dry_run() -> int:
    print("=" * 84)
    print("WITHIN-WINDOW CAPTURE-RATE PROXY -- dry-run on BTCUSDT 1d (one instrument/timeframe)")
    print("=" * 84)
    df = _load_btc_1d()
    print(f"  loaded {len(df)} bars: {df['date'].min().date()} -> {df['date'].max().date()}")

    # PAST-ONLY trigger (the setup): close exceeds prior-20-bar max close (a breakout confirmed at close).
    prior_max = df["close"].rolling(20).max().shift(1)
    df["breakout"] = (df["close"] > prior_max).fillna(False)
    print(f"  trigger = 20-bar breakout; firings = {int(df['breakout'].sum())}")

    res = capture_proxy(df, "breakout", horizon=5, win_radius=3, cost=TAKER)
    paths = write_artifacts(res, tag="BTCUSDT_1d_breakout_h5")
    print(f"\n  artifact CSV : {paths['csv']}")
    print(f"  artifact JSON: {paths['json']}")

    print("\n  per-window mean capture (held-out = OOS+UNSEEN) vs membership+regime-matched null p95:")
    for w in WINDOWS:
        s = res.per_window_summary[w]
        print(f"    {w:7} n={s['n_windows']:<4} mean_cap={s['mean_capture']}  "
              f"null_p95={s['null_p95']}  beats_null={s['beats_null']}")
    print("\n  per-regime (past-only vol tercile) mean capture:")
    for b, s in res.per_regime_summary.items():
        print(f"    bin {b}: n={s['n_windows']:<4} mean_cap={s['mean_capture']}  null_p95={s['null_p95']}  "
              f"beats_null={s['beats_null']}")
    hs = res.held_summary
    print(f"\n  HELD-OUT pooled: n={hs['n_windows']} mean_cap={hs['mean_capture']} "
          f"null_p95={hs['null_p95']} beats_null={hs['beats_null']}")

    ok, msg = verify_artifact(paths["csv"])
    print("\n" + "-" * 84)
    print(f"  VERIFY: {msg}")
    print(f"[within_window_capture_proxy] {'PASS' if ok else 'FAIL'} -- artifact emitted with all "
          f"per-window capture values bounded in [0,1].")
    return 0 if ok else 1


# ===========================================================================
# Synthetic two-sided soundness selftest (no market data)
# ===========================================================================
def _make_synth(seed=3, n=1400):
    """Daily OHLC with dip->bounce moves so a 'buy the dip' trigger has REAL within-window timing skill:
    entering at the dip bottom (best timing) beats entering at the window edges."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.010, n)
    bounce = 0
    is_dip = np.zeros(n, bool)
    for t in range(1, n):
        if bounce > 0:
            rets[t] += 0.012
            bounce -= 1
        elif rng.random() < 0.05:
            rets[t] -= 0.05
            is_dip[t] = True
            bounce = 6
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    df["dip"] = is_dip
    return df


def _selftest() -> int:
    print("=" * 84)
    print("[within_window_capture_proxy selftest] synthetic two-sided soundness (no market data)")
    print("=" * 84)
    df = _make_synth()
    win = Windows(train_end="2023-06-01", val_end="2024-01-01", oos_end="2024-09-01")

    # (A) a SKILLED trigger: buy the dip bottom -> should land HIGH capture vs random timing
    res = capture_proxy(df, "dip", horizon=5, win_radius=3, cost=TAKER, windows=win)
    skilled_mean = res.held_summary["mean_capture"]
    skilled_beats = res.held_summary["beats_null"]

    # (B) a NO-SKILL trigger: random bars at the same rate -> capture ~ null, should NOT beat null
    rng = np.random.default_rng(99)
    rate = float(df["dip"].mean())
    df["rand"] = rng.random(len(df)) < rate
    res_r = capture_proxy(df, "rand", horizon=5, win_radius=3, cost=TAKER, windows=win)
    rand_mean = res_r.held_summary["mean_capture"]
    rand_beats = res_r.held_summary["beats_null"]

    # bounds check on both artifacts
    paths = write_artifacts(res, tag="selftest_skilled")
    paths_r = write_artifacts(res_r, tag="selftest_random")
    ok_s, msg_s = verify_artifact(paths["csv"])
    ok_r, msg_r = verify_artifact(paths_r["csv"])

    print(f"\n  (A) SKILLED (buy-the-dip) held-out mean_capture={skilled_mean}  beats_null={skilled_beats}")
    print(f"  (B) NO-SKILL (random)     held-out mean_capture={rand_mean}  beats_null={rand_beats}")
    print(f"  (bounds) skilled artifact: {msg_s}")
    print(f"  (bounds) random  artifact: {msg_r}")

    bounds_ok = ok_s and ok_r
    discriminates = bool(skilled_beats) and not bool(rand_beats)
    ordered = (skilled_mean is not None and rand_mean is not None and skilled_mean > rand_mean)
    ok = bounds_ok and discriminates and ordered
    print("\n" + "-" * 84)
    print("SOUNDNESS:")
    print(f"  bounds [0,1] hold on both artifacts                 : {bounds_ok}")
    print(f"  SKILLED beats null AND NO-SKILL does not (two-sided): {discriminates}")
    print(f"  SKILLED mean capture > NO-SKILL mean capture        : {ordered}")
    print(f"\n[within_window_capture_proxy selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    else:
        sys.exit(_dry_run())
