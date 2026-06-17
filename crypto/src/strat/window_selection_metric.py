"""src/strat/window_selection_metric.py -- the WINDOW-SELECTION metric (standalone artifact).

WHAT THIS IS (and how it is the COMPLEMENT of within_window_capture_proxy.py)
-----------------------------------------------------------------------------
The apparatus already has the WITHIN-WINDOW timing lens:
  - within_window_capture_proxy.py / firewall.random_entry_null(membership_matched=True) draw the random
    null from WITHIN the SAME move-window the trigger fires in. They answer (b): "GIVEN you are present in
    this move, did your TRIGGER time the entry better than random timing inside the same move?" -- pure
    trigger-TIMING value, with move-MEMBERSHIP held fixed.

THIS module is the missing complement -- the SELECTION lens. It answers (a): "did the trigger SELECT
better move-windows to be present in at all, vs being dropped at RANDOM into other windows of the SAME
regime and the SAME horizon?" The random baseline here is matched on HORIZON + REGIME but explicitly
**NOT** on membership: a random window may be ANY same-regime bar in the period, not the trigger's own
move. So this isolates move-/window-SELECTION value (a), the exact thing the membership-matched null
holds constant.

THE UNIT (MEMORY.md founding framing): the unit of trading is a SETUP across a MOVE (multiple candles),
and the objective is robust held-out COMPOUND return -- NOT per-bar IC. This metric is reported as the
COMPOUND return of the trigger-selected move-windows vs the COMPOUND return of the regime+horizon-matched
random windows, per window (TRAIN/VAL/OOS/UNSEEN) with the held-out (OOS+UNSEEN) pool as the verdict.

THE METRIC (compound of selected windows vs regime+horizon-matched random windows)
----------------------------------------------------------------------------------
For each trigger firing at bar t, the move-window is a FIXED-HORIZON move: enter at next-bar open
opens[t+1], exit at opens[t+1+horizon]. The realized net return of that window is
    ret(t) = opens[t+1+horizon] / opens[t+1] - 1 - cost            (next-bar-open fill; round-trip cost)
The TRIGGER-SELECTED compound for a window W = product over all trigger firings in W of (1+ret) - 1.

RANDOM BASELINE -- matched on HORIZON + REGIME, NOT membership:
  For each trigger firing at bar t (regime bin b, window W), the null draws a random entry bar e from the
  pool of ALL valid entry bars in window W that share the SAME past-only regime bin b (the WHOLE-window
  same-regime pool, not the move band around t). Every null entry is chased to the SAME fixed horizon.
  n_books Monte-Carlo draws give a null distribution of compound return per window; the trigger beats the
  null iff its real compound exceeds the null's 95th percentile. Because the per-bar regime bin is a
  strictly past-only label (vol tercile, shift(1)) and the horizon is held constant, a positive result is
  SELECTION value: the trigger picked better-than-random same-regime moves -- not a regime artifact and
  not a horizon artifact.

WHY "NOT membership": drawing from the whole same-regime window pool (rather than the move band around t)
is the deliberate inverse of firewall.random_entry_null(membership_matched=True). Beating THIS null means
the trigger's CHOICE of which moves to be present in adds compound; beating the membership null means the
trigger's TIMING inside a chosen move adds compound. Reporting both decomposes the setup's edge into
SELECTION (a) + TIMING (b).

ARTIFACT: a JSON summary under runs/strat/window_selection_<tag>.json with per-window real vs null
compound, the held-out verdict, the full config, and a REPRODUCIBILITY fingerprint of the held-out null
(seed + p50/p95 + a sha1 of the null compound draws). The random baseline is fully reproducible from the
fixed seed -- re-running with the same seed reproduces the identical null draws (the verify path proves it).

RWYB:
    python src/strat/window_selection_metric.py            # BTC 1d dry-run -> writes + verifies artifact
    python src/strat/window_selection_metric.py --selftest # synthetic two-sided soundness (no market data)
    python src/strat/window_selection_metric.py --verify   # the verify_cmd: exits 0 iff metric computes
                                                            #   on a fixture AND the random baseline is
                                                            #   reproducible (fixed seed). NO market data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
ARTIFACT_DIR = ROOT / "runs" / "strat"

# Reuse the EXACT same window labelling + past-only regime binning as the within-window proxy so the
# SELECTION metric and the membership-matched TIMING proxy decompose the same edge on the same axes.
try:
    from .within_window_capture_proxy import Windows, past_only_regime_bins
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from within_window_capture_proxy import Windows, past_only_regime_bins


def _compound(nets) -> float:
    """Compound (geometric) return in PERCENT of a sequence of per-trade net returns."""
    nets = np.asarray(nets, float)
    return float((np.prod(1.0 + nets) - 1.0) * 100) if nets.size else 0.0


# ---------------------------------------------------------------------------
@dataclass
class WindowSelectionResult:
    per_window_summary: dict     # {WINDOW: {...}}
    per_regime_summary: dict     # {bin: {...}}
    held_summary: dict
    held_null_draws: np.ndarray  # the held-out null compound draws (for the reproducibility fingerprint)
    config: dict
    rng_seed: int
    n_trigger_windows: int


# ---------------------------------------------------------------------------
# THE METRIC
# ---------------------------------------------------------------------------
def window_selection_metric(df: pd.DataFrame, trigger_col: str, *, horizon: int = 5, cost: float = TAKER,
                            n_regime_bins: int = 3, n_books: int = 2000, seed: int = 11,
                            windows: Windows | None = None,
                            regime_bins: np.ndarray | None = None) -> WindowSelectionResult:
    """Compound return of trigger-selected fixed-horizon move-windows vs a HORIZON+REGIME-matched
    (NOT membership-matched) random baseline, per window.

    df         : DataFrame with date, open, high, low, close (trigger_col past-only by construction).
    trigger_col: boolean/0-1 column; True = the setup is CONFIRMED at the close of this bar (the trigger).
    horizon    : bars held -- enter opens[t+1], exit opens[t+1+horizon] (HELD CONSTANT -> horizon match).
    n_books    : Monte-Carlo random-baseline books (the null distribution of compound return).
    seed       : fixed RNG seed -> the random baseline is fully reproducible.
    """
    windows = windows or Windows()
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    if regime_bins is None:
        regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)

    # A bar i is a VALID entry iff its fill (i+1) and its fixed-horizon exit (i+1+horizon) are in range.
    def net_for_entry(i: int) -> float:
        ef = i + 1
        xf = ef + horizon
        return opens[xf] / opens[ef] - 1.0 - cost

    last_valid = n - 2 - horizon  # i <= last_valid  <=>  i+1+horizon <= n-1
    wlab = np.array([windows.label(dates.iloc[i]) for i in range(n)])

    trig = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    trig_idx = [int(t) for t in np.flatnonzero(trig) if t <= last_valid]

    # trigger firings as (t, regime_bin, window)
    trgs = [(t, int(regime_bins[t]), wlab[t]) for t in trig_idx]

    # whole-window, same-regime eligible entry pools (the NON-membership baseline pool).
    # pool[W][b] = every valid entry bar in window W whose past-only regime bin == b.
    valid_bars = np.array([i for i in range(0, last_valid + 1)], dtype=int)
    pools: dict[str, dict[int, np.ndarray]] = {w: {} for w in WINDOWS}
    for w in WINDOWS:
        bars_w = valid_bars[wlab[valid_bars] == w]
        for b in range(n_regime_bins):
            sel = bars_w[regime_bins[bars_w] == b]
            if sel.size:
                pools[w][b] = sel

    def _null_band(trg_list, books: int, band_seed: int):
        """Reproducible regime+horizon-matched (NOT membership) random-baseline compound draws.

        For each trigger (t,b,w) draw a random entry from pool[w][b] (whole-window same-regime pool) and
        chase it to the SAME fixed horizon; compound across the trigger set; repeat `books` times. Same
        seed + same trg_list -> byte-identical draws (proved by the verify path)."""
        if not trg_list:
            return None, None, np.empty(0)
        rng = np.random.default_rng(band_seed)
        comps = np.empty(books, dtype=float)
        for bk in range(books):
            nets = []
            for (t, b, w) in trg_list:
                pool = pools[w].get(b)
                e = int(rng.choice(pool)) if (pool is not None and pool.size) else int(t)  # fallback: t itself
                nets.append(net_for_entry(e))
            comps[bk] = _compound(nets)
        return float(np.percentile(comps, 50)), float(np.percentile(comps, 95)), comps

    def _summ(trg_list, band_seed: int):
        if not trg_list:
            return ({"n_windows": 0, "real_compound_pct": None, "null_p50": None, "null_p95": None,
                     "selection_edge_pp": None, "beats_null": None}, np.empty(0))
        real_nets = [net_for_entry(t) for (t, _, _) in trg_list]
        real_comp = _compound(real_nets)
        p50, p95, draws = _null_band(trg_list, n_books, band_seed)
        summary = {
            "n_windows": len(trg_list),
            "real_compound_pct": round(real_comp, 4),
            "null_p50": round(p50, 4) if p50 is not None else None,
            "null_p95": round(p95, 4) if p95 is not None else None,
            "selection_edge_pp": round(real_comp - p50, 4) if p50 is not None else None,
            "beats_null": bool(real_comp > p95) if p95 is not None else None,
        }
        return summary, draws

    # Per-window. Each window uses the SAME seed -> independent deterministic stream per window (the call
    # creates a fresh default_rng(seed) each time, so results are reproducible and order-independent).
    per_window = {w: _summ([g for g in trgs if g[2] == w], seed)[0] for w in WINDOWS}

    bins_present = sorted({b for (_, b, _) in trgs})
    per_regime = {int(b): _summ([g for g in trgs if g[1] == b], seed)[0] for b in bins_present}

    held_summary, held_draws = _summ([g for g in trgs if g[2] in HELD], seed)

    config = {
        "metric": "window_selection_compound",
        "definition": "compound return of trigger-selected fixed-horizon move-windows vs a "
                      "HORIZON+REGIME-matched (NOT membership-matched) random baseline, per window",
        "trigger_col": trigger_col,
        "horizon": horizon,
        "cost": cost,
        "n_regime_bins": n_regime_bins,
        "n_books": n_books,
        "matched_on": ["horizon", "regime(past_only_vol_tercile)"],
        "NOT_matched_on": ["membership(move_window)"],
        "complement_of": "firewall.random_entry_null(membership_matched=True) / within_window_capture_proxy",
        "isolates": "move-/window-SELECTION value (a), with regime+horizon held constant",
    }
    return WindowSelectionResult(per_window_summary=per_window, per_regime_summary=per_regime,
                                 held_summary=held_summary, held_null_draws=held_draws,
                                 config=config, rng_seed=seed, n_trigger_windows=len(trgs))


# ---------------------------------------------------------------------------
# Reproducibility fingerprint + artifact emit + verify
# ---------------------------------------------------------------------------
def _draws_fingerprint(draws: np.ndarray) -> str:
    """sha1 of the held-out null compound draws (rounded to 8 dp) -- a stable cross-run fingerprint."""
    if draws.size == 0:
        return "EMPTY"
    return hashlib.sha1(np.round(draws, 8).tobytes()).hexdigest()


def write_artifact(res: WindowSelectionResult, tag: str, out_dir: Path = ARTIFACT_DIR) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"window_selection_{tag}.json"
    summary = {
        "tag": tag,
        "n_trigger_windows": res.n_trigger_windows,
        "config": res.config,
        "rng_seed": res.rng_seed,
        "per_window_summary": res.per_window_summary,
        "per_regime_summary": res.per_regime_summary,
        "held_summary": res.held_summary,
        "reproducibility": {
            "seed": res.rng_seed,
            "held_null_p50": res.held_summary.get("null_p50"),
            "held_null_p95": res.held_summary.get("null_p95"),
            "held_null_n_draws": int(res.held_null_draws.size),
            "held_null_fingerprint_sha1": _draws_fingerprint(res.held_null_draws),
            "note": "re-running window_selection_metric with this seed reproduces an identical "
                    "held_null_fingerprint_sha1 (the random baseline is deterministic).",
        },
    }
    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.replace(json_path)
    return str(json_path)


def verify_artifact(json_path: str | Path) -> tuple[bool, str]:
    """Assert the JSON artifact exists, the metric COMPUTED (finite real+null on the held-out pool),
    and it carries a reproducibility fingerprint of the (seeded) random baseline."""
    json_path = Path(json_path)
    if not json_path.exists():
        return False, f"artifact missing: {json_path}"
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    held = d.get("held_summary") or {}
    if held.get("n_windows", 0) <= 0:
        return False, "metric did not compute: zero held-out trigger windows"
    for k in ("real_compound_pct", "null_p50", "null_p95"):
        v = held.get(k)
        if v is None or not np.isfinite(float(v)):
            return False, f"metric did not compute: held_summary['{k}'] is not finite ({v})"
    rep = d.get("reproducibility") or {}
    if not rep.get("held_null_fingerprint_sha1") or rep["held_null_fingerprint_sha1"] == "EMPTY":
        return False, "no reproducibility fingerprint on the random baseline"
    return True, (f"OK: held real_compound={held['real_compound_pct']}%  null_p95={held['null_p95']}%  "
                  f"beats_null={held['beats_null']}  fingerprint={rep['held_null_fingerprint_sha1'][:12]}")


# ===========================================================================
# Synthetic fixture (no market data) -- shared by --verify and --selftest
# ===========================================================================
def _make_fixture(seed: int = 3, n: int = 1600) -> pd.DataFrame:
    """Daily OHLC with dip->bounce moves so a 'buy-the-dip' trigger SELECTS genuinely better move-windows:
    a dip bar precedes a multi-bar up-drift, so entering on the dip selects an up-move that a random
    same-regime bar (mostly flat/noise) does not -> the SELECTION metric has a real two-sided signal."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.010, n)
    bounce = 0
    is_dip = np.zeros(n, bool)
    for t in range(1, n):
        if bounce > 0:
            rets[t] += 0.012
            bounce -= 1
        elif rng.random() < 0.045:
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


def _fixture_windows() -> Windows:
    return Windows(train_end="2023-06-01", val_end="2024-01-01", oos_end="2024-09-01")


# ===========================================================================
# THE VERIFY PATH: exits 0 iff the metric COMPUTES on a fixture AND the random
# baseline is REPRODUCIBLE (fixed seed). No market data.
# ===========================================================================
def _verify() -> int:
    print("=" * 88)
    print("[window_selection_metric --verify]  metric computes on a fixture + random baseline reproducible")
    print("=" * 88)
    df = _make_fixture()
    win = _fixture_windows()
    seed = 11

    # (1) METRIC COMPUTES on the fixture -> write + verify the JSON artifact.
    res = window_selection_metric(df, "dip", horizon=5, cost=TAKER, n_books=1000, seed=seed, windows=win)
    path = write_artifact(res, tag="FIXTURE")
    ok_compute, msg = verify_artifact(path)
    print(f"\n  (1) metric computes on fixture : {ok_compute}")
    print(f"      artifact: {path}")
    print(f"      {msg}")

    # (2) RANDOM BASELINE REPRODUCIBLE: re-run with the SAME seed -> identical held-out null draws.
    res2 = window_selection_metric(df, "dip", horizon=5, cost=TAKER, n_books=1000, seed=seed, windows=win)
    same_draws = bool(res.held_null_draws.shape == res2.held_null_draws.shape
                      and np.array_equal(res.held_null_draws, res2.held_null_draws))
    same_fp = _draws_fingerprint(res.held_null_draws) == _draws_fingerprint(res2.held_null_draws)
    # also confirm the on-disk fingerprint round-trips
    with open(path, encoding="utf-8") as f:
        disk_fp = json.load(f)["reproducibility"]["held_null_fingerprint_sha1"]
    disk_match = disk_fp == _draws_fingerprint(res.held_null_draws)

    # (3) DIFFERENT seed -> DIFFERENT draws (proves the seed actually drives the baseline, not a constant).
    res3 = window_selection_metric(df, "dip", horizon=5, cost=TAKER, n_books=1000, seed=seed + 1, windows=win)
    seed_matters = not np.array_equal(res.held_null_draws, res3.held_null_draws)

    print(f"\n  (2) random baseline reproducible (same seed -> identical draws) : {same_draws}")
    print(f"      held-null fingerprint (run1==run2)                          : {same_fp}")
    print(f"      on-disk fingerprint round-trips                             : {disk_match}")
    print(f"      different seed -> different draws (seed truly drives it)     : {seed_matters}")
    print(f"      fingerprint: {_draws_fingerprint(res.held_null_draws)[:16]}  "
          f"(seed={seed}, n_draws={res.held_null_draws.size})")

    ok = bool(ok_compute and same_draws and same_fp and disk_match and seed_matters)
    print("\n" + "-" * 88)
    print(f"[window_selection_metric --verify] {'PASS' if ok else 'FAIL'} -- "
          f"{'metric computed on the fixture and the seeded random baseline is reproducible.' if ok else 'see flags above.'}")
    return 0 if ok else 1


# ===========================================================================
# Two-sided SOUNDNESS selftest: a SELECTING trigger beats null; a random one does not.
# ===========================================================================
def _selftest() -> int:
    print("=" * 88)
    print("[window_selection_metric --selftest] two-sided soundness (no market data)")
    print("=" * 88)
    df = _make_fixture()
    win = _fixture_windows()

    # (A) SELECTING trigger: buy the dip -> selects up-moves -> beats regime+horizon-matched random null.
    res = window_selection_metric(df, "dip", horizon=5, n_books=1500, seed=11, windows=win)
    sel_real = res.held_summary["real_compound_pct"]
    sel_beats = res.held_summary["beats_null"]

    # (B) NO-SELECTION trigger: random bars at the same rate -> should NOT beat the null.
    rng = np.random.default_rng(99)
    df = df.copy()
    df["rand"] = rng.random(len(df)) < float(df["dip"].mean())
    res_r = window_selection_metric(df, "rand", horizon=5, n_books=1500, seed=11, windows=win)
    rnd_real = res_r.held_summary["real_compound_pct"]
    rnd_beats = res_r.held_summary["beats_null"]

    print(f"\n  (A) SELECTING (buy-the-dip) held real_compound={sel_real}%  beats_null={sel_beats}")
    print(f"      null_p50={res.held_summary['null_p50']}%  null_p95={res.held_summary['null_p95']}%")
    print(f"  (B) NO-SELECTION (random)  held real_compound={rnd_real}%  beats_null={rnd_beats}")
    print(f"      null_p50={res_r.held_summary['null_p50']}%  null_p95={res_r.held_summary['null_p95']}%")

    discriminates = bool(sel_beats) and not bool(rnd_beats)
    ordered = (sel_real is not None and rnd_real is not None and sel_real > rnd_real)
    ok = discriminates and ordered
    print("\n" + "-" * 88)
    print("SOUNDNESS (two-sided):")
    print(f"  SELECTING beats null AND NO-SELECTION does not : {discriminates}")
    print(f"  SELECTING real compound > NO-SELECTION         : {ordered}")
    print(f"\n[window_selection_metric --selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


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
    print("=" * 88)
    print("WINDOW-SELECTION METRIC -- dry-run on BTCUSDT 1d (one instrument/timeframe)")
    print("=" * 88)
    df = _load_btc_1d()
    print(f"  loaded {len(df)} bars: {df['date'].min().date()} -> {df['date'].max().date()}")
    prior_max = df["close"].rolling(20).max().shift(1)
    df["breakout"] = (df["close"] > prior_max).fillna(False)
    print(f"  trigger = 20-bar breakout; firings = {int(df['breakout'].sum())}")

    res = window_selection_metric(df, "breakout", horizon=5, cost=TAKER, n_books=2000, seed=11)
    path = write_artifact(res, tag="BTCUSDT_1d_breakout_h5")
    print(f"\n  artifact JSON: {path}")
    print("\n  per-window real compound vs HORIZON+REGIME-matched (NOT membership) random null:")
    for w in WINDOWS:
        s = res.per_window_summary[w]
        print(f"    {w:7} n={s['n_windows']:<4} real={s['real_compound_pct']}%  null_p50={s['null_p50']}%  "
              f"null_p95={s['null_p95']}%  beats_null={s['beats_null']}")
    hs = res.held_summary
    print(f"\n  HELD-OUT pooled (OOS+UNSEEN): n={hs['n_windows']} real={hs['real_compound_pct']}% "
          f"null_p95={hs['null_p95']}% selection_edge={hs['selection_edge_pp']}pp beats_null={hs['beats_null']}")

    ok, msg = verify_artifact(path)
    print("\n" + "-" * 88)
    print(f"  VERIFY: {msg}")
    print(f"[window_selection_metric] {'PASS' if ok else 'FAIL'} -- SELECTION artifact emitted "
          f"(complement of the membership-matched TIMING null).")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(_verify())
    elif "--selftest" in sys.argv:
        sys.exit(_selftest())
    else:
        sys.exit(_dry_run())
