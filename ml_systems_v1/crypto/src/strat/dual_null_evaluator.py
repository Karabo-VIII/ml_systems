"""src/strat/dual_null_evaluator.py -- the DUAL-NULL EVALUATOR (p-values + effect sizes, held-out).

WHAT THIS IS
------------
A single evaluator that decomposes a setup-trigger's edge into its TWO independent sources and reports,
for EACH, an empirical Monte-Carlo p-value + an effect size on HELD-OUT data (OOS+UNSEEN). It unifies the
two complementary nulls that already live in this package into one verdict:

  NULL-1  SELECTION  (which move-windows the trigger chose to be present in)
    Statistic : COMPOUND net return of the trigger-selected fixed-horizon move-windows.
    Null      : random windows matched on HORIZON + REGIME but NOT on membership -- a random same-regime
                entry bar anywhere in the same TRAIN/VAL/OOS/UNSEEN window, chased to the same fixed
                horizon. Beating it = the trigger's CHOICE of moves adds compound return.
    (delegated verbatim to window_selection_metric.window_selection_metric -- its held-out null draws are
     the Monte-Carlo null distribution from which the p-value + z are computed.)

  NULL-2  TIMING    (where inside a chosen move the trigger entered)
    Statistic : mean WITHIN-WINDOW CAPTURE = realized fraction of the best-vs-worst-entry spread, in [0,1].
    Null      : random entry timing inside the SAME move-window, under the TRIPLE confound control
                (HORIZON: one common exit per window; REGIME: candidate pool restricted to the trigger's
                past-only vol-tercile bin; WINDOW-MEMBERSHIP: candidates drawn only from the trigger's own
                move-window). Beating it = the trigger's ENTRY TIMING inside a chosen move adds capture.
    (a full per-window candidate-capture Monte-Carlo, in the style of within_window_capture_gate.)

WHY TWO NULLS: a setup can win by SELECTING better moves, by TIMING entries better inside moves, or both.
A single null conflates them. Reporting both, each with its own p-value + effect size, says exactly WHICH
lever (if any) carries the edge. The objective is robust held-out COMPOUND return on the SETUP/MOVE unit
(MEMORY.md founding framing) -- NOT per-bar IC. Both statistics are move-/window-level, never per-bar.

WHAT IT RETURNS (the ask): per-component p-values + effect sizes on held-out data.
    null_1_selection: {real, null_mean/p50/p95, p_value, effect:{edge_pp, z}, n_windows}
    null_2_timing   : {real, null_mean/p50/p95, p_value, effect:{capture_edge, realized_fraction, z}, n}
p-value  = empirical one-sided right-tail, +1 corrected: (1 + #{null >= real}) / (1 + n_books)
           (Davison & Hinkley; never 0, conservative). Also raw exceedance reported.
effect z = (real - null_mean) / null_std   (standardized distance from the null centre, sign = direction).

RWYB:
    python src/strat/dual_null_evaluator.py            # BTC 1d dry-run -> writes + verifies artifact
    python src/strat/dual_null_evaluator.py --selftest # synthetic two-sided soundness (no market data)
    python src/strat/dual_null_evaluator.py --verify   # computes on a fixture + nulls reproducible (no data)
"""
from __future__ import annotations

import hashlib
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

# Reuse the EXACT window labelling + past-only regime binning + the tested SELECTION metric so the two
# nulls decompose the SAME edge on the SAME axes (no divergent definitions).
try:
    from .within_window_capture_proxy import Windows, past_only_regime_bins
    from .window_selection_metric import window_selection_metric, _compound
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from within_window_capture_proxy import Windows, past_only_regime_bins
    from window_selection_metric import window_selection_metric, _compound


# ---------------------------------------------------------------------------
# p-value + effect-size helpers (shared by both components)
# ---------------------------------------------------------------------------
def _mc_p_value(null_draws: np.ndarray, real: float) -> tuple[float, int, int]:
    """Empirical one-sided right-tail p-value with the +1 correction.

    p = (1 + #{null >= real}) / (1 + B)  -- Davison & Hinkley (1997): never 0, conservative.
    Returns (p_value, n_exceed, n_books)."""
    null_draws = np.asarray(null_draws, float)
    null_draws = null_draws[np.isfinite(null_draws)]
    b = int(null_draws.size)
    if b == 0 or not np.isfinite(real):
        return float("nan"), 0, 0
    n_exceed = int(np.sum(null_draws >= real))
    return (1 + n_exceed) / (1 + b), n_exceed, b


def _z_effect(null_draws: np.ndarray, real: float) -> float:
    """Standardized effect size: (real - null_mean) / null_std. Sign carries direction."""
    null_draws = np.asarray(null_draws, float)
    null_draws = null_draws[np.isfinite(null_draws)]
    if null_draws.size == 0 or not np.isfinite(real):
        return float("nan")
    sd = float(null_draws.std(ddof=1)) if null_draws.size > 1 else 0.0
    if sd == 0.0:
        return float("inf") if real > float(null_draws.mean()) else (0.0 if real == float(null_draws.mean()) else float("-inf"))
    return float((real - float(null_draws.mean())) / sd)


# ===========================================================================
# NULL-2: within-window timing capture with a FULL Monte-Carlo random-timing null
# (recomputed here so we own the per-window candidate-capture curves needed for a real p-value;
#  identical confound control + conventions as within_window_capture_proxy/_gate.)
# ===========================================================================
def _timing_null(df: pd.DataFrame, trigger_col: str, *, horizon: int, win_radius: int, cost: float,
                 min_spread: float, n_regime_bins: int, n_books: int, seed: int,
                 windows: Windows, regime_bins: np.ndarray | None):
    """Per-window within-window capture + a per-window-candidate Monte-Carlo random-timing null on the
    HELD-OUT pool. Returns the held-out real mean capture, the null book-mean draws, and per-window n."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    if regime_bins is None:
        regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)

    trig = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    trig_idx = np.flatnonzero(trig)

    held_curves: list[np.ndarray] = []   # candidate-entry capture curves for HELD windows
    held_caps: list[float] = []          # the trigger's realized capture per HELD window
    per_window_caps = {w: [] for w in WINDOWS}

    for t in trig_idx:
        win_lo = max(0, t - win_radius)
        win_hi = min(n - 1, t + win_radius)
        exit_bar = win_hi + horizon
        if exit_bar >= n:
            continue
        reg_t = int(regime_bins[t])
        cand = [s for s in range(win_lo, win_hi + 1)
                if s + 1 < n and (int(regime_bins[s]) == reg_t or s == t)]
        cand = np.array(sorted(set(cand)), dtype=int)
        if cand.size < 2:
            continue
        rets = opens[exit_bar] / opens[cand + 1] - 1.0 - cost
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        if spread < min_spread:
            continue                                            # degenerate -> no timing decision
        caps_curve = (rets - worst) / spread                    # capture of each candidate entry, in [0,1]
        ti = int(np.flatnonzero(cand == t)[0])
        trig_cap = float(min(1.0, max(0.0, caps_curve[ti])))
        w = windows.label(dates.iloc[t])
        per_window_caps[w].append(trig_cap)
        if w in HELD:
            held_curves.append(caps_curve)
            held_caps.append(trig_cap)

    n_held = len(held_caps)
    real_mean = float(np.mean(held_caps)) if n_held else float("nan")

    # Monte-Carlo random-timing null: per book, redraw a uniform candidate's capture in each held window.
    if n_held:
        sizes = np.array([len(c) for c in held_curves])
        null_means = np.empty(n_books)
        for b in range(n_books):
            cb = np.empty(n_held)
            for j, curve in enumerate(held_curves):
                cb[j] = curve[rng.integers(0, sizes[j])]
            null_means[b] = cb.mean()
    else:
        null_means = np.empty(0)

    per_window_n = {w: len(per_window_caps[w]) for w in WINDOWS}
    per_window_mean = {w: (round(float(np.mean(per_window_caps[w])), 4) if per_window_caps[w] else None)
                       for w in WINDOWS}
    return real_mean, null_means, n_held, per_window_n, per_window_mean


# ===========================================================================
# THE DUAL-NULL EVALUATOR
# ===========================================================================
@dataclass
class DualNullResult:
    null_1_selection: dict
    null_2_timing: dict
    config: dict
    seed: int

    def verdict_line(self) -> str:
        s, tm = self.null_1_selection, self.null_2_timing
        return (f"SELECTION p={s['p_value']} z={s['effect']['z']} edge={s['effect']['edge_pp']}pp | "
                f"TIMING p={tm['p_value']} z={tm['effect']['z']} cap={tm['effect']['realized_fraction']}")


def dual_null_evaluate(df: pd.DataFrame, trigger_col: str, *, horizon: int = 5, win_radius: int = 3,
                       cost: float = TAKER, min_spread: float = 0.003, n_regime_bins: int = 3,
                       n_books: int = 2000, seed: int = 11, windows: Windows | None = None,
                       regime_bins: np.ndarray | None = None, alpha: float = 0.05) -> DualNullResult:
    """Evaluate a past-only setup TRIGGER against BOTH nulls on the HELD-OUT pool (OOS+UNSEEN).

    Returns per-component empirical p-values + effect sizes (the ask). Shares window labelling + past-only
    regime bins between the two components so they decompose the same edge.
    """
    windows = windows or Windows()
    df = df.reset_index(drop=True)
    if regime_bins is None:
        regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)

    # ----- NULL-1 SELECTION (delegated to the tested metric; its held null draws -> p-value + z) -----
    sel = window_selection_metric(df, trigger_col, horizon=horizon, cost=cost, n_regime_bins=n_regime_bins,
                                  n_books=n_books, seed=seed, windows=windows, regime_bins=regime_bins)
    real_comp = sel.held_summary.get("real_compound_pct")
    sel_draws = sel.held_null_draws
    p1, ex1, b1 = _mc_p_value(sel_draws, real_comp if real_comp is not None else float("nan"))
    z1 = _z_effect(sel_draws, real_comp if real_comp is not None else float("nan"))
    sel_null_mean = float(np.mean(sel_draws)) if sel_draws.size else None
    null_1 = {
        "component": "SELECTION (which move-windows the trigger chose)",
        "statistic": "compound_net_return_pct (held-out OOS+UNSEEN)",
        "real": real_comp,
        "null_mean": round(sel_null_mean, 4) if sel_null_mean is not None else None,
        "null_p50": sel.held_summary.get("null_p50"),
        "null_p95": sel.held_summary.get("null_p95"),
        "p_value": round(p1, 5) if np.isfinite(p1) else None,
        "n_exceed": ex1,
        "n_books": b1,
        "significant": bool(np.isfinite(p1) and p1 < alpha),
        "effect": {
            "edge_pp": (round(real_comp - sel_null_mean, 4)
                        if (real_comp is not None and sel_null_mean is not None) else None),
            "z": round(z1, 4) if np.isfinite(z1) else None,
        },
        "n_windows": sel.held_summary.get("n_windows", 0),
        "matched_on": ["horizon", "regime(past_only_vol_tercile)"],
        "NOT_matched_on": ["membership(move_window)"],
    }

    # ----- NULL-2 TIMING (full per-window-candidate MC random-timing null -> p-value + z) -----
    real_cap, cap_draws, n_held, pw_n, pw_mean = _timing_null(
        df, trigger_col, horizon=horizon, win_radius=win_radius, cost=cost, min_spread=min_spread,
        n_regime_bins=n_regime_bins, n_books=n_books, seed=seed, windows=windows, regime_bins=regime_bins)
    p2, ex2, b2 = _mc_p_value(cap_draws, real_cap)
    z2 = _z_effect(cap_draws, real_cap)
    cap_null_mean = float(np.mean(cap_draws)) if cap_draws.size else None
    null_2 = {
        "component": "TIMING (where inside a chosen move the trigger entered)",
        "statistic": "mean_within_window_capture in [0,1] (held-out OOS+UNSEEN)",
        "real": round(real_cap, 4) if np.isfinite(real_cap) else None,
        "null_mean": round(cap_null_mean, 4) if cap_null_mean is not None else None,
        "null_p50": round(float(np.percentile(cap_draws, 50)), 4) if cap_draws.size else None,
        "null_p95": round(float(np.percentile(cap_draws, 95)), 4) if cap_draws.size else None,
        "p_value": round(p2, 5) if np.isfinite(p2) else None,
        "n_exceed": ex2,
        "n_books": b2,
        "significant": bool(np.isfinite(p2) and p2 < alpha),
        "effect": {
            "realized_fraction": round(real_cap, 4) if np.isfinite(real_cap) else None,  # headline capture
            "capture_edge": (round(real_cap - cap_null_mean, 4)
                             if (np.isfinite(real_cap) and cap_null_mean is not None) else None),
            "z": round(z2, 4) if np.isfinite(z2) else None,
        },
        "n_windows": n_held,
        "per_window_n": pw_n,
        "per_window_mean_capture": pw_mean,
        "confound_control": ["horizon(single common exit per window)",
                             "regime(candidate pool = trigger past-only vol-tercile bin)",
                             "window_membership(candidates drawn from trigger's own move-window)"],
    }

    config = {
        "evaluator": "dual_null",
        "trigger_col": trigger_col,
        "horizon": horizon, "win_radius": win_radius, "cost": cost, "min_spread": min_spread,
        "n_regime_bins": n_regime_bins, "n_books": n_books, "alpha": alpha,
        "held_pool": HELD,
        "p_value_estimator": "(1 + #{null>=real}) / (1 + n_books)  [one-sided right tail, +1 corrected]",
        "effect_z": "(real - null_mean) / null_std",
        "objective": "robust held-out COMPOUND return on the SETUP/MOVE unit (not per-bar IC)",
    }
    return DualNullResult(null_1_selection=null_1, null_2_timing=null_2, config=config, seed=seed)


# ---------------------------------------------------------------------------
# Artifact emit + verify
# ---------------------------------------------------------------------------
def _fingerprint(res: DualNullResult) -> str:
    payload = json.dumps({"s": res.null_1_selection.get("p_value"),
                          "t": res.null_2_timing.get("p_value"),
                          "sr": res.null_1_selection.get("real"),
                          "tr": res.null_2_timing.get("real")}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def write_artifact(res: DualNullResult, tag: str, out_dir: Path = ARTIFACT_DIR) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"dual_null_{tag}.json"
    summary = {
        "tag": tag,
        "config": res.config,
        "seed": res.seed,
        "null_1_selection": res.null_1_selection,
        "null_2_timing": res.null_2_timing,
        "reproducibility": {"seed": res.seed, "fingerprint_sha1": _fingerprint(res),
                            "note": "same seed -> identical p-values + reals (nulls are deterministic)."},
    }
    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.replace(json_path)
    return str(json_path)


def verify_artifact(json_path: str | Path) -> tuple[bool, str]:
    """Assert the artifact exists and BOTH components computed a finite p-value + effect size on held-out."""
    json_path = Path(json_path)
    if not json_path.exists():
        return False, f"artifact missing: {json_path}"
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    for key in ("null_1_selection", "null_2_timing"):
        comp = d.get(key) or {}
        if comp.get("n_windows", 0) <= 0:
            return False, f"{key}: zero held-out windows (did not compute)"
        p = comp.get("p_value")
        if p is None or not np.isfinite(float(p)) or not (0.0 < float(p) <= 1.0):
            return False, f"{key}: p_value not a valid probability ({p})"
        z = (comp.get("effect") or {}).get("z")
        if z is None or not np.isfinite(float(z)):
            return False, f"{key}: effect z not finite ({z})"
    s, t = d["null_1_selection"], d["null_2_timing"]
    return True, (f"OK: SELECTION p={s['p_value']} z={s['effect']['z']} (n={s['n_windows']}) | "
                  f"TIMING p={t['p_value']} z={t['effect']['z']} cap={t['effect']['realized_fraction']} "
                  f"(n={t['n_windows']})")


# ===========================================================================
# Synthetic fixture (shared) -- dip->bounce moves carry BOTH selection + timing signal
# ===========================================================================
def _make_fixture(seed: int = 3, n: int = 1600) -> pd.DataFrame:
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
# VERIFY: both components compute on a fixture AND the nulls are reproducible (no market data)
# ===========================================================================
def _verify() -> int:
    print("=" * 92)
    print("[dual_null_evaluator --verify] both nulls compute on a fixture + p-values reproducible")
    print("=" * 92)
    df = _make_fixture()
    win = _fixture_windows()
    seed = 11

    res = dual_null_evaluate(df, "dip", horizon=5, n_books=1000, seed=seed, windows=win)
    path = write_artifact(res, tag="FIXTURE")
    ok_compute, msg = verify_artifact(path)
    print(f"\n  (1) both components compute on fixture : {ok_compute}")
    print(f"      artifact: {path}")
    print(f"      {msg}")

    # reproducible: same seed -> identical p-values + reals
    res2 = dual_null_evaluate(df, "dip", horizon=5, n_books=1000, seed=seed, windows=win)
    same = (_fingerprint(res) == _fingerprint(res2))
    # seed truly drives the nulls: different seed -> (generally) different null bands/p-values
    res3 = dual_null_evaluate(df, "dip", horizon=5, n_books=1000, seed=seed + 7, windows=win)
    seed_matters = (_fingerprint(res) != _fingerprint(res3)) or (
        res.null_1_selection["null_p95"] != res3.null_1_selection["null_p95"])

    print(f"\n  (2) reproducible (same seed -> identical p-values+reals) : {same}")
    print(f"      seed truly drives the nulls (diff seed -> diff)        : {seed_matters}")
    print(f"      fingerprint: {_fingerprint(res)[:16]}")

    ok = bool(ok_compute and same and seed_matters)
    print("\n" + "-" * 92)
    print(f"[dual_null_evaluator --verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ===========================================================================
# SELFTEST: two-sided soundness -- a genuinely skilled trigger is significant on BOTH;
# a random trigger is significant on NEITHER (per-component p-values discriminate).
# ===========================================================================
def _selftest() -> int:
    print("=" * 92)
    print("[dual_null_evaluator --selftest] two-sided soundness (no market data)")
    print("=" * 92)
    df = _make_fixture()
    win = _fixture_windows()

    # (A) SKILLED: buy-the-dip -> selects up-moves (NULL-1) AND times near the dip bottom (NULL-2).
    skl = dual_null_evaluate(df, "dip", horizon=5, win_radius=3, n_books=1500, seed=11, windows=win)
    # (B) NO-SKILL: random bars at the same rate -> neither component significant.
    rng = np.random.default_rng(99)
    df = df.copy()
    df["rand"] = rng.random(len(df)) < float(df["dip"].mean())
    rnd = dual_null_evaluate(df, "rand", horizon=5, win_radius=3, n_books=1500, seed=11, windows=win)

    def _show(name, r):
        s, t = r.null_1_selection, r.null_2_timing
        print(f"\n  {name}")
        print(f"    NULL-1 SELECTION : real={s['real']}%  null_p95={s['null_p95']}%  "
              f"p={s['p_value']}  z={s['effect']['z']}  sig={s['significant']}  n={s['n_windows']}")
        print(f"    NULL-2 TIMING    : real={t['real']}  null_p95={t['null_p95']}  "
              f"p={t['p_value']}  z={t['effect']['z']}  cap_edge={t['effect']['capture_edge']}  "
              f"sig={t['significant']}  n={t['n_windows']}")

    _show("(A) SKILLED (buy-the-dip)", skl)
    _show("(B) NO-SKILL (random)", rnd)

    # soundness: skilled significant on >=1 component (selection is the strong one here); random on neither.
    skl_any = skl.null_1_selection["significant"] or skl.null_2_timing["significant"]
    rnd_none = not (rnd.null_1_selection["significant"] or rnd.null_2_timing["significant"])
    # effect ordering: skilled selection edge strictly exceeds random's
    edge_ordered = (skl.null_1_selection["effect"]["edge_pp"] is not None
                    and rnd.null_1_selection["effect"]["edge_pp"] is not None
                    and skl.null_1_selection["effect"]["edge_pp"] > rnd.null_1_selection["effect"]["edge_pp"])
    ok = bool(skl_any and rnd_none and edge_ordered)
    print("\n" + "-" * 92)
    print("SOUNDNESS (two-sided):")
    print(f"  SKILLED significant on >=1 component        : {skl_any}")
    print(f"  NO-SKILL significant on NEITHER component    : {rnd_none}")
    print(f"  SKILLED selection edge > NO-SKILL edge       : {edge_ordered}")
    print(f"\n[dual_null_evaluator --selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


# ===========================================================================
# DRY-RUN on ONE instrument/timeframe (BTC 1d): write + verify the artifact
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
    print("=" * 92)
    print("DUAL-NULL EVALUATOR -- dry-run on BTCUSDT 1d (one instrument/timeframe)")
    print("=" * 92)
    df = _load_btc_1d()
    print(f"  loaded {len(df)} bars: {df['date'].min().date()} -> {df['date'].max().date()}")
    prior_max = df["close"].rolling(20).max().shift(1)
    df["breakout"] = (df["close"] > prior_max).fillna(False)
    print(f"  trigger = 20-bar breakout; firings = {int(df['breakout'].sum())}")

    res = dual_null_evaluate(df, "breakout", horizon=5, win_radius=3, cost=TAKER, n_books=2000, seed=11)
    path = write_artifact(res, tag="BTCUSDT_1d_breakout_h5")
    print(f"\n  artifact JSON: {path}")

    s, t = res.null_1_selection, res.null_2_timing
    print("\n  HELD-OUT (OOS+UNSEEN) per-component p-values + effect sizes:")
    print(f"    NULL-1 SELECTION : n={s['n_windows']}  real={s['real']}%  null_p95={s['null_p95']}%  "
          f"p={s['p_value']}  edge={s['effect']['edge_pp']}pp  z={s['effect']['z']}  sig={s['significant']}")
    print(f"    NULL-2 TIMING    : n={t['n_windows']}  real_capture={t['real']}  null_p95={t['null_p95']}  "
          f"p={t['p_value']}  cap_edge={t['effect']['capture_edge']}  z={t['effect']['z']}  sig={t['significant']}")

    ok, msg = verify_artifact(path)
    print("\n" + "-" * 92)
    print(f"  VERIFY: {msg}")
    print(f"[dual_null_evaluator] {'PASS' if ok else 'FAIL'} -- dual-null p-values + effect sizes emitted.")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(_verify())
    elif "--selftest" in sys.argv:
        sys.exit(_selftest())
    else:
        sys.exit(_dry_run())
