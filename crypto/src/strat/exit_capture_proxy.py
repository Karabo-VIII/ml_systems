"""src/strat/exit_capture_proxy.py -- the EXIT-METHODOLOGY capture-rate proxy on REAL data (the DUAL of
within_window_capture_proxy.py).

WHY (user steer 2026-06-08): the ENTRY-timing axis was already oracle-decomposed and found NULL held-out at
daily/4h (within_window_capture_proxy + the MA-oracle run). The user's insight: "the EXIT methodology selection is
what gives pure returns" -- a different, UNEXPLORED axis. So this module fixes the ENTRY and varies the EXIT:
given a move you are IN, does a CAUSAL exit RULE land closer to the BEST-possible exit than random exit timing
would? That is exit-policy SKILL, measured as a bounded [0,1] capture rate, two-sided + held-out -- the same rigor
as the entry proxy.

THE PROXY (best-vs-worst exit, per move; entry FIXED -> pure EXIT-timing skill)
------------------------------------------------------------------------------
A setup fires at bar t; we ENTER at open[t+1]. The candidate EXIT bars are the holding window
[t+1+min_hold, t+1+max_hold] (clamped). With next-bar fills already in the entry, every candidate exit shares
the SAME entry, so:
    ret(e) = open[e] / open[t+1] - 1 - cost      for each candidate exit bar e in the holding window
    best = max_e ret(e)  (sell at the peak -- the exit ORACLE)   worst = min_e ret(e)   spread = best - worst
    capture = (ret(rule_exit) - worst) / spread   in [0,1]
capture = 1.0 -> the rule exited at the BEST possible bar; 0.0 -> the worst; 0.5 -> middle of the exit range.
The exit ORACLE = best (perfect-foresight peak sell). The realizable exit = a CAUSAL rule (decision at bar e uses
only info up to e). The null = RANDOM exit timing inside the same holding window (membership matched).

CONFOUND CONTROLS (mirrors the entry proxy)
-------------------------------------------
1. HOLD WINDOW held constant: every rule + the null draw the exit from the SAME [min_hold,max_hold] pool, so a
   "smart" rule cannot secretly get a longer/shorter hold than the null it is compared to.
2. REGIME: each move is labelled by the ENTRY bar's PAST-ONLY vol tercile; capture is reported stratified by bin,
   so exit skill is not just "this move was in an easy regime".
3. ENTRY FIXED: capture scores WHERE-to-exit only, never which-move or when-to-enter (those are the entry proxy).

EXIT RULES (all CAUSAL): fixed-horizon (the no-skill baseline), trailing-stop (chandelier), MA-cross-exit (the
user's MA/EMA config as an EXIT condition). If a smart rule's held-out capture BEATS the random-exit null, exit
methodology carries real, tradeable skill at this cadence -- the edge the entry axis did not have.

RWYB:
    python src/strat/exit_capture_proxy.py            # BTCUSDT 1d dry-run -> writes + verifies artifact
    python src/strat/exit_capture_proxy.py --selftest # synthetic two-sided soundness (no market data)
No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from src.strat.within_window_capture_proxy import (  # noqa: E402  reuse the SAME window/regime/cost machinery
    Windows, past_only_regime_bins, WINDOWS, HELD, TAKER, ARTIFACT_DIR)


# ---------------------------------------------------------------------------
# CAUSAL exit rules. Each returns the chosen exit-bar index given (entry bar e0, candidate exit bars).
# The decision at any bar uses ONLY closes/opens up to that bar (causal). All clamp to the window end.
# ---------------------------------------------------------------------------
def exit_fixed(df, e0: int, cand_exits: np.ndarray, *, hold: int = 5) -> int:
    """No-skill baseline: exit a FIXED `hold` bars after entry (clamped into the window)."""
    target = e0 + hold
    return int(cand_exits[np.argmin(np.abs(cand_exits - target))])


def exit_trailing_stop(df, e0: int, cand_exits: np.ndarray, *, drop: float = 0.06) -> int:
    """Chandelier/trailing stop: track the running MAX close since entry; exit the first bar whose close has
    fallen >= `drop` below that running max. Causal. Falls through to the last candidate if never triggered."""
    close = df["close"].to_numpy(float)
    run_max = close[e0]
    for e in cand_exits:
        run_max = max(run_max, close[e])
        if close[e] <= run_max * (1.0 - drop):
            return int(e)
    return int(cand_exits[-1])


def exit_ma_cross(df, e0: int, cand_exits: np.ndarray, *, fast: int = 5, slow: int = 20) -> int:
    """The user's MA/EMA config as an EXIT condition: exit the first bar in the window where the fast EMA crosses
    BELOW the slow EMA (momentum rolled over). Causal (EMAs use closes up to that bar). Fall through to last."""
    close = df["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean().to_numpy(float)
    ema_s = close.ewm(span=slow, adjust=False).mean().to_numpy(float)
    for e in cand_exits:
        if ema_f[e] < ema_s[e]:
            return int(e)
    return int(cand_exits[-1])


EXIT_RULES = {
    "fixed_h5": (exit_fixed, {"hold": 5}),
    "trailing_6pct": (exit_trailing_stop, {"drop": 0.06}),
    "ma_cross_5_20": (exit_ma_cross, {"fast": 5, "slow": 20}),
}


@dataclass
class ExitCaptureResult:
    rows: list
    per_rule_window: dict      # {rule: {WINDOW: summary}}
    per_rule_held: dict        # {rule: held summary}
    config: dict
    rng_seed: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def exit_capture_proxy(df: pd.DataFrame, entry_trigger_col: str, *, min_hold: int = 1, max_hold: int = 20,
                       cost: float = TAKER, min_spread: float = 0.003, windows: Windows | None = None,
                       regime_bins: np.ndarray | None = None, n_regime_bins: int = 3,
                       rules: dict | None = None, seed: int = 11) -> ExitCaptureResult:
    """Compute the EXIT-timing capture rate for each rule, for every move the entry trigger fires in."""
    windows = windows or Windows()
    rules = rules or EXIT_RULES
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    if regime_bins is None:
        regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)
    trig = pd.to_numeric(df[entry_trigger_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    trig_idx = np.flatnonzero(trig)

    rows = []
    for t in trig_idx:
        e0 = t + 1                                    # entry bar (we buy at open[e0])
        if e0 + min_hold >= n:
            continue
        lo = e0 + min_hold
        hi = min(n - 1, e0 + max_hold)
        cand = np.arange(lo, hi + 1, dtype=int)
        if cand.size < 2:
            continue
        rets = opens[cand] / opens[e0] - 1.0 - cost
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        if spread < min_spread:
            continue
        reg_t = int(regime_bins[t])
        null_mean_capture = float(((rets - worst) / spread).mean())   # random-exit-timing null (analytic mean)
        row = {"entry_idx": int(e0), "entry_ts": str(pd.Timestamp(dates.iloc[t])),
               "window": windows.label(dates.iloc[t]), "regime_bin": reg_t,
               "n_cand_exits": int(cand.size), "best_ret": round(best, 6), "worst_ret": round(worst, 6),
               "spread": round(spread, 6), "null_mean_capture": round(null_mean_capture, 6)}
        for rname, (fn, kw) in rules.items():
            e_star = int(fn(df, e0, cand, **kw))
            e_star = int(min(max(e_star, lo), hi))    # clamp into the window
            rule_ret = float(opens[e_star] / opens[e0] - 1.0 - cost)
            cap = float(min(1.0, max(0.0, (rule_ret - worst) / spread)))
            row[f"cap_{rname}"] = round(cap, 6)
            row[f"exit_off_{rname}"] = int(e_star - e0)
        rows.append(row)

    def _mc_null(sub, n_books=2000):
        if not sub:
            return None, None
        caps = np.array([r["null_mean_capture"] for r in sub])
        idx = np.arange(len(caps)); means = np.empty(n_books)
        for b in range(n_books):
            means[b] = caps[rng.choice(idx, size=len(idx), replace=True)].mean()
        return float(np.percentile(means, 50)), float(np.percentile(means, 95))

    def _summ(sub, rname):
        if not sub:
            return {"n": 0, "mean_capture": None, "null_p95": None, "beats_null": None}
        caps = np.array([r[f"cap_{rname}"] for r in sub])
        p50, p95 = _mc_null(sub)
        m = float(caps.mean())
        return {"n": len(sub), "mean_capture": round(m, 4),
                "null_analytic": round(float(np.mean([r["null_mean_capture"] for r in sub])), 4),
                "null_p95": round(p95, 4) if p95 is not None else None,
                "beats_null": bool(m > p95) if p95 is not None else None}

    per_rule_window = {rn: {w: _summ([r for r in rows if r["window"] == w], rn) for w in WINDOWS} for rn in rules}
    per_rule_held = {rn: _summ([r for r in rows if r["window"] in HELD], rn) for rn in rules}
    config = {"entry_trigger_col": entry_trigger_col, "min_hold": min_hold, "max_hold": max_hold,
              "cost": cost, "min_spread": min_spread, "rules": {k: v[1] for k, v in rules.items()},
              "confound_control": {"hold_window": "fixed [min_hold,max_hold] -> all rules+null share the exit pool",
                                   "regime": "entry-bar past-only vol tercile", "entry": "fixed -> pure EXIT skill"}}
    return ExitCaptureResult(rows=rows, per_rule_window=per_rule_window, per_rule_held=per_rule_held,
                             config=config, rng_seed=seed)


def timing_skill_vs_baseline(res: ExitCaptureResult, baseline: str = "fixed_h5", window_set=HELD) -> dict:
    """The FAIR exit-TIMING test (added 2026-06-08 after a hold-length artifact was caught): `beats_null` alone is
    CONFOUNDED by hold-length -- the random-exit null over the full [min_hold,max_hold] window includes late bars, so
    a NO-SKILL short fixed hold "beats" it whenever the move fades (a horizon effect, not timing skill). To isolate
    TIMING, compare each rule's capture to a NO-SKILL fixed-hold baseline of similar length: a rule has genuine
    exit-timing skill only if its held-out capture exceeds the baseline's. Returns per-rule {delta_vs_baseline,
    frac_moves_better, timing_skill(bool)}."""
    rows = [r for r in res.rows if r["window"] in window_set]
    bcol = f"cap_{baseline}"
    out = {}
    if not rows or bcol not in rows[0]:
        return out
    base = np.array([r[bcol] for r in rows])
    for rn in res.per_rule_held:
        if rn == baseline:
            continue
        rcol = f"cap_{rn}"
        rv = np.array([r[rcol] for r in rows])
        delta = float((rv - base).mean())
        frac_better = float((rv > base).mean())
        # genuine timing skill requires BOTH a positive mean delta AND beating the no-skill baseline on a MAJORITY of
        # moves (frac_better > 0.5). A positive delta alone can be a few large moves = noise; the majority criterion
        # is the median-sign check that the small-n mean can't fake. (Tightened 2026-06-08 after a +0.022/49% blip.)
        out[rn] = {"delta_vs_noskill_baseline": round(delta, 4),
                   "frac_moves_better": round(frac_better, 3),
                   "timing_skill": bool(delta > 0.0 and frac_better > 0.5)}
    return out


def write_artifacts(res: ExitCaptureResult, tag: str, out_dir: Path = ARTIFACT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"exit_capture_{tag}.csv"
    json_path = out_dir / f"exit_capture_{tag}.json"
    res.to_frame().to_csv(csv_path, index=False)
    summary = {"tag": tag, "n_moves": len(res.rows), "config": res.config, "rng_seed": res.rng_seed,
               "per_rule_window": res.per_rule_window, "per_rule_held": res.per_rule_held}
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    tmp.replace(json_path)
    return {"csv": str(csv_path), "json": str(json_path)}


def verify_artifact(csv_path: str | Path) -> tuple[bool, str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, f"artifact missing: {csv_path}"
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return False, "artifact has zero rows"
    capcols = [c for c in df.columns if c.startswith("cap_")]
    if not capcols:
        return False, "no cap_ columns"
    for c in capcols:
        v = df[c].to_numpy(float)
        if not np.all(np.isfinite(v)) or v.min() < 0.0 or v.max() > 1.0:
            return False, f"{c} out of [0,1]: [{v.min()}, {v.max()}]"
    return True, f"OK: {len(df)} moves, {len(capcols)} exit rules, all captures bounded in [0,1]"


# ---------------------------------------------------------------------------
def _load_1d(sym="BTCUSDT") -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(sym, cadence="1d")
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                         "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                         "close": np.asarray(d["close"], float)})


def _dry_run() -> int:
    print("=" * 88)
    print("EXIT-METHODOLOGY CAPTURE PROXY -- dry-run on BTCUSDT 1d (does exit SKILL beat random exit, held-out?)")
    print("=" * 88)
    df = _load_1d("BTCUSDT")
    print(f"  loaded {len(df)} bars: {df['date'].min().date()} -> {df['date'].max().date()}")
    prior_max = df["close"].rolling(20).max().shift(1)
    df["breakout"] = (df["close"] > prior_max).fillna(False)
    print(f"  entry trigger = 20-bar breakout; firings = {int(df['breakout'].sum())}")
    res = exit_capture_proxy(df, "breakout", min_hold=1, max_hold=20, cost=TAKER)
    paths = write_artifacts(res, tag="BTCUSDT_1d_breakout")
    print(f"  artifact: {paths['csv']}")
    print("\n  HELD-OUT (OOS+UNSEEN) exit-capture per rule vs random-exit null p95:")
    for rn, s in res.per_rule_held.items():
        print(f"    {rn:14} n={s['n']:<4} mean_cap={s['mean_capture']}  null_p95={s['null_p95']}  "
              f"beats_null={s['beats_null']}")
    ok, msg = verify_artifact(paths["csv"])
    print("\n  FAIR exit-TIMING test (smart rule vs NO-SKILL fixed-hold baseline -- isolates timing from hold-length):")
    ts = timing_skill_vs_baseline(res, baseline="fixed_h5")
    for rn, s in ts.items():
        print(f"    {rn:14} delta_vs_noskill={s['delta_vs_noskill_baseline']:+.4f}  "
              f"better_on={s['frac_moves_better']:.0%} of moves  timing_skill={s['timing_skill']}")
    print("\n" + "-" * 88)
    print(f"  VERIFY: {msg}")
    real_timing_skill = any(s["timing_skill"] for s in ts.values())
    print(f"  VERDICT (FAIR): genuine exit-TIMING skill held-out = {real_timing_skill}. "
          f"NOTE: 'beats_null' alone is hold-length-CONFOUNDED (a no-skill short fixed hold beats the fade-dragged "
          f"random-exit null) -- trust the smart-vs-noskill delta, not beats_null.")
    print(f"[exit_capture_proxy] {'PASS' if ok else 'FAIL'} -- artifact emitted, all captures in [0,1].")
    return 0 if ok else 1


def _make_synth(seed=3, n=1400):
    """Moves that RUN UP then FADE (peak mid-hold) -> a trailing-stop / MA-cross exit should capture HIGH vs random
    exit timing or a too-long fixed hold (which gives the gains back)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.008, n)
    run = 0
    is_entry = np.zeros(n, bool)
    for t in range(1, n):
        if run > 0:
            # an up-then-down pulse: first half up, second half down (a peak inside the hold window)
            phase = run
            rets[t] += 0.018 if phase > 5 else -0.020
            run -= 1
        elif rng.random() < 0.04:
            is_entry[t] = True
            run = 10
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    df["entry"] = is_entry
    return df


def _selftest() -> int:
    print("=" * 88)
    print("[exit_capture_proxy selftest] synthetic two-sided soundness: a SMART exit beats random; a fixed long")
    print("hold does NOT (it gives the peak back). No market data.")
    print("=" * 88)
    df = _make_synth()
    win = Windows(train_end="2023-06-01", val_end="2024-01-01", oos_end="2024-09-01")
    # fixed_h20 = hold to the window end (gives the fade back -> low capture, ~ the worst); trailing/ma = smart
    rules = {"fixed_h20": (exit_fixed, {"hold": 20}),
             "trailing_5pct": (exit_trailing_stop, {"drop": 0.05}),
             "ma_cross_5_20": (exit_ma_cross, {"fast": 5, "slow": 20})}
    res = exit_capture_proxy(df, "entry", min_hold=1, max_hold=20, cost=TAKER, windows=win, rules=rules)
    paths = write_artifacts(res, tag="selftest_exit")
    ok_b, msg_b = verify_artifact(paths["csv"])
    h = res.per_rule_held
    print(f"\n  HELD-OUT exit-capture:")
    for rn, s in h.items():
        print(f"    {rn:14} mean_cap={s['mean_capture']}  beats_null={s['beats_null']}  (n={s['n']})")
    smart_beats = bool(h["trailing_5pct"]["beats_null"]) or bool(h["ma_cross_5_20"]["beats_null"])
    smart_over_fixed = ((h["trailing_5pct"]["mean_capture"] or 0) > (h["fixed_h20"]["mean_capture"] or 0))
    print("\n" + "-" * 88)
    print("SOUNDNESS:")
    print(f"  bounds [0,1] hold                                   : {ok_b}")
    print(f"  a SMART exit (trailing/MA) BEATS random-exit null   : {smart_beats}")
    print(f"  smart-exit capture > naive long-hold capture        : {smart_over_fixed}")
    ok = ok_b and smart_beats and smart_over_fixed
    print(f"\n[exit_capture_proxy selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else _dry_run())
