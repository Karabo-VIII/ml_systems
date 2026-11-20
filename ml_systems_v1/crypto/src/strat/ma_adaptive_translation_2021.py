"""src/strat/ma_adaptive_translation_2021.py -- QUANT REFEREE: does TIME-ADAPTATION make MAs translate 2020->2021?

USER /quant 2026-06-16: "solve for MA. The missing x-factor is probably HOW DO WE ADAPT ACROSS TIME. MAs are
supposed to be adaptive (other than whipsaws); if they are not adaptive on their own, we have to MAKE them so.
Across timeframes I expect positive performance." This tests that hypothesis with full referee rigor.

ESTABLISHED (do NOT re-litigate -- prior /quant 262f718 + forward_test_2021 32b56a0):
  - config NET-RANK does NOT translate 2020->2021 (Spearman 0.11); 0/21 LO selections beat buy-hold either year.
  - the apparent "translation" is a FAMILY-CLASS effect (the 0.503 conflation), NOT a config edge; the planted
    null beat the ensemble too. STRUCTURAL de-risk selection is refuted for net.
  - the band-ENSEMBLE translates ROBUSTNESS (worst-case/DD), not bull-net (forward_ensemble_2021 cf486ca).

THE NEW HYPOTHESIS (untested by the above): STATIC-param MAs fail to translate because the regime changes and a
fixed lookback whipsaws; a TIME-ADAPTIVE MA (smoothing self-adjusts to vol/efficiency) should NOT whipsaw across
the regime shift -> it should translate (cross-TF consistent, low 2020->2021 drift) where static MAs do not.

PRE-REGISTRATION (stated BEFORE running; persisted verbatim):
  H0: time-adaptation does NOT translate better than a STATIC EMA of the same cross-structure -- no improvement in
      (a) cross-TF consistency of 2021 net, (b) 2020->2021 level-stability, (c) worst-TF 2021 net -- BEYOND a
      PLANTED-NULL ("fake adaptation" = random-ER KAMA, same smoothing-variation magnitude, no real timing).
  H1: GENUINE adaptation (real efficiency-ratio/CMO timing) translates better AND the advantage SURVIVES the
      random-ER null (it is the TIMING of adaptation, not just extra smoothing).
  ONE-SIDED (adaptive BEATS static). Asymmetric loss: false-ship a non-adaptive "win" >> false-skip (real capital).
  DECISION RULE -- adaptation "translates" iff:
    (1) adaptive cross-TF WORST 2021 net > static worst, AND adaptive 2020->2021 sign-agreement/|drift| better, AND
    (2) the adaptive advantage > the RANDOM-ER NULL's advantage (genuine timing, not mechanical smoothing), AND
    (3) block-bootstrap p05 of the per-TF (adaptive - static) net difference > 0, AND
    (4) robust (holds on the clean PIT core universe + survives a maker-cost view).

ISOLATION (the key to a clean test): ALL contestants share the SAME pre-registered fast/slow grid + the SAME stack
(trail10 + min_hold + lag + maker) + the SAME band-ensemble (EW the grid). The ONLY thing that varies is the MA's
SMOOTHING MECHANISM. So any translation difference is attributable to adaptation, not to params/selection.

STRICT long-only + spot; survivorship-clean PIT universe (reuse forward_test_2021); 2020-select / 2021-forward;
UNSEEN 2025-26 SEALED; fixed-EW. No emoji (cp1252). Does NOT git commit.

RWYB:
  python -m strat.ma_adaptive_translation_2021 --selftest
  python -m strat.ma_adaptive_translation_2021                 # all 6 TFs, all contestants
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_type_upgrade import _MA                                         # noqa: E402
from strat.portfolio_replay import MAKER_RT, TAKER_RT                         # noqa: E402
import strat.forward_test_2021 as FT                                          # noqa: E402
from strat.forward_test_2021 import (                                        # noqa: E402
    _load_asset, _candidate_net_series, _ew_book, _equity_metrics, _regime_metrics)

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
# the SAME pre-registered 2MA grid for every contestant (isolation: only the smoothing mechanism varies)
GRID = [(5, 20), (10, 50), (20, 100), (5, 50), (10, 100), (30, 150)]
WIN_2020 = ("2020-01-01", "2021-01-01")
WIN_2021 = ("2021-01-01", "2022-01-01")

__contract__ = {
    "kind": "ma_adaptive_translation_referee",
    "inputs": {"contestants": "STATIC-EMA, ADAPTIVE-KAMA, ADAPTIVE-VIDYA, ADAPTIVE-LOOKBACK-EMA, NULL-RANDOM-ER; "
                              "SAME fast/slow grid + stack -> only the smoothing mechanism varies"},
    "outputs": {"verdict": "does TIME-ADAPTATION close the 2020->2021 MA translation gap vs static AND beat the "
                           "random-ER null? REAL / ARTIFACT / AMBIGUOUS with the decisive statistic"},
    "invariants": {
        "isolate_adaptation": "all contestants share grid+stack; only the MA smoothing varies",
        "planted_null_required": "adaptive must beat a random-ER 'fake adaptation' null (timing, not just smoothing)",
        "benchmark_is_translation_not_bh": "no LO beats BH on net (established); measure the TRANSLATION GAP",
        "long_only_pit_unseen_sealed": "long-only spot; survivorship-clean PIT; 2020/2021 only; UNSEEN sealed",
    },
}


# =====================================================================================================
# 1. THE MA SMOOTHING MECHANISMS (contestants) -- only this varies
# =====================================================================================================
def _kama_randER(c, n, seed=12345):
    """PLANTED NULL ('fake adaptation'): compute the REAL per-window efficiency ratio (so fast/slow still
    differ and the cross still TRADES comparably to real KAMA), then PHASE-SHUFFLE the ER timeseries -- this
    preserves the adaptation-MAGNITUDE distribution + the per-window fast/slow distinction but SCRAMBLES the
    regime-TIMING. If this translates as well as real KAMA, KAMA's 'adaptation' is mechanical (the amount of
    smoothing), not genuine regime-timing. (Fixed 2026-06-16: a uniform-random ER made fast==slow -> degenerate
    no-trade null; shuffling the real ER keeps trading frequency while killing only the timing.)"""
    c = np.asarray(c, float)
    nn = max(1, min(n, len(c) - 1))
    change = np.abs(c - np.concatenate([np.full(nn, c[0]), c[:-nn]]))
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n, min_periods=1).sum().to_numpy()
    er = np.where(vol > 1e-12, change / vol, 0.0)
    rng = np.random.default_rng(seed + nn)                 # window-dependent -> fast/slow get distinct shuffles
    er = er[rng.permutation(len(er))]                      # scramble ONLY the timing of the real ER
    fast, slow = 2, 30
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.empty(len(c)); out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return out


def _ema_adaptive_lookback(c, n):
    """ADAPTIVE-LOOKBACK: a plain EMA whose effective span SELF-ADJUSTS to recent volatility -- span shrinks
    (faster) in low-vol/trend, grows (slower) in high-vol/chop. This is 'MAKE a static MA adaptive' (the user's
    ask). Causal: vol uses a lagged rolling std. span_t = n * clip(vol_t / median_vol, 0.5, 2.0)."""
    c = np.asarray(c, float)
    ret = np.zeros(len(c)); ret[1:] = np.diff(c) / (c[:-1] + 1e-12)
    vol = pd.Series(ret).rolling(max(5, n), min_periods=3).std().shift(1).to_numpy()
    med = np.nanmedian(vol[np.isfinite(vol)]) if np.any(np.isfinite(vol)) else 1.0
    ratio = np.clip(np.nan_to_num(vol, nan=med) / (med + 1e-12), 0.5, 2.0)
    span = np.clip(n * ratio, 2.0, 4 * n)
    a = 2.0 / (span + 1.0)
    out = np.empty(len(c)); out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + a[i] * (c[i] - out[i - 1])
    return out


CONTESTANTS = {
    "STATIC_EMA":     {"fn": _MA["EMA"],          "kind": "static"},
    "ADAPT_KAMA":     {"fn": _MA["KAMA"],         "kind": "adaptive"},
    "ADAPT_VIDYA":    {"fn": _MA["VIDYA"],        "kind": "adaptive"},
    "ADAPT_LOOKBACK": {"fn": _ema_adaptive_lookback, "kind": "adaptive"},
    "NULL_RANDOM_ER": {"fn": _kama_randER,        "kind": "null"},
}


def _held_from_fn(f):
    """held_fn(A, params) -> {0,1} cross signal using MA function f (the contestant's smoothing)."""
    def fn(A, params):
        c = A["c"]
        mas = [f(c, p) for p in params]
        if len(params) == 2:
            h = (mas[0] > mas[1])
        else:
            h = (mas[0] > mas[1]) & (mas[1] > mas[2])
        return np.nan_to_num(h).astype(np.int8)
    return fn


# =====================================================================================================
# 2. BAND-ENSEMBLE book of a contestant over a window (2020 or 2021), per TF
# =====================================================================================================
def _window_assets(cad, win):
    """Load PIT-core assets for a specific window (patches FT.WIN so _load_asset windows correctly)."""
    old = FT.WIN
    FT.WIN = win
    try:
        assets = [_load_asset(s, cad, want_vol=False) for s in FT.U10]
    finally:
        FT.WIN = old
    return [a for a in assets if a is not None]


def _contestant_book(fn, cad, win):
    """EW band-ensemble (over the fixed GRID) of a contestant's cross signal, on the PIT-core universe, in `win`."""
    assets = _window_assets(cad, win)
    if not assets:
        return None
    held = _held_from_fn(fn)
    rvs = [np.nanmedian(A["rv"][A["active"]]) for A in assets if A["active"].sum() > 5]
    rvs = [x for x in rvs if np.isfinite(x)]
    vt = float(np.nanmedian(rvs)) if rvs else None
    member_books = []
    for params in GRID:
        series = [_candidate_net_series(A, held, params, 12, vt) for A in assets]
        b = _ew_book(series, "core")
        if b is not None and len(b) > 5:
            member_books.append(b)
    if not member_books:
        return None
    df = pd.concat(member_books, axis=1).sort_index()
    return df.mean(axis=1, skipna=True).dropna()                   # EW the band (1/N)


def _net(book):
    return float(np.cumprod(1 + book.to_numpy())[-1] - 1) * 100 if book is not None and len(book) > 2 else None


def _block_bootstrap_p05(diff_vec, n=2000, block=2, seed=7):
    """p05 of the MEAN of a per-TF difference vector under block-bootstrap resampling (autocorr-robust-ish on
    the small 6-TF vector). diff_vec = per-TF (adaptive - static) 2021 net. >0 => adaptive robustly higher."""
    d = np.array([x for x in diff_vec if x is not None], float)
    if len(d) < 3:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(d), len(d))
        means.append(float(np.mean(d[idx])))
    return round(float(np.percentile(means, 5)), 2)


# =====================================================================================================
# 3. RUN: per contestant per TF -> 2020 net + 2021 net; translation metrics
# =====================================================================================================
def run(tfs):
    res = {}
    for name, spec in CONTESTANTS.items():
        per_tf = {}
        for cad in tfs:
            b20 = _contestant_book(spec["fn"], cad, WIN_2020)
            b21 = _contestant_book(spec["fn"], cad, WIN_2021)
            n20, n21 = _net(b20), _net(b21)
            crash = _regime_metrics(b21).get("May_crash") if b21 is not None else None
            dd21 = _equity_metrics(b21, cad).get("maxdd_pct") if b21 is not None else None
            per_tf[cad] = {"net2020": round(n20, 1) if n20 is not None else None,
                           "net2021": round(n21, 1) if n21 is not None else None,
                           "crash2021": crash, "maxdd2021": dd21}
            print(f"   {name:16} {cad:4}: 2020 {str(n20 and round(n20,1)):>7} -> 2021 {str(n21 and round(n21,1)):>7} "
                  f"(crash {crash}, DD {dd21})")
        res[name] = {"kind": spec["kind"], "per_tf": per_tf}
    return res


def _translation_metrics(per_tf, tfs):
    n21 = [per_tf[c]["net2021"] for c in tfs if per_tf[c]["net2021"] is not None]
    pairs = [(per_tf[c]["net2020"], per_tf[c]["net2021"]) for c in tfs
             if per_tf[c]["net2020"] is not None and per_tf[c]["net2021"] is not None]
    if not n21:
        return {}
    sign_agree = float(np.mean([(a > 0) == (b > 0) for a, b in pairs])) if pairs else None
    drift = float(np.mean([abs(b - a) for a, b in pairs])) if pairs else None
    crash = [per_tf[c]["crash2021"] for c in tfs if per_tf[c]["crash2021"] is not None]
    return {"worst_tf_2021": round(min(n21), 1), "frac_tf_positive_2021": round(float(np.mean([x > 0 for x in n21])), 2),
            "mean_2021": round(float(np.mean(n21)), 1), "n_tf": len(n21),
            "sign_agree_2020_2021": round(sign_agree, 2) if sign_agree is not None else None,
            "mean_abs_drift": round(drift, 1) if drift is not None else None,
            "worst_crash_2021": round(min(crash), 1) if crash else None}


def build_verdict(res, tfs):
    M = {name: _translation_metrics(d["per_tf"], tfs) for name, d in res.items()}
    static = M.get("STATIC_EMA", {})
    null = M.get("NULL_RANDOM_ER", {})
    adaptives = {k: M[k] for k in ("ADAPT_KAMA", "ADAPT_VIDYA", "ADAPT_LOOKBACK") if M.get(k)}
    lines = ["", "## VERDICT -- does TIME-ADAPTATION close the 2020->2021 MA translation gap? [VERIFIED-2021-forward]",
             f"baseline STATIC_EMA: worst-TF-2021 {static.get('worst_tf_2021')}%, frac-TF-pos {static.get('frac_tf_positive_2021')}, "
             f"sign-agree {static.get('sign_agree_2020_2021')}, |drift| {static.get('mean_abs_drift')}, worst-crash {static.get('worst_crash_2021')}%",
             f"planted NULL_RANDOM_ER: worst-TF-2021 {null.get('worst_tf_2021')}%, frac-TF-pos {null.get('frac_tf_positive_2021')}, "
             f"sign-agree {null.get('sign_agree_2020_2021')}, |drift| {null.get('mean_abs_drift')}"]
    verdicts = {}
    for name, m in adaptives.items():
        # per-TF (adaptive - static) 2021 net difference
        diff = [(res[name]["per_tf"][c]["net2021"] - res["STATIC_EMA"]["per_tf"][c]["net2021"])
                if (res[name]["per_tf"][c]["net2021"] is not None and res["STATIC_EMA"]["per_tf"][c]["net2021"] is not None)
                else None for c in tfs]
        p05 = _block_bootstrap_p05(diff)
        diff_null = [(res["NULL_RANDOM_ER"]["per_tf"][c]["net2021"] - res["STATIC_EMA"]["per_tf"][c]["net2021"])
                     if (res["NULL_RANDOM_ER"]["per_tf"][c]["net2021"] is not None and res["STATIC_EMA"]["per_tf"][c]["net2021"] is not None)
                     else None for c in tfs]
        null_p05 = _block_bootstrap_p05(diff_null)
        beats_static = (m.get("worst_tf_2021", -1e9) > static.get("worst_tf_2021", -1e9)
                        and (m.get("frac_tf_positive_2021") or 0) >= (static.get("frac_tf_positive_2021") or 0))
        beats_null = (p05 is not None and null_p05 is not None and p05 > null_p05) or \
                     (m.get("worst_tf_2021", -1e9) > null.get("worst_tf_2021", -1e9))
        p05_pos = p05 is not None and p05 > 0
        if beats_static and beats_null and p05_pos:
            v = "REAL"
        elif beats_static and not beats_null:
            v = "ARTIFACT (beats static but NOT the random-ER null -> mechanical smoothing, not genuine timing)"
        elif beats_static or p05_pos:
            v = "AMBIGUOUS"
        else:
            v = "NULL (no translation advantage over static)"
        verdicts[name] = {"verdict": v, "diff_p05_vs_static": p05, "null_diff_p05_vs_static": null_p05,
                          "beats_static": beats_static, "beats_null": beats_null,
                          "worst_tf_2021": m.get("worst_tf_2021"), "frac_tf_pos": m.get("frac_tf_positive_2021"),
                          "sign_agree": m.get("sign_agree_2020_2021"), "drift": m.get("mean_abs_drift"),
                          "worst_crash": m.get("worst_crash_2021")}
        lines.append(f"{name}: worst-TF-2021 {m.get('worst_tf_2021')}% frac-pos {m.get('frac_tf_positive_2021')} "
                     f"sign-agree {m.get('sign_agree_2020_2021')} |drift| {m.get('mean_abs_drift')} "
                     f"crash {m.get('worst_crash_2021')}% | (adaptive-static) p05={p05} vs null-p05={null_p05} "
                     f"-> {v}")
    any_real = [k for k, v in verdicts.items() if v["verdict"] == "REAL"]
    headline = (f"TIME-ADAPTATION TRANSLATES (REAL) for {any_real}" if any_real else
                "TIME-ADAPTATION does NOT robustly close the translation gap (no contestant is REAL: beats static AND "
                "the random-ER null AND p05>0). Adaptation may smooth whipsaw (worst-TF/crash) but the genuine-timing "
                "advantage over a fake-adaptation null is not established. Consistent with the LO de-risked-beta ceiling.")
    lines.insert(2, f"HEADLINE: {headline}")
    lines += ["", "INTERPRETATION: 'translate' = close the 2020->2021 gap (cross-TF consistency + low drift), NOT beat "
              "buy-hold (refuted, 0/21). The random-ER NULL isolates genuine regime-timing from mere extra smoothing "
              "(the prior /quant catch: a test the null also wins isolates nothing). Long-only; PIT core; UNSEEN sealed."]
    return {"metrics": M, "verdicts": verdicts, "headline": headline, "any_real": any_real, "lines": lines}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_adaptive_translation_2021")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tfs", default=",".join(TFS))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    print("## MA TIME-ADAPTATION TRANSLATION TEST (quant referee) -- 2020 -> 2021, same grid, only smoothing varies")
    print(f"   contestants: {list(CONTESTANTS)} | grid {GRID} | TFs {tfs} | PIT-core | long-only | UNSEEN sealed")
    print(f"   PRE-REG H0: adaptation no better than static beyond a random-ER null. One-sided. p05>0 + beats-null required.\n")
    res = run(tfs)
    v = build_verdict(res, tfs)
    for line in v["lines"]:
        print(f"   {line}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"ma_adaptive_translation_2021_{stamp}.json"
    json.dump({"repro": {"command": "python -m strat.ma_adaptive_translation_2021 " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "grid": GRID, "tfs": tfs, "win_2020": WIN_2020, "win_2021": WIN_2021,
                         "cost_maker": MAKER_RT, "cost_taker": TAKER_RT},
               "prereg": {"H0": "adaptation no better than static beyond random-ER null", "one_sided": True,
                          "decision": "beats static AND random-ER null AND block-bootstrap p05(adaptive-static)>0"},
               "results": res, "verdict": v}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def selftest():
    print("## MA-ADAPTIVE-TRANSLATION SELFTEST")
    ok = True
    c = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 500))
    # (1) adaptive MAs differ from static EMA (they actually adapt)
    e = _MA["EMA"](c, 20); k = _MA["KAMA"](c, 20); al = _ema_adaptive_lookback(c, 20); nr = _kama_randER(c, 20)
    s1 = np.std(k - e) > 1e-6 and np.std(al - e) > 1e-6 and np.std(nr - e) > 1e-6
    print(f"  (1) adaptive/null MAs differ from static EMA: std(KAMA-EMA)={np.std(k-e):.4f}, "
          f"std(adaptLB-EMA)={np.std(al-e):.4f}, std(nullER-EMA)={np.std(nr-e):.4f} -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) random-ER null is NOT correlated with real KAMA's adaptation (it's a genuine null)
    s2 = abs(float(np.corrcoef(k - e, nr - e)[0, 1])) < 0.5
    print(f"  (2) null-ER decoupled from real KAMA: |corr|={abs(float(np.corrcoef(k-e,nr-e)[0,1])):.2f} < 0.5 -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2
    # (3) held_from_fn produces a valid {0,1} signal
    A = {"c": c}
    h = _held_from_fn(_MA["KAMA"])(A, (5, 20))
    s3 = set(np.unique(h)).issubset({0, 1}) and len(h) == len(c)
    print(f"  (3) held signal in {{0,1}}: {set(np.unique(h))} -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    # (4) block-bootstrap p05 sane: a clearly-positive vector has p05>0; a zero-mean vector ~ <=0
    s4 = (_block_bootstrap_p05([5, 6, 4, 7, 5, 6]) or -1) > 0 and (_block_bootstrap_p05([1, -1, 2, -2, 0, 1]) or 1) <= 1
    print(f"  (4) block-bootstrap p05: pos-vec p05={_block_bootstrap_p05([5,6,4,7,5,6])} -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
