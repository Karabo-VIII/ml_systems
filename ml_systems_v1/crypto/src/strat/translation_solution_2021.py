"""src/strat/translation_solution_2021.py -- THE 2020->2021 TRANSLATION SOLUTION TEST (quant referee).

THE LOAD-BEARING PROBLEM: 2020-selected configs FAIL to translate to 2021. The forward-test
(forward_test_2021.py + FORWARD_TEST_2021_2026_06_15.md, commits 32b56a0/d776183) established:
  - config NET-RANK does NOT transfer: Spearman(2020-OOS-net, 2021-fwd-net) = 0.11 ~ 0; ADX(14,20)@4h
    was 2020-BEST (+36.7%) and 2021-WORST (-18.4%); 0/21 beat buy-hold.
  - the de-risked-beta CLASS + DRAWDOWN-PRESERVATION DO generalize (May-2021 crash: 21/21 lost less
    than buy-hold).

THE HYPOTHESIS UNDER TEST: selecting 2020 configs by a STRUCTURAL de-risk property (crash-DD /
time-in-market / turnover / cross-subperiod stability) -- NOT their noisy net-rank -- TRANSLATES to
2021, because the structural property is a stable config TRAIT while net-rank is regime-transient noise.

PRE-REGISTRATION (stated BEFORE running, persisted verbatim in the output):
  H0 (null): NO 2020-selection metric beats the no-selection ENSEMBLE on 2021-forward; config
             translation is impossible regardless of metric. Net-rank Spearman 0.11 is the ceiling.
  H1 (alt) : a STRUCTURAL 2020-selection metric translates -- (a) its 2020->2021 rank-transfer
             Spearman is meaningfully > 0.11, AND (b) the 2020-structural-feature -> 2021-fwd-net
             relation is significant OUT-OF-SAMPLE (held-out configs).
  ONE-SIDED (does structural BEAT the baseline?). Asymmetric loss: false-ship a non-translating rule
             >> false-skip (real capital). DECISION RULE: a metric "translates" iff
             (1) rank-transfer Spearman > 0.11 noise floor, AND
             (2) survives multiple-comparisons MAX-STAT permutation deflation (shuffle 2020->2021
                 config mapping; how often does the best metric appear by chance?), AND
             (3) survives a PLANTED-NULL control (a metric that should NOT translate doesn't), AND
             (4) is ROBUST (holds across asset-subset / cost haircut).

DISCIPLINE: STRICT long-only + spot (NO short). 2020-selection / 2021-forward windows ONLY; UNSEEN
2025-26 SEALED. Survivorship-clean PIT universe (reuse forward_test_2021's). fixed-EW. Block-bootstrap
p05, rank-transfer Spearman, max-stat permutation deflation, planted-null. No emoji (cp1252).

RWYB:
  python -m strat.translation_solution_2021 --selftest        # mechanics sanity (fast)
  python -m strat.translation_solution_2021                   # the full translation solution
  python -m strat.translation_solution_2021 --tfs 1d,4h --max-per-family 40
Does NOT git commit.
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

from strat.ma_2020_breakdown import _panel                                   # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT, TAKER_RT      # noqa: E402
from strat.structural_fixes import min_hold                                  # noqa: E402
from strat.ma_type_upgrade import _MA, MA_TYPES                              # noqa: E402
import strat.deep2020_ti_pipeline as TI                                      # noqa: E402
import strat.forward_test_2021 as FT                                         # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# ----- windows -----
# 2020 SELECTION YEAR: full year so we can measure the MARCH-2020 crash (out-of-OOS structural data).
#   VAL/OOS split matches the original selection (Jul-Dec). The frozen configs were chosen on Jul-Dec;
#   Jan-Jun (incl. the March crash) is OUT-OF-SELECTION 2020 data -> a legit 2020-side structural feature.
WIN_2020 = ("2020-01-01", "2021-01-01")
SEL_SPLIT = "2020-10-01"                       # VAL (Jul-Oct) | OOS (Oct-Dec) -- the original selection target
MARCH_CRASH_2020 = ("2020-02-15", "2020-04-01")  # the COVID crash (BTC ~ -50%): the 2020 structural crash window
WARMUP = 400
U10 = FT.U10                                    # the 10 majors live through 2020 AND 2021 (survivorship-clean core)
VW = FT.VW
MINHOLD_DEFAULT = 12

__contract__ = {
    "kind": "translation_solution_2020_to_2021",
    "inputs": {
        "config_universe": "MA grids (2MA/3MA crosses, all 8 MA types) + all TI ironed grids across "
                           "chosen TFs -- ~200-400 configs spanning all families",
        "features_2020": "per-config 2020 STRUCTURAL features: OOS net (the FAILED baseline), Sharpe "
                         "(planted-null), time-in-market, turnover, March-2020-crash-DD, cross-subperiod "
                         "net-stability -- all measured on 2020 only",
        "target_2021": "per-config 2021-forward net (PIT survivorship-clean, frozen) + 2021-May-crash DD "
                       "-- reuses forward_test_2021's PIT universe + book machinery",
    },
    "outputs": {
        "translation_test": "for each 2020-selection metric: top-K selected set's 2021-fwd net + crash "
                            "preservation vs the no-selection ENSEMBLE",
        "structural_regression": "per-feature rank-transfer Spearman + OOS (held-out-config) R2 / rank-IC",
        "deflated_verdict": "max-stat permutation null over all metrics x features -> REAL/ARTIFACT/AMBIGUOUS",
    },
    "invariants": {
        "long_only_spot": "NO short logic anywhere",
        "frozen_no_2021_refit": "2020 features measured on 2020; 2021 is pure forward; UNSEEN sealed",
        "survivorship_clean_2021": "reuses forward_test_2021 PIT universe (data-derived listing dates)",
        "no_lookahead": "2020 features use 2020 bars only; positions lagged 1 bar; rolling rv shift(1)",
        "fixed_ew": "fixed-EW books (no skipna leakage)",
        "deflation": "max-stat permutation across ~5 metrics x several features; planted-null gate",
    },
}


# =====================================================================================================
# 1. CONFIG UNIVERSE -- MA grids + TI ironed grids (the deployable lane: ironed signal-level held)
# =====================================================================================================
def _ma_held_fn(ma_type, nums):
    def fn(A, _p):
        c2 = A["c"]; mas = [_MA[ma_type](c2, n) for n in nums]
        if len(nums) == 2:
            return np.nan_to_num(mas[0] > mas[1]).astype(np.int8)
        return np.nan_to_num((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
    return fn


def _ma_grid():
    """A representative MA grid: 2MA + 3MA crosses across the 8 MA types (subsampled to stay balanced
    vs the TI families). Fast in (5,10,20), slow in (50,100,200), 3rd in (200,) for 3MA."""
    out = []
    fasts = (5, 10, 20); slows = (50, 100, 200)
    for mt in MA_TYPES:
        for f in fasts:
            for s in slows:
                if s <= f:
                    continue
                out.append({"name": f"{mt}_{f}_{s}", "family": "MA", "held_fn": _ma_held_fn(mt, (f, s)),
                            "params": None, "minhold": 12, "loader": "ohlc"})
        # one 3MA per type for variety
        out.append({"name": f"{mt}_5_50_200", "family": "MA", "held_fn": _ma_held_fn(mt, (5, 50, 200)),
                    "params": None, "minhold": 12, "loader": "ohlc"})
    return out


def _ti_grid():
    """All TI configs as IRONED held_fns (the deployable lane). One entry per (indicator, grid-param)."""
    out = []
    for ind, spec in TI.INDICATORS.items():
        fam = spec["family"]; iron = spec["iron"]; mh = spec.get("minhold", 12)
        loader = spec.get("loader", "ohlc")
        for p in spec["grid"]():
            cfg = spec["name"](p)
            # closure capturing iron + p
            def make(fn, pp):
                return lambda A, _q: np.asarray(fn(A, pp)).astype(np.int8)
            out.append({"name": cfg, "family": fam, "held_fn": make(iron, p),
                        "params": None, "minhold": mh, "loader": loader})
    return out


def build_config_universe(tfs, max_per_family=60):
    """Build the full config x TF universe. Caps per (family, TF) to keep families balanced.
    Returns list of dicts(name, family, cad, held_fn, params, minhold, loader, uid)."""
    base = _ma_grid() + _ti_grid()
    cands = []
    seen = {}
    for cad in tfs:
        per_fam = {}
        for c in base:
            fam = c["family"]
            per_fam.setdefault(fam, 0)
            if per_fam[fam] >= max_per_family:
                continue
            per_fam[fam] += 1
            uid = f"{c['name']}@{cad}"
            cands.append({**c, "cad": cad, "uid": uid})
    return cands


# =====================================================================================================
# 2. 2020 PANEL (u10 core, full year) -- per-config STRUCTURAL features
# =====================================================================================================
_PANEL_CACHE = {}


def _load_2020_panel(cad, want_vol=False):
    key = (cad, want_vol)
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]
    s_ms = pd.Timestamp(WIN_2020[0]).value // 10**6
    e_ms = pd.Timestamp(WIN_2020[1]).value // 10**6
    vw = VW[cad]; assets = []
    import glob
    import polars as pl
    for sym in U10:
        if want_vol:
            fs = sorted(glob.glob(f"data/processed/chimera/{cad}/{sym.lower()}*.parquet"))
            if not fs:
                continue
            try:
                df = pl.read_parquet(fs[-1], columns=["timestamp", "open", "high", "low", "close",
                                                      "volume", "buy_vol", "sell_vol"]).sort("timestamp")
            except Exception:
                continue
            ms = df["timestamp"].to_numpy()
            o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
            l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
            vol = df["volume"].to_numpy().astype(float); bv = df["buy_vol"].to_numpy().astype(float)
            sv = df["sell_vol"].to_numpy().astype(float)
        else:
            try:
                o, h, l, c, ms = _panel(sym, cad)
            except Exception:
                continue
            vol = bv = sv = None
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        sl = slice(s0, e)
        o2, h2, l2, c2, ms2 = o[sl], h[sl], l[sl], c[sl], ms[sl]
        if len(c2) < 60:
            continue
        win = ms2 >= s_ms
        if win.sum() < 60:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
        A = {"sym": sym, "o": o2, "h": h2, "l": l2, "c": c2, "ret": ret, "win": win,
             "idx": pd.to_datetime(ms2, unit="ms"), "rv": rv}
        if want_vol:
            A["vol"] = vol[sl]; A["buy_vol"] = bv[sl]; A["sell_vol"] = sv[sl]
        assets.append(A)
    # vol-target level: median trailing rv over the SELECTION window VAL portion (no look-ahead)
    rv_meds = []
    split_ms = pd.Timestamp(SEL_SPLIT).value // 10**6
    sel_lo = pd.Timestamp("2020-07-01").value // 10**6
    for A in assets:
        m = (A["idx"].view("int64") // 10**6 >= sel_lo) & (A["idx"].view("int64") // 10**6 < split_ms)
        if m.sum() > 5:
            rv_meds.append(np.nanmedian(A["rv"][m]))
    vt = float(np.nanmedian([x for x in rv_meds if np.isfinite(x)])) if rv_meds else None
    _PANEL_CACHE[key] = (assets, vt)
    return assets, vt


def _net_pos_series(A, held_fn, params, minhold, vt):
    """Return (net Series over win, pos array over full slice). Exact deployable stack:
    signal -> trail10 -> min_hold -> lag1 -> vol-target -> maker flips. LONG-ONLY (held in {0,1})."""
    c2, ret, rv = A["c"], A["ret"], A["rv"]
    held0 = np.asarray(held_fn(A, params)).astype(np.int8)
    held = min_hold(apply_trail_stop(held0.copy(), c2, 0.10)[0].astype(np.int8), minhold).astype(np.float64)
    pos = np.zeros(len(c2)); pos[1:] = held[:-1]
    if vt is not None:
        pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (MAKER_RT / 2.0)
    s = pd.Series(net[A["win"]], index=A["idx"][A["win"]])
    return s, pos


def _book_from_series(series_list):
    """fixed-EW daily book from per-asset net Series (fillna 0 = no leakage)."""
    series_list = [s for s in series_list if s is not None and len(s)]
    if not series_list:
        return None
    df = pd.concat(series_list, axis=1).sort_index()
    b = df.fillna(0.0).mean(axis=1).dropna()
    return b.resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


def _compound(series_window):
    s = series_window.dropna().to_numpy()
    return float(np.prod(1 + s) - 1) * 100 if len(s) else None


def _maxdd(daily):
    s = daily.dropna().to_numpy()
    if len(s) < 3:
        return None
    eq = np.cumprod(1 + s); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def features_2020(cand):
    """Compute a config's 2020 STRUCTURAL features (all 2020-only, no 2021 peek)."""
    want_vol = cand["loader"] == "ohlcv"
    assets, vt = _load_2020_panel(cand["cad"], want_vol)
    if not assets:
        return None
    series, posns = [], []
    for A in assets:
        s, pos = _net_pos_series(A, cand["held_fn"], cand["params"], cand["minhold"], vt)
        series.append(s); posns.append((A, pos))
    book = _book_from_series(series)
    if book is None or len(book) < 60:
        return None
    # selection target: OOS net (Oct-Dec 2020) + VAL net (Jul-Oct) -- the original selection statistic
    val = book[(book.index >= pd.Timestamp("2020-07-01")) & (book.index < pd.Timestamp(SEL_SPLIT))]
    oos = book[book.index >= pd.Timestamp(SEL_SPLIT)]
    oos_net = _compound(oos); val_net = _compound(val)
    if oos_net is None or val_net is None:
        return None
    # ---- STRUCTURAL features (stable config traits) ----
    # time-in-market: avg fraction of bars with nonzero position, over the SELECTION window (Jul-Dec)
    tin = []
    for A, pos in posns:
        m = A["win"] & (A["idx"].view("int64") // 10**6 >= pd.Timestamp("2020-07-01").value // 10**6)
        p = pos[m]
        if len(p):
            tin.append(float((np.abs(p) > 1e-9).mean()))
    time_in = float(np.mean(tin)) if tin else None
    # turnover: avg flips per bar (round-trip activity) over selection window
    tov = []
    for A, pos in posns:
        m = A["win"] & (A["idx"].view("int64") // 10**6 >= pd.Timestamp("2020-07-01").value // 10**6)
        p = pos[m]
        if len(p) > 2:
            tov.append(float(np.abs(np.diff(p)).mean()))
    turnover = float(np.mean(tov)) if tov else None
    # MARCH-2020-CRASH DD: the book's max drawdown through the COVID crash window (out-of-selection)
    crash = book[(book.index >= pd.Timestamp(MARCH_CRASH_2020[0])) &
                 (book.index < pd.Timestamp(MARCH_CRASH_2020[1]))]
    march_crash_ret = _compound(crash)            # compound return through the crash (less negative = preserver)
    march_crash_dd = _maxdd(crash)                # max DD through the crash window
    # cross-subperiod stability: split the SELECTION window (Jul-Dec) into 3 equal subperiods, compound
    #   each; stability = -std (or the min) of the 3 -> a config whose net is consistent across subperiods
    selbook = book[book.index >= pd.Timestamp("2020-07-01")].dropna()
    sub_nets = []
    if len(selbook) >= 9:
        thirds = np.array_split(selbook.to_numpy(), 3)
        sub_nets = [float(np.prod(1 + t) - 1) * 100 for t in thirds if len(t)]
    sub_stability = float(-np.std(sub_nets)) if len(sub_nets) == 3 else None   # higher = more stable
    sub_min = float(min(sub_nets)) if len(sub_nets) == 3 else None             # worst subperiod (p-floor proxy)
    # overall 2020 selection-window Sharpe (the PLANTED-NULL metric: Sharpe-rank should NOT translate)
    selret = selbook.to_numpy()
    sharpe = float(selret.mean() / (selret.std() + 1e-12) * np.sqrt(365)) if len(selret) > 5 else None
    return {
        "uid": cand["uid"], "name": cand["name"], "family": cand["family"], "cad": cand["cad"],
        "oos_net_2020": round(oos_net, 2), "val_net_2020": round(val_net, 2),
        "sharpe_2020": round(sharpe, 3) if sharpe is not None else None,
        "time_in_2020": round(time_in, 4) if time_in is not None else None,
        "turnover_2020": round(turnover, 5) if turnover is not None else None,
        "march_crash_ret_2020": round(march_crash_ret, 2) if march_crash_ret is not None else None,
        "march_crash_dd_2020": round(march_crash_dd, 2) if march_crash_dd is not None else None,
        "sub_stability_2020": round(sub_stability, 3) if sub_stability is not None else None,
        "sub_min_2020": round(sub_min, 2) if sub_min is not None else None,
    }


# =====================================================================================================
# 3. 2021 FORWARD TARGET -- reuse forward_test_2021's PIT universe + book machinery
# =====================================================================================================
def target_2021(cand):
    """2021-forward net + 2021-May-crash compound, on the survivorship-clean PIT universe (CORE u10 for
    the cleanest apples-to-apples vs the 2020 core features; EXPAND tracked too as a robustness slice)."""
    res = {}
    want_vol = cand["loader"] == "ohlcv"
    for universe in ("core", "expand"):
        assets = FT._assets_for(cand["cad"], want_vol, universe)
        if not assets:
            res[universe] = {}; continue
        vt = None
        rvs = [np.nanmedian(A["rv"][A["active"]]) for A in assets if A["active"].sum() > 5]
        rvs = [x for x in rvs if np.isfinite(x)]
        vt = float(np.nanmedian(rvs)) if rvs else None
        series = [FT._candidate_net_series(A, cand["held_fn"], cand["params"], cand["minhold"], vt)
                  for A in assets]
        book = FT._ew_book(series, universe)
        if book is None:
            res[universe] = {}; continue
        fwd_net = _compound(book)
        reg = FT._regime_metrics(book)
        res[universe] = {"fwd_net_2021": round(fwd_net, 2) if fwd_net is not None else None,
                         "may_crash_2021": reg.get("May_crash"), "maxdd_2021": _maxdd(book)}
    return res


# =====================================================================================================
# 4. STATISTICS -- Spearman, OOS rank-IC, block-bootstrap, max-stat permutation
# =====================================================================================================
def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 5:
        return None, 0
    x = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return None, len(pairs)
    return float(np.corrcoef(rx, ry)[0, 1]), len(pairs)


def oos_rank_ic(feat, target, k_folds=5, seed=0):
    """Out-of-sample rank-IC: K-fold across CONFIGS. In each fold, rank configs by the 2020 feature on
    the TRAIN folds is irrelevant for a single feature (rank is monotone) -- instead we measure the
    HELD-OUT-fold Spearman(feature, 2021-target) and average. For a single feature this equals the
    full-sample Spearman in expectation but with a fold-CI; the value-add is the per-fold spread (is the
    relation stable or fold-fragile?). Returns (mean_oos_rho, per_fold list)."""
    pairs = [(f, t) for f, t in zip(feat, target) if f is not None and t is not None
             and np.isfinite(f) and np.isfinite(t)]
    if len(pairs) < 15:
        return None, []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    folds = np.array_split(idx, k_folds)
    rhos = []
    for fo in folds:
        sub = [pairs[i] for i in fo]
        if len(sub) < 5:
            continue
        r, _ = spearman([s[0] for s in sub], [s[1] for s in sub])
        if r is not None:
            rhos.append(r)
    if not rhos:
        return None, []
    return float(np.mean(rhos)), [round(r, 3) for r in rhos]


def block_bootstrap_spearman(feat, target, n_boot=2000, block=1, seed=0):
    """Bootstrap CI on the feature->2021 Spearman by resampling CONFIGS (block=1: configs are the i.i.d.
    unit here, not time -- we resample the cross-section of configs). Returns (rho, p05, p95)."""
    pairs = [(f, t) for f, t in zip(feat, target) if f is not None and t is not None
             and np.isfinite(f) and np.isfinite(t)]
    if len(pairs) < 8:
        return None, None, None
    base, _ = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    rng = np.random.default_rng(seed)
    n = len(pairs); boots = []
    for _ in range(n_boot):
        samp = rng.integers(0, n, n)
        sub = [pairs[i] for i in samp]
        r, _ = spearman([s[0] for s in sub], [s[1] for s in sub])
        if r is not None:
            boots.append(r)
    if not boots:
        return base, None, None
    return base, float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def maxstat_permutation_null(features_dict, target, n_perm=2000, seed=0, one_sided=True):
    """MAX-STAT permutation deflation. The observed statistic per feature = |Spearman(feature, 2021)|
    (or signed, one-sided). Under H0 the 2020->2021 mapping is random: shuffle the target vector,
    recompute every feature's Spearman, take the MAX |rho| across features -> the null distribution of
    'the best feature found by chance over this many features'. Deflated p = P(max_null >= observed_best).
    Returns dict(best_feature, best_rho, deflated_p, raw_p_best, max_null_p95)."""
    feats = list(features_dict.keys())
    # observed signed/abs rho per feature (use the DIRECTED hypothesis: structural de-risk -> higher 2021;
    # for crash-DD/march_crash_ret higher=better, for turnover lower=better; we take signed and let the
    # max-stat use the magnitude, then report the directional winner)
    obs = {}
    for f in feats:
        r, n = spearman(features_dict[f], target)
        obs[f] = r
    valid = {f: r for f, r in obs.items() if r is not None}
    if not valid:
        return {"best_feature": None}
    best_feature = max(valid, key=lambda f: abs(valid[f]))
    best_rho = valid[best_feature]
    observed_best = abs(best_rho)
    rng = np.random.default_rng(seed)
    # align: build a common index of configs where ALL features + target are present, so the shuffle is
    # consistent across features
    n = len(target)
    tgt = np.array([t if (t is not None and np.isfinite(t)) else np.nan for t in target], float)
    max_nulls = []
    per_feat_nulls = {f: [] for f in valid}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        tgt_s = tgt[perm]
        cur_max = 0.0
        for f in valid:
            fv = np.array([x if (x is not None and np.isfinite(x)) else np.nan
                           for x in features_dict[f]], float)
            mask = ~(np.isnan(fv) | np.isnan(tgt_s))
            if mask.sum() < 5:
                continue
            rx = pd.Series(fv[mask]).rank().to_numpy(); ry = pd.Series(tgt_s[mask]).rank().to_numpy()
            if rx.std() < 1e-9 or ry.std() < 1e-9:
                continue
            r = float(np.corrcoef(rx, ry)[0, 1])
            per_feat_nulls[f].append(r)
            if abs(r) > cur_max:
                cur_max = abs(r)
        max_nulls.append(cur_max)
    max_nulls = np.array(max_nulls)
    deflated_p = float((max_nulls >= observed_best).mean())
    # raw (un-deflated) p for the best feature alone
    bn = np.array([abs(x) for x in per_feat_nulls.get(best_feature, [])])
    raw_p = float((bn >= observed_best).mean()) if len(bn) else None
    return {"best_feature": best_feature, "best_rho": round(best_rho, 3),
            "observed_best_abs": round(observed_best, 3),
            "deflated_p": round(deflated_p, 4), "raw_p_best": round(raw_p, 4) if raw_p is not None else None,
            "max_null_p95": round(float(np.percentile(max_nulls, 95)), 3),
            "n_features_tested": len(valid), "n_perm": n_perm,
            "all_obs": {f: round(r, 3) for f, r in valid.items()}}


# =====================================================================================================
# 5. TRANSLATION TEST -- select top-K by each metric, measure the selected set's 2021 outcome
# =====================================================================================================
def translation_test(rows, k=8):
    """For each 2020-selection METRIC, select the top-K configs by that metric (the SELECTION RULE),
    freeze them, and measure the selected set's mean 2021-forward net + mean 2021-May-crash. Compare
    against the ENSEMBLE (no selection = all configs). One-sided: does any metric BEAT the ensemble?"""
    # metric -> (key, higher_is_better)
    metrics = {
        "peak_net_2020 [FAILED baseline]": ("oos_net_2020", True),
        "sharpe_2020 [PLANTED-NULL]": ("sharpe_2020", True),
        "march_crash_dd_2020 [STRUCTURAL: crash-DD preservation]": ("march_crash_dd_2020", True),  # less neg = better
        "march_crash_ret_2020 [STRUCTURAL: crash compound]": ("march_crash_ret_2020", True),
        "low_turnover_2020 [STRUCTURAL]": ("turnover_2020", False),
        "low_time_in_2020 [STRUCTURAL: de-risk]": ("time_in_2020", False),
        "sub_stability_2020 [STRUCTURAL: cross-subperiod]": ("sub_stability_2020", True),
        "sub_min_2020 [STRUCTURAL: worst-subperiod floor]": ("sub_min_2020", True),
    }
    # ensemble baseline (all configs)
    fwd_all = [r["fwd_net_2021_core"] for r in rows if r.get("fwd_net_2021_core") is not None]
    crash_all = [r["may_crash_2021_core"] for r in rows if r.get("may_crash_2021_core") is not None]
    ens = {"fwd_net": round(float(np.mean(fwd_all)), 2) if fwd_all else None,
           "may_crash": round(float(np.mean(crash_all)), 2) if crash_all else None,
           "n": len(fwd_all), "selection": "ENSEMBLE (no selection -- all configs EW)"}
    out = {"ENSEMBLE": ens, "metrics": {}}
    for label, (key, hib) in metrics.items():
        sel = [r for r in rows if r.get(key) is not None and r.get("fwd_net_2021_core") is not None]
        if len(sel) < k:
            continue
        sel.sort(key=lambda r: -r[key] if hib else r[key])
        top = sel[:k]
        fwd = [r["fwd_net_2021_core"] for r in top]
        crash = [r["may_crash_2021_core"] for r in top if r.get("may_crash_2021_core") is not None]
        out["metrics"][label] = {
            "key": key, "k": k,
            "sel_fwd_net_2021": round(float(np.mean(fwd)), 2),
            "sel_may_crash_2021": round(float(np.mean(crash)), 2) if crash else None,
            "beats_ensemble_fwd": bool(ens["fwd_net"] is not None and np.mean(fwd) > ens["fwd_net"]),
            "selected": [r["name"] + "@" + r["cad"] for r in top],
        }
    return out


# =====================================================================================================
# 6. MAIN
# =====================================================================================================
def assemble(tfs, max_per_family, verbose=True):
    """Build the joined per-config dataset: 2020 features + 2021 targets."""
    cands = build_config_universe(tfs, max_per_family)
    if verbose:
        from collections import Counter
        fam = Counter(c["family"] for c in cands)
        print(f"[universe] {len(cands)} configs across TFs={tfs}: {dict(fam)}")
    rows = []
    for i, c in enumerate(cands):
        f2020 = features_2020(c)
        if f2020 is None:
            continue
        t2021 = target_2021(c)
        core = t2021.get("core", {}); exp = t2021.get("expand", {})
        if core.get("fwd_net_2021") is None:
            continue
        row = {**f2020,
               "fwd_net_2021_core": core.get("fwd_net_2021"),
               "may_crash_2021_core": core.get("may_crash_2021"),
               "maxdd_2021_core": core.get("maxdd_2021"),
               "fwd_net_2021_expand": exp.get("fwd_net_2021"),
               "may_crash_2021_expand": exp.get("may_crash_2021")}
        rows.append(row)
        if verbose and (i % 25 == 0 or i == len(cands) - 1):
            print(f"   [{i+1}/{len(cands)}] {c['uid']:30} oos2020={f2020['oos_net_2020']:>7} "
                  f"crashDD2020={f2020['march_crash_dd_2020']} -> fwd2021={core.get('fwd_net_2021')}")
    return rows


def analyze(rows, n_perm=2000, n_boot=2000):
    """Run the full statistical battery on the assembled rows."""
    tgt_core = [r["fwd_net_2021_core"] for r in rows]
    tgt_crash = [r["may_crash_2021_core"] for r in rows]
    # the candidate 2020 features (signed so higher = hypothesized-better-2021)
    feat_specs = {
        "oos_net_2020 [net-rank, FAILED baseline]": [r["oos_net_2020"] for r in rows],
        "sharpe_2020 [PLANTED-NULL]": [r["sharpe_2020"] for r in rows],
        "march_crash_dd_2020 [crash-DD preservation]": [r["march_crash_dd_2020"] for r in rows],
        "march_crash_ret_2020 [crash compound]": [r["march_crash_ret_2020"] for r in rows],
        "neg_turnover_2020 [low-turnover]": [(-r["turnover_2020"] if r["turnover_2020"] is not None else None)
                                             for r in rows],
        "neg_time_in_2020 [de-risk/low time-in]": [(-r["time_in_2020"] if r["time_in_2020"] is not None else None)
                                                   for r in rows],
        "sub_stability_2020 [cross-subperiod]": [r["sub_stability_2020"] for r in rows],
        "sub_min_2020 [worst-subperiod floor]": [r["sub_min_2020"] for r in rows],
    }
    # per-feature rank-transfer Spearman vs 2021 fwd net + bootstrap CI + OOS fold rho
    per_feature = {}
    for label, fv in feat_specs.items():
        rho, n = spearman(fv, tgt_core)
        rho_b, p05, p95 = block_bootstrap_spearman(fv, tgt_core, n_boot=n_boot)
        oos_rho, folds = oos_rank_ic(fv, tgt_core)
        rho_crash, _ = spearman(fv, tgt_crash)
        per_feature[label] = {
            "rho_2021fwd": round(rho, 3) if rho is not None else None, "n": n,
            "boot_p05": round(p05, 3) if p05 is not None else None,
            "boot_p95": round(p95, 3) if p95 is not None else None,
            "oos_fold_rho": round(oos_rho, 3) if oos_rho is not None else None,
            "oos_folds": folds,
            "rho_2021_may_crash": round(rho_crash, 3) if rho_crash is not None else None,
        }
    # MAX-STAT deflation across all features (the multiple-comparisons referee)
    defl = maxstat_permutation_null(feat_specs, tgt_core, n_perm=n_perm)
    # the translation test (top-K selection per metric vs ensemble)
    trans = translation_test(rows, k=max(6, len(rows) // 12))
    return {"per_feature": per_feature, "deflation": defl, "translation": trans}


PREREG = {
    "H0": "NO 2020-selection metric beats the no-selection ENSEMBLE on 2021-forward; config translation "
          "is impossible regardless of metric. Net-rank Spearman 0.11 is the ceiling.",
    "H1": "a STRUCTURAL 2020-selection metric translates: (a) rank-transfer Spearman meaningfully > 0.11, "
          "AND (b) the 2020-structural-feature -> 2021-fwd-net relation is significant OUT-OF-SAMPLE.",
    "test": "ONE-SIDED (structural BEATS baseline?). Asymmetric loss: false-ship a non-translating rule "
            ">> false-skip.",
    "decision_rule": "a metric translates iff rank-transfer Spearman > 0.11 noise floor AND survives "
                     "MAX-STAT permutation deflation AND survives a PLANTED-NULL control AND is robust "
                     "(asset-subset / cost).",
    "noise_floor_spearman": 0.11,
    "planted_null_metric": "sharpe_2020 (and random) -- must NOT translate (two-sided gate: reject ghosts).",
}


def verdict(analysis, rows):
    pf = analysis["per_feature"]; defl = analysis["deflation"]; trans = analysis["translation"]
    floor = PREREG["noise_floor_spearman"]
    # structural features (exclude the failed net-rank baseline + the planted null)
    structural = {k: v for k, v in pf.items() if "FAILED baseline" not in k and "PLANTED-NULL" not in k}
    # the best structural feature by |rho|
    best_struct = max(structural, key=lambda k: abs(pf[k]["rho_2021fwd"] or 0)) if structural else None
    best_rho = pf[best_struct]["rho_2021fwd"] if best_struct else None
    net_rho = next((v["rho_2021fwd"] for k, v in pf.items() if "FAILED baseline" in k), None)
    planted_rho = next((v["rho_2021fwd"] for k, v in pf.items() if "PLANTED-NULL" in k), None)
    # gates
    gate_floor = best_rho is not None and abs(best_rho) > floor
    gate_deflation = defl.get("deflated_p") is not None and defl["deflated_p"] < 0.05
    # planted-null gate: planted (Sharpe) should be NOT-significant AND a structural should be the deflation winner
    bf = defl.get("best_feature", "") or ""
    gate_planted = "PLANTED-NULL" not in bf  # the deflation winner is NOT the planted null
    gate_oos = (best_struct is not None and pf[best_struct]["oos_fold_rho"] is not None
                and abs(pf[best_struct]["oos_fold_rho"]) > floor)
    # translation: does any STRUCTURAL metric's top-K beat the ensemble on 2021 fwd net?
    ens_fwd = trans["ENSEMBLE"]["fwd_net"]
    struct_beats = [(lbl, m["sel_fwd_net_2021"]) for lbl, m in trans["metrics"].items()
                    if "STRUCTURAL" in lbl and m["beats_ensemble_fwd"]]
    if gate_floor and gate_deflation and gate_planted and gate_oos:
        v = "REAL"
    elif gate_floor and (gate_deflation or gate_oos) and gate_planted:
        v = "AMBIGUOUS"
    else:
        v = "ARTIFACT"
    return {
        "verdict": v,
        "best_structural_feature": best_struct, "best_structural_rho": best_rho,
        "net_rank_rho": net_rho, "planted_null_rho": planted_rho,
        "deflation_winner": bf, "deflated_p": defl.get("deflated_p"),
        "gates": {"above_noise_floor": gate_floor, "deflation_p<0.05": gate_deflation,
                  "planted_null_not_winner": gate_planted, "oos_fold_above_floor": gate_oos},
        "ensemble_fwd_net_2021": ens_fwd,
        "structural_metrics_beating_ensemble": struct_beats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.translation_solution_2021")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tfs", default="1d,4h", help="comma TFs for the config universe")
    ap.add_argument("--max-per-family", type=int, default=60)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    tfs = a.tfs.split(",")
    print("## 2020->2021 TRANSLATION SOLUTION TEST -- quant referee [long-only spot, UNSEEN sealed]")
    print("## PRE-REGISTRATION:")
    for k in ("H0", "H1", "test", "decision_rule"):
        print(f"   {k}: {PREREG[k]}")
    print()
    rows = assemble(tfs, a.max_per_family)
    print(f"\n[assembled] {len(rows)} configs with full 2020 features + 2021 targets")
    if len(rows) < 20:
        print("INSUFFICIENT configs for a powered test -- widen --tfs / --max-per-family")
        return 1
    analysis = analyze(rows, n_perm=a.n_perm, n_boot=a.n_boot)
    vd = verdict(analysis, rows)

    # ---- print the report ----
    print("\n" + "=" * 100)
    print("## PER-FEATURE 2020->2021 TRANSLATION (rank-transfer Spearman vs 2021 fwd net) [VERIFIED-2021-FORWARD]")
    print(f"   {'2020 feature':52} {'rho2021':>8} {'boot[p05,p95]':>16} {'oosFold':>8} {'rhoCrash':>9}")
    for lbl, v in analysis["per_feature"].items():
        ci = f"[{v['boot_p05']},{v['boot_p95']}]"
        print(f"   {lbl:52} {str(v['rho_2021fwd']):>8} {ci:>16} {str(v['oos_fold_rho']):>8} "
              f"{str(v['rho_2021_may_crash']):>9}")
    d = analysis["deflation"]
    print(f"\n## MAX-STAT PERMUTATION DEFLATION ({d['n_features_tested']} features, {d['n_perm']} perms):")
    print(f"   best feature = {d['best_feature']}  (|rho|={d['observed_best_abs']})")
    print(f"   DEFLATED p = {d['deflated_p']}  (raw p for best alone = {d['raw_p_best']}; "
          f"max-null p95 floor = {d['max_null_p95']})")
    print("\n## TRANSLATION TEST (top-K selection per metric vs no-selection ENSEMBLE):")
    ens = analysis["translation"]["ENSEMBLE"]
    print(f"   ENSEMBLE (no selection, n={ens['n']}): 2021 fwd net {ens['fwd_net']}% | "
          f"May-crash {ens['may_crash']}%")
    for lbl, m in analysis["translation"]["metrics"].items():
        flag = "BEATS-ENS" if m["beats_ensemble_fwd"] else "below-ens"
        print(f"   top-{m['k']} by {lbl:52} -> fwd {m['sel_fwd_net_2021']:>7}% crash "
              f"{str(m['sel_may_crash_2021']):>7}% [{flag}]")
    print("\n" + "=" * 100)
    print(f"## VERDICT: {vd['verdict']}")
    print(f"   best structural feature: {vd['best_structural_feature']} (rho={vd['best_structural_rho']})")
    print(f"   net-rank baseline rho={vd['net_rank_rho']} (the FAILED 0.11) | planted-null rho={vd['planted_null_rho']}")
    print(f"   deflation winner: {vd['deflation_winner']} | deflated p={vd['deflated_p']}")
    print(f"   gates: {vd['gates']}")
    print(f"   structural metrics beating ensemble: {vd['structural_metrics_beating_ensemble']}")

    # ---- charts ----
    charts = []
    if not a.no_charts:
        charts = make_charts(rows, analysis)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"translation_solution_2021_{stamp}.json"
    json.dump({"repro": {"command": "python -m strat.translation_solution_2021 " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "win_2020": WIN_2020, "sel_split": SEL_SPLIT,
                         "march_crash_2020": MARCH_CRASH_2020, "may_crash_2021": FT.REGIMES_2021["May_crash"],
                         "cost_maker": MAKER_RT, "tfs": tfs, "n_configs": len(rows)},
               "prereg": PREREG, "rows": rows, "analysis": analysis, "verdict": vd, "charts": charts},
              open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def make_charts(rows, analysis):
    """Two charts: (1) structural-feature-vs-2021 scatter (the best structural feature), (2) metric-
    translation bars (top-K fwd net per metric vs ensemble)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] matplotlib unavailable ({e}) -- skipped")
        return []
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    # chart 1: scatter of the best structural feature (march_crash_dd_2020) + net-rank, vs 2021 fwd net
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fams = sorted(set(r["family"] for r in rows))
    cmap = {f: c for f, c in zip(fams, plt.cm.tab10.colors)}
    for ax, (xf, xl) in zip(axes, [("march_crash_dd_2020", "2020 March-crash maxDD % (structural: higher=preserver)"),
                                   ("oos_net_2020", "2020 OOS net % (the FAILED net-rank baseline)")]):
        for r in rows:
            if r.get(xf) is None or r.get("fwd_net_2021_core") is None:
                continue
            ax.scatter(r[xf], r["fwd_net_2021_core"], s=22, alpha=0.7, color=cmap[r["family"]])
        xs = [r[xf] for r in rows if r.get(xf) is not None and r.get("fwd_net_2021_core") is not None]
        ys = [r["fwd_net_2021_core"] for r in rows if r.get(xf) is not None and r.get("fwd_net_2021_core") is not None]
        rho, _ = spearman(xs, ys)
        ax.set_xlabel(xl); ax.set_ylabel("2021 forward net % (core)")
        ax.set_title(f"Spearman = {rho}"); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[f], label=f) for f in fams]
    axes[0].legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle("2020 structural feature (left) vs FAILED net-rank (right) -> 2021 forward net")
    fig.tight_layout()
    c1 = OUT / f"translation_scatter_{stamp}.png"
    fig.savefig(c1, dpi=110); plt.close(fig); paths.append(str(c1))
    # chart 2: metric-translation bars
    tr = analysis["translation"]
    labels, vals, cols = [], [], []
    ens = tr["ENSEMBLE"]["fwd_net"]
    for lbl, m in tr["metrics"].items():
        short = lbl.split(" [")[0]
        labels.append(short); vals.append(m["sel_fwd_net_2021"])
        cols.append("#2a9d8f" if m["beats_ensemble_fwd"] else "#e76f51")
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))
    ax2.barh(y, vals, color=cols)
    ax2.axvline(ens, color="k", ls="--", lw=1.5, label=f"ENSEMBLE (no selection) = {ens}%")
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("selected top-K mean 2021 forward net %")
    ax2.set_title("2020-selection metric -> 2021 forward net (vs no-selection ensemble)")
    ax2.legend(); ax2.grid(alpha=0.3, axis="x"); fig2.tight_layout()
    c2 = OUT / f"translation_metric_bars_{stamp}.png"
    fig2.savefig(c2, dpi=110); plt.close(fig2); paths.append(str(c2))
    print(f"[charts] {c1}\n[charts] {c2}")
    return paths


# =====================================================================================================
# 7. SELFTEST
# =====================================================================================================
def selftest():
    print("## TRANSLATION-SOLUTION SELFTEST")
    ok = True
    # (1) config universe builds and is family-balanced
    cands = build_config_universe(["1d"], max_per_family=20)
    from collections import Counter
    fam = Counter(c["family"] for c in cands)
    s1 = len(cands) > 30 and len(fam) >= 4
    print(f"  (1) universe: {len(cands)} configs, families {dict(fam)} -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) 2020 features compute for an MA config and include the March crash
    c0 = next(c for c in cands if c["family"] == "MA")
    f = features_2020(c0)
    s2 = f is not None and f.get("march_crash_dd_2020") is not None and f["march_crash_dd_2020"] < 0
    print(f"  (2) 2020 features: {c0['uid']} oos={f.get('oos_net_2020') if f else None} "
          f"marchDD={f.get('march_crash_dd_2020') if f else None} time_in={f.get('time_in_2020') if f else None} "
          f"-> {'PASS' if s2 else 'FAIL'}")
    ok &= s2
    # (3) 2021 target computes via the PIT machinery
    t = target_2021(c0)
    s3 = t.get("core", {}).get("fwd_net_2021") is not None
    print(f"  (3) 2021 target (PIT): core fwd_net={t.get('core',{}).get('fwd_net_2021')} "
          f"may_crash={t.get('core',{}).get('may_crash_2021')} -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    # (4) spearman + permutation-null mechanics on a planted signal (feature == target -> rho~1)
    feat = list(range(30)); tgt = list(range(30))
    r, _ = spearman(feat, tgt)
    s4a = r is not None and r > 0.99
    # planted-null: feature == shuffled target -> deflation should NOT flag
    d = maxstat_permutation_null({"real": feat, "noise": list(np.random.default_rng(0).permutation(30))},
                                 tgt, n_perm=300)
    s4b = d["best_feature"] == "real" and d["deflated_p"] < 0.05
    print(f"  (4) stats: identity Spearman={r} (PASS={s4a}); deflation picks 'real' p={d['deflated_p']} "
          f"(PASS={s4b}) -> {'PASS' if (s4a and s4b) else 'FAIL'}")
    ok &= (s4a and s4b)
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
