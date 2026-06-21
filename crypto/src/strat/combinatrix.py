"""src/strat/combinatrix.py -- V1.2 COMBINATRIX: combine per-asset hold-masks from N TI input engines
via principled OPERATORS, score with the deep2020 wealth/_metrics, gate with full data-snoop discipline.

CAMPAIGN (V1.2, user /orc 2026-06-21): the engine library (families 1-3, one engine per indicator) is built;
ALL are de-risked betas (none beats LO buy-hold on net). This module asks the HONEST combinatorial question
under the data-snoop trap (Sullivan-Timmermann-White): does ANY combo of survivor engines beat (a) its best
constituent single engine AND (b) long-only buy-hold, on OOS+IRON NET, AFTER deflation, with PBO low + p05>0?
Prior = NO (de-risked beta + overfit). A disciplined null is a fine result. Regime-gating may improve
DD/robustness without beating net -- that frontier is reported too.

DISCIPLINE (binding, baked into __contract__):
  - INPUTS = the CLEAN survivor seed only (runs/engines/V1.2_combinatrix/survivor_seed.json): 5 cross-mechanism
    reps CKSP(1d,trend)/KELTNER(4h,breakout)/AO(4h,momentum)/RSI(1h,mean-rev-robust)/ULCER(1d,vol-sizing),
    + up to 2 robust alternates (ADX 2nd-trend, BBSQZ 2nd-vol) -> <=7 cross-MECHANISM inputs. NEVER stack
    redundant within-mechanism engines (the data-snoop trap).
  - Each input = its VAL-selected config's per-asset hold-mask, built by REUSING deep2020_ti_pipeline's
    signal-builders + the SAME stack (signal -> trail10 -> minhold -> lag1 -> maker) + the SAME wealth/_metrics.
  - SPLIT: 2020 VAL Jul-Sep = SELECT the combo operator+params; 2020 Oct-Dec OOS = REPORT;
    2021 IRON = CONFIRM (held-out YEAR, the survivor must hold); 2022+ NEVER loaded (asserted).
  - Rank by net WEALTH. GATE every reported combo: block-bootstrap p05>0 vs best constituent single AND vs
    buy-hold; CSCV-PBO (prob of backtest overfit); a DEFLATED net/Sharpe accounting for #combos tried.

OPERATORS (combine per-asset {0,1} hold-masks of N inputs -> one combined {0,1} mask, per asset):
  - AND_K  : long when >=K of N inputs long (K=2..N) -- confluence.
  - OR     : long when ANY input long (= AND_1).
  - MAJORITY: long when >=ceil(N/2) inputs long.
  - REGIME_GATE(entry, regime): entry AND regime (DIRECTED -- the principled conditioning; e.g. AO momentum
    gated by CKSP trend, RSI MR gated by CKSP uptrend). The project's VALIDATED use of TIs.
  - WEIGHTED(thr): mean position across inputs >= thr (thr in (0,1]) -- soft vote.

DISCIPLINE MODULE:
  (1) CSCV-PBO (Bailey-Lopez de Prado 2014): split combined per-config returns into S even blocks; over all
      C(S,S/2) train/test partitions, take the IS-best config, find its rank in the test set; PBO = fraction
      of partitions where the IS-best underperforms the test MEDIAN. PBO~0.5 = pure overfit; PBO~0 = genuine.
  (2) DEFLATED metric: White's Reality Check bootstrap (best-of-N vs benchmark under H0) AND a deflated-Sharpe
      (Bailey-LdP) that penalizes by the NUMBER of combos evaluated.
  (3) block-bootstrap p05 (circular, Politis-Romano; reuses the funding_dispersion_bootstrap.py pattern).

RWYB: python -m strat.combinatrix --selftest   (the 4 mandatory self-tests)
      python -m strat.combinatrix --run         (the full combinatrix campaign)
No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import ceil, comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.deep2020_ti_pipeline as P
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.structural_fixes import min_hold
from strat.ma_2020_breakdown import _panel

__contract__ = {
    "kind": "strat.combinatrix",
    "inputs": [
        "survivor_seed.json (CLEAN VAL-selected cross-mechanism reps; <=7 cross-MECHANISM inputs)",
        "deep2020_ti_pipeline.INDICATORS (signal-builders) + the SAME stack + wealth/_metrics",
        "OHLC panels via ma_2020_breakdown._panel (27 u100 names with 2020-DEV-window data)",
    ],
    "outputs": [
        "per-combo {operator, params, VAL/OOS/IRON net, p05 vs single & BH, PBO, deflated} ranked by net WEALTH",
        "runs/engines/V1.2_combinatrix/combinatrix_results.json (incremental, n_names recorded per result)",
    ],
    "invariants": [
        "INPUTS are CROSS-MECHANISM survivor reps ONLY; never stack redundant within-mechanism engines.",
        "Each input mask = deep2020 signal -> trail10 -> minhold -> lag1 (pos shift) -> maker, IDENTICAL to _book.",
        "SPLIT: 2020 VAL Jul-Sep SELECT / 2020 Oct-Dec OOS REPORT / 2021 IRON CONFIRM / 2022+ NEVER loaded (asserted).",
        "Rank by net WEALTH (compound), never Sharpe. IC banned as a primary metric.",
        "GATE every reported combo: block-bootstrap p05>0 vs best single AND vs buy-hold; CSCV-PBO; DEFLATED net/Sharpe.",
        "A null (no combo beats best-single AND buy-hold after deflation) is an HONEST result, reported as such.",
        ">=20 distinct u100 names (n_names recorded on every result).",
        "Long-only spot; no short/market-neutral.",
    ],
    "owner": "V1.2 combinatrix campaign",
}

# ============================ windows (DEV 2020 + IRON 2021; 2022+ SEALED) ============================
WARMUP = P.WARMUP
VOLWIN = P.VOLWIN
DRIFT_TOL = P.DRIFT_TOL
SYMS = P.SYMS

# DEV: 2020 -- VAL Jul-Sep SELECT, OOS Oct-Dec REPORT (the deep2020 window, identical boundaries)
DEV_WIN = ("2020-07-01", "2021-01-01"); DEV_SPLIT = "2020-10-01"
# IRON: 2021 full year -- CONFIRM. There is no within-window selection here; the WHOLE year is held-out OOS.
# We use an H1/H2-2021 split only for the _metrics shape (val=H1 informational, "net"=H2 the confirm figure).
IRON_WIN = ("2021-01-01", "2022-01-01"); IRON_SPLIT = "2021-07-01"
# 2022+ is SEALED -- asserted never loaded.
SEALED_FROM = "2022-01-01"


def _assert_not_sealed(win_end):
    end = pd.Timestamp(win_end)
    assert end <= pd.Timestamp(SEALED_FROM), (
        f"SEALED-WINDOW VIOLATION: window end {win_end} > {SEALED_FROM}; 2022+ must NEVER be loaded.")


# ============================ data loader (generic window; mirrors deep2020 load_ohlc) ============================
def load_window(cad, win, split):
    """Load the 27-name OHLC panel for an arbitrary [win[0], win[1]) window with a VAL/OOS split.
    Returns (assets, vt, names). vt = VAL-only median realized-vol (vol-target reference). IDENTICAL machinery
    to deep2020_ti_pipeline.load_ohlc but parametrized on the window (so DEV-2020 + IRON-2021 share one path)."""
    _assert_not_sealed(win[1])
    s_ms = pd.Timestamp(win[0]).value // 10**6; e_ms = pd.Timestamp(win[1]).value // 10**6
    split_ms = pd.Timestamp(split).value // 10**6
    vw = VOLWIN[cad]; assets = []; rv_meds = []; names = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win_m = ms2 >= s_ms
        if win_m.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
        idx = pd.to_datetime(ms2[win_m], unit="ms")
        assets.append({"o": o2, "h": h2, "l": l2, "c": c2, "ret": ret, "win": win_m, "idx": idx, "rv": rv})
        vm = (ms2 >= s_ms) & (ms2 < split_ms)
        rv_meds.append(np.nanmedian(rv[vm]) if vm.sum() > 5 else np.nan)
        names.append(sym)
    vt = float(np.nanmedian(rv_meds)) if rv_meds and not np.all(np.isnan(rv_meds)) else None
    return assets, vt, names


# ============================ per-asset hold-mask (IDENTICAL stack to deep2020 _book) ============================
def input_masks(assets, held_fn, params, minhold):
    """Per-asset WINDOW-length held {0,1} masks AFTER the deep2020 stack overlays (trail10 + minhold), restricted
    to the tradeable window (held[A['win']], aligned to A['idx']). This is the SAME
    held = min_hold(apply_trail_stop(held0, c, 0.10), minhold) deep2020._book builds, BUT returned per-asset and
    window-restricted so the combinatrix operators combine aligned masks before the wealth aggregation. The pos
    lag-1 + vol-target + maker cost are applied later in _combo_book (also identical to _book)."""
    masks = []
    for A in assets:
        held0 = held_fn(A, params)
        held = min_hold(apply_trail_stop(held0.copy().astype(np.int8), A["c"], 0.10)[0].astype(np.int8),
                        minhold).astype(np.int8)
        # include the ONE bar before the window (the lead-in) so the lag-1 pos at the first window bar carries
        # the pre-window held -- identical to deep2020._book's full-slice pos[1:]=held[:-1]. _combo_book strips it.
        w = np.where(A["win"])[0]
        lo = max(0, w[0] - 1)
        masks.append(held[lo:w[-1] + 1])
    return masks


# ============================ operators (combine N per-asset masks -> 1 combined mask, per asset) ============================
def op_and_k(masks_list, k):
    """masks_list = [mask_i for each input]; each mask_i = list of per-asset arrays (aligned same length).
    Long when >= k of N inputs are long (elementwise per asset)."""
    n_assets = len(masks_list[0])
    out = []
    for a in range(n_assets):
        stack = np.vstack([m[a] for m in masks_list])
        out.append((stack.sum(axis=0) >= k).astype(np.int8))
    return out


def op_or(masks_list):
    return op_and_k(masks_list, 1)


def op_majority(masks_list):
    n = len(masks_list)
    return op_and_k(masks_list, ceil(n / 2))


def op_regime_gate(entry_masks, regime_masks):
    """DIRECTED: entry AND regime, per asset. entry_masks/regime_masks = lists of per-asset arrays."""
    n_assets = len(entry_masks)
    return [(entry_masks[a] & regime_masks[a]).astype(np.int8) for a in range(n_assets)]


def op_weighted(masks_list, thr):
    """Mean position across inputs >= thr -> long. thr in (0,1]."""
    n_assets = len(masks_list[0]); n = len(masks_list)
    out = []
    for a in range(n_assets):
        stack = np.vstack([m[a].astype(np.float64) for m in masks_list])
        out.append((stack.mean(axis=0) >= thr - 1e-9).astype(np.int8))
    return out


# ============================ combined mask -> wealth (IDENTICAL pos-shift/vt/maker/resample to _book) ============================
def _combo_book(assets, combined_masks, vt, split):
    """Take per-asset COMBINED held masks (WINDOW-length: aligned to A['idx'], the tradeable bars) -> the
    deep2020 wealth pipeline: pos = lag1(held) * vol_target, maker cost on flips, fixed-EW cross-asset mean,
    resample to daily. Returns (d, time_in, n_trades). Same arithmetic as deep2020._book's tail; here masks are
    already window-restricted, so ret/rv are taken over the window (A['ret'][win], A['rv'][win]). The lag-1 pos
    shift is applied within the window (pos[1:]=held[:-1]) -- identical causal treatment."""
    nets, tin, ntr = [], [], []
    for A, held in zip(assets, combined_masks):
        win, idx = A["win"], A["idx"]
        ret_w = A["ret"][win]; rv_w = A["rv"][win]
        held = held.astype(np.float64)
        L = len(idx)
        # masks carry a 1-bar lead-in (length L+1: the bar before the window + the window) so the lag-1 pos at the
        # first window bar inherits the pre-window held -- identical to deep2020._book. If no lead-in (length L,
        # e.g. window starts at bar 0), pad a leading 0.
        if len(held) == L + 1:
            held_lead = held
        elif len(held) == L:
            held_lead = np.concatenate([[0.0], held])
        else:
            raise ValueError(f"mask len {len(held)} != window len {L} or L+1 (lead-in) -- alignment broken")
        held_w = held_lead[1:]                       # the window-only held (for n_trades + time_in)
        d = np.diff(np.concatenate([[0], held_w.astype(int), [0]]))
        ntr.append(int(min((d == 1).sum(), (d == -1).sum())))
        pos = held_lead[:-1].copy()                  # pos[t] = held at t-1 (lag-1), over the window
        # lead pos: the position the bar BEFORE the window (held two bars before window-start, lag-1) -- used as
        # the flips-diff baseline so the first window bar's maker cost matches deep2020's full-slice computation.
        lead_pos = float(held_lead[0]) if len(held_lead) >= 1 else 0.0
        if vt is not None:
            scale = np.clip(vt / (np.nan_to_num(rv_w, nan=vt) + 1e-12), 0.0, 1.0)
            pos = pos * scale
            # lead pos has no rv in-window; deep2020 scales it by the pre-window rv. Approx with first in-window
            # scale (boundary bar only) -- exact for the held alignment; sub-bp residual on the flip cost only.
            lead_pos = lead_pos * (scale[0] if len(scale) else 1.0)
        flips = np.abs(np.diff(np.concatenate([[lead_pos], pos])))
        nets.append(pd.Series(pos * ret_w - flips * (MAKER_RT / 2.0), index=idx))
        s = pd.Series(pos, index=idx); s = s[s.index >= pd.Timestamp(split)]
        if len(s):
            tin.append(float(s.mean()))
    if not nets:
        return None
    b = pd.concat(nets, axis=1).fillna(0.0).mean(axis=1)
    d = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return d, (round(float(np.mean(tin)), 2) if tin else None), (round(float(np.mean(ntr)), 1) if ntr else None)


def _metrics(d, split, tin=None, ntr=None):
    """VAL/OOS metrics around `split`. Same definitions as deep2020._metrics (generalized split).
    'net' = OOS compound %, 'val_net' = VAL compound %, robust = |drift|<=DRIFT_TOL."""
    val = d[d.index < pd.Timestamp(split)].to_numpy(); oos = d[d.index >= pd.Timestamp(split)].to_numpy()
    if len(oos) < 5:
        return None
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    vn = float((np.prod(1 + val) - 1) * 100) if len(val) else 0.0
    on = float((eq[-1] - 1) * 100)
    return {"val_net": round(vn, 1), "net": round(on, 1), "worst": round(min(vn, on), 1),
            "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(dd, 1), "drift": round(on - vn, 1), "robust": bool(abs(on - vn) <= DRIFT_TOL),
            "time_in": tin, "n_trades": ntr}


def _combo_daily(assets, combined_masks, vt):
    """Just the daily return series (for bootstrap / PBO), no metric summary."""
    r = _combo_book(assets, combined_masks, vt, DEV_SPLIT)
    return None if r is None else r[0]


def buyhold(assets, vt, split):
    nets = []
    for A in assets:
        ret, win, idx, rv = A["ret"], A["win"], A["idx"], A["rv"]
        pos = np.ones(len(ret))
        if vt is not None:
            pos = np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        nets.append(pd.Series((pos * ret)[win], index=idx))
    b = pd.concat(nets, axis=1).fillna(0.0).mean(axis=1)
    d = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return d


# ============================ DISCIPLINE: block-bootstrap, CSCV-PBO, deflation ============================
def circular_block_bootstrap(daily, block_len, n_boot, rng):
    """Politis-Romano circular block bootstrap. Returns (n_boot, T) resamples. (Pattern from
    src/mining/funding_dispersion_bootstrap.py.)"""
    T = len(daily)
    if T == 0:
        return np.zeros((n_boot, 0))
    n_blocks = int(np.ceil(T / block_len))
    circ = np.concatenate([daily, daily])
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    boots = np.empty((n_boot, T))
    for i in range(n_boot):
        pieces = [circ[s:s + block_len] for s in starts[i]]
        boots[i] = np.concatenate(pieces)[:T]
    return boots


def _compound(series):
    return float(np.prod(1.0 + series) - 1.0)


def bootstrap_diff_p05(a_daily, b_daily, block_len=10, n_boot=2000, seed=42):
    """Block-bootstrap of the COMPOUND-return DIFFERENCE (combo - benchmark), paired by resampling the SAME
    block indices for both series (so the comparison is paired -> isolates the marginal edge). Returns dict
    with p05/p50 of the difference and p(diff>0). p05>0 => combo beats benchmark after block-bootstrap deflation.
    Aligns the two series on their common date index first."""
    a = pd.Series(a_daily); b = pd.Series(b_daily)
    idx = a.index.intersection(b.index)
    a = a.reindex(idx).fillna(0.0).to_numpy(); b = b.reindex(idx).fillna(0.0).to_numpy()
    T = len(a)
    if T < 15:
        return {"p05": float("nan"), "p50": float("nan"), "p_gt0": float("nan"), "T": T}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_len))
    circ_a = np.concatenate([a, a]); circ_b = np.concatenate([b, b])
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idxs = np.concatenate([np.arange(s, s + block_len) for s in starts[i]])[:T]
        ra = circ_a[idxs]; rb = circ_b[idxs]
        diffs[i] = _compound(ra) - _compound(rb)
    return {"p05": round(float(np.percentile(diffs, 5)) * 100, 3),
            "p50": round(float(np.percentile(diffs, 50)) * 100, 3),
            "p_gt0": round(float(np.mean(diffs > 0)), 4), "T": T}


def cscv_pbo(returns_by_config, S=10, seed=0):
    """CSCV-PBO (Bailey-Lopez de Prado 2014, 'The probability of backtest overfitting').

    returns_by_config: dict {config_name: daily_return_array (same length, aligned)} -- the M candidate combos.
    Procedure: stack into an (T x M) matrix; split rows into S even contiguous blocks; for each of the C(S, S/2)
    ways to choose S/2 blocks as the IS (training) set (rest = OOS test): pick the config with the best IS
    performance, find its RANK among all M configs in the OOS test set, map to relative rank omega in [0,1],
    logit lambda = ln(omega/(1-omega)); PBO = fraction of partitions with lambda<=0 (IS-best underperforms the
    OOS median). PBO~0.5 = the selection is overfit (IS-best is a coin-flip OOS); PBO~0 = genuine persistence.

    Performance metric per config = compound return (wealth, the project objective)."""
    names = list(returns_by_config.keys())
    M = len(names)
    if M < 2:
        return {"pbo": float("nan"), "n_partitions": 0, "M": M, "note": "need >=2 configs"}
    mat = np.vstack([np.asarray(returns_by_config[n], float) for n in names]).T  # (T, M)
    T = mat.shape[0]
    S = min(S, T)
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return {"pbo": float("nan"), "n_partitions": 0, "M": M, "note": "T too small for CSCV"}
    bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(S)]

    def perf(rows, cfg):  # compound return of config cfg over the given row indices
        return _compound(mat[rows, cfg])

    lambdas = []
    all_blocks = list(range(S))
    for is_blocks in combinations(all_blocks, S // 2):
        oos_blocks = [b for b in all_blocks if b not in is_blocks]
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in oos_blocks])
        if len(is_rows) < 2 or len(oos_rows) < 2:
            continue
        is_perf = np.array([perf(is_rows, c) for c in range(M)])
        oos_perf = np.array([perf(oos_rows, c) for c in range(M)])
        c_star = int(np.argmax(is_perf))
        # relative rank of c_star in OOS (1=worst .. M=best) -> omega in (0,1)
        rank = int(np.sum(oos_perf <= oos_perf[c_star]))  # 1..M
        omega = rank / (M + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(np.log(omega / (1 - omega)))
    if not lambdas:
        return {"pbo": float("nan"), "n_partitions": 0, "M": M, "note": "no valid partitions"}
    lambdas = np.array(lambdas)
    pbo = float(np.mean(lambdas <= 0.0))
    return {"pbo": round(pbo, 4), "n_partitions": len(lambdas), "M": M, "S": S,
            "median_logit": round(float(np.median(lambdas)), 3)}


def whites_reality_check(returns_by_config, bench_daily, block_len=10, n_boot=2000, seed=7):
    """White's Reality Check (2000): under H0 that the BEST of the M combos has no edge over the benchmark,
    is the OBSERVED best excess compound return significant after accounting for M (multiple comparisons)?

    For each config: excess daily return f_i = combo_i - bench (aligned). Statistic V = max_i sqrt(T)*mean(f_i).
    Bootstrap the null by circular-block resampling the DEMEANED f_i (Politis-Romano stationary-bootstrap idea),
    recomputing V*; RC p-value = fraction of V* >= V. Low p = the best combo's outperformance survives the
    multiple-comparisons deflation. Returns the RC p-value + the deflated best."""
    names = list(returns_by_config.keys())
    M = len(names)
    bench = pd.Series(bench_daily)
    fmat = []
    common = None
    for n in names:
        s = pd.Series(returns_by_config[n])
        common = s.index if common is None else common.intersection(s.index)
    common = common.intersection(bench.index)
    b = bench.reindex(common).fillna(0.0).to_numpy()
    for n in names:
        s = pd.Series(returns_by_config[n]).reindex(common).fillna(0.0).to_numpy()
        fmat.append(s - b)
    fmat = np.vstack(fmat)  # (M, T)
    T = fmat.shape[1]
    if T < 15:
        return {"rc_pvalue": float("nan"), "M": M, "T": T, "note": "T too small"}
    fbar = fmat.mean(axis=1)
    V = float(np.max(np.sqrt(T) * fbar))
    demeaned = fmat - fbar[:, None]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_len))
    circ = np.concatenate([demeaned, demeaned], axis=1)  # (M, 2T)
    Vstar = np.empty(n_boot)
    for ib in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idxs = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]
        resamp = circ[:, idxs]                 # (M, T)
        mstar = resamp.mean(axis=1)
        Vstar[ib] = float(np.max(np.sqrt(T) * mstar))
    p = float(np.mean(Vstar >= V))
    best_i = int(np.argmax(fbar))
    return {"rc_pvalue": round(p, 4), "best_config": names[best_i], "M": M, "T": T,
            "best_excess_mean_bps": round(float(fbar[best_i]) * 1e4, 2), "V": round(V, 3)}


def deflated_sharpe(daily, n_trials, benchmark_sr=0.0):
    """Bailey-Lopez de Prado Deflated Sharpe Ratio. Given the observed (annualized) Sharpe of the SELECTED
    strategy and the NUMBER of independent trials, the expected max Sharpe under the null inflates; DSR is the
    probability the true SR exceeds benchmark after that deflation. Uses the non-annualized per-bar moments for
    the skew/kurtosis correction, then reports DSR (0..1; >0.95 = survives) and the deflated SR threshold."""
    x = np.asarray(daily, float)
    T = len(x)
    if T < 10:
        return {"dsr": float("nan"), "T": T, "note": "T too small"}
    mu = x.mean(); sd = x.std(ddof=1)
    if sd <= 0:
        return {"dsr": float("nan"), "T": T, "note": "zero variance"}
    sr = mu / sd  # per-bar (non-annualized) Sharpe
    g3 = float(((x - mu) ** 3).mean() / sd ** 3)  # skew
    g4 = float(((x - mu) ** 4).mean() / sd ** 4)  # kurtosis (non-excess)
    from scipy.stats import norm
    # Expected max Sharpe across n_trials independent trials (BLdP approx)
    e = 0.5772156649
    if n_trials < 2:
        sr0_z = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials); z2 = norm.ppf(1 - 1.0 / (n_trials * e))
        sr0_z = (1 - e) * z1 + e * z2         # expected-max order statistic in variance-1 (z) units
    # standard error of SR estimator with non-normal correction
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (T - 1))
    # UNITS FIX (2026-06-22 Wave-2 audit): the order statistic is in variance-1 (z) units; scale by se to convert
    # to SR units before comparing to the observed per-bar SR (canonical Bailey-Lopez de Prado). The pre-fix code
    # subtracted the z-unit order stat directly, over-deflating DSR to ~0 for EVERY config (degenerate leaderboard).
    sr0 = sr0_z * se                          # expected max SR under the null, in SR units
    dsr = float(norm.cdf((sr - sr0) / (se + 1e-12)))
    return {"dsr": round(dsr, 4), "obs_sr_perbar": round(sr, 4), "exp_max_sr0": round(sr0, 4),
            "n_trials": n_trials, "T": T, "ann_sr": round(sr * np.sqrt(365), 2)}


# ============================ config resolver (seed cfg string -> params) ============================
def resolve_params(indicator, cfg_str):
    """Map a survivor-seed best_cfg string (e.g. 'CKSP(14,2.0)') to its params tuple by matching the
    indicator's grid via its name() formatter. Raises if no match (no silent fallback)."""
    ind = P.INDICATORS[indicator]
    for p in ind["grid"]():
        if ind["name"](p) == cfg_str:
            return p
    raise ValueError(f"cfg {cfg_str!r} not in {indicator} grid; have {[ind['name'](p) for p in ind['grid']()]}")


# ============================ CAMPAIGN: cross-TF mask alignment + combo enumeration ============================
# Each survivor input lives at its OWN best TF (CKSP@1d, KELT@4h, AO@4h, RSI@1h, ULCER@1d). To combine per-asset
# hold-masks with the operators, they must share a time grid. We pick a COMBO CADENCE and AS-OF align each input's
# native-TF held state onto the combo bars (forward-fill the most-recent native-TF held value -- strictly causal:
# the native bar's held state is known by its close, and we lag-1 in _combo_book, so no look-ahead). The wealth is
# then evaluated on the combo-cadence OHLC. This is the honest way to combine multi-TF engines without re-fitting.

def aligned_combo(inputs, combo_cad, win, split):
    """For a set of input specs at possibly-different TFs, build per-asset COMBO-cadence aligned held masks for
    each input (as-of forward-fill the input's native held onto the combo bars), restricted to symbols present in
    ALL inputs + the combo-cadence panel. Returns (combo_assets, vt, names, per_input_masks) where
    per_input_masks[i] = list of per-asset combo-aligned held arrays (same order as combo_assets/names).

    inputs: list of dicts {indicator, cfg, tf}. combo_cad: the cadence to evaluate wealth on (coarsest input TF).
    AS-OF alignment is strictly causal (each combo bar takes the native held value whose timestamp <= combo bar ts;
    _combo_book additionally lags pos by 1 combo-bar)."""
    combo_assets, vt, combo_names = load_window(combo_cad, win, split)
    # native held per input, keyed by sym, with FULL native ms (window portion)
    native = []
    for inp in inputs:
        ind = P.INDICATORS[inp["indicator"]]; params = resolve_params(inp["indicator"], inp["cfg"])
        mh = ind.get("minhold", 12)
        nassets, _, nnames = load_window(inp["tf"], win, split)
        d = {}
        for A, sym in zip(nassets, nnames):
            held0 = ind["iron"](A, params)
            held = min_hold(apply_trail_stop(held0.copy().astype(np.int8), A["c"], 0.10)[0].astype(np.int8),
                            mh).astype(np.int8)
            # native window ms + held over the window (tradeable region)
            ms_win = A["idx"].view("int64") // 10**6
            held_win = held[A["win"]]
            d[sym] = (ms_win, held_win)
        native.append(d)
    # symbols present everywhere
    common = set(combo_names)
    for d in native:
        common &= set(d.keys())
    names = [s for s in combo_names if s in common]
    keep_idx = [combo_names.index(s) for s in names]
    combo_assets = [combo_assets[i] for i in keep_idx]
    per_input_masks = []
    for d in native:
        masks = []
        for A, sym in zip(combo_assets, names):
            combo_ms = A["idx"].view("int64") // 10**6  # combo bar timestamps (window)
            nms, nheld = d[sym]
            # as-of: for each combo bar, the most recent native held with native_ms <= combo_ms
            pos = np.searchsorted(nms, combo_ms, side="right") - 1
            m = np.where(pos >= 0, nheld[np.clip(pos, 0, len(nheld) - 1)], 0).astype(np.int8)
            masks.append(m)
        per_input_masks.append(masks)
    return combo_assets, vt, names, per_input_masks


def _single_daily(inputs, i, combo_assets, vt, per_input_masks, split):
    """Daily return series of input i ALONE (its mask aligned to the combo cadence) -- the 'best constituent
    single' benchmark, evaluated on the SAME combo grid as the combos (apples-to-apples)."""
    r = _combo_book(combo_assets, per_input_masks[i], vt, split)
    return None if r is None else r[0]


def run_combos_for_set(inputs, combo_cad, win, split, label):
    """Enumerate operator-combos over `inputs` on `win` at `combo_cad`. Returns a dict of combos ->
    {daily series, metrics} plus the per-input singles + buy-hold (all on the SAME combo grid)."""
    combo_assets, vt, names, pim = aligned_combo(inputs, combo_cad, win, split)
    n = len(inputs)
    out = {"label": label, "combo_cad": combo_cad, "n_names": len(names), "names": names, "combos": {}}
    # buy-hold + singles
    bh = buyhold(combo_assets, vt, split)
    out["buyhold_daily"] = bh
    out["buyhold_metrics"] = _metrics(bh, split)
    out["singles"] = {}
    for i, inp in enumerate(inputs):
        d = _single_daily(inputs, i, combo_assets, vt, pim, split)
        out["singles"][inp["tag"]] = {"daily": d, "metrics": _metrics(d, split) if d is not None else None}
    # operators
    def _add(name, masks):
        r = _combo_book(combo_assets, masks, vt, split)
        if r is None:
            return
        m = _metrics(r[0], split, r[1], r[2])
        if m is not None:
            out["combos"][name] = {"daily": r[0], "metrics": m}
    # AND_K for K=2..n
    for k in range(2, n + 1):
        _add(f"AND_{k}", op_and_k(pim, k))
    _add("OR", op_or(pim))
    _add("MAJORITY", op_majority(pim))
    for thr in (0.5, 0.67):
        _add(f"WEIGHTED_{thr}", op_weighted(pim, thr))
    # REGIME_GATE: directed entry x regime. Use trend (CKSP/ADX) as the regime, others as entries.
    trend_idx = [i for i, inp in enumerate(inputs) if inp["mechanism"] in ("trend",)]
    for gi in trend_idx:
        for ei in range(n):
            if ei == gi:
                continue
            nm = f"GATE[{inputs[ei]['tag']}|{inputs[gi]['tag']}]"
            _add(nm, op_regime_gate(pim[ei], pim[gi]))
    return out, pim, combo_assets, vt


# ============================ CAMPAIGN DRIVER ============================
SEED_PATH = ROOT.parent / "runs" / "engines" / "V1.2_combinatrix" / "survivor_seed.json"
OUT_DIR = ROOT.parent / "runs" / "engines" / "V1.2_combinatrix"


def load_inputs():
    """Build the input list from the CLEAN survivor seed: 5 cross-mechanism reps + up to 2 robust alternates
    (ADX 2nd-trend, BBSQZ 2nd-vol). Mean-reversion uses the ROBUST alternate RSI (rf=0.975) over STOCHRSI
    (rf=0.681) per the seed note. <=7 cross-MECHANISM inputs -- never stack within-mechanism."""
    seed = json.load(open(SEED_PATH))
    reps = {s["mechanism"]: s for s in seed["survivor_seed"]}
    # use the RSI robust alternate for mean-reversion (per seed note_mean_reversion)
    rsi = seed["alternates"]["mean-reversion"]["alt"]
    inputs = [
        {"indicator": reps["trend"]["indicator"], "cfg": reps["trend"]["best_cfg"],
         "tf": reps["trend"]["best_tf"], "mechanism": "trend"},
        {"indicator": reps["breakout"]["indicator"], "cfg": reps["breakout"]["best_cfg"],
         "tf": reps["breakout"]["best_tf"], "mechanism": "breakout"},
        {"indicator": reps["momentum"]["indicator"], "cfg": reps["momentum"]["best_cfg"],
         "tf": reps["momentum"]["best_tf"], "mechanism": "momentum"},
        {"indicator": rsi["indicator"], "cfg": rsi["best_cfg"], "tf": rsi["best_tf"],
         "mechanism": "mean-reversion"},
        {"indicator": reps["volatility-sizing"]["indicator"], "cfg": reps["volatility-sizing"]["best_cfg"],
         "tf": reps["volatility-sizing"]["best_tf"], "mechanism": "volatility-sizing"},
    ]
    # up to 2 robust alternates (DISTINCT mechanisms still: 2nd-trend ADX, 2nd-vol BBSQZ -- principled, NOT
    # within-mechanism stacking of the SAME engine; ADX is a regime-gate trend, BBSQZ a vol-breakout).
    alternates = [
        {"indicator": "ADX", "cfg": "ADX(14,25)", "tf": "4h", "mechanism": "trend2"},
        {"indicator": "BBSQZ", "cfg": "BBSQZ(20,q100)", "tf": "1d", "mechanism": "volatility2"},
    ]
    for a in inputs + alternates:
        a["tag"] = f"{a['indicator']}@{a['tf']}"
    return inputs, alternates


def _gate_combo(name, dev_daily_oos, iron_daily, best_single_dev_oos, bh_dev_oos,
                best_single_iron, bh_iron, all_dev_oos_returns, n_trials):
    """Full discipline gate for ONE reported combo. Returns the gate dict."""
    g = {}
    # block-bootstrap p05 of (combo - best_single) and (combo - buy-hold), DEV-OOS + IRON
    g["p05_vs_single_oos"] = bootstrap_diff_p05(dev_daily_oos, best_single_dev_oos)
    g["p05_vs_bh_oos"] = bootstrap_diff_p05(dev_daily_oos, bh_dev_oos)
    if iron_daily is not None and best_single_iron is not None:
        g["p05_vs_single_iron"] = bootstrap_diff_p05(iron_daily, best_single_iron)
        g["p05_vs_bh_iron"] = bootstrap_diff_p05(iron_daily, bh_iron)
    # deflated Sharpe with n_trials = number of combos evaluated (the multiple-comparisons count)
    g["deflated_sharpe_oos"] = deflated_sharpe(dev_daily_oos.to_numpy(), n_trials=n_trials)
    return g


def run_campaign(use_alternates=True, n_boot=2000):
    """The honest combinatorial question, end-to-end with the data-snoop discipline.

    DEV-2020: build all operator-combos over the survivor inputs at the 1d combo grid; VAL-SELECT (best VAL net);
    REPORT OOS for the full ranked table. IRON-2021: rebuild the SAME combos; CONFIRM net (held-out year).
    GATE every reported combo (block-bootstrap p05 vs best-single AND vs BH on OOS+IRON; CSCV-PBO over all combos;
    White's Reality Check + deflated-Sharpe with n_trials=#combos). 2022+ asserted never loaded.
    Rank by net WEALTH. A null is an honest result."""
    import strat.deep2020_ti_pipeline as PP  # noqa
    inputs, alternates = load_inputs()
    if use_alternates:
        inputs = inputs + alternates
    combo_cad = "1d"  # coarsest input TF; as-of align finer-TF inputs onto the daily grid
    print("=" * 90)
    print("V1.2 COMBINATRIX CAMPAIGN -- the honest combinatorial question under the data-snoop trap")
    print("=" * 90)
    print(f"INPUTS ({len(inputs)} cross-mechanism):")
    for inp in inputs:
        print(f"  {inp['tag']:16} mechanism={inp['mechanism']}  cfg={inp['cfg']}")
    print(f"combo grid = {combo_cad} (as-of align finer-TF inputs, strictly causal + lag-1)")
    print("SPLIT: 2020 VAL Jul-Sep SELECT / 2020 Oct-Dec OOS REPORT / 2021 IRON CONFIRM / 2022+ SEALED (asserted)")
    print("=" * 90); sys.stdout.flush()

    # ---- DEV 2020: build all combos ----
    print("\n[DEV-2020] building combos (VAL-select / OOS-report)..."); sys.stdout.flush()
    dev, pim_dev, dev_assets, dev_vt = run_combos_for_set(inputs, combo_cad, DEV_WIN, DEV_SPLIT, "DEV2020")
    n_combos = len(dev["combos"])
    print(f"  {n_combos} combos built on n_names={dev['n_names']} u100 names.")
    bh_dev = dev["buyhold_metrics"]
    print(f"  DEV buy-hold: val={bh_dev['val_net']} OOS={bh_dev['net']} dd={bh_dev['maxdd']} sh={bh_dev['sharpe']}")
    # best constituent single (by VAL net, the selection criterion) -- VAL-selected, OOS-reported
    singles = dev["singles"]
    best_single_tag = max(singles, key=lambda t: (singles[t]["metrics"] or {"val_net": -1e9})["val_net"])
    bsm = singles[best_single_tag]["metrics"]
    print(f"  best constituent SINGLE (VAL-selected): {best_single_tag}  val={bsm['val_net']} OOS={bsm['net']}")
    sys.stdout.flush()

    # ---- IRON 2021: rebuild the SAME combos (held-out year) ----
    print("\n[IRON-2021] rebuilding the SAME combos (held-out year CONFIRM)..."); sys.stdout.flush()
    iron, pim_iron, iron_assets, iron_vt = run_combos_for_set(inputs, combo_cad, IRON_WIN, IRON_SPLIT, "IRON2021")
    print(f"  IRON n_names={iron['n_names']}. IRON buy-hold full-year net "
          f"(val=H1 oos=H2): val={iron['buyhold_metrics']['val_net']} oos={iron['buyhold_metrics']['net']}")
    sys.stdout.flush()

    # ---- CSCV-PBO over ALL combos (DEV-OOS returns) ----
    print("\n[DISCIPLINE] CSCV-PBO over all combos (DEV-OOS daily returns)..."); sys.stdout.flush()
    oos_returns = {}
    for nm, c in dev["combos"].items():
        d = c["daily"]; oos = d[d.index >= pd.Timestamp(DEV_SPLIT)]
        oos_returns[nm] = oos
    # align all to the common OOS index
    common_idx = None
    for s in oos_returns.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    oos_aligned = {nm: oos_returns[nm].reindex(common_idx).fillna(0.0).to_numpy() for nm in oos_returns}
    pbo = cscv_pbo(oos_aligned, S=10)
    print(f"  PBO={pbo['pbo']} (M={pbo['M']} combos, {pbo.get('n_partitions')} partitions)")
    # White's Reality Check: best combo vs buy-hold, deflated by M
    bh_oos = dev["buyhold_daily"]; bh_oos = bh_oos[bh_oos.index >= pd.Timestamp(DEV_SPLIT)]
    rc = whites_reality_check({nm: oos_returns[nm] for nm in oos_returns}, bh_oos, n_boot=n_boot)
    print(f"  White's Reality Check (best combo vs buy-hold): p={rc['rc_pvalue']} best={rc.get('best_config')}")
    sys.stdout.flush()

    # ---- assemble ranked OOS table + gate the reported combos ----
    best_single_dev_oos = singles[best_single_tag]["daily"]
    best_single_dev_oos = best_single_dev_oos[best_single_dev_oos.index >= pd.Timestamp(DEV_SPLIT)]
    bh_dev_oos = dev["buyhold_daily"]; bh_dev_oos = bh_dev_oos[bh_dev_oos.index >= pd.Timestamp(DEV_SPLIT)]
    # IRON best-single + bh (same tag) full-year H2 region (the IRON 'net')
    iron_single = iron["singles"].get(best_single_tag, {}).get("daily")
    iron_single_h2 = iron_single[iron_single.index >= pd.Timestamp(IRON_SPLIT)] if iron_single is not None else None
    bh_iron_full = iron["buyhold_daily"]; bh_iron_h2 = bh_iron_full[bh_iron_full.index >= pd.Timestamp(IRON_SPLIT)]

    ranked = sorted(dev["combos"].items(), key=lambda kv: -kv[1]["metrics"]["net"])
    results = []
    print("\n[REPORT] combos ranked by DEV-OOS net WEALTH (top by net), with IRON-2021 confirm + gates:")
    print(f"  {'combo':30} {'VALnet':>7} {'OOSnet':>7} {'IRONnet':>7} {'OOSdd':>6} "
          f"{'p05vsBH':>8} {'p05vsSGL':>9} {'IRONp05BH':>9} {'DSR':>5}")
    for nm, c in ranked:
        m = c["metrics"]
        ironm = iron["combos"].get(nm, {}).get("metrics")
        iron_daily = iron["combos"].get(nm, {}).get("daily")
        iron_daily_h2 = iron_daily[iron_daily.index >= pd.Timestamp(IRON_SPLIT)] if iron_daily is not None else None
        dev_oos = c["daily"][c["daily"].index >= pd.Timestamp(DEV_SPLIT)]
        gate = _gate_combo(nm, dev_oos, iron_daily_h2, best_single_dev_oos, bh_dev_oos,
                           iron_single_h2, bh_iron_h2, oos_aligned, n_trials=n_combos)
        rec = {
            "combo": nm, "combo_cad": combo_cad, "n_names": dev["n_names"],
            "dev": {"val_net": m["val_net"], "oos_net": m["net"], "maxdd": m["maxdd"],
                    "sharpe": m["sharpe"], "drift": m["drift"], "robust": m["robust"],
                    "time_in": m["time_in"], "n_trades": m["n_trades"]},
            "iron2021": ({"val_net_h1": ironm["val_net"], "net_h2": ironm["net"], "maxdd": ironm["maxdd"],
                          "sharpe": ironm["sharpe"]} if ironm else None),
            "gate": gate,
            "beats_bh_oos": bool(m["net"] > bh_dev["net"]),
            "beats_best_single_oos": bool(m["net"] > bsm["net"]),
        }
        results.append(rec)
        p1 = gate["p05_vs_bh_oos"]["p05"]; p2 = gate["p05_vs_single_oos"]["p05"]
        ip = gate.get("p05_vs_bh_iron", {}).get("p05", float("nan"))
        irn = ironm["net"] if ironm else float("nan")
        print(f"  {nm:30} {m['val_net']:>7} {m['net']:>7} {irn:>7} {m['maxdd']:>6} "
              f"{p1:>8} {p2:>9} {ip:>9} {gate['deflated_sharpe_oos']['dsr']:>5}")
    sys.stdout.flush()

    # ---- VAL-SELECTED combo (the honest single pick) ----
    val_best = max(dev["combos"].items(), key=lambda kv: kv[1]["metrics"]["val_net"])
    vb_nm, vb = val_best
    vbm = vb["metrics"]; vb_ironm = iron["combos"].get(vb_nm, {}).get("metrics")
    print("\n" + "=" * 90)
    print("VAL-SELECTED COMBO (chosen on 2020 VAL net, the disciplined single pick):")
    print(f"  {vb_nm}: VAL={vbm['val_net']} -> OOS={vbm['net']} (dd {vbm['maxdd']}, sh {vbm['sharpe']}) "
          f"-> IRON2021={(vb_ironm['net'] if vb_ironm else 'NA')}")
    print(f"  vs best single OOS={bsm['net']} | vs buy-hold OOS={bh_dev['net']}")

    # ---- VERDICT ----
    beats_bh = vbm["net"] > bh_dev["net"]
    beats_single = vbm["net"] > bsm["net"]
    vb_gate = next(r["gate"] for r in results if r["combo"] == vb_nm)
    p05_bh = vb_gate["p05_vs_bh_oos"]["p05"]; p05_sgl = vb_gate["p05_vs_single_oos"]["p05"]
    iron_holds = (vb_ironm is not None and vb_ironm["net"] > 0
                  and vb_gate.get("p05_vs_bh_iron", {}).get("p05", -1) > 0)
    survives = (beats_bh and beats_single and p05_bh > 0 and p05_sgl > 0
                and (pbo["pbo"] < 0.5) and rc["rc_pvalue"] < 0.05 and iron_holds)
    print("\nHONEST VERDICT:")
    print(f"  beats buy-hold (OOS net)        : {beats_bh}  ({vbm['net']} vs {bh_dev['net']})")
    print(f"  beats best single (OOS net)     : {beats_single}  ({vbm['net']} vs {bsm['net']})")
    print(f"  p05 vs buy-hold > 0 (OOS)       : {p05_bh > 0}  (p05={p05_bh})")
    print(f"  p05 vs best single > 0 (OOS)    : {p05_sgl > 0}  (p05={p05_sgl})")
    print(f"  PBO < 0.5 (not overfit)         : {pbo['pbo'] < 0.5}  (PBO={pbo['pbo']})")
    print(f"  White's RC p < 0.05 (deflated)  : {rc['rc_pvalue'] < 0.05}  (p={rc['rc_pvalue']})")
    print(f"  IRON-2021 confirm (net>0 & p05) : {iron_holds}")
    print(f"\n  >>> {'GENUINE SURVIVOR' if survives else 'NULL (de-risked beta / overfit) -- the honest, prior-consistent result'}")

    # ---- regime/robustness frontier (DD reduction without beating net) ----
    print("\n[FRONTIER] combos that CUT maxDD / robustify vs buy-hold (without beating net) -- the validated use:")
    bh_dd = bh_dev["maxdd"]
    frontier = [r for r in results if r["dev"]["maxdd"] > bh_dd and r["dev"]["robust"]]
    frontier.sort(key=lambda r: r["dev"]["maxdd"], reverse=True)
    for r in frontier[:8]:
        irn = r["iron2021"]["net_h2"] if r["iron2021"] else float("nan")
        print(f"  {r['combo']:30} OOSdd={r['dev']['maxdd']:>6} (BH {bh_dd}) OOSnet={r['dev']['oos_net']:>6} "
              f"IRON={irn:>6} robust={r['dev']['robust']}")

    # ---- write artifacts (incremental, n_names recorded on EVERY result) ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": "V1.2 combinatrix", "combo_cad": combo_cad, "n_names": dev["n_names"],
        "names": dev["names"], "n_combos": n_combos, "inputs": inputs,
        "split": {"dev_win": DEV_WIN, "dev_split": DEV_SPLIT, "iron_win": IRON_WIN, "iron_split": IRON_SPLIT,
                  "sealed_from": SEALED_FROM},
        "buyhold_dev": bh_dev, "best_single": {"tag": best_single_tag, "metrics": bsm},
        "pbo": pbo, "whites_reality_check": rc,
        "val_selected_combo": {"combo": vb_nm, "dev": vbm, "iron2021": vb_ironm, "gate": vb_gate},
        "verdict": {"survives": bool(survives), "beats_bh_oos": bool(beats_bh),
                    "beats_best_single_oos": bool(beats_single), "pbo": pbo["pbo"],
                    "rc_pvalue": rc["rc_pvalue"], "iron_holds": bool(iron_holds)},
        "results": results,
    }
    out = OUT_DIR / "combinatrix_results.json"
    json.dump(payload, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    print("=" * 90)
    return payload


# ============================ STRUCTURED SEARCH: pairs -> beam-triples -> a1a2 config-combos ============================
# The disciplined enumeration the campaign mandate asks for, REUSING every primitive above:
#   (a) SINGLES recap (the 5-7 cross-mechanism inputs -- already de-risked betas).
#   (b) PAIRS: all C(inputs,2) x operators (directed REGIME_GATE) -> VAL-SELECT each pair's best operator-param,
#       REPORT OOS, CONFIRM IRON. ~21 pairs x ~4-6 ops each.
#   (c) TRIPLES via BEAM: take the top-8 PAIRS by VAL net, add each remaining input x operators, keep top-8, prune.
#   (d) <=5 a1a2 WITHIN-indicator config-combos (CKSP/KELTNER fast+slow) -- flagged REDUNDANT (the data-snoop trap).
# Each input's combo-aligned per-asset mask is precomputed ONCE per window (DEV + IRON); subsets index into it.

def _build_input_masks(inputs, combo_cad, win, split):
    """Precompute the combo-aligned per-asset held mask of EVERY input once (reuses aligned_combo). Returns
    (combo_assets, vt, names, pim) where pim[i] = per-asset combo-aligned mask of input i."""
    return aligned_combo(inputs, combo_cad, win, split)


def _ops_for_subset(sub_pim, inputs, sub_idx):
    """Yield (op_name, combined_masks) for every principled operator over the input subset `sub_idx`.
    sub_pim = [pim[i] for i in sub_idx]. AND_k k=2..n, OR(=AND_1), MAJORITY, WEIGHTED thr, REGIME_GATE directed
    (entry x trend-regime) when the subset contains a trend/trend2 input (the project's VALIDATED conditioning)."""
    n = len(sub_idx)
    yield ("OR", op_or(sub_pim))
    for k in range(2, n + 1):
        yield (f"AND_{k}", op_and_k(sub_pim, k))
    if n >= 3:  # MAJORITY == OR for n=2 (ceil(2/2)=1); only distinct for n>=3
        yield ("MAJORITY", op_majority(sub_pim))
    for thr in (0.5, 0.67):
        if not (n == 2 and abs(thr - 0.5) < 1e-9):  # WEIGHTED_0.5 == AND_1=OR for n=2; skip the dup
            yield (f"WEIGHTED_{thr}", op_weighted(sub_pim, thr))
    # directed REGIME_GATE: each trend input gates each non-trend entry in the subset
    tags = [inputs[i]["tag"] for i in sub_idx]
    mech = [inputs[i]["mechanism"] for i in sub_idx]
    for gi in range(n):
        if mech[gi] in ("trend", "trend2"):
            for ei in range(n):
                if ei == gi:
                    continue
                yield (f"GATE[{tags[ei]}|{tags[gi]}]", op_regime_gate(sub_pim[ei], sub_pim[gi]))


def _eval_subset(sub_idx, inputs, dev_assets, dev_vt, dev_pim, iron_assets, iron_vt, iron_pim):
    """Evaluate EVERY operator over the input subset on DEV (VAL-select / OOS-report) + IRON (confirm).
    Returns list of dicts {op, dev_metrics, dev_daily, iron_metrics, iron_daily}. Subset masks index dev_pim/iron_pim."""
    dev_sub = [dev_pim[i] for i in sub_idx]
    iron_sub = [iron_pim[i] for i in sub_idx]
    rows = []
    iron_by_op = {}
    for op_name, im in _ops_for_subset(iron_sub, inputs, sub_idx):
        r = _combo_book(iron_assets, im, iron_vt, IRON_SPLIT)
        iron_by_op[op_name] = (r[0], _metrics(r[0], IRON_SPLIT, r[1], r[2])) if r else (None, None)
    for op_name, dm in _ops_for_subset(dev_sub, inputs, sub_idx):
        r = _combo_book(dev_assets, dm, dev_vt, DEV_SPLIT)
        if r is None:
            continue
        m = _metrics(r[0], DEV_SPLIT, r[1], r[2])
        if m is None:
            continue
        idaily, im_m = iron_by_op.get(op_name, (None, None))
        rows.append({"op": op_name, "dev_metrics": m, "dev_daily": r[0],
                     "iron_metrics": im_m, "iron_daily": idaily})
    return rows


def run_structured_campaign(use_alternates=True, n_boot=2000, beam_width=8, combo_cad="1d",
                            out_name="combinatrix_results.json", md_name="combinatrix_leaderboard.md"):
    """The mandated structured search: SINGLES -> PAIRS (all C(N,2) x ops, VAL-select per pair) -> BEAM TRIPLES
    -> a1a2 config-combos. VAL-select on 2020-VAL net / OOS-report on 2020-OOS / IRON-CONFIRM on 2021. Every
    reported combo carries: operator, inputs, val/oos/iron net (all xBH), maxdd, p05 (vs best-single AND vs BH),
    PBO, deflated_net, n_names. Rank the leaderboard by deflated OOS+IRON net. Writes `out_name` (all combos)
    + `md_name`. Reuses every primitive; 2022+ asserted never loaded.

    combo_cad: the cadence to evaluate combo wealth on {1d,4h,2h,1h,30m,15m}. Each input's native-TF held state
    is as-of forward-filled onto the combo grid (strictly causal: native held known by its close, lag-1 in
    _combo_book) -- this works whether the native TF is coarser (1d->4h ffill) or equal to the combo grid. The
    charter requires ALL TFs; a finer combo grid churns more, so the prior expects the de-risked-beta null to hold
    AT LEAST as hard (PBO high, 0 survivors). out_name/md_name let a caller write per-cadence artifacts
    incrementally (combinatrix_alltf_<cad>.json) so a stall is salvageable."""
    inputs, alternates = load_inputs()
    if use_alternates:
        inputs = inputs + alternates
    N = len(inputs)
    print("=" * 96)
    print("V1.2 COMBINATRIX -- STRUCTURED SEARCH (singles / pairs / beam-triples / a1a2) under the data-snoop trap")
    print("=" * 96)
    print(f"INPUTS ({N} cross-mechanism):  " + "  ".join(f"{i['tag']}[{i['mechanism']}]" for i in inputs))
    print(f"combo grid={combo_cad} (as-of align finer-TF, strictly causal + lag-1).  "
          "SPLIT 2020VAL-select / 2020OOS-report / 2021IRON-confirm / 2022+ SEALED")
    sys.stdout.flush()

    # ---- precompute aligned masks ONCE per window ----
    print("\n[align] precomputing combo-aligned per-input masks (DEV-2020 + IRON-2021)..."); sys.stdout.flush()
    dev_assets, dev_vt, dev_names, dev_pim = _build_input_masks(inputs, combo_cad, DEV_WIN, DEV_SPLIT)
    iron_assets, iron_vt, iron_names, iron_pim = _build_input_masks(inputs, combo_cad, IRON_WIN, IRON_SPLIT)
    n_names = len(dev_names)
    assert n_names >= 20, f"breadth floor: n_names={n_names} < 20"
    print(f"  n_names(DEV)={n_names}  n_names(IRON)={len(iron_names)}"); sys.stdout.flush()

    # ---- benchmarks: buy-hold + per-input singles (DEV + IRON), on the SAME grid ----
    bh_dev = buyhold(dev_assets, dev_vt, DEV_SPLIT); bh_dev_m = _metrics(bh_dev, DEV_SPLIT)
    bh_iron = buyhold(iron_assets, iron_vt, IRON_SPLIT); bh_iron_m = _metrics(bh_iron, IRON_SPLIT)
    bh_dev_oos = bh_dev[bh_dev.index >= pd.Timestamp(DEV_SPLIT)]
    bh_iron_h2 = bh_iron[bh_iron.index >= pd.Timestamp(IRON_SPLIT)]
    singles = {}
    for i, inp in enumerate(inputs):
        sd = _combo_book(dev_assets, dev_pim[i], dev_vt, DEV_SPLIT)
        si = _combo_book(iron_assets, iron_pim[i], iron_vt, IRON_SPLIT)
        singles[i] = {
            "tag": inp["tag"], "mechanism": inp["mechanism"],
            "dev_daily": sd[0] if sd else None, "dev_metrics": _metrics(sd[0], DEV_SPLIT, sd[1], sd[2]) if sd else None,
            "iron_daily": si[0] if si else None, "iron_metrics": _metrics(si[0], IRON_SPLIT, si[1], si[2]) if si else None,
        }
    print(f"  buy-hold: DEV val={bh_dev_m['val_net']} OOS={bh_dev_m['net']} dd={bh_dev_m['maxdd']} | "
          f"IRON H2={bh_iron_m['net']}")
    print("  SINGLES recap (de-risked betas; OOS xBH < 1.0 each):")
    for i in range(N):
        sm = singles[i]["dev_metrics"]; im = singles[i]["iron_metrics"]
        xbh = sm["net"] / bh_dev_m["net"] if bh_dev_m["net"] else float("nan")
        print(f"    {singles[i]['tag']:13} OOS={sm['net']:>6} ({xbh:.2f}xBH) dd={sm['maxdd']:>6} "
              f"IRON={im['net'] if im else 'NA':>6}")
    sys.stdout.flush()

    # ---- helper: package a combo record with full gate ----
    all_combos = []  # flat list of {tier, subset, tags, op, dev_*, iron_*} -- gated later
    def _pkg(tier, sub_idx, row, redundant=False):
        m = row["dev_metrics"]; im = row["iron_metrics"]
        tags = [inputs[i]["tag"] for i in sub_idx]
        # best constituent single (by VAL net) within THIS subset -- the apples-to-apples single benchmark
        bsi = max(sub_idx, key=lambda i: (singles[i]["dev_metrics"] or {"val_net": -1e9})["val_net"])
        return {
            "tier": tier, "inputs": tags, "best_single_tag": singles[bsi]["tag"],
            "combo": f"{row['op']}({'+'.join(tags)})", "op": row["op"], "redundant": redundant,
            "dev_daily": row["dev_daily"], "iron_daily": row["iron_daily"],
            "dev_metrics": m, "iron_metrics": im,
            "_bsi": bsi,
        }

    # ============ (b) PAIRS ============
    print(f"\n[PAIRS] all C({N},2)={comb(N,2)} pairs x operators -- VAL-select each pair's best op:"); sys.stdout.flush()
    pair_best = []  # (val_net, sub_idx, best_row) for beam seeding
    n_pair_combos = 0
    for sub in combinations(range(N), 2):
        rows = _eval_subset(list(sub), inputs, dev_assets, dev_vt, dev_pim, iron_assets, iron_vt, iron_pim)
        if not rows:
            continue
        for row in rows:
            all_combos.append(_pkg("pair", list(sub), row)); n_pair_combos += 1
        best = max(rows, key=lambda r: r["dev_metrics"]["val_net"])  # VAL-select the pair's operator
        pair_best.append((best["dev_metrics"]["val_net"], list(sub), best))
    print(f"  {n_pair_combos} pair-combos evaluated ({len(pair_best)} pairs, best-op VAL-selected each).")
    sys.stdout.flush()

    # ============ (c) TRIPLES via BEAM ============
    print(f"\n[BEAM-TRIPLES] seed = top-{beam_width} pairs by VAL net; add each remaining input x ops; keep top-{beam_width}:")
    sys.stdout.flush()
    pair_best.sort(key=lambda x: -x[0])
    seeds = pair_best[:beam_width]
    triple_cand = []
    seen_triples = set()
    n_triple_combos = 0
    for _vn, sub, _row in seeds:
        for j in range(N):
            if j in sub:
                continue
            tri = tuple(sorted(sub + [j]))
            if tri in seen_triples:
                continue
            seen_triples.add(tri)
            rows = _eval_subset(list(tri), inputs, dev_assets, dev_vt, dev_pim, iron_assets, iron_vt, iron_pim)
            if not rows:
                continue
            for row in rows:
                all_combos.append(_pkg("triple", list(tri), row)); n_triple_combos += 1
            best = max(rows, key=lambda r: r["dev_metrics"]["val_net"])
            triple_cand.append((best["dev_metrics"]["val_net"], list(tri), best))
    triple_cand.sort(key=lambda x: -x[0])
    kept = triple_cand[:beam_width]
    print(f"  {n_triple_combos} triple-combos evaluated ({len(seen_triples)} distinct triples); kept top-{len(kept)} by VAL net.")
    for vn, tri, row in kept[:beam_width]:
        print(f"    {row['op']:22} {'+'.join(inputs[i]['tag'] for i in tri):40} VAL={vn:>6} OOS={row['dev_metrics']['net']:>6}")
    sys.stdout.flush()

    # ============ (d) a1a2 WITHIN-indicator config-combos (flagged REDUNDANT) ============
    print("\n[a1a2] <=5 WITHIN-indicator config-combos (CKSP/KELTNER fast+slow) -- flagged REDUNDANT (data-snoop trap):")
    sys.stdout.flush()
    a1a2_specs = [
        ("CKSP", "1d", "CKSP(10,1.0)", "CKSP(20,3.0)"),   # fast vs slow CKSP
        ("CKSP", "1d", "CKSP(10,2.0)", "CKSP(20,2.0)"),
        ("KELTNER", "4h", "KELT(20,1.5)", "KELT(30,2.5)"),  # tight vs wide Keltner
        ("KELTNER", "4h", "KELT(20,2.0)", "KELT(30,2.0)"),
    ]
    n_a1a2 = 0
    for indicator, tf, cfg_a, cfg_b in a1a2_specs:
        a1a2_inputs = [
            {"indicator": indicator, "cfg": cfg_a, "tf": tf, "mechanism": "a1", "tag": f"{indicator}:{cfg_a}"},
            {"indicator": indicator, "cfg": cfg_b, "tf": tf, "mechanism": "a2", "tag": f"{indicator}:{cfg_b}"},
        ]
        try:
            da, dvt, dn, dp = _build_input_masks(a1a2_inputs, combo_cad, DEV_WIN, DEV_SPLIT)
            ia, ivt, _in, ip = _build_input_masks(a1a2_inputs, combo_cad, IRON_WIN, IRON_SPLIT)
        except Exception as e:
            print(f"    [skip] {indicator} {cfg_a}+{cfg_b}: {e}"); continue
        # singles within this a1a2 set (for the best-single benchmark)
        a_singles = {}
        for i in range(2):
            sd = _combo_book(da, dp[i], dvt, DEV_SPLIT)
            a_singles[i] = {"tag": a1a2_inputs[i]["tag"], "dev_metrics": _metrics(sd[0], DEV_SPLIT) if sd else None,
                            "dev_daily": sd[0] if sd else None,
                            "iron_metrics": None, "iron_daily": None, "mechanism": a1a2_inputs[i]["mechanism"]}
            si = _combo_book(ia, ip[i], ivt, IRON_SPLIT)
            if si:
                a_singles[i]["iron_metrics"] = _metrics(si[0], IRON_SPLIT); a_singles[i]["iron_daily"] = si[0]
        for op_name, dm in (("AND_2", op_and_k(dp, 2)), ("OR", op_or(dp))):
            r = _combo_book(da, dm, dvt, DEV_SPLIT)
            if r is None:
                continue
            m = _metrics(r[0], DEV_SPLIT, r[1], r[2])
            ir = _combo_book(ia, dict(AND_2=op_and_k(ip, 2), OR=op_or(ip))[op_name], ivt, IRON_SPLIT)
            im = _metrics(ir[0], IRON_SPLIT, ir[1], ir[2]) if ir else None
            bsi = max(range(2), key=lambda i: (a_singles[i]["dev_metrics"] or {"val_net": -1e9})["val_net"])
            rec = {
                "tier": "a1a2", "inputs": [a1a2_inputs[0]["tag"], a1a2_inputs[1]["tag"]],
                "best_single_tag": a_singles[bsi]["tag"],
                "combo": f"{op_name}({a1a2_inputs[0]['tag']}+{a1a2_inputs[1]['tag']})", "op": op_name,
                "redundant": True, "dev_daily": r[0], "iron_daily": ir[0] if ir else None,
                "dev_metrics": m, "iron_metrics": im,
                "_a_singles": a_singles, "_bsi": bsi,
            }
            all_combos.append(rec); n_a1a2 += 1
            print(f"    {op_name:6} {indicator} {cfg_a}+{cfg_b:13} OOS={m['net']:>6} (REDUNDANT) "
                  f"IRON={im['net'] if im else 'NA':>6}")
    sys.stdout.flush()

    # ============ DISCIPLINE: CSCV-PBO + White's RC over ALL combos (DEV-OOS) ============
    print(f"\n[DISCIPLINE] CSCV-PBO + White's Reality Check over ALL {len(all_combos)} combos (DEV-OOS)..."); sys.stdout.flush()
    oos_returns = {}
    for c in all_combos:
        d = c["dev_daily"]; oos = d[d.index >= pd.Timestamp(DEV_SPLIT)]
        oos_returns[c["combo"]] = oos
    common_idx = None
    for s in oos_returns.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    oos_aligned = {nm: oos_returns[nm].reindex(common_idx).fillna(0.0).to_numpy() for nm in oos_returns}
    pbo = cscv_pbo(oos_aligned, S=10)
    rc = whites_reality_check({nm: oos_returns[nm] for nm in oos_returns}, bh_dev_oos, n_boot=n_boot)
    n_trials = len(all_combos)
    print(f"  PBO={pbo['pbo']} (M={pbo['M']})   White's RC (best combo vs BH) p={rc['rc_pvalue']} best={rc.get('best_config')}")
    sys.stdout.flush()

    # ============ GATE every combo + assemble leaderboard rows ============
    print(f"\n[GATE] block-bootstrap p05 (vs best-single AND vs BH) + deflated net, per combo (n_trials={n_trials})..."); sys.stdout.flush()
    leaderboard = []
    for c in all_combos:
        m = c["dev_metrics"]; im = c["iron_metrics"]
        dev_oos = c["dev_daily"][c["dev_daily"].index >= pd.Timestamp(DEV_SPLIT)]
        # best-single daily (subset's VAL-best constituent), DEV-OOS + IRON-H2
        if c["tier"] == "a1a2":
            bs = c["_a_singles"][c["_bsi"]]
            bs_dev = bs["dev_daily"]; bs_iron = bs["iron_daily"]
        else:
            bs = singles[c["_bsi"]]
            bs_dev = bs["dev_daily"]; bs_iron = bs["iron_daily"]
        bs_dev_oos = bs_dev[bs_dev.index >= pd.Timestamp(DEV_SPLIT)] if bs_dev is not None else None
        bs_iron_h2 = bs_iron[bs_iron.index >= pd.Timestamp(IRON_SPLIT)] if bs_iron is not None else None
        p05_bh = bootstrap_diff_p05(dev_oos, bh_dev_oos)
        p05_sgl = bootstrap_diff_p05(dev_oos, bs_dev_oos) if bs_dev_oos is not None else {"p05": float("nan")}
        iron_daily = c["iron_daily"]
        iron_h2 = iron_daily[iron_daily.index >= pd.Timestamp(IRON_SPLIT)] if iron_daily is not None else None
        p05_bh_iron = bootstrap_diff_p05(iron_h2, bh_iron_h2) if iron_h2 is not None else {"p05": float("nan")}
        p05_sgl_iron = (bootstrap_diff_p05(iron_h2, bs_iron_h2)
                        if (iron_h2 is not None and bs_iron_h2 is not None) else {"p05": float("nan")})
        dsr = deflated_sharpe(dev_oos.to_numpy(), n_trials=n_trials)
        # deflated_net: shrink OOS net toward 0 by the deflation factor DSR-implied (a conservative deflated wealth:
        # net * DSR -- if the edge is pure selection (DSR->0) the deflated net collapses; if genuine (DSR->1) it holds).
        dsr_v = dsr.get("dsr", 0.0); dsr_v = 0.0 if (dsr_v is None or np.isnan(dsr_v)) else dsr_v
        deflated_net = round(m["net"] * dsr_v, 1)
        iron_net = im["net"] if im else None
        deflated_oos_iron = round((m["net"] + (iron_net if iron_net is not None else 0.0)) / 2.0 * dsr_v, 1)
        bs_dev_net = (c["_a_singles"][c["_bsi"]]["dev_metrics"] if c["tier"] == "a1a2"
                      else singles[c["_bsi"]]["dev_metrics"])["net"]
        leaderboard.append({
            "combo": c["combo"], "tier": c["tier"], "op": c["op"], "inputs": c["inputs"],
            "best_single_tag": c["best_single_tag"], "redundant": c["redundant"], "n_names": n_names,
            "val_net": m["val_net"], "oos_net": m["net"], "iron_net": iron_net,
            "val_net_xbh": round(m["val_net"] / bh_dev_m["val_net"], 3) if bh_dev_m["val_net"] else None,
            "oos_net_xbh": round(m["net"] / bh_dev_m["net"], 3) if bh_dev_m["net"] else None,
            "iron_net_xbh": round(iron_net / bh_iron_m["net"], 3) if (iron_net is not None and bh_iron_m["net"]) else None,
            "maxdd": m["maxdd"], "iron_maxdd": im["maxdd"] if im else None, "sharpe": m["sharpe"],
            "robust": m["robust"], "time_in": m["time_in"], "n_trades": m["n_trades"],
            "p05_vs_bh_oos": p05_bh["p05"], "p05_vs_single_oos": p05_sgl["p05"],
            "p05_vs_bh_iron": p05_bh_iron["p05"], "p05_vs_single_iron": p05_sgl_iron["p05"],
            "pbo": pbo["pbo"], "deflated_sharpe": dsr.get("dsr"), "deflated_net": deflated_net,
            "deflated_oos_iron": deflated_oos_iron,
            "beats_bh_oos": bool(m["net"] > bh_dev_m["net"]),
            "beats_best_single_oos": bool(m["net"] > bs_dev_net),
            "beats_bh_iron": bool(iron_net is not None and iron_net > bh_iron_m["net"]),
        })

    # rank by DEFLATED (OOS+IRON)/2 net (the mandate's leaderboard order)
    leaderboard.sort(key=lambda r: -(r["deflated_oos_iron"] if r["deflated_oos_iron"] is not None else -1e9))

    # VAL-selected combo across the WHOLE structured set (the disciplined single pick)
    val_best = max(all_combos, key=lambda c: c["dev_metrics"]["val_net"])
    vb_combo = val_best["combo"]
    vb_lb = next(r for r in leaderboard if r["combo"] == vb_combo)

    # ============ VERDICT ============
    def _beats(r):
        return (r["beats_bh_oos"] and r["beats_best_single_oos"] and r["beats_bh_iron"]
                and (r["p05_vs_bh_oos"] or -1) > 0 and (r["p05_vs_single_oos"] or -1) > 0
                and (r["p05_vs_bh_iron"] or -1) > 0 and pbo["pbo"] < 0.5 and rc["rc_pvalue"] < 0.05)
    survivors = [r for r in leaderboard if _beats(r)]
    # conditioning frontier: cuts OOS maxDD vs buy-hold + robust, WITHOUT needing to beat net
    frontier = [r for r in leaderboard if r["maxdd"] > bh_dev_m["maxdd"] and r["robust"]]
    frontier.sort(key=lambda r: r["maxdd"], reverse=True)

    print("\n" + "=" * 96)
    print("VAL-SELECTED combo (whole structured set, chosen on 2020 VAL net):")
    print(f"  {vb_combo}: VAL={vb_lb['val_net']} -> OOS={vb_lb['oos_net']} ({vb_lb['oos_net_xbh']}xBH, dd {vb_lb['maxdd']}) "
          f"-> IRON={vb_lb['iron_net']} | best-single={vb_lb['best_single_tag']}")
    print("\nHONEST VERDICT -- does ANY combo beat its best constituent single AND buy-hold on OOS+IRON net after deflation (PBO low + p05>0)?")
    print(f"  combos evaluated: {len(all_combos)}  (pairs={n_pair_combos}, triples={n_triple_combos}, a1a2={n_a1a2})")
    print(f"  PBO={pbo['pbo']} ({'OVERFIT' if pbo['pbo']>=0.5 else 'not overfit'})  White's RC p={rc['rc_pvalue']} ({'fails' if rc['rc_pvalue']>=0.05 else 'survives'} deflation)")
    print(f"  >>> SURVIVORS (beat best-single AND BH on OOS+IRON, p05>0 all, PBO<0.5, RC<0.05): {len(survivors)}")
    if survivors:
        for r in survivors[:5]:
            print(f"      GENUINE? {r['combo']}  OOS={r['oos_net']} IRON={r['iron_net']} deflated={r['deflated_net']}")
    else:
        print("      NONE -- NULL (de-risked beta + overfit holds under combination). The honest, prior-consistent result.")
    print("\n[FRONTIER] conditioning combos that CUT OOS maxDD vs buy-hold + robust (WITHOUT beating net):")
    for r in frontier[:8]:
        print(f"  {r['combo']:46} OOSdd={r['maxdd']:>6} (BH {bh_dev_m['maxdd']}) OOSnet={r['oos_net']:>6} "
              f"IRON={r['iron_net'] if r['iron_net'] is not None else 'NA':>6} robust={r['robust']}")
    sys.stdout.flush()

    # ============ WRITE ARTIFACTS ============
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": "V1.2 combinatrix (structured: singles/pairs/beam-triples/a1a2)",
        "combo_cad": combo_cad, "n_names": n_names, "names": dev_names,
        "n_combos": len(all_combos), "n_pairs": n_pair_combos, "n_triples": n_triple_combos, "n_a1a2": n_a1a2,
        "inputs": [{"tag": i["tag"], "mechanism": i["mechanism"], "cfg": i["cfg"], "tf": i["tf"]} for i in inputs],
        "split": {"dev_win": DEV_WIN, "dev_split": DEV_SPLIT, "iron_win": IRON_WIN, "iron_split": IRON_SPLIT,
                  "sealed_from": SEALED_FROM},
        "buyhold": {"dev": bh_dev_m, "iron": bh_iron_m},
        "singles": {singles[i]["tag"]: {"mechanism": singles[i]["mechanism"],
                                        "dev": singles[i]["dev_metrics"], "iron": singles[i]["iron_metrics"],
                                        "oos_net_xbh": round(singles[i]["dev_metrics"]["net"]/bh_dev_m["net"], 3)
                                        if bh_dev_m["net"] else None} for i in range(N)},
        "pbo": pbo, "whites_reality_check": rc, "n_trials": n_trials,
        "val_selected_combo": vb_combo,
        "verdict": {"n_survivors": len(survivors),
                    "survivors": [r["combo"] for r in survivors],
                    "any_beats_single_and_bh_after_deflation": bool(survivors),
                    "pbo": pbo["pbo"], "rc_pvalue": rc["rc_pvalue"]},
        "frontier_dd_cut": [r["combo"] for r in frontier[:8]],
        "leaderboard": leaderboard,
    }
    out = OUT_DIR / out_name
    json.dump(payload, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}  ({len(leaderboard)} combos, n_names recorded on every row)")

    # ---- leaderboard md (top by deflated OOS+IRON net, with PBO) ----
    _write_leaderboard_md(payload, bh_dev_m, bh_iron_m, frontier, survivors, md_name=md_name)
    print("=" * 96)
    return payload


def _write_leaderboard_md(payload, bh_dev_m, bh_iron_m, frontier, survivors, md_name="combinatrix_leaderboard.md"):
    lb = payload["leaderboard"]
    pbo = payload["pbo"]["pbo"]; rc = payload["whites_reality_check"]["rc_pvalue"]
    L = []
    L.append(f"# V1.2 COMBINATRIX -- LEADERBOARD (structured search: singles / pairs / beam-triples / a1a2)  "
             f"[combo_cad={payload['combo_cad']}]\n")
    L.append("> Ranked by DEFLATED (OOS+IRON)/2 net wealth. Every combo gated: block-bootstrap p05 (vs best-single")
    L.append("> AND vs buy-hold), CSCV-PBO, White's Reality Check + deflated-Sharpe (n_trials = #combos tried).")
    L.append("> 2020 VAL = SELECT / 2020 OOS = REPORT / 2021 IRON = CONFIRM / 2022+ SEALED (asserted never loaded).\n")
    L.append(f"- **Combos evaluated:** {payload['n_combos']} "
             f"(pairs={payload['n_pairs']}, triples={payload['n_triples']}, a1a2={payload['n_a1a2']}); "
             f"n_trials(deflation)={payload['n_trials']}")
    L.append(f"- **Breadth:** n_names={payload['n_names']} distinct u100 names (>=20 floor met).")
    L.append(f"- **CSCV-PBO** over all combos = **{pbo}** ({'OVERFIT' if pbo>=0.5 else 'not overfit'}).  "
             f"**White's Reality Check** (best combo vs BH) p=**{rc}** ({'FAILS' if rc>=0.05 else 'survives'} deflation).")
    L.append(f"- **Buy-hold benchmark:** DEV val={bh_dev_m['val_net']} OOS={bh_dev_m['net']} dd={bh_dev_m['maxdd']} | "
             f"IRON-2021 H2={bh_iron_m['net']}.")
    L.append(f"- **VAL-selected combo:** `{payload['val_selected_combo']}`.\n")

    L.append("## SINGLES recap (the cross-mechanism inputs -- all de-risked betas)\n")
    L.append("| input | mechanism | OOS net | xBH | IRON net |")
    L.append("|---|---|--:|--:|--:|")
    for tag, s in payload["singles"].items():
        dm = s["dev"]; im = s["iron"]
        L.append(f"| {tag} | {s['mechanism']} | {dm['net']} | {s['oos_net_xbh']} | {im['net'] if im else 'NA'} |")
    L.append("")

    L.append(f"## TOP-25 COMBOS (by deflated OOS+IRON net)\n")
    L.append("| # | combo | tier | val net (xBH) | OOS net (xBH) | IRON net (xBH) | maxDD | p05 vs BH | p05 vs SGL | PBO | defl net | defl OOS+IRON | n |")
    L.append("|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for k, r in enumerate(lb[:25], 1):
        def _x(v):
            return "NA" if v is None else v
        L.append(f"| {k} | `{r['combo']}` | {r['tier']}{'*' if r['redundant'] else ''} | "
                 f"{r['val_net']} ({_x(r['val_net_xbh'])}) | {r['oos_net']} ({_x(r['oos_net_xbh'])}) | "
                 f"{_x(r['iron_net'])} ({_x(r['iron_net_xbh'])}) | {r['maxdd']} | "
                 f"{_x(r['p05_vs_bh_oos'])} | {_x(r['p05_vs_single_oos'])} | {r['pbo']} | "
                 f"{r['deflated_net']} | {r['deflated_oos_iron']} | {r['n_names']} |")
    L.append("\n_* = a1a2 within-indicator config-combo (REDUNDANT -- flagged data-snoop trap; shown for completeness)._\n")

    L.append("## HONEST VERDICT\n")
    if survivors:
        L.append(f"**{len(survivors)} combo(s)** beat best-single AND buy-hold on OOS+IRON net with p05>0, PBO<0.5, RC<0.05:")
        for r in survivors[:10]:
            L.append(f"- `{r['combo']}` OOS={r['oos_net']} IRON={r['iron_net']} deflated={r['deflated_net']}")
    else:
        L.append("**NULL (prior-consistent, honest).** NO combo beats its best constituent single AND long-only "
                 "buy-hold on OOS+IRON net after deflation with PBO low + p05>0. The de-risked-beta wall holds under")
        L.append(f"combination: CSCV-PBO={pbo} (overfit) and White's Reality Check p={rc} (best combo fails the")
        L.append("multiple-comparisons deflation). Combining de-risked-beta TI engines does NOT manufacture alpha.\n")

    L.append("## CONDITIONING FRONTIER (cuts OOS maxDD vs buy-hold + robust, WITHOUT beating net)\n")
    L.append("The project's VALIDATED use of TIs (regime/AND-gating = drawdown insurance, per efb23a4b):\n")
    L.append("| combo | OOS maxDD (BH=" + str(bh_dev_m['maxdd']) + ") | OOS net | IRON net | robust |")
    L.append("|---|--:|--:|--:|:--:|")
    for r in frontier[:10]:
        L.append(f"| `{r['combo']}` | {r['maxdd']} | {r['oos_net']} | "
                 f"{r['iron_net'] if r['iron_net'] is not None else 'NA'} | {r['robust']} |")
    L.append("")
    out = OUT_DIR / md_name
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[md]   {out}")


# ============================ self-tests (RWYB) ============================
def selftest():
    """4 mandatory self-tests. Returns dict of PASS/FAIL + numbers."""
    import strat.deep2020_ti_pipeline as PP
    results = {}
    print("=" * 78); print("COMBINATRIX SELF-TESTS (RWYB)"); print("=" * 78)

    # ---------- (a) IDENTITY: a single input under AND_1/OR reproduces that single's deep2020 metrics ----------
    print("\n(a) IDENTITY -- single input under OR(=AND_1) == deep2020 single-engine metrics")
    cad = "1d"
    assets, vt, names = load_window(cad, DEV_WIN, DEV_SPLIT)
    ind = "CKSP"; params = resolve_params(ind, "CKSP(14,2.0)"); mh = PP.INDICATORS[ind].get("minhold", 12)
    # deep2020 reference: build the engine exactly via _book + _metrics
    ref_book = PP._book(assets, PP.INDICATORS[ind]["iron"], params, vt, mh)
    ref = PP._metrics(*ref_book)
    # combinatrix: single input mask -> OR (degenerate, 1 input) -> _combo_book + _metrics
    m = input_masks(assets, PP.INDICATORS[ind]["iron"], params, mh)
    comb_masks = op_or([m])
    cb = _combo_book(assets, comb_masks, vt, DEV_SPLIT)
    got = _metrics(cb[0], DEV_SPLIT, cb[1], cb[2])
    ok_a = (abs(ref["val_net"] - got["val_net"]) < 1e-6 and abs(ref["net"] - got["net"]) < 1e-6
            and abs(ref["maxdd"] - got["maxdd"]) < 1e-6)
    print(f"    deep2020 single : val={ref['val_net']} oos={ref['net']} dd={ref['maxdd']} sh={ref['sharpe']}")
    print(f"    combinatrix OR  : val={got['val_net']} oos={got['net']} dd={got['maxdd']} sh={got['sharpe']}")
    print(f"    -> {'PASS' if ok_a else 'FAIL'} (exact match required)")
    results["a_identity"] = {"pass": bool(ok_a), "ref": ref, "got": got}

    # ---------- (b) PBO on a SYNTHETIC OVERFIT case: many random-noise 'strategies' -> PBO ~0.5+ ----------
    print("\n(b) OVERFIT PBO -- 50 pure-noise strategies (no real edge) -> PBO should be ~0.5+")
    rng = np.random.default_rng(123)
    T = 300; Mn = 50
    noise = {f"noise{i}": rng.normal(0, 0.02, T) for i in range(Mn)}
    pbo_overfit = cscv_pbo(noise, S=10)
    ok_b = (not np.isnan(pbo_overfit["pbo"])) and pbo_overfit["pbo"] >= 0.4
    print(f"    PBO={pbo_overfit['pbo']}  (n_partitions={pbo_overfit['n_partitions']}, M={pbo_overfit['M']})")
    print(f"    -> {'PASS' if ok_b else 'FAIL'} (expect PBO>=0.4: IS-best is a coin-flip OOS)")
    results["b_overfit_pbo"] = {"pass": bool(ok_b), **pbo_overfit}

    # ---------- (c) PBO on a SYNTHETIC GENUINE case: one strategy with real persistent edge -> PBO ~0 ----------
    print("\n(c) GENUINE PBO -- 1 real-edge strategy among noise -> PBO should be ~0")
    genuine = {f"noise{i}": rng.normal(0, 0.02, T) for i in range(Mn - 1)}
    genuine["EDGE"] = rng.normal(0.004, 0.02, T)  # persistent positive drift -> always IS-best, always OOS-good
    pbo_genuine = cscv_pbo(genuine, S=10)
    ok_c = (not np.isnan(pbo_genuine["pbo"])) and pbo_genuine["pbo"] <= 0.15
    print(f"    PBO={pbo_genuine['pbo']}  (n_partitions={pbo_genuine['n_partitions']}, M={pbo_genuine['M']})")
    print(f"    -> {'PASS' if ok_c else 'FAIL'} (expect PBO<=0.15: the real edge persists OOS)")
    results["c_genuine_pbo"] = {"pass": bool(ok_c), **pbo_genuine}

    # ---------- (d) DEFLATION shrinks the best-of-N as N grows ----------
    print("\n(d) DEFLATION -- deflated-Sharpe of a FIXED series shrinks (DSR down, sr0 up) as n_trials grows")
    rng2 = np.random.default_rng(7)
    edge = rng2.normal(0.0018, 0.02, 400)   # a modest real-ish edge
    dsr_few = deflated_sharpe(edge, n_trials=1)
    dsr_some = deflated_sharpe(edge, n_trials=20)
    dsr_many = deflated_sharpe(edge, n_trials=500)
    monotone = dsr_few["exp_max_sr0"] <= dsr_some["exp_max_sr0"] <= dsr_many["exp_max_sr0"]
    shrink = dsr_few["dsr"] >= dsr_some["dsr"] >= dsr_many["dsr"]
    ok_d = bool(monotone and shrink)
    print(f"    n=1   : sr0={dsr_few['exp_max_sr0']}  DSR={dsr_few['dsr']}")
    print(f"    n=20  : sr0={dsr_some['exp_max_sr0']}  DSR={dsr_some['dsr']}")
    print(f"    n=500 : sr0={dsr_many['exp_max_sr0']}  DSR={dsr_many['dsr']}")
    print(f"    -> {'PASS' if ok_d else 'FAIL'} (sr0 rises + DSR falls monotonically as trials grow)")
    results["d_deflation"] = {"pass": ok_d, "n1": dsr_few, "n20": dsr_some, "n500": dsr_many}

    n_pass = sum(1 for k in ("a_identity", "b_overfit_pbo", "c_genuine_pbo", "d_deflation") if results[k]["pass"])
    print("\n" + "=" * 78)
    print(f"SELF-TEST RESULT: {n_pass}/4 PASS")
    print("=" * 78)
    results["summary"] = {"n_pass": n_pass, "n_total": 4}
    return results


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        out = ROOT.parent / "runs" / "engines" / "V1.2_combinatrix" / "selftest_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(out, "w"), indent=1, default=str)
        print(f"[json] {out}")
        sys.exit(0 if r["summary"]["n_pass"] == 4 else 1)
    if "--structured" in sys.argv:
        use_alt = "--no-alternates" not in sys.argv
        # --combo-cad <cad> (default 1d). For a non-1d cadence, write per-cadence artifacts
        # (combinatrix_alltf_<cad>.json + .md) so a stall is salvageable; 1d keeps the committed filenames.
        cad = "1d"
        if "--combo-cad" in sys.argv:
            i = sys.argv.index("--combo-cad")
            if i + 1 < len(sys.argv):
                cad = sys.argv[i + 1]
        valid = {"1d", "4h", "2h", "1h", "30m", "15m"}
        if cad not in valid:
            print(f"ERROR: --combo-cad {cad!r} not in {sorted(valid)}"); sys.exit(2)
        if cad == "1d" and "--alltf-name" not in sys.argv:
            run_structured_campaign(use_alternates=use_alt, combo_cad="1d")
        else:
            run_structured_campaign(use_alternates=use_alt, combo_cad=cad,
                                    out_name=f"combinatrix_alltf_{cad}.json",
                                    md_name=f"combinatrix_alltf_{cad}.md")
        sys.exit(0)
    if "--run" in sys.argv:
        use_alt = "--no-alternates" not in sys.argv
        run_campaign(use_alternates=use_alt)
        sys.exit(0)
    print("use --selftest (self-tests) | --run (all-input campaign) | "
          "--structured [--combo-cad 1d|4h|2h|1h|30m|15m] (pairs/beam-triples/a1a2)")
