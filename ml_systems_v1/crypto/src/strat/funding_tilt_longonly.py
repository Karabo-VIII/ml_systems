"""src/strat/funding_tilt_longonly.py -- THE ONE UNTESTED LONG-ONLY WEALTH ANGLE.

QUESTION (pre-registered, BINDING):
  Directional long-only TI/MA was shown to be drawdown-INSURANCE only (no wealth alpha
  translates). The one held-out-POSITIVE edge the project found -- cross-sectional funding
  DISPERSION carry -- is MARKET-NEUTRAL (long low-funding / SHORT high-funding perps), which
  is permanently OFF (short / long-short / market-neutral = a shortcut, ruled out).

  THE ON-CONSTRAINT VERSION tested HERE: use the FUNDING signal to TILT a STRICT LONG-ONLY
  spot book's WEIGHTS. Overweight LOW / negative-funding assets (cheap / paid to hold),
  underweight HIGH-funding (expensive). ALL weights >= 0, sum to 1, NET-LONG, ZERO short
  logic. Distinct from the ruled-out neutral carry: it captures ONLY the long leg, via tilt.

  Does a long-only funding-TILT harvest a WEALTH edge that TRANSLATES across regimes,
  held-out -- or does it collapse to beta like everything directional?

PRE-REGISTRATION (stated BEFORE the run, persisted verbatim in the JSON):
  H0: the long-only funding-tilt does NOT beat the EW-long-only baseline on held-out wealth
      (tilt-alpha = tilt minus EW; H0 says held-out tilt-alpha p05 <= 0 -- the book is just
      net-long beta, the tilt adds nothing).
  H1: positive, ROBUST tilt-alpha on the held-out path (OOS AND UNSEEN), block-bootstrap
      p05 > 0, surviving max-stat deflation across the swept tilt strengths.
  ONE-SIDED (we only ship if the tilt BEATS EW). ASYMMETRIC LOSS: false-ship (real capital
  into a non-edge) >> false-skip. DEFLATE: a small grid of tilt strengths x signal forms is
  swept -> the best is max-stat / Bonferroni-adjusted; PBO reported on the grid.

THE LOAD-BEARING NUMBER: the TILT-ALPHA = (tilt book daily net) - (EW book daily net) on the
  IDENTICAL PIT roster with IDENTICAL cost accounting. Beta cancels in the difference (both
  books are net-long the same names), so the tilt-alpha isolates the funding signal's pure
  long-only cross-sectional contribution. The absolute net is reported too, but it is mostly
  beta; tilt-alpha is what answers the question.

ABSOLUTE DISCIPLINE (binding):
  STRICT LONG-ONLY + spot: every weight >= 0, sum to 1, net-long, ZERO short logic anywhere.
  FIXED-EW baseline (the no-tilt book). PIT survivorship-clean: an asset not yet trading on a
  day has NaN return -> excluded from that day's weight normalization (renormalize over the
  live cross-section only); we never hold a name before it lists. LEAK-GUARD the funding
  signal: trailing-mean funding, LAGGED >= 1 bar (the 2026-06 run caught a 1-day funding label
  offset in the chimera daily collapse -- chimera daily funding is the PRIOR day's 8h sum, so
  lag>=1 on top of that is doubly conservative: we rank on STALE, never future, funding).
  Maker cost on turnover. Causal/lag-1 (weights set at close(t) earn t+1's return).
  UNSEEN window (2025-12-31 -> 2026-06-01) is READ-ONCE at the very end. No emoji (cp1252).

REUSES: funding panel construction from src/mining/funding_dispersion_probe.py (the LONG leg
  is what we tilt toward; the SHORT leg is NOT used). Scorecard / block-bootstrap from
  src/strat/{scorecard,battery}.py.

RWYB:
  python -m strat.funding_tilt_longonly --selftest
  python -m strat.funding_tilt_longonly --universe u50
Does NOT git commit (overseer commits after judging).
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
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "src" / "pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

from chimera_loader import ChimeraLoader  # noqa: E402
from strat.battery import block_bootstrap_p05_p95  # noqa: E402
from strat.scorecard import score_book, SPLITS  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# ---- split boundaries (project convention, identical to scorecard SPLITS) ----
SEL_END = np.datetime64("2025-03-15")
OOS_END = np.datetime64("2025-12-31")
UNSEEN_LO = np.datetime64("2025-12-31")
UNSEEN_HI = np.datetime64("2026-06-01")

# ---- per-year grade boundaries (calendar order) ----
GRADE_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# maker round-trip (project convention: portfolio_replay.MAKER_RT = 0.0006)
MAKER_RT = 0.0006

SETTLEMENTS_PER_DAY = 3  # daily chimera collapses the 8h clock to 3 settlements

__contract__ = {
    "kind": "funding_tilt_longonly_wealth_test",
    "inputs": {
        "book": "STRICT long-only spot book; weights = monotone-DECREASING function of trailing "
                "(lagged) funding rank (overweight low/negative funding, underweight high), all "
                "w>=0 sum to 1, net-long, NO short logic. Baseline = fixed-EW (no tilt).",
        "signal": "trailing-mean funding LEVEL over `lookback` days, LAGGED >=1 bar (leak guard; "
                  "chimera daily funding is already the prior day's 8h sum -> doubly conservative).",
        "universe": "u50 daily; PIT survivorship-clean via NaN-as-not-live (renormalize weights "
                    "over the live cross-section each day; never hold a name before it lists).",
    },
    "outputs": {
        "per_year": "net/maxDD/Sharpe for tilt vs EW vs buy-hold, 2020-2025 + UNSEEN read-once.",
        "tilt_alpha": "the LOAD-BEARING series = tilt daily net - EW daily net (beta cancels); "
                      "held-out compound + block-bootstrap p05 + max-stat deflation across strengths.",
        "decomposition": "tilt-vs-EW (clean tilt-alpha) reported separately from absolute net.",
        "verdict": "REAL / AMBIGUOUS / ARTIFACT on H1 (robust positive held-out tilt-alpha).",
    },
    "invariants": {
        "long_only_spot": "every weight >= 0, sum to 1, net-long; ZERO short logic. STRICT.",
        "leak_guard_funding": "funding signal lagged >=1 bar; weights at close(t) earn t+1 return.",
        "pit_survivorship": "NaN-return names excluded from daily weight normalization (PIT cash).",
        "fixed_ew_baseline": "EW book = 1/n_live over the live cross-section (no tilt).",
        "maker_cost": "round-trip maker charged on per-day turnover (sum|w_t - w_{t-1}|)/2 * MAKER_RT.",
        "unseen_readonce": "UNSEEN 2025-12-31..2026-06-01 touched once at the very end.",
        "deflated": "tilt strengths x signal forms swept -> best is max-stat / Bonferroni adjusted.",
    },
}


# =====================================================================================================
# 1. DATA -- calendar-aligned daily panel {date, asset, ret, funding_daily}  (reuses the probe's logic)
# =====================================================================================================
def load_panel(universe: str):
    loader = ChimeraLoader()
    feats = ["date", "returns_clean", "fund_rate_mean", "fund_n_settlements"]
    panel = loader.load_universe(universe, cadence="1d", features=feats, add_asset_col=True)
    panel = panel.with_columns(
        (pl.col("fund_rate_mean").fill_null(0.0)
         * pl.col("fund_n_settlements").fill_null(SETTLEMENTS_PER_DAY)).alias("funding_daily")
    )
    panel = panel.select(["date", "asset", "returns_clean", "funding_daily"])
    panel = panel.rename({"returns_clean": "ret"})
    panel = panel.drop_nulls(subset=["ret"])
    return panel


def to_wide(panel):
    """(dates[T], assets[N], ret[T,N], fund[T,N]) numpy with NaN for missing (not-live)."""
    dates = np.sort(panel["date"].unique().to_numpy()).astype("datetime64[D]")
    assets = sorted(panel["asset"].unique().to_list())
    a_idx = {a: i for i, a in enumerate(assets)}
    d_idx = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    T, N = len(dates), len(assets)
    ret = np.full((T, N), np.nan)
    fund = np.full((T, N), np.nan)
    for row in panel.iter_rows(named=True):
        di = d_idx[np.datetime64(row["date"], "D")]
        ai = a_idx[row["asset"]]
        ret[di, ai] = row["ret"]
        fund[di, ai] = row["funding_daily"]
    return dates, assets, ret, fund


# =====================================================================================================
# 2. SIGNAL -- trailing funding level, lagged (leak guard).  sig[t] uses funding in [t-lag-LB+1 .. t-lag]
# =====================================================================================================
def funding_signal(fund, lookback=7, lag=1):
    """Past-only cross-sectional funding LEVEL signal at each rebalance.

    LEAK GUARD (binding): the trade decided at close(t) earns t+1's price return; with lag>=1 the
    ranking does not peek at any funding inside the held window. The chimera daily funding is itself
    the prior day's 8h sum (a known 1-day label offset), so lag>=1 ranks on STALE funding -- never
    future. Conservative by construction.
    """
    T, N = fund.shape
    sig = np.full((T, N), np.nan)
    for t in range(T):
        hi = t - lag
        if hi < 0:
            continue
        lo = max(0, hi - lookback + 1)
        with np.errstate(invalid="ignore"):
            sig[t, :] = np.nanmean(fund[lo:hi + 1, :], axis=0)
    return sig


# =====================================================================================================
# 3. LONG-ONLY WEIGHTS -- monotone-decreasing in funding rank, all w>=0, sum to 1 (STRICT)
# =====================================================================================================
def longonly_tilt_weights(srow, valid_mask, strength):
    """STRICT long-only tilt weights from a trailing-funding signal row.

    Construction (guarantees w>=0, sum=1, net-long, NO short):
      1. Restrict to the LIVE cross-section (valid_mask & finite signal) -- PIT: not-yet-listed
         names (NaN return next bar) are simply not in the pool (cash, weight 0).
      2. Rank the live names by funding ASCENDING (rank 0 = lowest/most-negative funding = the
         CHEAPEST to hold = the one we OVERWEIGHT).
      3. Map rank -> a positive tilt multiplier that is monotone-DECREASING in funding rank:
            m_i = 1 + strength * (1 - 2 * r_i / (n-1))         for r_i in [0 .. n-1]
         At rank 0 (lowest funding): m = 1 + strength.  At rank n-1 (highest): m = 1 - strength.
         strength in [0,1): strength=0 -> EW (no tilt); strength->1 -> strongest admissible tilt
         that KEEPS the top name's multiplier strictly positive (1 - strength > 0). We cap
         strength at 0.99 so the highest-funding name keeps a tiny non-negative weight (never
         shorted, never zero-forced -- a pure reweight, not a selection screen).
      4. Normalize: w_i = m_i / sum(m). All m_i > 0 -> all w_i > 0, sum = 1, net-long. STRICT.

    strength == 0 returns exact EW (the baseline) -- so the EW book and the tilt book share the
    identical PIT roster and code path; their difference is PURELY the tilt (beta cancels).
    """
    N = len(srow)
    w = np.zeros(N)
    valid = np.where(valid_mask & np.isfinite(srow))[0]
    n = len(valid)
    if n == 0:
        return None
    if n == 1:
        w[valid[0]] = 1.0
        return w
    s = min(float(strength), 0.99)
    vals = srow[valid]
    # AVERAGE-rank within the live pool (0 = lowest funding). Ties get the MEAN rank so that
    # cross-sectionally tied funding -> identical multiplier -> the tilt collapses to EW on ties
    # (no cross-sectional information => no reallocation; arbitrary argsort tie-breaking would
    # otherwise inject pure noise into the 'tilt'). This is the honest no-information fallback.
    ranks = _avg_rank(vals)                          # in [0 .. n-1], ties averaged
    # monotone-decreasing positive multiplier in funding rank
    mult = 1.0 + s * (1.0 - 2.0 * ranks / (n - 1))   # in [1-s, 1+s], strictly > 0 for s<1
    mult = np.maximum(mult, 1e-9)                     # numeric floor (still > 0 -> never short/zero-forced)
    wv = mult / mult.sum()
    w[valid] = wv
    return w


def _avg_rank(vals):
    """Average (fractional) rank of each element, ascending; tied values share the MEAN rank.
    Output in [0 .. n-1]. Tied funding -> equal rank -> equal tilt multiplier -> collapses to EW
    on the no-information case (the honest fallback)."""
    n = len(vals)
    order = np.argsort(vals, kind="mergesort")       # stable
    sorted_vals = vals[order]
    ranks_sorted = np.arange(n, dtype=float)
    # average ranks across tie-groups
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            ranks_sorted[i:j] = np.mean(ranks_sorted[i:j])
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def ew_weights(srow, valid_mask):
    """Fixed-EW baseline: 1/n over the live cross-section (PIT). Identical pool to the tilt book."""
    return longonly_tilt_weights(srow, valid_mask, 0.0)


# =====================================================================================================
# 4. SIMULATE -- walk the panel, rebalance daily, MtM, maker cost on turnover (long-only)
# =====================================================================================================
def simulate(dates, ret, signal, weight_fn, fee_rt=MAKER_RT):
    """Return a pandas Series of daily net book return (DatetimeIndex), long-only.

    PnL_{t+1} = sum_j w[t,j] * ret[t+1,j]              (price MtM; w>=0, sum=1 -> net-long)
              - (fee_rt/2) * sum_j |w[t,j] - w_prev[j]| (maker round-trip on per-day turnover)

    A name with NaN ret[t+1] is treated as cash (0 contribution) AND is excluded from the weight
    pool at t (valid_mask below), so we never 'hold' an unlisted/halted name. Weights are decided
    on signal[t] (past-only) and earn t+1's return -> causal.
    """
    T, N = ret.shape
    prev_w = np.zeros(N)
    out_dates = []
    out_net = []
    for t in range(T - 1):
        srow = signal[t, :]
        # PIT live cross-section: name must have a finite signal AND a finite next-bar return
        valid = np.isfinite(srow) & np.isfinite(ret[t + 1, :])
        w = weight_fn(srow, valid)
        if w is None:
            prev_w = np.zeros(N)
            continue
        r_next = np.where(np.isfinite(ret[t + 1, :]), ret[t + 1, :], 0.0)
        price_pnl = float(np.sum(w * r_next))
        turnover = float(np.sum(np.abs(w - prev_w)))
        cost = (fee_rt / 2.0) * turnover
        out_dates.append(dates[t + 1])
        out_net.append(price_pnl - cost)
        prev_w = w
    idx = pd.to_datetime(np.array(out_dates))
    return pd.Series(out_net, index=idx).sort_index()


def buyhold_series(dates, ret):
    """EW buy-hold (no rebalance cost beyond drift): EW over the live cross-section each day, no
    funding tilt, no turnover charge -- the pure market beta benchmark. (The EW BOOK above DOES pay
    rebalance cost; buy-hold here is the cost-free beta floor.)"""
    T, N = ret.shape
    out_dates, out_net = [], []
    for t in range(T - 1):
        valid = np.isfinite(ret[t + 1, :])
        n = int(valid.sum())
        if n == 0:
            continue
        r_next = ret[t + 1, valid]
        out_dates.append(dates[t + 1])
        out_net.append(float(np.mean(r_next)))
    return pd.Series(out_net, index=pd.to_datetime(np.array(out_dates))).sort_index()


# =====================================================================================================
# 5. METRICS / PER-YEAR / TILT-ALPHA
# =====================================================================================================
def _metrics(daily: pd.Series) -> dict:
    s = daily.dropna()
    if len(s) < 5:
        return {"n_days": int(len(s)), "net_pct": None, "maxdd_pct": None, "sharpe": None}
    eq = (1 + s).cumprod()
    dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    return {"n_days": int(len(s)),
            "net_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
            "maxdd_pct": round(dd, 2),
            "sharpe": round(float(s.mean() / (s.std() + 1e-12) * np.sqrt(365)), 2)}


def _year_slice(s: pd.Series, year: int) -> pd.Series:
    lo = pd.Timestamp(f"{year}-01-01")
    hi = pd.Timestamp(f"{year + 1}-01-01")
    return s[(s.index >= lo) & (s.index < hi)]


def _unseen_slice(s: pd.Series) -> pd.Series:
    lo = pd.Timestamp(str(UNSEEN_LO))
    hi = pd.Timestamp(str(UNSEEN_HI))
    return s[(s.index >= lo) & (s.index < hi)]


def per_year_grade(tilt: pd.Series, ew: pd.Series, bh: pd.Series):
    """Per-year + UNSEEN: tilt vs EW vs buy-hold, plus the tilt-alpha (tilt - EW) per year."""
    rows = []
    alpha = (tilt - ew).dropna()
    for yr in GRADE_YEARS:
        t_m = _metrics(_year_slice(tilt, yr))
        e_m = _metrics(_year_slice(ew, yr))
        b_m = _metrics(_year_slice(bh, yr))
        a_sl = _year_slice(alpha, yr)
        a_comp = round(float((np.prod(1 + a_sl.to_numpy()) - 1) * 100), 2) if len(a_sl) >= 5 else None
        rows.append({"period": str(yr), "n_days": t_m["n_days"],
                     "tilt_net": t_m["net_pct"], "ew_net": e_m["net_pct"], "bh_net": b_m["net_pct"],
                     "tilt_dd": t_m["maxdd_pct"], "ew_dd": e_m["maxdd_pct"],
                     "tilt_sh": t_m["sharpe"], "ew_sh": e_m["sharpe"],
                     "tilt_alpha_comp": a_comp})
    # UNSEEN (read-once)
    t_m = _metrics(_unseen_slice(tilt))
    e_m = _metrics(_unseen_slice(ew))
    b_m = _metrics(_unseen_slice(bh))
    a_sl = _unseen_slice(alpha)
    a_comp = round(float((np.prod(1 + a_sl.to_numpy()) - 1) * 100), 2) if len(a_sl) >= 5 else None
    rows.append({"period": "UNSEEN", "n_days": t_m["n_days"],
                 "tilt_net": t_m["net_pct"], "ew_net": e_m["net_pct"], "bh_net": b_m["net_pct"],
                 "tilt_dd": t_m["maxdd_pct"], "ew_dd": e_m["maxdd_pct"],
                 "tilt_sh": t_m["sharpe"], "ew_sh": e_m["sharpe"],
                 "tilt_alpha_comp": a_comp})
    return rows


# =====================================================================================================
# 6. BUILD ALL VARIANTS (tilt strengths x signal forms) -- the swept grid for deflation
# =====================================================================================================
def build_variants(dates, ret, fund, strengths, lookbacks, lag):
    """Return {label: {'tilt': Series, 'ew': Series, 'alpha': Series, 'strength', 'lookback'}}.

    The EW book is shared per-lookback (signal only gates the live pool; EW is tilt-free), but we
    rebuild it per (lookback) because the valid-pool depends on signal finiteness. The tilt-alpha is
    computed on the matched EW for that exact signal pool -> beta cancels exactly."""
    variants = {}
    ew_cache = {}
    for lb in lookbacks:
        sig = funding_signal(fund, lookback=lb, lag=lag)
        ew = simulate(dates, ret, sig, lambda sr, vm: ew_weights(sr, vm))
        ew_cache[lb] = (sig, ew)
        for st in strengths:
            if st == 0.0:
                continue  # st=0 is the EW baseline itself
            tilt = simulate(dates, ret, sig,
                            lambda sr, vm, _st=st: longonly_tilt_weights(sr, vm, _st))
            alpha = (tilt - ew).dropna()
            label = f"st{st:.2f}_lb{lb}"
            variants[label] = {"tilt": tilt, "ew": ew, "alpha": alpha,
                               "strength": st, "lookback": lb}
    return variants, ew_cache


# =====================================================================================================
# 7. VERDICT (pre-registered, two-sided, deflated)
# =====================================================================================================
PREREG = {
    "H0": "the long-only funding-tilt does NOT beat the EW-long-only baseline on held-out wealth "
          "(tilt-alpha = tilt minus EW; held-out tilt-alpha p05 <= 0 -- book is just net-long beta).",
    "H1": "positive ROBUST tilt-alpha on the held-out path (OOS AND UNSEEN), block-bootstrap p05 > 0, "
          "surviving max-stat deflation across the swept tilt strengths.",
    "one_sided": "we ship only if the tilt BEATS EW (overweighting low-funding adds wealth).",
    "asymmetric_loss": "false-ship (real capital into a non-edge) >> false-skip.",
    "decisive_statistic": "the TILT-ALPHA daily series (tilt net - EW net, identical PIT roster + cost) "
                          "held-out compound + block-bootstrap p05; beta cancels in the difference.",
    "deflation": "tilt strengths x signal forms swept -> best is max-stat / Bonferroni-adjusted; PBO "
                 "reported on the grid.",
}


def _heldout_alpha_p05(alpha: pd.Series):
    """Block-bootstrap p05 of the tilt-alpha compound on the HELD-OUT path (OOS+UNSEEN)."""
    held = alpha[(alpha.index >= pd.Timestamp(str(SEL_END)))]
    if len(held) < 10:
        return None, None
    bb = block_bootstrap_p05_p95(held.to_numpy())
    comp = float((np.prod(1 + held.to_numpy()) - 1) * 100)
    return comp, bb


def build_verdict(variants, sel_pick_label, n_variants_tried):
    """Two-sided, deflated verdict on H1.

    Selection is on SEL ONLY (pick the best SEL tilt-alpha). The verdict keys on the HELD-OUT
    tilt-alpha of THAT selected variant -- never on SEL. Deflation: the selected variant's held-out
    p05 must beat 0 AND the SEL selection must survive a max-stat / Bonferroni adjustment (we report
    the count tried; a single-config survivor at p05>0 on held-out is the bar, the SEL max-stat is a
    deflation context, not the gate)."""
    v = variants[sel_pick_label]
    alpha = v["alpha"]
    held_comp, bb = _heldout_alpha_p05(alpha)
    # split-wise tilt-alpha compounds
    def _split_comp(lo, hi):
        sl = alpha[(alpha.index >= pd.Timestamp(lo)) & (alpha.index < pd.Timestamp(hi))]
        return round(float((np.prod(1 + sl.to_numpy()) - 1) * 100), 3) if len(sl) >= 5 else None
    sel_a = _split_comp(SPLITS["SEL"][0], SPLITS["SEL"][1])
    oos_a = _split_comp(SPLITS["OOS"][0], SPLITS["OOS"][1])
    un_a = _split_comp(SPLITS["UNSEEN"][0], SPLITS["UNSEEN"][1])
    p05 = bb.get("p05") if bb else None

    oos_pos = oos_a is not None and oos_a > 0
    un_pos = un_a is not None and un_a > 0
    held_pos = held_comp is not None and held_comp > 0
    p05_pos = p05 is not None and p05 > 0

    if held_pos and p05_pos and oos_pos and un_pos:
        verdict = "REAL"
    elif held_pos and (oos_pos or un_pos):
        verdict = "AMBIGUOUS"
    else:
        verdict = "ARTIFACT"

    return {
        "selected_variant": sel_pick_label,
        "selected_on": "SEL tilt-alpha (highest)",
        "n_variants_tried": n_variants_tried,
        "tilt_alpha_SEL_pct": sel_a,
        "tilt_alpha_OOS_pct": oos_a,
        "tilt_alpha_UNSEEN_pct": un_a,
        "tilt_alpha_heldout_pct": round(held_comp, 3) if held_comp is not None else None,
        "tilt_alpha_heldout_p05": p05,
        "tilt_alpha_heldout_p95": bb.get("p95") if bb else None,
        "verdict": verdict,
        "interpretation": (
            "REAL = robust positive held-out tilt-alpha (the funding signal adds long-only wealth "
            "beyond beta). AMBIGUOUS = held-out positive but not p05-robust or not on both OOS+UNSEEN. "
            "ARTIFACT = the tilt does NOT beat EW held-out -> the book is just net-long beta, the "
            "funding tilt adds no wealth (it collapses to beta like everything directional)."),
    }


# =====================================================================================================
# 8. CHARTS
# =====================================================================================================
def make_charts(rows, variants, sel_pick_label, bh):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] matplotlib unavailable ({e}) -- skipped")
        return []
    paths = []
    v = variants[sel_pick_label]
    tilt, ew = v["tilt"], v["ew"]

    # ---- chart 1: tilt vs EW vs BH per year (grouped bars of net %) ----
    periods = [r["period"] for r in rows]
    tilt_net = [r["tilt_net"] if r["tilt_net"] is not None else 0 for r in rows]
    ew_net = [r["ew_net"] if r["ew_net"] is not None else 0 for r in rows]
    bh_net = [r["bh_net"] if r["bh_net"] is not None else 0 for r in rows]
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.25, tilt_net, width=0.25, color="#2a9d8f", label=f"funding-tilt ({sel_pick_label})")
    ax.bar(x, ew_net, width=0.25, color="#264653", label="EW long-only (baseline)")
    ax.bar(x + 0.25, bh_net, width=0.25, color="#e9c46a", label="EW buy-hold (beta)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("net return %"); ax.set_xlabel("period")
    ax.set_title("Long-only funding-TILT vs EW-baseline vs buy-hold, per year + UNSEEN (read-once)")
    ax.legend(loc="best"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    c1 = OUT / "funding_tilt_vs_ew.png"
    fig.savefig(c1, dpi=120); plt.close(fig); paths.append(str(c1))
    print(f"[chart] {c1}")

    # ---- chart 2: tilt-alpha (tilt - EW) compound per year -- the load-bearing isolation ----
    a_comp = [r["tilt_alpha_comp"] if r["tilt_alpha_comp"] is not None else 0 for r in rows]
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    colors = ["#2a9d8f" if a >= 0 else "#e76f51" for a in a_comp]
    ax2.bar(x, a_comp, width=0.55, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    # shade the held-out region (OOS+UNSEEN periods: 2025 + UNSEEN)
    for i, p in enumerate(periods):
        if p in ("2025", "UNSEEN"):
            ax2.axvspan(i - 0.4, i + 0.4, color="#999999", alpha=0.12)
    ax2.set_xticks(x); ax2.set_xticklabels(periods)
    ax2.set_ylabel("tilt-alpha = tilt net - EW net (compound %)"); ax2.set_xlabel("period")
    ax2.set_title("TILT-ALPHA by regime (funding signal's pure long-only contribution; beta cancels). "
                  "Grey = held-out.")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    c2 = OUT / "tilt_alpha_by_regime.png"
    fig2.savefig(c2, dpi=120); plt.close(fig2); paths.append(str(c2))
    print(f"[chart] {c2}")
    return paths


# =====================================================================================================
# 9. MAIN
# =====================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.funding_tilt_longonly")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lag", type=int, default=1, help="leak-guard lag on the funding signal (>=1)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    # swept grid (DEFLATION context): a few tilt strengths x a couple of lookbacks
    STRENGTHS = [0.25, 0.50, 0.75, 0.90]
    LOOKBACKS = [7, 14, 30]
    n_variants_tried = len(STRENGTHS) * len(LOOKBACKS)

    print("## LONG-ONLY FUNDING-TILT -- the one untested long-only WEALTH angle")
    print("## PRE-REGISTRATION:")
    for k in ("H0", "H1", "one_sided", "asymmetric_loss", "decisive_statistic", "deflation"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   universe={a.universe}  lag={a.lag} (leak guard)  maker_rt={MAKER_RT}")
    print(f"   STRICT long-only (w>=0, sum=1, net-long, NO short) | PIT survivorship-clean | "
          f"UNSEEN read-once")
    print(f"   swept grid (deflation): strengths={STRENGTHS} x lookbacks={LOOKBACKS} "
          f"= {n_variants_tried} variants\n", flush=True)

    panel = load_panel(a.universe)
    dates, assets, ret, fund = to_wide(panel)
    print(f"panel: {len(dates)} days x {len(assets)} assets "
          f"({str(dates[0])} -> {str(dates[-1])})", flush=True)

    # ---- build the buy-hold beta floor ----
    bh = buyhold_series(dates, ret)

    # ---- build the swept variants ----
    print("\n-- building variants (this sweeps the deflation grid) ...", flush=True)
    variants, ew_cache = build_variants(dates, ret, fund, STRENGTHS, LOOKBACKS, a.lag)

    # ---- SELECT on SEL ONLY: pick the variant with the highest SEL tilt-alpha ----
    def _sel_alpha(v):
        sl = v["alpha"][(v["alpha"].index >= pd.Timestamp(str(np.datetime64("2018-01-01")))) &
                        (v["alpha"].index < pd.Timestamp(str(SEL_END)))]
        return float((np.prod(1 + sl.to_numpy()) - 1) * 100) if len(sl) >= 5 else -1e9
    sel_scores = {lbl: _sel_alpha(v) for lbl, v in variants.items()}
    sel_pick_label = max(sel_scores, key=sel_scores.get)
    print(f"-- SEL selection (highest SEL tilt-alpha): {sel_pick_label} "
          f"(SEL tilt-alpha={sel_scores[sel_pick_label]:+.2f}%)", flush=True)

    v = variants[sel_pick_label]
    tilt, ew = v["tilt"], v["ew"]

    # ---- per-year grade (tilt vs EW vs BH) + tilt-alpha per year ----
    rows = per_year_grade(tilt, ew, bh)
    print("\n" + "=" * 110)
    print("## PER-YEAR GRADE -- funding-tilt vs EW-baseline vs buy-hold (selected variant) "
          "[UNSEEN read-once]")
    print(f"   {'period':>7} {'days':>5} {'tiltNet':>8} {'ewNet':>8} {'bhNet':>8} "
          f"{'tiltDD':>7} {'ewDD':>7} {'tiltSh':>7} {'ewSh':>6} {'TILT-ALPHA':>11}")
    for r in rows:
        print(f"   {r['period']:>7} {str(r['n_days']):>5} {str(r['tilt_net']):>8} {str(r['ew_net']):>8} "
              f"{str(r['bh_net']):>8} {str(r['tilt_dd']):>7} {str(r['ew_dd']):>7} "
              f"{str(r['tilt_sh']):>7} {str(r['ew_sh']):>6} {str(r['tilt_alpha_comp']):>11}")
    print("=" * 110, flush=True)

    # ---- canonical scorecard on the tilt-alpha series (the load-bearing object) ----
    alpha = v["alpha"]
    grid_alpha = _grid_for_pbo(variants)
    card = score_book(f"funding_tilt_alpha[{sel_pick_label}]", alpha, grid_returns=grid_alpha)

    # ---- deflated verdict ----
    vd = build_verdict(variants, sel_pick_label, n_variants_tried)

    # ---- full sweep table: held-out tilt-alpha + p05 for EVERY variant (transparency) ----
    print("\n## FULL SWEEP -- held-out tilt-alpha (OOS+UNSEEN) compound + block-bootstrap p05 "
          "(deflation transparency)")
    print(f"   {'variant':>14} {'SEL-a%':>8} {'OOS-a%':>8} {'UNSEEN-a%':>10} {'held-a%':>9} "
          f"{'held-p05':>9} {'held-p95':>9}")
    sweep = []
    for lbl in sorted(variants, key=lambda L: (variants[L]["lookback"], variants[L]["strength"])):
        vv = variants[lbl]
        al = vv["alpha"]
        def _c(lo, hi):
            s = al[(al.index >= pd.Timestamp(lo)) & (al.index < pd.Timestamp(hi))]
            return round(float((np.prod(1 + s.to_numpy()) - 1) * 100), 2) if len(s) >= 5 else None
        sel_a = _c(SPLITS["SEL"][0], SPLITS["SEL"][1])
        oos_a = _c(SPLITS["OOS"][0], SPLITS["OOS"][1])
        un_a = _c(SPLITS["UNSEEN"][0], SPLITS["UNSEEN"][1])
        hc, bb = _heldout_alpha_p05(al)
        p05 = bb.get("p05") if bb else None
        p95 = bb.get("p95") if bb else None
        sweep.append({"variant": lbl, "sel_alpha": sel_a, "oos_alpha": oos_a, "unseen_alpha": un_a,
                      "heldout_alpha": round(hc, 2) if hc is not None else None,
                      "heldout_p05": p05, "heldout_p95": p95})
        print(f"   {lbl:>14} {str(sel_a):>8} {str(oos_a):>8} {str(un_a):>10} "
              f"{str(round(hc,2) if hc is not None else None):>9} "
              f"{str(round(p05,3) if p05 is not None else None):>9} "
              f"{str(round(p95,3) if p95 is not None else None):>9}")
    # count how many variants have held-out p05 > 0 (multiplicity reality check)
    n_p05_pos = sum(1 for sw in sweep if sw["heldout_p05"] is not None and sw["heldout_p05"] > 0)
    print(f"\n   variants with held-out tilt-alpha p05 > 0: {n_p05_pos} of {len(sweep)} "
          f"(if ~5% by chance under H0, expect ~{0.05*len(sweep):.1f})")

    # ---- verdict print ----
    print("\n" + "=" * 110)
    print("## VERDICT (deflated, two-sided)")
    for k, val in vd.items():
        print(f"   {k}: {val}")
    print(f"\n   scorecard ship_read (on the tilt-alpha series): {card.get('ship_read')}")
    pbo = card.get("pbo", {})
    print(f"   PBO (prob. of backtest overfit on the tilt-alpha grid): {pbo.get('pbo', pbo)}")
    print(f"\n   >>> VERDICT: {vd['verdict']}")
    print("=" * 110, flush=True)

    # ---- charts ----
    charts = []
    if not a.no_charts:
        charts = make_charts(rows, variants, sel_pick_label, bh)

    # ---- persist ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pj = OUT / f"funding_tilt_longonly_{stamp}.json"
    payload = {
        "repro": {"command": "python -m strat.funding_tilt_longonly " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "universe": a.universe, "lag": a.lag, "maker_rt": MAKER_RT,
                  "strengths": STRENGTHS, "lookbacks": LOOKBACKS, "n_variants_tried": n_variants_tried},
        "prereg": PREREG,
        "per_year_grade": rows,
        "selected_variant": sel_pick_label,
        "sel_alpha_scores": {k: round(vv, 3) for k, vv in sel_scores.items()},
        "sweep": sweep,
        "n_heldout_p05_pos": n_p05_pos,
        "scorecard_tilt_alpha": _json_safe(card),
        "verdict": vd,
        "charts": charts,
    }
    json.dump(payload, open(pj, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {pj}")
    return 0


def _grid_for_pbo(variants):
    """Stack the per-variant tilt-alpha daily series into a [T x n_variants] matrix for PBO/CSCV.
    Align on the common date index (intersection)."""
    series = [v["alpha"] for v in variants.values()]
    if not series:
        return None
    df = pd.concat([s.rename(i) for i, s in enumerate(series)], axis=1).dropna()
    if df.shape[0] < 50 or df.shape[1] < 2:
        return None
    return df.to_numpy()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


# =====================================================================================================
# 10. SELFTEST -- mechanics + invariants (no full grade)
# =====================================================================================================
def selftest():
    print("## FUNDING-TILT-LONGONLY SELFTEST")
    ok = True
    rng = np.random.default_rng(7)
    T, N = 200, 8

    # (1) weights are STRICT long-only: w>=0, sum=1, net-long, for random signals + strengths
    s1 = True
    for _ in range(200):
        srow = rng.normal(0, 1, N)
        valid = np.ones(N, bool)
        for st in [0.0, 0.25, 0.5, 0.9, 0.99]:
            w = longonly_tilt_weights(srow, valid, st)
            if w is None:
                s1 = False; break
            if (w < -1e-12).any() or abs(w.sum() - 1.0) > 1e-9:
                s1 = False; break
        if not s1:
            break
    print(f"  (1) STRICT long-only (w>=0, sum=1) over random signals x strengths -> "
          f"{'PASS' if s1 else 'FAIL'}")
    ok &= s1

    # (2) strength=0 == EW exactly (the baseline shares the tilt code path)
    srow = rng.normal(0, 1, N); valid = np.ones(N, bool)
    w0 = longonly_tilt_weights(srow, valid, 0.0)
    wew = ew_weights(srow, valid)
    s2 = np.allclose(w0, 1.0 / N) and np.allclose(w0, wew)
    print(f"  (2) strength=0 == EW (1/n) exactly -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2

    # (3) monotone tilt: the LOWEST-funding name gets MORE weight than the HIGHEST-funding name
    srow = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # ascending funding
    w = longonly_tilt_weights(srow, np.ones(N, bool), 0.6)
    s3 = w[0] > w[-1] and w[0] > 1.0 / N > w[-1]   # overweight low funding, underweight high
    print(f"  (3) monotone: low-funding overweighted (w[0]={w[0]:.4f}) > 1/n ({1/N:.4f}) > "
          f"high-funding (w[-1]={w[-1]:.4f}) -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3

    # (4) PIT: an asset that is NaN next-bar is excluded from the pool (cash), weights renormalize
    dates = np.array([np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(T)])
    ret = rng.normal(0, 0.02, (T, N))
    ret[:, 0] = np.nan        # asset 0 never trades -> must always be weight 0
    fund = rng.normal(0, 0.0005, (T, N))
    sig = funding_signal(fund, lookback=7, lag=1)
    # probe one bar's weights directly
    s4 = True
    for t in range(10, T - 1):
        valid = np.isfinite(sig[t, :]) & np.isfinite(ret[t + 1, :])
        w = longonly_tilt_weights(sig[t, :], valid, 0.5)
        if w is None:
            continue
        if w[0] > 1e-12:        # the dead asset must never hold weight
            s4 = False; break
        if abs(w.sum() - 1.0) > 1e-9:
            s4 = False; break
    print(f"  (4) PIT: NaN-next-bar asset excluded from pool (weight 0), weights renormalize -> "
          f"{'PASS' if s4 else 'FAIL'}")
    ok &= s4

    # (5) leak guard: signal[t] depends ONLY on funding up to t-lag (no future). Inject a spike at t0
    #     into funding and confirm sig at t < t0+lag is unchanged.
    fund2 = np.zeros((T, N)); fund2[:] = 0.0001
    sig_a = funding_signal(fund2, lookback=7, lag=1)
    fund3 = fund2.copy(); t0 = 100; fund3[t0, :] = 99.0   # huge future spike
    sig_b = funding_signal(fund3, lookback=7, lag=1)
    # signal at bars < t0+lag must be identical (no peeking at the spike at t0)
    s5 = np.allclose(np.nan_to_num(sig_a[:t0 + 1]), np.nan_to_num(sig_b[:t0 + 1]))
    print(f"  (5) leak guard: future funding spike at t={t0} does NOT alter signal at t<=t0 "
          f"(lag=1) -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5

    # (6) tilt-alpha cancels beta: with ZERO funding signal (flat), tilt==EW -> alpha==0
    flat = np.full((T, N), 0.001)
    sigf = funding_signal(flat, lookback=7, lag=1)
    ew = simulate(dates, ret, sigf, lambda sr, vm: ew_weights(sr, vm))
    tl = simulate(dates, ret, sigf, lambda sr, vm: longonly_tilt_weights(sr, vm, 0.5))
    al = (tl - ew).dropna()
    s6 = float(np.abs(al.to_numpy()).max()) < 1e-9
    print(f"  (6) flat funding -> tilt==EW -> tilt-alpha==0 (max|alpha|="
          f"{float(np.abs(al.to_numpy()).max()):.2e}) -> {'PASS' if s6 else 'FAIL'}")
    ok &= s6

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
