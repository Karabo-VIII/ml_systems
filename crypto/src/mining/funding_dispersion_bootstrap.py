"""funding_dispersion_bootstrap.py -- Block-bootstrap deflation + dispersion-gate
for the 3c DEPLOYABLE funding-dispersion carry series.

TASK (Wave 2B statistical gate):
  (1) Re-derive the 3c daily return series from scratch (no trust of reported figures).
  (2) Run a CIRCULAR block-bootstrap on compound return AND Sharpe, sweeping block
      lengths 5,10,15,20 trading days. Report p05/p50/p95 per window (SEL/OOS/UNSEEN).
      Verdict: does p05 > 0 survive deflation?
  (3) Dispersion-gate hypothesis: deploy only when cross-sectional funding std >
      TRAIN-FIT threshold. Threshold pre-registered on SEL. Report OOS/UNSEEN rescue.

INFERENCE DESIGN (pre-registered before seeing results):
  - Null: H0 = strategy compound return <= 0 (one-sided upper-tail test).
  - Alt:  H1 = strategy compound return > 0.
  - Test stat: compound return (primary), Sharpe (secondary / informational).
  - Block bootstrap: CIRCULAR (Politis & Romano 1994) to preserve autocorrelation.
    Block length swept: 5,10,15,20 days. Conservative = longest block length.
  - n_bootstrap = 2000 per window per block length.
  - p-value = fraction of bootstrap samples where compound return <= 0.
    (Conservative: we test whether p05 > 0, not just whether mean > 0.)
  - n_eff estimate: T / block_length (lower bound; AC structure tightens further).
  - Asymmetric loss: false ship >> false skip. Use p05 (5th percentile of bootstrap
    distribution) as the go/no-go gate. If p05 > 0, edge survives deflation.
  - Multiple comparisons: NONE here -- single pre-registered series (3c). The probe
    selected k on SEL Sharpe; that selection is the comparison; we report DSR-style
    deflation via the block-bootstrap itself.
  - Dispersion gate: ONE threshold, fit on SEL (median cross-sectional funding std),
    zero additional comparisons. Applied to OOS/UNSEEN intact.

CRITICAL: UNSEEN is touched ONCE at the end. All threshold selection is SEL-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

from chimera_loader import ChimeraLoader  # noqa: E402

# ---- split boundaries (project convention) -----------------------------------
SEL_END = np.datetime64("2025-03-15")
OOS_END = np.datetime64("2025-12-31")
UNSEEN_END = np.datetime64("2026-05-28")

# ---- 3c DEPLOYABLE parameters (from funding_dispersion_frictions.py) ---------
LOOKBACK = 7
LAG = 1
K = 5
SETTLEMENTS_PER_DAY = 3
TAKER_BASE_BPS = 4.5
SLIP_LIQUID_BPS = 2.0
SLIP_MID_BPS = 8.0
SLIP_ILLIQUID_BPS = 25.0
SHORT_ILLIQUID_BORROW_BPS_PER_8H = 1.0
LIQ_TOP_FRACTION = 0.40
HYST = 2.0

# ---- bootstrap parameters ---------------------------------------------------
N_BOOTSTRAP = 2000
BLOCK_LENGTHS = [5, 10, 15, 20]
RNG_SEED = 42


def _flush():
    sys.stdout.flush()


# =============================================================================
# DATA LOADING (re-derive from scratch -- do not trust reported figures)
# =============================================================================

def load_daily_panel(universe: str):
    loader = ChimeraLoader()
    feats = ["date", "returns_clean", "fund_rate_mean", "fund_n_settlements",
             "volume_usd", "s3_oi_usd"]
    panel = loader.load_universe(universe, cadence="1d", features=feats,
                                 add_asset_col=True)
    panel = panel.with_columns(
        (pl.col("fund_rate_mean").fill_null(0.0)
         * pl.col("fund_n_settlements").fill_null(SETTLEMENTS_PER_DAY))
        .alias("funding_daily")
    )
    panel = panel.select(["date", "asset", "returns_clean", "funding_daily",
                          "volume_usd", "s3_oi_usd"])
    panel = panel.rename({"returns_clean": "ret"})
    panel = panel.drop_nulls(subset=["ret"])
    return panel


def to_wide(panel: pl.DataFrame):
    dates = np.sort(panel["date"].unique().to_numpy()).astype("datetime64[D]")
    assets = sorted(panel["asset"].unique().to_list())
    a_idx = {a: i for i, a in enumerate(assets)}
    d_idx = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    T, N = len(dates), len(assets)
    ret = np.full((T, N), np.nan)
    fund = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    for row in panel.iter_rows(named=True):
        di = d_idx[np.datetime64(row["date"], "D")]
        ai = a_idx[row["asset"]]
        ret[di, ai] = row["ret"]
        fund[di, ai] = row["funding_daily"]
        v = row["volume_usd"]
        vol[di, ai] = v if v is not None else np.nan
    return dates, assets, ret, fund, vol


def load_funding_8h(assets, dates):
    T, N = len(dates), len(assets)
    fund_8h = np.full((T, N), np.nan)
    d_idx = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    slot_of = {0: 0, 8: 1, 16: 2}
    for ai, sym in enumerate(assets):
        d = PROJECT_ROOT / "data" / "raw" / sym / "funding"
        if not d.exists():
            continue
        try:
            df = (pl.read_parquet(str(d / "*.parquet"),
                                  columns=["timestamp", "funding_rate"])
                  .drop_nulls()
                  .unique(subset=["timestamp"]))
        except Exception:
            continue
        if df.height == 0:
            continue
        ts = df["timestamp"].to_numpy().astype("int64")
        fr = df["funding_rate"].to_numpy()
        dt = ts.astype("datetime64[ms]")
        day = dt.astype("datetime64[D]")
        hour = ((ts // 3600000) % 24).astype("int64")
        for day_d, hr, rate in zip(day, hour, fr):
            di = d_idx.get(np.datetime64(day_d, "D"))
            if di is None:
                continue
            v = fund_8h[di, ai]
            fund_8h[di, ai] = (0.0 if np.isnan(v) else v) + float(rate)
    return fund_8h


def funding_signal(fund, lookback=7, lag=1):
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


def liquidity_rank_pct(vol_row, valid_mask):
    N = len(vol_row)
    pct = np.full(N, np.nan)
    vi = np.where(valid_mask & ~np.isnan(vol_row))[0]
    if len(vi) < 2:
        return pct
    order = vi[np.argsort(vol_row[vi])]
    for r, idx in enumerate(order):
        pct[idx] = r / (len(order) - 1)
    return pct


def per_name_cost_bps(liq_pct_value):
    if np.isnan(liq_pct_value):
        slip = SLIP_MID_BPS
    elif liq_pct_value >= 0.75:
        slip = SLIP_LIQUID_BPS
    elif liq_pct_value >= 0.25:
        slip = SLIP_MID_BPS
    else:
        slip = SLIP_ILLIQUID_BPS
    return (TAKER_BASE_BPS + slip) * 1e-4


def build_weights(srow, valid_mask, liq_pct, short_floor, long_floor):
    N = len(srow)
    w = np.zeros(N)
    base_valid = valid_mask & ~np.isnan(srow)
    long_ok = base_valid.copy()
    short_ok = base_valid.copy()
    if long_floor is not None:
        long_ok = long_ok & ~np.isnan(liq_pct) & (liq_pct >= long_floor)
    if short_floor is not None:
        short_ok = short_ok & ~np.isnan(liq_pct) & (liq_pct >= short_floor)
    long_idx = np.where(long_ok)[0]
    short_idx = np.where(short_ok)[0]
    if len(long_idx) < K or len(short_idx) < K:
        return None
    longs = long_idx[np.argsort(srow[long_idx])][:K]
    shorts = short_idx[np.argsort(srow[short_idx])][-K:]
    if set(longs) & set(shorts):
        return None
    w[longs] = 0.5 / K
    w[shorts] = -0.5 / K
    return w


def apply_hysteresis(w_new, prev_w, srow, valid, liq_pct, band, short_floor):
    if band <= 1.0 or w_new is None:
        return w_new
    base_valid = valid & ~np.isnan(srow)
    long_ok = base_valid.copy()
    short_ok = base_valid.copy()
    if short_floor is not None:
        short_ok = short_ok & ~np.isnan(liq_pct) & (liq_pct >= short_floor)
    long_idx = np.where(long_ok)[0]
    short_idx = np.where(short_ok)[0]
    wide = int(np.ceil(band * K))
    if len(long_idx) < K or len(short_idx) < K:
        return w_new
    long_sorted = long_idx[np.argsort(srow[long_idx])]
    short_sorted = short_idx[np.argsort(srow[short_idx])]
    long_wide = set(long_sorted[:wide].tolist())
    short_wide = set(short_sorted[-wide:].tolist())
    held_long = [j for j in np.where(prev_w > 0)[0] if j in long_wide]
    held_short = [j for j in np.where(prev_w < 0)[0] if j in short_wide]
    longs = list(held_long)
    for j in long_sorted:
        if len(longs) >= K:
            break
        if j not in longs:
            longs.append(int(j))
    longs = longs[:K]
    shorts = list(held_short)
    for j in short_sorted[::-1]:
        if len(shorts) >= K:
            break
        if j not in shorts:
            shorts.append(int(j))
    shorts = shorts[:K]
    if len(longs) < K or len(shorts) < K or (set(longs) & set(shorts)):
        return w_new
    N = len(w_new)
    w = np.zeros(N)
    w[longs] = 0.5 / K
    w[shorts] = -0.5 / K
    return w


def simulate_3c(dates, ret, fund_8h, vol, signal):
    """Reproduce the 3c DEPLOYABLE simulation exactly.
    Returns (dates_of_returns, daily_returns, daily_fund_pnl, cs_fund_std).
    cs_fund_std[t] = cross-sectional std of funding signal at t (for dispersion gate).
    """
    short_floor = 1.0 - LIQ_TOP_FRACTION
    T, N = ret.shape
    prev_w = np.zeros(N)
    out_dates = []
    out_returns = []
    out_fund_pnl = []
    out_cs_std = []   # cross-sectional std of funding signal (for gate)

    for t in range(T - 1):
        srow = signal[t, :]
        valid = (~np.isnan(srow) & ~np.isnan(ret[t + 1, :])
                 & ~np.isnan(fund_8h[t + 1, :]))
        liq_pct = liquidity_rank_pct(vol[t, :], valid)

        w = build_weights(srow, valid, liq_pct, short_floor=short_floor, long_floor=None)
        if w is None:
            prev_w = np.zeros(N)
            out_dates.append(dates[t + 1])
            out_returns.append(0.0)
            out_fund_pnl.append(0.0)
            # still compute cs_std even when no trade
            vi = np.where(valid & ~np.isnan(srow))[0]
            cs_std = float(np.nanstd(srow[vi])) if len(vi) > 1 else np.nan
            out_cs_std.append(cs_std)
            continue

        w = apply_hysteresis(w, prev_w, srow, valid, liq_pct, HYST, short_floor)

        r_next = np.where(np.isnan(ret[t + 1, :]), 0.0, ret[t + 1, :])
        f_next = np.where(np.isnan(fund_8h[t + 1, :]), 0.0, fund_8h[t + 1, :])

        price_pnl = float(np.sum(w * r_next))
        funding_pnl = float(-np.sum(w * f_next))

        # borrow drag on illiquid shorts
        borrow_drag = 0.0
        liq_next = liquidity_rank_pct(vol[t + 1, :], ~np.isnan(vol[t + 1, :]))
        for j in np.where(w < 0)[0]:
            lp = liq_next[j]
            if np.isnan(lp) or lp < 0.25:
                borrow_drag += (abs(w[j]) * SHORT_ILLIQUID_BORROW_BPS_PER_8H
                                * 1e-4 * SETTLEMENTS_PER_DAY)

        # tiered taker cost
        dw = w - prev_w
        liq_now = liquidity_rank_pct(vol[t, :], ~np.isnan(vol[t, :]))
        cost = 0.0
        for j in np.where(np.abs(dw) > 1e-12)[0]:
            cost += abs(dw[j]) * per_name_cost_bps(liq_now[j])
        cost += borrow_drag

        total = price_pnl + funding_pnl - cost

        out_dates.append(dates[t + 1])
        out_returns.append(total)
        out_fund_pnl.append(funding_pnl)
        prev_w = w.copy()

        # cross-sectional std of the funding signal (gate feature)
        vi = np.where(valid & ~np.isnan(srow))[0]
        cs_std = float(np.nanstd(srow[vi])) if len(vi) > 1 else np.nan
        out_cs_std.append(cs_std)

    return (np.array(out_dates, dtype="datetime64[D]"),
            np.array(out_returns, dtype=float),
            np.array(out_fund_pnl, dtype=float),
            np.array(out_cs_std, dtype=float))


# =============================================================================
# WINDOW SELECTION
# =============================================================================

def window_of(d):
    d = np.datetime64(d, "D")
    if d < SEL_END:
        return "SEL"
    if d < OOS_END:
        return "OOS"
    return "UNSEEN"


def split_by_window(dates, daily):
    out = {}
    for w in ["SEL", "OOS", "UNSEEN"]:
        mask = np.array([window_of(d) == w for d in dates])
        out[w] = daily[mask]
    return out


def split_returns_with_dates(dates, daily):
    out = {}
    for w in ["SEL", "OOS", "UNSEEN"]:
        mask = np.array([window_of(d) == w for d in dates])
        out[w] = (dates[mask], daily[mask])
    return out


# =============================================================================
# CIRCULAR BLOCK BOOTSTRAP
# =============================================================================

def circular_block_bootstrap(daily, block_len, n_boot, rng):
    """Politis-Romano circular block bootstrap.

    The series is treated as circular: blocks wrap around the end. This preserves
    the autocorrelation structure up to block_len lags without requiring any iid
    assumption. Each bootstrap resample has the same length as the original.

    Returns shape (n_boot, T) array of resampled series.
    """
    T = len(daily)
    # number of blocks needed (ceil)
    n_blocks = int(np.ceil(T / block_len))
    # circular extension: wrap the series
    circ = np.concatenate([daily, daily])  # length 2T (wrap-safe)
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    boots = np.empty((n_boot, T))
    for i in range(n_boot):
        pieces = []
        for s in starts[i]:
            pieces.append(circ[s:s + block_len])
        boot = np.concatenate(pieces)[:T]
        boots[i] = boot
    return boots


def bootstrap_stats(boots):
    """Given (n_boot, T) bootstrap samples, compute compound return and Sharpe."""
    # compound return
    comp = np.prod(1.0 + boots, axis=1) - 1.0  # shape (n_boot,)
    # Sharpe
    mu = np.mean(boots, axis=1)
    sd = np.std(boots, axis=1)
    sharpe = mu / (sd + 1e-12) * np.sqrt(365)
    return comp, sharpe


def point_stats(daily):
    T = len(daily)
    comp = float(np.prod(1.0 + daily) - 1.0)
    ann = float((1.0 + comp) ** (365.0 / T) - 1.0) if T > 5 else np.nan
    vol = float(np.std(daily) * np.sqrt(365)) if T > 1 else np.nan
    sharpe = float(np.mean(daily) / (np.std(daily) + 1e-12) * np.sqrt(365)) if T > 1 else np.nan
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq / peak - 1.0))
    # n_eff estimate: using autocorrelation (variance-inflation factor)
    # n_eff = T / (1 + 2 * sum_k rho_k) [Newey-West style estimate, lag 1..20]
    if T > 40:
        mu = np.mean(daily)
        demeaned = daily - mu
        var0 = np.var(demeaned)
        if var0 > 0:
            # Bartlett kernel for lags 1..min(T//4, 40)
            max_lag = min(T // 4, 40)
            rho_sum = 0.0
            for lag in range(1, max_lag + 1):
                w_k = 1.0 - lag / (max_lag + 1)  # Bartlett weight
                rho_k = np.mean(demeaned[:-lag] * demeaned[lag:]) / var0
                rho_sum += w_k * rho_k
            vif = max(1.0, 1.0 + 2.0 * rho_sum)  # variance inflation factor
            n_eff = T / vif
        else:
            n_eff = float(T)
    else:
        n_eff = float(T)
    return {"T": T, "comp": comp, "ann": ann, "vol": vol, "sharpe": sharpe,
            "maxdd": maxdd, "n_eff": n_eff}


# =============================================================================
# DISPERSION GATE
# =============================================================================

def dispersion_gate_test(dates, daily, cs_std, gate_threshold):
    """Apply gate: trade only when cs_std > gate_threshold.
    Returns daily returns series with 0 on gated-out days.
    """
    gated = np.where(cs_std > gate_threshold, daily, 0.0)
    # also track n_days active
    n_active = int(np.sum(cs_std > gate_threshold))
    n_total = len(daily)
    return gated, n_active, n_total


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    print("=" * 78)
    print("FUNDING-DISPERSION BLOCK-BOOTSTRAP DEFLATION + DISPERSION-GATE")
    print("Wave 2B statistical gate -- quant referee analysis")
    print("=" * 78)
    print(f"universe={args.universe}  k={K}  lookback={LOOKBACK}d  lag={LAG}")
    print(f"n_bootstrap={args.n_boot}  block_lengths={BLOCK_LENGTHS}  seed={args.seed}")
    print()
    print("INFERENCE DESIGN (pre-registered):")
    print("  H0: compound return <= 0 (one-sided upper-tail)")
    print("  Test: circular block-bootstrap (Politis-Romano), p05>0 = SURVIVES")
    print("  Dispersion gate: threshold fit on SEL-only, applied unchanged to OOS/UNSEEN")
    print("  UNSEEN touched once at the end.")
    print("=" * 78)
    _flush()

    # ---- Load data -----------------------------------------------------------
    print("\nLoading data...")
    panel = load_daily_panel(args.universe)
    dates, assets, ret, fund_daily, vol = to_wide(panel)
    btc_idx = assets.index("BTCUSDT") if "BTCUSDT" in assets else None
    print(f"Panel: {len(dates)} days x {len(assets)} assets "
          f"({str(dates[0])} -> {str(dates[-1])})")
    _flush()

    print("Loading raw 8h funding...")
    fund_8h = load_funding_8h(assets, dates)
    n_8h_cells = int(np.isfinite(fund_8h).sum())
    print(f"  8h funding cells: {n_8h_cells}")
    _flush()

    # ---- Compute signal + simulate 3c ----------------------------------------
    print("\nComputing funding signal + simulating 3c DEPLOYABLE series...")
    sig_8h = funding_signal(fund_8h, lookback=LOOKBACK, lag=LAG)
    out_dates, daily_returns, fund_pnl, cs_std = simulate_3c(dates, ret, fund_8h, vol, sig_8h)
    _flush()

    # ---- Point stats per window (re-derived) ---------------------------------
    windows = split_by_window(out_dates, daily_returns)
    print("\nRE-DERIVED POINT STATISTICS (3c DEPLOYABLE, no bootstrap):")
    print(f"{'window':8s} {'T':>5s} {'comp%':>8s} {'ann%':>8s} {'Sh':>7s} "
          f"{'maxDD%':>8s} {'n_eff':>8s}")
    for w in ["SEL", "OOS", "UNSEEN"]:
        d = windows[w]
        if len(d) == 0:
            continue
        ps = point_stats(d)
        print(f"{w:8s} {ps['T']:5d} {ps['comp']*100:8.2f} {ps['ann']*100:8.2f} "
              f"{ps['sharpe']:7.2f} {ps['maxdd']*100:8.2f} {ps['n_eff']:8.1f}")
    print()
    print("  [Verify against log: OOS comp~7.93%, UNSEEN comp~10.31%]")
    _flush()

    # ---- CIRCULAR BLOCK BOOTSTRAP -------------------------------------------
    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 78)
    print("BLOCK-BOOTSTRAP RESULTS")
    print("=" * 78)
    print(f"{'window':8s} {'blk':>4s} {'T':>5s} {'n_eff':>7s} "
          f"{'point_comp%':>12s} {'p05_comp%':>10s} {'p50_comp%':>10s} "
          f"{'p95_comp%':>10s} {'p05_Sh':>8s} {'p50_Sh':>8s} "
          f"{'p95_Sh':>8s} {'p(>0)':>7s} {'SURVIVES':>10s}")
    print("-" * 78)

    bootstrap_summary = {}
    for w in ["SEL", "OOS", "UNSEEN"]:
        d = windows[w]
        if len(d) < 20:
            continue
        ps = point_stats(d)
        for bl in BLOCK_LENGTHS:
            boot_samples = circular_block_bootstrap(d, bl, args.n_boot, rng)
            comp_boot, sh_boot = bootstrap_stats(boot_samples)
            p05_c = float(np.percentile(comp_boot, 5)) * 100
            p50_c = float(np.percentile(comp_boot, 50)) * 100
            p95_c = float(np.percentile(comp_boot, 95)) * 100
            p05_s = float(np.percentile(sh_boot, 5))
            p50_s = float(np.percentile(sh_boot, 50))
            p95_s = float(np.percentile(sh_boot, 95))
            # p-value: fraction of bootstrap samples where compound <= 0
            p_gt0 = float(np.mean(comp_boot > 0))
            survives = "YES" if p05_c > 0 else "NO"
            print(f"{w:8s} {bl:4d} {ps['T']:5d} {ps['n_eff']:7.1f} "
                  f"{ps['comp']*100:12.2f} {p05_c:10.2f} {p50_c:10.2f} "
                  f"{p95_c:10.2f} {p05_s:8.2f} {p50_s:8.2f} "
                  f"{p95_s:8.2f} {p_gt0:7.3f} {survives:>10s}")
            key = f"{w}_bl{bl}"
            bootstrap_summary[key] = {
                "window": w, "block_len": bl, "T": ps["T"], "n_eff": ps["n_eff"],
                "point_comp_pct": round(ps["comp"] * 100, 3),
                "p05_comp_pct": round(p05_c, 3), "p50_comp_pct": round(p50_c, 3),
                "p95_comp_pct": round(p95_c, 3),
                "p05_sharpe": round(p05_s, 3), "p50_sharpe": round(p50_s, 3),
                "p95_sharpe": round(p95_s, 3),
                "p_gt0": round(p_gt0, 4), "survives_p05": survives == "YES"
            }
        print()
    _flush()

    # ---- Conservative verdict (longest block length = most conservative) -----
    print("=" * 78)
    print("DEFLATION VERDICT (conservative = largest block length = BL=20)")
    print("=" * 78)
    for w in ["SEL", "OOS", "UNSEEN"]:
        key = f"{w}_bl20"
        if key not in bootstrap_summary:
            continue
        s = bootstrap_summary[key]
        survives_str = "SURVIVES" if s["survives_p05"] else "DOES-NOT-SURVIVE"
        print(f"  {w:8s}: point={s['point_comp_pct']:+7.2f}%  "
              f"p05={s['p05_comp_pct']:+7.2f}%  p50={s['p50_comp_pct']:+7.2f}%  "
              f"p(boot>0)={s['p_gt0']:.3f}  n_eff={s['n_eff']:.0f}  "
              f"-> {survives_str}")
    _flush()

    # ---- AUTOCORRELATION DIAGNOSTIC -----------------------------------------
    print("\n" + "=" * 78)
    print("AUTOCORRELATION DIAGNOSTIC (justify block length choice)")
    print("=" * 78)
    print(f"{'window':8s} {'T':>5s}  AC(1)   AC(5)  AC(10)  AC(20)  "
          f"VIF(NW40)  n_eff")
    for w in ["SEL", "OOS", "UNSEEN"]:
        d = windows[w]
        if len(d) < 25:
            continue
        T = len(d)
        mu = np.mean(d)
        dm = d - mu
        var0 = np.var(dm) + 1e-12
        ac_lags = [1, 5, 10, 20]
        acs = []
        for lag in ac_lags:
            if lag >= T:
                acs.append(np.nan)
            else:
                ac = float(np.mean(dm[:-lag] * dm[lag:]) / var0)
                acs.append(ac)
        ps = point_stats(d)
        ac_str = "  ".join(f"{a:+.3f}" if not np.isnan(a) else "  nan " for a in acs)
        # VIF from n_eff
        vif = T / ps["n_eff"] if ps["n_eff"] > 0 else np.nan
        print(f"{w:8s} {T:5d}  {ac_str}  {vif:8.2f}  {ps['n_eff']:6.0f}")
    _flush()

    # =========================================================================
    # DISPERSION GATE
    # SEL-ONLY threshold selection: median cross-sectional funding std on active days.
    # Pre-registration: SINGLE threshold (median of SEL cs_std), no sweep.
    # =========================================================================
    print("\n" + "=" * 78)
    print("DISPERSION GATE ANALYSIS")
    print("=" * 78)
    print("Pre-registered design:")
    print("  - Threshold = MEDIAN of cross-sectional funding std on SEL days")
    print("  - Single threshold (no sweep = no multiple comparisons)")
    print("  - Applied unchanged to OOS and UNSEEN")
    print("  - Gate condition: deploy when cs_funding_std > threshold")
    print()

    # ---- Fit threshold on SEL-only ------------------------------------------
    sel_mask_all = np.array([window_of(d) == "SEL" for d in out_dates])
    sel_cs_std = cs_std[sel_mask_all]
    sel_daily = daily_returns[sel_mask_all]

    # filter to days where strategy was actually active (cs_std > 0)
    active_mask = ~np.isnan(sel_cs_std) & (sel_cs_std > 0)
    gate_threshold = float(np.median(sel_cs_std[active_mask]))
    print(f"SEL cross-sectional funding std:")
    print(f"  n_sel={int(active_mask.sum())}  "
          f"min={np.nanmin(sel_cs_std[active_mask]):.4e}  "
          f"p25={np.nanpercentile(sel_cs_std[active_mask], 25):.4e}  "
          f"median={gate_threshold:.4e}  "
          f"p75={np.nanpercentile(sel_cs_std[active_mask], 75):.4e}  "
          f"max={np.nanmax(sel_cs_std[active_mask]):.4e}")
    print(f"  -> PRE-REGISTERED gate_threshold = {gate_threshold:.6e}")
    print()

    # ---- Apply gate to each window, compare gated vs ungated -----------------
    gate_results = {}
    print(f"{'window':8s} {'regime':16s} {'n_days':>7s} {'pct_active':>10s} "
          f"{'comp%':>8s} {'ann%':>8s} {'Sh':>7s} {'maxDD%':>7s}")
    print("-" * 78)

    for w in ["SEL", "OOS", "UNSEEN"]:
        w_mask = np.array([window_of(d) == w for d in out_dates])
        w_daily = daily_returns[w_mask]
        w_cs = cs_std[w_mask]
        w_dates_w = out_dates[w_mask]

        if len(w_daily) == 0:
            continue

        # ungated (all days)
        ps_all = point_stats(w_daily)
        print(f"{w:8s} {'ungated':16s} {ps_all['T']:7d} {'100.0':>10s} "
              f"{ps_all['comp']*100:8.2f} {ps_all['ann']*100:8.2f} "
              f"{ps_all['sharpe']:7.2f} {ps_all['maxdd']*100:7.2f}")

        # gated
        active = ~np.isnan(w_cs) & (w_cs > gate_threshold)
        n_active = int(np.sum(active))
        pct = n_active / len(w_daily) * 100.0 if len(w_daily) > 0 else 0.0

        if n_active > 5:
            gated_daily = np.where(active, w_daily, 0.0)
            ps_gated = point_stats(gated_daily)
            print(f"{w:8s} {'gated(std>thr)':16s} {n_active:7d} {pct:10.1f} "
                  f"{ps_gated['comp']*100:8.2f} {ps_gated['ann']*100:8.2f} "
                  f"{ps_gated['sharpe']:7.2f} {ps_gated['maxdd']*100:7.2f}")
            gate_results[w] = {
                "ungated_comp": round(ps_all["comp"] * 100, 3),
                "ungated_ann": round(ps_all["ann"] * 100, 3),
                "ungated_sharpe": round(ps_all["sharpe"], 3),
                "gated_comp": round(ps_gated["comp"] * 100, 3),
                "gated_ann": round(ps_gated["ann"] * 100, 3),
                "gated_sharpe": round(ps_gated["sharpe"], 3),
                "n_active": n_active, "pct_active": round(pct, 1),
                "gate_threshold": round(gate_threshold, 8)
            }
        else:
            print(f"{w:8s} {'gated(std>thr)':16s} {n_active:7d} {pct:10.1f}  (too few)")

        # subdecade breakdown (for OOS only: 2023-25 era rescue check)
        if w == "SEL":
            # show 2020-22 vs 2023-25 within SEL
            for era_lbl, lo, hi in [
                ("2020-22", np.datetime64("2020-01-01"), np.datetime64("2023-01-01")),
                ("2023-25", np.datetime64("2023-01-01"), SEL_END)
            ]:
                era_m = np.array([lo <= np.datetime64(d, "D") < hi for d in w_dates_w])
                if era_m.sum() < 10:
                    continue
                ed = w_daily[era_m]
                ec = w_cs[era_m]
                ega = ~np.isnan(ec) & (ec > gate_threshold)
                n_era_act = int(np.sum(ega))
                pct_era = n_era_act / era_m.sum() * 100.0
                # gated
                eg_d = np.where(ega, ed, 0.0)
                ps_era_ung = point_stats(ed)
                ps_era_g = point_stats(eg_d)
                print(f"  {era_lbl:8s}  ungated: comp={ps_era_ung['comp']*100:+.2f}%  "
                      f"ann={ps_era_ung['ann']*100:+.2f}%  | "
                      f"gated(act={n_era_act}/{era_m.sum()},{pct_era:.0f}%): "
                      f"comp={ps_era_g['comp']*100:+.2f}%  ann={ps_era_g['ann']*100:+.2f}%")
        print()

    # ---- Gate bootstrap validation (OOS only, BL=10 as primary) -------------
    print("=" * 78)
    print("GATE BOOTSTRAP VALIDATION (OOS, block_len=10, n=500)")
    print("  Confirm: gated OOS p05 > 0 AND gated > ungated at p05")
    print("=" * 78)
    oos_mask = np.array([window_of(d) == "OOS" for d in out_dates])
    oos_daily = daily_returns[oos_mask]
    oos_cs = cs_std[oos_mask]

    active_oos = ~np.isnan(oos_cs) & (oos_cs > gate_threshold)
    gated_oos = np.where(active_oos, oos_daily, 0.0)

    rng2 = np.random.default_rng(args.seed + 1)
    boot_oos_ung = circular_block_bootstrap(oos_daily, 10, 500, rng2)
    boot_oos_gat = circular_block_bootstrap(gated_oos, 10, 500, rng2)

    c_ung, _ = bootstrap_stats(boot_oos_ung)
    c_gat, _ = bootstrap_stats(boot_oos_gat)

    print(f"  OOS ungated : p05={np.percentile(c_ung,5)*100:+.2f}%  "
          f"p50={np.percentile(c_ung,50)*100:+.2f}%  "
          f"p(>0)={np.mean(c_ung>0):.3f}")
    print(f"  OOS gated   : p05={np.percentile(c_gat,5)*100:+.2f}%  "
          f"p50={np.percentile(c_gat,50)*100:+.2f}%  "
          f"p(>0)={np.mean(c_gat>0):.3f}")
    _flush()

    # ---- Save JSON output ---------------------------------------------------
    out_path = PROJECT_ROOT / "runs" / "funding_bootstrap_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "bootstrap_summary": bootstrap_summary,
        "gate_results": gate_results,
        "gate_threshold": float(gate_threshold),
        "settings": {
            "universe": args.universe, "k": K, "lookback": LOOKBACK, "lag": LAG,
            "n_bootstrap": args.n_boot, "block_lengths": BLOCK_LENGTHS,
            "seed": args.seed
        }
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 78)
    print("FINAL STATISTICAL VERDICT")
    print("=" * 78)

    # Primary gate: OOS and UNSEEN with BL=20 (most conservative)
    oos_bl20 = bootstrap_summary.get("OOS_bl20", {})
    un_bl20 = bootstrap_summary.get("UNSEEN_bl20", {})

    oos_survives = oos_bl20.get("survives_p05", False)
    un_survives = un_bl20.get("survives_p05", False)

    print(f"\n  OOS   (BL=20): point={oos_bl20.get('point_comp_pct',float('nan')):+.2f}%  "
          f"p05={oos_bl20.get('p05_comp_pct',float('nan')):+.2f}%  "
          f"p(boot>0)={oos_bl20.get('p_gt0',float('nan')):.3f}  "
          f"n_eff={oos_bl20.get('n_eff',float('nan')):.0f}  "
          f"-> {'SURVIVES' if oos_survives else 'DOES-NOT-SURVIVE'}")
    print(f"  UNSEEN(BL=20): point={un_bl20.get('point_comp_pct',float('nan')):+.2f}%  "
          f"p05={un_bl20.get('p05_comp_pct',float('nan')):+.2f}%  "
          f"p(boot>0)={un_bl20.get('p_gt0',float('nan')):.3f}  "
          f"n_eff={un_bl20.get('n_eff',float('nan')):.0f}  "
          f"-> {'SURVIVES' if un_survives else 'DOES-NOT-SURVIVE'}")

    if oos_survives and un_survives:
        verdict = "REAL: edge survives block-bootstrap deflation on both held-out windows"
    elif oos_survives or un_survives:
        verdict = "AMBIGUOUS: survives deflation on one held-out window only"
    else:
        verdict = "ARTIFACT: does not survive block-bootstrap deflation"

    print(f"\n  BLOCK-BOOTSTRAP VERDICT: {verdict}")

    # Dispersion gate summary
    oos_gate = gate_results.get("OOS", {})
    un_gate = gate_results.get("UNSEEN", {})
    if oos_gate and un_gate:
        gate_helps_oos = oos_gate.get("gated_ann", 0) > oos_gate.get("ungated_ann", 0)
        gate_helps_un = un_gate.get("gated_ann", 0) > un_gate.get("ungated_ann", 0)
        gate_verdict = (
            "GATE HELPS both" if (gate_helps_oos and gate_helps_un)
            else "GATE HELPS OOS only" if gate_helps_oos
            else "GATE HELPS UNSEEN only" if gate_helps_un
            else "GATE HURTS (ungated dominates)"
        )
        print(f"\n  DISPERSION GATE: {gate_verdict}")
        print(f"    threshold={gate_threshold:.4e}")
        print(f"    OOS:    ungated_ann={oos_gate.get('ungated_ann',float('nan')):+.2f}%  "
              f"gated_ann={oos_gate.get('gated_ann',float('nan')):+.2f}%  "
              f"active={oos_gate.get('pct_active',float('nan')):.0f}%")
        print(f"    UNSEEN: ungated_ann={un_gate.get('ungated_ann',float('nan')):+.2f}%  "
              f"gated_ann={un_gate.get('gated_ann',float('nan')):+.2f}%  "
              f"active={un_gate.get('pct_active',float('nan')):.0f}%")

    print("\n  DEPLOYMENT READINESS (statistical validity only):")
    print("  [X] Market-neutral (violates LO+spot) -- infrastructure blocker")
    if oos_survives and un_survives:
        print("  [OK] Block-bootstrap p05 > 0 on both OOS and UNSEEN (BL=20)")
    else:
        print("  [FAIL] Block-bootstrap p05 <= 0 -- statistical validity NOT confirmed")
    print("  Note: Sharpe~5 pre-deflation; post-deflation Sharpe is the p50 above.")
    _flush()


if __name__ == "__main__":
    main()
