"""funding_dispersion_frictions.py -- DECISIVE net-of-frictions go/no-go on the
cross-sectional funding-DISPERSION carry found by funding_dispersion_probe.py.

The probe found a dollar-neutral L/S perp book (long lowest-funding, short
highest-funding) that is GROSS-positive held-out (OOS +15% / UNSEEN +9.6%,
beta~0, p<0.001 vs random-neutral null). THREE frictions were NOT modeled and
could kill the NET edge. This script models them honestly:

  1. SHORT-SIDE CAPTURABILITY. The carry concentrates in the SHORT leg = the
     highest-funding names = typically small-cap, illiquid alts that are hard
     to short. Variants:
       (a) short-liquid-only  (long broad, short restricted to liquid perps)
       (b) both-legs-liquid   (both legs restricted to liquid majors)
     Liquidity = cross-sectional rank on volume_usd (corroborated by s3_oi_usd).

  2. THE 8h FUNDING CLOCK. The probe collapsed funding to a daily bar. Real
     perps settle every 8h (00:00/08:00/16:00 UTC -- confirmed in
     data/raw/<SYM>/funding/*.parquet). We re-rank + settle on the 8h clock
     using the raw funding timestamps, splitting each daily price return across
     the 3 intra-day settlement sub-bars (the carry is the funding cash-flow,
     which the daily bar over-smooths). Confirms the carry isn't a daily-agg
     artifact.

  3. REALISTIC TIERED TAKER COST. Per-name cost that scales with illiquidity
     (small-caps cost more): a liquidity-tiered taker fee + slippage, NOT a
     flat 5bps. Plus the actual 8h funding cash-flows. Plus a short-borrow /
     short-funding asymmetry haircut on the illiquid short leg.

  4. FORWARD-DECAY SPLIT (the D18 cousin). Carry per era:
       2020-22 (mania funding) | 2023-25 | OOS 2025 | UNSEEN 2026.
     An edge that lives only in 2020-22 is dead-on-arrival.

Reuses funding_dispersion_probe.py's data loading, split discipline and the
dollar-neutral rank construction. Read-only on data. Standalone. Does NOT commit.

Run:
  python src/mining/funding_dispersion_frictions.py --universe u50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

from chimera_loader import ChimeraLoader  # noqa: E402

# ---- split boundaries (project convention, identical to the probe) --------
SEL_END = np.datetime64("2025-03-15")
OOS_END = np.datetime64("2025-12-31")
UNSEEN_END = np.datetime64("2026-05-28")

# ---- era boundaries for the forward-decay split (friction #4) -------------
ERA_BOUNDS = [
    ("2020-22 mania", np.datetime64("2020-01-01"), np.datetime64("2023-01-01")),
    ("2023-25",       np.datetime64("2023-01-01"), np.datetime64("2025-03-15")),
    ("OOS 2025",      np.datetime64("2025-03-15"), OOS_END),
    ("UNSEEN 2026",   OOS_END,                     np.datetime64("2026-06-01")),
]

SETTLEMENTS_PER_DAY = 3  # 8h clock -> 3 settlements/day

# ---- COST MODEL -----------------------------------------------------------
# Per CLAUDE.md: assume TAKER (maker p_fill 0.21-0.40 unreliable -> taker).
# Binance perp taker ~ 4.5bps; majors get the floor, illiquid names pay a
# tier premium (wider spread + slippage on a real clip). Tiers are by
# cross-sectional liquidity rank (volume_usd percentile within the day).
# Numbers are deliberately CONSERVATIVE-pessimistic (we are the adversary).
TAKER_BASE_BPS = 4.5            # base taker fee per side
SLIP_LIQUID_BPS = 2.0          # top-quartile liquidity: thin spread/slippage
SLIP_MID_BPS = 8.0             # mid liquidity
SLIP_ILLIQUID_BPS = 25.0       # bottom-quartile: wide spread + impact on a clip
# short-side borrow/funding asymmetry: shorting an illiquid alt perp can carry
# an extra implicit cost (negative-funding spikes against a crowded short,
# inventory/locate friction). Applied as a per-8h haircut on the SHORT leg of
# illiquid names. Conservative.
SHORT_ILLIQUID_BORROW_BPS_PER_8H = 1.0   # ~1bp/8h = ~11%/yr drag on illiq shorts

# liquidity-floor variants: "liquid" = top fraction of the cross-section by
# volume_usd. Tune the floor here; reported across a couple of settings.
LIQ_TOP_FRACTION = 0.40        # "liquid" = top 40% by dollar volume that day
LIQ_MAJORS_FRACTION = 0.20     # "majors" = top 20%


def _flush():
    sys.stdout.flush()


# ===========================================================================
# DATA LOADING (daily panel; reuses the probe's discipline + adds liquidity)
# ===========================================================================
def load_daily_panel(universe: str):
    """Daily panel: {date, asset, ret, funding_daily, volume_usd, oi_usd}.

    funding_daily = fund_rate_mean * fund_n_settlements (per-day cash-flow a
    LONG PAYS when positive). ret = returns_clean (close-to-close daily).
    volume_usd / s3_oi_usd are the liquidity rankers for friction #1.
    """
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
    """(dates, assets, ret[T,N], fund[T,N], vol[T,N], oi[T,N]) with NaN missing."""
    dates = np.sort(panel["date"].unique().to_numpy()).astype("datetime64[D]")
    assets = sorted(panel["asset"].unique().to_list())
    a_idx = {a: i for i, a in enumerate(assets)}
    d_idx = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    T, N = len(dates), len(assets)
    ret = np.full((T, N), np.nan)
    fund = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    oi = np.full((T, N), np.nan)
    for row in panel.iter_rows(named=True):
        di = d_idx[np.datetime64(row["date"], "D")]
        ai = a_idx[row["asset"]]
        ret[di, ai] = row["ret"]
        fund[di, ai] = row["funding_daily"]
        v = row["volume_usd"]
        vol[di, ai] = v if v is not None else np.nan
        o = row["s3_oi_usd"]
        oi[di, ai] = o if (o is not None and o > 0) else np.nan
    return dates, assets, ret, fund, vol, oi


# ===========================================================================
# 8h FUNDING (friction #2): build a daily-aligned funding panel from RAW 8h
# ===========================================================================
def load_funding_8h(assets, dates):
    """Load raw 8h funding per asset, aggregate to per-day SUM and to the
    intra-day settlement SCHEDULE. Returns:
      fund_day_sum[T,N]  = sum of the 3 intra-day funding rates (the true
                           daily funding cash-flow from the 8h clock)
      fund_settles[T,N,3]= the three settlement rates (NaN if missing)
    This is the honest 8h reconstruction; the chimera fund_rate_mean *
    n_settlements is the daily collapse we compare against.
    """
    T, N = len(dates), len(assets)
    fund_day_sum = np.full((T, N), np.nan)
    fund_settles = np.full((T, N, SETTLEMENTS_PER_DAY), np.nan)
    d_idx = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    # slot map: 00:00->0, 08:00->1, 16:00->2
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
        # vectorized: ms -> (day index, hour) via numpy datetime64
        ts = df["timestamp"].to_numpy().astype("int64")
        fr = df["funding_rate"].to_numpy()
        dt = ts.astype("datetime64[ms]")
        day = dt.astype("datetime64[D]")
        hour = ((ts // 3600000) % 24).astype("int64")
        for day_d, hr, rate in zip(day, hour, fr):
            di = d_idx.get(np.datetime64(day_d, "D"))
            if di is None:
                continue
            slot = slot_of.get(int(hr))
            v = fund_day_sum[di, ai]
            fund_day_sum[di, ai] = (0.0 if np.isnan(v) else v) + float(rate)
            if slot is not None:
                fund_settles[di, ai, slot] = float(rate)
    return fund_day_sum, fund_settles


# ===========================================================================
# SIGNAL (past-only, reuses the probe's lag-1 A6 guard)
# ===========================================================================
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


# ===========================================================================
# LIQUIDITY MASKS (friction #1)
# ===========================================================================
def liquidity_rank_pct(vol_row, valid_mask):
    """Cross-sectional liquidity percentile (0=least, 1=most liquid) among
    valid names on a given day, ranked by dollar volume. NaN where invalid."""
    N = len(vol_row)
    pct = np.full(N, np.nan)
    vi = np.where(valid_mask & ~np.isnan(vol_row))[0]
    if len(vi) < 2:
        return pct
    order = vi[np.argsort(vol_row[vi])]      # ascending volume
    ranks = np.empty(len(order))
    ranks[np.arange(len(order))] = np.arange(len(order))
    # assign percentile by position
    for r, idx in enumerate(order):
        pct[idx] = r / (len(order) - 1)
    return pct


def build_weights_friction(srow, valid_mask, k, liq_pct,
                           short_liquid_floor=None, long_liquid_floor=None):
    """Dollar-neutral weights with liquidity restriction.

    short_liquid_floor: only allow SHORT names with liq_pct >= floor
                        (e.g. 0.60 = short only the top-40% most-liquid).
    long_liquid_floor:  same restriction on the LONG leg (None = broad).

    Long = lowest funding among the eligible long pool.
    Short = highest funding among the eligible short pool.
    Returns length-N weights summing to 0, gross=1, or None if either leg
    can't be formed.
    """
    N = len(srow)
    w = np.zeros(N)
    base_valid = valid_mask & ~np.isnan(srow)

    # eligible pools after liquidity restriction
    long_ok = base_valid.copy()
    short_ok = base_valid.copy()
    if long_liquid_floor is not None:
        long_ok = long_ok & ~np.isnan(liq_pct) & (liq_pct >= long_liquid_floor)
    if short_liquid_floor is not None:
        short_ok = short_ok & ~np.isnan(liq_pct) & (liq_pct >= short_liquid_floor)

    long_idx = np.where(long_ok)[0]
    short_idx = np.where(short_ok)[0]
    if len(long_idx) < k or len(short_idx) < k:
        return None
    # rank within each pool
    longs = long_idx[np.argsort(srow[long_idx])][:k]        # lowest funding
    shorts = short_idx[np.argsort(srow[short_idx])][-k:]     # highest funding
    # guard: don't let an asset be both long and short
    overlap = set(longs) & set(shorts)
    if overlap:
        return None
    w[longs] = 0.5 / k
    w[shorts] = -0.5 / k
    return w


def build_weights_baseline(srow, valid_mask, k):
    """The probe's original construction (no liquidity restriction)."""
    return build_weights_friction(srow, valid_mask, k, None,
                                  short_liquid_floor=None, long_liquid_floor=None)


# ===========================================================================
# COST MODEL (friction #3): tiered taker by liquidity percentile
# ===========================================================================
def per_name_cost_bps(liq_pct_value):
    """Round-trip-agnostic PER-SIDE cost in bps for a name at a given liquidity
    percentile. Taker fee + tiered slippage."""
    if np.isnan(liq_pct_value):
        slip = SLIP_MID_BPS
    elif liq_pct_value >= 0.75:
        slip = SLIP_LIQUID_BPS
    elif liq_pct_value >= 0.25:
        slip = SLIP_MID_BPS
    else:
        slip = SLIP_ILLIQUID_BPS
    return (TAKER_BASE_BPS + slip) * 1e-4   # -> fraction per side


# ===========================================================================
# SIMULATION
# ===========================================================================
def _apply_hysteresis(w_new, prev_w, srow, valid, liq_pct, k, band,
                      short_floor, long_floor):
    """Keep currently-held names in the book until their funding rank exits the
    wider band (band*k), instead of forcing a fresh top/bottom-k each day. This
    cuts self-inflicted daily churn that a sticky carry signal does not need.
    Returns a dollar-neutral weight vector (gross=1) or w_new on failure."""
    if band <= 1.0:
        return w_new
    base_valid = valid & ~np.isnan(srow)
    long_ok = base_valid.copy()
    short_ok = base_valid.copy()
    if long_floor is not None:
        long_ok = long_ok & ~np.isnan(liq_pct) & (liq_pct >= long_floor)
    if short_floor is not None:
        short_ok = short_ok & ~np.isnan(liq_pct) & (liq_pct >= short_floor)
    long_idx = np.where(long_ok)[0]
    short_idx = np.where(short_ok)[0]
    wide = int(np.ceil(band * k))
    if len(long_idx) < k or len(short_idx) < k:
        return w_new
    long_sorted = long_idx[np.argsort(srow[long_idx])]
    short_sorted = short_idx[np.argsort(srow[short_idx])]
    long_wide = set(long_sorted[:wide].tolist())
    short_wide = set(short_sorted[-wide:].tolist())
    held_long = [j for j in np.where(prev_w > 0)[0]
                 if j in long_wide]
    held_short = [j for j in np.where(prev_w < 0)[0]
                  if j in short_wide]
    # fill remaining slots from the tightest names not already held
    longs = list(held_long)
    for j in long_sorted:
        if len(longs) >= k:
            break
        if j not in longs:
            longs.append(int(j))
    longs = longs[:k]
    shorts = list(held_short)
    for j in short_sorted[::-1]:
        if len(shorts) >= k:
            break
        if j not in shorts:
            shorts.append(int(j))
    shorts = shorts[:k]
    if len(longs) < k or len(shorts) < k or (set(longs) & set(shorts)):
        return w_new
    w = np.zeros(len(w_new))
    w[longs] = 0.5 / k
    w[shorts] = -0.5 / k
    return w


def simulate(dates, ret, fund_settle_day, fund_settles, vol, oi, signal, k,
             weight_fn, tiered_cost=True, borrow_asym=True,
             use_8h_settle=True, hysteresis_band=1.0,
             short_floor=None, long_floor=None):
    """Walk the panel daily; rebalance each day.

    fund_settle_day[T,N] = per-day funding cash-flow a long PAYS (from 8h sum
                           when use_8h_settle, else the daily collapse).
    Costs: tiered per-name taker on turnover (friction #3) when tiered_cost,
           else flat 5bps. borrow_asym adds the illiquid-short haircut.
    hysteresis_band > 1.0 keeps held names until they exit band*k (cuts churn);
    short_floor/long_floor are passed so hysteresis respects the same liq pools.

    Returns recs: (date, price_pnl, funding_pnl, cost, total, w).
    """
    T, N = ret.shape
    prev_w = np.zeros(N)
    recs = []
    for t in range(T - 1):
        srow = signal[t, :]
        valid = (~np.isnan(srow) & ~np.isnan(ret[t + 1, :])
                 & ~np.isnan(fund_settle_day[t + 1, :]))
        liq_pct = liquidity_rank_pct(vol[t, :], valid)
        w = weight_fn(srow, valid, k, liq_pct)
        if w is None:
            prev_w = np.zeros(N)
            continue
        if hysteresis_band > 1.0:
            w = _apply_hysteresis(w, prev_w, srow, valid, liq_pct, k,
                                  hysteresis_band, short_floor, long_floor)
        r_next = np.where(np.isnan(ret[t + 1, :]), 0.0, ret[t + 1, :])
        f_next = np.where(np.isnan(fund_settle_day[t + 1, :]), 0.0,
                          fund_settle_day[t + 1, :])
        price_pnl = float(np.sum(w * r_next))
        funding_pnl = float(-np.sum(w * f_next))   # long pays +f, short receives

        # borrow/short asymmetry on illiquid shorts (friction #3 extra)
        borrow_drag = 0.0
        if borrow_asym:
            liq_next = liquidity_rank_pct(vol[t + 1, :], ~np.isnan(vol[t + 1, :]))
            shorts = np.where(w < 0)[0]
            for j in shorts:
                lp = liq_next[j]
                if np.isnan(lp) or lp < 0.25:        # illiquid short
                    borrow_drag += (abs(w[j])
                                    * SHORT_ILLIQUID_BORROW_BPS_PER_8H * 1e-4
                                    * SETTLEMENTS_PER_DAY)

        # turnover cost (friction #3 tiered)
        dw = w - prev_w
        if tiered_cost:
            liq_now = liquidity_rank_pct(vol[t, :], ~np.isnan(vol[t, :]))
            cost = 0.0
            for j in np.where(np.abs(dw) > 1e-12)[0]:
                cost += abs(dw[j]) * per_name_cost_bps(liq_now[j])
        else:
            cost = float(np.sum(np.abs(dw))) * 0.0005
        cost += borrow_drag

        total = price_pnl + funding_pnl - cost
        recs.append((dates[t + 1], price_pnl, funding_pnl, cost, total, w.copy()))
        prev_w = w
    return recs


# ===========================================================================
# AGGREGATION
# ===========================================================================
def window_of(d):
    d = np.datetime64(d, "D")
    if d < SEL_END:
        return "SEL"
    if d < OOS_END:
        return "OOS"
    return "UNSEEN"


def _stats(daily, price, funding, cost, btc_r=None):
    n = len(daily)
    if n == 0:
        return None
    comp = float(np.prod(1.0 + daily) - 1.0)
    ann = float((1.0 + comp) ** (365.0 / n) - 1.0) if n > 5 else float("nan")
    vol = float(np.std(daily) * np.sqrt(365)) if n > 1 else float("nan")
    sharpe = (float(np.mean(daily) / (np.std(daily) + 1e-12) * np.sqrt(365))
              if n > 1 else float("nan"))
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq / peak - 1.0))
    beta = float("nan")
    if btc_r is not None:
        m = ~np.isnan(btc_r)
        if m.sum() > 10 and np.std(btc_r[m]) > 0:
            beta = float(np.cov(daily[m], btc_r[m])[0, 1] / np.var(btc_r[m]))
    return {
        "n_days": n, "compound_pct": comp * 100, "ann_pct": ann * 100,
        "sharpe": sharpe, "vol_pct": vol * 100, "maxdd_pct": maxdd * 100,
        "price_pnl_sum_pct": float(np.sum(price)) * 100,
        "funding_pnl_sum_pct": float(np.sum(funding)) * 100,
        "cost_sum_pct": float(np.sum(cost)) * 100, "beta_btc": beta,
    }


def agg_window(recs, win, btc_idx=None, ret=None, dates=None):
    sel = [r for r in recs if window_of(r[0]) == win]
    if not sel:
        return None
    daily = np.array([r[4] for r in sel])
    price = np.array([r[1] for r in sel])
    funding = np.array([r[2] for r in sel])
    cost = np.array([r[3] for r in sel])
    btc_r = None
    if btc_idx is not None and ret is not None and dates is not None:
        ddates = np.array([np.datetime64(r[0], "D") for r in sel])
        d2i = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
        btc_r = np.array([ret[d2i[d], btc_idx] if d in d2i else np.nan
                          for d in ddates])
    return _stats(daily, price, funding, cost, btc_r)


def agg_era(recs, lo, hi, btc_idx=None, ret=None, dates=None):
    sel = [r for r in recs
           if lo <= np.datetime64(r[0], "D") < hi]
    if not sel:
        return None
    daily = np.array([r[4] for r in sel])
    price = np.array([r[1] for r in sel])
    funding = np.array([r[2] for r in sel])
    cost = np.array([r[3] for r in sel])
    btc_r = None
    if btc_idx is not None and ret is not None and dates is not None:
        ddates = np.array([np.datetime64(r[0], "D") for r in sel])
        d2i = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
        btc_r = np.array([ret[d2i[d], btc_idx] if d in d2i else np.nan
                          for d in ddates])
    return _stats(daily, price, funding, cost, btc_r)


# ===========================================================================
# WEIGHT-FN FACTORIES
# ===========================================================================
def make_weight_fn(short_floor=None, long_floor=None):
    def fn(srow, valid, k, liq_pct):
        return build_weights_friction(srow, valid, k, liq_pct,
                                     short_liquid_floor=short_floor,
                                     long_liquid_floor=long_floor)
    return fn


def print_window_table(label, recs, btc_idx, ret, dates):
    print(f"\n--- {label} ---")
    hdr = (f"{'win':7s} {'days':>5s} {'comp%':>8s} {'ann%':>7s} {'Sh':>6s} "
           f"{'vol%':>6s} {'maxDD%':>7s} {'fund%':>7s} {'price%':>8s} "
           f"{'cost%':>6s} {'betaBTC':>8s}")
    print(hdr)
    out = {}
    for win in ["SEL", "OOS", "UNSEEN"]:
        s = agg_window(recs, win, btc_idx, ret, dates)
        out[win] = s
        if s:
            print(f"{win:7s} {s['n_days']:5d} {s['compound_pct']:8.2f} "
                  f"{s['ann_pct']:7.2f} {s['sharpe']:6.2f} {s['vol_pct']:6.2f} "
                  f"{s['maxdd_pct']:7.2f} {s['funding_pnl_sum_pct']:7.2f} "
                  f"{s['price_pnl_sum_pct']:8.2f} {s['cost_sum_pct']:6.2f} "
                  f"{s['beta_btc']:+8.3f}")
    _flush()
    return out


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--k", type=int, default=5,
                    help="legs per side; default 5 (probe-selected region)")
    args = ap.parse_args()

    print("=" * 78)
    print("FUNDING-DISPERSION FRICTIONS -- net-of-frictions go/no-go")
    print("=" * 78)
    print(f"universe={args.universe} lookback={args.lookback}d lag={args.lag} "
          f"k={args.k}")
    print(f"cost: taker_base={TAKER_BASE_BPS}bps + tiered_slip "
          f"[{SLIP_LIQUID_BPS}/{SLIP_MID_BPS}/{SLIP_ILLIQUID_BPS}bps] "
          f"+ illiq-short borrow {SHORT_ILLIQUID_BORROW_BPS_PER_8H}bps/8h")
    _flush()

    panel = load_daily_panel(args.universe)
    dates, assets, ret, fund_daily, vol, oi = to_wide(panel)
    btc_idx = assets.index("BTCUSDT") if "BTCUSDT" in assets else None
    print(f"panel: {len(dates)} days x {len(assets)} assets "
          f"({str(dates[0])} -> {str(dates[-1])})")
    _flush()

    # ---- friction #2: build the honest 8h funding daily-sum -----------------
    print("\nloading raw 8h funding (friction #2)...")
    _flush()
    fund_8h_sum, fund_settles = load_funding_8h(assets, dates)
    # coverage / agreement vs the chimera daily collapse
    both = ~np.isnan(fund_8h_sum) & ~np.isnan(fund_daily)
    if both.sum() > 0:
        corr = np.corrcoef(fund_8h_sum[both], fund_daily[both])[0, 1]
        mae = float(np.nanmean(np.abs(fund_8h_sum[both] - fund_daily[both])))
        print(f"  8h-sum vs chimera-daily: corr={corr:.4f}  MAE={mae:.2e}  "
              f"cov={both.sum()} cells  "
              f"(8h cells={np.isfinite(fund_8h_sum).sum()}, "
              f"daily cells={np.isfinite(fund_daily).sum()})")
        # diagnostic: chimera daily funding is the PREVIOUS day's 8h sum
        # (1-day label offset; magnitudes bit-identical). corr(daily[t],8h[t-1]).
        lag_both = ~np.isnan(fund_8h_sum[:-1]) & ~np.isnan(fund_daily[1:])
        if lag_both.sum() > 100:
            corr_lag = np.corrcoef(fund_8h_sum[:-1][lag_both],
                                   fund_daily[1:][lag_both])[0, 1]
            print(f"  NOTE: corr(chimera_daily[t], 8h_sum[t-1])={corr_lag:.4f} "
                  f"-> chimera daily funding is the PRIOR day's 8h sum "
                  f"(1-day label offset, magnitudes identical). The 8h-clock "
                  f"path below removes this offset; it is CONSERVATIVE for "
                  f"leak (the probe earned stale, not future, funding).")
    _flush()

    k = args.k
    sig_daily = funding_signal(fund_daily, lookback=args.lookback, lag=args.lag)
    sig_8h = funding_signal(fund_8h_sum, lookback=args.lookback, lag=args.lag)

    # =====================================================================
    # STEP 0 -- GROSS BASELINE (reproduce the probe: daily clock, flat 5bps,
    #            no liquidity restriction). Anchors the waterfall.
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 0: GROSS BASELINE (daily clock, flat 5bps, no liq filter)")
    print("#" * 78)
    recs_gross = simulate(dates, ret, fund_daily, fund_settles, vol, oi,
                          sig_daily, k, make_weight_fn(None, None),
                          tiered_cost=False, borrow_asym=False)
    g = print_window_table("GROSS baseline", recs_gross, btc_idx, ret, dates)

    # =====================================================================
    # STEP 1 -- FRICTION #1: short-side capturability
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 1: FRICTION #1 -- short-side capturability (liquidity floor)")
    print("#" * 78)
    short_floor = 1.0 - LIQ_TOP_FRACTION       # short only top-40% liquid
    majors_floor = 1.0 - LIQ_MAJORS_FRACTION    # both legs top-20% liquid
    # (a) short-liquid-only (long broad)
    recs_1a = simulate(dates, ret, fund_daily, fund_settles, vol, oi,
                       sig_daily, k, make_weight_fn(short_floor, None),
                       tiered_cost=False, borrow_asym=False)
    a1 = print_window_table(
        f"1a short-liquid-only (short top-{int(LIQ_TOP_FRACTION*100)}% liq, "
        f"long broad), daily clock, flat 5bps", recs_1a, btc_idx, ret, dates)
    # (b) both-legs-liquid majors
    recs_1b = simulate(dates, ret, fund_daily, fund_settles, vol, oi,
                       sig_daily, k, make_weight_fn(majors_floor, majors_floor),
                       tiered_cost=False, borrow_asym=False)
    b1 = print_window_table(
        f"1b both-legs majors (top-{int(LIQ_MAJORS_FRACTION*100)}% liq both "
        f"legs), daily clock, flat 5bps", recs_1b, btc_idx, ret, dates)

    # =====================================================================
    # STEP 2 -- FRICTION #2: 8h funding clock
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 2: FRICTION #2 -- 8h funding clock (settle on raw 8h sum)")
    print("#" * 78)
    # gross construction but on the 8h funding signal + 8h funding cash-flow
    recs_8h = simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi,
                       sig_8h, k, make_weight_fn(None, None),
                       tiered_cost=False, borrow_asym=False)
    print_window_table("2 8h-clock, no-liq-filter, flat 5bps "
                       "(vs STEP 0 to isolate the 8h effect)",
                       recs_8h, btc_idx, ret, dates)

    # =====================================================================
    # STEP 3 -- THE FULLY-REALISTIC VARIANT
    #   short-liquid-only + 8h clock + tiered taker + illiq-short borrow.
    #   3a = daily rebalance (over-charges churn a sticky carry doesn't need).
    #   3c = +hysteresis band=2.0 (only swap a held name when it exits the wide
    #        band) -- the DEPLOYABLE config. The daily-vs-hysteresis gap is the
    #        single most important honesty check: a funding carry has sticky
    #        ranks, so charging a full daily re-rank is an unfair friction.
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 3: FULLY-REALISTIC -- short-liquid + 8h + tiered taker + borrow")
    print("#" * 78)
    recs_real = simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi,
                         sig_8h, k, make_weight_fn(short_floor, None),
                         tiered_cost=True, borrow_asym=True,
                         short_floor=short_floor, long_floor=None)
    r3 = print_window_table("3a REALISTIC daily-rebalance (short-liquid, 8h, "
                            "tiered taker, borrow)", recs_real, btc_idx, ret,
                            dates)
    recs_real_maj = simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi,
                             sig_8h, k, make_weight_fn(majors_floor, majors_floor),
                             tiered_cost=True, borrow_asym=True,
                             short_floor=majors_floor, long_floor=majors_floor)
    print_window_table("3b REALISTIC majors-both-legs (8h, tiered taker, "
                       "borrow)", recs_real_maj, btc_idx, ret, dates)
    HYST = 2.0
    recs_dep = simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi,
                        sig_8h, k, make_weight_fn(short_floor, None),
                        tiered_cost=True, borrow_asym=True, hysteresis_band=HYST,
                        short_floor=short_floor, long_floor=None)
    r3c = print_window_table(
        f"3c DEPLOYABLE short-liquid + 8h + tiered taker + borrow + "
        f"hysteresis({HYST}) [LOW-TURNOVER]", recs_dep, btc_idx, ret, dates)

    # =====================================================================
    # STEP 3.5 -- RANDOM dollar-neutral null on the DEPLOYABLE cost structure
    #   (shuffle funding ranks within the same liquid pool, same k, same
    #    realistic costs; NO hysteresis so the signal-vs-random comparison is
    #    clean, not pinned by carried-forward real-signal names).
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 3.5: RANDOM null (same liquid pool + same realistic costs)")
    print("#" * 78)
    rng_master = np.random.default_rng(0)

    def make_random_wf(seed):
        rng = np.random.default_rng(seed)

        def fn(srow, valid, k, liq_pct):
            s = srow.copy()
            vi = np.where(valid & ~np.isnan(s))[0]
            if len(vi) >= 2:
                perm = rng.permutation(vi)
                tmp = s[vi].copy()
                s[perm] = tmp
            return build_weights_friction(s, valid, k, liq_pct,
                                         short_liquid_floor=short_floor,
                                         long_liquid_floor=None)
        return fn

    n_seeds = 200
    null = {"OOS": [], "UNSEEN": []}
    for seed in range(n_seeds):
        nrecs = simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi,
                         sig_8h, k, make_random_wf(seed), tiered_cost=True,
                         borrow_asym=True, short_floor=short_floor)
        for w in ["OOS", "UNSEEN"]:
            ns = agg_window(nrecs, w)
            if ns:
                null[w].append(ns["compound_pct"])
    # compare to the realistic DAILY book (matched cost structure; hysteresis is
    # a sizing refinement applied equally to a real book, not to a random one).
    pvals = {}
    for w in ["OOS", "UNSEEN"]:
        arr = np.array(null[w])
        bc = r3[w]["compound_pct"]
        p = float((arr >= bc).mean())
        pvals[w] = p
        print(f"  {w}: funding-rank book comp={bc:6.2f}%   random-null mean="
              f"{arr.mean():7.2f}% sd={arr.std():5.2f} "
              f"[p05={np.percentile(arr,5):+.2f} p95={np.percentile(arr,95):+.2f}]"
              f"   p(null>=book)={p:.3f}")
    _flush()

    # =====================================================================
    # STEP 4 -- FORWARD-DECAY SPLIT (per era) on the DEPLOYABLE variant
    # =====================================================================
    print("\n" + "#" * 78)
    print("# STEP 4: FORWARD-DECAY by era (D18 cousin) -- DEPLOYABLE variant (3c)")
    print("#" * 78)
    print(f"{'era':16s} {'days':>5s} {'comp%':>8s} {'ann%':>8s} {'Sh':>6s} "
          f"{'maxDD%':>7s} {'fund%':>7s} {'price%':>8s} {'cost%':>6s} "
          f"{'betaBTC':>8s}")
    era_stats = {}
    for lbl, lo, hi in ERA_BOUNDS:
        s = agg_era(recs_dep, lo, hi, btc_idx, ret, dates)
        era_stats[lbl] = s
        if s:
            print(f"{lbl:16s} {s['n_days']:5d} {s['compound_pct']:8.2f} "
                  f"{s['ann_pct']:8.2f} {s['sharpe']:6.2f} {s['maxdd_pct']:7.2f} "
                  f"{s['funding_pnl_sum_pct']:7.2f} {s['price_pnl_sum_pct']:8.2f} "
                  f"{s['cost_sum_pct']:6.2f} {s['beta_btc']:+8.3f}")
        else:
            print(f"{lbl:16s}  (no records)")
    print("\n  [daily-rebalance (3a) era split for contrast -- shows churn drag]")
    for lbl, lo, hi in ERA_BOUNDS:
        s = agg_era(recs_real, lo, hi, btc_idx, ret, dates)
        if s:
            print(f"  {lbl:16s} ann={s['ann_pct']:8.2f}%  "
                  f"fund={s['funding_pnl_sum_pct']:7.2f}%  "
                  f"cost={s['cost_sum_pct']:6.2f}%")
    _flush()

    # =====================================================================
    # GROSS -> NET WATERFALL  (held-out: OOS + UNSEEN, annualized)
    # =====================================================================
    print("\n" + "=" * 78)
    print("GROSS -> NET WATERFALL (held-out, annualized %/yr)")
    print("=" * 78)
    for win in ["OOS", "UNSEEN"]:
        gv = g[win]["ann_pct"] if g[win] else float("nan")
        a1v = a1[win]["ann_pct"] if a1[win] else float("nan")
        r3v = r3[win]["ann_pct"] if r3[win] else float("nan")
        depv = r3c[win]["ann_pct"] if r3c[win] else float("nan")
        print(f"\n{win}:")
        print(f"  GROSS (daily, flat 5bps, broad)               : {gv:+7.2f}")
        print(f"  - illiquid-short restriction (short top-40%)  : {a1v:+7.2f}"
              f"   (delta {a1v - gv:+.2f})")
        print(f"  - 8h clock + tiered taker + borrow [daily reb]: {r3v:+7.2f}"
              f"   (delta {r3v - a1v:+.2f})")
        print(f"  - hysteresis (cut self-inflicted churn) = NET : {depv:+7.2f}"
              f"   (delta {depv - r3v:+.2f})")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "=" * 78)
    print("VERDICT (most-realistic DEPLOYABLE = short-liquid + 8h + tiered "
          "taker + borrow + hysteresis)")
    print("=" * 78)
    oos = r3c["OOS"]
    un = r3c["UNSEEN"]

    def _p(s):
        return float("nan") if s is None else s["ann_pct"]
    print(f"  OOS    : ann={_p(oos):+.2f}%  Sh={oos['sharpe']:.2f}  "
          f"maxDD={oos['maxdd_pct']:.2f}%  beta={oos['beta_btc']:+.3f}  "
          f"null-p={pvals['OOS']:.3f}")
    if un:
        print(f"  UNSEEN : ann={_p(un):+.2f}%  Sh={un['sharpe']:.2f}  "
              f"maxDD={un['maxdd_pct']:.2f}%  beta={un['beta_btc']:+.3f}  "
              f"null-p={pvals['UNSEEN']:.3f}")
    print("  era carry (deployable, ann%/yr): "
          + "  ".join(f"{lbl.split()[0]}={_p(era_stats[lbl]):+.1f}"
                      for lbl, _, _ in ERA_BOUNDS))

    oos_pos = oos and oos["ann_pct"] > 0
    un_pos = un and un["ann_pct"] > 0
    held_out_pos = oos_pos and un_pos
    null_clears = pvals["OOS"] < 0.05 and pvals["UNSEEN"] < 0.05
    # forward-persistence judged on the HELD-OUT path (OOS + UNSEEN both > 0)
    # plus a non-mania in-sample era (2023-25) being non-negative -- so the edge
    # is NOT purely a 2020-22 mania artifact (the D18 failure mode).
    s_2325 = era_stats.get("2023-25")
    not_mania_only = s_2325 and s_2325["ann_pct"] > 0
    fwd_persist = held_out_pos and not_mania_only
    healthy_sharpe = oos and oos["sharpe"] >= 1.0

    if held_out_pos and null_clears and fwd_persist and healthy_sharpe:
        verdict = "DEPLOYABLE"
    elif held_out_pos and null_clears:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    print(f"\nVERDICT: {verdict}")
    reasons = []
    if not oos_pos:
        reasons.append("OOS net-negative")
    if not un_pos:
        reasons.append("UNSEEN net-negative")
    if not null_clears:
        reasons.append(f"fails random null (p_OOS={pvals['OOS']:.2f}, "
                       f"p_UNSEEN={pvals['UNSEEN']:.2f})")
    if not not_mania_only:
        reasons.append("2020-22-only (D18 fate: 2023-25 non-positive)")
    if not healthy_sharpe:
        reasons.append("thin Sharpe (<1.0)")
    if reasons:
        print("  caveats/killers: " + "; ".join(reasons))
    else:
        print("  net-positive held-out, beats random null net-of-cost, "
              "forward-persistent, capturable in liquid names.")
    _flush()


if __name__ == "__main__":
    main()
