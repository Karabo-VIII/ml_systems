"""funding_dispersion_probe.py -- CHEAP existence test of a cross-sectional
funding-rate DISPERSION edge, traded DOLLAR-NEUTRAL long/short on perps.

NEW object (cite-checked vs docs/MARKET_FRAMEWORK/01_DEAD_LIST.md):
  - NOT D18 (single-asset delta-neutral carry; decayed ~0 post-2024).
  - NOT D42/D54 (funding-extreme as a per-asset DIRECTIONAL filter).
  - NOT D17/D40/D68 (cross-sectional PRICE-MOMENTUM selection; long-only/directional).
  - NOT D37 (crypto pairs MR).
  This is the RELATIVE funding DISPERSION across the universe, harvested
  dollar-neutral (long lowest-funding, short highest-funding). A structural
  cross-sectional carry that can persist even if AVERAGE funding is arbed to 0.

CONSTRAINTS honored:
  - RELAXES long-only-spot ON PURPOSE: market-neutral needs perp shorts;
    gross lev ~1, net ~0. (This is the whole point.)
  - Cost-honest: TAKER fee per side (no flat-30bps lie); funding cash-flows
    are modeled explicitly (they ARE the carry).
  - No look-ahead: at rebalance close(d) we rank on PAST-only funding; we earn
    day d+1 price return + day d+1 funding cash-flow. A6-class guard: the daily
    funding cash-flow is applied on the bar it accrues, not forward-filled into
    the ranking.

Benchmark (critical -- beta-benchmark is WRONG for a ~0-beta book):
  (a) RANDOM dollar-neutral null: shuffle the funding ranks -> random long/short
      split, same k, same costs, many seeds. Edge is real only if the
      funding-ranked book beats this null OUT OF SAMPLE.
  (b) realized BTC-beta ~ 0 confirmation.

Splits (project convention, src/strat/alt_bar_trend_lab.py):
  SEL (TRAIN+VAL): start -> 2025-03-15   (pick k + cost on SEL ONLY)
  OOS:             2025-03-15 -> 2025-12-31   (report)
  UNSEEN:          2025-12-31 -> 2026-05-28   (peek ONCE at the end)

Read-only on data. Standalone. Does NOT commit.

Run:
  python src/mining/funding_dispersion_probe.py --universe u50 --rebalance 1d
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

# ---- split boundaries (project convention) -------------------------------
SEL_END = np.datetime64("2025-03-15")
OOS_END = np.datetime64("2025-12-31")
UNSEEN_END = np.datetime64("2026-05-28")

SETTLEMENTS_PER_DAY = 3  # daily chimera collapses the 8h clock to 3 settlements

# Conservative TAKER assumption (per CLAUDE.md: maker p_fill 0.21-0.40 -> assume
# taker for a conservative probe). Binance perp taker ~ 4-5 bps/side. We use 5bps.
TAKER_FEE_PER_SIDE = 0.0005


def _to_npdate(d):
    return np.array(d, dtype="datetime64[D]")


def load_panel(universe: str):
    """Load a calendar-aligned daily panel of {date, asset, ret, funding_daily}.

    funding_daily = fund_rate_mean * fund_n_settlements = the per-day funding
    cash-flow a LONG PAYS (short receives) when positive.
    ret = returns_clean (close-to-close on the daily dollar-bar).
    """
    loader = ChimeraLoader()
    feats = ["date", "returns_clean", "fund_rate_mean", "fund_n_settlements"]
    panel = loader.load_universe(universe, cadence="1d", features=feats,
                                 add_asset_col=True)
    # daily funding cash-flow (long pays when +). Guard nulls.
    panel = panel.with_columns(
        (pl.col("fund_rate_mean").fill_null(0.0)
         * pl.col("fund_n_settlements").fill_null(SETTLEMENTS_PER_DAY))
        .alias("funding_daily")
    )
    panel = panel.select(["date", "asset", "returns_clean", "funding_daily"])
    panel = panel.rename({"returns_clean": "ret"})
    panel = panel.drop_nulls(subset=["ret"])
    return panel


def to_wide(panel: pl.DataFrame):
    """Return (dates, assets, ret[T,N], fund[T,N]) numpy with NaN for missing."""
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


def funding_signal(fund, lookback=7, lag=1):
    """Past-only cross-sectional funding signal at each rebalance.

    We rank on a trailing-mean funding LEVEL (level, not z; level is what you
    actually pay/receive). signal[t] uses funds in [t-lag-lookback+1 .. t-lag].

    lag=1 (HONEST DEFAULT, A6-class guard): the signal is computed strictly from
    funding that accrued BEFORE the day whose funding cash-flow we then earn. The
    trade decided at close(t) earns day t+1's price return AND day t+1's funding;
    with lag=1 the ranking does not peek at any funding inside the held window.
    Verified: lag 0->1->2 barely changes the result (no look-ahead dependence).
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


def build_weights_from_rank(rank_signal_row, valid_mask, k):
    """Dollar-neutral weights: long bottom-k (lowest funding), short top-k.

    Returns a length-N weight vector summing to 0, gross = 1
    (0.5 long notional + 0.5 short notional).
    """
    N = len(rank_signal_row)
    w = np.zeros(N)
    valid = np.where(valid_mask)[0]
    if len(valid) < 2 * k:
        return None  # not enough cross-section to form the book
    vals = rank_signal_row[valid]
    order = valid[np.argsort(vals)]          # ascending funding
    longs = order[:k]                         # lowest / most negative funding
    shorts = order[-k:]                       # highest funding
    w[longs] = 0.5 / k
    w[shorts] = -0.5 / k
    return w


def simulate(dates, ret, fund, signal, k, fee_per_side,
             weight_fn, seed=None):
    """Walk the panel, rebalancing each day. weight_fn(signal_row, valid, k, rng).

    PnL_{t+1} = sum_j w[t,j] * ret[t+1,j]            (price PnL, dollar-neutral)
              - sum_j (w[t,j] * funding[t+1, j])      (funding cash-flow:
                  long w>0 PAYS funding>0 -> subtract; short w<0 RECEIVES -> adds)
              - fee_per_side * turnover                (taker cost on rebalance)

    Returns dict with daily series split by window, and the long/short weights
    history for beta.
    """
    rng = np.random.default_rng(seed)
    T, N = ret.shape
    prev_w = np.zeros(N)
    recs = []  # (date_of_return, price_pnl, funding_pnl, cost, total)
    w_hist = []
    for t in range(T - 1):
        srow = signal[t, :]
        valid = ~np.isnan(srow) & ~np.isnan(ret[t + 1, :]) & ~np.isnan(fund[t + 1, :])
        w = weight_fn(srow, valid, k, rng)
        if w is None:
            prev_w = np.zeros(N)
            continue
        r_next = np.where(np.isnan(ret[t + 1, :]), 0.0, ret[t + 1, :])
        f_next = np.where(np.isnan(fund[t + 1, :]), 0.0, fund[t + 1, :])
        price_pnl = float(np.sum(w * r_next))
        funding_pnl = float(-np.sum(w * f_next))   # long pays +funding, short receives
        turnover = float(np.sum(np.abs(w - prev_w)))
        cost = fee_per_side * turnover
        total = price_pnl + funding_pnl - cost
        recs.append((dates[t + 1], price_pnl, funding_pnl, cost, total, w.copy()))
        w_hist.append(w.copy())
        prev_w = w
    return recs


def window_of(d):
    d = np.datetime64(d, "D")
    if d < SEL_END:
        return "SEL"
    if d < OOS_END:
        return "OOS"
    return "UNSEEN"


def agg_window(recs, win, btc_idx=None, ret=None, dates=None):
    """Aggregate a window's daily records into stats."""
    sel = [r for r in recs if window_of(r[0]) == win]
    if not sel:
        return None
    daily = np.array([r[4] for r in sel])
    price = np.array([r[1] for r in sel])
    funding = np.array([r[2] for r in sel])
    cost = np.array([r[3] for r in sel])
    n = len(daily)
    comp = float(np.prod(1.0 + daily) - 1.0)
    ann = float((1.0 + comp) ** (365.0 / n) - 1.0) if n > 5 else float("nan")
    vol = float(np.std(daily) * np.sqrt(365)) if n > 1 else float("nan")
    sharpe = float(np.mean(daily) / (np.std(daily) + 1e-12) * np.sqrt(365)) if n > 1 else float("nan")
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq / peak - 1.0))
    # realized beta-to-BTC: regress daily book return on BTC daily return
    beta = float("nan")
    if btc_idx is not None and ret is not None and dates is not None:
        ddates = np.array([np.datetime64(r[0], "D") for r in sel])
        d2i = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
        btc_r = np.array([ret[d2i[d], btc_idx] if d in d2i else np.nan for d in ddates])
        m = ~np.isnan(btc_r)
        if m.sum() > 10 and np.std(btc_r[m]) > 0:
            beta = float(np.cov(daily[m], btc_r[m])[0, 1] / np.var(btc_r[m]))
    return {
        "n_days": n, "compound_pct": comp * 100, "ann_pct": ann * 100,
        "sharpe": sharpe, "vol_pct": vol * 100, "maxdd_pct": maxdd * 100,
        "price_pnl_sum_pct": float(np.sum(price)) * 100,
        "funding_pnl_sum_pct": float(np.sum(funding)) * 100,
        "cost_sum_pct": float(np.sum(cost)) * 100,
        "beta_btc": beta,
    }


def funding_weight_fn(srow, valid, k, rng):
    return build_weights_from_rank(srow, valid, k)


def random_weight_fn(srow, valid, k, rng):
    """Shuffle the funding ranks -> random dollar-neutral split, same k."""
    shuffled = srow.copy()
    vi = np.where(valid)[0]
    if len(vi) >= 2:
        perm = rng.permutation(vi)
        tmp = shuffled[vi].copy()
        shuffled[perm] = tmp
    return build_weights_from_rank(shuffled, valid, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--lookback", type=int, default=7,
                    help="trailing days for the funding signal")
    ap.add_argument("--lag", type=int, default=1,
                    help="A6-class leak guard: days before held bar (default 1)")
    ap.add_argument("--n-null-seeds", type=int, default=300)
    args = ap.parse_args()

    print("=" * 78)
    print("CROSS-SECTIONAL FUNDING-DISPERSION PROBE (dollar-neutral L/S perps)")
    print("=" * 78)
    print(f"universe={args.universe}  lookback={args.lookback}d  "
          f"taker={TAKER_FEE_PER_SIDE*1e4:.1f}bps/side  null_seeds={args.n_null_seeds}")

    panel = load_panel(args.universe)
    dates, assets, ret, fund = to_wide(panel)
    btc_idx = assets.index("BTCUSDT") if "BTCUSDT" in assets else None
    print(f"panel: {len(dates)} days x {len(assets)} assets  "
          f"({str(dates[0])} -> {str(dates[-1])})")

    sig = funding_signal(fund, lookback=args.lookback, lag=args.lag)
    print(f"funding signal: trailing-{args.lookback}d level, lag={args.lag} "
          f"(A6 leak guard)")

    # ---- 1) SELECT k on SEL ONLY -------------------------------------------
    print("\n--- SEL selection (pick k by SEL compound; cost fixed @ taker) ---")
    k_grid = [3, 4, 5, 6, 8, 10]
    sel_results = {}
    for k in k_grid:
        if 2 * k > len(assets):
            continue
        recs = simulate(dates, ret, fund, sig, k, TAKER_FEE_PER_SIDE,
                        funding_weight_fn)
        s = agg_window(recs, "SEL", btc_idx, ret, dates)
        sel_results[k] = (s, recs)
        print(f"  k={k:2d}: SEL comp={s['compound_pct']:7.2f}%  "
              f"ann={s['ann_pct']:6.2f}%  Sh={s['sharpe']:5.2f}  "
              f"fund={s['funding_pnl_sum_pct']:6.2f}%  price={s['price_pnl_sum_pct']:7.2f}%  "
              f"cost={s['cost_sum_pct']:5.2f}%  beta={s['beta_btc']:+.3f}")
    best_k = max(sel_results, key=lambda k: sel_results[k][0]["sharpe"])
    print(f"  -> selected k={best_k} (best SEL Sharpe)")

    recs = sel_results[best_k][1]

    # ---- 2) Report OOS + UNSEEN for the selected k -------------------------
    print(f"\n--- Funding-ranked book (k={best_k}) by window ---")
    hdr = (f"{'win':7s} {'days':>5s} {'comp%':>8s} {'ann%':>7s} {'Sh':>6s} "
           f"{'vol%':>6s} {'maxDD%':>7s} {'fundPnL%':>9s} {'pricePnL%':>10s} "
           f"{'cost%':>6s} {'betaBTC':>8s}")
    print(hdr)
    book_stats = {}
    for win in ["SEL", "OOS", "UNSEEN"]:
        s = agg_window(recs, win, btc_idx, ret, dates)
        book_stats[win] = s
        if s:
            print(f"{win:7s} {s['n_days']:5d} {s['compound_pct']:8.2f} "
                  f"{s['ann_pct']:7.2f} {s['sharpe']:6.2f} {s['vol_pct']:6.2f} "
                  f"{s['maxdd_pct']:7.2f} {s['funding_pnl_sum_pct']:9.2f} "
                  f"{s['price_pnl_sum_pct']:10.2f} {s['cost_sum_pct']:6.2f} "
                  f"{s['beta_btc']:+8.3f}")

    # ---- 3) RANDOM dollar-neutral null (same k, same costs, many seeds) -----
    print(f"\n--- RANDOM dollar-neutral null (k={best_k}, {args.n_null_seeds} seeds) ---")
    null_comp = {"SEL": [], "OOS": [], "UNSEEN": []}
    null_fund = {"SEL": [], "OOS": [], "UNSEEN": []}
    for seed in range(args.n_null_seeds):
        nrecs = simulate(dates, ret, fund, sig, best_k, TAKER_FEE_PER_SIDE,
                         random_weight_fn, seed=seed)
        for win in ["SEL", "OOS", "UNSEEN"]:
            ns = agg_window(nrecs, win)
            if ns:
                null_comp[win].append(ns["compound_pct"])
                null_fund[win].append(ns["funding_pnl_sum_pct"])
    for win in ["SEL", "OOS", "UNSEEN"]:
        arr = np.array(null_comp[win])
        farr = np.array(null_fund[win])
        book_c = book_stats[win]["compound_pct"]
        # one-sided p-value: P(null >= funding-ranked book)
        pval = float((arr >= book_c).mean())
        print(f"{win:7s}  book_comp={book_c:7.2f}%   null_comp "
              f"mean={arr.mean():6.2f}%  sd={arr.std():5.2f}  "
              f"[p05={np.percentile(arr,5):6.2f}, p95={np.percentile(arr,95):6.2f}]   "
              f"p(null>=book)={pval:.3f}   "
              f"null_fundPnL_mean={farr.mean():6.2f}%")

    # ---- VERDICT -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    oos = book_stats["OOS"]
    arr_oos = np.array(null_comp["OOS"])
    p_oos = float((arr_oos >= oos["compound_pct"]).mean())
    edge_oos = (oos["compound_pct"] > arr_oos.mean()) and (p_oos < 0.05)
    print(f"OOS: book comp={oos['compound_pct']:.2f}%  vs  null mean={arr_oos.mean():.2f}%"
          f"  (p={p_oos:.3f})  beta_BTC={oos['beta_btc']:+.3f}")
    print(f"OOS funding-PnL={oos['funding_pnl_sum_pct']:.2f}%  "
          f"price-PnL={oos['price_pnl_sum_pct']:.2f}%  cost={oos['cost_sum_pct']:.2f}%")
    if "UNSEEN" in book_stats and book_stats["UNSEEN"]:
        un = book_stats["UNSEEN"]
        arr_un = np.array(null_comp["UNSEEN"])
        p_un = float((arr_un >= un["compound_pct"]).mean())
        print(f"UNSEEN (peeked once): book comp={un['compound_pct']:.2f}%  vs null "
              f"mean={arr_un.mean():.2f}% (p={p_un:.3f})  beta_BTC={un['beta_btc']:+.3f}  "
              f"fundPnL={un['funding_pnl_sum_pct']:.2f}%")
    print(f"\nEDGE@OOS = {'YES' if edge_oos else 'NO (null/decayed)'}")


if __name__ == "__main__":
    main()
