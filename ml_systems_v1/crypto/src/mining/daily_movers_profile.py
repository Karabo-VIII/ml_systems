"""
daily_movers_profile.py -- DESCRIPTIVE characterization of the daily TOP-25% movers.

Purpose: answer the user's "review and tell me about the daily movers (top 25%)".
This is DESCRIPTIVE review only -- NOT a strategy hunt. RWYB on real chimera data.

Two definitions of "top-25% daily mover" (both reported):
  (a) SIGNED : top quartile of the cross-section each day by signed daily return
               (the long-harvestable up-movers).
  (b) ABS    : top quartile each day by |daily return| (magnitude movers, up or down).

Sections computed (real numbers):
  1. SIZE + FREQUENCY  -- return distribution, count/day, up-vs-down split.
  2. IDIOSYNCRATIC vs BETA -- rolling beta to BTC; residual (idio) fraction of the move.
  3. INTRADAY TIMING   -- hourly bars; share of the daily move in the single biggest hour,
                          overnight-gap vs intraday-burst vs continuous-drift split.
  4. DAY-TO-DAY PERSISTENCE -- mover-status autocorr; mean NEXT-day return | top-mover today
                               (continuation vs reversal), split up vs down.
  5. PRECURSOR STATE   -- state on the day BEFORE a mover day vs a random day
                          (volume / funding / OI / whale / vol-compression).

Regime eras reported separately: 2020-22, 2023-24, 2025-26, plus FULL history.

Usage:
  python -m src.mining.daily_movers_profile --universe u100
  python -m src.mining.daily_movers_profile --universe u50 --no-intraday

NO git commit. NO emoji (Windows cp1252).
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # crypto/ (mining->src->crypto)
DAILY_DIR = os.path.join(ROOT, "data/processed/chimera/1d")
HOURLY_DIR = os.path.join(ROOT, "data/processed/chimera/1h")

# --- era boundaries (by date string) ---
ERAS = [
    ("2020-22", "2020-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
    ("2025-26", "2025-01-01", "2026-12-31"),
    ("FULL",    "2000-01-01", "2099-12-31"),
]


def _pct(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _sym_of(path):
    base = os.path.basename(path)
    return base.split("_")[0].upper()  # e.g. btcusdt_v51_... -> BTCUSDT


def load_daily_panel(universe: str):
    """Return a long DataFrame: [date, sym, ret, plus precursor cols] for universe members.

    ret = same-day close-to-close return realized ON `date` (= close[t]/close[t-1]-1).
    """
    flag = {"u10": "is_u10", "u50": "is_u50", "u100": "is_u100"}[universe]
    precursor_cols = [
        "volume_usd", "fund_rate_mean", "fund_rate_abs_mean", "s3_oi_usd",
        "norm_oi_change", "wh_whale_net_usd", "norm_whale", "liq_total_usd",
        "norm_vol_cluster", "norm_yz_volatility", "norm_funding",
    ]
    base_cols = ["date", "close", flag] + precursor_cols
    frames = []
    for f in sorted(glob.glob(os.path.join(DAILY_DIR, "*.parquet"))):
        try:
            head = pl.read_parquet(f, columns=[flag])
        except Exception:
            continue
        if len(head) == 0 or not bool(head[flag][0]):
            continue
        avail = pl.read_parquet_schema(f)
        cols = [c for c in base_cols if c in avail]
        df = pl.read_parquet(f, columns=cols).sort("date")
        sym = _sym_of(f)
        c = df["close"].to_numpy().astype(float)
        ret = np.full(len(c), np.nan)
        ret[1:] = c[1:] / c[:-1] - 1.0
        out = {"date": df["date"].to_numpy(), "sym": np.array([sym] * len(df)), "ret": ret}
        for pc in precursor_cols:
            out[pc] = df[pc].to_numpy().astype(float) if pc in df.columns else np.full(len(df), np.nan)
        frames.append(pl.DataFrame(out))
    panel = pl.concat(frames, how="vertical_relaxed")
    panel = panel.filter(pl.col("ret").is_finite())
    # guard absurd returns (data glitches): cap at +/-300% for stats robustness flag
    return panel


def assign_mover_flags(panel: pl.DataFrame, min_xsec: int = 8):
    """For each date, flag top-quartile movers under SIGNED and ABS definitions.

    Returns the panel with columns: top_signed (bool), top_abs (bool),
    xsec_n (assets that day), rank_signed_pct, rank_abs_pct.
    Days with < min_xsec assets are dropped (cross-section too thin to rank a quartile).
    """
    panel = panel.with_columns(pl.col("ret").abs().alias("absret"))
    panel = panel.with_columns(pl.len().over("date").alias("xsec_n"))
    panel = panel.filter(pl.col("xsec_n") >= min_xsec)
    # rank within day; rank 1 = smallest. pct in [0,1].
    panel = panel.with_columns([
        (pl.col("ret").rank("ordinal").over("date") / pl.col("xsec_n")).alias("rank_signed_pct"),
        (pl.col("absret").rank("ordinal").over("date") / pl.col("xsec_n")).alias("rank_abs_pct"),
    ])
    panel = panel.with_columns([
        (pl.col("rank_signed_pct") > 0.75).alias("top_signed"),
        (pl.col("rank_abs_pct") > 0.75).alias("top_abs"),
    ])
    return panel


def era_mask(dates_str: np.ndarray, lo: str, hi: str):
    return (dates_str >= lo) & (dates_str <= hi)


# ----------------------------------------------------------------------------
# SECTION 1: SIZE + FREQUENCY
# ----------------------------------------------------------------------------
def section1(panel: pl.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 1 -- SIZE + FREQUENCY of the daily top-25% movers")
    print("=" * 78)
    d = panel["date"].cast(pl.Utf8).to_numpy()
    for name, lo, hi in ERAS:
        m = era_mask(d, lo, hi)
        sub = panel.filter(pl.Series(m))
        if len(sub) == 0:
            continue
        # per-day cross-section size
        per_day = sub.group_by("date").agg([
            pl.len().alias("xsec_n"),
            pl.col("top_signed").sum().alias("n_signed"),
            pl.col("top_abs").sum().alias("n_abs"),
        ])
        xsec = per_day["xsec_n"].to_numpy()
        n_sig = per_day["n_signed"].to_numpy()
        n_abs = per_day["n_abs"].to_numpy()
        # mover returns
        sig = sub.filter(pl.col("top_signed"))["ret"].to_numpy()
        absm = sub.filter(pl.col("top_abs"))
        absm_ret = absm["ret"].to_numpy()
        up = absm_ret[absm_ret > 0]
        dn = absm_ret[absm_ret < 0]
        print(f"\n[{name}]  days={len(per_day)}  median assets/day={np.median(xsec):.0f}  "
              f"(min {xsec.min():.0f}, max {xsec.max():.0f})")
        print(f"  movers/day: SIGNED median={np.median(n_sig):.0f}   ABS median={np.median(n_abs):.0f}")
        print("  (a) SIGNED up-movers (top quartile by signed return) -- the long-harvestable set:")
        print(f"      ret %:  median={np.median(sig)*100:+.2f}  p25={_pct(sig,25)*100:+.2f}  "
              f"p75={_pct(sig,75)*100:+.2f}  p90={_pct(sig,90)*100:+.2f}  max={np.nanmax(sig)*100:+.1f}")
        print(f"      frac of signed-top that are actually >0: {(sig>0).mean()*100:.1f}%")
        print("  (b) ABS magnitude-movers (top quartile by |return|):")
        print(f"      |ret| %: median={_pct(np.abs(absm_ret),50)*100:.2f}  p75={_pct(np.abs(absm_ret),75)*100:.2f}  "
              f"p90={_pct(np.abs(absm_ret),90)*100:.2f}  max={np.nanmax(np.abs(absm_ret))*100:.1f}")
        print(f"      UP/DOWN split: up={len(up)/len(absm_ret)*100:.1f}%  down={len(dn)/len(absm_ret)*100:.1f}%   "
              f"(up median {np.median(up)*100:+.2f}%, down median {np.median(dn)*100:+.2f}%)")


# ----------------------------------------------------------------------------
# SECTION 2: IDIOSYNCRATIC vs BETA
# ----------------------------------------------------------------------------
def section2(panel: pl.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 2 -- IDIOSYNCRATIC vs BETA (rolling 60d beta to BTC)")
    print("=" * 78)
    # build per-date BTC return series
    btc = panel.filter(pl.col("sym") == "BTCUSDT").select(["date", "ret"]).rename({"ret": "btc_ret"})
    pj = panel.join(btc, on="date", how="left")
    # rolling beta per asset over trailing 60 days
    pj = pj.sort(["sym", "date"])
    d = pj["date"].cast(pl.Utf8).to_numpy()
    sym = pj["sym"].to_numpy()
    ret = pj["ret"].to_numpy()
    bret = pj["btc_ret"].to_numpy()
    top_sig = pj["top_signed"].to_numpy()
    top_abs = pj["top_abs"].to_numpy()

    WIN = 60
    beta = np.full(len(ret), np.nan)
    resid = np.full(len(ret), np.nan)
    # group by symbol contiguous
    order = np.argsort(sym, kind="stable")
    # simpler: iterate unique syms
    for s in np.unique(sym):
        idx = np.where(sym == s)[0]
        r = ret[idx]; b = bret[idx]
        for k in range(len(idx)):
            if k < WIN:
                continue
            rw = r[k - WIN:k]; bw = b[k - WIN:k]
            ok = np.isfinite(rw) & np.isfinite(bw)
            if ok.sum() < 20:
                continue
            bv = np.var(bw[ok])
            if bv <= 0:
                continue
            bt = np.cov(rw[ok], bw[ok])[0, 1] / bv
            beta[idx[k]] = bt
            # residual fraction of THIS day's move: |ret - beta*btc_ret| / |ret|
            if np.isfinite(r[k]) and np.isfinite(b[k]) and abs(r[k]) > 1e-9:
                pred = bt * b[k]
                resid[idx[k]] = abs(r[k] - pred) / abs(r[k])

    for name, lo, hi in ERAS:
        m = era_mask(d, lo, hi) & np.isfinite(beta)
        for label, flag in [("SIGNED up-movers", top_sig), ("ABS magnitude-movers", top_abs)]:
            sel = m & flag.astype(bool)
            if sel.sum() < 50:
                continue
            bb = beta[sel]
            rr = resid[sel & np.isfinite(resid)]
            # also: corr of mover-day ret to btc-ret (are they market bursts?)
            mr = ret[sel]; mb = bret[sel]
            ok = np.isfinite(mr) & np.isfinite(mb)
            corr = np.corrcoef(mr[ok], mb[ok])[0, 1] if ok.sum() > 5 else float("nan")
            # fraction of mover-days where the move is SAME sign as BTC that day
            samesign = (np.sign(mr[ok]) == np.sign(mb[ok])).mean()
            print(f"\n[{name}] {label}  (n={sel.sum()})")
            print(f"   rolling beta to BTC: median={np.median(bb):.2f}  p25={_pct(bb,25):.2f}  p75={_pct(bb,75):.2f}")
            print(f"   IDIOSYNCRATIC fraction of the move |ret-beta*btc|/|ret|: median={np.median(rr)*100:.1f}%  "
                  f"p25={_pct(rr,25)*100:.1f}%  p75={_pct(rr,75)*100:.1f}%")
            print(f"   corr(mover-day ret, BTC ret)={corr:.2f}   same-sign-as-BTC days={samesign*100:.0f}%")


# ----------------------------------------------------------------------------
# SECTION 3: INTRADAY TIMING (hourly)
# ----------------------------------------------------------------------------
def section3(panel: pl.DataFrame, universe: str, max_assets: int = 30):
    print("\n" + "=" * 78)
    print("SECTION 3 -- INTRADAY TIMING: when does the daily move happen? (hourly bars)")
    print("=" * 78)
    flag = {"u10": "is_u10", "u50": "is_u50", "u100": "is_u100"}[universe]
    # for the mover days, what % of the daily |move| is in the single biggest hour?
    # also: overnight-gap share = first-bar return contribution.
    # We need, per (sym, date) that was a top_abs mover, the hourly path.
    movers = panel.filter(pl.col("top_abs")).select(["date", "sym", "ret"])
    mover_set = set(zip(movers["sym"].to_numpy(), movers["date"].cast(pl.Utf8).to_numpy()))
    # also a random-day baseline for contrast handled implicitly by sampling mover days only.

    biggest_hour_share = []
    overnight_share = []
    top3_share = []
    n_processed = 0
    files = sorted(glob.glob(os.path.join(HOURLY_DIR, "*.parquet")))
    # restrict to universe members present in panel
    panel_syms = set(panel["sym"].unique().to_list())
    for f in files:
        sym = _sym_of(f)
        if sym not in panel_syms:
            continue
        n_processed += 1
        if n_processed > max_assets:
            break
        try:
            h = pl.read_parquet(f, columns=["timestamp", "date", "open", "high", "low", "close"]).sort("timestamp")
        except Exception:
            continue
        h = h.with_columns(pl.col("date").cast(pl.Utf8).alias("dstr"))
        # only days that are mover-days for this sym
        ds = h["dstr"].to_numpy()
        keep = np.array([(sym, dd) in mover_set for dd in ds])
        if keep.sum() == 0:
            continue
        h = h.filter(pl.Series(keep))
        for (dd,), grp in h.group_by(["dstr"], maintain_order=True):
            o = grp["open"].to_numpy().astype(float)
            c = grp["close"].to_numpy().astype(float)
            if len(c) < 6 or o[0] <= 0 or (c <= 0).any():
                continue
            # IMPORTANT: these "1h" bars are dollar/event bars that do NOT tile the hour
            # cleanly -- open[i+1] != close[i], and the bar-to-bar GAPS carry MORE of the
            # price path than the intra-bar moves (verified ~3x). So a bar's true contribution
            # to the daily path is its CLOSE-to-CLOSE increment (which absorbs the entry gap),
            # NOT open-to-close. First bar's contribution = day-open -> first-close.
            cc = np.empty(len(c))
            cc[0] = np.log(c[0] / o[0])          # day-open to first bar close
            cc[1:] = np.log(c[1:] / c[:-1])      # close-to-close chained (absorbs gaps)
            barlr = cc
            day_lr = np.log(c[-1] / o[0])         # = sum(cc), the day's net move
            if abs(day_lr) < 1e-9:
                continue
            # share of daily move in each bar (signed in direction of the day)
            # biggest single-hour |contribution| relative to total |day move|
            biggest = np.max(np.abs(barlr)) / abs(day_lr)
            biggest_hour_share.append(min(biggest, 5.0))  # cap pathological
            # overnight/open gap: first bar's open vs prior close not available here;
            # proxy "first hour share" = first bar contribution
            overnight_share.append(barlr[0] / day_lr)
            # top-3 hours cumulative |share|
            top3 = np.sum(np.sort(np.abs(barlr))[-3:]) / abs(day_lr)
            top3_share.append(min(top3, 5.0))

    bh = np.array(biggest_hour_share)
    t3 = np.array(top3_share)
    on = np.array(overnight_share)
    print(f"\n  (computed over {len(bh)} mover-days across first {min(n_processed,max_assets)} universe assets)")
    if len(bh):
        print(f"  Largest single HOUR's share of the daily net move:")
        print(f"     median={np.median(bh)*100:.0f}%  p25={_pct(bh,25)*100:.0f}%  p75={_pct(bh,75)*100:.0f}%  p90={_pct(bh,90)*100:.0f}%")
        print(f"  Top-3 hours' combined share of the daily net move:")
        print(f"     median={np.median(t3)*100:.0f}%  p75={_pct(t3,75)*100:.0f}%")
        print(f"  First-hour share (open-of-day burst proxy): median={np.median(on)*100:+.0f}%")
        # classify days
        burst = (bh >= 0.5).mean()       # >=50% in one hour = single burst
        drift = (bh < 0.25).mean()       # <25% in biggest hour = continuous drift
        print(f"  Day-shape mix:  SINGLE-HOUR-BURST (>=50%% in 1h)={burst*100:.0f}%   "
              f"CONTINUOUS-DRIFT (<25%% in biggest hour)={drift*100:.0f}%   middle={ (1-burst-drift)*100:.0f}%")


# ----------------------------------------------------------------------------
# SECTION 4: DAY-TO-DAY PERSISTENCE
# ----------------------------------------------------------------------------
def section4(panel: pl.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 4 -- DAY-TO-DAY PERSISTENCE (continuation vs reversal)")
    print("=" * 78)
    pj = panel.sort(["sym", "date"])
    d = pj["date"].cast(pl.Utf8).to_numpy()
    sym = pj["sym"].to_numpy()
    ret = pj["ret"].to_numpy()
    top_sig = pj["top_signed"].to_numpy().astype(bool)
    top_abs = pj["top_abs"].to_numpy().astype(bool)

    # next-day return per row (shift within sym)
    next_ret = np.full(len(ret), np.nan)
    next_top_abs = np.full(len(ret), np.nan)
    next_top_sig = np.full(len(ret), np.nan)
    for s in np.unique(sym):
        idx = np.where(sym == s)[0]
        # idx is already date-sorted because pj sorted by sym,date
        next_ret[idx[:-1]] = ret[idx[1:]]
        next_top_abs[idx[:-1]] = top_abs[idx[1:]].astype(float)
        next_top_sig[idx[:-1]] = top_sig[idx[1:]].astype(float)

    for name, lo, hi in ERAS:
        m = era_mask(d, lo, hi) & np.isfinite(next_ret)
        # baseline next-day mean for ALL asset-days (the unconditional mean to compare against)
        base = np.nanmean(ret[m])
        # status autocorrelation: P(mover tomorrow | mover today) vs base rate
        for label, flag, nxt in [("ABS magnitude-mover", top_abs, next_top_abs),
                                 ("SIGNED up-mover", top_sig, next_top_sig)]:
            sel = m & flag
            if sel.sum() < 50:
                continue
            base_rate = np.nanmean(nxt[m])  # unconditional P(mover tomorrow)
            cond_rate = np.nanmean(nxt[sel])
            print(f"\n[{name}] {label}  (n today={sel.sum()})")
            print(f"   status persistence: P(mover tomorrow | mover today)={cond_rate*100:.1f}%  "
                  f"vs base rate {base_rate*100:.1f}%  (lift {cond_rate/base_rate:.2f}x)")
        # THE decision number: mean next-day return | top-mover today, split up vs down
        print(f"   unconditional mean next-day return (all asset-days): {base*100:+.3f}%")
        # ABS movers split by direction of today's move
        sel_abs = m & top_abs
        up_today = sel_abs & (ret > 0)
        dn_today = sel_abs & (ret < 0)
        nr_up = next_ret[up_today]; nr_dn = next_ret[dn_today]
        if len(nr_up) > 20:
            print(f"   mean NEXT-day ret | top-ABS-mover UP today  : {np.nanmean(nr_up)*100:+.3f}%  "
                  f"(median {np.nanmedian(nr_up)*100:+.3f}%, n={len(nr_up)})  "
                  f"[{'CONTINUATION' if np.nanmean(nr_up)>base else 'REVERSAL'} vs base]")
        if len(nr_dn) > 20:
            print(f"   mean NEXT-day ret | top-ABS-mover DOWN today: {np.nanmean(nr_dn)*100:+.3f}%  "
                  f"(median {np.nanmedian(nr_dn)*100:+.3f}%, n={len(nr_dn)})  "
                  f"[{'CONTINUATION' if np.nanmean(nr_dn)<base else 'REVERSAL'} vs base]")
        # SIGNED up-movers (the long-harvestable set) next-day
        sel_sig = m & top_sig
        nr_sig = next_ret[sel_sig]
        if len(nr_sig) > 20:
            print(f"   mean NEXT-day ret | top-SIGNED up-mover today: {np.nanmean(nr_sig)*100:+.3f}%  "
                  f"(median {np.nanmedian(nr_sig)*100:+.3f}%, n={len(nr_sig)})  "
                  f"[{'CONTINUATION' if np.nanmean(nr_sig)>base else 'REVERSAL'} vs base]")


# ----------------------------------------------------------------------------
# SECTION 5: PRECURSOR STATE (day before a mover day)
# ----------------------------------------------------------------------------
def section5(panel: pl.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 5 -- PRECURSOR STATE (state the day BEFORE a top-mover day)")
    print("=" * 78)
    pj = panel.sort(["sym", "date"])
    d = pj["date"].cast(pl.Utf8).to_numpy()
    sym = pj["sym"].to_numpy()
    top_abs = pj["top_abs"].to_numpy().astype(bool)

    # precursor candidate features
    feats = ["volume_usd", "fund_rate_abs_mean", "norm_oi_change", "norm_whale",
             "liq_total_usd", "norm_vol_cluster", "norm_yz_volatility", "norm_funding"]
    feats = [f for f in feats if f in pj.columns]
    fvals = {f: pj[f].to_numpy().astype(float) for f in feats}

    # pre_mover[i] = True if the NEXT day (same sym) is a top_abs mover
    pre_mover = np.zeros(len(sym), dtype=bool)
    for s in np.unique(sym):
        idx = np.where(sym == s)[0]
        pre_mover[idx[:-1]] = top_abs[idx[1:]]

    # to compare apples-to-apples, z-score each feature WITHIN sym (cross-time),
    # so a "big volume" is relative to that asset's own history (volume_usd scale differs hugely).
    z = {}
    for f in feats:
        v = fvals[f].copy()
        zz = np.full(len(v), np.nan)
        for s in np.unique(sym):
            idx = np.where(sym == s)[0]
            x = v[idx]
            ok = np.isfinite(x)
            if ok.sum() > 30:
                mu = np.nanmean(x[ok]); sd = np.nanstd(x[ok])
                if sd > 0:
                    zz[idx] = (x - mu) / sd
        z[f] = zz

    print("\n  Per-asset z-scored state. Compares the DAY BEFORE a top-ABS-mover day")
    print("  vs all other (non-pre-mover) asset-days. Delta = pre-mover mean - baseline mean (in std units).")
    for name, lo, hi in ERAS:
        m = era_mask(d, lo, hi)
        pre = m & pre_mover
        base = m & (~pre_mover)
        if pre.sum() < 100:
            continue
        print(f"\n[{name}]  pre-mover days n={pre.sum()}   baseline days n={base.sum()}")
        rows = []
        for f in feats:
            zz = z[f]
            a = zz[pre]; b = zz[base]
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            if len(a) < 50 or len(b) < 50:
                continue
            delta = np.mean(a) - np.mean(b)
            # crude effect-size (Cohen's d-ish)
            sd = np.sqrt((np.var(a) + np.var(b)) / 2) + 1e-9
            cohen = delta / sd
            rows.append((abs(cohen), f, delta, cohen, np.mean(a)))
        rows.sort(reverse=True)
        for _, f, delta, cohen, premean in rows:
            tag = "  <-- separating" if abs(cohen) >= 0.15 else ""
            print(f"   {f:<22} pre-mean(z)={premean:+.3f}  delta={delta:+.3f}sd  effect-d={cohen:+.3f}{tag}")
        print("   (effect-d magnitude: <0.1 negligible, 0.1-0.2 small, >0.2 modest. "
              "Modest+ => some ex-ante separability.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u100", choices=["u10", "u50", "u100"])
    ap.add_argument("--min-xsec", type=int, default=8)
    ap.add_argument("--no-intraday", action="store_true")
    ap.add_argument("--intraday-assets", type=int, default=30)
    args = ap.parse_args()

    print("Loading daily panel for universe", args.universe, "...")
    panel = load_daily_panel(args.universe)
    panel = assign_mover_flags(panel, min_xsec=args.min_xsec)
    n_days = panel.select(pl.col("date").n_unique()).item()
    n_syms = panel.select(pl.col("sym").n_unique()).item()
    print(f"Panel: {len(panel)} asset-days, {n_syms} assets, {n_days} trading days, "
          f"{panel['date'].cast(pl.Utf8).min()} -> {panel['date'].cast(pl.Utf8).max()}")

    section1(panel)
    section2(panel)
    if not args.no_intraday:
        section3(panel, args.universe, max_assets=args.intraday_assets)
    section4(panel)
    section5(panel)
    print("\nDONE.")


if __name__ == "__main__":
    main()
