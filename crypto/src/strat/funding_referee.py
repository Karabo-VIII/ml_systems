"""src/strat/funding_referee.py -- FUNDING-SIGNAL EXTENSION of the canonical referee harness.

META-FOLD NL-CYCLE 2 deliverable: revisit the prior held-out-positive (cross-sectional
funding-dispersion carry) in a LONG-ONLY context, evaluated on the SAME 7d-slice canonical
harness as every other engine.

HYPOTHESES TESTED
-----------------
H1 (SELECTION): Does funding-rate rank predict 7d RELATIVE performance across assets?
    -> Long-only tilt: each week hold the K lowest-funding-rate names (crowded-short =
       squeeze-prone, cheapest carry cost = most underweighted by perp traders).
    -> Judged by: pos_rate + mean_pct on the canonical 7d slicer vs BH 52.3% / +0.46%.

H2 (TIMING / REGIME): Does the cross-sectional MEAN funding level predict the NEXT WEEK's
    EW book return? i.e. enter the market (go long EW) when mean funding is low/negative
    (market underweight), exit to cash when mean funding is extremely high (crowded long,
    overheated).
    -> Judged by: pos_rate + mean_pct (timing around the EW book) vs BH.

H3 (CONTROL -- does it survive?): Same-exposure shuffle on the top strategy. Null keeps
    the book's daily exposure/name-count identical but picks assets at random. If shuffle
    ~ strategy -> edge is structural (de-risked-beta via cash) not genuine selection.

H4 (HONEST REPORT): also run the long-short (LS) version for reference -- but flag clearly
    that LS is LO-BLOCKED per project memory.  We DO NOT route capital to it; we just report
    the number so the "real edge is LO-blocked" claim can be quantified.

LEAK-FREE PROTOCOL (same as referee_harness):
    - Chimera columns are LAGGED 1 bar before use (feature at day d acts on d+1).
    - BH = fixed-EW cadence-invariant (fillna(0)=cash for pre-listing).
    - 7 consecutive trading-day slices, N=5000, seeds=[11,23,42].
    - OOS_START = 2022-01-01 (canonical).
    - SKIP xex_ columns (poisoned per memory).

Run: C:\\Users\\karab\\Documents\\coding\\ml_systems\\.venv\\Scripts\\python.exe -m strat.funding_referee
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as rh

COST = lab.COST
CHIMERA_DIR = ROOT.parent / "data" / "processed" / "chimera" / "1d"

# ===================================================================
# LOAD FUNDING PANEL (cross-sectional, causal)
# ===================================================================
def load_funding_panel(ind: dict) -> dict:
    """Load per-asset funding columns from chimera parquets.

    Returns a dict of DataFrames aligned to ind['C'] index x assets.
    Columns loaded (raw, NOT xex_):
      fund_rate_mean  -- daily mean funding rate (8h settlements averaged)
      fund_rate_z30   -- 30d z-score of fund_rate_mean (dispersion proxy)
      fund_avg_apr    -- annualized carry rate
      fund_extreme_long_count  -- count of extreme-long settlements
      norm_funding    -- chimera's normalized funding feature
    """
    C = ind["C"]
    idx = C.index
    syms = list(C.columns)

    # map parquet files by sym
    files = list(CHIMERA_DIR.glob("*.parquet"))
    # file naming: <sym>_v51_chimera_1d_<date>.parquet; sym is lowercase without 'usdt' prefix handled
    # build map sym -> file
    file_map: dict[str, Path] = {}
    for f in files:
        parts = f.name.split("_")
        sym_lower = parts[0]  # e.g. 'btcusdt'
        sym_upper = sym_lower.upper()
        # prefer latest file if multiple (sorted by date suffix)
        if sym_upper not in file_map or f.name > file_map[sym_upper].name:
            file_map[sym_upper] = f

    wanted_cols = [
        "date",
        "fund_rate_mean",
        "fund_rate_z30",
        "fund_avg_apr",
        "fund_extreme_long_count",
        "fund_extreme_short_count",
        "norm_funding",
        "norm_funding_momentum",
    ]

    panels: dict[str, dict[str, pd.Series]] = {c: {} for c in wanted_cols[1:]}
    coverage_report = []

    for sym in syms:
        if sym not in file_map:
            coverage_report.append(f"  MISS: {sym}")
            continue
        try:
            import polars as pl
            df = pl.read_parquet(file_map[sym])
            # filter wanted cols that actually exist
            avail = [c for c in wanted_cols if c in df.schema]
            df_sub = df.select(avail).to_pandas()
            df_sub["date"] = pd.to_datetime(df_sub["date"])
            df_sub = df_sub.set_index("date").sort_index()
            # reindex to the ind['C'] index
            df_sub = df_sub.reindex(idx)
            for col in wanted_cols[1:]:
                if col in df_sub.columns:
                    panels[col][sym] = df_sub[col]
        except Exception as e:
            coverage_report.append(f"  ERR {sym}: {e}")

    result = {}
    for col, d in panels.items():
        if d:
            result[col] = pd.DataFrame(d, index=idx)

    if coverage_report:
        print("[funding] coverage warnings:")
        for w in coverage_report:
            print(w)
    print(f"[funding] loaded {len(panels['fund_rate_mean'])} assets for fund_rate_mean")
    return result


# ===================================================================
# H1: SELECTION -- long-only funding tilt (low-funding names)
# ===================================================================
def build_funding_selection_weights(
    fund: pd.DataFrame, ind: dict, K: int = 5, rebal_days: int = 7,
    require_gate: bool = True, low_funding: bool = True,
) -> pd.DataFrame:
    """Each rebal date: pick K assets with LOWEST funding rate (crowded-short, squeeze-prone).
    low_funding=False -> pick HIGHEST (for reference / inverted test).
    Gate: only from assets above SMA200 if require_gate=True.
    Causal: fund_rate_mean at day d is LAGGED -> used to set positions for day d+1.
    The actual lag is handled by referee_harness.book_daily_returns (positions shifted 1 bar).
    So W at day d represents a signal observed at close of day d -> acts at d+1 open.
    To be SAFE we lag fund_rate_mean by 1 day before ranking (signal at d-1 -> position at d).
    """
    C = ind["C"]
    gate = ind["gate"] if require_gate else pd.DataFrame(True, index=C.index, columns=C.columns)

    # CAUSAL LAG: fund rate at d-1 is visible at end of d-1 -> rank at end of d-1 -> weight at d
    # book_daily_returns will shift this weight by 1 more bar -> effective trade at d+1
    # So fund rank at d-1 -> trade at d+1 (safe: 2-bar lag from funding observation to fill)
    fund_lag = fund.shift(1)  # fund at d -> visible at d (so this shift = d-1's funding visible at d)

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last_rebal = -999
    cur: dict[str, float] = {}

    for i, d in enumerate(C.index):
        if i - last_rebal >= rebal_days:
            row_fund = fund_lag.loc[d] if d in fund_lag.index else pd.Series(dtype=float)
            row_gate = gate.loc[d]
            # eligible: gated + valid funding
            elig = [
                s for s in C.columns
                if s in row_fund.index and pd.notna(row_fund[s])
                and bool(row_gate.get(s, False))
            ]
            if len(elig) >= K:
                ranked = sorted(elig, key=lambda s: row_fund[s], reverse=not low_funding)
                pick = ranked[:K]
                cur = {s: 1.0 / K for s in pick}
            elif len(elig) > 0:
                cur = {s: 1.0 / len(elig) for s in elig}
            else:
                cur = {}
            last_rebal = i

        for s, w in cur.items():
            if s in W.columns:
                W.loc[d, s] = w

    return W


# ===================================================================
# H2: TIMING -- market-entry gated by funding STATE
# ===================================================================
def build_funding_timing_weights(
    fund: pd.DataFrame, ind: dict,
    entry_pct: float = 0.25,  # enter when cross-sec mean funding < 25th percentile (cheap = underweighted)
    exit_pct: float = 0.85,   # exit to cash when cross-sec mean funding > 85th pctile (overheated)
    lookback: int = 252,       # rolling window for percentile computation (causal)
) -> pd.DataFrame:
    """Timing: hold EW book when cross-sectional mean funding is low (cheap to be long);
    go to cash when funding is extremely high (crowded long, expensive, reversion risk).

    This is a MARKET-TIMING signal, not asset-selection.
    Causal: fund mean at d-1 -> position at d -> executed at d+1 (via 1-bar book lag).
    """
    C = ind["C"]
    gate = ind["gate"]

    # cross-sectional mean funding per day (lag 1 for causality)
    fund_mean = fund.mean(axis=1).shift(1)  # lag: yesterday's cross-sec mean

    # rolling percentile rank (causal: uses only past data up to d-1)
    fund_rank = fund_mean.rolling(lookback, min_periods=60).rank(pct=True)

    # build gated EW as the base allocation
    gate_f = gate.astype(float)
    g_sum = gate_f.sum(axis=1).replace(0, np.nan)
    gated_ew = gate_f.div(g_sum, axis=0).fillna(0.0)

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i, d in enumerate(C.index):
        rank = fund_rank.loc[d] if d in fund_rank.index else np.nan
        if pd.isna(rank):
            # not enough history -> stay in market (default)
            W.loc[d] = gated_ew.loc[d]
        elif rank <= entry_pct:
            # cheap funding -> full gated EW long
            W.loc[d] = gated_ew.loc[d]
        elif rank >= exit_pct:
            # expensive funding -> cash
            W.loc[d] = 0.0
        else:
            # neutral zone -> full gated EW long (conservative: stay in)
            W.loc[d] = gated_ew.loc[d]

    return W


# ===================================================================
# H4: LONG-SHORT VERSION (for reference only, LO-BLOCKED)
# ===================================================================
def build_ls_funding_weights(
    fund: pd.DataFrame, ind: dict, K: int = 5, rebal_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (long_W, short_W) for reference. NOT for capital allocation.
    Long = low-funding names (crowded-short), Short = high-funding names (crowded-long).
    Net book return = long - short (simulated separately, subtract).
    """
    long_W = build_funding_selection_weights(fund, ind, K=K, rebal_days=rebal_days,
                                              require_gate=False, low_funding=True)
    short_W = build_funding_selection_weights(fund, ind, K=K, rebal_days=rebal_days,
                                               require_gate=False, low_funding=False)
    return long_W, short_W


# ===================================================================
# SHUFFLE CONTROL (same-exposure)
# ===================================================================
def shuffle_control_posrate(
    W: pd.DataFrame, ind: dict, oos_start: str, oos_end: str,
    n_slices: int, n_shuffles: int, seed: int
) -> dict:
    """Shuffle control: keep daily exposure+#names, randomize WHICH assets.
    p-value = fraction of shuffles whose pos_rate >= strategy (one-sided).
    """
    C = ind["C"]
    bh_b = rh.book_daily_returns(rh.bh_ew_weights(ind), ind)
    eng_b = rh.book_daily_returns(W, ind)

    rng = np.random.default_rng(seed)
    present = C.notna()
    cols = list(C.columns)
    col_pos = {c: j for j, c in enumerate(cols)}
    daily_expo = W.sum(axis=1)
    n_names = (W > 0).sum(axis=1)
    R = ind["R"].reindex(index=C.index, columns=C.columns).fillna(0.0).values

    sh_pr, sh_mn = [], []
    for _ in range(n_shuffles):
        Wsh = np.zeros((len(C.index), len(cols)))
        for i, d in enumerate(C.index):
            e = daily_expo.iloc[i]; k = int(n_names.iloc[i])
            if e <= 0 or k <= 0:
                continue
            avail = [c for c in cols if bool(present.loc[d, c])]
            if not avail:
                continue
            k = min(k, len(avail))
            pick = rng.choice(len(avail), size=k, replace=False)
            w = e / k
            for pj in pick:
                Wsh[i, col_pos[avail[pj]]] = w
        pos = np.vstack([np.zeros((1, len(cols))), Wsh[:-1]])
        turn = np.abs(np.vstack([pos[:1], np.diff(pos, axis=0)])).sum(axis=1)
        bret_sh = pd.Series((pos * R).sum(axis=1) - turn * (COST / 2.0), index=C.index)
        st = rh.slice_stats(bret_sh, bh_b, oos_start, oos_end, n_slices, 7, seed + _)
        sh_pr.append(st["pos_rate"])
        sh_mn.append(st["mean_pct"])

    eng_stats = rh.slice_stats(eng_b, bh_b, oos_start, oos_end, n_slices, 7, seed)
    eng_pr = eng_stats["pos_rate"]
    eng_mn = eng_stats["mean_pct"]
    sh_pr = np.array(sh_pr); sh_mn = np.array(sh_mn)
    p_pr = float((sh_pr >= eng_pr).mean())
    p_mn = float((sh_mn >= eng_mn).mean())

    return {
        "strategy_pr": eng_pr,
        "strategy_mn": eng_mn,
        "shuffle_pr_mean": round(float(sh_pr.mean()), 1),
        "shuffle_pr_p05": round(float(np.percentile(sh_pr, 5)), 1),
        "shuffle_pr_p95": round(float(np.percentile(sh_pr, 95)), 1),
        "shuffle_mn_mean": round(float(sh_mn.mean()), 2),
        "p_posrate": round(p_pr, 3),
        "p_mean": round(p_mn, 3),
        "n_shuffles": n_shuffles,
        "verdict": "SELECTION SKILL" if (p_pr < 0.10 or p_mn < 0.10) else "NO SELECTION SKILL (de-risked beta only)",
    }


# ===================================================================
# RELATIVE PERFORMANCE TEST (does funding rank predict 7d RELATIVE return?)
# ===================================================================
def test_funding_relative_performance(
    fund: pd.DataFrame, ind: dict, oos_start: str, oos_end: str,
    n_slices: int, seeds: list,
) -> dict:
    """Direct test: at each week start, rank assets by lagged fund_rate_mean (low=rank 1).
    Measure 7d forward return of low-funding tercile vs high-funding tercile (long-only: mean
    of bottom tercile absolute return).
    Spearman rank-IC: fund_rate_rank vs 7d fwd return, OOS only.
    """
    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1  # 7d forward return (label, not used for decisions)

    oos_mask = C.index >= pd.Timestamp(oos_start)
    oos_dates = C.index[oos_mask]

    # fund rank per day (causal lag)
    fund_lag = fund.shift(1)

    ics = []
    low_rets, high_rets, mid_rets = [], [], []

    for d in oos_dates[:-8]:  # need 7 more bars
        fr = fund_lag.loc[d].dropna()
        fw = fwd7.loc[d]
        # assets with both
        common = fr.index.intersection(fw.dropna().index)
        if len(common) < 6:
            continue
        fr_c = fr[common]; fw_c = fw[common]
        # Spearman IC: corr(fund_rank, fwd_ret)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(fr_c.values, fw_c.values)
        ics.append(rho)
        # tercile split
        n = len(common)
        t = n // 3
        ranked = fr_c.sort_values()
        low_names = list(ranked.index[:t])
        high_names = list(ranked.index[-t:])
        mid_names = list(ranked.index[t:-t])
        low_rets.append(float(fw_c[low_names].mean()))
        high_rets.append(float(fw_c[high_names].mean()))
        mid_rets.append(float(fw_c[mid_names].mean()))

    ics = np.array(ics)
    low_rets = np.array(low_rets)
    high_rets = np.array(high_rets)

    from scipy import stats as scipy_stats
    t_stat, p_ic = scipy_stats.ttest_1samp(ics, 0.0)
    t_lo, p_lo = scipy_stats.ttest_1samp(low_rets, 0.0)
    t_hi, p_hi = scipy_stats.ttest_1samp(high_rets, 0.0)

    return {
        "n_dates": len(ics),
        "rank_IC_mean": round(float(ics.mean()), 4),
        "rank_IC_std": round(float(ics.std()), 4),
        "rank_IC_p": round(float(p_ic), 4),
        "low_fund_7d_ret_mean": round(100 * float(low_rets.mean()), 2),
        "low_fund_7d_ret_p": round(float(p_lo), 4),
        "high_fund_7d_ret_mean": round(100 * float(high_rets.mean()), 2),
        "high_fund_7d_ret_p": round(float(p_hi), 4),
        "lo_minus_hi_mean": round(100 * float((low_rets - high_rets).mean()), 2),
        "interpretation": (
            "low-fund OUTPERFORMS high-fund" if ics.mean() < 0 else
            "low-fund UNDERPERFORMS high-fund (crowded-short = squeeze, NOT a buy)"
        ),
    }


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END = "2026-06-01"
    N = 5000
    SEEDS = [11, 23, 42]

    print("=" * 76)
    print("FUNDING REFEREE -- long-only funding-signal extension (NL-CYCLE 2)")
    print(f"OOS: {OOS_START} -> {OOS_END} | N={N} | seeds={SEEDS}")
    print("=" * 76)

    # Load price data
    print("\n[1] Loading price indicators...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    bh_W = rh.bh_ew_weights(ind)
    bh_b = rh.book_daily_returns(bh_W, ind)

    # BH baseline
    bh_stats = {}
    for s in SEEDS:
        bh_stats[s] = rh.bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s)
    bh_pr = [bh_stats[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_stats[s]["mean_pct"] for s in SEEDS]
    print(f"\n[BH] canonical baseline: pos_rate={round(np.mean(bh_pr),1)}% seeds={bh_pr}")
    print(f"     mean_pct={round(np.mean(bh_mn),2)}% seeds={bh_mn}")

    # Load funding panel
    print("\n[2] Loading chimera funding panel...")
    fp = load_funding_panel(ind)
    fund = fp.get("fund_rate_mean")
    if fund is None or fund.empty:
        print("ERROR: no funding data loaded")
        return

    # Coverage summary
    n_assets = (fund > -1e9).sum(axis=1)
    oos_coverage = n_assets[C.index >= pd.Timestamp(OOS_START)]
    print(f"  Funding coverage OOS: min={int(oos_coverage.min())} max={int(oos_coverage.max())} "
          f"median={int(oos_coverage.median())} assets per day")

    # ===========================================================
    # TEST 1: Rank-IC -- does funding rank predict 7d fwd return?
    # ===========================================================
    print("\n[H1a] RANK-IC: does fund_rate rank predict 7d relative return? (OOS Spearman)")
    ic_result = test_funding_relative_performance(fund, ind, OOS_START, OOS_END, N, SEEDS)
    print(f"  n_dates={ic_result['n_dates']}")
    print(f"  Rank IC (fund_rank vs 7d fwd_ret): mean={ic_result['rank_IC_mean']} "
          f"std={ic_result['rank_IC_std']} p={ic_result['rank_IC_p']}")
    print(f"  Low-funding tercile 7d ret: {ic_result['low_fund_7d_ret_mean']}% (p={ic_result['low_fund_7d_ret_p']})")
    print(f"  High-funding tercile 7d ret: {ic_result['high_fund_7d_ret_mean']}% (p={ic_result['high_fund_7d_ret_p']})")
    print(f"  Lo-Hi spread: {ic_result['lo_minus_hi_mean']}%")
    print(f"  Interpretation: {ic_result['interpretation']}")

    # ===========================================================
    # TEST 2: H1 selection -- long-only funding-tilt book
    # ===========================================================
    print("\n[H1b] SELECTION: long-only top-K low-funding book vs BH")
    sel_results = {}
    configs = [
        ("K5_rebal7_gate",   dict(K=5,  rebal_days=7,  require_gate=True,  low_funding=True)),
        ("K5_rebal7_nogate", dict(K=5,  rebal_days=7,  require_gate=False, low_funding=True)),
        ("K3_rebal7_gate",   dict(K=3,  rebal_days=7,  require_gate=True,  low_funding=True)),
        ("K10_rebal7_gate",  dict(K=10, rebal_days=7,  require_gate=True,  low_funding=True)),
        ("K5_rebal3_gate",   dict(K=5,  rebal_days=3,  require_gate=True,  low_funding=True)),
        ("K5_rebal14_gate",  dict(K=5,  rebal_days=14, require_gate=True,  low_funding=True)),
        ("K5_HIGH_gate",     dict(K=5,  rebal_days=7,  require_gate=True,  low_funding=False)),  # inverted control
    ]

    for tag, cfg in configs:
        W = build_funding_selection_weights(fund, ind, **cfg)
        b = book_daily_returns_with_cost(W, ind)
        prs = [rh.slice_stats(b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs]
        mn = [x["mean_pct"] for x in prs]
        bw = [x["beat_bh_pct"] for x in prs]
        expo = float((W.sum(axis=1) > 0).loc[C.index >= pd.Timestamp(OOS_START)].mean())
        sel_results[tag] = {
            "pos_rate": round(float(np.mean(pr)), 1), "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "beat_bh_pct": round(float(np.mean(bw)), 1),
            "avg_expo": round(expo, 2),
        }
        print(f"  {tag:20s}: pos_rate={sel_results[tag]['pos_rate']}% (seeds {pr}) "
              f"mean={sel_results[tag]['mean_pct']}% beat_bh={sel_results[tag]['beat_bh_pct']}% "
              f"expo={sel_results[tag]['avg_expo']:.2f}")

    # ===========================================================
    # TEST 3: H2 -- timing by funding STATE
    # ===========================================================
    print("\n[H2] TIMING: gated-EW entry/exit by funding level (percentile threshold)")
    timing_results = {}
    timing_configs = [
        ("entry25_exit85", dict(entry_pct=0.25, exit_pct=0.85)),
        ("entry15_exit90", dict(entry_pct=0.15, exit_pct=0.90)),
        ("entry35_exit75", dict(entry_pct=0.35, exit_pct=0.75)),
        ("entry50_exit95", dict(entry_pct=0.50, exit_pct=0.95)),  # only exit extreme overheating
    ]
    for tag, cfg in timing_configs:
        W = build_funding_timing_weights(fund, ind, **cfg)
        b = book_daily_returns_with_cost(W, ind)
        prs = [rh.slice_stats(b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs]
        mn = [x["mean_pct"] for x in prs]
        bw = [x["beat_bh_pct"] for x in prs]
        dn = [x.get("down_wk_eng_mean") for x in prs]
        expo = float((W.sum(axis=1) > 0).loc[C.index >= pd.Timestamp(OOS_START)].mean())
        timing_results[tag] = {
            "pos_rate": round(float(np.mean(pr)), 1),
            "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "beat_bh_pct": round(float(np.mean(bw)), 1),
            "down_wk_mean": round(float(np.mean([x for x in dn if x is not None])), 2) if any(x is not None for x in dn) else None,
            "avg_expo": round(expo, 2),
        }
        print(f"  {tag:20s}: pos_rate={timing_results[tag]['pos_rate']}% (seeds {pr}) "
              f"mean={timing_results[tag]['mean_pct']}% beat_bh={timing_results[tag]['beat_bh_pct']}% "
              f"expo={timing_results[tag]['avg_expo']:.2f}")

    # Gated EW (no funding) for reference
    g = ind["gate"].astype(float)
    gW = g.div(g.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gb = rh.book_daily_returns(gW, ind)
    gpr = [rh.slice_stats(gb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    g_pr = [x["pos_rate"] for x in gpr]; g_mn = [x["mean_pct"] for x in gpr]
    print(f"  [ref] gated_EW_no_funding: pos_rate={round(np.mean(g_pr),1)}% "
          f"mean={round(np.mean(g_mn),2)}%")

    # ===========================================================
    # SHUFFLE CONTROL on the best selection strategy
    # ===========================================================
    print("\n[H3] SHUFFLE CONTROL on best selection config (K5_rebal7_gate)")
    best_W = build_funding_selection_weights(fund, ind, K=5, rebal_days=7,
                                              require_gate=True, low_funding=True)
    sh_result = shuffle_control_posrate(best_W, ind, OOS_START, OOS_END,
                                         n_slices=N, n_shuffles=30, seed=99)
    print(f"  strategy: pos_rate={sh_result['strategy_pr']}% mean={sh_result['strategy_mn']}%")
    print(f"  shuffle (n=30): pos_rate mean={sh_result['shuffle_pr_mean']}% "
          f"[p05={sh_result['shuffle_pr_p05']} p95={sh_result['shuffle_pr_p95']}]")
    print(f"  one-sided p(shuffle>=strategy): pos_rate p={sh_result['p_posrate']} mean p={sh_result['p_mean']}")
    print(f"  VERDICT: {sh_result['verdict']}")

    # ===========================================================
    # H4: LS reference (LO-BLOCKED -- reported for completeness)
    # ===========================================================
    print("\n[H4] LONG-SHORT REFERENCE (LO-BLOCKED -- informational only)")
    long_W, short_W = build_ls_funding_weights(fund, ind, K=5, rebal_days=7)
    long_b = book_daily_returns_with_cost(long_W, ind)
    short_b = book_daily_returns_with_cost(short_W, ind)
    ls_b = long_b - short_b  # net LS return (NOT a real book -- dollar-neutral simulation)
    ls_prs = [rh.slice_stats(ls_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    ls_pr = [x["pos_rate"] for x in ls_prs]; ls_mn = [x["mean_pct"] for x in ls_prs]
    print(f"  LS (long low-fund, short high-fund): pos_rate={round(np.mean(ls_pr),1)}% "
          f"seeds={ls_pr} mean={round(np.mean(ls_mn),2)}%")
    print(f"  NOTE: LS is LO-BLOCKED per project memory -- capital NOT routed here.")

    # ===========================================================
    # SUMMARY TABLE
    # ===========================================================
    print("\n" + "=" * 76)
    print("SUMMARY TABLE (canonical 7d slicer, OOS 2022+, N=5000)")
    print("=" * 76)
    print(f"  {'Engine':<30} {'pos_rate':>10} {'mean_pct':>10} {'beat_bh':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'BH_EW (baseline)':<30} {round(np.mean(bh_pr),1):>10.1f}% {round(np.mean(bh_mn),2):>10.2f}%  --")
    for tag, r in sel_results.items():
        print(f"  {'SEL_'+tag:<30} {r['pos_rate']:>10.1f}% {r['mean_pct']:>10.2f}% {r['beat_bh_pct']:>10.1f}%")
    for tag, r in timing_results.items():
        print(f"  {'TIM_'+tag:<30} {r['pos_rate']:>10.1f}% {r['mean_pct']:>10.2f}% {r['beat_bh_pct']:>10.1f}%")
    print(f"  {'LS_K5_ref (LO-BLOCKED)':<30} {round(np.mean(ls_pr),1):>10.1f}% {round(np.mean(ls_mn),2):>10.2f}%  --")

    # ===========================================================
    # SAVE
    # ===========================================================
    out = {
        "meta": {
            "oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS,
            "description": "Funding-signal LO referee extension (NL-CYCLE 2)",
            "runtime_s": round(time.time() - t0, 1),
        },
        "bh": {"pos_rate": round(float(np.mean(bh_pr)), 1), "mean_pct": round(float(np.mean(bh_mn)), 2)},
        "rank_ic": ic_result,
        "selection": sel_results,
        "timing": timing_results,
        "shuffle_control": sh_result,
        "ls_reference": {
            "pos_rate": round(float(np.mean(ls_pr)), 1),
            "mean_pct": round(float(np.mean(ls_mn)), 2),
            "blocked": True,
        },
    }
    outp = ROOT.parent / "runs" / "strat" / "funding_referee_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['meta']['runtime_s']}s)")
    return out


def book_daily_returns_with_cost(W: pd.DataFrame, ind: dict) -> pd.Series:
    """Convenience wrapper -- identical to rh.book_daily_returns."""
    return rh.book_daily_returns(W, ind)


if __name__ == "__main__":
    main()
