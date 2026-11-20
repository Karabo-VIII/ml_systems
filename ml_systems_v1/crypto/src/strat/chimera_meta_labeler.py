"""src/strat/chimera_meta_labeler.py -- EXOGENOUS SIGNAL as the pos-rate lever.

META-FOLD NL-CYCLE 2 deliverable: extend referee_harness with chimera exogenous features.
KEY QUESTION: does exogenous signal forecast 7d direction where price could NOT?

C1 referee verdict:
  - BH pos_rate = 52.3% (2022+, canonical)
  - ML (price only, HGB): AUC ~0.50-0.51, pos_rate <= BH (7d direction UNFORECASTABLE from price)
  - ROUTER wins MEAN (+0.83 vs +0.46) and TAIL (down-wk -3 vs -7) via selection, NOT pos-rate

THIS MODULE:
  1. Loads chimera parquets for all available symbols, aligns to price dates, SKIPS xex_ columns.
  2. Builds exo feature matrix per (date, asset) with AS-OF alignment (bar d = data known at
     end of day d, used to predict d->d+7 which is observed at d+7).
  3. STRICT walk-forward (same label-closure rule as referee_harness: row p eligible at
     predict-pos i iff p+7 <= i-8 trading bars).
  4. Trains chimera-only AND chimera+price models (HistGB + LogReg).
  5. Reports: date-block-permutation AUC (NOT iid), random-slice pos-rate vs BH 52.3%,
     per-family feature importance.
  6. Saves results JSON to runs/strat/chimera_meta_labeler_results.json.

ALIGNMENT HONESTY:
  - chimera parquet 'date' column = the trading day (UTC, same as price bar close).
  - We join on date. A chimera row for date=d contains metrics computed FROM data UP TO
    end-of-day d (funding rate THAT day, buy_vol THAT day, etc.).
  - The label is fwd_7d = C[d+7]/C[d] - 1, known at d+7.
  - So feature at d -> label at d+7: NO forward look. Confirmed by using C.shift(-7)/C-1
    for label and chimera features un-shifted.
  - ETF flow: etf_btc_etf_total_usdm only available post-2024-01-12 (~54% OOS coverage).
    We include but flag low coverage; the ML imputer handles NaNs (HGB native NaN support).

COVERAGE NOTE (confirmed from parquet inspection):
  - fund_rate_mean, fund_rate_z30: 100% from 2020-01-07 (BEST coverage)
  - bs_basis_pct, bs_basis_z30: 98%+ from 2020-01-08
  - buy_vol, sell_vol (order flow): 100% from 2020
  - stbl_total_zscore_30d: 100% from 2020 (stablecoin supply)
  - wh_whale_net_usd: 100% from 2020
  - te_in, te_out (transfer entropy): 96%+ from 2020-03-31
  - dv_dvol_close: BTC/ETH only, ~81%, from 2021-03-25
  - s3_global_lsr (long-short ratio): 99% OOS (starts 2022-01-20)
  - etf_btc_etf_total_usdm: 54% OOS (post-2024-01-12, BTC/ETH only)
  - lob features: <10% OOS (post-2026-01, skip)

No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import glob as glob_mod
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as rh

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CHIMERA_DIR = ROOT.parent / "data" / "processed" / "chimera" / "1d"

# Feature families and which columns to use (skip xex_, skip lob_bgf_, skip bd_, skip lob_l1/l5 -- low coverage)
FAMILY_COLS = {
    "funding": [
        "fund_rate_mean", "fund_rate_z30", "fund_rate_abs_mean",
        "fund_extreme_long_count", "fund_extreme_short_count", "fund_sign_flip",
        "fund_avg_apr", "premium_vol30", "premium_persistence30", "premium_z90",
    ],
    "basis": [
        "bs_basis_pct", "bs_basis_z30", "bs_basis_delta_1d", "bs_basis_delta_3d",
        "bs_basis_xsec_z",
    ],
    "order_flow": [
        "buy_vol", "sell_vol", "norm_flow_imbalance", "norm_vpin",
        "hbr_eta_total", "hbr_eta_buy", "hbr_eta_sell", "hbr_eta_imbalance",
    ],
    "lsr": [
        "s3_global_lsr", "s3_top_pos_lsr", "s3_taker_lsr",
        "s3_smart_vs_retail", "s3_smart_bullish", "s3_smart_bearish",
    ],
    "stablecoin": [
        "stbl_total_zscore_30d", "stbl_total_delta_7d_pct", "stbl_total_delta_30d_pct",
        "stbl_stable_shock", "stbl_stable_crash",
    ],
    "oi_liquidations": [
        "norm_oi_change", "liq_total_usd", "liq_delta_usd",
        "liq_long_z30", "liq_short_z30", "liq_capitulation", "liq_short_panic",
    ],
    "dvol": [
        "dv_dvol_close",
    ],
    "transfer_entropy": [
        "te_in", "te_out", "te_imb", "te_in_btc", "te_out_btc", "te_btc_imb",
    ],
    "whale": [
        "wh_whale_net_usd", "wh_whale_buy_usd", "wh_whale_sell_usd",
    ],
    "etf": [
        "etf_btc_etf_total_usdm", "etf_btc_etf_total_z30",
        "etf_btc_etf_inflow_shock", "etf_btc_etf_outflow_shock",
        "etf_eth_etf_total_usdm",
    ],
    "realized_vol": [
        "rv_rv_5m", "rv_bpv_5m", "rv_jump_frac", "rv_jump_count",
    ],
    "cross_exog": [
        "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
        "xd_cross_return_mean", "xd_cross_vol_mean", "xd_momentum_rank",
        "xrel_rv_rv_5m_xrank", "xrel_rv_bpv_5m_xrank",
        "xrel_hbr_eta_total_xrank", "xrel_liq_long_usd_xrank",
        "xrel_wh_whale_net_usd_xrank",
    ],
}

ALL_EXOG_COLS = [c for cols in FAMILY_COLS.values() for c in cols]


def load_chimera_panel(syms: list[str], start: str, end: str) -> pd.DataFrame:
    """Load chimera parquets for given symbols, return stacked (date, asset) DataFrame.
    AS-OF alignment: chimera row date=d uses features from bar d (no forward fill from future).
    Date column is cast to pd.Timestamp for join with price index.
    SKIPS xex_ columns (name-collision, poisoned per project memory).
    """
    try:
        import polars as pl
        HAS_POLARS = True
    except ImportError:
        HAS_POLARS = False

    frames = []
    parquets = sorted(CHIMERA_DIR.glob("*.parquet"))
    sym_lower = {s.lower(): s for s in syms}

    available_cols = None

    for pf in parquets:
        stem = pf.stem  # e.g. btcusdt_v51_chimera_1d_20260528
        parts = stem.split("_")
        sym_raw = parts[0]  # e.g. btcusdt
        sym_upper = sym_raw.upper() + "T" if not sym_raw.upper().endswith("T") else sym_raw.upper()
        # match: chimera file symname + USDT
        if sym_raw not in sym_lower and sym_upper not in syms:
            # try mapping btcusdt -> BTCUSDT
            candidate = sym_raw.upper()
            if candidate not in syms:
                continue
            sym_key = candidate
        else:
            sym_key = sym_lower.get(sym_raw, sym_upper)
            if sym_key not in syms:
                sym_key = sym_raw.upper()
                if sym_key not in syms:
                    continue

        if HAS_POLARS:
            df_pl = pl.read_parquet(pf)
            # cast date to python date, filter range
            if "date" not in df_pl.columns:
                continue
            df_pd = df_pl.to_pandas()
        else:
            df_pd = pd.read_parquet(pf)

        # cast date
        if not pd.api.types.is_datetime64_any_dtype(df_pd["date"]):
            df_pd["date"] = pd.to_datetime(df_pd["date"])
        else:
            df_pd["date"] = pd.to_datetime(df_pd["date"])

        df_pd = df_pd[(df_pd["date"] >= pd.Timestamp(start)) & (df_pd["date"] < pd.Timestamp(end))].copy()
        if len(df_pd) == 0:
            continue

        df_pd["asset"] = sym_key

        # filter to exog cols that exist (skip xex_ poisoned, skip lob_bgf_, bd_ low coverage)
        exog_present = [c for c in ALL_EXOG_COLS if c in df_pd.columns]
        # make sure no xex_ sneak in
        exog_present = [c for c in exog_present if not c.startswith("xex_")]

        if available_cols is None:
            available_cols = exog_present
        else:
            available_cols = [c for c in available_cols if c in exog_present]

        keep = ["date", "asset"] + exog_present
        frames.append(df_pd[keep].set_index(["date", "asset"]))

    if not frames:
        raise RuntimeError("No chimera parquets matched the loaded symbols")

    panel = pd.concat(frames, axis=0).sort_index()
    # restrict to columns present across ALL loaded assets
    if available_cols:
        panel = panel[[c for c in available_cols if c in panel.columns]]

    print(f"  chimera panel: {len(panel)} rows, {panel.shape[1]} exog features, "
          f"{panel.index.get_level_values('asset').nunique()} assets, "
          f"{start} -> {end}")
    return panel


def build_chimera_features(ind: dict, chimera_panel: pd.DataFrame,
                           include_price: bool = False) -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    """Build (date, asset) feature matrix from chimera exogenous + optionally price features.
    Label: fwd 7-bar return > 0.
    AS-OF: chimera features at bar d are used AS-IS (computed from data up to d).
    """
    C = ind["C"]
    eps = 1e-8

    # Label: fwd 7-bar compound return from bar d (known at d+7)
    fwd = (C.shift(-7) / C - 1)
    fl = fwd.stack(dropna=False); fl.index.names = ["date", "asset"]
    label = (fl > 0).astype(float).where(fl.notna(), np.nan)

    # chimera exog cols
    exog_cols = [c for c in chimera_panel.columns if not c.startswith("xex_")]

    feat = chimera_panel[exog_cols].copy()

    if include_price:
        # price features from referee_harness.build_features (causal)
        panels_price = {
            "dist_sma200": C / (ind["sma200"] + eps) - 1,
            "dist_sma50":  C / (ind["sma50"] + eps) - 1,
            "range_pos":   (C - ind["ll14"]) / ((ind["hh14"] - ind["ll14"]) + eps),
            "rsi14":       ind["rsi14"],
            "vol20":       ind["vol20"],
            "mom7":        ind["mom7"],
            "mom14":       ind["mom14"],
            "mom30":       ind["mom30"],
            "ret1":        ind["ret1"],
            "ret3":        C / C.shift(3) - 1,
        }
        breadth = (C > ind["sma50"]).astype(float).mean(axis=1)
        btc_reg = (C["BTCUSDT"] > ind["sma200"]["BTCUSDT"]).astype(float).fillna(0.0)

        price_stacked = {}
        for name, df_p in panels_price.items():
            s = df_p.stack(dropna=False); s.index.names = ["date", "asset"]; price_stacked[name] = s
        price_df = pd.DataFrame(price_stacked)
        dl = price_df.index.get_level_values("date")
        price_df["breadth"] = breadth.reindex(dl).values
        price_df["btc_regime"] = btc_reg.reindex(dl).values
        price_cols = list(panels_price.keys()) + ["breadth", "btc_regime"]

        # align to chimera index
        price_df = price_df.reindex(feat.index)
        feat = pd.concat([feat, price_df[price_cols]], axis=1)
        all_cols = exog_cols + price_cols
    else:
        all_cols = exog_cols

    return feat, all_cols, label, fl


def strict_chimera_walk_forward(ind: dict, chimera_panel: pd.DataFrame,
                                oos_start: str, include_price: bool = False,
                                retrain_every: int = 90, min_train: int = 300,
                                model_type: str = "hgb") -> pd.DataFrame:
    """Strict walk-forward over chimera features. Same label-closure rule as referee_harness:
    row at date-pos p is train-eligible at predict-pos i iff p+7 <= i-1 (=> p <= i-8).
    HGB handles NaN natively (no imputation needed, no look-ahead in imputer).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    feat, cols, label, fwd = build_chimera_features(ind, chimera_panel, include_price=include_price)

    all_dates = list(ind["C"].index)
    pos_of = {d: i for i, d in enumerate(all_dates)}

    dl_pos = np.array([pos_of.get(d, -1) for d in feat.index.get_level_values("date")])
    valid_pos = dl_pos >= 0
    feat = feat[valid_pos]
    label_aligned = label.reindex(feat.index)
    fwd_aligned = fwd.reindex(feat.index)
    dl_pos = dl_pos[valid_pos]

    Xv = feat[cols].values.astype(float)
    yv = label_aligned.values.astype(float)
    fwd_v = fwd_aligned.values.astype(float)
    valid_lbl = ~np.isnan(yv)
    valid_feat = np.isfinite(Xv).any(axis=1)  # at least 1 non-nan feature

    oos_pos = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(oos_start)), len(all_dates))

    model = None; scaler = None; imputer = None; last_retrain = -10**9
    preds = []

    for i in range(oos_pos, len(all_dates)):
        T = all_dates[i]
        if i - last_retrain >= retrain_every:
            tr = (dl_pos <= i - 8) & valid_feat & valid_lbl
            if tr.sum() >= min_train:
                Xtr = Xv[tr]; ytr = yv[tr]
                if model_type == "hgb":
                    # HGB handles NaN natively -- no imputer needed, no look-ahead
                    model = HistGradientBoostingClassifier(
                        max_iter=300, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=30, l2_regularization=1.0,
                        random_state=42)
                    model.fit(Xtr, ytr); scaler = None; imputer = None
                else:
                    # LogReg: impute then scale (imputer fitted on train only = no leak)
                    imputer = SimpleImputer(strategy="median")
                    Ximp = imputer.fit_transform(Xtr)
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(Ximp)
                    model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                    model.fit(Xs, ytr)
                last_retrain = i

        if model is None:
            continue

        day_mask = (dl_pos == i) & valid_feat
        if not day_mask.any():
            continue

        Xday = Xv[day_mask]
        if scaler is not None and imputer is not None:
            Xday = scaler.transform(imputer.transform(Xday))
        try:
            p = model.predict_proba(Xday)[:, 1]
        except Exception:
            continue

        idx_day = feat.index[day_mask]
        for j, (d, sym) in enumerate(idx_day):
            preds.append({
                "date": d, "asset": sym,
                "prob": float(p[j]),
                "label": float(yv[day_mask][j]) if not np.isnan(yv[day_mask][j]) else np.nan,
                "fwd": float(fwd_v[day_mask][j]) if not np.isnan(fwd_v[day_mask][j]) else np.nan,
            })

    return pd.DataFrame(preds)


def date_block_permutation_auc(pred_df: pd.DataFrame, block_weeks: int = 8,
                                n_perm: int = 2000, seed: int = 42) -> dict:
    """Date-block-permutation AUC (NOT iid).
    Permute BLOCKS of consecutive weeks to destroy temporal autocorrelation while
    preserving within-block structure. One-sided p-value: fraction of permuted AUCs >= observed.
    """
    from sklearn.metrics import roc_auc_score
    va = pred_df[pred_df["label"].notna() & pred_df["prob"].notna()].copy()
    if len(va) < 100:
        return {"auc": np.nan, "p_block_perm": np.nan, "n_obs": len(va)}

    obs_auc = float(roc_auc_score(va["label"], va["prob"]))

    # build blocks of consecutive dates
    dates = pd.to_datetime(va["date"])
    va = va.copy(); va["_date"] = dates
    va = va.sort_values("_date")
    week_bins = (va["_date"].dt.isocalendar().week +
                 va["_date"].dt.year * 53).values
    unique_weeks = np.unique(week_bins)
    # group into block_weeks-sized chunks
    n_blocks = int(np.ceil(len(unique_weeks) / block_weeks))
    week_to_block = {}
    for bi in range(n_blocks):
        for w in unique_weeks[bi * block_weeks: (bi + 1) * block_weeks]:
            week_to_block[w] = bi
    va["_block"] = [week_to_block[w] for w in week_bins]
    block_ids = va["_block"].unique()

    rng = np.random.default_rng(seed)
    perm_aucs = []
    y_true = va["label"].values
    y_prob = va["prob"].values
    block_arr = va["_block"].values

    for _ in range(n_perm):
        perm_order = rng.permutation(len(block_ids))
        # shuffle y_true by reassigning each row to the permuted block's y_true
        # Build lookup: block_id -> sorted row indices
        block_rows = {b: np.where(block_arr == b)[0] for b in block_ids}
        perm_block_ids = block_ids[perm_order]
        new_y = y_true.copy()
        for orig_b, perm_b in zip(block_ids, perm_block_ids):
            orig_rows = block_rows[orig_b]
            perm_rows = block_rows[perm_b]
            n = min(len(orig_rows), len(perm_rows))
            if n == 0:
                continue
            # assign perm_b's labels to orig_b's positions (truncate to shorter)
            new_y[orig_rows[:n]] = y_true[perm_rows[:n]]
        try:
            perm_aucs.append(float(roc_auc_score(new_y, y_prob)))
        except Exception:
            pass

    perm_aucs = np.array(perm_aucs)
    p_val = float((perm_aucs >= obs_auc).mean())
    return {
        "auc": round(obs_auc, 4),
        "perm_auc_mean": round(float(perm_aucs.mean()), 4),
        "perm_auc_p95": round(float(np.percentile(perm_aucs, 95)), 4),
        "p_block_perm": round(p_val, 4),
        "n_obs": len(va),
        "n_blocks": n_blocks,
        "block_weeks": block_weeks,
    }


def feature_importance_by_family(model, cols: list[str], X_val: np.ndarray = None,
                                  y_val: np.ndarray = None, n_repeats: int = 5,
                                  seed: int = 42) -> dict:
    """Extract feature importances and aggregate by family.
    HGB (sklearn 1.7) does not expose feature_importances_ directly;
    we use permutation_importance on a held-in validation window (pre-OOS).
    """
    from sklearn.inspection import permutation_importance as perm_imp
    if X_val is None or y_val is None or len(X_val) < 50:
        return {}
    # Use a subset for speed
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
    Xv = X_val[idx]; yv = y_val[idx]
    valid = ~np.isnan(yv)
    Xv = Xv[valid]; yv = yv[valid]
    if len(Xv) < 50:
        return {}
    try:
        result = perm_imp(model, Xv, yv, n_repeats=n_repeats, random_state=seed,
                          scoring="roc_auc", n_jobs=1)
        imps = result.importances_mean
    except Exception:
        # fallback: use zero importance
        imps = np.zeros(len(cols))

    col_imp = dict(zip(cols, imps))
    family_imp = {}
    for fam, fam_cols in FAMILY_COLS.items():
        total = sum(max(col_imp.get(c, 0.0), 0.0) for c in fam_cols if c in col_imp)
        family_imp[fam] = round(total, 4)
    # individual top-10
    top10 = sorted(col_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"by_family": family_imp, "top10": [(c, round(v, 4)) for c, v in top10]}


def get_last_model(ind: dict, chimera_panel: pd.DataFrame,
                   oos_start: str, include_price: bool = False,
                   retrain_every: int = 90, min_train: int = 300) -> tuple:
    """Re-train to get the last fitted model + feature columns + validation data (for importance).
    Returns (model, cols, X_val, y_val).
    Train window: all data with label closed before oos_start.
    Val window: 90-day pre-OOS window (for permutation importance, not a future leak).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    feat, cols, label, fwd = build_chimera_features(ind, chimera_panel, include_price=include_price)
    all_dates = list(ind["C"].index)
    pos_of = {d: i for i, d in enumerate(all_dates)}
    dl_pos = np.array([pos_of.get(d, -1) for d in feat.index.get_level_values("date")])
    valid_pos = dl_pos >= 0
    feat = feat[valid_pos]; label = label.reindex(feat.index)
    dl_pos = dl_pos[valid_pos]
    Xv = feat[cols].values.astype(float)
    yv = label.values.astype(float)
    valid_lbl = ~np.isnan(yv)
    valid_feat = np.isfinite(Xv).any(axis=1)
    oos_pos = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(oos_start)), len(all_dates))
    # train on all data strictly before oos_start (label closed)
    tr = (dl_pos <= oos_pos - 8) & valid_feat & valid_lbl
    if tr.sum() < min_train:
        return None, cols, None, None
    # val = last 90 days of training window (for permutation importance)
    tr_pos = np.where(tr)[0]
    n_val = min(90 * 10, len(tr_pos) // 4)  # ~10 assets x 90 days or 25% of train
    val_idx = tr_pos[-n_val:]
    tr_idx = tr_pos[:-n_val] if len(tr_pos) > n_val + 100 else tr_pos
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0, random_state=42)
    model.fit(Xv[tr_idx], yv[tr_idx])
    return model, cols, Xv[val_idx], yv[val_idx]


def coverage_report(chimera_panel: pd.DataFrame, oos_start: str) -> dict:
    """Report % non-null per family in OOS region."""
    oos_mask = chimera_panel.index.get_level_values("date") >= pd.Timestamp(oos_start)
    oos = chimera_panel[oos_mask]
    report = {}
    for fam, fam_cols in FAMILY_COLS.items():
        present = [c for c in fam_cols if c in oos.columns]
        if not present:
            report[fam] = {"coverage_pct": 0.0, "n_cols": 0}
            continue
        cov = round(float(oos[present].notna().mean().mean() * 100), 1)
        report[fam] = {"coverage_pct": cov, "n_cols": len(present)}
    return report


def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END = "2026-06-01"
    N = 5000
    SEEDS = [11, 23, 42]
    SLICE_DAYS = 7

    print("=" * 76)
    print("CHIMERA META-LABELER -- exogenous signal as pos-rate lever")
    print(f"OOS: {OOS_START} -> {OOS_END} | N={N} slices | SLICE={SLICE_DAYS}d")
    print("C1 canonical BH pos_rate = 52.3% (win bar to beat)")
    print("=" * 76)

    # 1. Load price data
    print("\n[1] Loading price data (mover_lab)...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    syms = list(C.columns)
    print(f"    {len(syms)} symbols: {syms[:5]}...{syms[-3:]}")

    bh_W = rh.bh_ew_weights(ind)
    bh_b = rh.book_daily_returns(bh_W, ind)

    # BH baseline (canonical re-confirm)
    print("\n[BH] Canonical baseline (re-confirm at N=5000):")
    bh_stats = {s: rh.bh_slice_stats(bh_b, OOS_START, OOS_END, N, SLICE_DAYS, s) for s in SEEDS}
    bh_pr = [bh_stats[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_stats[s]["mean_pct"] for s in SEEDS]
    print(f"  BH pos_rate: seeds={bh_pr} mean={round(float(np.mean(bh_pr)),1)}%")
    print(f"  BH mean_pct: seeds={bh_mn} mean={round(float(np.mean(bh_mn)),2)}%")

    # 2. Load chimera panel
    print("\n[2] Loading chimera panel...")
    chimera = load_chimera_panel(syms, "2020-01-01", OOS_END)

    # 3. Coverage report
    print("\n[3] Exogenous feature coverage in OOS (2022+):")
    cov = coverage_report(chimera, OOS_START)
    for fam, info in sorted(cov.items(), key=lambda x: -x[1]["coverage_pct"]):
        print(f"  {fam:<20}: {info['coverage_pct']:5.1f}% ({info['n_cols']} cols)")

    # 4. Chimera-only walk-forward (HGB)
    print("\n[4] Chimera-only HGB walk-forward (strict label-closure)...")
    pred_exog = strict_chimera_walk_forward(
        ind, chimera, OOS_START, include_price=False, model_type="hgb",
        retrain_every=90, min_train=300)
    print(f"    OOS predictions: {len(pred_exog)}")

    # AUC (iid, then block-permutation)
    from sklearn.metrics import roc_auc_score
    va = pred_exog[pred_exog["label"].notna() & pred_exog["prob"].notna()]
    auc_iid = float(roc_auc_score(va["label"], va["prob"])) if len(va) > 50 else np.nan
    print(f"    AUC (iid, BIASED): {auc_iid:.4f}")

    print("    Running date-block-permutation AUC (N=2000 permutations)...")
    auc_block = date_block_permutation_auc(pred_exog, block_weeks=8, n_perm=2000, seed=42)
    print(f"    AUC (block-perm) = {auc_block['auc']:.4f} | "
          f"perm_mean={auc_block['perm_auc_mean']:.4f} | "
          f"p95_null={auc_block['perm_auc_p95']:.4f} | "
          f"p={auc_block['p_block_perm']:.4f}")

    # Build weight matrix + slice stats
    exog_results = {}
    ml_configs = [(3, 0.50), (2, 0.55), (1, 0.60), (5, 0.50)]
    for (k, thr) in ml_configs:
        W = rh.ml_weight_matrix(pred_exog, ind, k, thr)
        b = rh.book_daily_returns(W, ind)
        prs_list = [rh.slice_stats(b, bh_b, OOS_START, OOS_END, N, SLICE_DAYS, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs_list]
        mn = [x["mean_pct"] for x in prs_list]
        bw = [x["beat_bh_pct"] for x in prs_list]
        dw = [x["down_wk_eng_mean"] for x in prs_list]
        p05 = [x["p05_pct"] for x in prs_list]
        avg_expo = round(float((W.sum(axis=1) > 0).loc[C.index >= OOS_START].mean()), 3)
        res = {
            "pos_rate": round(float(np.mean(pr)), 1), "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "p05_pct": round(float(np.mean(p05)), 2),
            "beat_bh": round(float(np.mean(bw)), 1),
            "down_wk_mean": round(float(np.mean([x for x in dw if x is not None])), 2) if any(x is not None for x in dw) else None,
            "avg_expo": avg_expo,
        }
        exog_results[f"exog_top{k}_thr{str(thr).replace('.','p')}"] = res
        print(f"  EXOG_ONLY top{k} thr{thr}: pos_rate={res['pos_rate']}% (seeds {pr}) "
              f"mean={res['mean_pct']}% p05={res['p05_pct']}% beat_bh={res['beat_bh']}% expo={avg_expo}")

    # 5. Chimera + price walk-forward (HGB)
    print("\n[5] Chimera+PRICE HGB walk-forward (strict label-closure)...")
    pred_both = strict_chimera_walk_forward(
        ind, chimera, OOS_START, include_price=True, model_type="hgb",
        retrain_every=90, min_train=300)
    print(f"    OOS predictions: {len(pred_both)}")

    va_b = pred_both[pred_both["label"].notna() & pred_both["prob"].notna()]
    auc_iid_both = float(roc_auc_score(va_b["label"], va_b["prob"])) if len(va_b) > 50 else np.nan
    print(f"    AUC (iid, BIASED): {auc_iid_both:.4f}")

    print("    Running date-block-permutation AUC (N=2000 permutations)...")
    auc_block_both = date_block_permutation_auc(pred_both, block_weeks=8, n_perm=2000, seed=42)
    print(f"    AUC (block-perm) = {auc_block_both['auc']:.4f} | "
          f"perm_mean={auc_block_both['perm_auc_mean']:.4f} | "
          f"p95_null={auc_block_both['perm_auc_p95']:.4f} | "
          f"p={auc_block_both['p_block_perm']:.4f}")

    both_results = {}
    for (k, thr) in ml_configs:
        W = rh.ml_weight_matrix(pred_both, ind, k, thr)
        b = rh.book_daily_returns(W, ind)
        prs_list = [rh.slice_stats(b, bh_b, OOS_START, OOS_END, N, SLICE_DAYS, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs_list]
        mn = [x["mean_pct"] for x in prs_list]
        bw = [x["beat_bh_pct"] for x in prs_list]
        dw = [x["down_wk_eng_mean"] for x in prs_list]
        p05 = [x["p05_pct"] for x in prs_list]
        avg_expo = round(float((W.sum(axis=1) > 0).loc[C.index >= OOS_START].mean()), 3)
        res = {
            "pos_rate": round(float(np.mean(pr)), 1), "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "p05_pct": round(float(np.mean(p05)), 2),
            "beat_bh": round(float(np.mean(bw)), 1),
            "down_wk_mean": round(float(np.mean([x for x in dw if x is not None])), 2) if any(x is not None for x in dw) else None,
            "avg_expo": avg_expo,
        }
        both_results[f"both_top{k}_thr{str(thr).replace('.','p')}"] = res
        print(f"  EXOG+PRICE top{k} thr{thr}: pos_rate={res['pos_rate']}% (seeds {pr}) "
              f"mean={res['mean_pct']}% p05={res['p05_pct']}% beat_bh={res['beat_bh']}% expo={avg_expo}")

    # 6. Feature importance (permutation importance on pre-OOS validation window)
    print("\n[6] Feature importance (permutation on pre-OOS window, n_repeats=5)...")
    model_imp, imp_cols, Xv_val, yv_val = get_last_model(ind, chimera, OOS_START, include_price=False)
    fi = {}
    if model_imp is not None:
        fi = feature_importance_by_family(model_imp, imp_cols, Xv_val, yv_val, n_repeats=5)
        print("  By family (exog-only model):")
        if "by_family" in fi:
            for fam, score in sorted(fi["by_family"].items(), key=lambda x: -x[1]):
                print(f"    {fam:<20}: {score:.4f}")
            print("  Top-10 individual features:")
            for col, score in fi.get("top10", []):
                print(f"    {col:<35}: {score:.4f}")
        else:
            print("  (no importances available)")
    else:
        print("  Insufficient training data for importance extraction.")

    model_imp_both, imp_cols_both, Xv_val_b, yv_val_b = get_last_model(
        ind, chimera, OOS_START, include_price=True)
    fi_both = {}
    if model_imp_both is not None:
        fi_both = feature_importance_by_family(model_imp_both, imp_cols_both, Xv_val_b, yv_val_b, n_repeats=5)
        print("  Top-10 features (exog+price model):")
        for col, score in fi_both.get("top10", []):
            print(f"    {col:<35}: {score:.4f}")

    # 7. Save results
    out = {
        "meta": {
            "oos_start": OOS_START, "oos_end": OOS_END, "n_slices": N,
            "seeds": SEEDS, "slice_days": SLICE_DAYS,
            "win_bar_bh_posrate_pct": 52.3,
            "canonical_bh_posrate": round(float(np.mean(bh_pr)), 1),
            "canonical_bh_mean_pct": round(float(np.mean(bh_mn)), 2),
        },
        "coverage": cov,
        "exog_only": {
            "auc_iid_biased": round(auc_iid, 4) if not np.isnan(auc_iid) else None,
            "auc_block_perm": auc_block,
            "configs": exog_results,
        },
        "exog_plus_price": {
            "auc_iid_biased": round(auc_iid_both, 4) if not np.isnan(auc_iid_both) else None,
            "auc_block_perm": auc_block_both,
            "configs": both_results,
        },
        "feature_importance_exog_only": fi,
        "feature_importance_exog_plus_price": fi_both,
        "runtime_s": round(time.time() - t0, 1),
    }

    outp = ROOT.parent / "runs" / "strat" / "chimera_meta_labeler_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
