"""xsec_variants_daily_equity ENHANCED — XGB ranker with frontier features.

Extends the baseline by joining 3 validated frontier features into the panel:
    dib_flow_imbalance (IC +0.19 t=13.28 pooled top-10 aligned)
    vol_runs_buy_sell_ratio (IC +0.19 t=13.53)
    hawkes_imbalance_vol_7d (IC -0.04 t=-4.06 vs ret_5d)

Seeds produced (separate from baseline to allow A/B comparison):
    pt_xsec_K5_5_FULL_dneut_enh
    pt_xgb_K3_long_WEALTH40_enh
    pt_cat_K1_stop_no_macro_enh

Baseline comparison: see scratch/xsec_variants_daily_equity.py output.
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle, time, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
SEEDS_DIR = ROOT / 'logs' / 'paper_trader_v2' / 'seeds'
MAKER_RT = 0.08
TRAIN_END = '2024-10-01'
TEST_START = '2025-01-01'
TEST_END = '2026-04-16'
CAPITAL = 10000.0

# Frontier feature data
FRONTIER_DIB_DIR = ROOT / 'data' / 'frontier' / 'dib'
FRONTIER_RUNS_DIR = ROOT / 'data' / 'frontier' / 'runs_bars'
FRONTIER_HAWKES = ROOT / 'data' / 'frontier' / 'hawkes_enh' / 'hawkes_enh_daily.parquet'

META_PKL = ROOT / 'models' / 'meta_labeler' / 'v8_catboost.pkl'
meta_obj = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else None
meta_model = meta_obj['model'] if isinstance(meta_obj, dict) and 'model' in meta_obj else None
meta_features = meta_obj.get('feature_names', []) if isinstance(meta_obj, dict) else []
print(f"[info] meta-labeler loaded: {meta_model is not None}")

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]
print(f"[info] universe: {len(assets)} assets")

btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet', columns=['timestamp','close']).to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close':'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['btc_ret_1d'] = btc_d['close'].pct_change(1)
btc_d['btc_ret_7d'] = btc_d['close'].pct_change(7)
btc_d['date'] = pd.to_datetime(btc_d['date'])

DNA_MAP = {
    'BTC': 'BLUE', 'ETH': 'BLUE',
    'BNB': 'STEADY', 'SOL': 'STEADY', 'XRP': 'STEADY',
    'ADA': 'STEADY', 'AVAX': 'STEADY', 'LINK': 'STEADY', 'LTC': 'STEADY',
    'DOGE': 'VOLATILE', 'TRX': 'STEADY', 'DOT': 'STEADY',
    'NEAR': 'VOLATILE', 'ATOM': 'STEADY',
}


def load_dib_daily_panel() -> pd.DataFrame:
    """Load DIB bars for top-10, aggregate daily, compute flow_imbalance."""
    fps = sorted(glob.glob(str(FRONTIER_DIB_DIR / '*_dib_2025.parquet')))
    frames = []
    for fp in fps:
        asset = Path(fp).stem.replace('_dib_2025', '').replace('USDT', '')
        df = pl.read_parquet(fp).to_pandas()
        ts_arr = df['bar_end_ts'].values
        ts_norm = np.where(ts_arr > 1e15, ts_arr / 1000, ts_arr).astype(np.int64)
        df['date'] = pd.to_datetime(ts_norm, unit='ms').floor('D')
        daily = df.groupby('date').agg({
            'signed_usd': 'sum', 'buy_usd': 'sum', 'sell_usd': 'sum',
        }).reset_index()
        daily['dib_flow_imbalance'] = daily['signed_usd'] / (daily['buy_usd'] + daily['sell_usd']).replace(0, np.nan)
        daily['asset'] = asset
        frames.append(daily[['date', 'asset', 'dib_flow_imbalance']])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_vol_runs_daily_panel() -> pd.DataFrame:
    """Load vol_runs bars, aggregate daily, compute buy/sell ratio."""
    fps = sorted(glob.glob(str(FRONTIER_RUNS_DIR / '*_vol_runs.parquet')))
    frames = []
    for fp in fps:
        asset = Path(fp).stem.replace('_vol_runs', '').replace('USDT', '')
        df = pl.read_parquet(fp).to_pandas()
        ts_arr = df['bar_end_ts'].values
        ts_norm = np.where(ts_arr > 1e15, ts_arr / 1000, ts_arr).astype(np.int64)
        df['date'] = pd.to_datetime(ts_norm, unit='ms').floor('D')
        daily = df.groupby('date').agg({
            'buy_usd': 'sum', 'sell_usd': 'sum',
        }).reset_index()
        daily['vol_runs_buy_sell_ratio'] = daily['buy_usd'] / daily['sell_usd'].replace(0, np.nan)
        daily['asset'] = asset
        frames.append(daily[['date', 'asset', 'vol_runs_buy_sell_ratio']])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_hawkes_enh() -> pd.DataFrame:
    """Load Hawkes enhanced features."""
    if not FRONTIER_HAWKES.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FRONTIER_HAWKES)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'asset', 'hawkes_imbalance_vol_7d']]


# Load frontier features
print("[info] loading frontier features...", flush=True)
t0 = time.time()
dib_panel = load_dib_daily_panel()
vr_panel = load_vol_runs_daily_panel()
hawkes_panel = load_hawkes_enh()
print(f"  DIB: {len(dib_panel)} rows, {dib_panel['asset'].nunique() if not dib_panel.empty else 0} assets")
print(f"  VolRuns: {len(vr_panel)} rows, {vr_panel['asset'].nunique() if not vr_panel.empty else 0} assets")
print(f"  Hawkes: {len(hawkes_panel)} rows, {hawkes_panel['asset'].nunique() if not hawkes_panel.empty else 0} assets")

# Build baseline panel (same as original)
print("[info] building baseline panel...", flush=True)
t0 = time.time()
rows = []
for a in assets:
    fp = DATA/f'{a.lower()}usdt_v50_chimera.parquet'
    try:
        df = pl.read_parquet(fp).to_pandas()
    except Exception:
        continue
    if len(df) < 1000: continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close':'last','open':'first','high':'max','low':'min','volume':'sum'}
    for c in df.columns:
        if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime':
            agg[c] = 'last'
    d = df.groupby('date').agg(agg).reset_index()
    d['ret_1d'] = d['close'].pct_change()
    d['ret_3d'] = d['close'].pct_change(3)
    d['ret_7d'] = d['close'].pct_change(7)
    d['ret_14d'] = d['close'].pct_change(14)
    d['vol_7d'] = d['ret_1d'].rolling(7).std()
    d['vol_14d'] = d['ret_1d'].rolling(14).std()
    d['vol_30d'] = d['ret_1d'].rolling(30).std()
    d['hl'] = (d['high']-d['low'])/d['open']
    d['fwd_1d'] = d['close'].shift(-1)/d['close']-1
    d['fwd_2d'] = d['close'].shift(-2)/d['close']-1
    d['fwd_3d'] = d['close'].shift(-3)/d['close']-1
    d['max_fwd_3d'] = np.maximum.reduce([d['fwd_1d'],d['fwd_2d'],d['fwd_3d']])
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'],d['fwd_2d'],d['fwd_3d']])
    d['bc_ratio'] = (d['volume']/d['volume'].rolling(30).mean()).fillna(1.0)
    d['bc_ratio_trend_3d'] = d['bc_ratio'] - d['bc_ratio'].shift(3)
    d['flow_persistence'] = d.get('norm_flow_imbalance', pd.Series(np.zeros(len(d)))).rolling(7).mean()
    d['vpin_spike'] = ((d.get('norm_vpin', pd.Series(np.zeros(len(d)))) -
                       d.get('norm_vpin', pd.Series(np.zeros(len(d)))).rolling(30).mean()) > 0).astype(float)
    d['asset'] = a
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date','btc_30d','btc_ret_1d','btc_ret_7d']], on='date', how='left')
panel['asset_vs_btc_7d'] = panel['ret_7d'] - panel['btc_ret_7d']
panel['btc_regime'] = np.sign(panel['btc_ret_7d'].fillna(0))
for b in ['BLUE', 'STEADY', 'VOLATILE', 'DEGEN']:
    panel[f'bucket_{b}'] = panel['asset'].apply(lambda a: 1.0 if DNA_MAP.get(a, 'VOLATILE') == b else 0.0)

# ============================ ENHANCEMENT STARTS HERE ============================
# IMPORTANT: DIB + vol_runs only exist 2024-01-01+. Ranker trains pre-2024-10-01
# would see them as all-zero, then at test time they suddenly have values -> erratic.
# Include ONLY hawkes_imbalance_vol_7d (computed from chimera, full 2020-2026 history).
# DIB/vol_runs deferred until we extend their history back to 2020.
print("[info] merging frontier features into panel (Hawkes only, full-history)...", flush=True)
print(f"  panel before merge: {panel.shape}")

if not hawkes_panel.empty:
    panel = panel.merge(hawkes_panel, on=['date', 'asset'], how='left')

for c in ['hawkes_imbalance_vol_7d']:
    if c in panel.columns:
        panel[c] = panel[c].fillna(0)

n_with_haw = int((panel['hawkes_imbalance_vol_7d'].fillna(0) != 0).sum()) if 'hawkes_imbalance_vol_7d' in panel.columns else 0
print(f"  non-zero Hawkes: {n_with_haw}/{len(panel)}")
# ============================ ENHANCEMENT ENDS HERE ==============================

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d','ret_3d','ret_7d','ret_14d','vol_7d','vol_30d','hl','hurst_regime']
# ADD FRONTIER FEATURE (hawkes only, full 2020-2026 history)
feat_list += ['hawkes_imbalance_vol_7d']
feat_list = [c for c in feat_list if c in panel.columns]

panel = panel.sort_values(['date','asset']).reset_index(drop=True)
panel['rank_target'] = panel.groupby('date')['fwd_3d'].rank(pct=True).apply(lambda x: int(x*31))
print(f"[info] ENHANCED panel: {panel.shape}, {len(feat_list)} features (was 30-40 baseline)")


def score_meta(te_df):
    if meta_model is None:
        return np.ones(len(te_df)) * 0.5
    X_cols = []
    for f in meta_features:
        if f in te_df.columns:
            X_cols.append(te_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(te_df)))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(te_df), 1))
    try:
        p = meta_model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
    except Exception:
        return np.ones(len(te_df)) * 0.5


import xgboost as xgb
tr = panel[panel['date'] < TRAIN_END]
te = panel[(panel['date'] >= TEST_START) & (panel['date'] < TEST_END)].copy()
print(f"[info] train={len(tr)} rows, test={len(te)} rows")

tr_g = tr.groupby('date').size().values
print("[info] training XGBRanker with enhanced feature set...", flush=True)
t1 = time.time()
ranker = xgb.XGBRanker(
    objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
    max_depth=6, n_estimators=500, random_state=42, eval_metric='ndcg@5'
)
ranker.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values, group=tr_g, verbose=False)
te['pred'] = ranker.predict(te[feat_list].fillna(0).values)
print(f"[info] trained in {time.time()-t1:.1f}s")

# Feature importance audit
imps = ranker.feature_importances_
imp_df = pd.DataFrame({'feat': feat_list, 'importance': imps}).sort_values('importance', ascending=False)
print("\n[feature importance] top 15:")
print(imp_df.head(15).to_string(index=False))
print("\n[frontier feature importances]:")
for f in ['dib_flow_imbalance', 'vol_runs_buy_sell_ratio', 'hawkes_imbalance_vol_7d']:
    if f in imp_df['feat'].values:
        row = imp_df[imp_df['feat'] == f].iloc[0]
        rank = int((imp_df['feat'] == f).idxmax()) + 1 if f in imp_df['feat'].values else -1
        pos = imp_df.reset_index(drop=True).query(f'feat == "{f}"').index[0] + 1
        print(f"  {f}: importance={row['importance']:.4f}, rank {pos}/{len(imp_df)}")

print("[info] scoring meta-labeler...", flush=True)
te['p_win'] = score_meta(te)


def simulate_daily_equity(
    te: pd.DataFrame,
    K_long: int,
    K_short: int,
    stop: float = 0.10,
    regime_gate: bool = False,
    meta_gate: bool = False,
    meta_thresh: float = 0.45,
    bear_thresh: float = -0.15,
    variant_name: str = "",
) -> pd.DataFrame:
    dates = sorted(te['date'].unique())
    daily_rets = []
    for d in dates:
        grp = te[te['date'] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0)
            continue
        btc30 = float(grp['btc_30d'].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            daily_rets.append(0.0)
            continue
        long_r = short_r = 0.0
        if K_long > 0:
            top = grp.sort_values('pred', ascending=False).head(K_long).copy()
            if meta_gate:
                kept = top[top['p_win'] >= meta_thresh]
                if len(kept) == 0:
                    daily_rets.append(0.0)
                    continue
                top = kept
            rs = []
            for _, r in top.iterrows():
                p = r['fwd_3d']
                if stop and r['min_fwd_3d'] < -stop:
                    p = -stop
                rs.append(p)
            long_r = (np.mean(rs) * 100 - MAKER_RT) if rs else 0.0
        if K_short > 0:
            bot = grp.sort_values('pred', ascending=True).head(K_short)
            rs = []
            for _, r in bot.iterrows():
                p = -r['fwd_3d']
                if stop and r['max_fwd_3d'] > stop:
                    p = -stop
                rs.append(p)
            short_r = (np.mean(rs) * 100 - MAKER_RT) if rs else 0.0
        n_sides = (1 if K_long > 0 else 0) + (1 if K_short > 0 else 0)
        daily_rets.append((long_r + short_r) / max(n_sides, 1) / 3)

    r = np.array(daily_rets) / 100.0
    eq = CAPITAL * np.cumprod(1 + r)
    out = pd.DataFrame({
        'date': dates,
        'bar_idx': np.arange(len(dates)),
        'bar_ts': [int(pd.Timestamp(d).timestamp() * 1000) for d in dates],
        'total_equity': eq,
        'swing_equity': eq,
        'short_equity': np.zeros(len(dates)),
        'total_ret_pct': (eq / CAPITAL - 1) * 100,
        'swing_ret_pct': (eq / CAPITAL - 1) * 100,
        'short_ret_pct': np.zeros(len(dates)),
        'swing_open_positions': np.zeros(len(dates), dtype=int),
    })
    total_ret = (eq[-1] / CAPITAL - 1) * 100
    n_days = len(dates)
    cagr = ((eq[-1] / CAPITAL) ** (365.0 / n_days) - 1) * 100 if n_days > 0 else 0
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    cum_max = np.maximum.accumulate(eq)
    dd = (eq - cum_max) / cum_max
    max_dd = dd.min() * 100
    print(f"  {variant_name:<35} n_days={n_days} total={total_ret:+.2f}% CAGR={cagr:+.1f}% Sharpe={sharpe:+.2f} DD={max_dd:+.2f}%")
    return out


variants = [
    ("xsec_K5_5_FULL_dneut_enh",  {'K_long': 5, 'K_short': 5, 'stop': 0.10,
                                     'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ("xgb_K3_long_WEALTH40_enh",  {'K_long': 3, 'K_short': 0, 'stop': 0.10,
                                     'regime_gate': False, 'meta_gate': False}),
    ("cat_K1_stop_no_macro_enh",  {'K_long': 1, 'K_short': 0, 'stop': 0.10,
                                     'regime_gate': False, 'meta_gate': False}),
]

print("\n=== XSEC ENHANCED VARIANTS (2025-01-01 -> 2026-04-16) ===")
for name, kwargs in variants:
    seed_dir = SEEDS_DIR / f"pt_{name}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    df = simulate_daily_equity(te, variant_name=name, **kwargs)
    df.to_csv(seed_dir / "daily_snapshot.csv", index=False)

print("\n[info] All 3 enhanced variants saved to logs/paper_trader_v2/seeds/pt_xsec_*_enh/")
