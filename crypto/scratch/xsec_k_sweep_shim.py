"""Seed 3 xsec variants with DAILY equity curves for portfolio aggregation.

Variants (from dual_axis_ranking_2026_04_22.md):
  1. xsec_K5_5_FULL_dneut   -- Sharpe 3.70, 161%/yr, DD -8%, WF 3/3 positive
  2. xgb_K3_long_WEALTH40   -- Sharpe 2.58, 252%/yr, DD -40%, WF 3/3 positive
  3. cat_K1_stop_no_macro    -- Sharpe 1.69, 334%/yr, DD -78%, WF 3/3 positive

Each outputs logs/paper_trader_v2/seeds/xsec_<variant>/daily_snapshot.csv in
paper_trader_v2-compatible format so the portfolio aggregator can treat
them uniformly.

Training end: 2024-10-01. Test period: 2025-01-01 -> 2026-04-16 (matches
the paper_trader seed window for blend-ready comparison).

Cost: maker RT 0.08% per trade (conservative retail; matches xsec_walkforward).
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle, time, sys, os
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
SEEDS_DIR = ROOT / 'logs' / 'paper_trader_v2' / 'seeds'
MAKER_RT = 0.08  # % per round trip
TRAIN_END = '2024-10-01'
TEST_START = os.environ.get('PT_TEST_START', '2025-01-01')
# TEST_END: default to latest available data if unset (for live paper trading)
# Set PT_TEST_END=YYYY-MM-DD to reproduce a historical snapshot (audit/replay).
TEST_END = os.environ.get('PT_TEST_END', '2099-12-31')
CAPITAL = 10000.0

# Load meta-labeler (CatBoost v8) for FULL stack gates
META_PKL = ROOT / 'models' / 'meta_labeler' / 'v8_catboost.pkl'
meta_obj = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else None
meta_model = meta_obj['model'] if isinstance(meta_obj, dict) and 'model' in meta_obj else None
meta_features = meta_obj.get('feature_names', []) if isinstance(meta_obj, dict) else []

print(f"[info] meta-labeler loaded: {meta_model is not None}")

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]

# Optional universe restriction via PT_UNIVERSE (comma-separated, e.g. "BTC,ETH,SOL,...")
_universe_env = os.environ.get('PT_UNIVERSE', '').strip()
if _universe_env:
    allow = {a.strip().upper() for a in _universe_env.split(',') if a.strip()}
    assets = [a for a in assets if a in allow]
    print(f"[info] universe restricted via PT_UNIVERSE: {len(assets)} assets kept: {sorted(assets)}")

print(f"[info] universe: {len(assets)} assets")

# BTC regime feature
btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet', columns=['timestamp','close']).to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close':'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['btc_ret_1d'] = btc_d['close'].pct_change(1)
btc_d['btc_ret_7d'] = btc_d['close'].pct_change(7)
btc_d['date'] = pd.to_datetime(btc_d['date'])

# DNA bucket map
DNA_MAP = {
    'BTC': 'BLUE', 'ETH': 'BLUE',
    'BNB': 'STEADY', 'SOL': 'STEADY', 'XRP': 'STEADY',
    'ADA': 'STEADY', 'AVAX': 'STEADY', 'LINK': 'STEADY', 'LTC': 'STEADY',
    'DOGE': 'VOLATILE', 'TRX': 'STEADY', 'DOT': 'STEADY',
    'NEAR': 'VOLATILE', 'ATOM': 'STEADY',
}

print("[info] building panel...", flush=True)
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
feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d','ret_3d','ret_7d','ret_14d','vol_7d','vol_30d','hl','hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]

panel = panel.sort_values(['date','asset']).reset_index(drop=True)
panel['rank_target'] = panel.groupby('date')['fwd_3d'].rank(pct=True).apply(lambda x: int(x*31))
print(f"[info] panel built: {panel.shape} in {time.time()-t0:.1f}s, {len(feat_list)} features")

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

# Train ONCE on pre-2024-10-01 data (all variants use same ranker)
tr = panel[panel['date'] < TRAIN_END]
te = panel[(panel['date'] >= TEST_START) & (panel['date'] < TEST_END)].copy()
print(f"[info] train={len(tr)} rows, test={len(te)} rows")

tr_g = tr.groupby('date').size().values
print("[info] training XGBRanker...", flush=True)
t1 = time.time()
ranker = xgb.XGBRanker(
    objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
    max_depth=6, n_estimators=500, random_state=42, eval_metric='ndcg@5'
)
ranker.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values, group=tr_g, verbose=False)
te['pred'] = ranker.predict(te[feat_list].fillna(0).values)
print(f"[info] trained in {time.time()-t1:.1f}s")

# Add meta-labeler p_win for FULL gates
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
    """Run daily backtest, return dataframe with date, daily_ret, equity.

    Mirrors bt() in xsec_xgb_walkforward.py but persists per-day equity."""
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
        # /3 accounts for 3-day hold overlap (each day's position held ~3 days)
        daily_rets.append((long_r + short_r) / max(n_sides, 1) / 3)

    r = np.array(daily_rets) / 100.0  # convert % to decimal
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
    # Summary
    total_ret = (eq[-1] / CAPITAL - 1) * 100
    n_days = len(dates)
    cagr = ((eq[-1] / CAPITAL) ** (365.0 / n_days) - 1) * 100 if n_days > 0 else 0
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    cum_max = np.maximum.accumulate(eq)
    dd = (eq - cum_max) / cum_max
    max_dd = dd.min() * 100
    print(f"  {variant_name:<30} n_days={n_days} total={total_ret:+.2f}% CAGR={cagr:+.1f}% Sharpe={sharpe:+.2f} DD={max_dd:+.2f}%")
    return out


variants = [
    ('xsec_K1_1_FULL_dneut_sweep',  {'K_long': 1, 'K_short': 1, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K3_3_FULL_dneut_sweep',  {'K_long': 3, 'K_short': 3, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K5_5_FULL_dneut_sweep',  {'K_long': 5, 'K_short': 5, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K7_7_FULL_dneut_sweep',  {'K_long': 7, 'K_short': 7, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K10_10_FULL_dneut_sweep',  {'K_long': 10, 'K_short': 10, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K3_long_FULL_dneut_sweep',  {'K_long': 3, 'K_short': 0, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K5_long_FULL_dneut_sweep',  {'K_long': 5, 'K_short': 0, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
    ('xsec_K10_long_FULL_dneut_sweep',  {'K_long': 10, 'K_short': 0, 'stop': 0.10,
                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}),
]
print("\n=== XSEC VARIANTS DAILY EQUITY (2025-01-01 -> 2026-04-16) ===")
for name, kwargs in variants:
    seed_dir = SEEDS_DIR / f"pt_{name}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    df = simulate_daily_equity(te, variant_name=name, **kwargs)
    df.to_csv(seed_dir / "daily_snapshot.csv", index=False)

print("\n[info] All 3 variants persisted to logs/paper_trader_v2/seeds/pt_xsec_*/daily_snapshot.csv")
