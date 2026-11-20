"""
xsec_xgb_walkforward.py + SETUP classifier gating.

Extends the WF-validated XGB + FULL stack to ALSO gate each pick by
setup classifiers (bounce_ml / fade_ml / swing_ml).

Gating rules:
  LONG leg (top-K): require max(p_bounce, p_swing) >= setup_thresh
  SHORT leg (bottom-K): require p_fade >= setup_thresh

Baseline: K=5+5 FULL d-neut WF (regime+stop+meta 0.45) = +314%/Sh 3.70/DD-8%.
Target: setup gate adds 15-30% Sharpe uplift.
"""
import glob
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

# === Load setup classifiers (priority 1) ===
SETUP_MODELS = {}
for name in ['bounce_ml', 'fade_ml', 'swing_ml']:
    fp = ROOT / 'models' / name / f'{name}_v1.pkl'
    if fp.exists():
        SETUP_MODELS[name] = pickle.load(open(fp, 'rb'))
        print(f'  loaded {name}: {len(SETUP_MODELS[name]["features"])} features')

# === Load meta-labeler ===
META_PKL = ROOT / 'models' / 'meta_labeler' / 'v8_catboost.pkl'
meta_obj = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else None
meta_model = meta_obj['model'] if isinstance(meta_obj, dict) and 'model' in meta_obj else None
meta_features = meta_obj.get('feature_names', []) if isinstance(meta_obj, dict) else []
print(f'  loaded meta_labeler v8: {len(meta_features)} features' if meta_model else '  meta_labeler not found')

# === Load panel ===
all_fps = sorted(glob.glob(str(DATA / '*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera', '').upper() for f in all_fps]
btc = pl.read_parquet(DATA / 'btcusdt_v50_chimera.parquet').to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close': 'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['btc_ret_1d'] = btc_d['close'].pct_change()
btc_d['btc_ret_7d'] = btc_d['close'].pct_change(7)
btc_d['date'] = pd.to_datetime(btc_d['date'])

rows = []
for a in assets:
    fp = DATA / f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000:
        continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'volume': 'sum'}
    for c in df.columns:
        if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime':
            agg[c] = 'last'
    d = df.groupby('date').agg(agg).reset_index()
    d['ret_d'] = d['close'].pct_change()  # matches setup models' feature name
    d['ret_1d'] = d['ret_d']
    d['ret_3d'] = d['close'].pct_change(3)
    d['ret_7d'] = d['close'].pct_change(7)
    d['ret_14d'] = d['close'].pct_change(14)
    d['vol_7d'] = d['ret_1d'].rolling(7).std()
    d['vol_14d'] = d['ret_1d'].rolling(14).std()
    d['vol_30d'] = d['ret_1d'].rolling(30).std()
    d['hl'] = (d['high'] - d['low']) / d['open']
    d['fwd_1d'] = d['close'].shift(-1) / d['close'] - 1
    d['fwd_2d'] = d['close'].shift(-2) / d['close'] - 1
    d['fwd_3d'] = d['close'].shift(-3) / d['close'] - 1
    d['max_fwd_3d'] = np.maximum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['bc_ratio'] = (d['volume'] / d['volume'].rolling(30).mean()).fillna(1.0)
    d['bc_ratio_trend_3d'] = d['bc_ratio'] - d['bc_ratio'].shift(3)
    d['flow_persistence'] = d.get('norm_flow_imbalance', pd.Series(np.zeros(len(d)))).rolling(7).mean()
    d['vpin_spike'] = (
        (d.get('norm_vpin', pd.Series(np.zeros(len(d)))) -
         d.get('norm_vpin', pd.Series(np.zeros(len(d)))).rolling(30).mean()) > 0
    ).astype(float)
    d['asset'] = a
    rows.append(d)

panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date', 'btc_30d', 'btc_ret_1d', 'btc_ret_7d']], on='date', how='left')
panel['asset_vs_btc_7d'] = panel['ret_7d'] - panel['btc_ret_7d']
panel['btc_regime'] = np.sign(panel['btc_ret_7d'].fillna(0))

DNA_MAP = {
    'BTC': 'BLUE', 'ETH': 'BLUE',
    'BNB': 'STEADY', 'SOL': 'STEADY', 'XRP': 'STEADY',
    'ADA': 'STEADY', 'AVAX': 'STEADY', 'LINK': 'STEADY', 'LTC': 'STEADY',
    'DOGE': 'VOLATILE', 'TRX': 'STEADY', 'DOT': 'STEADY', 'MATIC': 'VOLATILE',
    'NEAR': 'VOLATILE', 'FTM': 'VOLATILE', 'ATOM': 'STEADY',
}
for b in ['BLUE', 'STEADY', 'VOLATILE', 'DEGEN']:
    panel[f'bucket_{b}'] = panel['asset'].apply(
        lambda a, b=b: 1.0 if DNA_MAP.get(a, 'VOLATILE') == b else 0.0)
feat_list_ranker = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list_ranker += ['ret_1d', 'ret_3d', 'ret_7d', 'ret_14d', 'vol_7d', 'vol_30d', 'hl', 'hurst_regime']
feat_list_ranker = [c for c in feat_list_ranker if c in panel.columns]

panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)
panel['rank_target'] = panel.groupby('date')['fwd_3d'].rank(pct=True).apply(lambda x: int(x * 31))


def _score_setup(model_obj, rows_df):
    """Score p(setup) for each row in rows_df using model_obj."""
    feats = model_obj['features']
    X_cols = []
    for f in feats:
        if f in rows_df.columns:
            X_cols.append(rows_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(rows_df)))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(rows_df), 1))
    try:
        p = model_obj['clf'].predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        return p.flatten()
    except Exception:
        return np.ones(len(rows_df)) * 0.5


def _score_meta(rows_df):
    if meta_model is None:
        return np.ones(len(rows_df)) * 0.5
    X_cols = []
    for f in meta_features:
        if f in rows_df.columns:
            X_cols.append(rows_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(rows_df)))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(rows_df), 1))
    try:
        p = meta_model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
    except Exception:
        return np.ones(len(rows_df)) * 0.5


import xgboost as xgb


def train_test_window(train_end, test_start, test_end, label):
    tr = panel[panel['date'] < train_end].copy()
    te = panel[(panel['date'] >= test_start) & (panel['date'] < test_end)].copy()
    if len(tr) < 5000 or len(te) < 500:
        return None
    tr_g = tr.groupby('date').size().values
    ranker = xgb.XGBRanker(
        objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
        max_depth=6, n_estimators=500, random_state=42, eval_metric='ndcg@5',
    )
    ranker.fit(tr[feat_list_ranker].fillna(0).values, tr['rank_target'].values,
               group=tr_g, verbose=False)
    te['pred'] = ranker.predict(te[feat_list_ranker].fillna(0).values)
    return te


def bt(te, K_long, K_short, stop=0.10,
       regime_gate=False, meta_gate=False, meta_thresh=0.45,
       setup_gate=False, setup_thresh=0.45,
       bear_thresh=-0.15):
    """Backtest with FULL stack + optional setup classifier gate."""
    daily = []
    stats = {'n_regime_blk': 0, 'n_meta_rej_long': 0, 'n_meta_rej_short': 0,
             'n_setup_rej_long': 0, 'n_setup_rej_short': 0, 'n_active': 0}
    for d in sorted(te['date'].unique()):
        grp = te[te['date'] == d]
        if len(grp) < K_long + K_short:
            daily.append(0); continue
        btc30 = float(grp['btc_30d'].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            stats['n_regime_blk'] += 1
            daily.append(0); continue
        long_r = short_r = 0
        if K_long > 0:
            top = grp.sort_values('pred', ascending=False).head(K_long).copy()
            if meta_gate:
                p_win = _score_meta(top)
                top = top.assign(p_win=p_win)
                kept = top[top['p_win'] >= meta_thresh]
                stats['n_meta_rej_long'] += len(top) - len(kept)
                top = kept
            if setup_gate and len(top) > 0:
                # Require max(p_bounce, p_swing) >= setup_thresh for long picks
                p_b = _score_setup(SETUP_MODELS['bounce_ml'], top) if 'bounce_ml' in SETUP_MODELS else np.ones(len(top))
                p_s = _score_setup(SETUP_MODELS['swing_ml'], top) if 'swing_ml' in SETUP_MODELS else np.ones(len(top))
                p_setup = np.maximum(p_b, p_s)
                mask = p_setup >= setup_thresh
                stats['n_setup_rej_long'] += (~mask).sum()
                top = top[mask]
            if len(top) > 0:
                rs = []
                for _, r in top.iterrows():
                    p = r['fwd_3d']
                    if stop and r['min_fwd_3d'] < -stop:
                        p = -stop
                    rs.append(p)
                long_r = np.mean(rs) * 100 - MAKER_RT
        if K_short > 0:
            bot = grp.sort_values('pred', ascending=True).head(K_short).copy()
            if setup_gate and len(bot) > 0:
                p_f = _score_setup(SETUP_MODELS['fade_ml'], bot) if 'fade_ml' in SETUP_MODELS else np.ones(len(bot))
                mask = p_f >= setup_thresh
                stats['n_setup_rej_short'] += (~mask).sum()
                bot = bot[mask]
            if len(bot) > 0:
                rs = []
                for _, r in bot.iterrows():
                    p = -r['fwd_3d']
                    if stop and r['max_fwd_3d'] > stop:
                        p = -stop
                    rs.append(p)
                short_r = np.mean(rs) * 100 - MAKER_RT
        n_s = (1 if K_long > 0 and long_r != 0 else 0) + (1 if K_short > 0 and short_r != 0 else 0)
        if n_s > 0:
            daily.append((long_r + short_r) / n_s / 3)
            stats['n_active'] += 1
        else:
            daily.append(0)
    r = np.array(daily)
    if r.std() == 0:
        return {'total': 0, 'sharpe': 0, 'dd': 0, **stats}
    eq = np.cumprod(1 + r / 100)
    cm = np.maximum.accumulate(eq)
    return {'total': (eq[-1] - 1) * 100,
            'sharpe': r.mean() * 252 / (r.std() * np.sqrt(252)),
            'dd': ((eq - cm) / cm).min() * 100,
            **stats}


print('\n=== Priority 1 Test: Setup-classifier gate on xsec K=5+5 FULL d-neut ===')
print(f"{'window':<30} {'K=5+5 FULL (base)':>28} {'K=5+5 FULL+SETUP 45':>28} {'K=5+5 FULL+SETUP 55':>28}")

windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25 (5mo)'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25 (6mo)'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26 (7mo)'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]

for train_end, ts, te_end, label in windows:
    te = train_test_window(train_end, ts, te_end, label)
    if te is None:
        continue
    # Baseline: K=5+5 FULL (regime+stop+meta) — the current champion
    r_base = bt(te, 5, 5, stop=0.10, regime_gate=True,
                meta_gate=True, meta_thresh=0.45)
    # + setup gate at threshold 0.45
    r_setup45 = bt(te, 5, 5, stop=0.10, regime_gate=True,
                    meta_gate=True, meta_thresh=0.45,
                    setup_gate=True, setup_thresh=0.45)
    # + setup gate at 0.55 (stricter)
    r_setup55 = bt(te, 5, 5, stop=0.10, regime_gate=True,
                    meta_gate=True, meta_thresh=0.45,
                    setup_gate=True, setup_thresh=0.55)
    print(f"{label:<30} {r_base['total']:>+11.0f}% Sh{r_base['sharpe']:>+.2f} DD{r_base['dd']:>+.0f}% "
          f"{r_setup45['total']:>+11.0f}% Sh{r_setup45['sharpe']:>+.2f} DD{r_setup45['dd']:>+.0f}% "
          f"{r_setup55['total']:>+11.0f}% Sh{r_setup55['sharpe']:>+.2f} DD{r_setup55['dd']:>+.0f}%")
    # Print stats on rejection rates for combined
    if 'COMBINED' in label:
        print(f"  setup45 stats: long_rej={r_setup45['n_setup_rej_long']}, "
              f"short_rej={r_setup45['n_setup_rej_short']}, active_days={r_setup45['n_active']}")
        print(f"  setup55 stats: long_rej={r_setup55['n_setup_rej_long']}, "
              f"short_rej={r_setup55['n_setup_rej_short']}, active_days={r_setup55['n_active']}")
