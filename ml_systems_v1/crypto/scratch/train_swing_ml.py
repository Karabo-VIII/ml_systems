"""ML on intraday swing-bounce signals (hl>10% + red close, hold 2d)."""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'models' / 'swing_ml'
OUT.mkdir(parents=True, exist_ok=True)

all_fps = sorted(glob.glob(str(DATA / '*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]

all_rows = []
for a in assets:
    fp = DATA/f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000: continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close':'last','open':'first','high':'max','low':'min','volume':'sum'}
    feat_cols = [c for c in df.columns if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime']
    for fc in feat_cols: agg[fc] = 'last'
    d = df.groupby('date').agg(agg).reset_index()
    d['ret_d'] = d['close'].pct_change()
    d['ret_3d'] = d['close'].pct_change(3)
    d['ret_7d'] = d['close'].pct_change(7)
    d['hl'] = (d['high']-d['low'])/d['open']
    d['asset']=a
    sig = d[(d['hl']>0.10) & (d['close']<d['open'])].copy()
    if len(sig)==0: continue
    fr = []
    for i in sig.index:
        if i+2 >= len(d): fr.append(np.nan)
        else: fr.append((d.iloc[i+2]['close']/d.iloc[i]['close']-1)*100)
    sig['forward_ret_2d'] = fr
    sig['net_pnl'] = sig['forward_ret_2d'] - 0.08
    sig['label'] = (sig['net_pnl']>0.5).astype(int)
    all_rows.append(sig)

all_sig = pd.concat(all_rows, ignore_index=True).dropna(subset=['forward_ret_2d'])
all_sig['date'] = pd.to_datetime(all_sig['date'])
print(f"Total swing-bounce signals: {len(all_sig)}")
print(f"Baseline: mean_net={all_sig['net_pnl'].mean():.2f}%, WR={(all_sig['net_pnl']>0).mean():.2%}")
all_sig = all_sig.sort_values('date').reset_index(drop=True)
tr = all_sig[all_sig['date']<'2024-10-01']
va = all_sig[(all_sig['date']>='2024-10-01')&(all_sig['date']<'2025-03-16')]
te = all_sig[all_sig['date']>='2025-03-16']
print(f"Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")

feat_list = [c for c in all_sig.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_d','ret_3d','ret_7d','hl','hurst_regime']
feat_list = [c for c in feat_list if c in all_sig.columns]

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

X_tr=tr[feat_list].fillna(0).values; y_tr=tr['label'].values
X_va=va[feat_list].fillna(0).values; y_va=va['label'].values
X_te=te[feat_list].fillna(0).values; y_te=te['label'].values

clf = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                         random_seed=42, verbose=0, eval_metric='AUC', early_stopping_rounds=30)
clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=0)
p_te = clf.predict_proba(X_te)[:,1]
print(f"Test AUC: {roc_auc_score(y_te, p_te):.4f}")

te_df = te.copy().reset_index(drop=True)
te_df['p_win'] = p_te
print()
print(f"{'gate':>6}  {'n':>5}  {'wr':>4}  {'mean_net%':>10}  {'frac':>5}")
for g in [0.0,0.40,0.45,0.50,0.55,0.60]:
    sub = te_df[te_df['p_win']>=g]
    if len(sub)<5: continue
    wr=(sub['net_pnl']>0).mean()*100
    ev=sub['net_pnl'].mean()
    print(f"{g:>6.2f}  {len(sub):>5}  {wr:>3.0f}  {ev:>+9.2f}  {len(sub)/len(te_df)*100:>4.0f}%")

(OUT/'swing_ml_v1.pkl').write_bytes(pickle.dumps({'clf':clf,'features':feat_list}))
print(f"\nSaved to {OUT/'swing_ml_v1.pkl'}")
