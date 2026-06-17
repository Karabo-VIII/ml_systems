"""
Bravo turn 008 -- three probes + one verification pass, using Alpha's turn-007 artifacts:

  (P1) R2 rescue: per-sleeve conditional sizing on the MVP failure.
       Does gating the 4 sleeves INDIVIDUALLY by (stable_z, etf_z) beat flat?
  (P2) 30-min BTC.D probe: does BTC-dominance regime predict blend forward return?
  (P3) Orthogonality regression: blend_ret ~ stable_z + etf_z + funding_z + btc_d_chg
       + cycle_regime_onehot. If R^2 < 0.05, ship the "blend is regime-orthogonal"
       finding as canonical.
  (V1) Verification of Alpha's A11 funding-regime gate (-7.6pp CAGR claim) --
       recompute independently from the saved replay CSV.

SPOT-only interpretation: multiplier = capital allocation % with residue in cash.
No leverage (no mult > 1 unless accompanied by paired floor).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

ROOT = 'c:/Users/karab/Documents/coding/v4_crypto_stystem'

def ann_stats(r, label=''):
    r = np.asarray(r, dtype=float)
    mu = r.mean() * 365
    sd = r.std(ddof=1) * np.sqrt(365)
    sharpe = mu/sd if sd>0 else np.nan
    eq = (1+r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = ((eq-peak)/peak).min()
    days = len(r)
    cagr = eq[-1]**(365.0/days)-1 if days>0 else np.nan
    return dict(label=label, days=days, cagr=cagr, sharpe=sharpe, dd=dd, total=(eq[-1]-1))

# ---------- Load ----------
per_sleeve = pd.read_csv(f'{ROOT}/logs/portfolio_aggregator/recommended_4sleeve_per_sleeve_returns.csv',
                         parse_dates=['date']).sort_values('date').reset_index(drop=True)
SLEEVES = ['xsec_K10_10_FULL_dneut_U50', 'frontier_dib_flow_both', 'asym_breakout', 'asym_vol_expansion']
# convert pct to decimal + fill tail NaN (3 missing days in xsec)
for c in SLEEVES + ['blend_EW']:
    per_sleeve[c] = per_sleeve[c].fillna(0.0) / 100.0

stable = pd.read_parquet(f'{ROOT}/data/frontier/defillama/stable_flow_features.parquet')[['date','total_zscore_30d']].rename(columns={'total_zscore_30d':'stable_z'})
etf = pd.read_parquet(f'{ROOT}/data/frontier/etf/etf_flow_features.parquet')[['date','btc_etf_total_7d_z']].rename(columns={'btc_etf_total_7d_z':'etf_z'})

# BTC.D
btcd = pd.read_csv(f'{ROOT}/logs/frontier/btc_dominance/btc_dominance_daily.csv', parse_dates=['date'])
btcd = btcd[['date','btc_d_proxy','btc_d_sma30','btc_d_sma90','btc_d_chg_30d','regime']].rename(columns={'regime':'btcd_regime'})

# Cycle + funding (from Alpha replays)
cyc = pd.read_csv(f'{ROOT}/logs/frontier/cycle_gate/cycle_gate_replay.csv', parse_dates=['date'])[['date','regime','multiplier']].rename(columns={'regime':'cycle_regime','multiplier':'cycle_mult'})
fund = pd.read_csv(f'{ROOT}/logs/frontier/futures_data_gate/funding_regime_replay.csv', parse_dates=['date'])[['date','fund_z30','multiplier','daily_ret_pct','gated_ret_pct']].rename(columns={'multiplier':'fund_mult','fund_z30':'fund_z'})
fund['daily_ret']  = fund['daily_ret_pct']  / 100.0
fund['gated_ret']  = fund['gated_ret_pct']  / 100.0

# Align all on blend window
df = per_sleeve[['date','blend_EW']+SLEEVES].merge(stable, on='date', how='left').merge(etf, on='date', how='left').merge(btcd, on='date', how='left').merge(cyc, on='date', how='left').merge(fund[['date','fund_z','fund_mult','daily_ret','gated_ret']], on='date', how='left')
for c in ['stable_z','etf_z','btc_d_chg_30d']:
    df[c] = df[c].fillna(0.0)
df['btcd_regime'] = df['btcd_regime'].fillna('UNKNOWN')
df['cycle_regime'] = df['cycle_regime'].fillna('NORMAL')
df['cycle_mult'] = df['cycle_mult'].fillna(1.0)

n = len(df)
print(f'merged: {n} days, {df.date.min().date()} -> {df.date.max().date()}')
print()

# ---------- V1: Verify Alpha's A11 funding-regime gate -7.6pp CAGR claim ----------
print('='*70)
print('V1: Verify Alpha A11 funding-regime gate (-7.6pp CAGR claim)')
print('='*70)
base_v1 = ann_stats(df['daily_ret'].fillna(0), 'flat')
gate_v1 = ann_stats(df['gated_ret'].fillna(0), 'A11_fund_gate')
print(f'  FLAT    : CAGR={base_v1["cagr"]*100:6.2f}%  Sharpe={base_v1["sharpe"]:5.2f}  DD={base_v1["dd"]*100:6.2f}%')
print(f'  A11 gate: CAGR={gate_v1["cagr"]*100:6.2f}%  Sharpe={gate_v1["sharpe"]:5.2f}  DD={gate_v1["dd"]*100:6.2f}%')
delta_cagr_v1 = (gate_v1['cagr']-base_v1['cagr'])*100
print(f'  DELTA   : CAGR {delta_cagr_v1:+.2f}pp  Sharpe {gate_v1["sharpe"]-base_v1["sharpe"]:+.2f}')
print(f'  Alpha claim: -7.6pp CAGR, -0.12 Sharpe')
print(f'  Verdict: {"CONFIRMED" if abs(delta_cagr_v1 - (-7.6)) < 2 else "DISCREPANCY"} (within 2pp tolerance)')
print()

# ---------- P1: R2 rescue -- per-sleeve conditional sizing ----------
print('='*70)
print('P1 (R2 rescue): per-sleeve conditional sizing')
print('='*70)
# Baseline: equal-weight blend
r_blend = df['blend_EW'].values
base = ann_stats(r_blend, 'flat_EW_blend')
print(f'  FLAT EW   : CAGR={base["cagr"]*100:6.2f}%  Sharpe={base["sharpe"]:5.2f}  DD={base["dd"]*100:6.2f}%')

# Per-sleeve signal-to-forward-return correlations
print('\n  Per-sleeve IC to NEXT-DAY return (should identify which signal helps which sleeve):')
for slv in SLEEVES:
    fwd = df[slv].shift(-1)
    cs = df[['stable_z']].assign(y=fwd).dropna().corr().iloc[0,1]
    ce = df[['etf_z']].assign(y=fwd).dropna().corr().iloc[0,1]
    print(f'    {slv:<32s}  stable_z IC={cs:+.4f}  etf_z IC={ce:+.4f}')

# R2 sizing: scale each sleeve individually by its best signal (continuous, small alpha)
def sleeve_scaled_blend(df, sleeve_signals, alpha=0.25, lo=0.5, hi=1.5, weights=None):
    """
    sleeve_signals: dict sleeve_name -> column in df to use as z-signal (None = flat)
    weights: dict sleeve_name -> weight (None = equal)
    """
    if weights is None:
        weights = {s: 0.25 for s in SLEEVES}
    blend_ret = np.zeros(len(df))
    for slv in SLEEVES:
        sig_col = sleeve_signals.get(slv)
        if sig_col is None:
            mult = np.ones(len(df))
        else:
            mult = np.clip(1.0 + alpha * df[sig_col].values, lo, hi)
        blend_ret += weights[slv] * df[slv].values * mult
    return blend_ret

# Sensible signal assignments based on mechanism:
# xsec: broad cross-sec, use stable_z (macro capital flow)
# dib_flow: BTC/ETH flow, use etf_z (BTC+ETH institutional flow)
# asym_breakout: momentum, use stable_z (risk-on tide)
# asym_vol_expansion: vol regime, use etf_z
assignments = [
    ('signal-matched (xsec->stable, dib->etf, brk->stable, volx->etf)', {'xsec_K10_10_FULL_dneut_U50':'stable_z','frontier_dib_flow_both':'etf_z','asym_breakout':'stable_z','asym_vol_expansion':'etf_z'}),
    ('all-sleeves-by-stable_z', {s:'stable_z' for s in SLEEVES}),
    ('all-sleeves-by-etf_z',    {s:'etf_z' for s in SLEEVES}),
    ('xsec-only-by-etf', {'xsec_K10_10_FULL_dneut_U50':'etf_z'}),
]

print()
for label, sig_map in assignments:
    for a_k, lo, hi in [(0.25, 0.5, 1.5), (0.5, 0.25, 1.75)]:
        r = sleeve_scaled_blend(df, sig_map, alpha=a_k, lo=lo, hi=hi)
        s = ann_stats(r, label)
        tag = f'{label}  alpha={a_k} clip=[{lo},{hi}]'
        print(f'  {tag:<80s} CAGR={s["cagr"]*100:6.2f}%  Sh={s["sharpe"]:5.2f}  DD={s["dd"]*100:6.2f}%')
print()

# ---------- P2: 30-min BTC.D probe ----------
print('='*70)
print('P2: BTC.D regime probe')
print('='*70)
# Regime distribution in blend window
print('  BTC.D regime distribution (blend window):')
print(df.groupby('btcd_regime').size().to_string())
print()
print('  Per-regime blend_EW daily return:')
for reg, g in df.groupby('btcd_regime'):
    r = g['blend_EW'].values
    mu = r.mean()*10000
    sd = r.std()*10000
    hit = (r>0).mean()*100
    print(f'    {reg:<15s} n={len(g):4d}  mean={mu:7.1f} bps  std={sd:7.1f} bps  hit={hit:5.1f}%')

# Per-sleeve x regime
print('\n  Per-sleeve mean daily return by BTC.D regime (bps):')
print(f'    {"regime":<15s} {"xsec":>8s} {"dib":>8s} {"brk":>8s} {"volx":>8s}')
for reg, g in df.groupby('btcd_regime'):
    vals = []
    for slv in SLEEVES:
        vals.append(f'{g[slv].mean()*10000:8.1f}')
    print(f'    {reg:<15s} ' + ' '.join(vals))

# Sizing gate: scale to 1.25 during BTC_UP (if xsec favors BTC-concentrated), 0.75 in BTC_DOWN
print('\n  BTC.D sizing-gate variants (on blend_EW):')
gates = {
    'gate v1 (BTC_LEADERSHIP=1.25, ALT_SEASON=0.75, SIDEWAYS=1.0)': {'BTC_LEADERSHIP': 1.25, 'ALT_SEASON': 0.75, 'SIDEWAYS': 1.0},
    'gate v2 (ALT_SEASON=1.25, BTC_LEADERSHIP=0.75, SIDEWAYS=1.0)': {'ALT_SEASON': 1.25, 'BTC_LEADERSHIP': 0.75, 'SIDEWAYS': 1.0},
    'gate v3 (SIDEWAYS=0.5, else 1.0 -- avoid chop)': {'SIDEWAYS': 0.5, 'BTC_LEADERSHIP': 1.0, 'ALT_SEASON': 1.0},
    'gate v4 (ALT_SEASON=1.5 aggressive-alt, else 1.0)': {'ALT_SEASON': 1.5, 'BTC_LEADERSHIP': 1.0, 'SIDEWAYS': 1.0},
    'gate v5 (BTC_LEADERSHIP=1.5 aggressive-BTC, else 1.0)': {'BTC_LEADERSHIP': 1.5, 'ALT_SEASON': 1.0, 'SIDEWAYS': 1.0},
}
for label, regime_mult in gates.items():
    mult = df['btcd_regime'].map(regime_mult).fillna(1.0).values
    r = df['blend_EW'].values * mult
    s = ann_stats(r, label)
    print(f'    {label:<60s} CAGR={s["cagr"]*100:6.2f}%  Sh={s["sharpe"]:5.2f}  DD={s["dd"]*100:6.2f}%')

# Baseline for comparison
base_blend = ann_stats(df['blend_EW'].values, 'flat_blend')
print(f'    {"FLAT (baseline)":<60s} CAGR={base_blend["cagr"]*100:6.2f}%  Sh={base_blend["sharpe"]:5.2f}  DD={base_blend["dd"]*100:6.2f}%')
print()

# ---------- P3: Orthogonality regression ----------
print('='*70)
print('P3: Orthogonality regression')
print('='*70)
# Dependent: blend_EW daily return
# Independents: stable_z, etf_z, fund_z, btc_d_chg_30d, cycle_regime one-hot
Xcols = ['stable_z','etf_z','fund_z','btc_d_chg_30d']
# cycle regime one-hot
cyc_regimes = df['cycle_regime'].fillna('NORMAL').unique()
for reg in cyc_regimes:
    col = f'cyc_{reg}'
    df[col] = (df['cycle_regime']==reg).astype(float)
    Xcols.append(col)
# btcd regime one-hot
for reg in df['btcd_regime'].fillna('UNKNOWN').unique():
    col = f'btcd_{reg}'
    df[col] = (df['btcd_regime']==reg).astype(float)
    Xcols.append(col)

X = df[Xcols].fillna(0.0).values
y = df['blend_EW'].fillna(0.0).values

# OLS via numpy
X1 = np.hstack([np.ones((len(X),1)), X])
# solve beta = (X'X)^-1 X'y
beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
y_hat = X1 @ beta
ss_res = ((y - y_hat)**2).sum()
ss_tot = ((y - y.mean())**2).sum()
r2 = 1 - ss_res/ss_tot

# Per-regressor contributions (univariate R^2)
print('  Univariate R^2 per regressor (blend_EW daily):')
for i, col in enumerate(Xcols):
    x = X[:,i]
    if x.std() > 1e-9:
        b = np.polyfit(x, y, 1)
        y_pred = b[0]*x + b[1]
        r2_u = 1 - ((y-y_pred)**2).sum() / ((y-y.mean())**2).sum()
        corr = np.corrcoef(x, y)[0,1]
        print(f'    {col:<20s}  R^2={r2_u:.4f}  corr={corr:+.4f}')

print(f'\n  Multivariate R^2 (all regressors): {r2:.4f}')
print(f'  Orthogonality finding threshold: R^2 < 0.05')
if r2 < 0.05:
    print(f'  VERDICT: CONFIRMED -- blend is regime-orthogonal (R^2 = {r2:.4f} << 0.05)')
else:
    print(f'  VERDICT: NOT YET -- some regime signal remains (R^2 = {r2:.4f} >= 0.05)')

# Also do 1-day-forward regression (does signal PREDICT next-day return?)
y_fwd = df['blend_EW'].shift(-1).fillna(0.0).values[:-1]
X_fwd = X[:-1]
X1_fwd = np.hstack([np.ones((len(X_fwd),1)), X_fwd])
beta_fwd, *_ = np.linalg.lstsq(X1_fwd, y_fwd, rcond=None)
y_hat_fwd = X1_fwd @ beta_fwd
ss_res_f = ((y_fwd - y_hat_fwd)**2).sum()
ss_tot_f = ((y_fwd - y_fwd.mean())**2).sum()
r2_fwd = 1 - ss_res_f/ss_tot_f
print(f'\n  FORWARD (fwd_1d ~ signals) R^2: {r2_fwd:.4f}')
if r2_fwd < 0.05:
    print(f'  No predictable macro-signal structure in next-day blend return.')
