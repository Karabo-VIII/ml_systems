"""
Bravo turn 006 - Supply-flow Meta-Multiplier Probe (MVP).

Question: Does using stable-supply + ETF inflow as sizing multipliers on the
4-sleeve champion blend improve Sharpe / CAGR versus flat sizing?

Signals used (continuous z-scores):
  - stable_flow: total_zscore_30d from defillama stable aggregates
  - etf_flow: btc_etf_total_7d_z

DIB flow deferred to next iteration (per-asset per-year panel rollup pending).

Regime construction (MVP):
  risk_on_count = (stable_z > 0) + (etf_z > 0)    # 0, 1, or 2
  sizing_multiplier map:
    2 / 2 risk-on -> 1.5x
    1 / 2 risk-on -> 1.0x
    0 / 2 risk-on -> 0.5x

Metric: scaled blend daily returns -> Sharpe, CAGR, DD, Sortino.
Compare scaled vs flat.

SPOT-only interpretation: the "multiplier" is applied by increasing/decreasing
capital deployed to the alpha stack, with idle capital in stables (no leverage,
just cash vs risk).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

ROOT = 'c:/Users/karab/Documents/coding/ml_systems'

# Load blend daily returns
blend = pd.read_csv(f'{ROOT}/logs/portfolio_aggregator/recommended_4sleeve_alpha_stack_daily.csv',
                    parse_dates=['date']).sort_values('date').reset_index(drop=True)
# NOTE: portfolio_ret_pct is CUMULATIVE (total_ret_from_inception), not daily.
# Derive daily returns from the equity curve.
blend['ret'] = blend['portfolio_equity'].pct_change().fillna(0.0)
print(f'blend: {len(blend)} days, {blend.date.min().date()} -> {blend.date.max().date()}')

# Load signals
stable = pd.read_parquet(f'{ROOT}/data/frontier/defillama/stable_flow_features.parquet')
stable = stable[['date', 'total_zscore_30d', 'usdt_zscore_30d']].rename(columns={'total_zscore_30d':'stable_z','usdt_zscore_30d':'usdt_z'})
print(f'stable: {len(stable)} days')

etf = pd.read_parquet(f'{ROOT}/data/frontier/etf/etf_flow_features.parquet')
etf = etf[['date', 'btc_etf_total_7d_z']].rename(columns={'btc_etf_total_7d_z':'etf_z'})
print(f'etf: {len(etf)} days')

# Align
df = blend[['date','ret']].merge(stable, on='date', how='left').merge(etf, on='date', how='left')
df['stable_z'] = df['stable_z'].fillna(0.0)
df['etf_z']    = df['etf_z'].fillna(0.0)
print(f'merged: {len(df)} days, NA stable={df.stable_z.isna().sum()} NA etf={df.etf_z.isna().sum()}')

# Regime count: 0/1/2 of {stable_z > 0, etf_z > 0}
df['stable_on'] = (df['stable_z'] > 0).astype(int)
df['etf_on']    = (df['etf_z'] > 0).astype(int)
df['risk_on']   = df['stable_on'] + df['etf_on']

# Multiplier map (MVP)
mult_map = {0: 0.5, 1: 1.0, 2: 1.5}
df['mult'] = df['risk_on'].map(mult_map)
df['ret_scaled'] = df['ret'] * df['mult']

# Metrics
def stats(rets, name):
    r = np.asarray(rets, dtype=float)
    mu = r.mean() * 365
    sd = r.std(ddof=1) * np.sqrt(365)
    sharpe = mu / sd if sd > 0 else np.nan
    eq = (1 + r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    days = len(r)
    total_ret = eq[-1] - 1
    cagr = (eq[-1] ** (365.0/days)) - 1 if days > 0 else np.nan
    neg = r[r < 0]
    sortino = (mu / (neg.std(ddof=1)*np.sqrt(365))) if len(neg) > 1 and neg.std(ddof=1) > 0 else np.nan
    return dict(name=name, days=days, cagr=cagr, sharpe=sharpe, sortino=sortino, dd=dd, total_ret=total_ret)

base = stats(df['ret'], 'flat')
scaled = stats(df['ret_scaled'], 'meta_scaled')

print('\n=== HEAD ===')
print(df[['date','ret','stable_z','etf_z','stable_on','etf_on','risk_on','mult']].head(3))
print('\n=== TAIL ===')
print(df[['date','ret','stable_z','etf_z','risk_on','mult']].tail(3))

# Regime distribution
print('\n=== Regime distribution ===')
print(df['risk_on'].value_counts().sort_index())

# Per-regime return stats
print('\n=== Per-regime mean daily return (bps) ===')
for k, g in df.groupby('risk_on'):
    n = len(g)
    m = g['ret'].mean() * 10000
    s = g['ret'].std() * 10000
    pos = (g['ret'] > 0).mean() * 100
    print(f'  risk_on={k}  n={n:4d}  mean_bps={m:7.1f}  std_bps={s:7.1f}  hit_rate={pos:5.1f}%')

print('\n=== Flat vs meta-scaled ===')
b_cagr, b_sh, b_sor, b_dd, b_tot = base['cagr'], base['sharpe'], base['sortino'], base['dd'], base['total_ret']
s_cagr, s_sh, s_sor, s_dd, s_tot = scaled['cagr'], scaled['sharpe'], scaled['sortino'], scaled['dd'], scaled['total_ret']
print(f'  FLAT   : CAGR={b_cagr*100:6.2f}%  Sharpe={b_sh:5.2f}  Sortino={b_sor:6.2f}  DD={b_dd*100:6.2f}%  totalRet={b_tot*100:6.2f}%')
print(f'  SCALED : CAGR={s_cagr*100:6.2f}%  Sharpe={s_sh:5.2f}  Sortino={s_sor:6.2f}  DD={s_dd*100:6.2f}%  totalRet={s_tot*100:6.2f}%')
print(f'  DELTA  : CAGR {(s_cagr-b_cagr)*100:+.2f} pp  Sharpe {s_sh-b_sh:+.2f}  DD {(s_dd-b_dd)*100:+.2f} pp')

# Variants: also try a 3-level aggressive mapping and a conservative mapping
variants = {
    'conservative (0.75/1.0/1.25)': {0: 0.75, 1: 1.0, 2: 1.25},
    'aggressive (0.25/1.0/1.75)':   {0: 0.25, 1: 1.0, 2: 1.75},
    'on_off (0.0/1.0/1.5)':         {0: 0.0,  1: 1.0, 2: 1.5},
}

print('\n=== Variant comparisons ===')
hdr_v, hdr_c, hdr_s, hdr_d = 'variant', 'CAGR%', 'Sharpe', 'DD%'
print(f'{hdr_v:<34s} {hdr_c:>7s} {hdr_s:>7s} {hdr_d:>7s}')
flat_lbl = 'FLAT'
print(f'{flat_lbl:<34s} {b_cagr*100:>7.2f} {b_sh:>7.2f} {b_dd*100:>7.2f}')
for vname, vmap in variants.items():
    rs = df['ret'] * df['risk_on'].map(vmap)
    vs = stats(rs, vname)
    vc, vsh, vdd = vs['cagr'], vs['sharpe'], vs['dd']
    print(f'{vname:<34s} {vc*100:>7.2f} {vsh:>7.2f} {vdd*100:>7.2f}')

# Continuous sizing: mult = clip(1 + alpha * (stable_z + etf_z)/2, floor, ceil)
print('\n=== Continuous sizing (mult = clip(1 + alpha * avg_z, lo, hi)) ===')
df['avg_z'] = (df['stable_z'] + df['etf_z']) / 2.0
for alpha_k, lo, hi in [(0.25, 0.5, 1.5), (0.5, 0.25, 1.75), (1.0, 0.0, 2.0)]:
    mult_c = (1.0 + alpha_k * df['avg_z']).clip(lo, hi)
    rs = df['ret'] * mult_c
    vs = stats(rs, 'cont')
    label = f'alpha={alpha_k} clip=[{lo},{hi}]'
    vc, vsh, vdd = vs['cagr'], vs['sharpe'], vs['dd']
    print(f'{label:<34s} {vc*100:>7.2f} {vsh:>7.2f} {vdd*100:>7.2f}')

# Correlation of signals to forward 1d blend return
print('\n=== Signal-to-return correlations ===')
c0 = df[['stable_z','ret']].corr().iloc[0,1]
c1 = df[['etf_z','ret']].corr().iloc[0,1]
c2 = df[['risk_on','ret']].corr().iloc[0,1]
print(f'  corr(stable_z, ret_today) = {c0:+.4f}')
print(f'  corr(etf_z, ret_today)    = {c1:+.4f}')
print(f'  corr(risk_on, ret_today)  = {c2:+.4f}')
# Forward returns (1d ahead, 3d, 7d)
for k in [1, 3, 7]:
    df[f'fwd_{k}'] = df['ret'].shift(-k).rolling(k).sum().shift(1-k) if k > 1 else df['ret'].shift(-1)
c_s1 = df[['stable_z','fwd_1']].corr().iloc[0,1]
c_e1 = df[['etf_z','fwd_1']].corr().iloc[0,1]
c_s3 = df[['stable_z','fwd_3']].corr().iloc[0,1]
c_e3 = df[['etf_z','fwd_3']].corr().iloc[0,1]
c_s7 = df[['stable_z','fwd_7']].corr().iloc[0,1]
c_e7 = df[['etf_z','fwd_7']].corr().iloc[0,1]
print(f'  corr(stable_z, fwd_1d)  = {c_s1:+.4f}')
print(f'  corr(etf_z,    fwd_1d)  = {c_e1:+.4f}')
print(f'  corr(stable_z, fwd_3d)  = {c_s3:+.4f}')
print(f'  corr(etf_z,    fwd_3d)  = {c_e3:+.4f}')
print(f'  corr(stable_z, fwd_7d)  = {c_s7:+.4f}')
print(f'  corr(etf_z,    fwd_7d)  = {c_e7:+.4f}')
