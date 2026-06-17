"""Aggregate J1 + J4 + xsec_stack WF results into logs/post_fix_ranking_v3.csv.

Reads each fixj1_* / unseen_* / fixed_* seed from logs/paper_trader_v2/seeds
and emits per-profile: total_ret_pct, sharpe, max_dd, n_trades, mean_pnl_pct.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = ROOT / 'logs' / 'paper_trader_v2' / 'seeds'
OUT = ROOT / 'logs' / 'post_fix_ranking_v3.csv'


def summarize_seed(seed_dir: Path) -> dict | None:
    ds_fp = seed_dir / 'daily_snapshot.csv'
    tl_fp = seed_dir / 'trade_log.csv'
    if not ds_fp.exists():
        return None
    ds = pl.read_csv(ds_fp)
    if len(ds) == 0:
        return None
    init_eq = float(ds['total_equity'][0])
    final_eq = float(ds['total_equity'][-1])
    # daily returns from total_equity
    eq = ds['total_equity'].to_numpy()
    if len(eq) > 1:
        daily_ret = eq[1:] / eq[:-1] - 1  # fractional daily return
        valid = daily_ret[np.isfinite(daily_ret)]
        if len(valid) > 2 and valid.std() > 0:
            sharpe = valid.mean() * np.sqrt(365) / valid.std()
        else:
            sharpe = 0
        peak = np.maximum.accumulate(eq)
        max_dd = ((eq - peak) / peak).min() * 100
    else:
        sharpe = 0
        max_dd = 0
    total_ret = (final_eq / init_eq - 1) * 100
    # trade count
    n_trades = 0
    mean_pnl_pct = 0
    if tl_fp.exists():
        tl = pl.read_csv(tl_fp)
        n_trades = len(tl)
        if n_trades > 0 and 'pnl_pct' in tl.columns:
            mean_pnl_pct = float(tl['pnl_pct'].mean())
    return {
        'seed': seed_dir.name,
        'init_eq': init_eq,
        'final_eq': final_eq,
        'total_ret_pct': total_ret,
        'sharpe': sharpe,
        'max_dd_pct': max_dd,
        'n_trades': n_trades,
        'mean_pnl_pct': mean_pnl_pct,
        'n_bars': len(ds),
    }


def main():
    results = []
    for sd in sorted(SEEDS_DIR.iterdir()):
        if not sd.is_dir():
            continue
        # Filter to our seeds of interest
        if not (sd.name.startswith('fixj1_')
                or sd.name.startswith('fix_')
                or sd.name.startswith('fixed_')
                or sd.name.startswith('unseen_')):
            continue
        r = summarize_seed(sd)
        if r is None:
            continue
        results.append(r)
    if not results:
        print('No matching seeds found.')
        return
    df = pl.DataFrame(results)
    # sort by total_ret desc
    df = df.sort('total_ret_pct', descending=True)
    df.write_csv(OUT)
    print(f"\nPost-fix ranking written to {OUT}")
    print(f"\n{'seed':<35} {'total%':>9} {'Sh':>6} {'DD%':>7} {'n_tr':>5} {'net%/t':>8} {'bars':>5}")
    print('-' * 85)
    for r in df.to_dicts():
        print(f"{r['seed']:<35} {r['total_ret_pct']:>+8.2f}% "
              f"{r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f}% "
              f"{r['n_trades']:>5} {r['mean_pnl_pct']:>+7.3f} {r['n_bars']:>5}")


if __name__ == "__main__":
    main()
