"""
Reusable event-study harness for event-triggered sleeves.

Built turn 019 for: p11 Announcement-Volatility (delisting-rebound, monitoring,
margin, etc.), HODLer campaigns, p10 IEO retrospective, future multi-CEX work.

API (minimal):

    from src.frontier.utils.event_study import run_event_study, load_kline_cached

    events = pd.DataFrame([
        {'event_date': '2025-03-01 14:30:00', 'symbol': 'XRP', 'category': 'delisting'},
        ...
    ])

    results = run_event_study(
        events_df=events,
        horizons_h=[1, 6, 12, 24, 48, 72, 168],    # hours after entry
        interval='1h',                              # '1h' for intraday, '1d' for daily
        use_spot=True,                              # True=spot api, False=futures fapi
        entry_lag_min=60,                           # minutes after event before entry
        cost_rt_pct=0.0020,                         # 20 bps round-trip
        splits={
            'TRAIN': ('2020-01-01', '2023-12-31'),
            'VAL':   ('2024-01-01', '2024-12-31'),
            'OOS':   ('2025-01-01', '2026-12-31'),
        },
        shuffle_null_n=20,                          # 0 to skip null
    )

    # results = {
    #   'per_event': DataFrame [symbol, event_date, h1_ret, h6_ret, ...],
    #   'per_split_horizon': {
    #     'TRAIN': {1: {n, mean, t_stat, hit, std}, 6: {...}, ...},
    #     'VAL':   {...},
    #     'OOS':   {...},
    #   },
    #   'per_asset_horizon': {
    #     'XRP': {1: {...across splits}},
    #     ...
    #   },
    #   'null': {1: {null_t_p95, ...}},
    # }

Design notes:
  - Klines are cached per-symbol-per-interval to avoid re-fetch.
  - Cost applied as entry + exit (round-trip); applied to net forward return.
  - Entry timing: event_date + entry_lag_min; exit = entry + horizon_h.
  - Forward return = (exit_close / entry_close) - 1 - cost_rt_pct
  - Shuffle null: randomize event_dates over the full span, preserve count.
  - t-stat: mean / (std/sqrt(n)). hit = (r > 0).mean().
  - Chronological split by event_date.
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


# Binance endpoints
SPOT_KLINES_URL = 'https://api.binance.com/api/v3/klines'
FUTURES_KLINES_URL = 'https://fapi.binance.com/fapi/v1/klines'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / 'data' / 'frontier' / 'event_study_klines'
DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)


def fetch_klines(symbol: str, start_ms: int, end_ms: int,
                 interval: str = '1h', use_spot: bool = True,
                 max_retries: int = 3) -> list[list[Any]] | None:
    """Fetch klines from Binance, handling pagination + rate-limits."""
    url = SPOT_KLINES_URL if use_spot else FUTURES_KLINES_URL
    out: list[list[Any]] = []
    cursor = int(start_ms)
    end_ms = int(end_ms)
    backoff = 1.0
    retries = 0
    while cursor < end_ms:
        params = {'symbol': symbol, 'interval': interval, 'startTime': cursor, 'endTime': end_ms, 'limit': 1500}
        try:
            r = requests.get(url, params=params, timeout=15, headers={'User-Agent': UA})
            if r.status_code == 429 or r.status_code == 418:
                if retries >= max_retries:
                    return None
                time.sleep(backoff)
                backoff *= 2
                retries += 1
                continue
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        if not data:
            break
        out.extend(data)
        if len(data) < 1500:
            break
        cursor = data[-1][0] + 1
        time.sleep(0.1)
    return out


def load_kline_cached(symbol: str, start_ms: int, end_ms: int,
                      interval: str = '1h', use_spot: bool = True,
                      cache_dir: Path = DEFAULT_CACHE) -> pd.DataFrame | None:
    """Load klines with disk cache. Returns [open_time, open, high, low, close, volume] DataFrame.

    Cache granularity: per (symbol, interval). Cached file is the FULL span
    ever requested; returns the requested window from it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{symbol.upper()}_{interval}.parquet'
    df_existing: pd.DataFrame | None = None
    need_fetch_start = start_ms
    need_fetch_end = end_ms
    if cache.exists():
        df_existing = pd.read_parquet(cache)
        if len(df_existing) > 0:
            have_start = int(df_existing['open_time'].min())
            have_end = int(df_existing['open_time'].max())
            # only fetch the uncovered slice (simple: fetch if request extends either end)
            if have_start <= start_ms and have_end >= end_ms:
                return df_existing[(df_existing['open_time'] >= start_ms) & (df_existing['open_time'] <= end_ms)].copy()
            # otherwise, fetch a superset
            need_fetch_start = min(start_ms, have_start)
            need_fetch_end = max(end_ms, have_end)
    raw = fetch_klines(symbol.upper(), need_fetch_start, need_fetch_end, interval=interval, use_spot=use_spot)
    if raw is None or len(raw) == 0:
        return df_existing[(df_existing['open_time'] >= start_ms) & (df_existing['open_time'] <= end_ms)].copy() if df_existing is not None else None
    df = pd.DataFrame(raw, columns=['open_time','open','high','low','close','volume',
                                     'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore'])
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col])
    df['open_time'] = df['open_time'].astype('int64')
    df = df[['open_time','open','high','low','close','volume']].drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    df.to_parquet(cache)
    return df[(df['open_time'] >= start_ms) & (df['open_time'] <= end_ms)].copy()


def event_forward_returns(events_df: pd.DataFrame, horizons_h: list[int],
                          interval: str = '1h', use_spot: bool = True,
                          entry_lag_min: int = 60,
                          cache_dir: Path = DEFAULT_CACHE,
                          symbol_suffix: str = 'USDT',
                          verbose: bool = False) -> pd.DataFrame:
    """For each event, compute forward returns at each horizon."""
    out_rows = []
    max_h_ms = max(horizons_h) * 3600 * 1000 + 24*3600*1000  # buffer
    entry_lag_ms = entry_lag_min * 60 * 1000
    for i, row in events_df.iterrows():
        sym_base = str(row['symbol']).upper()
        symbol = f'{sym_base}{symbol_suffix}'
        ev_date = pd.Timestamp(row['event_date'])
        ev_ms = int(ev_date.timestamp() * 1000)
        start = ev_ms - 3600*1000  # 1h before
        end = ev_ms + max_h_ms
        klines = load_kline_cached(symbol, start, end, interval=interval, use_spot=use_spot, cache_dir=cache_dir)
        if klines is None or len(klines) < 3:
            if verbose:
                print(f'  [skip] {symbol} @ {ev_date}: no klines')
            continue
        # entry: first kline with open_time >= ev_ms + entry_lag_ms
        entry_target_ms = ev_ms + entry_lag_ms
        entry_rows = klines[klines['open_time'] >= entry_target_ms]
        if len(entry_rows) == 0:
            if verbose:
                print(f'  [skip] {symbol} @ {ev_date}: no entry kline')
            continue
        entry = entry_rows.iloc[0]
        entry_close = float(entry['close'])
        out = {'symbol': sym_base, 'event_date': ev_date, 'entry_time': pd.Timestamp(entry['open_time'], unit='ms'), 'entry_close': entry_close}
        for h in horizons_h:
            exit_target_ms = entry['open_time'] + h * 3600 * 1000
            exit_rows = klines[klines['open_time'] >= exit_target_ms]
            if len(exit_rows) == 0:
                out[f'h{h}_ret'] = np.nan
                continue
            exit_close = float(exit_rows.iloc[0]['close'])
            out[f'h{h}_ret'] = (exit_close / entry_close) - 1.0
        # preserve any extra columns
        for col in events_df.columns:
            if col not in ('symbol','event_date'):
                out[col] = row[col]
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def _stats(arr: np.ndarray, cost_rt: float) -> dict:
    net = arr - cost_rt
    net = net[~np.isnan(net)]
    n = len(net)
    if n < 3:
        return {'n': n, 'status': 'thin'}
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if n > 1 else 0.0
    t = mean / (std/np.sqrt(n)) if std > 0 else 0.0
    hit = float((net > 0).mean())
    return {'n': n, 'mean_pct': mean*100, 't_stat': t, 'hit_rate': hit, 'std_pct': std*100}


def run_event_study(events_df: pd.DataFrame, horizons_h: list[int],
                    interval: str = '1h', use_spot: bool = True,
                    entry_lag_min: int = 60, cost_rt_pct: float = 0.0020,
                    splits: dict[str, tuple[str,str]] | None = None,
                    shuffle_null_n: int = 0, rng_seed: int = 42,
                    cache_dir: Path = DEFAULT_CACHE,
                    symbol_suffix: str = 'USDT',
                    verbose: bool = False) -> dict:
    """End-to-end event study with per-split + per-asset + optional shuffle null."""
    # 1. forward returns
    per_event = event_forward_returns(events_df, horizons_h, interval=interval,
                                      use_spot=use_spot, entry_lag_min=entry_lag_min,
                                      cache_dir=cache_dir, symbol_suffix=symbol_suffix,
                                      verbose=verbose)
    if len(per_event) == 0:
        return {'per_event': per_event, 'per_split_horizon': {}, 'per_asset_horizon': {}, 'null': {}}

    # 2. per-split + per-horizon aggregate
    per_split_horizon: dict = {}
    if splits is None:
        splits = {'ALL': (per_event['event_date'].min().strftime('%Y-%m-%d'),
                          per_event['event_date'].max().strftime('%Y-%m-%d'))}
    for split_name, (lo, hi) in splits.items():
        mask = (per_event['event_date'] >= lo) & (per_event['event_date'] <= hi)
        sub = per_event[mask]
        per_split_horizon[split_name] = {}
        for h in horizons_h:
            col = f'h{h}_ret'
            if col in sub.columns:
                per_split_horizon[split_name][h] = _stats(sub[col].values, cost_rt_pct)
            else:
                per_split_horizon[split_name][h] = {'n': 0, 'status': 'no_data'}

    # 3. per-asset aggregate (across all events)
    per_asset_horizon: dict = {}
    for sym, g in per_event.groupby('symbol'):
        per_asset_horizon[sym] = {}
        for h in horizons_h:
            col = f'h{h}_ret'
            if col in g.columns:
                per_asset_horizon[sym][h] = _stats(g[col].values, cost_rt_pct)

    # 4. Shuffle null (if requested) -- RANDOMIZE ENTRY DATES PER SYMBOL
    #
    # Prior bug (fixed turn 021): permuted pool of returns, which is invariant
    # under mean/std -> null t_stat always = real t_stat. That null was non-
    # informative.
    #
    # Correct null: randomize each event's date uniformly within that symbol's
    # trading history (keep symbol + category fixed). Re-compute forward
    # returns from randomized dates via the same harness. Aggregate t_stats
    # across n_shuffles trials per horizon.
    #
    # Interpretation: if random dates on the same symbols give the same t_stat,
    # the "event" is not providing signal beyond asset-selection base rate.
    null_out: dict = {}
    if shuffle_null_n > 0 and len(per_event) >= 10:
        rng = np.random.default_rng(rng_seed)
        # Build the date span per symbol from observed events (simple + safe)
        # Better: use the cached klines' date range for each symbol
        sym_span = {}
        for sym in per_event['symbol'].unique():
            # prefer cached klines date range for proper span
            cache_file = cache_dir / f'{sym.upper()}{symbol_suffix}_{interval}.parquet'
            if cache_file.exists():
                kl = pd.read_parquet(cache_file)
                if len(kl) > 0:
                    sym_span[sym] = (int(kl['open_time'].min()), int(kl['open_time'].max()))
                    continue
            # fallback: use event_date range across all events
            ev_dates = pd.to_datetime(per_event['event_date'])
            sym_span[sym] = (int(ev_dates.min().timestamp()*1000),
                             int(ev_dates.max().timestamp()*1000))

        ts_by_h: dict[int, list[float]] = {h: [] for h in horizons_h}
        max_h_ms = max(horizons_h) * 3600 * 1000
        for trial in range(shuffle_null_n):
            # Build randomized events DataFrame (same symbols, random dates in each symbol's span)
            null_rows = []
            for _, row in per_event.iterrows():
                sym = row['symbol']
                lo, hi = sym_span[sym]
                hi_safe = max(lo + max_h_ms + 24*3600*1000, hi - max_h_ms)  # ensure forward-horizon fits
                if hi_safe <= lo:
                    continue
                rand_ms = int(rng.uniform(lo, hi_safe))
                null_rows.append({'symbol': sym,
                                  'event_date': pd.Timestamp(rand_ms, unit='ms'),
                                  'category': row.get('category', 'null')})
            if not null_rows:
                continue
            null_events_df = pd.DataFrame(null_rows)
            null_per_event = event_forward_returns(
                null_events_df, horizons_h, interval=interval, use_spot=use_spot,
                entry_lag_min=entry_lag_min, cache_dir=cache_dir,
                symbol_suffix=symbol_suffix, verbose=False)
            if len(null_per_event) == 0:
                continue
            for h in horizons_h:
                col = f'h{h}_ret'
                if col not in null_per_event.columns:
                    continue
                arr = null_per_event[col].dropna().values
                if len(arr) < 3:
                    continue
                s = _stats(arr, cost_rt_pct)
                if 't_stat' in s:
                    ts_by_h[h].append(s['t_stat'])
        for h in horizons_h:
            ts = ts_by_h[h]
            if len(ts) < 2:
                null_out[h] = {'status': 'thin', 'n_shuffles': len(ts)}
                continue
            ts_arr = np.array(ts)
            null_out[h] = {
                'null_t_mean': float(ts_arr.mean()),
                'null_t_std': float(ts_arr.std()),
                'null_t_p5':  float(np.percentile(ts_arr, 5)),
                'null_t_p95': float(np.percentile(ts_arr, 95)),
                'null_t_p99': float(np.percentile(ts_arr, 99)),
                'n_shuffles': len(ts_arr),
            }

    return {
        'per_event': per_event,
        'per_split_horizon': per_split_horizon,
        'per_asset_horizon': per_asset_horizon,
        'null': null_out,
    }


def pretty_report(results: dict, horizons_h: list[int] | None = None) -> str:
    """Format run_event_study output as a readable report string."""
    lines = []
    per_event = results.get('per_event')
    if per_event is None or len(per_event) == 0:
        return '(no events processed)'
    if horizons_h is None:
        horizons_h = sorted([int(c[1:-4]) for c in per_event.columns if c.startswith('h') and c.endswith('_ret')])
    lines.append(f'Events processed: {len(per_event)}')
    lines.append(f'Assets covered:   {per_event["symbol"].nunique()}')
    lines.append(f'Date span:        {per_event["event_date"].min()} -> {per_event["event_date"].max()}')
    lines.append('')

    psh = results.get('per_split_horizon', {})
    lines.append('=== Per-split × per-horizon ===')
    for split_name, by_h in psh.items():
        lines.append(f'\n{split_name}:')
        for h in horizons_h:
            s = by_h.get(h, {})
            if s.get('status'):
                lines.append(f'  h{h:>3d}h: {s["status"]} (n={s.get("n",0)})')
                continue
            star = '*' if s.get('t_stat',0) > 2 and s.get('mean_pct',0) > 0 and s.get('hit_rate',0) > 0.5 else ' '
            lines.append(f'  h{h:>3d}h: n={s["n"]:4d}  mean={s["mean_pct"]:+6.2f}%  t={s["t_stat"]:+5.2f}  hit={s["hit_rate"]:.2f}  std={s["std_pct"]:.2f}%{star}')

    nl = results.get('null', {})
    if nl:
        lines.append('\n=== Shuffle null (t_stat percentiles) ===')
        for h in horizons_h:
            nb = nl.get(h, {})
            if nb.get('status'):
                continue
            lines.append(f'  h{h:>3d}h: null_p95={nb.get("null_t_p95",0):+.2f}  null_p99={nb.get("null_t_p99",0):+.2f}  (n_shuffles={nb.get("n_shuffles",0)})')
    return '\n'.join(lines)


if __name__ == '__main__':
    # Smoke test: synthetic events on BTC
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    events = pd.DataFrame([
        {'symbol':'BTC', 'event_date':'2025-03-01 10:00:00', 'category':'test'},
        {'symbol':'BTC', 'event_date':'2025-06-15 10:00:00', 'category':'test'},
        {'symbol':'ETH', 'event_date':'2025-08-01 10:00:00', 'category':'test'},
    ])
    r = run_event_study(events, horizons_h=[1, 6, 24, 72], interval='1h',
                       use_spot=True, splits={'ALL': ('2020-01-01','2026-12-31')},
                       shuffle_null_n=0, verbose=True)
    print(pretty_report(r))
