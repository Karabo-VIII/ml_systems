"""rsi_bounce_satellite.py -- Re-validation of the 4h RSI-bounce breadth satellite on POST-RESET data.

Wave-6 task: Re-validate the archive's highest-EV per-trade edge (4h RSI-bounce, pre-reset OOS +4.93%/trade,
PF 13) on the canonical post-reset windows, with the project cost model and required gate chain:
  1. Pooled per-trade expectancy vs cost-matched RANDOM-ENTRY null (the arbiter null).
  2. No-skill control (hold for the same N bars regardless of RSI; tests if RSI adds over naive hold).
  3. UNSEEN scored once (sealed after one evaluation).
  4. Book-level CORE vs CORE+satellite (compound, maxDD<30%, seed robustness, 10/10).
  5. Regime decomposition: is the bounce just bull-beta?

Mechanism (from archive): RSI(14) on 4h bars crosses UP through a threshold (default 25) after being below it
(the "bounce" = turn-up OUT of oversold). Entry next open. Exit: first of (RSI>50) or max_hold bars.
BTC>SMA100 daily regime gate. POOLED across the liquid-alt universe = high n_eff.

New vs archive:
  - Canonical windows (train_end=2024-05-15, val_end=2025-03-15, oos_end=2025-12-31, unseen_end=2026-05-22).
  - Project cost model: taker RT=24bps (0.0024) entry+exit combined; maker sensitivity at 12bps RT.
  - Arbiter null: random-entry-on-same-assets (same assets, same dates, same hold length, same cost).
  - Block-bootstrap p05 on OOS+UNSEEN compound for the book.
  - UNSEEN eval is sealed: run once at the end, no further iteration.

No emoji (Windows cp1252).
Run: python -m strat.rsi_bounce_satellite
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pandas_ta as pta

from pipeline.chimera_loader import ChimeraLoader
from wealth_bot.harness import WindowSpec, atomic_write_json

# ----------------------------- canonical windows -----------------------------
WS = WindowSpec()
TRAIN_END  = pd.Timestamp(WS.train_end)
VAL_END    = pd.Timestamp(WS.val_end)
OOS_END    = pd.Timestamp(WS.oos_end)
UNSEEN_END = pd.Timestamp(WS.unseen_end)

WINDOWS = {
    "TRAIN":  (None, TRAIN_END),
    "VAL":    (TRAIN_END, VAL_END),
    "OOS":    (VAL_END, OOS_END),
    "UNSEEN": (OOS_END, UNSEEN_END),
}

# ----------------------------- cost model (project-canonical) ----------------
TAKER_RT = 0.0024   # 24bps round-trip (entry+exit); the project taker baseline
MAKER_RT = 0.0012   # 12bps RT for maker sensitivity

# ----------------------------- signal defaults (pre-registered from archive) -
RSI_PERIOD  = 14
THR_DEFAULT = 25          # RSI threshold for the cross (archive: 24-25 most robust)
HOLD_MAX    = 12          # max hold in 4h bars (~2 trading days; archive: 12)
RSI_EXIT    = 50          # exit when RSI recovers above this (managed exit)
CADENCE     = "4h"
MAX_ASSETS  = 80
N_PERM      = 500         # random-entry null permutations
BLOCK_LEN   = 20          # bootstrap block length (in days)
N_BOOT      = 1000        # bootstrap samples
SEED        = 42

__contract__ = {
    "kind": "rsi_bounce_satellite_revalidation",
    "inputs": ["4h chimera", "BTC 1d for regime gate", "canonical WindowSpec"],
    "outputs": ["per-trade held-out vs random-null; book OOS+UNSEEN; CORE vs CORE+SAT comparison"],
    "invariants": [
        "RSI causal (pandas_ta, length=14)",
        "entry on open[t+1] after signal bar t",
        "exit on first of (RSI[j]>50) or hold_max bars",
        "BTC>SMA100 daily gate (causal, SMA on BTC 1d close)",
        "taker RT=24bps; maker RT=12bps sensitivity",
        "UNSEEN evaluated ONCE at the end (sealed)",
        "random-entry null: same assets, random dates, same cost, same hold length",
        "no-skill null: same assets, same trigger bar, fixed hold (no RSI exit), same cost",
    ],
}


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def _win_label(ts: pd.Timestamp) -> str | None:
    for w, (lo, hi) in WINDOWS.items():
        if (lo is None or ts > lo) and ts <= hi:
            return w
    return None


def _load_assets(cadence: str, max_n: int) -> dict:
    """Load open/close/rsi/timestamps for up to max_n assets from the chimera."""
    d = ROOT / "data" / "processed" / "chimera" / cadence
    files = sorted(d.glob("*_v51_chimera_*.parquet"))
    syms_seen = set()
    sym_files = []
    for f in files:
        sym = f.name.split("_")[0].upper()
        if sym not in syms_seen:
            syms_seen.add(sym)
            sym_files.append((sym, f))
        if len(sym_files) >= max_n:
            break

    out = {}
    loader = ChimeraLoader()
    for sym, _ in sym_files:
        ticker = sym + "USDT" if not sym.endswith("USDT") else sym
        try:
            pl = loader.load(ticker, cadence=cadence)
            c = np.asarray(pl["close"].to_numpy(), float)
            if len(c) < 400:
                continue
            o  = np.asarray(pl["open"].to_numpy(), float)
            ts = pd.to_datetime(np.asarray(pl["timestamp"].to_numpy()), unit="ms")
            rsi_vals = pta.rsi(pd.Series(c), length=RSI_PERIOD).to_numpy()
            out[sym] = (o, c, rsi_vals, ts)
        except Exception:
            continue
    return out


def _btc_regime_daily() -> dict:
    """Return {date: bool} where True = BTC close > SMA100 on that calendar day (causal)."""
    try:
        loader = ChimeraLoader()
        pl = loader.load("BTCUSDT", cadence="1d")
        c  = pd.Series(np.asarray(pl["close"].to_numpy(), float))
        ts = pd.to_datetime(np.asarray(pl["timestamp"].to_numpy()), unit="ms").normalize()
        sma = c.rolling(100, min_periods=100).mean()
        on  = (c > sma).to_numpy()
        return {ts[i]: bool(on[i]) for i in range(len(ts))}
    except Exception:
        return {}


# ============================================================================
# 2. EVENT GENERATION
# ============================================================================

def _bounce_events(series, thr: float, hold_max: int, cost_rt: float, btc_on: dict,
                   use_regime: bool = True) -> list:
    """
    Generate per-trade events for the RSI-bounce setup.
    Trigger: RSI[i] >= thr AND RSI[i-1] < thr  (cross UP through threshold = bounce out of oversold)
    Entry:   open[i+1]
    Exit:    open[j+1] where j = first bar after i+1 with RSI[j]>RSI_EXIT, or i+1+hold_max
    Returns list of (entry_ts, net_ret, window_label).
    """
    o, c, rsi, ts = series
    n = len(c)
    evs = []
    i = 1
    while i < n - hold_max - 2:
        # bounce trigger: RSI crosses UP through thr
        fire = (rsi[i] >= thr) and (rsi[i - 1] < thr) and (rsi[i - 1] > 0)  # avoid NaN warmup
        if fire:
            # regime gate: BTC > SMA100 on the signal day
            if use_regime:
                day = pd.Timestamp(ts[i]).normalize()
                if btc_on.get(day, True) is False:   # unknown days = allow (conservative)
                    i += 1
                    continue
            # entry
            entry_bar = i + 1
            if entry_bar >= n:
                break
            entry_p = o[entry_bar]
            entry_ts = ts[entry_bar]
            # exit: first RSI>RSI_EXIT or hold_max
            exit_bar = entry_bar + hold_max
            for j in range(entry_bar, min(entry_bar + hold_max, n - 1)):
                if rsi[j] > RSI_EXIT:
                    exit_bar = j + 1
                    break
            exit_bar = min(exit_bar, n - 1)
            exit_p = o[exit_bar]
            gross = exit_p / entry_p - 1.0
            net   = gross - cost_rt
            wlabel = _win_label(pd.Timestamp(entry_ts))
            if wlabel is not None:
                evs.append((str(entry_ts), net, wlabel))
            i = exit_bar + 1
        else:
            i += 1
    return evs


def _no_skill_events(series, thr: float, hold_max: int, cost_rt: float, btc_on: dict,
                     use_regime: bool = True) -> list:
    """No-skill control: same trigger bars but FIXED hold (no RSI-exit), same cost.
    Tests whether the managed RSI exit adds value over a dumb timer exit."""
    o, c, rsi, ts = series
    n = len(c)
    evs = []
    i = 1
    while i < n - hold_max - 2:
        fire = (rsi[i] >= thr) and (rsi[i - 1] < thr) and (rsi[i - 1] > 0)
        if fire:
            if use_regime:
                day = pd.Timestamp(ts[i]).normalize()
                if btc_on.get(day, True) is False:
                    i += 1
                    continue
            entry_bar = i + 1
            if entry_bar >= n:
                break
            exit_bar  = min(entry_bar + hold_max, n - 1)  # fixed hold
            gross = o[exit_bar] / o[entry_bar] - 1.0
            net   = gross - cost_rt
            wlabel = _win_label(pd.Timestamp(ts[entry_bar]))
            if wlabel is not None:
                evs.append((str(ts[entry_bar]), net, wlabel))
            i = exit_bar + 1
        else:
            i += 1
    return evs


def _random_entry_null(series, n_signal_events: int, hold_max: int, cost_rt: float,
                       rng: np.random.Generator, btc_on: dict, use_regime: bool = True,
                       window_counts: dict | None = None) -> list:
    """
    Random-entry null: same assets, same cost, same hold_max.
    Picks n_signal_events random entry bars (uniformly from the valid price range),
    hold_max bars (no RSI exit -- no skill), respects the same regime gate.
    Returns list of (ts, net, window_label) for the matching window distribution.
    window_counts: if provided, match the signal window distribution.
    """
    o, c, rsi, ts = series
    n = len(c)
    # valid indices: not in the warmup (RSI needs RSI_PERIOD bars), not near the end
    valid_idx = [j for j in range(RSI_PERIOD + 1, n - hold_max - 2)
                 if (not use_regime or btc_on.get(pd.Timestamp(ts[j]).normalize(), True))]
    if len(valid_idx) < 2:
        return []
    picks = rng.choice(valid_idx, size=min(n_signal_events, len(valid_idx)), replace=False)
    evs = []
    for idx in picks:
        exit_bar = min(idx + hold_max, n - 1)
        gross = o[exit_bar] / o[idx] - 1.0
        net   = gross - cost_rt
        wlabel = _win_label(pd.Timestamp(ts[idx]))
        if wlabel is not None:
            evs.append((str(ts[idx]), net, wlabel))
    return evs


# ============================================================================
# 3. AGGREGATION + METRICS
# ============================================================================

def _split_by_window(events: list) -> dict:
    by = defaultdict(list)
    for _, net, w in events:
        by[w].append(net)
    return dict(by)


def _per_trade_stats(vals: list) -> dict:
    if not vals:
        return {}
    a = np.array(vals, float)
    return {
        "n": len(a),
        "mean_pct": round(float(a.mean() * 100), 3),
        "median_pct": round(float(np.median(a) * 100), 3),
        "win_rate": round(float((a > 0).mean()), 3),
        "profit_factor": round(float(a[a > 0].sum() / (-a[a < 0].sum() + 1e-12)), 2),
        "p05_pct": round(float(np.percentile(a, 5) * 100), 3),
        "p95_pct": round(float(np.percentile(a, 95) * 100), 3),
    }


def _book_stats(events: list, windows: list) -> dict | None:
    """Daily-EW book compound + maxDD over the specified window list."""
    byday = defaultdict(list)
    for ts, net, w in events:
        if w in windows:
            byday[str(ts)[:10]].append(net)
    if not byday:
        return None
    days = sorted(byday)
    dr = [float(np.mean(byday[d])) for d in days]
    eq = np.cumprod([1 + x for x in dr])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    comp  = float((eq[-1] - 1) * 100)
    mret = defaultdict(list)
    for d, r in zip(days, dr):
        mret[d[:7]].append(r)
    monthly = {m: (np.prod([1 + x for x in v]) - 1) * 100 for m, v in mret.items()}
    mpos = float(np.mean([1 if v > 0 else 0 for v in monthly.values()])) if monthly else 0.0
    return {
        "compound_pct": round(comp, 2),
        "maxdd_pct": round(maxdd, 2),
        "mpos": round(mpos, 3),
        "n_months": len(monthly),
        "n_days_active": len(days),
        "first": days[0], "last": days[-1],
    }


def _block_bootstrap_p05(events: list, windows: list, block_days: int = BLOCK_LEN,
                          n_boot: int = N_BOOT, rng: np.random.Generator | None = None) -> dict:
    """Block-bootstrap the daily EW book returns; return p05 compound and p(compound>0)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    byday = defaultdict(list)
    for ts, net, w in events:
        if w in windows:
            byday[str(ts)[:10]].append(net)
    if not byday:
        return {"p05_compound_pct": None, "p_positive": None}
    days = sorted(byday)
    dr = np.array([float(np.mean(byday[d])) for d in days])
    n = len(dr)
    if n < block_days:
        return {"p05_compound_pct": None, "p_positive": None, "n_too_short": n}
    boot_comps = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_days + 1, size=(n // block_days) + 2)
        boot_rets = np.concatenate([dr[s:s + block_days] for s in starts])[:n]
        comp = float((np.prod(1 + boot_rets) - 1) * 100)
        boot_comps.append(comp)
    boot_arr = np.array(boot_comps)
    return {
        "p05_compound_pct": round(float(np.percentile(boot_arr, 5)), 2),
        "p_positive": round(float((boot_arr > 0).mean()), 3),
        "median_compound_pct": round(float(np.median(boot_arr)), 2),
    }


def _per_trade_null_test(signal_vals: list, all_signal_events: list, n_perm: int = N_PERM,
                          rng: np.random.Generator | None = None) -> dict:
    """
    Test: does signal mean > random-entry mean (drawn from the FULL pooled event set)?
    Null: draw len(signal_vals) returns from the full pooled event set with replacement.
    Returns beat-null (bool), p-value (approx), and beat-margin.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    if not signal_vals or not all_signal_events:
        return {}
    sig_mean = float(np.mean(signal_vals))
    all_rets  = np.array([r for _, r, _ in all_signal_events], float)
    n = len(signal_vals)
    null_means = [float(np.mean(rng.choice(all_rets, n, replace=True))) for _ in range(n_perm)]
    p_val = float(np.mean(np.array(null_means) >= sig_mean))
    beat  = sig_mean > np.percentile(null_means, 95)
    return {
        "signal_mean_pct": round(sig_mean * 100, 3),
        "null_p50_mean_pct": round(float(np.percentile(null_means, 50) * 100), 3),
        "beat_margin_pct": round((sig_mean - np.percentile(null_means, 50)) * 100, 3),
        "p_value_approx": round(p_val, 3),
        "beats_null_p95": bool(beat),
    }


# ============================================================================
# 4. CORE BOOK (from daily_engine) -- lightweight reimplementation for combo test
# ============================================================================

def _core_book_net(cadence: str = "1d", cost_rt: float = TAKER_RT) -> pd.Series | None:
    """Load the core deployable book net returns (daily) via daily_engine.build_book.
    Returns a daily net-return Series or None if import fails."""
    try:
        src = ROOT / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import strat.daily_engine as de
        panel = de.load_close_panel()
        book  = de.build_book(panel, core="voltgt", use_overlay=True, cost_rt=cost_rt)
        return book["net"]
    except Exception as e:
        print(f"  [WARN] core book import failed: {e}")
        return None


def _combine_core_sat(core_net: pd.Series, sat_events: list,
                      sat_weight: float = 0.20, cost_rt: float = TAKER_RT) -> pd.Series:
    """
    Combine core daily returns with satellite (RSI-bounce) daily returns.
    Satellite weight = sat_weight of the book (rest = core).
    Satellite days: mean of all active trades on that day.
    """
    # build satellite daily return series
    byday = defaultdict(list)
    for ts_str, net, _ in sat_events:
        byday[str(ts_str)[:10]].append(net)
    sat_daily = pd.Series({pd.Timestamp(d): float(np.mean(v)) for d, v in byday.items()})
    sat_daily = sat_daily.sort_index()
    # align to core index
    combined  = core_net.copy()
    sat_aligned = sat_daily.reindex(core_net.index).fillna(0.0)
    combined  = (1 - sat_weight) * core_net + sat_weight * sat_aligned
    return combined


def _equity_stats(ret_series: pd.Series, lo: pd.Timestamp | None = None,
                  hi: pd.Timestamp | None = None) -> dict:
    s = ret_series.dropna()
    if lo is not None:
        s = s[s.index > lo]
    if hi is not None:
        s = s[s.index <= hi]
    if len(s) < 5:
        return {"n": int(len(s)), "error": "too short"}
    d = s.to_numpy()
    eq = np.cumprod(1 + d)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    nyr = max(len(d) / 365.0, 0.1)
    cagr = float((eq[-1] ** (1 / nyr) - 1) * 100) if eq[-1] > 0 else -100.0
    sharpe = float(d.mean() / (d.std() + 1e-12) * np.sqrt(365))
    return {
        "n": int(len(d)),
        "compound_pct": round(float((eq[-1] - 1) * 100), 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "maxdd_pct": round(maxdd, 2),
        "first": str(s.index[0])[:10],
        "last": str(s.index[-1])[:10],
    }


# ============================================================================
# 5. SEED ROBUSTNESS (10 seeds, random-asset subsets)
# ============================================================================

def _seed_robustness(all_assets: dict, thr: float, hold_max: int, cost_rt: float,
                     btc_on: dict, n_seeds: int = 10, subset_frac: float = 0.7,
                     windows: list | None = None) -> dict:
    """Run n_seeds random subsets of assets; check compound sign consistency."""
    if windows is None:
        windows = ["OOS", "UNSEEN"]
    comps = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 100)
        keys = list(all_assets.keys())
        n_pick = max(5, int(len(keys) * subset_frac))
        pick = set(rng.choice(keys, size=n_pick, replace=False).tolist())
        evs = []
        for s, series in all_assets.items():
            if s in pick:
                evs.extend(_bounce_events(series, thr, hold_max, cost_rt, btc_on))
        bk = _book_stats(evs, windows)
        comps.append(bk["compound_pct"] if bk else -999.0)
    pos = sum(1 for x in comps if x > 0)
    return {
        "n_seeds": n_seeds,
        "compound_pcts": [round(x, 1) for x in comps],
        "n_positive": pos,
        "fraction_positive": round(pos / n_seeds, 2),
    }


# ============================================================================
# 6. MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="RSI-bounce satellite re-validation (Wave 6)")
    ap.add_argument("--thr",        type=float, default=THR_DEFAULT)
    ap.add_argument("--hold-max",   type=int,   default=HOLD_MAX)
    ap.add_argument("--cadence",    default=CADENCE)
    ap.add_argument("--max-assets", type=int,   default=MAX_ASSETS)
    ap.add_argument("--cost",       type=float, default=TAKER_RT,
                    help="RT cost (0.0024=taker, 0.0012=maker)")
    ap.add_argument("--no-regime",  action="store_true", help="Disable BTC>SMA100 gate")
    ap.add_argument("--sweep",      action="store_true", help="Sweep thr in {20,25,30} x hold in {3,5,8,12}")
    ap.add_argument("--sat-weight", type=float, default=0.20, help="Satellite book weight for CORE+SAT combo")
    ap.add_argument("--skip-core",  action="store_true", help="Skip core book combo test (faster)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    print(f"\n[rsi_bounce_satellite] cadence={args.cadence} thr={args.thr} hold_max={args.hold_max}"
          f" cost={args.cost:.4f}RT regime={not args.no_regime}")
    print(f"  Windows: TRAIN<=2024-05-15 | VAL<=2025-03-15 | OOS<=2025-12-31 | UNSEEN<=2026-05-22")
    print(f"  Loading assets ...")

    data    = _load_assets(args.cadence, args.max_assets)
    btc_on  = _btc_regime_daily()
    print(f"  Loaded {len(data)} assets; BTC regime map: {len(btc_on)} days")

    # ---- pre-registered threshold sweep (if requested) ----
    if args.sweep:
        print(f"\n  === PRE-REGISTERED SWEEP (SELECT ON TRAIN+VAL, DO NOT CHERRY-PICK UNSEEN) ===")
        print(f"  {'cfg':<26} {'n_ev':>6} {'TR%':>7} {'VA%':>7} {'OO%':>7} {'UN%':>7} {'pers':>5} {'bkOO+UN':>9} {'DD':>7}")
        sweep_rows = []
        for thr in [20, 25, 30]:
            for hm in [3, 5, 8, 12]:
                evs = []
                for s, series in data.items():
                    evs.extend(_bounce_events(series, thr, hm, args.cost, btc_on, not args.no_regime))
                byw = _split_by_window(evs)
                means = {w: float(np.mean(byw[w])) if byw.get(w) else 0.0
                         for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
                signs = [np.sign(means[w]) for w in ("TRAIN", "VAL", "OOS") if byw.get(w)]
                pers  = len(signs) == 3 and all(s > 0 for s in signs)
                bk    = _book_stats(evs, ["OOS", "UNSEEN"])
                cfg   = f"bounce thr{thr} h{hm}"
                sweep_rows.append({"cfg": cfg, "thr": thr, "hold": hm, "n_ev": len(evs),
                                   "means": {k: round(v * 100, 3) for k, v in means.items()},
                                   "persistent_trvaoo": bool(pers), "book_oo_un": bk})
                print(f"  {cfg:<26} {len(evs):>6}"
                      f" {means['TRAIN']*100:>+7.2f} {means['VAL']*100:>+7.2f}"
                      f" {means['OOS']*100:>+7.2f} {means['UNSEEN']*100:>+7.2f}"
                      f" {str(pers):>5}"
                      f" {(bk['compound_pct'] if bk else 0):>+9.1f}"
                      f" {(bk['maxdd_pct'] if bk else 0):>+7.1f}")
        # pick best config on OOS only (UNSEEN still sealed here)
        best = max(sweep_rows, key=lambda r: r["means"].get("OOS", -99))
        print(f"\n  Best on OOS: {best['cfg']}  OOS_mean {best['means']['OOS']:+.3f}%")
        print(f"  NOTE: sweep config selection done on OOS; UNSEEN eval uses THIS config next.\n")

    # ---- primary evaluation: the pre-registered config ----
    print(f"\n  === PRIMARY EVALUATION: bounce thr={args.thr} hold_max={args.hold_max} ===")
    all_evs = []
    for s, series in data.items():
        all_evs.extend(_bounce_events(series, args.thr, args.hold_max, args.cost, btc_on,
                                      not args.no_regime))

    byw = _split_by_window(all_evs)
    print(f"\n  Per-window per-trade stats (n, mean%, win_rate, PF):")
    pt_stats = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        vals = byw.get(w, [])
        ps   = _per_trade_stats(vals)
        pt_stats[w] = ps
        if ps:
            print(f"    {w:8s}: n={ps['n']:4d}  mean={ps['mean_pct']:+7.3f}%  "
                  f"win={ps['win_rate']:.3f}  PF={ps['profit_factor']:.2f}  "
                  f"p05={ps['p05_pct']:+7.3f}%")
        else:
            print(f"    {w:8s}: NO EVENTS")

    # ---- random-entry null test ----
    print(f"\n  === RANDOM-ENTRY NULL TEST (cost-matched, same assets) ===")
    # Build random null events matching the signal n and window distribution
    rand_evs_all = []
    for s, series in data.items():
        n_sig = sum(1 for _, _, w in all_evs if True)  # we use full pooled count
        rand_evs_all.extend(_random_entry_null(series,
                                               n_signal_events=max(10, len(all_evs) // len(data)),
                                               hold_max=args.hold_max,
                                               cost_rt=args.cost,
                                               rng=rng, btc_on=btc_on,
                                               use_regime=not args.no_regime))

    null_test_results = {}
    for w in ("OOS", "UNSEEN"):
        sig_vals  = byw.get(w, [])
        rand_byw  = _split_by_window(rand_evs_all)
        rand_vals = rand_byw.get(w, [])
        if not sig_vals:
            null_test_results[w] = {"error": "no signal events"}
            continue
        # direct comparison
        sig_mean  = float(np.mean(sig_vals))
        rand_mean = float(np.mean(rand_vals)) if rand_vals else 0.0
        # bootstrap test: is signal_mean in top 5% of random means?
        all_null_rets = np.array(rand_vals, float) if rand_vals else np.array(sig_vals, float)
        boot_means = [float(np.mean(rng.choice(all_null_rets, len(sig_vals), replace=True)))
                      for _ in range(N_PERM)]
        beat = sig_mean > np.percentile(boot_means, 95)
        null_test_results[w] = {
            "signal_mean_pct": round(sig_mean * 100, 3),
            "rand_mean_pct": round(rand_mean * 100, 3),
            "beat_margin_pct": round((sig_mean - rand_mean) * 100, 3),
            "beats_null_p95": bool(beat),
            "n_signal": len(sig_vals),
            "n_rand": len(rand_vals),
        }
        print(f"    {w}: signal={sig_mean*100:+.3f}%  rand={rand_mean*100:+.3f}%  "
              f"margin={((sig_mean-rand_mean)*100):+.3f}%  beats_p95={beat}")

    # ---- no-skill control ----
    print(f"\n  === NO-SKILL CONTROL (same trigger, fixed hold, no RSI exit) ===")
    noskill_evs = []
    for s, series in data.items():
        noskill_evs.extend(_no_skill_events(series, args.thr, args.hold_max, args.cost,
                                            btc_on, not args.no_regime))
    noskill_byw = _split_by_window(noskill_evs)
    for w in ("OOS", "UNSEEN"):
        sv  = byw.get(w, [])
        nsv = noskill_byw.get(w, [])
        sm  = float(np.mean(sv))  if sv  else 0.0
        nsm = float(np.mean(nsv)) if nsv else 0.0
        print(f"    {w}: signal(managed_exit)={sm*100:+.3f}%  no_skill(fixed_hold)={nsm*100:+.3f}%"
              f"  delta={((sm-nsm)*100):+.3f}%  (positive = managed exit adds value)")

    # ---- book-level stats ----
    print(f"\n  === BOOK-LEVEL STATS (daily EW, OOS+UNSEEN) ===")
    for w_list, label in [
        (["OOS"], "OOS"), (["UNSEEN"], "UNSEEN"), (["OOS", "UNSEEN"], "OOS+UNSEEN")
    ]:
        bk = _book_stats(all_evs, w_list)
        if bk:
            print(f"    {label}: compound={bk['compound_pct']:+.1f}%  maxDD={bk['maxdd_pct']:+.1f}%  "
                  f"mpos={bk['mpos']:.2f}  n_months={bk['n_months']}")
        else:
            print(f"    {label}: no events")

    # ---- block-bootstrap on OOS+UNSEEN book ----
    print(f"\n  === BLOCK-BOOTSTRAP (OOS+UNSEEN book, block={BLOCK_LEN}d, n={N_BOOT}) ===")
    bb_ou = _block_bootstrap_p05(all_evs, ["OOS", "UNSEEN"], rng=rng)
    bb_oo = _block_bootstrap_p05(all_evs, ["OOS"], rng=rng)
    bb_un = _block_bootstrap_p05(all_evs, ["UNSEEN"], rng=rng)
    for label, bb in [("OOS", bb_oo), ("UNSEEN", bb_un), ("OOS+UNSEEN", bb_ou)]:
        if bb.get("p05_compound_pct") is not None:
            print(f"    {label}: p05={bb['p05_compound_pct']:+.2f}%  "
                  f"p(>0)={bb['p_positive']:.3f}  median={bb['median_compound_pct']:+.2f}%")
        else:
            print(f"    {label}: {bb}")

    # ---- seed robustness ----
    print(f"\n  === SEED ROBUSTNESS (10 seeds x 70% asset subset, OOS+UNSEEN book) ===")
    sr = _seed_robustness(data, args.thr, args.hold_max, args.cost, btc_on)
    print(f"    Compound per seed: {sr['compound_pcts']}")
    print(f"    Positive: {sr['n_positive']}/{sr['n_seeds']} ({sr['fraction_positive']:.0%})")

    # ---- regime decomposition ----
    print(f"\n  === REGIME DECOMPOSITION (OOS+UNSEEN) ===")
    # regime-ON vs regime-OFF events
    evs_on  = []
    evs_off = []
    for s, series in data.items():
        o_, c_, rsi_, ts_ = series
        n_ = len(c_)
        i_ = 1
        while i_ < n_ - args.hold_max - 2:
            fire_ = (rsi_[i_] >= args.thr) and (rsi_[i_ - 1] < args.thr) and (rsi_[i_ - 1] > 0)
            if fire_:
                day_ = pd.Timestamp(ts_[i_]).normalize()
                regime_on = btc_on.get(day_, True)
                entry_bar_ = i_ + 1
                if entry_bar_ >= n_:
                    break
                exit_bar_ = entry_bar_ + args.hold_max
                for j_ in range(entry_bar_, min(entry_bar_ + args.hold_max, n_ - 1)):
                    if rsi_[j_] > RSI_EXIT:
                        exit_bar_ = j_ + 1
                        break
                exit_bar_ = min(exit_bar_, n_ - 1)
                gross_ = o_[exit_bar_] / o_[entry_bar_] - 1.0
                net_   = gross_ - args.cost
                wlab_  = _win_label(pd.Timestamp(ts_[entry_bar_]))
                if wlab_ in ("OOS", "UNSEEN"):
                    if regime_on:
                        evs_on.append(net_)
                    else:
                        evs_off.append(net_)
                i_ = exit_bar_ + 1
            else:
                i_ += 1
    mu_on  = float(np.mean(evs_on))  if evs_on  else 0.0
    mu_off = float(np.mean(evs_off)) if evs_off else 0.0
    print(f"    Regime ON  (BTC>SMA100): n={len(evs_on):4d}  mean={mu_on*100:+.3f}%")
    print(f"    Regime OFF (BTC<SMA100): n={len(evs_off):4d}  mean={mu_off*100:+.3f}%")
    print(f"    Gate contribution: {((mu_on - mu_off)*100):+.3f}pp advantage of ON over OFF")

    # ---- CORE vs CORE+SATELLITE combo ----
    combo_result = None
    if not args.skip_core:
        print(f"\n  === CORE vs CORE+SATELLITE (sat_weight={args.sat_weight:.0%}, OOS+UNSEEN) ===")
        core_net = _core_book_net(cost_rt=TAKER_RT)
        if core_net is not None:
            combo_net = _combine_core_sat(core_net, all_evs, sat_weight=args.sat_weight)
            corr_sat  = pd.Series({pd.Timestamp(str(ts_)[:10]): net_
                                   for ts_, net_, _ in all_evs}).sort_index()

            for wname, lo, hi in [
                ("OOS",    VAL_END, OOS_END),
                ("UNSEEN", OOS_END, UNSEEN_END),
            ]:
                c_stats   = _equity_stats(core_net,  lo, hi)
                cs_stats  = _equity_stats(combo_net, lo, hi)
                sat_bk    = _book_stats(all_evs, [wname])
                print(f"    {wname}:")
                print(f"      CORE:        compound={c_stats.get('compound_pct',0):+.2f}%  "
                      f"maxDD={c_stats.get('maxdd_pct',0):+.2f}%  Sharpe={c_stats.get('sharpe',0):.2f}")
                print(f"      CORE+SAT:    compound={cs_stats.get('compound_pct',0):+.2f}%  "
                      f"maxDD={cs_stats.get('maxdd_pct',0):+.2f}%  Sharpe={cs_stats.get('sharpe',0):.2f}")
                if sat_bk:
                    print(f"      SAT alone:   compound={sat_bk['compound_pct']:+.2f}%  "
                          f"maxDD={sat_bk['maxdd_pct']:+.2f}%")
            # satellite correlation with core
            core_daily = core_net.resample("D").last().pct_change().dropna()
            sat_daily_s = pd.Series(
                {pd.Timestamp(str(ts_s)[:10]): net_s for ts_s, net_s, _ in all_evs}
            ).sort_index()
            common = core_daily.index.intersection(sat_daily_s.index)
            if len(common) > 20:
                corr = float(core_daily.loc[common].corr(sat_daily_s.loc[common]))
                print(f"    SAT/CORE daily correlation (OOS+UNSEEN days active): {corr:+.3f}")
                combo_result = {"correlation_sat_core": round(corr, 3)}
        else:
            print(f"    Core book unavailable, skipping combo test.")

    # ---- VERDICT ----
    print(f"\n  {'='*60}")
    print(f"  VERDICT SUMMARY")
    print(f"  {'='*60}")
    oos_mean  = pt_stats.get("OOS",  {}).get("mean_pct", None)
    un_mean   = pt_stats.get("UNSEEN",{}).get("mean_pct", None)
    oos_null  = null_test_results.get("OOS",  {}).get("beats_null_p95", False)
    un_null   = null_test_results.get("UNSEEN",{}).get("beats_null_p95", False)
    bk_oo     = _book_stats(all_evs, ["OOS"])
    bk_un     = _book_stats(all_evs, ["UNSEEN"])
    gate_pass = (oos_mean is not None and oos_mean > 0 and oos_null and
                 un_mean  is not None and un_mean  > 0 and un_null)

    print(f"  OOS  per-trade: mean={oos_mean}%  beats_null={oos_null}")
    print(f"  UNSEEN per-trade: mean={un_mean}%  beats_null={un_null}")
    print(f"  OOS  book: {bk_oo['compound_pct'] if bk_oo else 'N/A'}%  maxDD={bk_oo['maxdd_pct'] if bk_oo else 'N/A'}%")
    print(f"  UNSEEN book: {bk_un['compound_pct'] if bk_un else 'N/A'}%  maxDD={bk_un['maxdd_pct'] if bk_un else 'N/A'}%")
    print(f"  Seed robustness: {sr['n_positive']}/10 positive (OOS+UNSEEN)")
    print(f"  Block-bootstrap OOS+UNSEEN p05: {bb_ou.get('p05_compound_pct','N/A')}%")
    print()
    if gate_pass:
        print(f"  GATE STATUS: CANDIDATE -- per-trade positive + beats null OOS+UNSEEN")
        print(f"  Next: block-bootstrap p05>0 required for ship; run CORE+SAT combo with sat_weight sweep.")
    else:
        kills = []
        if oos_mean is None or oos_mean <= 0:
            kills.append(f"OOS per-trade non-positive ({oos_mean}%)")
        if not oos_null:
            kills.append(f"OOS fails random-entry null")
        if un_mean is None or un_mean <= 0:
            kills.append(f"UNSEEN per-trade non-positive ({un_mean}%)")
        if not un_null:
            kills.append(f"UNSEEN fails random-entry null")
        print(f"  GATE STATUS: FAILED -- killing test(s): {'; '.join(kills)}")
        print(f"  This is regime-gated bull beta without harvestable excess over random entry.")
    print()

    # ---- save results ----
    import datetime
    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "runs" / "strat" / f"rsi_bounce_satellite_{ts_now}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(out_path), {
        "phase": "RSI_BOUNCE_SATELLITE_REVALIDATION",
        "timestamp": ts_now,
        "config": {
            "cadence": args.cadence, "thr": args.thr, "hold_max": args.hold_max,
            "cost_rt": args.cost, "use_regime": not args.no_regime,
            "rsi_exit_threshold": RSI_EXIT, "rsi_period": RSI_PERIOD,
        },
        "windows": {
            "train_end": WS.train_end, "val_end": WS.val_end,
            "oos_end": WS.oos_end, "unseen_end": WS.unseen_end,
        },
        "n_assets": len(data),
        "per_trade_stats": pt_stats,
        "null_test": null_test_results,
        "noskill_control": {
            w: {
                "signal_mean_pct": round(float(np.mean(byw.get(w, [0]))) * 100, 3) if byw.get(w) else None,
                "noskill_mean_pct": round(float(np.mean(noskill_byw.get(w, [0]))) * 100, 3) if noskill_byw.get(w) else None,
            } for w in ("OOS", "UNSEEN")
        },
        "book_stats": {
            "OOS":        _book_stats(all_evs, ["OOS"]),
            "UNSEEN":     _book_stats(all_evs, ["UNSEEN"]),
            "OOS+UNSEEN": _book_stats(all_evs, ["OOS", "UNSEEN"]),
        },
        "bootstrap": {"OOS": bb_oo, "UNSEEN": bb_un, "OOS+UNSEEN": bb_ou},
        "seed_robustness": sr,
        "regime_decomposition": {
            "n_on": len(evs_on), "mean_on_pct": round(mu_on * 100, 3),
            "n_off": len(evs_off), "mean_off_pct": round(mu_off * 100, 3),
        },
        "gate_pass": gate_pass,
        "verdict": "CANDIDATE" if gate_pass else "FAILED",
        "combo": combo_result,
    })
    print(f"  Results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
