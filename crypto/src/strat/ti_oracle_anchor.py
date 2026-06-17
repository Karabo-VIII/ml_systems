"""PRICE-ORACLE vs TI-ORACLE anchoring tool.

Anchors the discovery phase by quantifying, in hindsight, how much of an
available price move an EMA/MA (technical-indicator) family could have captured.

This is an ANCHOR, not a pass/fail gate. The TI-oracle being below the
price-oracle is EXPECTED ("it's a start"). No "null" verdict is emitted.

Spec (encoded literally):
  PRICE ORACLE (per event):
    - Scan the chart for ALL distinct move-events. A move-event = a 7-14 day
      window in which a LONG round-trip of 2-10% was available: enter at a local
      low, exit at the highest high AFTER that low within the window.
    - PRICE-ORACLE ROI = (highest_high_after_low - low) / low. Keep events where
      this is in [2%, 10%].
    - 7-14 days -> bars per cadence: 1d 7-14, 4h 42-84, 1h 168-336, 15m 672-1344.
    - NON-OVERLAPPING events ("genuinely anywhere" without double-counting).
    - Report the COUNT of events per timeframe.

  TI ORACLE (per event, EMA/MA only):
    - Sweep MA/EMA configs: type in {SMA, EMA} x (fast,slow) grid with
      fast in {5,10,20,50}, slow in {20,50,100,200}, fast < slow.
    - For each config simulate a CAUSAL long: enter on golden cross (fast crosses
      above slow), exit on death cross or window end; next-bar-open fills; net
      taker 0.24% RT.
    - The config's captured ROI = realized long return within the event window.
    - TI-ORACLE ROI for the event = MAX captured ROI over all configs
      (best-in-HINDSIGHT). Record the WINNING config (the DNA).
    - MA warmup may look back BEFORE the window (causal signal, only the config
      CHOICE is hindsight).

  REPORT (per timeframe AND aggregate):
    - N events; mean & median PRICE-ORACLE ROI; mean & median TI-ORACLE ROI;
      CAPTURE RATIO (TI-oracle / price-oracle); winning-config DISTRIBUTION (DNA).
    - JSON artifact + clean printed table.

DISCIPLINE:
  - HINDSIGHT by design (config selection uses future knowledge -- that IS the
    TI-oracle; UNSEEN-once is NOT imposed on the oracle).
  - Each config's signal is CAUSAL within its window (no look-ahead in MA
    computation; next-bar fills).
  - cp1252-safe (no emoji).

Usage:
    python src/strat/ti_oracle_anchor.py --asset BTCUSDT --cadences 1d,4h,1h,15m
    python src/strat/ti_oracle_anchor.py --selftest

__contract__ = {
    "kind": "research_anchor",
    "inputs": ["chimera OHLC via ChimeraLoader"],
    "outputs": ["runs/strat/ti_oracle_anchor_<ASSET>.json", "stdout table"],
    "invariants": [
        "MA signals causal within window (warmup lookback only)",
        "next-bar-open fills",
        "non-overlapping price-oracle events",
        "config selection is hindsight by design (the TI-oracle)",
        "no pass/fail or null verdict -- this is an ANCHOR",
        "price-oracle ROI clamped to [2%, 10%]",
    ],
}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---- spec constants ---------------------------------------------------------

# 7-14 days -> bars per cadence (literal from spec).
WINDOW_BARS = {
    "1d": (7, 14),
    "4h": (42, 84),
    "1h": (168, 336),
    "15m": (672, 1344),
}

PRICE_ROI_LO = 0.02   # 2%
PRICE_ROI_HI = 0.10   # 10%

# MA/EMA sweep grid (literal from spec).
MA_TYPES = ("SMA", "EMA")
FAST_GRID = (5, 10, 20, 50)
SLOW_GRID = (20, 50, 100, 200)

TAKER_RT = 0.0024     # net taker 0.24% round-trip


# ---- MA primitives (causal) -------------------------------------------------

def sma(x: np.ndarray, n: int) -> np.ndarray:
    """Causal simple moving average. out[i] uses x[i-n+1..i]. NaN until warm."""
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (csum[n:] - csum[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Causal exponential moving average. Seeded by SMA(n) at index n-1.

    out[i] depends only on x[0..i] (no look-ahead).
    """
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    seed = float(np.mean(x[:n]))
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(x)):
        prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def moving_avg(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    return sma(x, n) if kind == "SMA" else ema(x, n)


# ---- price oracle -----------------------------------------------------------

@dataclass
class MoveEvent:
    start: int           # window start index (global)
    end: int             # window end index (global, exclusive)
    low_idx: int         # index of the entry low (global)
    high_idx: int        # index of the exit high after the low (global)
    price_roi: float     # (high - low)/low, clamped target in [0.02, 0.10]


def find_price_oracle_events(
    high: np.ndarray,
    low: np.ndarray,
    win_lo: int,
    win_hi: int,
) -> list[MoveEvent]:
    """Scan the chart for ALL distinct, NON-OVERLAPPING 7-14 day move-events
    where a LONG round-trip of 2-10% was available.

    Greedy left-to-right: at each cursor, search window lengths in [win_lo, win_hi]
    for the best available long (lowest low, then highest high AFTER that low) whose
    ROI falls in [2%, 10%]. Take the first qualifying window (smallest length that
    qualifies, maximizing its ROI), record it, and advance the cursor PAST it so
    events never overlap. If none qualifies at the cursor, advance by one bar.
    """
    n = len(low)
    events: list[MoveEvent] = []
    cursor = 0
    while cursor + win_lo <= n:
        best: MoveEvent | None = None
        # Try each window length; keep the first length that yields an in-band ROI,
        # choosing within that length the (low -> highest-high-after-low) pair.
        for wlen in range(win_lo, win_hi + 1):
            end = cursor + wlen
            if end > n:
                break
            lo_slice = low[cursor:end]
            hi_slice = high[cursor:end]
            # Entry at the lowest low in the window.
            li = int(np.argmin(lo_slice))
            lo_val = float(lo_slice[li])
            if lo_val <= 0:
                continue
            # Exit at the highest high AFTER the low (within window).
            if li + 1 >= len(hi_slice):
                continue
            hi_after = hi_slice[li + 1:]
            hj = int(np.argmax(hi_after))
            hi_val = float(hi_after[hj])
            roi = (hi_val - lo_val) / lo_val
            if PRICE_ROI_LO <= roi <= PRICE_ROI_HI:
                best = MoveEvent(
                    start=cursor,
                    end=end,
                    low_idx=cursor + li,
                    high_idx=cursor + li + 1 + hj,
                    price_roi=roi,
                )
                break  # smallest qualifying window length wins
        if best is not None:
            events.append(best)
            cursor = best.end       # NON-OVERLAPPING: jump past the window
        else:
            cursor += 1
    return events


# ---- TI oracle --------------------------------------------------------------

def precompute_configs(close_full: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute (f, s) MA pairs ONCE per cadence for every config in the grid.

    Avoids recomputing full-series MAs per event (O(events*configs*N) -> O(configs*N)).
    Memoizes each unique (kind, length) MA so EMA's Python loop runs at most once
    per (type, length).
    """
    ma_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_ma(kind: str, length: int) -> np.ndarray:
        key = (kind, length)
        if key not in ma_cache:
            ma_cache[key] = moving_avg(close_full, length, kind)
        return ma_cache[key]

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in MA_TYPES:
        for fast in FAST_GRID:
            for slow in SLOW_GRID:
                if fast >= slow:
                    continue
                out[f"{kind}_{fast}_{slow}"] = (get_ma(kind, fast), get_ma(kind, slow))
    return out


def causal_ma_long_return(
    f: np.ndarray,
    s: np.ndarray,
    open_full: np.ndarray,
    win_start: int,
    win_end: int,
) -> float:
    """Simulate ONE causal MA/EMA long over [win_start, win_end) given precomputed
    full-series fast (f) and slow (s) MA arrays.

    - MAs are computed on the FULL series (warmup lookback before the window is
      allowed -- causal; only later bars feed each MA point).
    - Enter on golden cross (fast crosses above slow) detected at bar t; fill at
      NEXT bar OPEN (t+1). If already-long at the window's first actionable bar
      (cross fired during warmup), enter at the window's first open.
    - Exit on death cross detected at bar t (fill next bar open) OR at window end
      (fill at the last in-window open if still long).
    - Multiple in-window crosses are traded sequentially; total return accumulates.
    - Net taker 0.24% RT charged per completed round trip.

    Returns realized long return (fraction) within the window.
    """
    # cross state at bar t: fast > slow ?
    above = f > s
    n = len(open_full)

    # We only ACT on signals whose fill bar lands inside [win_start, win_end).
    total_ret = 0.0
    in_pos = False
    entry_px = 0.0

    # Window-start state entry: if the MA state is ALREADY long at the first
    # actionable in-window bar (the golden cross fired during warmup, before the
    # window), enter at the window's first open. This lets the TI-oracle capture a
    # trend already in progress -- still causal (the cross is a past fact), only the
    # config CHOICE is hindsight. Charge entry; the round-trip cost is applied on
    # exit/force-close below.
    ws = win_start
    if (ws < n and not np.isnan(f[ws]) and not np.isnan(s[ws])
            and ws - 1 >= 0 and not np.isnan(f[ws - 1]) and not np.isnan(s[ws - 1])
            and above[ws]):
        entry_px = float(open_full[ws])
        if entry_px > 0:
            in_pos = True

    # Start scanning a little before the window so a cross that fires at win_start-1
    # can fill at win_start. But fills must land within the window to count.
    t0 = max(1, win_start - 1)
    for t in range(t0, win_end):
        if np.isnan(f[t]) or np.isnan(s[t]) or np.isnan(f[t - 1]) or np.isnan(s[t - 1]):
            continue
        golden = (not above[t - 1]) and above[t]
        death = above[t - 1] and (not above[t])
        fill_idx = t + 1  # next-bar-open fill
        if not in_pos and golden:
            if win_start <= fill_idx < win_end:
                entry_px = float(open_full[fill_idx])
                if entry_px > 0:
                    in_pos = True
        elif in_pos and death:
            if fill_idx < win_end and fill_idx < n:
                exit_px = float(open_full[fill_idx])
            else:
                exit_px = float(open_full[win_end - 1])
            if entry_px > 0:
                gross = exit_px / entry_px - 1.0
                total_ret += gross - TAKER_RT
            in_pos = False

    # Force-close at window end if still long.
    if in_pos and entry_px > 0:
        exit_px = float(open_full[win_end - 1])
        gross = exit_px / entry_px - 1.0
        total_ret += gross - TAKER_RT

    return total_ret


def ti_oracle_for_event(
    configs: dict[str, tuple[np.ndarray, np.ndarray]],
    open_full: np.ndarray,
    ev: MoveEvent,
) -> tuple[float, str]:
    """Best-in-hindsight captured ROI over the full MA/EMA config grid for one
    event. `configs` = precomputed {label: (f_arr, s_arr)}.
    Returns (best_roi, winning_config_label).
    """
    best_roi = -np.inf
    best_cfg = "NONE"
    for label, (f, s) in configs.items():
        roi = causal_ma_long_return(f, s, open_full, ev.start, ev.end)
        if roi > best_roi:
            best_roi = roi
            best_cfg = label
    if best_roi == -np.inf:
        best_roi = 0.0
        best_cfg = "NONE"
    return best_roi, best_cfg


# ---- per-cadence driver -----------------------------------------------------

@dataclass
class CadenceResult:
    cadence: str
    n_events: int
    price_roi: list[float] = field(default_factory=list)
    ti_roi: list[float] = field(default_factory=list)
    winning_cfgs: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        pr = np.array(self.price_roi, dtype=np.float64)
        ti = np.array(self.ti_roi, dtype=np.float64)
        if len(pr) == 0:
            return {
                "cadence": self.cadence,
                "n_events": 0,
                "price_oracle_mean": None,
                "price_oracle_median": None,
                "ti_oracle_mean": None,
                "ti_oracle_median": None,
                "capture_ratio_of_means": None,
                "capture_ratio_median_of_per_event": None,
                "winning_config_distribution": {},
            }
        pr_mean = float(np.mean(pr))
        ti_mean = float(np.mean(ti))
        # per-event capture ratio (guard div-by-zero with the in-band floor 0.02).
        per_event = ti / np.where(pr == 0.0, np.nan, pr)
        cap_median = float(np.nanmedian(per_event))
        dist = dict(Counter(self.winning_cfgs).most_common())
        return {
            "cadence": self.cadence,
            "n_events": self.n_events,
            "price_oracle_mean": pr_mean,
            "price_oracle_median": float(np.median(pr)),
            "ti_oracle_mean": ti_mean,
            "ti_oracle_median": float(np.median(ti)),
            "capture_ratio_of_means": (ti_mean / pr_mean) if pr_mean != 0 else None,
            "capture_ratio_median_of_per_event": cap_median,
            "winning_config_distribution": dist,
        }


def run_cadence(open_a, high_a, low_a, close_a, cadence: str) -> CadenceResult:
    win_lo, win_hi = WINDOW_BARS[cadence]
    events = find_price_oracle_events(high_a, low_a, win_lo, win_hi)
    res = CadenceResult(cadence=cadence, n_events=len(events))
    configs = precompute_configs(close_a)
    for ev in events:
        ti_roi, cfg = ti_oracle_for_event(configs, open_a, ev)
        res.price_roi.append(ev.price_roi)
        res.ti_roi.append(ti_roi)
        res.winning_cfgs.append(cfg)
    return res


def aggregate(results: list[CadenceResult]) -> dict:
    all_pr, all_ti, all_cfg = [], [], []
    for r in results:
        all_pr.extend(r.price_roi)
        all_ti.extend(r.ti_roi)
        all_cfg.extend(r.winning_cfgs)
    agg = CadenceResult(cadence="AGGREGATE", n_events=len(all_pr))
    agg.price_roi = all_pr
    agg.ti_roi = all_ti
    agg.winning_cfgs = all_cfg
    return agg.summary()


# ---- reporting --------------------------------------------------------------

def _fmt(v, pct=True):
    if v is None:
        return "   n/a"
    if pct:
        return f"{v * 100:6.2f}%"
    return f"{v:6.3f}"


def print_table(per_cadence: list[dict], agg: dict) -> None:
    print("")
    print("=" * 92)
    print("PRICE-ORACLE vs TI-ORACLE ANCHOR (EMA/MA family, hindsight by design)")
    print("=" * 92)
    hdr = (f"{'cadence':>10} | {'N':>4} | {'price mean':>10} {'price med':>9} | "
           f"{'ti mean':>10} {'ti med':>9} | {'cap(means)':>10} {'cap(med/ev)':>11}")
    print(hdr)
    print("-" * 92)
    for s in per_cadence + [agg]:
        print(f"{s['cadence']:>10} | {s['n_events']:>4} | "
              f"{_fmt(s['price_oracle_mean']):>10} {_fmt(s['price_oracle_median']):>9} | "
              f"{_fmt(s['ti_oracle_mean']):>10} {_fmt(s['ti_oracle_median']):>9} | "
              f"{_fmt(s['capture_ratio_of_means'], pct=False):>10} "
              f"{_fmt(s['capture_ratio_median_of_per_event'], pct=False):>11}")
    print("-" * 92)
    print("winning-config DNA (best-capturing MA/EMA configs):")
    for s in per_cadence + [agg]:
        dist = s["winning_config_distribution"]
        if not dist:
            print(f"  {s['cadence']:>10}: (no events)")
            continue
        top = list(dist.items())[:6]
        top_str = ", ".join(f"{k}:{v}" for k, v in top)
        print(f"  {s['cadence']:>10}: {top_str}")
    print("=" * 92)
    print("NOTE: ANCHOR (not a gate). TI-oracle < price-oracle is EXPECTED. No "
          "pass/fail / null verdict.")
    print("")


# ---- data loading -----------------------------------------------------------

def load_ohlc(asset: str, cadence: str):
    from pipeline.chimera_loader import ChimeraLoader
    loader = ChimeraLoader()
    df = loader.load(asset, cadence=cadence,
                     features=["open", "high", "low", "close"])
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    lo = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    return o, h, lo, c


# ---- selftest ---------------------------------------------------------------

def _synth_clean_trend(n=400, seed=0):
    """A clean staircase: long smooth up-legs an MA can ride end-to-end."""
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        # gentle persistent uptrend with tiny noise -> MA captures most of it
        drift = 0.004
        price.append(price[-1] * (1.0 + drift + rng.normal(0, 0.0008)))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0008
    lo = p * 0.9992
    return o, h, lo, c


def _synth_choppy(n=400, seed=1):
    """Choppy mean-reverting noise: 2-10% moves exist but MA whipsaws -> low capture."""
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        # oscillation + noise, no persistent trend
        osc = 0.03 * np.sin(i / 3.0)
        price.append(price[-1] * (1.0 + osc * 0.2 + rng.normal(0, 0.006)))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.002
    lo = p * 0.998
    return o, h, lo, c


def selftest() -> bool:
    """Synthetic clean-trend -> high TI capture; choppy -> TI << price-oracle."""
    ok = True

    # Use 1d window bars (7-14) on synthetic series.
    win_lo, win_hi = WINDOW_BARS["1d"]

    # Clean trend
    o, h, lo, c = _synth_clean_trend()
    ev_clean = find_price_oracle_events(h, lo, win_lo, win_hi)
    cfg_clean = precompute_configs(c)
    caps_clean = []
    for ev in ev_clean:
        ti, _ = ti_oracle_for_event(cfg_clean, o, ev)
        caps_clean.append(ti / ev.price_roi if ev.price_roi > 0 else 0.0)
    mean_cap_clean = float(np.mean(caps_clean)) if caps_clean else 0.0

    # Choppy
    o2, h2, lo2, c2 = _synth_choppy()
    ev_chop = find_price_oracle_events(h2, lo2, win_lo, win_hi)
    cfg_chop = precompute_configs(c2)
    caps_chop = []
    for ev in ev_chop:
        ti, _ = ti_oracle_for_event(cfg_chop, o2, ev)
        caps_chop.append(ti / ev.price_roi if ev.price_roi > 0 else 0.0)
    mean_cap_chop = float(np.mean(caps_chop)) if caps_chop else 0.0

    print(f"[selftest] clean-trend: {len(ev_clean)} events, "
          f"mean capture-ratio={mean_cap_clean:.3f}")
    print(f"[selftest] choppy:      {len(ev_chop)} events, "
          f"mean capture-ratio={mean_cap_chop:.3f}")

    # Assertions: clean must have events and meaningfully higher capture than choppy.
    if len(ev_clean) == 0:
        print("[selftest] FAIL: clean-trend produced no price-oracle events")
        ok = False
    if len(ev_chop) == 0:
        print("[selftest] FAIL: choppy produced no price-oracle events")
        ok = False
    if ok and not (mean_cap_clean > mean_cap_chop):
        print("[selftest] FAIL: clean capture not greater than choppy capture")
        ok = False
    if ok and not (mean_cap_clean > 0.5):
        print("[selftest] FAIL: clean-trend capture-ratio not high (>0.5)")
        ok = False

    line = ("[selftest] PASS: clean-trend -> high TI capture; choppy -> TI << "
            "price-oracle") if ok else "[selftest] FAIL"
    print(line)
    return ok


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PRICE-ORACLE vs TI-ORACLE anchor")
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cadences", default="1d,4h,1h,15m")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    results: list[CadenceResult] = []
    per_cadence_summ: list[dict] = []
    for cad in cadences:
        print(f"[run] {args.asset} {cad}: loading + scanning ...", flush=True)
        o, h, lo, c = load_ohlc(args.asset, cad)
        res = run_cadence(o, h, lo, c, cad)
        results.append(res)
        s = res.summary()
        per_cadence_summ.append(s)
        print(f"[run] {args.asset} {cad}: {res.n_events} events", flush=True)

    agg = aggregate(results)
    print_table(per_cadence_summ, agg)

    asset_short = args.asset.upper().replace("USDT", "")
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" /
                f"ti_oracle_anchor_{asset_short}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "ti_oracle_anchor",
        "anchor_not_gate": True,
        "asset": args.asset,
        "cadences": cadences,
        "spec": {
            "price_roi_band": [PRICE_ROI_LO, PRICE_ROI_HI],
            "window_bars": {c: WINDOW_BARS[c] for c in cadences},
            "ma_types": list(MA_TYPES),
            "fast_grid": list(FAST_GRID),
            "slow_grid": list(SLOW_GRID),
            "taker_rt": TAKER_RT,
            "fills": "next_bar_open",
            "events": "non_overlapping",
            "hindsight": "config_choice_only (signals causal)",
        },
        "per_cadence": per_cadence_summ,
        "aggregate": agg,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
