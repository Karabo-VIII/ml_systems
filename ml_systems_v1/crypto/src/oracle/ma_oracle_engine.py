"""MA-family ORACLE / HINDSIGHT ENGINE.

================================================================================
HINDSIGHT UPPER BOUND -- descriptive, NOT a tradeable signal.
================================================================================

For any day D, this engine answers, for each top-25 performer as of D:
    "What is the best MA-family TI (config + entry day, which may be days before
     D) that would have captured the MAXIMUM realized return to D, and how close
     is that to the perfect-entry oracle?"

This is HINDSIGHT. The hindsight is ONLY in *selecting the best config after the
fact* -- that is the allowed oracle move. Each individual config's entry SIGNAL
is computed CAUSALLY (past-only up to each day; the moving averages at day t use
closes up to and including t, and a cross is detected from the sign change at
t-1 -> t with no future leak). The "best config" pick is the upper bound; the
mechanics that produce each candidate are honest.

Value of the engine:
  - the descriptive answer (which entry day, what captured return, how close to
    perfect) -- an UPPER BOUND on what an MA-family long entry could have done.
  - the DNA: which configs recur as "best" across days/assets -> the seed for a
    *predictive* adaptive-MA process later (out of scope here).

--------------------------------------------------------------------------------
KEY DEFINITIONS (all debatable choices are stated explicitly):

1. Daily key: we key everything off the chimera `date` column (a clean Date),
   NOT the raw `timestamp` (which carries intraday offsets in this dataset).

2. Top-performer ranking (rank_top_performers):
     perf(sym, D) = close[D] / close[D - lookback_days] - 1   (trailing return)
   ranked descending; top-25 returned. `lookback_days` is configurable and is
   ALSO the window used by best_ma_capture for the perfect-entry oracle, so the
   two are consistent (the "move that made it a top performer" and the "window
   we hunt the best entry in" are the same window).

3. MA grid (default): BOTH SMA and EMA crossovers.
     fast in {5, 10, 20}, slow in {20, 50, 100}, with fast < slow.
   -> 8 SMA pairs + 8 EMA pairs = 16 configs.

4. Entry / exit cross logic (CAUSAL, per config):
     - MA_fast(t), MA_slow(t) computed from closes up to and including day t.
     - GOLDEN CROSS (entry) at day t: spread(t-1) = fast(t-1)-slow(t-1) <= 0 AND
       spread(t) = fast(t)-slow(t) > 0  (fast crosses ABOVE slow).
     - DEATH CROSS (exit) at day t: spread(t-1) > 0 AND spread(t) <= 0.
     - "In position at D": the MOST RECENT golden cross at some day e <= D such
       that NO death cross occurs in (e, D]. i.e. the position opened at the last
       golden cross and has not been closed by a death cross on or before D.
       (If the last cross before/at D is a death cross, the config is FLAT at D
       and contributes no captured return.)
     - entry_date = day of that golden cross; captured_return =
       (close[D] - close[entry_date]) / close[entry_date].
   Debatable choice flagged: a config that is "in position at D" but whose entry
   is OUTSIDE the lookback window (cross older than D - lookback_days) is still
   allowed -- the cross "3 days ago" framing wants real (possibly older) entries.
   We do NOT clamp the entry into the window. days_back can exceed lookback_days.

5. Perfect-entry oracle (the denominator):
     perfect_return = (close[D] - min(close over [D-lookback_days, D])) /
                       min(close over that window)
   = the best possible LONG entry within the ranking window (buy the low close,
     hold to D).
   capture_rate = clip(best_config_captured / perfect_return, 0, 1).
   If perfect_return <= 0 (no up-move in window) capture_rate is defined as 0.0
   and flagged.

6. best_ma_capture picks the config with the MAX captured_return among configs
   that are "in position at D". If NO config is in position at D, returns a row
   with best_ti=None, captured_return=0.0, capture_rate=0.0.

--------------------------------------------------------------------------------
CLI:
    python src/oracle/ma_oracle_engine.py --date 2026-05-20 [--universe u100]
        [--lookback 30] [--out runs/oracle/ma_oracle_<date>.csv]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date, datetime as _dt
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "oracle_engine",
    "inputs": ["chimera 1d via pipeline.chimera_loader.ChimeraLoader"],
    "outputs": {
        "callable": "oracle(date, universe, lookback_days) -> pl.DataFrame",
        "csv": "runs/oracle/ma_oracle_<date>.csv",
    },
    "invariants": [
        "per-config MA + cross signal is CAUSAL (past-only up to each day)",
        "best-config selection is hindsight (the allowed oracle move)",
        "capture_rate in [0,1]; entry_date <= date; days_back >= 0",
        "output labeled HINDSIGHT UPPER BOUND -- not a tradeable signal",
        "no emoji in prints (cp1252)",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for p in (str(SRC), str(SRC / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

HINDSIGHT_LABEL = "HINDSIGHT UPPER BOUND -- descriptive, not a tradeable signal."

# Default MA grid: fast < slow, over both families.
DEFAULT_FAST = (5, 10, 20)
DEFAULT_SLOW = (20, 50, 100)


def _build_ma_grid(fast=DEFAULT_FAST, slow=DEFAULT_SLOW):
    """Return list of (family, fast, slow) with fast < slow, for SMA and EMA."""
    grid = []
    for fam in ("SMA", "EMA"):
        for f in fast:
            for s in slow:
                if f < s:
                    grid.append((fam, f, s))
    return grid


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    """Causal simple moving average; NaN until w-1 warmup is satisfied.

    Each output[t] uses only x[t-w+1 .. t] (past-only)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (csum[w:] - csum[:-w]) / w
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal exponential moving average, span-based (alpha = 2/(span+1)).

    Seeded with the SMA of the first `span` points so output[t] depends only on
    x[0..t] (past-only). NaN until the seed window (t < span-1) is filled."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    seed = x[:span].mean()
    out[span - 1] = seed
    prev = seed
    for t in range(span, n):
        prev = alpha * x[t] + (1.0 - alpha) * prev
        out[t] = prev
    return out


def _crosses(spread: np.ndarray):
    """Given spread = fast - slow (with NaN warmup), return (golden_idx, death_idx).

    golden cross at t: spread[t-1] <= 0 and spread[t] > 0   (both non-NaN)
    death  cross at t: spread[t-1] >  0 and spread[t] <= 0   (both non-NaN)
    Indices are into the same array. Causal: uses only t-1 and t."""
    n = len(spread)
    golden, death = [], []
    for t in range(1, n):
        a, b = spread[t - 1], spread[t]
        if np.isnan(a) or np.isnan(b):
            continue
        if a <= 0.0 and b > 0.0:
            golden.append(t)
        elif a > 0.0 and b <= 0.0:
            death.append(t)
    return golden, death


class MAOracleEngine:
    """Queryable MA-family hindsight engine.

    Loads daily closes lazily per asset (cached) from the chimera loader.
    """

    def __init__(self, loader: ChimeraLoader | None = None,
                 fast=DEFAULT_FAST, slow=DEFAULT_SLOW):
        self.loader = loader or ChimeraLoader()
        self.grid = _build_ma_grid(fast, slow)
        self._cache: dict[str, pl.DataFrame] = {}

    # ---- data access ----------------------------------------------------
    def _load_daily(self, sym: str) -> pl.DataFrame | None:
        """Return a (date, close) frame sorted ascending, or None if unavailable."""
        if sym in self._cache:
            return self._cache[sym]
        try:
            df = self.loader.load(sym, cadence="1d", features=["close", "date"])
        except Exception:
            self._cache[sym] = None
            return None
        if "date" not in df.columns or "close" not in df.columns:
            self._cache[sym] = None
            return None
        df = (df.select(["date", "close"])
                .drop_nulls()
                .unique(subset=["date"], keep="last")
                .sort("date"))
        self._cache[sym] = df
        return df

    def data_date_range(self, universe: str = "u100"):
        """Return (min_date, max_date) across all loadable assets in the universe."""
        mins, maxs = [], []
        for sym in self.loader.universes.list(universe):
            df = self._load_daily(sym)
            if df is not None and len(df):
                mins.append(df["date"].min())
                maxs.append(df["date"].max())
        if not mins:
            return None, None
        return min(mins), max(maxs)

    # ---- ranking --------------------------------------------------------
    def rank_top_performers(self, date, universe: str = "u100",
                            lookback_days: int = 30, top_n: int = 25):
        """Rank assets by trailing `lookback_days` return ending at `date`.

        perf(sym) = close[on/at D] / close[on/at (D - lookback_days)] - 1
        Uses the last available trading day <= date for D, and the last available
        day <= (D's date index - lookback_days positions) for the base. To keep
        it robust to gaps we index by POSITION in the asset's own date series:
        D_idx = last row with date <= `date`; base_idx = D_idx - lookback_days.
        Returns list[(sym, perf)] sorted desc, length <= top_n.
        """
        d = _to_date(date)
        rows = []
        for sym in self.loader.universes.list(universe):
            df = self._load_daily(sym)
            if df is None or len(df) == 0:
                continue
            dates = df["date"].to_list()
            closes = df["close"].to_numpy()
            d_idx = _last_idx_le(dates, d)
            if d_idx is None:
                continue
            base_idx = d_idx - lookback_days
            if base_idx < 0:
                continue
            c0 = closes[base_idx]
            cD = closes[d_idx]
            if c0 is None or c0 <= 0 or np.isnan(c0) or np.isnan(cD):
                continue
            perf = float(cD) / float(c0) - 1.0
            rows.append((sym, perf))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:top_n]

    # ---- per-asset best capture ----------------------------------------
    def best_ma_capture(self, sym: str, date, lookback_days: int = 30,
                        ma_grid=None) -> dict:
        """Best MA-family config capture for `sym` as of `date`.

        See module docstring section 4-6 for exact definitions.
        Returns a dict with keys:
          sym, best_ti, family, fast, slow, entry_date, days_back,
          captured_return, perfect_return, capture_rate, in_position, note.
        """
        grid = ma_grid if ma_grid is not None else self.grid
        d = _to_date(date)
        base = {
            "sym": sym, "best_ti": None, "family": None, "fast": None,
            "slow": None, "entry_date": None, "days_back": None,
            "captured_return": 0.0, "perfect_return": 0.0,
            "capture_rate": 0.0, "in_position": False, "note": "",
        }
        df = self._load_daily(sym)
        if df is None or len(df) == 0:
            base["note"] = "no data"
            return base
        dates = df["date"].to_list()
        closes = df["close"].to_numpy().astype(float)
        d_idx = _last_idx_le(dates, d)
        if d_idx is None:
            base["note"] = "date before first bar"
            return base

        # Perfect-entry oracle over the ranking window [D - lookback_days, D].
        win_lo = max(0, d_idx - lookback_days)
        window_closes = closes[win_lo:d_idx + 1]
        c_min = float(np.min(window_closes))
        cD = float(closes[d_idx])
        perfect_return = (cD - c_min) / c_min if c_min > 0 else 0.0
        base["perfect_return"] = perfect_return

        # Per-config causal signal over the FULL past series up to D (so a cross
        # older than the lookback window is still detectable -- the "crossed days
        # ago" case). We slice closes[:d_idx+1] so nothing past D is ever seen.
        past_closes = closes[:d_idx + 1]
        best = None  # (captured, family, fast, slow, entry_idx)
        for (fam, f, s) in grid:
            if fam == "SMA":
                ma_f = _sma(past_closes, f)
                ma_s = _sma(past_closes, s)
            else:
                ma_f = _ema(past_closes, f)
                ma_s = _ema(past_closes, s)
            spread = ma_f - ma_s
            golden, death = _crosses(spread)
            if not golden:
                continue
            last_golden = golden[-1]              # index into past_closes
            # in position at D iff no death cross strictly after the golden one.
            later_death = [dx for dx in death if dx > last_golden]
            if later_death:
                continue  # closed before/at D -> flat at D
            entry_idx = last_golden
            c_entry = past_closes[entry_idx]
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            captured = (cD - c_entry) / c_entry
            if best is None or captured > best[0]:
                best = (captured, fam, f, s, entry_idx)

        if best is None:
            base["note"] = "no config in position at D"
            return base

        captured, fam, f, s, entry_idx = best
        entry_date = dates[entry_idx]
        days_back = (d - entry_date).days
        cap_rate = 0.0
        if perfect_return > 0:
            cap_rate = max(0.0, min(1.0, captured / perfect_return))
        base.update({
            "best_ti": f"{fam}({f},{s})",
            "family": fam, "fast": f, "slow": s,
            "entry_date": entry_date,
            "days_back": int(days_back),
            "captured_return": float(captured),
            "perfect_return": float(perfect_return),
            "capture_rate": float(cap_rate),
            "in_position": True,
            "note": "" if perfect_return > 0 else "no up-move in window; cap_rate=0",
        })
        return base

    # ---- tie it together ------------------------------------------------
    def oracle(self, date, universe: str = "u100", lookback_days: int = 30,
               top_n: int = 25) -> pl.DataFrame:
        """Rank top-N performers, run best_ma_capture per asset, return a table."""
        ranked = self.rank_top_performers(date, universe, lookback_days, top_n)
        out_rows = []
        for rank, (sym, perf) in enumerate(ranked, start=1):
            cap = self.best_ma_capture(sym, date, lookback_days)
            out_rows.append({
                "sym": sym,
                "perf_rank": rank,
                "trailing_perf": round(perf, 6),
                "best_ti": cap["best_ti"],
                "entry_date": str(cap["entry_date"]) if cap["entry_date"] else None,
                "days_back": cap["days_back"],
                "captured_return": round(cap["captured_return"], 6),
                "perfect_return": round(cap["perfect_return"], 6),
                "capture_rate": round(cap["capture_rate"], 6),
                "in_position": cap["in_position"],
                "note": cap["note"],
            })
        if not out_rows:
            return pl.DataFrame()
        return pl.DataFrame(out_rows).sort("perf_rank")


# ---- helpers ------------------------------------------------------------
def _print_table_ascii(table: pl.DataFrame, max_rows: int = 25) -> None:
    """Print a polars DataFrame as a plain-ASCII table.

    Avoids polars' Unicode box-drawing glyphs, which crash on Windows cp1252
    stdout. ASCII-only -> safe everywhere."""
    cols = table.columns
    rows = table.head(max_rows).rows()
    cells = [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [len(c) for c in cols]
    for row in cells:
        for i, v in enumerate(row):
            if len(v) > widths[i]:
                widths[i] = len(v)
    sep = "-+-".join("-" * w for w in widths)
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(header)
    print(sep)
    for row in cells:
        print(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))


def _to_date(d) -> _date:
    if isinstance(d, _date) and not isinstance(d, _dt):
        return d
    if isinstance(d, _dt):
        return d.date()
    return _date.fromisoformat(str(d))


def _last_idx_le(dates: list, d: _date):
    """Index of the last date <= d in an ascending date list, or None."""
    lo, hi, ans = 0, len(dates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] <= d:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _reconcile_one(engine: MAOracleEngine, sym: str, date, lookback_days: int):
    """Print a hand reconciliation for one asset: close[D], close[entry], return."""
    cap = engine.best_ma_capture(sym, date, lookback_days)
    df = engine._load_daily(sym)
    dates = df["date"].to_list()
    closes = df["close"].to_numpy().astype(float)
    d_idx = _last_idx_le(dates, _to_date(date))
    cD = closes[d_idx]
    print("\n--- HAND RECONCILIATION (one asset) ---")
    print(f"  asset            : {sym}")
    print(f"  query date D     : {date}  (resolved trading day: {dates[d_idx]})")
    print(f"  close[D]         : {cD:.8f}")
    if cap["entry_date"] is None:
        print("  (no config in position at D -- nothing to reconcile)")
        return
    e_idx = _last_idx_le(dates, cap["entry_date"])
    c_entry = closes[e_idx]
    manual = (cD - c_entry) / c_entry
    print(f"  best_ti          : {cap['best_ti']}")
    print(f"  entry_date       : {cap['entry_date']}  (days_back={cap['days_back']})")
    print(f"  close[entry]     : {c_entry:.8f}")
    print(f"  manual return    : (close[D]-close[entry])/close[entry] = "
          f"({cD:.8f}-{c_entry:.8f})/{c_entry:.8f} = {manual:.6f}")
    print(f"  engine captured  : {cap['captured_return']:.6f}")
    print(f"  match            : {abs(manual - cap['captured_return']) < 1e-9}")
    print(f"  perfect_return   : {cap['perfect_return']:.6f}  "
          f"capture_rate={cap['capture_rate']:.6f}")


def main():
    ap = argparse.ArgumentParser(description=HINDSIGHT_LABEL)
    ap.add_argument("--date", required=True, help="query day D, YYYY-MM-DD")
    ap.add_argument("--universe", default="u100")
    ap.add_argument("--lookback", type=int, default=30,
                    help="trailing-return ranking window AND perfect-entry window")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reconcile", action="store_true",
                    help="also print a hand reconciliation for the rank-1 asset")
    args = ap.parse_args()

    engine = MAOracleEngine()
    table = engine.oracle(args.date, args.universe, args.lookback, args.top_n)

    print("=" * 78)
    print(HINDSIGHT_LABEL)
    print(f"MA-ORACLE  date={args.date}  universe={args.universe}  "
          f"lookback={args.lookback}d  top_n={args.top_n}")
    print(f"grid: {len(engine.grid)} configs (SMA+EMA, "
          f"fast{list(DEFAULT_FAST)} x slow{list(DEFAULT_SLOW)}, fast<slow)")
    print("=" * 78)
    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return
    _print_table_ascii(table, max_rows=args.top_n)

    out = args.out or str(PROJECT_ROOT / "runs" / "oracle" /
                          f"ma_oracle_{args.date}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    print(f"\nwrote: {out}")

    if args.reconcile and not table.is_empty():
        top_sym = table["sym"][0]
        _reconcile_one(engine, top_sym, args.date, args.lookback)


if __name__ == "__main__":
    main()
