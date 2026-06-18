"""capture_rate_probe.py -- Minimal, transparent capture-rate lens.

Setup: Donchian-20 breakout entry + Donchian-10 trailing stop (or max-hold 60 bars).
Assets: BTC + ETH, daily bars, 2023-01-01 to 2025-12-31.

Capture-rate = realized_return / available_move per trade (cost-free).
Coverage = fraction of big UP-moves (>= +10% from a local low) during which
           the setup held a capturing position.

Run:
    python crypto/src/strat/capture_rate_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import polars as pl
import polars.selectors as cs

# --- path setup (works both as script and module) ---
_CRYPTO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_CRYPTO_SRC) not in sys.path:
    sys.path.insert(0, str(_CRYPTO_SRC))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

# ------------------------------------------------------------------ config ---
ASSETS      = ["BTC", "ETH"]
START_DATE  = "2023-01-01"
END_DATE    = "2025-12-31"
DC_ENTRY    = 20    # breakout window (bars)
DC_STOP     = 10    # trailing stop window (bars)
MAX_HOLD    = 60    # bars (hard cap, ~2 months)
BIG_MOVE_PCT = 10.0  # a "big up-move" threshold (%)
TAKER_RT_BPS = 24   # round-trip taker cost (bps)

def load_daily(sym: str) -> pl.DataFrame:
    loader = ChimeraLoader()
    df = loader.load(
        sym, cadence="1d",
        features=["timestamp", "open", "high", "low", "close"],
        date_range=(START_DATE, END_DATE),
    )
    # timestamp is Unix milliseconds (Int64) -- convert to Date
    df = df.with_columns(
        (pl.col("timestamp") * 1_000)          # ms -> us
        .cast(pl.Datetime("us"))
        .cast(pl.Date)
        .alias("date")
    )
    df = df.sort("date")
    return df


def compute_donchian(df: pl.DataFrame, window: int, col: str) -> pl.Series:
    """Rolling max(high) or min(low) over the PRIOR `window` bars (causal, shift-1)."""
    if col == "high":
        rolling = df["high"].rolling_max(window_size=window)
    else:
        rolling = df["low"].rolling_min(window_size=window)
    # shift by 1 so bar t uses only bars 0..t-1
    return rolling.shift(1)


def run_trades(df: pl.DataFrame, asset: str) -> list[dict]:
    """Simulate the Donchian breakout + trailing-stop strategy.

    Entry signal on bar t: close[t] > max(high[t-20..t-1])
    Enter at open[t+1] (next bar open, lag-1 causal)
    Exit: close[t] < min(low[t-10..t-1]) OR max_hold reached
    """
    closes  = df["close"].to_list()
    highs   = df["high"].to_list()
    lows    = df["low"].to_list()
    opens   = df["open"].to_list()
    dates   = df["date"].to_list()
    n       = len(closes)

    trades = []
    in_trade = False
    entry_bar = None
    entry_price = None
    entry_date = None
    peak_close = None

    for i in range(DC_ENTRY + 1, n - 1):  # need i+1 for entry open
        if not in_trade:
            # entry signal: today's close > max of prior DC_ENTRY highs
            prior_high = max(highs[i - DC_ENTRY: i])
            if closes[i] > prior_high:
                in_trade = True
                entry_bar = i + 1         # enter at NEXT bar's open
                entry_price = opens[i + 1]
                entry_date = dates[i + 1]
                peak_close = closes[i + 1]
        else:
            j = i  # current bar index (we entered at entry_bar)
            # update running peak
            if closes[j] > peak_close:
                peak_close = closes[j]

            # exit conditions
            prior_low = min(lows[j - DC_STOP: j]) if j >= DC_STOP else lows[0]
            hit_stop  = closes[j] < prior_low
            hit_cap   = (j - entry_bar + 1) >= MAX_HOLD

            if hit_stop or hit_cap or j == n - 2:
                exit_price = closes[j]
                exit_date  = dates[j]
                duration   = j - entry_bar + 1

                realized   = (exit_price - entry_price) / entry_price
                available  = (peak_close - entry_price) / entry_price
                # available is always >= 0 if peak >= entry (almost always true)
                # if price never exceeded entry, available = 0 -> capture = 0
                if available > 0.001:   # avoid division noise
                    capture_rate = max(0.0, min(realized / available, 1.0))
                elif realized > 0:
                    capture_rate = 1.0  # tiny move, consider 100% captured
                else:
                    capture_rate = 0.0

                trades.append({
                    "asset":        asset,
                    "entry_date":   entry_date,
                    "exit_date":    exit_date,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "peak_close":   peak_close,
                    "realized_pct": realized * 100,
                    "available_pct": available * 100,
                    "capture_rate": capture_rate,
                    "duration_bars": duration,
                    "exit_reason":  "stop" if hit_stop else ("cap" if hit_cap else "eod"),
                })
                in_trade = False
                entry_bar = None
                entry_price = None
                peak_close = None

    return trades


def find_big_moves(df: pl.DataFrame, threshold_pct: float) -> list[dict]:
    """Identify forward swings >= threshold_pct from a local low.

    A swing starts at any bar that is a local minimum (lower than the bars
    either side), and ends at the highest close in the next 30 bars.
    Report swings whose magnitude >= threshold_pct.
    """
    closes = df["close"].to_list()
    dates  = df["date"].to_list()
    n      = len(closes)
    moves  = []

    i = 1
    while i < n - 1:
        # local low: lower than both neighbours
        if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
            base_price = closes[i]
            base_date  = dates[i]
            # find peak in next 30 bars
            window = min(30, n - i - 1)
            future_closes = closes[i + 1: i + 1 + window]
            peak_idx_rel  = future_closes.index(max(future_closes))
            peak_price    = future_closes[peak_idx_rel]
            peak_date     = dates[i + 1 + peak_idx_rel]
            swing_pct     = (peak_price - base_price) / base_price * 100
            if swing_pct >= threshold_pct:
                moves.append({
                    "start_date": base_date,
                    "peak_date":  peak_date,
                    "swing_pct":  swing_pct,
                })
            i += peak_idx_rel + 2  # skip to after the peak
        else:
            i += 1

    return moves


def coverage_of_big_moves(
    trades: list[dict],
    big_moves: list[dict],
) -> float:
    """Fraction of big moves during which at least one trade was open."""
    if not big_moves:
        return float("nan")

    covered = 0
    for bm in big_moves:
        s, e = bm["start_date"], bm["peak_date"]
        for t in trades:
            if t["entry_date"] <= e and t["exit_date"] >= s:
                covered += 1
                break

    return covered / len(big_moves)


def summarise_trades(trades: list[dict]) -> dict:
    if not trades:
        return {}
    realized   = [t["realized_pct"]  for t in trades]
    available  = [t["available_pct"] for t in trades]
    captures   = [t["capture_rate"]  for t in trades]
    durations  = [t["duration_bars"] for t in trades]

    def med(x): return sorted(x)[len(x) // 2]
    def mean(x): return sum(x) / len(x)

    wins = sum(1 for r in realized if r > 0)

    return {
        "n_trades":          len(trades),
        "win_rate":          wins / len(trades),
        "median_realized_pct": med(realized),
        "mean_realized_pct":   mean(realized),
        "median_available_pct": med(available),
        "median_capture_rate": med(captures),
        "mean_capture_rate":   mean(captures),
        "avg_duration_bars":   mean(durations),
    }


def pick_examples(trades: list[dict], n: int = 5) -> list[dict]:
    """Return n diverse examples spread across the sorted trade list."""
    if len(trades) <= n:
        return trades
    step = len(trades) // n
    return [trades[i * step] for i in range(n)]


def print_report(all_results: dict) -> None:
    divider = "-" * 72

    print()
    print("=" * 72)
    print("  CAPTURE-RATE PROBE  --  Donchian-20 Breakout, Daily Bars")
    print("=" * 72)

    # 1. Spec recap
    print()
    print("SPEC RECAP")
    print(divider)
    print(f"  Assets        : BTC, ETH")
    print(f"  Span          : {START_DATE} to {END_DATE}  (in-sample / illustrative)")
    print(f"  Entry setup   : Daily close > max(high) of prior {DC_ENTRY} bars")
    print(f"  Entry fill    : NEXT bar's open  (lag-1, causal)")
    print(f"  Exit policy   : Close < min(low) of prior {DC_STOP} bars  OR  {MAX_HOLD}-bar cap")
    print(f"  Cost          : NONE in capture_rate; taker RT noted separately")
    print(f"  Big-move def  : >= +{BIG_MOVE_PCT}% forward swing from a local daily low")
    print()
    print(f"  Capture-rate  = realized / available (entry->peak) per trade, [0,1]")
    print(f"  Coverage      = fraction of big up-moves during which setup was LONG")

    # 2. Summary table
    print()
    print("SUMMARY TABLE")
    print(divider)
    hdr = (f"{'Asset':<6}  {'Trades':>6}  {'WinRate':>7}  {'MedReal%':>8}  "
           f"{'MedAvail%':>9}  {'MedCap':>7}  {'AvgCap':>7}  {'Coverage':>8}  {'AvgDur(d)':>9}")
    print(hdr)
    print(divider)
    for sym in ASSETS:
        r = all_results[sym]
        s = r["summary"]
        cov = r["coverage"]
        cov_str = f"{cov:.1%}" if cov == cov else "N/A"  # nan check
        print(
            f"{sym:<6}  {s['n_trades']:>6}  {s['win_rate']:>7.1%}  "
            f"{s['median_realized_pct']:>8.1f}  {s['median_available_pct']:>9.1f}  "
            f"{s['median_capture_rate']:>7.2f}  {s['mean_capture_rate']:>7.2f}  "
            f"{cov_str:>8}  {s['avg_duration_bars']:>9.1f}"
        )

    # 3. Example trades
    print()
    print("EXAMPLE TRADES  (5 per asset, spread across the period)")
    print(divider)
    ehdr = (f"{'Asset':<6}  {'Entry':>10}  {'Exit':>10}  {'Avail%':>7}  "
            f"{'Real%':>7}  {'Cap':>5}  {'Dur':>4}  Reason")
    print(ehdr)
    print(divider)
    for sym in ASSETS:
        for t in all_results[sym]["examples"]:
            print(
                f"{sym:<6}  {str(t['entry_date']):>10}  {str(t['exit_date']):>10}  "
                f"{t['available_pct']:>7.1f}  {t['realized_pct']:>7.1f}  "
                f"{t['capture_rate']:>5.2f}  {t['duration_bars']:>4}  {t['exit_reason']}"
            )
        print()

    # 4. Cost note
    print()
    print("COST NOTE")
    print(divider)
    for sym in ASSETS:
        s = all_results[sym]["summary"]
        mean_real = s["mean_realized_pct"]
        cost_pct  = TAKER_RT_BPS / 100.0
        net       = mean_real - cost_pct
        verdict   = "survives" if net > 0 else "does NOT survive"
        print(
            f"  {sym}: mean realized {mean_real:.1f}%  |  taker RT {cost_pct:.2f}%  "
            f"=> net ~ {net:.1f}%  [{verdict} taker cost]"
        )
    print(f"  (Maker RT ~0.08%; all long trades -- entry + exit each side.)")

    # 5. Honest read
    print()
    print("WHAT THE CAPTURE LENS REVEALS")
    print(divider)
    for sym in ASSETS:
        r  = all_results[sym]
        s  = r["summary"]
        cov = r["coverage"]
        n_big = r["n_big_moves"]
        cov_str = f"{cov:.0%}" if cov == cov else "N/A"
        print(
            f"  {sym}: The setup opened {s['n_trades']} trades over the 3-year span. "
            f"Median realized {s['median_realized_pct']:.1f}% on a median available "
            f"move of {s['median_available_pct']:.1f}% -> median capture "
            f"{s['median_capture_rate']:.2f}. {n_big} big moves (>={BIG_MOVE_PCT}%) "
            f"identified; setup was long for {cov_str} of them. Average hold "
            f"{s['avg_duration_bars']:.0f} days."
        )
    print()
    print(
        "  Wealth-vs-BH asks: did we compound more than just holding? That metric is "
        "dominated by how much TIME we spend in the market and the 2023-2024 bull run. "
        "The capture lens asks the SHARPER question: when a move actually happened, "
        "what fraction of it did we bank? A setup can LOOK great on wealth-vs-BH simply "
        "because it was long during a bull run -- but if capture_rate is low (< 0.40), "
        "the exit policy is bleeding value. Conversely a high capture_rate with low "
        "coverage means the setup misses most big moves entirely. Both failure modes "
        "are invisible to the wealth-vs-BH headline."
    )
    print()
    print("  NOTE: All figures are IN-SAMPLE / ILLUSTRATIVE. No UNSEEN holdout ceremony.")
    print()
    print("=" * 72)


def main():
    all_results = {}

    for sym in ASSETS:
        print(f"[probe] loading {sym}...", flush=True)
        df = load_daily(sym)
        print(f"[probe] {sym}: {len(df)} daily bars ({df['date'][0]} to {df['date'][-1]})", flush=True)

        trades    = run_trades(df, sym)
        big_moves = find_big_moves(df, BIG_MOVE_PCT)
        cov       = coverage_of_big_moves(trades, big_moves)
        summary   = summarise_trades(trades)
        examples  = pick_examples(trades, n=5)

        all_results[sym] = {
            "trades":     trades,
            "big_moves":  big_moves,
            "coverage":   cov,
            "n_big_moves": len(big_moves),
            "summary":    summary,
            "examples":   examples,
        }

    print_report(all_results)


if __name__ == "__main__":
    main()
