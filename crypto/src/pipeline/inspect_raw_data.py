"""
Raw Data Inspector — Comprehensive summary + diagnostic plots for all raw assets.

Outputs:
  - Console: Per-asset summary table (files, date range, gaps, timestamps, sizes)
  - Plots:   plots/raw_data_*_YYYY-MM-DD.png (date-stamped)
"""
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, date, timedelta
import sys
import argparse

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Date subfolder: same day overwrites, different days get new folder
_DATE_SUBDIR = date.today().isoformat()


def _save_plot(fig, name: str):
    """Save plot into date subfolder and close figure."""
    stem = name.removesuffix(".png")
    out_dir = PLOTS_DIR / _DATE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved: plots/{_DATE_SUBDIR}/{stem}.png")

DATA_TYPES = {
    "aggTrades": {"expected_cols": ["timestamp", "price", "qty", "is_buyer_maker"], "label": "Trades"},
    "funding":   {"expected_cols": ["timestamp", "funding_rate"], "label": "Funding"},
    "metrics":   {"expected_cols": ["timestamp", "open_interest_val"], "label": "Metrics/OI"},
}


def ts_to_date(ts_ms):
    """Convert millisecond timestamp to datetime."""
    try:
        return datetime(1970, 1, 1) + timedelta(milliseconds=int(ts_ms))
    except Exception:
        return None


def normalize_ts(ts):
    """Normalize timestamp to milliseconds."""
    ts = int(ts)
    if ts > 1_000_000_000_000_000:
        return ts // 1000  # microseconds -> ms
    if ts < 1_000_000_000_000:
        return ts * 1000   # seconds -> ms
    return ts


def inspect_folder(symbol, folder_name, info):
    """Inspect a raw data folder and return summary dict."""
    path = RAW_DIR / symbol / folder_name
    result = {
        "exists": False, "files": 0, "total_rows": 0,
        "ts_start": None, "ts_end": None, "date_start": None, "date_end": None,
        "days_span": 0, "days_with_data": 0, "gap_days": 0, "gap_pct": 0.0,
        "missing_cols": [], "ts_format": "N/A", "avg_rows_per_file": 0,
        "total_size_mb": 0.0, "sample_schema": {},
    }

    if not path.exists():
        return result

    files = sorted(list(path.glob("*.parquet")))
    if not files:
        return result

    result["exists"] = True
    result["files"] = len(files)

    # Total disk size
    total_bytes = sum(f.stat().st_size for f in files)
    result["total_size_mb"] = total_bytes / (1024 * 1024)

    # Sample first and last
    try:
        df_first = pl.read_parquet(files[0])
        df_last = pl.read_parquet(files[-1])
        result["sample_schema"] = dict(df_first.schema)
    except Exception:
        return result

    # Check expected columns
    result["missing_cols"] = [c for c in info["expected_cols"] if c not in df_first.columns]

    # Timestamp analysis
    t_col = "timestamp" if "timestamp" in df_first.columns else None
    if not t_col:
        return result

    ts_first = int(df_first[t_col][0])
    ts_last = int(df_last[t_col][-1])

    # Detect format
    if ts_first > 1_000_000_000_000_000:
        result["ts_format"] = "Microseconds (us)"
    elif ts_first > 1_000_000_000_000:
        result["ts_format"] = "Milliseconds (ms)"
    elif ts_first > 1_000_000_000:
        result["ts_format"] = "Seconds (s)"
    else:
        result["ts_format"] = "Unknown"

    ts_start_ms = normalize_ts(ts_first)
    ts_end_ms = normalize_ts(ts_last)
    result["ts_start"] = ts_start_ms
    result["ts_end"] = ts_end_ms
    result["date_start"] = ts_to_date(ts_start_ms)
    result["date_end"] = ts_to_date(ts_end_ms)

    if result["date_start"] and result["date_end"]:
        result["days_span"] = (result["date_end"] - result["date_start"]).days + 1

    # Count rows via sampling (first + last + 10 random midpoints)
    sample_files = [files[0], files[-1]]
    step = max(1, len(files) // 10)
    sample_files += files[step::step]
    sample_files = list(set(sample_files))
    avg_rows = np.mean([len(pl.read_parquet(f)) for f in sample_files[:12]])
    result["total_rows"] = int(avg_rows * len(files))
    result["avg_rows_per_file"] = int(avg_rows)

    # Gap analysis: extract dates from filenames
    file_dates = set()
    for f in files:
        parts = f.stem.split("-")
        # filename like BTCUSDT-aggTrades-2020-01-01 or BTCUSDT-funding-2020-01-01
        try:
            date_str = "-".join(parts[-3:])
            file_dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())
        except Exception:
            pass

    if file_dates:
        result["days_with_data"] = len(file_dates)
        if result["days_span"] > 0:
            result["gap_days"] = result["days_span"] - len(file_dates)
            result["gap_pct"] = (result["gap_days"] / result["days_span"]) * 100

    return result


def print_summary(all_results):
    """Print comprehensive console summary."""
    print("=" * 100)
    print("  RAW DATA INSPECTION REPORT")
    print("=" * 100)

    for symbol, dtype_results in sorted(all_results.items()):
        print(f"\n  [{symbol}]")
        print(f"  {'Type':<12} {'Files':>7} {'Est.Rows':>12} {'Size(MB)':>10} "
              f"{'Date Start':>12} {'Date End':>12} {'Days':>6} {'Gaps':>6} {'Gap%':>6} {'TS Format':<20} {'Status'}")
        print(f"  {'-'*120}")

        for dtype, r in dtype_results.items():
            label = DATA_TYPES[dtype]["label"]
            if not r["exists"]:
                print(f"  {label:<12} {'---':>7} {'---':>12} {'---':>10} "
                      f"{'---':>12} {'---':>12} {'---':>6} {'---':>6} {'---':>6} {'N/A':<20} NO DATA")
                continue

            d_start = r["date_start"].strftime("%Y-%m-%d") if r["date_start"] else "N/A"
            d_end = r["date_end"].strftime("%Y-%m-%d") if r["date_end"] else "N/A"

            # Status
            issues = []
            if r["missing_cols"]:
                issues.append(f"MISSING:{r['missing_cols']}")
            if r["gap_pct"] > 5:
                issues.append(f"HIGH GAPS")
            if "Micro" in r["ts_format"]:
                issues.append("NEEDS NORMALIZATION")
            status = ", ".join(issues) if issues else "OK"

            print(f"  {label:<12} {r['files']:>7,} {r['total_rows']:>12,} {r['total_size_mb']:>10.1f} "
                  f"{d_start:>12} {d_end:>12} {r['days_span']:>6} {r['gap_days']:>6} {r['gap_pct']:>5.1f}% "
                  f"{r['ts_format']:<20} {status}")

    # Cross-asset summary table
    print(f"\n{'=' * 100}")
    print("  CROSS-ASSET SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {'Asset':<12} {'Trade Files':>12} {'Fund Files':>12} {'OI Files':>12} "
          f"{'Total Size':>12} {'Date Range':<25} {'Coverage'}")
    print(f"  {'-'*100}")

    total_files = 0
    total_size = 0.0
    for symbol, dtype_results in sorted(all_results.items()):
        t = dtype_results.get("aggTrades", {})
        f = dtype_results.get("funding", {})
        m = dtype_results.get("metrics", {})
        sym_files = t.get("files", 0) + f.get("files", 0) + m.get("files", 0)
        sym_size = t.get("total_size_mb", 0) + f.get("total_size_mb", 0) + m.get("total_size_mb", 0)
        total_files += sym_files
        total_size += sym_size

        d_start = t.get("date_start")
        d_end = t.get("date_end")
        date_range = "N/A"
        if d_start and d_end:
            date_range = f"{d_start.strftime('%Y-%m-%d')} to {d_end.strftime('%Y-%m-%d')}"

        has_all = all(dtype_results.get(d, {}).get("exists", False) for d in DATA_TYPES)
        coverage = "COMPLETE" if has_all else "PARTIAL"

        print(f"  {symbol:<12} {t.get('files', 0):>12,} {f.get('files', 0):>12,} "
              f"{m.get('files', 0):>12,} {sym_size:>10.1f} MB {date_range:<25} {coverage}")

    print(f"  {'-'*100}")
    print(f"  {'TOTAL':<12} {total_files:>12,} {'':>12} {'':>12} {total_size:>10.1f} MB")


def generate_plots(all_results):
    """Generate diagnostic plots for raw data."""

    symbols = sorted(all_results.keys())
    if not symbols:
        return

    # ── Plot 1: File count comparison across assets ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(symbols))
    width = 0.25
    trade_counts = [all_results[s].get("aggTrades", {}).get("files", 0) for s in symbols]
    fund_counts = [all_results[s].get("funding", {}).get("files", 0) for s in symbols]
    oi_counts = [all_results[s].get("metrics", {}).get("files", 0) for s in symbols]

    ax.bar(x - width, trade_counts, width, label="aggTrades", color="#2196F3", alpha=0.85)
    ax.bar(x, fund_counts, width, label="Funding", color="#FF9800", alpha=0.85)
    ax.bar(x + width, oi_counts, width, label="Metrics/OI", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Number of Files (Days)")
    ax.set_title("Raw Data: File Counts per Asset & Type")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(trade_counts):
        if v > 0:
            ax.text(i - width, v + 20, f"{v:,}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save_plot(fig, "raw_data_file_counts.png")

    # ── Plot 2: Disk size comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    trade_sizes = [all_results[s].get("aggTrades", {}).get("total_size_mb", 0) for s in symbols]
    fund_sizes = [all_results[s].get("funding", {}).get("total_size_mb", 0) for s in symbols]
    oi_sizes = [all_results[s].get("metrics", {}).get("total_size_mb", 0) for s in symbols]

    ax.bar(x - width, trade_sizes, width, label="aggTrades", color="#2196F3", alpha=0.85)
    ax.bar(x, fund_sizes, width, label="Funding", color="#FF9800", alpha=0.85)
    ax.bar(x + width, oi_sizes, width, label="Metrics/OI", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Raw Data: Disk Size per Asset & Type")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, "raw_data_disk_sizes.png")

    # ── Plot 3: Date coverage timeline (Gantt-style) ─────────────────────────
    fig, ax = plt.subplots(figsize=(16, max(6, len(symbols) * 1.2)))
    colors = {"aggTrades": "#2196F3", "funding": "#FF9800", "metrics": "#4CAF50"}
    y_pos = 0
    y_labels = []
    y_ticks = []

    for sym in symbols:
        for dtype in ["aggTrades", "funding", "metrics"]:
            r = all_results[sym].get(dtype, {})
            d_start = r.get("date_start")
            d_end = r.get("date_end")
            if d_start and d_end:
                ax.barh(y_pos, (d_end - d_start).days, left=mdates.date2num(d_start),
                        height=0.6, color=colors[dtype], alpha=0.8,
                        label=DATA_TYPES[dtype]["label"] if sym == symbols[0] else "")
            y_labels.append(f"{sym} / {dtype}")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between assets

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.set_xlabel("Date")
    ax.set_title("Raw Data: Date Coverage Timeline")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_plot(fig, "raw_data_coverage_timeline.png")

    # ── Plot 4: Gap percentage heatmap ───────────────────────────────────────
    dtype_list = list(DATA_TYPES.keys())
    gap_matrix = np.zeros((len(symbols), len(dtype_list)))
    for i, sym in enumerate(symbols):
        for j, dtype in enumerate(dtype_list):
            gap_matrix[i, j] = all_results[sym].get(dtype, {}).get("gap_pct", 100.0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(symbols) * 0.6)))
    im = ax.imshow(gap_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=20)
    ax.set_xticks(range(len(dtype_list)))
    ax.set_xticklabels([DATA_TYPES[d]["label"] for d in dtype_list])
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols)
    ax.set_title("Raw Data: Gap Percentage (lower = better)")
    for i in range(len(symbols)):
        for j in range(len(dtype_list)):
            val = gap_matrix[i, j]
            color = "white" if val > 10 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="Gap %")
    fig.tight_layout()
    _save_plot(fig, "raw_data_gap_heatmap.png")

    # ── Plot 5: Estimated rows per asset (stacked) ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    trade_rows = [all_results[s].get("aggTrades", {}).get("total_rows", 0) / 1e6 for s in symbols]
    fund_rows = [all_results[s].get("funding", {}).get("total_rows", 0) / 1e6 for s in symbols]
    oi_rows = [all_results[s].get("metrics", {}).get("total_rows", 0) / 1e6 for s in symbols]

    ax.bar(x - width, trade_rows, width, label="aggTrades", color="#2196F3", alpha=0.85)
    ax.bar(x, fund_rows, width, label="Funding", color="#FF9800", alpha=0.85)
    ax.bar(x + width, oi_rows, width, label="Metrics/OI", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Estimated Rows (Millions)")
    ax.set_title("Raw Data: Estimated Total Rows per Asset & Type")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, "raw_data_estimated_rows.png")


def run_strict_gates(all_results):
    """
    Run critical gate checks on raw data. Returns (all_pass, failures).

    Gates:
      1. Trade files exist for each asset
      2. Gap percentage < 5% for trades
    """
    failures = []
    for sym, dtype_results in all_results.items():
        trades = dtype_results.get("aggTrades", {})
        if not trades.get("exists", False):
            failures.append(f"{sym}: No trade files found")
        elif trades.get("gap_pct", 100) > 5.0:
            failures.append(
                f"{sym}: Trade gap percentage {trades['gap_pct']:.1f}% exceeds 5% threshold"
            )
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Raw Data Inspector")
    parser.add_argument("--strict", action="store_true",
                        help="Return exit code 1 if critical checks fail (CI gate mode)")
    args = parser.parse_args()

    symbols = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
    if not symbols:
        print(f"[ERROR] No data found in {RAW_DIR}")
        if args.strict:
            sys.exit(1)
        return

    print(f"[START] Inspecting raw data for {len(symbols)} assets...")

    all_results = {}
    for sym in symbols:
        print(f"  Scanning {sym}...", end="\r")
        all_results[sym] = {}
        for dtype, info in DATA_TYPES.items():
            all_results[sym][dtype] = inspect_folder(sym, dtype, info)

    print_summary(all_results)
    generate_plots(all_results)

    # Strict gate checks
    if args.strict:
        print(f"\n{'='*70}")
        print(f"  STRICT GATE CHECKS")
        print(f"{'='*70}")
        all_pass, failures = run_strict_gates(all_results)
        if all_pass:
            print(f"  [PASS] All {len(symbols)} assets passed raw data gates")
        else:
            for fail in failures:
                print(f"  [GATE FAIL] {fail}")
            print(f"\n  [FAIL] {len(failures)} gate failures")
        sys.exit(0 if all_pass else 1)

    print(f"\n[DONE] Inspection complete. Plots saved to: {PLOTS_DIR / _DATE_SUBDIR}")


if __name__ == "__main__":
    main()
