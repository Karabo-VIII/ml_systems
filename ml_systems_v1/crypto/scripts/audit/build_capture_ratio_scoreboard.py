"""build_capture_ratio_scoreboard.py — Phase 1 anchor of the 1-5%/d No-Failure
Baseline Campaign (charter: docs/NO_FAILURE_1_5_PCT_CAMPAIGN_CHARTER_2026_05_18.md).

Builds a per-day + per-quarter scoreboard joining:
  - Oracle daily availability (ideal_k5_1d_ret from outcome_catalog.parquet)
  - Strategy realized daily PnL (parsed from v3 paper_trade_replay stdout logs)

Outputs:
  runs/audit/capture_ratio_scoreboard.parquet       (per-day, per-strategy)
  runs/audit/capture_ratio_quarter_rollup.parquet   (per-quarter)
  runs/audit/CAPTURE_RATIO_SCOREBOARD_V1_2026_05_18.md  (verdict)

Acceptance gates (per charter):
  - Reproduces ~1.3% capture ratio claim (Phase 4 Forensics) within +/- 0.5pp
  - Reproduces STRICT_LO_SETUP60 +20.95% / Sh 0.63 / DD -10.3% / 6/8 pos Q rollup

Strategy baseline this build: REGIME_ROUTER_STRICT_LO_SETUP60.
Source logs: runs/rerun_walkforward_topN/REGIME_ROUTER_STRICT_LO_SETUP60-<Q>_*.stdout.log
"""
from __future__ import annotations
import re
import math
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- stdout log parser ----------------------------------------------------

DATE_RE = re.compile(r"\[v3\] === (\d{4}-\d{2}-\d{2}) ===")
PNL_RE = re.compile(
    r"day_pnl=([\+\-][\d\.]+)%.*nav=([\d\.]+).*new_entries=(\d+).*open=(\d+).*closed_today=(\d+)"
)
# alt order — some logs put new_entries/open/closed_today in different order
ENTRIES_RE = re.compile(r"new_entries=(\d+)")
OPEN_RE = re.compile(r"open=(\d+)")
CLOSED_RE = re.compile(r"closed_today=(\d+)")
PNL_ONLY_RE = re.compile(r"day_pnl=([\+\-][\d\.]+)%.*nav=([\d\.]+)")

QUARTERS = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]
LOG_DIR = ROOT / "runs" / "rerun_walkforward_topN"
LOG_PATTERN = "REGIME_ROUTER_STRICT_LO_SETUP60-{q}_20260517_204256.stdout.log"

STRATEGY_ID = "REGIME_ROUTER_STRICT_LO_SETUP60"


def parse_one_log(path: Path, strategy_id: str, quarter_label: str) -> list[dict]:
    rows: list[dict] = []
    current_date: str | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DATE_RE.search(line)
            if m:
                current_date = m.group(1)
                continue
            m2 = PNL_ONLY_RE.search(line)
            if m2 and current_date is not None:
                day_pnl_pct = float(m2.group(1))
                nav = float(m2.group(2))
                ent = ENTRIES_RE.search(line)
                opn = OPEN_RE.search(line)
                cls = CLOSED_RE.search(line)
                rows.append({
                    "date": current_date,
                    "strategy_id": strategy_id,
                    "quarter": quarter_label,
                    "day_pnl_pct": day_pnl_pct,
                    "nav": nav,
                    "n_entries": int(ent.group(1)) if ent else 0,
                    "n_open": int(opn.group(1)) if opn else 0,
                    "n_closed": int(cls.group(1)) if cls else 0,
                })
                current_date = None
    return rows


def parse_all_quarters() -> pl.DataFrame:
    all_rows: list[dict] = []
    for q in QUARTERS:
        log = LOG_DIR / LOG_PATTERN.format(q=q)
        if not log.exists():
            print(f"  WARN: missing log for {q}: {log}")
            continue
        rows = parse_one_log(log, STRATEGY_ID, q)
        print(f"  parsed {q}: {len(rows)} days, last_nav={rows[-1]['nav']:.2f}" if rows else f"  parsed {q}: EMPTY")
        all_rows.extend(rows)
    df = pl.DataFrame(all_rows)
    df = df.with_columns(pl.col("date").str.to_date())
    return df


# ----- Oracle availability join --------------------------------------------

def load_oracle() -> pl.DataFrame:
    o = pl.read_parquet(str(ROOT / "data" / "processed" / "outcome_catalog.parquet"))
    return o.select([
        "date",
        pl.col("ideal_k5_1d_ret").alias("oracle_k5_1d_ret"),
        pl.col("gross_k5_1d_ret").alias("oracle_gross_k5_1d_ret"),
        pl.col("n_assets_available_1d"),
        pl.col("day_class_1d"),
        pl.col("day_class_q"),
    ])


def build_scoreboard(strategy_df: pl.DataFrame, oracle_df: pl.DataFrame) -> pl.DataFrame:
    s = strategy_df.join(oracle_df, on="date", how="left")
    # capture ratio: realized / oracle (oracle in fraction; day_pnl in pct)
    s = s.with_columns(
        (pl.col("day_pnl_pct") / 100.0).alias("realized_frac"),
    )
    s = s.with_columns([
        pl.when(pl.col("oracle_k5_1d_ret") > 0)
          .then(pl.col("realized_frac") / pl.col("oracle_k5_1d_ret"))
          .otherwise(None)
          .alias("capture_ratio_signed"),
    ])
    # also: capped capture ratio at [0, 1] for stat aggregation
    s = s.with_columns(
        pl.col("capture_ratio_signed").clip(0.0, 1.0).alias("capture_ratio_clipped")
    )
    return s


def quarter_rollup(scoreboard: pl.DataFrame) -> pl.DataFrame:
    g = scoreboard.group_by("quarter").agg([
        pl.col("date").min().alias("q_start"),
        pl.col("date").max().alias("q_end"),
        pl.len().alias("n_days"),
        pl.col("day_pnl_pct").mean().alias("mean_day_pnl_pct"),
        pl.col("day_pnl_pct").std().alias("std_day_pnl_pct"),
        ((pl.col("day_pnl_pct") > 0).sum()).alias("n_pos_days"),
        ((pl.col("day_pnl_pct") < 0).sum()).alias("n_neg_days"),
        pl.col("n_entries").sum().alias("total_entries"),
        pl.col("nav").last().alias("end_nav"),
        pl.col("nav").min().alias("min_nav_in_q"),
        pl.col("oracle_k5_1d_ret").mean().alias("mean_oracle_avail_d"),
        pl.col("oracle_k5_1d_ret").median().alias("median_oracle_avail_d"),
        pl.col("capture_ratio_signed").mean().alias("mean_capture_signed"),
        pl.col("capture_ratio_clipped").mean().alias("mean_capture_clipped"),
        pl.col("realized_frac").sum().alias("total_realized_frac"),
        pl.col("oracle_k5_1d_ret").sum().alias("total_oracle_avail_frac"),
    ])
    # Quarter return from final NAV (each quarter restarts from $10000 in this run)
    g = g.with_columns([
        ((pl.col("end_nav") / 10000.0 - 1.0) * 100.0).alias("q_return_pct"),
        # Quarter-internal DD: (min NAV - 10000) / 10000
        ((pl.col("min_nav_in_q") / 10000.0 - 1.0) * 100.0).alias("q_dd_pct"),
        # Quarter Sharpe (daily) annualized over the quarter days (not calendar)
        (pl.col("mean_day_pnl_pct") / pl.col("std_day_pnl_pct") * math.sqrt(365)).alias("q_sharpe_d_ann"),
        # Capture-ratio aggregate: total realized / total oracle
        (pl.col("total_realized_frac") / pl.col("total_oracle_avail_frac")).alias("capture_ratio_q_agg"),
    ])
    g = g.sort("quarter")
    return g


# ----- main -----------------------------------------------------------------

def main():
    print("[scoreboard] parsing v3 stdout logs ...")
    strat_df = parse_all_quarters()
    print(f"[scoreboard] strategy rows: {len(strat_df)}")
    print(f"[scoreboard] date range: {strat_df['date'].min()} -> {strat_df['date'].max()}")

    print("[scoreboard] loading oracle outcome_catalog ...")
    oracle_df = load_oracle()
    print(f"[scoreboard] oracle rows: {len(oracle_df)}")

    print("[scoreboard] joining + computing capture ratio ...")
    sb = build_scoreboard(strat_df, oracle_df)
    sb_path = OUT_DIR / "capture_ratio_scoreboard.parquet"
    sb.write_parquet(str(sb_path))
    print(f"[scoreboard] wrote {sb_path}  rows={len(sb)}")

    qr = quarter_rollup(sb)
    qr_path = OUT_DIR / "capture_ratio_quarter_rollup.parquet"
    qr.write_parquet(str(qr_path))
    print(f"[scoreboard] wrote {qr_path}  rows={len(qr)}")

    # --- 8Q rollup summary ---
    print()
    print("=" * 80)
    print("8Q WALK-FORWARD ROLLUP (STRICT_LO_SETUP60)")
    print("=" * 80)
    # COMP from chained quarter returns
    q_rets = qr["q_return_pct"].to_list()
    comp = 1.0
    for r in q_rets:
        comp *= (1.0 + r / 100.0)
    comp_pct = (comp - 1.0) * 100.0

    # Walk-forward DD: chain quarters and compute running max NAV vs current NAV
    # Approximation: per-quarter DD already computed; we report worst-quarter DD
    worst_dd = qr["q_dd_pct"].min()
    n_pos_q = sum(1 for r in q_rets if r > 0)
    n_neg_q = sum(1 for r in q_rets if r < 0)
    daily_pnls = sb["day_pnl_pct"].to_list()
    sh_8q = (np.mean(daily_pnls) / np.std(daily_pnls)) * math.sqrt(365)
    mean_d = float(np.mean(daily_pnls))

    print(f"  COMP 8Q:           {comp_pct:+.2f}%")
    print(f"  Sharpe (daily):    {sh_8q:+.2f}")
    print(f"  Worst Q DD:        {worst_dd:+.2f}%")
    print(f"  Positive quarters: {n_pos_q}/8")
    print(f"  Mean day PnL:      {mean_d:+.4f}%")
    print()

    # --- Capture ratio summary ---
    print("=" * 80)
    print("CAPTURE RATIO SUMMARY")
    print("=" * 80)
    # Aggregate capture across all 8Q
    total_realized = sb["realized_frac"].sum()
    total_oracle = sb["oracle_k5_1d_ret"].sum()
    cap_agg = total_realized / total_oracle if total_oracle > 0 else float("nan")
    cap_per_day_mean = sb["capture_ratio_clipped"].mean()
    cap_per_day_median = sb["capture_ratio_clipped"].median()

    print(f"  Total realized (sum of fractions):   {total_realized:+.4f}")
    print(f"  Total oracle avail (sum of fractions): {total_oracle:+.4f}")
    print(f"  Capture ratio aggregate (realized/oracle): {cap_agg*100:+.2f}%")
    print(f"  Mean per-day capture (clipped [0,1]):     {cap_per_day_mean*100:+.2f}%")
    print(f"  Median per-day capture (clipped [0,1]):   {cap_per_day_median*100:+.2f}%")
    print()

    # --- Per-quarter table ---
    print("=" * 80)
    print("PER-QUARTER TABLE")
    print("=" * 80)
    print(f"  {'Q':5s} {'days':>5s} {'ret%':>8s} {'DD%':>8s} {'pos/tot':>9s} {'sh_ann':>8s} {'cap%':>8s} {'orac%':>8s} {'real%':>8s}")
    for row in qr.iter_rows(named=True):
        print(
            f"  {row['quarter']:5s} {row['n_days']:>5d} "
            f"{row['q_return_pct']:>+8.2f} "
            f"{row['q_dd_pct']:>+8.2f} "
            f"{row['n_pos_days']:>4d}/{row['n_days']:<4d} "
            f"{row['q_sharpe_d_ann']:>+8.2f} "
            f"{row['capture_ratio_q_agg']*100:>+8.2f} "
            f"{row['total_oracle_avail_frac']*100:>+8.2f} "
            f"{row['total_realized_frac']*100:>+8.2f}"
        )
    print()

    # --- ACCEPTANCE GATES ---
    print("=" * 80)
    print("ACCEPTANCE GATES (per charter)")
    print("=" * 80)
    # Gate 1: reproduce 1.3% capture ratio claim within +/- 0.5pp
    gate1_pass = abs(cap_agg * 100 - 1.3) <= 0.5
    # Gate 2: reproduce STRICT_LO_SETUP60 +20.95% / Sh 0.63 / DD -10.3% / 6/8 within tolerance
    gate2_comp = abs(comp_pct - 20.95) <= 3.0  # +/- 3pp tolerance (different cost model possible)
    gate2_sh = abs(sh_8q - 0.63) <= 0.30
    gate2_pos = n_pos_q == 6
    gate2_dd = worst_dd <= -8.0 and worst_dd >= -15.0  # within +/- 5pp of -10.3%

    print(f"  Gate 1 (capture ratio ~1.3% +/- 0.5pp):   measured={cap_agg*100:+.2f}%  pass={gate1_pass}")
    print(f"  Gate 2a (COMP ~+20.95% +/- 3pp):          measured={comp_pct:+.2f}%  pass={gate2_comp}")
    print(f"  Gate 2b (Sharpe ~0.63 +/- 0.30):          measured={sh_8q:+.2f}     pass={gate2_sh}")
    print(f"  Gate 2c (6/8 positive quarters):          measured={n_pos_q}/8     pass={gate2_pos}")
    print(f"  Gate 2d (worst DD ~-10% +/- 5pp):         measured={worst_dd:+.2f}%  pass={gate2_dd}")

    all_pass = gate1_pass and gate2_comp and gate2_sh and gate2_pos and gate2_dd
    print()
    print(f"  ALL GATES PASS: {all_pass}")

    return {
        "comp_pct": comp_pct, "sh_8q": sh_8q, "worst_dd": worst_dd, "n_pos_q": n_pos_q,
        "cap_agg_pct": cap_agg * 100, "cap_per_day_mean_pct": cap_per_day_mean * 100,
        "all_pass": all_pass, "qr": qr, "sb": sb,
    }


if __name__ == "__main__":
    main()
