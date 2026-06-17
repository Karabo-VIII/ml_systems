"""CANONICAL HONEST SCORECARD -- the one uniform evaluator every strategy flows through.

User mandate (2026-06-11 /orc): re-grade EVERY strategy ever generated under the CORRECT
methodology, honest across the board; 1d/3d ROI are a REPORTED soft-benchmark, never an
eliminator. This is the SOTA elevation of "regrade them": instead of ad-hoc per-candidate
numbers, every candidate produces the SAME honest metric block so the leaderboard is uniform
and deflation-aware.

Two candidate flavors, one scorecard:
  BOOK   (daily portfolio net-return series, split-indexed): compound/ann/maxDD/Sharpe per
         split + rolling 1d/3d/7d ROI soft-benchmark (median + %positive) + block-bootstrap p05.
  TRADES (per-trade net list, each tagged split [+ optional sym, ts]): per-trade mean +- SE,
         win, PF, fraction-matched jackknife (drop-top-5%), n_eff (Herfindahl), breadth
         (assets-positive), block-bootstrap p05 on the trade stream.
A candidate may supply both; the scorecard emits whichever metrics apply.

DISCIPLINE baked in (the failure modes we fixed this session):
  - Splits SEL(pre-OOS, selection) / OOS (validate) / UNSEEN (test-once). Numbers are
    reported per split; SHIP verdict keys on UNSEEN + full-cycle block-bootstrap p05, NEVER on
    the selection window.
  - Concentration is fraction-matched (drop-top-5%) + n_eff, NOT a fixed-count jackknife
    (the regime_dna artifact).
  - Per-trade comparison uses mean +- SE (NOT sum / breadth, which reward trade-count).
  - 1d/3d/7d ROI is a SOFT BENCHMARK column (reported, flagged), never a gate.
  - PBO (pbo_cscv) is reported when a config grid is supplied (selection deflation).
  - Every number carries the data window; claim-tag is the caller's (VERIFIED iff RWYB here).
No emoji (cp1252).

Use:
  from strat.scorecard import score_book, score_trades, SPLITS
  card = score_book("wave1_GOLD100", daily_net_series)         # pandas Series, DatetimeIndex
  card = score_trades("regime_dna_SYS_A", trades)              # list[{net, split, sym?, ts?}]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strat.battery import (compound, herfindahl_neff, block_bootstrap_p05_p95,  # noqa: E402
                           win_rate, profit_factor)

ANN = 365.0
# canonical splits (match WIN in entry_signal_lab): SEL = everything < val_end
SPLITS = {"SEL": ("2018-01-01", "2025-03-15"),   # pre-OOS selection window
          "OOS": ("2025-03-15", "2025-12-31"),   # validation
          "UNSEEN": ("2025-12-31", "2026-06-01")}  # test-once


def _drop_top_pct(a: np.ndarray, pct: float = 0.05) -> float | None:
    a = np.asarray(a, float)
    k = max(1, int(round(len(a) * pct)))
    return float(np.sort(a)[:-k].mean()) if len(a) > k else None


def _maxdd_pct(net: pd.Series) -> float:
    eq = (1 + net).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min() * 100) if len(eq) else 0.0


def _sl(series: pd.Series, lo: str, hi: str) -> pd.Series:
    return series[(series.index >= pd.Timestamp(lo)) & (series.index < pd.Timestamp(hi))]


def rolling_roi_softbench(net: pd.Series) -> dict:
    """1d/3d/7d compounded rolling-window ROI distribution -- REPORTED soft-benchmark only.
    median %/window + fraction of windows positive (non-overlapping windows)."""
    out = {}
    for h, label in [(1, "1d"), (3, "3d"), (7, "7d")]:
        if len(net) < h:
            out[label] = {"median_pct": None, "frac_positive": None, "n": 0}
            continue
        # non-overlapping h-day compounded windows
        w = [(np.prod(1 + net.iloc[i:i + h].to_numpy()) - 1) * 100
             for i in range(0, len(net) - h + 1, h)]
        w = np.array(w)
        out[label] = {"median_pct": round(float(np.median(w)), 3),
                      "frac_positive": round(float((w > 0).mean()), 3), "n": len(w)}
    return out


def score_book(name: str, daily_net: pd.Series, grid_returns=None) -> dict:
    """Book-level scorecard from a daily net-return Series (DatetimeIndex).
    grid_returns (optional): np.ndarray [T x n_configs] for PBO deflation."""
    daily_net = daily_net.dropna().sort_index()
    card = {"name": name, "kind": "BOOK", "n_days": int(len(daily_net))}
    per = {}
    for sp, (lo, hi) in SPLITS.items():
        s = _sl(daily_net, lo, hi)
        if len(s) < 5:
            per[sp] = {"n": len(s)}
            continue
        eq = (1 + s).cumprod()
        nyr = len(s) / ANN
        per[sp] = {
            "n": int(len(s)), "compound_pct": round(compound(s.to_numpy()), 2),
            "ann_pct": round(float((eq.iloc[-1] ** (1 / nyr) - 1) * 100) if eq.iloc[-1] > 0 else -100.0, 2),
            "maxdd_pct": round(_maxdd_pct(s), 2),
            "sharpe": round(float(s.mean() / (s.std() + 1e-12) * np.sqrt(ANN)), 2),
            "softbench_roi": rolling_roi_softbench(s),
        }
    card["per_split"] = per
    card["full_block_bootstrap"] = block_bootstrap_p05_p95(daily_net.to_numpy())
    held = _sl(daily_net, SPLITS["OOS"][0], SPLITS["UNSEEN"][1])
    card["heldout_block_bootstrap"] = block_bootstrap_p05_p95(held.to_numpy()) if len(held) > 10 else {}
    if grid_returns is not None:
        try:
            from strat.pbo_cscv import pbo_cscv
            card["pbo"] = pbo_cscv(np.asarray(grid_returns, float))
        except Exception as e:
            card["pbo"] = {"error": str(e)[:80]}
    # honest ship read (UNSEEN + full p05; selection window EXCLUDED from the gate)
    u = per.get("UNSEEN", {})
    fp = card["full_block_bootstrap"].get("p05")
    hp = card["heldout_block_bootstrap"].get("p05") if card["heldout_block_bootstrap"] else None
    card["ship_read"] = {
        "unseen_compound_pos": bool(u.get("compound_pct", -1) > 0),
        "full_p05_pos": bool(fp is not None and fp > 0),
        "heldout_p05_pos": bool(hp is not None and hp > 0),
        "ship": bool(u.get("compound_pct", -1) > 0 and fp is not None and fp > 0
                     and hp is not None and hp > 0),
    }
    return card


def score_trades(name: str, trades: list[dict], grid_returns=None) -> dict:
    """Trade-level scorecard. trades: list of {net, split, sym?, ts?}."""
    card = {"name": name, "kind": "TRADES", "n_trades": len(trades)}
    per = {}
    for sp in SPLITS:
        nets = np.array([t["net"] for t in trades if t.get("split") == sp], dtype=float)
        if len(nets) < 3:
            per[sp] = {"n": int(len(nets))}
            continue
        per_asset = {}
        for t in trades:
            if t.get("split") == sp and t.get("sym"):
                per_asset.setdefault(t["sym"], []).append(t["net"])
        breadth = {s: float(np.mean(v)) for s, v in per_asset.items() if len(v) >= 3}
        per[sp] = {
            "n": int(len(nets)),
            "mean_pct": round(float(nets.mean() * 100), 3),
            "se_pct": round(float(nets.std() / np.sqrt(len(nets)) * 100), 3),
            "median_pct": round(float(np.median(nets) * 100), 3),
            "win": round(win_rate(nets), 3), "pf": round(profit_factor(nets), 2),
            "jk_drop_top5pct_mean_pct": (round(_drop_top_pct(nets) * 100, 3)
                                         if _drop_top_pct(nets) is not None else None),
            "n_eff": round(herfindahl_neff(nets), 1),
            "breadth_pos": sum(1 for v in breadth.values() if v > 0),
            "breadth_tot": len(breadth),
            "bootstrap_p05_pct": block_bootstrap_p05_p95(nets).get("p05"),
        }
    card["per_split"] = per
    if grid_returns is not None:
        try:
            from strat.pbo_cscv import pbo_cscv
            card["pbo"] = pbo_cscv(np.asarray(grid_returns, float))
        except Exception as e:
            card["pbo"] = {"error": str(e)[:80]}
    u = per.get("UNSEEN", {})
    card["ship_read"] = {
        "unseen_mean_pos": bool(u.get("mean_pct", -1) > 0),
        "unseen_mean_gt_2se": bool(u.get("mean_pct") is not None and u.get("se_pct")
                                   and u["mean_pct"] > 2 * u["se_pct"]),
        "unseen_jk_pos": bool((u.get("jk_drop_top5pct_mean_pct") or -1) > 0),
        "unseen_breadth_majority": bool(u.get("breadth_tot", 0) and
                                        u.get("breadth_pos", 0) / u["breadth_tot"] >= 0.5),
        "ship": bool(u.get("mean_pct", -1) > 0 and (u.get("jk_drop_top5pct_mean_pct") or -1) > 0
                     and u.get("breadth_tot", 0) and u.get("breadth_pos", 0) / u["breadth_tot"] >= 0.5),
    }
    return card


def _selftest() -> bool:
    idx = pd.date_range("2021-01-01", "2026-05-01", freq="D")
    rng = np.random.default_rng(7)
    # positive-drift book (drift well above sampling noise: SE of mean ~ 0.02/sqrt(N) << 0.004)
    pos = pd.Series(rng.normal(0.004, 0.02, len(idx)), index=idx)
    cp = score_book("pos_drift", pos)
    # zero-edge book
    zero = pd.Series(rng.normal(0.0, 0.02, len(idx)), index=idx)
    cz = score_book("zero_edge", zero)
    ok = cp["per_split"]["SEL"]["ann_pct"] > cz["per_split"]["SEL"]["ann_pct"]
    # trade-level: planted edge
    tr = [{"net": float(rng.normal(0.03, 0.05)), "split": "UNSEEN", "sym": f"A{i%6}"} for i in range(120)]
    ct = score_trades("planted", tr)
    ok = ok and ct["per_split"]["UNSEEN"]["mean_pct"] > 0
    print(f"[scorecard selftest] pos ann {cp['per_split']['SEL']['ann_pct']} > zero {cz['per_split']['SEL']['ann_pct']}; "
          f"planted UNSEEN mean {ct['per_split']['UNSEEN']['mean_pct']}% -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys as _s
    _s.exit(0 if _selftest() else 1)
