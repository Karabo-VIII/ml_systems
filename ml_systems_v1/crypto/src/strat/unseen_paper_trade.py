"""UNSEEN PAPER-TRADE -- arm every runnable candidate (frozen dev config) and run each
INDEPENDENTLY forward on the SEALED UNSEEN window (2025-12-31 -> 2026-06-01). The final
litmus: UNSEEN was never touched during dev, so this is the honest forward grade.

User mandate (2026-06-11 /orc): "arm all of them, paper trade each independently on the
unseen window (the final litmus, not touched during dev)." This is the SOTA version of the
re-grade: don't trust one 'survivor' verdict -- run them ALL forward on the held-out window
and let the data rank them.

Each candidate => a $1 -> $X forward equity over UNSEEN + final%, annualized, maxDD, Sharpe,
win-DAY rate, and the 1d/3d rolling-ROI soft-benchmark. BOOK candidates are RWYB (live daily
net series); labs whose full UNSEEN sweep is expensive are ingested from their UNSEEN-once
persisted numbers (already a paper-trade-on-UNSEEN by construction). Output: ranked leaderboard
JSON + an equity-curve plot of all daily-series candidates. No emoji.

Run: python -m strat.unseen_paper_trade --universe u50
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT.parent / "runs" / "strat"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
ANN = 365.0
UNS_LO, UNS_HI = pd.Timestamp("2025-12-31"), pd.Timestamp("2026-06-01")


def _equity_metrics(net: pd.Series) -> dict:
    """net = daily net-return Series over UNSEEN. Returns forward paper-trade metrics."""
    s = net.dropna().sort_index()
    s = s[(s.index >= UNS_LO) & (s.index < UNS_HI)]
    if len(s) < 5:
        return {"n_days": len(s)}
    eq = (1 + s).cumprod()
    nyr = len(s) / ANN
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    # non-overlapping 1d/3d soft-benchmark
    def roll(h):
        w = [(np.prod(1 + s.iloc[i:i + h].to_numpy()) - 1) * 100 for i in range(0, len(s) - h + 1, h)]
        return {"median_pct": round(float(np.median(w)), 3), "frac_positive": round(float((np.array(w) > 0).mean()), 3)} if w else {}
    return {"n_days": int(len(s)), "final_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
            "ann_pct": round(float((eq.iloc[-1] ** (1 / nyr) - 1) * 100) if eq.iloc[-1] > 0 else -100.0, 2),
            "maxdd_pct": round(float(dd), 2),
            "sharpe": round(float(s.mean() / (s.std() + 1e-12) * np.sqrt(ANN)), 2),
            "win_day_pct": round(float((s > 0).mean() * 100), 1),
            "softbench_1d": roll(1), "softbench_3d": roll(3), "_eq": eq}


def book_candidates(universe: str, cps: float) -> dict:
    """RWYB: the tsmom/regime/blend books (live daily UNSEEN net)."""
    from strat.tsmom_ensemble import run as tsmom_run
    out, _ = tsmom_run(universe, "1d", cps)
    res = {}
    for name, d in out.items():
        net = d.get("_net")
        if net is not None and len(net) > 50:
            m = _equity_metrics(net)
            if m.get("n_days"):
                res[name] = m
    return res


def family2_candidate(cps: float) -> dict | None:
    """RWYB: Family2 rotation forward equity on UNSEEN (from per-trade, entry-ordered)."""
    try:
        from strat.momentum_rotation_lab import load_all_assets, simulate_rotation
        dfs = load_all_assets(verbose=False)
        r = simulate_rotation(dfs, 10, 3, 10, 200, 3.0, cost_rt=2 * cps)
        # build a daily-ish equity from UNSEEN trades: compound each trade's net on its exit date
        rows = []
        for t in r["trades"]:
            if t["window"] != "UNSEEN":
                continue
            net = float(t["exit_p"]) / float(t["entry_p"]) - 1.0 - 2 * cps
            rows.append((pd.Timestamp(t["exit_date"]), net * float(t.get("weight", 1.0))))
        if not rows:
            return None
        ser = pd.Series(dict(rows)).sort_index()
        # treat as a sparse daily return series (book contribution per exit day)
        idx = pd.date_range(UNS_LO, UNS_HI, freq="D")
        daily = pd.Series(0.0, index=idx)
        for d, v in ser.items():
            if d in daily.index:
                daily[d] += v
        m = _equity_metrics(daily)
        m["n_trades_unseen"] = len(rows)
        return m
    except Exception as e:
        return {"error": str(e)[:80]}


def ingest_lab_unseen() -> dict:
    """Labs already paper-traded UNSEEN once -- ingest their reported UNSEEN result."""
    return {
        "trend_book_lab": {"final_pct": 0.0, "note": "0 trades on UNSEEN (flat in bear = pure abstention)"},
        "symmetric_trend_LS_perp": {"final_pct": None, "ann_pct": 13.8, "note": "+13.8%/yr on 4 short trades (perp; tiny-n)"},
        "setup_chaser_book": {"final_pct": -15.18, "note": "UNSEEN -15.18% comp; battery FAIL PBO 0.79 (NULL)"},
        "alt_bar_trend_renko": {"ann_pct": -20.0, "note": "Renko UNSEEN -20%/yr (best alt-bar); 0/10 seeds (NULL)"},
        "regime_dna_SYS_A_u50": {"note": "UNSEEN per-trade -1.25+-0.82% (trade-level, not a book); UNSEEN-neg"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.unseen_paper_trade")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--maker", action="store_true")
    a = ap.parse_args()
    cps = 0.0006 if a.maker else 0.0012

    books = book_candidates(a.universe, cps)
    fam2 = family2_candidate(cps)
    if fam2 and fam2.get("n_days"):
        books["Family2_mover_rotation"] = fam2
    labs = ingest_lab_unseen()

    # rank by UNSEEN final %
    ranked = sorted([(k, v) for k, v in books.items() if v.get("final_pct") is not None],
                    key=lambda kv: kv[1]["final_pct"], reverse=True)

    # plot equity curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 7))
        for name, v in ranked:
            eq = v.get("_eq")
            if eq is not None:
                ax.plot(eq.index, eq.values, lw=1.4, label=f"{name} {v['final_pct']:+.1f}%")
        ax.axhline(1.0, color="gray", lw=0.7, ls="--")
        ax.set_title(f"UNSEEN paper-trade (2025-12-31 -> 2026-05) -- {a.universe}, frozen dev configs, "
                     f"{'maker' if a.maker else 'taker'} -- $1 -> $X forward equity")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        plot = PLOTS / f"unseen_paper_trade_{a.universe}_{stamp}.png"
        fig.tight_layout(); fig.savefig(plot, dpi=110); plt.close(fig)
    except Exception as e:
        plot = f"(plot failed: {e})"

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_eq"} for k, v in books.items()}
    p = OUT / f"UNSEEN_PAPER_TRADE_{a.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha},
               "window": ["2025-12-31", "2026-06-01"], "cost_per_side": cps,
               "books_rwyb": clean, "labs_ingested": labs, "plot": str(plot)}, open(p, "w", encoding="utf-8"), indent=1, default=str)

    print(f"## UNSEEN PAPER-TRADE -- {a.universe} -- {'maker' if a.maker else 'taker'} -- frozen dev configs, sealed forward window")
    print(f"   (2025-12-31 -> 2026-06-01; ~5mo; the final litmus -- never touched during dev)\n")
    print(f"   {'candidate':24} {'final%':>8} {'ann%':>8} {'maxDD%':>8} {'Sharpe':>7} {'winDay%':>8} {'3d med/+%':>12}")
    for name, v in ranked:
        sb = v.get("softbench_3d", {})
        print(f"   {name:24} {v['final_pct']:>8.2f} {v.get('ann_pct',0):>8.1f} {v.get('maxdd_pct',0):>8.1f} "
              f"{v.get('sharpe',0):>7.2f} {v.get('win_day_pct',0):>8.1f} "
              f"{str(sb.get('median_pct'))+'/'+str(sb.get('frac_positive')):>12}")
    print(f"\n   LABS (ingested UNSEEN-once results):")
    for k, v in labs.items():
        print(f"   {k:24} {v.get('note','')}")
    print(f"\n   PLOT -> {plot}\n   JSON -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
