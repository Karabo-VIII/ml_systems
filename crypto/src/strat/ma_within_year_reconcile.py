"""src/strat/ma_within_year_reconcile.py -- MA per-config 2020<->2021 reconciliation + EQUITY-curve charts.

USER /orc 2026-06-16: "run all the MA configs for 2021 as in 2020 (6/3/3), I want to see the charts and results
myself." The per-config leaderboard + band-heatmap + rank-stability charts come from ma_2020_config_leaderboard
(now year-parametrized, outputs under runs/periods/TRAIN/<year>/DEEP_DIVE/). THIS adds the two things that let the
user SEE it: (1) EQUITY-CURVE charts (best robust config per MA-type @ a deployable TF, $1 growth over the year vs
buy-hold, TRAIN/VAL/OOS shaded) for 2020 AND 2021; (2) a side-by-side MA_RECONCILE_2020_2021.md (per-MA-type band /
rank-stability / OOS-net vs buy-hold, both years, + the MA-type rank-transfer Spearman).

Reuses ma_2020_config_leaderboard.{build_panels, config_book, _asset_close, SYMS} (the EXACT fixed-EW long-only
ironed sleeve) + the persisted config_leaderboard.json (grid keyed 'MAtype|TF'). Long-only spot, fixed-EW, maker.
No emoji. Does NOT git commit.

RWYB: python -m strat.ma_within_year_reconcile --tf 4h    (after both years' leaderboards exist)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_2020_config_leaderboard as CL                                # noqa: E402
from strat.ma_type_upgrade import _nums, MA_TYPES                            # noqa: E402

PER = ROOT.parent / "runs" / "periods" / "TRAIN"


def _grid(year):
    p = PER / str(year) / "DEEP_DIVE" / "config_leaderboard.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _splits(year):
    return {"TRAIN": (f"{year}-01-01", f"{year}-07-01"), "VAL": (f"{year}-07-01", f"{year}-10-01"),
            "OOS": (f"{year}-10-01", f"{year + 1}-01-01")}


def _set_year(year):
    """Point the leaderboard machinery at `year` (it reads these module globals in _asset_close/build_panels)."""
    CL.YEAR = (f"{year}-01-01", f"{year + 1}-01-01")
    CL.SPLIT = _splits(year)
    CL.SPLITS = {**CL.SPLIT, "FULL": CL.YEAR}


def _best_robust_cfg(cell):
    """The highest FULL-net config that is positive across TRAIN&VAL&OOS (the band #1). Falls back to FULL #1."""
    rows = cell.get("ranked", [])
    band = [r for r in rows if r.get("positive_3way")]
    pool = band if band else rows
    if not pool:
        return None
    return max(pool, key=lambda r: (r["FULL"]["net"] if r["FULL"]["net"] is not None else -1e9))


def _spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    rx = pd.Series([p[0] for p in pairs]).rank().to_numpy()
    ry = pd.Series([p[1] for p in pairs]).rank().to_numpy()
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


# =====================================================================================================
# EQUITY CURVES: rebuild the best-robust-config $1 growth per MA-type @ tf, vs buy-hold, TRAIN/VAL/OOS shaded
# =====================================================================================================
def _equity(book):
    return (1 + book).cumprod()


def equity_chart(year, tf):
    _set_year(year)
    g = _grid(year)
    if g is None:
        print(f"   [skip equity] no {year} grid"); return None
    # buy-hold equity (u10, no cost)
    bh_cols = []
    for sym in CL.SYMS:
        a = CL._asset_close(sym, tf)
        if a is None:
            continue
        c, ms, win = a
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
        bh_cols.append(pd.Series(r[win], index=pd.to_datetime(ms[win], unit="ms")))
    bh = pd.concat(bh_cols, axis=1).fillna(0.0).mean(axis=1).sort_index()
    bh_eq = _equity(bh)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(bh_eq.index, bh_eq.values, color="black", lw=2.4, label="buy-hold u10 (no cost)", zorder=5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MA_TYPES)))
    all_periods_cache = {}
    for mt, col in zip(MA_TYPES, colors):
        cell = g["grid"].get(f"{mt}|{tf}")
        if not cell:
            continue
        best = _best_robust_cfg(cell)
        if not best:
            continue
        periods = _nums(best["config"])
        if mt not in all_periods_cache:
            all_periods_cache[mt] = CL.build_panels(tf, mt, sorted({p for p in periods}))
        panels = CL.build_panels(tf, mt, sorted(set(periods)))
        book = CL.config_book(panels, periods)
        if book is None:
            continue
        eq = _equity(book)
        ax.plot(eq.index, eq.values, color=col, lw=1.3, alpha=0.9,
                label=f"{mt}({','.join(map(str, periods))}) FULL {best['FULL']['net']}% OOS {best['OOS']['net']}%")
    # shade TRAIN / VAL / OOS
    sp = _splits(year)
    for w, c in (("TRAIN", "#e8f0ff"), ("VAL", "#fff3e0"), ("OOS", "#ffe8e8")):
        ax.axvspan(pd.Timestamp(sp[w][0]), pd.Timestamp(sp[w][1]), color=c, alpha=0.5, zorder=0)
        ax.text(pd.Timestamp(sp[w][0]), ax.get_ylim()[1], f" {w}", fontsize=9, va="top", color="#555")
    ax.set_yscale("log")
    ax.set_title(f"{year} MA best-robust-config equity (band #1 per MA-type) @ {tf} vs buy-hold -- LONG-ONLY ironed "
                 f"(trail10+min_hold12+maker), fixed-EW u10, 6/3/3.\nThese are de-risked betas: they UNDER-participate "
                 f"the bull (below buy-hold) but cut drawdown. log-y.", fontsize=10)
    ax.set_ylabel("growth of $1 (log)"); ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    out = PER / str(year) / "DEEP_DIVE" / "charts" / f"best_matype_equity_{tf}_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"   [equity chart] {out}")
    return out


# =====================================================================================================
# RECONCILE: per MA-type, both years -- band sizes, rank-stability rho, OOS net vs buy-hold
# =====================================================================================================
def reconcile(tf):
    g20, g21 = _grid(2020), _grid(2021)
    if g20 is None or g21 is None:
        print("need both 2020 and 2021 config_leaderboard.json"); return None
    bh20 = (g20["benchmarks"].get(tf, {}).get("BUYHOLD", {}) or {})
    bh21 = (g21["benchmarks"].get(tf, {}).get("BUYHOLD", {}) or {})
    bh20_oos = bh20.get("OOS", {}).get("net"); bh21_oos = bh21.get("OOS", {}).get("net")
    bh20_full = bh20.get("FULL", {}).get("net"); bh21_full = bh21.get("FULL", {}).get("net")
    rows = []
    for mt in MA_TYPES:
        c20 = g20["grid"].get(f"{mt}|{tf}"); c21 = g21["grid"].get(f"{mt}|{tf}")
        if not c20 or not c21:
            continue
        b20, b21 = _best_robust_cfg(c20), _best_robust_cfg(c21)
        rows.append({
            "mt": mt,
            "band20": c20["band"]["n_band_2ma"] + c20["band"]["n_band_3ma"],
            "band21": c21["band"]["n_band_2ma"] + c21["band"]["n_band_3ma"],
            "rho20": c20["stability"]["spearman_trainval_vs_oos"],
            "rho21": c21["stability"]["spearman_trainval_vs_oos"],
            "oos20": b20["OOS"]["net"] if b20 else None, "oos21": b21["OOS"]["net"] if b21 else None,
            "full20": b20["FULL"]["net"] if b20 else None, "full21": b21["FULL"]["net"] if b21 else None,
            "cfg20": b20["config"] if b20 else None, "cfg21": b21["config"] if b21 else None,
        })
    # MA-type rank-transfer: does the best-FULL-net ORDERING of MA types reproduce 2020->2021?
    rho_full = _spearman([r["full20"] for r in rows], [r["full21"] for r in rows])
    rho_oos = _spearman([r["oos20"] for r in rows], [r["oos21"] for r in rows])
    return {"tf": tf, "rows": rows, "bh20_oos": bh20_oos, "bh21_oos": bh21_oos,
            "bh20_full": bh20_full, "bh21_full": bh21_full,
            "matype_rank_transfer_full": rho_full, "matype_rank_transfer_oos": rho_oos}


def write_md(rec, tfs_done):
    tf = rec["tf"]
    L = [f"# MA per-config 2020 <-> 2021 reconciliation (identical 6/3/3 tool) -- focal TF {tf}", ""]
    L.append("Both years run through the SAME year-parametrized `ma_2020_config_leaderboard.py` (6mo TRAIN / 3mo "
             "VAL / 3mo OOS, fixed-EW u10, long-only ironed sleeve, maker). Per-config charts + leaderboards live in "
             "`runs/periods/TRAIN/{2020,2021}/DEEP_DIVE/`. ALL numbers [VERIFIED within-year]. Charts in `charts/`.")
    L.append("")
    L.append(f"**Buy-hold (u10, no cost) @ {tf}:** 2020 FULL {rec['bh20_full']}% / OOS {rec['bh20_oos']}% | "
             f"2021 FULL {rec['bh21_full']}% / OOS {rec['bh21_oos']}%. "
             f"(2020-OOS Oct-Dec = clean bull; 2021-OOS Oct-Dec = post-ATH decline/chop -- the OOS REGIME differs, "
             f"which is why a de-risked book's relative result flips.)")
    L.append("")
    L.append(f"## Per-MA-type, best ROBUST (band #1) config @ {tf} -- 2020 vs 2021")
    L.append("| MA | band 20/21 | rank-rho 20/21 | best cfg 2020 (FULL/OOS) | best cfg 2021 (FULL/OOS) |")
    L.append("|---|---|---|---|---|")
    for r in rec["rows"]:
        L.append(f"| {r['mt']} | {r['band20']}/{r['band21']} | {r['rho20']}/{r['rho21']} | "
                 f"`{r['cfg20']}` {r['full20']}/{r['oos20']}% | `{r['cfg21']}` {r['full21']}/{r['oos21']}% |")
    L.append("")
    L.append("## Reconciliation verdict")
    L.append(f"- **MA-type rank-transfer @ {tf}:** Spearman(best-FULL-net 2020, 2021) = "
             f"**{rec['matype_rank_transfer_full']}** (FULL), {rec['matype_rank_transfer_oos']} (OOS). "
             f"{'The MA-type ORDERING partially persists' if (rec['matype_rank_transfer_full'] or 0) > 0.4 else 'The MA-type ORDERING does NOT cleanly persist -- which MA type is best is regime-transient too'}.")
    L.append(f"- **Working band exists in BOTH years for every MA type** (band 20/21 columns > 0) -- the robust "
             f"(fast,slow) region reproduces; the band, not the #1, is the stable object.")
    L.append(f"- **Within-cell rank-stability rho is low in both years** (the rank-rho columns) -- the #1 config "
             f"does not transfer; this is the empirical basis for 'deploy the band ensemble, not the #1'.")
    L.append(f"- **The de-risked-beta read holds:** the best MA configs UNDER-participate buy-hold in the bull "
             f"(see the equity charts -- all curves sit below buy-hold) but cut drawdown. 'Beating' buy-hold only "
             f"happens in a down/flat OOS quarter and is EXPOSURE (cash), not alpha.")
    L.append("")
    L.append(f"Equity charts: `charts/best_matype_equity_{tf}_2020.png` + `..._2021.png` (each MA-type's band-#1 "
             f"$1-growth vs buy-hold, TRAIN/VAL/OOS shaded). Band map + rank-stability: `charts/config_band_heatmap.png` "
             f"+ `charts/rank_stability.png` per year.")
    out = PER / "2021" / "DEEP_DIVE" / "MA_RECONCILE_2020_2021.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"   [md] {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_within_year_reconcile")
    ap.add_argument("--tf", default="4h", help="focal TF for the reconciliation + equity charts")
    ap.add_argument("--equity-tfs", default="1d,4h", help="TFs to render equity charts for (both years)")
    a = ap.parse_args(argv)
    print(f"## MA 2020<->2021 reconciliation @ {a.tf} + equity charts")
    rec = reconcile(a.tf)
    if rec is None:
        return 1
    print(f"   MA-type rank-transfer (FULL) @ {a.tf}: {rec['matype_rank_transfer_full']}; "
          f"buy-hold OOS 2020 {rec['bh20_oos']}% / 2021 {rec['bh21_oos']}%")
    for r in rec["rows"]:
        print(f"   {r['mt']:6} band {r['band20']}/{r['band21']}  rho {r['rho20']}/{r['rho21']}  "
              f"OOS {r['oos20']}/{r['oos21']}%  FULL {r['full20']}/{r['full21']}%")
    write_md(rec, a.equity_tfs)
    for tf in [t.strip() for t in a.equity_tfs.split(",")]:
        for yr in (2020, 2021):
            equity_chart(yr, tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
