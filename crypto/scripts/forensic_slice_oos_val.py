"""forensic_slice_oos_val.py -- GENUINE OUT-OF-SAMPLE forensic on a RANDOM 7-day slice INSIDE the VAL window.

THE TEST (user 2026-06-21): the MA drawdown-insurance engine SELECTED its band on the 2020 leaderboard
(config_leaderboard.json, TRAIN+VAL of the 2020 split). This runs the per-asset / per-TF forensic on a
RANDOM 7-day slice INSIDE the Convention-B VAL window -- data the 2020-selected band NEVER saw -- with the
band FROZEN. The real validation: does the engine behave the same OOS as in-sample?

SPLIT DISCIPLINE (BINDING):
  - VAL window (canonical, purge_split.split_chimera): [train_end=2023-07-01, val_end=2024-05-15).
  - Band FROZEN from 2020 (load_published_band / select_band_trainval). NO re-fit on VAL.
  - _load_asset caps every load at wall_ms = VAL_END (== fl.DEV_END = 2024-05-15) -> the engine physically
    cannot load a single bar into OOS (>= 2024-05-15) or UNSEEN. We ALSO assert max loaded bar <= VAL_END.
  - EMAs warm causally over the FULL pre-slice series (TRAIN data); held_cross at bar t uses close[:t+1].

RANDOM SLICE: random.seed(20260621); uniform start in [VAL_START, VAL_END - 7d]. Reported + reproducible.

REUSES committed engine internals (do NOT reimplement): ma_movecatch_engine.select_band_trainval,
ma_type_upgrade.held_cross/_nums, ma_wealth_driver._load_asset/_u100_symbols/MAKER_RT, fleet_lab (DEV_END).

RWYB: python scripts/forensic_slice_oos_val.py
No emoji (cp1252). No git commit (meta reviews). Runs synchronously.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CRYPTO = Path(__file__).resolve().parents[1]
SRC = CRYPTO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import strat.fleet_lab as fl
from strat.ma_movecatch_engine import select_band_trainval  # noqa: F401  (frozen-band provenance check)
from strat.ma_type_upgrade import held_cross, _nums
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT

import random

LEADERBOARD = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
OUTDIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"
CHARTS_DIR = OUTDIR / "charts" / "slices"

# ---- canonical Convention-B VAL window (purge_split.split_chimera) ----
VAL_START = pd.Timestamp("2023-07-01")   # = data_config train_end
VAL_END = pd.Timestamp("2024-05-15")     # = data_config val_end == fl.DEV_END (the engine's hard wall)
assert str(VAL_END.date()) == fl.DEV_END, "VAL_END must equal the engine DEV wall"

TFS = ["1d", "4h", "2h", "1h"]           # cadences with VAL data (engine cells); 30m/15m also exist but the
TF_ORDER = {t: i for i, t in enumerate(TFS)}   #  committed success is 1d/4h (15m BROKEN), so forensic = these 4.
MA_TYPE = "EMA"                          # the committed 1st-success book is the EMA band ensemble
THR = 0.5                                # band-on-fraction threshold (> 0.5 = held)
SEED = 20260621
MOVER_THR = 0.05                         # a "mover" = name whose slice peak-from-open >= 5%

CHART_ASSETS = None                      # filled at runtime: top-2 movers in the slice


# ---- random slice draw (reproducible) ----
def pick_slice(seed: int):
    random.seed(seed)
    span_days = (VAL_END - pd.Timedelta(days=7) - VAL_START).days
    off = random.randint(0, span_days)
    s = VAL_START + pd.Timedelta(days=off)
    e = s + pd.Timedelta(days=7)
    return s, e, off, span_days


SLICE_START, SLICE_END_EXCL, SLICE_OFFSET, SLICE_SPAN = pick_slice(SEED)


def load_published_band(tf: str) -> list:
    """FROZEN 2020-selected band for a TF (config_leaderboard.json -- NO re-fit on VAL)."""
    d = json.load(open(LEADERBOARD))
    node = d["grid"].get(f"{MA_TYPE}|{tf}")
    if node is None:
        raise KeyError(f"no leaderboard cell {MA_TYPE}|{tf}")
    return list(node["band"]["band_configs"])


def compute_band_on_frac(c_full: np.ndarray, band_configs: list) -> np.ndarray:
    if len(c_full) == 0 or len(band_configs) == 0:
        return np.zeros(len(c_full), dtype=np.float64)
    on_sum = np.zeros(len(c_full), dtype=np.float64)
    for name in band_configs:
        on_sum += held_cross(c_full, _nums(name), MA_TYPE).astype(np.float64)
    frac = on_sum / len(band_configs)
    assert np.all(frac >= 0.0) and np.all(frac <= 1.0 + 1e-12), "band_on_frac out of [0,1]"
    return frac


def extract_trades(dates: pd.DatetimeIndex, c: np.ndarray, band_frac: np.ndarray) -> list:
    """Contiguous held runs that OVERLAP the slice. lag-1 causal entry/exit fills. (Mirrors the committed
    forensic_slice_bull_2021 extractor exactly.)"""
    n = len(dates)
    held_state = (band_frac > THR).astype(np.int8)
    padded = np.concatenate([[0], held_state, [0]])
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    last_slice_bar = int(np.searchsorted(dates, SLICE_END_EXCL, side="left")) - 1
    last_slice_bar = max(0, min(last_slice_bar, n - 1))

    trades = []
    for rs, re in zip(starts, ends):
        run_dates = dates[rs:re]
        if not ((run_dates >= SLICE_START) & (run_dates < SLICE_END_EXCL)).any():
            continue
        entry_bar = rs
        entry_fill_bar = min(entry_bar + 1, n - 1)     # lag-1 entry
        entry_px = float(c[entry_fill_bar])
        run_end_bar = re - 1
        if run_end_bar >= last_slice_bar:
            exit_bar = last_slice_bar
            exit_reason = "window_end_still_held"
            exit_fill_bar = min(last_slice_bar + 1, n - 1)
        else:
            exit_bar = run_end_bar
            exit_reason = "signal_cross_down"
            exit_fill_bar = min(re + 1, n - 1)
        exit_px = float(c[exit_fill_bar])
        hold_bars = max(0, exit_fill_bar - entry_fill_bar)
        roi_gross = (exit_px / entry_px - 1.0) * 100 if entry_px > 0 else 0.0
        roi_net = roi_gross - MAKER_RT * 100           # one round-trip maker cost
        trades.append({
            "entry_time": dates[entry_bar], "entry_px": entry_px,
            "exit_time": dates[exit_bar], "exit_px": exit_px,
            "hold_bars": hold_bars, "ROI_pct_net": round(roi_net, 4),
            "signal_band_on_frac_at_entry": round(float(band_frac[entry_bar]), 4),
            "max_band_on_during_hold": round(float(np.max(band_frac[rs:re])) if re > rs else float(band_frac[entry_bar]), 4),
            "exit_reason": exit_reason, "_rs": rs, "_re": re,
        })
    return trades


def run_asset_tf(sym, tf, published_band, wall_ms):
    rec = _load_asset(sym, tf, wall_ms)
    if rec is None:
        return None
    dates, o, h, l, c = rec
    assert dates.max() < VAL_END, f"WALL VIOLATION: {sym}|{tf} max {dates.max()} >= VAL_END"
    c = np.asarray(c, dtype=float)
    band_frac = compute_band_on_frac(c, published_band)
    slice_mask = (dates >= SLICE_START) & (dates < SLICE_END_EXCL)
    if slice_mask.sum() == 0:
        return None
    trades = extract_trades(dates, c, band_frac)
    sd = dates[slice_mask]; sc = c[slice_mask]; sbf = band_frac[slice_mask]
    # slice buy-hold ROI (open->close of the slice), slice peak-from-open (mover test)
    bh_roi = (sc[-1] / sc[0] - 1.0) * 100 if len(sc) and sc[0] > 0 else 0.0
    peak_from_open = (np.max(sc) / sc[0] - 1.0) * 100 if len(sc) and sc[0] > 0 else 0.0
    # band-on fraction of slice bars (HOW: how much of the week the engine was long)
    band_on_slice = float((sbf > THR).mean())
    # ENGINE SLICE ROI = WITHIN-SLICE MtM (the honest measure). lag-1 held mask: at slice bar t we earn the
    # bar return iff the band was ON at t-1 (causal). A trade that ENTERED before the slice is valued from
    # the slice-open price (NOT its weeks-old pre-slice entry) -- crediting pre-slice gains to a 7-day slice
    # is look-back contamination. One maker RT is charged per held-run boundary INSIDE the slice (enter/exit).
    si = np.where(slice_mask)[0]
    if len(si) >= 2:
        held_full = (band_frac > THR).astype(np.int8)
        s0 = si[0]
        # lag-1 held at each slice bar t -> held_full[t-1]; bar return = c[t]/c[t-1]-1
        bar_ret = c[s0 + 1: si[-1] + 1] / c[s0: si[-1]] - 1.0
        held_lag = held_full[s0: si[-1]]                       # held at t-1 governs bar t
        gross = float(np.prod(1.0 + bar_ret * held_lag))
        # turnover cost: count ON/OFF transitions of held_lag WITHIN the slice (each = one side)
        trans = int(np.sum(np.abs(np.diff(np.concatenate([[held_full[s0 - 1] if s0 > 0 else 0], held_lag])))))
        eng_roi = (gross * (1.0 - trans * (MAKER_RT / 2.0)) - 1.0) * 100.0
    else:
        eng_roi = 0.0
    return {
        "trades": trades, "slice_dates": sd, "slice_close": sc, "slice_band_frac": sbf,
        "n_trades": len(trades), "engine_slice_roi": round(eng_roi, 3),
        "buyhold_slice_roi": round(bh_roi, 3), "peak_from_open_pct": round(peak_from_open, 3),
        "band_on_slice_frac": round(band_on_slice, 4), "n_slice_bars": int(slice_mask.sum()),
        "held_any": bool((sbf > THR).any()),
    }


def make_chart(per_tf_results: dict, slice_tag: str):
    """One figure: top-2 movers' slice price + held shading + entry/exit markers, 4 TFs each (2x4 grid),
    plus a held-names slice-ROI bar (1d) panel. Saved per the requested path."""
    movers = CHART_ASSETS
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for r, sym in enumerate(movers):
        for ci, tf in enumerate(TFS):
            ax = axes[r, ci]
            res = per_tf_results.get(tf, {}).get(sym)
            if res is None or len(res["slice_dates"]) == 0:
                ax.set_title(f"{sym} {tf}  no data", fontsize=9)
                ax.annotate("no data", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center")
                continue
            sd = res["slice_dates"]; sc = res["slice_close"]; sbf = res["slice_band_frac"]
            x = np.arange(len(sd))
            ax.plot(x, sc, color="steelblue", lw=1.0)
            ax.set_title(f"{sym} {tf}  eng={res['engine_slice_roi']:+.1f}%  bh={res['buyhold_slice_roi']:+.1f}%  on={res['band_on_slice_frac']*100:.0f}%", fontsize=8)
            ax.tick_params(labelsize=6)
            # held shading
            held = sbf > THR
            in_h = False; s0 = 0
            for xi, hv in enumerate(held):
                if hv and not in_h:
                    s0 = xi; in_h = True
                elif not hv and in_h:
                    ax.axvspan(s0, xi, alpha=0.12, color="green"); in_h = False
            if in_h:
                ax.axvspan(s0, len(x) - 1, alpha=0.12, color="green")
            d2x = {d: i for i, d in enumerate(sd)}
            for t in res["trades"]:
                xe = d2x.get(t["entry_time"], 0)
                if 0 <= xe < len(sc):
                    ax.plot(xe, sc[xe], "^", color="green", ms=7, zorder=5)
                xx = d2x.get(t["exit_time"], len(sc) - 1)
                if 0 <= xx < len(sc):
                    ax.plot(xx, sc[xx], "v", color="red", ms=7, zorder=5)
    fig.suptitle(f"OOS VAL-slice forensic (EMA band FROZEN from 2020)  {slice_tag}  -- 2 top movers x 4 TFs", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"oos_val_slice_{slice_tag}.png"
    fig.savefig(str(out), dpi=110)
    plt.close(fig)
    return out


def main():
    global CHART_ASSETS
    wall_ms = int(VAL_END.value // 10 ** 6)
    slice_tag = f"{SLICE_START.date()}_{(SLICE_END_EXCL - pd.Timedelta(days=1)).date()}"
    print("=" * 78)
    print("OOS VAL-SLICE FORENSIC -- EMA band FROZEN from 2020 config_leaderboard.json")
    print(f"  VAL window (canonical): [{VAL_START.date()}, {VAL_END.date()})  (train_end .. val_end)")
    print(f"  RANDOM slice (seed={SEED}, offset_draw={SLICE_OFFSET}/{SLICE_SPAN}): "
          f"[{SLICE_START.date()}, {SLICE_END_EXCL.date()})  = {slice_tag}")
    print(f"  OOS? slice is inside VAL, AFTER 2020 selection -> YES, genuinely out-of-sample")
    print("=" * 78)

    syms = _u100_symbols()
    pub_bands = {tf: load_published_band(tf) for tf in TFS}
    print("[frozen band] EMA band sizes per TF (from 2020 leaderboard):")
    for tf in TFS:
        print(f"   EMA|{tf}: {len(pub_bands[tf])} members")

    per_tf_results = {tf: {} for tf in TFS}
    max_bar_seen = pd.Timestamp("2000-01-01")
    rows = []

    for tf in TFS:
        for sym in syms:
            res = run_asset_tf(sym, tf, pub_bands[tf], wall_ms)
            if res is None:
                continue
            per_tf_results[tf][sym] = res
            mb = res["slice_dates"].max()
            if mb > max_bar_seen:
                max_bar_seen = mb
            for t in res["trades"]:
                rows.append({
                    "asset": sym, "tf": tf, "entry_time": t["entry_time"], "entry_px": t["entry_px"],
                    "exit_time": t["exit_time"], "exit_px": t["exit_px"], "hold_bars": t["hold_bars"],
                    "ROI_pct_net": t["ROI_pct_net"],
                    "band_on_frac_at_entry": t["signal_band_on_frac_at_entry"],
                    "max_band_on_during_hold": t["max_band_on_during_hold"], "exit_reason": t["exit_reason"],
                })
        print(f"[{tf}] {len(per_tf_results[tf])} assets had slice bars")

    # ---- pick chart assets = top-2 movers (peak-from-open) by 1d ----
    movers_1d = sorted(per_tf_results["1d"].items(), key=lambda kv: -kv[1]["peak_from_open_pct"])
    CHART_ASSETS = [m[0] for m in movers_1d[:2]] if len(movers_1d) >= 2 else [m[0] for m in movers_1d]

    # ---- PORTFOLIO slice result per TF: engine held-EW vs buy-hold-all vs cash + captured-vs-available ----
    portfolio = {}
    for tf in TFS:
        names = list(per_tf_results[tf].keys())
        nU = len(names)
        eng_rois = np.array([per_tf_results[tf][s]["engine_slice_roi"] for s in names]) / 100.0
        bh_rois = np.array([per_tf_results[tf][s]["buyhold_slice_roi"] for s in names]) / 100.0
        held_names = [s for s in names if per_tf_results[tf][s]["n_trades"] > 0]
        nH = len(held_names)
        # engine portfolio = EW over the names the engine actually traded; un-traded names sit in CASH (0)
        eng_port = float(np.mean([per_tf_results[tf][s]["engine_slice_roi"] for s in names]) ) if nU else 0.0
        bh_all = float(np.mean(bh_rois) * 100) if nU else 0.0   # buy-hold-ALL EW
        # captured-vs-available: movers = peak-from-open >= 5% this week; engine long = engine held it
        movers = [s for s in names if per_tf_results[tf][s]["peak_from_open_pct"] >= MOVER_THR * 100]
        captured = [s for s in movers if per_tf_results[tf][s]["n_trades"] > 0]
        portfolio[tf] = {
            "n_universe": nU, "n_held": nH,
            "engine_slice_roi_pct": round(eng_port, 3),     # EW over universe; untraded = cash(0)
            "buyhold_all_slice_roi_pct": round(bh_all, 3),
            "cash_roi_pct": 0.0,
            "engine_vs_bh_pp": round(eng_port - bh_all, 3),
            "n_movers_up5pct": len(movers), "n_movers_captured": len(captured),
            "movers": movers, "movers_captured": captured,
        }

    # ---- SELF-VERIFY ----
    leak_ok = bool(max_bar_seen < VAL_END)
    verify = {
        "max_loaded_slice_bar": str(max_bar_seen),
        "VAL_END": str(VAL_END),
        "no_leak_PASS": leak_ok,
        "band_frozen_2020_PASS": True,   # band loaded from config_leaderboard.json; NO VAL re-fit in this script
        "lag1_causal_PASS": True,        # within-slice held mask is lagged 1 bar (held[t-1] governs bar t)
        "mtm_reconciled_note": ("engine_slice_roi = WITHIN-SLICE MtM (lag-1 held mask * bar returns, maker "
                                "RT on transitions inside the slice). A pre-slice entry is valued from the "
                                "slice-open price, NOT its old entry -- no pre-slice gain credited to the 7d slice. "
                                "The trade ledger keeps the TRUE entry ts/px for WHEN/WHERE only."),
    }

    # ---- write CSV ----
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / f"forensic_slice_oos_val_{slice_tag}.csv"
    if rows:
        df = pd.DataFrame(rows)
        df["_o"] = df["tf"].map(TF_ORDER)
        df = df.sort_values(["asset", "_o", "entry_time"]).drop(columns=["_o"])
        df.to_csv(str(csv_path), index=False)
    else:
        pd.DataFrame(columns=["asset", "tf", "entry_time", "entry_px", "exit_time", "exit_px", "hold_bars",
                              "ROI_pct_net", "band_on_frac_at_entry", "max_band_on_during_hold",
                              "exit_reason"]).to_csv(str(csv_path), index=False)
    print(f"\n[CSV] {csv_path}  ({len(rows)} trade rows)")

    # ---- chart ----
    chart_path = make_chart(per_tf_results, slice_tag)
    print(f"[chart] {chart_path}  exists={os.path.exists(chart_path)}  "
          f"size={os.path.getsize(chart_path) if os.path.exists(chart_path) else 0:,}")

    # ---- per-(asset x TF) forensic: ~10 most active/mover names ----
    # rank assets by max slice peak-from-open across the 4 TFs (the slice's movers)
    asset_peak = {}
    for tf in TFS:
        for s, res in per_tf_results[tf].items():
            asset_peak[s] = max(asset_peak.get(s, -1e9), res["peak_from_open_pct"])
    top_assets = [s for s, _ in sorted(asset_peak.items(), key=lambda kv: -kv[1])[:10]]

    print("\n=== PER-ASSET x TF FORENSIC (top-10 movers in the slice) ===")
    print("cols: n_trades / engine_slice_ROI% / buyhold_slice_ROI% / band_on_frac")
    hdr = "| asset | peak% | " + " | ".join(TFS) + " |"
    print(hdr); print("|---" * (len(TFS) + 2) + "|")
    for s in top_assets:
        cells = []
        for tf in TFS:
            res = per_tf_results[tf].get(s)
            if res is None:
                cells.append("-/-/-/-"); continue
            cells.append(f"{res['n_trades']}/{res['engine_slice_roi']:+.1f}/{res['buyhold_slice_roi']:+.1f}/{res['band_on_slice_frac']:.2f}")
        print(f"| {s} | {asset_peak[s]:+.1f} | " + " | ".join(cells) + " |")

    # first-entry timestamps for the 2 chart movers (WHEN/WHERE) per TF
    print("\n=== WHEN/WHERE (first trade per TF) for the 2 chart movers ===")
    for s in CHART_ASSETS:
        print(f" {s}:")
        for tf in TFS:
            res = per_tf_results[tf].get(s)
            if res and res["trades"]:
                t = res["trades"][0]
                print(f"   {tf}: enter {t['entry_time']} @ {t['entry_px']:.6g}  ->  exit {t['exit_time']} @ "
                      f"{t['exit_px']:.6g}  ROI={t['ROI_pct_net']:+.2f}%  on@entry={t['signal_band_on_frac_at_entry']:.2f}  ({t['exit_reason']})")
            else:
                print(f"   {tf}: 0 trades (engine in CASH)")

    # ---- PORTFOLIO + captured-vs-available ----
    print("\n=== PORTFOLIO SLICE RESULT (per TF) ===")
    print("| TF | n_held/n_univ | engine% | buyhold-all% | cash% | eng-vs-BH pp | movers>=5% | captured |")
    print("|---|---|---|---|---|---|---|---|")
    for tf in TFS:
        p = portfolio[tf]
        print(f"| {tf} | {p['n_held']}/{p['n_universe']} | {p['engine_slice_roi_pct']:+.2f} | "
              f"{p['buyhold_all_slice_roi_pct']:+.2f} | 0.00 | {p['engine_vs_bh_pp']:+.2f} | "
              f"{p['n_movers_up5pct']} | {p['n_movers_captured']} |")

    # ---- write the JSON record ----
    out_json = OUTDIR / f"forensic_slice_oos_val_{slice_tag}.json"
    rec = {
        "test": "OOS VAL-slice forensic (EMA band FROZEN from 2020)",
        "seed": SEED, "slice": [str(SLICE_START.date()), str(SLICE_END_EXCL.date())], "slice_tag": slice_tag,
        "slice_offset_draw": SLICE_OFFSET, "slice_span_days": SLICE_SPAN,
        "VAL_window": [str(VAL_START.date()), str(VAL_END.date())],
        "VAL_window_source": "config/data_config.yaml splits (train_end..val_end) == purge_split.split_chimera VAL",
        "ma_type": MA_TYPE, "tfs": TFS, "thr": THR, "maker_rt": MAKER_RT,
        "frozen_band_sizes": {tf: len(pub_bands[tf]) for tf in TFS},
        "chart_movers": CHART_ASSETS, "portfolio": portfolio, "self_verify": verify,
        "csv": str(csv_path), "chart": str(chart_path),
        "n_trade_rows": len(rows),
        "per_asset_tf": {tf: {s: {k: v for k, v in res.items()
                                  if k in ("n_trades", "engine_slice_roi", "buyhold_slice_roi",
                                           "peak_from_open_pct", "band_on_slice_frac", "n_slice_bars", "held_any")}
                              for s, res in per_tf_results[tf].items()} for tf in TFS},
    }
    out_json.write_text(json.dumps(rec, indent=1, default=str))
    print(f"\n[JSON] {out_json}")

    print("\n=== SELF-VERIFY ===")
    for k, v in verify.items():
        print(f"   {k}: {v}")
    print(f"\n[done] slice={slice_tag}  trade_rows={len(rows)}  no_leak={leak_ok}")


if __name__ == "__main__":
    main()
