"""Chimera DECOMPOSER / VIEWER -- select (period, timeframe, asset|universe) and VIEW how every chimera feature
behaved over that slice. DESCRIPTIVE inspection tool (NOT signal-mining).

This is the "pick a window and see all the feature behaviours" capability:
  - asset mode    : every chimera feature for one asset over [start,end] at a cadence -- level, trend within the
                    window, min/max, percentile-vs-full-history (is it unusually high/low now?), spike count, family.
  - universe mode : for a (period, cadence) across ALL assets -- per-asset performance ranking + each feature's
                    cross-asset behaviour (median / dispersion / which assets are extreme) for that window.

Reuses the narrate feature_map (classify/group_columns/FEATURES) for family taxonomy + human titles (100% coverage),
and ChimeraLoader for canonical data access. Output: a readable table to stdout + a CSV/JSON under runs/mining/.

Run:
  python -m mining.decompose --asset BTC --cadence 4h --start 2025-01-01 --end 2025-02-01
  python -m mining.decompose --universe u100 --cadence 1d --start 2025-10-01 --end 2025-11-01 --top-n 25
  python -m mining.decompose --asset ETH --cadence 1h --start 2025-03-01 --end 2025-03-15 --json
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader          # noqa: E402
from narrate import feature_map as fm                      # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

_META = {"timestamp", "date", "bar_id", "open", "high", "low", "close", "volume", "volume_usd",
         "buy_vol", "sell_vol", "tick_count", "tick_seq", "asset_dna", "is_u10", "is_u50", "is_u100",
         "returns_clean", "fp_fund_panel"}


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def _load(sym: str, cadence: str) -> pl.DataFrame:
    df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence)
    return df.sort("date")


def _feature_cols(df: pl.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in _META or c.startswith("target_"):
            continue
        if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            out.append(c)
    return out


def _window_mask(df: pl.DataFrame, start: str | None, end: str | None) -> np.ndarray:
    d = df["date"].cast(pl.Utf8).to_numpy()
    m = np.ones(len(d), bool)
    if start:
        m &= (d >= start)
    if end:
        m &= (d <= end)
    return m


def decompose_asset(sym: str, cadence: str, start=None, end=None) -> dict:
    df = _load(sym, cadence)
    mask = _window_mask(df, start, end)
    n_win = int(mask.sum())
    if n_win < 2:
        return {"error": f"window [{start},{end}] has {n_win} bars for {sym}/{cadence}"}
    close = df["close"].to_numpy().astype(float)
    cw = close[mask]
    price_move = float((cw[-1] / cw[0] - 1) * 100)
    peak = np.maximum.accumulate(cw); dd = float(((cw - peak) / peak).min() * 100)
    rows = []
    for col in _feature_cols(df):
        x = df[col].to_numpy().astype(float)
        xf = x[np.isfinite(x)]                 # full history (percentile context)
        xw = x[mask]; xw = xw[np.isfinite(xw)]
        if len(xw) < 1:
            continue
        hist_mean = float(np.mean(xf)) if len(xf) else float("nan")
        hist_std = float(np.std(xf)) if len(xf) else float("nan")
        win_mean = float(np.mean(xw))
        # percentile of the window-MEAN within the full-history distribution of this feature
        pct = float((xf < win_mean).mean() * 100) if len(xf) else float("nan")
        # trend within window (first vs last finite value)
        trend = float(xw[-1] - xw[0])
        # spikes: bars in window > 3 sigma from history
        spikes = int(np.sum(np.abs(xw - hist_mean) > 3 * hist_std)) if hist_std and hist_std > 0 else 0
        feat = fm.FEATURES.get(col)
        rows.append({"feature": col, "family": fm.classify(col),
                     "title": feat.title if feat else col,
                     "win_mean": round(win_mean, 5), "pctile_vs_history": round(pct, 1),
                     "win_min": round(float(np.min(xw)), 5), "win_max": round(float(np.max(xw)), 5),
                     "win_trend_last_minus_first": round(trend, 5), "spikes_3sig": spikes,
                     "n_obs": int(len(xw))})
    # period lenses
    retw = np.diff(np.log(np.clip(cw, 1e-12, None)))
    ac1 = float(np.corrcoef(retw[:-1], retw[1:])[0, 1]) if len(retw) > 50 and np.std(retw) > 0 else float("nan")
    regimes = {}
    if "regime_label" in df.columns:
        rl = df["regime_label"].to_numpy()[mask]
        for k in (0, 1, 2):
            regimes[f"regime{k}_share"] = round(float(np.mean(rl == k)), 3)
    return {"asset": _norm_sym(sym), "cadence": cadence, "start": start, "end": end, "n_bars": n_win,
            "price_move_pct": round(price_move, 2), "max_drawdown_pct": round(dd, 2),
            "window_ac1": round(ac1, 4) if np.isfinite(ac1) else None, "regime_shares": regimes,
            "features": rows}


def render_asset_text(d: dict) -> str:
    if "error" in d:
        return d["error"]
    lines = [f"## {d['asset']} -- {d['cadence']} -- {d['start']} -> {d['end']}  ({d['n_bars']} bars)",
             f"price move {d['price_move_pct']:+.2f}%  | maxDD {d['max_drawdown_pct']:.2f}%  | "
             f"window AC1 {d['window_ac1']}  | regimes {d['regime_shares']}",
             "",
             "ALL chimera feature behaviours over the window (grouped by family; pctile = window-mean vs full history):"]
    by_fam: dict[str, list] = {}
    for r in d["features"]:
        by_fam.setdefault(r["family"], []).append(r)
    for fam in fm.FAMILY_ORDER + [k for k in by_fam if k not in fm.FAMILY_ORDER]:
        rs = by_fam.get(fam)
        if not rs:
            continue
        lines.append(f"\n[{fam}]  ({len(rs)} features)")
        for r in sorted(rs, key=lambda x: -abs(x["pctile_vs_history"] - 50)):  # most-unusual first
            flag = "  <== UNUSUAL" if abs(r["pctile_vs_history"] - 50) > 40 else ""
            sp = f"  spikes={r['spikes_3sig']}" if r["spikes_3sig"] else ""
            lines.append(f"  {r['title'][:42]:42s} p{r['pctile_vs_history']:.0f}  "
                         f"mean={r['win_mean']:+.4g} trend={r['win_trend_last_minus_first']:+.4g}{sp}{flag}")
    return "\n".join(lines)


def decompose_universe(cadence: str, start=None, end=None, universe="u100", top_n=25) -> dict:
    import glob
    files = sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cadence /
                                 f"*_v51_chimera_{cadence}_*.parquet")))
    perf = {}; feat_win = {}     # sym -> %move ; feat -> {sym: window-mean}
    for f in files:
        sym = Path(f).stem.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        if universe in ("u10", "u50", "u100") and universe in df.columns:
            if not bool(df[universe][0]):
                continue
        mask = _window_mask(df, start, end)
        if mask.sum() < 2:
            continue
        close = df["close"].to_numpy().astype(float)[mask]
        perf[sym] = float((close[-1] / close[0] - 1) * 100)
        for col in _feature_cols(df):
            xw = df[col].to_numpy().astype(float)[mask]
            xw = xw[np.isfinite(xw)]
            if len(xw):
                feat_win.setdefault(col, {})[sym] = float(np.mean(xw))
    if not perf:
        return {"error": f"no assets with data in window [{start},{end}] at {cadence}"}
    ranked = sorted(perf.items(), key=lambda x: -x[1])
    # per-feature cross-asset behaviour (median + dispersion + extreme assets), this window
    feat_rows = []
    for col, d in feat_win.items():
        vals = np.array(list(d.values()))
        if len(vals) < 5:
            continue
        med = float(np.median(vals))
        items = sorted(d.items(), key=lambda x: -x[1])
        feat_rows.append({"feature": col, "family": fm.classify(col),
                          "title": (fm.FEATURES[col].title if col in fm.FEATURES else col),
                          "cross_asset_median": round(med, 5), "cross_asset_std": round(float(np.std(vals)), 5),
                          "n_assets": len(vals),
                          "highest": [f"{s}({v:.3g})" for s, v in items[:3]],
                          "lowest": [f"{s}({v:.3g})" for s, v in items[-3:]]})
    return {"cadence": cadence, "start": start, "end": end, "universe": universe, "n_assets": len(perf),
            "top_performers": [{"sym": s, "move_pct": round(p, 2)} for s, p in ranked[:top_n]],
            "bottom_performers": [{"sym": s, "move_pct": round(p, 2)} for s, p in ranked[-5:]],
            "feature_cross_asset": sorted(feat_rows, key=lambda r: r["family"])}


def render_universe_text(d: dict) -> str:
    if "error" in d:
        return d["error"]
    lines = [f"## UNIVERSE {d['universe']} -- {d['cadence']} -- {d['start']} -> {d['end']}  ({d['n_assets']} assets)",
             "", f"TOP {len(d['top_performers'])} PERFORMERS over the window:"]
    lines += [f"  {r['sym']:14s} {r['move_pct']:+8.1f}%" for r in d["top_performers"]]
    lines.append("\nFEATURE cross-asset behaviour this window (median across assets + the 3 highest/lowest assets):")
    by_fam: dict[str, list] = {}
    for r in d["feature_cross_asset"]:
        by_fam.setdefault(r["family"], []).append(r)
    for fam in fm.FAMILY_ORDER + [k for k in by_fam if k not in fm.FAMILY_ORDER]:
        rs = by_fam.get(fam)
        if not rs:
            continue
        lines.append(f"\n[{fam}]")
        for r in rs:
            lines.append(f"  {r['title'][:38]:38s} med={r['cross_asset_median']:+.4g}  hi:{','.join(r['highest'])}")
    return "\n".join(lines)


# ------------------------------------------------------------------ VISUALS (match src/pipeline/inspect_dataset.py)
PLOTS_DIR = ROOT / "plots"


def _save_plot(fig, name: str) -> str:
    import matplotlib.pyplot as plt
    from datetime import date
    out_dir = PLOTS_DIR / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name.removesuffix('.png')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rel = f"plots/{date.today().isoformat()}/{name.removesuffix('.png')}.png"
    print(f"  [PLOT] {rel}")
    return str(path)


def plot_asset(sym: str, cadence: str, start, end) -> list:
    """Visual decomposition for one asset over a window: feature x time z-score heatmap (every feature's
    behaviour at a glance, grouped by family), price+regime ribbon, and percentile-vs-history bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _load(sym, cadence)
    mask = _window_mask(df, start, end)
    if mask.sum() < 3:
        print("  [PLOT] window too small"); return []
    cols = _feature_cols(df)
    # group cols by family (FAMILY_ORDER), keep only features with variance in-window
    grouped, ylabels, rows = [], [], []
    bymask = {}
    for c in cols:
        bymask.setdefault(fm.classify(c), []).append(c)
    Z = []
    sep_positions, fam_label_pos = [], []
    for fam in fm.FAMILY_ORDER + [k for k in bymask if k not in fm.FAMILY_ORDER]:
        fcols = bymask.get(fam, [])
        block_start = len(Z)
        for c in fcols:
            x = df[c].to_numpy().astype(float)[mask]
            mu, sd = np.nanmean(x), np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-9:
                continue
            Z.append(np.clip((x - mu) / sd, -3, 3))
            feat = fm.FEATURES.get(c)
            ylabels.append((feat.title if feat else c)[:34])
        if len(Z) > block_start:
            fam_label_pos.append((block_start + len(Z) - 1) / 2.0)
            grouped.append(fam)
            sep_positions.append(len(Z) - 0.5)
    if not Z:
        print("  [PLOT] no varying features"); return []
    Z = np.array(Z)
    close = df["close"].to_numpy().astype(float)[mask]
    dates = df["date"].cast(pl.Utf8).to_numpy()[mask]
    saved = []
    # --- Fig 1: feature x time heatmap (the 'view all behaviours' visual) ---
    fig, (axp, axh) = plt.subplots(2, 1, figsize=(16, max(10, len(Z) * 0.16)),
                                   gridspec_kw={"height_ratios": [1, 9]}, sharex=True)
    axp.plot(np.arange(len(close)), close, color="#111", lw=1.0)
    axp.set_ylabel("close"); axp.set_title(
        f"{_norm_sym(sym)} -- {cadence} -- {start} -> {end}  ({int(mask.sum())} bars, {len(Z)} live features)")
    if "regime_label" in df.columns:
        rl = df["regime_label"].to_numpy()[mask]
        cmap = {0: "#d65f5f", 1: "#dddddd", 2: "#5fa55f"}
        for i in range(len(rl)):
            axp.axvspan(i - 0.5, i + 0.5, color=cmap.get(int(rl[i]) if np.isfinite(rl[i]) else 1, "#fff"), alpha=0.18, lw=0)
    im = axh.imshow(Z, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3, interpolation="nearest")
    axh.set_yticks(range(len(ylabels))); axh.set_yticks(np.arange(len(ylabels)))
    axh.set_yticklabels(ylabels, fontsize=5)
    for sp in sep_positions[:-1]:
        axh.axhline(sp, color="black", lw=0.4)
    for pos, fam in zip(fam_label_pos, grouped):
        axh.text(-0.06, pos, fam.upper(), transform=axh.get_yaxis_transform(),
                 ha="right", va="center", fontsize=6, fontweight="bold", color="#333")
    nx = len(close); step = max(1, nx // 12)
    axh.set_xticks(range(0, nx, step)); axh.set_xticklabels([str(dates[i])[:10] for i in range(0, nx, step)],
                                                            rotation=45, fontsize=6, ha="right")
    axh.set_xlabel("bar (window)")
    fig.colorbar(im, ax=axh, shrink=0.5, label="z within window (clipped +-3)")
    saved.append(_save_plot(fig, f"decompose_{_norm_sym(sym)}_{cadence}_{start}_{end}_heatmap"))
    # --- Fig 2: percentile-vs-history bars (what stood out this window) ---
    d = decompose_asset(sym, cadence, start, end)
    feats = sorted(d["features"], key=lambda r: -abs(r["pctile_vs_history"] - 50))[:30]
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    names = [f["title"][:34] for f in feats][::-1]
    devs = [f["pctile_vs_history"] - 50 for f in feats][::-1]
    ax2.barh(range(len(names)), devs, color=["#d65f5f" if v < 0 else "#3b7dd8" for v in devs])
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=7)
    ax2.axvline(0, color="black", lw=0.6); ax2.set_xlim(-50, 50)
    ax2.set_xlabel("percentile vs full history  (-50 = lowest ever ... +50 = highest ever)")
    ax2.set_title(f"{_norm_sym(sym)} {cadence} {start}->{end}: most UNUSUAL features this window")
    saved.append(_save_plot(fig2, f"decompose_{_norm_sym(sym)}_{cadence}_{start}_{end}_unusual"))
    return saved


def plot_universe(cadence, start, end, universe, d) -> list:
    """Visual: top performers bar + feature x asset window-mean heatmap (z across assets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "error" in d:
        print("  [PLOT]", d["error"]); return []
    saved = []
    tp = d["top_performers"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(tp) * 0.3)))
    names = [r["sym"] for r in tp][::-1]; moves = [r["move_pct"] for r in tp][::-1]
    ax.barh(range(len(names)), moves, color=["#d65f5f" if v < 0 else "#3b7dd8" for v in moves])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("% move over window"); ax.set_title(f"{universe} {cadence} {start}->{end}: top performers")
    saved.append(_save_plot(fig, f"decompose_{universe}_{cadence}_{start}_{end}_topperf"))
    # feature x asset heatmap -- rank features by cross-asset variation (drop globally-constant/sparse ones so the
    # grid is dense + informative; e.g. BTC-global ETF/USDT features have no cross-asset variance -> excluded)
    fr = sorted([r for r in d["feature_cross_asset"] if r["cross_asset_std"] and r["cross_asset_std"] > 1e-9],
                key=lambda r: -abs(r["cross_asset_std"]))[:40]
    assets = [r["sym"] for r in tp[:20]]
    # rebuild a matrix from the JSON's per-feature highest/lowest is lossy; reload window-means compactly
    import glob
    files = sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cadence /
                                 f"*_v51_chimera_{cadence}_*.parquet")))
    fcols = [r["feature"] for r in fr]
    M = np.full((len(fcols), len(assets)), np.nan)
    aidx = {a: j for j, a in enumerate(assets)}
    for f in files:
        s = Path(f).stem.split("_v51_")[0].upper()
        if s not in aidx:
            continue
        try:
            dfx = pl.read_parquet(f, columns=["date"] + [c for c in fcols if True])
        except Exception:
            continue
        m = _window_mask(dfx, start, end)
        for i, c in enumerate(fcols):
            if c in dfx.columns:
                v = dfx[c].to_numpy().astype(float)[m]
                v = v[np.isfinite(v)]
                if len(v):
                    M[i, aidx[s]] = float(np.mean(v))
    # z across assets per feature
    Mz = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        row = M[i]; ok = np.isfinite(row)
        if ok.sum() > 2 and np.nanstd(row[ok]) > 1e-9:
            Mz[i] = (row - np.nanmean(row[ok])) / np.nanstd(row[ok])
    fig2, ax2 = plt.subplots(figsize=(max(10, len(assets) * 0.6), max(8, len(fcols) * 0.3)))
    im = ax2.imshow(np.nan_to_num(Mz), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax2.set_xticks(range(len(assets))); ax2.set_xticklabels(assets, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(fcols)))
    ax2.set_yticklabels([(fm.FEATURES[c].title if c in fm.FEATURES else c)[:30] for c in fcols], fontsize=6)
    ax2.set_title(f"{universe} {cadence} {start}->{end}: feature window-mean (z across top assets)")
    fig2.colorbar(im, ax=ax2, shrink=0.5, label="z across assets")
    saved.append(_save_plot(fig2, f"decompose_{universe}_{cadence}_{start}_{end}_featheatmap"))
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mining.decompose",
                                 description="VIEW all chimera feature behaviours for a (period, timeframe, asset|universe).")
    ap.add_argument("--asset", help="asset symbol (e.g. BTC, ETHUSDT). Omit for --universe mode.")
    ap.add_argument("--universe", help="u10|u50|u100 -- cross-asset view for the period")
    ap.add_argument("--cadence", default="4h", help="timeframe: 1d|4h|1h|30m|15m")
    ap.add_argument("--start", help="ISO date window start")
    ap.add_argument("--end", help="ISO date window end")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--plots", action="store_true", help="also render visual decomposition to plots/<date>/")
    a = ap.parse_args(argv)
    if not a.asset and not a.universe:
        ap.error("provide --asset SYM or --universe u100")
    if a.asset:
        d = decompose_asset(a.asset, a.cadence, a.start, a.end)
        tag = f"{_norm_sym(a.asset)}_{a.cadence}_{a.start}_{a.end}"
        text = render_asset_text(d)
    else:
        d = decompose_universe(a.cadence, a.start, a.end, a.universe, a.top_n)
        tag = f"{a.universe}_{a.cadence}_{a.start}_{a.end}"
        text = render_universe_text(d)
    (OUT / f"decompose_{tag}.json").write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    print(json.dumps(d, indent=2, default=str) if a.json else text)
    print(f"\n[written] {OUT / ('decompose_'+tag+'.json')}")
    if a.plots:
        print("\n[rendering visuals -> plots/<date>/]")
        if a.asset:
            plot_asset(a.asset, a.cadence, a.start, a.end)
        else:
            plot_universe(a.cadence, a.start, a.end, a.universe, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
