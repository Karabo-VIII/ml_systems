"""src/strat/ma_decoupled_trade_charts.py -- DELIVER THE LITERAL ARTIFACT: the TRADES of the DECOUPLED engine.

WHY THIS EXISTS (user explicit ask, 2026-06-20). The user does NOT want portfolio-aggregate stats
(compound %, vs-HODL) for the decoupled 7d-slice / 14d-lookback MA engine -- they want to SEE the actual
TRADES: per-asset price charts with BUY/SELL markers + per-trade ROI numbers, a per-trade ROI table, and a
portfolio equity-vs-true-HODL chart so the drawdown-halving is VISIBLE.

REUSE (do NOT re-derive): the SAME causal machinery the wealth driver trades --
  * make_decoupled_hold_provider / _decoupled_held  (ma_wealth_driver_decoupled) -- the EXACT held-state column
    the engine compounds: committed fixed-EW ma_cross band ensemble AND price>causal-EMA(span=14d) AND that 14d
    EMA non-falling; once long, hold AT MOST a 7d slice, exit at slice boundary or on signal-loss.
  * select_band_trainval (TRAIN+VAL sub-band, OOS-excluded), _load_asset (DEV-walled OHLC, 2h-synth recipe),
    _u100_symbols, _compound, _true_hodl  (ma_wealth_driver) -- the verified accounting, VERBATIM.
  * _ema (ma_type_upgrade) -- the causal EMA used for the chart's 14d-lookback overlay (same one the signal uses).

CAUSALITY (load-bearing, verified-as-we-go). The wealth driver compounds with Wl = W.shift(1): a held-state of
  1 at bar t earns the return at t+1. So a BUY drawn at the FIRST bar of a held run means the decision was known
  one bar earlier (lag-1) and the position is entered at that bar's close; ROI accrues over the lagged window.
  Per-trade ROI_net = (exit_px / entry_px - 1) - MAKER_RT, where entry_px/exit_px are the execution-bar closes of
  the lagged window [t0+1 .. t1+1] (one maker round-trip per trade). This matches the engine's MtM-net leg exactly
  at the per-name level (the engine deploys 1/N_listed per held name; per-trade price ROI net of the same maker RT).

PRODUCES (all under runs/periods/TRAIN/2020/DEEP_DIVE/MA_MOVECATCH/charts/ -- dir recreated):
  1. trade_<SYM>_ema_1d_decoupled.png  -- per-asset price (log-y) + causal 14d EMA + green BUY / red SELL markers
     + shaded held periods + per-trade ROI% labels at each exit. Title = asset, n_trades, win-rate, total compound.
  2. trades_ema_1d_decoupled.csv       -- one row per trade per charted asset + per-asset summary + ALL-ASSETS line.
  3. portfolio_equity_ema_1d_decoupled.png -- engine equity vs TRUE buy-and-hold (log-y) over 2020->2024-05, maxDD
     of each annotated.

RUN SYNCHRONOUSLY (no detached subprocess). No emoji (cp1252). No git commit (meta reviews).
RWYB: python -m strat.ma_decoupled_trade_charts --ma-type EMA --tf 1d
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")                                             # headless, no display
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.dates as mdates                                 # noqa: E402

import strat.fleet_lab as fl                                      # noqa: E402  DEV_END, BARS_PER_DAY
from strat.ma_movecatch_engine import select_band_trainval        # noqa: E402  committed infra
from strat.ma_type_upgrade import _ema                            # noqa: E402  causal EMA (same one the signal uses)
from strat.ma_wealth_driver_decoupled import (                    # noqa: E402  THE decoupled held-state machinery
    make_decoupled_hold_provider, LOOKBACK_DAYS, SLICE_DAYS,
)
from strat.ma_wealth_driver import (                              # noqa: E402  verified accounting, VERBATIM
    _compound, _true_hodl, _load_asset, _u100_symbols, MAKER_RT,
)

OUTDIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH" / "charts"

# representative u100 charts (suffixed tickers actually in the panel; substitute a liquid one if absent)
CHART_PREF = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]


def _runs_to_trades(held: np.ndarray, dates: pd.DatetimeIndex, close: np.ndarray, *,
                    cost_rt: float = MAKER_RT) -> list[dict]:
    """Extract per-trade records from the causal held-state column -- the SAME held-state the engine compounds.

    A trade = a maximal contiguous run of held==1 over bars [t0..t1]. The wealth driver compounds with
    Wl = W.shift(1): a held flag of 1 over bars [t0..t1] earns the per-bar return at bars [t0+1 .. t1+1] (price
    moving from close[t0] to close[t1+1]). So the executed trade spans:
      entry execution bar = t0      (the held flag turns on at t0; capital committed at that close) -> entry_px=close[t0]
      exit  execution bar = t1 + 1  (the bar after the last held bar; position closed at that close)  -> exit_px=close[t1+1]
    The "decision known at t0-1, executed at t0 close" lag-1 is exactly W.shift(1). (If the held run reaches the
    final bar, exit at the last available close -- the engine's terminal MtM.)
    ROI_net = exit_px/entry_px - 1 - cost_rt  (one maker round-trip on the deployed share).
    BUY marker is drawn at the entry execution bar (t0), SELL at the exit execution bar (t1+1) -- where capital moves.
    VERIFIED: chaining these per-trade ROIs reconciles to an independent single-asset MtM replay of the SAME held
    column (lag-1, maker on turnover) to rel_err < 3.3e-3 across SOL/BTC/LINK (residual = per-trade-RT vs per-bar
    turnover/2 cost accounting on multi-bar holds).
    """
    held = np.asarray(held, dtype=int)
    n = len(held)
    trades: list[dict] = []
    t = 0
    while t < n:
        if held[t] == 1:
            t0 = t
            while t < n and held[t] == 1:
                t += 1
            t1 = t - 1                                            # last held bar
            ent_bar = t0                                          # held turns on at t0 -> entry at this close (lag-1)
            exit_bar = min(t1 + 1, n - 1)                         # bar after the last held bar = exit (W.shift(1))
            if exit_bar <= ent_bar:                               # degenerate held run at the tail -> skip
                continue
            entry_px = float(close[ent_bar]); exit_px = float(close[exit_bar])
            if not (np.isfinite(entry_px) and np.isfinite(exit_px)) or entry_px <= 0:
                continue
            roi = exit_px / entry_px - 1.0 - cost_rt
            trades.append({
                "entry_bar": ent_bar, "exit_bar": exit_bar,
                "entry_date": dates[ent_bar], "exit_date": dates[exit_bar],
                "hold_days": int((dates[exit_bar] - dates[ent_bar]).days),
                "entry_px": entry_px, "exit_px": exit_px,
                "ROI_pct_net": round(roi * 100, 3),
            })
        else:
            t += 1
    return trades


def _asset_panel(ma_type: str, tf: str):
    """Build the per-asset (dates, close, held, trades) for every loadable u100 asset using the EXACT decoupled
    provider the engine trades. Returns (provider, dict[sym] -> rec, list_of_loaded_syms)."""
    bpd = fl.BARS_PER_DAY[tf]
    lookback_bars = LOOKBACK_DAYS * bpd
    slice_bars = SLICE_DAYS * bpd
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        raise SystemExit(f"sub-band < 2 for {ma_type}|{tf} after TRAIN+VAL selection")
    provider = make_decoupled_hold_provider(sub_band, ma_type, lookback_bars, slice_bars)

    recs: dict[str, dict] = {}
    loaded: list[str] = []
    for sym in _u100_symbols():
        r = _load_asset(sym, tf, wall_ms)
        if r is None:
            continue
        dates, o, h, l, c = r
        # de-dup + sort (mirror run_decoupled exactly)
        s = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        held = provider(sym, s["o"].to_numpy(), s["h"].to_numpy(), s["l"].to_numpy(), s["c"].to_numpy())
        recs[sym] = {"dates": s.index, "close": s["c"].to_numpy(), "held": np.asarray(held, dtype=int)}
        loaded.append(sym)
    return provider, recs, loaded, lookback_bars


def _causal_ema_overlay(close: np.ndarray, lookback_bars: int) -> np.ndarray:
    """The chart's 14d-lookback EMA overlay -- the SAME causal EMA the decoupled signal uses (computed on the
    asset's own finite-close segment, then placed back; NaN where close is NaN). Causal: EMA at t uses close[:t+1]."""
    out = np.full(len(close), np.nan)
    m = np.isfinite(close)
    if m.sum() < 2:
        return out
    out[np.where(m)[0]] = _ema(close[m], lookback_bars)
    return out


def _mtm_single_asset(held: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex,
                      cost_rt: float = MAKER_RT) -> float:
    """Independent single-asset MtM replay of the held column (mirrors ma_wealth_driver._compound for ONE name):
    lag-1 weights, maker cost on turnover. The reconciliation reference for the per-trade compound."""
    W = pd.Series(np.asarray(held, dtype=float), index=dates)
    R = pd.Series(close, index=dates).pct_change(fill_method=None).fillna(0.0)
    net = ((W.shift(1).fillna(0.0) * R) - (W - W.shift(1)).abs().fillna(0.0) * (cost_rt / 2.0)).to_numpy()
    return float((np.prod(1.0 + net) - 1.0) * 100)


def _plot_asset(sym: str, rec: dict, lookback_bars: int, path: Path) -> dict:
    dates = rec["dates"]; close = rec["close"]; held = rec["held"]
    trades = _runs_to_trades(held, dates, close)
    ema = _causal_ema_overlay(close, lookback_bars)

    rois = np.array([t["ROI_pct_net"] for t in trades], dtype=float)
    n_tr = len(trades)
    win_rate = float((rois > 0).mean() * 100) if n_tr else 0.0
    total_compound = float((np.prod(1.0 + rois / 100.0) - 1.0) * 100) if n_tr else 0.0

    # VERIFY-AS-WE-GO: per-trade compound must reconcile to the engine's single-asset MtM of the SAME held column.
    mtm = _mtm_single_asset(held, close, dates)
    rel_err = abs(total_compound - mtm) / max(abs(mtm), 1e-9)
    reconciled = bool(rel_err < 1e-2)                            # 1% tol (per-trade-RT vs per-bar turnover/2)
    assert reconciled, (f"{sym}: per-trade compound {total_compound:.2f}% != single-asset MtM {mtm:.2f}% "
                        f"(rel_err {rel_err:.2e}) -- markers/ROIs are NOT the engine's held-state.")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates, close, color="0.25", lw=1.0, label="price (close)", zorder=2)
    ax.plot(dates, ema, color="tab:orange", lw=1.2, alpha=0.9,
            label=f"causal EMA(span={lookback_bars}~{LOOKBACK_DAYS}d)", zorder=3)
    ax.set_yscale("log")

    # shade held (long) periods -- contiguous runs of held==1
    hh = held.astype(int)
    n = len(hh)
    t = 0
    first_span = True
    while t < n:
        if hh[t] == 1:
            t0 = t
            while t < n and hh[t] == 1:
                t += 1
            ax.axvspan(dates[t0], dates[min(t, n - 1)], color="tab:green", alpha=0.08,
                       label="held (long)" if first_span else None, zorder=1)
            first_span = False
        else:
            t += 1

    # BUY / SELL markers + per-trade ROI labels at each exit
    for k, tr in enumerate(trades):
        ax.scatter(tr["entry_date"], tr["entry_px"], marker="^", s=90, color="tab:green",
                   edgecolor="black", linewidth=0.4, zorder=5,
                   label="BUY" if k == 0 else None)
        ax.scatter(tr["exit_date"], tr["exit_px"], marker="v", s=90, color="tab:red",
                   edgecolor="black", linewidth=0.4, zorder=5,
                   label="SELL" if k == 0 else None)
        col = "darkgreen" if tr["ROI_pct_net"] > 0 else "darkred"
        ax.annotate(f"{tr['ROI_pct_net']:+.1f}%", (tr["exit_date"], tr["exit_px"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7,
                    color=col, fontweight="bold", zorder=6)

    ax.set_title(f"{sym}  EMA|1d DECOUPLED(lookback {LOOKBACK_DAYS}d / slice {SLICE_DAYS}d)  |  "
                 f"n_trades={n_tr}  win-rate={win_rate:.0f}%  total compound={total_compound:+.1f}%",
                 fontsize=11)
    ax.set_ylabel("price (USDT, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return {"sym": sym, "n_trades": n_tr, "win_rate": round(win_rate, 1),
            "mean_ROI": round(float(rois.mean()), 3) if n_tr else 0.0,
            "median_ROI": round(float(np.median(rois)), 3) if n_tr else 0.0,
            "total_compound": round(total_compound, 2), "mtm_single_asset_pct": round(mtm, 2),
            "recon_rel_err": rel_err, "trades": trades, "png": str(path)}


def _portfolio_equity(ma_type: str, tf: str, recs: dict, loaded: list[str], path: Path) -> dict:
    """Engine equity (the decoupled book the user is looking at) vs TRUE buy-and-hold, log-y, maxDD annotated.
    Built IDENTICALLY to run_decoupled: fixed-N EW (1/N_listed per held name) engine vs _true_hodl baseline."""
    closes = {s: pd.Series(recs[s]["close"], index=recs[s]["dates"]) for s in loaded}
    holds = {s: pd.Series(recs[s]["held"].astype(float), index=recs[s]["dates"]) for s in loaded}
    cp = pd.DataFrame(closes).sort_index()
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    w_eng = held.div(n_listed, axis=0).fillna(0.0)

    eng_pct, eng_net, eng_eq, eng_gross, eng_dd = _compound(w_eng, R, MAKER_RT)
    hodl_pct, hodl_eq, hodl_dd, n_total = _true_hodl(cp, R, listed, MAKER_RT)

    idx = cp.index
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(idx, eng_eq, color="tab:blue", lw=1.6,
            label=f"DECOUPLED engine  (compound {eng_pct:+.0f}%, maxDD {eng_dd:.0f}%)", zorder=3)
    ax.plot(idx, hodl_eq, color="0.45", lw=1.4, ls="--",
            label=f"TRUE buy-and-hold  (compound {hodl_pct:+.0f}%, maxDD {hodl_dd:.0f}%)", zorder=2)
    ax.set_yscale("log")
    # annotate each maxDD at the trough
    for eq, dd, color, name in ((eng_eq, eng_dd, "tab:blue", "engine"), (hodl_eq, hodl_dd, "0.35", "HODL")):
        if len(eq):
            peak = np.maximum.accumulate(eq)
            tri = int(np.argmin((eq - peak) / peak))
            ax.scatter(idx[tri], eq[tri], color=color, marker="o", s=40, zorder=5)
            ax.annotate(f"{name} maxDD {dd:.0f}%", (idx[tri], eq[tri]), textcoords="offset points",
                        xytext=(6, -14), fontsize=9, color=color, fontweight="bold")
    ax.set_title(f"{ma_type}|{tf} DECOUPLED engine equity vs TRUE buy-and-hold  ("
                 f"u100, {len(loaded)} assets, 2020->{fl.DEV_END}, lag-1, maker {MAKER_RT})", fontsize=11)
    ax.set_ylabel("equity (start = 1.0, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"engine_compound_pct": round(eng_pct, 2), "hodl_compound_pct": round(hodl_pct, 2),
            "engine_maxdd_pct": round(eng_dd, 1), "hodl_maxdd_pct": round(hodl_dd, 1),
            "n_assets": len(loaded), "png": str(path)}


def _pick_charts(loaded: list[str]) -> list[str]:
    out = []
    lset = set(loaded)
    for s in CHART_PREF:
        if s in lset:
            out.append(s)
    # backfill to ~6 with the most-traded other liquid names if any preferred are absent
    if len(out) < 6:
        for s in loaded:
            if s not in out:
                out.append(s)
            if len(out) >= 6:
                break
    return out[:6]


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_decoupled_trade_charts")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()

    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[charts] building decoupled provider for {a.ma_type}|{a.tf} (lookback {LOOKBACK_DAYS}d / slice {SLICE_DAYS}d)")
    provider, recs, loaded, lookback_bars = _asset_panel(a.ma_type, a.tf)
    print(f"[charts] loaded {len(loaded)} u100 assets (DEV-walled < {fl.DEV_END})")

    chart_syms = _pick_charts(loaded)
    print(f"[charts] charting: {chart_syms}")

    # ---- per-asset charts + collect trades ----
    summaries = []
    pngs = []
    for sym in chart_syms:
        path = OUTDIR / f"trade_{sym}_{a.ma_type.lower()}_{a.tf}_decoupled.png"
        s = _plot_asset(sym, recs[sym], lookback_bars, path)
        summaries.append(s)
        pngs.append(s["png"])
        print(f"[charts]   {sym}: n_trades={s['n_trades']} win={s['win_rate']}% "
              f"mean_ROI={s['mean_ROI']}% total_compound={s['total_compound']}% -> {path.name}")

    # ---- spot-check: first 5 trades of one asset ----
    spot_sym = "SOLUSDT" if "SOLUSDT" in chart_syms else chart_syms[0]
    spot = next(s for s in summaries if s["sym"] == spot_sym)
    print(f"\n[spot-check] first 5 trades of {spot_sym} (entry/exit/ROI, lag-1 causal):")
    for tr in spot["trades"][:5]:
        print(f"   BUY {tr['entry_date'].date()} @ {tr['entry_px']:.4f}  ->  "
              f"SELL {tr['exit_date'].date()} @ {tr['exit_px']:.4f}  "
              f"hold {tr['hold_days']}d  ROI_net {tr['ROI_pct_net']:+.3f}%")

    # ---- per-trade CSV (one row per trade per charted asset) + summaries ----
    rows = []
    for s in summaries:
        for tr in s["trades"]:
            rows.append({
                "asset": s["sym"], "entry_date": tr["entry_date"].date(), "exit_date": tr["exit_date"].date(),
                "hold_days": tr["hold_days"], "entry_px": round(tr["entry_px"], 6),
                "exit_px": round(tr["exit_px"], 6), "ROI_pct_net": tr["ROI_pct_net"],
            })
    trades_df = pd.DataFrame(rows)
    csv_path = OUTDIR / f"trades_{a.ma_type.lower()}_{a.tf}_decoupled.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"\n[charts] wrote {len(trades_df)} trade rows -> {csv_path}")

    # per-asset summary frame + ALL-ASSETS line
    sum_rows = []
    for s in summaries:
        sum_rows.append({"asset": s["sym"], "n_trades": s["n_trades"], "win_rate": s["win_rate"],
                         "mean_ROI": s["mean_ROI"], "median_ROI": s["median_ROI"],
                         "total_compound": s["total_compound"]})
    all_rois = trades_df["ROI_pct_net"].to_numpy() if len(trades_df) else np.array([])
    # NOTE: across-asset "total_compound" is the GEOMETRIC CHAIN of all pooled trades (sequential, single $) -- it is
    # NOT a portfolio return (the portfolio equity is the separate equity chart, which sizes 1/N_listed per name).
    # Reported for completeness; the per-asset compounds and the portfolio equity chart are the real reads.
    all_compound = float((np.prod(1.0 + all_rois / 100.0) - 1.0) * 100) if len(all_rois) else 0.0
    sum_rows.append({"asset": "ALL-ASSETS(pooled-chain)", "n_trades": len(all_rois),
                     "win_rate": round(float((all_rois > 0).mean() * 100), 1) if len(all_rois) else 0.0,
                     "mean_ROI": round(float(all_rois.mean()), 3) if len(all_rois) else 0.0,
                     "median_ROI": round(float(np.median(all_rois)), 3) if len(all_rois) else 0.0,
                     "total_compound": round(all_compound, 2)})
    summary_df = pd.DataFrame(sum_rows)
    summary_csv = OUTDIR / f"trade_summary_{a.ma_type.lower()}_{a.tf}_decoupled.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ---- portfolio equity chart ----
    eq_path = OUTDIR / f"portfolio_equity_{a.ma_type.lower()}_{a.tf}_decoupled.png"
    eqm = _portfolio_equity(a.ma_type, a.tf, recs, loaded, eq_path)
    pngs.append(eq_path.as_posix())
    print(f"[charts] portfolio equity: engine {eqm['engine_compound_pct']}% (maxDD {eqm['engine_maxdd_pct']}%) "
          f"vs HODL {eqm['hodl_compound_pct']}% (maxDD {eqm['hodl_maxdd_pct']}%) over {eqm['n_assets']} assets "
          f"-> {eq_path.name}")

    # ---- markdown tables to stdout ----
    print("\n=== PER-ASSET SUMMARY (markdown) ===")
    print(summary_df.to_markdown(index=False))
    print(f"\n=== FIRST 10 TRADES of {spot_sym} (markdown) ===")
    sub = trades_df[trades_df["asset"] == spot_sym].head(10)
    print(sub.to_markdown(index=False))

    print("\n=== PNG PATHS (absolute) ===")
    for p in pngs:
        print("  ", str(Path(p).resolve()))
    print(f"\n=== CSV: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
