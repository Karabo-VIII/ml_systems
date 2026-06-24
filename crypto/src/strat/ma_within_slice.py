"""src/strat/ma_within_slice.py -- the WITHIN-SLICE view of the committed MA drawdown-insurance engine.

WHY (2026-06-20 user directive). The committed 1st-success (efb23a4b) reports a 2020->2024 AGGREGATE compound.
The user wants the literal "trade ANY slice across u100" view: take ONE 7-day calendar week and report what the
committed EMA|1d engine DID and MADE inside that week across the u100 universe -- which names it was long, the
in-week entry/exit dates, per-asset slice ROI (net maker), portfolio slice ROI vs buy-hold-ALL vs all-cash, and
how many of that week's >=5% up-movers it captured.

REUSES THE COMMITTED ENGINE (no fresh backtest): ma_wealth_driver._load_asset (same DEV-walled u100 panel, 2h
synth recipe), ma_movecatch_engine.{select_band_trainval, make_hold_provider} (the committed fixed-EW ma_cross
band ENSEMBLE hold provider, TRAIN+VAL-selected, OOS never read), and the SAME fixed-N EW + lag-1 + maker-cost
MtM accounting as ma_wealth_driver._compound -- so a within-slice portfolio ROI reconciles to the committed engine.

ENGINE = EMA, tf=1d, long-while-band-holds / cash-otherwise; band selected on 2020 TRAIN+VAL; DEV wall < 2024-05-15.
COST = maker 0.0006 RT (same leg as the committed wealth driver). No emoji (cp1252). No git commit.

RWYB: python -m strat.ma_within_slice --ma-type EMA --tf 1d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                       # noqa: E402
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT  # noqa: E402

OUT_DIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"
CHART_DIR = OUT_DIR / "charts" / "slices"

# The 3 representative weeks (all < DEV_END 2024-05-15). Dates are the 7-day window [start, end) inclusive of the
# 7 calendar days start..start+6. Picked for canonical regime archetypes; rationale stated per slice in the output.
SLICES = [
    {"id": "BULL_2021-01",   "start": "2021-01-02", "days": 7,
     "why": "strong-bull alt-ramp week: BTC broke $30k->$40k early Jan-2021, alts ripped. Canonical bull."},
    {"id": "BEAR_2022-05",   "start": "2022-05-09", "days": 7,
     "why": "BEAR-SELLOFF: the LUNA/UST de-peg collapse week (2022-05-09..05-15). Canonical crypto crash."},
    {"id": "CHOP_2023-08",   "start": "2023-08-07", "days": 7,
     "why": "CHOP/sideways: mid-Aug-2023 low-vol range, BTC pinned ~$29-30k, no trend. Canonical chop."},
]


def _build_panel(ma_type: str, tf: str):
    """Load the full DEV-walled u100 panel ONCE with the committed engine's hold provider. Returns
    (cp close-panel, hp hold-panel, n_total_admitted). cp/hp are aligned DataFrames indexed by date."""
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    provider = make_hold_provider(sub_band, ma_type)
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    closes, holds = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]
    cp = pd.DataFrame(closes).sort_index()
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    return cp, hp, len(sel_meta), sub_band, sel_meta


def _slice_window(cp: pd.DataFrame, hp: pd.DataFrame, start: str, days: int):
    """Restrict to the [start, start+days) 7-day calendar window. Keep ONE extra pre-bar so we have the entry
    reference price (the close the day BEFORE the window starts) for an asset already held at slice start.
    Returns (cp_w, hp_w, pre_close_row) where cp_w/hp_w are the in-window bars."""
    t0 = pd.Timestamp(start)
    t1 = t0 + pd.Timedelta(days=days)
    in_win = (cp.index >= t0) & (cp.index < t1)
    idx = np.where(in_win)[0]
    if len(idx) == 0:
        return None
    first_i = idx[0]
    pre_close = cp.iloc[first_i - 1] if first_i > 0 else None       # the bar just before the window
    return cp.loc[in_win], hp.loc[in_win], pre_close


def analyze_slice(cp: pd.DataFrame, hp: pd.DataFrame, sl: dict, cost_rt: float):
    """Full within-slice analysis for one 7-day window. Returns a dict deliverable."""
    out = {"id": sl["id"], "start": sl["start"], "days": sl["days"], "why": sl["why"]}
    res = _slice_window(cp, hp, sl["start"], sl["days"])
    if res is None:
        out["error"] = "no bars in window"; return out, None
    cp_w, hp_w, pre_close = res
    dates = cp_w.index
    out["window_dates"] = [str(dates.min().date()), str(dates.max().date())]
    out["n_bars"] = int(len(dates))

    # ----- universe LIVE this week: a name is live if it has any finite close in the window -----
    live = cp_w.columns[cp_w.notna().any(axis=0)].tolist()
    n_universe = len(live)
    out["n_universe"] = n_universe

    # ----- AVAILABLE MOVES: weekly return per live name = last finite / first finite - 1 over the window -----
    weekly_ret = {}
    for s in live:
        col = cp_w[s].dropna()
        if len(col) >= 2:
            weekly_ret[s] = float(col.iloc[-1] / col.iloc[0] - 1.0)
    wr = pd.Series(weekly_ret).sort_values(ascending=False)
    out["movers_ge2pct_up"] = int((wr >= 0.02).sum())
    out["movers_ge5pct_up"] = int((wr >= 0.05).sum())
    out["movers_ge10pct_up"] = int((wr >= 0.10).sum())
    out["top5_movers"] = [{"sym": s, "weekly_ret_pct": round(wr[s] * 100, 2)} for s in wr.head(5).index]

    # ----- ENGINE ACTIONS: which names the engine was long during the week + in-week entry/exit -----
    # hold-state per bar (the committed provider's int8 holds). held at slice start = held on bar0.
    # entry mid-week = 0->1 transition inside the window; exit mid-week = 1->0 inside the window.
    held_any = []
    actions = {}
    for s in live:
        hcol = hp_w[s].fillna(0.0).to_numpy()
        ccol = cp_w[s]
        if hcol.sum() == 0:
            continue
        held_any.append(s)
        # detect transitions within window (compare to prior bar; bar0's prior = the state implied by pre-window)
        # held_at_start = state on bar0
        held_at_start = bool(hcol[0] > 0.5)
        # entries/exits inside window
        entries, exits = [], []
        for i in range(1, len(hcol)):
            if hcol[i] > 0.5 and hcol[i - 1] <= 0.5:
                entries.append(str(dates[i].date()))
            if hcol[i] <= 0.5 and hcol[i - 1] > 0.5:
                exits.append(str(dates[i].date()))
        actions[s] = {
            "held_at_start": held_at_start,
            "entries_in_week": entries,
            "exits_in_week": exits,
            "bars_held": int(hcol.sum()),
        }
    out["n_held"] = len(held_any)
    out["held_names"] = held_any

    # ----- PER-ASSET slice ROI for each held position (lag-1, maker cost, MtM) -----
    # Reuse the committed accounting: weight = held-state (1 while long), lag 1 bar, charge maker on turnover.
    # Per-asset ROI = compound of (W_{t-1} * r_t) - turnover*cost/2 over the window for THAT name alone.
    per_asset = {}
    for s in held_any:
        col = cp_w[s]
        r = col.pct_change(fill_method=None).fillna(0.0).to_numpy()
        w = hp_w[s].fillna(0.0).to_numpy()
        wl = np.concatenate([[0.0], w[:-1]])                        # lag-1
        turn = np.abs(np.diff(np.concatenate([[0.0], w]))) [:len(w)]
        net = wl * r - turn * (cost_rt / 2.0)
        roi = float(np.prod(1.0 + net) - 1.0)
        per_asset[s] = {
            "slice_roi_pct": round(roi * 100, 3),
            "weekly_ret_pct": round(weekly_ret.get(s, float("nan")) * 100, 2),
            **actions[s],
        }
    out["per_asset"] = per_asset

    # ----- PORTFOLIO slice ROI: fixed-N EW across HELD names (same as committed engine leg) -----
    # engine: 1/N_listed per HELD name each bar; lag-1; maker cost on turnover; MtM-no-double-count.
    R = cp_w.pct_change(fill_method=None).fillna(0.0)
    listed = cp_w.notna().astype(float)
    held = hp_w.reindex(columns=cp_w.columns).fillna(0.0) * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    w_eng = held.div(n_listed, axis=0).fillna(0.0)
    eng_pct = _book_compound(w_eng, R, cost_rt)

    # buy-hold-ALL-u100 this week = fixed-N EW LONG every live name (no timing), same lag-1+cost.
    w_bh = listed.div(n_listed, axis=0).fillna(0.0)
    bh_pct = _book_compound(w_bh, R, cost_rt)

    out["portfolio_engine_roi_pct"] = round(eng_pct * 100, 3)
    out["buyhold_all_roi_pct"] = round(bh_pct * 100, 3)
    out["allcash_roi_pct"] = 0.0
    out["engine_avg_gross_exposure"] = round(float(w_eng.sum(axis=1).mean()), 3)

    # ----- CAPTURE: of names that moved up >=5% this week, how many was the engine long for? -----
    up5 = set(wr[wr >= 0.05].index)
    held_set = set(held_any)
    captured = sorted(up5 & held_set)
    missed = sorted(up5 - held_set)
    out["up5_movers"] = sorted(up5)
    out["up5_captured"] = captured
    out["up5_missed"] = missed
    out["up5_capture_rate"] = round(len(captured) / max(1, len(up5)), 3) if up5 else None

    return out, (cp_w, hp_w, held_any, per_asset, wr)


def _book_compound(W: pd.DataFrame, R: pd.DataFrame, cost_rt: float) -> float:
    """MtM-no-double-count compound of a weight book over the window (mirrors ma_wealth_driver._compound)."""
    Wl = W.shift(1).fillna(0.0)
    gross = (Wl * R).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = (gross - turn * (cost_rt / 2.0)).to_numpy()
    eq = np.cumprod(1.0 + net)
    return float(eq[-1] - 1.0) if len(eq) else 0.0


def _chart(sl_out: dict, extra, path: Path):
    """Horizontal bar of per-asset slice-ROI for HELD names, colored win/loss; title shows portfolio vs hold vs cash."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cp_w, hp_w, held_any, per_asset, wr = extra
    if not held_any:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, f"{sl_out['id']}: engine held 0 names (all cash)\n"
                          f"engine {sl_out['portfolio_engine_roi_pct']}% vs hold-all "
                          f"{sl_out['buyhold_all_roi_pct']}% vs cash 0.0%",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
        return
    rois = pd.Series({s: per_asset[s]["slice_roi_pct"] for s in held_any}).sort_values()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in rois.values]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(rois) + 1.5)))
    ax.barh(range(len(rois)), rois.values, color=colors)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois.index, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-asset slice ROI % (net maker, lag-1)")
    ax.set_title(f"{sl_out['id']}  ({sl_out['window_dates'][0]}..{sl_out['window_dates'][1]})  "
                 f"n_held={sl_out['n_held']}/{sl_out['n_universe']}\n"
                 f"engine {sl_out['portfolio_engine_roi_pct']}%  vs hold-all "
                 f"{sl_out['buyhold_all_roi_pct']}%  vs cash 0.0%   |  "
                 f">=5% up-movers captured {len(sl_out['up5_captured'])}/{len(sl_out['up5_movers'])}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_within_slice")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    cost = MAKER_RT
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[within-slice] building DEV-walled u100 panel for {a.ma_type}|{a.tf} (committed engine) ...")
    cp, hp, n_band, sub_band, sel_meta = _build_panel(a.ma_type, a.tf)
    print(f"[within-slice] panel: {cp.shape[1]} assets x {cp.shape[0]} bars  "
          f"({cp.index.min().date()}..{cp.index.max().date()})  band={len(sub_band)} members")

    results, csv_rows = [], []
    for sl in SLICES:
        sl_out, extra = analyze_slice(cp, hp, sl, cost)
        results.append(sl_out)
        if extra is None:
            print(f"[{sl['id']}] ERROR: {sl_out.get('error')}")
            continue
        chart_path = CHART_DIR / f"slice_{sl['id']}.png"
        _chart(sl_out, extra, chart_path)
        sl_out["chart_path"] = str(chart_path)
        print(f"\n=== {sl_out['id']}  {sl_out['window_dates'][0]}..{sl_out['window_dates'][1]} ===")
        print(f"  n_held/n_universe = {sl_out['n_held']}/{sl_out['n_universe']}   "
              f"avg_gross_exposure={sl_out['engine_avg_gross_exposure']}")
        print(f"  PORTFOLIO engine {sl_out['portfolio_engine_roi_pct']}%  vs hold-all "
              f"{sl_out['buyhold_all_roi_pct']}%  vs cash 0.0%")
        print(f"  movers up >=2%/{sl_out['movers_ge2pct_up']}  >=5%/{sl_out['movers_ge5pct_up']}  "
              f">=10%/{sl_out['movers_ge10pct_up']}")
        print(f"  top5 movers: {[(m['sym'], m['weekly_ret_pct']) for m in sl_out['top5_movers']]}")
        print(f"  >=5% up-movers CAPTURED {len(sl_out['up5_captured'])}/{len(sl_out['up5_movers'])}  "
              f"captured={sl_out['up5_captured']}  missed={sl_out['up5_missed']}")
        print(f"  chart -> {chart_path}")
        for s, pa in sl_out["per_asset"].items():
            csv_rows.append({
                "slice": sl_out["id"], "window_start": sl_out["window_dates"][0],
                "window_end": sl_out["window_dates"][1], "sym": s,
                "slice_roi_pct": pa["slice_roi_pct"], "weekly_ret_pct": pa["weekly_ret_pct"],
                "held_at_start": pa["held_at_start"], "entries_in_week": ";".join(pa["entries_in_week"]),
                "exits_in_week": ";".join(pa["exits_in_week"]), "bars_held": pa["bars_held"],
                "captured_up5": s in set(sl_out["up5_captured"]),
            })

    csv_path = OUT_DIR / "slices_within_window.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    json_path = OUT_DIR / "slices_within_window.json"
    json.dump({"engine": f"{a.ma_type}|{a.tf}", "cost_rt": cost, "cost_kind": "maker",
               "dev_wall": str(fl.DEV_END), "band_members": len(sub_band),
               "panel_assets": int(cp.shape[1]), "panel_bars": int(cp.shape[0]),
               "slices": results}, open(json_path, "w"), indent=1, default=str)
    print(f"\n[within-slice] CSV  -> {csv_path}")
    print(f"[within-slice] JSON -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
