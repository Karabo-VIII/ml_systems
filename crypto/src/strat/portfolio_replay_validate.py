"""VALIDATE + PLOT the portfolio_replay engine -- establish its validity, with figures + a table.

User /orc (2026-06-11): "plot these figures, tabulate the results -- establish the validity of the
replay." Not just pretty curves: a VALIDATION SUITE that proves the engine is correct.

VALIDITY CHECKS (numerical, must pass):
  A. MtM self-consistency : independently recompute net = (W.shift(1)*ret).sum - turnover*(cost/2)
                            from the engine's OWN persisted weights+returns; must match the engine's
                            equity to ~1e-12 (no hidden double-count / post-hoc tampering).
  B. Cost reconciliation  : taker vs maker on the SAME book -> the log-equity gap must equal
                            sum(turnover) * (taker/2 - maker/2) independently computed (costs real).
  C. Determinism          : re-run -> identical final equity (no RNG / order drift).
BENCHMARK: each equity curve is plotted against the INDEPENDENT equal-weight buy&hold of the same
universe/window (computed from raw closes, not the engine) -- the sanity anchor + the thing the book
is judged against.

Figures: one equity-curve panel per (universe, window) with the B&H overlay + a cost-recon panel.
Table: every run tabulated (final %, ann, DD, Sharpe, turnover, vs B&H) + the validity verdicts.
No emoji (cp1252).

Run: python -m strat.portfolio_replay_validate
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from strat.portfolio_replay import run, WIN, TAKER_RT, MAKER_RT, _mask  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                       # noqa: E402
from mining.family_regime_map import _norm_sym                          # noqa: E402

OUTDIR = ROOT.parent / "runs" / "strat" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

STRATS_2_3 = "ema_50_100,ema_10_50_100"   # MA(x,y) + MA(x,y,z)


def equal_weight_bh(universe, cadence, window):
    """INDEPENDENT equal-weight buy&hold equity over the window (from raw closes, floored-aligned)."""
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    if "assets" in spec:
        syms = [a["symbol"] for a in spec["assets"]]
    else:
        u50 = yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u50.yaml"))
        syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
        syms = [s for s in dict.fromkeys(syms) if s not in set(spec.get("excluded_assets") or [])]
    freq = {"1d": "D", "4h": "4h", "1h": "h"}.get(cadence, "D")
    closes = {}
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence, features=["close"])
        except Exception:
            continue
        idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor(freq)
        s = pd.Series(df["close"].to_numpy().astype(float), index=idx)
        closes[sym] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(closes).sort_index()
    ret = panel.pct_change()
    ew = ret.mean(axis=1)                                  # equal-weight basket daily return
    ms = np.array([int(t.value // 10**6) for t in panel.index])
    m = _mask(ms, window)
    ewn = ew[m].fillna(0.0)
    eq = (1 + ewn).cumprod()
    return eq, ewn.index


def metrics(eq, cadence):
    eqa = np.asarray(eq)
    if len(eqa) < 2:
        return {}
    ann = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}.get(cadence, 365)
    nyr = len(eqa) / ann
    dd = float(((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa)).min() * 100)
    return {"final_pct": float((eqa[-1] - 1) * 100), "ann_pct": round(float((eqa[-1] ** (1 / nyr) - 1) * 100), 1) if eqa[-1] > 0 else -100.0,
            "maxdd_pct": round(dd, 1)}


def main():
    configs = [
        ("u10", "CUSTOM", ("2022-01-01", "2022-02-01"), "Jan 2022 (u10)"),
        ("u50", "CUSTOM", ("2022-01-01", "2022-02-01"), "Jan 2022 (u50)"),
        ("u10", "ALL", None, "Full cycle 2020-25 (u10)"),
        ("u10", "UNSEEN", None, "UNSEEN bear 2026 (u10)"),
    ]
    rows = []
    panels = []
    for uni, win, custom, title in configs:
        if custom:
            WIN["CUSTOM"] = custom
        r = run(uni, "1d", STRATS_2_3.split(","), win, TAKER_RT, False, 0.02, 0.15)
        if "error" in r:
            print(f"[{title}] {r['error']}"); continue
        dates = pd.to_datetime(r["_dates"])
        eq = np.array(r["_equity"])
        # independent buy&hold benchmark
        bh_eq, bh_idx = equal_weight_bh(uni, "1d", win)
        bh_m = metrics(bh_eq.to_numpy(), "1d")
        # VALIDITY A: MtM self-consistency -- recompute net from the FULL weights+returns
        # (so the boundary lag is correct), then mask to the window and compare to engine _net.
        W = r["_W_full"]; retp = r["_ret_full"]; wmask = np.array(r["_wmask"])
        Wl = W.shift(1).fillna(0.0)
        gross = (Wl * retp).sum(axis=1)
        turn = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
        net_all_recon = (gross - turn * (TAKER_RT / 2)).to_numpy()
        net_recon = net_all_recon[wmask]
        eng_net = np.array(r["_net"])
        max_diff = float(np.max(np.abs(net_recon - eng_net))) if len(eng_net) and len(net_recon) == len(eng_net) else 9.9
        rows.append({
            "scenario": title, "n_bars": r["n_bars"], "book_final%": round(r["final_pct"], 1),
            "book_ann%": r["ann_pct"], "book_DD%": r["maxdd_pct"], "book_Sharpe": r["sharpe"],
            "turnover": r["avg_daily_turnover"], "BH_final%": round(bh_m.get("final_pct", 0), 1),
            "BH_DD%": bh_m.get("maxdd_pct"), "MtM_recon_maxdiff": f"{max_diff:.1e}",
        })
        panels.append((title, dates, eq, bh_eq))
        print(f"[{title}] book {r['final_pct']:+.1f}% vs B&H {bh_m.get('final_pct',0):+.1f}% | "
              f"MtM recon max-diff {max_diff:.1e}")

    # VALIDITY B: cost reconciliation (ALL u10 taker vs maker)
    rt = run("u10", "1d", STRATS_2_3.split(","), "ALL", TAKER_RT, False, 0.02, 0.15)
    rm = run("u10", "1d", STRATS_2_3.split(","), "ALL", MAKER_RT, False, 0.02, 0.15)
    sum_turn = float(np.sum(rt["_turnover"]))
    expected_cost_gap = sum_turn * (TAKER_RT / 2 - MAKER_RT / 2)
    actual_logdiff = float(np.log(rm["final_equity"]) - np.log(rt["final_equity"]))   # maker - taker (>0)
    cost_ok = abs(actual_logdiff - expected_cost_gap) < 0.02 * abs(expected_cost_gap) + 1e-6
    # VALIDITY C: determinism
    rt2 = run("u10", "1d", STRATS_2_3.split(","), "ALL", TAKER_RT, False, 0.02, 0.15)
    determ_ok = abs(rt2["final_equity"] - rt["final_equity"]) < 1e-9
    mtm_ok = all(float(x["MtM_recon_maxdiff"]) < 1e-9 for x in rows)

    # ---- FIGURE ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for ax, (title, dates, eq, bh_eq) in zip(axes, panels):
        ax.plot(dates, eq, lw=1.6, color="tab:blue", label="replay book (2MA+3MA)")
        ax.plot(bh_eq.index, bh_eq.to_numpy(), lw=1.2, ls="--", color="gray", label="equal-wt buy&hold")
        ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
        ax.set_title(title, fontsize=10); ax.set_ylabel("equity ($1 ->)"); ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        if eq.max() / max(eq.min(), 1e-6) > 30:
            ax.set_yscale("log")
    # cost-recon panel
    axc = axes[4]
    axc.plot(np.cumsum(rt["_turnover"]) * (TAKER_RT / 2) * 100, label="taker cumulative cost %", color="tab:red")
    axc.plot(np.cumsum(rm["_turnover"]) * (MAKER_RT / 2) * 100, label="maker cumulative cost %", color="tab:green")
    axc.set_title(f"Cost reconciliation (ALL u10)\nexpected gap {expected_cost_gap:.4f} vs actual {actual_logdiff:.4f} -> "
                  f"{'PASS' if cost_ok else 'FAIL'}", fontsize=9)
    axc.set_ylabel("cumulative cost (%)"); axc.legend(fontsize=8); axc.grid(alpha=0.25)
    # validity text panel
    axv = axes[5]; axv.axis("off")
    txt = ("VALIDITY VERDICT\n\n"
           f"A. MtM self-consistency : {'PASS' if mtm_ok else 'FAIL'}\n"
           f"   (recompute net from weights+returns;\n    max diff < 1e-9 across all scenarios)\n\n"
           f"B. Cost reconciliation  : {'PASS' if cost_ok else 'FAIL'}\n"
           f"   (taker-maker gap = turnover x rate-delta)\n\n"
           f"C. Determinism          : {'PASS' if determ_ok else 'FAIL'}\n"
           f"   (re-run -> identical equity)\n\n"
           f"Engine: lagged weights x next-bar ret\n - turnover x (cost/2)  [MtM, no double-count]")
    axv.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle("Portfolio Replay Engine -- VALIDITY (book vs independent buy&hold + numerical checks)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = OUTDIR / f"portfolio_replay_validity_{stamp}.png"
    fig.savefig(fpath, dpi=110); plt.close(fig)

    # ---- TABLE ----
    print("\n## RESULTS TABLE (2MA ema_50_100 + 3MA ema_10_50_100, taker, inverse-vol)")
    hdr = f"| {'scenario':28} | {'bars':>4} | {'book%':>8} | {'ann%':>7} | {'DD%':>7} | {'Sh':>5} | {'turn':>6} | {'B&H%':>8} | {'MtM diff':>9} |"
    print(hdr); print("|" + "-" * (len(hdr) - 2) + "|")
    for x in rows:
        print(f"| {x['scenario']:28} | {x['n_bars']:>4} | {x['book_final%']:>+8.1f} | {x['book_ann%']:>+7.1f} | "
              f"{x['book_DD%']:>7.1f} | {x['book_Sharpe']:>5} | {x['turnover']:>6} | {x['BH_final%']:>+8.1f} | {x['MtM_recon_maxdiff']:>9} |")
    print(f"\nVALIDITY: MtM self-consistency {'PASS' if mtm_ok else 'FAIL'} | "
          f"Cost reconciliation {'PASS' if cost_ok else 'FAIL'} (expected {expected_cost_gap:.4f} vs actual {actual_logdiff:.4f}) | "
          f"Determinism {'PASS' if determ_ok else 'FAIL'}")
    out = {"rows": rows, "validity": {"mtm_self_consistency": mtm_ok, "cost_reconciliation": cost_ok,
                                      "cost_expected_gap": expected_cost_gap, "cost_actual_gap": actual_logdiff,
                                      "determinism": determ_ok}, "figure": str(fpath)}
    jpath = ROOT.parent / "runs" / "strat" / f"portfolio_replay_validity_{stamp}.json"
    json.dump(out, open(jpath, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[figure]  {fpath}\n[json]    {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
