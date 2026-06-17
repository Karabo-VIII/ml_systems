"""src/strat/ma_tf_trend_book.py -- the canonical MA x TF TREND SLEEVE book over u50.

WHAT THIS IS (construction node, 2026-06-13): the honest BOOK-LEVEL baseline for the ACTIVE
multi-lookback crossover ENTRY sleeve across the u50 universe -- the COMPLEMENT to daily_engine.py's
vol-target BUY-HOLD core. daily_engine is the always-in beta core; THIS is the active trend ENTRY
sleeve (in only while the crossover says trend), pooled across all 48 present u50 names into ONE
daily net-return book that the canonical scorecard grades.

REUSE (no reinvention): the signal/sizing/replay engine IS strat.portfolio_replay.run() -- it already
does per-asset EMA/Donchian crossover holding-states, inverse-vol / vol-target sizing with gross +
per-name caps, lag-1 causal MtM, taker/maker cost, and pools the universe into one daily net series
(its _net/_dates). We add: (1) a PRE-REGISTERED config set, (2) the scorecard.score_book grade across
SEL/OOS/UNSEEN, (3) trade-level BREADTH + CONCENTRATION (the "across 50 it hits somewhere" question),
(4) persistence so the next build cycle reuses the book net series + card.

PRE-REGISTERED config set (causal, lag-1, real TAKER_RT cost, vol-target sizing):
    EMA(20/50), EMA(50/100), EMA(50/200), Donchian(20/10)   [+ optional SMA(50/200)]
The book holds an asset while ANY config in the set is long (the union "trend on" state in
portfolio_replay), inverse-vol-sized, gross-capped at 1.0, per-name-capped.

FRAME (construction, per CAMPAIGN_CHARTER_2026_06_10): the UNIT graded is the BOOK (one pooled equity
curve), not the per-asset config. Beta/trend-premium is a PRODUCT, not a kill. We do NOT eliminate for
"not UNSEEN-positive in a sustained bear" -- we grade book-level capture + risk envelope + the
rolling-ROI soft-benchmark (REPORTED, never an eliminator), then ENUMERATE weaknesses as solve-targets.

RWYB:
  python -m strat.ma_tf_trend_book                          # default: u50, the pre-registered set, taker
  python -m strat.ma_tf_trend_book --configs ema_50_100,donch20
  python -m strat.ma_tf_trend_book --maker

No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from strat.portfolio_replay import run, holding_state, TAKER_RT, MAKER_RT          # noqa: E402
from strat.portfolio_replay_per_asset import per_asset_trades                      # noqa: E402
from strat.scorecard import score_book, SPLITS                                     # noqa: E402
from strat.battery import herfindahl_neff                                          # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                                  # noqa: E402
from mining.family_regime_map import _norm_sym                                     # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# PRE-REGISTERED multi-lookback crossover config set (names resolve in portfolio_replay.STRATS).
# A small, fixed set -- 2 EMA pairs + 1 long EMA pair + 1 Donchian breakout. Causal by construction.
DEFAULT_CONFIGS = ["ema_20_50", "ema_50_100", "ema_50_200", "donch20"]
# ema_20_50 is not in the base STRATS table; register it (EMA 20/50 cross, same 2MA family).
import strat.portfolio_replay as PR                                               # noqa: E402
PR.STRATS.setdefault("ema_20_50", ("2MA", dict(type="EMA", fast=20, slow=50)))


def _split_of(ts: pd.Timestamp) -> str:
    for sp, (lo, hi) in SPLITS.items():
        if pd.Timestamp(lo) <= ts < pd.Timestamp(hi):
            return sp
    return "PRE"   # before the earliest split lo (shouldn't happen given SEL starts 2018)


def build_book(universe="u50", configs=None, cost_rt=TAKER_RT, vol_target=0.02, max_per_name=0.15):
    """Build the pooled book daily net Series + the per-asset trade list (for breadth/concentration).
    Returns (net_series, meta_dict, trades_list). The net series is the FULL-history pooled book."""
    configs = configs or DEFAULT_CONFIGS
    r = run(universe, "1d", configs, "ALL", cost_rt, use_spine=False,
            vol_target=vol_target, max_per_name=max_per_name)
    if "error" in r:
        raise RuntimeError(r["error"])
    net = pd.Series(r["_net"], index=pd.to_datetime(r["_dates"])).sort_index()
    meta = {k: v for k, v in r.items() if not k.startswith("_")}

    # --- per-asset round-trip trades for breadth + concentration (the "hits somewhere" claim) ---
    # The book's per-asset holding = union of the configs (ANY long). Reconstruct that union state
    # per asset, extract round-trip trades, tag each trade's split by ENTRY date.
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    syms = [a["symbol"] for a in spec["assets"]]
    trades = []
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence="1d",
                                      features=["open", "high", "low", "close"])
        except Exception:
            continue
        idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor("D")
        # dedup to the aligned daily grid (same contract as the engine)
        keep = ~pd.Index(idx).duplicated(keep="last")
        o = df["open"].to_numpy().astype(float)[keep]
        h = df["high"].to_numpy().astype(float)[keep]
        l = df["low"].to_numpy().astype(float)[keep]
        c = df["close"].to_numpy().astype(float)[keep]
        ms = (df["timestamp"].to_numpy())[keep]
        if len(c) < 60:
            continue
        union = np.zeros(len(c), dtype=np.int8)
        for nm in configs:
            union = ((union + holding_state(nm, o, h, l, c)) > 0).astype(np.int8)
        for t in per_asset_trades(o, c, union, ms, cost_rt):
            ts = pd.Timestamp(int(t["entry_ms"]), unit="ms")
            trades.append({"net": float(t["ret"]), "split": _split_of(ts), "sym": sym,
                           "ts": str(ts)[:10], "hold": int(t["hold"])})
    return net, meta, trades


def concentration_breadth(trades):
    """Trade-level concentration + breadth, per split AND full. The 'across 50 it hits somewhere'
    claim is a breadth question; the 'few winners carry it' risk is a concentration question.

    NB: trades within an asset OVERLAP / are interleaved across assets in calendar time, so a
    cross-trade compound PRODUCT is meaningless (the book-level scorecard equity curve is the
    correct compound). Concentration here uses the scorecard convention: mean per-trade net +- and
    the drop-top-5%-by-|net| MEAN (the 'do a few outliers carry the average?' test), n_eff
    (Herfindahl on |net|), and breadth = assets whose mean (or summed) per-trade net is positive."""
    out = {}
    for scope in list(SPLITS) + ["ALL"]:
        tr = trades if scope == "ALL" else [t for t in trades if t["split"] == scope]
        if len(tr) < 5:
            out[scope] = {"n_trades": len(tr)}
            continue
        nets = np.array([t["net"] for t in tr], float)
        per_asset = {}
        for t in tr:
            per_asset.setdefault(t["sym"], []).append(t["net"])
        asset_mean = {s: float(np.mean(v)) for s, v in per_asset.items()}
        asset_sum = {s: float(np.sum(v)) for s, v in per_asset.items()}      # naive sum (attribution)
        breadth_pos = sum(1 for v in asset_mean.values() if v > 0)
        breadth_tot = len(asset_mean)
        # concentration on the MEAN: drop the top 5% of trades by |net|, recompute the mean
        k = max(1, int(round(len(nets) * 0.05)))
        order = np.argsort(np.abs(nets))
        kept = nets[order[:-k]]
        mean_full = float(nets.mean() * 100)
        mean_drop5 = float(kept.mean() * 100) if len(kept) else None
        out[scope] = {
            "n_trades": len(tr),
            "breadth_pos_assets": breadth_pos, "breadth_tot_assets": breadth_tot,
            "breadth_frac": round(breadth_pos / breadth_tot, 3) if breadth_tot else None,
            "n_eff_trades": round(herfindahl_neff(nets), 1),
            "mean_net_pct": round(mean_full, 3),
            "se_net_pct": round(float(nets.std() / np.sqrt(len(nets)) * 100), 3),
            "mean_drop_top5pct": round(mean_drop5, 3) if mean_drop5 is not None else None,
            "drop5_flips_sign": bool(mean_full > 0 and mean_drop5 is not None and mean_drop5 <= 0),
            "win_rate": round(float((nets > 0).mean()), 3),
            "top5_asset_sumnet": {s: round(v, 3) for s, v
                                  in sorted(asset_sum.items(), key=lambda kv: -kv[1])[:5]},
        }
    return out


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_tf_trend_book")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="comma list of crossover config names (portfolio_replay.STRATS keys)")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--vol-target", type=float, default=0.02)
    ap.add_argument("--max-per-name", type=float, default=0.15)
    a = ap.parse_args(argv)
    cost = MAKER_RT if a.maker else TAKER_RT
    cost_name = "maker" if a.maker else "taker"
    configs = [c.strip() for c in a.configs.split(",") if c.strip() in PR.STRATS]
    if not configs:
        print(f"no valid configs; choose from {list(PR.STRATS)}"); return 2

    print(f"## MA x TF TREND BOOK -- {a.universe} 1d -- configs={configs} -- {cost_name} "
          f"-- vol_target={a.vol_target} max_per_name={a.max_per_name}")
    net, meta, trades = build_book(a.universe, configs, cost, a.vol_target, a.max_per_name)
    print(f"   pooled book: {len(net)} days, {meta['n_assets']} assets, {len(trades)} per-asset trades")
    print(f"   engine full-window: final {meta['final_pct']:+.1f}% | ann {meta['ann_pct']:+.1f}% | "
          f"maxDD {meta['maxdd_pct']:.1f}% | Sharpe {meta['sharpe']} | avg_gross {meta['avg_gross']}")

    card = score_book(f"ma_tf_trend_book_{a.universe}", net)
    print("\n   --- SCORECARD (book-level, SEL/OOS/UNSEEN) ---")
    print(f"   {'split':7} {'n':>4} {'compound%':>10} {'ann%':>8} {'maxDD%':>8} {'Sharpe':>7} "
          f"{'1dROI_med%':>10} {'1d_%pos':>8} {'3dROI_med%':>10} {'3d_%pos':>8}")
    for sp in ("SEL", "OOS", "UNSEEN"):
        m = card["per_split"].get(sp, {})
        if "compound_pct" not in m:
            print(f"   {sp:7} {m.get('n', 0):>4} (too short)"); continue
        sb = m["softbench_roi"]
        print(f"   {sp:7} {m['n']:>4} {m['compound_pct']:>10} {m['ann_pct']:>8} {m['maxdd_pct']:>8} "
              f"{m['sharpe']:>7} {str(sb['1d']['median_pct']):>10} {str(sb['1d']['frac_positive']):>8} "
              f"{str(sb['3d']['median_pct']):>10} {str(sb['3d']['frac_positive']):>8}")
    fb = card["full_block_bootstrap"]; hb = card.get("heldout_block_bootstrap", {})
    print(f"   full-cycle block-bootstrap compound p05/p50/p95: {fb.get('p05')}/{fb.get('p50')}/{fb.get('p95')}")
    print(f"   held-out (OOS+UNSEEN) block-bootstrap p05/p50/p95: "
          f"{hb.get('p05')}/{hb.get('p50')}/{hb.get('p95')}")
    print(f"   ship_read (UNSEEN+p05 gate -- ALPHA test, NOT the construction verdict): {card['ship_read']}")

    cb = concentration_breadth(trades)
    print("\n   --- BREADTH + CONCENTRATION (the 'across 50 it hits somewhere' question; per-trade) ---")
    print(f"   {'scope':7} {'nTr':>5} {'breadth+/tot':>13} {'frac':>6} {'n_eff':>7} "
          f"{'meanNet%':>9} {'+-SE':>7} {'drop-top5%':>11} {'win':>5}")
    for scope in ("SEL", "OOS", "UNSEEN", "ALL"):
        m = cb.get(scope, {})
        if "breadth_pos_assets" not in m:
            print(f"   {scope:7} {m.get('n_trades', 0):>5} (too short)"); continue
        print(f"   {scope:7} {m['n_trades']:>5} "
              f"{str(m['breadth_pos_assets'])+'/'+str(m['breadth_tot_assets']):>13} "
              f"{str(m['breadth_frac']):>6} {m['n_eff_trades']:>7} {m['mean_net_pct']:>9} "
              f"{m['se_net_pct']:>7} {str(m['mean_drop_top5pct']):>11} {m['win_rate']:>5}")

    # persist book net series + card + breadth so the next cycle reuses it
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"ma_tf_trend_book_{a.universe}_{stamp}.json"
    out = {
        "repro": {"command": "python -m strat.ma_tf_trend_book " + " ".join(argv or sys.argv[1:]),
                  "git_sha": _git_sha(), "cost_rt": cost, "cost_name": cost_name,
                  "universe": a.universe, "configs": configs, "cadence": "1d",
                  "vol_target": a.vol_target, "max_per_name": a.max_per_name,
                  "splits": {k: list(v) for k, v in SPLITS.items()}},
        "engine_meta": meta,
        "scorecard": card,
        "breadth_concentration": cb,
        "book_net_series": {str(d)[:10]: round(float(v), 8) for d, v in net.items()},
    }
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n   [persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
