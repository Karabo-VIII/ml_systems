"""experiments/adaptive_ma/expert/run_u100.py -- honest adaptive-MA backtest over u100 (1d, taker 0.0024).

Pipeline per asset:
  ChimeraLoader.load(sym,'1d') -> causal features (vol / Kaufman-ER trend / xs-dispersion) ->
  deterministic feature->MA-config map -> adaptive_fast/slow columns -> CanonicalHarness
  (LONG-ONLY, next-bar-open fill, ONE uniform exit = opposite-cross, src/strat taker cost 0.0024).

THREE-WAY honest comparison (the brief's bar -- adaptation must EARN its keep):
  (A) ADAPTIVE   : config adapts per-bar from causal features.
  (B) FIXED      : a single fixed-config MA baseline (10/30 SMA) + every constituent config of the map.
  (C) NULL       : cost-matched random-entry firewall (src/strat/firewall.py) -- beta-in-disguise test.

Splits TRAIN/VAL/OOS/UNSEEN via the harness WindowSpec; UNSEEN + OOS are held-out (never used to pick
anything -- the config map + constants were fixed up-front). Reports per-trade win-rate + held-out
compound, pooled and per-asset, with a sign test for adaptive>fixed. Honest about nulls.

RWYB:  python experiments/adaptive_ma/expert/run_u100.py [--quick] [--firewall]
No emoji (cp1252). numpy/pandas only + the kept apparatus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_ma as A  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec  # noqa: E402
from strat.firewall import random_entry_null  # noqa: E402

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
FIXED_FAST, FIXED_SLOW, FIXED_TYPE = 10, 30, "sma"   # canonical fixed baseline (chosen up-front, NOT on held-out)
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")


def _load_ohlc(loader: ChimeraLoader, sym: str) -> pd.DataFrame | None:
    try:
        g = loader.load(sym, cadence="1d")
    except Exception:
        return None
    df = pd.DataFrame({
        "date": pd.to_datetime(g["date"].to_list()),
        "open": g["open"].to_numpy().astype(float),
        "high": g["high"].to_numpy().astype(float),
        "low": g["low"].to_numpy().astype(float),
        "close": g["close"].to_numpy().astype(float),
    })
    return df


def _spec(fast_col, slow_col, ma_signal="crossover"):
    # ONE uniform exit: opposite cross (signal_flip). Spot LONG-ONLY -> no funding. Taker cost.
    return StrategySpec(
        fast_col=fast_col, slow_col=slow_col, signal=ma_signal,
        filter_col=None, exit_policy="signal_flip", cost_rt=TAKER,
        use_funding=False, funding_col="fund_rate_mean", funding_scale=0.0,
        max_hold_bars=None, max_hold_ext_bars=None,
    )


def _window_trades(res, w):
    return [t["net_pnl"] for t in res.trades if t["window"] == w]


def _compound(nets):
    a = np.asarray(nets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0


def build_xs_dispersion(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Cross-sectional std of 1-bar returns across the universe, per date. (market-state feature.)"""
    rets = {}
    for sym, df in frames.items():
        r = df.set_index("date")["close"].pct_change()
        rets[sym] = r
    panel = pd.DataFrame(rets)               # index=date, cols=assets
    disp = panel.std(axis=1, ddof=0)         # per-date cross-sectional dispersion
    disp.name = "xs_dispersion"
    return disp


def run(quick: bool, do_firewall: bool) -> dict:
    loader = ChimeraLoader()
    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:20]
    print(f"[adaptive-MA expert] u100 1d | assets={len(syms)} | taker={TAKER} | "
          f"fixed-baseline={FIXED_FAST}/{FIXED_SLOW} {FIXED_TYPE} | exit=opposite-cross", flush=True)

    # 1. load all OHLC once (reused for dispersion + per-asset runs)
    frames = {}
    for s in syms:
        df = _load_ohlc(loader, s)
        if df is not None and len(df) > 400:
            frames[s] = df
    print(f"[load] {len(frames)}/{len(syms)} assets with >400 daily bars", flush=True)

    # 2. cross-sectional dispersion panel
    disp = build_xs_dispersion(frames)
    print(f"[xs_dispersion] dates={disp.notna().sum()} median={disp.median():.4f} "
          f"p10={disp.quantile(.1):.4f} p90={disp.quantile(.9):.4f}", flush=True)

    # 3. per-asset adaptive vs fixed (+ firewall)
    per_asset = {}
    constituent_cfgs = A.all_configs()  # for the 'beats best constituent fixed' check
    for k, (s, df) in enumerate(frames.items(), 1):
        xs = df["date"].map(disp)  # align dispersion by date
        feat = A.compute_features(df, xs_disp=pd.Series(xs.values, index=df.index))
        adp = A.build_adaptive_columns(feat)
        fixd = A.build_fixed_columns(adp, FIXED_FAST, FIXED_SLOW, FIXED_TYPE)

        # adaptive harness
        h_ad = CanonicalHarness(fixd, _spec("adaptive_fast", "adaptive_slow"), WIN, chimera_path=f"adaptive::{s}")
        r_ad = h_ad.run()
        # fixed harness (same df, fixed cols)
        h_fx = CanonicalHarness(fixd, _spec("fix_fast", "fix_slow"), WIN, chimera_path=f"fixed::{s}")
        r_fx = h_fx.run()

        rec = {"adaptive": {}, "fixed": {}}
        for w in WINDOWS:
            an = _window_trades(r_ad, w)
            fn = _window_trades(r_fx, w)
            rec["adaptive"][w] = {"comp": round(_compound(an), 2), "n": len(an),
                                  "wr": round(float((np.asarray(an) > 0).mean()) if an else 0.0, 3),
                                  "exp": round(float(np.mean(an) * 100) if an else 0.0, 4)}
            rec["fixed"][w] = {"comp": round(_compound(fn), 2), "n": len(fn),
                               "wr": round(float((np.asarray(fn) > 0).mean()) if fn else 0.0, 3),
                               "exp": round(float(np.mean(fn) * 100) if fn else 0.0, 4)}
        # per-window nets for pooling (all windows)
        rec["adaptive_nets"] = {w: _window_trades(r_ad, w) for w in WINDOWS}
        rec["fixed_nets"] = {w: _window_trades(r_fx, w) for w in WINDOWS}

        if do_firewall:
            try:
                fw = random_entry_null(h_ad, n_books=200, seed=7)
                rec["firewall"] = {w: {"real": fw["per_window"][w]["real"],
                                       "null_p95": fw["per_window"][w]["null_p95"],
                                       "beats_null": fw["per_window"][w]["beats_null"]} for w in WINDOWS}
                rec["firewall_beats_held"] = bool(fw["beats_held"])
            except Exception as e:  # noqa: BLE001
                rec["firewall"] = {"error": repr(e)}
                rec["firewall_beats_held"] = None
        per_asset[s] = rec
        if k % 10 == 0:
            print(f"[run] {k}/{len(frames)} assets done", flush=True)

    # 4. universe aggregation
    agg = _aggregate(per_asset, do_firewall)
    out = {"config": {"n_assets": len(frames), "taker": TAKER, "fixed_baseline": [FIXED_FAST, FIXED_SLOW, FIXED_TYPE],
                      "exit": "opposite_cross", "windows": WIN.__dict__,
                      "feature_consts": {"RV_WIN": A.RV_WIN, "ER_WIN": A.ER_WIN, "PCT_WIN": A.PCT_WIN,
                                         "LO_BAND": A.LO_BAND, "HI_BAND": A.HI_BAND},
                      "map": {f"{tr}_{vh}": A.config_for(tr, vh) for tr in (0, 1, 2) for vh in (0, 1)}},
           "aggregate": agg, "per_asset": per_asset}
    _print_report(agg, len(frames), do_firewall)
    outpath = Path(__file__).resolve().parent / ("results_quick.json" if quick else "results_u100.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    return out


def _aggregate(per_asset: dict, do_firewall: bool) -> dict:
    syms = list(per_asset.keys())
    res = {}
    for w in WINDOWS:
        a_comp = np.array([per_asset[s]["adaptive"][w]["comp"] for s in syms], float)
        f_comp = np.array([per_asset[s]["fixed"][w]["comp"] for s in syms], float)
        # pooled per-trade nets across assets (all windows)
        a_pool, f_pool = [], []
        for s in syms:
            a_pool += per_asset[s]["adaptive_nets"][w]; f_pool += per_asset[s]["fixed_nets"][w]
        a_pool = np.asarray(a_pool, float); f_pool = np.asarray(f_pool, float)
        res[w] = {
            "adaptive_mean_comp": round(float(a_comp.mean()), 2),
            "adaptive_median_comp": round(float(np.median(a_comp)), 2),
            "fixed_mean_comp": round(float(f_comp.mean()), 2),
            "fixed_median_comp": round(float(np.median(f_comp)), 2),
            "n_assets_adaptive_gt_fixed": int((a_comp > f_comp).sum()),
            "n_assets": len(syms),
            "pooled_trade_exp_adaptive_pct": (round(float(a_pool.mean() * 100), 4) if a_pool.size else None),
            "pooled_trade_exp_fixed_pct": (round(float(f_pool.mean() * 100), 4) if f_pool.size else None),
            "pooled_winrate_adaptive": (round(float((a_pool > 0).mean()), 3) if a_pool.size else None),
            "pooled_winrate_fixed": (round(float((f_pool > 0).mean()), 3) if f_pool.size else None),
            "pooled_n_adaptive": int(a_pool.size), "pooled_n_fixed": int(f_pool.size),
        }
    # held-out sign test (adaptive UNSEEN comp > fixed UNSEEN comp), binomial 2-sided p
    a_u = np.array([per_asset[s]["adaptive"]["UNSEEN"]["comp"] for s in syms], float)
    f_u = np.array([per_asset[s]["fixed"]["UNSEEN"]["comp"] for s in syms], float)
    diff = a_u - f_u
    nz = diff[diff != 0.0]
    wins = int((nz > 0).sum()); n = int(nz.size)
    p_binom = _binom_two_sided(wins, n) if n else None
    res["UNSEEN_signtest"] = {"adaptive_beats_fixed": wins, "n_decisive": n,
                              "two_sided_p": (round(p_binom, 4) if p_binom is not None else None),
                              "mean_diff_pp": round(float(diff.mean()), 2),
                              "median_diff_pp": round(float(np.median(diff)), 2)}
    if do_firewall:
        fbh = [per_asset[s].get("firewall_beats_held") for s in syms]
        res["firewall"] = {"n_assets_beat_null_held": int(sum(1 for x in fbh if x is True)),
                           "n_assets_evaluated": int(sum(1 for x in fbh if x is not None)),
                           "n_assets": len(syms)}
    return res


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    from math import comb
    probs = [comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(sum(pr for pr in probs if pr <= obs + 1e-12))


def _print_report(agg: dict, n_assets: int, do_firewall: bool):
    print("\n" + "=" * 78)
    print(f"ADAPTIVE-MA vs FIXED-MA vs RANDOM-NULL  |  u100 1d  |  {n_assets} assets  |  taker {TAKER}")
    print("=" * 78)
    print(f"{'window':8} {'ADAPT mean':>11} {'FIX mean':>10} {'ADAPT med':>10} {'FIX med':>9} "
          f"{'A>F assets':>11} {'A exp%':>8} {'F exp%':>8} {'A wr':>6} {'F wr':>6}")
    for w in WINDOWS:
        r = agg[w]
        ax = r["pooled_trade_exp_adaptive_pct"]; fx = r["pooled_trade_exp_fixed_pct"]
        aw = r["pooled_winrate_adaptive"]; fw = r["pooled_winrate_fixed"]
        print(f"{w:8} {r['adaptive_mean_comp']:>11} {r['fixed_mean_comp']:>10} "
              f"{r['adaptive_median_comp']:>10} {r['fixed_median_comp']:>9} "
              f"{str(r['n_assets_adaptive_gt_fixed'])+'/'+str(r['n_assets']):>11} "
              f"{(ax if ax is not None else 0):>8} {(fx if fx is not None else 0):>8} "
              f"{(aw if aw is not None else 0):>6} {(fw if fw is not None else 0):>6}")
    st = agg["UNSEEN_signtest"]
    print("-" * 78)
    print(f"UNSEEN sign test (adaptive comp > fixed comp): {st['adaptive_beats_fixed']}/{st['n_decisive']} "
          f"decisive | 2-sided p={st['two_sided_p']} | mean diff={st['mean_diff_pp']}pp | median diff={st['median_diff_pp']}pp")
    if do_firewall and "firewall" in agg:
        fw = agg["firewall"]
        print(f"FIREWALL (adaptive beats cost-matched random-entry null on OOS+UNSEEN): "
              f"{fw['n_assets_beat_null_held']}/{fw['n_assets_evaluated']} assets")
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 20 assets only")
    ap.add_argument("--firewall", action="store_true", help="run the random-entry null firewall per asset (slower)")
    args = ap.parse_args()
    run(quick=args.quick, do_firewall=args.firewall)
