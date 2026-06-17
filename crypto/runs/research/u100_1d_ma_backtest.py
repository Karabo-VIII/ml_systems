"""RWYB: full MA-crossover backtest on u100 @ 1d -- held-out per-trade edge vs two baselines.

Subject  : best fixed-config MA crossover (SAME params for ALL assets), LONG-ONLY SPOT, taker 0.24% RT.
Baseline a: cost-matched random-ENTRY null via strat.firewall.random_entry_null (the firewall floor).
Baseline b: best fixed-config MA -- selected on DEV (TRAIN+VAL+OOS) pooled per-trade net, NEVER on UNSEEN.

Honesty guards:
  - config selection uses ONLY dev windows; UNSEEN is reported, never used to pick the "best" config.
  - taker 0.24% round-trip; no funding (spot); no leverage; no filter (pure price-MA baseline).
  - per-trade DISTRIBUTION reported (not just mean); n_trades per asset; aggregate compound = equal-weight
    mean of per-asset window compound (each asset = one independent sleeve, cash when flat).
No commit. Emits JSON + console table.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path("src")))
import numpy as np
import pandas as pd

from pipeline.chimera_loader import ChimeraLoader
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only, ema_past_only
from strat.firewall import random_entry_null

COST_RT = 0.0024            # taker round-trip (honest baseline)
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
DEV = ["TRAIN", "VAL", "OOS"]
HELD = "UNSEEN"
N_BOOKS = 200

GRID = [   # (label, kind, fast, slow) -- fixed across ALL assets
    ("SMA_5_20",   "sma", 5,  20),
    ("SMA_10_30",  "sma", 10, 30),
    ("SMA_20_50",  "sma", 20, 50),
    ("SMA_10_50",  "sma", 10, 50),
    ("SMA_50_100", "sma", 50, 100),
    ("EMA_12_26",  "ema", 12, 26),
    ("EMA_20_50",  "ema", 20, 50),
]


def load_ohlc(L, sym):
    d = L.load(sym, cadence="1d").to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                         "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                         "close": np.asarray(d["close"], float)})


def make_harness(base, kind, fast, slow):
    df = base.copy()
    fn = sma_past_only if kind == "sma" else ema_past_only
    df["f"] = fn(df["close"], fast)
    df["s"] = fn(df["close"], slow)
    spec = StrategySpec(fast_col="f", slow_col="s", signal="crossover", filter_col=None,
                        exit_policy="signal_flip_or_filter", cost_rt=COST_RT, use_funding=False,
                        max_hold_bars=None, max_hold_ext_bars=None)
    return CanonicalHarness(df, spec, WIN, chimera_path=f"u100_1d::{kind}_{fast}_{slow}")


def dist(nets):
    a = np.asarray(nets, float)
    if a.size == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None, "std_pct": None, "win_rate": None,
                "p5_pct": None, "p25_pct": None, "p75_pct": None, "p95_pct": None, "min_pct": None, "max_pct": None}
    return {"n": int(a.size), "mean_pct": round(float(a.mean()) * 100, 4),
            "median_pct": round(float(np.median(a)) * 100, 4), "std_pct": round(float(a.std()) * 100, 4),
            "win_rate": round(float((a > 0).mean()), 4),
            "p5_pct": round(float(np.percentile(a, 5)) * 100, 3), "p25_pct": round(float(np.percentile(a, 25)) * 100, 3),
            "p75_pct": round(float(np.percentile(a, 75)) * 100, 3), "p95_pct": round(float(np.percentile(a, 95)) * 100, 3),
            "min_pct": round(float(a.min()) * 100, 3), "max_pct": round(float(a.max()) * 100, 3)}


def pooled_random_pertrade(harness, window, seed=7, n_books=N_BOOKS):
    """Per-trade net distribution of the cost-matched random-entry null for `window`.
    Mirrors strat.firewall.random_entry_null inner loop (opens[xf]/opens[ef]-1-cost), drawing entries
    uniformly from eligible window bars and durations from the REAL trade holding distribution."""
    rng = np.random.default_rng(seed)
    real = harness.run()
    df = harness.df
    opens = df["open"].to_numpy(float)
    n = len(opens)
    cost = float(harness.spec.cost_rt)
    wlab = np.array([harness._window_label(pd.Timestamp(df["date"].iloc[i])) for i in range(n)])
    durs = [max(1, int(t["duration_bars"])) for t in real.trades if t["window"] == window]
    nw = len(durs)
    elig = np.array([i for i in range(1, n - 2) if wlab[i] == window])
    if nw == 0 or elig.size == 0:
        return []
    durs = np.array(durs)
    nets = []
    for _ in range(n_books):
        entries = rng.choice(elig, size=nw, replace=True)
        dsamp = rng.choice(durs, size=nw, replace=True)
        for e, dd in zip(entries, dsamp):
            ef = e + 1
            xf = min(ef + int(dd), n - 1)
            if xf <= ef:
                continue
            nets.append(opens[xf] / opens[ef] - 1.0 - cost)
    return nets


def main():
    t0 = time.time()
    L = ChimeraLoader()
    syms = L.universes.list("u100")
    print(f"[u100-1d-MA] loading OHLC for {len(syms)} assets ...", flush=True)
    bases = {}
    for s in syms:
        try:
            bases[s] = load_ohlc(L, s)
        except Exception as e:
            print(f"  SKIP {s}: {type(e).__name__} {e}", flush=True)
    syms = list(bases.keys())
    print(f"[u100-1d-MA] loaded {len(syms)} assets in {time.time()-t0:.1f}s", flush=True)

    # ---- sweep grid, collect per-window pooled trades + per-asset compound ----
    grid_results = {}
    per_asset_trades = {}   # label -> {sym: CanonicalResults}
    for label, kind, fast, slow in GRID:
        per_asset_trades[label] = {}
        pooled = {w: [] for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]}
        per_asset_comp = {w: {} for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]}
        for s in syms:
            res = make_harness(bases[s], kind, fast, slow).run()
            per_asset_trades[label][s] = res
            for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]:
                pooled[w].extend([t["net_pnl"] for t in res.trades if t["window"] == w])
                per_asset_comp[w][s] = res.window_stats[w].compound_pct
        dev_nets = pooled["TRAIN"] + pooled["VAL"] + pooled["OOS"]
        grid_results[label] = {
            "pooled": pooled, "per_asset_comp": per_asset_comp,
            "dev_pertrade_mean_pct": round(float(np.mean(dev_nets)) * 100, 4) if dev_nets else None,
            "dev_n": len(dev_nets),
            "unseen_pertrade": dist(pooled["UNSEEN"]),
            "oos_pertrade": dist(pooled["OOS"]),
            "unseen_agg_comp_mean": round(float(np.mean(list(per_asset_comp["UNSEEN"].values()))), 3),
            "oos_agg_comp_mean": round(float(np.mean(list(per_asset_comp["OOS"].values()))), 3),
        }
        print(f"  {label:11s} dev_pertrade_mean={grid_results[label]['dev_pertrade_mean_pct']}%  "
              f"UNSEEN per-trade mean={grid_results[label]['unseen_pertrade']['mean_pct']}%  "
              f"n={grid_results[label]['unseen_pertrade']['n']}  "
              f"UNSEEN agg_comp(mean per-asset)={grid_results[label]['unseen_agg_comp_mean']}%", flush=True)

    # ---- select BEST fixed config on DEV ONLY (max dev pooled per-trade mean net) ----
    best = max((l for l in grid_results if grid_results[l]["dev_pertrade_mean_pct"] is not None),
               key=lambda l: grid_results[l]["dev_pertrade_mean_pct"])
    print(f"\n[SELECT] best fixed config on DEV (TRAIN+VAL+OOS pooled per-trade mean): {best}  "
          f"(dev_pertrade_mean={grid_results[best]['dev_pertrade_mean_pct']}%)", flush=True)

    bl, bk, bf, bs = next(g for g in GRID if g[0] == best)

    # ---- baseline (a): cost-matched random-entry null via firewall.py, per asset, aggregated ----
    print(f"[firewall] running random_entry_null (n_books={N_BOOKS}) for {best} on {len(syms)} assets ...", flush=True)
    fw_real_uns, fw_p50_uns, fw_p95_uns = {}, {}, {}
    fw_real_oos, fw_p95_oos = {}, {}
    beats_uns = 0
    null_pertrade_uns = []
    for s in syms:
        h = make_harness(bases[s], bk, bf, bs)
        fw = random_entry_null(h, n_books=N_BOOKS, seed=7)
        u = fw["per_window"]["UNSEEN"]; o = fw["per_window"]["OOS"]
        fw_real_uns[s] = u["real"]; fw_p50_uns[s] = u["null_p50"]; fw_p95_uns[s] = u["null_p95"]
        fw_real_oos[s] = o["real"]; fw_p95_oos[s] = o["null_p95"]
        if u.get("beats_null") is True:
            beats_uns += 1
        null_pertrade_uns.extend(pooled_random_pertrade(h, "UNSEEN", seed=7))

    # aggregate firewall (only assets that actually traded in UNSEEN, i.e. null defined)
    uns_traded = [s for s in syms if fw_p95_uns[s] is not None]
    agg_real_uns = round(float(np.mean([fw_real_uns[s] for s in uns_traded])), 3) if uns_traded else None
    agg_null_p50 = round(float(np.mean([fw_p50_uns[s] for s in uns_traded])), 3) if uns_traded else None
    agg_null_p95 = round(float(np.mean([fw_p95_uns[s] for s in uns_traded])), 3) if uns_traded else None

    # ---- per-asset n_trades (UNSEEN) for best config ----
    best_res = per_asset_trades[best]
    n_per_asset_uns = {s: best_res[s].window_stats["UNSEEN"].n_trades for s in syms}
    comp_per_asset_uns = {s: round(best_res[s].window_stats["UNSEEN"].compound_pct, 2) for s in syms}
    n_traders = sum(1 for s in syms if n_per_asset_uns[s] > 0)
    pos_assets = sum(1 for s in syms if comp_per_asset_uns[s] > 0 and n_per_asset_uns[s] > 0)

    strat_pertrade_uns = grid_results[best]["unseen_pertrade"]
    null_pertrade_dist = dist(null_pertrade_uns)

    comp_vals_all = list(comp_per_asset_uns.values())
    comp_vals_traded = [comp_per_asset_uns[s] for s in syms if n_per_asset_uns[s] > 0]

    out = {
        "task": "u100 1d MA-crossover full backtest -- held-out per-trade edge vs random-null + best-MA",
        "universe": "u100", "cadence": "1d", "n_assets": len(syms),
        "cost_rt_taker": COST_RT, "long_only_spot": True, "funding": False, "filter": None,
        "windows": vars(WIN), "n_books_firewall": N_BOOKS,
        "selection": {"rule": "max DEV(TRAIN+VAL+OOS) pooled per-trade mean net; UNSEEN never used to select",
                      "best_config": best, "best_dev_pertrade_mean_pct": grid_results[best]["dev_pertrade_mean_pct"]},
        "grid": {l: {"dev_pertrade_mean_pct": grid_results[l]["dev_pertrade_mean_pct"],
                     "unseen_pertrade_mean_pct": grid_results[l]["unseen_pertrade"]["mean_pct"],
                     "unseen_n_trades": grid_results[l]["unseen_pertrade"]["n"],
                     "unseen_agg_comp_mean_pct": grid_results[l]["unseen_agg_comp_mean"],
                     "oos_agg_comp_mean_pct": grid_results[l]["oos_agg_comp_mean"]} for l in grid_results},
        "HELD_OUT_UNSEEN": {
            "best_MA_config": best,
            "per_trade_edge": strat_pertrade_uns,
            "total_n_trades": strat_pertrade_uns["n"],
            "n_assets_traded": n_traders,
            "aggregate_compound_equalweight_mean_pct": round(float(np.mean(comp_vals_all)), 3),
            "aggregate_compound_equalweight_median_pct": round(float(np.median(comp_vals_all)), 3),
            "aggregate_compound_traded_only_mean_pct": round(float(np.mean(comp_vals_traded)), 3) if comp_vals_traded else None,
            "pct_assets_positive_of_traded": round(pos_assets / max(1, n_traders), 3),
            "n_trades_per_asset": n_per_asset_uns,
            "compound_per_asset_pct": comp_per_asset_uns,
        },
        "BASELINE_a_random_entry_null_firewall": {
            "via": "strat.firewall.random_entry_null",
            "per_trade_edge": null_pertrade_dist,
            "agg_real_compound_mean_pct_traded": agg_real_uns,
            "agg_null_p50_compound_mean_pct_traded": agg_null_p50,
            "agg_null_p95_compound_mean_pct_traded": agg_null_p95,
            "n_assets_real_beats_null_p95_unseen": beats_uns,
            "n_assets_traded_unseen": len(uns_traded),
        },
        "BASELINE_b_best_fixed_MA": {  # same object as HELD_OUT_UNSEEN.best_MA_config (the MA family's best single config)
            "config": best,
            "unseen_per_trade_mean_pct": strat_pertrade_uns["mean_pct"],
            "unseen_agg_comp_mean_pct": grid_results[best]["unseen_agg_comp_mean"],
        },
        "wallclock_s": round(time.time() - t0, 1),
    }
    outpath = Path("runs/research/u100_1d_ma_backtest_result.json")
    outpath.write_text(json.dumps(out, indent=2, default=str))

    # ---- console summary ----
    print("\n" + "=" * 78)
    print(f"HELD-OUT (UNSEEN {WIN.oos_end}->{WIN.unseen_end}) | u100 1d | LONG-ONLY SPOT | taker {COST_RT*100:.2f}% RT")
    print(f"Best fixed MA config (selected on DEV only): {best}")
    print("=" * 78)
    sp, nu = strat_pertrade_uns, null_pertrade_dist
    print(f"{'metric':<26}{'BEST-MA (b)':>16}{'RANDOM-NULL (a)':>18}")
    print(f"{'per-trade mean %':<26}{str(sp['mean_pct']):>16}{str(nu['mean_pct']):>18}")
    print(f"{'per-trade median %':<26}{str(sp['median_pct']):>16}{str(nu['median_pct']):>18}")
    print(f"{'per-trade std %':<26}{str(sp['std_pct']):>16}{str(nu['std_pct']):>18}")
    print(f"{'per-trade win rate':<26}{str(sp['win_rate']):>16}{str(nu['win_rate']):>18}")
    print(f"{'per-trade p5 %':<26}{str(sp['p5_pct']):>16}{str(nu['p5_pct']):>18}")
    print(f"{'per-trade p25 %':<26}{str(sp['p25_pct']):>16}{str(nu['p25_pct']):>18}")
    print(f"{'per-trade p75 %':<26}{str(sp['p75_pct']):>16}{str(nu['p75_pct']):>18}")
    print(f"{'per-trade p95 %':<26}{str(sp['p95_pct']):>16}{str(nu['p95_pct']):>18}")
    print(f"{'per-trade max %':<26}{str(sp['max_pct']):>16}{str(nu['max_pct']):>18}")
    print(f"{'n trades (pooled)':<26}{str(sp['n']):>16}{str(nu['n']):>18}")
    print("-" * 78)
    print(f"agg compound (eq-wt mean per-asset, all {len(syms)}): {out['HELD_OUT_UNSEEN']['aggregate_compound_equalweight_mean_pct']}%")
    print(f"agg compound (eq-wt median per-asset):              {out['HELD_OUT_UNSEEN']['aggregate_compound_equalweight_median_pct']}%")
    print(f"agg compound (traded-only mean):                   {out['HELD_OUT_UNSEEN']['aggregate_compound_traded_only_mean_pct']}%")
    print(f"firewall agg real compound (traded): {agg_real_uns}%  vs null_p50 {agg_null_p50}%  null_p95 {agg_null_p95}%")
    print(f"assets where real BEATS null_p95 (UNSEEN): {beats_uns}/{len(uns_traded)} traded")
    print(f"assets total/traded/positive(of traded): {len(syms)}/{n_traders}/{pos_assets}")
    print(f"\nwrote {outpath}  ({out['wallclock_s']}s)")


if __name__ == "__main__":
    main()
