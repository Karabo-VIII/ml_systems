"""runs/research/u100_1d_ma_vs_null_rwyb.py  --  RWYB backtest (NO COMMIT, read-only analysis).

TASK: full backtest on u100 1d. Emit held-out (UNSEEN) after-cost PER-TRADE edge + trade count,
with two baselines side-by-side on the SAME split+costs:
  (a) cost-matched random-entry NULL  -- via strat.firewall.random_entry_null  (the "is it timing or beta?" null)
  (b) best fixed-config MA baseline   -- the single MA crossover config that maximises IN-SAMPLE
                                          (TRAIN+VAL) equal-weight basket compound, applied identically
                                          to every u100 asset (NO per-asset tuning, NO held-out peeking).

Everything is LONG-ONLY, SPOT, taker round-trip cost 0.24% (TAKER_COST_RT), funding OFF (spot).
Indicators are past-only by construction (sma_past_only / ema_past_only -> fill at opens[i+1]).
Held-out windows: OOS [2025-03-15, 2025-12-31), UNSEEN [2025-12-31, 2026-05-22]. UNSEEN = the headline held-out.

Reproduce:  python runs/research/u100_1d_ma_vs_null_rwyb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader
from wealth_bot.harness import (CanonicalHarness, StrategySpec, sma_past_only, ema_past_only)
from strat import DEFAULT_WINDOWS, random_entry_null
from strat.candidate_gate import TAKER_COST_RT

WINDOWS = DEFAULT_WINDOWS  # train_end 2024-05-15 / val_end 2025-03-15 / oos_end 2025-12-31 / unseen_end 2026-05-22
HELD = "UNSEEN"

# Fixed MA-crossover grid (long-only, pure price, no exogenous filter). (kind, fast, slow).
GRID = [
    ("sma", 5, 20), ("sma", 10, 30), ("sma", 20, 50), ("sma", 10, 50),
    ("sma", 30, 100), ("sma", 50, 200),
    ("ema", 10, 30), ("ema", 20, 50), ("ema", 12, 26), ("ema", 50, 200),
]


def load_u100_1d():
    cl = ChimeraLoader()
    syms = cl.universes.list("u100")
    out = {}
    for s in syms:
        try:
            g = cl.load(s, cadence="1d")
        except Exception:
            continue
        d = g.to_dict(as_series=False)
        raw = np.asarray(d["date"])
        # polars Date -> object/datetime; coerce robustly
        dt = pd.to_datetime(raw)
        df = pd.DataFrame({
            "date": dt,
            "open": np.asarray(d["open"], float),
            "high": np.asarray(d.get("high", d["close"]), float),
            "low": np.asarray(d.get("low", d["close"]), float),
            "close": np.asarray(d["close"], float),
        })
        df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
        if len(df) >= 60:
            out[s] = df
    return out


def add_ma(df, kind, fast, slow):
    df = df.copy()
    fn = sma_past_only if kind == "sma" else ema_past_only
    df["ma_fast"] = fn(df["close"], fast)
    df["ma_slow"] = fn(df["close"], slow)
    return df


def make_spec():
    return StrategySpec(
        fast_col="ma_fast", slow_col="ma_slow", signal="crossover",
        filter_col=None, exit_policy="signal_flip",
        cost_rt=TAKER_COST_RT, use_funding=False,
        max_hold_bars=None, max_hold_ext_bars=None,
    )


def run_one(df, kind, fast, slow):
    """Run harness for one asset/config. Returns CanonicalResults or None if MA undefined / too short."""
    d = add_ma(df, kind, fast, slow)
    if d["ma_slow"].notna().sum() < 5:
        return None
    h = CanonicalHarness(d, make_spec(), WINDOWS, chimera_path=f"u100_1d::{kind}{fast}_{slow}")
    return h.run()


def basket_compound(comp_fracs):
    """Equal-weight basket: split capital 1/N across N independent per-asset sleeves.
    portfolio_mult = mean_i (1 + comp_i). Returns basket compound in %."""
    if not comp_fracs:
        return 0.0
    return float((np.mean([1.0 + c for c in comp_fracs]) - 1.0) * 100)


# ---------------------------------------------------------------------------
# STEP 1: select best fixed config IN-SAMPLE (TRAIN+VAL only -- no held-out peek)
# ---------------------------------------------------------------------------
def insample_basket(data, kind, fast, slow):
    fracs = []
    for s, df in data.items():
        res = run_one(df, kind, fast, slow)
        if res is None:
            continue
        # TRAIN+VAL combined compound (chain the two in-sample windows)
        ct = res.window_stats["TRAIN"].compound_pct / 100.0
        cv = res.window_stats["VAL"].compound_pct / 100.0
        comb = (1 + ct) * (1 + cv) - 1.0
        # only count assets that actually traded in-sample
        if (res.window_stats["TRAIN"].n_trades + res.window_stats["VAL"].n_trades) > 0:
            fracs.append(comb)
    return basket_compound(fracs), len(fracs)


# ---------------------------------------------------------------------------
# null per-trade edge (firewall logic, returns pooled per-trade nets for UNSEEN)
# ---------------------------------------------------------------------------
def null_per_trade_unseen(harness, n_books=60, seed=11):
    """Mirror firewall.random_entry_null's inner draw but POOL per-trade nets for the UNSEEN window,
    so we get a cost-matched random-entry per-trade distribution (apples-to-apples with the candidate)."""
    rng = np.random.default_rng(seed)
    real = harness.run()
    df = harness.df
    opens = df["open"].to_numpy(float)
    n = len(opens)
    cost = float(harness.spec.cost_rt)
    wlab = np.array([harness._window_label(pd.Timestamp(df["date"].iloc[i])) for i in range(n)])
    durs = [max(1, int(t["duration_bars"])) for t in real.trades if t["window"] == HELD]
    nw = len(durs)
    if nw == 0:
        return []
    durs = np.array(durs)
    eligible = np.array([i for i in range(1, n - 2) if wlab[i] == HELD])
    if len(eligible) == 0:
        return []
    nets = []
    for _ in range(n_books):
        entries = rng.choice(eligible, size=nw, replace=True)
        dsamp = rng.choice(durs, size=nw, replace=True)
        for e, d in zip(entries, dsamp):
            ef = e + 1
            xf = min(ef + int(d), n - 1)
            if xf <= ef:
                continue
            nets.append(opens[xf] / opens[ef] - 1.0 - cost)
    return nets


def dist(arr):
    a = np.asarray(arr, float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean_pct": round(float(a.mean()) * 100, 4),
        "mean_bps": round(float(a.mean()) * 1e4, 1),
        "std_pct": round(float(a.std()) * 100, 3),
        "median_pct": round(float(np.median(a)) * 100, 4),
        "p10_pct": round(float(np.percentile(a, 10)) * 100, 3),
        "p25_pct": round(float(np.percentile(a, 25)) * 100, 3),
        "p75_pct": round(float(np.percentile(a, 75)) * 100, 3),
        "p90_pct": round(float(np.percentile(a, 90)) * 100, 3),
        "min_pct": round(float(a.min()) * 100, 2),
        "max_pct": round(float(a.max()) * 100, 2),
        "win_rate": round(float((a > 0).mean()), 4),
    }


def main():
    print("=" * 96)
    print("RWYB: u100 1d  --  best fixed-config MA crossover  vs  cost-matched random-entry null")
    print(f"LONG-ONLY SPOT | taker cost_rt={TAKER_COST_RT} | funding OFF | exit=signal_flip | no filter")
    print(f"windows: TRAIN<{WINDOWS.train_end} VAL<{WINDOWS.val_end} OOS<{WINDOWS.oos_end} UNSEEN<={WINDOWS.unseen_end}")
    print("=" * 96)

    data = load_u100_1d()
    print(f"\nloaded u100 1d assets with >=60 bars: {len(data)}")

    # STEP 1 -- in-sample config selection
    print("\n[STEP 1] selecting best fixed MA config on IN-SAMPLE (TRAIN+VAL) equal-weight basket compound:")
    sel = []
    for (kind, f, s) in GRID:
        bc, na = insample_basket(data, kind, f, s)
        sel.append((bc, na, kind, f, s))
        print(f"  {kind}{f:>3}/{s:<3}  in-sample basket={bc:+8.2f}%  (assets traded={na})")
    sel.sort(reverse=True)
    best = sel[0]
    BKIND, BF, BS = best[2], best[3], best[4]
    print(f"\n  -> BEST FIXED CONFIG (in-sample): {BKIND}{BF}/{BS}  basket={best[0]:+.2f}%")

    # STEP 2 -- evaluate the chosen fixed config on held-out + run the firewall null per asset
    print(f"\n[STEP 2] held-out evaluation of {BKIND}{BF}/{BS} across {len(data)} assets + firewall null (SAME split+cost)")
    per_asset = {}            # asset -> {oos_n, oos_comp, uns_n, uns_comp, beats_null, null_p50, null_p95}
    pooled_uns_trades = []    # candidate per-trade nets (UNSEEN)
    pooled_oos_trades = []
    pooled_null_trades = []   # random-entry per-trade nets (UNSEEN)
    uns_comp_fracs = []
    n_beat = 0; n_eval_null = 0
    for i, (s, df) in enumerate(sorted(data.items())):
        res = run_one(df, BKIND, BF, BS)
        if res is None:
            continue
        uns = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
        oos = [t["net_pnl"] for t in res.trades if t["window"] == "OOS"]
        uns_c = res.window_stats["UNSEEN"].compound_pct
        oos_c = res.window_stats["OOS"].compound_pct
        pooled_uns_trades += uns
        pooled_oos_trades += oos
        if res.window_stats["UNSEEN"].n_trades > 0:
            uns_comp_fracs.append(uns_c / 100.0)
        # firewall (compound-level null) -- plain null over all UNSEEN bars
        h = CanonicalHarness(add_ma(df, BKIND, BF, BS), make_spec(), WINDOWS, chimera_path="fw")
        fw = random_entry_null(h, n_books=200, seed=7, regime_matched=False)
        u = fw["per_window"]["UNSEEN"]
        beats = u.get("beats_null")
        if beats is not None:
            n_eval_null += 1
            n_beat += int(bool(beats))
        per_asset[s] = {
            "oos_n": res.window_stats["OOS"].n_trades, "oos_comp": round(oos_c, 2),
            "uns_n": res.window_stats["UNSEEN"].n_trades, "uns_comp": round(uns_c, 2),
            "null_p50": u.get("null_p50"), "null_p95": u.get("null_p95"), "beats_null": beats,
        }
        # null per-trade pooled
        pooled_null_trades += null_per_trade_unseen(h, n_books=40, seed=13)

    # STEP 3 -- aggregate report
    print("\n" + "=" * 96)
    print("RESULTS  (held-out = UNSEEN unless noted)")
    print("=" * 96)

    cand_uns = dist(pooled_uns_trades)
    null_uns = dist(pooled_null_trades)
    cand_oos = dist(pooled_oos_trades)

    print("\n-- PER-TRADE EDGE (after-cost, pooled across assets) --")
    print(f"  {'metric':<12} {'CANDIDATE MA (UNSEEN)':>24} {'RANDOM-NULL (UNSEEN)':>24} {'CANDIDATE MA (OOS)':>22}")
    keys = ["n", "mean_bps", "mean_pct", "median_pct", "std_pct", "p10_pct", "p25_pct",
            "p75_pct", "p90_pct", "min_pct", "max_pct", "win_rate"]
    for k in keys:
        print(f"  {k:<12} {str(cand_uns.get(k)):>24} {str(null_uns.get(k)):>24} {str(cand_oos.get(k)):>22}")

    print("\n-- AGGREGATE COMPOUND (UNSEEN) --")
    eq_basket = basket_compound(uns_comp_fracs)
    mean_comp = float(np.mean([c * 100 for c in uns_comp_fracs])) if uns_comp_fracs else 0.0
    med_comp = float(np.median([c * 100 for c in uns_comp_fracs])) if uns_comp_fracs else 0.0
    pos_assets = sum(1 for c in uns_comp_fracs if c > 0)
    print(f"  equal-weight basket compound (1/N per asset) : {eq_basket:+.2f}%   (N traded={len(uns_comp_fracs)})")
    print(f"  mean per-asset compound                       : {mean_comp:+.2f}%")
    print(f"  median per-asset compound                     : {med_comp:+.2f}%")
    print(f"  assets positive on UNSEEN                      : {pos_assets}/{len(uns_comp_fracs)}")

    print("\n-- FIREWALL: candidate vs cost-matched random-entry null (per-asset, UNSEEN) --")
    print(f"  assets where real UNSEEN compound BEATS null p95 : {n_beat}/{n_eval_null}")
    null_p50s = [v["null_p50"] for v in per_asset.values() if v["null_p50"] is not None]
    reals = [v["uns_comp"] for v in per_asset.values() if v["null_p50"] is not None]
    if null_p50s:
        print(f"  mean per-asset real UNSEEN compound              : {np.mean(reals):+.2f}%")
        print(f"  mean per-asset NULL p50 compound                 : {np.mean(null_p50s):+.2f}%")
        print(f"  candidate per-trade mean - null per-trade mean   : "
              f"{(cand_uns.get('mean_bps') or 0) - (null_uns.get('mean_bps') or 0):+.1f} bps")

    print("\n-- n_trades + compound PER ASSET (UNSEEN; assets with >=1 UNSEEN trade) --")
    print(f"  {'asset':<12} {'uns_n':>6} {'uns_comp%':>10} {'null_p50':>9} {'null_p95':>9} {'beats':>6}")
    rows = [(s, v) for s, v in per_asset.items() if v["uns_n"] > 0]
    rows.sort(key=lambda kv: kv[1]["uns_comp"], reverse=True)
    for s, v in rows:
        print(f"  {s:<12} {v['uns_n']:>6} {v['uns_comp']:>10} {str(v['null_p50']):>9} "
              f"{str(v['null_p95']):>9} {str(v['beats_null']):>6}")
    print(f"\n  (assets with 0 UNSEEN trades: {sum(1 for v in per_asset.values() if v['uns_n']==0)})")
    total_uns_trades = sum(v["uns_n"] for v in per_asset.values())
    total_oos_trades = sum(v["oos_n"] for v in per_asset.values())
    print(f"  TOTAL UNSEEN trades across u100: {total_uns_trades} | TOTAL OOS trades: {total_oos_trades}")

    print("\n" + "=" * 96)
    print("VERDICT INPUTS (for the RESULT line):")
    print(f"  cand UNSEEN per-trade mean = {cand_uns.get('mean_bps')} bps (n={cand_uns.get('n')}, "
          f"win={cand_uns.get('win_rate')})")
    print(f"  null UNSEEN per-trade mean = {null_uns.get('mean_bps')} bps (n={null_uns.get('n')})")
    print(f"  equal-weight basket UNSEEN compound = {eq_basket:+.2f}% | beats-null assets {n_beat}/{n_eval_null}")
    print("=" * 96)


if __name__ == "__main__":
    main()
