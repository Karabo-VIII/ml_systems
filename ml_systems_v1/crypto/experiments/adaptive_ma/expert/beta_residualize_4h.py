"""experiments/adaptive_ma/expert/beta_residualize_4h.py -- FALSIFIER (-k): is the ER-gated 4h
fixed-MA return TIMING ALPHA or LONG-BETA-DURING-TRENDS?

TASK (auditor RED-team -k falsifier): take the LOCKED config of er_gate_4h.py (8/21 EMA, Kaufman
ER(20)>0.4 gate, ATR-trail x3.0 + 42-bar cap, taker 0.0024, 4h) and re-measure its P&L on
BETA-NEUTRALIZED / market-residual returns: per bar, subtract beta * (BTC return), where beta is the
asset's OLS sensitivity to BTC over open-to-open 4h returns. If the held-out compound advantage over the
regime-matched (ER>0.4) random-entry null COLLAPSES to ~0 once beta is removed, the "return" is
long-beta-during-trends, NOT entry-timing alpha.

METHOD (faithful to "the same locked config"):
  1. Build the EXACT same entry/ATR columns and run the EXACT same SetupHarness -> the strategy picks the
     SAME trades (same entry_fill_idx, exit_idx, window). We do NOT change a single trade decision.
  2. Per-bar open-to-open simple returns:  a[t]=aopen[t+1]/aopen[t]-1 (asset), b[t]=bopen[t+1]/bopen[t]-1
     (BTC, aligned to the asset's bars by floored-4h timestamp bucket, ffilled across rare gaps).
  3. beta = OLS slope of a on b over the FULL sample (a MEASUREMENT transform, not a trade decision;
     full-sample is the *conservative* choice -- it gives BTC the best chance to absorb the return, so the
     residual test is HARDER to pass).  Residual per bar: rr[t] = a[t] - beta*b[t]  (== a daily-rebalanced
     beta-hedged long: long asset, short beta*BTC each bar).
  4. RAW replay net  = prod_{t in [ef,xf)} (1+a[t]) - 1 - cost   (== open[xf]/open[ef]-1-cost; matches the
     harness exactly on TIME exits, approximates stop/target by open-to-open -- but the null uses the SAME
     open-to-open replay, so RAW-vs-RESID and REAL-vs-NULL are both apples-to-apples).
     RESID replay net = prod_{t in [ef,xf)} (1+rr[t]) - 1 - cost.
  5. Regime-matched null: random entries drawn ONLY from ER>0.4 bars (the SAME gate), held for durations
     sampled from the strategy's OWN per-window distribution, SAME cost -- replayed on BOTH a[] (raw null)
     and rr[] (residual null). Real-vs-null compared per held-out window (OOS, UNSEEN).
  6. Report: per-trade expectancy (pooled held-out) raw vs residual, and block-bootstrap p05 (battery's own
     stationary block-bootstrap) on the residualized per-trade returns.

All past-only / leak-safe by reusing the audited SetupHarness (entry fill=opens[i+1]; ATR=atr[j-1]).
No emoji (cp1252). numpy/pandas only. SAFETY: analysis + JSON only; no commit/deploy/capital.

RWYB:  python experiments/adaptive_ma/expert/beta_residualize_4h.py [--quick] [--probe ETHUSDT]
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

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy  # noqa: E402
from strat.battery import block_bootstrap_p05_p95  # noqa: E402

# reuse the LOCKED config + column builder verbatim from the strategy under test
import er_gate_4h as EG  # type: ignore  # noqa: E402  (same directory)

WINDOWS = EG.WINDOWS
HELD = EG.HELD
TAKER = EG.TAKER
WIN = EG.WIN
ER_THRESH = EG.ER_THRESH
GRID_MS = 4 * 3600 * 1000  # 4h grid for cross-asset timestamp alignment


def load_4h_ts(loader: ChimeraLoader, sym: str) -> pd.DataFrame | None:
    """Like er_gate_4h.load_4h but ALSO carries the raw ms timestamp + a floored-4h bucket key."""
    try:
        g = loader.load(sym, cadence="4h")
    except Exception:
        return None
    ts = g["timestamp"].to_numpy().astype("int64")
    df = pd.DataFrame({
        "timestamp": ts,
        "bucket": ts // GRID_MS,
        "date": pd.to_datetime(ts, unit="ms"),
        "open": g["open"].to_numpy().astype(float),
        "high": g["high"].to_numpy().astype(float),
        "low": g["low"].to_numpy().astype(float),
        "close": g["close"].to_numpy().astype(float),
    })
    return df


def btc_open_series(loader: ChimeraLoader) -> pd.Series:
    """BTC open indexed by floored-4h bucket (sorted, unique) -- the market factor price."""
    b = load_4h_ts(loader, "BTCUSDT")
    s = pd.Series(b["open"].to_numpy(), index=b["bucket"].to_numpy())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def bar_returns(open_arr: np.ndarray) -> np.ndarray:
    """Open-to-open simple return r[t]=open[t+1]/open[t]-1, length n (last entry NaN)."""
    o = np.asarray(open_arr, float)
    r = np.full(o.shape, np.nan)
    r[:-1] = o[1:] / o[:-1] - 1.0
    return r


def ols_beta(a: np.ndarray, b: np.ndarray) -> float:
    """Full-sample OLS slope of a on b over finite pairs. NaN-safe; returns nan if degenerate."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan")
    am = a[m] - a[m].mean()
    bm = b[m] - b[m].mean()
    den = float((bm * bm).sum())
    return float((am * bm).sum() / den) if den > 0 else float("nan")


def expanding_beta(a: np.ndarray, b: np.ndarray, min_obs: int = 100) -> np.ndarray:
    """PAST-ONLY expanding OLS slope: beta_t = cov_{0..t-1}(a,b)/var_{0..t-1}(b), using ONLY bars strictly
    before t (leak-free hedge ratio). Before min_obs valid pairs -> beta=1.0 (neutral fallback). Returns a
    per-bar beta array aligned to a/b. (Defends the falsifier against the 'full-sample beta peeks at
    held-out' attack: an expanding beta fits worse -> leaves MORE beta in the residual, a STRICTER test.)"""
    n = a.size
    af = np.where(np.isfinite(a) & np.isfinite(b), a, 0.0)
    bf = np.where(np.isfinite(a) & np.isfinite(b), b, 0.0)
    valid = (np.isfinite(a) & np.isfinite(b)).astype(float)
    c = np.cumsum(valid)
    sa, sb = np.cumsum(af), np.cumsum(bf)
    sab, sbb = np.cumsum(af * bf), np.cumsum(bf * bf)
    beta = np.ones(n)
    for t in range(n):
        k = t - 1  # stats through bar t-1 (strictly past)
        if k < 0 or c[k] < min_obs:
            continue
        cnt = c[k]
        cov = sab[k] - sa[k] * sb[k] / cnt
        var = sbb[k] - sb[k] * sb[k] / cnt
        if var > 0:
            beta[t] = cov / var
    return beta


def _replay_compound(nets: list) -> float:
    return float((np.prod(1.0 + np.asarray(nets, float)) - 1.0) * 100) if nets else 0.0


def trade_net(per_bar: np.ndarray, ef: int, xf: int, cost: float) -> float | None:
    """Replay a single trade's P&L as prod_{t in [ef,xf)} (1+per_bar[t]) - 1 - cost. None if any NaN."""
    if xf <= ef:
        return None
    seg = per_bar[ef:xf]
    if not np.all(np.isfinite(seg)):
        return None
    return float(np.prod(1.0 + seg) - 1.0 - cost)


def residual_null(harness, a: np.ndarray, rr: np.ndarray, cost: float,
                  n_books: int, seed: int) -> dict:
    """Regime-matched (ER>0.4) random-entry null, replayed on BOTH raw a[] and residual rr[] per-bar
    returns. Mirrors strat.firewall.random_entry_null exactly (same eligible-bar / duration-sampling
    logic) but computes nets from the supplied per-bar return arrays instead of opens[xf]/opens[ef]."""
    rng = np.random.default_rng(seed)
    real = harness.run()
    df = harness.df
    n = len(df)
    dates = df["date"]
    wlab = np.array([harness._window_label(pd.Timestamp(dates.iloc[i])) for i in range(n)])

    real_n = {w: 0 for w in WINDOWS}
    real_durs = {w: [] for w in WINDOWS}
    for t in real.trades:
        w = t["window"]
        real_n[w] += 1
        real_durs[w].append(max(1, int(t["duration_bars"])))

    # gate mask: ER>0.4 (the harness spec was overridden to filter_col='er', gt, 0.4 by the caller)
    gate_vals = df[harness.spec.filter_col].to_numpy(float)
    with np.errstate(invalid="ignore"):
        gate_mask = gate_vals > float(harness.spec.filter_val)
    eligible = {w: np.array([i for i in range(1, n - 2) if wlab[i] == w and gate_mask[i]]) for w in WINDOWS}

    out = {}
    for w in WINDOWS:
        nw = real_n[w]
        if nw == 0 or len(eligible[w]) == 0:
            out[w] = {"raw_p50": None, "raw_p95": None, "resid_p50": None, "resid_p95": None, "n": nw}
            continue
        durs = np.array(real_durs[w]) if real_durs[w] else np.array([3])
        raw_comps, resid_comps = [], []
        for _ in range(n_books):
            entries = rng.choice(eligible[w], size=nw, replace=True)
            dsamp = rng.choice(durs, size=nw, replace=True)
            rnets, rrnets = [], []
            for e, d in zip(entries, dsamp):
                ef = e + 1
                xf = min(ef + int(d), n - 1)
                rn = trade_net(a, ef, xf, cost)
                rrn = trade_net(rr, ef, xf, cost)
                if rn is not None:
                    rnets.append(rn)
                if rrn is not None:
                    rrnets.append(rrn)
            raw_comps.append(_replay_compound(rnets))
            resid_comps.append(_replay_compound(rrnets))
        out[w] = {"raw_p50": round(float(np.percentile(raw_comps, 50)), 2),
                  "raw_p95": round(float(np.percentile(raw_comps, 95)), 2),
                  "resid_p50": round(float(np.percentile(resid_comps, 50)), 2),
                  "resid_p95": round(float(np.percentile(resid_comps, 95)), 2),
                  "n": nw}
    return out


def run_asset(loader: ChimeraLoader, sym: str, btc_open_by_bucket: pd.Series,
              n_books: int, seed: int, beta_mode: str = "full") -> dict | None:
    df = load_4h_ts(loader, sym)
    if df is None or len(df) < 1000:
        return None
    cols = EG.build_cols(df, entry_style="state")  # carries timestamp/bucket through df.copy()
    if int(cols["entry"].sum()) < 4:
        return None

    # align BTC open to this asset's bars by bucket (ffill across rare missing buckets)
    btc_aligned = btc_open_by_bucket.reindex(cols["bucket"].to_numpy()).ffill().to_numpy(float)
    a = bar_returns(cols["open"].to_numpy(float))    # asset open-to-open per-bar return
    b = bar_returns(btc_aligned)                     # BTC   open-to-open per-bar return (aligned)
    if beta_mode == "past":
        beta_arr = expanding_beta(a, b)              # PAST-ONLY per-bar hedge ratio (leak-free)
        beta = float(np.nanmedian(beta_arr))         # reported summary stat
        rr = a - beta_arr * b
    else:
        beta = ols_beta(a, b)                        # full-sample OLS slope (conservative diagnostic)
        if not np.isfinite(beta):
            return None
        rr = a - beta * b                            # market-residual per-bar return

    policy = ExitPolicy(atr_trail_mult=EG.ATR_TRAIL_MULT, atr_col="atr", max_hold_bars=EG.MAX_HOLD)
    h = SetupHarness(cols, "entry", policy, WIN, cost_rt=TAKER, regime_match_on_entry=False)
    h.spec.filter_col = "er"
    h.spec.filter_op = "gt"
    h.spec.filter_val = ER_THRESH

    res = h.run()
    cost = TAKER

    # replay every REAL trade on raw a[] and residual rr[]; keep per-window net lists
    raw_nets = {w: [] for w in WINDOWS}
    resid_nets = {w: [] for w in WINDOWS}
    resid_pairs = {w: [] for w in WINDOWS}  # (entry_ts, resid_net) for diagnostics
    for t in res.trades:
        w = t["window"]
        ef, xf = int(t["entry_fill_idx"]), int(t["exit_idx"])
        rn = trade_net(a, ef, xf, cost)
        rrn = trade_net(rr, ef, xf, cost)
        if rn is not None:
            raw_nets[w].append(rn)
        if rrn is not None:
            resid_nets[w].append(rrn)
            resid_pairs[w].append((t["entry_ts"], rrn))

    raw_comp = {w: _replay_compound(raw_nets[w]) for w in WINDOWS}
    resid_comp = {w: _replay_compound(resid_nets[w]) for w in WINDOWS}
    nulld = residual_null(h, a, rr, cost, n_books=n_books, seed=seed)

    # ADVANTAGE over the regime-matched null (the literal quantity the falsifier asks about):
    # real_compound - null_p50_compound, per held-out window, raw vs residual.
    def adv(real_c, w, key):
        nv = nulld[w].get(key)
        return None if nv is None else round(real_c - nv, 2)
    advantage = {w: {"raw": adv(raw_comp[w], w, "raw_p50"), "resid": adv(resid_comp[w], w, "resid_p50")}
                 for w in HELD}

    # held-out pooled per-trade returns
    held_resid = np.array([x for w in HELD for x in resid_nets[w]], float)
    held_raw = np.array([x for w in HELD for x in raw_nets[w]], float)
    bb_resid = block_bootstrap_p05_p95(held_resid) if held_resid.size >= 10 else {"p05": None, "p50": None, "p95": None}

    def beats(real_comp, null_block, key95):
        return bool(null_block.get(key95) is not None and real_comp > null_block[key95])

    raw_beats_held = all(beats(raw_comp[w], nulld[w], "raw_p95") for w in HELD)
    resid_beats_held = all(beats(resid_comp[w], nulld[w], "resid_p95") for w in HELD)
    raw_pos_held = all(raw_comp[w] > 0 for w in HELD)
    resid_pos_held = all(resid_comp[w] > 0 for w in HELD)

    bb_raw = block_bootstrap_p05_p95(held_raw) if held_raw.size >= 10 else {"p05": None}
    return {
        "beta_vs_btc": round(beta, 3),
        "n_trades_held": int(held_resid.size),
        "windows_raw": {w: round(raw_comp[w], 2) for w in WINDOWS},
        "windows_resid": {w: round(resid_comp[w], 2) for w in WINDOWS},
        "advantage_over_null": advantage,   # real - null_p50, per held-out window, raw vs resid
        "null": nulld,
        "raw_beats_null_held": raw_beats_held, "raw_pos_held": raw_pos_held,
        "resid_beats_null_held": resid_beats_held, "resid_pos_held": resid_pos_held,
        "held_raw_exp_pct": round(float(held_raw.mean() * 100), 4) if held_raw.size else None,
        "held_resid_exp_pct": round(float(held_resid.mean() * 100), 4) if held_resid.size else None,
        "held_raw_p05_blockboot": bb_raw["p05"],
        "held_resid_p05_blockboot": bb_resid["p05"],
        "held_resid_p50_blockboot": bb_resid["p50"],
        "raw_p05_positive": bool(bb_raw["p05"] is not None and bb_raw["p05"] > 0),
        "resid_p05_positive": bool(bb_resid["p05"] is not None and bb_resid["p05"] > 0),
        # pass raw trade-return lists up for the pooled aggregate
        "_held_resid_rets": held_resid.tolist(),
        "_held_raw_rets": held_raw.tolist(),
    }


def main(quick: bool, probe: str | None, beta_mode: str = "full"):
    loader = ChimeraLoader()
    btc_open = btc_open_series(loader)
    n_books = 200

    if probe:
        rec = run_asset(loader, probe, btc_open, n_books=300, seed=7, beta_mode=beta_mode)
        print(f"[probe {probe}]")
        if rec is None:
            print("  (insufficient data / setups)")
            return
        prn = {k: v for k, v in rec.items() if not k.startswith("_")}
        print(json.dumps(prn, indent=2, default=str))
        return

    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:20]
    print(f"[beta-residualize 4h falsifier] u100 4h | assets={len(syms)} | taker={TAKER} | "
          f"locked={EG.FAST}/{EG.SLOW}EMA ER>{ER_THRESH} ATRx{EG.ATR_TRAIL_MULT}+{EG.MAX_HOLD} | "
          f"factor=BTC open-to-open, full-sample OLS beta", flush=True)

    per_asset = {}
    for k, s in enumerate(syms, 1):
        try:
            rec = run_asset(loader, s, btc_open, n_books=n_books, seed=7)
        except Exception as e:  # noqa: BLE001
            rec = {"error": repr(e)[:160]}
        if rec is not None:
            per_asset[s] = rec
        if k % 10 == 0:
            print(f"[run] {k}/{len(syms)} processed, {len([x for x in per_asset.values() if 'beta_vs_btc' in x])} evaluated", flush=True)

    ev = {s: r for s, r in per_asset.items() if "beta_vs_btc" in r}
    n = len(ev)
    raw_beat_pos = [s for s, r in ev.items() if r["raw_beats_null_held"] and r["raw_pos_held"]]
    resid_beat_pos = [s for s, r in ev.items() if r["resid_beats_null_held"] and r["resid_pos_held"]]
    raw_pos = [s for s, r in ev.items() if r["raw_pos_held"]]
    resid_pos = [s for s, r in ev.items() if r["resid_pos_held"]]

    def arr(key, win):
        return np.array([r[key][win] for r in ev.values()], float)

    pooled_raw = np.array([x for r in ev.values() for x in r["_held_raw_rets"]], float)
    pooled_resid = np.array([x for r in ev.values() for x in r["_held_resid_rets"]], float)

    def expectancy_bootstrap(rets, n=5000, seed=7):
        """Resample per-trade returns WITH replacement, take the MEAN each draw -> p05/p50/p95 of the
        per-trade expectancy. (Pooling cross-asset trades into a single COMPOUND curve is meaningless --
        they are not one portfolio's sequence; the mean-expectancy bootstrap is the correct pooled test.)"""
        a = np.asarray(rets, float)
        if a.size < 10:
            return {"p05": None, "p50": None, "p95": None}
        rng = np.random.default_rng(seed)
        means = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1) * 100
        return {"p05": round(float(np.percentile(means, 5)), 4),
                "p50": round(float(np.percentile(means, 50)), 4),
                "p95": round(float(np.percentile(means, 95)), 4)}

    eb_raw = expectancy_bootstrap(pooled_raw)
    eb_resid = expectancy_bootstrap(pooled_resid)
    # per-asset block-bootstrap-of-compound p05>0 counts (the held-out robustness gate, per asset)
    raw_p05_pos = [s for s, r in ev.items() if r.get("raw_p05_positive")]
    resid_p05_pos = [s for s, r in ev.items() if r.get("resid_p05_positive")]
    # advantage over null (real - null_p50), aggregated across assets, raw vs resid
    def adv_arr(win, key):
        return np.array([r["advantage_over_null"][win][key] for r in ev.values()
                         if r["advantage_over_null"][win][key] is not None], float)

    betas = np.array([r["beta_vs_btc"] for r in ev.values()], float)
    agg = {
        "n_assets_evaluated": n,
        "beta_vs_btc_median": round(float(np.median(betas)), 3),
        "beta_vs_btc_mean": round(float(np.mean(betas)), 3),
        # RAW (cross-check vs er_gate_4h.json: expect ~0/77 beat, ~13 pos_held)
        "raw_n_beat_null_AND_pos_held": len(raw_beat_pos),
        "raw_n_pos_held": len(raw_pos), "raw_pos_held_assets": raw_pos,
        "raw_UNSEEN_mean": round(float(arr("windows_raw", "UNSEEN").mean()), 2),
        "raw_UNSEEN_median": round(float(np.median(arr("windows_raw", "UNSEEN"))), 2),
        "raw_OOS_mean": round(float(arr("windows_raw", "OOS").mean()), 2),
        "raw_OOS_median": round(float(np.median(arr("windows_raw", "OOS"))), 2),
        # RESIDUAL (beta-neutralized) -- the falsifier surface
        "resid_n_beat_null_AND_pos_held": len(resid_beat_pos), "resid_beat_pos_assets": resid_beat_pos,
        "resid_n_pos_held": len(resid_pos), "resid_pos_held_assets": resid_pos,
        "resid_UNSEEN_mean": round(float(arr("windows_resid", "UNSEEN").mean()), 2),
        "resid_UNSEEN_median": round(float(np.median(arr("windows_resid", "UNSEEN"))), 2),
        "resid_OOS_mean": round(float(arr("windows_resid", "OOS").mean()), 2),
        "resid_OOS_median": round(float(np.median(arr("windows_resid", "OOS"))), 2),
        # advantage over the regime-matched null (real - null_p50), aggregated across assets
        "advantage_OOS_raw_median": round(float(np.median(adv_arr("OOS", "raw"))), 2) if adv_arr("OOS", "raw").size else None,
        "advantage_OOS_resid_median": round(float(np.median(adv_arr("OOS", "resid"))), 2) if adv_arr("OOS", "resid").size else None,
        "advantage_UNSEEN_raw_median": round(float(np.median(adv_arr("UNSEEN", "raw"))), 2) if adv_arr("UNSEEN", "raw").size else None,
        "advantage_UNSEEN_resid_median": round(float(np.median(adv_arr("UNSEEN", "resid"))), 2) if adv_arr("UNSEEN", "resid").size else None,
        # per-asset held-out block-bootstrap-of-compound p05 > 0 counts
        "raw_n_p05_positive": len(raw_p05_pos), "raw_p05_positive_assets": raw_p05_pos,
        "resid_n_p05_positive": len(resid_p05_pos), "resid_p05_positive_assets": resid_p05_pos,
        # pooled held-out per-trade expectancy + bootstrap of the MEAN (correct pooled test)
        "pooled_held_n_trades": int(pooled_resid.size),
        "pooled_held_raw_exp_pct": round(float(pooled_raw.mean() * 100), 4) if pooled_raw.size else None,
        "pooled_held_resid_exp_pct": round(float(pooled_resid.mean() * 100), 4) if pooled_resid.size else None,
        "pooled_raw_exp_bootstrap": eb_raw,
        "pooled_resid_exp_bootstrap": eb_resid,
    }
    out = {"config": {"cadence": "4h", "locked_from": "er_gate_4h.py", "fast": EG.FAST, "slow": EG.SLOW,
                      "er_thresh": ER_THRESH, "atr_trail_mult": EG.ATR_TRAIL_MULT, "max_hold": EG.MAX_HOLD,
                      "taker": TAKER, "factor": "BTCUSDT open-to-open 4h, full-sample OLS beta",
                      "null": "regime_matched_ER>0.4 random-entry, replayed on residual returns",
                      "n_books": n_books},
           "aggregate": agg,
           "per_asset": {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in per_asset.items()}}

    print("\n" + "=" * 80)
    print(f"BETA-RESIDUALIZATION FALSIFIER -- ER-gated fixed-MA @ 4h  |  {n} assets")
    print("=" * 80)
    print(f"  beta vs BTC: median={agg['beta_vs_btc_median']}  mean={agg['beta_vs_btc_mean']}")
    print(f"  RAW      held-out: OOS mean={agg['raw_OOS_mean']}% med={agg['raw_OOS_median']}% | "
          f"UNSEEN mean={agg['raw_UNSEEN_mean']}% med={agg['raw_UNSEEN_median']}% | "
          f"pos_held={agg['raw_n_pos_held']}/{n}  beat_null&pos={agg['raw_n_beat_null_AND_pos_held']}/{n}")
    print(f"  RESIDUAL held-out: OOS mean={agg['resid_OOS_mean']}% med={agg['resid_OOS_median']}% | "
          f"UNSEEN mean={agg['resid_UNSEEN_mean']}% med={agg['resid_UNSEEN_median']}% | "
          f"pos_held={agg['resid_n_pos_held']}/{n}  beat_null&pos={agg['resid_n_beat_null_AND_pos_held']}/{n}")
    print(f"  advantage over null (real-null_p50) median: OOS raw={agg['advantage_OOS_raw_median']} "
          f"resid={agg['advantage_OOS_resid_median']} | UNSEEN raw={agg['advantage_UNSEEN_raw_median']} "
          f"resid={agg['advantage_UNSEEN_resid_median']}")
    print(f"  per-asset held-out block-boot p05>0: RAW={agg['raw_n_p05_positive']}/{n}  "
          f"RESID={agg['resid_n_p05_positive']}/{n}")
    print(f"  pooled held-out per-trade expectancy:  RAW={agg['pooled_held_raw_exp_pct']}%  ->  "
          f"RESID={agg['pooled_held_resid_exp_pct']}%   (n={agg['pooled_held_n_trades']} trades)")
    print(f"  pooled per-trade-EXPECTANCY bootstrap RESID: p05={eb_resid['p05']}%  p50={eb_resid['p50']}%  "
          f"p95={eb_resid['p95']}%   (RAW p05={eb_raw['p05']}%)")
    print("=" * 80)
    outpath = Path(__file__).resolve().parent / ("beta_resid_4h_quick.json" if quick else "beta_resid_4h_u100.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 20 u100 assets only")
    ap.add_argument("--probe", type=str, default=None, help="single-asset probe (e.g. ETHUSDT)")
    args = ap.parse_args()
    main(quick=args.quick, probe=args.probe)
