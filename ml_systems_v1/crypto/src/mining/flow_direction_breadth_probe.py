"""src/mining/flow_direction_breadth_probe.py -- go/no-go on the order-flow DIRECTIONAL micro-edge.

THE CLAIM UNDER TEST (from a 3-asset DIB pilot)
-----------------------------------------------
Trading the DIRECTION of the order-flow imbalance (sign of norm_flow_imbalance) at the extreme tail
(|flow| >= p90, hold ~N bars) was OOS net-positive 3/3 (BTC/ETH/PEPE), threshold-robust + monotonic,
and survived regressing out price momentum (the flow carried DIRECTIONAL info ORTHOGONAL to momentum).
BUT it was cost-fragile (majors died at ~0.25-0.30% round-trip = the maker cost cliff) and n=3.

THE QUESTION
------------
Does that momentum-orthogonal order-flow DIRECTIONAL edge survive BREADTH (~10 liquid assets) + the
REAL cost model? The edge is claimed to live in the FLOW FEATURES (norm_flow_imbalance, norm_vpin),
NOT the imbalance-bar SAMPLING -> so it must reproduce on the full-universe TIME / DOLLAR chimera that
carries those columns. If it survives -> the session's first buildable directional micro-edge. If it
dies on cost or breadth -> the order-flow-direction axis is closed too.

WHAT THIS PROBE DOES (RWYB, held-out, no commit)
------------------------------------------------
SIGNAL  : at each bar t, signal = sign(flow_feature[t]). TRADE only when |flow_feature[t]| >= thr,
          where thr = the p90 of |flow_feature| measured on SEL ONLY (TRAIN+VAL), per asset
          (no look-ahead). Enter at close of t (signal known at close), hold N bars, capture the
          SIGNED forward log return * sign (long if sign>0, short if sign<0).
          Two flow features reported:
            raw      = norm_flow_imbalance (the signed z-scored order-flow imbalance)
            residual = the part of norm_flow_imbalance ORTHOGONAL to price momentum. We fit, on SEL,
                       an OLS of flow on [past 1-bar ret, past N-bar momentum, intercept]; the residual
                       flow = flow - fitted. We trade sign(residual) at |residual|>=p90(SEL). This is
                       the LOAD-BEARING momentum-orthogonal test: if sign(residual) still pays, the
                       directional info is NOT just momentum in disguise.

COST    : canonical src/strat/fill_model.py::MODES (NOT a flat-30bps lie). Reported at:
            taker  : cost_rt 0.0024, p_fill 1.00, adverse 0.00   (solid spot taker)
            maker  : cost_rt 0.0010, adverse 0.96, at p_fill in {1.00 ideal, 0.40, 0.21}
          Expected NET per ATTEMPTED trade at p_fill pf:
            net = pf * mean( gross_directional - adverse*|gross_directional| - cost_rt )
          (with prob 1-pf the maker order does not fill -> 0 exposure, 0 pnl.) For taker pf=1, adv=0.

NULL    : cost-matched RANDOM-DIRECTION null. Same trigger bars, same hold, same cost -- but the
          direction is a coin flip (random +/-1), averaged over many seeds. The edge is the EXCESS of
          the signal's net over this random-direction net. A positive net that does not beat the
          random-direction null is NOT a direction edge (it is just the asset's drift / the cost sign).

HELD-OUT: thr + the momentum regression are fit on SEL. Reported separately on SEL / OOS / UNSEEN.
          UNSEEN is touched ONCE. Splits are time-ordered fractional with a purge gap.

BREADTH : u10 (BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC). Report how many of 10 are OOS+UNSEEN
          net-positive at each cost scenario; the decisive number is the momentum-orthogonal
          residual-flow net of REAL cost across breadth.

Run:
  python src/mining/flow_direction_breadth_probe.py --universe u10 --cadence 4h --hold 16
  python src/mining/flow_direction_breadth_probe.py --universe u10 --cadence dollar --hold 16
No emoji. Do NOT git commit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from strat.fill_model import MODES  # noqa: E402

OUT = ROOT / "runs" / "mining"

# Canonical cost legs (from src/strat/fill_model.py::MODES -- do NOT hardcode fees).
TAKER_COST_RT = MODES["taker"]["cost_rt"]              # 0.0024
TAKER_ADVERSE = MODES["taker"]["adverse"]              # 0.00
MAKER_COST_RT = MODES["maker_pessimistic"]["cost_rt"]  # 0.0010
MAKER_ADVERSE = MODES["maker_pessimistic"]["adverse"]  # 0.96

# Held-out split fractions (time-ordered): SEL 0.70 (TRAIN+VAL) / OOS 0.20 / UNSEEN 0.10.
SEL_FRAC = 0.70
OOS_FRAC = 0.20
# UNSEEN = remainder (0.10)
PURGE = 50          # bars dropped at each boundary to kill normalization/overlap leakage
P_TAIL = 90.0       # |flow| >= p90 (TRAIN-fit) trigger
WARMUP = 64         # need past momentum window
MOM_WINDOW = 16     # past-N-bar momentum used in the orthogonalization regression


def _log_ret(close: np.ndarray) -> np.ndarray:
    r = np.zeros(len(close))
    r[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    return r


def _fwd_hold_ret(ret: np.ndarray, t: int, hold: int) -> float | None:
    """Signed forward log return from entering at close of t, holding `hold` bars."""
    end = t + hold
    if end >= len(ret):
        return None
    seg = ret[t + 1: end + 1]
    if not np.all(np.isfinite(seg)):
        return None
    return float(np.sum(seg))


def _splits(n: int) -> dict[str, np.ndarray]:
    """Time-ordered index ranges for SEL / OOS / UNSEEN with purge gaps and warmup."""
    sel_end = int(n * SEL_FRAC)
    oos_end = int(n * (SEL_FRAC + OOS_FRAC))
    sel = np.arange(WARMUP, sel_end - PURGE)
    oos = np.arange(sel_end + PURGE, oos_end - PURGE)
    unseen = np.arange(oos_end + PURGE, n)
    return {"SEL": sel, "OOS": oos, "UNSEEN": unseen}


def _residualize_flow(flow: np.ndarray, ret: np.ndarray, sel_idx: np.ndarray) -> np.ndarray:
    """Return residual flow = flow - OLS(flow ~ [past_1ret, past_mom, 1]) fit on SEL only.

    past_1ret[t]  = ret[t]            (the bar's own return -- contemporaneous momentum proxy, known at t)
    past_mom[t]   = sum ret[t-MOM_WINDOW+1 .. t]   (trailing momentum, known at t)
    Both are PAST-only at decision time t (we trade at close of t). Coeffs fit on SEL, applied to ALL.
    """
    n = len(flow)
    past_mom = np.full(n, np.nan)
    csum = np.cumsum(ret)
    for t in range(MOM_WINDOW, n):
        past_mom[t] = csum[t] - csum[t - MOM_WINDOW]
    X = np.column_stack([ret, past_mom, np.ones(n)])
    # fit on SEL rows where everything is finite
    m = sel_idx[np.isfinite(flow[sel_idx]) & np.isfinite(X[sel_idx]).all(axis=1)]
    if m.size < 50:
        return np.full(n, np.nan)
    beta, *_ = np.linalg.lstsq(X[m], flow[m], rcond=None)
    fitted = X @ beta
    resid = flow - fitted
    # rows with non-finite X -> residual undefined
    bad = ~np.isfinite(X).all(axis=1)
    resid[bad] = np.nan
    return resid


def _signal_trades(signal_feat: np.ndarray, ret: np.ndarray, thr: float,
                   idx: np.ndarray, hold: int):
    """For bars in idx with |signal_feat|>=thr, collect (gross_directional, abs_gross) arrays.

    gross_directional = sign(signal_feat[t]) * forward_hold_return  (the realizable directional pnl).
    Also returns the entry bar indices (for the random-direction null to reuse the SAME bars/holds).
    """
    g_dir, g_fwd, ent = [], [], []
    for t in idx:
        v = signal_feat[t]
        if not np.isfinite(v) or abs(v) < thr:
            continue
        f = _fwd_hold_ret(ret, int(t), hold)
        if f is None:
            continue
        s = 1.0 if v > 0 else -1.0
        g_dir.append(s * f)
        g_fwd.append(f)
        ent.append(int(t))
    return np.array(g_dir), np.array(g_fwd), np.array(ent, dtype=int)


def _net_per_attempt(gross_dir: np.ndarray, cost_rt: float, adverse: float, p_fill: float) -> float:
    """Expected NET per ATTEMPTED trade. net_per_fill = gross - adverse*|gross| - cost_rt; *p_fill."""
    if gross_dir.size == 0:
        return float("nan")
    net_per_fill = gross_dir - adverse * np.abs(gross_dir) - cost_rt
    return float(p_fill * np.mean(net_per_fill))


def _random_dir_null_net(g_fwd: np.ndarray, cost_rt: float, adverse: float, p_fill: float,
                         rng: np.random.Generator, n_seed: int = 500) -> float:
    """Cost-matched random-DIRECTION null: random +/-1 sign on the SAME forward returns, same cost.

    Returns the mean (over n_seed coin-flip sign assignments) of the net-per-attempt. This is the
    'no direction skill' baseline -- the signal must beat THIS, not just beat zero.
    """
    if g_fwd.size == 0:
        return float("nan")
    vals = np.empty(n_seed)
    for i in range(n_seed):
        sgn = rng.choice(np.array([-1.0, 1.0]), size=g_fwd.size)
        gross = sgn * g_fwd
        net_per_fill = gross - adverse * np.abs(gross) - cost_rt
        vals[i] = p_fill * np.mean(net_per_fill)
    return float(np.mean(vals))


# Cost scenarios to report: (label, cost_rt, adverse, p_fill)
COST_SCENARIOS = [
    ("taker",          TAKER_COST_RT, TAKER_ADVERSE, 1.00),
    ("maker_pf1.00",   MAKER_COST_RT, MAKER_ADVERSE, 1.00),
    ("maker_pf0.40",   MAKER_COST_RT, MAKER_ADVERSE, 0.40),
    ("maker_pf0.21",   MAKER_COST_RT, MAKER_ADVERSE, 0.21),
]


def _eval_feature(feat: np.ndarray, ret: np.ndarray, splits: dict, hold: int,
                  rng: np.random.Generator) -> dict:
    """Fit thr on SEL |feat| p90, evaluate net/null on SEL/OOS/UNSEEN at all cost scenarios."""
    sel = splits["SEL"]
    sel_finite = feat[sel][np.isfinite(feat[sel])]
    if sel_finite.size < 50:
        return {"error": "insufficient SEL"}
    thr = float(np.percentile(np.abs(sel_finite), P_TAIL))

    out = {"thr_p90_on_SEL": round(thr, 4)}
    for seg in ["SEL", "OOS", "UNSEEN"]:
        g_dir, g_fwd, ent = _signal_trades(feat, ret, thr, splits[seg], hold)
        seg_res = {
            "n_trades": int(g_dir.size),
            "gross_dir_bps": round(1e4 * float(np.mean(g_dir)), 2) if g_dir.size else None,
            "up_rate_correct": round(float(np.mean(g_dir > 0)), 3) if g_dir.size else None,
            "cost": {},
        }
        for label, c, adv, pf in COST_SCENARIOS:
            net = _net_per_attempt(g_dir, c, adv, pf)
            null = _random_dir_null_net(g_fwd, c, adv, pf, rng)
            excess = (net - null) if (np.isfinite(net) and np.isfinite(null)) else float("nan")
            seg_res["cost"][label] = {
                "net_bps": round(1e4 * net, 3) if np.isfinite(net) else None,
                "null_dir_bps": round(1e4 * null, 3) if np.isfinite(null) else None,
                "excess_bps": round(1e4 * excess, 3) if np.isfinite(excess) else None,
                "positive": bool(np.isfinite(net) and net > 0),
                "beats_null": bool(np.isfinite(excess) and excess > 0),
            }
        out[seg] = seg_res
    return out


def run(universe: str, cadence: str, hold: int, seed: int) -> dict:
    cl = ChimeraLoader()
    u = UniverseLoader.load()
    syms = u.list(universe)
    rng = np.random.default_rng(seed)

    feats_needed = ["close", "norm_flow_imbalance", "norm_vpin"]
    per_asset = {}
    skipped = []
    for sym in syms:
        try:
            df = cl.load(sym, cadence=cadence, features=feats_needed)
        except Exception as e:
            skipped.append((sym, f"load:{type(e).__name__}"))
            continue
        if df is None or df.height < 1000 or "close" not in df.columns:
            skipped.append((sym, "too_short"))
            continue
        close = df["close"].to_numpy().astype(float)
        flow = df["norm_flow_imbalance"].to_numpy().astype(float)
        vpin = df["norm_vpin"].to_numpy().astype(float) if "norm_vpin" in df.columns else np.full_like(close, np.nan)
        ret = _log_ret(close)
        n = len(close)
        splits = _splits(n)
        if splits["OOS"].size < 30 or splits["UNSEEN"].size < 20:
            skipped.append((sym, "split_too_small"))
            continue
        resid = _residualize_flow(flow, ret, splits["SEL"])

        per_asset[sym] = {
            "n_bars": n,
            "n_SEL": int(splits["SEL"].size),
            "n_OOS": int(splits["OOS"].size),
            "n_UNSEEN": int(splits["UNSEEN"].size),
            "raw_flow":      _eval_feature(flow, ret, splits, hold, rng),
            "residual_flow": _eval_feature(resid, ret, splits, hold, rng),
            "vpin_signed":   _eval_feature(vpin, ret, splits, hold, rng),
        }

    # ---- BREADTH AGGREGATION ----
    def breadth(feature_key: str, seg: str, cost_label: str):
        pos, beat, vals, excs = 0, 0, [], []
        for sym, d in per_asset.items():
            fd = d.get(feature_key, {})
            sg = fd.get(seg, {})
            cd = sg.get("cost", {}).get(cost_label) if isinstance(sg, dict) else None
            if cd and cd["net_bps"] is not None:
                vals.append(cd["net_bps"])
                excs.append(cd["excess_bps"] if cd["excess_bps"] is not None else float("nan"))
                if cd["positive"]:
                    pos += 1
                if cd["beats_null"]:
                    beat += 1
        n = len([1 for sym, d in per_asset.items()
                 if d.get(feature_key, {}).get(seg, {}).get("cost", {}).get(cost_label)])
        return {
            "n_assets": n,
            "n_net_positive": pos,
            "n_beats_null": beat,
            "mean_net_bps": round(float(np.nanmean(vals)), 3) if vals else None,
            "median_net_bps": round(float(np.nanmedian(vals)), 3) if vals else None,
            "mean_excess_bps": round(float(np.nanmean(excs)), 3) if excs else None,
        }

    summary = {}
    for fkey in ["raw_flow", "residual_flow", "vpin_signed"]:
        summary[fkey] = {}
        for seg in ["OOS", "UNSEEN"]:
            summary[fkey][seg] = {lab: breadth(fkey, seg, lab) for (lab, *_rest) in COST_SCENARIOS}

    return {
        "universe": universe,
        "cadence": cadence,
        "hold_bars": hold,
        "seed": seed,
        "trigger": f"|flow_feature| >= p90(SEL); trade sign(flow_feature); hold {hold} bars",
        "cost_source": "src/strat/fill_model.py::MODES (taker + maker_pessimistic)",
        "splits": f"SEL {SEL_FRAC} / OOS {OOS_FRAC} / UNSEEN remainder, purge {PURGE}, warmup {WARMUP}",
        "n_assets_used": len(per_asset),
        "assets_used": list(per_asset.keys()),
        "skipped": skipped,
        "BREADTH_SUMMARY": summary,
        "per_asset": per_asset,
    }


def _fmt_breadth(b: dict) -> str:
    return (f"pos {b['n_net_positive']}/{b['n_assets']}  beats_null {b['n_beats_null']}/{b['n_assets']}  "
            f"mean_net {b['mean_net_bps']} bps  mean_excess {b['mean_excess_bps']} bps")


def _report(res: dict) -> None:
    print("=" * 92)
    print("ORDER-FLOW DIRECTIONAL EDGE -- BREADTH + REAL-COST GO/NO-GO")
    print("=" * 92)
    print(f"universe={res['universe']}  cadence={res['cadence']}  hold={res['hold_bars']} bars  "
          f"assets_used={res['n_assets_used']}")
    print(f"trigger : {res['trigger']}")
    print(f"cost    : {res['cost_source']}")
    print(f"splits  : {res['splits']}")
    if res["skipped"]:
        print(f"skipped : {res['skipped']}")
    s = res["BREADTH_SUMMARY"]
    for fkey, label in [("raw_flow", "RAW FLOW (norm_flow_imbalance)"),
                        ("residual_flow", "RESIDUAL FLOW (momentum-orthogonal)  <-- LOAD-BEARING"),
                        ("vpin_signed", "VPIN (signed)")]:
        print("-" * 92)
        print(label)
        for seg in ["OOS", "UNSEEN"]:
            print(f"  [{seg}]")
            for lab, *_ in COST_SCENARIOS:
                print(f"    {lab:14}: {_fmt_breadth(s[fkey][seg][lab])}")
    print("=" * 92)
    # decisive verdict line
    rf = s["residual_flow"]
    dec_oos = rf["OOS"]["maker_pf0.21"]
    dec_uns = rf["UNSEEN"]["maker_pf0.21"]
    taker_oos = rf["OOS"]["taker"]
    taker_uns = rf["UNSEEN"]["taker"]
    print("DECISIVE (momentum-orthogonal residual-flow, net of REAL cost):")
    print(f"  taker      OOS pos {taker_oos['n_net_positive']}/{taker_oos['n_assets']} beats_null "
          f"{taker_oos['n_beats_null']}/{taker_oos['n_assets']} mean_net {taker_oos['mean_net_bps']}bps | "
          f"UNSEEN pos {taker_uns['n_net_positive']}/{taker_uns['n_assets']} beats_null "
          f"{taker_uns['n_beats_null']}/{taker_uns['n_assets']} mean_net {taker_uns['mean_net_bps']}bps")
    print(f"  maker p21  OOS pos {dec_oos['n_net_positive']}/{dec_oos['n_assets']} beats_null "
          f"{dec_oos['n_beats_null']}/{dec_oos['n_assets']} mean_net {dec_oos['mean_net_bps']}bps | "
          f"UNSEEN pos {dec_uns['n_net_positive']}/{dec_uns['n_assets']} beats_null "
          f"{dec_uns['n_beats_null']}/{dec_uns['n_assets']} mean_net {dec_uns['mean_net_bps']}bps")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="4h", help="dollar|1d|4h|1h|30m|15m")
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    res = run(args.universe, args.cadence, args.hold, args.seed)
    res["_elapsed_s"] = round(time.time() - t0, 1)
    _report(res)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUT / f"flow_direction_breadth_{args.universe}_{args.cadence}_hold{args.hold}.json")
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
