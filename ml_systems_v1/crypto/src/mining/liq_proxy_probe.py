"""LIQUIDATION-HEATMAP PROXY PROBE -- test the Coinglass thesis FOR FREE before buying.

User question (2026-06-11): "what does Coinglass get us, and is there a proxy before we buy in?
If the thesis is unproven and I buy, that wastes money."

Coinglass gets us the FORWARD liquidation HEATMAP: where leveraged-position liquidation levels
CLUSTER ahead of price (the 'magnets' price gets pulled toward), aggregated across exchanges.
The D72 meta-labeler (OOS AUC 0.52, the continuation discriminator that's the only lock on the
+6%/event intraday prize) used realized/coincident liq + OI-deltas + funding -- but NOT (a) LSR
CROWDING (which side is over-positioned) nor (b) a liquidation-MAGNET PROXIMITY (where the magnet
sits ahead). Those two ARE the heatmap's information, and we can PROXY them from on-disk data:
  - long_short_ratio (5m metrics)  -> crowding side
  - open_interest_val (5m metrics) -> fuel
  - squeeze_up   = oi_z / lsr   (shorts crowded + fuel -> upside squeeze magnet above)
  - liq_below    = oi_z * lsr   (longs crowded + fuel -> downside liq magnet below)
  - magnet proximity proxy: distance from price to the recent OI-weighted price node.

TEST: add these proxy features to the EXACT D72 continuation label (does a +1.5% up-trigger
continue to a profitable trail?) and re-measure OOS AUC. DECISION RULE (pre-registered):
  - proxy OOS AUC >= 0.56  -> Coinglass (a cleaner/aggregated version of the same signal) is
    worth the $29/mo test. The thesis has free support.
  - proxy OOS AUC < 0.54   -> the positioning-magnet signal does NOT discriminate at our
    resolution; Coinglass unlikely to help -> SKIP (save the money).
  - 0.54-0.56 -> marginal; report both, lean skip.
Honest caveat: our proxy is a ROUGH estimate of the heatmap (no per-position entry prices); a
null proxy is suggestive-not-decisive, but a POSITIVE proxy is a real green light. UNSEEN sealed.
No emoji. Run: python -m mining.liq_proxy_probe --universe u10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mining.cascade_oracle import load_panel          # noqa: E402 (has OI + LSR + price)
from mining.mover_ride import trail_exit              # noqa: E402
from mining.mover_metalabel import _norm_sym, TRAIN_END_MS, OOS_END_MS, T_TRIG, TRAIL_K, GATE_COST, MIN_MINUTES_LEFT, DAY_MS  # noqa: E402

OUT = ROOT / "runs" / "mining"

# baseline (D72) features + the NEW positioning-magnet proxy features
BASE_FEATS = ["t2trig", "overshoot", "pre_vol", "oi_d1h", "oi_d4h", "oi_d24h", "funding", "liq_ratio", "regime"]
PROXY_FEATS = ["lsr", "lsr_z", "oi_z", "squeeze_up", "liq_below", "magnet_prox"]


def split_of(ms): return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def extract(sym: str) -> list[dict]:
    df = load_panel(sym)
    ms = df["minute_ts"].to_numpy()
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    vol = df["vol_usd"].to_numpy()
    oi = df["open_interest_val"].to_numpy() if "open_interest_val" in df.columns else np.full(len(df), np.nan)
    lsr = df["long_short_ratio"].to_numpy() if "long_short_ratio" in df.columns else np.full(len(df), np.nan)
    pre_vol = df["pre_vol"].to_numpy()
    oi1 = df["oi_d1h"].to_numpy(); oi4 = df["oi_d4h"].to_numpy(); oi24 = df["oi_d24h"].to_numpy()
    fund = df["funding"].to_numpy(); liqr = df["liq_ratio"].to_numpy()
    regime = df["regime_above_sma200"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    n = len(df)
    # trailing 24h OI mean/std (causal) for oi_z
    oi_mean = np.full(n, np.nan); oi_std = np.full(n, np.nan)
    # rolling via cumsum approx (1440-min window)
    W = 1440
    for nm_arr, src in [("m", oi)]:
        pass
    # compute rolling mean/std of OI causally
    valid_oi = np.where(np.isfinite(oi), oi, np.nan)
    s = np.zeros(n + 1); s2 = np.zeros(n + 1); cnt = np.zeros(n + 1)
    fill = np.nan_to_num(valid_oi)
    isv = np.isfinite(valid_oi).astype(float)
    s[1:] = np.cumsum(fill); s2[1:] = np.cumsum(fill * fill); cnt[1:] = np.cumsum(isv)
    for t in range(W, n):
        k = cnt[t] - cnt[t - W]
        if k >= 100:
            mu = (s[t] - s[t - W]) / k
            var = max(0.0, (s2[t] - s2[t - W]) / k - mu * mu)
            oi_mean[t] = mu; oi_std[t] = np.sqrt(var)
    # recent OI-weighted price node (magnet proxy): VWAP over trailing 24h
    px_node = np.full(n, np.nan)
    sv = np.zeros(n + 1); spv = np.zeros(n + 1)
    sv[1:] = np.cumsum(np.nan_to_num(vol)); spv[1:] = np.cumsum(np.nan_to_num(vol) * c)
    for t in range(W, n):
        vv = sv[t] - sv[t - W]
        if vv > 0:
            px_node[t] = (spv[t] - spv[t - W]) / vv

    day_ids = ms // DAY_MS
    dstarts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    dends = np.append(dstarts[1:] - 1, n - 1)
    ev = []
    for ds, de in zip(dstarts, dends):
        if de - ds < 1200 or real[ds:de + 1].mean() < 0.90:
            continue
        d_open = o[ds]
        if not np.isfinite(d_open) or d_open <= 0:
            continue
        sp = split_of(int(ms[ds]))
        rel = c[ds:de + 1] / d_open - 1.0
        hit = np.flatnonzero(rel >= T_TRIG)
        if len(hit) == 0:
            continue
        m = ds + int(hit[0]); f = m + 1
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]) or o[f] <= 0:
            continue
        g = trail_exit(o, c, f, de, TRAIL_K)
        y = 1 if (g - GATE_COST) > 0 else 0
        lsr_v = lsr[m] if np.isfinite(lsr[m]) and lsr[m] > 0 else np.nan
        oi_z = ((oi[m] - oi_mean[m]) / oi_std[m]) if np.isfinite(oi_mean[m]) and oi_std[m] > 0 else np.nan
        lsr_z = (lsr_v - 1.0) if np.isfinite(lsr_v) else np.nan
        squeeze_up = (oi_z / lsr_v) if (np.isfinite(oi_z) and np.isfinite(lsr_v) and lsr_v > 0) else np.nan
        liq_below = (oi_z * lsr_v) if (np.isfinite(oi_z) and np.isfinite(lsr_v)) else np.nan
        magnet_prox = ((c[m] - px_node[m]) / c[m]) if np.isfinite(px_node[m]) and c[m] > 0 else np.nan
        ev.append({"sym": sym, "split": sp, "y": y,
                   "t2trig": float(m - ds), "overshoot": float(rel[m - ds] - T_TRIG),
                   "pre_vol": float(pre_vol[m]) if np.isfinite(pre_vol[m]) else np.nan,
                   "oi_d1h": float(oi1[m]) if np.isfinite(oi1[m]) else np.nan,
                   "oi_d4h": float(oi4[m]) if np.isfinite(oi4[m]) else np.nan,
                   "oi_d24h": float(oi24[m]) if np.isfinite(oi24[m]) else np.nan,
                   "funding": float(fund[m]) if fund[m] is not None and np.isfinite(fund[m]) else np.nan,
                   "liq_ratio": float(liqr[m]) if np.isfinite(liqr[m]) else np.nan,
                   "regime": (1.0 if regime[m] else 0.0) if regime[m] is not None else np.nan,
                   "lsr": float(lsr_v) if np.isfinite(lsr_v) else np.nan, "lsr_z": float(lsr_z) if np.isfinite(lsr_z) else np.nan,
                   "oi_z": float(oi_z) if np.isfinite(oi_z) else np.nan,
                   "squeeze_up": float(squeeze_up) if np.isfinite(squeeze_up) else np.nan,
                   "liq_below": float(liq_below) if np.isfinite(liq_below) else np.nan,
                   "magnet_prox": float(magnet_prox) if np.isfinite(magnet_prox) else np.nan})
    return ev


def auc_for(events, feats, seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    tr = [e for e in events if e["split"] == "TRAIN"]
    oo = [e for e in events if e["split"] == "OOS"]
    if len(tr) < 50 or len(oo) < 30:
        return None
    Xtr = np.array([[e[f] for f in feats] for e in tr], float); ytr = np.array([e["y"] for e in tr])
    Xoo = np.array([[e[f] for f in feats] for e in oo], float); yoo = np.array([e["y"] for e in oo])
    if len(set(ytr)) < 2 or len(set(yoo)) < 2:
        return None
    clf = HistGradientBoostingClassifier(max_iter=250, random_state=seed, early_stopping=False)
    clf.fit(Xtr, ytr)
    return {"train": round(float(roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])), 3),
            "oos": round(float(roc_auc_score(yoo, clf.predict_proba(Xoo)[:, 1])), 3), "n_tr": len(tr), "n_oo": len(oo)}


def main():
    ap = argparse.ArgumentParser(prog="python -m mining.liq_proxy_probe")
    ap.add_argument("--universe", default="u10")
    a = ap.parse_args()
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{a.universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]
    allev = []
    for s in syms:
        try:
            allev += extract(s)
        except Exception as e:
            print(f"[{s}] skip: {type(e).__name__}: {str(e)[:60]}")
    base = auc_for(allev, BASE_FEATS)
    proxy = auc_for(allev, BASE_FEATS + PROXY_FEATS)
    proxy_only = auc_for(allev, PROXY_FEATS)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"liq_proxy_probe_{a.universe}_{stamp}.json"
    json.dump({"base_feats": BASE_FEATS, "proxy_feats": PROXY_FEATS, "n_events": len(allev),
               "baseline": base, "base_plus_proxy": proxy, "proxy_only": proxy_only}, open(p, "w"), indent=1)
    print(f"\n## LIQUIDATION-HEATMAP PROXY PROBE -- {a.universe} -- continuation label (D72), n_events={len(allev)}")
    print(f"   BASELINE (D72 feats):      OOS AUC {base['oos'] if base else '-'}  (train {base['train'] if base else '-'})")
    print(f"   + MAGNET PROXY feats:      OOS AUC {proxy['oos'] if proxy else '-'}  (train {proxy['train'] if proxy else '-'})")
    print(f"   PROXY-ONLY (positioning):  OOS AUC {proxy_only['oos'] if proxy_only else '-'}")
    if base and proxy:
        delta = proxy['oos'] - base['oos']
        verdict = ("BUY Coinglass (proxy >= 0.56 OR moves AUC materially)" if proxy['oos'] >= 0.56 or delta >= 0.03
                   else "SKIP Coinglass (proxy < 0.54, no lift)" if proxy['oos'] < 0.54 and delta < 0.02
                   else "MARGINAL -- lean skip")
        print(f"\n   DELTA from proxy: {delta:+.3f} AUC.  DECISION: {verdict}")
    print(f"   JSON -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
