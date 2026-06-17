"""MOVER-RIDE META-LABELER -- the discrimination test the mover_ride decomposition exposed.

CONTEXT (2026-06-10): mover_ride (1m event-clock) established two facts on u10:
  (a) MEAT EXISTS: on >=5% mover days, after a causal +1.5% trigger the oracle ceiling
      is +5.3-5.7% median (n=2662 TRAIN / 1043 OOS mover-events) -- the bar-level
      "too late" mechanism (D67) does NOT survive at 1m resolution.
  (b) The UNCONDITIONAL rider bleeds: all 18 pre-registered cells negative at 24bps
      (false-positive days swamp mover capture; win rate 37-41%).
  => The capture problem reduces to DISCRIMINATION at trigger time:
     P(ride pays | what is observable at the trigger minute). Prior is already ~0.38.

THIS TOOL is the one framework-endorsed ML use (META-LABELER on a proven trigger
mechanism -- never a signal generator, cf D16/D17) applied to that question. It is the
LAST untested angle of the mover-capture family with internal data.

PRE-REGISTRATION (stated before any result was read):
  Trigger: T = +1.5% from UTC day open (close-cross, fill next-minute open). Chosen by
    MECHANISM before this study: median fire minute ~119 on movers (early), prior
    P(mover5|fired) ~ 0.38, n large. NOT tuned on grid results (all cells were
    negative anyway; there is no winner to cherry-pick).
  Exits: trail3pct (primary; widest pre-registered trail = least noise-tagged) and
    to_close (secondary arm, reported).
  Label: y = (trail3 gross - 24bps) > 0.
  Features (ALL causal at the trigger close): minutes-to-trigger, overshoot,
    prev-day return, pre_vol (24h, ends 1h before now), realized day vol-so-far ratio,
    volume surge (cum day volume vs trailing-30d expectation pro-rata), aggressor
    imbalance day-so-far, oi_d1h/oi_d4h/oi_d24h, funding, liq_ratio,
    regime_above_sma200, BTC rel-from-day-open at the same minute, BTC 24h return,
    hour-of-day (UTC). Per-(asset,feature) z-scores from TRAIN stats only.
  Models: HistGradientBoostingClassifier (primary; native NaN handling, seed 7) +
    LogisticRegression (sanity; TRAIN-median imputation).
  Operating point: tau = 67th percentile of TRAIN predicted P (trade top ~1/3).
  OOS ALIVE GATES (all required): portfolio net/yr @24bps > 0 (FIXED split-span
    annualization, 1/n_assets sizing), breadth_pos >= 6 of the FULL universe,
    drop-top-3 jackknife net mean > 0, OOS AUC > 0.52.
  UNSEEN: untouched entirely.
Costs {0,6,12,24}bps RT; 24bps canonical. Splits as D71/mover_ride. Seeded; lineage
in JSON. No emoji (cp1252).

Run:
  python -m mining.mover_metalabel --universe u10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mining.cascade_oracle import load_panel          # noqa: E402
from mining.mover_ride import trail_exit              # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"liq_subbar_1m": "via cascade_oracle.load_panel"},
    "outputs": {"study_json": "runs/mining/mover_metalabel_<tag>_<stamp>.json"},
    "invariants": {
        "ml_as_metalabeler_only": "classifier gates a fixed causal trigger; never generates signals",
        "train_only_fit": "model, z-stats, imputation medians, tau ALL from TRAIN only",
        "causal_features": "every feature computed from data <= trigger close",
        "unseen_untouched": "no UNSEEN rows enter features, labels, or reports",
        "fixed_span_annualization": "portfolio %/yr uses the split calendar span",
    },
}

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DATA_START_MS = int(dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY_MS = 86_400_000
T_TRIG = 0.015
TRAIL_K = 0.03
GATE_COST = 0.0024
COSTS_RT = [0.0, 0.0006, 0.0012, 0.0024]
MIN_MINUTES_LEFT = 30
TAU_Q = 0.67
FEATURES = ["t2trig", "overshoot", "prevday_ret", "pre_vol", "dayvol_ratio",
            "vol_surge", "aggr_imb", "oi_d1h", "oi_d4h", "oi_d24h", "funding",
            "liq_ratio", "regime", "btc_rel", "btc_r24h", "hour_utc"]


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def extract_events(sym: str, btc_ctx: dict | None) -> list[dict]:
    """One event per asset-day where close crosses day_open*(1+T). Causal features at
    the trigger close; outcomes via the audited trail mechanics."""
    df = load_panel(sym)
    ms = df["minute_ts"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    vol = df["vol_usd"].to_numpy()
    buy = df["buy_aggr_usd"].to_numpy()
    sell = df["sell_aggr_usd"].to_numpy()
    pre_vol = df["pre_vol"].to_numpy()
    ret1m = df["ret_1m"].to_numpy()
    oi1 = df["oi_d1h"].to_numpy()
    oi4 = df["oi_d4h"].to_numpy()
    oi24 = df["oi_d24h"].to_numpy()
    fund = df["funding"].to_numpy()
    liqr = df["liq_ratio"].to_numpy()
    regime = df["regime_above_sma200"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    n = len(df)

    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)

    # trailing 30d mean daily volume (shifted: yesterday back 30 days)
    day_tot = np.array([vol[ds:de + 1].sum() for ds, de in zip(day_starts, day_ends)])
    day_close_arr = np.array([closes[de] for de in day_ends])
    avg30 = np.full(len(day_tot), np.nan)
    for i in range(len(day_tot)):
        lo = max(0, i - 30)
        if i - lo >= 10:
            avg30[i] = day_tot[lo:i].mean()

    events = []
    for di, (ds, de) in enumerate(zip(day_starts, day_ends)):
        if de - ds < 1200 or real[ds:de + 1].mean() < 0.90:
            continue
        d_open = opens[ds]
        if not np.isfinite(d_open) or d_open <= 0:
            continue
        sp = split_of(int(ms[ds]))
        if sp == "UNSEEN":
            continue
        rel = closes[ds:de + 1] / d_open - 1.0
        hit = np.flatnonzero(rel >= T_TRIG)
        if len(hit) == 0:
            continue
        m = ds + int(hit[0])
        f = m + 1
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]):
            continue
        entry = opens[f]
        if not np.isfinite(entry) or entry <= 0:
            continue
        # outcomes
        trail_g = trail_exit(opens, closes, f, de, TRAIL_K)
        to_close_g = float(closes[de] / entry - 1.0)
        d_high = float(np.max(closes[f:de + 1]))
        # causal features at close of m
        elapsed = (m - ds + 1) / 1440.0
        vsum = float(vol[ds:m + 1].sum())
        bsum, ssum = float(buy[ds:m + 1].sum()), float(sell[ds:m + 1].sum())
        dvol_sofar = float(np.nanstd(ret1m[ds:m + 1])) if m - ds >= 30 else np.nan
        btc_rel = btc_r24h = np.nan
        if btc_ctx is not None:
            bi = int((ms[m] - btc_ctx["ms0"]) // 60_000)
            if 0 <= bi < len(btc_ctx["rel"]):
                btc_rel = btc_ctx["rel"][bi]
                btc_r24h = btc_ctx["r24h"][bi]
        events.append({
            "sym": sym, "day_ms": int(ms[ds]), "t0_ms": int(ms[m]), "split": sp,
            "trail3_gross": float(trail_g), "to_close_gross": to_close_g,
            "oracle_ceiling": float(d_high / entry - 1.0),
            "runup_day": float(np.max(rel)),
            "feats": {
                "t2trig": float(m - ds),
                "overshoot": float(rel[m - ds] - T_TRIG),
                "prevday_ret": (float(day_close_arr[di - 1] / day_close_arr[di - 2] - 1.0)
                                if di >= 2 else np.nan),
                "pre_vol": float(pre_vol[m]) if np.isfinite(pre_vol[m]) else np.nan,
                "dayvol_ratio": (dvol_sofar / pre_vol[m]
                                 if np.isfinite(pre_vol[m]) and pre_vol[m] > 0
                                 and np.isfinite(dvol_sofar) else np.nan),
                "vol_surge": (vsum / (avg30[di] * elapsed)
                              if np.isfinite(avg30[di]) and avg30[di] > 0 else np.nan),
                "aggr_imb": (bsum - ssum) / (bsum + ssum) if bsum + ssum > 0 else np.nan,
                "oi_d1h": float(oi1[m]) if np.isfinite(oi1[m]) else np.nan,
                "oi_d4h": float(oi4[m]) if np.isfinite(oi4[m]) else np.nan,
                "oi_d24h": float(oi24[m]) if np.isfinite(oi24[m]) else np.nan,
                "funding": float(fund[m]) if fund[m] is not None and np.isfinite(fund[m]) else np.nan,
                "liq_ratio": float(liqr[m]) if np.isfinite(liqr[m]) else np.nan,
                "regime": (1.0 if regime[m] else 0.0) if regime[m] is not None else np.nan,
                "btc_rel": btc_rel, "btc_r24h": btc_r24h,
                "hour_utc": float((ms[m] % DAY_MS) / 3_600_000),
            },
        })
    return events


def btc_context() -> dict:
    df = load_panel("BTCUSDT")
    ms = df["minute_ts"].to_numpy()
    closes = df["close"].to_numpy()
    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    d_open = np.repeat(closes[day_starts],
                       np.diff(np.append(day_starts, len(ms))))
    # rel from day's first close (proxy for day open; identical convention each minute)
    rel = closes / d_open - 1.0
    r24h = np.full(len(closes), np.nan)
    r24h[1440:] = closes[1440:] / closes[:-1440] - 1.0
    return {"ms0": int(ms[0]), "rel": rel, "r24h": r24h}


def econ(evs: list[dict], n_assets: int, gross_key: str = "trail3_gross") -> dict:
    if not evs:
        return {"n": 0}
    g = np.array([e[gross_key] for e in evs])
    sp = evs[0]["split"]
    span = (TRAIN_END_MS - DATA_START_MS) if sp == "TRAIN" else (OOS_END_MS - TRAIN_END_MS)
    years = span / (365.25 * DAY_MS)
    out = {"n": len(g), "gross_mean": float(g.mean()), "win_gross": float((g > 0).mean())}
    for c in COSTS_RT:
        key = f"{round(c*1e4)}bps"
        out[f"net_mean_{key}"] = float((g - c).mean())
        out[f"portfolio_net_per_year_{key}"] = float((g - c).sum() / n_assets / years)
    per = {}
    for e in evs:
        per.setdefault(e["sym"], []).append(e[gross_key] - GATE_COST)
    rows = {s: float(np.mean(v)) for s, v in per.items() if len(v) >= 3}
    out["breadth_pos"] = sum(1 for v in rows.values() if v > 0)
    out["breadth_tot"] = n_assets
    out["per_asset_net24_mean"] = rows
    net = np.sort(g - GATE_COST)
    for K in (1, 3, 5):
        out[f"jk_drop_top{K}"] = float(net[:-K].mean()) if len(net) > K else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta-labeler on the 1m mover trigger")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
        tag = args.tag or "_".join(s[:-4] for s in syms[:4]).lower()
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        tag = args.tag or args.universe
    else:
        ap.error("provide --assets or --universe")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    print("loading BTC context...")
    bctx = btc_context()
    all_events = []
    for sym in syms:
        try:
            evs = extract_events(sym, None if sym == "BTCUSDT" else bctx)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        n_tr = sum(1 for e in evs if e["split"] == "TRAIN")
        print(f"[{sym}] events TRAIN/OOS = {n_tr}/{len(evs)-n_tr}")
        all_events.extend(evs)
    n_assets = len({e["sym"] for e in all_events}) or 1

    # matrices; per-(asset,feature) z from TRAIN only
    def mat(evs):
        X = np.array([[e["feats"][f] for f in FEATURES] for e in evs], dtype=float)
        y = np.array([(e["trail3_gross"] - GATE_COST) > 0 for e in evs], dtype=int)
        return X, y

    train = [e for e in all_events if e["split"] == "TRAIN"]
    oos = [e for e in all_events if e["split"] == "OOS"]
    Xtr_raw, ytr = mat(train)
    Xoo_raw, yoo = mat(oos)
    sym_tr = np.array([e["sym"] for e in train])
    sym_oo = np.array([e["sym"] for e in oos])
    Xtr = Xtr_raw.copy()
    Xoo = Xoo_raw.copy()
    for s in set(sym_tr):
        mtr, moo = sym_tr == s, sym_oo == s
        for j in range(len(FEATURES)):
            col = Xtr_raw[mtr, j]
            mu = np.nanmean(col)
            sd = np.nanstd(col)
            if np.isfinite(sd) and sd > 0:
                Xtr[mtr, j] = (col - mu) / sd
                if moo.any():
                    Xoo[moo, j] = (Xoo_raw[moo, j] - mu) / sd

    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=args.seed,
                                         early_stopping=False)
    hgb.fit(Xtr, ytr)
    p_tr = hgb.predict_proba(Xtr)[:, 1]
    p_oo = hgb.predict_proba(Xoo)[:, 1]
    med = np.nanmedian(Xtr, axis=0)
    Xtr_imp = np.where(np.isnan(Xtr), med, Xtr)
    Xoo_imp = np.where(np.isnan(Xoo), med, Xoo)
    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xtr_imp, ytr)
    pl_tr = logit.predict_proba(Xtr_imp)[:, 1]
    pl_oo = logit.predict_proba(Xoo_imp)[:, 1]

    auc = {"hgb_train": float(roc_auc_score(ytr, p_tr)),
           "hgb_oos": float(roc_auc_score(yoo, p_oo)),
           "logit_train": float(roc_auc_score(ytr, pl_tr)),
           "logit_oos": float(roc_auc_score(yoo, pl_oo))}

    tau = float(np.quantile(p_tr, TAU_Q))
    sel_tr = [e for e, p in zip(train, p_tr) if p >= tau]
    sel_oo = [e for e, p in zip(oos, p_oo) if p >= tau]
    rej_oo = [e for e, p in zip(oos, p_oo) if p < tau]

    # TRAIN decile lift (diagnostic)
    order = np.argsort(p_tr)
    deciles = []
    for d in range(10):
        idx = order[int(d / 10 * len(order)): int((d + 1) / 10 * len(order))]
        deciles.append({"decile": d + 1,
                        "mean_net24": float(np.mean([train[i]["trail3_gross"] for i in idx]) - GATE_COST),
                        "win": float(np.mean([train[i]["trail3_gross"] - GATE_COST > 0 for i in idx]))})

    res = {
        "auc": auc, "tau": tau, "tau_quantile": TAU_Q,
        "TRAIN_selected": econ(sel_tr, n_assets),
        "OOS_selected": econ(sel_oo, n_assets),
        "OOS_selected_to_close": econ(sel_oo, n_assets, "to_close_gross"),
        "OOS_rejected": econ(rej_oo, n_assets),
        "OOS_all": econ(oos, n_assets),
        "train_decile_lift": deciles,
        "feature_importance_perm_proxy": None,
    }
    g = res["OOS_selected"]
    gates = {
        "net_pos_24bps": bool(g.get("portfolio_net_per_year_24bps", -1) > 0),
        "breadth_ge6_full_denominator": bool(g.get("breadth_pos", 0) >= 6),
        "jk_drop_top3_pos": bool((g.get("jk_drop_top3") or -1) > 0),
        "auc_oos_gt_052": bool(auc["hgb_oos"] > 0.52),
    }
    gates["alive"] = all(gates.values())
    res["oos_gates"] = gates

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"tool": "mover_metalabel", "git_sha": sha, "seed": args.seed,
               "params": {"T": T_TRIG, "trail": TRAIL_K, "tau_q": TAU_Q,
                          "features": FEATURES, "label": "trail3 net@24bps > 0"},
               "n_assets": n_assets, "result": res,
               "caveats": ["u10 current membership (survivorship on absolute levels)",
                           "single pre-registered operating point; no threshold sweep"]}
    out_path = OUT / f"mover_metalabel_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    print("\n" + "=" * 78)
    print("MOVER META-LABELER (T+1.5pct trigger, trail 3pct) -- STORY")
    print("=" * 78)
    print(f"AUC: HGB train {auc['hgb_train']:.3f} -> OOS {auc['hgb_oos']:.3f} | "
          f"logit train {auc['logit_train']:.3f} -> OOS {auc['logit_oos']:.3f}")
    print("TRAIN decile lift (net@24bps, win):")
    for d in deciles:
        print(f"  d{d['decile']:>2}: {d['mean_net24']*100:+.3f}%  win {d['win']*100:.0f}%")
    for k in ["TRAIN_selected", "OOS_selected", "OOS_rejected", "OOS_all"]:
        s = res[k]
        if s.get("n"):
            print(f"{k}: n={s['n']} net24 {s['net_mean_24bps']*100:+.3f}%/ev | "
                  f"{s['portfolio_net_per_year_24bps']*100:+.2f}%/yr | win {s['win_gross']*100:.0f}% | "
                  f"breadth {s['breadth_pos']}/{s['breadth_tot']} | jk3 "
                  f"{(s.get('jk_drop_top3') or 0)*100:+.3f}%")
    sc = res["OOS_selected_to_close"]
    if sc.get("n"):
        print(f"OOS_selected (to-close exit): net24 {sc['net_mean_24bps']*100:+.3f}%/ev | "
              f"{sc['portfolio_net_per_year_24bps']*100:+.2f}%/yr")
    print(f"\nOOS GATES: {gates}")
    print(f"({time.time()-t0:.0f}s)  JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
