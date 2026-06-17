"""CONFIG-MAP LAB -- can a PREDETERMINED generator map (asset, week) -> MA config
well enough to net 1-5% per traded week? (user mandate 2026-06-10)

PRIORS (cited, not ignored): the RICH factor->config mapping is dead-listed four ways
(D45 oracle-decomposition 0/14; vol->config ML NULL; matched-filter REFUTED;
regime-switch adaptation NULL -- mechanism: weekly non-persistence + config
interchangeability; the in-flight weekly oracle shows the modal winner changing every
week). What those tests did NOT measure is the USER'S bar: net +1-5% per traded
move/window, per instrument -- NOT beat-passive, NOT compound-to-2x. A generator that
abstains in chop and rides slow configs in trend could meet THIS bar while failing
all prior ones. That cell is tested here, once, pre-registered.

DESIGN (per asset in u10, per cadence in {4h, 1h, 15m}):
  WEEKS: epoch-aligned UTC 7d blocks. Splits: TRAIN <2024-01-01, OOS <2025-07-01,
         UNSEEN >=2025-07-01 (UNTOUCHED -- weeks never simulated).
  ORACLE per week: best of the anchor's 26-config MA/EMA grid, each simulated
         CAUSALLY within the week (golden/death cross, next-bar-open fills, taker
         24bps RT, week-start already-long entry) -- only the config choice is
         hindsight. Imported from ti_oracle_anchor (read-only).
  GENERATORS (all causal at week start):
    M0 fixed      : per-asset TRAIN-best single config, traded every week.
    M1 persistence: trade THIS week the config that won the oracle LAST week
                    (abstain if last week's oracle was <= 0).
    M2 regime math (PREDETERMINED, zero fitting): daily close > SMA200 at week
                    start AND ER(20d) >= TRAIN-median(ER) -> trade EMA_50_200,
                    else ABSTAIN. Config + structure fixed a priori (the DNA's
                    dominant slow config; thresholds = TRAIN medians, not tuned).
    M3 learned    : GBM multiclass factors -> {top-6 TRAIN oracle-winning configs,
                    ABSTAIN}; factors = trailing 30d realized vol, ER20, SMA200
                    state, autocorr(1) 30d, 7d/30d range compression, prior-week
                    asset return, prior-week oracle-winner net; per-asset z from
                    TRAIN. Fit TRAIN, applied OOS once.
  METRICS (the user's bar): per TRADED week net distribution (median/mean), the
    +1-5% band check, % weeks traded, capture-vs-oracle, per-asset breadth
    (>=6/10 positive median), drop-top-3 jackknife, TRAIN -> OOS.
No emoji (cp1252). Seeded. Lineage in JSON.

Run:
  python -m mining.config_map_lab --universe u10 --cadences 4h,1h,15m
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from strat.ti_oracle_anchor import (  # noqa: E402  (read-only import of the anchor)
    precompute_configs, causal_ma_long_return,
)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"chimera_ohlc": "ChimeraLoader per cadence + 1d for factors"},
    "outputs": {"study_json": "runs/mining/config_map_lab_<tag>_<stamp>.json"},
    "invariants": {
        "causal_generators": "every generator decision uses data strictly before the week",
        "anchor_sim": "trading sim imported from ti_oracle_anchor (causal, next-bar fills, 24bps RT)",
        "train_only_fit": "M0 selection, M2 thresholds, M3 model + z-stats from TRAIN only",
        "unseen_untouched": "UNSEEN weeks never simulated",
        "user_bar": "verdict keyed to median net per traded week in [+1%, +5%]",
    },
}

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
M2_CONFIG = "EMA_50_200"   # predetermined (dominant slow DNA); never tuned here
TOP_K = 6


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def efficiency_ratio(closes: np.ndarray, n: int = 20) -> float:
    if len(closes) < n + 1:
        return np.nan
    seg = closes[-(n + 1):]
    direction = abs(seg[-1] - seg[0])
    path = np.sum(np.abs(np.diff(seg)))
    return float(direction / path) if path > 0 else 0.0


def asset_weeks(sym: str, cadence: str) -> dict:
    """Per-week oracle table + causal factors for one (asset, cadence)."""
    df = ChimeraLoader().load(sym, cadence=cadence,
                              features=["open", "high", "low", "close"])
    ts = df["timestamp"].to_numpy()
    o = df["open"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    n = len(c)
    configs = precompute_configs(c)
    labels = list(configs.keys())

    dd = ChimeraLoader().load(sym, cadence="1d", features=["close"])
    dts = dd["timestamp"].to_numpy()
    dc = dd["close"].to_numpy().astype(np.float64)

    week_ids = ts // WEEK_MS
    uniq = np.unique(week_ids)
    rows = []
    for w in uniq:
        idx = np.flatnonzero(week_ids == w)
        ws, we = int(idx[0]), int(idx[-1]) + 1
        # full week + enough MA warmup history before it
        if we - ws < {"4h": 40, "1h": 160, "15m": 640}[cadence] or ws < 210:
            continue
        w_start_ms = int(w) * WEEK_MS
        sp = split_of(w_start_ms)
        if sp == "UNSEEN":
            continue  # untouched
        # oracle: net of EVERY config this week (causal sim; choice = hindsight)
        nets = np.array([causal_ma_long_return(*configs[lab], o, ws, we)
                         for lab in labels])
        bi = int(np.argmax(nets))
        # causal factors from STRICTLY BEFORE the week (daily frame)
        dmask = dts < w_start_ms
        dhist = dc[dmask]
        if len(dhist) < 210:
            continue
        sma200 = float(np.mean(dhist[-200:]))
        rets30 = np.diff(np.log(dhist[-31:]))
        rows.append({
            "week_ms": w_start_ms, "split": sp, "ws": ws, "we": we,
            "oracle_cfg": labels[bi], "oracle_net": float(nets[bi]),
            "nets": {lab: float(v) for lab, v in zip(labels, nets)},
            "week_ret": float(c[we - 1] / o[ws] - 1.0),
            "factors": {
                "above_sma200": float(dhist[-1] > sma200),
                "er20": efficiency_ratio(dhist, 20),
                "vol30": float(np.std(rets30)) if len(rets30) >= 10 else np.nan,
                "ac1_30": (float(np.corrcoef(rets30[:-1], rets30[1:])[0, 1])
                           if len(rets30) >= 11 else np.nan),
                "range_comp": (float((np.max(dhist[-7:]) - np.min(dhist[-7:]))
                               / (np.max(dhist[-30:]) - np.min(dhist[-30:]) + 1e-12))
                               if len(dhist) >= 30 else np.nan),
                "prev_ret_7d": (float(dhist[-1] / dhist[-8] - 1.0)
                                if len(dhist) >= 8 else np.nan),
            },
        })
    # chain prior-week oracle info (causal at trade time: PRIOR week is past)
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 and rows[i - 1]["week_ms"] == r["week_ms"] - WEEK_MS else None
        r["prev_oracle_cfg"] = prev["oracle_cfg"] if prev else None
        r["prev_oracle_net"] = prev["oracle_net"] if prev else None
    return {"sym": sym, "labels": labels, "rows": rows}


# ----------------------------------------------------------------- generators

def decide_all(assets: list[dict], seed: int) -> None:
    """Attach M0/M1/M2/M3 decisions (config label or None=abstain) to every row."""
    # M0: per-asset TRAIN-best fixed config
    for a in assets:
        tr = [r for r in a["rows"] if r["split"] == "TRAIN"]
        if not tr:
            continue
        means = {lab: float(np.mean([r["nets"][lab] for r in tr])) for lab in a["labels"]}
        a["m0_cfg"] = max(means, key=means.get)
        for r in a["rows"]:
            r["M0"] = a["m0_cfg"]
            # M1: last week's oracle winner (abstain if none/<=0)
            r["M1"] = (r["prev_oracle_cfg"]
                       if r["prev_oracle_cfg"] and (r["prev_oracle_net"] or 0) > 0 else None)

    # M2: predetermined regime math; ER threshold = TRAIN median (pooled, not tuned)
    er_train = [r["factors"]["er20"] for a in assets for r in a["rows"]
                if r["split"] == "TRAIN" and np.isfinite(r["factors"]["er20"])]
    er_med = float(np.median(er_train))
    for a in assets:
        for r in a["rows"]:
            f = r["factors"]
            r["M2"] = (M2_CONFIG if (f["above_sma200"] > 0.5
                                     and np.isfinite(f["er20"]) and f["er20"] >= er_med)
                       else None)

    # M3: learned factor->config (top-K configs by TRAIN oracle-win frequency + ABSTAIN)
    from collections import Counter
    from sklearn.ensemble import HistGradientBoostingClassifier
    win_counts = Counter(r["oracle_cfg"] for a in assets for r in a["rows"]
                         if r["split"] == "TRAIN" and r["oracle_net"] > 0)
    top_cfgs = [c for c, _ in win_counts.most_common(TOP_K)]
    classes = top_cfgs + ["ABSTAIN"]

    feat_names = ["above_sma200", "er20", "vol30", "ac1_30", "range_comp", "prev_ret_7d",
                  "prev_oracle_net"]

    def feats(r):
        f = r["factors"]
        return [f["above_sma200"], f["er20"], f["vol30"], f["ac1_30"], f["range_comp"],
                f["prev_ret_7d"],
                r["prev_oracle_net"] if r["prev_oracle_net"] is not None else np.nan]

    Xtr, ytr = [], []
    for a in assets:
        for r in a["rows"]:
            if r["split"] != "TRAIN":
                continue
            cand = {lab: r["nets"][lab] for lab in top_cfgs}
            best = max(cand, key=cand.get)
            label = best if cand[best] > 0.005 else "ABSTAIN"  # floor: cost + margin
            Xtr.append(feats(r))
            ytr.append(classes.index(label))
    Xtr = np.array(Xtr, dtype=float)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed,
                                         early_stopping=False)
    clf.fit(Xtr, np.array(ytr))
    for a in assets:
        for r in a["rows"]:
            pred = classes[int(clf.predict(np.array([feats(r)], dtype=float))[0])]
            r["M3"] = None if pred == "ABSTAIN" else pred
    return None


# ----------------------------------------------------------------- evaluation

def evaluate(assets: list[dict], gen: str, split: str) -> dict:
    """User-bar metrics for one generator on one split."""
    traded, all_weeks, per_asset = [], 0, {}
    for a in assets:
        for r in a["rows"]:
            if r["split"] != split:
                continue
            all_weeks += 1
            cfg = r.get(gen)
            if cfg is None:
                continue
            net = r["nets"][cfg]
            cap = net / r["oracle_net"] if r["oracle_net"] > 0.005 else None
            traded.append({"sym": a["sym"], "net": net, "cap": cap,
                           "week_ms": r["week_ms"]})
            per_asset.setdefault(a["sym"], []).append(net)
    if not traded:
        return {"n_traded": 0, "n_weeks": all_weeks}
    nets = np.array([t["net"] for t in traded])
    caps = [t["cap"] for t in traded if t["cap"] is not None]
    med = float(np.median(nets))
    rows = {s: float(np.median(v)) for s, v in per_asset.items() if len(v) >= 5}
    srt = np.sort(nets)
    # the user's bar is per-MOVE: weeks where the chosen config never opened a
    # position (net exactly 0) are not moves -- report the nonzero-only median too
    nz = nets[np.abs(nets) > 1e-12]
    return {
        "n_traded": len(nets), "n_weeks": all_weeks,
        "n_actual_trades": int(len(nz)),
        "net_median_actual_trades": float(np.median(nz)) if len(nz) else None,
        "in_user_band_actual_trades": bool(len(nz) and 0.01 <= float(np.median(nz)) <= 0.05),
        "pct_weeks_traded": round(len(nets) / max(1, all_weeks) * 100, 1),
        "net_median": med, "net_mean": float(nets.mean()),
        "win_rate": float((nets > 0).mean()),
        "in_user_band_1_5pct": bool(0.01 <= med <= 0.05),
        "capture_median": float(np.median(caps)) if caps else None,
        "breadth_pos_median": sum(1 for v in rows.values() if v > 0),
        "breadth_tot": len(assets),
        "jk_drop_top3_mean": float(srt[:-3].mean()) if len(srt) > 3 else None,
        "per_asset_median": rows,
        "sum_net": float(nets.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="config-map lab: generators vs weekly oracle")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--cadences", default="4h,1h,15m")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
        tag = "_".join(s[:-4] for s in syms[:3]).lower()
    else:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        tag = args.universe
    cadences = args.cadences.split(",")

    report = {}
    for cad in cadences:
        t0 = time.time()
        assets = []
        for sym in syms:
            try:
                assets.append(asset_weeks(sym, cad))
            except Exception as ex:
                print(f"[{cad}][{sym}] SKIP: {type(ex).__name__}: {str(ex)[:90]}")
        if not assets:
            continue
        decide_all(assets, args.seed)
        cad_rep = {"oracle": {}, "generators": {}}
        for sp in ["TRAIN", "OOS"]:
            o_nets = [r["oracle_net"] for a in assets for r in a["rows"] if r["split"] == sp]
            cad_rep["oracle"][sp] = {
                "n_weeks": len(o_nets),
                "oracle_net_median": float(np.median(o_nets)) if o_nets else None,
                "oracle_net_mean": float(np.mean(o_nets)) if o_nets else None,
                "pct_weeks_oracle_pos": (round(float(np.mean(np.array(o_nets) > 0)) * 100, 1)
                                          if o_nets else None),
            }
            for gen in ["M0", "M1", "M2", "M3"]:
                cad_rep["generators"][f"{gen}|{sp}"] = evaluate(assets, gen, sp)
        cad_rep["m0_choices"] = {a["sym"]: a.get("m0_cfg") for a in assets}
        report[cad] = cad_rep
        print(f"[{cad}] done ({time.time()-t0:.0f}s)")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"tool": "config_map_lab", "git_sha": sha, "seed": args.seed,
               "params": {"cadences": cadences, "m2_config": M2_CONFIG, "top_k": TOP_K,
                          "user_bar": "median net per traded week in [1%,5%]"},
               "priors_cited": ["D45", "vol->config NULL", "matched-filter REFUTED",
                                 "regime-switch NULL", "weekly oracle non-persistence (prelim)"],
               "report": report}
    out_path = OUT / f"config_map_lab_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\n" + "=" * 104)
    print("CONFIG-MAP LAB -- can a causal generator hit the user bar (median net +1..5% per traded week)?")
    print("=" * 104)
    for cad, rep in report.items():
        print(f"\n[{cad}]  weekly ORACLE: " + " | ".join(
            f"{sp}: med {rep['oracle']['{0}'.format(sp)]['oracle_net_median']*100:+.2f}% "
            f"({rep['oracle'][sp]['pct_weeks_oracle_pos']}% wks>0)"
            for sp in ["TRAIN", "OOS"] if rep["oracle"][sp]["n_weeks"]))
        hdr = (f"  {'gen|split':<10} {'traded':>12} {'med net':>8} {'mean':>8} {'win':>5} "
               f"{'capture':>8} {'breadth':>8} {'jk3':>8} {'USER BAND':>9}")
        print(hdr)
        for key, s in rep["generators"].items():
            if s.get("n_traded", 0) == 0:
                print(f"  {key:<10} {'0 (abstained all)':>12}")
                continue
            mnz = s.get("net_median_actual_trades")
            print(f"  {key:<10} {s['n_traded']:>5}/{s['n_weeks']:<5} "
                  f"{s['net_median']*100:+7.2f}% {s['net_mean']*100:+7.2f}% "
                  f"{s['win_rate']*100:4.0f}% "
                  f"{(s['capture_median'] if s['capture_median'] is not None else float('nan')):8.2f} "
                  f"{s['breadth_pos_median']:>4}/{s['breadth_tot']:<3} "
                  f"{(s['jk_drop_top3_mean'] or 0)*100:+7.2f}% "
                  f"{'YES' if s['in_user_band_1_5pct'] else 'no':>9}"
                  f"  | actual-trades n={s.get('n_actual_trades')} med "
                  f"{(mnz*100 if mnz is not None else float('nan')):+.2f}% "
                  f"band={'YES' if s.get('in_user_band_actual_trades') else 'no'}")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
