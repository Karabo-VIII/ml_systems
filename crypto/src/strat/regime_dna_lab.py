"""REGIME-DNA REALITY CHECK -- the fork that decides the whole architecture.

User framing (2026-06-11): per-asset systems, REACTIVE regime-adaptation (NOT predictive
config selection; D73 killed forecasting next-window's optimum -- this is "detect the
regime we are in NOW, run the config that historically suits THIS regime on THIS asset,
switch reactively so the strat does not decay"). Movers span all regimes, so a single
long-trend config structurally misses non-trend opportunity -> regime coverage matters.
Per-asset edge is solved FIRST; basket/correlation decoupling is a later deployment layer.

THREE QUESTIONS, ONE EXPERIMENT (predetermined configs, zero tuning; honest splits):
  Q1 DNA real?      -- do per-asset configs beat ONE pooled config-for-all, out of sample?
  Q2 Adapt helps?   -- does regime-REACTIVE switching beat a single fixed per-asset config?
  Q3 Stable/anti-decay? -- does a TRAIN-chosen (asset x regime)->config STILL hold OOS+UNSEEN,
                           or does it flip (= overfit noise, the D73 failure mode)?

SYSTEMS (all built on the 12 predetermined family configs from family_regime_map):
  SYS_C pooled    : the single config with best TRAIN mean-net POOLED across all assets,
                    applied to every asset (the "1 robust system x N assets" hypothesis).
  SYS_B per_asset : per asset, the single config with best TRAIN mean-net (asset DNA, no regime).
  SYS_A regime_sw : per (asset, regime), the config with best TRAIN mean-net; a trade counts
                    for SYS_A iff it was ENTERED while in that regime (reactive switching;
                    no mid-trade config change, no forecasting).

DISCIPLINE (fixes this morning's wave1 OOS-selection bug):
  - ALL selection on TRAIN only (< 2024-01-01). OOS [2024-01-01, 2025-07-01) is VALIDATION.
    UNSEEN [2025-07-01, end) is the test -- evaluated once, NEVER selected on.
  - Regime detected causally (data <= t; trailing percentiles, no full-sample stats).
  - Fills next-bar open; taker 0.24% RT. Trades tagged with regime-at-entry + split.
  - Honesty (r2, audit-corrected): split-boundary force-close (no cross-split hold leak);
    compare on PER-TRADE expectancy +- se (NOT sum/breadth, which reward trade-count);
    count-INVARIANT concentration = drop-top-5pct + n_eff (not a fixed drop-3); Q3
    survival vs a null drawn from the SAME in-regime eligibility, evaluated identically;
    SYS_A decomposed UP vs DOWN (DOWN = long entries below SMA150 = D58 bear-bounce
    suspects). (A matched-hold random-entry firewall is deferred to the full battery --
    NOT claimed as an in-line guard here.)
No emoji (cp1252).

Run:
  python -m strat.regime_dna_lab --universe u10 --cadence 1d --regime trend
  python -m strat.regime_dna_lab --universe u50 --cadence 1d --regime trendvol
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
from mining.family_regime_map import (  # noqa: E402  (predetermined primitives, unmodified)
    fam_signals, atr14, sma, FAMILIES, FAMILY_CLASS, COST_RT, _norm_sym, DAY_MS,
)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
MIN_TRADES = 8          # per (asset, regime, config) for selection eligibility
STOPS = (False, True)   # signal-exit and +2xATR-trail variants both in the config space


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def detect_regime(c: np.ndarray, mode: str) -> np.ndarray:
    """Causal regime state per bar (string labels). data <= t only.
    trend:    UP/DOWN by close vs SMA150.
    trendvol: {UP,DOWN} x {CALM,VOL} by SMA150 and trailing-vol vs its trailing median."""
    n = len(c)
    sma150 = sma(c, 150)
    trend = np.where(np.isfinite(sma150) & (c > sma150), "UP", "DOWN")
    if mode == "trend":
        return trend.astype(object)
    # vol axis: 30d realized vol ranked vs its own trailing 180d median (causal)
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    rv = np.full(n, np.nan)
    for i in range(30, n):
        rv[i] = np.std(ret[i - 29:i + 1])
    rvmed = np.full(n, np.nan)
    for i in range(210, n):
        rvmed[i] = np.median(rv[i - 179:i + 1])
    volstate = np.where(np.isfinite(rvmed) & (rv > rvmed), "VOL", "CALM")
    return np.array([f"{t}_{v}" for t, v in zip(trend, volstate)], dtype=object)


def simulate_tagged(name, stop, o, h, l, c, regime, split_arr) -> list[dict]:
    """Long-only; signal close t -> fill open t+1; +2xATR trail if stop. Each trade tagged
    with regime-at-entry, split-at-entry, hold length, entry index."""
    ent, exi = fam_signals(name, o, h, l, c)
    atr = atr14(h, l, c)
    n = len(c)
    trades = []
    in_pos = False
    entry_px = hw = 0.0
    e_reg = e_split = e_idx = None
    for t in range(1, n - 1):
        if not in_pos:
            if ent[t] and np.isfinite(o[t + 1]) and o[t + 1] > 0 and not np.isnan(atr[t]):
                in_pos, entry_px, hw = True, o[t + 1], c[t]
                e_reg, e_split, e_idx = regime[t], split_arr[t], t + 1
        else:
            if split_arr[t] != e_split:  # FIX(audit): force-close at split boundary -> no
                # cross-split hold leak; realized leg is fully within the entry split
                trades.append({"net": o[t] / entry_px - 1.0 - COST_RT, "regime": e_reg,
                               "split": e_split, "hold": t - e_idx, "entry": e_idx})
                in_pos = False
                continue
            hw = max(hw, c[t])
            stop_hit = stop and np.isfinite(atr[t]) and (c[t] < hw - 2.0 * atr[t])
            if exi[t] or stop_hit:
                trades.append({"net": o[t + 1] / entry_px - 1.0 - COST_RT, "regime": e_reg,
                               "split": e_split, "hold": (t + 1) - e_idx, "entry": e_idx})
                in_pos = False
    if in_pos:
        trades.append({"net": o[n - 1] / entry_px - 1.0 - COST_RT, "regime": e_reg,
                       "split": e_split, "hold": (n - 1) - e_idx, "entry": e_idx})
    return trades


def _cfgs():
    return [(f, s) for f in FAMILIES for s in STOPS]


def _drop_top_pct(a: np.ndarray, pct: float = 0.05):
    """FIX(audit): count-INVARIANT concentration test -- drop the top pct fraction
    (not a fixed 3) so systems with different trade counts compare fairly."""
    k = max(1, int(round(len(a) * pct)))
    return float(np.sort(a)[:-k].mean()) if len(a) > k else None


def _n_eff_pos(a: np.ndarray) -> float:
    """Effective number of positive contributors (1/HHI). Low = a few trades carry it."""
    w = a[a > 0]
    if len(w) == 0 or w.sum() == 0:
        return 0.0
    p = w / w.sum()
    return float(1.0 / np.sum(p ** 2))


def _stats(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0}
    a = np.array(nets)
    w, lo = a[a > 0], a[a <= 0]
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "se": float(a.std() / np.sqrt(len(a))),  # per-trade expectancy std error (audit: compare on THIS, not sum)
            "win": float((a > 0).mean()),
            "pf": float(w.sum() / -lo.sum()) if len(lo) and lo.sum() < 0 else None,
            "sum": float(a.sum()),
            "jk_drop_top5pct_mean": _drop_top_pct(a, 0.05),  # FAIR concentration metric
            "n_eff_pos": _n_eff_pos(a)}


def run(universe: str, cadence: str, regime_mode: str, seed: int) -> dict:
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    if "assets" in spec:
        syms = [a["symbol"] for a in spec["assets"]]
    else:  # u100-style inherit
        u50 = yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u50.yaml"))
        syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
        syms = [s for s in dict.fromkeys(syms) if s not in set(spec.get("excluded_assets") or [])]

    # per-asset trade book: {asset: {(fam,stop): [trades]}}
    book = {}
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence,
                                      features=["open", "high", "low", "close"])
        except Exception:
            continue
        ts = df["timestamp"].to_numpy()
        if (ts < OOS_END_MS).sum() < 250:
            continue
        o = df["open"].to_numpy().astype(float)
        h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float)
        c = df["close"].to_numpy().astype(float)
        reg = detect_regime(c, regime_mode)
        split_arr = np.array([split_of(int(t)) for t in ts], dtype=object)
        book[sym] = {cfg: simulate_tagged(cfg[0], cfg[1], o, h, l, c, reg, split_arr)
                     for cfg in _cfgs()}
    assets = list(book.keys())
    regimes = sorted({tr["regime"] for a in assets for cfg in book[a]
                      for tr in book[a][cfg] if tr["regime"] is not None})

    def trnet(sym, cfg, split, regime=None):
        return [tr["net"] for tr in book[sym][cfg]
                if tr["split"] == split and (regime is None or tr["regime"] == regime)]

    # ---- SELECTION (TRAIN ONLY) ----
    # SYS_C pooled: best config by TRAIN pooled mean-net
    pooled_train = {cfg: [tr["net"] for s in assets for tr in book[s][cfg] if tr["split"] == "TRAIN"]
                    for cfg in _cfgs()}
    pooled_cfg = max([c for c in _cfgs() if len(pooled_train[c]) >= MIN_TRADES] or _cfgs(),
                     key=lambda c: np.mean(pooled_train[c]) if pooled_train[c] else -9)
    # SYS_B per_asset: best config per asset by TRAIN mean-net
    perasset_cfg = {}
    for s in assets:
        cand = [(c, np.mean(trnet(s, c, "TRAIN"))) for c in _cfgs() if len(trnet(s, c, "TRAIN")) >= MIN_TRADES]
        perasset_cfg[s] = max(cand, key=lambda x: x[1])[0] if cand else pooled_cfg
    # SYS_A regime_sw: best config per (asset, regime) by TRAIN mean-net
    regime_cfg = {}
    for s in assets:
        for r in regimes:
            cand = [(c, np.mean(trnet(s, c, "TRAIN", r))) for c in _cfgs()
                    if len(trnet(s, c, "TRAIN", r)) >= MIN_TRADES]
            if cand:
                regime_cfg[(s, r)] = max(cand, key=lambda x: x[1])[0]

    # ---- EVALUATION (per split; selection is fixed from TRAIN) ----
    def sys_trades(system: str, split: str) -> list[float]:
        out = []
        for s in assets:
            if system == "C":
                out += trnet(s, pooled_cfg, split)
            elif system == "B":
                out += trnet(s, perasset_cfg[s], split)
            else:  # A: regime-switch -- a trade counts iff entered in its mapped regime
                for r in regimes:
                    cfg = regime_cfg.get((s, r))
                    if cfg:
                        out += trnet(s, cfg, split, r)
        return out

    def breadth(system: str, split: str) -> dict:
        per = {}
        for s in assets:
            if system == "C":
                v = trnet(s, pooled_cfg, split)
            elif system == "B":
                v = trnet(s, perasset_cfg[s], split)
            else:
                v = [x for r in regimes if regime_cfg.get((s, r))
                     for x in trnet(s, regime_cfg[(s, r)], split, r)]
            if len(v) >= 3:
                per[s] = float(np.mean(v))
        return {"pos": sum(1 for x in per.values() if x > 0), "tot": len(per)}

    systems = {}
    for name, code in [("SYS_C_pooled", "C"), ("SYS_B_per_asset", "B"), ("SYS_A_regime_sw", "A")]:
        systems[name] = {sp: _stats(sys_trades(code, sp)) for sp in ["TRAIN", "OOS", "UNSEEN"]}
        systems[name]["breadth_OOS"] = breadth(code, "OOS")
        systems[name]["breadth_UNSEEN"] = breadth(code, "UNSEEN")

    # ---- Q3 stability: do TRAIN-best (asset,regime) configs stay positive OOS / UNSEEN? ----
    # FIX(audit): observed survival is now regime-FILTERED (trnet ...,r) to match the
    # null exactly, and the null draws from the SAME per-(s,r) in-regime eligibility pool.
    surv = {"n_cells": len(regime_cfg)}
    cells_evaluable = {sp: [(s, r, cfg) for (s, r), cfg in regime_cfg.items()
                            if len(trnet(s, cfg, sp, r)) >= 3] for sp in ["OOS", "UNSEEN"]}
    for sp in ["OOS", "UNSEEN"]:
        means = [np.mean(trnet(s, cfg, sp, r)) for (s, r, cfg) in cells_evaluable[sp]]
        surv[f"{sp}_cells_evaluable"] = len(means)
        surv[f"{sp}_frac_positive"] = float(np.mean([m > 0 for m in means])) if means else None
    rng = np.random.default_rng(seed)
    # in-regime eligibility per (s,r): configs with >=MIN_TRADES TRAIN trades IN regime r
    elig_r = {(s, r): [c for c in _cfgs() if len(trnet(s, c, "TRAIN", r)) >= MIN_TRADES]
              for (s, r) in regime_cfg}
    null_surv = []
    for _ in range(200):
        flags = []
        for (s, r, _cfg) in cells_evaluable["OOS"]:        # SAME fixed evaluable cell set
            pool = elig_r[(s, r)] or [_cfg]
            cfg = pool[rng.integers(len(pool))]
            v = trnet(s, cfg, "OOS", r)                     # SAME in-regime eval
            if len(v) >= 3:
                flags.append(np.mean(v) > 0)
        if flags:
            null_surv.append(np.mean(flags))
    surv["OOS_null_frac_positive_mean"] = float(np.mean(null_surv)) if null_surv else None

    # ---- FIX(audit): SYS_A UP-vs-DOWN decomposition (DOWN cells = D58 bear-bounce
    # suspects in a long-only book; expose whether 'coverage' is the spurious half) ----
    def sys_a_by_trend(split: str, want_up: bool) -> list[float]:
        out = []
        for s in assets:
            for r in regimes:
                cfg = regime_cfg.get((s, r))
                if cfg and (r.startswith("UP") == want_up):
                    out += trnet(s, cfg, split, r)
        return out
    sysA_decomp = {sp: {"UP": _stats(sys_a_by_trend(sp, True)),
                        "DOWN": _stats(sys_a_by_trend(sp, False))}
                   for sp in ["TRAIN", "OOS", "UNSEEN"]}

    return {"universe": universe, "cadence": cadence, "regime_mode": regime_mode,
            "sysA_up_down": sysA_decomp,
            "n_assets": len(assets), "regimes": regimes,
            "selection": {"pooled_cfg": f"{pooled_cfg[0]}/{'stop' if pooled_cfg[1] else 'sig'}",
                          "per_asset_cfg": {s: f"{c[0]}/{'stop' if c[1] else 'sig'}" for s, c in perasset_cfg.items()},
                          "n_regime_cells": len(regime_cfg)},
            "systems": systems, "stability_Q3": surv}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.regime_dna_lab")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--regime", default="trend", choices=["trend", "trendvol"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    r = run(a.universe, a.cadence, a.regime, a.seed)

    print(f"## REGIME-DNA REALITY CHECK -- {a.universe} {a.cadence} -- regime={a.regime} "
          f"-- {r['n_assets']} assets, regimes={r['regimes']}")
    print(f"   pooled(SYS_C) config = {r['selection']['pooled_cfg']}; "
          f"{r['selection']['n_regime_cells']} (asset x regime) cells")
    print(f"\n   {'system':18} | {'OOS mean+-se / fair-jk / n_eff':>34} | {'UNSEEN mean+-se / n':>22}")
    for name in ["SYS_C_pooled", "SYS_B_per_asset", "SYS_A_regime_sw"]:
        d = r["systems"][name]
        def f(s):
            if not s.get("n"):
                return "-"
            jk = s.get("jk_drop_top5pct_mean")
            return (f"{s['mean']*100:+.2f}+-{s['se']*100:.2f}% jk{(jk*100 if jk is not None else 0):+.2f}% "
                    f"neff{s['n_eff_pos']:.0f} n{s['n']}")
        print(f"   {name:18} | {f(d['OOS']):>34} | {f(d['UNSEEN']):>22}")
    # SYS_A UP vs DOWN (the coverage claim's honesty check)
    ad = r["sysA_up_down"]
    print(f"\n   SYS_A by trend-regime (is 'coverage' real or the D58 DOWN half?):")
    for sp in ["OOS", "UNSEEN"]:
        up, dn = ad[sp]["UP"], ad[sp]["DOWN"]
        gu = lambda x: f"{x['mean']*100:+.2f}% (n{x['n']})" if x.get("n") else "-"
        print(f"      {sp:6}  UP: {gu(up):<18}  DOWN: {gu(dn)}")
    s = r["stability_Q3"]
    print(f"\n   Q3 STABILITY (regime-FILTERED, fair null) -- TRAIN-best cfg still positive:")
    print(f"      OOS:    {s.get('OOS_frac_positive')} of {s.get('OOS_cells_evaluable')} cells "
          f"vs random-config null {s.get('OOS_null_frac_positive_mean')}")
    print(f"      UNSEEN: {s.get('UNSEEN_frac_positive')} of {s.get('UNSEEN_cells_evaluable')} cells")
    print(f"\n   READ (fair metrics): compare PER-TRADE mean+-se (not sum/breadth); fair jk = "
          f"drop-top-5pct; DNA real iff per-cell survival > null; UNSEEN is the verdict.")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"regime_dna_{a.universe}_{a.cadence}_{a.regime}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "result": r},
              open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
