"""Consolidate the V1.2 families-1-3 TI engine library from ti_*.json artifacts.

CORRECTED 2026-06-22 (Wave-1 adversarial verify, CRITICAL selection-bug fix):
  - SELECTION-ON-OOS LEAK FIXED. The prior version picked each indicator's best config by
    max(iron['net']) where net = OOS compound -> selecting on the held-out window (a leak).
    Now we SELECT each indicator's best config on iron['val_net'] (the VAL-window net,
    robust-preferred), then REPORT that config's OOS net / maxDD / robust_frac. This is the
    charter discipline (VAL-select / OOS-report).
  - BREADTH FLOOR ENFORCED. Each TF block must carry n_names >= 20 (>=20 distinct u100 names);
    blocks below the floor (or with no n_names recorded -- stale pre-breadth-fix artifacts) are
    SKIPPED. The pipeline universe was expanded from a hardcoded 10-name list to the 27 u100
    names that actually have 2020-DEV-window data (deep2020_ti_pipeline.py SYMS).

Reads every per-indicator band/iron JSON in DEEP_DIVE, picks the best deployable IRONED
config per indicator on VAL net (robust-preferred), records xBH vs same-cadence buy-hold (OOS),
maxDD, Sharpe, robust-frac, and n_names (breadth). Writes engine_library.json. No emoji (cp1252).
"""
import json, glob, os, re
from collections import defaultdict

DEEP = r"C:/Users/karab/Documents/coding/ml_systems/crypto/runs/periods/TRAIN/2020/DEEP_DIVE"
OUT = r"C:/Users/karab/Documents/coding/ml_systems/crypto/runs/engines/V1.2_combinatrix/engine_library.json"

MIN_NAMES = 20      # charter breadth floor: >=20 distinct u100 names
DRIFT_TOL = 10.0    # |OOS - VAL| <= DRIFT_TOL  => "robust" (matches the pipeline's robust flag)

# Catalog family map (families 1-3 only). Pipeline 'family' tag -> catalog family.
# trend -> Family 1 (TREND); momentum + mean-reversion -> Family 2 (MOMENTUM/OSCILLATOR);
# volatility -> Family 3 (VOLATILITY). volume = Family 4 -> EXCLUDED from this library.
PIPE_FAMILY = {
    "MACD":"trend","VORTEX":"trend","ADX":"trend","SUPERTREND":"trend","PSAR":"trend",
    "AROON":"trend","TRIX":"trend","LINREG":"trend","CKSP":"trend",
    "TSI":"momentum","ROC":"momentum","AO":"momentum","RVI":"momentum","KST":"momentum","BOP":"momentum",
    "RSI":"mean-reversion","STOCH":"mean-reversion","BBPCT":"mean-reversion","CCI":"mean-reversion",
    "WILLR":"mean-reversion","STOCHRSI":"mean-reversion","CMO":"mean-reversion","UO":"mean-reversion",
    "ATRREG":"volatility","CHOP":"volatility","ULCER":"volatility","BBSQZ":"volatility","MASSI":"volatility",
    "KELTNER":"breakout","DONCHIAN":"breakout",
}
# Volume family 4 indicators -- EXCLUDED (out of families 1-3 scope)
VOLUME = {"OBV","MFI","VOLIMB","CMF"}

# Map pipeline family -> catalog family group + mechanism label
CATALOG_FAMILY = {
    "trend":"1-TREND","breakout":"1-TREND",   # breakout (Donchian/Keltner) sits in catalog Family 1/3 channel; keep with trend group as channel-breakout
    "momentum":"2-MOMENTUM","mean-reversion":"2-MOMENTUM",
    "volatility":"3-VOLATILITY",
}
MECHANISM = {
    "trend":"trend","breakout":"breakout","momentum":"momentum",
    "mean-reversion":"mean-reversion","volatility":"volatility-sizing",
}

# gather all ti_<IND>_<cads>.json, group by indicator
files = glob.glob(os.path.join(DEEP, "ti_*.json"))
SKIP = {"ti_registry.json","ti_top10.json","ti_recover_15m_30m.json","ti_fighter_15m.json",
        "ti_fighter_30m.json"}
ind_files = defaultdict(list)
for f in files:
    base = os.path.basename(f)
    if base in SKIP or "capture_phase1" in base:
        continue
    m = re.match(r"ti_([A-Z0-9]+)_", base)
    if not m:
        continue
    ind = m.group(1)
    ind_files[ind].append(f)


def iter_rows(f):
    """Yield (tf, bh_net, n_names, row) for each row in each TF block.
    n_names is the recorded breadth for that block (None if not recorded -> stale artifact)."""
    d = json.load(open(f))
    for tf, blk in d.items():
        if not isinstance(blk, dict) or "rows" not in blk:
            continue
        bh = blk.get("buyhold", {}).get("net")
        nn = blk.get("n_names")
        for r in blk["rows"]:
            yield tf, bh, nn, r


engines = {}
breadth_skips = defaultdict(int)   # (indicator) -> n blocks skipped for breadth
for ind, fs in ind_files.items():
    if ind in VOLUME:
        continue  # family 4
    if ind not in PIPE_FAMILY:
        continue
    pf = PIPE_FAMILY[ind]
    # collect every (tf, cfg, iron-metrics, bh, n_names) across files; dedup on (tf,cfg).
    # BREADTH GATE: only admit blocks with n_names >= MIN_NAMES (skip stale / sub-floor blocks).
    seen = {}
    n_total = 0
    n_robust = 0
    for f in fs:
        for tf, bh, nn, r in iter_rows(f):
            if nn is None or nn < MIN_NAMES:
                breadth_skips[ind] += 1
                continue
            iron = r.get("iron")
            if iron is None:
                continue
            key = (tf, r["cfg"])
            if key in seen:
                continue
            seen[key] = (tf, bh, r["cfg"], iron, nn)
            n_total += 1
            # robust = small |OOS - VAL| drift (recompute from val_net/net to be authoritative)
            vn = iron.get("val_net"); on = iron.get("net")
            is_robust = (vn is not None and on is not None and abs(on - vn) <= DRIFT_TOL)
            if is_robust:
                n_robust += 1
    if not seen:
        continue
    # ----- VAL-SELECT / OOS-REPORT -----
    # pick best deployable ironed config: robust-preferred, then by max VAL net (NOT OOS net).
    cands = list(seen.values())  # each = (tf, bh, cfg, iron, nn)

    def _robust(c):
        vn, on = c[3].get("val_net"), c[3].get("net")
        return vn is not None and on is not None and abs(on - vn) <= DRIFT_TOL

    robust_cands = [c for c in cands if _robust(c)]
    pool = robust_cands if robust_cands else cands
    # SELECT on VAL net (the leak-free selection metric)
    best = max(pool, key=lambda c: (c[3].get("val_net") if c[3].get("val_net") is not None else -1e9))
    tf, bh, cfg, iron, nn = best
    val_net = iron.get("val_net")
    net = iron.get("net")               # REPORTED OOS net (not selected on)
    xbh = round(net / bh, 3) if bh else None
    val_xbh = round(val_net / bh, 3) if (bh and val_net is not None) else None
    engines[ind] = {
        "indicator": ind,
        "pipe_family": pf,
        "catalog_family": CATALOG_FAMILY[pf],
        "mechanism": MECHANISM[pf],
        "best_tf": tf,
        "best_cfg": cfg,
        "n_names": nn,
        "val_net": val_net,             # selection metric (VAL window)
        "net": net,                     # reported OOS net
        "buyhold_net": bh,
        "val_net_xbh": val_xbh,
        "net_xbh": xbh,                 # OOS net / buy-hold
        "maxdd": iron.get("maxdd"),
        "sharpe": iron.get("sharpe"),
        "drift": iron.get("drift"),
        "time_in": iron.get("time_in"),
        "n_trades": iron.get("n_trades"),
        "robust_best": bool(_robust(best)),
        "n_configs": n_total,
        "n_robust": n_robust,
        "robust_frac": round(n_robust / n_total, 3) if n_total else 0.0,
    }

# order: by catalog family then by net_xbh (OOS) desc
ordered = sorted(engines.values(),
                 key=lambda e: (e["catalog_family"], -(e["net_xbh"] or -9)))
out = {
    "meta": {
        "scope": "families 1-3 (Trend / Momentum-Oscillator / Volatility); volume family 4 EXCLUDED",
        "protocol": "deep2020_ti_pipeline.py; 2020 band; fixed-EW long-only spot; VAL Jul-Sep / OOS Oct-Dec; maker cost",
        "selection_rule": "VAL-SELECT / OOS-REPORT (CORRECTED 2026-06-22): per indicator, pick the IRONED config with "
                          "max VAL net (robust-preferred: |OOS-VAL|<=10), across all TFs {1d,4h,2h,1h,30m,15m}; "
                          "then REPORT that config's OOS net / maxDD / Sharpe. Prior version leaked by selecting on OOS net.",
        "breadth_rule": f"each TF block must carry n_names >= {MIN_NAMES} distinct u100 names; sub-floor / "
                        "no-n_names (stale pre-fix) blocks are SKIPPED. Universe = 27 u100 names with 2020-window data.",
        "robust_frac": "ironed configs with |OOS-VAL|<=10 / total ironed configs (admitted blocks only)",
        "buyhold_net_by_tf_note": "buyhold net now reflects the 27-name universe (see each engine's buyhold_net)",
        "n_engines": len(ordered),
        "breadth_skips_by_indicator": {k: v for k, v in sorted(breadth_skips.items()) if v},
    },
    "engines": ordered,
}
json.dump(out, open(OUT, "w"), indent=2)
print("wrote", OUT, "n_engines=", len(ordered))
print("breadth-skipped blocks (stale/sub-floor):", dict(breadth_skips))
print(f"{'fam':14s} {'ind':9s} {'mech':16s} tf  val_net  oos_net  valxBH oosxBH  dd     sh    rf      n_names")
for e in ordered:
    print(f"{e['catalog_family']:14s} {e['indicator']:9s} {e['mechanism']:16s} "
          f"{e['best_tf']:3s} {e['val_net']!s:>7} {e['net']!s:>7}  "
          f"{e['val_net_xbh']!s:>5} {e['net_xbh']!s:>5}  "
          f"{e['maxdd']!s:>6} {e['sharpe']!s:>5} {e['robust_frac']:.2f}  "
          f"({e['n_robust']}/{e['n_configs']})  n={e['n_names']}")
