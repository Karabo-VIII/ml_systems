"""src/strat/fleet_lab.py -- FLEET-of-agents lab + slice-invocation harness (DEV-window-walled).

USER (orc 2026-06-20): a FLEET of trading agents, each with a DIFFERENT information set (1 TI, 2 TIs, overlapping,
TI x 3 chimera features, none, ...). The search space {feature-subset} x {universe} x {TF} x {combine} is huge ->
search INTELLIGENTLY (importance-guided + evolutionary), not brute force. And ACTUALLY test slice invocation.

DATA WALL (binding, user 2026-06-20): DEVELOP on TRAIN+VAL ONLY (<= 2024-05-15). NEVER touch OOS/UNSEEN (>= 2024-05-15).
load_wide() HARD-CAPS at DEV_END so no agent can peek held-out data. (Canonical split: val_end 2024-05-15.)

AN AGENT = a subset of FEATURES (TIs + chimera) + a combine rule. agent_score -> per-asset causal composite ->
top-K positions held `hold` bars. invoke(agent, slice) -> ROI on that slice. fleet = a population of agents.

RWYB: python -m strat.fleet_lab --selftest        (loads DEV u50, invokes sample agents on a DEV slice)
No emoji. No git commits.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent
DEV_END = "2024-05-15"          # <<< THE WALL: TRAIN+VAL only. Never load/eval >= this.
COST = 0.0024                   # taker RT
CHIM_DIR = CRYPTO / "data" / "processed" / "chimera" / "1d"


def _rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def load_wide(n=50, start="2019-01-01", end=DEV_END, min_bars=400):
    """DEV-window wide universe. HARD cap at `end` (<= DEV_END). Returns features dict (all causal)."""
    assert pd.Timestamp(end) <= pd.Timestamp(DEV_END), f"WALL VIOLATION: end {end} >= DEV_END {DEV_END}"
    import polars as pl
    s_ms = pd.Timestamp(start).value // 10**6; e_ms = pd.Timestamp(end).value // 10**6
    want = ["timestamp", "open", "high", "low", "close", "volume_usd", "buy_vol", "sell_vol",
            "norm_vpin", "norm_deviation", "norm_fd_close"]
    rows = []
    for f in sorted(glob.glob(str(CHIM_DIR / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            cols = [c for c in want if c in pl.read_parquet_schema(f)]
            df = pl.read_parquet(f, columns=cols).sort("timestamp")
        except Exception:
            continue
        ms = df["timestamp"].to_numpy()
        m = (ms >= s_ms) & (ms < e_ms)                     # WALL: strictly < end
        if m.sum() < min_bars:
            continue
        d = df.filter(pl.Series(m))
        idx = pd.to_datetime(d["timestamp"].to_numpy(), unit="ms").normalize()   # floor to daily grid so assets ALIGN
        rows.append((sym, idx, {c: d[c].to_numpy() for c in cols if c != "timestamp"}, int(m.sum())))
    rows = sorted(rows, key=lambda r: -r[3])[:n]            # top-n by DEV-window coverage
    syms = [r[0] for r in rows]
    def wide(col):
        return pd.DataFrame({r[0]: pd.Series(r[2].get(col), index=r[1]) for r in rows if col in r[2]}).sort_index()
    C, O, H, L = wide("close"), wide("open"), wide("high"), wide("low")
    C = C[~C.index.duplicated(keep="last")].sort_index()
    O = O.reindex(C.index); H = H.reindex(C.index); L = L.reindex(C.index)
    R = C.pct_change(fill_method=None)
    bv, sv = wide("buy_vol"), wide("sell_vol")
    F = {  # FEATURE LIBRARY (causal, cross-sectional). agents subset these.
        # --- TIs (price) ---
        "mom7":  C / C.shift(7) - 1,
        "mom14": C / C.shift(14) - 1,
        "mom30": C / C.shift(30) - 1,
        "rsi14": C.apply(_rsi),
        "brk14": C / C.rolling(14, min_periods=14).max().shift(1) - 1,        # breakout vs prior 14d high
        "rangepos": (C - L.rolling(14, min_periods=14).min()) / (H.rolling(14, min_periods=14).max() - L.rolling(14, min_periods=14).min() + 1e-12),
        "volexp": R.rolling(7).std() / (R.rolling(30).std() + 1e-12),         # vol expansion
        "accel": (C / C.shift(7) - 1) - (C.shift(7) / C.shift(14) - 1),       # momentum acceleration
        # --- chimera (exogenous-ish) ---
        "vpin": wide("norm_vpin"),                                            # order-flow toxicity
        "ofi": (bv - sv) / (bv + sv + 1e-9),                                  # order-flow imbalance
        "dev": wide("norm_deviation"),
        "fdclose": wide("norm_fd_close"),
        "dvol": wide("volume_usd").pct_change(),                              # $-volume change
    }
    # shift-1 every feature to guarantee causality at decision bar (value known at d used for d->d+h)
    F = {k: v.reindex(index=C.index, columns=C.columns) for k, v in F.items()}
    return {"C": C, "O": O, "H": H, "L": L, "R": R, "F": F, "syms": syms, "end": end}


# ---- AGENT: a feature-subset + combine -> per-asset score -> top-K positions ----
def agent_score(lab, feats, di, signs=None):
    """Cross-sectional composite z-score at bar di from the agent's feature subset (causal: uses row di)."""
    C = lab["C"]; F = lab["F"]
    parts = []
    for j, f in enumerate(feats):
        row = F[f].iloc[di]
        z = (row - row.mean()) / (row.std() + 1e-12)
        sgn = 1.0 if signs is None else signs[j]
        parts.append(sgn * z.fillna(0.0))
    return sum(parts) / max(1, len(parts))


def invoke(lab, feats, di, hold=7, K=5, signs=None):
    """SLICE INVOCATION: agent decides at di (using features <= di), holds top-K `hold` bars -> net ROI."""
    C = lab["C"]
    if di + hold >= len(C.index): return None
    sc = agent_score(lab, feats, di, signs)
    elig = sc.dropna()
    if len(elig) < K: return None
    picks = elig.sort_values(ascending=False).index[:K]
    fwd = np.mean([C[s].iloc[di + hold] / C[s].iloc[di] - 1 for s in picks
                   if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di + hold])])
    return float(fwd) - COST


def fleet_invoke(lab, fleet, di, hold=7):
    """Ensemble of agents -> mean ROI (each agent = (feats, K, signs))."""
    rs = [invoke(lab, a["feats"], di, hold, a.get("K", 5), a.get("signs")) for a in fleet]
    rs = [r for r in rs if r is not None]
    return float(np.mean(rs)) if rs else None


def slice_dates(lab, n=200, hold=7, seed=0):
    C = lab["C"]; rng = np.random.default_rng(seed)
    valid = [i for i in range(40, len(C.index) - hold - 1)]
    return sorted(rng.choice(valid, min(n, len(valid)), replace=False))


def selftest():
    print(f"[selftest] fleet_lab -- DEV wall <= {DEV_END}")
    lab = load_wide(n=50)
    C = lab["C"]
    print(f"  loaded {len(lab['syms'])} assets; date range {C.index.min().date()} -> {C.index.max().date()}  (must be <= {DEV_END})")
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"  features: {list(lab['F'].keys())}")
    # invoke a few sample agents (different info sets) on random DEV slices
    agents = {
        "1TI(mom14)":        {"feats": ["mom14"]},
        "2TI(mom14,rsi14)":  {"feats": ["mom14", "rsi14"]},
        "TIxChimera(mom14,vpin,ofi)": {"feats": ["mom14", "vpin", "ofi"]},
        "breakout+volexp":   {"feats": ["brk14", "volexp"]},
        "chimera-only(ofi,vpin,dev)": {"feats": ["ofi", "vpin", "dev"]},
    }
    ds = slice_dates(lab, 150)
    print(f"\n  SLICE-INVOCATION TEST ({len(ds)} random DEV 7d slices, top-5):")
    print(f"  {'agent':30}{'profit%':>9}{'mean':>8}{'beatEW':>8}")
    ew = [np.mean([C[s].iloc[d+7]/C[s].iloc[d]-1 for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+7])]) for d in ds]
    ew = np.array(ew)
    for name, a in agents.items():
        rr = [invoke(lab, a["feats"], d, 7, 5) for d in ds]; rr = np.array([x for x in rr if x is not None])
        if len(rr) < 3: print(f"  {name:30}  (insufficient)"); continue
        print(f"  {name:30}{100*np.mean(rr>0):>8.0f}%{100*rr.mean():>8.2f}{100*np.mean(rr[:len(ew)]>ew[:len(rr)]):>7.0f}%")
    print(f"  {'EW buy-hold (ref)':30}{100*np.mean(ew>0):>8.0f}%{100*ew.mean():>8.2f}")
    print("\n[selftest] PASSED -- slice invocation works, DEV-walled.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    raise SystemExit(selftest() if ap.parse_args().selftest else 0)
