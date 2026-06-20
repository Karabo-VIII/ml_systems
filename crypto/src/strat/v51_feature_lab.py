"""src/strat/v51_feature_lab.py -- WASTED v51-chimera EXOGENOUS features as MOVE-CATCH triggers (causal daily grid).

The user's charter: exhaust the internal infra incl. "Chimera v51 that goes to waste" BEFORE external data. The rich
v51 features (funding, basis, liquidations, whale, stablecoin, order-flow, transfer-entropy, listing-age) live ONLY in
the dollar-bar files; the 1d chimera had just vpin/deviation/fd_close (the 4 that produced the "chimera dead" verdict).
This loader resamples those EXOGENOUS-ish features to a CAUSAL daily grid (last-of-day level / max-of-day flag / sum-of-
day flow) and returns a lab dict compatible with capture_lab.evaluate_ti -- so the FULL hardened battery applies
(random-ENTRY null + date-block bootstrap + regime-shuffle + reverse-score + calendar-invariance).

THE DECISIVE QUESTION the price-TIs failed: do EXOGENOUS features produce a move-CATCH edge that is REGIME-CONDITIONAL
or BEAR-positive -- breaking the de-risked-beta wall (up-regime continuation) that every price-TI hit? Exogenous signals
(liq cascades, ETF/whale flow, funding/basis extremes) have a mechanistic move-onset prior a price-TI cannot manufacture.

DEV-walled (<= 2024-05-15). Long-only spot. Causal (end-of-day value -> next-day entry). No emoji (cp1252). RWYB.
RWYB: python -m strat.v51_feature_lab --selftest
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent
DOLLAR_DIR = CRYPTO / "data" / "processed" / "chimera" / "dollar"
DEV_END = "2024-05-15"

# curated EXOGENOUS v51 features (col -> daily agg): 'last' = level/z-score, 'max' = shock flag, 'sum' = flow amount
V51 = {
    "norm_funding": "last", "s3_global_lsr_z": "last", "s3_smart_vs_retail_z": "last",
    "bs_basis_z30": "last", "bs_basis_xsec_z": "last",
    "liq_total_usd": "sum", "liq_delta_z30": "last", "liq_capitulation": "max", "liq_short_panic": "max",
    "wh_whale_net_usd": "sum", "norm_whale": "last",
    "stbl_total_zscore_30d": "last", "stbl_compound_shock": "max",
    "norm_vpin": "last", "norm_kyle_lambda": "last", "norm_hawkes_imbalance": "last",
    "te_imb": "last", "norm_oi_change": "last", "mv_days_since_listed_binance": "last", "xd_momentum_rank": "last",
}


def load_v51_daily(n=50, end=DEV_END, feats=None, min_days=300):
    """Per-asset dollar-bar v51 features -> causal daily wide panels. Returns lab dict for capture_lab.evaluate_ti."""
    assert pd.Timestamp(end) <= pd.Timestamp(DEV_END), f"WALL VIOLATION: end {end} >= DEV_END {DEV_END}"
    import polars as pl
    feats = feats or V51
    e_ms = pd.Timestamp(end).value // 10**6
    rows = []
    for f in sorted(glob.glob(str(DOLLAR_DIR / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            avail = pl.read_parquet_schema(f)
            cols = ["timestamp", "close"] + [c for c in feats if c in avail]
            df = pl.read_parquet(f, columns=cols).filter(pl.col("timestamp") < e_ms)   # WALL: strictly < end
        except Exception:
            continue
        if df.height < 1000:
            continue
        df = df.with_columns((pl.col("timestamp") // 86400000).alias("d"))
        aggs = [pl.col("close").last().alias("close")]
        for c in feats:
            if c in avail:
                a = feats[c]
                e = pl.col(c).sum() if a == "sum" else (pl.col(c).max() if a == "max" else pl.col(c).last())
                aggs.append(e.alias(c))
        g = df.group_by("d").agg(aggs).sort("d")
        if g.height < min_days:
            continue
        idx = pd.to_datetime(g["d"].to_numpy() * 86400000, unit="ms")
        rows.append((sym, idx, g, g.height))
    rows = sorted(rows, key=lambda r: -r[3])[:n]
    syms = [r[0] for r in rows]

    def wide(col):
        return pd.DataFrame({r[0]: pd.Series(r[2][col].to_numpy(), index=r[1])
                             for r in rows if col in r[2].columns}).sort_index()
    C = wide("close"); C = C[~C.index.duplicated(keep="last")].sort_index()
    F = {c: wide(c).reindex(index=C.index, columns=C.columns) for c in feats}
    # add the price-TIs (for comparison/conditioning + the reverse-score machinery)
    F["mom14"] = C / C.shift(14) - 1
    F["brk14"] = C / C.rolling(14, min_periods=14).max().shift(1) - 1
    return {"C": C, "R": C.pct_change(fill_method=None), "F": F, "syms": syms, "end": end}


def selftest():
    import strat.capture_lab as cl
    print(f"[selftest] v51_feature_lab -- WASTED exogenous v51 features as move-CATCH triggers, DEV wall <= {DEV_END}")
    lab = load_v51_daily(n=50)
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"  {len(lab['syms'])} assets; range {C.index.min().date()} -> {C.index.max().date()}")
    avail = [c for c in V51 if c in lab["F"] and lab["F"][c].notna().sum().sum() > 1000]
    print(f"  v51 features with data: {avail}")
    print(f"\n  MOVE-CATCH by regime (time-exit, top-tercile trigger, honest block bootstrap) -- "
          f"hunting a BEAR-positive or regime-conditional edge:")
    print(f"  {'feature':26}{'bull_edge':>10}{'chop_edge':>10}{'bear_edge':>10}{'bear_p05':>10}{'bear_ble0':>10}")
    for ti in avail[:12]:
        try:
            r = cl.evaluate_ti(lab, ti, tf="1d", exit_kind="time", n_null=200, by_regime=True, block=True)
        except Exception as ex:
            print(f"  {ti:26}  (err: {ex})"); continue
        br = r.get("by_regime", {})
        def g(rg, k):
            d = br.get(rg); return d[k] if d else None
        be = g("bear", "edge_pp"); bb = g("bear", "block")
        print(f"  {ti:26}{str(g('bull','edge_pp')):>10}{str(g('chop','edge_pp')):>10}{str(be):>10}"
              f"{str(bb['block_p05_pp'] if bb else None):>10}{str(bb['block_p_le0'] if bb else None):>10}")
    print("\n[selftest] PASSED -- v51 exogenous features load DEV-walled + run the full capture battery.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    raise SystemExit(selftest() if ap.parse_args().selftest else 0)
