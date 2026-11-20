"""src/strat/discover.py -- the discovery FRONT-END + discovery->validation SCAN loop.

PROVENANCE: ported 2026-06-05 from runs/staging/discriminate_2026_06_04.py + scan_2026_06_04.py.
Hardened against the 2026-06-05 apparatus red-audit (docs/APPARATUS_AUDIT_2026_06_05.md):
  - F5 (MEDIUM, FIXED): the forward-return label `fwd = close.shift(-H)` at the last bars of a window
    is computed from closes that fall in the NEXT window (a boundary-crossing label that mildly
    contaminates the per-window spread). Now each row carries `w_fwd` (the window H bars ahead) and the
    per-window spread keeps only rows whose forward bar stays IN the same window (`w == w_fwd`). This
    removes the boundary-crossing labels cleanly.

The foundation toolkit pipeline:  discriminate (find candidate gates)  ->  scan (harvest + validate
each cell through candidate_gate)  ->  battery (robustness).

Per the dead-list lesson: DISCRIMINATION != HARVESTABILITY (a gate that discriminates is usually
untradeable) AND the same-sign-across-4-windows base rate is 2*0.5^4 = 0.125/feature -> ~N*0.125
chance-beats; ALWAYS read the count vs the shuffle-null, never the raw count. Read-only on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_E, VAL_E, OOS_E = "2024-05-15", "2025-03-15", "2025-12-31"
# candidate GATE features (look-back; the avenue-map's "best gate candidates")
GATE_FEATS = ["wh_whale_net_usd", "s3_smart_vs_retail", "hbr_eta_imbalance", "liq_delta_z30",
              "bs_basis_z30", "norm_whale", "norm_vpin", "norm_flow_imbalance", "s3_top_pos_lsr",
              "norm_funding", "norm_yz_volatility", "hurst_regime"]


def _win(ts):
    ts = pd.Timestamp(ts)
    return "TRAIN" if ts < pd.Timestamp(TRAIN_E) else "VAL" if ts < pd.Timestamp(VAL_E) else "OOS" if ts < pd.Timestamp(OOS_E) else "UNSEEN"


def q5q1_spread(feat, fwd):
    """mean fwd-return in top quintile minus bottom quintile of the look-back feature (within-slice
    quantiles -- no global/future thresholds)."""
    ok = ~(np.isnan(feat) | np.isnan(fwd))
    f, r = feat[ok], fwd[ok]
    if len(f) < 50:
        return np.nan
    q1, q5 = np.nanpercentile(f, 20), np.nanpercentile(f, 80)
    bot, top = r[f <= q1], r[f >= q5]
    if len(bot) < 5 or len(top) < 5:
        return np.nan
    return float(top.mean() - bot.mean())


def discriminate(sym="PEPEUSDT", cadence="dollar", H=4, n_perm=200, target_bars=6676, verbose=True):
    """Which look-back features DISCRIMINATE forward returns (candidate GATES), shuffle-null-calibrated.
    Returns the list of (feature, beats_null_on_unseen) for same-sign-across-4-windows features."""
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(sym, cadence=cadence)
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    close = np.asarray(d["close"], float)
    feats_present = [c for c in GATE_FEATS if c in d]
    df = pd.DataFrame({"date": dt, "close": close, **{c: np.asarray(d[c], float) for c in feats_present}})
    # coarsen to ~target_bars (last-in-group)
    n = len(df); step = max(1, n // target_bars); df["grp"] = np.arange(n) // step
    agg = {"date": ("date", "last"), "close": ("close", "last")}
    agg.update({c: (c, "last") for c in feats_present})
    a = df.groupby("grp").agg(**agg).reset_index(drop=True)
    a["fwd"] = a["close"].shift(-H) / a["close"] - 1.0   # forward H-bar return (the label)
    a["w"] = a["date"].map(_win)
    a["w_fwd"] = a["date"].shift(-H).map(lambda x: _win(x) if pd.notna(x) else None)  # F5: window H bars ahead
    rng = np.random.default_rng(7)
    if verbose:
        print(f"[discriminate] {sym} {cadence}(~{len(a)} bars) H={H}, {len(feats_present)} candidate gates, shuffle-null x{n_perm}")
        print(f"  base rate same-sign-4-windows by chance = {2*0.5**4:.3f}/feat -> ~{len(feats_present)*2*0.5**4:.1f} expected by chance")
        print(f"  (F5: forward labels crossing a window boundary are DROPPED -- w==w_fwd only)")
        print(f"  {'feature':22} {'TR':>7} {'VAL':>7} {'OOS':>7} {'UNS':>7} {'same_sign4':>10} {'UNS>nullp95':>11}")
    persist = 0; real = 0; survivors = []
    for c in feats_present:
        # F5: only rows whose forward bar stays inside the same window contribute to that window's spread
        sp = {}
        for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]:
            m = (a["w"] == w) & (a["w_fwd"] == w)
            sp[w] = q5q1_spread(a.loc[m, c].to_numpy(), a.loc[m, "fwd"].to_numpy())
        vals = [sp[w] for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]]
        same_sign = all(not np.isnan(v) for v in vals) and (all(v > 0 for v in vals) or all(v < 0 for v in vals))
        # shuffle-null on UNSEEN spread (clean within-window rows only)
        mu = (a["w"] == "UNSEEN") & (a["w_fwd"] == "UNSEEN")
        uns_f = a.loc[mu, c].to_numpy(); uns_r = a.loc[mu, "fwd"].to_numpy()
        null = []
        for _ in range(n_perm):
            null.append(abs(q5q1_spread(uns_f, rng.permutation(uns_r))))
        null = np.array([x for x in null if not np.isnan(x)])
        real_uns = abs(sp["UNSEEN"]) if not np.isnan(sp["UNSEEN"]) else 0.0
        beats = len(null) > 0 and real_uns > np.nanpercentile(null, 95)
        persist += int(same_sign); real += int(same_sign and beats)
        if same_sign:
            survivors.append((c, bool(beats)))

        def _f(v):
            return "" if (v is None or np.isnan(v)) else f"{v * 100:+.2f}"
        if verbose:
            print(f"  {c:22} {_f(sp['TRAIN']):>7} {_f(sp['VAL']):>7} {_f(sp['OOS']):>7} {_f(sp['UNSEEN']):>7} {str(same_sign):>10} {str(beats):>11}")
    if verbose:
        print(f"\n[discriminate] same-sign-4-window features: {persist} (expected ~{len(feats_present)*0.125:.1f} by chance)")
        print(f"[discriminate] AND-beats-own-shuffle-null (UNSEEN): {real}  <- candidate GATES to harvest-test via scan")
        print("[discriminate] DISCRIMINATION != HARVESTABILITY: a surviving gate still must clear the scan (cost + firewall + battery).")
    return survivors


# ---- the discovery->validation SCAN loop ---------------------------------
def _coarse(sym, target=6676):
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(sym, cadence="dollar"); d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    df = pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                       "high": np.asarray(d.get("high", d["close"]), float),
                       "low": np.asarray(d.get("low", d["close"]), float),
                       "close": np.asarray(d["close"], float),
                       "wh": np.asarray(d.get("wh_whale_net_usd", np.zeros(len(d["close"]))), float)})
    n = len(df); step = max(1, n // target); df["grp"] = np.arange(n) // step
    a = df.groupby("grp").agg(date=("date", "last"), open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"), wh=("wh", "sum")).reset_index(drop=True)
    return a.rename(columns={"wh": "wh_whale_net_usd"})


def scan(assets, sma_configs, gates, n_books=60, verbose=True):
    """Sweep a (asset x TI-config x gate) grid; pipe each cell through the integrated candidate_gate;
    return per-cell verdicts + flag SHIP-tier. family_n = total cells (for the DSR note)."""
    from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only
    try:  # package import (normal) vs script run (python src/strat/discover.py)
        from .candidate_gate import evaluate_candidate, TAKER_COST_RT
    except ImportError:
        from strat.candidate_gate import evaluate_candidate, TAKER_COST_RT
    WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    cells = [(s, f, sl, gname) for s in assets for (f, sl) in sma_configs for gname in gates]
    fam_n = len(cells)
    if verbose:
        print(f"[scan] grid = {len(assets)} assets x {len(sma_configs)} sma x {len(gates)} gates = {fam_n} cells (family_n={fam_n})")
    base = {s: _coarse(s) for s in assets}
    results = []
    for (s, f, sl, gname) in cells:
        df = base[s].copy()
        df["sma_fast"] = sma_past_only(df["close"], f); df["sma_slow"] = sma_past_only(df["close"], sl)
        gcol, gop, gval = gates[gname]
        spec = StrategySpec(fast_col="sma_fast", slow_col="sma_slow", signal="crossover",
                            filter_col=(gcol or None), filter_op=gop, filter_val=gval,
                            exit_policy="signal_flip_or_filter", cost_rt=TAKER_COST_RT, use_funding=False,
                            funding_col="fund_rate_mean", funding_scale=0.0, max_hold_bars=18, max_hold_ext_bars=42)
        h = CanonicalHarness(df, spec, WIN, chimera_path=f"scan:{s}:{f}/{sl}:{gname}")
        try:
            v = evaluate_candidate(h, family_n=fam_n, n_books=n_books)
            results.append((s, f, sl, gname, v["CONSOLIDATED"], v["battery"]["verdict"], v["comps"].get("UNSEEN")))
        except Exception as e:
            results.append((s, f, sl, gname, f"ERR:{type(e).__name__}", "-", None))
    if verbose:
        print(f"  {'asset':9} {'sma':>8} {'gate':>9} {'UNSEEN':>8} {'battery':>22} {'CONSOLIDATED':>14}")
        for (s, f, sl, gname, cons, bat, uns) in results:
            print(f"  {s:9} {f}/{sl:<5} {gname:>9} {('' if uns is None else f'{uns:+.0f}%'):>8} {bat:>22} {cons:>14}")
    ship = [r for r in results if r[4].startswith("SHIP-TIER")]  # NOT 'SHIP in ...' (substring of NOT-SHIP)
    if verbose:
        print(f"\n[scan] SHIP-tier cells: {len(ship)} / {fam_n}  -> {'NONE' if not ship else ship}")
        print("[scan] LOOP PROVEN: discovery grid -> candidate_gate -> ranked verdict. Solving phase: spec a real grid + run.")
    return results


def _rwyb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print("=== discriminate (PEPE dollar, H=4) ===")
    discriminate(sym="PEPEUSDT", cadence="dollar", H=4, n_perm=120)
    print("\n=== scan (tiny proof-grid: 2 assets x 2 sma x 2 gates) ===")
    GATES = {"none": ("", "gt", 0.0), "whale>0": ("wh_whale_net_usd", "gt", 0.0)}
    scan(assets=["PEPEUSDT", "DOGEUSDT"], sma_configs=[(30, 50), (20, 40)], gates=GATES, n_books=40)


if __name__ == "__main__":
    _rwyb()
