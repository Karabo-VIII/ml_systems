"""src/strat/cycle3_final_referee.py -- CYCLE-3 INDEPENDENT ADVERSARIAL REFEREE.

Does NOT trust either lane. Re-derives from mover_lab primitives ONLY:
  (1) SELECTION EDGE: block-bootstrap p + walk-forward consistency. ALPHA or BETA.
  (2) CHOP-GATE year matrix: did any exposure overlay fix 2025 without breaking 2022/bull.
  (3) FINAL DEPLOYABLE SPEC with per-year + full-cycle + maxDD.

PRE-REGISTERED (stated before running):
  - Config under test: mom14, K=5, rebal=3, per-asset-SMA200 gate (Cycle-2 deployable).
  - Null: matched-exposure random-gated-5 (same gate, same rebal cadence, random pick).
  - Decision threshold: one-sided block-bootstrap p < 0.05 to call ALPHA; else BETA.
  - Block length: 21 trading days primary; also report 10 and 42 for robustness.
  - K=3 independent derivations of the headline p (3 disjoint null-seed banks + 3 block lengths).

RWYB: python -m strat.cycle3_final_referee
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.mover_lab as ml

COST = ml.COST
OOS_S, OOS_E = "2023-01-01", "2026-06-01"


# ---------------------------------------------------------------------------
# primitives (self-contained, mirror ml.evaluate mechanics exactly)
# ---------------------------------------------------------------------------
def book_returns(W, ind):
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    return bret, pos


def win(bret, s, e):
    s = pd.Timestamp(s); e = pd.Timestamp(e)
    return bret[(bret.index >= s) & (bret.index < e)]


def comp(arr):
    return (np.prod(1.0 + arr) - 1.0) * 100.0


def mdd(arr):
    eq = np.cumprod(1.0 + arr); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100.0)


def random_gated_k(ind, K, rebal, rng):
    C = ind["C"]; g = ind["gate"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns); last = -999
    cols = list(C.columns)
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in cols if bool(g.loc[d, s])]
            if elig:
                k = min(K, len(elig))
                pick = list(rng.choice(elig, size=k, replace=False))
                W.loc[d, :] = 0.0
                for s in pick:
                    W.loc[d, s] = 1.0 / k
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


# ---------------------------------------------------------------------------
# (1) SELECTION EDGE -- block bootstrap on the daily differential
# ---------------------------------------------------------------------------
def moving_block_boot(diff_daily, block_len, n_boot, rng):
    """One-sided p = P(bootstrap mean of (strat - null) <= 0). Stationary moving-block."""
    x = np.asarray(diff_daily)
    n = len(x)
    nb = int(np.ceil(n / block_len))
    max_start = n - block_len
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=nb)
        res = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        means[b] = res.mean()
    return {
        "true_mean_day": float(x.mean()),
        "true_ann_pct": float(x.mean()) * 365 * 100,
        "p05_ann": float(np.percentile(means, 5)) * 365 * 100,
        "p95_ann": float(np.percentile(means, 95)) * 365 * 100,
        "p_onesided": float(np.mean(means <= 0.0)),
        "block_len": block_len, "n_boot": n_boot, "n_days": n,
    }


def build_null_bank(ind, K, rebal, seeds):
    """Build a bank of null book-return series for a list of seeds."""
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        Wn = random_gated_k(ind, K, rebal, rng)
        bn, _ = book_returns(Wn, ind)
        out.append(bn)
    return out


def selection_edge(ind):
    print("=" * 74)
    print("(1) SELECTION EDGE -- mom14 K5 r3 per-asset-gate vs random-gated-5")
    print("=" * 74)
    W = ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=3)
    bret, _ = book_returns(W, ind)

    # K=3 INDEPENDENT derivations: 3 disjoint null-seed banks x 3 block lengths.
    # Each derivation uses its own null bank (median series) AND its own block bootstrap rng.
    seed_banks = {
        "bankA": list(range(0, 300)),
        "bankB": list(range(1000, 1300)),
        "bankC": list(range(5000, 5300)),
    }
    derivations = []
    null_med_for_wf = None
    for di, (bname, seeds) in enumerate(seed_banks.items()):
        bank = build_null_bank(ind, K=5, rebal=3, seeds=seeds)
        null_mat = pd.DataFrame({i: bank[i] for i in range(len(bank))})
        null_med = null_mat.median(axis=1)
        if null_med_for_wf is None:
            null_med_for_wf = null_med
            null_bank_for_wf = bank
        diff = win(bret - null_med, OOS_S, OOS_E)
        block_len = [10, 21, 42][di]  # each derivation gets a different block length too
        res = moving_block_boot(diff, block_len=block_len, n_boot=3000,
                                rng=np.random.default_rng(777 + di))
        res["bank"] = bname
        derivations.append(res)
        print(f"  [{bname}, block={block_len:>2}d] true={res['true_ann_pct']:>+7.2f}%/yr  "
              f"p05={res['p05_ann']:>+7.2f}  p95={res['p95_ann']:>+7.2f}  p={res['p_onesided']:.4f}")

    ps = [d["p_onesided"] for d in derivations]
    p_med = float(np.median(ps))
    n_pass = sum(1 for p in ps if p < 0.05)
    print(f"\n  K=3 derivation p-values: {[round(p,4) for p in ps]}")
    print(f"  median p = {p_med:.4f} | derivations passing p<0.05: {n_pass}/3")
    verdict = "ALPHA" if p_med < 0.05 else "BETA"
    print(f"  >> SELECTION EDGE VERDICT: {verdict} (median block-bootstrap p={p_med:.4f})")

    # Walk-forward consistency: strategy vs null distribution per year, using the wf null bank
    print("\n  --- Walk-forward per-year (strategy vs null bankA distribution) ---")
    print(f"  {'Year':<6}{'Strat%':>10}{'NullMed%':>10}{'NullP05':>10}{'NullP95':>10}{'p_year':>9}{'Beat?':>7}")
    wf = []
    for y in [2021, 2023, 2024, 2025]:
        s_sl = win(bret, f"{y}-01-01", f"{y+1}-01-01")
        s_c = comp(s_sl.to_numpy())
        yr_nulls = np.array([comp(win(br, f"{y}-01-01", f"{y+1}-01-01").to_numpy()) for br in null_bank_for_wf])
        nm = float(np.median(yr_nulls)); p05 = float(np.percentile(yr_nulls, 5)); p95 = float(np.percentile(yr_nulls, 95))
        py = float(np.mean(yr_nulls >= s_c))
        beat = "YES" if s_c > nm else "no"
        sig = "YES" if py < 0.05 else "no"
        print(f"  {y:<6}{s_c:>10.1f}{nm:>10.1f}{p05:>10.1f}{p95:>10.1f}{py:>9.3f}{beat:>7}  sig<.05:{sig}")
        wf.append({"year": y, "strat": s_c, "null_med": nm, "p": py, "beat": beat == "YES", "sig": py < 0.05})
    n_beat = sum(1 for r in wf if r["beat"])
    n_sig = sum(1 for r in wf if r["sig"])
    print(f"  beat null median: {n_beat}/4 years | individually significant (p<.05): {n_sig}/4")

    return {"verdict": verdict, "p_med": p_med, "derivations": derivations, "wf": wf,
            "n_beat": n_beat, "n_sig": n_sig, "bret_strat": bret}


# ---------------------------------------------------------------------------
# (2) CHOP-GATE year matrix -- exposure overlays on the mom14-K5-r3 engine
# ---------------------------------------------------------------------------
def btc_market_gate_W(ind, base_W):
    """Whole book to cash when BTC < its own SMA200 (BTC-MARKET gate)."""
    C = ind["C"]
    btc_on = (C["BTCUSDT"] > ind["sma200"]["BTCUSDT"]).fillna(False)
    return base_W.mul(btc_on.astype(float), axis=0)


def breadth_scale_W(ind, base_W):
    """Scale book by fraction of assets above SMA50 (breadth)."""
    breadth = (ind["C"] > ind["sma50"]).fillna(False).mean(axis=1)
    return base_W.mul(breadth, axis=0)


def vol_target_W(ind, base_W, target=0.25):
    """Scale book to target annualised vol using trailing-20d book vol (causal, lagged)."""
    bret, _ = book_returns(base_W, ind)
    rv = bret.rolling(20, min_periods=10).std() * np.sqrt(365)
    scale = (target / rv.replace(0, np.nan)).clip(upper=1.0).shift(1).fillna(0.0)
    return base_W.mul(scale, axis=0)


def dist_ramp_W(ind, base_W, lo=1.0, hi=1.1):
    """Ramp exposure by BTC distance above its SMA200: 0 at/below lo, full at/above hi."""
    C = ind["C"]
    ratio = (C["BTCUSDT"] / ind["sma200"]["BTCUSDT"]).fillna(0.0)
    scale = ((ratio - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)
    return base_W.mul(scale, axis=0)


def per_asset_vol_gate_W(ind, base_W, target=0.80):
    """NEW lane: per-asset vol gating -- downweight each held asset by its own trailing vol.
    Caps each asset's contribution so a single calming/crashing alt doesn't dominate the book."""
    rv = ind["R"].rolling(20, min_periods=10).std() * np.sqrt(365)
    cap = (target / rv.replace(0, np.nan)).clip(upper=1.0).shift(1).fillna(0.0)
    scaled = base_W.mul(cap)
    # renormalise row-sum to <= original gross (do NOT add leverage)
    return scaled


def yearmatrix(ind, label, W):
    bret, pos = book_returns(W, ind)
    row = {"rule": label}
    for y in range(2020, 2026):
        sl = win(bret, f"{y}-01-01", f"{y+1}-01-01")
        row[str(y)] = comp(sl.to_numpy()) if len(sl) > 2 else None
    full = win(bret, "2020-01-01", OOS_E)
    row["FULL"] = comp(full.to_numpy())
    row["maxDD"] = mdd(full.to_numpy())
    row["expo"] = float(pos.sum(axis=1).mean())
    return row, bret


def chop_gate(ind):
    print("\n" + "=" * 74)
    print("(2) CHOP-GATE year matrix -- can an overlay fix 2025 w/o breaking 2022/bull?")
    print("=" * 74)
    base = ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=3)  # per-asset gate already
    btcg = btc_market_gate_W(ind, base)

    rules = {
        "(ref) per-asset-gate only": base,
        "(a) +BTC-market gate":      btcg,
        "(b) +BTC-gate +breadth":    breadth_scale_W(ind, btcg),
        "(c) +BTC-gate +voltgt25":   vol_target_W(ind, btcg, 0.25),
        "(d) +BTC-gate +dist-ramp":  dist_ramp_W(ind, btcg),
        "(e) +BTC-gate +per-asset-vol": per_asset_vol_gate_W(ind, btcg, 0.80),
    }
    rows = []
    brets = {}
    for lab, W in rules.items():
        r, b = yearmatrix(ind, lab, W)
        rows.append(r); brets[lab] = b

    cols = ["2020", "2021", "2022", "2023", "2024", "2025", "FULL", "maxDD", "expo"]
    hdr = f"  {'rule':<30}" + "".join(f"{c:>9}" for c in cols)
    print(hdr); print("  " + "-" * (28 + 9 * len(cols)))
    for r in rows:
        def f(k):
            v = r.get(k)
            if v is None: return "--"
            if k == "FULL": return f"{v:,.0f}"
            if k == "expo": return f"{v:.2f}"
            return f"{v:,.1f}"
        print(f"  {r['rule']:<30}" + "".join(f"{f(c):>9}" for c in cols))

    # decisive test: which rules keep 2022<=0 AND improve 2025 AND don't lose >50% of bull (2021)?
    print("\n  --- chop-gate decision (vs (a) BTC-market gate baseline) ---")
    a = next(r for r in rows if r["rule"].startswith("(a)"))
    for r in rows:
        if r["rule"].startswith("(ref)") or r["rule"].startswith("(a)"):
            continue
        d2025 = r["2025"] - a["2025"]
        d2021 = r["2021"] - a["2021"]
        dfull = r["FULL"] - a["FULL"]
        dd = r["maxDD"] - a["maxDD"]
        fixed = "FIXED-2025" if d2025 > 5 else "no-2025-help"
        bull_cost = f"2021 {d2021:+,.0f}pp"
        print(f"  {r['rule']:<30} 2025 {d2025:+5.1f}pp [{fixed}]  {bull_cost}  FULL {dfull:+,.0f}pp  DD {dd:+.1f}pp")

    return rows, a


# ---------------------------------------------------------------------------
# (3) FINAL DEPLOYABLE SPEC
# ---------------------------------------------------------------------------
def final_spec(ind, sel, rows):
    print("\n" + "=" * 74)
    print("(3) FINAL DEPLOYABLE SPEC")
    print("=" * 74)
    # The deployable = mom14 K5 r3 + per-asset-gate + BTC-market gate (= rule (a)).
    # Compare with the BUY-HOLD EW and the random-gated-5 + BTC-gate baseline for honesty.
    base = ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=3)
    deploy = btc_market_gate_W(ind, base)
    dr, db = yearmatrix(ind, "DEPLOY mom14-K5-r3 +per-asset +BTC gate", deploy)

    # honest control: SAME spec but random selection (random-gated-5 + BTC gate), median of 200
    rng_seeds = list(range(2000, 2200))
    ctrl_rows = []
    for sd in rng_seeds:
        rng = np.random.default_rng(sd)
        Wn = random_gated_k(ind, K=5, rebal=3, rng=rng)
        Wn = btc_market_gate_W(ind, Wn)
        b, _ = book_returns(Wn, ind)
        ctrl_rows.append({str(y): comp(win(b, f"{y}-01-01", f"{y+1}-01-01").to_numpy()) for y in range(2020, 2026)}
                         | {"FULL": comp(win(b, "2020-01-01", OOS_E).to_numpy()),
                            "maxDD": mdd(win(b, "2020-01-01", OOS_E).to_numpy())})
    ctrl = {k: float(np.median([c[k] for c in ctrl_rows])) for k in ctrl_rows[0]}

    ew = pd.DataFrame(1.0 / ind["C"].shape[1], index=ind["C"].index, columns=ind["C"].columns)
    er, _ = yearmatrix(ind, "EW buy-hold", ew)

    print(f"  {'spec':<42}" + "".join(f"{c:>9}" for c in ["2020","2021","2022","2023","2024","2025","FULL","maxDD"]))
    print("  " + "-" * 114)
    for r, nm in [(dr, dr["rule"]), (None, "  CONTROL random-gated-5 +BTC gate (med)"), (er, er["rule"])]:
        src = ctrl if r is None else r
        def f(k):
            v = src.get(k)
            if v is None: return "--"
            return f"{v:,.0f}" if k == "FULL" else f"{v:,.1f}"
        print(f"  {nm:<42}" + "".join(f"{f(c):>9}" for c in ["2020","2021","2022","2023","2024","2025","FULL","maxDD"]))

    print(f"\n  DEPLOY vs CONTROL FULL: {dr['FULL']:,.0f}% vs {ctrl['FULL']:,.0f}% "
          f"(selection adds {dr['FULL']-ctrl['FULL']:+,.0f}pp on full-cycle)")
    print(f"  DEPLOY vs EW buy-hold FULL: {dr['FULL']:,.0f}% vs {er['FULL']:,.0f}%  "
          f"(maxDD {dr['maxDD']:.1f}% vs {er['maxDD']:.1f}%)")
    return dr, ctrl, er


def main():
    t0 = time.time()
    print("loading mover_lab u10 2020-01..2026-05 (daily) ...")
    ind = ml.load("2020-01-01", "2026-06-01")
    print(f"assets={ind['C'].shape[1]} days={ind['C'].shape[0]}\n")

    sel = selection_edge(ind)
    rows, a = chop_gate(ind)
    dr, ctrl, er = final_spec(ind, sel, rows)

    print(f"\n[done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
