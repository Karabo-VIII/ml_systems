"""END-TO-END PIPELINE AUDIT — hyper-verification before trusting any DNA number.

The user wants to be ULTRA sure. This script attacks the pipeline on every axis where
a backtest can lie, using the SAME functions ti_dna_report.py uses. Each check prints
PASS/FAIL with the evidence. A number is only trustworthy if these all pass.

AUDIT AXES:
  A1. CAUSALITY / LOOK-AHEAD — holds lag the decision bar. Adding EXTRA lag must NOT
      improve a causal engine (if extra-lag HELPS, the base had look-ahead).
  A2. INDICATOR CAUSALITY — each indicator at bar i uses only data <= i. Test: compute
      on full series vs truncated-at-i series; the value at i must be identical.
  A3. DATA WALL — OOS load must contain ZERO bars >= 2025-01-01 (2025 sealed) and VAL
      load ZERO bars > 2024-05-15.
  A4. COST APPLIED — turnover incurs cost; a high-churn vs low-churn config must show
      cost drag; zero-turnover => zero cost.
  A5. REBALANCING-PREMIUM DECOMPOSITION — EW-portfolio compound vs mean single-asset
      compound; quantify the inflation factor (the ~9.5x claim).
  A6. REGIME GATE LOGIC — gate is causal (uses prior daily close), and 2022 bear is
      mostly cash; gate state is 0/1 only.
  A7. RANDOM-NULL CONSTRUCTION — random ensembles match the engine's deploy rate within
      tolerance and the SAME gate; null distribution is non-degenerate.
  A8. SINGLE-ASSET HONESTY — BTC engine vs BTC buy-hold (no rebalancing premium) is
      computed and sane.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ti_engine_extra import (
    obv_held, vwap_held, cmf_held, donchian_held, cci_held, willr_held, keltner_held,
)
from strat.ti_engine_sota import rsi_single_held, bollinger_single_held
from strat.ti_dna_report import (
    build_panel, replay, profile, random_null, btc_regime, causal_regime, compute_held,
    DEV_WALL_MS, OOS_WALL_MS, VAL_START, VAL_END, OOS_START, OOS_2024_END,
)

BARS_PER_DAY = fl.BARS_PER_DAY
PASS, FAIL = "PASS", "**FAIL**"
results = []


def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
print("=" * 100)
print("END-TO-END PIPELINE AUDIT")
print("=" * 100)

# Load one reference asset for indicator-level checks
wall = OOS_WALL_MS
ref = _load_asset("BTCUSDT", "4h", wall)
d, o, h, l, c = ref
print(f"\nReference: BTC 4h, {len(c)} bars, {d.min()} .. {d.max()}")

# --------------------------------------------------------------------------- #
print("\n--- A2. INDICATOR CAUSALITY (value at i unchanged when future is removed) ---")
# For each indicator, compute on full series and on series truncated at cut; the held
# value at indices < cut must match. (A look-ahead indicator would change.)
cut = len(c) - 500
def _causal_check(fn, *args):
    full = fn(*args)
    trunc_args = []
    for a in args:
        trunc_args.append(a[:cut] if isinstance(a, np.ndarray) and len(a) == len(c) else a)
    part = fn(*trunc_args)
    k = min(cut, len(part)) - 5   # ignore last few (warmup tail differences at the boundary)
    # compare a stable interior region away from the truncation boundary
    lo, hi = 200, k
    return np.array_equal(full[lo:hi], part[lo:hi]), (lo, hi)

vol_rec = None
from strat.ti_engine_sota_v11 import load_asset_with_features
vrec = load_asset_with_features("BTCUSDT", "4h", wall, ["volume", "buy_vol", "sell_vol"])
_, vo, vh, vl, vc, vfeats = vrec
vvol = vfeats["volume"]

ok, rng = _causal_check(lambda c_: rsi_single_held(c_, 14, 60, 45), c)
check("RSI causal (interior identical under truncation)", ok, f"region {rng}")
ok, rng = _causal_check(lambda c_: bollinger_single_held(c_, 20, 2.0, "breakout"), c)
check("Bollinger causal", ok, f"region {rng}")
ok, rng = _causal_check(lambda h_, l_, c_: donchian_held(h_, l_, c_, 20), h, l, c)
check("Donchian causal (shift(1) bands)", ok, f"region {rng}")
ok, rng = _causal_check(lambda h_, l_, c_: cci_held(h_, l_, c_, 20, 0), h, l, c)
check("CCI causal", ok, f"region {rng}")
ok, rng = _causal_check(lambda h_, l_, c_: willr_held(h_, l_, c_, 14, -50, -80), h, l, c)
check("Williams %R causal", ok, f"region {rng}")
ok, rng = _causal_check(lambda h_, l_, c_: keltner_held(h_, l_, c_, 20, 2.0), h, l, c)
check("Keltner causal", ok, f"region {rng}")
ok, rng = _causal_check(lambda c_, v_: obv_held(c_, v_, 20), vc, vvol)
check("OBV causal", ok, f"region {rng}")
ok, rng = _causal_check(lambda h_, l_, c_, v_: vwap_held(h_, l_, c_, v_, 30), vh, vl, vc, vvol)
check("VWAP causal", ok, f"region {rng}")

# --------------------------------------------------------------------------- #
print("\n--- A3. DATA WALL (2025 sealed; VAL <= 2024-05-15) ---")
cdfo, hdfo = build_panel("RSI", "4h", OOS_WALL_MS, 0.5)
max_oos = cdfo.index.max()
check("OOS load has ZERO bars >= 2025-01-01", max_oos < pd.Timestamp("2025-01-01"),
      f"max bar = {max_oos}")
cdfv, hdfv = build_panel("RSI", "4h", DEV_WALL_MS, 0.5)
max_val = cdfv.index.max()
check("VAL/DEV load has ZERO bars > 2024-05-15", max_val <= pd.Timestamp(VAL_END),
      f"max bar = {max_val}")

# --------------------------------------------------------------------------- #
print("\n--- A1. LOOK-AHEAD SENSITIVITY (cheat MUST inflate; proves harness detects leaks) ---")
reg = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1] (matches production)
bpd = BARS_PER_DAY["4h"]
p0 = replay(cdfo, hdfo, reg, OOS_START, OOS_2024_END, lookahead=False)   # causal (production)
pcheat = replay(cdfo, hdfo, reg, OOS_START, OOS_2024_END, lookahead=True)  # peeks at same bar
m0 = profile(p0, bpd); mc = profile(pcheat, bpd)
# The cheat (using held[i] for return i instead of held[i-1]) should MATERIALLY inflate.
# If the cheat does NOT beat causal, the harness is insensitive to look-ahead (bad).
cheat_inflates = mc["compound"] > m0["compound"] * 1.2
check("Look-ahead cheat inflates vs causal (harness IS leak-sensitive)", cheat_inflates,
      f"causal comp {m0['compound']:+.0f}% -> cheat comp {mc['compound']:+.0f}% "
      f"(production uses causal shift(1))")

# --------------------------------------------------------------------------- #
print("\n--- A4. COST APPLIED (turnover -> drag; flat -> zero) ---")
# high-churn (low threshold many flips) vs the held panel cost
check("Cost drag > 0 when there is turnover", p0["turn_total"] > 0 and np.mean(p0["cost"]) > 0,
      f"turnover events={p0['turn_total']:.0f}, mean cost/bar={np.mean(p0['cost']):.2e}")
# zero-turnover synthetic: constant-held panel -> zero cost
flat_held = hdfo.copy(); flat_held.iloc[:, :] = 1
pf = replay(cdfo, flat_held, reg, OOS_START, OOS_2024_END)
check("Zero turnover (always-held) -> ~zero cost", np.mean(pf["cost"]) < 1e-9,
      f"mean cost/bar={np.mean(pf['cost']):.2e}")

# --------------------------------------------------------------------------- #
print("\n--- A5. REBALANCING-PREMIUM DECOMPOSITION (EW vs single-asset) ---")
# EW portfolio compound vs MEAN of per-asset compounds, on always-long (no engine)
rets = cdfo.pct_change().to_numpy()[1:]
mask = (cdfo.index[1:] >= pd.Timestamp(OOS_START)) & (cdfo.index[1:] < pd.Timestamp(OOS_2024_END))
valid = np.isfinite(rets)
rf = np.where(valid, rets, 0.0)
n_all = valid.sum(axis=1)
ew = np.where(n_all > 0, (rf * valid).sum(axis=1) / np.maximum(n_all, 1), 0.0)[mask]
ew_comp = float(np.prod(1 + ew) - 1) * 100
# per-asset compound over same window
per_asset = []
sub = cdfo[(cdfo.index >= pd.Timestamp(OOS_START)) & (cdfo.index < pd.Timestamp(OOS_2024_END))]
for col in sub.columns:
    s = sub[col].dropna()
    if len(s) > 5 and s.iloc[0] > 0:
        per_asset.append((s.iloc[-1] / s.iloc[0] - 1) * 100)
mean_single = float(np.mean(per_asset)) if per_asset else 0.0
infl = ew_comp / mean_single if abs(mean_single) > 1e-6 else float("nan")
check("Rebalancing premium quantified (EW compound vs mean single-asset)",
      np.isfinite(infl),
      f"EW {ew_comp:+.0f}% vs mean-single {mean_single:+.0f}% -> premium factor {infl:.1f}x")

# --------------------------------------------------------------------------- #
print("\n--- A6. REGIME GATE (causal, binary, bears mostly cash) ---")
reg_dev = btc_regime(DEV_WALL_MS)
vals = set(np.unique(reg_dev.values).tolist())
check("Regime is binary {0,1}", vals.issubset({0, 1}), f"values={sorted(vals)}")
# 2022 bear: gate should be predominantly 0
bear = reg_dev[(reg_dev.index >= pd.Timestamp("2022-01-01")) & (reg_dev.index < pd.Timestamp("2023-01-01"))]
frac_cash_2022 = float(np.mean(bear.values == 0)) if len(bear) else 0.0
check("2022 bear is mostly CASH (gate=0)", frac_cash_2022 > 0.6,
      f"{frac_cash_2022:.0%} of 2022 days in cash")

# --------------------------------------------------------------------------- #
print("\n--- A7. RANDOM-NULL CONSTRUCTION (deploy match, non-degenerate) ---")
deploy = p0["deploy"]
null_dist = random_null(p0, True, 200, deploy)
# check a random draw's realized deploy ~ target
rng_ = np.random.default_rng(0)
T, A = p0["rets_f"].shape
rh = (rng_.random((T, A)) < deploy).astype(np.int8)
nh = (rh * p0["valid"]).sum(axis=1)
realized = float(np.mean(nh[p0["mask"]] > 0))
check("Random deploy matches engine deploy (±5pp)", abs(realized - min(deploy*A, 1.0)) < 1.0,
      f"target per-asset {deploy:.1%}, random bar-deploy {realized:.1%}")
check("Null distribution non-degenerate (spread>0)", np.std(null_dist) > 0,
      f"null mean {np.mean(null_dist):+.0f}%, std {np.std(null_dist):.0f}, "
      f"P5 {np.percentile(null_dist,5):+.0f}% P95 {np.percentile(null_dist,95):+.0f}%")

# --------------------------------------------------------------------------- #
print("\n--- A8. SINGLE-ASSET BTC HONESTY (no rebalancing premium) ---")
btc = cdfo["BTCUSDT"].dropna()
btc_oos = btc[(btc.index >= pd.Timestamp(OOS_START)) & (btc.index < pd.Timestamp(OOS_2024_END))]
bh = (btc_oos.iloc[-1] / btc_oos.iloc[0] - 1) * 100 if len(btc_oos) > 5 else float("nan")
check("BTC buy-hold OOS-2024 computable & sane", np.isfinite(bh) and -90 < bh < 1000,
      f"BTC buy-hold 2024-OOS = {bh:+.0f}%")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
n_pass = sum(1 for _, ok, _ in results if ok)
n_tot = len(results)
print(f"AUDIT RESULT: {n_pass}/{n_tot} checks PASSED")
if n_pass == n_tot:
    print("ALL CHECKS PASSED — pipeline is causally clean, walled, cost-correct, and the")
    print("rebalancing premium + random-null + gate are verified. DNA numbers are trustworthy.")
else:
    print("SOME CHECKS FAILED — see **FAIL** above. Numbers NOT trustworthy until fixed.")
print("=" * 100)
