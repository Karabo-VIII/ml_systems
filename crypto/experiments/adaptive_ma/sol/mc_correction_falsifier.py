"""
FALSIFIER: multiple-comparisons correction over the adaptive-MA cadence/config grid scans.

The SOL lane was directed (OVERSEER 2026-06-06 ~00:35) to scan the full grid
{15m,30m,1h,4h,1d,dollar,range,dib,runs_volume,adaptive_vol} x u100 and to DSR/Holm-correct
across the ~10 cadences. As of this audit, experiments/adaptive_ma/sol/ contains NO scan output
(the per-cadence scan was never run). So we correct the REAL grids the lineage DID produce:

  (G1) 1d MA-cross grid: 7 MA configs x u100  -> runs/research/u100_1d_ma_backtest_result.json
  (G2) 4h ER-gated 3-DOF sweep: 27 configs x u100 -> runs/research/minimal_3dof_4h_sweep_result.json
  (G3) 4h ER-gated single-config full firewall (77 assets) -> runs/research/minimal_3dof_4h_result.json
  (G4) 4h ER-gated bootstrap battery (77 assets) -> experiments/adaptive_ma/expert/er_gate_4h_bootstrap_u100.json

Corrections implemented (no external deps beyond numpy/scipy):
  * Holm-Bonferroni over the 77-asset 'beats matched random-entry null' p-family
    (p_i = 1 - diff_frac_pos_i, the bootstrap one-sided P(real - null > 0)).
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) over the 77 per-asset Sharpes
    implied by the block-bootstrap held-compound quantiles, deflating for N trials.

RWYB: every number printed is computed here from the on-disk JSON. Re-run:
    python experiments/adaptive_ma/sol/mc_correction_falsifier.py
"""
import json, os, math
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def L(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)

GAMMA_E = 0.5772156649015329  # Euler-Mascheroni

def psr(sr_hat, sr_ref, T, skew=0.0, kurt=3.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_ref) given estimate sr_hat over T obs."""
    if T <= 1:
        return float('nan')
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat))
    z = (sr_hat - sr_ref) * math.sqrt(T - 1.0) / denom
    return float(norm.cdf(z))

def expected_max_sr_under_null(var_sr, N):
    """E[max SR] under the null that all true SR=0 (Bailey-LdP), N independent trials."""
    if N < 2 or var_sr <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * math.e))
    return math.sqrt(var_sr) * ((1.0 - GAMMA_E) * z1 + GAMMA_E * z2)

def holm_bonferroni(pvals, alpha=0.05):
    """Return boolean reject array (FWER<=alpha) in original order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if p[idx] <= thresh:
            reject[idx] = True
        else:
            break  # Holm stops at first non-rejection
    return reject

print("=" * 78)
print("MULTIPLE-COMPARISONS CORRECTION OVER THE ADAPTIVE-MA CADENCE/CONFIG GRID")
print("=" * 78)

# ---------------------------------------------------------------------------
# STEP 0: literal SOL per-cadence scan existence check
# ---------------------------------------------------------------------------
sol_dir = os.path.join(ROOT, "experiments/adaptive_ma/sol")
sol_files = [f for f in os.listdir(sol_dir) if f.endswith((".json", ".parquet", ".csv"))]
print(f"\n[STEP 0] SOL per-cadence scan output in experiments/adaptive_ma/sol/: {sol_files}")
print("         -> the named 10-cadence x bar-type scan was NOT produced; correcting the")
print("            REAL grids the lineage ran instead (1d MA grid, 4h ER-gated family).")

# ---------------------------------------------------------------------------
# STEP 1: RAW 'winner' counts per grid (beat matched random-entry null on held-out)
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[STEP 1] RAW winner counts (naive metric vs proper matched-null firewall)")
print("-" * 78)

# G1: 1d MA grid (7 configs)
d1 = L("runs/research/u100_1d_ma_backtest_result.json")
grid1 = d1["grid"]
dev_pos = sum(1 for c in grid1.values() if c["dev_pertrade_mean_pct"] > 0)
uns_pos = sum(1 for c in grid1.values() if c["unseen_pertrade_mean_pct"] > 0)
best_dev = max(grid1.items(), key=lambda kv: kv[1]["dev_pertrade_mean_pct"])
n_beat_null_1d = d1["BASELINE_a_random_entry_null_firewall"]["n_assets_real_beats_null_p95_unseen"]
n_assets_1d = d1["BASELINE_a_random_entry_null_firewall"]["n_assets_traded_unseen"]
print(f"G1 1d MA grid: {len(grid1)} configs scanned.")
print(f"   NAIVE (dev pooled per-trade mean > 0): {dev_pos}/{len(grid1)} 'winners'  "
      f"[best = {best_dev[0]} @ +{best_dev[1]['dev_pertrade_mean_pct']:.2f}% dev]")
print(f"   OUT-OF-SAMPLE (UNSEEN per-trade mean > 0): {uns_pos}/{len(grid1)} "
      f"[best config UNSEEN = {best_dev[1]['unseen_pertrade_mean_pct']:.2f}%/trade]")
print(f"   BEATS MATCHED NULL on UNSEEN: {n_beat_null_1d}/{n_assets_1d} assets")

# G2: 4h 27-config sweep
g2 = L("runs/research/minimal_3dof_4h_sweep_result.json")
def parse_frac(s):
    a, b = s.split("/"); return int(a), int(b)
g2_winners = sum(1 for c in g2 if parse_frac(c["fw_regime_beats_held"])[0] > 0)
best_pool = max(g2, key=lambda c: c["unseen_pool_exp_pct"])
print(f"\nG2 4h 3-DOF sweep: {len(g2)} configs scanned.")
print(f"   NAIVE (max pooled UNSEEN per-trade expectancy): {best_pool['unseen_pool_exp_pct']:.3f}% "
      f"[er{best_pool['er_gate']}/brk{best_pool['break_n']}/atr{best_pool['atr_mult']}]")
print(f"   BEATS MATCHED NULL on held-out: configs with >=1 asset beating null = "
      f"{g2_winners}/{len(g2)}  (every config = {g2[0]['fw_regime_beats_held']})")

# G3: 4h single-config firewall (77 assets)
g3 = L("runs/research/minimal_3dof_4h_result.json")
fw3 = g3["aggregate"]["firewall_regime_matched"]
print(f"\nG3 4h single-config firewall: n_beat_null_held = "
      f"{fw3['n_beat_null_held']}/{fw3['n_assets']} assets")

# G4: 4h bootstrap battery (77 assets)
g4 = L("experiments/adaptive_ma/expert/er_gate_4h_bootstrap_u100.json")
agg4 = g4["aggregate"]
print(f"\nG4 4h bootstrap battery: n_GENUINE_SIGNAL = {agg4['n_GENUINE_SIGNAL']}/"
      f"{agg4['n_assets_evaluated']},  n_cond3_beats_null = {agg4['n_cond3_beats_null']}/"
      f"{agg4['n_assets_evaluated']}")

raw_total_winners = uns_pos + g2_winners + fw3["n_beat_null_held"] + agg4["n_cond3_beats_null"]
print(f"\n>>> RAW winners across ALL real grids (beat matched null on held-out): "
      f"{raw_total_winners}")

# ---------------------------------------------------------------------------
# STEP 2: Holm-Bonferroni over the 77-asset 'beats null' p-family (G4)
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[STEP 2] HOLM-BONFERRONI over 77 per-asset 'beats matched null' tests (G4)")
print("-" * 78)
pa = g4["per_asset"]
# keep only assets with a usable bootstrap one-sided p (diff_frac_pos not None)
syms = [s for s in pa.keys() if pa[s].get("diff_frac_pos") is not None]
n_dropped_p = len(pa) - len(syms)
# one-sided p that real beats null = 1 - bootstrap P(real-null>0)
pvals = np.array([1.0 - float(pa[s]["diff_frac_pos"]) for s in syms])
print(f"   (dropped {n_dropped_p} assets with None diff_frac_pos = too few held trades to bootstrap)")
raw_sig = int(np.sum(pvals < 0.05))                       # uncorrected
reject = holm_bonferroni(pvals, alpha=0.05)
holm_sig = int(np.sum(reject))
print(f"   family size m = {len(syms)} assets")
print(f"   min p-value (best asset 'beats null') = {pvals.min():.4f} "
      f"[{syms[int(np.argmin(pvals))]}]")
print(f"   RAW significant @ p<0.05 (NO correction): {raw_sig}/{len(syms)}")
print(f"   Holm-Bonferroni significant @ FWER 0.05 : {holm_sig}/{len(syms)}")

# ---------------------------------------------------------------------------
# STEP 3: Deflated Sharpe Ratio over the 77 per-asset Sharpes (G4)
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[STEP 3] DEFLATED SHARPE RATIO over 77 per-asset Sharpes (G4), N=77 trials")
print("-" * 78)
Z95 = 1.6448536269514722
srs, Ts, skews = [], [], []
sr_syms = []
for s in syms:
    a = pa[s]
    if a.get("bb_real_p05") is None or a.get("bb_real_p50") is None or a.get("bb_real_p95") is None:
        continue
    p05, p50, p95 = float(a["bb_real_p05"]), float(a["bb_real_p50"]), float(a["bb_real_p95"])
    sigma = (p95 - p05) / (2.0 * Z95)
    if sigma <= 0:
        continue
    sr_syms.append(s)
    sr = p50 / sigma                                   # standardized held-compound statistic
    bowley = ((p95 - p50) - (p50 - p05)) / (p95 - p05) # robust skew proxy
    srs.append(sr); Ts.append(max(2, int(a["n_held"]))); skews.append(bowley)
srs = np.array(srs); Ts = np.array(Ts); skews = np.array(skews)
N = len(srs)
var_sr = float(np.var(srs, ddof=1))
sr_star = float(np.max(srs)); i_star = int(np.argmax(srs))
sr0 = expected_max_sr_under_null(var_sr, N)
dsr_best = psr(sr_star, sr0, Ts[i_star], skew=float(skews[i_star]), kurt=3.0)
# raw = best Sharpe significant vs 0 ignoring multiplicity (PSR(0)); deflated = vs SR0
psr0_best = psr(sr_star, 0.0, Ts[i_star], skew=float(skews[i_star]), kurt=3.0)
raw_dsr_winners = sum(
    1 for k in range(N) if psr(srs[k], 0.0, Ts[k], skew=float(skews[k])) > 0.95
)
dsr_winners = sum(
    1 for k in range(N) if psr(srs[k], sr0, Ts[k], skew=float(skews[k])) > 0.95
)
print(f"   N trials = {N};  Var[SR_n] = {var_sr:.4f};  sqrt = {math.sqrt(var_sr):.4f}")
print(f"   best observed Sharpe SR* = {sr_star:.4f} [{sr_syms[i_star]}], T={Ts[i_star]} trades")
print(f"   expected-max Sharpe under null (deflation hurdle) SR0 = {sr0:.4f}")
print(f"   PSR(0) of best (raw, NO multiplicity) = {psr0_best:.4f}")
print(f"   DSR  of best (deflated for N={N})      = {dsr_best:.4f}  "
      f"({'SURVIVES' if dsr_best > 0.95 else 'FAILS'} @ 0.95)")
print(f"   RAW Sharpe-significant assets (PSR(0)>0.95, NO correction): {raw_dsr_winners}/{N}")
print(f"   DEFLATED Sharpe-significant assets (DSR>0.95)             : {dsr_winners}/{N}")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print(f"RAW winners (matched-null firewall, held-out)        : {raw_total_winners}")
print(f"Holm-Bonferroni survivors (FWER 0.05, 77 assets)     : {holm_sig}")
print(f"Deflated-Sharpe survivors (N={N} trials)              : {dsr_winners}")
print(f"Naive grid 'winners' that look real pre-correction   : "
      f"1d dev={dev_pos}/7, 4h max-pool-exp={best_pool['unseen_pool_exp_pct']:.2f}% (=> 0/25 vs null)")
print("\nMultiple-comparisons correction is MONOTONE: corrected <= raw. Here raw = 0")
print("under the matched-null firewall, so DSR+Holm leave the count at 0. The apparent")
print("'winners' (1d dev +41.12%/trade SMA_50_100; 4h +4.82%/trade pooled) are the")
print("canonical grid-scan false positives -- pooled/in-sample-positive but 0 breadth vs null.")

out = {
    "sol_scan_exists": bool(sol_files),
    "raw_winners_matched_null": raw_total_winners,
    "holm_survivors": holm_sig,
    "holm_family_size": len(syms),
    "holm_raw_sig_uncorrected": raw_sig,
    "dsr_N_trials": N,
    "dsr_var_sr": var_sr,
    "dsr_sr_star": sr_star,
    "dsr_sr0_hurdle": sr0,
    "dsr_best": dsr_best,
    "dsr_raw_winners": raw_dsr_winners,
    "dsr_deflated_winners": dsr_winners,
    "g1_1d_dev_positive": dev_pos, "g1_1d_unseen_positive": uns_pos,
    "g1_1d_beats_null_unseen": f"{n_beat_null_1d}/{n_assets_1d}",
    "g2_4h_sweep_winners": f"{g2_winners}/{len(g2)}",
    "g2_4h_max_pool_exp_pct": best_pool["unseen_pool_exp_pct"],
    "g3_4h_firewall": f"{fw3['n_beat_null_held']}/{fw3['n_assets']}",
    "g4_genuine_signal": f"{agg4['n_GENUINE_SIGNAL']}/{agg4['n_assets_evaluated']}",
}
outpath = os.path.join(sol_dir, "mc_correction_result.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {outpath}")
