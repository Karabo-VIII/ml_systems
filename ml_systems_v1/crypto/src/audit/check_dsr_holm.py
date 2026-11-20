"""
CDAP Layer 8 — Holm-Corrected Deflated Sharpe Ratio (DSR) Sweep Gate
=====================================================================

Per researcher-expert audit 2026-05-25 19:35 SAST (Bailey 2014, AMS Notices):
when N variants are tested in a sweep, expected false positives at p=0.05
unadjusted = N × 0.05. For the 2026-05-25 INST-A session: N=122 variants,
n=141 UNSEEN days → expected ~6 false positives by chance. No candidate
can be considered statistically credible without family-wise correction.

This module computes the Holm-corrected DSR for every audit JSON found in
a sweep directory and flags candidates that fail the family-wise corrected
significance bar. Bailey 2014 + Holm 1979 step-down procedure.

Wired into `check_invariants.py:run_audit()` alongside Layers 1-7.

Methodology summary:
  1. For each audit JSON with a `ship_claim` block, extract per-trade returns.
  2. Compute Sharpe Ratio (annualized for 4h cadence: sqrt(252*6/fwd_bars)).
  3. Apply Bailey's Deflated Sharpe formula correcting for skewness, kurtosis,
     and number of trials in the sweep (N inferred from sweep-directory size).
  4. Convert DSR to p-value (z-score → normal CDF).
  5. Apply Holm step-down correction: sort p-values ascending, reject H_i if
     p_i < alpha / (N - i + 1).
  6. Flag candidates failing the Holm threshold as CRIT (exit 2).

Configurable thresholds:
  HOLM_ALPHA = 0.05  (family-wise error rate)
  MIN_TRIALS_FOR_CORRECTION = 20  (per Bailey 2014, deflation kicks in)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HOLM_ALPHA = 0.05
MIN_TRIALS_FOR_CORRECTION = 20

# Default sweep paths to scan (each ship-tier audit JSON contributes one trial)
DEFAULT_SWEEP_PATHS = [
    "runs/audit/**/data/r*.json",
]


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _compute_dsr(returns: list[float], n_trials: int, cadence_bars_per_year: float = 2190.0,
                  fwd_bars: int = 7) -> dict:
    """Compute Bailey's Deflated Sharpe Ratio for a single candidate.

    Formula (Bailey & Lopez de Prado 2014, SSRN 2460551, eq 9):
      DSR = (SR - SR_threshold) × sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)

    Where:
      SR_threshold = sqrt(2 * ln(N)) - sqrt(2*ln(2*pi*ln(N))) ≈ SR_zero of N trials
      T = number of trades
      skew, kurt = sample skewness + sample excess kurtosis of returns

    Returns dict {sr, dsr, p_value, n_trades, n_trials}
    """
    if len(returns) < 5:
        return {"n_trades": len(returns), "sr": 0.0, "dsr": 0.0, "p_value": 1.0,
                "note": "insufficient_trades_for_dsr"}

    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 1e-12

    # Sample skewness + excess kurtosis (unbiased estimators)
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    skew = m3 / (std ** 3) if std > 0 else 0.0
    kurt_excess = (m4 / (std ** 4)) - 3.0 if std > 0 else 0.0

    # Periods per year for annualization (4h cadence, fwd_bars=7 hold)
    # Effective trade rate = bars_per_year / fwd_bars
    periods_per_year = cadence_bars_per_year / max(fwd_bars, 1)
    sr_per_trade = mean / std
    sr_annual = sr_per_trade * math.sqrt(periods_per_year)

    # Bailey's SR_threshold for N trials (eq 4 of SSRN 2460551)
    if n_trials >= MIN_TRIALS_FOR_CORRECTION:
        ln_n = math.log(max(n_trials, 2))
        sr_thresh = math.sqrt(2.0 * ln_n) - math.sqrt(2.0 * math.log(2.0 * math.pi * ln_n))
    else:
        sr_thresh = 0.0

    # DSR (eq 9, slightly simplified — assumes IID-corrected std)
    denom = 1.0 - skew * sr_per_trade + ((kurt_excess + 1.0) / 4.0) * (sr_per_trade ** 2)
    denom = max(denom, 1e-12)
    dsr_numer = (sr_per_trade - sr_thresh) * math.sqrt(max(n - 1, 1))
    dsr = dsr_numer / math.sqrt(denom)

    p_value = 1.0 - _normal_cdf(dsr)  # one-sided

    return {
        "n_trades": n,
        "n_trials": n_trials,
        "sr_per_trade": float(sr_per_trade),
        "sr_annual": float(sr_annual),
        "sr_threshold_bailey": float(sr_thresh),
        "skewness": float(skew),
        "excess_kurtosis": float(kurt_excess),
        "dsr": float(dsr),
        "p_value_one_sided": float(p_value),
    }


def _holm_correction(p_values: list[float], alpha: float = HOLM_ALPHA,
                     family_size: int | None = None) -> list[bool]:
    """Holm step-down procedure. Returns list of bool[reject H_i].

    family_size (LD-3 FIX 2026-06-05): the TRUE number of hypotheses tested in the sweep, which may
    EXCEED the number of observed p-values (NULL/REFUTED rounds + aggregation DoF that were tested but
    never written as ship-claim JSONs). Holm divides alpha by (family_size - rank). Defaults to
    len(p_values) for backward compatibility, but the gate now passes the manifest-declared family-N so
    the correction is honest. Unobserved hypotheses are conservatively treated as p >= threshold."""
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    n_obs = len(p_values)
    m = max(family_size or n_obs, n_obs)  # never smaller than the observed count
    rejected = [False] * n_obs
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / max(m - rank, 1)
        if p < threshold:
            rejected[orig_idx] = True
        else:
            break  # step-down stops at first non-rejection
    return rejected


def _is_ship_claim(audit_data: dict) -> bool:
    """LD-3 FIX: a candidate is a SHIP-CLAIM (must HALT the commit if it fails Holm) vs a NULL/REFUTED
    round (informational only). Positively identify a ship verdict; default to NOT-ship when ambiguous
    (conservative -- ambiguous JSONs warn, they do not block)."""
    # explicit ship flags
    for k in ("is_ship", "ship", "shipped"):
        if audit_data.get(k) is True:
            return True
    # verdict strings anywhere obvious
    for k in ("CONSOLIDATED", "consolidated", "verdict", "tier", "ship_tier"):
        v = audit_data.get(k)
        if isinstance(v, str) and "SHIP" in v.upper() and "NOT-SHIP" not in v.upper():
            return True
    block = audit_data.get("ship_claim") or audit_data.get("claim") or audit_data.get("ship_candidate_block")
    if isinstance(block, dict):
        for k in ("is_ship", "ship", "shipped"):
            if block.get(k) is True:
                return True
        v = block.get("verdict") or block.get("tier")
        if isinstance(v, str) and "SHIP" in v.upper() and "NOT-SHIP" not in v.upper():
            return True
    return False


def _declared_family_n(audit_data: dict) -> int:
    """A per-JSON declared family-N, if present (e.g. sample_size_discipline.n_variants_tested)."""
    for path in (("sample_size_discipline", "n_variants_tested"), ("family_n",), ("n_variants_tested",),
                 ("sweep", "n_variants_tested")):
        node = audit_data
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)) and node > 0:
            return int(node)
    return 0


def _manifest_family_n(scan_paths: list[str]) -> int:
    """Largest n_variants_tested declared in any _sweep_manifest.json near the scanned dirs."""
    best = 0
    seen_dirs = set()
    for pattern in scan_paths:
        # strip glob tail to a base dir to look for a manifest
        base = str(PROJECT_ROOT / pattern).split("**")[0].split("*")[0]
        for cand_dir in (Path(base), Path(base).parent):
            if cand_dir in seen_dirs:
                continue
            seen_dirs.add(cand_dir)
            for mf in glob.glob(str(cand_dir / "**" / "_sweep_manifest.json"), recursive=True):
                try:
                    md = json.loads(Path(mf).read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    continue
                for k in ("n_variants_tested", "family_n", "n_trials"):
                    v = md.get(k) if isinstance(md, dict) else None
                    if isinstance(v, (int, float)) and v > best:
                        best = int(v)
    return best


def _extract_returns_from_audit(audit_data: dict) -> list[float] | None:
    """Pull per-trade returns from the canonical ship_claim block."""
    for key in ("ship_claim", "claim", "ship_candidate_block"):
        block = audit_data.get(key)
        if isinstance(block, dict) and "per_trade_returns_sorted_desc" in block:
            return list(block["per_trade_returns_sorted_desc"])
    return None


def evaluate_candidates(candidates: list[dict], family_n: int) -> tuple[list[dict], int]:
    """Core gate logic (testable, no file IO). candidates = [{file, returns, is_ship}].
    LD-3 FIX (2026-06-05): a candidate that CLAIMS ship-tier AND fails Holm is CRITICAL (exit 2, halts
    the commit) -- this is the gate the docstring always promised but never delivered (it emitted only
    'warn'). A NULL/REFUTED round failing Holm stays informational. family_n is the TRUE family size
    (>= written count) so the Bailey threshold + Holm divisor are honest."""
    if not candidates:
        return [], 0

    n_written = len(candidates)
    n_trials = max(int(family_n), n_written)
    findings: list[dict] = []
    dsr_results: list[dict] = []
    p_values: list[float] = []
    for cand in candidates:
        res = _compute_dsr(cand["returns"], n_trials=n_trials)
        dsr_results.append({"file": cand.get("file", "?"), "is_ship": bool(cand.get("is_ship")), **res})
        p_values.append(res["p_value_one_sided"])

    rejections = _holm_correction(p_values, alpha=HOLM_ALPHA, family_size=n_trials)
    n_survive_holm = sum(rejections)

    for cand_res, reject in zip(dsr_results, rejections):
        if reject:
            continue
        # Failed Holm. CRITICAL iff it CLAIMS ship-tier; else informational (NULL/REFUTED round).
        is_ship = cand_res["is_ship"]
        findings.append({
            "severity": "critical" if is_ship else "warn",
            "name": "ship_claim_fails_holm_corrected_dsr" if is_ship else "round_fails_holm_corrected_dsr",
            "file": cand_res["file"],
            "detail": (
                f"{'SHIP-CLAIM' if is_ship else 'non-ship round'}: DSR={cand_res['dsr']:.2f}, "
                f"p={cand_res['p_value_one_sided']:.4f}, n_trades={cand_res['n_trades']}, "
                f"family_N={n_trials} (written={n_written}). Fails Holm-corrected alpha={HOLM_ALPHA}. "
                + ("HALT: a ship-tier claim below the multiple-testing-corrected significance bar is NOT "
                   "deploy-credible (Bailey 2014)." if is_ship else
                   "Informational: a NULL/REFUTED round, not a ship attempt.")
            ),
        })

    findings.insert(0, {
        "severity": "info",
        "name": "dsr_holm_sweep_summary",
        "detail": (
            f"family_N={n_trials} (written={n_written}), survived_holm_alpha={HOLM_ALPHA}: {n_survive_holm}. "
            f"ship_claims={sum(1 for c in dsr_results if c['is_ship'])}. "
            f"Per Bailey 2014: at N={n_trials} variants tested without correction, expected false "
            f"positives at p=0.05 unadjusted = {n_trials * 0.05:.1f}."
        ),
    })

    n_crit = sum(1 for f in findings if f["severity"] == "critical")
    n_warn = sum(1 for f in findings if f["severity"] == "warn")
    return findings, (2 if n_crit else 1 if n_warn else 0)


def run_audit(scan_paths: list[str] | None = None, family_n: int | None = None) -> tuple[list[dict], int]:
    """Compute Holm-corrected DSR across all audit JSONs found. family_n (optional) overrides the
    resolved family size; otherwise family_N = max(written candidates, manifest n_variants_tested,
    per-JSON declared n_variants_tested) -- the LD-3 fix so the correction is not under-counted."""
    paths = scan_paths or DEFAULT_SWEEP_PATHS
    candidates: list[dict] = []
    declared = 0
    for pattern in paths:
        for fp in glob.glob(str(PROJECT_ROOT / pattern), recursive=True):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            rets = _extract_returns_from_audit(data)
            if rets is None or len(rets) < 5:
                continue
            declared = max(declared, _declared_family_n(data))
            candidates.append({
                "file": str(Path(fp).relative_to(PROJECT_ROOT)),
                "returns": rets,
                "is_ship": _is_ship_claim(data),
            })

    if not candidates:
        return [], 0  # no candidates with per-trade-returns block; nothing to gate

    resolved_family_n = family_n if family_n is not None else max(len(candidates), declared, _manifest_family_n(paths))
    return evaluate_candidates(candidates, family_n=resolved_family_n)


# ---- RWYB self-test (synthetic; no file IO) ------------------------------
def _selftest() -> int:
    import random
    rng = random.Random(0)
    # (a) a STRONG genuine edge claimed as ship, tested ALONE (family_n=1) -> should survive Holm -> exit 0
    strong = [0.03 + rng.gauss(0, 0.01) for _ in range(40)]
    f_a, e_a = evaluate_candidates([{"file": "strong_ship.json", "returns": strong, "is_ship": True}], family_n=1)
    # (b) the SAME strong edge but at a large declared family_N=200 -> Bailey threshold rises -> may fail
    f_b, e_b = evaluate_candidates([{"file": "strong_ship.json", "returns": strong, "is_ship": True}], family_n=200)
    # (c) a WEAK ship-claim at family_N=200 -> must FAIL Holm and HALT (critical, exit 2)
    weak = [0.002 + rng.gauss(0, 0.02) for _ in range(30)]
    f_c, e_c = evaluate_candidates([{"file": "weak_ship.json", "returns": weak, "is_ship": True}], family_n=200)
    # (d) the SAME weak result but a NULL round (not a ship-claim) -> warn only (exit 1), NOT a halt
    f_d, e_d = evaluate_candidates([{"file": "null_round.json", "returns": weak, "is_ship": False}], family_n=200)
    print("[check_dsr_holm selftest]")
    print(f"  (a) strong ship, family_N=1   -> exit {e_a}  (expect 0: survives Holm)")
    print(f"  (b) strong ship, family_N=200 -> exit {e_b}  (Bailey threshold rises with family_N)")
    print(f"  (c) WEAK ship,   family_N=200 -> exit {e_c}  (EXPECT 2: ship-claim fails Holm -> HALT)")
    print(f"  (d) weak NULL round, fam=200  -> exit {e_d}  (EXPECT 1: warn only, NOT a halt)")
    ok = (e_a == 0 and e_c == 2 and e_d == 1)
    print(f"[check_dsr_holm selftest] {'OK' if ok else '*** FAIL ***'} -- the gate now HALTS a ship-claim that "
          f"fails family-wise correction (was always 'warn' before LD-3 fix), and family_N is honest.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Holm-corrected DSR sweep gate (CDAP Layer 8)")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic RWYB self-test and exit")
    ap.add_argument("--family-n", type=int, default=None, help="override the resolved family-N")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    findings, exit_code = run_audit(family_n=args.family_n)
    if exit_code == 0 and not findings:
        print("[check_dsr_holm] OK - no SHIP-tier candidates with per_trade_returns block; nothing to gate")
        return 0
    label = "CRIT" if exit_code == 2 else "WARN" if exit_code == 1 else "INFO"
    print(f"[check_dsr_holm] {label} - {len(findings)} findings")
    for f in findings:
        prefix = {"critical": "FAIL", "warn": "WARN", "info": "INFO"}.get(f["severity"], "?")
        loc = f"  [{f.get('file')}]" if f.get("file") else ""
        print(f"  {prefix} {f['name']}{loc}: {f['detail']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
