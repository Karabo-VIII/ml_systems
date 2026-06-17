"""
Wealth-Bot Audit-JSON Claim Contract (canonical fields enforcer)
=================================================================

Every audit JSON that emits a SHIP-tier or all-4-positive candidate MUST
include the canonical fields defined in REQUIRED_FIELDS_FOR_SHIP_CLAIM
below. This contract is enforced by:

  1. `check_wealth_bot_claims.py` (CDAP pre-commit gate, exit 2 on violation)
  2. Auditor RED-team brief (mechanism falsifier check)
  3. Manual review of WEALTH_BOT_LEADERBOARD.md promotions

Why: 2026-05-25 INST-A session inflated P4_route_basis_pos_only as a
"near-breakthrough" candidate. RED-team audit revealed the filter
*kept* the top-3 ABC_AND trades and *dropped* the diversifying ones --
opposite of the claimed mechanism. The pattern P (top-trade
concentration) and Q (mechanism verification) memory entries codify
the lesson. This module is the code-level enforcement.

Usage (in any audit-emitting script):

    from wealth_bot.framework.claim_contract import (
        build_ship_claim_block, validate_claim_block, ClaimContractError
    )

    rets = compute_per_trade_returns_for_unseen(...)
    claim = build_ship_claim_block(
        candidate_id="my_candidate_name",
        per_trade_returns=rets,           # list of floats per trade
        jackknife_compound={"K=0": ..., "K=1": ..., "K=2": ..., "K=3": ..., "K=5": ...},
        combined_K2_plus_S9=...,
        n_unseen=len(rets),
        baseline_compound_pct=...,
        mechanism_claim="brief text describing why it works",
        all_window_compounds={"TRAIN": ..., "VAL": ..., "OOS": ..., "UNSEEN": ...},
    )
    out_dict["ship_claim"] = claim
    # CDAP will validate this on commit; failure halts.
"""
from __future__ import annotations

import json
import math
from typing import Any


class ClaimContractError(ValueError):
    """Raised when a SHIP-tier claim is missing required fields or fails validation."""


# ---------------------------------------------------------------------------
# Required-field schema
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = (
    "candidate_id",
    "n_unseen",
    "baseline_compound_pct",
    "all_window_compounds",        # dict {TRAIN, VAL, OOS, UNSEEN: float}
    "all_4_positive",
    "per_trade_returns_sorted_desc",
    "top_3_pct_of_compound",       # retained for backward-compat; superseded by concentration_metrics
    "concentration_metrics",       # NEW v1.1: herfindahl + gini + top_K_abs (sign-pathology-free)
    "jackknife",                   # dict K=0..K=5
    "combined_K2_plus_S9_pct",
    "sample_size_discipline",      # dict with ship-threshold + stressed-compound check
    "phase1_n_eff_gate",           # NEW v1.2 (2026-05-25): hard-block at n_eff<15
    "compound_discounted_pct",     # NEW v1.2 (SR1.4): compound * min(1, n_eff/12)
    "mechanism_claim",
    "mechanism_falsifier_check",   # dict with what_filter_keeps / drops / verified_by
    "passes_strict_gate",          # bool, derived
    "contract_version",
)

# Phase 1 minimum effective sample size (from validator+oracle consensus 2026-05-25):
# at n_eff < 15 the per-trade return distribution is too sparse for block-bootstrap p05/p95
# or jackknife-K to be reliable. Below this floor no candidate can be Phase-1-promoted.
PHASE1_N_EFF_MIN = 15
# SR1.4 reference n for full credit: at or above this n, no discount is applied.
SR1_4_FULL_CREDIT_N = 12

CONTRACT_VERSION = "1.2"  # 2026-05-25 r2: Phase 1 n_eff gate + SR1.4 auto-discount


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_ship_claim_block(
    *,
    candidate_id: str,
    per_trade_returns: list[float],       # raw per-trade net returns (post-cost), e.g. 0.025 = +2.5%
    jackknife_compound: dict[str, float], # MUST include K=0,1,2,3,5
    combined_K2_plus_S9: float,
    n_unseen: int,
    baseline_compound_pct: float,
    mechanism_claim: str,
    all_window_compounds: dict[str, float],   # {TRAIN, VAL, OOS, UNSEEN: %}
    mechanism_falsifier_check: dict[str, Any] | None = None,
    n_unseen_threshold_relaxed: int = 30,
    n_unseen_threshold_binding: int = 20,
    baseline_for_escalation_pct: float = 39.30,  # 1-strat PATCHED ensemble baseline
) -> dict:
    """Build a canonical SHIP-claim block for inclusion in an audit JSON.

    Validates required inputs at construction time. Computes derived fields.
    Raises ClaimContractError if any required input is malformed.

    The output dict satisfies validate_claim_block(); writing it to disk is
    sufficient for CDAP pre-commit to pass.
    """
    if not per_trade_returns:
        raise ClaimContractError(f"per_trade_returns is empty for {candidate_id}")

    rets_sorted = sorted(per_trade_returns, reverse=True)
    n = len(rets_sorted)

    if n != n_unseen:
        raise ClaimContractError(
            f"n_unseen={n_unseen} doesn't match len(per_trade_returns)={n} for {candidate_id}"
        )

    # Compute top-3 % of compound (kept for backward-compat; flagged by validator
    # as sign-pathological — values >100% possible when compound is near zero).
    if n >= 3:
        top_3_comp = math.prod(1 + r for r in rets_sorted[:3]) - 1.0
    else:
        top_3_comp = math.prod(1 + r for r in rets_sorted) - 1.0
    overall_comp = math.prod(1 + r for r in rets_sorted) - 1.0
    if abs(overall_comp) > 1e-9:
        top_3_pct = 100.0 * top_3_comp / overall_comp
    else:
        top_3_pct = float("inf") if top_3_comp > 0 else float("-inf") if top_3_comp < 0 else 0.0

    # NEW v1.1 (2026-05-25 trust-stack item #5): concentration_metrics —
    # sign-pathology-free measures of trade-level dominance. Per validator finding:
    # Herfindahl-on-|contributions| is bounded [1/n, 1], scale-invariant, no
    # compound-denominator artifact. Gate: H > 0.50 at n<30 flags concentration.
    abs_rets = [abs(r) for r in rets_sorted]
    sum_abs = sum(abs_rets) if abs_rets else 1.0
    if sum_abs > 1e-12:
        shares = [a / sum_abs for a in abs_rets]
        herfindahl = sum(s * s for s in shares)            # in [1/n, 1]
        # top-3 absolute contribution share (signs ignored)
        top_3_abs_share = sum(shares[:3]) if len(shares) >= 3 else sum(shares)
        # Gini coefficient on absolute contributions
        if len(abs_rets) >= 2:
            sorted_abs = sorted(abs_rets)
            cum = 0.0
            for i, a in enumerate(sorted_abs):
                cum += (i + 1) * a
            gini = (2.0 * cum) / (len(sorted_abs) * sum_abs) - (len(sorted_abs) + 1) / len(sorted_abs)
        else:
            gini = 0.0
    else:
        herfindahl = 1.0 / max(n, 1)
        top_3_abs_share = 0.0
        gini = 0.0

    concentration_metrics = {
        "herfindahl_on_abs_contribution": float(herfindahl),
        "top_3_abs_share": float(top_3_abs_share),
        "gini_on_abs_contribution": float(gini),
        "n_effective_at_herfindahl": float(1.0 / herfindahl) if herfindahl > 0 else float(n),
        "concentrated_flag": bool(herfindahl > 0.50 and n < 30),
        "rationale": ("Herfindahl > 0.50 at n<30 indicates concentration. "
                       "Effective sample size = 1/H. "
                       "Bounded [1/n, 1], sign-pathology-free; replaces top_3_pct_of_compound for gating."),
    }

    # All-4-positive check
    all_4_positive = all(v > 0 for v in all_window_compounds.values())

    # Validate jackknife dict
    required_K = {"K=0", "K=1", "K=2", "K=3"}
    if not required_K.issubset(jackknife_compound):
        missing = required_K - set(jackknife_compound)
        raise ClaimContractError(
            f"jackknife_compound for {candidate_id} missing keys: {missing}. "
            f"Required: K=0, K=1, K=2, K=3 (K=5 recommended)."
        )

    # Sample-size discipline (compared against post-stress)
    if n_unseen < n_unseen_threshold_binding:
        ship_threshold_pp = 25.0
        binding = True
        discipline_label = f"n<{n_unseen_threshold_binding} (BINDING)"
    elif n_unseen < n_unseen_threshold_relaxed:
        ship_threshold_pp = 15.0
        binding = False
        discipline_label = f"n<{n_unseen_threshold_relaxed} (relaxed)"
    else:
        ship_threshold_pp = 10.0
        binding = False
        discipline_label = f"n>={n_unseen_threshold_relaxed} (relaxed)"

    ship_threshold_compound = baseline_for_escalation_pct + ship_threshold_pp
    stressed_compound = combined_K2_plus_S9
    passes_baseline_gate = baseline_compound_pct >= ship_threshold_compound
    passes_stressed_gate = stressed_compound >= ship_threshold_compound

    # Phase 1 n_eff gate (v1.2 — 2026-05-25 trust-stack item, validator+oracle consensus):
    # use Herfindahl-derived effective sample size as the floor for Phase 1 promotion.
    # n_eff_at_herfindahl = 1/H. When dominated by 1-2 trades, n_eff approaches 1-2 even
    # if n_unseen=20, and the candidate has no genuine signal — only a few lucky bars.
    n_eff_at_herfindahl = concentration_metrics["n_effective_at_herfindahl"]
    phase1_n_eff_passes = (n_eff_at_herfindahl >= PHASE1_N_EFF_MIN) and (n_unseen >= PHASE1_N_EFF_MIN)
    phase1_n_eff_gate_block = {
        "n_unseen": int(n_unseen),
        "n_effective_at_herfindahl": float(n_eff_at_herfindahl),
        "phase1_min_required": int(PHASE1_N_EFF_MIN),
        "passes": bool(phase1_n_eff_passes),
        "rationale": (
            f"Phase 1 promotion requires BOTH n_unseen >= {PHASE1_N_EFF_MIN} AND "
            f"n_eff (= 1/Herfindahl) >= {PHASE1_N_EFF_MIN}. "
            "Below this floor block-bootstrap CIs and jackknife-K are unreliable."
        ),
    }

    # SR1.4 auto-discount (2026-05-25): linearly discount headline compound by
    # min(1, n_eff / SR1_4_FULL_CREDIT_N). At n_eff < 12, compound is haircut by
    # the fraction of the reference n it covers. Leaderboard rendering uses
    # discounted compound for ranking; raw compound retained for audit transparency.
    sr1_4_credit_fraction = min(1.0, n_eff_at_herfindahl / float(SR1_4_FULL_CREDIT_N))
    overall_compound_pct = (math.prod(1 + r for r in rets_sorted) - 1.0) * 100.0
    compound_discounted = overall_compound_pct * sr1_4_credit_fraction

    # Strict gate (all required):
    #   1. all 4 windows positive
    #   2. jackknife K=2 > 0
    #   3. combined K2+S9 > 0 AND >= sample-size-escalated ship threshold
    #   4. n >= 20 OR baseline_compound clears n<20 binding gate
    #   5. Phase 1 n_eff gate passes (v1.2)
    passes_strict_gate = (
        all_4_positive
        and jackknife_compound["K=2"] > 0
        and combined_K2_plus_S9 > 0
        and passes_stressed_gate  # this catches "stressed-not-baseline" gate
        and n_unseen >= 20  # n<20 cannot pass strict gate even if baseline does
        and phase1_n_eff_passes  # n_eff floor for genuine signal
    )

    # If no mechanism_falsifier_check provided AND concentration is high
    # (herfindahl > 0.50 OR top_3_pct > 70%) AND n<30,
    # the claim cannot ship without explicit empirical verification.
    high_concentration = concentration_metrics["concentrated_flag"] or top_3_pct > 70.0
    if mechanism_falsifier_check is None and high_concentration and n_unseen < 30:
        mechanism_falsifier_check = {
            "what_filter_keeps": "REQUIRED_BUT_NOT_PROVIDED",
            "what_filter_drops": "REQUIRED_BUT_NOT_PROVIDED",
            "verified_by": "NOT_YET_VERIFIED",
            "note": (
                f"top_3_pct_of_compound={top_3_pct:.1f}% > 70% at n_unseen={n_unseen} < 30. "
                "Mechanism claim REQUIRES empirical trade-level falsification before SHIP. "
                "Per Pattern Q (mechanism_verification_rule), this claim must be "
                "downgraded to INCONCLUSIVE pending auditor review."
            ),
        }

    return {
        "contract_version": CONTRACT_VERSION,
        "candidate_id": candidate_id,
        "n_unseen": int(n_unseen),
        "baseline_compound_pct": float(baseline_compound_pct),
        "all_window_compounds": dict(all_window_compounds),
        "all_4_positive": bool(all_4_positive),
        "per_trade_returns_sorted_desc": [float(r) for r in rets_sorted],
        "top_3_pct_of_compound": float(top_3_pct),
        "concentration_metrics": concentration_metrics,
        "jackknife": dict(jackknife_compound),
        "combined_K2_plus_S9_pct": float(combined_K2_plus_S9),
        "sample_size_discipline": {
            "n_unseen": int(n_unseen),
            "discipline_label": discipline_label,
            "ship_threshold_pp_over_baseline": float(ship_threshold_pp),
            "ship_threshold_compound_required": float(ship_threshold_compound),
            "baseline_compound_against_threshold": float(baseline_compound_pct),
            "stressed_compound_against_threshold": float(stressed_compound),
            "passes_baseline_gate": bool(passes_baseline_gate),
            "passes_stressed_gate": bool(passes_stressed_gate),
            "binding_at_this_n": bool(binding),
        },
        "phase1_n_eff_gate": phase1_n_eff_gate_block,
        "compound_discounted_pct": float(compound_discounted),
        "mechanism_claim": str(mechanism_claim),
        "mechanism_falsifier_check": dict(mechanism_falsifier_check) if mechanism_falsifier_check else {
            "what_filter_keeps": "<list trade indices kept by filter>",
            "what_filter_drops": "<list trade indices dropped by filter>",
            "verified_by": "<auditor identity + timestamp>",
        },
        "passes_strict_gate": bool(passes_strict_gate),
    }


# v1.2-only fields that existing v1.1 audits do not have. Accepted as
# missing on v1.1 claims (backward-compat); enforced on v1.2 claims.
V1_2_ONLY_FIELDS = frozenset({"phase1_n_eff_gate", "compound_discounted_pct"})

# Accepted contract versions (newest first). v1.1 stays accepted as honest
# retrofit-frozen audits; v1.2 is required for all NEW claims.
ACCEPTED_CONTRACT_VERSIONS = ("1.2", "1.1")


def validate_claim_block(claim: dict) -> list[str]:
    """Return a list of validation errors. Empty list = valid.

    Used by CDAP `check_wealth_bot_claims.py` to scan audit JSONs.

    Backward-compat: contract_version="1.1" is ACCEPTED without the v1.2-only
    fields (phase1_n_eff_gate, compound_discounted_pct). NEW claims (i.e.
    anything written after 2026-05-25 20:45 SAST) MUST emit contract_version
    "1.2" via build_ship_claim_block — which auto-populates the new fields.
    """
    errors: list[str] = []
    if not isinstance(claim, dict):
        return [f"claim is not a dict: {type(claim)}"]

    claim_version = claim.get("contract_version")
    is_v1_1 = (claim_version == "1.1")

    for field in REQUIRED_FIELDS:
        if field not in claim:
            # v1.2-only fields are tolerated as missing on v1.1 claims
            if is_v1_1 and field in V1_2_ONLY_FIELDS:
                continue
            errors.append(f"missing required field: {field}")

    if "contract_version" in claim and claim["contract_version"] not in ACCEPTED_CONTRACT_VERSIONS:
        errors.append(
            f"contract_version not accepted: claim={claim['contract_version']!r} "
            f"accepted={list(ACCEPTED_CONTRACT_VERSIONS)}"
        )

    # Cross-field consistency
    if "jackknife" in claim and isinstance(claim["jackknife"], dict):
        required_K = {"K=0", "K=2"}
        if not required_K.issubset(claim["jackknife"]):
            errors.append(f"jackknife missing required keys: {required_K - set(claim['jackknife'])}")

    if "top_3_pct_of_compound" in claim and "mechanism_falsifier_check" in claim:
        top_3 = claim["top_3_pct_of_compound"]
        n_uns = claim.get("n_unseen", 0)
        mfc = claim.get("mechanism_falsifier_check", {})
        cm = claim.get("concentration_metrics", {})
        # v1.1 trust-stack item #5: use Herfindahl (sign-pathology-free) as
        # primary gate. Keep top_3 as a backstop for legacy claims.
        herfindahl = cm.get("herfindahl_on_abs_contribution", 0.0)
        high_concentration = herfindahl > 0.50 or top_3 > 70.0
        if high_concentration and n_uns < 30:
            if mfc.get("verified_by") in (None, "NOT_YET_VERIFIED", "<auditor identity + timestamp>"):
                errors.append(
                    f"high concentration (herfindahl={herfindahl:.2f}, top_3_pct={top_3:.1f}%) "
                    f"at n={n_uns} < 30 requires "
                    f"mechanism_falsifier_check.verified_by to be set (got: {mfc.get('verified_by')})"
                )

    if claim.get("passes_strict_gate") is True:
        # Sanity-check: strict-gate implies all internal gates pass
        ssd = claim.get("sample_size_discipline", {})
        if not ssd.get("passes_stressed_gate"):
            errors.append(
                "passes_strict_gate=True but sample_size_discipline.passes_stressed_gate=False"
            )
        # v1.2: also enforce phase1_n_eff gate when strict-gate is claimed.
        # On v1.1 claims (pre-2026-05-25-evening retrofit), phase1_n_eff_gate
        # is absent; skip this sub-check rather than reject.
        if not is_v1_1:
            p1g = claim.get("phase1_n_eff_gate", {})
            if not p1g.get("passes"):
                errors.append(
                    f"passes_strict_gate=True but phase1_n_eff_gate.passes=False "
                    f"(n_eff={p1g.get('n_effective_at_herfindahl')}, "
                    f"required>={p1g.get('phase1_min_required')})"
                )

    return errors


# ---------------------------------------------------------------------------
# Helper: bootstrap p05/p95 from per-trade returns (block bootstrap)
# ---------------------------------------------------------------------------
def block_bootstrap_p05_p95(
    returns: list[float], n_boot: int = 5000, block_size: int = 3, seed: int = 42,
) -> dict[str, float]:
    """Stationary block bootstrap. Used by audit-emitting scripts to fill
    additional verification fields. Pure-Python — no numpy required.
    """
    import random

    rng = random.Random(seed)
    n = len(returns)
    if n == 0:
        return {"p05": 0.0, "p95": 0.0, "median": 0.0, "positive_pct": 0.0}

    n_blocks = (n + block_size - 1) // block_size
    boots: list[float] = []
    for _ in range(n_boot):
        starts = [rng.randrange(n) for _ in range(n_blocks)]
        sampled = []
        for s in starts:
            for k in range(block_size):
                sampled.append(returns[(s + k) % n])
        sampled = sampled[:n]
        compound = 1.0
        for r in sampled:
            compound *= (1 + r)
        boots.append((compound - 1.0) * 100.0)
    boots.sort()
    return {
        "p05": boots[max(0, int(0.05 * len(boots)))],
        "p95": boots[min(len(boots) - 1, int(0.95 * len(boots)))],
        "median": boots[len(boots) // 2],
        "positive_pct": 100.0 * sum(1 for b in boots if b > 0) / len(boots),
    }


# ---------------------------------------------------------------------------
# CLI for spot-checking an audit JSON
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Validate a wealth-bot audit JSON against the SHIP claim contract.")
    ap.add_argument("json_path", help="path to an audit JSON")
    ap.add_argument("--ship-claim-key", default="ship_claim",
                    help="dict key under which the ship_claim block lives (default: ship_claim)")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    block = data.get(args.ship_claim_key)
    if block is None:
        print(f"NO ship_claim block found at key '{args.ship_claim_key}' in {args.json_path}")
        return 2

    errors = validate_claim_block(block)
    if errors:
        print(f"FAIL: {args.json_path}")
        for e in errors:
            print(f"  - {e}")
        return 2
    print(f"OK: {args.json_path} passes claim-contract validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
