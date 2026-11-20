"""src/agents/_shared/shuffled_market_control.py -- the A2 policy-overfit detector (the RL analog of ShIC).

WHAT (doc AGENT_LAYER_ARCHITECTURE_2026_06_11 S4.2 H3 / S4.1 H3): a forecaster that MEMORIZES shows
contiguous-IC high / ShIC ~ 0 (the V22/V25 incident: ic1=+0.21, ShIC=0.000). The POLICY analog is
INVISIBLE by inspection -- a memorizing policy's only symptom is a profitable backtest curve,
indistinguishable from a real edge. This module IS the policy-side ShIC test, and per the spec it "does
NOT yet exist for policies." It is the A2 keystone gate (S1.6: A2 == A1's gate MINUS the source-F clause
PLUS this control).

THE TEST. Given a policy's realized per-trade (or per-bar) return stream on the REAL market, re-evaluate
the SAME policy on SURROGATE markets whose MARGINAL return distribution is preserved but whose TEMPORAL
structure is destroyed. A GENUINE structure-exploiting policy -> ~0 on the surrogate (its edge came from
real temporal structure that no longer exists). A MEMORIZER / curve-fit policy -> still "profits" on the
surrogate (it memorized the path, not the structure) -> that residual is the policy ShIC=0 signature.

  realized_real  -  realized_surrogate   ==   the genuine, structure-derived part of the edge.
  realized_surrogate (as a fraction of realized_real)  ==   the OVERFIT signature.

SURROGATE ROLES (the load-bearing detail -- finding #9: the WRONG surrogate is a broken gate):
  PRIMARY -- PREDICTABILITY-DESTROYING (these DRIVE the verdict):
  - perm (index permute) : full random permutation (== block_shuffle at block=1). PRESERVES the marginal
                      EXACTLY; DESTROYS ALL temporal predictability -- linear AND nonlinear. The most
                      aggressive destroyer; needs no length tuning.
  - block_shuffle   : shuffle contiguous BLOCKS of length L (L < the setup->payoff horizon). PRESERVES
                      the marginal EXACTLY and only the SHORT-RANGE autocorr INSIDE each block; DESTROYS
                      predictable structure longer than L.
  A GENUINE policy exploits SOME temporal predictability (linear OR nonlinear) -- so it COLLAPSES on
  perm/block. A MEMORIZER/curve-fitter still 'profits' on them (it keyed the path/index, not structure).

  SECONDARY -- MECHANISM DIAGNOSTICS (computed + reported, but they do NOT drive the verdict):
  - phase_randomize : FFT, randomize the phases, invert. PRESERVES the power spectrum == the full linear
                      autocorrelation; DESTROYS phase coupling / nonlinear structure. Marginal Gaussianized.
  - iaaft           : Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz 1996). PRESERVES
                      BOTH the power spectrum (autocorrelation) AND the exact amplitude (marginal).
  WHY NOT VERDICT DRIVERS (finding #9): phase/iaaft PRESERVE linear autocorrelation, so a GENUINE policy
  that exploits autocorrelation SURVIVES on them -> it would be FALSE-FLAGGED OVERFIT. Their value is
  MECHANISM attribution: a policy that survives iaaft but collapses on perm is a LINEAR (autocorr) exploit;
  one that collapses on BOTH perm and iaaft is a NONLINEAR / higher-order exploit.

TWO-SIDED VALIDATION (MANDATORY -- a gate that always says OVERFIT is as useless as one that never does):
  - POSITIVE A (NONLINEAR): a multi-bar setup->payoff policy on a ZERO-DRIFT structured market ->
                      profits on REAL, collapses to ~0 on the PREDICTABILITY-DESTROYING perm/block
                      surrogates -> GENUINE.
  - POSITIVE B (LINEAR autocorr -- finding #9): a lag-1 momentum policy on a ZERO-DRIFT AR(1) market ->
                      profits on REAL by exploiting linear autocorrelation. phase/iaaft PRESERVE that
                      autocorrelation so it SURVIVES on them (the OLD false-flag) -- but perm/block
                      collapse it -> GENUINE. The demo asserts BOTH the GENUINE verdict AND that iaaft
                      WOULD have false-flagged it, proving iaaft must NOT drive the verdict.
  - NEGATIVE control: a beta/marginal-harvester (long every bar; no timing skill) on a +drift market ->
                      profits on BOTH real AND every surrogate (perm/block keep the SAME marginal, so a
                      no-timing harvester earns it identically) -> verdict OVERFIT (flagged).

REUSED APPARATUS (this module does NOT reinvent): the percentile-band detection convention and the
_compound() helper mirror src/strat/{firewall.py, synthetic_positive_control.py}; the two-sided
positive/negative control PATTERN mirrors src/strat/{positive_control.py, pbo_cscv.py}; this control is
the policy-stream sibling of those gates and is designed to be wired alongside battery.py / pbo_cscv.py /
firewall.py when the real A2 policy lands. It deliberately operates on an ABSTRACT trade-return stream
(a callable `policy(returns) -> per-trade nets`), so it is agnostic to whether the policy is a DT, PPO,
SAC, or evolutionary champion.

PROJECT INVARIANTS honored: taker cost 0.0024 round-trip is the default charged INSIDE each trade
(never an optimistic maker headline); no look-ahead -- the surrogate transform is applied to the WHOLE
held-out stream the policy already produced (we re-score an existing policy, we do not refit on the
future); IC is NOT used as an objective anywhere (this is the SETUP/MOVE-level compound analog of ShIC);
the unit is a per-trade/per-setup net return, not a per-bar IC.

CLASS TAG: NONE, BY DESIGN. This is pure shared APPARATUS (a validation primitive) -- it contains NO
policy/agent logic, emits no actions, owns no held-out compound number. The `agent_class_declared` CDAP
invariant scopes __class_tag__ to agent-LOGIC entry points only (config/_invariants.yaml
agent_logic_globs: a1_wm_consuming entry modules + a2_raw_data/*_agent.py + a1h_hybrid/*_agent.py); a
file under src/agents/_shared/ that is not a *_agent.py is intentionally OUT of scope (README.md: "_shared
... is a utility module, not an agent"). A tag here would be a category error (it would assert this gate
IS an A2 agent). __class_tag__ is therefore set to None with this rationale, not omitted silently.

RWYB: `python src/agents/_shared/shuffled_market_control.py`  (exit 0 == both controls land correctly:
positive -> GENUINE, negative -> OVERFIT). CPU + tiny synthetic inputs -- this verifies the gate LOGIC
and its two-sidedness, NOT GPU/throughput. The real run happens when V1.1 lands and a real A2 policy
exists. No emoji (cp1252-safe).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

# Pure apparatus, NOT an agent -- see module docstring "CLASS TAG" for why this is None (and why a tag
# here would be a category error). The agent_class_declared invariant does not scope src/agents/_shared/.
__class_tag__ = None

TAKER_COST_RT = 0.0024  # project invariant: taker round-trip is the gate value, never an optimistic maker headline

__contract__ = {
    "kind": "validation_primitive",
    "inputs": (
        "a real market return series (1-D per-bar simple returns) + a policy callable "
        "`policy(returns: np.ndarray) -> np.ndarray` that returns the policy's per-trade NET return stream; "
        "surrogate kind(s) in {phase, block, iaaft}; n_surrogates; cost_rt; seed"
    ),
    "outputs": (
        "dict{verdict in {GENUINE, OVERFIT, INCONCLUSIVE}, real_compound, surrogate_compound_p50/p95, "
        "surrogate_vs_real_gap, overfit_fraction, per_kind, beats_surrogate}"
    ),
    "invariants": [
        "the VERDICT is driven ONLY by PREDICTABILITY-DESTROYING surrogates (perm/block) -- they scramble "
        "ALL temporal predictability (linear AND nonlinear) while preserving the marginal EXACTLY",
        "phase/iaaft are SECONDARY MECHANISM diagnostics (they PRESERVE linear autocorrelation, so an "
        "autocorr-exploiting GENUINE policy survives on them -- they must NOT drive the verdict, finding #9)",
        "phase_randomize preserves the power spectrum (full linear autocorrelation); marginal Gaussianized",
        "block_shuffle/perm preserve the marginal EXACTLY (perm = block=1, destroys ALL temporal structure)",
        "iaaft preserves BOTH the power spectrum AND the exact amplitude (marginal) distribution",
        "each surrogate KIND uses its OWN seeded rng -> bands are independent of kind iteration ORDER (finding #10)",
        "GENUINE iff the policy's real compound beats the predictability-destroying band AND its residual is ~0",
        "OVERFIT iff the policy still profits after predictability is destroyed (surrogate compound not ~0)",
        "two-sided: a structure-exploiting policy -> GENUINE; a curve-fit/memorizing policy -> OVERFIT",
        "NECESSARY NOT SUFFICIENT (finding #11): destroys path-memorization evidence, does NOT prove OOS "
        "generalization; an index-keyed curve-fitter can still 'profit' on a surrogate -- compose with "
        "held-out/walk-forward refit (battery.py, pbo_cscv.py)",
        "cost (taker 0.0024 default) is charged INSIDE the policy's trades, not bolted on after",
        "no look-ahead: the policy is RE-SCORED on a surrogate of an already-produced held-out stream",
        "IC is never an objective; the unit is a per-trade/per-setup net return (the SETUP/MOVE-level ShIC analog)",
    ],
}


# ======================================================================================================
# COMPOUND HELPER (mirrors src/strat/firewall.py:_compound / synthetic_positive_control.py:_compound)
# ======================================================================================================
def _compound(per_trade: Sequence[float]) -> float:
    """Compound a stream of per-trade NET returns into a single fraction (e.g. +0.12 == +12%)."""
    a = np.asarray(per_trade, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.prod(1.0 + a) - 1.0) if a.size else 0.0


# ======================================================================================================
# SURROGATE GENERATORS -- "make the surrogate correct"
# ======================================================================================================
def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate. Preserves the POWER SPECTRUM (== full linear autocorrelation),
    randomizes the Fourier phases (destroys phase coupling / nonlinear temporal structure). The marginal
    is Gaussianized as a side effect -- use `iaaft` when the exact marginal must be preserved.

    Construction: FFT -> keep |X(f)|, replace arg(X(f)) with uniform random phases drawn antisymmetrically
    so the inverse transform is real; the DC term and (for even N) the Nyquist term keep their original
    (real) phase to guarantee a real-valued output."""
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    mag = np.abs(X)
    n_freq = X.size
    phases = rng.uniform(0.0, 2.0 * np.pi, n_freq)
    phases[0] = 0.0  # DC term must stay real
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist term must stay real for even-length signals
    X_surr = mag * np.exp(1j * phases)
    return np.fft.irfft(X_surr, n=n)


def block_shuffle(x: np.ndarray, rng: np.random.Generator, block: int = 8) -> np.ndarray:
    """Block-shuffled surrogate. Preserves the marginal EXACTLY (it is a permutation of the original
    blocks) and the SHORT-RANGE autocorrelation INSIDE each block; destroys structure longer than
    `block`. A trailing remainder block (< block) is kept intact and shuffled with the rest.

    This is a PREDICTABILITY-DESTROYING surrogate: at small `block` it scrambles essentially ALL
    temporal predictability (linear AND nonlinear) while keeping the marginal exact -- making it the
    PRIMARY genuine/overfit discriminator (a policy that still profits here memorized the path/index,
    not the structure). IAAFT/phase, by contrast, PRESERVE linear autocorrelation, so they are NOT
    valid primary discriminators (an autocorr-exploiting genuine policy survives on them)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    block = max(1, int(block))
    blocks = [x[i:i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def index_permute(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full index-permutation surrogate (== block_shuffle at block=1). Preserves the marginal
    EXACTLY and destroys ALL temporal structure -- linear AND nonlinear -- the most aggressive
    PREDICTABILITY-DESTROYING surrogate. Any policy that still profits here is harvesting the
    marginal/index, not temporal structure (the cleanest OVERFIT signature)."""
    x = np.asarray(x, dtype=float)
    return x[rng.permutation(x.size)]


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Iterative Amplitude Adjusted Fourier Transform surrogate (Schreiber & Schmitz 1996).

    The GOLD-STANDARD surrogate: preserves BOTH the power spectrum (autocorrelation) AND the exact
    amplitude (marginal) distribution. Alternates two projections until convergence:
      (1) impose the target power spectrum (keep target |X(f)|, keep current phases);
      (2) impose the target marginal (rank-remap the values onto the sorted original amplitudes).
    A policy that beats a phase + block + IAAFT trio is exploiting temporal structure those three -- which
    between them fix the spectrum AND the marginal -- cannot reproduce."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sorted_amp = np.sort(x)                      # target marginal (exact amplitudes)
    target_mag = np.abs(np.fft.rfft(x))          # target power spectrum
    # init: a random permutation of x (correct marginal, scrambled spectrum)
    s = x[rng.permutation(n)].copy()
    prev_err = np.inf
    for _ in range(n_iter):
        # (1) impose the target spectrum, keep current phases
        S = np.fft.rfft(s)
        phases = np.angle(S)
        s = np.fft.irfft(target_mag * np.exp(1j * phases), n=n)
        # (2) impose the exact marginal by rank-remap onto the sorted target amplitudes
        ranks = np.argsort(np.argsort(s))
        s = sorted_amp[ranks]
        # convergence on the spectrum-matching error
        err = float(np.mean((np.abs(np.fft.rfft(s)) - target_mag) ** 2))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return s


_SURROGATES: Dict[str, Callable] = {
    "perm": index_permute,    # PRIMARY predictability-destroyer (block=1; destroys ALL temporal structure)
    "block": block_shuffle,   # PRIMARY predictability-destroyer (small block; destroys structure > block)
    "phase": phase_randomize, # SECONDARY mechanism diagnostic (preserves linear autocorr)
    "iaaft": iaaft,           # SECONDARY mechanism diagnostic (preserves spectrum AND marginal)
}

# The surrogate families split into two ROLES (finding #9):
#   PREDICTABILITY-DESTROYING (perm, block): destroy temporal predictability (linear AND nonlinear)
#       while preserving the marginal EXACTLY. These are the VALID primary genuine/overfit
#       discriminators -- a GENUINE policy (which exploits SOME temporal structure) COLLAPSES on them.
#   MECHANISM DIAGNOSTICS (phase, iaaft): PRESERVE the power spectrum == linear autocorrelation. An
#       autocorr-exploiting GENUINE policy SURVIVES on these, so they must NOT drive the verdict (they
#       would FALSE-FLAG it OVERFIT). They are reported to diagnose WHICH structure the edge uses:
#       survives-iaaft-but-collapses-on-perm == a LINEAR (autocorrelation) exploit; collapses on BOTH
#       == a NONLINEAR/higher-order exploit.
_PREDICTABILITY_DESTROYING = ("perm", "block")
_MECHANISM_DIAGNOSTIC = ("phase", "iaaft")


def _make_surrogate(kind: str, x: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    if kind == "perm":
        return index_permute(x, rng)
    if kind == "block":
        return block_shuffle(x, rng, block=block)
    if kind == "phase":
        return phase_randomize(x, rng)
    if kind == "iaaft":
        return iaaft(x, rng)
    raise ValueError(f"unknown surrogate kind {kind!r}; expected one of {sorted(_SURROGATES)}")


# ======================================================================================================
# THE GATE
# ======================================================================================================
@dataclass
class ShuffledMarketResult:
    verdict: str                       # GENUINE / OVERFIT / INCONCLUSIVE
    real_compound: float
    surrogate_compound_p50: float
    surrogate_compound_p95: float
    surrogate_vs_real_gap: float       # real - surrogate_p50 (the structure-derived part of the edge)
    overfit_fraction: float            # surrogate_p50 / real  (the OVERFIT signature; ~0 == genuine)
    beats_surrogate: bool              # real compound beats the surrogate p95 band
    per_kind: Dict[str, dict] = field(default_factory=dict)
    n_surrogates: int = 0
    cost_rt: float = TAKER_COST_RT

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def shuffled_market_control(
    real_returns: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    kinds: Sequence[str] = ("perm", "block", "phase", "iaaft"),
    verdict_kinds: Sequence[str] = ("perm", "block"),
    n_surrogates: int = 200,
    seed: int = 7,
    block: int = 4,
    overfit_frac_threshold: float = 0.35,
    genuine_frac_threshold: float = 0.20,
) -> ShuffledMarketResult:
    """Re-score `policy` on the REAL return stream and on a distribution of SURROGATE markets, then issue
    a GENUINE / OVERFIT verdict.

    real_returns : 1-D per-bar SIMPLE returns of the real (held-out) market.
    policy       : a callable `policy(returns) -> per-trade NET return stream` (cost already inside).
                   The SAME policy object is applied to real and to every surrogate (no refit).
    kinds        : which surrogate families to COMPUTE + report (all four by default).
    verdict_kinds: which families DRIVE the verdict. Default = (perm, block) -- the PREDICTABILITY-
                   DESTROYING surrogates (finding #9). They scramble ALL temporal predictability (linear
                   AND nonlinear) while preserving the marginal EXACTLY, so a GENUINE policy (which exploits
                   SOME temporal structure -- linear or nonlinear) COLLAPSES to ~0 on them. `phase` and
                   `iaaft` are computed + reported but EXCLUDED from the verdict because they PRESERVE the
                   power spectrum == linear autocorrelation: an autocorrelation-exploiting GENUINE policy
                   SURVIVES on them and would be FALSE-FLAGGED OVERFIT. They are kept as SECONDARY MECHANISM
                   diagnostics (does the edge survive iaaft? -> it is a LINEAR/autocorr exploit; does it
                   collapse on both? -> NONLINEAR). (phase additionally GAUSSIANIZES the marginal, inflating
                   the band variance on fat-tailed returns -- a second reason it is not a verdict driver.)
    block        : block-shuffle length. CORRECTNESS REQUIREMENT (RWYB finding): block MUST be SHORTER than
                   the policy's setup->payoff horizon, else the shuffled blocks keep the pattern intact
                   inside each block and the surrogate fails to destroy it (a block == pattern length gave
                   overfit_fraction ~0.96 on a genuine pattern policy). Default 4; lower it below the setup
                   span for finer multi-bar patterns. (`perm` is block=1 -- the most aggressive destroyer --
                   and needs no length tuning.)
    overfit_frac_threshold : if surrogate_p50 / real >= this on the verdict surrogates, the policy keeps too
                   much profit on the surrogate -> OVERFIT (memorized the path/index, not the structure).
    genuine_frac_threshold : if surrogate_p50 / real <= this AND real beats the verdict surrogates' p95 band,
                   the surrogate residual is ~0 -> GENUINE (the edge came from real temporal structure).

    SCOPE -- NECESSARY, NOT SUFFICIENT (finding #11): this gate is a NECESSARY condition for a genuine
    edge, NOT a sufficient one. The predictability-destroying surrogate preserves the marginal AT THE SAME
    INDICES, so an INDEX-KEYED curve-fitter (a lookup that replays in-sample-best entries by bar position)
    can STILL 'profit' on the surrogate and be correctly flagged OVERFIT -- but a policy that COVERTLY keys
    on the index while LOOKING structure-driven could pass. Compose this gate with the held-out / walk-
    forward refit discipline (battery.py, pbo_cscv.py): GENUINE here means 'the edge did not survive
    predictability destruction', which REFUTES path-memorization but does NOT by itself prove out-of-sample
    generalization. Treat a GENUINE verdict as one necessary hurdle cleared, not a ship signal.

    VERDICT LOGIC (two-sided by construction; driven by the PREDICTABILITY-DESTROYING `verdict_kinds`):
      GENUINE  <- real (>0) AND overfit_fraction (the MEDIAN surrogate / real) <= genuine_frac AND real beats
                  the p95 band on AT LEAST ONE verdict surrogate. The MEDIAN ratio is the primary, tail-robust
                  signal (the predictability-destroying surrogate kills the edge); `beats_p95` on >=1 verdict
                  surrogate is the secondary robustness check. We require >=1 (not ALL) because a marginal-
                  preserving surrogate with a FAT-TAILED return distribution (jumps) produces a wide p95 band
                  that can veto a genuine edge even when its MEDIAN retains ~0 profit. perm/block (which
                  destroy predictability most cleanly) carry the beats_p95 evidence.
      OVERFIT  <- overfit_fraction >= overfit_frac (still profits after predictability is destroyed).
      INCONCLUSIVE otherwise (the middle band -- treat as a soft fail; do NOT ship)."""
    real_returns = np.asarray(real_returns, dtype=float)
    if real_returns.ndim != 1:
        raise ValueError(f"real_returns must be 1-D per-bar returns; got shape {real_returns.shape}")
    vk = [k for k in verdict_kinds if k in kinds]
    if not vk:
        raise ValueError(f"verdict_kinds {tuple(verdict_kinds)} must be a non-empty subset of kinds {tuple(kinds)}")
    if not all(k in _PREDICTABILITY_DESTROYING for k in vk):
        raise ValueError(
            f"verdict_kinds must be PREDICTABILITY-DESTROYING surrogates {_PREDICTABILITY_DESTROYING} "
            f"(finding #9: spectrum-preserving surrogates phase/iaaft false-flag an autocorr-exploiting "
            f"genuine policy as OVERFIT); got {tuple(verdict_kinds)}. phase/iaaft are mechanism diagnostics.")

    real_stream = np.asarray(policy(real_returns), dtype=float)
    real_compound = _compound(real_stream)

    per_kind: Dict[str, dict] = {}
    verdict_compounds: List[float] = []
    # FINDING #10: each surrogate KIND gets its OWN seeded rng (derived deterministically from the
    # base seed + kind), so the result is INDEPENDENT of the ORDER kinds are iterated. A single shared
    # rng advanced across kinds made the per-kind bands order-dependent (iaaft p50 varied ~15x).
    base_ss = np.random.SeedSequence(seed)
    kind_seeds = {k: int(s.generate_state(1)[0])
                  for k, s in zip(_SURROGATES.keys(), base_ss.spawn(len(_SURROGATES)))}
    for kind in kinds:
        rng_k = np.random.default_rng(kind_seeds[kind])   # per-kind independent stream
        comps = []
        for _ in range(n_surrogates):
            surr = _make_surrogate(kind, real_returns, rng_k, block)
            stream = np.asarray(policy(surr), dtype=float)
            comps.append(_compound(stream))
        comps = np.asarray(comps, dtype=float)
        p50, p95 = float(np.percentile(comps, 50)), float(np.percentile(comps, 95))
        per_kind[kind] = {
            "surrogate_p50": round(p50, 6),
            "surrogate_p95": round(p95, 6),
            "beats_p95": bool(real_compound > p95),
            "overfit_fraction": round((p50 / real_compound) if abs(real_compound) > 1e-12 else 0.0, 4),
            "drives_verdict": bool(kind in vk),
            "role": ("predictability_destroying" if kind in _PREDICTABILITY_DESTROYING
                     else "mechanism_diagnostic"),
        }
        if kind in vk:
            verdict_compounds.extend(comps.tolist())

    surr = np.asarray(verdict_compounds, dtype=float)
    s_p50, s_p95 = float(np.percentile(surr, 50)), float(np.percentile(surr, 95))
    gap = real_compound - s_p50
    overfit_fraction = (s_p50 / real_compound) if abs(real_compound) > 1e-12 else 0.0
    # secondary robustness: real beats the p95 band on AT LEAST ONE verdict surrogate (the cleanest
    # structure-destroyer, typically block-shuffle, carries this; a fat-tailed IAAFT band may not, and
    # must not be allowed to veto a genuine median-zero edge -- see VERDICT LOGIC in the docstring).
    beats_surrogate = bool(real_compound > 0 and any(per_kind[k]["beats_p95"] for k in vk))

    if real_compound > 0 and overfit_fraction <= genuine_frac_threshold and beats_surrogate:
        verdict = "GENUINE"
    elif overfit_fraction >= overfit_frac_threshold:
        verdict = "OVERFIT"
    else:
        verdict = "INCONCLUSIVE"

    return ShuffledMarketResult(
        verdict=verdict,
        real_compound=round(real_compound, 6),
        surrogate_compound_p50=round(s_p50, 6),
        surrogate_compound_p95=round(s_p95, 6),
        surrogate_vs_real_gap=round(gap, 6),
        overfit_fraction=round(float(overfit_fraction), 4),
        beats_surrogate=beats_surrogate,
        per_kind=per_kind,
        n_surrogates=n_surrogates,
        cost_rt=TAKER_COST_RT,
    )


# ======================================================================================================
# SURROGATE-CORRECTNESS SELF-CHECK -- prove the surrogates do what the docstring claims
# ======================================================================================================
def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-18 or lag >= x.size:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def verify_surrogate_correctness(seed: int = 0) -> dict:
    """Confirm: (a) IAAFT preserves the marginal (sorted values ~ equal) AND the lag-1 autocorr;
    (b) phase_randomize preserves the power spectrum (lag-1 autocorr ~ equal); (c) block_shuffle
    preserves the marginal EXACTLY (it is a permutation)."""
    rng = np.random.default_rng(seed)
    # an autocorrelated series (AR(1)) so there IS temporal structure to preserve/destroy
    n = 512
    ar = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + eps[t]
    ac1_real = _autocorr(ar, 1)

    ph = phase_randomize(ar, rng)
    bl = block_shuffle(ar, rng, block=8)
    ia = iaaft(ar, rng)

    out = {
        "real_lag1_autocorr": round(ac1_real, 4),
        "phase_lag1_autocorr": round(_autocorr(ph, 1), 4),
        "iaaft_lag1_autocorr": round(_autocorr(ia, 1), 4),
        "block_marginal_exact": bool(np.allclose(np.sort(bl), np.sort(ar))),
        "iaaft_marginal_match": round(float(np.max(np.abs(np.sort(ia) - np.sort(ar)))), 6),
        "phase_marginal_gaussianized": True,  # documented side effect (mean/var preserved, shape Gaussian)
    }
    # the structure-preserving claims (loose bands -- surrogates are stochastic)
    out["phase_preserves_autocorr"] = bool(abs(out["phase_lag1_autocorr"] - ac1_real) < 0.20)
    out["iaaft_preserves_autocorr"] = bool(abs(out["iaaft_lag1_autocorr"] - ac1_real) < 0.25)
    out["iaaft_preserves_marginal"] = bool(out["iaaft_marginal_match"] < 1e-6)
    return out


# ======================================================================================================
# TWO-SIDED CONTROLS (the MANDATORY positive + negative demonstration)
#
# A CLEAN positive control must satisfy BOTH:
#   (i)  it profits on REAL purely from exploiting TEMPORAL STRUCTURE -- so its marginal is ZERO-DRIFT
#        (a blind long-everything harvester makes ~0 after cost -> the only way to profit is timing);
#   (ii) the structure it exploits is NONLINEAR (a multi-bar pattern), so it is destroyed by the
#        marginal-preserving surrogates (block-shuffle below the pattern length, and IAAFT, which
#        preserves only the 2nd-order/linear spectrum). A purely LINEAR (lag-1) momentum edge is NOT a
#        good positive control: phase/IAAFT preserve the power spectrum == all linear autocorrelation,
#        so a lag-1 linear edge SURVIVES on them (RWYB finding). The nonlinear pattern is the honest test.
# ======================================================================================================
def make_structured_market(seed: int = 1, n_blocks: int = 200, vol: float = 0.004,
                           down: float = 0.03) -> np.ndarray:
    """ZERO-DRIFT market with a NONLINEAR, multi-bar SETUP->PAYOFF pattern (the positive control's world).

    Each 4-bar block is [down, down, up, flat] with up == 2*down, so the block (and the whole series) has
    ~ZERO net drift: a blind long-everything harvester earns ~0 (negative after cost). The ONLY way to
    profit is to TIME the 3rd bar by recognizing the two preceding down bars -- a genuine multi-candle
    SETUP across a MOVE (the founding framing's unit of trading). This pattern is destroyed by block-shuffle
    (block < 4) and by IAAFT (it is higher-order, not in the power spectrum)."""
    rng = np.random.default_rng(seed)
    up = 2.0 * down
    r: List[float] = []
    for _ in range(n_blocks):
        r += [-down + rng.normal(0, vol), -down + rng.normal(0, vol),
              up + rng.normal(0, vol), rng.normal(0, vol)]
    return np.asarray(r, dtype=float)


def make_drift_market(seed: int = 2, n: int = 800, drift: float = 0.004, vol: float = 0.01) -> np.ndarray:
    """A market with POSITIVE DRIFT but NO exploitable temporal structure (i.i.d. + drift). The negative
    control's world: profit here is pure MARGINAL/beta, which every marginal-preserving surrogate keeps."""
    rng = np.random.default_rng(seed)
    return drift + rng.normal(0.0, vol, n)


def genuine_pattern_policy(cost_rt: float = TAKER_COST_RT):
    """POSITIVE control policy. PAST-ONLY: it recognizes the SETUP (two consecutive down bars) and goes
    LONG the next bar (the payoff), paying round-trip cost INSIDE the trade. On the REAL pattern market it
    captures the engineered up-bar -> profits. On a surrogate that destroys the down-down->up linkage
    (block-shuffle below the pattern length, IAAFT) the trigger fires on coin-flip continuations and it
    degrades to ~0 (paying cost on noise) -> the GENUINE signature (surrogate residual ~0)."""
    def policy(returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, float)
        nets = []
        for t in range(2, returns.size):
            if returns[t - 1] < 0 and returns[t - 2] < 0:  # past-only multi-bar setup
                nets.append(returns[t] - cost_rt)          # long the payoff bar, cost inside
        return np.asarray(nets, float)
    return policy


def make_autocorr_market(seed: int = 5, n: int = 2000, ar: float = 0.70, vol: float = 0.012) -> np.ndarray:
    """ZERO-MEAN AR(1) market (strong positive lag-1 autocorrelation, NO drift). The LINEAR positive
    control's world: a blind long-everything harvester earns ~0 (no drift), so the only edge is
    TIMING off the LINEAR autocorrelation (yesterday-up -> today-up). The autocorr (ar=0.70) and vol
    are sized so the conditional mean after an up-bar (~ar * 0.8 * sigma_x, sigma_x = vol/sqrt(1-ar^2))
    CLEARS the round-trip cost -> the genuine lag-1 policy actually PROFITS on REAL. This is the case
    finding #9 is about: phase/iaaft PRESERVE this autocorrelation, so the autocorr policy SURVIVES on
    them (would be false-flagged OVERFIT) but COLLAPSES on the predictability-destroying perm/block."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = ar * x[t - 1] + rng.normal(0.0, vol)
    return x - x.mean()  # strip any residual drift so profit can ONLY come from timing


def autocorr_momentum_policy(cost_rt: float = TAKER_COST_RT):
    """LINEAR positive control policy. PAST-ONLY lag-1 momentum: go LONG the next bar after an UP
    bar. On the AR(1) market it captures the positive autocorrelation -> profits on REAL. It is a
    GENUINE temporal-structure exploit, so it MUST verdict GENUINE -- but because phase/iaaft
    PRESERVE the autocorrelation it exploits, it would SURVIVE on them (the finding-#9 false-flag).
    The predictability-destroying perm/block surrogates correctly collapse it -> GENUINE."""
    def policy(returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, float)
        nets = []
        for t in range(1, returns.size):
            if returns[t - 1] > 0:                 # past-only lag-1 up -> ride the autocorrelation
                nets.append(returns[t] - cost_rt)
        return np.asarray(nets, float)
    return policy


def beta_harvester_policy(cost_rt: float = TAKER_COST_RT):
    """NEGATIVE control policy. The policy ShIC=0 failure mode in its purest form: a NO-TIMING-SKILL
    policy that just goes LONG EVERY bar (harvests the MARGINAL / beta). On a positive-drift market it
    'profits', but that profit is a property of the MARGINAL, which phase/block/IAAFT all PRESERVE -> it
    profits IDENTICALLY on the surrogate (overfit_fraction ~ 1.0) -> the gate MUST flag OVERFIT. This is
    the cleanest possible negative control: a beta bet wearing a backtest curve, with zero genuine
    temporal-structure edge for any surrogate to remove."""
    def policy(returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, float)
        return returns - cost_rt  # long every bar, cost inside -- pure marginal harvest, no timing
    return policy


def run_two_sided_demo(verbose: bool = True, seed: int = 1, n_surrogates: int = 200) -> dict:
    """The mandatory two-sided demonstration. Returns the verdict for the controls + the surrogate
    self-correctness check. Exit-0 from __main__ iff positives -> GENUINE and negative -> OVERFIT.

    Positive A (NONLINEAR): a genuine multi-bar setup->payoff policy on a ZERO-DRIFT structured market
      -> profits on real, collapses on the PREDICTABILITY-DESTROYING surrogates (perm/block) -> GENUINE.
    Positive B (LINEAR autocorr -- the finding-#9 case): a lag-1 momentum policy on a ZERO-DRIFT AR(1)
      market. It exploits LINEAR autocorrelation, which phase/iaaft PRESERVE -> it would SURVIVE on iaaft
      (the OLD false-flag), but the PREDICTABILITY-DESTROYING perm/block collapse it -> GENUINE. We assert
      BOTH that the verdict (perm/block-driven) is GENUINE AND that it WOULD have survived iaaft -- proving
      the demotion of iaaft to a non-verdict diagnostic is what makes the gate sound.
    Negative: a beta/marginal-harvester on a POSITIVE-DRIFT i.i.d. market -> profits on real AND on every
      surrogate (perm/block keep the marginal) -> OVERFIT. `block=2` is below the 4-bar pattern span."""
    pos_market = make_structured_market(seed=seed)
    pos = shuffled_market_control(pos_market, genuine_pattern_policy(),
                                  n_surrogates=n_surrogates, seed=seed, block=2)

    # LINEAR autocorr positive control (finding #9): verdict from perm/block (predictability-destroying);
    # ALSO compute an iaaft-driven verdict to SHOW it would have false-flagged this genuine policy OVERFIT.
    lin_market = make_autocorr_market(seed=seed + 4)
    lin = shuffled_market_control(lin_market, autocorr_momentum_policy(),
                                  n_surrogates=n_surrogates, seed=seed, block=2)
    lin_iaaft_overfit_frac = lin.per_kind["iaaft"]["overfit_fraction"]  # ~1 => iaaft keeps the edge

    neg_market = make_drift_market(seed=seed + 1)
    neg = shuffled_market_control(neg_market, beta_harvester_policy(),
                                  n_surrogates=n_surrogates, seed=seed, block=8)

    surr_check = verify_surrogate_correctness(seed=seed)

    if verbose:
        print("=" * 100)
        print("A2 SHUFFLED-MARKET CONTROL -- the policy-overfit detector (RL analog of ShIC). RWYB (CPU, synthetic).")
        print("=" * 100)
        print("\n[surrogate correctness] make-the-surrogate-correct self-check:")
        print(f"  real lag-1 autocorr      = {surr_check['real_lag1_autocorr']:+.4f}")
        print(f"  phase  lag-1 autocorr    = {surr_check['phase_lag1_autocorr']:+.4f}  "
              f"(preserves power spectrum: {surr_check['phase_preserves_autocorr']})")
        print(f"  iaaft  lag-1 autocorr    = {surr_check['iaaft_lag1_autocorr']:+.4f}  "
              f"(preserves spectrum: {surr_check['iaaft_preserves_autocorr']})")
        print(f"  block  marginal EXACT    = {surr_check['block_marginal_exact']}  (permutation of the original)")
        print(f"  iaaft  marginal max|dif| = {surr_check['iaaft_marginal_match']:.2e}  "
              f"(preserves marginal: {surr_check['iaaft_preserves_marginal']})")

        print("\n[POSITIVE A: NONLINEAR] genuine multi-bar setup->payoff policy on a ZERO-DRIFT market:")
        print(f"  real_compound        = {pos.real_compound*100:+.3f}%")
        print(f"  surrogate p50 / p95  = {pos.surrogate_compound_p50*100:+.3f}% / {pos.surrogate_compound_p95*100:+.3f}%  "
              f"(verdict surrogates = perm+block, PREDICTABILITY-DESTROYING)")
        print(f"  surrogate-vs-real gap= {pos.surrogate_vs_real_gap*100:+.3f}%   overfit_fraction(median)={pos.overfit_fraction:+.4f}")
        print(f"  beats_surrogate(>=1 p95) = {pos.beats_surrogate}   per_kind={json.dumps(pos.per_kind)}")
        print(f"  ==> VERDICT: {pos.verdict}   (expected GENUINE: collapses on perm/block)")

        print("\n[POSITIVE B: LINEAR autocorr -- finding #9] lag-1 momentum on a ZERO-DRIFT AR(1) market:")
        print(f"  real_compound        = {lin.real_compound*100:+.3f}%")
        print(f"  verdict (perm/block) overfit_fraction = {lin.overfit_fraction:+.4f}  ==> VERDICT: {lin.verdict}")
        print(f"  iaaft overfit_fraction = {lin_iaaft_overfit_frac:+.4f}  (>~{genuine_frac_default()} => iaaft "
              f"KEEPS this autocorr edge: it WOULD have FALSE-FLAGGED it OVERFIT -- WHY iaaft is demoted)")

        print("\n[NEGATIVE control] beta/marginal-harvester policy (long every bar; no timing skill) on +drift market:")
        print(f"  real_compound        = {neg.real_compound*100:+.3f}%")
        print(f"  surrogate p50 / p95  = {neg.surrogate_compound_p50*100:+.3f}% / {neg.surrogate_compound_p95*100:+.3f}%")
        print(f"  surrogate-vs-real gap= {neg.surrogate_vs_real_gap*100:+.3f}%   overfit_fraction={neg.overfit_fraction:+.3f}")
        print(f"  beats_surrogate(p95) = {neg.beats_surrogate}   per_kind={json.dumps(neg.per_kind)}")
        print(f"  ==> VERDICT: {neg.verdict}   (expected OVERFIT: still profits after predictability destroyed)")

    pos_ok = (pos.verdict == "GENUINE")
    lin_ok = (lin.verdict == "GENUINE")
    # finding #9 PROOF: the linear edge SURVIVES iaaft (would be false-flagged) but the primary catches it.
    lin_iaaft_would_false_flag = bool(lin_iaaft_overfit_frac >= 0.35)
    neg_ok = (neg.verdict == "OVERFIT")
    surr_ok = (surr_check["block_marginal_exact"] and surr_check["iaaft_preserves_marginal"]
               and surr_check["phase_preserves_autocorr"] and surr_check["iaaft_preserves_autocorr"])

    if verbose:
        print("\n" + "=" * 100)
        print("TWO-SIDED VALIDATION:")
        print(f"  [{'PASS' if pos_ok else 'FAIL'}] positive A (nonlinear) -> GENUINE  (perm/block collapse it)")
        print(f"  [{'PASS' if lin_ok else 'FAIL'}] positive B (linear autocorr) -> GENUINE  (finding #9: perm/block "
              f"catch it; iaaft would NOT)")
        print(f"  [{'PASS' if lin_iaaft_would_false_flag else 'FAIL'}] finding #9 PROOF: iaaft KEEPS the linear edge "
              f"(overfit_frac={lin_iaaft_overfit_frac:+.3f}) -> it would have FALSE-FLAGGED it; demotion is correct")
        print(f"  [{'PASS' if neg_ok else 'FAIL'}] negative control -> OVERFIT  (the gate FLAGS a curve-fit memorizer)")
        print(f"  [{'PASS' if surr_ok else 'FAIL'}] surrogate correctness (marginal + spectrum preserved)")
        ok = pos_ok and lin_ok and lin_iaaft_would_false_flag and neg_ok and surr_ok
        print("\n" + ("ALL TWO-SIDED CHECKS HOLD -- the gate ACCEPTS genuine (linear AND nonlinear) AND FLAGS "
                      "overfit; the predictability-destroying primary is sound where iaaft was not."
                      if ok else "*** TWO-SIDED CHECK VIOLATED -- gate is mis-calibrated, inspect above ***"))
        print("JSON_SUMMARY " + json.dumps({
            "positive_verdict": pos.verdict, "linear_verdict": lin.verdict, "negative_verdict": neg.verdict,
            "positive": pos.to_dict(), "linear": lin.to_dict(), "negative": neg.to_dict(),
            "linear_iaaft_overfit_fraction": lin_iaaft_overfit_frac,
            "surrogate_correctness": surr_check, "all_pass": ok,
        }, default=str))

    return {
        "positive": pos.to_dict(), "linear": lin.to_dict(), "negative": neg.to_dict(),
        "surrogate_correctness": surr_check,
        "linear_iaaft_overfit_fraction": lin_iaaft_overfit_frac,
        "pos_ok": pos_ok, "lin_ok": lin_ok, "lin_iaaft_would_false_flag": lin_iaaft_would_false_flag,
        "neg_ok": neg_ok, "surr_ok": surr_ok,
        "all_pass": bool(pos_ok and lin_ok and lin_iaaft_would_false_flag and neg_ok and surr_ok),
    }


def genuine_frac_default() -> float:
    """The default genuine_frac_threshold (for the demo's explanatory print)."""
    return 0.20


def main() -> int:
    res = run_two_sided_demo(verbose=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
