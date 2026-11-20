"""src/firm/trader_mindset.py -- the TRADING-MINDSET engine: how a great discretionary-quant trader JUDGES a setup
(the WHY / WHETHER), BEFORE the mechanical decision_spine sizes it (the HOW-MUCH). The mechanical layer answers "given a
forecast, how big?"; this answers "is this a setup a great trader would even take?" -- encoding the judgment that keeps a
desk alive: convex asymmetry, a defined invalidation, a NAMED edge source, regime-fit, survival-first, and second-order
(what's-priced-in) thinking, with crypto-specific lenses (reflexivity, narrative, liquidation fuel, funding-as-crowding).

WHY (user, 2026-06-06): "there are TRADING MINDSET ENGINES you can build." The mechanical engines are necessary but not
sufficient -- a desk's edge is as much JUDGMENT (which pitches to swing at, when you're wrong, why the edge exists) as
math. This makes that judgment mechanical + checkable. CRITICAL principles (no defined invalidation, unnamed edge
source, ruin-risk) VETO a setup regardless of how good the payoff looks -- the discipline that separates traders from
gamblers. Composes with decision_spine (sizing), problem_framing (breadth), market_state (regime). Pure / deterministic.
__contract__ for CDAP. No emoji (cp1252).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["Setup(reward, risk, invalidation_level, edge_source, regime_fit, crowdedness, conviction, max_loss_pct)"],
    "outputs": ["MindsetVerdict(verdict TAKE|MARGINAL|PASS, score, principle_scores, flags, vetoes)"],
    "invariants": [
        "CRITICAL principles (no invalidation / unnamed edge source / ruin-risk) VETO -> PASS regardless of score",
        "asymmetry uses reward:risk; what's-priced-in penalizes consensus (crowded) theses; deterministic",
    ],
}

RUIN_RISK_PCT = 0.25     # a single setup risking more than this fraction of equity is a survival veto


@dataclass(frozen=True)
class Setup:
    reward: float                 # expected favourable move (e.g. 0.08)
    risk: float                   # expected adverse move to the invalidation (e.g. 0.03) -- the bounded loss
    edge_source: str = ""         # WHY the edge exists / who is on the other side (empty = unnamed)
    invalidation_level: float | None = None   # the price/level where the thesis is WRONG (None = undefined)
    regime_fit: float = 0.5       # 0..1: does the setup match the current regime (1 = with the tape)
    crowdedness: float = 0.5      # 0..1: how CONSENSUS the thesis is (1 = everyone's on it, priced in)
    conviction: float = 0.5       # 0..1: is this a FAT PITCH (1) or a marginal setup (0)
    max_loss_pct: float = 0.05    # fraction of equity at risk on this setup (survival check)
    crypto_note: str = ""         # optional: reflexivity / narrative / liquidation / funding context


@dataclass(frozen=True)
class MindsetVerdict:
    verdict: str                  # TAKE | MARGINAL | PASS
    score: float                  # 0..1 weighted mindset quality
    principle_scores: dict
    flags: list = field(default_factory=list)     # soft concerns
    vetoes: list = field(default_factory=list)     # hard disqualifiers (force PASS)


# (key, weight, is_critical)
_PRINCIPLES = [
    ("asymmetry", 0.22, False),
    ("invalidation_defined", 0.18, True),
    ("edge_source_named", 0.18, True),
    ("regime_fit", 0.12, False),
    ("survival", 0.10, True),
    ("not_crowded", 0.12, False),
    ("selectivity", 0.08, False),
]


def assess(s: Setup) -> MindsetVerdict:
    ps, flags, vetoes = {}, [], []

    # asymmetry: reward:risk; 3:1+ is excellent, <1.5:1 is a flag
    rr = (s.reward / s.risk) if s.risk > 1e-9 else 0.0
    ps["asymmetry"] = max(0.0, min(1.0, rr / 3.0))
    if rr < 1.5:
        flags.append(f"weak asymmetry (reward:risk {rr:.2f} < 1.5) -- not convex enough")

    # invalidation defined (CRITICAL): a trade with no 'where am I wrong + out' is a gamble
    if s.invalidation_level is None:
        ps["invalidation_defined"] = 0.0
        vetoes.append("NO INVALIDATION LEVEL -- where are you wrong + out? (a trade without a stop-thesis is a gamble)")
    else:
        ps["invalidation_defined"] = 1.0

    # edge source named (CRITICAL): an edge with no named counterparty/inefficiency is usually noise
    if not s.edge_source.strip():
        ps["edge_source_named"] = 0.0
        vetoes.append("UNNAMED EDGE SOURCE -- who is on the other side, what inefficiency, why does it persist?")
    else:
        ps["edge_source_named"] = 1.0

    # regime fit: don't fight the tape
    ps["regime_fit"] = max(0.0, min(1.0, s.regime_fit))
    if s.regime_fit < 0.4:
        flags.append(f"poor regime fit ({s.regime_fit:.2f}) -- fighting the tape")

    # survival (CRITICAL): ruin-avoidance dominates everything
    ps["survival"] = 1.0 if s.max_loss_pct <= RUIN_RISK_PCT else max(0.0, 1.0 - (s.max_loss_pct - RUIN_RISK_PCT))
    if s.max_loss_pct > RUIN_RISK_PCT:
        vetoes.append(f"RUIN RISK -- {s.max_loss_pct:.0%} of equity on one setup (> {RUIN_RISK_PCT:.0%}); survive to compound")

    # what's priced in (second-order): consensus theses are already in the price
    ps["not_crowded"] = max(0.0, min(1.0, 1.0 - s.crowdedness))
    if s.crowdedness > 0.7:
        flags.append(f"crowded/consensus thesis ({s.crowdedness:.2f}) -- likely priced in (reflexivity: who's left to buy?)")

    # selectivity: wait for the fat pitch
    ps["selectivity"] = max(0.0, min(1.0, s.conviction))
    if s.conviction < 0.5:
        flags.append(f"marginal conviction ({s.conviction:.2f}) -- not a fat pitch; patience > activity")

    score = sum(ps[k] * w for k, w, _ in _PRINCIPLES)
    if vetoes:
        verdict = "PASS"
    elif score >= 0.65 and not flags:
        verdict = "TAKE"
    elif score >= 0.55:
        verdict = "MARGINAL"
    else:
        verdict = "PASS"
    return MindsetVerdict(verdict, round(score, 4), {k: round(v, 3) for k, v in ps.items()}, flags, vetoes)


def _selftest():
    print("=== trader_mindset selftest ===")
    # 1) a great setup: convex, defined invalidation, named edge, with-tape, not crowded, conviction
    good = Setup(reward=0.09, risk=0.03, edge_source="forced liquidation cascade -> reflexive overshoot, MM steps away",
                 invalidation_level=0.95, regime_fit=0.8, crowdedness=0.3, conviction=0.75, max_loss_pct=0.03)
    vg = assess(good)
    print(f"  great setup -> {vg.verdict} score={vg.score} flags={vg.flags} vetoes={vg.vetoes}")
    assert vg.verdict == "TAKE"
    # 2) no invalidation -> VETO regardless of a juicy payoff
    nostop = Setup(reward=0.20, risk=0.02, edge_source="momentum", invalidation_level=None, conviction=0.9)
    vn = assess(nostop)
    print(f"  no-invalidation (20:1!) -> {vn.verdict} :: {vn.vetoes}")
    assert vn.verdict == "PASS" and any("INVALIDATION" in x for x in vn.vetoes)
    # 3) crowded + unnamed edge + weak asymmetry -> PASS
    crowd = Setup(reward=0.02, risk=0.03, edge_source="", invalidation_level=0.9, crowdedness=0.9, conviction=0.4)
    vc = assess(crowd)
    print(f"  crowded/unnamed -> {vc.verdict} flags={len(vc.flags)} vetoes={vc.vetoes}")
    assert vc.verdict == "PASS"
    # 4) ruin risk -> VETO
    ruin = Setup(reward=0.10, risk=0.03, edge_source="x", invalidation_level=0.9, max_loss_pct=0.40)
    assert assess(ruin).verdict == "PASS"
    print("  ALL PASS -- critical vetoes (invalidation/edge-source/ruin) dominate; asymmetry+crowding scored.")


if __name__ == "__main__":
    _selftest()
