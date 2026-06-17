"""src/firm/portfolio.py -- PORTFOLIO CONSTRUCTION: combine many per-setup bets (each sized by the decision_spine)
into ONE risk-budgeted portfolio. A firm does not bet each setup independently at full size -- it allocates a TOTAL
risk budget across correlated bets, vol-targets the book, and caps gross + per-name exposure.

Composes directly with decision_spine: decision_spine.decide() sizes each candidate bet; allocate() here turns the set
of bets into portfolio weights under a single risk budget. Pure / deterministic / no look-ahead / long-only-capable.

__contract__ declared for CDAP. No emoji (cp1252).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["positions [(asset, raw_fraction, vol)]", "correlation matrix (optional)", "PortfolioLimits"],
    "outputs": ["Allocation(weights, gross, port_vol, scaled_by, notes)"],
    "invariants": [
        "portfolio vol is targeted to vol_target (scale all weights by vol_target/port_vol)",
        "gross exposure (sum |w|) never exceeds max_gross; per-name |w| never exceeds max_per_name",
        "long_only -> all weights >= 0; correlated bets do NOT get double-counted (uses the covariance, not sum of vols)",
    ],
}


@dataclass(frozen=True)
class Position:
    asset: str
    raw_fraction: float   # the per-bet stake from decision_spine.decide().fraction (>=0)
    vol: float            # per-period return std of the asset/setup (the bet's risk)


@dataclass(frozen=True)
class PortfolioLimits:
    vol_target: float = 0.02      # target per-period portfolio volatility (e.g. 2%/period)
    max_gross: float = 1.0        # cap on sum of |weights| (1.0 = fully invested, long-only, no leverage)
    max_per_name: float = 0.25    # cap on any single name's |weight|
    long_only: bool = True


@dataclass(frozen=True)
class Allocation:
    weights: dict
    gross: float
    port_vol: float
    scaled_by: float
    notes: list = field(default_factory=list)


def _cov(positions: list, corr: list | None) -> list:
    n = len(positions)
    vols = [p.vol for p in positions]
    if corr is None:
        # assume independence -> diagonal covariance
        return [[(vols[i] ** 2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    return [[corr[i][j] * vols[i] * vols[j] for j in range(n)] for i in range(n)]


def _port_vol(w: list, cov: list) -> float:
    n = len(w)
    var = sum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n))
    return math.sqrt(max(var, 0.0))


def allocate(positions: list, corr: list | None = None, limits: PortfolioLimits | None = None) -> Allocation:
    limits = limits or PortfolioLimits()
    notes = []
    if not positions:
        return Allocation({}, 0.0, 0.0, 0.0, ["no positions"])
    n = len(positions)
    w = [max(p.raw_fraction, 0.0) if limits.long_only else p.raw_fraction for p in positions]
    cov = _cov(positions, corr)

    # 1) vol-target the book: scale all weights so portfolio vol == vol_target (correlation-aware via the covariance)
    pv = _port_vol(w, cov)
    scaled_by = (limits.vol_target / pv) if pv > 1e-12 else 0.0
    w = [wi * scaled_by for wi in w]
    if pv <= 1e-12:
        notes.append("zero portfolio vol (no risk) -> flat book")

    # 2) per-name cap
    capped = False
    for i in range(n):
        if abs(w[i]) > limits.max_per_name:
            w[i] = math.copysign(limits.max_per_name, w[i]); capped = True
    if capped:
        notes.append(f"per-name cap {limits.max_per_name} bound")

    # 3) gross cap (sum |w|)
    gross = sum(abs(wi) for wi in w)
    if gross > limits.max_gross and gross > 0:
        shrink = limits.max_gross / gross
        w = [wi * shrink for wi in w]
        notes.append(f"gross cap {limits.max_gross} bound (shrank x{shrink:.3f})")

    weights = {positions[i].asset: round(w[i], 5) for i in range(n)}
    final_gross = round(sum(abs(wi) for wi in w), 5)
    return Allocation(weights, final_gross, round(_port_vol(w, cov), 5), round(scaled_by, 4), notes)


def _selftest():
    print("=== portfolio selftest ===")
    # 3 bets; A and B are highly correlated (0.9), C independent
    pos = [Position("A", 0.20, 0.05), Position("B", 0.20, 0.05), Position("C", 0.20, 0.08)]
    corr = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
    al = allocate(pos, corr, PortfolioLimits(vol_target=0.02, max_gross=1.0, max_per_name=0.25))
    print(f"  weights={al.weights} gross={al.gross} port_vol={al.port_vol} scaled_by={al.scaled_by}")
    print(f"  notes={al.notes}")
    assert abs(al.port_vol - 0.02) < 1e-3 or al.gross == 1.0, "vol target or gross cap should bind"
    assert all(abs(v) <= 0.25 + 1e-9 for v in al.weights.values()), "per-name cap"
    assert al.gross <= 1.0 + 1e-9, "gross cap"
    # independence vs correlation: correlated book needs MORE downscaling for the same vol target
    al_indep = allocate(pos, None, PortfolioLimits(vol_target=0.02))
    print(f"  indep scaled_by={al_indep.scaled_by} vs corr scaled_by={al.scaled_by} (corr should scale DOWN more)")
    assert al.scaled_by <= al_indep.scaled_by + 1e-9, "correlated book is riskier -> smaller scale"
    print("  ALL PASS -- vol-target + correlation-awareness + gross/per-name caps hold.")


if __name__ == "__main__":
    _selftest()
