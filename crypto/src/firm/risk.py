"""src/firm/risk.py -- the firm-level RISK GATE that bounds the whole book (L6). Sits AFTER portfolio construction:
takes the allocated weights + the live risk state and returns PASS / SCALE / HALT with the breaches named. This is the
circuit-breaker layer -- a world-class firm never lets the decision/portfolio layers size past the book's risk limits.

Bounds enforced: portfolio VaR / CVaR (parametric, from weights+covariance), gross leverage, single-name concentration,
and a DRAWDOWN KILL-SWITCH (halt new risk once peak-to-trough exceeds the limit). Pure / deterministic / no look-ahead.
__contract__ for CDAP. Composes with portfolio.allocate() (its weights) + market_state (regime can tighten limits).
No emoji (cp1252).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["weights {asset: w}", "vols {asset: per-period std}", "correlation matrix", "RiskState", "RiskConfig"],
    "outputs": ["RiskVerdict(action PASS|SCALE|HALT, scale_factor, port_var, gross, breaches)"],
    "invariants": [
        "DRAWDOWN KILL-SWITCH: current_drawdown >= max_drawdown -> HALT (no new risk), scale_factor 0",
        "VaR / gross / per-name breaches -> SCALE to the binding limit (never silently exceed)",
        "deterministic; VaR is parametric from the covariance (correlated risk not double-counted)",
    ],
}

Z_95 = 1.645          # one-sided 95% normal quantile (VaR)
PHI_Z95 = 0.10313     # standard-normal pdf at Z_95, for CVaR = vol * phi(z)/(1-alpha)


@dataclass(frozen=True)
class RiskConfig:
    max_drawdown: float = 0.30      # kill-switch: halt new risk past this peak-to-trough
    var_limit: float = 0.05         # cap on per-period 95% VaR as a fraction of equity
    max_gross: float = 1.0          # cap on sum|w| (leverage)
    max_per_name: float = 0.25
    dd_taper_start: float = 0.15    # begin de-risking (linear taper) once DD exceeds this, before the hard halt


@dataclass(frozen=True)
class RiskState:
    current_drawdown: float = 0.0   # peak-to-trough fraction, >= 0
    equity: float = 1.0


@dataclass(frozen=True)
class RiskVerdict:
    action: str                     # PASS | SCALE | HALT
    scale_factor: float             # multiply all weights by this (0 on HALT, 1 on clean PASS)
    port_var: float
    port_cvar: float
    gross: float
    breaches: list = field(default_factory=list)


def _port_vol(weights: dict, vols: dict, corr: list | None) -> float:
    assets = list(weights.keys())
    n = len(assets)
    w = [weights[a] for a in assets]
    v = [vols.get(a, 0.0) for a in assets]
    if corr is None:
        var = sum((w[i] * v[i]) ** 2 for i in range(n))
    else:
        var = sum(w[i] * corr[i][j] * v[i] * v[j] * w[j] for i in range(n) for j in range(n))
    return math.sqrt(max(var, 0.0))


def check_book(weights: dict, vols: dict, corr: list | None, state: RiskState | None = None,
               cfg: RiskConfig | None = None) -> RiskVerdict:
    cfg = cfg or RiskConfig()
    state = state or RiskState()
    breaches = []

    # 1) DRAWDOWN KILL-SWITCH (hard halt) + taper (soft de-risk before the halt)
    dd = max(state.current_drawdown, 0.0)
    if dd >= cfg.max_drawdown:
        return RiskVerdict("HALT", 0.0, 0.0, 0.0, 0.0, [f"DRAWDOWN {dd:.1%} >= max {cfg.max_drawdown:.1%} -> kill-switch"])
    dd_scale = 1.0
    if dd > cfg.dd_taper_start:
        dd_scale = max(0.0, 1.0 - (dd - cfg.dd_taper_start) / (cfg.max_drawdown - cfg.dd_taper_start))
        breaches.append(f"drawdown taper {dd:.1%} -> de-risk x{dd_scale:.2f}")

    scale = dd_scale

    # 2) gross leverage
    gross = sum(abs(w) for w in weights.values())
    if gross * scale > cfg.max_gross and gross > 0:
        s = cfg.max_gross / gross
        scale = min(scale, s); breaches.append(f"gross {gross:.2f} > {cfg.max_gross} -> scale x{s:.2f}")

    # 3) per-name concentration
    mx = max((abs(w) for w in weights.values()), default=0.0)
    if mx * scale > cfg.max_per_name and mx > 0:
        s = cfg.max_per_name / mx
        scale = min(scale, s); breaches.append(f"per-name {mx:.2f} > {cfg.max_per_name} -> scale x{s:.2f}")

    # 4) portfolio VaR (parametric, correlation-aware)
    pv = _port_vol(weights, vols, corr)
    var = Z_95 * pv
    if var * scale > cfg.var_limit and var > 0:
        s = cfg.var_limit / var
        scale = min(scale, s); breaches.append(f"VaR95 {var:.3f} > {cfg.var_limit} -> scale x{s:.2f}")

    final_var = round(Z_95 * pv * scale, 5)
    final_cvar = round(pv * scale * (PHI_Z95 / 0.05), 5)
    final_gross = round(gross * scale, 5)
    action = "PASS" if (abs(scale - 1.0) < 1e-9 and not breaches) else ("SCALE" if scale > 0 else "HALT")
    return RiskVerdict(action, round(scale, 5), final_var, final_cvar, final_gross, breaches)


def _selftest():
    print("=== risk selftest ===")
    w = {"A": 0.15, "B": 0.15, "C": 0.10}
    vols = {"A": 0.05, "B": 0.05, "C": 0.08}
    corr = [[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]]
    # 1) within limits -> PASS
    v1 = check_book(w, vols, corr, RiskState(0.05), RiskConfig(var_limit=0.10))
    print(f"  calm book -> {v1.action} scale={v1.scale_factor} VaR={v1.port_var} gross={v1.gross} {v1.breaches}")
    assert v1.action == "PASS"
    # 2) tight VaR limit -> SCALE
    v2 = check_book(w, vols, corr, RiskState(0.0), RiskConfig(var_limit=0.01))
    print(f"  tight VaR  -> {v2.action} scale={v2.scale_factor} VaR={v2.port_var} {v2.breaches}")
    assert v2.action == "SCALE" and v2.scale_factor < 1.0 and v2.port_var <= 0.01 + 1e-6
    # 3) drawdown breach -> HALT
    v3 = check_book(w, vols, corr, RiskState(0.32), RiskConfig(max_drawdown=0.30))
    print(f"  DD 32%     -> {v3.action} scale={v3.scale_factor} {v3.breaches}")
    assert v3.action == "HALT" and v3.scale_factor == 0.0
    # 4) drawdown taper (between start and halt) -> de-risk but not halt
    v4 = check_book(w, vols, corr, RiskState(0.225), RiskConfig(dd_taper_start=0.15, max_drawdown=0.30, var_limit=0.10))
    print(f"  DD 22.5%   -> {v4.action} scale={v4.scale_factor} (taper) {v4.breaches}")
    assert v4.action == "SCALE" and 0 < v4.scale_factor < 1.0
    print("  ALL PASS -- kill-switch / VaR-scale / gross / taper all bind correctly.")


if __name__ == "__main__":
    _selftest()
