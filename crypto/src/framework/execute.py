"""src/framework/execute.py -- the EXECUTOR for the generic model engine.

solve.py::plan_solve() is a PLANNER (it routes, names the adapter/spine, prints an honest
ceiling -- but never trains). This module adds the EXECUTE path the planner deferred:

    execute_solve(problem, segments=None, **kw) -> SolveResult

It runs the real loop:
    REPRESENT -> ROUTE (reuse plan_solve for routing + ceiling) -> LEARN (actually call
    general_trainer.train_layer_b on the problem's segments) -> VALIDATE (run the
    src/strat/battery.py robustness spine on the held-out directional returns) -> return a
    REAL structured SolveResult (held-out IC, n, shuffled IC, battery verdict, honest ceiling).

LANE DISCIPLINE (honest):
  - ONLY Layer B (general forecasting) executes end-to-end here. It is the in-lane, CPU-friendly,
    domain-agnostic forecasting layer.
  - Layer A (games/self-play) and Layer C (crypto WM) are OUT-OF-LANE / compute-bound for this
    executor: it returns an HONEST `executed=False` SolveResult with
    `execution_deferred=<reason>` rather than fabricating a number. (Layer A needs a self-play
    GPU loop + champion gate; Layer C needs the chimera pipeline + cost model + the project's WM
    zoo. Both are reachable via plan_solve()'s next_actions -- this executor refuses to fake them.)

SPINE-WIRING CAVEAT (E2, stated honestly in the result):
  battery.py is a RETURNS / COMPOUND-oriented gate (block-bootstrap p05, jackknife, Lens A/B/C).
  A Layer-B target_return_<h> forecast is NOT itself a tradeable per-trade book. To run the spine
  WITHOUT misapplying it, the executor forms the most defensible tradeable object: a SIGN-OF-FORECAST
  directional strategy on the held-out UNSEEN bars --
        trade_ret[i] = sign(pred[i]) * realized_return[i].
  That IS a real return series, so block-bootstrap p05 / jackknife / compound / expectancy / PF are
  APPLICABLE and produce real numbers. What is N/A is marked N/A and WHY:
    - Lens A `all_4_positive` across TRAIN/VAL/OOS/UNSEEN: the minimal trainer exposes only the
      UNSEEN held-out arrays, so the per-split directional compounds for TRAIN/VAL/OOS are NOT
      computed here -> all_4_positive is reported as N/A (not silently faked as True).
    - Lens C monthly gate: only meaningful when the series carries real calendar timestamps spanning
      >=3 months. For synthetic / non-calendar generic series it is reported but flagged
      `calendar_meaningful=False`.
    - maxDD on the proxy equity curve is a PER-BAR proxy (each window = one bar-decision), not a
      real per-trade book, and is labelled as such.
  If the Layer-B target is a NON-RETURN generic signal (e.g. a physical quantity, not a price
  return), `sign(pred)*target` is NOT a tradeable return at all -> the executor still computes the
  IC diagnostics but marks the battery block `returns_interpretable=False` and does NOT assert any
  trading verdict. The caller declares this via problem['target_is_return'] (default True for
  crypto/time_series price-return targets).

No emoji (Windows cp1252). torch via general_trainer; numpy; battery is numpy-only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# CDAP contract
# ---------------------------------------------------------------------------
__contract__ = {
    "kind": "executor",
    "module": "execute_solve -> SolveResult (the engine's EXECUTE path)",
    "inputs": [
        "problem: dict (router contract) + optional segments: List[dict] "
        "(GeneralAdapter.to_segments() output)",
    ],
    "outputs": [
        "SolveResult: {executed, layer, held_out_ic, shuffled_ic, n_unseen, battery, "
        "honest_ceiling, execution_deferred, notes}",
    ],
    "invariants": {
        "layer_b_only_executes": "only Layer B trains end-to-end; A/C return executed=False + "
                                 "execution_deferred reason (never a fabricated number)",
        "spine_runs": "battery.py (block-bootstrap p05 + jackknife + lens gates) runs on the "
                      "held-out directional returns and produces REAL numbers",
        "honest_na": "non-applicable spine pieces are marked N/A with a reason, never silently "
                     "misapplied (returns_interpretable flag gates the trading verdict)",
        "no_lookahead": "held-out arrays come from the UNSEEN split only (train_layer_b guarantee)",
        "planner_intact": "reuses plan_solve() for routing+ceiling; does not replace it",
    },
}


# ---------------------------------------------------------------------------
# SolveResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    """Structured output of execute_solve(). Serialisable to JSON.

    A REAL run result (vs solve.SolvePlan which is a plan). For deferred layers most numeric
    fields are None and `executed=False` with an `execution_deferred` reason.
    """
    problem: Dict[str, Any]
    routing: Dict[str, Any]                 # layer/method/variant/adapter_cls/rationale
    executed: bool                          # True only when a real train+validate ran
    layer: str                              # "A" | "B" | "C"
    execution_deferred: Optional[str]       # reason string when executed=False; else None

    # --- learn (real numbers when executed) ---
    held_out_ic: Optional[float] = None     # IC on UNSEEN (the make-or-break diagnostic)
    shuffled_ic: Optional[float] = None     # row-permuted shuffled IC on UNSEEN (memorization probe)
    val_ic: Optional[float] = None
    n_unseen_windows: Optional[int] = None
    n_params: Optional[int] = None
    epochs_trained: Optional[int] = None
    model_kind: Optional[str] = None
    wall_time_s: Optional[float] = None
    trainer_result: Dict[str, Any] = field(default_factory=dict)  # full scalar dict from trainer

    # --- validate (the spine) ---
    battery: Dict[str, Any] = field(default_factory=dict)         # real numbers + applicability flags

    # --- honesty ---
    honest_ceiling: str = ""                # carried from plan_solve()
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spine wiring (E2): run battery.py on the held-out directional returns
# ---------------------------------------------------------------------------

def _run_battery_on_holdout(
    pred: np.ndarray,
    real: np.ndarray,
    unseen_ts: Optional[np.ndarray],
    target_is_return: bool,
) -> Dict[str, Any]:
    """Run the src/strat/battery.py spine on a SIGN-OF-FORECAST directional strategy.

    trade_ret[i] = sign(pred[i]) * real[i] over the held-out UNSEEN windows.

    Returns a dict carrying the REAL block-bootstrap p05/jackknife/compound numbers + clear
    applicability flags (returns_interpretable, all_4_positive=N/A, calendar_meaningful).
    """
    import strat.battery as battery  # numpy-only, no heavy deps

    out: Dict[str, Any] = {
        "spine": "src/strat/battery.py",
        "strategy": "sign(forecast) directional on UNSEEN bars: trade_ret = sign(pred)*real",
        "returns_interpretable": bool(target_is_return),
    }

    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[mask], real[mask]
    n = int(pred.size)
    out["n_unseen_trades"] = n

    if n < 10:
        out["error"] = f"too few held-out points (n={n}) to run the spine (need >=10)."
        out["ran"] = False
        return out

    # directional strategy returns. sign(0) -> 0 (no position): treat flat preds as no-trade.
    side = np.sign(pred)
    trade_ret = side * real
    # drop no-trade bars (side==0) so expectancy/PF reflect ACTUAL positions taken
    taken = side != 0.0
    trade_ret_taken = trade_ret[taken]
    n_taken = int(trade_ret_taken.size)
    out["n_positions_taken"] = n_taken

    if n_taken < 10:
        out["error"] = f"too few positions taken (n_taken={n_taken}); model predicts flat."
        out["ran"] = False
        return out

    # --- APPLICABLE pieces (need only the UNSEEN return vector) ------------------
    unseen_compound = battery.compound(trade_ret_taken)
    bb = battery.block_bootstrap_p05_p95(trade_ret_taken)
    jk2 = battery.jackknife(trade_ret_taken, 2)
    jk3 = battery.jackknife(trade_ret_taken, 3)
    neff = battery.herfindahl_neff(trade_ret_taken)
    exp = battery.expectancy(trade_ret_taken)
    wr = battery.win_rate(trade_ret_taken)
    pf = battery.profit_factor(trade_ret_taken)

    # proxy equity curve maxDD (PER-BAR proxy, labelled as such)
    eq = np.cumprod(1.0 + trade_ret_taken)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    maxdd_pct = float(dd.min() * 100.0) if dd.size else 0.0

    out["ran"] = True
    out["unseen_compound_pct"] = round(float(unseen_compound), 3)
    out["block_bootstrap"] = bb                 # {p05, p50, p95}
    out["p05"] = bb["p05"]
    out["p05_positive"] = bool(bb["p05"] is not None and bb["p05"] > 0)
    out["jackknife_k2_pct"] = round(float(jk2), 3)
    out["jackknife_k3_pct"] = round(float(jk3), 3)
    out["jk_robust"] = bool(jk2 > 0 and jk3 > 0)
    out["n_eff"] = round(float(neff), 1)
    out["expectancy_pct"] = round(float(exp), 4)
    out["win_rate"] = round(float(wr), 3)
    out["profit_factor"] = round(float(pf), 3)
    out["proxy_maxdd_pct"] = round(maxdd_pct, 2)

    # --- HONEST N/A markers -----------------------------------------------------
    # Lens A all_4_positive across TRAIN/VAL/OOS/UNSEEN: only UNSEEN directional returns are
    # available from the minimal trainer -> cannot compute the per-split directional compounds.
    out["all_4_positive"] = "N/A"
    out["all_4_positive_reason"] = (
        "minimal Layer-B trainer exposes ONLY the UNSEEN held-out arrays; TRAIN/VAL/OOS "
        "directional compounds are not computed here -> Lens A all-4-positive cannot be asserted "
        "(reported N/A, never faked True). Wire per-split directional returns to enable it."
    )

    # Lens C monthly calendar gate: meaningful only with real calendar timestamps spanning months.
    calendar_meaningful = False
    monthly = {"n_months": 0, "mpos": 0.0, "worst_month_pct": 0.0}
    have_ts = unseen_ts is not None and len(unseen_ts) > 0
    if have_ts and len(unseen_ts) == n and target_is_return:
        ts = np.asarray(unseen_ts)[mask][taken]
        try:
            span_ms = int(ts.max() - ts.min())
            span_days = span_ms / 86_400_000.0
            if span_days >= 60:  # at least ~2 months of span before calling Lens C meaningful
                pairs = list(zip(ts.tolist(), trade_ret_taken.tolist()))
                monthly = battery.monthly(pairs)
                calendar_meaningful = monthly["n_months"] >= 3
        except Exception as exc:  # pragma: no cover -- calendar grouping is best-effort
            out["monthly_error"] = str(exc)
    out["monthly"] = monthly
    out["calendar_meaningful"] = bool(calendar_meaningful)
    if not have_ts:
        out["calendar_reason"] = (
            "N/A: the index-based walk-forward split does not carry per-bar timestamps into the "
            "held-out segment, so Lens C monthly grouping cannot be computed. (Not a defect -- the "
            "spine's returns-based gates p05/jackknife/compound run regardless.)"
        )

    # --- VERDICT (only when the target is genuinely a tradeable return) ----------
    if target_is_return:
        # An UNSEEN-only pragmatic read: positive held-out compound + p05>0 + jk-robust + DD ok.
        # NOT the full Lens B (which needs all-4-positive); reported as an UNSEEN-ONLY verdict so
        # nobody mistakes it for the institutional Lens A/B pass.
        dd_ok = maxdd_pct > -30.0
        unseen_only_pass = bool(
            unseen_compound > 0 and out["p05_positive"] and out["jk_robust"]
            and dd_ok and neff >= 8.0
        )
        out["unseen_only_verdict"] = "PASS (UNSEEN-only)" if unseen_only_pass else "FAIL (UNSEEN-only)"
        out["unseen_only_pass"] = unseen_only_pass
        out["verdict_scope"] = (
            "UNSEEN-only pragmatic read (positive compound + block-bootstrap p05>0 + jk2/jk3>0 + "
            "n_eff>=8 + maxDD<30% on the held-out directional book). NOT the full Lens A/B "
            "(all-4-window-positive is N/A here). Treat as a directional-edge SCREEN, not a ship gate."
        )
    else:
        out["unseen_only_verdict"] = "N/A (target is not a tradeable return)"
        out["unseen_only_pass"] = None
        out["verdict_scope"] = (
            "target_is_return=False -> sign(pred)*target is NOT a tradeable return; the spine's "
            "compound/p05/jackknife numbers are computed for completeness but carry NO trading "
            "meaning. Use the IC diagnostics instead. (Honest: not misapplying a returns gate.)"
        )

    return out


# ---------------------------------------------------------------------------
# Synthetic RETURN-scaled controls (for a faithful spine demonstration)
# ---------------------------------------------------------------------------
# general_trainer's make_positive_control plants a STANDARDIZED signal (target ~ N(0,~1.4)) --
# perfect for the IC control, but those magnitudes are NOT fractional returns: compounding
# prod(1 + 1.5) is nonsensical. For the EXECUTOR's returns-spine demonstration we need a target
# in a realistic fractional-return range (~1-3% per bar). These generators produce exactly that,
# so battery.py's compound/p05/jackknife numbers are genuinely INTERPRETABLE.

def _make_return_segment(features: np.ndarray, ret_target: np.ndarray,
                         asset_name: str) -> List[dict]:
    """Pack features + a fractional-return target into the segment contract (horizon 1)."""
    n = len(features)
    base_ms = 1_600_000_000_000
    cadence_ms = 86_400_000  # 1 day -> gives real calendar span if a downstream split kept ts
    ts = (base_ms + np.arange(n, dtype=np.int64) * cadence_ms).astype(np.int64)
    seg = {
        "asset_idx": 0,
        "asset_name": asset_name,
        "timestamp": ts,
        "features": features.astype(np.float32),
    }
    for h in (1, 4, 16, 64):
        seg[f"target_return_{h}"] = (ret_target.astype(np.float32) if h == 1
                                     else np.zeros(n, dtype=np.float32))
    return [seg]


def make_positive_return_control(n: int = 6000, n_features: int = 4, lag: int = 1,
                                 ret_scale: float = 0.02, noise_frac: float = 0.6,
                                 seed: int = 0) -> List[dict]:
    """Planted lagged signal, target in a realistic per-bar RETURN range (~+/-2%).

    target[t] = ret_scale * tanh(driver[t-lag]) + noise. tanh bounds the driver so a single bar's
    'return' stays plausible (no +300% bars), making the compound/p05 spine numbers meaningful.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, n_features)).astype(np.float32)
    driver = feats[:, 0]
    target = np.zeros(n, dtype=np.float32)
    target[lag:] = (ret_scale * np.tanh(driver[:-lag])).astype(np.float32)
    target += (rng.standard_normal(n).astype(np.float32) * ret_scale * noise_frac)
    return _make_return_segment(feats, target, "positive_return_control")


def make_negative_return_control(n: int = 6000, n_features: int = 4,
                                 ret_scale: float = 0.02, seed: int = 0) -> List[dict]:
    """Pure-noise returns: target ~ N(0, ret_scale), independent of every feature."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, n_features)).astype(np.float32)
    target = (rng.standard_normal(n).astype(np.float32) * ret_scale)
    return _make_return_segment(feats, target, "negative_return_control")


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------

def execute_solve(
    problem: dict,
    segments: Optional[List[dict]] = None,
    trainer_overrides: Optional[Dict[str, Any]] = None,
    run_spine: bool = True,
    **kw: Any,
) -> SolveResult:
    """Execute the engine on a problem: represent -> route -> LEARN -> VALIDATE -> SolveResult.

    Parameters
    ----------
    problem : dict
        The router/plan_solve contract (name/domain/objective_type/...). Optional key
        `target_is_return` (default inferred True for crypto/time_series price-return targets)
        gates whether the returns-oriented spine verdict is asserted.
    segments : List[dict], optional
        GeneralAdapter.to_segments() output. REQUIRED for a Layer-B end-to-end execution.
        If None and the problem routes to Layer B, the result is executed=False with a clear
        "segments required" deferral (the executor does not invent data).
    trainer_overrides : dict, optional
        Passed through to train_layer_b (e.g. {"horizon": 4, "model_kind": "mlp", "seq_len": 16}).
    run_spine : bool, default True
        If True, run battery.py on the held-out directional returns (E2). If False, skip the spine
        (IC-only result).
    **kw :
        Reserved for future use; currently ignored (kept for forward-compat signature).

    Returns
    -------
    SolveResult
    """
    from framework.solve import plan_solve  # reuse the planner for routing + ceiling (intact)

    # 1. REPRESENT + 2. ROUTE (via the planner -- single source of routing + ceiling truth)
    plan = plan_solve(problem)
    layer = plan.routing["layer"]
    routing = dict(plan.routing)
    honest_ceiling = plan.honest_ceiling
    notes: List[str] = list(plan.plan_notes)

    # --- LANE GUARD: only Layer B executes end-to-end here ---------------------
    if layer == "A":
        return SolveResult(
            problem=dict(problem), routing=routing, executed=False, layer="A",
            execution_deferred=(
                "Layer A (games/self-play) is OUT-OF-LANE for this executor: it requires a self-play "
                "RL loop + champion gate + (for strength) a GPU self-play budget. The engine ROUTES it "
                "(see routing) and the adapter PLUGS IN, but training is compute-bound and not run here. "
                "Use plan_solve()'s Layer-A next_actions (chess_zero pipeline) to execute."
            ),
            honest_ceiling=honest_ceiling,
            notes=notes + ["execution_deferred: Layer A self-play not run by the generic executor."],
        )

    if layer == "C":
        return SolveResult(
            problem=dict(problem), routing=routing, executed=False, layer="C",
            execution_deferred=(
                "Layer C (crypto WM) is OUT-OF-LANE for this executor (a forked instance owns src/wm/ "
                "+ the chimera pipeline + cost model). It is also compound-return-bound per MEMORY.md. "
                "The engine ROUTES it; execution flows through the crypto stack (chimera_loader -> WM zoo "
                "-> battery on real returns), not this generic Layer-B trainer."
            ),
            honest_ceiling=honest_ceiling,
            notes=notes + ["execution_deferred: Layer C crypto WM not run by the generic executor."],
        )

    # --- Layer B: structured_prediction is a named-but-unimplemented bucket ----
    if routing.get("method") == "geometric_dl":
        return SolveResult(
            problem=dict(problem), routing=routing, executed=False, layer="B",
            execution_deferred=(
                "Layer B / structured_prediction (equivariant/geometric DL, AlphaFold-class) is NOT "
                "IMPLEMENTED: no equivariant net family and no structural data pipeline exist in-repo. "
                "Honest scope gap -- the router names the bucket, the implementation does not exist."
            ),
            honest_ceiling=honest_ceiling,
            notes=notes + ["execution_deferred: structured_prediction not implemented."],
        )

    # --- Layer B forecasting: the in-lane end-to-end path ----------------------
    if segments is None:
        return SolveResult(
            problem=dict(problem), routing=routing, executed=False, layer="B",
            execution_deferred=(
                "Layer B forecasting routed correctly, but NO segments were supplied. The executor "
                "does not invent data. Build segments via GeneralAdapter(data_source, target_col=...)"
                ".to_segments() and pass them as execute_solve(problem, segments=...)."
            ),
            honest_ceiling=honest_ceiling,
            notes=notes + ["execution_deferred: segments=None for a Layer-B problem."],
        )

    # 3. LEARN: real training on the supplied segments (held-out arrays for the spine)
    from framework.general_trainer import train_layer_b

    overrides = dict(trainer_overrides or {})
    tr = train_layer_b(segments, return_arrays=run_spine, **overrides)

    result = SolveResult(
        problem=dict(problem), routing=routing, executed=True, layer="B",
        execution_deferred=None,
        held_out_ic=float(tr["held_out_ic"]),
        shuffled_ic=float(tr["shuffled_ic"]),
        val_ic=float(tr.get("val_ic", float("nan"))),
        n_unseen_windows=int(tr.get("n_unseen_windows", 0)),
        n_params=int(tr.get("n_params", 0)),
        epochs_trained=int(tr.get("epochs_trained", 0)),
        model_kind=str(tr.get("model_kind", "")),
        wall_time_s=float(tr.get("wall_time_s", 0.0)),
        trainer_result={k: v for k, v in tr.items()
                        if k not in ("unseen_pred", "unseen_real", "unseen_ts")},
        honest_ceiling=honest_ceiling,
        notes=notes,
    )

    # 4. VALIDATE: run the spine on the held-out directional returns (E2)
    if run_spine:
        target_is_return = bool(problem.get("target_is_return",
                                            problem.get("domain") in ("crypto", "time_series")
                                            or problem.get("objective_type") == "forecasting"))
        try:
            result.battery = _run_battery_on_holdout(
                pred=tr["unseen_pred"], real=tr["unseen_real"],
                unseen_ts=tr.get("unseen_ts"), target_is_return=target_is_return,
            )
            if result.battery.get("ran"):
                result.notes.append(
                    f"spine ran on {result.battery.get('n_positions_taken')} held-out positions; "
                    f"p05={result.battery.get('p05')} "
                    f"unseen_verdict={result.battery.get('unseen_only_verdict')}"
                )
            else:
                result.notes.append(
                    f"spine did NOT run: {result.battery.get('error', 'unknown')}"
                )
        except Exception as exc:  # spine failure must not crash the executor
            result.battery = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
            result.notes.append(f"spine raised: {type(exc).__name__}: {exc}")
    else:
        result.battery = {"ran": False, "error": "run_spine=False (IC-only result)."}

    return result


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

_SEP = "-" * 70


def format_result(res: SolveResult) -> str:
    """Human-readable rendering of a SolveResult."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  SOLVE RESULT  (EXECUTE path)")
    lines.append("=" * 70)
    lines.append(f"  problem        : {res.problem.get('name', 'unnamed')}")
    lines.append(f"  layer          : {res.layer}  "
                 f"(method={res.routing.get('method')}, variant={res.routing.get('variant')})")
    lines.append(f"  executed       : {res.executed}")
    if not res.executed:
        lines.append(_SEP)
        lines.append("  EXECUTION DEFERRED (honest -- no fabricated number):")
        for cl in _wrap(res.execution_deferred or "(no reason given)"):
            lines.append("    " + cl)
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.append(_SEP)
    lines.append("  LEARN (held-out diagnostics):")
    lines.append(f"    held_out_IC   : {res.held_out_ic:+.4f}   (memorization probe shuffled_IC="
                 f"{res.shuffled_ic:+.4f})")
    lines.append(f"    val_IC        : {res.val_ic:+.4f}")
    lines.append(f"    model         : {res.model_kind}  ({res.n_params} params, "
                 f"{res.epochs_trained} epochs, {res.wall_time_s}s)")
    lines.append(f"    n_unseen_win  : {res.n_unseen_windows}")

    lines.append(_SEP)
    lines.append("  VALIDATE (src/strat/battery.py spine on held-out directional returns):")
    b = res.battery
    if not b.get("ran"):
        lines.append(f"    spine did NOT run: {b.get('error', 'unknown')}")
    else:
        lines.append(f"    strategy          : {b.get('strategy')}")
        lines.append(f"    returns_interpretable: {b.get('returns_interpretable')}")
        lines.append(f"    n_positions_taken : {b.get('n_positions_taken')}")
        lines.append(f"    unseen_compound % : {b.get('unseen_compound_pct')}")
        lines.append(f"    block-bootstrap   : p05={b.get('p05')} "
                     f"p50={b.get('block_bootstrap', {}).get('p50')} "
                     f"p95={b.get('block_bootstrap', {}).get('p95')}  "
                     f"(p05_positive={b.get('p05_positive')})")
        lines.append(f"    jackknife         : k2={b.get('jackknife_k2_pct')}% "
                     f"k3={b.get('jackknife_k3_pct')}%  (jk_robust={b.get('jk_robust')})")
        lines.append(f"    expectancy/PF/WR  : exp={b.get('expectancy_pct')}% "
                     f"PF={b.get('profit_factor')} WR={b.get('win_rate')}  n_eff={b.get('n_eff')}")
        lines.append(f"    proxy maxDD %     : {b.get('proxy_maxdd_pct')}  (PER-BAR proxy, not a book)")
        lines.append(f"    all_4_positive    : {b.get('all_4_positive')}  "
                     f"({b.get('all_4_positive_reason', '')[:60]}...)")
        lines.append(f"    calendar_meaningful: {b.get('calendar_meaningful')}  "
                     f"(Lens C monthly: {b.get('monthly')})")
        lines.append(f"    UNSEEN-only verdict: {b.get('unseen_only_verdict')}")
        for cl in _wrap("scope: " + str(b.get("verdict_scope", ""))):
            lines.append("      " + cl)

    lines.append(_SEP)
    lines.append("  HONEST CEILING (carried from the planner):")
    for cl in _wrap(res.honest_ceiling):
        lines.append("    " + cl)
    lines.append("=" * 70)
    return "\n".join(lines)


def _wrap(text: str, width: int = 64) -> List[str]:
    import textwrap
    return textwrap.wrap(str(text), width=width) or [""]


# ---------------------------------------------------------------------------
# Executor selftest (two-sided + lane-deferral)
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True, seed: int = 0) -> int:
    """End-to-end executor selftest. Returns 0 on pass.

    Cases:
      1. Layer-B POSITIVE control (planted lagged signal) -> executed=True, held_out_IC>0.15,
         spine RAN with real p05/jackknife numbers.
      2. Layer-B NEGATIVE control (pure noise) -> executed=True, |held_out_IC|<0.10, spine RAN
         but does NOT pass the UNSEEN-only verdict (no hallucinated edge).
      3. Layer A (games) -> executed=False, execution_deferred populated (honest deferral).
      4. Layer C (crypto) -> executed=False, execution_deferred populated.
    """
    failures = 0

    def _ck(name: str, cond: bool) -> None:
        nonlocal failures
        ok = bool(cond)
        if not ok:
            failures += 1
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    ov = {"seed": seed, "verbose": False, "model_kind": "gru",
          "seq_len": 16, "max_epochs": 60, "purge_gap_bars": 16}

    # -- Case 1: positive control (return-scaled target so the spine is meaningful) --
    if verbose:
        print("\n[executor selftest] Case 1: Layer-B POSITIVE control (planted signal)")
    pos_problem = {"name": "selftest_positive", "domain": "time_series",
                   "objective_type": "forecasting", "target_is_return": True}
    pos = execute_solve(pos_problem, segments=make_positive_return_control(seed=seed),
                        trainer_overrides=ov)
    _ck("pos: executed", pos.executed)
    _ck("pos: layer B", pos.layer == "B")
    _ck(f"pos: held_out_IC>0.15 (got {pos.held_out_ic:+.4f})", pos.held_out_ic > 0.15)
    _ck("pos: spine ran", pos.battery.get("ran") is True)
    _ck("pos: p05 is a real number", isinstance(pos.battery.get("p05"), (int, float)))
    _ck("pos: jackknife k2/k3 present",
        "jackknife_k2_pct" in pos.battery and "jackknife_k3_pct" in pos.battery)
    _ck("pos: all_4_positive marked N/A (not faked)", pos.battery.get("all_4_positive") == "N/A")
    if verbose:
        print(f"        -> held_out_IC={pos.held_out_ic:+.4f} | unseen_compound="
              f"{pos.battery.get('unseen_compound_pct')}% | p05={pos.battery.get('p05')} | "
              f"verdict={pos.battery.get('unseen_only_verdict')}")

    # -- Case 2: negative control --
    if verbose:
        print("\n[executor selftest] Case 2: Layer-B NEGATIVE control (pure noise)")
    neg_problem = {"name": "selftest_negative", "domain": "time_series",
                   "objective_type": "forecasting", "target_is_return": True}
    neg = execute_solve(neg_problem, segments=make_negative_return_control(seed=seed),
                        trainer_overrides=ov)
    _ck("neg: executed", neg.executed)
    _ck(f"neg: |held_out_IC|<0.10 (got {neg.held_out_ic:+.4f})", abs(neg.held_out_ic) < 0.10)
    _ck("neg: spine ran", neg.battery.get("ran") is True)
    _ck("neg: does NOT pass UNSEEN-only verdict (no hallucinated edge)",
        neg.battery.get("unseen_only_pass") is False)
    if verbose:
        print(f"        -> held_out_IC={neg.held_out_ic:+.4f} | unseen_compound="
              f"{neg.battery.get('unseen_compound_pct')}% | p05={neg.battery.get('p05')} | "
              f"verdict={neg.battery.get('unseen_only_verdict')}")

    # -- Case 3: Layer A deferral --
    if verbose:
        print("\n[executor selftest] Case 3: Layer A (games) honest deferral")
    a_problem = {"name": "tictactoe", "domain": "games", "objective_type": "decision",
                 "has_exact_simulator": True, "action_space": "discrete_small"}
    a_res = execute_solve(a_problem)
    _ck("A: executed=False", a_res.executed is False)
    _ck("A: execution_deferred populated", bool(a_res.execution_deferred))
    _ck("A: no fabricated IC", a_res.held_out_ic is None)

    # -- Case 4: Layer C deferral --
    if verbose:
        print("\n[executor selftest] Case 4: Layer C (crypto) honest deferral")
    c_problem = {"name": "BTCUSDT 4h", "domain": "crypto", "objective_type": "forecasting"}
    c_res = execute_solve(c_problem)
    _ck("C: executed=False", c_res.executed is False)
    _ck("C: execution_deferred populated", bool(c_res.execution_deferred))
    _ck("C: no fabricated IC", c_res.held_out_ic is None)

    # -- two-sided soundness: positive screen PASS-able, negative MUST NOT --
    pos_screen = pos.battery.get("unseen_only_pass")
    neg_screen = neg.battery.get("unseen_only_pass")
    _ck("two-sided: negative does not screen-pass while showing it CAN run",
        neg_screen is False and neg.battery.get("ran") is True)

    if verbose:
        print()
        if failures:
            print(f"[executor selftest] FAIL: {failures} check(s) failed")
        else:
            print("[executor selftest] PASS: all checks passed (two-sided + lane-deferral)")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="framework.execute",
        description="The generic engine EXECUTOR: execute_solve(problem, segments) -> SolveResult. "
                    "Runs represent->route->LEARN->VALIDATE end-to-end on Layer B; honestly defers A/C.",
    )
    ap.add_argument("--selftest", action="store_true",
                    help="Run the executor selftest (positive + negative control + A/C deferral).")
    ap.add_argument("--controls", action="store_true",
                    help="Run the positive + negative Layer-B controls through the EXECUTOR and print "
                         "the full SolveResult report for each.")
    ap.add_argument("--json-out", action="store_true", help="Emit JSON instead of a report.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(verbose=True, seed=args.seed) == 0 else 1)

    if args.controls or True:  # default action: run the two controls through the executor
        ov = {"seed": args.seed, "verbose": False, "model_kind": "gru",
              "seq_len": 16, "max_epochs": 60, "purge_gap_bars": 16}

        pos = execute_solve(
            {"name": "EXECUTOR positive control", "domain": "time_series",
             "objective_type": "forecasting", "target_is_return": True},
            segments=make_positive_return_control(seed=args.seed), trainer_overrides=ov)
        neg = execute_solve(
            {"name": "EXECUTOR negative control", "domain": "time_series",
             "objective_type": "forecasting", "target_is_return": True},
            segments=make_negative_return_control(seed=args.seed), trainer_overrides=ov)

        if args.json_out:
            print(json.dumps({"positive": asdict(pos), "negative": asdict(neg)},
                             indent=2, ensure_ascii=False, default=str))
        else:
            print(format_result(pos))
            print()
            print(format_result(neg))
        sys.exit(0)
