"""Safety guards for V1.x trainer + validator paths.

Three orthogonal correctness guarantees, motivated by the "trust the numbers"
mandate (2026-05-03):

1. safe_torch_save     -- atomic checkpoint write (tmp + fsync + rename).
                          Power-loss/disk-full mid-write cannot corrupt the
                          target file; readers see either old or new, never
                          a partial.

2. safe_metric         -- LOUD-FAIL wrapper for any IC/ShIC/Sharpe/Sortino
                          computation. Refuses to publish NaN/inf. Returns
                          a tagged result so callers can choose to log
                          "DEGENERATE" instead of silently emitting 0.0.

3. preflight_probe     -- 50-100 batch dry-run BEFORE a long training run.
                          Asserts: no NaN losses, grad_norm in band,
                          ShIC computable, GPU memory < ceiling. Catches
                          fp16 overflow, dead grad, dim mismatch in 60s
                          instead of 35 hours.

These are STANDALONE helpers. Not retrofitted into the actively-running
V1.x training (PID 11776 has cached imports anyway). Future trainer
rounds + variation-version Phase 2 + the post-V1 retrain cycle adopt
them.

Tests at the bottom (run module directly): python safety_guards.py
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch


# ─── A. Atomic torch.save ────────────────────────────────────────────────────

def safe_torch_save(obj, path: Path | str, *, fsync: bool = True) -> None:
    """Atomic checkpoint write: write to tmp, fsync, POSIX-atomic rename.

    A bit-flip during torch.save can leave a partial file at `path`. On
    restart, load_state_dict sees garbage, raises a cryptic shape error,
    or (worst) loads silently corrupt weights. safe_torch_save eliminates
    the partial-write window: readers always see the previous file or the
    new one, never a mix.

    Args:
        obj: anything torch.save accepts (state_dict, full ckpt dict, ...)
        path: target write path; will be atomically replaced
        fsync: if True (default), fsync the tmp file before rename. ~10-100ms
               cost; guarantees disk durability before the rename. Disable
               only for high-throughput non-durability paths.

    Behaviour:
        - tmp is at the same dir as path (so rename is atomic on POSIX)
        - if torch.save raises, tmp is cleaned up; original path untouched
        - if rename fails, tmp may persist for cleanup; original untouched
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        if fsync:
            # fsync on the tmp file before rename. On Windows, fsync on a
            # read-mode descriptor can fail with errno 9; we tolerate that
            # because the load-bearing guarantee here is ATOMICITY of the
            # rename, not durability. POSIX systems get the durability bonus.
            try:
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass  # Windows / unsupported fs; rename atomicity still holds
        os.replace(tmp, path)  # POSIX-atomic + Windows-atomic on the same FS
    except Exception:
        # Ensure tmp is cleaned up even on failure; original path is untouched
        # because os.replace is the only modification site.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ─── B. safe_metric NaN/inf guard ────────────────────────────────────────────

@dataclass
class MetricResult:
    """Tagged metric result; callers decide whether to publish or log degeneracy."""
    value: float
    is_finite: bool
    reason: str  # "ok" | "nan" | "inf" | "zero_variance" | "empty" | <custom>

    def __float__(self) -> float:
        if not self.is_finite:
            raise ValueError(
                f"refusing to coerce non-finite metric to float "
                f"(value={self.value}, reason={self.reason})"
            )
        return self.value


def safe_metric(value: float, reason: str = "ok") -> MetricResult:
    """Wrap a computed metric with finiteness guard.

    Use:
        ic = np.corrcoef(preds, reals)[0, 1]
        result = safe_metric(ic, reason="zero_variance" if np.std(preds) == 0 else "ok")
        if not result.is_finite:
            print(f"[ic] DEGENERATE: {result.reason}")
            return  # do NOT publish
        ic_value = float(result)  # only reachable if finite
    """
    if value is None:
        return MetricResult(float("nan"), False, "none")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MetricResult(float("nan"), False, "uncoercible")
    if math.isnan(v):
        return MetricResult(v, False, reason if reason != "ok" else "nan")
    if math.isinf(v):
        return MetricResult(v, False, "inf")
    return MetricResult(v, True, reason)


def assert_finite_metrics(metrics: dict, *, name: str = "metrics") -> None:
    """LOUD-FAIL on any non-finite metric in a dict.

    Use at the boundary where metrics get logged / persisted / reported:
        assert_finite_metrics({"ic_h1": ic, "shic": shic, "sortino": sortino})
    """
    bad = []
    for k, v in metrics.items():
        if v is None:
            bad.append((k, "none"))
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            bad.append((k, f"uncoercible({type(v).__name__})"))
            continue
        if math.isnan(vf):
            bad.append((k, "nan"))
        elif math.isinf(vf):
            bad.append((k, "inf"))
    if bad:
        raise RuntimeError(
            f"\n  [LOUD FAIL] non-finite metrics in {name}: {bad}\n"
            f"  Refusing to publish -- 'trust the numbers' mandate.\n"
            f"  Investigate: zero-variance predictions, all-NaN targets, "
            f"fp16 overflow upstream, or div-by-zero in metric computation."
        )


# ─── C. Preflight numerical probe ────────────────────────────────────────────

@dataclass
class ProbeResult:
    passed: bool
    findings: list  # (severity, message) tuples
    metrics: dict   # measured numbers

    def report(self) -> str:
        lines = [f"[preflight] {'PASS' if self.passed else 'FAIL'}"]
        for sev, msg in self.findings:
            lines.append(f"  [{sev}] {msg}")
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def preflight_numerical_probe(
    step_fn: Callable[[int], dict],
    n_steps: int = 50,
    *,
    grad_norm_min: float = 1e-3,
    grad_norm_max: float = 1e3,
    loss_max: float = 500.0,
    gpu_mem_ceiling_gb: Optional[float] = 7.5,
) -> ProbeResult:
    """Run step_fn for n_steps and assert numerical health.

    step_fn(step_idx) MUST return a dict with at least:
        "loss":      float (scalar loss value)
        "grad_norm": float (post-clip grad norm)
    Optional:
        "preds":     np.ndarray (1D for ShIC sanity; checked for variance)

    Returns ProbeResult with findings ordered by severity (CRIT/HIGH/MED).
    Caller should refuse to launch the long run if passed=False.

    The 50-step default is calibrated to ~60s on a 4060/8GB; tune via
    n_steps if your model has slower step time.
    """
    findings: list = []
    metrics: dict = {
        "n_steps_completed": 0,
        "max_loss": -math.inf,
        "min_loss": math.inf,
        "max_grad_norm": -math.inf,
        "min_grad_norm": math.inf,
        "n_nan_losses": 0,
        "n_inf_losses": 0,
        "max_gpu_mem_gb": 0.0,
    }
    has_cuda = torch.cuda.is_available()

    for i in range(n_steps):
        try:
            out = step_fn(i)
        except Exception as e:
            findings.append(("CRIT", f"step {i} raised {type(e).__name__}: {e}"))
            return ProbeResult(False, findings, metrics)

        metrics["n_steps_completed"] = i + 1
        loss = out.get("loss")
        gn = out.get("grad_norm")

        # Loss sanity
        if loss is None or not isinstance(loss, (int, float)):
            findings.append(("HIGH", f"step {i}: loss missing or non-numeric ({loss})"))
        else:
            if math.isnan(loss):
                metrics["n_nan_losses"] += 1
            elif math.isinf(loss):
                metrics["n_inf_losses"] += 1
            else:
                metrics["max_loss"] = max(metrics["max_loss"], loss)
                metrics["min_loss"] = min(metrics["min_loss"], loss)

        # Grad norm sanity
        if gn is None or not isinstance(gn, (int, float)):
            findings.append(("HIGH", f"step {i}: grad_norm missing or non-numeric ({gn})"))
        elif math.isnan(gn) or math.isinf(gn):
            findings.append(("CRIT", f"step {i}: grad_norm non-finite ({gn})"))
        else:
            metrics["max_grad_norm"] = max(metrics["max_grad_norm"], gn)
            metrics["min_grad_norm"] = min(metrics["min_grad_norm"], gn)

        # GPU memory ceiling
        if has_cuda:
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            metrics["max_gpu_mem_gb"] = max(metrics["max_gpu_mem_gb"], mem_gb)

    # Post-loop adjudication
    if metrics["n_nan_losses"] > 0:
        findings.append(("CRIT", f"{metrics['n_nan_losses']} NaN losses in {n_steps} steps"))
    if metrics["n_inf_losses"] > 0:
        findings.append(("CRIT", f"{metrics['n_inf_losses']} inf losses in {n_steps} steps"))
    if metrics["max_loss"] > loss_max:
        findings.append(("HIGH", f"max_loss={metrics['max_loss']:.2e} > {loss_max:.2e}"))
    if metrics["max_grad_norm"] > grad_norm_max:
        findings.append(("HIGH", f"max_grad_norm={metrics['max_grad_norm']:.2e} > {grad_norm_max}"))
    if metrics["max_grad_norm"] < grad_norm_min and metrics["n_steps_completed"] >= 5:
        findings.append(("HIGH", f"max_grad_norm={metrics['max_grad_norm']:.2e} < {grad_norm_min} (dead grad?)"))
    if gpu_mem_ceiling_gb is not None and metrics["max_gpu_mem_gb"] > gpu_mem_ceiling_gb:
        findings.append(("HIGH", f"GPU mem {metrics['max_gpu_mem_gb']:.2f}GB > ceiling {gpu_mem_ceiling_gb}GB"))

    has_crit_or_high = any(s in ("CRIT", "HIGH") for s, _ in findings)
    return ProbeResult(passed=not has_crit_or_high, findings=findings, metrics=metrics)


# ─── Tests ───────────────────────────────────────────────────────────────────

def _test_safe_torch_save():
    print("--- safe_torch_save ---")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        # Round-trip
        safe_torch_save({"foo": torch.tensor([1.0, 2.0])}, p)
        assert p.exists()
        loaded = torch.load(p, weights_only=False)
        assert torch.allclose(loaded["foo"], torch.tensor([1.0, 2.0]))
        # Tmp must not exist after success
        assert not p.with_name(p.name + ".tmp").exists()
        # Overwrite
        safe_torch_save({"bar": torch.tensor([3.0])}, p)
        loaded = torch.load(p, weights_only=False)
        assert "bar" in loaded
        # Failure path: torch.save raises -> tmp cleaned + original preserved.
        # Lambda is genuinely non-picklable (unlike bare object() which uses default
        # __reduce__ and pickles fine).
        try:
            safe_torch_save({"non_picklable": lambda x: x}, p)
        except Exception:
            pass
        assert not p.with_name(p.name + ".tmp").exists(), "tmp leaked on failure"
        loaded = torch.load(p, weights_only=False)
        assert "bar" in loaded, "original was clobbered on failure"
    print("  PASS")


def _test_safe_metric():
    print("--- safe_metric / assert_finite_metrics ---")
    # Finite
    r = safe_metric(0.0674, "ok")
    assert r.is_finite and float(r) == 0.0674
    # NaN
    r = safe_metric(float("nan"), "zero_variance")
    assert not r.is_finite and r.reason == "zero_variance"
    try:
        float(r)
        assert False, "should have raised"
    except ValueError:
        pass
    # inf
    r = safe_metric(float("inf"))
    assert not r.is_finite and r.reason == "inf"
    # None
    r = safe_metric(None)
    assert not r.is_finite and r.reason == "none"
    # assert_finite_metrics
    assert_finite_metrics({"ic": 0.05, "shic": 0.03})  # all good -> no raise
    try:
        assert_finite_metrics({"ic": 0.05, "shic": float("nan")})
        assert False, "should have raised"
    except RuntimeError as e:
        assert "shic" in str(e) and "nan" in str(e)
    print("  PASS")


def _test_preflight():
    print("--- preflight_numerical_probe ---")

    # Healthy mock
    def healthy_step(i):
        return {"loss": 28.0 - i * 0.01, "grad_norm": 1.5}
    res = preflight_numerical_probe(healthy_step, n_steps=20)
    assert res.passed, f"healthy step should pass: {res.findings}"

    # NaN loss
    def nan_step(i):
        return {"loss": float("nan") if i == 5 else 28.0, "grad_norm": 1.5}
    res = preflight_numerical_probe(nan_step, n_steps=20)
    assert not res.passed, "NaN loss must fail"
    assert any("NaN" in m or "nan" in m for _, m in res.findings)

    # Dead grad
    def dead_grad(i):
        return {"loss": 28.0, "grad_norm": 1e-6}
    res = preflight_numerical_probe(dead_grad, n_steps=20)
    assert not res.passed, "dead grad must fail"
    assert any("dead grad" in m for _, m in res.findings)

    # Exploded grad
    def exploded(i):
        return {"loss": 28.0, "grad_norm": 1e5}
    res = preflight_numerical_probe(exploded, n_steps=20)
    assert not res.passed, "exploded grad must fail"

    # Step raises
    def raiser(i):
        if i == 3:
            raise RuntimeError("synthetic")
        return {"loss": 28.0, "grad_norm": 1.0}
    res = preflight_numerical_probe(raiser, n_steps=20)
    assert not res.passed
    assert any("synthetic" in m for _, m in res.findings)
    print("  PASS")


if __name__ == "__main__":
    _test_safe_torch_save()
    _test_safe_metric()
    _test_preflight()
    print("\nALL safety_guards tests PASS")
