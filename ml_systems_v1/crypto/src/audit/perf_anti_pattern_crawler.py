"""perf_anti_pattern_crawler.py -- detect performance hot-loop anti-patterns.

PURPOSE
-------
Born 2026-05-16 from the hawkes_branching_panel BTC+ETH MemoryError episode
that surfaced 3 critical Python-loop-over-data hotspots in the pipeline
(hawkes per_row_to_seconds 12x, lob_proxy _ts_to_ms 46x, range_bars
build_range_bars_day 184x). The fixes were obvious in hindsight; this
crawler ensures the same anti-patterns don't sneak back in.

Sister-crawler to pipeline_audit_crawler.py (correctness + data integrity);
this one is SPEED-only. Non-invasive, AST-based, reads code never invokes.

DETECTION AXES
--------------
A1. NP_EMPTY_LIKE_FOR    : np.empty_like(...) followed by for-loop populating
                            it. The exact hawkes / lob_proxy signature. CRITICAL.
A2. ITERROWS_APPLY_AXIS1 : .iterrows() / .itertuples() / .apply(axis=1) on a
                            DataFrame. Always 10-1000x slower than vectorized.
A3. CONCAT_IN_LOOP       : pd.concat / pl.concat inside a for/while body.
                            O(N^2) memory + time vs accumulating into a list.
A4. STATEFUL_LOOP_NO_JIT : for-loop over a numpy array indexing scalars in a
                            function without @njit / @jit decorator (heuristic
                            for stateful streaming algorithms that should be
                            JIT'd, like range_bars). MEDIUM (some are unavoidable
                            EWMA-style recurrences with N small).
A5. APPLY_LAMBDA_ROWWISE : .apply(lambda) on a Series where the lambda has
                            scalar conditional logic vectorizable via np.where.

USAGE
-----
    python src/audit/perf_anti_pattern_crawler.py
    python src/audit/perf_anti_pattern_crawler.py --root src/pipeline
    python src/audit/perf_anti_pattern_crawler.py --root src/pipeline --json
    python src/audit/perf_anti_pattern_crawler.py --critical-only

OUTPUT
------
    runs/audit/perf_anti_patterns_<DATE>.md  -- markdown findings + fixes
    Exit code: 0 = no findings; 1 = warnings; 2 = critical findings present

INVOKE FROM CDAP
----------------
    Add to .git/hooks/pre-commit -> src/audit/check_invariants.py orchestrator
    to fail commits that introduce new critical hot-loops.
"""
from __future__ import annotations

__contract__ = {
    "kind": "perf_anti_pattern_crawler",
    "owner": "audit/perf",
    "outputs": ["runs/audit/perf_anti_patterns_<DATE>.md"],
    "invariants": [
        "non-invasive: AST parse + grep only; never invokes producers",
        "every finding includes file:line + suggested fix",
        "exit code 2 on any CRITICAL finding for CI / pre-commit hook integration",
    ],
}

import argparse
import ast
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Severity levels.
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass
class Finding:
    file: str
    line: int
    axis: str
    severity: str
    snippet: str
    suggestion: str
    function_name: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "axis": self.axis,
            "severity": self.severity,
            "snippet": self.snippet,
            "suggestion": self.suggestion,
            "function_name": self.function_name,
        }


# ============================================================================
# AST visitors -- each detects one axis
# ============================================================================

class AntiPatternVisitor(ast.NodeVisitor):
    """Walks a module AST collecting Findings across all 5 axes."""

    def __init__(self, file_rel: str, source_lines: list[str]):
        self.file_rel = file_rel
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        # Track enclosing function name + whether it has @njit / @jit decorator
        self._fn_stack: list[tuple[str, bool]] = []
        # Track recently-seen np.empty_like assignments by target name + line
        self._empty_like_assigns: dict[str, int] = {}

    # ----- helpers -----------------------------------------------------------

    def _snippet_at(self, lineno: int) -> str:
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].rstrip()
        return ""

    def _current_fn(self) -> str:
        return self._fn_stack[-1][0] if self._fn_stack else ""

    def _current_fn_is_jit(self) -> bool:
        return self._fn_stack[-1][1] if self._fn_stack else False

    def _emit(self, line: int, axis: str, severity: str, suggestion: str) -> None:
        self.findings.append(Finding(
            file=self.file_rel,
            line=line,
            axis=axis,
            severity=severity,
            snippet=self._snippet_at(line),
            suggestion=suggestion,
            function_name=self._current_fn(),
        ))

    # ----- function tracking (for JIT-decorator context) ---------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_jit = any(self._is_jit_decorator(d) for d in node.decorator_list)
        self._fn_stack.append((node.name, is_jit))
        # Reset empty_like tracking per function -- patterns are function-local
        prev_empty = self._empty_like_assigns
        self._empty_like_assigns = {}
        self.generic_visit(node)
        self._empty_like_assigns = prev_empty
        self._fn_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _is_jit_decorator(self, node: ast.expr) -> bool:
        # @njit, @jit, @numba.njit, @numba.jit, @njit(cache=True), etc.
        name = ""
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        return name in {"njit", "jit"}

    # ----- A1: np.empty_like (or np.zeros/empty) + for-loop -----------------

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track LHS name = np.empty_like(...) / np.zeros_like(...) / np.empty(...)
        if isinstance(node.value, ast.Call):
            call = node.value
            fname = self._call_dotted_name(call)
            if fname in {"np.empty_like", "np.zeros_like", "np.empty",
                          "numpy.empty_like", "numpy.zeros_like", "numpy.empty"}:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self._empty_like_assigns[tgt.id] = node.lineno
        self.generic_visit(node)

    # ----- A1 + A4: for-loop detection --------------------------------------

    def visit_For(self, node: ast.For) -> None:
        # Detect: for i, t in enumerate(<array>):
        is_enumerate_idx_val = self._is_enumerate_index_value(node)
        is_range_len = self._is_range_len(node)
        is_range_only = self._is_range_call(node.iter)

        # A1: np.empty_like target referenced inside body via subscript assign
        if is_enumerate_idx_val or is_range_len:
            empty_target = self._find_subscript_assign_target(node)
            if empty_target and empty_target in self._empty_like_assigns:
                self._emit(
                    line=node.lineno,
                    axis="A1_NP_EMPTY_LIKE_FOR",
                    severity=CRITICAL,
                    suggestion=(
                        f"`{empty_target} = np.empty_like(...)` followed by a "
                        f"for-loop populating it row-by-row is THE hotspot pattern "
                        f"(hawkes/lob_proxy/range_bars). Vectorize with np.where "
                        f"or numba @njit (see commit b7e611b/a0ed97e/ab084ae)."
                    ),
                )

        # A4: stateful for-loop over a numpy/polars array w/o JIT
        if (is_enumerate_idx_val or is_range_len or is_range_only):
            if not self._current_fn_is_jit():
                # Heuristic: only flag if body has scalar arithmetic / conditional
                # assignment to an indexed buffer (typical streaming pattern).
                if self._body_has_scalar_state(node):
                    self._emit(
                        line=node.lineno,
                        axis="A4_STATEFUL_LOOP_NO_JIT",
                        severity=MEDIUM,
                        suggestion=(
                            "Stateful for-loop over array indices in a non-JIT'd "
                            "function. If the algorithm is intrinsically sequential "
                            "(streaming bar builder, etc.), wrap with @njit(cache=True) "
                            "for 50-200x speedup (see range_bars_fast.py commit "
                            "ab084ae). If it's a simple recurrence (EWMA), check if "
                            "polars.ewm_mean / pandas.ewm covers it."
                        ),
                    )

        # A3: pd.concat / pl.concat inside the for body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                fname = self._call_dotted_name(child)
                if fname in {"pd.concat", "pandas.concat", "pl.concat", "polars.concat"}:
                    # Only flag if INSIDE the for body, not the iter expression
                    # (ast.walk includes the For itself; filter)
                    if self._is_inside_for_body(node, child):
                        self._emit(
                            line=child.lineno,
                            axis="A3_CONCAT_IN_LOOP",
                            severity=HIGH,
                            suggestion=(
                                "concat() inside a loop is O(N^2) memory + time. "
                                "Append to a list inside the loop, then concat ONCE "
                                "outside the loop. Standard pattern in fetch_all.py, "
                                "binance_spot_klines.py, etc."
                            ),
                        )

        self.generic_visit(node)

    # ----- A2 + A5: pandas row-wise method calls ----------------------------

    def visit_Call(self, node: ast.Call) -> None:
        fname = self._call_dotted_name(node)
        # A2: df.iterrows / df.itertuples / df.apply(axis=1) / df.applymap
        if fname.endswith(".iterrows") or fname.endswith(".itertuples"):
            self._emit(
                line=node.lineno,
                axis="A2_ITERROWS",
                severity=CRITICAL,
                suggestion=(
                    f"`{fname}()` iterates rows in Python. 100-1000x slower than "
                    f"vectorized polars / numpy. Rewrite using polars expressions, "
                    f"pl.col().map_batches, or numpy boolean indexing."
                ),
            )
        elif fname.endswith(".applymap"):
            self._emit(
                line=node.lineno,
                axis="A2_APPLYMAP",
                severity=HIGH,
                suggestion="`.applymap()` is deprecated since pandas 2.1. Use "
                            ".map() or polars equivalents.",
            )
        elif fname.endswith(".apply"):
            # Detect axis=1 -- row-wise apply
            for kw in node.keywords:
                if kw.arg == "axis":
                    if isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                        self._emit(
                            line=node.lineno,
                            axis="A2_APPLY_AXIS1",
                            severity=HIGH,
                            suggestion=(
                                ".apply(axis=1) is row-wise = Python loop. "
                                "Replace with vectorized polars/numpy expressions."
                            ),
                        )
                        break
            # A5: .apply(lambda) where lambda has scalar conditional
            if node.args and isinstance(node.args[0], ast.Lambda):
                lam = node.args[0]
                if self._lambda_is_conditional(lam):
                    self._emit(
                        line=node.lineno,
                        axis="A5_APPLY_LAMBDA_CONDITIONAL",
                        severity=LOW,
                        suggestion=(
                            ".apply(lambda) with scalar conditional logic is "
                            "vectorizable via np.where(cond, true_expr, false_expr). "
                            "Small impact at panel scale but cleaner + faster."
                        ),
                    )

        self.generic_visit(node)

    # ----- AST helpers -------------------------------------------------------

    @staticmethod
    def _call_dotted_name(call: ast.Call) -> str:
        """Best-effort dotted name extraction for a Call node's .func."""
        parts: list[str] = []
        cur = call.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    @staticmethod
    def _is_enumerate_index_value(node: ast.For) -> bool:
        """Match: for i, t in enumerate(<expr>)."""
        if not (isinstance(node.iter, ast.Call)
                 and isinstance(node.iter.func, ast.Name)
                 and node.iter.func.id == "enumerate"):
            return False
        return (isinstance(node.target, ast.Tuple)
                 and len(node.target.elts) == 2)

    @staticmethod
    def _is_range_len(node: ast.For) -> bool:
        """Match: for i in range(len(<expr>))."""
        if not (isinstance(node.iter, ast.Call)
                 and isinstance(node.iter.func, ast.Name)
                 and node.iter.func.id == "range"):
            return False
        if not node.iter.args:
            return False
        a = node.iter.args[0]
        return (isinstance(a, ast.Call)
                 and isinstance(a.func, ast.Name)
                 and a.func.id == "len")

    @staticmethod
    def _is_range_call(iter_node: ast.expr) -> bool:
        return (isinstance(iter_node, ast.Call)
                 and isinstance(iter_node.func, ast.Name)
                 and iter_node.func.id == "range")

    @staticmethod
    def _find_subscript_assign_target(for_node: ast.For) -> str | None:
        """Walk for-body, return name X if there's `X[i] = ...` assignment."""
        for child in ast.walk(for_node):
            if isinstance(child, ast.Assign):
                for tgt in child.targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Name)):
                        return tgt.value.id
        return None

    @staticmethod
    def _is_inside_for_body(for_node: ast.For, target_node: ast.AST) -> bool:
        """Check if target_node is in for_node.body (not for_node.iter)."""
        for stmt in for_node.body:
            for child in ast.walk(stmt):
                if child is target_node:
                    return True
        return False

    @staticmethod
    def _body_has_scalar_state(for_node: ast.For) -> bool:
        """Tight heuristic for A4: requires ALL of
        (a) AugAssign on a non-subscripted name (cur_vol += ...) — accumulator
        (b) subscript-READ of a Name using the loop's index variable (price[i])
            — array iteration

        Without both, this is likely orchestration / file iteration / subprocess
        management — not a data hot-loop.

        Also requires the iter expression to be a single Name (not enumerate(
        sorted(...)) or other obvious orchestration patterns).
        """
        # Iter purity: must be enumerate(Name) or range(len(Name)) -- not
        # enumerate(sorted(...)) / enumerate(dict.values()) etc.
        iter_n = for_node.iter
        target_name: str | None = None
        if isinstance(iter_n, ast.Call) and isinstance(iter_n.func, ast.Name):
            if iter_n.func.id == "enumerate" and iter_n.args:
                if isinstance(iter_n.args[0], ast.Name):
                    target_name = iter_n.args[0].id
            elif iter_n.func.id == "range" and iter_n.args:
                a = iter_n.args[0]
                if (isinstance(a, ast.Call)
                        and isinstance(a.func, ast.Name)
                        and a.func.id == "len"
                        and a.args
                        and isinstance(a.args[0], ast.Name)):
                    target_name = a.args[0].id
        if target_name is None:
            return False

        # Loop index name
        idx_name: str | None = None
        tgt = for_node.target
        if isinstance(tgt, ast.Name):
            idx_name = tgt.id
        elif isinstance(tgt, ast.Tuple) and tgt.elts:
            if isinstance(tgt.elts[0], ast.Name):
                idx_name = tgt.elts[0].id
        if idx_name is None:
            return False

        has_scalar_accumulator = False
        has_array_index_read = False
        for child in ast.walk(for_node):
            # (a) AugAssign on a plain Name (NOT subscripted): "cur_vol += ..."
            if isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    has_scalar_accumulator = True
            # (b) Subscript READ: <SomeName>[<idx_name>]
            elif (isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Name)
                    and isinstance(child.ctx, ast.Load)):
                sub = child.slice
                if isinstance(sub, ast.Name) and sub.id == idx_name:
                    has_array_index_read = True
        return has_scalar_accumulator and has_array_index_read

    @staticmethod
    def _lambda_is_conditional(lam: ast.Lambda) -> bool:
        """Detect `lambda x: A if cond else B` or `lambda x: x // 1000 if x >= 1e15 else x`."""
        return isinstance(lam.body, ast.IfExp)


# ============================================================================
# File scanner
# ============================================================================

def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    source_lines = text.splitlines()
    # Best-effort relative path; fall back to absolute when scanning outside
    # PROJECT_ROOT (e.g., temp dirs for testing the crawler itself).
    try:
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(path).replace("\\", "/")
    visitor = AntiPatternVisitor(rel, source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_tree(root: Path, exclude_dirs: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        # Skip excluded dirs
        if any(p in exclude_dirs for p in path.parts):
            continue
        findings.extend(scan_file(path))
    return findings


# ============================================================================
# Output formatting
# ============================================================================

SEVERITY_RANK = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3}


def write_markdown(findings: list[Finding], out_path: Path,
                    root: str, scanned_files: int) -> None:
    today = dt.date.today().isoformat()
    by_axis: dict[str, list[Finding]] = {}
    by_severity: dict[str, int] = {CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0}
    for f in findings:
        by_axis.setdefault(f.axis, []).append(f)
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"# Pipeline Perf Anti-Pattern Audit -- {today}\n\n")
        out.write(f"Scanned: `{root}` ({scanned_files} .py files)\n\n")
        out.write(f"## Severity totals\n\n")
        for sev in (CRITICAL, HIGH, MEDIUM, LOW):
            out.write(f"- **{sev}**: {by_severity[sev]}\n")
        out.write(f"\nTotal findings: {len(findings)}\n\n")
        if not findings:
            out.write("**CLEAN** -- no perf anti-patterns detected.\n\n")
            out.write("See [docs/PERF_AUDIT_2026_05_16.md](../../docs/PERF_AUDIT_2026_05_16.md) "
                        "for the audit that ran before this crawler shipped.\n")
            return

        out.write("## Findings by axis\n\n")
        # Sort axis groups by max severity within
        ordered_axes = sorted(
            by_axis.keys(),
            key=lambda ax: min(SEVERITY_RANK[f.severity] for f in by_axis[ax]),
        )
        for axis in ordered_axes:
            axis_findings = sorted(
                by_axis[axis],
                key=lambda f: (SEVERITY_RANK[f.severity], f.file, f.line),
            )
            out.write(f"### {axis} ({len(axis_findings)})\n\n")
            for f in axis_findings:
                out.write(f"- **{f.severity}** `{f.file}:{f.line}`")
                if f.function_name:
                    out.write(f" in `{f.function_name}()`")
                out.write("\n")
                if f.snippet:
                    out.write(f"  ```\n  {f.snippet}\n  ```\n")
                out.write(f"  *Fix:* {f.suggestion}\n\n")

        out.write("## Severity exit-code policy\n\n")
        out.write("- Any CRITICAL = exit 2 (CI / pre-commit halt)\n")
        out.write("- HIGH only = exit 1 (warn, review)\n")
        out.write("- MEDIUM/LOW only = exit 0 (log + monitor)\n")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect perf anti-patterns in pipeline code via AST.")
    ap.add_argument("--root", default="src/pipeline",
                    help="Directory to scan (default: src/pipeline).")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="Directory name(s) to skip (repeatable).")
    ap.add_argument("--critical-only", action="store_true",
                    help="Filter to CRITICAL severity only.")
    ap.add_argument("--json", action="store_true",
                    help="Print findings as JSON to stdout (no markdown).")
    ap.add_argument("--out", default=None,
                    help="Markdown output path (default: runs/audit/perf_anti_patterns_<DATE>.md).")
    args = ap.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    if not root.exists():
        print(f"[perf-crawler] FATAL: root not found: {root}", file=sys.stderr)
        return 2

    exclude_dirs = set(args.exclude_dir) | {"__pycache__", "_archived", ".venv",
                                                "node_modules", "BKP_20260429_MODEL_HARMONIZATION"}
    findings = scan_tree(root, exclude_dirs)
    if args.critical_only:
        findings = [f for f in findings if f.severity == CRITICAL]

    scanned_files = sum(1 for _ in root.rglob("*.py")
                          if not any(p in exclude_dirs for p in _.parts))

    if args.json:
        json.dump([f.to_dict() for f in findings], sys.stdout, indent=2)
        print()
    else:
        out_path = (Path(args.out) if args.out
                     else OUT_DIR / f"perf_anti_patterns_{dt.date.today().isoformat()}.md")
        write_markdown(findings, out_path, str(root.relative_to(PROJECT_ROOT)),
                        scanned_files)
        print(f"[perf-crawler] scanned {scanned_files} files in "
              f"{root.relative_to(PROJECT_ROOT)}; {len(findings)} findings; "
              f"report: {out_path.relative_to(PROJECT_ROOT)}")

    # Exit code policy
    by_sev = {CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    if by_sev[CRITICAL] > 0:
        return 2
    if by_sev[HIGH] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
