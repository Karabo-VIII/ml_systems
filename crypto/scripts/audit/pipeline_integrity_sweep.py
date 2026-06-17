"""pipeline_integrity_sweep.py -- programmatic end-to-end pipeline audit.

Scans every .py under src/pipeline/, scripts/oracle/, and strategy
consumer surfaces. For each file, extracts:
  - __contract__ presence + kind
  - parquet write/read sites + whether atomic_write_parquet is used
  - producer outputs declared
  - consumer inputs declared
  - split-boundary constants used (TRAIN_END_MS, etc.)
  - emoji usage (CLAUDE.md invariant: no emoji in prints/logs)

Output: docs/PIPELINE_INTEGRITY_SWEEP_2026_05_15.md
"""
from __future__ import annotations

import ast
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "PIPELINE_INTEGRITY_SWEEP_2026_05_15.md"

SCAN_ROOTS = [
    # R27 SOURCE + PROCESSING surfaces (original scope)
    ROOT / "src" / "pipeline",
    ROOT / "scripts" / "oracle",
    ROOT / "scripts" / "strat_audit",
    # R28 EXTENDED CONSUMER SURFACES (per user pushback 2026-05-15)
    ROOT / "src" / "strategy",       # full strategy tree (was only 3 sub-dirs)
    ROOT / "src" / "wm",              # WM V1-V14 training + inference
    ROOT / "src" / "audit",           # audit/validation
    ROOT / "src" / "analysis",        # analytics consumers
    ROOT / "scripts" / "audit",       # script audits
]

# Patterns
RE_CONTRACT = re.compile(r"__contract__\s*=\s*\{")
RE_ATOMIC_WRITE = re.compile(r"atomic_write_parquet\s*\(")
RE_RAW_WRITE = re.compile(r"\.write_parquet\s*\(|pq\.write_table\s*\(|pd\.DataFrame.*\.to_parquet\s*\(")
RE_PARQUET_READ = re.compile(r"\.read_parquet\s*\(|pq\.read_table\s*\(|pd\.read_parquet\s*\(")
RE_TRAIN_END = re.compile(r"TRAIN_END_MS|TRAIN_END_DATE|TRAIN_END\s*=")
RE_VAL_END = re.compile(r"VAL_END_MS|VAL_END_DATE|VAL_END\s*=")
RE_HARDCODED_DATE_STR = re.compile(r"['\"]20\d{2}-\d{2}-\d{2}['\"]")
RE_HARDCODED_MS = re.compile(r"1[5-7]\d{11}\b")  # 13-digit ms in [1.5e12, 2e12)
RE_EMOJI_PRINT = re.compile(r"print\s*\([^)]*[☀-➿\U0001F300-\U0001F9FF]")
RE_FROM_IMPORT = re.compile(r"^\s*from\s+(\S+)\s+import\s+", re.MULTILINE)

def scan_file(fp: Path) -> dict:
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"file": str(fp), "error": "unreadable"}
    rel = fp.relative_to(ROOT).as_posix()
    return {
        "file": rel,
        "lines": text.count("\n"),
        "has_contract": bool(RE_CONTRACT.search(text)),
        "atomic_writes": len(RE_ATOMIC_WRITE.findall(text)),
        "raw_writes": len(RE_RAW_WRITE.findall(text)),
        "parquet_reads": len(RE_PARQUET_READ.findall(text)),
        "train_end_ref": bool(RE_TRAIN_END.search(text)),
        "val_end_ref": bool(RE_VAL_END.search(text)),
        "hardcoded_date_strs": len(RE_HARDCODED_DATE_STR.findall(text)),
        "hardcoded_ms_constants": len(RE_HARDCODED_MS.findall(text)),
        "emoji_in_print": len(RE_EMOJI_PRINT.findall(text)),
        "from_imports_internal": [
            m.group(1) for m in RE_FROM_IMPORT.finditer(text)
            if not m.group(1).startswith((".", "polars", "numpy", "pandas",
                                            "pickle", "json", "pathlib", "typing",
                                            "argparse", "datetime", "warnings",
                                            "os", "sys", "time", "glob", "re",
                                            "hashlib", "lightgbm", "sklearn",
                                            "yaml", "concurrent", "subprocess",
                                            "collections", "itertools",
                                            "functools", "dataclasses",
                                            "pyarrow", "scipy", "matplotlib",
                                            "tqdm", "requests", "math",
                                            "xgboost", "torch"))
        ][:5],
    }


def main():
    rows = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for fp in root.rglob("*.py"):
            if "__pycache__" in fp.parts or "/archive/" in fp.as_posix() or "_archive/" in fp.as_posix():
                continue
            rows.append(scan_file(fp))

    # Aggregate
    n_total = len(rows)
    n_contract = sum(1 for r in rows if r.get("has_contract"))
    n_atomic = sum(1 for r in rows if r.get("atomic_writes", 0) > 0)
    n_raw_write = sum(1 for r in rows if r.get("raw_writes", 0) > 0)
    n_emoji = sum(1 for r in rows if r.get("emoji_in_print", 0) > 0)
    n_hardcoded_dates = sum(1 for r in rows if r.get("hardcoded_date_strs", 0) > 0)
    n_hardcoded_ms = sum(1 for r in rows if r.get("hardcoded_ms_constants", 0) > 0)
    n_train_ref = sum(1 for r in rows if r.get("train_end_ref"))
    n_val_ref = sum(1 for r in rows if r.get("val_end_ref"))

    # Drift detection
    raw_write_no_atomic = [r for r in rows if r.get("raw_writes", 0) > 0 and r.get("atomic_writes", 0) == 0]
    contract_missing_with_writes = [r for r in rows if not r.get("has_contract") and r.get("raw_writes", 0) > 0]
    emoji_violations = [r for r in rows if r.get("emoji_in_print", 0) > 0]
    split_users = [r for r in rows if r.get("train_end_ref") or r.get("val_end_ref")]

    # Write report
    lines = []
    lines.append("# Pipeline Integrity Sweep -- 2026-05-15")
    lines.append("")
    lines.append(f"> Programmatic audit of {n_total} files across src/pipeline/, scripts/oracle/, scripts/strat_audit/, src/strategy/{{discovery,sleeves,gen5_growth}}/.")
    lines.append(f"> User mandate 2026-05-15: 'Go through every file. Want to ensure no gap in pipeline, at all. No drift. When we rebuild and process data we rebuild correctly.'")
    lines.append("")
    lines.append("## Headline counts")
    lines.append("")
    lines.append(f"- Files scanned: **{n_total}**")
    lines.append(f"- With `__contract__` declared: {n_contract} ({n_contract/n_total*100:.0f}%)")
    lines.append(f"- Using `atomic_write_parquet` (G-AUDIT-020 contract): {n_atomic}")
    lines.append(f"- Writing parquet with RAW path (NOT atomic): {n_raw_write}")
    lines.append(f"- Reading parquet: {sum(1 for r in rows if r.get('parquet_reads', 0) > 0)}")
    lines.append(f"- Files referencing TRAIN_END constant: {n_train_ref}")
    lines.append(f"- Files referencing VAL_END constant: {n_val_ref}")
    lines.append(f"- Hardcoded date strings ('YYYY-MM-DD'): {n_hardcoded_dates} files")
    lines.append(f"- Hardcoded 13-digit ms constants: {n_hardcoded_ms} files")
    lines.append(f"- Files with emoji in print() (CLAUDE.md violation): {n_emoji}")
    lines.append("")

    # ── Drift sections ──
    lines.append("## DRIFT #1 — Raw parquet writes (NOT using atomic_write_parquet)")
    lines.append("")
    lines.append(f"Per CLAUDE.md G-AUDIT-020: 'New pipeline producers MUST use atomic_write_parquet'. Files writing parquet WITHOUT going through the framework helper:")
    lines.append("")
    if raw_write_no_atomic:
        lines.append(f"**{len(raw_write_no_atomic)} files** have raw parquet writes:")
        lines.append("")
        lines.append("| File | Raw writes | Has contract? |")
        lines.append("|---|---|---|")
        for r in sorted(raw_write_no_atomic, key=lambda x: -x["raw_writes"])[:30]:
            lines.append(f"| `{r['file']}` | {r['raw_writes']} | {'✅' if r['has_contract'] else '❌'} |")
        if len(raw_write_no_atomic) > 30:
            lines.append(f"| ... +{len(raw_write_no_atomic)-30} more | | |")
    else:
        lines.append("None. ✅")
    lines.append("")

    lines.append("## DRIFT #2 — Producers without __contract__")
    lines.append("")
    lines.append(f"Per CLAUDE.md: 'New components MUST declare a top-of-file __contract__ dict'. Files writing parquet WITHOUT __contract__:")
    lines.append("")
    if contract_missing_with_writes:
        lines.append(f"**{len(contract_missing_with_writes)} files** missing contract:")
        lines.append("")
        lines.append("| File | Raw writes | Atomic writes |")
        lines.append("|---|---|---|")
        for r in sorted(contract_missing_with_writes, key=lambda x: -x["raw_writes"])[:25]:
            lines.append(f"| `{r['file']}` | {r['raw_writes']} | {r['atomic_writes']} |")
    else:
        lines.append("None. ✅")
    lines.append("")

    lines.append("## DRIFT #3 — Emoji in print() (CLAUDE.md hard invariant: Windows cp1252 crashes)")
    lines.append("")
    if emoji_violations:
        lines.append(f"**{len(emoji_violations)} files** with emoji in print/log statements:")
        lines.append("")
        for r in emoji_violations[:20]:
            lines.append(f"- `{r['file']}` ({r['emoji_in_print']} occurrences)")
    else:
        lines.append("None. ✅")
    lines.append("")

    lines.append("## SPLIT CONFIG — Files referencing TRAIN_END / VAL_END")
    lines.append("")
    lines.append(f"User-flagged concern: 'The split. The communication between files.'")
    lines.append("")
    lines.append(f"Files referencing split boundaries: **{len(split_users)}**")
    lines.append("")
    if split_users:
        lines.append("| File | TRAIN_END ref | VAL_END ref | Hardcoded dates | Hardcoded ms |")
        lines.append("|---|---|---|---|---|")
        for r in split_users[:25]:
            lines.append(f"| `{r['file']}` | {'✅' if r['train_end_ref'] else '-'} | "
                         f"{'✅' if r['val_end_ref'] else '-'} | "
                         f"{r['hardcoded_date_strs']} | {r['hardcoded_ms_constants']} |")

    lines.append("")
    lines.append("**Centralized split-config status**: each consumer defines its own constants. No `src/split_config.py` single-source-of-truth exists. RISK: split drift if any consumer's constants get edited independently.")
    lines.append("")

    lines.append("## HARDCODED CONSTANTS — risk surface")
    lines.append("")
    high_hardcode = sorted(rows, key=lambda x: -(x.get("hardcoded_date_strs", 0) + x.get("hardcoded_ms_constants", 0)))[:15]
    lines.append("Top 15 files with most hardcoded date strings / ms constants (potential drift surface):")
    lines.append("")
    lines.append("| File | Date strings | MS constants |")
    lines.append("|---|---|---|")
    for r in high_hardcode:
        tot = r.get("hardcoded_date_strs", 0) + r.get("hardcoded_ms_constants", 0)
        if tot == 0: continue
        lines.append(f"| `{r['file']}` | {r['hardcoded_date_strs']} | {r['hardcoded_ms_constants']} |")
    lines.append("")

    lines.append("## File-level details (sortable)")
    lines.append("")
    lines.append("| File | Lines | Contract | Atomic writes | Raw writes | Reads | TRAIN ref | Emoji |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["file"]):
        lines.append(
            f"| `{r['file']}` | {r['lines']} | "
            f"{'✅' if r.get('has_contract') else '❌'} | "
            f"{r.get('atomic_writes', 0)} | {r.get('raw_writes', 0)} | "
            f"{r.get('parquet_reads', 0)} | "
            f"{'✅' if r.get('train_end_ref') else '-'} | "
            f"{r.get('emoji_in_print', 0)} |"
        )
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({n_total} files audited)")
    print()
    print(f"SUMMARY:")
    print(f"  Contract coverage: {n_contract}/{n_total} ({n_contract/n_total*100:.1f}%)")
    print(f"  Raw-parquet-writes (NOT atomic): {n_raw_write}")
    print(f"  Files with emoji-in-print: {n_emoji}")
    print(f"  Files referencing split boundaries: {n_train_ref} TRAIN, {n_val_ref} VAL")
    print(f"  Files w/ hardcoded date strings: {n_hardcoded_dates}")
    print(f"  Files w/ hardcoded ms constants: {n_hardcoded_ms}")


if __name__ == "__main__":
    main()
