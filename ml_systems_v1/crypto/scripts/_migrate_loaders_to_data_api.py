"""Phase B.5 batch migration: replace `from anti_fragile import (..., load_full_data, ...)`
with the trimmed import + `from data_api import load_full_data_for_training as load_full_data`.

Skips: src/wm/v1/archive/** (frozen historical), src/wm/v15-v19 (no anti_fragile import).
Idempotent: skips files where data_api alias already exists.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

DATA_API_LINE = (
    "# Read-side contract: load_full_data goes through data_api so future\n"
    "# pipeline changes only touch one module. See src/data_api/__init__.py.\n"
    "from data_api import load_full_data_for_training as load_full_data\n"
)

# Match the multi-line `from anti_fragile import (\n  ...\n)` block when it
# contains `load_full_data` on its own line (with optional trailing comma /
# whitespace). Strip just that one entry.
PATTERN = re.compile(
    r"(from anti_fragile import \([^)]*?)(\n\s*load_full_data\s*,?\s*)([^)]*?\))",
    re.DOTALL,
)


def main() -> int:
    targets = list(ROOT.glob("src/wm/**/train_world_model.py"))
    targets = [t for t in targets if "archive" not in t.parts]

    n_changed = 0
    n_skipped = 0
    for p in targets:
        text = p.read_text(encoding="utf-8")
        if "data_api import load_full_data_for_training" in text:
            print(f"  already-migrated: {p.relative_to(ROOT)}")
            n_skipped += 1
            continue
        m = PATTERN.search(text)
        if not m:
            print(f"  NO-MATCH (manual review): {p.relative_to(ROOT)}")
            continue
        # Remove the load_full_data entry from the import block
        new_block = m.group(1) + ("\n" if not m.group(1).endswith("\n") else "") + m.group(3)
        # Clean up double commas / orphan newlines from the splice
        new_block = re.sub(r",\s*,", ",", new_block)
        new_block = re.sub(r"\(\s*,", "(", new_block)
        new_block = re.sub(r",\s*\)", "\n)", new_block)
        new_text = text[:m.start()] + new_block + text[m.end():]
        # Insert the data_api import after the anti_fragile block
        marker = new_block
        idx = new_text.find(marker)
        end_idx = idx + len(marker)
        # Skip past trailing newline if any
        if end_idx < len(new_text) and new_text[end_idx] == "\n":
            end_idx += 1
        new_text = new_text[:end_idx] + DATA_API_LINE + new_text[end_idx:]
        p.write_text(new_text, encoding="utf-8")
        print(f"  migrated: {p.relative_to(ROOT)}")
        n_changed += 1

    print(f"\nmigrated: {n_changed}  already-migrated: {n_skipped}  total: {len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
