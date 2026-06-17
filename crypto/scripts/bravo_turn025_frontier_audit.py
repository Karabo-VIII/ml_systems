"""Bravo turn 025 -- comprehensive frontier folder audit (per user pointer)."""
from __future__ import annotations

import os, re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')
FRONTIER = ROOT / 'src/frontier'
RESULTS_LOG = ROOT / 'docs/frontier/frontier_results_log.md'

# Walk all .py
all_modules = []
for fp in sorted(FRONTIER.rglob('*.py')):
    if '__pycache__' in str(fp): continue
    if fp.name == '__init__.py': continue
    rel = fp.relative_to(FRONTIER).as_posix()
    txt = fp.read_text(encoding='utf-8', errors='replace')
    loc = len(txt.splitlines())
    has_main = '__main__' in txt
    has_argparse = 'argparse' in txt
    # detect docstring summary
    m = re.match(r'\s*(?:"""|\'\'\')([^"\'\n]+)', txt)
    summary = m.group(1)[:80] if m else ''
    all_modules.append({
        'rel': rel, 'loc': loc, 'main': has_main, 'argparse': has_argparse,
        'summary': summary,
    })

# Logged modules
log_text = RESULTS_LOG.read_text(encoding='utf-8')
logged = set(re.findall(r'(?:strategies|features|pipeline|ingest|utils|backtest|blend|audit|live)/[a-z_0-9]+\.py', log_text))

# Categorize
print(f'{"Module":<60s} {"LOC":>5s} {"main":>5s} {"argp":>4s} {"Logged":>8s}  Summary')
print('-'*150)
unlogged_strats = []
unlogged_other = []
for m in all_modules:
    is_logged = m['rel'] in logged
    short = m['rel']
    status = 'LOGGED' if is_logged else 'UNLOGGED'
    main_s = 'yes' if m['main'] else '-'
    argp_s = 'yes' if m['argparse'] else '-'
    print(f'{short:<60s} {m["loc"]:>5d} {main_s:>5s} {argp_s:>4s} {status:>8s}  {m["summary"]}')
    if not is_logged:
        if 'strategies/' in m['rel']:
            unlogged_strats.append(m)
        else:
            unlogged_other.append(m)

print()
print(f'=== Unlogged strategy modules: {len(unlogged_strats)} ===')
for m in unlogged_strats:
    print(f'  {m["rel"]:<55s} ({m["loc"]} LOC, main={m["main"]})')

print(f'\n=== Unlogged non-strategy modules: {len(unlogged_other)} ===')
for m in unlogged_other:
    print(f'  {m["rel"]:<55s} ({m["loc"]} LOC, main={m["main"]})')

# Also check src/growth pillars for similar audit
print('\n=== Growth pillars audit ===')
GROWTH_PILLARS = ROOT / 'src/growth/pillars'
if GROWTH_PILLARS.exists():
    for fp in sorted(GROWTH_PILLARS.glob('*.py')):
        if fp.name == '__init__.py' or fp.name == 'base.py': continue
        txt = fp.read_text(encoding='utf-8', errors='replace')
        loc = len(txt.splitlines())
        m = re.match(r'\s*(?:"""|\'\'\')([^"\'\n]+)', txt)
        summary = m.group(1)[:80] if m else ''
        print(f'  {fp.name:<35s} ({loc} LOC)  {summary}')
