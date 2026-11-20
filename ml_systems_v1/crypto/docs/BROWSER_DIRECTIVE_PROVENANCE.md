# @browser Directive — Rule Provenance (sidecar to BROWSER_DIRECTIVE.md)

> Each B-series rule in [BROWSER_DIRECTIVE.md](BROWSER_DIRECTIVE.md) was added
> in response to a real shipped bug. This file holds the bug history; the
> main directive holds only the rules. Read this when DEBATING / EDITING a
> rule, not on every turn.

## B1 — Defaults are LOUD

`range_bars_fast.py` had a silent 0.5% default → SUI generated 58M bars
consuming 23 GB memory before the process hung. Rule requires every default
value (worker count, threshold, universe, fallback path, cadence, retry
count) to be **printed at start of run** so the user sees what they got vs
what they asked for, before runtime.

## B2 — Silent caps are FORBIDDEN

`build_bars.py --workers 8` silently became `--workers 4` because only 4
bartypes existed. User asked: "is `--workers` dynamic? I gave 8, it announces 4."
Rule requires explicit WARN, never silent clamp.

## B3 — Default fallbacks announce themselves

`RANGE_THRESHOLDS.get(symbol, 0.005)` for everything not BTC/ETH/SOL → uniform
0.5% threshold applied to PEPE/SUI/TRX without warning. Rule requires
`[FALLBACK] <key> not in <source>; using default <value>` print.

## B4 — Spawn parameters are explicit

`make_dataset.py _spawn` hardcoded `--workers 12` in the child process →
infinite recursion → 0xC0000005 segfaults. Rule requires every
`subprocess.Popen` / `ProcessPoolExecutor` re-launching the same script to
verify args break recursion (parent `workers > 1` → child `workers=1`).

## B5 — Universe propagation is checked

Hawkes panel hardcoded u10 while bar_fabric ran u50. 38 non-u10 assets
silently got NaN hawkes columns in the chimera join. Same pattern still
latent in `build_panels`' s3/te/rv_jumps sub-builders. CDAP invariant
`cli_universe_support` enforces this (warn).

## B6 — Hard caps preceded by explicit projection

Range bars had no projection probe → 58M bars committed before the runtime
memory hit blew up the process. Rule requires `projection N exceeds cap C;
auto-widening to X` or `SKIPPED` announcement. Never silently truncate.

## B7 — Output verification BEFORE declaring "done"

Chimera_v51 reported 50/50 OK from manifests dated `_20260427` even though
the current run produced zero new files. Mis-classifying stale outputs as
success is the worst class of false-OK. Rule: don't trust `exit_code=0`;
verify `mtime > run_started_epoch`, schema, row count.

## B8 — Smoke probes at REAL scale, not toy

V4 magnitude explosion only triggered at B=32 after 34 steps. Initial B=4
probe passed clean. Rule: smoke at the SAME scale as the bug.

## B9 — Comments documenting bugs cannot be regression-detected by regex

CDAP simulator regex `mtm_only_no_double_count` `ret_from_entry.*pnl_bar`
matched the COMMENT documenting the historical bug, not the bug itself.
Rule: audit regex against ACTUAL forbidden code, not documentation strings.

## B10 — `re.M` does not constrain `.` or `[^x]*` to a single line

CDAP simulator regex `pnl_bar.*ret_from_entry` matched across 4 lines of
unrelated code via `[^#]*` greedy newline traversal. Rule: iterate
`text.splitlines()` and apply the regex per line.

## B11 — External-state claims MUST be verified via WebSearch / WebFetch

Silent staleness from training cutoff. Without a hard rule, model defaults
to "I think I know this" on questions where the ground truth is moving
(Binance API changes, polars version updates, recent papers). Token cost
of 1 WebSearch is far below the cost of shipping a stale claim.

This rule is **additive to** `/un`'s "Mandatory External Research" directive
(2 WebSearch + 1 WebFetch on architecture decisions). B11 extends the binding
to ALL non-trivial factual claims, not just architecture.

## G — Response modes (added 2026-05-03)

Token audit (`docs/BROWSER_INVOCATION_OPTIMIZATION_2026_05_03.md`) showed
Research-mode ceremony was firing for casual lookups, costing ~3K output
tokens per turn unnecessarily. Quick-mode default fixes that.
