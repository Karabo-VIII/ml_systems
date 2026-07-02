---
name: chart
description: "Chart / data-visualization workflow for the V4 Crypto System. Codifies the ad-hoc draw_*.py one-off scripts into ONE reusable, honest, correctly-accounted charting path. Invoke for ANY request to plot / chart / graph / visualize a strategy, config, equity curve, return/drawdown distribution, or per-asset comparison: 'chart the EMA config vs buy-hold', 'plot the equity curves for all instruments', 'show me the return distribution across u16', 'draw the MA family leaderboard'. Bakes in: per-asset-by-default scope, the canonical output location (mirror where the user looks), MtM-correct next-open accounting, and the MANDATORY honest note (path-luck returns vs structural drawdown-reduction). NOT a market read (that is narrate) and NOT an edge hunt (that is discover) -- this is the render+deliver layer."
argument-hint: "what to plot / which instruments / vs what -- e.g. 'EMA(50,100) 1d equity vs buy-hold, BTC SOL DOGE' or 'return-edge distribution across all u16'"
metadata:
  schema_version: "2026-07-02"
  aliases: ["/plot", "/viz", "/draw"]
  node_type: skill
---

You are the **charting / data-visualization layer** for the V4 Crypto System. Given a thing to
plot, you render the LITERAL artifact the user can inspect, in the canonical place they look, with
correct accounting and an HONEST read attached. Apply [`_common/STANDARDS.md`](../_common/STANDARDS.md).
Real capital; the chart must not flatter the strategy. Cite file:line; verify the file exists after writing.

## Your task
$ARGUMENTS

## The binding rules — read before writing any plotting code

1. **PER-ASSET BY DEFAULT.** "all assets" / "all instruments" / "the universe" means **ONE chart (or
   one row/bar) per instrument** — NEVER silently collapse into a handful of basket charts. If a
   basket/aggregate view is genuinely wanted, produce it IN ADDITION to per-asset, and say which is which.
   If the scope is ambiguous (per-asset vs aggregated, which universe, how many), **state the scope you
   are using in one line before rendering** (`u16 = the 16 chimera-available majors, one bar each`) —
   do not guess silently and force a re-do. (Provenance: repeated per-asset-vs-basket scope rework.)

2. **DELIVER WHERE THE USER LOOKS.** Before choosing an output path, find where the analogous prior run
   wrote its artifacts and **MIRROR that exact location + format** — do not invent a new directory the
   user won't think to open. Canonical homes:
   - MA / config equity + distribution charts → `crypto/runs/ma_config_charts/` (numbered: `04_...`, `05_...`).
   - Per-period config leaderboards + their charts → `crypto/runs/periods/TRAIN/<year>/DEEP_DIVE/`
     (mirror the 2020 layout for any new year — `CONFIG_LEADERBOARD.md` + `config_leaderboard.json` + charts/).
   - Engine / tier studies → alongside their JSON under `crypto/runs/engines/<STUDY>/`.
   Use **native absolute paths** in code (`c:/Users/karab/Documents/coding/ml_systems/crypto/...` — forward
   slashes are fine on Windows in Python). `os.makedirs(..., exist_ok=True)` before `savefig`.
   See [[feedback-deliver-literal-artifacts-where-user-looks]].

3. **THE HONEST NOTE IS MANDATORY** (not optional garnish). Every strategy/return chart must separate:
   - **path-luck (returns)** — per-asset return-edge vs buy-hold is typically a **coin-flip** (≈half beat,
     median ≈0 = beta); a single asset "winning" is draw luck, not alpha.
   - **structural (drawdown / risk)** — drawdown reduction is the **consistent** effect that holds across
     most assets; that is the real, repeatable benefit.
   State it numerically on the chart AND in the summary: `beats BH on RETURN n/N (median +Xpp = coin-flip)` and
   `LOWER drawdown on m/N (median −Ypp = consistent de-risking)`. Never present a returns headline without it.
   See [[project-ma-family-robustness-audit-2026-06-29]], [[feedback-rank-by-wealth-not-sharpe]].

4. **SURVIVORSHIP-NEUTRAL framing.** Prefer **RELATIVE** comparisons (config vs buy-hold **on the same
   asset**) — that difference is survivorship-neutral. An absolute equity curve inherits the asset's
   survivorship + the favorable bull; label it as such (`full history incl the bull`) and never let it
   carry the conclusion alone.

## Accounting invariants (MtM-correct — do NOT reintroduce the fixed bugs)

Reuse the reference scripts as templates:
`crypto/runs/engines/STRAT_TIERS/draw_ma_equity.py` (single-asset equity vs BH) and
`draw_coinflip_distribution.py` (cross-instrument return/drawdown distribution).

- **Next-open fills, 1-bar lag.** Decide on the close of `t-1`, fill at the open of `t`:
  `pos[1:] = held[:-1]`; return stream is open-to-open `roo[:-1] = open[1:]/open[:-1] - 1`.
- **Cost on TURNOVER only.** `turn = |diff(pos)|`; `net = pos*roo - turn*(COST/2)`. Prod round-trip
  `COST = 0.0028` (0.28% RT) unless told otherwise; state the cost + fill model on the chart.
- **MtM, no double-count.** Compound the per-bar net return stream (`cumprod(1+net)`); do not also add
  cumulative ret-from-entry (the 2026-04-22 double-count bug — see crypto/CLAUDE.md Backtest Simulator Invariants).
- **Data via the canonical loader / latest parquet.** Read `open`/`close` from the newest
  `data/processed/chimera/<tf>/<sym>_v51_chimera_<tf>_*.parquet` (glob, take `[-1]`), sort by timestamp.
- Skip assets with `n < ~250` bars (too short for a fair full-history read); report which were skipped, never drop silently.

## Rendering + delivery

- `import matplotlib; matplotlib.use("Agg")` (headless). `dpi≈115`, `bbox_inches="tight"`, log-y (`semilogy`)
  for equity curves so a 100x and a 2x are both legible.
- **NO emoji anywhere in the script** — print statements included (Windows cp1252 crashes; hard project invariant).
- Titles/legends carry the honest numbers (return% + maxDD% per line), not just labels.
- **VERIFY after write:** confirm `os.path.exists(out)` and `print("SAVED", out)` (the reference scripts do this);
  then in your reply LIST every file created with its absolute path so the user can open them. A chart claimed
  but not on disk is a silent failure.

## The value chain — where charting sits

`narrate` (what the market is doing) → `discover` (is there an edge) → **`chart`** (render the result the
user inspects) + `trader` (act on it). This skill does NOT decide whether a strategy is good — it renders
the evidence faithfully, honest note attached, so the user (and `decide`/`validator`) can judge.

## When to invoke

| Trigger | Why |
|---|---|
| "chart / plot / graph / draw / visualize X" | The core use case |
| "show me the equity curves / return distribution" | Literal-artifact request — render + deliver, don't reframe into a verdict |
| "across all instruments / the universe" | Per-asset by default; confirm scope in one line |
| "compare config vs buy-hold" | Relative, survivorship-neutral, honest note mandatory |
| After a `discover` / engine run produces numbers | Turn the result JSON into the inspectable chart |

Do NOT invoke for a market state read (that is `narrate`), an edge search (`discover`), or a numeric-claim
audit (`validator` / `quant`). Chart renders; it does not conclude.

## Gotchas

- **Reframing a "run it and show me" into a prose verdict** — the #1 failure here. Produce the CHARTS first,
  layer the verdict ON TOP, never instead. [[feedback-deliver-literal-artifacts-where-user-looks]]
- **A returns headline with no drawdown/robustness context** — inflates path-luck into "alpha". Always pair.
- **Aggregating "all assets" into 4 baskets** — that is the scope-collapse to avoid; per-asset unless asked.
- **New output directory the user won't open** — mirror the canonical location; if unsure, ask where they look.
- **Emoji in a print** — crashes on cp1252. Keep the whole script ASCII.
