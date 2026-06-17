# TI Harness — the configured technical-indicator lens

**Purpose (2026-06-09).** Read a rolling window through a CONFIGURED set of technical indicators (RSI / MA / MACD /
Bollinger / ADX / ATR). Descriptive — what the indicators *say* over the window — NOT a strategy.

## Why this is a distinct lens (the load-bearing point)
`decompose` / `narrate` read chimera features that **already exist** (pre-computed, fixed). `econometric_signature`
computes **canonical** estimators (no choice). A technical indicator is **fundamentally different — it does not exist
until you CONFIGURE it:** you must pick the FAMILY *and* the PARAMETERS (period, fast/slow/signal, σ-multiple,
thresholds) before any number exists, and a different config gives a **different read of the same data.** So a TI read
is always *relative to a config* — the harness makes the config a first-class, echoed input, never implicit.

**Demonstration (same AVAX 2023-01-02→09 window, +12.5%):**
| | `default` (RSI-14, EMA-20/50, MACD-12/26/9, BB-20/2σ) | `fast` (RSI-7, EMA-9/21, MACD-6/13/5, BB-10/2σ) |
|---|---|---|
| MACD histogram | **positive (bullish)** | **negative (bearish)** |
| Bollinger %B | 0.69 (upper-half) | 0.49 (lower-half) |

Identical price data, opposite momentum read — purely the config. That is the whole reason TI is its own lens.

## Tool: `src/mining/ti_harness.py`
```
python -m mining.ti_harness --asset BTC --cadence 4h --start 2025-01-01 --end 2025-01-08         # default config
python -m mining.ti_harness --asset SOL --cadence 4h --start 2024-07-09 --end 2024-07-16 --config fast --json
python -m mining.ti_harness --asset ETH --cadence 4h --start 2022-02-13 --end 2022-02-20 --config my_cfg.json
python -m mining.ti_harness --selftest
```
**Config** (`--config`): a preset NAME (`default` | `fast` | `slow`) or a path to a JSON file with the same shape — a
list of `{"family", "params", "thresholds"}` specs. Adding an indicator or changing a parameter = editing the config.

**Rolling-window + warmup:** a 7-day 4h window is ~42 bars, but an MA-50 needs 50 bars of history. The harness loads
`max(period)*3` bars BEFORE `start` to warm the indicators, computes over `[start-warmup, end]`, and reports STATE
within `[start, end]` (crosses counted only inside the window). Warmup-loaded bars are reported.

**Per-indicator output:** RSI (value + zone + threshold crosses + frac-of-window in each zone); MA (fast-vs-slow
golden/death + distance% + the cross dates in-window); MACD (histogram sign + MACD/signal cross dates); Bollinger
(%B + position + band-walk fraction + mean bandwidth); ADX (value + TRENDING/ranging verdict + ±DI direction); ATR
(% of price). Plus a one-line **TI CONSENSUS** synthesis. Output: text + `runs/mining/ti_<tag>.json`.

**Backend / reuse:** pandas_ta — the SAME library + indicator math as the oracle's `INDICATOR_REGISTRY`
(`src/oracle/indicators_ta.py`); this is a window READER, the oracle is the capture-analyzer. Not reinvention.

**Verified:** `--selftest` (data-free) — synthetic uptrend → EMA golden + ADX TRENDING + RSI not-oversold; downtrend
→ EMA death. 4/4 PASS.

## Place in the harness
The 4th lens alongside `decompose` (chimera table) + `narrate` (prose) + `econometric_signature` (process math). Use
TI when you want what a *chosen* indicator configuration says over the window; the answer is always config-tagged.
Tracked in the framework store (`01_mining`). No emoji (cp1252).
