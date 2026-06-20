# CURRENT — live state (Claude-written each cycle; committed; <2KB)

PHASE: BUILD
UPDATED: 2026-06-20

## CURRENT GOAL
Build the ability to **trade ANY SLICE** — any setup from any asset across the **u100 universe** — compounding, with
~100% coverage of **2%+ daily moves**, a **7-day lookback** window (a known prior weakness to fix), reusing the whole
project's move-catch infra. This is the live acid test: it must NOT collapse to "trade BTC only."

## SURFACE TO COVER (all four, or the null is not earned)
- [ ] UNIVERSE: >=20 distinct u100 symbols touched in executed code (not BTC-only).
- [ ] WINDOW: 7d lookback as a real executed parameter.
- [ ] COVERAGE: the 2%+ daily-move set enumerated/measured (a coverage manifest: realized vs available capture).
- [ ] INFRA: extend `src/strat/move_catch_book.py` / `oracle_capture_lab.py` / chimera loader — not a fresh BTC script.

## STANDING TRUTH (the floor, not the verdict)
Internal long-only TI work has repeatedly landed on **de-risked beta** (preserves bears, doesn't beat buy-hold on net).
That is the FLOOR to build within, not a stop. The genuinely-untested corners: the move-CAPTURE-RATE lens, dynamic
regime->config-set switching, chimera x TI conditioning, the ironed FINE-TF band (cost doesn't bind a properly-ironed
fine-TF band; under-participation does). The held-out POSITIVE so far is funding-dispersion carry (dollar-neutral perps;
boxed pending the user). External-listing-data is the open user-gated frontier.

## NEXT ACTIONS
1. Run the u100 x 7d x all-TF move-capture sweep through `move_catch_book` + `oracle_capture_lab` capture KPI.
2. Build the 2%+ daily-move coverage manifest (realized/available per cell).
3. Honest gates (fixed-hold null, random-entry, p05); rank survivors by WEALTH; aggregate the book.

## DO NOT
Collapse to one asset. Default a single TF. Propose the short side. Stop at a "de-risked beta, done" verdict without a
constructed within-constraint book.
