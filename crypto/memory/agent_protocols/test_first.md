# Test-First Protocol

> For non-trivial code work, write the smoke test BEFORE building. Locks the
> success criterion. Devin / TDD-style discipline.

## Trigger

Apply to:

- New sleeve adapters (e.g., upcoming P8 listing momentum sleeve)
- New runners in `KNOWN_RUNNERS` (`src/strategy/gen5_growth/blend_composer.py`)
- New model architectures (any `src/wm/v*/` addition)
- New pipeline producers (`src/pipeline/`)
- Anything claiming "delivers feature X" where X is testable

Do NOT apply to:

- Bug fixes < 10 lines
- Doc / comment / log message changes
- Renames / refactors with no behavior change

## Steps

### 1. Define success in one sentence
> "<artifact> is COMPLETE when <observable condition>."

Examples:
- "`p8_listing_h1_sleeve.run_p8_listing_sleeve()` is COMPLETE when called with
  `pick_date='2026-04-22'` and returns ≥1 valid TradeIntent dict referencing
  a known recent listing."
- "Range bar coarsener is COMPLETE when input 30M raw range bars produce ≤300
  output bars per day with monotone ts and no NaN close."

### 2. Write the smoke test FIRST
Before building the artifact, write the test that defines success. The test
must FAIL meaningfully when the artifact doesn't exist or is broken.

Examples:
```python
# tests/smoke/test_p8_listing_sleeve.py
def test_p8_listing_sleeve_returns_intent():
    from strategy.sleeves.p8_listing_h1_sleeve import run_p8_listing_sleeve
    intents = run_p8_listing_sleeve("p8_smoke", blend_weight_pct=0.05,
                                      pick_date="2026-04-22")
    assert isinstance(intents, list), "must return list"
    assert all("asset" in i and "side" in i for i in intents), "intents schema"
    # At least one real (non-CASH) intent if listing event present
    non_cash = [i for i in intents if i.get("asset") != "USDC"]
    assert len(non_cash) >= 1, "expected at least 1 listing intent for known date"
```

### 3. Run the test — confirm it FAILS appropriately
Run it before building. Expected failure: `ImportError` or `FileNotFoundError`.
**If it passes immediately, the test isn't actually testing your change.**

### 4. Build until the test passes
Iterate. Do not declare done until the test passes cleanly.

### 5. Add the test to the project's smoke suite
- Sleeve adapters → `tests/smoke/test_<sleeve_name>.py`
- Pipeline producers → `tests/smoke/test_<producer>.py`
- Model architectures → smoke fixture in the model's settings/test file

### 6. Document the test as the success criterion in the work log
> "Done when `tests/smoke/test_<name>.py` passes. (Verified: 1/1 passing, 0.4s)"

## Why this matters here

The project has 12+ shipped strategy sleeves and ~10 model versions. Without
locked success criteria:

- "Done" gets defined retroactively to whatever the code happens to do
- Drift between intent and implementation is invisible
- Hand-offs to other Claude instances lose the success-criterion context

Locking the test FIRST means every artifact has an observable definition of
done that survives across sessions.

## Failure mode this prevents

A sleeve adapter built without test-first may emit intents that LOOK right
(non-empty list) but fail at the runner level (wrong field names, wrong
units). The test catches the actual integration surface; "I wrote some code"
doesn't.

## Special: ML training success criteria

For training runs:
- Smoke = "model trains 20 steps without NaN" — minimum
- Real success criterion = ShIC / IC gate thresholds from CLAUDE.md
- A training run is COMPLETE when:
  1. Smoke smoke passed
  2. After full epoch budget, IC and ShIC gates are clear
  3. Validation harness (`src/wm/v*/validate.py`) reports gate-pass

Don't claim "training done" without all three.

## Integration with CDAP

CDAP (`src/audit/check_invariants.py`) is a project-level invariant check.
This protocol is a per-artifact success criterion. They're complementary:
- CDAP says "no rule broken"
- test-first says "intended behavior delivered"

Both must pass for "done."
