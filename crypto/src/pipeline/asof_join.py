"""Bitemporal (as-of) join: attach a feature to a base row ONLY when it was
KNOWABLE.

THE GAP THIS CLOSES
-------------------
The project catches look-ahead REACTIVELY today: ad-hoc per-feature shift(1)
in producers, plus a downstream forward-return correlation / MI / same-day-
publication-race detector in src/pipeline/validate_chimera.py. Those detectors
fire AFTER a leak is already baked into chimera. What's missing is a STRUCTURAL
layer that records WHEN each datum became KNOWABLE, so vendor revisions /
backfills (ETF flows, Wikipedia/attention, funding) and same-day-publication
races cannot silently attach a value to a bar that closed BEFORE that value
existed.

THE BITEMPORAL CONTRACT
-----------------------
Every feature row has TWO times:
  - event_time      (feature_time_col): the timestamp the datum DESCRIBES
                    (e.g. the day an ETF flow refers to, the funding interval).
  - knowable_at     = event_time + publication_lag: the timestamp at which a
                    live trader could ACTUALLY have observed it (vendor publish
                    delay + same-day-publication race + backfill settling).

A feature value attaches to a base row at base_time ONLY when

        knowable_at <= base_time            (event_time + lag <= base_time)

i.e. a BACKWARD as-of join keyed on knowable_at, taking the most-recent value
that was already knowable. This is structurally leak-free: a value published
"on day T" but referring to day T (lag >= 1 bar) is excluded from bar T and
first appears on bar T+1, exactly as it would live.

PURE + DETERMINISTIC
--------------------
No I/O, no global state, no wall-clock. Same inputs -> same output, bit-exact.
load_publication_lags() is the only function that touches disk (reads the
declared-lag YAML); the join itself takes the lag as an explicit argument.

Self-test (positive-control leak test): run
    python src/pipeline/asof_join.py
It constructs a base frame + a feature frame where a NAIVE same-time join would
leak a future value into an earlier bar, and proves asof_join_knowable with the
correct publication_lag EXCLUDES that value (prints PASS / FAIL).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LAG_YAML = PROJECT_ROOT / "config" / "feature_publication_lag.yaml"

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_io",
    "inputs": {
        "args": [
            "base (pl.DataFrame), feature (pl.DataFrame), feature_cols (list[str]), "
            "publication_lag_ms (int), base_time_col (str), feature_time_col (str), "
            "by (optional join key e.g. 'asset')"
        ]
    },
    "outputs": {
        "returns": "pl.DataFrame: base rows with feature_cols attached only where "
                   "feature_time + publication_lag <= base_time (knowable-at contract)"
    },
    "invariants": {
        "no_lookahead": True,            # value never attaches before knowable_at
        "pure_deterministic": True,      # no I/O / clock in the join; bit-exact
        "backward_asof_only": True,      # strategy='backward' on knowable_at
        "lag_is_explicit": True,         # publication_lag_ms is a required arg
        "non_negative_lag": True,        # negative lag would re-introduce look-ahead
    },
    "rationale": "Structural look-ahead defense: bitemporal knowable-at join "
                 "replaces ad-hoc shift(1) + reactive correlation leak detection. "
                 "Closes vendor-revision / backfill / same-day-publication-race "
                 "paper-to-live divergence (ETF flows, Wikipedia, funding, onchain).",
}


def asof_join_knowable(
    base: pl.DataFrame,
    feature: pl.DataFrame,
    feature_cols: list[str],
    publication_lag_ms: int,
    *,
    base_time_col: str = "ts",
    feature_time_col: str = "ts",
    by: Optional[str] = None,
    knowable_col: str = "_knowable_at",
) -> pl.DataFrame:
    """Backward as-of join on the KNOWABLE-AT timestamp. No look-ahead.

    Each feature row's value becomes knowable at
        knowable_at = feature_time + publication_lag_ms
    and is attached to a base row at base_time ONLY when knowable_at <= base_time
    (the most-recent such value wins -- standard backward as-of semantics).

    This is the structural look-ahead defense: a value referring to time T but
    published with a >= 1-bar lag cannot attach to bar T (it first appears on
    the bar whose base_time >= knowable_at), exactly mirroring live trading.

    Args:
        base: left frame; one row per base observation. MUST contain
            base_time_col (and `by` if provided). Returned 1:1 (same row count).
        feature: right frame; the lag-prone source. MUST contain feature_time_col,
            feature_cols (and `by` if provided).
        feature_cols: feature columns to attach. Must NOT collide with base columns
            unless intentionally overwriting (collision raises).
        publication_lag_ms: publication delay in milliseconds. MUST be >= 0;
            negative lag would let a value attach BEFORE its event time
            (look-ahead) and raises ValueError. Use load_publication_lags() to
            source the declared per-family lag.
        base_time_col: timestamp column in base (13-digit ms epoch, per project
            invariant). Default 'ts'.
        feature_time_col: EVENT-time column in feature (the time the datum
            describes, NOT the publish time). Default 'ts'.
        by: optional grouping key for per-group as-of (e.g. 'asset'). When set,
            the join matches within each `by` group independently.
        knowable_col: name of the temporary knowable-at column (dropped before
            return). Override only if it would collide with a real column.

    Returns:
        pl.DataFrame: base with feature_cols attached (null where no knowable
        value exists yet). Row count == len(base); base row order preserved.

    Raises:
        ValueError: publication_lag_ms < 0; missing required columns; or a
            feature_col collides with an existing base column.
    """
    if publication_lag_ms < 0:
        raise ValueError(
            f"publication_lag_ms must be >= 0 (got {publication_lag_ms}); a "
            f"negative lag re-introduces look-ahead -- the value would attach "
            f"before its event_time."
        )
    if base_time_col not in base.columns:
        raise ValueError(f"base missing base_time_col '{base_time_col}'")
    if feature_time_col not in feature.columns:
        raise ValueError(f"feature missing feature_time_col '{feature_time_col}'")
    missing_feats = [c for c in feature_cols if c not in feature.columns]
    if missing_feats:
        raise ValueError(f"feature missing feature_cols: {missing_feats}")
    collisions = [c for c in feature_cols if c in base.columns]
    if collisions:
        raise ValueError(
            f"feature_cols collide with existing base columns: {collisions}; "
            f"rename or prefix the feature columns before joining."
        )
    if by is not None:
        if by not in base.columns:
            raise ValueError(f"base missing 'by' key '{by}'")
        if by not in feature.columns:
            raise ValueError(f"feature missing 'by' key '{by}'")
        if knowable_col == by:
            raise ValueError(f"knowable_col cannot equal 'by' key '{by}'")

    # Build the knowable-at column on the RIGHT (feature) side. polars join_asof
    # requires BOTH sides sorted on their respective `on` keys.
    feat = feature.with_columns(
        (pl.col(feature_time_col) + pl.lit(publication_lag_ms)).alias(knowable_col)
    )
    # Keep only the columns we need from the feature side: knowable_at, the
    # requested feature_cols, and (if grouping) the `by` key.
    keep_cols = [knowable_col] + list(feature_cols)
    if by is not None:
        keep_cols.append(by)
    feat = feat.select(keep_cols)

    # Tag base rows so we can restore original order after the sort the as-of
    # join requires (join_asof needs both sides sorted by the on-key).
    order_col = "_asof_orig_order"
    base_tagged = base.with_row_index(order_col)

    sort_keys_base = ([by, base_time_col] if by is not None else [base_time_col])
    sort_keys_feat = ([by, knowable_col] if by is not None else [knowable_col])
    base_sorted = base_tagged.sort(sort_keys_base)
    feat_sorted = feat.sort(sort_keys_feat)

    # Mark the as-of on-keys as sorted. We just sorted on them, so this is
    # correct (not a fib to polars).
    base_sorted = base_sorted.set_sorted(base_time_col)
    feat_sorted = feat_sorted.set_sorted(knowable_col)

    # On the grouped (`by`) path polars 1.x still emits a cosmetic
    # "Sortedness of columns cannot be checked when 'by' groups provided"
    # UserWarning even though we just sorted -- it skips the CHECK, not the
    # warning. Suppress only that specific message so a stray warning isn't
    # misread as a defect; correctness is unaffected (inputs ARE sorted).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked",
            category=UserWarning,
        )
        joined = base_sorted.join_asof(
            feat_sorted,
            left_on=base_time_col,
            right_on=knowable_col,
            by=by,
            strategy="backward",  # most-recent value whose knowable_at <= base_time
        )

    # join_asof leaves the right-side key (knowable_col) in the output; drop it,
    # restore the original base row order, drop the helper index.
    drop_cols = [c for c in (knowable_col, order_col) if c in joined.columns]
    return joined.sort(order_col).drop(drop_cols)


def load_publication_lags(
    yaml_path: Optional[Path | str] = None,
) -> dict[str, int]:
    """Load the declared per-family publication lag (ms) from the YAML registry.

    Reads config/feature_publication_lag.yaml (or an override path). Returns a
    flat {family_name: lag_ms} dict so callers can do
        lag = load_publication_lags()["etf_flows"]
    The YAML's nested {family: {lag_ms, note, ...}} structure is collapsed to
    just the integer lag; the human note is documentation-only.

    Args:
        yaml_path: override path; defaults to config/feature_publication_lag.yaml.

    Returns:
        dict[str, int]: family-name -> publication lag in milliseconds.

    Raises:
        FileNotFoundError: YAML missing.
        ValueError: a family entry has a missing / non-int / negative lag_ms.
    """
    import yaml  # local import: keeps the join path dependency-light

    path = Path(yaml_path) if yaml_path is not None else DEFAULT_LAG_YAML
    if not path.exists():
        raise FileNotFoundError(f"publication-lag YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    families = spec.get("families", {}) or {}
    out: dict[str, int] = {}
    for name, entry in families.items():
        if not isinstance(entry, dict) or "lag_ms" not in entry:
            raise ValueError(
                f"publication-lag family '{name}' must declare an integer lag_ms"
            )
        lag = entry["lag_ms"]
        if not isinstance(lag, int) or lag < 0:
            raise ValueError(
                f"publication-lag family '{name}': lag_ms must be a non-negative "
                f"int (got {lag!r})"
            )
        out[str(name)] = lag
    return out


# Common conversions for declaring lags in human terms.
MS_PER_SECOND = 1_000
MS_PER_MINUTE = 60 * MS_PER_SECOND
MS_PER_HOUR = 60 * MS_PER_MINUTE
MS_PER_DAY = 24 * MS_PER_HOUR


def _selftest() -> int:
    """Positive-control leak test. Returns 0 on PASS, 1 on FAIL.

    Construct a tiny base frame (5 daily bars) + a feature frame whose value
    AT day-2 (event_time) would, under a NAIVE same-time join, leak into the
    day-2 base bar -- even though the datum (e.g. an ETF flow referring to
    day-2) is only PUBLISHED the next day. With a 1-day publication lag, the
    leak-prone value must FIRST appear on day-3, never day-2.
    """
    print("=" * 68)
    print("asof_join positive-control leak test")
    print("=" * 68)

    one_day = MS_PER_DAY
    base_t0 = 1_700_000_000_000  # arbitrary 13-digit ms epoch (project invariant)

    # 5 consecutive daily base bars: day0..day4.
    base = pl.DataFrame({
        "ts": [base_t0 + i * one_day for i in range(5)],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    })

    # Feature: an ETF-flow-like value whose EVENT time is day2, but which a live
    # trader can only observe the NEXT day (publication lag = 1 day). The value
    # 999.0 is a sentinel: if it shows up on the day2 base bar, that's a leak.
    feature = pl.DataFrame({
        "ts": [base_t0 + 2 * one_day],   # event_time = day2
        "etf_flow": [999.0],
    })

    # --- NAIVE join (zero lag): the wrong, leaky behavior. ---
    naive = asof_join_knowable(
        base, feature, ["etf_flow"], publication_lag_ms=0,
        base_time_col="ts", feature_time_col="ts",
    )
    naive_day2 = naive.filter(pl.col("ts") == base_t0 + 2 * one_day)["etf_flow"][0]

    # --- CORRECT join (1-day publication lag): structurally leak-free. ---
    correct = asof_join_knowable(
        base, feature, ["etf_flow"], publication_lag_ms=one_day,
        base_time_col="ts", feature_time_col="ts",
    )
    correct_day2 = correct.filter(pl.col("ts") == base_t0 + 2 * one_day)["etf_flow"][0]
    correct_day3 = correct.filter(pl.col("ts") == base_t0 + 3 * one_day)["etf_flow"][0]

    print(f"  feature event_time      : day2 (value=999.0)")
    print(f"  publication lag         : 1 day -> knowable_at = day3")
    print()
    print(f"  NAIVE (lag=0)  day2 etf_flow = {naive_day2!r}   "
          f"<- LEAKED future value into day2 bar")
    print(f"  CORRECT(lag=1d) day2 etf_flow = {correct_day2!r}   "
          f"<- excluded (not yet knowable on day2)")
    print(f"  CORRECT(lag=1d) day3 etf_flow = {correct_day3!r}  "
          f"<- first appears on day3 (knowable), as it would live")
    print()

    # Assertions that define PASS:
    #  1. Naive join LEAKED the value into day2 (proves the test is real).
    #  2. Correct join EXCLUDED it from day2 (null) -- the look-ahead is
    #     structurally prevented.
    #  3. Correct join lets it appear on day3 (the join is not just nulling
    #     everything -- it attaches the value exactly when knowable).
    leak_demonstrated = (naive_day2 == 999.0)
    leak_prevented = (correct_day2 is None)
    value_appears_when_knowable = (correct_day3 == 999.0)

    ok = leak_demonstrated and leak_prevented and value_appears_when_knowable

    # Extra guard: a negative lag MUST be rejected (it would re-open look-ahead).
    neg_lag_rejected = False
    try:
        asof_join_knowable(base, feature, ["etf_flow"], publication_lag_ms=-one_day)
    except ValueError:
        neg_lag_rejected = True
    print(f"  negative-lag rejected   : {neg_lag_rejected}  "
          f"(guards against re-introducing look-ahead)")
    ok = ok and neg_lag_rejected

    # Extra guard: per-asset (`by`) path produces leak-free results too.
    base_a = pl.DataFrame({
        "ts": [base_t0 + i * one_day for i in range(3)] * 2,
        "asset": ["BTC"] * 3 + ["ETH"] * 3,
        "close": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    feat_a = pl.DataFrame({
        "ts": [base_t0 + 1 * one_day, base_t0 + 1 * one_day],
        "asset": ["BTC", "ETH"],
        "funding": [0.5, -0.5],
    })
    by_res = asof_join_knowable(
        base_a, feat_a, ["funding"], publication_lag_ms=one_day, by="asset",
    )
    # funding event=day1, lag=1d -> knowable day2; must be null on day0/day1,
    # present on day2, per asset, with no cross-asset bleed.
    btc_day1 = by_res.filter((pl.col("asset") == "BTC")
                             & (pl.col("ts") == base_t0 + one_day))["funding"][0]
    btc_day2 = by_res.filter((pl.col("asset") == "BTC")
                             & (pl.col("ts") == base_t0 + 2 * one_day))["funding"][0]
    eth_day2 = by_res.filter((pl.col("asset") == "ETH")
                             & (pl.col("ts") == base_t0 + 2 * one_day))["funding"][0]
    by_ok = (btc_day1 is None) and (btc_day2 == 0.5) and (eth_day2 == -0.5)
    print(f"  per-asset (by='asset')  : BTC day1={btc_day1!r} (excluded), "
          f"BTC day2={btc_day2!r}, ETH day2={eth_day2!r}  -> {'ok' if by_ok else 'BAD'}")
    ok = ok and by_ok

    print()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
