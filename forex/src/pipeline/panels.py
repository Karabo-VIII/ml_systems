"""Cross-pair panels (silver) — USD strength, currency strength, triangular no-arb.

These are the FX-native cross-sectional signals a per-pair feature frame cannot
see. All are LOOK-AHEAD-SAFE (per-bar contemporaneous aggregates + trailing
cumsum; no future info).

  usd_strength_ret / usd_index   average USD-side log return across the majors
                                 (sign-correct: +ret for USDxxx, -ret for xxxUSD),
                                 and its cumulative index (DXY-like).
  str_<CCY>                      per-currency strength: null-aware mean of that
                                 currency's signed log return across every pair
                                 it appears in.
  tri_<CROSS>_resid_pips         triangular no-arb residual:
                                 log(cross) - [log(BASE/USD) - log(QUOTE/USD)].
                                 ~0 up to spreads; a persistent nonzero = stale
                                 data or a real dislocation (feature + QC signal).
  tri_max_abs_resid_pips         data-quality roll-up across all triplets.

Output: data/processed/panels/<cadence>/panel_<YYYYMMDD>.parquet

    python src/pipeline/panels.py --universe u28 --cadences 1h 1d
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from functools import reduce
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.parquet_io import atomic_write_parquet
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "panels",
    "inputs": {"upstream": "data/processed/bars/<cadence>/<pair>_<cadence>_*.parquet",
               "args": ["--pairs", "--universe", "--cadences"]},
    "outputs": {"files": "data/processed/panels/<cadence>/panel_<YYYYMMDD>.parquet"},
    "invariants": {"causal_no_lookahead": True, "usd_sign_correct": True,
                   "triangular_consistency_is_qc_and_feature": True, "atomic_write": True},
    "rationale": "Cross-pair FX signals (USD strength, currency strength, no-arb).",
}

CURRENCIES = ("EUR", "USD", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD")
_EPS = 1e-12


def _load_close_wide(pairs: list[str], cadence: str) -> tuple[pl.DataFrame, list[str]]:
    """Outer-join every available pair's close on timestamp -> wide frame."""
    frames, present = [], []
    for p in pairs:
        path = layout.bars_latest(p, cadence)
        if path is None or not path.exists():
            continue
        d = pl.read_parquet(path, columns=["timestamp", "close"]).rename({"close": p})
        frames.append(d)
        present.append(p)
    if not frames:
        return pl.DataFrame(), []
    wide = reduce(lambda a, b: a.join(b, on="timestamp", how="full", coalesce=True), frames)
    return wide.sort("timestamp"), present


def _null_aware_mean(exprs: list[pl.Expr]) -> pl.Expr:
    """Mean over the given per-pair expressions, ignoring nulls."""
    total = pl.sum_horizontal([e.fill_null(0.0) for e in exprs])
    count = pl.sum_horizontal([e.is_not_null().cast(pl.Float64) for e in exprs])
    return pl.when(count > 0).then(total / count).otherwise(None)


def _usd_side_return(pair: str, ret: pl.Expr) -> pl.Expr:
    """USD-side log return: +ret if USD is the base (USDxxx), -ret if quote (xxxUSD)."""
    n = layout.normalize_pair(pair)
    if n.base == "USD":
        return ret
    if n.quote == "USD":
        return -ret
    return pl.lit(None, dtype=pl.Float64)


def _ccy_side_return(pair: str, ccy: str, ret: pl.Expr) -> pl.Expr | None:
    """Signed log return of `ccy` in `pair`: +ret if base, -ret if quote, else None."""
    n = layout.normalize_pair(pair)
    if n.base == ccy:
        return ret
    if n.quote == ccy:
        return -ret
    return None


# Standard cross triplets (BASE, QUOTE) to check, if all legs are present.
_TRIPLETS = [("EUR", "JPY"), ("GBP", "JPY"), ("AUD", "JPY"), ("CAD", "JPY"),
             ("CHF", "JPY"), ("NZD", "JPY"), ("EUR", "GBP"), ("EUR", "CHF"),
             ("EUR", "AUD"), ("EUR", "CAD"), ("EUR", "NZD"), ("GBP", "CHF"),
             ("GBP", "AUD"), ("GBP", "CAD"), ("GBP", "NZD"), ("AUD", "NZD"),
             ("AUD", "CAD"), ("CAD", "CHF"), ("NZD", "CAD")]


def _leg_over_usd(ccy: str, present: set[str], logclose: dict[str, pl.Expr]) -> pl.Expr | None:
    """log(ccy/USD) from an available USD pair (ccyUSD => +, USDccy => -)."""
    if f"{ccy}USD" in present:
        return logclose[f"{ccy}USD"]
    if f"USD{ccy}" in present:
        return -logclose[f"USD{ccy}"]
    return None


def build_panel(pairs: list[str], cadence: str, U: UniverseLoader) -> pl.DataFrame | None:
    wide, present = _load_close_wide(pairs, cadence)
    if wide.is_empty():
        return None
    present_set = set(present)
    rets = {p: (pl.col(p) / pl.col(p).shift(1)).log() for p in present}
    logclose = {p: pl.col(p).log() for p in present}

    out = [pl.col("timestamp")]

    # --- USD strength (majors present) ---
    majors = [p for p in present if p in
              ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")]
    if majors:
        usd_terms = [_usd_side_return(p, rets[p]) for p in majors]
        out.append(_null_aware_mean(usd_terms).alias("usd_strength_ret"))

    # --- per-currency strength ---
    for ccy in CURRENCIES:
        terms = [e for p in present
                 if (e := _ccy_side_return(p, ccy, rets[p])) is not None]
        if terms:
            out.append(_null_aware_mean(terms).alias(f"str_{ccy}"))

    # --- triangular no-arb residuals (pips of the cross) ---
    tri_cols = []
    for base, quote in _TRIPLETS:
        cross = f"{base}{quote}"
        if cross not in present_set:
            continue
        base_leg = _leg_over_usd(base, present_set, logclose)
        quote_leg = _leg_over_usd(quote, present_set, logclose)
        if base_leg is None or quote_leg is None:
            continue
        implied = base_leg - quote_leg                 # log(BASE/USD) - log(QUOTE/USD)
        resid = logclose[cross] - implied              # ~0 up to spreads
        pip = U.pip(cross)
        # resid is in log space; convert to pips via resid * price / pip ~ price move.
        col = f"tri_{cross}_resid_pips"
        out.append((resid * pl.col(cross) / pip).alias(col))
        tri_cols.append(col)

    panel = wide.select(out)
    if tri_cols:
        panel = panel.with_columns(
            pl.max_horizontal([pl.col(c).abs() for c in tri_cols]).alias("tri_max_abs_resid_pips"))
    # cumulative USD index (trailing cumsum of the per-bar USD strength return).
    if "usd_strength_ret" in panel.columns:
        panel = panel.with_columns(
            pl.col("usd_strength_ret").fill_null(0.0).cum_sum().alias("usd_index"))
    return panel


def build_cadence(pairs: list[str], cadence: str, U: UniverseLoader) -> dict:
    panel = build_panel(pairs, cadence, U)
    if panel is None or panel.is_empty():
        return {"status": "no_bars", "cadence": cadence}
    latest = datetime.fromtimestamp(panel["timestamp"].max() / 1000, tz=timezone.utc).date()
    out = layout.panel_path("panel", cadence, latest_date=latest)
    atomic_write_parquet(panel, out, required_cols={"timestamp"})
    n_tri = len([c for c in panel.columns if c.startswith("tri_") and c.endswith("resid_pips")])
    return {"status": "ok", "cadence": cadence, "rows": len(panel),
            "cols": len(panel.columns), "n_triplets": n_tri, "path": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-pair FX panels")
    cli.add_standard_args(ap, default_workers=1, date_window=False)
    ap.add_argument("--cadences", nargs="+", default=["1h", "1d"])
    args = ap.parse_args()
    # default the panel to u28 so triangulars/currency-strength have crosses.
    if not args.pairs and not args.universe:
        args.universe = "u28"
    pairs = cli.resolve_pairs(args, stage_name="panels")
    U = UniverseLoader.load()
    for cad in args.cadences:
        print(f"[panels] {build_cadence(pairs, cad, U)}", flush=True)


if __name__ == "__main__":
    main()
