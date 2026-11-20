"""Transfer Entropy daily panel -- directional information flow between u10 assets.

Per Schreiber (2000), TE(X->Y) measures how much knowing X's history reduces
uncertainty about Y's future, BEYOND what Y's own history tells you. Directional;
TE(X->Y) != TE(Y->X). Adds info-flow signal orthogonal to existing xd_*
(correlation-based) features.

Method:
  For each rolling 90-day window of daily returns:
    Compute TE(X -> Y) for each asset pair X,Y in u10 with binning n_bins=3.
    Aggregate per-asset:
      te_in_<asset>  = max over X of TE(X -> asset)
      te_out_<asset> = max over Y of TE(asset -> Y)
      te_in_btc      = TE(BTC -> asset)
      te_out_btc     = TE(asset -> BTC)
      te_imb         = te_in - te_out (positive => asset is a follower)
      te_btc_imb     = te_in_btc - te_out_btc

Output: data/processed/panels/daily/te_panel_<DATE>.parquet
Schema: date, asset, te_in, te_out, te_in_btc, te_out_btc, te_imb, te_btc_imb

Source: builds on src/strategy/ml/transfer_entropy.py (the te_matrix function).

Wired in:
  - src/pipeline/build_panels.py "te" entry
  - config/feature_registry.yaml source `transfer_entropy`
  - src/feature_sets.py FEATURE_LIST_133 = FEATURE_LIST_127 + TE_6
"""
from __future__ import annotations
import os

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
# 2026-05-29: the transfer_entropy kernel was orphaned when src/strategy/ml was
# archived (src/strategy -> archive/). It is a self-contained numpy kernel, so it
# now lives next to its only consumer at src/pipeline/transfer_entropy.py (found
# via the src/pipeline path insert above). The stale src/strategy/ml insert was
# removed -- it pointed at a directory that no longer exists.
import layout as _layout  # noqa: E402
from transfer_entropy import (  # noqa: E402
    transfer_entropy, bin_series, te_matrix_from_returns, _te_from_binned,
)

PANEL_NAME = "te_panel"

# u10 default -- TE matrix is N x N (N^2 directional pairs). Override at runtime
# via --universe / --assets per @browser B5 (universe propagation checked).
DEFAULT_U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
# U10 is mutated at main() entry from CLI args; module-level kept for back-compat.
# _TE_N_WORKERS is similarly mutated at main() entry from --workers (default 8).
_TE_N_WORKERS = 8
U10 = list(DEFAULT_U10)

# Rolling-window length for TE estimation. 90 days gives ~stable TE estimates
# while staying responsive to regime shifts. With 1-day stride, TE on
# n=90 returns at lag=1 has FWE ~0.05 in the LM-style null.
WINDOW_DAYS = 90
LAG = 1
N_BINS = 3


_DAILY_RETURNS_CACHE = (
    PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "te_daily_returns_cache.parquet"
)


def _aggregate_one_asset(asset: str, raw_root: Path) -> pl.DataFrame | None:
    """Per-asset daily log-return aggregation. Only re-reads files newer than cache."""
    asset_dir = raw_root / asset / "aggTrades"
    if not asset_dir.exists():
        return None
    files = sorted(asset_dir.glob("*.parquet"))
    if not files:
        return None
    # 2026-05-13: include timestamp + sort to handle 2026-Q1+ unsorted aggTrades
    # (see memory/fix_logs/pipeline_aggtrades_us_unsort_2026_05_13.md).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
    from _aggtrades_utils import prepare_aggtrades  # noqa: E402
    per_day = []
    for fp in files:
        try:
            df = pl.read_parquet(fp, columns=["timestamp", "price"])
            if df.height < 2:
                continue
            # Sort by timestamp so first/last price is chronological, even when
            # source aggTrades arrive out-of-order (Binance 2026-03+).
            df = prepare_aggtrades(df, ts_col="timestamp")
            p_first = float(df["price"][0])
            p_last = float(df["price"][-1])
            if p_first <= 0 or p_last <= 0:
                continue
            ret = float(np.log(p_last / p_first))
            stem = fp.stem
            date_part = "-".join(stem.split("-")[-3:])
            from datetime import date as _d
            d = _d.fromisoformat(date_part)
            # Strip USDT suffix to match the consolidator's asset filter
            # convention (other panels: BTC / ETH / ...).
            asset_root = asset.replace("USDT", "") if asset.endswith("USDT") else asset
            per_day.append({"date": d, "asset": asset_root, "target_return_1": ret})
        except Exception:
            continue
    if not per_day:
        return None
    return pl.DataFrame(per_day)


def _load_daily_returns_panel() -> pl.DataFrame:
    """Build (date, asset, ret_1d) panel from raw aggTrades. Cached.

    Self-sufficient -- no dependency on chimera_legacy or chimera_v51, so this
    panel can be built at T2 stage (same as RV) before chimera builds.

    Per-asset, per-day: log-return between first and last price of the day.
    Cache at te_daily_returns_cache.parquet (~10MB). Cache is invalidated by:
      (a) cache mtime older than newest aggTrades file across U10
      (b) cache asset set != requested U10
    """
    import time as _time
    raw_root = PROJECT_ROOT / "data" / "raw"

    # Cache check: must be fresher than newest raw aggTrades file across U10
    if _DAILY_RETURNS_CACHE.exists():
        try:
            cached = pl.read_parquet(_DAILY_RETURNS_CACHE)
            cached_assets = set(cached["asset"].unique().to_list())
            requested = set(U10)
            if requested.issubset(cached_assets):
                cache_mtime = _DAILY_RETURNS_CACHE.stat().st_mtime
                # Find newest raw file across requested assets
                newest_raw = 0.0
                for asset in U10:
                    d = raw_root / asset / "aggTrades"
                    if not d.exists():
                        continue
                    for fp in d.glob("*.parquet"):
                        m = fp.stat().st_mtime
                        if m > newest_raw:
                            newest_raw = m
                            if newest_raw > cache_mtime:
                                break
                    if newest_raw > cache_mtime:
                        break
                if newest_raw <= cache_mtime:
                    # Cache fresh -- filter to requested universe
                    cached = cached.filter(pl.col("asset").is_in(list(U10)))
                    print(f"[te] cache hit: {cached.height} rows from "
                          f"{_DAILY_RETURNS_CACHE.name} (skipping {len(U10)} asset re-aggs)",
                          flush=True)
                    return cached.sort(["asset", "date"])
        except Exception as e:
            print(f"[te] cache check failed ({e}); rebuilding", flush=True)

    print(f"[te] building daily-returns panel from raw aggTrades for {len(U10)} assets...",
          flush=True)
    t0 = _time.time()
    rows = []
    # ThreadPool over per-asset aggregates: each call reads ~700-2000 small
    # parquets and computes one log-return per day. I/O bound; threads share
    # OS file cache and polars releases GIL on read_parquet. 8 threads
    # saturate disk on most systems.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = min(_TE_N_WORKERS, len(U10))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_aggregate_one_asset, asset, raw_root): asset
                    for asset in U10}
        for fut in as_completed(futures):
            asset = futures[fut]
            completed += 1
            try:
                df = fut.result()
            except Exception as e:
                print(f"  [te load] {asset}: ERROR {type(e).__name__}: {e}", flush=True)
                continue
            if df is not None:
                rows.append(df)
            if completed % 10 == 0 or completed == len(U10):
                print(f"  [te load {completed}/{len(U10)}] elapsed {_time.time()-t0:.0f}s "
                      f"({n_workers} threads)", flush=True)
    if not rows:
        raise RuntimeError("No raw aggTrades found for TE panel build")
    panel = pl.concat(rows, how="vertical_relaxed").sort(["asset", "date"])

    # Write cache (atomic-tmp-rename pattern)
    try:
        _DAILY_RETURNS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DAILY_RETURNS_CACHE.with_suffix(".parquet.tmp")
        panel.write_parquet(tmp)
        if _DAILY_RETURNS_CACHE.exists():
            _DAILY_RETURNS_CACHE.unlink()
        os.replace(str(tmp), str(_DAILY_RETURNS_CACHE))  # atomic overwrite (Windows-safe)
        print(f"[te] cached daily-returns panel: {panel.height} rows -> "
              f"{_DAILY_RETURNS_CACHE.name}", flush=True)
    except Exception as e:
        print(f"[te] WARN: cache write failed ({e}); continuing without cache", flush=True)

    return panel


def _compute_te_window(returns: dict,
                        leader_subset: list | None = None) -> tuple[np.ndarray, list]:
    """TE matrix M[i,j] = TE(names[i] -> names[j]) over the input window.

    Vectorized: bins each asset ONCE per window and uses np.bincount for the
    joint-count phase. Replaces the per-pair Python for-loop with batched
    numpy ops -- 50-100x faster on 90-day windows.

    If `leader_subset` is given, M[i,j] is computed only when at least one
    of (i, j) is in the subset. Off-subset pairs stay 0. This caps the
    pairwise max-over-leaders signal at O(K*N) rather than O(N^2) for
    leader_subset of size K -- 12-24x faster at N=48, K=10. BTC-anchored
    columns/rows (te_in_btc, te_out_btc) are always exact since BTC is
    always in the leader subset.
    """
    names = sorted(returns.keys())
    N = len(names)
    M = np.zeros((N, N))
    if N < 2:
        return M, names

    # Bin each asset ONCE (Schreiber/3-quantile, per-asset quantiles).
    binned: dict = {}
    for n in names:
        arr = returns[n]
        if not isinstance(arr, np.ndarray) or len(arr) < LAG + 50:
            continue
        binned[n] = bin_series(arr, N_BINS)

    leader_set = set(leader_subset) if leader_subset is not None else None
    for i, ni in enumerate(names):
        if ni not in binned:
            continue
        for j, nj in enumerate(names):
            if i == j or nj not in binned:
                continue
            # If leader_subset is given, skip pairs where neither end is a leader.
            if leader_set is not None and ni not in leader_set and nj not in leader_set:
                continue
            xb = binned[ni]
            yb = binned[nj]
            T = min(len(xb), len(yb))
            M[i, j] = _te_from_binned(xb[:T], yb[:T], lag=LAG, K=N_BINS)
    return M, names


def _aggregate_per_asset(M: np.ndarray, names: list) -> dict:
    """For each asset, compute te_in / te_out / te_in_btc / te_out_btc / imb."""
    # names hold the canonical no-USDT asset roots after the
    # _load_daily_returns_panel fix. Match against the BTC root.
    btc_idx = (names.index("BTC") if "BTC" in names else
                names.index("BTCUSDT") if "BTCUSDT" in names else None)
    out = {}
    for i, asset in enumerate(names):
        # Inflow: max TE FROM others into asset (excluding self)
        inflow = M[:, i].copy()
        inflow[i] = 0
        # Outflow: max TE FROM asset to others
        outflow = M[i, :].copy()
        outflow[i] = 0
        te_in = float(inflow.max()) if inflow.size > 1 else 0.0
        te_out = float(outflow.max()) if outflow.size > 1 else 0.0
        te_in_btc = float(M[btc_idx, i]) if btc_idx is not None and btc_idx != i else 0.0
        te_out_btc = float(M[i, btc_idx]) if btc_idx is not None and btc_idx != i else 0.0
        out[asset] = {
            "te_in": te_in,
            "te_out": te_out,
            "te_in_btc": te_in_btc,
            "te_out_btc": te_out_btc,
            "te_imb": te_in - te_out,
            "te_btc_imb": te_in_btc - te_out_btc,
        }
    return out


def build_panel(window_days: int = WINDOW_DAYS,
                stride_days: int = 7,
                leader_subset: list | None = None,
                min_anchor_date=None) -> pl.DataFrame:
    """Iterate rolling windows and emit one row per (date, asset).

    Args:
        min_anchor_date: if provided, skip windows whose anchor_date
            (last date of the window) is <= this. Used by the delta path
            to only build windows newer than what's in the existing panel.
            Per-window TE is independent of prior windows (each window
            recomputes the 90-day matrix from scratch), so skipping old
            windows is bit-identical to a full rebuild for those rows.
    """
    panel = _load_daily_returns_panel()
    print(f"[te] loaded panel: {panel.height} rows, "
          f"{panel.select('asset').n_unique()} assets")

    # Pivot to (date, asset -> return) wide form
    wide = panel.pivot(values="target_return_1", index="date", on="asset",
                       aggregate_function="first").sort("date")
    dates = wide["date"].to_list()
    asset_cols = [c for c in wide.columns if c != "date"]
    n = len(dates)
    print(f"[te] {n} dates, {len(asset_cols)} assets, "
          f"window={window_days}, stride={stride_days}, "
          f"min_anchor_date={min_anchor_date}")

    rows = []
    n_windows = max(0, (n - window_days) // stride_days)
    win_count = 0
    n_skipped = 0
    for end_idx in range(window_days, n, stride_days):
        start_idx = end_idx - window_days
        anchor_date = dates[end_idx - 1]
        # Delta-mode skip: anchor_date <= min_anchor_date means this window's
        # row is already in the existing panel.
        if min_anchor_date is not None and anchor_date <= min_anchor_date:
            n_skipped += 1
            continue
        window_df = wide[start_idx:end_idx]
        returns = {}
        for a in asset_cols:
            # @browser + frontier-final-state invariant: NEVER fill_null(0.0)
            # on returns -- corrupts std/binning and creates phantom TE flows
            # (same family as cross-venue fillna(0) provenance). If the asset
            # has ANY null in this window, exclude it from the TE matrix for
            # this window. With stride_days=7 and a 90-day window, an asset
            # missing one day still gets included on most adjacent windows.
            col = window_df[a]
            if col.null_count() > 0:
                continue
            arr = col.to_numpy().astype(np.float64)
            if np.std(arr) > 1e-9:
                returns[a] = arr
        if len(returns) < 2:
            continue
        M, names = _compute_te_window(returns, leader_subset=leader_subset)
        agg = _aggregate_per_asset(M, names)
        for asset, vals in agg.items():
            rows.append({"date": anchor_date, "asset": asset, **vals})
        win_count += 1
        if win_count % 20 == 0:
            pct = 100.0 * win_count / max(n_windows, 1)
            print(f"  [te {win_count}/{n_windows}] window end {anchor_date} "
                  f"({pct:.0f}% complete, {len(returns)} assets in flight)",
                  flush=True)

    if min_anchor_date is not None and n_skipped:
        print(f"[te] delta: skipped {n_skipped} windows already in existing "
              f"panel (anchor_date <= {min_anchor_date})", flush=True)

    if not rows:
        # In delta-mode this can be legitimate (no new windows past the
        # existing panel's max). Caller decides whether to surface fresh.
        if min_anchor_date is not None:
            return pl.DataFrame()
        raise RuntimeError("TE panel produced 0 rows -- check upstream data")

    df = pl.DataFrame(rows).sort(["asset", "date"])
    return df


def main() -> int:
    # Framework primitives.
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
    from parquet_io import (atomic_write_parquet, validate_existing,
                              append_panel_parquet)
    from cli import add_standard_args, resolve_assets

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(ap, default_workers=8, date_window=False)
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--stride-days", type=int, default=7,
                    help="Re-compute TE every N days (default: weekly)")
    ap.add_argument("--leader-universe",
                    default="u10",
                    choices=["u10", "u50", "u100", "all"],
                    help="Subset of assets used as leaders for the te_in/te_out "
                         "max-over-pairs signal (default u10). For an asset i NOT "
                         "in the leader set, te_in[i] = max over leader L of "
                         "TE(L->i), and te_out[i] = max over leader L of TE(i->L). "
                         "BTC-anchored (te_in_btc / te_out_btc / te_btc_imb) is "
                         "always computed exactly. Caps O(N^2) -> O(K*N) at N=48, "
                         "K=10 -> 12-24x speedup. 'all' = full N^2 matrix.")
    args = ap.parse_args()

    # Propagate --workers into the module-level knob used by the
    # per-asset daily-returns ThreadPool inside _load_daily_returns_panel.
    global _TE_N_WORKERS
    _TE_N_WORKERS = max(1, args.workers)

    global U10
    U10 = resolve_assets(args, default=list(DEFAULT_U10), stage_name="te_panel")
    if args.universe and len(U10) > 1:
        print(f"  TE matrix = {len(U10)}x{len(U10)} = {len(U10)**2} pairs", flush=True)

    # Resolve leader_subset: BTC-anchored is always exact; te_in/te_out's
    # max-over-pairs uses this subset of "leaders". Default u10 -> 10 leaders
    # which is the same set the strategy primitives (te_flow_tilt + meta-
    # learner) actually condition on; downstream signal is unchanged.
    leader_subset = None
    if args.leader_universe != "all":
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader
            # Inner TE matrix uses no-USDT asset roots (per
            # _load_daily_returns_panel fix). Strip USDT here so the
            # leader_set membership check matches.
            leader_subset = [s.upper().replace("USDT", "")
                              for s in UniverseLoader.load().list(args.leader_universe)]
            if "BTC" not in leader_subset:
                leader_subset.append("BTC")
            # U10 module-level holds full BTCUSDT-form symbols; compare
            # the no-USDT roots.
            u_roots = {u.replace("USDT", "") for u in U10}
            leader_subset = [a for a in leader_subset if a in u_roots]
        except Exception as e:
            print(f"[FALLBACK] leader-universe={args.leader_universe} load failed ({e}); using full N^2")
            leader_subset = None

    print(f"\n{'='*70}")
    print(f"BUILD TE PANEL  window={args.window_days}d  stride={args.stride_days}d  "
          f"|U|={len(U10)}")
    if leader_subset is not None:
        n_leader = len(leader_subset)
        n_full = len(U10)
        # Pairs computed: union of (leader, any) and (any, leader); subtract
        # leader-leader pairs counted twice; subtract self-pairs.
        approx_pairs = (2 * n_leader * n_full - n_leader * n_leader - n_leader)
        full_pairs = n_full * (n_full - 1)
        if n_leader >= n_full or approx_pairs >= full_pairs:
            print(f"  leader_subset={args.leader_universe} ({n_leader} leaders) >= |U|={n_full}; "
                  f"computing all {full_pairs} pairs (no speedup)")
        else:
            speedup = full_pairs / max(approx_pairs, 1)
            print(f"  leader_subset={args.leader_universe} ({n_leader} leaders); "
                  f"~{approx_pairs} TE pairs/window vs {full_pairs} full ({speedup:.1f}x faster)")
    else:
        print(f"  leader_subset=ALL (full {len(U10)*(len(U10)-1)} pairs/window -- slowest)")
    print(f"{'='*70}\n")

    # Delta path: find latest existing TE panel; if it's not corrupt,
    # only build windows past its max date.
    existing_panel_path = _layout._pick_latest(_layout.panels_dir(), PANEL_NAME)
    min_anchor_date = None
    do_delta_append = False
    if not args.force and existing_panel_path is not None:
        ok, why = validate_existing(
            existing_panel_path,
            required_cols={"date", "asset", "te_in", "te_out",
                            "te_in_btc", "te_out_btc", "te_imb", "te_btc_imb"},
            max_null_rate={"te_in": 0.10, "te_out": 0.10})
        if not ok:
            print(f"[te] CORRUPT existing panel {existing_panel_path.name}: "
                  f"{why}; full rebuild", flush=True)
        else:
            try:
                ex_df = pl.read_parquet(existing_panel_path, columns=["date"])
                ex_max = ex_df["date"].max()
                if isinstance(ex_max, str):
                    ex_max = date.fromisoformat(ex_max[:10])
                elif hasattr(ex_max, "date"):
                    ex_max = ex_max.date() if not isinstance(ex_max, date) else ex_max
                if isinstance(ex_max, date):
                    min_anchor_date = ex_max
                    do_delta_append = True
                    print(f"[te] delta: existing panel max={ex_max}; "
                          f"building only windows past this", flush=True)
            except Exception as e:
                print(f"[te] could not read existing max date "
                      f"({type(e).__name__}: {e}); full rebuild", flush=True)

    df = build_panel(window_days=args.window_days, stride_days=args.stride_days,
                      leader_subset=leader_subset,
                      min_anchor_date=min_anchor_date)

    # Delta-fresh: nothing new past existing max -> exit cleanly.
    if do_delta_append and (df is None or len(df) == 0):
        print(f"[te] no new windows past {min_anchor_date}; existing panel "
              f"is fresh", flush=True)
        return 0

    # Atomic write OR delta append, whichever applies.
    d_max = df["date"].max()
    if not isinstance(d_max, date):
        d_max = datetime.now(timezone.utc).date()

    if do_delta_append and existing_panel_path is not None:
        # Delta append: write atomically to a NEW dated path so older readers
        # can pick it up via _pick_latest. Combine existing + new before write.
        ex = pl.read_parquet(existing_panel_path)
        # Drop overlapping (asset, date) keys from existing, then concat.
        new_keys = df.select(["asset", "date"]).unique()
        ex_keep = ex.join(new_keys, on=["asset", "date"], how="anti")
        union = pl.concat([ex_keep, df], how="vertical_relaxed").sort(["asset", "date"])
        out_path = _layout.panels_dir() / f"{PANEL_NAME}_{d_max.strftime('%Y%m%d')}.parquet"
        atomic_write_parquet(
            union, out_path,
            required_cols={"date", "asset", "te_in", "te_out",
                            "te_in_btc", "te_out_btc", "te_imb", "te_btc_imb"})
        _layout.gc_older_dated(_layout.panels_dir(), PANEL_NAME)
        print(f"[te] delta-append: {df.height} new rows, {union.height} total",
              flush=True)
        return 0

    out_path = _layout.panels_dir() / f"{PANEL_NAME}_{d_max.strftime('%Y%m%d')}.parquet"
    atomic_write_parquet(
        df, out_path,
        required_cols={"date", "asset", "te_in", "te_out",
                        "te_in_btc", "te_out_btc", "te_imb", "te_btc_imb"})
    _layout.gc_older_dated(_layout.panels_dir(), PANEL_NAME)

    print(f"\n[OK] Wrote {out_path.name}: {df.height} rows, "
          f"{df.select('asset').n_unique()} assets, "
          f"date range {df['date'].min()} -> {df['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
