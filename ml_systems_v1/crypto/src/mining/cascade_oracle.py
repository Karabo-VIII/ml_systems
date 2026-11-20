"""Sub-bar liquidation-cascade ORACLE study -- the first honest test of the only
un-refuted Fork-B avenue (D67: "the ONLY fork avenue that could help is LEADING DATA").

r2 (2026-06-10): full RED-team rework after the adversarial audit (1 CRITICAL, 7 HIGH
confirmed). Changes vs r1, all applied BEFORE any pooled multi-asset result was read:
  C1  Suffix-max oracle gross scales with POST-event vol (cascades elevate it), so an
      event-vs-null oracle win is NOT evidence of edge. PASS estimands are now the
      vol-robust ones: fixed-horizon DRIFT (mean, not max: r60/r240/r1440) and the
      LONG-MINUS-SHORT oracle ASYMMETRY (pure vol cancels; a bounce edge survives).
      The oracle-gross contrast is retained as a KILL-ONLY criterion.
  H1  btc_coincident is now CAUSAL (BTC event at-or-before the alt trigger, <=30m);
      None (excluded) for BTC itself; built from TRAIN/OOS BTC events only.
  H2  Null pools + vol-matching quantiles are PER SPLIT; UNSEEN minutes are excluded
      from all pools (no UNSEEN price path is ever priced into TRAIN/OOS stats).
  H3  UNSEEN events are stored as count-stubs only (sym, t0, split) -- no oracle, no
      DNA, nothing for a later instance to spend.
  M*  Disjoint event windows (cooldown >= horizon + entry_max + 2); 48h-cluster
      bootstrap (serial + cross-asset simultaneity); per-asset breadth sign-test as
      the headline gate; split-boundary purge (event + null windows may not cross a
      split edge); real-data coverage gates (trigger real, >=95% fwd, >=90% trailing
      30d) so data gaps can't manufacture events or stale-price nulls; NULL-A screens
      out quasi-events (ratio>=5); NULL-B pool widened to 0.6x drop with a relative
      caliper and match-dropout reporting; flow profile renamed flow_posthoc
      (hindsight-descriptive, NEVER a Phase C conditioner); ci_level labeled 0.90.

WHAT THIS MEASURES:
For every detected long-liquidation cascade EVENT (event-clock, 1m resolution):
  NULL-A (vol-matched random, same split): is there more forward drift / directional
          asymmetry after a cascade than at an ordinary minute of similar PRE-vol?
  NULL-B (same 60m price-drop, NO abnormal liq flow, same split): does the
          LIQUIDATION signature carry information BEYOND the drop itself?  <-- Fork-B.
If events do not beat the nulls on the DRIFT/ASYMMETRY estimands, no causal rule can
work and the sub-bar cascade cell joins the dead-list. Decision discipline:
  KILL  if drift+asym contrasts <= 0 (or oracle-gross contrast <= 0, which is even
        stronger since the max-statistic is biased UP for events).
  PASS  requires drift/asym contrasts > 0 on TRAIN AND OOS with breadth (>=6/10
        assets positive) -- never the oracle-gross contrast alone.

Splits: TRAIN 2021-01-01..2023-12-31 | OOS 2024-01-01..2025-06-30 | UNSEEN
2025-07-01..2026-05-28 (counts only; spent ONCE by the Phase C causal rule).
Costs: 0/6/12/24 bps RT (maker 6bps OPTIMISTIC: real p_fill 0.21-0.40 per D43).
Seeded; git lineage recorded. Survivorship caveat: u10 = CURRENT membership.

Run:
  python -m mining.cascade_oracle --universe u10
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SUBBAR = ROOT / "data" / "processed" / "liq_subbar"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {
        "liq_subbar_1m": "data/processed/liq_subbar/<SYM>_1m.parquet",
        "metrics_5m": "data/raw/<SYM>/metrics/*.parquet",
        "funding_8h": "data/raw/<SYM>/funding/*.parquet",
    },
    "outputs": {"study_json": "runs/mining/cascade_oracle_<tag>_<stamp>.json"},
    "invariants": {
        "causal_trigger": "event trigger uses data <= trigger minute; fills next-minute open",
        "baseline_strictly_prior": "30d flow baseline shift(1) before rolling window",
        "unseen_once": "UNSEEN: count-stubs only in JSON and console; no oracle/DNA computed",
        "null_pools_per_split": "null candidates + vol bins share the event's split; UNSEEN never pooled",
        "pass_estimand_vol_robust": "PASS gates on drift/asymmetry, never the max-statistic ceiling",
        "seeded_nulls": "numpy RNG seeded; seed recorded in output",
    },
}

MIN_MS, MAX_MS = int(1.5e12), int(2.0e12)
COSTS_RT = [0.0, 0.0006, 0.0012, 0.0024]
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
SPLIT_NAMES = ["TRAIN", "OOS", "UNSEEN"]
EST_KEYS = ["gross", "asym", "r60", "r240", "r1440"]
CLUSTER_MS = 48 * 3_600_000  # bootstrap cluster width (covers horizon+entry window)


def split_idx_of(ms: int) -> int:
    return 0 if ms < TRAIN_END_MS else (1 if ms < OOS_END_MS else 2)


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def _read_many(pattern: str) -> pl.DataFrame | None:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files]).sort("timestamp")


def _norm_ts_ms(df: pl.DataFrame, col: str = "timestamp") -> pl.DataFrame:
    mx = df[col].max()
    if mx is not None and mx >= MAX_MS:
        div = 1_000 if mx < 2e15 else 1_000_000
        df = df.with_columns((pl.col(col) // div).alias(col))
    return df


# --------------------------------------------------------------------- panel build

def load_panel(sym: str) -> pl.DataFrame:
    """1m grid panel: price/flows + causal rolling features + OI/LSR/funding as-of."""
    p = SUBBAR / f"{sym}_1m.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run mining.liq_subbar first")
    df = pl.read_parquet(p).sort("minute_ts")

    t0, t1 = int(df["minute_ts"].min()), int(df["minute_ts"].max())
    grid = pl.DataFrame({"minute_ts": np.arange(t0, t1 + 60_000, 60_000, dtype=np.int64)})
    df = grid.join(df, on="minute_ts", how="left", maintain_order="left")
    assert df["minute_ts"].is_sorted(), "grid join lost row order"
    df = df.with_columns(pl.col("n_trades").is_not_null().alias("is_real"))
    df = df.with_columns(pl.col("close").forward_fill())
    df = df.with_columns([
        pl.col("open").fill_null(pl.col("close")),
        pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")),
        pl.col("liq_long_usd").fill_null(0.0),
        pl.col("liq_short_usd").fill_null(0.0),
        pl.col("liq_long_cnt").fill_null(0),
        pl.col("vol_usd").fill_null(0.0),
    ])

    W30, D30 = 30, 43_200
    df = df.with_columns([
        pl.col("liq_long_usd").rolling_sum(W30, min_samples=W30).alias("liq30"),
        pl.col("liq_long_cnt").rolling_sum(W30, min_samples=W30).alias("liq_cnt30"),
    ])
    df = df.with_columns(
        pl.col("liq30").shift(1).rolling_mean(D30, min_samples=D30 // 2).alias("liq30_base"),
    )
    df = df.with_columns([
        (pl.col("liq30") / (pl.col("liq30_base") + 1.0)).alias("liq_ratio"),
        (pl.col("close") / pl.col("close").shift(60) - 1.0).alias("ret_60m"),
        pl.col("close").pct_change().alias("ret_1m"),
    ])
    # pre-event vol: std of 1m returns over [t-25h, t-1h] (excludes the cascade hour)
    df = df.with_columns(
        pl.col("ret_1m").shift(60).rolling_std(1440, min_samples=720).alias("pre_vol"),
    )
    # trailing 30d real-data coverage (data gaps must not depress flow baselines silently)
    df = df.with_columns(
        pl.col("is_real").cast(pl.Float64).shift(1).rolling_mean(D30, min_samples=D30 // 2)
        .alias("trail_cov"),
    )

    # daily SMA200 regime, causal (yesterday's state)
    daily = (
        df.with_columns(((pl.col("minute_ts") // 86_400_000) * 86_400_000).alias("day_ts"))
        .group_by("day_ts").agg(pl.col("close").last().alias("day_close")).sort("day_ts")
        .with_columns(
            (pl.col("day_close") > pl.col("day_close").rolling_mean(200, min_samples=200))
            .shift(1).alias("regime_above_sma200"))
    )
    df = df.with_columns(((pl.col("minute_ts") // 86_400_000) * 86_400_000).alias("day_ts"))
    df = df.join(daily.select(["day_ts", "regime_above_sma200"]), on="day_ts",
                 how="left", maintain_order="left")
    assert df["minute_ts"].is_sorted(), "regime join lost row order"

    met = _read_many(str(RAW / sym / "metrics" / "*.parquet"))
    if met is not None:
        met = _norm_ts_ms(met).rename({"timestamp": "minute_ts"}).sort("minute_ts")
        df = df.join_asof(met, on="minute_ts", strategy="backward")
        df = df.with_columns([
            (pl.col("open_interest_val") / pl.col("open_interest_val").shift(60) - 1).alias("oi_d1h"),
            (pl.col("open_interest_val") / pl.col("open_interest_val").shift(240) - 1).alias("oi_d4h"),
            (pl.col("open_interest_val") / pl.col("open_interest_val").shift(1440) - 1).alias("oi_d24h"),
            (pl.col("long_short_ratio") - pl.col("long_short_ratio").shift(1440)).alias("lsr_d24h"),
        ])
    else:
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in
                              ["open_interest_val", "long_short_ratio", "oi_d1h", "oi_d4h",
                               "oi_d24h", "lsr_d24h"]])
    fnd = _read_many(str(RAW / sym / "funding" / "*.parquet"))
    if fnd is not None:
        fnd = _norm_ts_ms(fnd).rename({"timestamp": "minute_ts", "funding_rate": "funding"}).sort("minute_ts")
        df = df.join_asof(fnd, on="minute_ts", strategy="backward")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding"))
    return df


# --------------------------------------------------------------------- estimands

def estimands(opens: np.ndarray, i: int, entry_max: int, horizon: int) -> dict | None:
    """Per-window estimands. Fills at next-minute open throughout.

    w[j] = open of minute i+1+j, j in [0, horizon].
    LONG oracle:  entry decision close of i+e (e<=entry_max) -> fill w[e];
                  exit decision close of i+x (x>e) -> fill w[x] with x>=e+1 -> sm_max[e+1].
    SHORT oracle: identical geometry on the suffix-min (vol-symmetric control).
    DRIFT r_h:    w[h]/w[0] - 1 (mean-class statistic, robust to vol inflation).
    """
    w = opens[i + 1: i + horizon + 2]
    if len(w) < horizon + 1 or not np.all(np.isfinite(w)):
        return None
    sm_max = np.maximum.accumulate(w[::-1])[::-1]
    sm_min = np.minimum.accumulate(w[::-1])[::-1]
    es = np.arange(0, entry_max + 1)
    long_gross = sm_max[es + 1] / w[es] - 1.0
    short_gross = w[es] / sm_min[es + 1] - 1.0
    e_star = int(np.argmax(long_gross))
    g = float(long_gross[e_star])
    lo_j = int(np.argmin(w))
    return {
        "gross": g,
        "short_gross": float(np.max(short_gross)),
        "asym": float(g - np.max(short_gross)),
        "r60": float(w[60] / w[0] - 1.0),
        "r240": float(w[240] / w[0] - 1.0),
        "r1440": float(w[horizon] / w[0] - 1.0),
        "entry_offset_min": e_star,
        "net": {f"{round(c*1e4)}bps": float(g - c) for c in COSTS_RT},
        "spike_buy_gross": float(sm_max[1] / w[0] - 1.0),
        "t_low_min": lo_j + 1,
        "low_depth": float(w[lo_j] / w[0] - 1.0),
    }


def flow_decay_posthoc(liq_long: np.ndarray, i: int) -> dict:
    """HINDSIGHT-ONLY post-trigger 10m-flow shape (descriptive; NEVER a Phase C
    conditioner -- it reads 6h of future flow)."""
    seg = liq_long[i: i + 361]
    if len(seg) < 361:
        return {"peak_offset_min": None, "half_decay_min": None}
    f10 = np.convolve(seg, np.ones(10), mode="valid")
    pk = int(np.argmax(f10))
    half = None
    for j in range(pk + 1, len(f10)):
        if f10[j] < 0.5 * f10[pk]:
            half = j
            break
    return {"peak_offset_min": pk, "half_decay_min": half}


# --------------------------------------------------------------------- detection

def detect_events(df: pl.DataFrame, args, split_arr: np.ndarray,
                  fwd_cov: np.ndarray, real: np.ndarray) -> list[int]:
    """Row indices of event triggers (causal, disjoint windows, coverage-gated,
    no split-boundary crossing)."""
    ratio = df["liq_ratio"].to_numpy()
    ret60 = df["ret_60m"].to_numpy()
    base = df["liq30_base"].to_numpy()
    cnt30 = df["liq_cnt30"].to_numpy()
    trail = df["trail_cov"].to_numpy()
    n = len(df)
    H = args.horizon_min
    cand = np.flatnonzero(
        (ratio >= args.r_mult) & (ret60 <= -args.drop_pct / 100.0)
        & np.isfinite(base) & (base > 0) & (cnt30 >= 5)
        & (real == 1) & (fwd_cov >= 0.95) & np.isfinite(trail) & (trail >= 0.90)
    )
    events, last = [], -10**18
    for i in cand:
        if i - last < args.cooldown_min:
            continue
        if i + H + 2 >= n or i < 1500:
            continue
        if split_arr[i] != split_arr[i + H + 1]:  # boundary purge
            continue
        events.append(int(i))
        last = i
    return events


def study_asset(sym: str, args, btc_events_ms: list[int] | None,
                rng: np.random.Generator) -> dict:
    t_start = time.time()
    df = load_panel(sym)
    opens = df["open"].to_numpy()
    liq_long = df["liq_long_usd"].to_numpy()
    ms = df["minute_ts"].to_numpy()
    n = len(df)
    H = args.horizon_min

    split_arr = np.full(n, 2, dtype=np.int8)
    split_arr[ms < OOS_END_MS] = 1
    split_arr[ms < TRAIN_END_MS] = 0

    real = df["is_real"].to_numpy().astype(np.int8)
    cum = np.concatenate([[0], np.cumsum(real)])
    # forward real coverage over fill window minutes i+1 .. i+H+1
    fwd_cov = np.full(n, 0.0)
    upper = np.minimum(np.arange(n) + H + 2, n)
    lower = np.minimum(np.arange(n) + 1, n)
    fwd_cov = (cum[upper] - cum[lower]) / float(H + 1)

    ev_idx = detect_events(df, args, split_arr, fwd_cov, real)

    ratio = df["liq_ratio"].to_numpy()
    ret60 = df["ret_60m"].to_numpy()
    base = df["liq30_base"].to_numpy()
    pre_vol = df["pre_vol"].to_numpy()
    trail = df["trail_cov"].to_numpy()

    idx = np.arange(n)
    valid = (idx > 1500) & (idx + H + 2 < n) & np.isfinite(base) & (base > 0) \
        & np.isfinite(pre_vol) & (real == 1) & (fwd_cov >= 0.95) \
        & np.isfinite(trail) & (trail >= 0.90) \
        & (split_arr == split_arr[np.minimum(idx + H + 1, n - 1)]) \
        & (split_arr < 2)  # H2: UNSEEN minutes never enter any pool
    near_event = np.zeros(n, dtype=bool)
    for i in ev_idx:
        near_event[max(0, i - (H + 61)): i + H + 2] = True
    # NULL-A: ordinary minutes (no quasi-events: ratio < 5)
    poolA_all = valid & ~near_event & (ratio < 5.0)
    # NULL-B: comparable drop, no abnormal liq (pool widened to 0.6x drop; caliper matches)
    poolB_all = valid & ~near_event & (ret60 <= -0.6 * args.drop_pct / 100.0) & (ratio < 5.0)

    pools = {}
    for s in (0, 1):
        pa = np.flatnonzero(poolA_all & (split_arr == s))
        pb = np.flatnonzero(poolB_all & (split_arr == s))
        vq = (np.nanquantile(pre_vol[pa], [0.2, 0.4, 0.6, 0.8]) if len(pa) >= 100
              else np.array([np.inf] * 4))
        pa_buckets = np.searchsorted(vq, pre_vol[pa]) if len(pa) else np.array([], dtype=int)
        pools[s] = {"A": pa, "A_buckets": pa_buckets, "B": pb, "vq": vq}

    def null_means(idx_list: list[int]) -> dict | None:
        if not idx_list:
            return None
        vals = {k: [] for k in EST_KEYS}
        for j in idx_list:
            o = estimands(opens, int(j), args.entry_max_min, H)
            if o:
                for k in EST_KEYS:
                    vals[k].append(o[k])
        if not vals["gross"]:
            return None
        return {k: float(np.mean(v)) for k, v in vals.items()} | {"n": len(vals["gross"])}

    events = []
    n_unseen = 0
    for i in ev_idx:
        ev_ms = int(ms[i])
        s = int(split_arr[i])
        if s == 2:  # H3: UNSEEN -> count-stub only
            n_unseen += 1
            events.append({"sym": sym, "t0_ms": ev_ms, "split": "UNSEEN"})
            continue
        est = estimands(opens, i, args.entry_max_min, H)
        if est is None:
            continue
        row = df.row(i, named=True)
        P = pools[s]
        # NULL-A: vol-matched random minutes, same split
        nullA = None
        if len(P["A"]) >= 5:
            b = int(np.searchsorted(P["vq"], pre_vol[i])) if np.isfinite(pre_vol[i]) else 2
            candA = P["A"][P["A_buckets"] == b]
            if len(candA) >= 5:
                take = rng.choice(candA, size=min(args.k_nulls, len(candA)), replace=False)
                nullA = null_means([int(x) for x in take])
        # NULL-B: drop-matched no-liq minutes, same split, relative caliper
        nullB = None
        if len(P["B"]) >= 5:
            caliper = max(0.0075, 0.2 * abs(ret60[i]))
            dmatch = P["B"][np.abs(ret60[P["B"]] - ret60[i]) <= caliper]
            if len(dmatch) >= 5:
                take = rng.choice(dmatch, size=min(args.k_nulls, len(dmatch)), replace=False)
                nullB = null_means([int(x) for x in take])
        # H1: causal BTC coincidence (BTC cascade at-or-before alt trigger, <=30m);
        # None for BTC itself (degenerate) and when BTC events unknown.
        if sym == "BTCUSDT" or btc_events_ms is None:
            btc_co = None
        else:
            btc_co = any(0 <= ev_ms - b_ <= 1_800_000 for b_ in btc_events_ms)
        events.append({
            "sym": sym, "t0_ms": ev_ms,
            "t0_iso": dt.datetime.fromtimestamp(ev_ms / 1000, dt.timezone.utc).isoformat(),
            "split": SPLIT_NAMES[s],
            "trigger_ratio": float(ratio[i]), "drop_60m": float(ret60[i]),
            "pre_vol": float(pre_vol[i]) if np.isfinite(pre_vol[i]) else None,
            "est": est,
            "flow_posthoc": flow_decay_posthoc(liq_long, i),
            "nullA": nullA, "nullB": nullB,
            "dna": {
                "oi_d1h": row.get("oi_d1h"), "oi_d4h": row.get("oi_d4h"),
                "oi_d24h": row.get("oi_d24h"), "lsr": row.get("long_short_ratio"),
                "lsr_d24h": row.get("lsr_d24h"), "funding": row.get("funding"),
                "regime_above_sma200": row.get("regime_above_sma200"),
                "btc_coincident": btc_co,
                "liq30_usd": float(df["liq30"][i]) if df["liq30"][i] is not None else None,
            },
        })
    n_tr = sum(1 for e in events if e["split"] == "TRAIN")
    n_oos = sum(1 for e in events if e["split"] == "OOS")
    print(f"[{sym}] events T/O/U={n_tr}/{n_oos}/{n_unseen} "
          f"poolA(T/O)={len(pools[0]['A'])}/{len(pools[1]['A'])} "
          f"poolB(T/O)={len(pools[0]['B'])}/{len(pools[1]['B'])} ({time.time()-t_start:.0f}s)")
    return {"sym": sym, "events": events}


# --------------------------------------------------------------------- aggregation

def paired_deltas(events: list[dict], null_key: str, est: str):
    """(delta, cluster_id) pairs for events with a usable null."""
    out = []
    for e in events:
        nl = e.get(null_key)
        if nl is None or "est" not in e:
            continue
        out.append((e["est"][est] - nl[est], e["t0_ms"] // CLUSTER_MS))
    return out


def cluster_contrast(events: list[dict], null_key: str, est: str) -> dict:
    """Mean paired delta with a 48h-CLUSTER bootstrap (serial overlap + cross-asset
    simultaneity are absorbed into clusters). ci_level=0.90 ([p05, p95])."""
    pairs = paired_deltas(events, null_key, est)
    if len(pairs) < 5:
        return {"n": len(pairs)}
    d = np.array([p[0] for p in pairs])
    cl = np.array([p[1] for p in pairs])
    uniq = np.unique(cl)
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(10_000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([d[cl == c] for c in pick])
        boots.append(sel.mean())
    boots = np.array(boots)
    return {
        "n": len(d), "n_clusters": int(len(uniq)), "ci_level": 0.90,
        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
        "p05": float(np.quantile(boots, 0.05)), "p95": float(np.quantile(boots, 0.95)),
        "frac_positive": float(np.mean(d > 0)),
    }


def breadth(events: list[dict], null_key: str, est: str, min_n: int = 3) -> dict:
    """Per-asset mean delta sign test -- the headline robustness gate."""
    per = {}
    for e in events:
        nl = e.get(null_key)
        if nl is None or "est" not in e:
            continue
        per.setdefault(e["sym"], []).append(e["est"][est] - nl[est])
    rows = {s: {"n": len(v), "mean_delta": float(np.mean(v))}
            for s, v in per.items() if len(v) >= min_n}
    pos = sum(1 for r in rows.values() if r["mean_delta"] > 0)
    return {"per_asset": rows, "assets_positive": pos, "assets_total": len(rows)}


def summarize_split(events: list[dict], split: str) -> dict:
    evs = [e for e in events if e["split"] == split and "est" in e]
    if split == "UNSEEN":
        return {"n_events": sum(1 for e in events if e["split"] == "UNSEEN"),
                "note": "UNSEEN: count only (spent once in Phase C)"}
    if not evs:
        return {"n_events": 0}
    g = np.array([e["est"]["gross"] for e in evs])
    out = {
        "n_events": len(evs),
        "n_nullB_unmatched": sum(1 for e in evs if e.get("nullB") is None),
        "drop_mean_all": float(np.mean([e["drop_60m"] for e in evs])),
        "drop_mean_nullB_matched": (float(np.mean([e["drop_60m"] for e in evs
                                                   if e.get("nullB") is not None]))
                                    if any(e.get("nullB") for e in evs) else None),
        "oracle_gross": {"mean": float(g.mean()), "median": float(np.median(g)),
                         "p25": float(np.quantile(g, 0.25)), "p75": float(np.quantile(g, 0.75))},
        "oracle_net_mean": {k: float(np.mean([e["est"]["net"][k] for e in evs]))
                            for k in evs[0]["est"]["net"]},
        "frac_net_pos": {k: float(np.mean([e["est"]["net"][k] > 0 for e in evs]))
                         for k in evs[0]["est"]["net"]},
        "drift_mean": {h: float(np.mean([e["est"][h] for e in evs]))
                       for h in ["r60", "r240", "r1440"]},
        "asym_mean": float(np.mean([e["est"]["asym"] for e in evs])),
        "entry_offset_min_median": float(np.median([e["est"]["entry_offset_min"] for e in evs])),
        "t_low_min_median": float(np.median([e["est"]["t_low_min"] for e in evs])),
        "low_depth_mean": float(np.mean([e["est"]["low_depth"] for e in evs])),
        "contrasts": {},
        "breadth": {},
    }
    for nk in ["nullA", "nullB"]:
        for est in ["gross", "asym", "r240", "r1440"]:
            out["contrasts"][f"{est}_vs_{nk}"] = cluster_contrast(evs, nk, est)
        out["breadth"][f"asym_vs_{nk}"] = breadth(evs, nk, "asym")
        out["breadth"][f"r1440_vs_{nk}"] = breadth(evs, nk, "r1440")
    return out


def dna_correlations(events: list[dict], split: str) -> dict:
    """Spearman rho of strictly-prior features vs the DRIFT estimand (r1440) and the
    ceiling (gross), per split. Conditioning candidates for Phase C come from r1440."""
    from scipy.stats import spearmanr
    evs = [e for e in events if e["split"] == split and "est" in e]
    feats = ["oi_d1h", "oi_d4h", "oi_d24h", "lsr", "lsr_d24h", "funding", "liq30_usd"]
    out = {}
    for target in ["r1440", "gross"]:
        y = np.array([e["est"][target] for e in evs])
        t_out = {}
        for f in feats:
            x = np.array([e["dna"][f] if e["dna"][f] is not None else np.nan for e in evs],
                         dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 10 and np.nanstd(x[m]) > 0:
                rho, p = spearmanr(x[m], y[m])
                t_out[f] = {"rho": float(rho), "p": float(p), "n": int(m.sum())}
        for f in ["regime_above_sma200", "btc_coincident"]:
            grp1 = [e["est"][target] for e in evs if e["dna"][f] is True]
            grp0 = [e["est"][target] for e in evs if e["dna"][f] is False]
            if len(grp1) >= 5 and len(grp0) >= 5:
                t_out[f] = {"mean_true": float(np.mean(grp1)), "mean_false": float(np.mean(grp0)),
                            "n_true": len(grp1), "n_false": len(grp0)}
        out[target] = t_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sub-bar liq-cascade oracle study (r2)")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--r-mult", type=float, default=25.0)
    ap.add_argument("--drop-pct", type=float, default=1.5)
    ap.add_argument("--cooldown-min", type=int, default=1802,
                    help=">= horizon + entry_max + 2 so event windows are disjoint")
    ap.add_argument("--horizon-min", type=int, default=1440)
    ap.add_argument("--entry-max-min", type=int, default=360)
    ap.add_argument("--k-nulls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="u10")
    args = ap.parse_args()
    assert args.cooldown_min >= args.horizon_min + args.entry_max_min + 2, \
        "cooldown must keep event windows disjoint (audit M1)"

    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        args.tag = args.universe
    else:
        ap.error("provide --assets or --universe")
    syms = sorted(set(syms), key=lambda s: (s != "BTCUSDT", s))

    rng = np.random.default_rng(args.seed)
    all_events: list[dict] = []
    per_asset = []
    btc_events_ms: list[int] | None = None
    for sym in syms:
        try:
            res = study_asset(sym, args, btc_events_ms, rng)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        per_asset.append({"sym": sym, "n_events": len(res["events"])})
        all_events.extend(res["events"])
        if sym == "BTCUSDT":
            # TRAIN/OOS triggers only (UNSEEN stubs excluded -> no UNSEEN bleed)
            btc_events_ms = [e["t0_ms"] for e in res["events"] if e["split"] != "UNSEEN"]

    summary = {
        "TRAIN": summarize_split(all_events, "TRAIN"),
        "OOS": summarize_split(all_events, "OOS"),
        "UNSEEN": summarize_split(all_events, "UNSEEN"),
        "dna_TRAIN": dna_correlations(all_events, "TRAIN"),
        "dna_OOS": dna_correlations(all_events, "OOS"),
    }

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "cascade_oracle_r2", "git_sha": sha, "seed": args.seed,
        "params": vars(args),
        "config_registry": {
            "preregistered": "r_mult=25, drop=1.5pct, horizon=1440m, entry_max=360m, "
                             "cnt30>=5 chosen a priori (r1); cooldown 720->1802 changed by "
                             "the RED-team audit BEFORE any pooled multi-asset result was read",
            "configs_run_at_these_params": 1,
        },
        "caveats": [
            "u10 = CURRENT universe membership: absolute ceilings are survivorship-"
            "biased; inference rests on within-asset matched contrasts only",
            "maker 6bps RT assumes fills; real p_fill 0.21-0.40 (D43)",
            "fixed-USD liq thresholds are non-stationary across 2021-2026; ratio "
            "baseline absorbs rate drift, not composition drift",
            "PASS requires drift/asym contrasts (vol-robust), never oracle-gross "
            "(max-statistic inflates with post-event vol; kill-only)",
        ],
        "splits": {"train": "<2024-01-01", "oos": "<2025-07-01", "unseen": ">=2025-07-01"},
        "costs_rt": COSTS_RT, "per_asset": per_asset,
        "summary": summary,
        "events": all_events,  # UNSEEN entries are count-stubs (sym, t0_ms, split) only
    }
    out_path = OUT / f"cascade_oracle_{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---------------- console story
    print("\n" + "=" * 78)
    print("SUB-BAR LIQUIDATION-CASCADE ORACLE (r2, vol-robust estimands) -- STORY")
    print("=" * 78)
    for sp in ["TRAIN", "OOS"]:
        s = summary[sp]
        if s.get("n_events", 0) == 0:
            print(f"\n[{sp}] no events")
            continue
        print(f"\n[{sp}] n={s['n_events']} events (pooled {args.tag}); "
              f"nullB unmatched: {s['n_nullB_unmatched']} "
              f"(drop all {s['drop_mean_all']*100:+.2f}% vs matched "
              f"{(s['drop_mean_nullB_matched'] or 0)*100:+.2f}%)")
        print(f"  DESCRIPTIVE ceiling (kill-only): gross mean {s['oracle_gross']['mean']*100:+.2f}% "
              f"median {s['oracle_gross']['median']*100:+.2f}%; "
              f"net-pos frac " + " ".join(f"{k}:{v*100:.0f}%" for k, v in s["frac_net_pos"].items()))
        print(f"  Timing: optimal entry median {s['entry_offset_min_median']:.0f}m after trigger; "
              f"low at median {s['t_low_min_median']:.0f}m; dip {s['low_depth_mean']*100:+.2f}%")
        print(f"  PASS estimands: drift r240 {s['drift_mean']['r240']*100:+.3f}%  "
              f"r1440 {s['drift_mean']['r1440']*100:+.3f}%  asym {s['asym_mean']*100:+.3f}%")
        for label, key in [("NULL-A(vol-matched)", "nullA"), ("NULL-B(drop,no-liq)", "nullB")]:
            for est in ["asym", "r1440", "gross"]:
                c = s["contrasts"][f"{est}_vs_{key}"]
                if "mean_delta" in c:
                    tagk = "KILL-ONLY" if est == "gross" else "PASS-grade"
                    print(f"    {label:<20} {est:<6} delta {c['mean_delta']*100:+.3f}pp "
                          f"[90% CI {c['p05']*100:+.3f},{c['p95']*100:+.3f}] "
                          f"frac>0 {c['frac_positive']*100:.0f}% n={c['n']}/cl={c['n_clusters']} ({tagk})")
        for key in ["nullA", "nullB"]:
            b = s["breadth"][f"asym_vs_{key}"]
            print(f"    breadth asym_vs_{key}: {b['assets_positive']}/{b['assets_total']} assets positive")
    print(f"\n[UNSEEN] n={summary['UNSEEN']['n_events']} events DETECTED (count-stub only)")
    print("\nDNA vs DRIFT r1440 (Phase C conditioning candidates), TRAIN -> OOS:")
    for f in sorted(set(summary["dna_TRAIN"].get("r1440", {})) | set(summary["dna_OOS"].get("r1440", {}))):
        a = summary["dna_TRAIN"]["r1440"].get(f)
        b = summary["dna_OOS"]["r1440"].get(f)
        def _fmt(x):
            if not x:
                return "--"
            if "rho" in x:
                return f"rho {x['rho']:+.2f} (p {x['p']:.3f}, n {x['n']})"
            return f"T {x['mean_true']*100:+.2f}% vs F {x['mean_false']*100:+.2f}%"
        print(f"  {f:<22} TRAIN {_fmt(a):<36} OOS {_fmt(b)}")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
