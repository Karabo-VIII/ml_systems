"""MOVER CAPTURE-RATE study -- the ASYMMETRIC-exit re-test of the D72/D61 null.

PROBLEM (decomposed mover problem #4): of a daily mover you are ALREADY IN
(entered at a +X% intraday confirmation trigger on a >=5%-runup day), what is the
maximum realizable CAPTURE-RATE = realized_exit_gain / available_MFE, and can ANY
exit get it net-positive, pushing toward the user's 50-80% target?

GROUNDING (trusted, from a just-completed characterization):
  (1) the daily mover move is a SINGLE-HOUR BURST (median ~51% of the daily net move
      lands in one hour; ~52% of mover-days are single-hour bursts);
  (2) post-move is REVERSAL in the MEDIAN but CONTINUATION in the FAT TAIL
      (mean positive, median negative -- a right skew);
  (3) D72: the post-trigger MEAT EXISTS at 1m (MFE ~+5.3-5.7% MEDIAN on >=5% runup days)
      but UNCONDITIONAL riding BLED (-35 to -88%/yr across 18 cells); D61: exit-timing
      null for daily breakouts. BOTH used SYMMETRIC exits.

FRESH ANGLE (why this is not a re-mine): the capture problem is ASYMMETRIC. Because the
move reverses in the median but runs in the tail, the right exit must CUT the typical
pullback fast AND let the rare tail-runner run. A symmetric trail/target is wrong by
CONSTRUCTION. We re-test capture as an asymmetric/convex EXIT optimization, hold the
ENTRY fixed (D72's trigger), and decompose how much of the bleed is exit-fixable.

DESIGN:
  ENTRY (fixed, = D72/mover_ride): first minute whose CLOSE >= day_open*(1+T) on a day,
    fill next-minute OPEN; one event per asset-day; day-bounded (no overnight). The
    trigger is UNCONDITIONAL (never knows the day is a mover). Only events with
    runup_day >= 5% are the "true mover" cohort, but we also report the all-trigger
    cohort to expose entry-quality bleed (false/too-late triggers).
  AVAILABLE MOVE (denominator): post-trigger MFE = (max close from fill..day-end)/entry-1.
    Hindsight; used ONLY for the capture-rate denominator, never by a causal exit.
  EXIT POLICIES (all strictly causal: decide at close of minute x, fill at open x+1;
    running stats use only [f..x]):
    (a) trail_pct      -- symmetric high-water % trail (the D72 baseline, for contrast)
    (b) chandelier     -- ATR-scaled trail: exit if close < runmax - m*ATR (Chandelier)
    (c) tp_frac_mfe    -- take-profit at a fixed % above entry (caps the tail; a control)
    (d) scaleout       -- ASYMMETRIC: bank HALF at +T1, ride the rest on a LOOSE trail
                          (the convex "cut median, ride tail" policy)
    (e) time_stop      -- exit N minutes after fill (the burst is ~1h; tests give-back)
    (f) vol_trail      -- volatility-scaled % trail: k * pre-event 1m vol (wider in
                          high-vol names so the trail is not noise-tripped)
    (g) breakeven_ride -- ASYMMETRIC convex: tight trail until +B locks profit, then
                          switch to a LOOSE trail (cut early reversers, ride survivors)
  NEUTRAL overlay: for the SELECTED policy, LONG the triggered mover and SHORT a
    matched hedge (BTC or equal-weight-universe proxy) over the identical hold -> tests
    whether the give-back is market-wide (hedge cleans it) and reports beta.

PRE-REGISTRATION (gates stated before any result is read):
  Selection: ON TRAIN ONLY -- per policy family, the parameter with the highest
    portfolio net %/yr at 24bps RT among the >=5% mover cohort with breadth >= 6/10.
    Exactly ONE (policy, param) advances to OOS.
  OOS "alive": portfolio net > 0 at 24bps AND breadth >= 6/10 AND survives
    drop-top-3-events jackknife. UNSEEN peeked exactly ONCE at the very end.
  Costs: net = gross - RT at {0,6,12,24}bps taker; ALSO report maker-best-case
    (entry as a limit during the burst, exit taker -> ~half the RT).
VERDICT printed: best realizable capture (per-move + aggregate net), whether ANY exit
  is net-positive, distance to 50-80%, and whether asymmetric/neutral cracks the null.

Run:
  python -m mining.mover_capture_rate --universe u10
No emoji (cp1252). Seeded. Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUBBAR = ROOT / "data" / "processed" / "liq_subbar"
OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"liq_subbar_1m": "data/processed/liq_subbar/<SYM>_1m.parquet"},
    "outputs": {"study_json": "runs/mining/mover_capture_<tag>_<stamp>.json"},
    "invariants": {
        "causal_exits": "decide at close of minute x -> fill open of x+1; running stats use only [f..x]",
        "available_move": "post-trigger MFE (hindsight) is the DENOMINATOR only, never used by a causal exit",
        "one_event_per_asset_day": "no re-entry; day-bounded exits",
        "train_selection": "ONE (policy,param) per family selected on TRAIN, advanced to OOS",
        "unseen_once": "UNSEEN peeked exactly once at the very end",
    },
}

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DATA_START_MS = int(dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
COSTS_RT = [0.0, 0.0006, 0.0012, 0.0024]
GATE_COST = 0.0024
MAKER_RT = 0.0006            # maker entry (limit during the burst) + taker exit ~ half taker RT
TRIGGER = 0.015             # +1.5% intraday confirmation -- D68 cadence-gradient sweet spot
MIN_MINUTES_LEFT = 30       # trigger must leave >=30m of day to ride
DAY_MS = 86_400_000
CLUSTER_MS = 7 * DAY_MS      # weekly clusters for the block bootstrap
MOVER_RUNUP = 0.05          # the >=5% "true mover" cohort
N_BOOT = 5000

# ----------------------------------------------------------------- exit policies
# Each returns a realized gross return (exit_px/entry - 1), strictly causal.
# Convention: decision at close of minute x in [f, end-1]; if a stop is breached at the
# close of x, exit fills at open[x+1] (x+1 <= end always, since MIN_MINUTES_LEFT >= 1).
# If never breached, exit at close[end] (market-on-close; identical across policies).


def _atr(highs, lows, closes, f, end, win=14):
    """Causal Wilder-ish ATR series aligned to [f..end] using true range; uses only past
    bars within the day. Returns an array atr[i] for i in [0, end-f] (atr at minute f+i,
    computed from TR up to and including f+i)."""
    n = end - f + 1
    h = highs[f:end + 1].astype(float)
    lo = lows[f:end + 1].astype(float)
    c = closes[f:end + 1].astype(float)
    prev_c = np.empty(n)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    # simple rolling mean ATR (causal): mean of last `win` TRs
    atr = np.empty(n)
    csum = np.cumsum(tr)
    for i in range(n):
        a = max(0, i - win + 1)
        atr[i] = (csum[i] - (csum[a - 1] if a > 0 else 0.0)) / (i - a + 1)
    return atr


def exit_trail_pct(opens, highs, lows, closes, f, end, k):
    entry = opens[f]
    c = closes[f:end + 1]
    runmax = np.maximum.accumulate(np.maximum(c, entry))
    breach = c < runmax * (1.0 - k)
    breach[end - f] = False  # last bar handled by MoC
    idx = int(np.argmax(breach))
    if breach[idx]:
        return float(opens[f + idx + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def exit_chandelier(opens, highs, lows, closes, f, end, mult, win=14):
    entry = opens[f]
    c = closes[f:end + 1]
    runmax = np.maximum.accumulate(np.maximum(c, entry))
    atr = _atr(highs, lows, closes, f, end, win)
    stop = runmax - mult * atr
    breach = c < stop
    breach[end - f] = False
    idx = int(np.argmax(breach))
    if breach[idx]:
        return float(opens[f + idx + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def exit_tp_frac(opens, highs, lows, closes, f, end, tp):
    """Fixed take-profit: exit when close >= entry*(1+tp). Caps the tail (a control)."""
    entry = opens[f]
    c = closes[f:end + 1]
    hit = c >= entry * (1.0 + tp)
    hit[end - f] = False
    idx = int(np.argmax(hit))
    if hit[idx]:
        return float(opens[f + idx + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def exit_scaleout(opens, highs, lows, closes, f, end, t1, loose_k):
    """ASYMMETRIC: bank HALF the position the first time close >= entry*(1+t1) (fill
    open next bar), ride the remaining HALF on a LOOSE high-water trail. If t1 never
    hit, the whole position rides the loose trail. Realized = 0.5*g_banked + 0.5*g_ride
    (or full g_ride if no scale)."""
    entry = opens[f]
    c = closes[f:end + 1]
    hit = c >= entry * (1.0 + t1)
    hit[end - f] = False
    bidx = int(np.argmax(hit))
    if not hit[bidx]:
        # never reached t1 -> whole position on loose trail
        return exit_trail_pct(opens, highs, lows, closes, f, end, loose_k)
    g_banked = float(opens[f + bidx + 1] / entry - 1.0)
    # remaining half rides the loose trail from the SAME entry, over the rest of the day
    g_ride = exit_trail_pct(opens, highs, lows, closes, f, end, loose_k)
    return 0.5 * g_banked + 0.5 * g_ride


def exit_time_stop(opens, highs, lows, closes, f, end, n_min):
    """Exit n_min minutes after fill (or MoC if shorter). The burst is ~1h."""
    x = min(f + n_min, end)
    return float(closes[x] / opens[f] - 1.0)


def exit_vol_trail(opens, highs, lows, closes, f, end, k, pre_vol):
    """Volatility-scaled % trail: trail width = clip(k * pre_vol, 0.005, 0.08)."""
    width = float(np.clip(k * pre_vol, 0.005, 0.08))
    return exit_trail_pct(opens, highs, lows, closes, f, end, width)


def exit_breakeven_ride(opens, highs, lows, closes, f, end, tight_k, lock_b, loose_k):
    """ASYMMETRIC CONVEX: trail TIGHT (tight_k) until running gain first reaches +lock_b
    (locks in a winner / cuts early reversers fast); once locked, switch to a LOOSE trail
    (loose_k) so the rare tail-runner can run. Strictly causal: the switch depends only on
    the running max so far."""
    entry = opens[f]
    n = end - f + 1
    c = closes[f:end + 1]
    runmax = entry
    locked = False
    for i in range(n - 1):  # last bar is MoC
        px = c[i]
        if px > runmax:
            runmax = px
        if not locked and (runmax / entry - 1.0) >= lock_b:
            locked = True
        k = loose_k if locked else tight_k
        if px < runmax * (1.0 - k):
            return float(opens[f + i + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def exit_ride_to_close(opens, highs, lows, closes, f, end):
    """The DUMB HOLD baseline (= D72 'ride to day close', entry+exit fixed). Every other
    policy must BEAT this to claim an exit edge. If it doesn't, the capture is the cohort's
    hold-to-close beta, not a smart exit."""
    return float(closes[end] / opens[f] - 1.0)


# Policy registry: name -> (fn, list of param-dicts). Param dicts are the swept grid.
def build_policies():
    pol = {}
    pol["ride_to_close"] = (exit_ride_to_close, [{}])  # the dumb-hold NULL baseline
    pol["trail_pct"] = (exit_trail_pct,
                        [{"k": k} for k in (0.01, 0.02, 0.03, 0.05)])
    pol["chandelier"] = (exit_chandelier,
                         [{"mult": m} for m in (2.0, 3.0, 4.0, 6.0)])
    pol["tp_frac"] = (exit_tp_frac,
                      [{"tp": t} for t in (0.02, 0.03, 0.05, 0.08)])
    pol["scaleout"] = (exit_scaleout,
                       [{"t1": t1, "loose_k": lk}
                        for t1 in (0.01, 0.02, 0.03) for lk in (0.04, 0.06)])
    pol["time_stop"] = (exit_time_stop,
                        [{"n_min": n} for n in (30, 60, 120, 240)])
    pol["vol_trail"] = (exit_vol_trail,
                        [{"k": k} for k in (2.0, 3.0, 5.0)])
    pol["breakeven_ride"] = (exit_breakeven_ride,
                             [{"tight_k": tk, "lock_b": lb, "loose_k": lk}
                              for tk in (0.01, 0.015) for lb in (0.01, 0.02)
                              for lk in (0.04, 0.06)])
    return pol


def param_key(p: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in p.items())


# ----------------------------------------------------------------- data + events

def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def load_raw(sym: str) -> pl.DataFrame:
    p = SUBBAR / f"{sym}_1m.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pl.read_parquet(p, columns=["minute_ts", "open", "high", "low", "close", "n_trades"]).sort("minute_ts")
    t0, t1 = int(df["minute_ts"].min()), int(df["minute_ts"].max())
    grid = pl.DataFrame({"minute_ts": np.arange(t0, t1 + 60_000, 60_000, dtype=np.int64)})
    df = grid.join(df, on="minute_ts", how="left", maintain_order="left")
    df = df.with_columns(pl.col("n_trades").is_not_null().alias("is_real"))
    df = df.with_columns(pl.col("close").forward_fill())
    df = df.with_columns([
        pl.col("open").fill_null(pl.col("close")),
        pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")),
    ])
    # pre-event 1m vol (causal): std of 1m returns over the prior day, shifted 60m
    df = df.with_columns(pl.col("close").pct_change().alias("ret_1m"))
    df = df.with_columns(
        pl.col("ret_1m").shift(60).rolling_std(1440, min_samples=720).alias("pre_vol"))
    return df


def study_asset(sym: str, policies: dict) -> dict:
    t0 = time.time()
    df = load_raw(sym)
    ms = df["minute_ts"].to_numpy()
    opens = df["open"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    real = df["is_real"].to_numpy().astype(bool)
    pre_vol = df["pre_vol"].to_numpy().astype(float)
    n = len(df)

    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)

    events = []
    days_meta = []
    for ds, de in zip(day_starts, day_ends):
        if de - ds < 1200:
            continue
        if real[ds:de + 1].mean() < 0.90:
            continue
        d_open = opens[ds]
        if not np.isfinite(d_open) or d_open <= 0:
            continue
        d_high = float(np.max(highs[ds:de + 1]))
        d_close = closes[de]
        runup = d_high / d_open - 1.0
        sp = split_of(int(ms[ds]))
        days_meta.append({"sym": sym, "split": sp, "runup": float(runup),
                          "day_ret": float(d_close / d_open - 1.0)})
        # NOTE: UNSEEN events ARE collected here but tagged; they are NEVER touched in
        # aggregation/selection -- only the final single peek reads them.
        rel = closes[ds:de + 1] / d_open - 1.0
        hit = np.flatnonzero(rel >= TRIGGER)
        if len(hit) == 0:
            continue
        m = ds + int(hit[0])
        f = m + 1
        if de - f < MIN_MINUTES_LEFT:
            continue
        entry = opens[f]
        if not np.isfinite(entry) or entry <= 0:
            continue
        # AVAILABLE MOVE = post-trigger MFE (hindsight, denominator only)
        ceil_px = float(np.max(closes[f:de + 1]))
        available_mfe = ceil_px / entry - 1.0
        pv = float(pre_vol[f]) if np.isfinite(pre_vol[f]) else 0.01
        ev = {
            "sym": sym, "day_ms": int(ms[ds]), "fill_ms": int(ms[f]), "split": sp,
            "runup_day": float(runup), "is_mover5": bool(runup >= MOVER_RUNUP),
            "available_mfe": float(available_mfe),
            "to_close_gross": float(d_close / entry - 1.0),
            "f": int(f), "de": int(de),
            "policies": {},
        }
        for pname, (fn, grid) in policies.items():
            for p in grid:
                if pname == "vol_trail":
                    g = fn(opens, highs, lows, closes, f, de, pre_vol=pv, **p)
                else:
                    g = fn(opens, highs, lows, closes, f, de, **p)
                ev["policies"][f"{pname}|{param_key(p)}"] = float(g)
        events.append(ev)
    n_movers_tr = sum(1 for d in days_meta if d["runup"] >= MOVER_RUNUP and d["split"] == "TRAIN")
    n_movers_oos = sum(1 for d in days_meta if d["runup"] >= MOVER_RUNUP and d["split"] == "OOS")
    print(f"[{sym}] days={len(days_meta)} events={len(events)} "
          f"movers5(T/O)={n_movers_tr}/{n_movers_oos} ({time.time()-t0:.0f}s)")
    return {"sym": sym, "events": events, "days": days_meta,
            "ms": ms, "opens": opens, "closes": closes}  # arrays kept for the neutral overlay


# ----------------------------------------------------------------- aggregation

def years_of(split: str) -> float:
    span = (TRAIN_END_MS - DATA_START_MS) if split == "TRAIN" else \
           ((OOS_END_MS - TRAIN_END_MS) if split == "OOS" else
            (int(dt.datetime(2026, 5, 29, tzinfo=dt.timezone.utc).timestamp() * 1000) - OOS_END_MS))
    return span / (365.25 * DAY_MS)


def cell_stats(events, split, pol_key, n_assets, mover_only=True):
    """Per-event + portfolio stats for one policy cell on one split."""
    evs = [e for e in events if e["split"] == split and (e["is_mover5"] or not mover_only)
           and pol_key in e["policies"]]
    if not evs:
        return {"n": 0}
    g = np.array([e["policies"][pol_key] for e in evs])
    mfe = np.array([e["available_mfe"] for e in evs])
    years = years_of(split)
    out = {"n": len(g), "gross_mean": float(g.mean()), "gross_median": float(np.median(g)),
           "win_rate_gross": float((g > 0).mean())}
    for c in COSTS_RT:
        key = f"{round(c*1e4)}bps"
        net = g - c
        out[f"net_mean_{key}"] = float(net.mean())
        out[f"portfolio_net_per_year_{key}"] = float(net.sum() / n_assets / years)
    out["net_mean_maker"] = float((g - MAKER_RT).mean())
    out["portfolio_net_per_year_maker"] = float((g - MAKER_RT).sum() / n_assets / years)
    # CAPITAL-HONEST compounded equity: each calendar day take at most one position per
    # asset, equal-weight 1/n_assets that day, daily portfolio ret = mean event-net that
    # day (0 if no event), compound across all calendar days in the split. This is the
    # capacity-bounded view (portfolio_net_per_year above is uncompounded sum-scaling and
    # OVER-states deployable wealth -- kept for cross-cell comparison only).
    day_idx = np.array([e["day_ms"] // DAY_MS for e in evs])
    nets24 = g - GATE_COST
    uniq_days = np.unique(day_idx)
    daily_ret = np.array([nets24[day_idx == d].mean() / n_assets for d in uniq_days])
    total_days = max(1, int((years * 365.25)))
    eq = float(np.prod(1.0 + daily_ret)) if daily_ret.size else 1.0
    out["compound_equity_24bps"] = eq           # x starting capital over the split
    out["compound_cagr_24bps"] = float(eq ** (365.25 / total_days) - 1.0) if eq > 0 else -1.0
    # CAPTURE RATE (realized gross / available MFE), only where MFE is materially positive
    valid = mfe > 0.005
    caps = (g[valid] / mfe[valid])
    if caps.size:
        out["capture_rate_median"] = float(np.median(caps))
        out["capture_rate_mean"] = float(np.mean(caps))
        out["capture_rate_p25"] = float(np.quantile(caps, 0.25))
        out["capture_rate_p75"] = float(np.quantile(caps, 0.75))
        out["n_capture"] = int(caps.size)
    # asymmetry diagnostic: does this cut the median reverser while keeping the tail?
    out["frac_negative"] = float((g < 0).mean())
    out["tail_kept_p90"] = float(np.quantile(g, 0.90))  # 90th pct realized gain
    out["available_mfe_median"] = float(np.median(mfe))
    # breadth at gate cost (>=3 events to qualify; full universe denominator)
    per = {}
    for e in evs:
        per.setdefault(e["sym"], []).append(e["policies"][pol_key] - GATE_COST)
    rows = {s: float(np.mean(v)) for s, v in per.items() if len(v) >= 3}
    out["breadth_pos"] = sum(1 for v in rows.values() if v > 0)
    out["breadth_tot"] = n_assets
    out["breadth_qualifying"] = len(rows)
    # jackknife on net @ gate cost
    net24 = np.sort(g - GATE_COST)
    for K in (1, 3, 5):
        out[f"jk_drop_top{K}_net24_mean"] = float(net24[:-K].mean()) if len(net24) > K else None
    return out


def block_boot_ci(deltas, cluster_ids, seed=11):
    """Weekly-cluster bootstrap CI on the mean of paired deltas."""
    d = np.asarray(deltas)
    cl = np.asarray(cluster_ids)
    uniq = np.unique(cl)
    if len(uniq) < 3:
        return None
    rng = np.random.default_rng(seed)
    boots = np.empty(N_BOOT)
    idx_by = {c: np.flatnonzero(cl == c) for c in uniq}
    for b in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[b] = np.concatenate([d[idx_by[c]] for c in pick]).mean()
    return {"mean": float(d.mean()), "p05": float(np.quantile(boots, 0.05)),
            "p95": float(np.quantile(boots, 0.95)), "n": int(len(d)), "n_clusters": int(len(uniq))}


# ----------------------------------------------------------------- neutral overlay

def neutral_overlay(events, asset_arrays, hedge_sym, split, pol_key, n_assets):
    """LONG the triggered mover (exit = selected policy gross), SHORT a matched BTC hedge
    over the IDENTICAL hold window [fill_ms .. exit-of-policy]. Since the policy exit time
    is path-dependent, we approximate the hold as fill_ms -> day-end close (the policy's
    longest possible hold; for a day-bounded policy the hedge is held the same calendar
    window). Reports the net long-short return and the realized beta of the long leg to
    the hedge over the holds."""
    h = asset_arrays.get(hedge_sym)
    if h is None:
        return {"error": f"hedge {hedge_sym} not loaded"}
    hms = h["ms"]; hclose = h["closes"]
    evs = [e for e in events if e["split"] == split and e["is_mover5"] and pol_key in e["policies"]
           and e["sym"] != hedge_sym]
    if not evs:
        return {"n": 0}
    longs, hedges, ls = [], [], []
    for e in evs:
        gL = e["policies"][pol_key]
        # hedge return over [fill_ms .. day-end]; day-end ms = fill_ms's day + ~end
        i0 = int(np.searchsorted(hms, e["fill_ms"]))
        # day end ms for the hedge: align to the same calendar minute index span as the long
        day_end_ms = (e["fill_ms"] // DAY_MS) * DAY_MS + (DAY_MS - 60_000)
        i1 = int(np.searchsorted(hms, day_end_ms))
        if i0 >= len(hms) or i1 >= len(hms) or i1 <= i0:
            continue
        gH = float(hclose[i1] / hclose[i0] - 1.0)
        longs.append(gL); hedges.append(gH); ls.append(gL - gH)
    if not ls:
        return {"n": 0}
    longs = np.array(longs); hedges = np.array(hedges); ls = np.array(ls)
    years = years_of(split)
    beta = float(np.cov(longs, hedges)[0, 1] / (np.var(hedges) + 1e-12))
    net = ls - GATE_COST * 2  # two legs, taker both sides (conservative)
    return {
        "n": len(ls), "hedge": hedge_sym,
        "ls_gross_mean": float(ls.mean()), "ls_gross_median": float(np.median(ls)),
        "long_leg_mean": float(longs.mean()), "hedge_leg_mean": float(hedges.mean()),
        "beta_long_to_hedge": beta,
        "net_mean_2x24bps": float(net.mean()),
        "portfolio_net_per_year_2x24bps": float(net.sum() / n_assets / years),
        "win_rate_ls": float((ls > 0).mean()),
    }


# ----------------------------------------------------------------- decomposition

def bleed_decomposition(events, split, n_assets):
    """Hold ENTRY fixed; how much of the unconditional ride loss is EXIT-fixable?
    Compares, on the >=5% mover cohort:
      - ride_to_close : the naive D72-style 'ride to day close' (entry+exit fixed)
      - best_train_exit: the best symmetric trail vs the best asymmetric exit
    plus the ENTRY-quality split: mover-cohort MFE vs all-trigger MFE (false/late triggers).
    """
    evs = [e for e in events if e["split"] == split]
    mov = [e for e in evs if e["is_mover5"]]
    allt = evs
    if not mov:
        return {"n_mover": 0}
    to_close = np.array([e["to_close_gross"] for e in mov])
    to_close_all = np.array([e["to_close_gross"] for e in allt])  # CAUSAL cohort (no day hindsight)
    mfe_mov = np.array([e["available_mfe"] for e in mov])
    mfe_all = np.array([e["available_mfe"] for e in allt])
    return {
        "n_mover": len(mov), "n_all_triggers": len(allt),
        "mover_share_of_triggers": float(len(mov) / max(1, len(allt))),
        "available_mfe_median_mover": float(np.median(mfe_mov)),
        "available_mfe_median_all": float(np.median(mfe_all)),
        "ride_to_close_mean": float(to_close.mean()),
        "ride_to_close_median": float(np.median(to_close)),
        # the CAUSAL bar: ride-to-close on ALL triggers (the trigger never knows the day's
        # outcome). The mover-only stats above CONDITION ON the future (>=5% runup is a
        # day-end fact) and overstate the edge -- this is the honest deployable number.
        "ride_to_close_net24_all_triggers_mean": float(to_close_all.mean() - GATE_COST),
        "ride_to_close_net24_movers_mean": float(to_close.mean() - GATE_COST),
        "give_back_median": float(np.median(mfe_mov) - np.median(to_close)),  # MFE not captured by hold-to-close
    }


# ----------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="1m mover capture-rate (asymmetric-exit) study")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="u10")
    ap.add_argument("--hedge", default="BTCUSDT")
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        args.tag = args.universe
    else:
        ap.error("provide --assets or --universe")

    policies = build_policies()
    all_events, all_days = [], []
    asset_arrays = {}
    for sym in syms:
        try:
            res = study_asset(sym, policies)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        all_events.extend(res["events"])
        all_days.extend(res["days"])
        asset_arrays[sym] = {"ms": res["ms"], "opens": res["opens"], "closes": res["closes"]}
    n_assets = len({e["sym"] for e in all_events}) or 1

    # opportunity surface
    surface = {}
    for sp in ["TRAIN", "OOS", "UNSEEN"]:
        ds = [d for d in all_days if d["split"] == sp]
        surface[sp] = {"asset_days": len(ds),
                       "mover5_days": sum(1 for d in ds if d["runup"] >= MOVER_RUNUP),
                       "n_events": sum(1 for e in all_events if e["split"] == sp),
                       "n_mover_events": sum(1 for e in all_events if e["split"] == sp and e["is_mover5"])}

    # full grid on TRAIN (mover cohort), all cells reported
    pol_keys = []
    for pname, (fn, grid) in policies.items():
        for p in grid:
            pol_keys.append(f"{pname}|{param_key(p)}")

    train_grid = {pk: cell_stats(all_events, "TRAIN", pk, n_assets, mover_only=True) for pk in pol_keys}

    # SELECTION: per policy family, best param on TRAIN by portfolio net/yr @24bps with
    # breadth >= 6/10 and n >= 30; then the single overall best family-winner -> OOS.
    family_winners = {}
    for pname in policies:
        best, bestv = None, -1e18
        for pk in pol_keys:
            if not pk.startswith(pname + "|"):
                continue
            s = train_grid[pk]
            if s.get("n", 0) < 30:
                continue
            if s.get("breadth_tot", 0) and s["breadth_pos"] / s["breadth_tot"] >= 0.6:
                v = s.get("portfolio_net_per_year_24bps", -1e18)
                if v > bestv:
                    bestv, best = v, pk
        if best:
            family_winners[pname] = {"cell": best, "train_net_per_year_24bps": bestv}
    # the champion is the best EXIT POLICY -- exclude the dumb-hold null baseline from
    # being crowned (it is reported as a family winner for contrast, never as champion).
    champ_pool = {k: v for k, v in family_winners.items() if k != "ride_to_close"}
    overall = max(champ_pool.items(), key=lambda kv: kv[1]["train_net_per_year_24bps"], default=None)

    # OOS for every family winner (so the grid is honest), + the overall champion detail
    oos_grid = {fw["cell"]: cell_stats(all_events, "OOS", fw["cell"], n_assets, mover_only=True)
                for fw in family_winners.values()}
    # the dumb-hold null on OOS, always computed (the bar every policy must clear)
    rtc_key = "ride_to_close|"
    oos_rtc = cell_stats(all_events, "OOS", rtc_key, n_assets, mover_only=True)
    train_rtc = train_grid.get(rtc_key, {})

    # null contrast for the champion on OOS: realized exit vs ride-to-close (the dumb hold)
    champion = overall[1]["cell"] if overall else None
    champ_oos_null = None
    if champion:
        oos_movers = [e for e in all_events if e["split"] == "OOS" and e["is_mover5"]
                      and champion in e["policies"]]
        if oos_movers:
            deltas = [e["policies"][champion] - e["to_close_gross"] for e in oos_movers]
            cl = [e["fill_ms"] // CLUSTER_MS for e in oos_movers]
            champ_oos_null = block_boot_ci(deltas, cl)

    # neutral overlay for the champion on OOS
    champ_neutral = neutral_overlay(all_events, asset_arrays, args.hedge, "OOS", champion, n_assets) \
        if champion else None

    # bleed decomposition
    bleed = {sp: bleed_decomposition(all_events, sp, n_assets) for sp in ["TRAIN", "OOS"]}

    # ---------------- UNSEEN: single peek at the very end, champion only ----------------
    unseen = None
    if champion:
        unseen = {
            "champion_cell": champion,
            "taker_24bps": cell_stats(all_events, "UNSEEN", champion, n_assets, mover_only=True),
            "neutral": neutral_overlay(all_events, asset_arrays, args.hedge, "UNSEEN", champion, n_assets),
        }

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_capture_rate", "git_sha": sha, "seed": args.seed,
        "params": {"trigger": TRIGGER, "min_minutes_left": MIN_MINUTES_LEFT,
                   "mover_runup": MOVER_RUNUP, "gate_cost": GATE_COST, "maker_rt": MAKER_RT,
                   "selection": "per-family best TRAIN net/yr@24bps with breadth>=0.6,n>=30; overall best -> OOS; UNSEEN peeked once"},
        "surface": surface,
        "train_grid": train_grid,
        "family_winners": family_winners,
        "overall_champion": overall[1] if overall else None,
        "ride_to_close_null": {"TRAIN": train_rtc, "OOS": oos_rtc},
        "oos_grid": oos_grid,
        "champion_oos_vs_ride_to_close_null": champ_oos_null,
        "champion_neutral_oos": champ_neutral,
        "bleed_decomposition": bleed,
        "unseen_single_peek": unseen,
        "n_assets": n_assets,
        "caveats": [
            "available MFE (hindsight) is the capture-rate DENOMINATOR only; never used by a causal exit",
            "u10 current membership = survivorship; contrasts are within-asset",
            "neutral hedge held [fill .. day-end]; the policy exit is path-dependent so this is an upper bound on hedge cost",
            "exit at final-minute close is a market-on-close approximation",
        ],
    }
    out_path = OUT / f"mover_capture_{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---------------- console story ----------------
    print("\n" + "=" * 80)
    print("MOVER CAPTURE-RATE -- ASYMMETRIC-EXIT RE-TEST (entry fixed = D72 +1.5% trigger)")
    print("=" * 80)
    for sp in ["TRAIN", "OOS", "UNSEEN"]:
        s = surface[sp]
        print(f"[{sp}] asset-days={s['asset_days']} mover5-days={s['mover5_days']} "
              f"trigger-events={s['n_events']} (mover-events={s['n_mover_events']})")

    print("\nBLEED DECOMPOSITION -- is the loss ENTRY (false/late triggers) or EXIT (give-back)?")
    for sp in ["TRAIN", "OOS"]:
        b = bleed[sp]
        if b.get("n_mover"):
            print(f"  [{sp}] {b['mover_share_of_triggers']*100:.0f}% of {b['n_all_triggers']} triggers are true movers "
                  f"(rest fizzle) | available MFE median: mover {b['available_mfe_median_mover']*100:+.2f}% "
                  f"vs all-trigger {b['available_mfe_median_all']*100:+.2f}%")
            print(f"        ENTRY effect -> ride-to-close net@24bps: MOVER-cohort {b['ride_to_close_net24_movers_mean']*100:+.3f}%/ev "
                  f"(conditions on future) BUT CAUSAL ALL-TRIGGER {b['ride_to_close_net24_all_triggers_mean']*100:+.3f}%/ev")
            print(f"        EXIT effect  -> give-back median {b['give_back_median']*100:.2f}pp of MFE left on the table by hold-to-close")

    print("\nTRAIN GRID -- per policy cell (mover cohort): net/yr@24bps | capture-rate med | "
          "win% | breadth | frac-neg | p90-tail:")
    for pname in policies:
        print(f"  [{pname}]")
        for pk in pol_keys:
            if not pk.startswith(pname + "|"):
                continue
            s = train_grid[pk]
            if s.get("n", 0) == 0:
                continue
            cr = s.get("capture_rate_median")
            crs = f"{cr*100:+.0f}%" if cr is not None else "--"
            print(f"    {pk.split('|',1)[1]:<24} {s.get('portfolio_net_per_year_24bps',0)*100:+8.1f}%/yr | "
                  f"cap {crs:>6} | win {s['win_rate_gross']*100:3.0f}% | "
                  f"breadth {s.get('breadth_pos',0)}/{s.get('breadth_tot',0)} | "
                  f"neg {s.get('frac_negative',0)*100:2.0f}% | p90 {s.get('tail_kept_p90',0)*100:+.1f}% (n={s['n']})")

    # the NULL bar: the dumb hold-to-close, the line every exit must beat
    print("\nNULL BAR -- RIDE-TO-CLOSE (dumb hold, entry+exit fixed; the D72/D61 baseline):")
    print(f"  TRAIN: net/yr@24bps {train_rtc.get('portfolio_net_per_year_24bps',0)*100:+.1f}% | "
          f"capture med {train_rtc.get('capture_rate_median')*100 if train_rtc.get('capture_rate_median') is not None else float('nan'):+.0f}% | "
          f"compound-CAGR {train_rtc.get('compound_cagr_24bps',0)*100:+.1f}%/yr (capital-honest)")
    print(f"  OOS:   net/yr@24bps {oos_rtc.get('portfolio_net_per_year_24bps',0)*100:+.1f}% | "
          f"capture med {oos_rtc.get('capture_rate_median')*100 if oos_rtc.get('capture_rate_median') is not None else float('nan'):+.0f}% | "
          f"compound-CAGR {oos_rtc.get('compound_cagr_24bps',0)*100:+.1f}%/yr (capital-honest)")

    print("\nFAMILY WINNERS (TRAIN-selected) -> OOS  [delta vs ride-to-close = the EXIT edge]:")
    for pname, fw in sorted(family_winners.items(), key=lambda kv: -kv[1]["train_net_per_year_24bps"]):
        oss = oos_grid.get(fw["cell"], {})
        cr = oss.get("capture_rate_median")
        crs = f"{cr*100:+.0f}%" if cr is not None else "--"
        d_vs_rtc = oss.get("net_mean_24bps", 0) - oos_rtc.get("net_mean_24bps", 0)
        tag = " (=NULL)" if pname == "ride_to_close" else ""
        print(f"  {pname:<16} {fw['cell'].split('|',1)[1]:<24} "
              f"TRAIN {fw['train_net_per_year_24bps']*100:+8.1f}%/yr -> "
              f"OOS {oss.get('portfolio_net_per_year_24bps',0)*100:+8.1f}%/yr "
              f"(cap {crs}, CAGR {oss.get('compound_cagr_24bps',0)*100:+.0f}%, "
              f"d-vs-hold {d_vs_rtc*100:+.3f}%/ev, breadth {oss.get('breadth_pos',0)}/{oss.get('breadth_tot',0)}, n={oss.get('n',0)}){tag}")

    if champion:
        print(f"\nCHAMPION (overall TRAIN best) -> {champion}")
        oss = oos_grid[champion]
        print(f"  OOS taker: net/yr@24bps {oss.get('portfolio_net_per_year_24bps',0)*100:+.1f}% "
              f"(uncompounded) | compound-CAGR {oss.get('compound_cagr_24bps',0)*100:+.1f}%/yr (capital-honest) | "
              f"per-event net {oss.get('net_mean_24bps',0)*100:+.3f}% | "
              f"net/yr maker {oss.get('portfolio_net_per_year_maker',0)*100:+.1f}% | "
              f"capture med {oss.get('capture_rate_median')*100 if oss.get('capture_rate_median') is not None else float('nan'):+.0f}% "
              f"mean {oss.get('capture_rate_mean')*100 if oss.get('capture_rate_mean') is not None else float('nan'):+.0f}% | "
              f"breadth {oss.get('breadth_pos',0)}/{oss.get('breadth_tot',0)} | "
              f"jk-3 {oss.get('jk_drop_top3_net24_mean')*100 if oss.get('jk_drop_top3_net24_mean') is not None else float('nan'):+.3f}%")
        print(f"  EXIT EDGE vs dumb hold: net/ev {(oss.get('net_mean_24bps',0)-oos_rtc.get('net_mean_24bps',0))*100:+.3f}% "
              f"(champion {oss.get('net_mean_24bps',0)*100:+.3f}% vs hold {oos_rtc.get('net_mean_24bps',0)*100:+.3f}%)")
        if champ_oos_null:
            c = champ_oos_null
            print(f"  OOS vs ride-to-close null: mean delta {c['mean']*100:+.3f}% "
                  f"[p05 {c['p05']*100:+.3f}, p95 {c['p95']*100:+.3f}] (n={c['n']}, clusters={c['n_clusters']})")
        if champ_neutral and champ_neutral.get("n"):
            cn = champ_neutral
            print(f"  OOS NEUTRAL (long mover / short {cn['hedge']}): LS gross mean {cn['ls_gross_mean']*100:+.3f}% "
                  f"median {cn['ls_gross_median']*100:+.3f}% | beta {cn['beta_long_to_hedge']:+.2f} | "
                  f"net/yr@2x24bps {cn['portfolio_net_per_year_2x24bps']*100:+.1f}% | win {cn['win_rate_ls']*100:.0f}%")
        if unseen:
            us = unseen["taker_24bps"]; un = unseen["neutral"]
            print(f"\n  UNSEEN single peek (champion {champion}):")
            print(f"    taker: net/yr@24bps {us.get('portfolio_net_per_year_24bps',0)*100:+.1f}% | "
                  f"per-event net {us.get('net_mean_24bps',0)*100:+.3f}% | "
                  f"capture med {us.get('capture_rate_median')*100 if us.get('capture_rate_median') is not None else float('nan'):+.0f}% | "
                  f"breadth {us.get('breadth_pos',0)}/{us.get('breadth_tot',0)} | n={us.get('n',0)}")
            if un and un.get("n"):
                print(f"    NEUTRAL: LS gross med {un['ls_gross_median']*100:+.3f}% beta {un['beta_long_to_hedge']:+.2f} "
                      f"net/yr@2x24bps {un['portfolio_net_per_year_2x24bps']*100:+.1f}% win {un['win_rate_ls']*100:.0f}%")
    else:
        print("\nNO TRAIN cell met the selection bar -- the grid is the verdict.")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
