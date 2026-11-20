"""src/strat/nlcycle3_timing.py -- NL-CYCLE 3 LANE (a): multi-TF (4h) entry timing overlay.

FRESH ANGLE: the router picks the basket on DAILY bars. The systematic price cycle tested daily
DIRECTION (dead). This tests EXECUTION TIMING -- a different axis: when the router newly ENTERS a
name on day d (acted at d+1), does timing the entry at 4h granularity within the entry day improve
the realized entry price, and does that flow through to a better 7d outcome?

MECHANISM (long-only-spot, internal-data, causal):
  Baseline router enters at the daily-bar mechanic (position lagged 1 bar -> filled at d's return).
  Overlay: for each NEW-ENTRY name (weight increase on day d), instead of taking the full daily
  return on the entry day, we simulate filling the position at an INTRADAY 4h price chosen by a
  timing rule over the entry day's six 4h bars:
    - 'pullback': fill at the LOWEST 4h close on the entry day (buy the intraday dip) -> entry edge
      if intraday lows are systematically re-touched (favorable fill).
    - 'breakout': fill only after a 4h close breaks the prior daily high (confirmation); if it never
      breaks, SKIP the entry that day (stay cash, re-evaluated next day) -> avoids false starts.
    - 'open': fill at the entry day's first 4h open (== ~baseline; control).
  The entry-day return is replaced by (close_d / fill_price - 1); subsequent days unchanged. Carried
  names (already held) are untouched. This ONLY changes the ENTRY-DAY economics of new positions.

  All causal: the timing rule at day d uses only day-d 4h bars (which close during day d); the
  resulting position is still only LONG and only on names the router already chose.

REFEREE: date-block-permute the per-entry timing CHOICE (shuffle which entries get the pullback fill
  in 7-day blocks). A real timing edge must beat the block-permuted null. Plus an 'open' control that
  must be ~flat vs baseline (sanity that the harness isn't manufacturing an edge).

No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as ref
import strat.adaptive_meta_engine as ame
from strat.ma_per_instrument import _panel

OOS_START = "2022-01-01"
OOS_END = "2026-06-01"
TRAIN_END = "2022-01-01"
N_SLICES = 500
SEEDS = [11, 23, 42]


# ============================================================
# 4h INTRADAY PANEL: for each (asset, day) -> the 4h OHLC bars of that day
# ============================================================
def load_4h_intraday(C: pd.DataFrame) -> dict:
    """Return {sym: DataFrame indexed by 4h timestamp with o/h/l/c}. Used to derive an intraday
    fill price for the entry day. Causal: we only ever read 4h bars that close on the entry day.
    """
    out = {}
    for sym in C.columns:
        try:
            o, h, l, c, ms = _panel(sym, "4h")
        except Exception:
            continue
        idx = pd.to_datetime(ms, unit="ms")
        df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[sym] = df
    return out


def intraday_fill_factor(C: pd.DataFrame, intraday: dict, rule: str, x: float = 0.02) -> pd.DataFrame:
    """For each (date d, asset s) return fill_ratio = fill_price / close_d for a NEW entry. The
    entry-day realized return becomes (close_d / fill_price - 1) = (1/fill_ratio - 1).

    REALIZABLE RULES ONLY (no intraday hindsight). `close_dm1` = prior-day daily close = the price
    known when the position is decided (close of d-1). All fills are achievable with a passive limit
    or a confirmation trigger:
      - 'limit_dip' : place a BUY LIMIT at (1-x)*close_dm1. Fill at that limit IF any 4h LOW on day d
                      touches it (limit orders fill at the limit, not the low). If never touched, the
                      dip didn't come -> CHASE: fill at day-d close (ratio=1, no edge that day).
                      Realizable: a resting limit order. NO look-ahead (the limit level is set from
                      d-1 close; the touch test uses day-d lows which is exactly how a limit fills).
      - 'open'      : fill at the entry day's first 4h OPEN (~baseline control; must be ~flat).
      - 'breakout'  : fill at the first 4h CLOSE on day d that exceeds the prior-DAY high (momentum
                      confirmation, realizable as a stop-entry). NaN -> never confirmed -> SKIP entry.
      - 'minclose'  : LEAKED CEILING (fill at the day's minimum 4h close) -- kept ONLY as an explicit
                      look-ahead upper bound to show how much of any 'edge' is hindsight. NOT tradeable.
    """
    fill_ratio = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    Cd = C  # daily closes (dates x assets), our causal d-1 reference
    for sym in C.columns:
        if sym not in intraday:
            continue
        df = intraday[sym]
        day = df.index.floor("D")
        g = df.groupby(day)
        opens = g["o"].first()
        last_close = g["c"].last()
        mins_close = g["c"].min()
        lows = g["l"].min()
        highs = g["h"].max()
        ph = highs.shift(1)               # prior-day high (causal)
        close_dm1 = Cd[sym].shift(1)      # prior daily close (known at decision time)
        col = pd.Series(index=C.index, dtype=float)
        if rule == "open":
            col = (opens / last_close).reindex(C.index)
        elif rule == "minclose":
            col = (mins_close / last_close).reindex(C.index)
        elif rule == "limit_dip":
            cdm1 = close_dm1.reindex(C.index)
            day_low = lows.reindex(C.index)
            day_close = last_close.reindex(C.index)
            limit = (1.0 - x) * cdm1
            touched = day_low <= limit
            # fill = limit if touched else day_close; ratio = fill/day_close
            fill = pd.Series(np.where(touched.values, limit.values, day_close.values), index=C.index)
            col = fill / day_close
        elif rule == "breakout":
            fills = {}
            for d, sub in g:
                pdh = ph.get(d, np.nan)
                if pd.isna(pdh):
                    continue
                brk = sub["c"][sub["c"] > pdh]
                if len(brk) > 0:
                    fills[d] = brk.iloc[0] / sub["c"].iloc[-1]
            col = pd.Series(fills).reindex(C.index)
        fill_ratio[sym] = col
    return fill_ratio


# ============================================================
# BUILD ENTRY-TIMED BOOK
# ============================================================
def book_with_entry_timing(Wr: pd.DataFrame, ind: dict, fill_ratio: pd.DataFrame,
                           rule: str, entry_mask: pd.DataFrame = None) -> pd.Series:
    """Recompute the router book return, but on NEW-ENTRY name-days replace the entry-day return.

    Baseline book (ref.book_daily_returns): pos = W.shift(1); bret_d = sum(pos_d * R_d) - cost.
    The entry day for a name is the FIRST day pos>0 after being 0 (i.e. W increased on d-1, acted d).
    On that day, instead of R_d (close_{d-1}->close_d), the realized leg is close_d / fill_price - 1,
    where fill_price = fill_ratio * close_d. We add the delta (close_d/fill - close_d/close_{d-1}) to
    that name's contribution on the entry day. For 'breakout' with NaN fill -> SKIP: the position is
    suppressed that entry day (pos forced 0), re-attempted next day naturally by the router carry.
    """
    R = ind["R"].reindex(index=Wr.index, columns=Wr.columns).fillna(0.0)
    C = ind["C"].reindex(index=Wr.index, columns=Wr.columns)
    pos = Wr.shift(1).fillna(0.0)
    # new-entry name-days at the BOOK level: pos>0 today and pos==0 yesterday
    new_entry = (pos > 0) & (pos.shift(1).fillna(0.0) <= 1e-12)
    if entry_mask is not None:
        new_entry = new_entry & entry_mask.reindex(index=Wr.index, columns=Wr.columns).fillna(False)

    fr = fill_ratio.reindex(index=Wr.index, columns=Wr.columns)
    pos_adj = pos.copy()
    R_adj = R.copy()

    # entry-day return replacement: realized = close_d/fill - 1 = (1/fill_ratio) - 1
    repl = (1.0 / fr) - 1.0   # the new entry-day return for that name
    # apply only where new_entry and fill available
    valid = new_entry & fr.notna()
    R_adj = R_adj.where(~valid, repl)

    if rule == "breakout":
        # where new_entry but fill is NaN (never broke prior high) -> suppress the position that day
        skip = new_entry & fr.isna()
        pos_adj = pos_adj.where(~skip, 0.0)

    turn = pos_adj.diff().abs().fillna(pos_adj.abs()).sum(axis=1)
    bret = (pos_adj * R_adj).sum(axis=1) - turn * (lab.COST / 2.0)
    return bret


def eval_book(b: pd.Series, bh_b: pd.Series) -> dict:
    prs = [ref.slice_stats(b, bh_b, OOS_START, OOS_END, N_SLICES, 7, s) for s in SEEDS]
    return {
        "pos_rate": round(float(np.mean([x["pos_rate"] for x in prs])), 2),
        "mean_pct": round(float(np.mean([x["mean_pct"] for x in prs])), 3),
        "p05_pct": round(float(np.mean([x["p05_pct"] for x in prs])), 2),
        "median_pct": round(float(np.mean([x["median_pct"] for x in prs])), 3),
        "beat_bh": round(float(np.mean([x["beat_bh_pct"] for x in prs])), 1),
        "down_wk_mean": round(float(np.mean([x["down_wk_eng_mean"] for x in prs])), 2),
    }


def block_permute_entrychoice(new_entry: pd.DataFrame, oos_start: str, block: int, seed: int):
    """Shuffle 7-day blocks of the entry-mask rows within OOS -> the timing overlay touches the same
    number of entry-days but at scrambled dates."""
    rng = np.random.default_rng(seed)
    idx = new_entry.index
    oos_pos = np.where(idx >= pd.Timestamp(oos_start))[0]
    blocks = [oos_pos[i:i + block] for i in range(0, len(oos_pos), block)]
    order = list(range(len(blocks))); rng.shuffle(order)
    new_pos = np.concatenate([blocks[o] for o in order])
    vals = new_entry.values.copy()
    vals[oos_pos] = new_entry.values[new_pos]
    return pd.DataFrame(vals, index=new_entry.index, columns=new_entry.columns)


def run_lane_a():
    t0 = time.time()
    print("=" * 78)
    print("NL-CYCLE 3 LANE (a): multi-TF (4h) ENTRY-TIMING overlay on router entries")
    print(f"OOS {OOS_START}->{OOS_END} | n={N_SLICES} | seeds={SEEDS}")
    print("=" * 78)

    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    bh_W = ref.bh_ew_weights(ind); bh_b = ref.book_daily_returns(bh_W, ind)
    tm = C.index < pd.Timestamp(TRAIN_END)
    vthr = float(ind["vol20"]["BTCUSDT"][tm].dropna().quantile(ame.VOL_HI_PCTILE))
    Wr = ame.build_weight_matrix(ind, vthr)
    rb = ref.book_daily_returns(Wr, ind)
    base = eval_book(rb, bh_b)
    print(f"\n[BASE ROUTER] pos={base['pos_rate']}% mean={base['mean_pct']}% p05={base['p05_pct']}% "
          f"down_wk={base['down_wk_mean']}% beat_bh={base['beat_bh']}%")

    print("\n[4h] loading intraday panel...")
    intraday = load_4h_intraday(C)
    print(f"[4h] loaded {len(intraday)} assets")

    # the book-level new-entry mask (for the permutation referee)
    pos = Wr.shift(1).fillna(0.0)
    new_entry = (pos > 0) & (pos.shift(1).fillna(0.0) <= 1e-12)

    # rules: 'open' = control (must be ~flat), 'minclose' = LEAKED ceiling (hindsight upper bound),
    # 'limit_dip_X' = REALIZABLE passive-limit dip-buy at X below prior close, 'breakout' = confirmation.
    results = {}
    rule_specs = [("open", None), ("minclose", None),
                  ("limit_dip", 0.01), ("limit_dip", 0.02), ("limit_dip", 0.03), ("limit_dip", 0.05),
                  ("breakout", None)]
    fr_cache = {}
    for rule, x in rule_specs:
        tag = rule if x is None else f"{rule}_{x}"
        fr = intraday_fill_factor(C, intraday, rule, x=x if x is not None else 0.02)
        fr_cache[tag] = (fr, rule)
        b = book_with_entry_timing(Wr, ind, fr, rule)
        st = eval_book(b, bh_b)
        results[tag] = st
        d_mean = st["mean_pct"] - base["mean_pct"]; d_p05 = st["p05_pct"] - base["p05_pct"]
        flag = " <-LEAKED" if rule == "minclose" else (" (control)" if rule == "open" else "")
        print(f"  {tag:13s}: pos={st['pos_rate']}% mean={st['mean_pct']}% (d{d_mean:+.3f}) "
              f"p05={st['p05_pct']}% (d{d_p05:+.2f}) down_wk={st['down_wk_mean']}% "
              f"beat_bh={st['beat_bh']}%{flag}")

    # ---- REFEREE: block-permute the entry-timing choice for the best REALIZABLE rule ----
    realizable = [t for t in results if t.startswith("limit_dip") or t == "breakout"]
    cand = max(realizable,
               key=lambda r: (results[r]["mean_pct"] - base["mean_pct"]) + (results[r]["p05_pct"] - base["p05_pct"]))
    fr_c, cand_rule = fr_cache[cand]
    real_st = results[cand]
    real_metric = (real_st["mean_pct"] - base["mean_pct"]) + (real_st["p05_pct"] - base["p05_pct"])
    print(f"\n[REFEREE] block-permutation null on '{cand}' (metric = mean+p05 delta vs base)...")
    n_perm = 200
    perm_metrics = []
    for ps in range(n_perm):
        em = block_permute_entrychoice(new_entry, OOS_START, 7, ps)
        bp = book_with_entry_timing(Wr, ind, fr_c, cand_rule, entry_mask=em)
        sp = ref.slice_stats(bp, bh_b, OOS_START, OOS_END, N_SLICES, 7, 42)
        pm = (sp["mean_pct"] - base["mean_pct"]) + (sp["p05_pct"] - base["p05_pct"])
        perm_metrics.append(pm)
    perm_metrics = np.array(perm_metrics)
    p_val = float((perm_metrics >= real_metric).mean())
    print(f"  real metric={real_metric:+.3f} | perm mean={perm_metrics.mean():+.3f} "
          f"std={perm_metrics.std():.3f} p95={np.percentile(perm_metrics,95):+.3f} | p={p_val:.4f}")

    out = {
        "lane": "a", "base_router": base, "rules": results, "referee_rule": cand,
        "referee": {"real_metric": round(real_metric, 3), "perm_mean": round(float(perm_metrics.mean()), 3),
                    "perm_p95": round(float(np.percentile(perm_metrics, 95)), 3), "p_value": round(p_val, 4),
                    "n_perm": n_perm},
        "runtime_s": round(time.time() - t0, 1),
    }
    outp = ROOT.parent / "runs" / "strat" / "nlcycle3_lane_a_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp} ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    run_lane_a()
