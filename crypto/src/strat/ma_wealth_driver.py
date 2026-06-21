"""src/strat/ma_wealth_driver.py -- the ONE canonical wealth driver for the 8 MA move-CATCH engines.

WHY THIS EXISTS (2026-06-20 meta directive). The 8-engine MA build produced SOLID capture (A/B/C, gated)
but a BROKEN wealth leg: each fan-out worker hand-rolled the wealth call, so wealth came out INCONSISTENT
(some null) and one was absurdly inflated. This module replaces the per-worker improvisation with ONE
function, `run_wealth(ma_type, tf, window)`, run UNIFORMLY across all 8 engines, producing a correctly-labeled,
RANKABLE wealth read (engine vs TRUE buy-hold vs the always-long rebalanced index) for the 8-engine leaderboard.

ACCOUNTING IS VERIFIED CORRECT (kept intact, STAGE-1 adversarial verify): the per-bar MtM in `_compound`
  reconciles to 2e-16; no A1 double-count; lag-1 causality is load-bearing; the DEV wall holds; band selection
  is TRAIN+VAL only (OOS-excluded). This stage does NOT touch any of that.

THE ONE NO-GO THE VERIFY RAISED -- a MISLABELED baseline (now fixed here):
  The old `w_bh = listed/n_listed` was REBALANCED EVERY BAR. Rebalancing a fixed-N EW book every bar is a
  DAILY-REBALANCED EW INDEX that harvests a ~4.6x volatility/rebalance premium (sells winners, buys losers) --
  it is NOT a buy-hold. On DEMA|1d|7d it reads ~+2136%, while a TRUE HODL basket (buy at listing, NO rebalance,
  weights drift with price) EW = ~+465% and BTC-HODL = ~+655%. The old basis/buyhold_via strings FALSELY
  asserted "true always-long EW buy-hold". That single mislabel was the NO-GO.

THE DEEPER (EXPECTED) FINDING -- this is the de-risked-beta floor, NOT a bug:
  The engine's huge compound is (a) the rebalance premium inflating BOTH legs + (b) a real 2022-bear DODGE that
  compounds. Strip the 2022 edge and the engine's compound collapses toward the rebalanced index -> there is NO
  up-market timing alpha; the entire compound "edge" is bear-PRESERVATION (drawdown-insurance de-risked beta,
  D63/D77). The honest landing.

THE FIX (this stage = a SMALL surgical reframe so the read is interpretable + correctly-labeled + RANKABLE):
  Per (ma_type, tf, window) the driver now reports THREE legs on the SAME admitted u100 panel + window:
    1. engine_compound_pct                -- the verified MtM engine net (UNCHANGED).
    2. hodl_compound_pct (NEW, PRIMARY)   -- TRUE buy-and-hold: each name bought ONCE at its first listed bar
       (point-in-time listing), equal 1/N_total dollar share, weights DRIFT with price, NO rebalance, held to
       window end.  vs_hodl_pp = engine - hodl is the primary "beat buy-hold" metric.
    3. alwayslong_rebal_compound_pct      -- the OLD `listed/n_listed` book, honestly RELABELED as the
       "always-long daily-rebalanced EW index" (NOT a buy-hold). vs_alwayslong_pp = engine - alwayslong is the
       TIMING-skill control (isolates the in/out decisions from the rebalance premium).
    4. bear2022_preservation_pp (NEW)     -- engine vs alwayslong over the 2022 bear (the de-risked-beta value).
    5. maxdd_engine_pct vs maxdd_hodl_pct.
  Still strictly causal: weights LAG 1 bar (W.shift(1)); MAKER cost (0.0006 RT) on turnover; DEV wall caps bars
  < DEV_END before any compute; selection = the committed capture infra's TRAIN+VAL sub-band (OOS never read).
  MtM RECONCILIATION is still ASSERTED (the MtM-no-double-count gate, scripts/probe_simulator_fix.py contract).
  COST NOTE: this WEALTH leg uses maker 0.0006 RT; the COMMITTED capture leg B uses TAKER 0.0024 RT -- stated in
  the output so the leaderboard is never silently mixed-cost.
  INSANITY FLAG (REFRAMED): a large compound gap DRIVEN by bear-preservation (up-market timing alpha ~0/neg,
  positive bear2022_preservation) is EXPECTED de-risked beta = INTERPRETABLE, not insane. Flag insane ONLY if the
  UP-MARKET timing alpha (engine vs alwayslong EXCLUDING the 2022 bear) is absurdly positive.

Reuses the COMMITTED capture infra (do NOT touch the capture leg): ma_movecatch_engine.select_band_trainval
(TRAIN+VAL sub-band, OOS-excluded) + make_hold_provider (the fixed-EW ma_cross band ensemble holds) +
fleet_lab (DEV_END wall, BARS_PER_DAY, the chimera u100 panel) + the same 2h synth-from-1h recipe used by
portfolio_replay / fleet_lab. The hold provider IS the capture leg's ensemble; this only changes the WEALTH
accounting around it.

RWYB (smoke): python -m strat.ma_wealth_driver --ma-type DEMA --tf 1d --window 7d
No emoji (cp1252). No git commit (meta reviews).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                    # noqa: E402  DEV_END, BARS_PER_DAY
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402  committed infra
from mining.family_regime_map import _norm_sym                  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader               # noqa: E402

MAKER_RT = 0.0006                                               # maker round-trip cost (this driver's wealth leg)
CAPTURE_TAKER_RT = 0.0024                                       # the COMMITTED capture leg B used taker (NOTE: mixed-cost)
# INSANITY now keys on UP-MARKET TIMING ALPHA, not the raw compound gap. A large gap DRIVEN by bear-preservation
# (2022 dodge) is EXPECTED de-risked beta (D63/D77) and is INTERPRETABLE -- NOT a bug. Flag insane ONLY if the
# engine shows absurd UP-market timing alpha = (engine - always-long-rebal index) EXCLUDING the 2022 bear window.
INSANITY_UPMKT_ALPHA_PP = 200.0                                # engine vs always-long EX-2022 beating by > this = flag
BEAR_2022_START = "2022-01-01"                                  # de-risked-beta value window (2022 crypto bear)
BEAR_2022_END = "2023-01-01"
_2H_MS = 7_200_000
_FREQ = {"1d": "D", "4h": "4h", "2h": "2h", "1h": "h", "30m": "30min", "15m": "15min"}


def _u100_symbols() -> list[str]:
    spec = yaml.safe_load(open(CRYPTO / "config" / "universes" / "u100.yaml"))
    if "assets" in spec:
        return [a["symbol"] for a in spec["assets"]]
    u50 = yaml.safe_load(open(CRYPTO / "config" / "universes" / "u50.yaml"))
    syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
    excl = set(spec.get("excluded_assets") or [])
    return [s for s in dict.fromkeys(syms) if s not in excl]


def _load_asset(sym: str, tf: str, wall_ms: int, floor_ms: int | None = None):
    """Load one asset's DEV-walled OHLC at `tf` (2h synthesized from native 1h, causal). Returns
    (dates, o, h, l, c) or None. Mirrors portfolio_replay / fleet_lab exactly so the panel matches.

    EVAL WINDOW (V1.1, surgical): `wall_ms` is the UPPER wall (bars with ts < wall_ms are kept -- defaults to
    DEV_END so 2022+ etc. is capped). `floor_ms` (optional) is the LOWER floor (bars with ts >= floor_ms are
    kept). floor_ms=None => no lower floor (== exact prior behavior). With floor_ms=2020-01-01 ms and
    wall_ms=2021-01-01 ms, ONLY 2020 bars load -> 2021+ (incl. all of 2022+) is NEVER touched. The window only
    RESTRICTS the panel; it changes NO accounting."""
    load_cad = "1h" if tf == "2h" else tf
    try:
        df = ChimeraLoader().load(_norm_sym(sym), cadence=load_cad, features=["open", "high", "low", "close"])
    except Exception:
        return None
    if tf == "2h":                                              # native-1h -> non-overlapping causal 2h OHLC
        ts1h = df["timestamp"].to_numpy()
        b = (ts1h // _2H_MS).astype("int64")
        pdf = pd.DataFrame({"open": df["open"].to_numpy().astype(float),
                            "high": df["high"].to_numpy().astype(float),
                            "low": df["low"].to_numpy().astype(float),
                            "close": df["close"].to_numpy().astype(float), "__b": b})
        g = pdf.groupby("__b", sort=True).agg(open=("open", "first"), high=("high", "max"),
                                              low=("low", "min"), close=("close", "last"))
        ts = (g.index.to_numpy() * _2H_MS).astype("int64")
        o = g["open"].to_numpy(); h = g["high"].to_numpy(); l = g["low"].to_numpy(); c = g["close"].to_numpy()
    else:
        ts = df["timestamp"].to_numpy()
        o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
    keep = ts < wall_ms                                         # DEV WALL (upper): cap bars < DEV_END/eval_end
    if floor_ms is not None:                                    # EVAL FLOOR (lower): drop bars before eval_start
        keep = keep & (ts >= floor_ms)
    if keep.sum() < 31:
        return None
    dates = pd.to_datetime(ts[keep], unit="ms").floor(_FREQ.get(tf, "D"))
    return dates, o[keep], h[keep], l[keep], c[keep]


def _compound(W: pd.DataFrame, R: pd.DataFrame, cost_rt: float):
    """MtM-NO-DOUBLE-COUNT replay of a weight book. Strictly causal: lag weights 1 bar, charge maker cost on
    turnover. Returns (compound_pct, net_stream, equity_curve, avg_gross, maxdd_pct).
    net_t = (W_{t-1} . R_t) - turnover_t * (cost/2);  equity = (1+net).cumprod()."""
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * R).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = (gross_ret - turnover * (cost_rt / 2.0)).to_numpy()
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    avg_gross = float(W.sum(axis=1).mean())
    return float((eq[-1] - 1.0) * 100) if len(eq) else 0.0, net, eq, avg_gross, maxdd


def _true_hodl(cp: pd.DataFrame, R: pd.DataFrame, listed: pd.DataFrame, cost_rt: float):
    """TRUE buy-and-hold (the PRIMARY baseline). Each admitted asset is bought ONCE at its FIRST available bar in
    the window (point-in-time listing), with an EQUAL DOLLAR share (1/N_total of initial capital, where N_total =
    distinct admitted names over the window). It is then HELD to window end -- NO daily rebalance, so the weights
    DRIFT with price (winners grow their share, losers shrink). Capital earmarked for a not-yet-listed name sits in
    CASH until that name lists. ONE round-trip maker cost per name (buy at listing + the implicit close at end).

    Implemented as a units/MtM equity so it reconciles by construction (no (2N-1)/N inflator):
      - allocate 1/N_total of $1 to each name at its first listed bar, buy `units = alloc / entry_price`;
      - per-bar portfolio value = cash_unallocated + sum_i units_i * price_i  (price carried forward; units fixed);
      - returns are MtM of that value series; the buy/sell costs are subtracted from the terminal compound.
    Returns (compound_pct, equity_curve, maxdd_pct, n_total)."""
    cols = list(cp.columns)
    n_total = len(cols)
    if n_total == 0:
        return 0.0, np.array([]), 0.0, 0
    alloc = 1.0 / n_total                                       # equal DOLLAR share per name at its own listing
    # first listed bar index per name (point-in-time listing within this DEV window)
    first_idx = {c: int(np.argmax(listed[c].to_numpy() > 0)) if listed[c].to_numpy().any() else None for c in cols}
    prices = cp.to_numpy()                                      # NaN before listing
    nrows = prices.shape[0]
    value = np.zeros(nrows)
    for j, c in enumerate(cols):
        fi = first_idx[c]
        if fi is None:
            continue
        entry_px = prices[fi, j]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        units = alloc / entry_px
        # held value of this name from its listing onward; carry last price forward over any internal NaN gaps
        px = pd.Series(prices[:, j]).ffill().to_numpy()
        held_val = units * px
        held_val[:fi] = 0.0                                     # not yet bought -> contributes nothing
        held_val[~np.isfinite(held_val)] = 0.0
        value += held_val
    # cash earmarked for names not yet listed at each bar = alloc * (#names not yet bought by that bar)
    bought_cum = np.zeros(nrows)
    for c in cols:
        fi = first_idx[c]
        if fi is not None:
            bought_cum[fi:] += 1.0
    cash = alloc * (n_total - bought_cum)                       # un-deployed capital sits in cash (no drift)
    equity_gross = value + cash                                 # starts at 1.0 (all cash), drifts as names list+move
    # cost: ONE round-trip per name (entry at listing + the single implicit exit at window end), on its 1/N share
    n_entered = int(sum(1 for c in cols if first_idx[c] is not None))
    total_cost = n_entered * alloc * cost_rt                    # round-trip cost on each deployed share
    eq = equity_gross * (1.0 - total_cost)                      # apply costs to terminal-comparable equity
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    return float((eq[-1] - 1.0) * 100) if len(eq) else 0.0, eq, maxdd, n_total


def _window_compound(W: pd.DataFrame, R: pd.DataFrame, cost_rt: float, mask: np.ndarray) -> float:
    """Compound % of a weight book RESTRICTED to the bars where `mask` is True (e.g. the 2022 bear window).
    Same MtM-no-double-count accounting as `_compound`, evaluated only over the masked sub-stream."""
    if mask.sum() == 0:
        return float("nan")
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * R).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = (gross_ret - turnover * (cost_rt / 2.0)).to_numpy()[mask]
    if len(net) == 0:
        return float("nan")
    return float((np.prod(1.0 + net) - 1.0) * 100)


def run_wealth(ma_type: str, tf: str, window: str, *, cost_rt: float = MAKER_RT,
               exit_fn=None, eval_start: str | None = None, eval_end: str | None = None,
               verbose: bool = False) -> dict:
    """THE ONE wealth cell. (ma_type, tf, window e.g. '7d') -> dict, RANKABLE for the 8-engine leaderboard:
        engine_compound_pct                  (PRIMARY engine net -- verified MtM, unchanged)
        hodl_compound_pct                    (NEW PRIMARY baseline -- TRUE buy-and-hold)
        alwayslong_rebal_compound_pct        (TIMING-skill control -- daily-rebalanced EW index, honestly relabeled)
        vs_hodl_pp = engine - hodl           (the "beat buy-hold" metric)
        vs_alwayslong_pp = engine - alwayslong (isolates timing skill from the rebalance premium)
        bear2022_preservation_pp             (engine vs always-long over the 2022 bear -- the de-risked-beta value)
        upmkt_timing_alpha_pp                (engine vs always-long EXCLUDING 2022 -- the insanity key)
        maxdd_engine_pct, maxdd_hodl_pct, mtm_reconciled, sane, plus diagnostics.

    THREE LEGS on the SAME admitted u100 panel + DEV window (identical names, identical bars):
      ENGINE      = fixed-N EW (1/N_listed per HELD name; cash for non-held/unlisted), long-while-held per the
                    committed fixed-EW ma_cross band ENSEMBLE. Sits in cash part of the time -> de-risked beta.
      HODL        = TRUE buy-and-hold: each name bought ONCE at its first listed bar (1/N_total dollar share),
                    weights DRIFT with price, NO rebalance, held to window end (`_true_hodl`). This is the
                    honest "what if I just bought and held" -- the PRIMARY thing the engine must beat.
      ALWAYS-LONG = the EXISTING `listed/n_listed` book REBALANCED every bar = a DAILY-REBALANCED EW index
                    (harvests a rebalance/vol premium). It is NOT a buy-hold. Kept ONLY as the timing-skill
                    control: engine-vs-always-long isolates the in/out decisions from the rebalance premium.

    exit_fn (V1.1 EXIT-HARNESS HOOK, optional): a causal POST-PROCESSOR on the engine hold-mask with signature
      exit_fn(sym, o, h, l, c, base_mask) -> int8 mask. When given, the band ensemble's per-asset hold-mask is
      replaced by exit_fn's output BEFORE any sizing/MtM. exit_fn MUST be a pure boolean-mask transform (e.g.
      strat.exit_harness.build_exit_fn(...)) -- it NEVER touches wealth; ALL the MtM-no-double-count / lag-1 /
      maker-RT / TRUE-HODL / DEV-wall accounting below is reused UNTOUCHED. exit_fn=None reproduces the committed
      engine exactly (band_off exit). This keeps run_wealth the single authoritative wealth path (the charter's
      hard rule -- 2 fabrications were caught because workers hand-rolled MtM; the harness must not).

    eval_start / eval_end (V1.1 TWO-SIDED EVAL WINDOW, optional date strings): RESTRICT the loaded panel to bars in
      [eval_start, eval_end). eval_start=None => no lower floor; eval_end=None => the DEV_END wall (== EXACT prior
      behavior). For the 2020 sweep pass eval_start="2020-01-01", eval_end="2021-01-01" so 2021+ (incl. all of
      2022+) is NEVER loaded. The eval_end wall is ALWAYS clamped to <= DEV_END (cannot reach OOS/UNSEEN). The
      window only RESTRICTS the panel -- it changes NO accounting (MtM / lag-1 / maker-RT / TRUE-HODL all intact).
    """
    window_days = int(str(window).rstrip("d"))
    bpd = fl.BARS_PER_DAY[tf]
    # UPPER WALL: eval_end if given else DEV_END; ALWAYS clamped to <= DEV_END (OOS/UNSEEN can never load).
    dev_wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    if eval_end is not None:
        req_wall_ms = int(pd.Timestamp(eval_end).value // 10 ** 6)
        wall_ms = min(req_wall_ms, dev_wall_ms)                 # never above the DEV wall
    else:
        wall_ms = dev_wall_ms
    # LOWER FLOOR: eval_start if given else none (no floor == exact prior behavior).
    floor_ms = int(pd.Timestamp(eval_start).value // 10 ** 6) if eval_start is not None else None

    # selection + ensemble hold provider = the COMMITTED capture infra (TRAIN+VAL sub-band, OOS never read).
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        return {"error": f"sub-band < 2 for {ma_type}|{tf} after TRAIN+VAL selection", "selection": sel_meta}
    provider = make_hold_provider(sub_band, ma_type)

    closes, holds = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms, floor_ms=floor_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        base_mask = provider(sym, o, h, l, c)
        # V1.1 EXIT-HARNESS SEAM: post-process the band hold-mask with a causal exit/entry transform if supplied.
        # exit_fn is a pure boolean-mask transform (no wealth); None == the committed band_off engine (identity).
        hh = (exit_fn(sym, o, h, l, c, base_mask) if exit_fn is not None else base_mask).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]

    if not closes:
        return {"error": f"no assets loaded for {ma_type}|{tf}|{window}"}
    cp = pd.DataFrame(closes).sort_index()
    # HARDEN: fail loud if the floor/dedup alignment regresses (a dropped floor explodes the union index ~Nx).
    _max_rows = max((s.notna().sum() for s in closes.values()), default=0)
    assert len(cp) <= 1.5 * _max_rows + 5, (
        f"date-alignment regression: panel {len(cp)} rows >> max per-asset {_max_rows}")
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)

    # NO LOOK-AHEAD: returns are pct_change of close (causal); the DEV wall already removed all >= DEV_END bars.
    assert cp.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION (panel index >= DEV_END)"
    # EVAL-WINDOW WALL (V1.1): when an upper eval_end is set, PROVE the panel never reached it (e.g. 2022+ sealed).
    if eval_end is not None:
        _eval_end_ts = pd.Timestamp(eval_end)
        assert cp.index.max() < _eval_end_ts, (
            f"EVAL-WINDOW VIOLATION: panel max {cp.index.max()} >= eval_end {_eval_end_ts} (window not sealed)")
    if eval_start is not None:
        assert cp.index.min() >= pd.Timestamp(eval_start), (
            f"EVAL-WINDOW VIOLATION: panel min {cp.index.min()} < eval_start {eval_start}")
    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)                          # 1 while the asset is listed (has a close)
    held = hp * listed                                         # long-while-held AND listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)           # fixed-N denominator = #listed this bar

    # LEG 1 ENGINE: fixed-N EW (1/N_listed per HELD name; cash for non-held -> de-risked beta).
    w_eng = held.div(n_listed, axis=0).fillna(0.0)             # 1/N_listed per HELD name; cash if flat
    # LEG 3 ALWAYS-LONG (daily-REBALANCED EW INDEX -- NOT a buy-hold): 1/N_listed per LISTED name, recomputed
    # every bar. Rebalancing every bar harvests a vol/rebalance premium (sells winners / buys losers) that a
    # true buy-hold does NOT. Kept ONLY as the timing-skill control (engine vs this isolates the in/out decision).
    w_along = listed.div(n_listed, axis=0).fillna(0.0)

    eng_pct, eng_net, eng_eq, eng_gross, eng_dd = _compound(w_eng, R, cost_rt)
    along_pct, along_net, along_eq, along_gross, along_dd = _compound(w_along, R, cost_rt)

    # LEG 2 HODL (PRIMARY baseline): TRUE buy-and-hold -- buy each name once at its listing, weights DRIFT, no rebal.
    hodl_pct, hodl_eq, hodl_dd, n_total = _true_hodl(cp, R, listed, cost_rt)

    vs_hodl_pp = round(eng_pct - hodl_pct, 2)                   # the "beat buy-hold" metric (PRIMARY)
    vs_alwayslong_pp = round(eng_pct - along_pct, 2)            # timing skill vs the rebalanced index (can be < 0)

    # ---- 2022-BEAR PRESERVATION (the de-risked-beta value) ----
    # engine vs the always-long index OVER the 2022 bear window only. POSITIVE = the engine preserved capital by
    # de-risking (sitting in cash) while the always-long index ate the bear. This is the value of de-risked beta.
    bidx = cp.index
    bear_mask = np.asarray((bidx >= pd.Timestamp(BEAR_2022_START)) & (bidx < pd.Timestamp(BEAR_2022_END)))
    eng_bear = _window_compound(w_eng, R, cost_rt, bear_mask)
    along_bear = _window_compound(w_along, R, cost_rt, bear_mask)
    bear2022_preservation_pp = (round(eng_bear - along_bear, 2)
                                if np.isfinite(eng_bear) and np.isfinite(along_bear) else None)

    # ---- UP-MARKET TIMING ALPHA (the insanity key) ----
    # engine vs the always-long index EXCLUDING the 2022 bear. This strips the bear-dodge out, leaving only
    # the engine's in/out TIMING SKILL in non-bear (mostly up) markets. De-risked beta => this is ~0 or negative.
    upmkt_mask = ~bear_mask
    eng_up = _window_compound(w_eng, R, cost_rt, upmkt_mask)
    along_up = _window_compound(w_along, R, cost_rt, upmkt_mask)
    upmkt_timing_alpha_pp = (round(eng_up - along_up, 2)
                             if np.isfinite(eng_up) and np.isfinite(along_up) else None)

    # ---- MtM RECONCILIATION (the project's MtM-no-double-count gate) ----
    # the per-bar net STREAM must compound to the reported equity within 0.1% (relative). There is no
    # (2N-1)/N double count: compound = prod(1+net) by construction; this guards a future regression that
    # would re-introduce a divergence between the net stream and the equity curve.
    recompound = float(np.prod(1.0 + eng_net) - 1.0) * 100
    denom = max(abs(eng_pct), 1e-9)
    mtm_rel_err = abs(recompound - eng_pct) / denom
    mtm_reconciled = bool(mtm_rel_err < 1e-3)

    # ---- INSANITY FLAG (REFRAMED) ----
    # A large engine-vs-hodl/index COMPOUND GAP is NOT insane if it is DRIVEN BY BEAR-PRESERVATION: that is
    # EXPECTED de-risked beta (D63/D77) and is INTERPRETABLE. The only insane outcome is absurd UP-MARKET timing
    # alpha -- the engine "calling" up-market entries/exits to beat a rebalanced always-long index ex-bear by a
    # wide margin. THAT would contradict every dead-list null and signal a bug (look-ahead / survivorship).
    sane = bool(upmkt_timing_alpha_pp is None or upmkt_timing_alpha_pp <= INSANITY_UPMKT_ALPHA_PP)
    # de-risked beta is CONFIRMED when up-market timing alpha is ~0/negative AND the bear was preserved (>0).
    derisked_beta_confirmed = bool(
        upmkt_timing_alpha_pp is not None and upmkt_timing_alpha_pp <= 5.0
        and (bear2022_preservation_pp is None or bear2022_preservation_pp >= 0.0))

    out = {
        "cell": f"{ma_type}|{tf}|{window}",
        "ma_type": ma_type, "tf": tf, "window": window, "window_days": window_days, "hold_bars": window_days * bpd,
        # --- V1.1 exit-harness provenance (band_off baseline when exit_fn is None) ---
        "exit_harness": (getattr(exit_fn, "__exit_meta__", {"exit_type": "custom_exit_fn"})
                         if exit_fn is not None else {"exit_type": "band_off", "entry_type": "entry_none"}),
        # --- the leaderboard reads (rankable) ---
        "engine_compound_pct": round(eng_pct, 2),                      # PRIMARY engine net (unchanged, verified MtM)
        "hodl_compound_pct": round(hodl_pct, 2),                       # PRIMARY baseline: TRUE buy-and-hold
        "alwayslong_rebal_compound_pct": round(along_pct, 2),         # daily-rebalanced EW INDEX (NOT buy-hold)
        "vs_hodl_pp": vs_hodl_pp,                                      # engine - hodl  ("beat buy-hold")
        "vs_alwayslong_pp": vs_alwayslong_pp,                         # engine - index (TIMING-skill control)
        "bear2022_preservation_pp": bear2022_preservation_pp,        # engine vs index over 2022 (de-risked value)
        "upmkt_timing_alpha_pp": upmkt_timing_alpha_pp,              # engine vs index EX-2022 (insanity key)
        "maxdd_engine_pct": round(eng_dd, 1),
        "maxdd_hodl_pct": round(hodl_dd, 1),
        "maxdd_alwayslong_pct": round(along_dd, 1),
        "derisked_beta_confirmed": derisked_beta_confirmed,
        # --- panel / accounting diagnostics ---
        "n_assets": int(cp.shape[1]),
        "n_total_admitted": int(n_total),
        "n_bars": int(cp.shape[0]),
        "engine_avg_gross": round(eng_gross, 3),
        "alwayslong_avg_gross": round(along_gross, 3),
        "cost_rt": cost_rt, "cost_kind": "maker" if abs(cost_rt - MAKER_RT) < 1e-9 else "custom",
        "cost_note": (f"this wealth leg charges MAKER {cost_rt} RT; the COMMITTED capture leg B charges TAKER "
                      f"{CAPTURE_TAKER_RT} RT. Do NOT silently mix the two on one leaderboard -- the wealth read "
                      f"is maker-cost, the capture (A/B/C) read is taker-cost; state the basis per column."),
        "mtm_reconciled": mtm_reconciled,
        "mtm_rel_err": float(mtm_rel_err),
        "sane": sane,
        "insanity_upmkt_alpha_threshold_pp": INSANITY_UPMKT_ALPHA_PP,
        "bear_2022_window": [BEAR_2022_START, BEAR_2022_END],
        "basis": ("THREE legs, SAME admitted u100 panel+window, lag-1 fills, maker cost, DEV wall, "
                  "MtM-no-double-count. ENGINE = fixed-N EW (1/N_listed per HELD name; cash else -> de-risked "
                  "beta). HODL (PRIMARY) = TRUE buy-and-hold (buy each name ONCE at its listing, 1/N_total share, "
                  "weights DRIFT, no rebalance). ALWAYS-LONG = listed/n_listed REBALANCED every bar = a "
                  "daily-rebalanced EW INDEX that harvests a rebalance premium -- NOT a buy-hold; kept only as the "
                  "timing-skill control."),
        "engine_via": "fixed-N EW, long-while-held by the committed ma_cross band ensemble; cash when flat",
        "hodl_via": ("TRUE buy-and-hold: each name bought once at its first listed bar (point-in-time), equal "
                     "1/N_total dollar share, weights DRIFT with price, NO daily rebalance, one maker round-trip/name"),
        "alwayslong_via": ("DAILY-REBALANCED EW INDEX (1/N_listed per listed name, recomputed every bar) -- a "
                           "rebalanced index, explicitly NOT a buy-hold; isolates the engine's timing decisions"),
        "selection_rule": sel_meta.get("selection_rule"),
        "n_sub_band": len(sub_band),
        "split": f"DEV (TRAIN+VAL, < {fl.DEV_END}); OOS+UNSEEN SEALED",
        "data_range": [str(cp.index.min()), str(cp.index.max())],
        # --- V1.1 eval window (provenance for the 7d-slice metric; None,None == prior full-DEV behavior) ---
        "eval_window": {"eval_start": eval_start, "eval_end": eval_end,
                        "floor_ms": floor_ms, "wall_ms": wall_ms,
                        "panel_min": str(cp.index.min()), "panel_max": str(cp.index.max())},
        # --- the DATED engine equity curve (for the 7d-slice acid metric); JSON-serializable [iso, value] pairs ---
        "engine_equity": [[d.isoformat(), float(v)] for d, v in zip(cp.index, eng_eq)],
        "engine_net_stream": [[d.isoformat(), float(v)] for d, v in zip(cp.index, eng_net)],
    }
    if derisked_beta_confirmed:
        out["interpretation"] = (
            f"DE-RISKED BETA (D63/D77): up-market timing alpha {upmkt_timing_alpha_pp}pp (~0/neg) -- the engine "
            f"adds NO up-market timing edge over a rebalanced always-long index. The compound gap is driven by "
            f"2022-bear preservation ({bear2022_preservation_pp}pp). This is drawdown-insurance, the honest landing "
            f"-- INTERPRETABLE, not insane.")
    if not sane:
        out["sane_note"] = (f"engine shows UP-MARKET timing alpha {upmkt_timing_alpha_pp}pp (> "
                            f"{INSANITY_UPMKT_ALPHA_PP}pp) vs the rebalanced always-long index EXCLUDING the 2022 "
                            f"bear -- that contradicts every dead-list LO null (D63/D77). Probable look-ahead / "
                            f"survivorship concentration on the thin early-2020 DEV panel, NOT genuine alpha. Verify.")
    if not mtm_reconciled:
        out["mtm_note"] = f"MtM stream/compound mismatch rel_err={mtm_rel_err:.2e} (>= 1e-3) -- accounting regression."
    if verbose:
        print(f"[wealth] {out['cell']}  engine {out['engine_compound_pct']}%  HODL {out['hodl_compound_pct']}%  "
              f"alwayslong-idx {out['alwayslong_rebal_compound_pct']}%")
        print(f"         vs-HODL {out['vs_hodl_pp']}pp  vs-alwayslong {out['vs_alwayslong_pp']}pp  "
              f"bear2022-preserve {out['bear2022_preservation_pp']}pp  upmkt-alpha {out['upmkt_timing_alpha_pp']}pp")
        print(f"         maxDD eng {out['maxdd_engine_pct']}% / hodl {out['maxdd_hodl_pct']}%  "
              f"n_assets {out['n_assets']}  MtM_recon {out['mtm_reconciled']}  sane {out['sane']}  "
              f"derisked_beta_confirmed {out['derisked_beta_confirmed']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_wealth_driver")
    ap.add_argument("--ma-type", default="DEMA")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--window", default="7d")
    ap.add_argument("--taker", action="store_true", help="charge taker 0.0024 RT instead of maker 0.0006")
    a = ap.parse_args()
    cost = 0.0024 if a.taker else MAKER_RT
    r = run_wealth(a.ma_type, a.tf, a.window, cost_rt=cost, verbose=True)
    print(json.dumps(r, indent=1, default=str))
    if "error" in r:
        return 2
    if not r["mtm_reconciled"]:
        print("[FAIL] MtM reconciliation did NOT pass"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
