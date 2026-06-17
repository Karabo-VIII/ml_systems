"""
!!! QUARANTINED -- DO NOT USE / DO NOT TRUST RESULTS (2026-06-13) !!!
====================================================================
This v1 produced a "+0.90 lead, 3/3" Test-1 result that was an ARTIFACT:
definitional-lead + hindsight mover-day conditioning (outcome-conditioning).
Kept ONLY as a cautionary record. The honest re-test is `dib_mover_axis_v2.py`,
and the axis was ultimately CLOSED by the breadth+real-cost kill in
`flow_direction_breadth_probe.py` (dead-list D75 /
FINDING_chart_type_axis_orderflow_2026-06-13). Do not import, run, or cite this file.
====================================================================

DIB MOVER-AXIS GO/NO-GO  (standalone, do-not-commit pilot)
==========================================================

Re-attack the daily-mover problem on DOLLAR IMBALANCE BARS (DIB) -- information
bars that sample faster when order flow is imbalanced.

PRIOR SPRINT (1m TIME bars) concluded:
  - TIMING null  : burst is COINCIDENT not leading (median lead ~20-30 min;
                   the +1.5% trigger detects the move underway).
  - CONTINUATION : directional OOS AUC 0.52 (internal info-bound; clear bar 0.58).
  - CAPTURE      : entry-bound; only ~31% of triggers convert; ~0 net.

USER HYPOTHESIS: on imbalance bars these may differ -- a DIB COMPRESSES clock
time during a flow burst, so "the first bar of the burst" is much EARLIER in
clock time than a 1m time bar. A DIB trigger could be genuinely LEADING.

This is a 3-asset PILOT (BTC + ETH + PEPE). GO/NO-GO on whether the
imbalance-bar AXIS is worth building for the full universe -- not a deployable
result. Adversarial stance: 3 assets = tiny n + multiple-comparison risk;
require CLEAN + mechanism-plausible, not one lucky asset, vs a shuffled /
time-bar null.

METHODOLOGY (held-out, no look-ahead, cost-honest)
--------------------------------------------------
Data: data/processed/chimera/dib/<sym>_v51_chimera_dib_20260529.parquet
      (BTC, ETH, PEPE). 204 cols incl. norm_vpin, norm_flow_imbalance, buy_vol,
      sell_vol, tick_count, target_return_{1,4,16,64}, timestamp(ms), close.

Both triggers see the IDENTICAL price path (the DIB close series). The DIB
trigger uses the native imbalance feature on the DIB index; the time-bar
baseline uses a CAUSAL forward-fill of DIB close onto a fixed clock grid
(1-min), then the classic +THRESH% momentum trigger. The ONLY difference is
bar-type / trigger rule -> a clean test of "does DIB give more lead".

  Daily mover  : a UTC day whose intraday move (first DIB bar -> the day's
                 extreme in the move direction) is >= MOVE_THRESH (5%).
                 Direction = sign of (last bar - first bar).
  DIB trigger  : the FIRST DIB bar on the mover day at which a run of RUN_LEN
                 consecutive same-direction flow_imbalance bars completes AND
                 cumulative-from-open in the move direction is still small
                 (signal, not confirmation). Causal: uses bars <= trigger only.
  Time trigger : first 1-min clock bar at which trailing TB_WIN-min return in
                 the move direction first exceeds TB_THRESH (=1.5%). Causal.
  LEAD         : at each trigger, "move still ahead" =
                   price lead  = (day_extreme - trigger_price)/trigger_price
                                 in the move direction (fraction of the WHOLE
                                 move that remains), and
                   clock lead  = minutes from trigger to the bar that realizes
                                 the day extreme.
                 More lead = better. The crux: DIB price-lead vs time-bar.

TRAIN/OOS split: chronological 60/40 per asset (purge gap of PURGE_DAYS days).
  - Continuation classifier (logistic) is FIT ON TRAIN ONLY, scored on OOS.
  - Lead + capture stats reported on OOS movers only (out of sample).

NULLS:
  - time-bar baseline (the prior sprint's apparatus, same price path).
  - shuffled-label null for continuation AUC (target permuted within OOS).
  - random-entry null for capture (enter at a random bar, same horizon/cost).

COSTS: round-trip taker+slippage charged per trade. Frequency = DIB trigger
count = the cost. Reported per asset.

Run:  python src/mining/dib_mover_axis.py
"""

import sys
import numpy as np
import polars as pl

# ----------------------------- CONFIG --------------------------------------
from pathlib import Path as _P
DIB_DIR = str(_P(__file__).resolve().parents[2] / "data/processed/chimera/dib")  # crypto/data/...
ASSETS = {
    "btcusdt": f"{DIB_DIR}/btcusdt_v51_chimera_dib_20260529.parquet",
    "ethusdt": f"{DIB_DIR}/ethusdt_v51_chimera_dib_20260529.parquet",
    "pepeusdt": f"{DIB_DIR}/pepeusdt_v51_chimera_dib_20260529.parquet",
}

MOVE_THRESH = 0.05      # daily mover = >= 5% intraday move
RUN_LEN = 3             # DIB trigger: run of N consecutive same-sign flow bars
FLOW_COL = "norm_flow_imbalance"   # signed, normalized (std~1)
VPIN_COL = "norm_vpin"
EARLY_CAP = 0.015       # DIB trigger only valid while cum-from-open in dir < this
TB_WIN_MIN = 30         # time-bar trailing window (minutes) for momentum trigger
TB_THRESH = 0.015       # time-bar +1.5% trigger (prior sprint apparatus)
PURGE_DAYS = 5          # purge gap between train and oos (days)
TRAIN_FRAC = 0.60       # chronological 60/40 split

# capture / continuation horizon: hold from trigger until day extreme is realized
# (oracle exit at the extreme is reported as a CEILING; a fixed-policy exit is the
#  HONEST realizable number). cost is round-trip.
COST_RT = {             # round-trip taker+slippage (fraction), per asset
    "btcusdt": 0.0008,  # ~4bps/side taker + slip
    "ethusdt": 0.0010,
    "pepeusdt": 0.0020, # wider for the small-cap
}
HOLD_BARS = 16          # fixed-policy exit: hold N DIB bars after entry (honest)
SEED = 7
np.random.seed(SEED)


# ----------------------------- LOADER --------------------------------------
def load(sym, path):
    df = pl.read_parquet(path).sort("timestamp")
    ts = df["timestamp"].to_numpy().astype(np.int64)
    out = {
        "ts": ts,
        "close": df["close"].to_numpy().astype(float),
        "flow": df[FLOW_COL].to_numpy().astype(float),
        "vpin": df[VPIN_COL].to_numpy().astype(float),
        "buy_vol": df["buy_vol"].to_numpy().astype(float),
        "sell_vol": df["sell_vol"].to_numpy().astype(float),
        "tick_count": df["tick_count"].to_numpy().astype(float),
        "norm_tick": df["norm_tick_count"].to_numpy().astype(float),
        "day": (ts // 1000 // 86400).astype(np.int64),
    }
    return out


# ------------------------- DAILY MOVER DETECTION ---------------------------
def find_movers(d):
    """Return list of (day, dir, i_open, i_ext, ext_price, open_price, move_frac).
    i_* are global indices into the DIB arrays. move = open->extreme in dir."""
    ts, close, day = d["ts"], d["close"], d["day"]
    movers = []
    for dd in np.unique(day):
        idx = np.where(day == dd)[0]
        if len(idx) < 6:
            continue
        c = close[idx]
        o = c[0]
        last = c[-1]
        direction = 1 if last >= o else -1
        if direction > 0:
            i_ext_local = int(np.argmax(c))
            ext = c[i_ext_local]
            move = ext / o - 1.0
        else:
            i_ext_local = int(np.argmin(c))
            ext = c[i_ext_local]
            move = o / ext - 1.0   # positive magnitude
        if move >= MOVE_THRESH and i_ext_local > 0:
            movers.append({
                "day": int(dd),
                "dir": direction,
                "i_open": int(idx[0]),
                "i_ext": int(idx[i_ext_local]),
                "idx": idx,
                "i_ext_local": i_ext_local,
                "open_price": float(o),
                "ext_price": float(ext),
                "move": float(move),
            })
    return movers


# ------------------------ DIB IMBALANCE TRIGGER ----------------------------
def dib_trigger(d, mv):
    """First DIB bar on the mover day where a run of RUN_LEN consecutive
    same-sign-as-move flow_imbalance bars completes AND cum-from-open in the
    move direction is still < EARLY_CAP. Causal. Returns local-in-day index or
    None."""
    idx = mv["idx"]
    c = d["close"][idx]
    f = d["flow"][idx]
    o = c[0]
    direction = mv["dir"]
    # cum move from open in the move direction (positive = toward the move)
    cum_dir = direction * (c / o - 1.0)
    sgn = np.sign(f) == direction
    run = 0
    for k in range(len(idx)):
        run = run + 1 if sgn[k] else 0
        if run >= RUN_LEN and cum_dir[k] < EARLY_CAP:
            return k   # local index where the run completes
    return None


# ------------------------ TIME-BAR (+1.5%) TRIGGER -------------------------
def time_trigger(d, mv):
    """Causal forward-fill of DIB close onto a 1-min clock grid; first grid bar
    where trailing TB_WIN_MIN-min return in move dir first exceeds TB_THRESH.
    Returns (trigger_clock_ms, trigger_price) or None."""
    idx = mv["idx"]
    t = d["ts"][idx]
    c = d["close"][idx]
    direction = mv["dir"]
    t0 = (t[0] // 60000) * 60000
    grid = np.arange(t0, t[-1] + 60000, 60000)
    if len(grid) < TB_WIN_MIN + 2:
        return None
    gi = np.searchsorted(t, grid, side="right") - 1
    gi = np.clip(gi, 0, len(c) - 1)
    gc = c[gi]                       # causal price on clock grid
    w = TB_WIN_MIN
    trailing = np.full(len(gc), 0.0)
    trailing[w:] = direction * (gc[w:] / gc[:-w] - 1.0)
    hit = np.where(trailing >= TB_THRESH)[0]
    if len(hit) == 0:
        return None
    j = hit[0]
    return float(grid[j]), float(gc[j])


# -------------------------- LEAD MEASUREMENT -------------------------------
def lead_stats(d, mv, k_local):
    """At DIB trigger local index k_local, how much of the move is ahead."""
    idx = mv["idx"]
    c = d["close"][idx]
    t = d["ts"][idx]
    direction = mv["dir"]
    p_trig = c[k_local]
    ext = mv["ext_price"]
    t_trig = t[k_local]
    t_ext = t[mv["i_ext_local"]]
    if direction > 0:
        price_ahead = (ext - p_trig) / p_trig
    else:
        price_ahead = (p_trig - ext) / p_trig
    clock_ahead_min = (t_ext - t_trig) / 1000.0 / 60.0
    frac_of_move_ahead = price_ahead / mv["move"] if mv["move"] > 0 else np.nan
    return price_ahead, clock_ahead_min, frac_of_move_ahead, t_trig


def time_lead_stats(d, mv, trig_clock_ms, trig_price):
    idx = mv["idx"]
    direction = mv["dir"]
    ext = mv["ext_price"]
    t_ext = d["ts"][mv["i_ext"]]
    if direction > 0:
        price_ahead = (ext - trig_price) / trig_price
    else:
        price_ahead = (trig_price - ext) / trig_price
    clock_ahead_min = (t_ext - trig_clock_ms) / 1000.0 / 60.0
    frac_ahead = price_ahead / mv["move"] if mv["move"] > 0 else np.nan
    return price_ahead, clock_ahead_min, frac_ahead


# --------------------------- LOGISTIC (no sklearn dep) ---------------------
def fit_logistic(X, y, l2=1.0, iters=300, lr=0.1):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    n, p = Xs.shape
    w = np.zeros(p)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        pr = 1.0 / (1.0 + np.exp(-z))
        g = pr - y
        gw = Xs.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return (w, b, mu, sd)


def predict_logistic(model, X):
    w, b, mu, sd = model
    Xs = (np.asarray(X, float) - mu) / sd
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # rank-based AUC
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# ----------------------------- MAIN ----------------------------------------
def run_asset(sym, path):
    d = load(sym, path)
    movers = find_movers(d)
    n_days = len(np.unique(d["day"]))
    print(f"\n========================  {sym.upper()}  ========================")
    print(f"DIB bars={len(d['ts'])}  days={n_days}  bars/day~{len(d['ts'])/n_days:.0f}  "
          f"movers(|>={MOVE_THRESH*100:.0f}%)={len(movers)}")
    if len(movers) < 10:
        print("  too few movers -- skip")
        return None

    # chronological 60/40 split by day, purge gap
    udays = np.array(sorted(set(m["day"] for m in movers)))
    split_day = np.percentile([m["day"] for m in movers], TRAIN_FRAC * 100)
    train = [m for m in movers if m["day"] <= split_day - PURGE_DAYS]
    oos = [m for m in movers if m["day"] > split_day]
    print(f"  split: train movers={len(train)}  oos movers={len(oos)}  (purge {PURGE_DAYS}d)")

    # ---------- TEST 1: LEAD (DIB vs time-bar), OOS movers ----------
    dib_fracs, dib_clock, dib_price = [], [], []
    tb_fracs, tb_clock, tb_price = [], [], []
    n_dib_trig = 0
    n_tb_trig = 0
    paired = []   # rows where BOTH triggers fire -> apples-to-apples
    for mv in oos:
        kt = dib_trigger(d, mv)
        tt = time_trigger(d, mv)
        if kt is not None:
            n_dib_trig += 1
            pa, ca, fa, _ = lead_stats(d, mv, kt)
            dib_fracs.append(fa); dib_clock.append(ca); dib_price.append(pa)
        if tt is not None:
            n_tb_trig += 1
            pa2, ca2, fa2 = time_lead_stats(d, mv, tt[0], tt[1])
            tb_fracs.append(fa2); tb_clock.append(ca2); tb_price.append(pa2)
        if kt is not None and tt is not None:
            pa, ca, fa, _ = lead_stats(d, mv, kt)
            pa2, ca2, fa2 = time_lead_stats(d, mv, tt[0], tt[1])
            paired.append((fa, fa2, pa, pa2))

    def msum(a):
        a = np.array(a, float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return "n=0"
        return f"n={len(a)} med={np.median(a):+.2f} mean={np.mean(a):+.2f}"

    print("\n  --- TEST 1: LEAD (fraction of the move STILL AHEAD at trigger) ---")
    print(f"    DIB imbalance trigger : fires on {n_dib_trig}/{len(oos)} oos movers  "
          f"frac_ahead {msum(dib_fracs)}")
    print(f"    DIB clock-ahead (min) : {msum(dib_clock)}")
    print(f"    TIME +{TB_THRESH*100:.1f}% trigger  : fires on {n_tb_trig}/{len(oos)} oos movers  "
          f"frac_ahead {msum(tb_fracs)}")
    print(f"    TIME clock-ahead (min): {msum(tb_clock)}")
    if paired:
        pf = np.array(paired)
        d_frac = pf[:, 0] - pf[:, 1]
        # paired sign test
        wins = int((d_frac > 0).sum()); n = len(d_frac)
        from math import comb
        # two-sided binomial p (exact)
        k = max(wins, n - wins)
        p = 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n <= 60 else np.nan
        print(f"    PAIRED (both fire, n={n}): DIB more lead in {wins}/{n} "
              f"(mean frac_ahead diff {d_frac.mean():+.3f}, sign-test p={p if np.isfinite(p) else float('nan'):.3f})")
    lead_dib_med = np.nanmedian(np.array(dib_fracs, float)) if dib_fracs else np.nan
    lead_tb_med = np.nanmedian(np.array(tb_fracs, float)) if tb_fracs else np.nan

    # ---------- TEST 2: CONTINUATION (OOS AUC) ----------
    # Build a classification dataset at the DIB trigger bar:
    #   label = 1 if, AFTER the trigger, the move continues in dir by >= CONT_THRESH
    #           before reversing by REV_THRESH (a triple-barrier-style continuation).
    #   features = order-flow STATE at the trigger (causal): vpin, flow, buy/sell
    #              ratio, tick acceleration, run length, cum-so-far.
    CONT_THRESH = 0.01
    def build_cont(rows):
        Xs, ys = [], []
        for mv in rows:
            kt = dib_trigger(d, mv)
            if kt is None:
                continue
            idx = mv["idx"]
            g = idx[kt]               # global trigger index
            direction = mv["dir"]
            c = d["close"]
            p0 = c[g]
            # forward path within the day after the trigger
            fwd = idx[idx > g]
            if len(fwd) < 2:
                continue
            cf = c[fwd]
            dir_ret = direction * (cf / p0 - 1.0)
            # label: did it continue >= CONT_THRESH (before, in the path, dipping below 0 doesn't disqualify;
            # honest forward continuation: max forward dir-return >= CONT_THRESH)
            label = 1 if dir_ret.max() >= CONT_THRESH else 0
            # causal features at trigger
            bv = d["buy_vol"][g]; sv = d["sell_vol"][g]
            ratio = bv / (bv + sv + 1e-9)
            tick = d["norm_tick"][g]
            vpin = d["vpin"][g]
            flow = d["flow"][g]
            cum_dir = direction * (p0 / mv["open_price"] - 1.0)
            # tick acceleration over last few DIB bars
            kk = max(0, kt - 3)
            tick_prev = d["norm_tick"][idx[kk]]
            tick_acc = tick - tick_prev
            Xs.append([direction * flow, vpin, direction * (ratio - 0.5), tick, tick_acc, cum_dir])
            ys.append(label)
        return np.array(Xs), np.array(ys)

    Xtr, ytr = build_cont(train)
    Xte, yte = build_cont(oos)
    cont_auc = np.nan; cont_base = np.nan; cont_null = np.nan
    if len(Xtr) >= 20 and len(Xte) >= 15 and ytr.min() != ytr.max() and yte.min() != yte.max():
        model = fit_logistic(Xtr, ytr, l2=2.0)
        sc = predict_logistic(model, Xte)
        cont_auc = auc(yte, sc)
        cont_base = yte.mean()
        # shuffled-label null AUC (permute oos labels)
        nulls = []
        for _ in range(200):
            yp = np.random.permutation(yte)
            nulls.append(auc(yp, sc))
        cont_null = np.nanmean(nulls)
        null_p = (np.array(nulls) >= cont_auc).mean()
    else:
        null_p = np.nan
    print("\n  --- TEST 2: CONTINUATION (OOS AUC, order-flow state at DIB onset) ---")
    print(f"    train n={len(Xtr)} (pos={ytr.mean()*100:.0f}%)  oos n={len(Xte)} (pos rate={cont_base*100:.0f}%)"
          if len(Xtr) else "    insufficient data")
    print(f"    OOS AUC = {cont_auc:.3f}   shuffled-null AUC = {cont_null:.3f}   p(null>=auc)={null_p:.3f}")
    print(f"    benchmark to beat: time-bar 0.52 / clear-signal 0.58")

    # ---------- TEST 3: CAPTURE (cost-honest, OOS) ----------
    # Enter at the DIB trigger, fixed-policy exit = hold HOLD_BARS DIB bars
    # (honest, no oracle). Also report oracle-to-extreme ceiling. Net of
    # round-trip cost. Compare to random-entry null on the same movers.
    cost = COST_RT.get(sym, 0.0015)
    rets_honest, rets_oracle, rand_rets = [], [], []
    n_cap_trig = 0
    for mv in oos:
        kt = dib_trigger(d, mv)
        if kt is None:
            continue
        n_cap_trig += 1
        idx = mv["idx"]
        g = idx[kt]
        direction = mv["dir"]
        c = d["close"]
        p0 = c[g]
        # honest exit
        g_exit = min(g + HOLD_BARS, idx[-1])
        r_h = direction * (c[g_exit] / p0 - 1.0) - cost
        rets_honest.append(r_h)
        # oracle ceiling (exit at the day extreme if it's after entry)
        if mv["i_ext"] > g:
            r_o = direction * (mv["ext_price"] / p0 - 1.0) - cost
        else:
            r_o = -cost   # extreme already passed -> no capture
        rets_oracle.append(r_o)
        # random-entry null: pick a random bar on the same day, same hold+cost
        rcand = idx[(idx >= mv["i_open"]) & (idx < idx[-1] - 1)]
        if len(rcand) > 0:
            gr = int(np.random.choice(rcand))
            gr_exit = min(gr + HOLD_BARS, idx[-1])
            rr = direction * (c[gr_exit] / c[gr] - 1.0) - cost
            rand_rets.append(rr)

    def stat(a):
        a = np.array(a, float)
        if len(a) == 0:
            return "n=0"
        return (f"n={len(a)} mean={a.mean()*100:+.2f}% med={np.median(a)*100:+.2f}% "
                f"win={ (a>0).mean()*100:.0f}% sum={a.sum()*100:+.1f}%")

    # trade frequency / cost realism
    span_days = (d["ts"].max() - d["ts"].min()) / 1000 / 86400
    trig_per_day = n_cap_trig / (len(oos) if len(oos) else 1)   # ~1 setup per mover-day
    print("\n  --- TEST 3: CAPTURE (cost-honest, OOS movers) ---")
    print(f"    cost round-trip = {cost*100:.2f}%  hold = {HOLD_BARS} DIB bars  "
          f"triggers = {n_cap_trig}/{len(oos)} oos movers (~1/mover-day; trade freq is per mover-day)")
    print(f"    HONEST (fixed {HOLD_BARS}-bar exit) : {stat(rets_honest)}")
    print(f"    ORACLE ceiling (exit @ extreme)   : {stat(rets_oracle)}")
    print(f"    RANDOM-entry null (same day)      : {stat(rand_rets)}")
    honest = np.array(rets_honest, float)
    rnd = np.array(rand_rets, float)
    cap_edge = honest.mean() - (rnd.mean() if len(rnd) else 0.0)
    print(f"    edge over random-entry null = {cap_edge*100:+.2f}%   "
          f"honest mean {'POSITIVE' if honest.mean()>0 else 'NEGATIVE'} net")

    return {
        "sym": sym,
        "n_movers": len(movers),
        "n_oos": len(oos),
        "lead_dib_med": lead_dib_med,
        "lead_tb_med": lead_tb_med,
        "n_dib_trig": n_dib_trig,
        "n_tb_trig": n_tb_trig,
        "cont_auc": cont_auc,
        "cont_null": cont_null,
        "cont_null_p": null_p,
        "cap_honest_mean": float(honest.mean()) if len(honest) else np.nan,
        "cap_rand_mean": float(rnd.mean()) if len(rnd) else np.nan,
        "cap_edge": cap_edge,
        "cap_honest_win": float((honest > 0).mean()) if len(honest) else np.nan,
    }


def main():
    print("=" * 72)
    print("DIB MOVER-AXIS GO/NO-GO  (3-asset pilot; held-out; cost-honest)")
    print(f"MOVE_THRESH={MOVE_THRESH*100:.0f}%  RUN_LEN={RUN_LEN}  EARLY_CAP={EARLY_CAP*100:.1f}%  "
          f"TB_THRESH={TB_THRESH*100:.1f}%  HOLD={HOLD_BARS}  seed={SEED}")
    print("=" * 72)
    results = []
    for sym, path in ASSETS.items():
        try:
            r = run_asset(sym, path)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR {sym}: {e}")
            traceback.print_exc()

    # ---------------- VERDICT ----------------
    print("\n" + "=" * 72)
    print("VERDICT  (adversarial -- require CLEAN across assets, not one lucky one)")
    print("=" * 72)
    if not results:
        print("no results")
        return
    print(f"\n{'asset':10s} {'lead_DIB':>9s} {'lead_TB':>8s} {'cont_AUC':>9s} "
          f"{'null':>6s} {'p':>5s} {'cap_honest':>11s} {'cap_rand':>9s} {'edge':>7s}")
    for r in results:
        print(f"{r['sym']:10s} {r['lead_dib_med']:>9.2f} {r['lead_tb_med']:>8.2f} "
              f"{r['cont_auc']:>9.3f} {r['cont_null']:>6.3f} {r['cont_null_p']:>5.2f} "
              f"{r['cap_honest_mean']*100:>+10.2f}% {r['cap_rand_mean']*100:>+8.2f}% "
              f"{r['cap_edge']*100:>+6.2f}%")

    # decision logic
    lead_crack = all(np.isfinite(r["lead_dib_med"]) and np.isfinite(r["lead_tb_med"])
                     and r["lead_dib_med"] > r["lead_tb_med"] for r in results)
    lead_crack_majority = sum(1 for r in results
                              if np.isfinite(r["lead_dib_med"]) and np.isfinite(r["lead_tb_med"])
                              and r["lead_dib_med"] > r["lead_tb_med"])
    cont_crack = all(np.isfinite(r["cont_auc"]) and r["cont_auc"] >= 0.58
                     and r["cont_null_p"] < 0.05 for r in results)
    cont_crack_majority = sum(1 for r in results
                              if np.isfinite(r["cont_auc"]) and r["cont_auc"] >= 0.58)
    cap_crack = all(np.isfinite(r["cap_honest_mean"]) and r["cap_honest_mean"] > 0
                    and r["cap_edge"] > 0 for r in results)
    cap_crack_majority = sum(1 for r in results
                             if np.isfinite(r["cap_honest_mean"]) and r["cap_honest_mean"] > 0)
    n = len(results)
    print(f"\n  LEADING-TIMING : DIB beats time-bar lead on {lead_crack_majority}/{n} assets "
          f"-> {'CLEAN CRACK' if lead_crack else ('partial' if lead_crack_majority else 'NULL')}")
    print(f"  CONTINUATION   : AUC>=0.58 on {cont_crack_majority}/{n} assets "
          f"-> {'CLEAN CRACK' if cont_crack else ('partial' if cont_crack_majority else 'NULL')}")
    print(f"  CAPTURE>0 net  : positive+beats-random on {cap_crack_majority}/{n} assets "
          f"-> {'CLEAN CRACK' if cap_crack else ('partial' if cap_crack_majority else 'NULL')}")
    any_clean = lead_crack or cont_crack or cap_crack
    print(f"\n  GO/NO-GO: {'GO -- build DIB for full universe (a test cracked CLEANLY)' if any_clean else 'NO-GO -- imbalance-bar axis is NULL on the pilot; chart-type axis CLOSED'}")
    print("  (3 assets = tiny n; a 'partial' is NOT a crack -- require all-asset clean + mechanism.)")


if __name__ == "__main__":
    main()
