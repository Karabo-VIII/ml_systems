"""runs/research/firewall_4h_regime_matched.py -- THE -k FALSIFIER (CORE).

Re-test the 1d MA-cross failure mode (0/69 beat the random-entry firewall on held-out) at the NEW 4h
cadence, against the minimal-honest p1 baseline (ER-GATED fixed-MA crossover), using the REGIME-MATCHED
random-entry null (random entries drawn ONLY from gate-ON / ER>=thr trending bars -> isolates WITHIN-gate
entry TIMING from the gate's regime SELECTION). Decision = NET PER-TRADE EXPECTANCY of the baseline vs the
regime-matched null distribution (null mean/quantiles + baseline percentile rank), on the SAME held-out
split the apparatus uses everywhere. Also runs a positive_control@4h power-check (a PLANTED within-gate
timing edge MUST pass) so the null is only trusted once the harness is shown to have power at 4h.

NO commit / NO deploy. RWYB: `python runs/research/firewall_4h_regime_matched.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, ema_past_only
from strat.firewall import random_entry_null, _gate_on_mask

# ---- fixed, pre-registered config (the minimal honest p1 baseline) -------------------------
ER_WIN = 20
ER_THR = 0.40          # researcher prescription: trade the cross ONLY when ER > ~0.4 (trending)
EMA_FAST, EMA_SLOW = 8, 21
MAX_HOLD = 42          # <7d at 4h
COST_RT = 0.0024       # spot taker round-trip (honest)
N_BOOKS = 1000
SEED = 7
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
HELD = ["OOS", "UNSEEN"]


def kaufman_er(close: pd.Series, n: int = ER_WIN) -> pd.Series:
    """Past-only Kaufman Efficiency Ratio at bar t over closes[t-n..t] (close-of-bar; fill next open => past-only)."""
    c = close.astype(float)
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change / vol.replace(0, np.nan)


def make_harness(df: pd.DataFrame) -> CanonicalHarness:
    df = df.copy()
    df["er"] = kaufman_er(df["close"], ER_WIN)
    df["ema_fast"] = ema_past_only(df["close"], EMA_FAST)
    df["ema_slow"] = ema_past_only(df["close"], EMA_SLOW)
    spec = StrategySpec(fast_col="ema_fast", slow_col="ema_slow", signal="crossover",
                        filter_col="er", filter_op="gte", filter_val=ER_THR,
                        exit_policy="signal_flip_or_filter", cost_rt=COST_RT,
                        use_funding=False, funding_col="fund_rate_mean", funding_scale=0.0,
                        max_hold_bars=MAX_HOLD, max_hold_ext_bars=None)
    return CanonicalHarness(df, spec, WIN, chimera_path="firewall_4h_rm")


# ---------------------------------------------------------------------------
def per_trade_expectancy_null(harness, n_books: int = N_BOOKS, seed: int = SEED,
                              regime_matched: bool = True) -> dict:
    """The task's decision surface: a REGIME-MATCHED random-entry null distribution of NET PER-TRADE
    EXPECTANCY (not compound). Same machinery as strat.firewall.random_entry_null (matched trade count +
    holding-duration distribution + cost; entries drawn from gate-ON bars when regime_matched) but the
    per-book statistic is the MEAN net return per trade, so we can report the null mean/quantiles and the
    baseline's percentile rank in per-trade-expectancy terms -- the exact lens the 1d failure was reported
    in (-2.09% adaptive / -2.50% fixed)."""
    rng = np.random.default_rng(seed)
    real = harness.run()
    df = harness.df
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    cost = float(harness.spec.cost_rt)
    windows = list(harness.WINDOWS)
    wlab = np.array([harness._window_label(pd.Timestamp(dates.iloc[i])) for i in range(n)])

    real_durs = {w: [] for w in windows}
    real_nets = {w: [] for w in windows}
    for t in real.trades:
        w = t["window"]
        real_durs[w].append(max(1, int(t["duration_bars"])))
        real_nets[w].append(float(t["net_pnl"]))

    gate_mask = _gate_on_mask(harness, n) if regime_matched else None
    regime_mode = ("regime_matched_gate_on" if gate_mask is not None else
                   "plain_all_bars" + ("" if not regime_matched else "(no_filter_to_match)"))
    eligible = {w: np.array([i for i in range(1, n - 2)
                             if wlab[i] == w and (gate_mask is None or gate_mask[i])]) for w in windows}

    def draw_book_nets(w):
        nw = len(real_nets[w])
        if nw == 0 or len(eligible[w]) == 0:
            return None
        durs = np.array(real_durs[w]) if real_durs[w] else np.array([3])
        entries = rng.choice(eligible[w], size=nw, replace=True)
        dsamp = rng.choice(durs, size=nw, replace=True)
        nets = []
        for e, d in zip(entries, dsamp):
            ef = e + 1
            xf = min(ef + int(d), n - 1)
            if xf <= ef:
                continue
            nets.append(opens[xf] / opens[ef] - 1.0 - cost)
        return nets

    # per-window null of MEAN per-trade net
    out = {}
    null_books_per_w = {}
    for w in windows:
        nw = len(real_nets[w])
        if nw == 0 or len(eligible[w]) == 0:
            out[w] = {"real_exp_pct": None, "n_trades": nw, "null_mean_pct": None,
                      "null_p5_pct": None, "null_p50_pct": None, "null_p95_pct": None, "pctile_rank": None}
            null_books_per_w[w] = None
            continue
        books = []
        for _ in range(n_books):
            nets = draw_book_nets(w)
            books.append(np.mean(nets) if nets else 0.0)
        books = np.array(books)
        null_books_per_w[w] = books
        real_exp = float(np.mean(real_nets[w]))
        pr = float((books < real_exp).mean())
        out[w] = {"real_exp_pct": round(real_exp * 100, 4), "n_trades": nw,
                  "null_mean_pct": round(float(books.mean()) * 100, 4),
                  "null_p5_pct": round(float(np.percentile(books, 5)) * 100, 4),
                  "null_p50_pct": round(float(np.percentile(books, 50)) * 100, 4),
                  "null_p95_pct": round(float(np.percentile(books, 95)) * 100, 4),
                  "pctile_rank": round(pr, 4)}

    # held-out COMBINED (pool OOS+UNSEEN per book, matched counts per window) -- the decisive number
    real_held_nets = [x for w in HELD for x in real_nets[w]]
    combined = None
    if real_held_nets and all(len(eligible[w]) > 0 for w in HELD if len(real_nets[w]) > 0):
        rng2 = np.random.default_rng(seed + 1)
        # re-bind draw to rng2 for the combined pass (independent stream)
        def draw_book_nets2(w):
            nw = len(real_nets[w])
            if nw == 0 or len(eligible[w]) == 0:
                return []
            durs = np.array(real_durs[w]) if real_durs[w] else np.array([3])
            entries = rng2.choice(eligible[w], size=nw, replace=True)
            dsamp = rng2.choice(durs, size=nw, replace=True)
            nets = []
            for e, d in zip(entries, dsamp):
                ef = e + 1; xf = min(ef + int(d), n - 1)
                if xf <= ef:
                    continue
                nets.append(opens[xf] / opens[ef] - 1.0 - cost)
            return nets
        books = []
        for _ in range(n_books):
            pooled = []
            for w in HELD:
                pooled.extend(draw_book_nets2(w))
            books.append(np.mean(pooled) if pooled else 0.0)
        books = np.array(books)
        real_exp = float(np.mean(real_held_nets))
        pr = float((books < real_exp).mean())
        combined = {"real_exp_pct": round(real_exp * 100, 4), "n_trades": len(real_held_nets),
                    "null_mean_pct": round(float(books.mean()) * 100, 4),
                    "null_p5_pct": round(float(np.percentile(books, 5)) * 100, 4),
                    "null_p50_pct": round(float(np.percentile(books, 50)) * 100, 4),
                    "null_p95_pct": round(float(np.percentile(books, 95)) * 100, 4),
                    "pctile_rank": round(pr, 4),
                    "beats_null_p95": bool(real_exp > np.percentile(books, 95))}

    # PASS (per-trade-expectancy): mirror the firewall's "beats on held-out" -> require BOTH held windows
    # individually above their null p95 AND the combined above its null p95.
    held_ok = all(out[w]["pctile_rank"] is not None and out[w]["pctile_rank"] > 0.95 for w in HELD)
    comb_ok = bool(combined and combined["beats_null_p95"])
    pos_held = all((out[w]["real_exp_pct"] or -1) > 0 for w in HELD)
    pass_pte = bool(held_ok and comb_ok and pos_held)
    return {"per_window": out, "combined_held_out": combined, "regime_mode": regime_mode,
            "PASS_per_trade_expectancy": pass_pte, "n_books": n_books, "cost_rt": cost}


# ---------------------------------------------------------------------------
def make_positive_control_4h(seed: int = 11) -> pd.DataFrame:
    """Synthetic 4h OHLC with a GENUINE WITHIN-GATE entry-timing edge (the power check@4h).

    Construction: mostly flat noise (low ER -> gate OFF). Occasionally a FRONT-LOADED up-impulse: M bars
    of decaying positive drift (big early, fading late). During the impulse ER is high (trending -> gate
    ON) and the EMA-fast crosses above EMA-slow near the START. A random entry drawn from gate-ON bars
    lands UNIFORMLY across the impulse (often late, after most of the front-loaded gain) and, held for the
    strategy's own duration, exits into the post-impulse flat -> captures less. So the CROSS's entry timing
    genuinely beats random-among-trending. If the apparatus cannot detect THIS at 4h, a null on real data
    is untrustworthy. (FOUNDATION verification -- synthetic, no market claim.)"""
    dates = pd.date_range(start="2020-01-07", end="2026-05-28", freq="4h")
    n = len(dates)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, n)          # quiet noisy background (low ER)
    impulse_left = 0
    decay = None
    k = 0
    t = 1
    while t < n:
        if impulse_left > 0:
            rets[t] += decay[k]               # front-loaded decaying up-drift
            k += 1
            impulse_left -= 1
        elif rng.random() < 0.012:            # ~1.2% of bars start an impulse
            M = 12
            decay = np.linspace(0.030, 0.002, M)   # +3.0%/bar fading to +0.2%/bar (front-loaded)
            impulse_left = M
            k = 0
            continue                           # this bar is the first impulse bar next loop
        t += 1
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0015, n)))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
def _fmt_window_row(w, r):
    if r["real_exp_pct"] is None:
        return f"  {w:8} n={r['n_trades']:<4} (no trades / no eligible gate-ON bars)"
    return (f"  {w:8} real_exp={r['real_exp_pct']:>+8.4f}%  null[mean={r['null_mean_pct']:+.4f} "
            f"p5={r['null_p5_pct']:+.4f} p50={r['null_p50_pct']:+.4f} p95={r['null_p95_pct']:+.4f}]%  "
            f"pctile_rank={r['pctile_rank']}  n={r['n_trades']}")


def run_one(df, tag, verbose=True):
    h = make_harness(df)
    res = h.run()
    # canonical compound firewall (existing apparatus verdict), regime-matched
    fw = random_entry_null(h, n_books=300, seed=SEED, regime_matched=True)
    # per-trade-expectancy null (the task's decision surface), regime-matched
    pte = per_trade_expectancy_null(h, n_books=N_BOOKS, seed=SEED, regime_matched=True)
    if verbose:
        print(f"\n===== {tag} =====")
        print(f"  [regime_mode={pte['regime_mode']}]  per-window NET per-trade expectancy vs regime-matched null:")
        for w in h.WINDOWS:
            print(_fmt_window_row(w, pte["per_window"][w]))
        c = pte["combined_held_out"]
        if c:
            print(f"  HELD-OUT(OOS+UNSEEN) real_exp={c['real_exp_pct']:+.4f}%  null[mean={c['null_mean_pct']:+.4f} "
                  f"p50={c['null_p50_pct']:+.4f} p95={c['null_p95_pct']:+.4f}]%  pctile_rank={c['pctile_rank']}  "
                  f"beats_p95={c['beats_null_p95']}  n={c['n_trades']}")
        print(f"  --> compound-firewall (canonical) beats_held={fw['beats_held']} pos_held={fw['pos_held']} :: {fw['verdict']}")
        print(f"  --> PER-TRADE-EXPECTANCY PASS = {pte['PASS_per_trade_expectancy']}")
    return {"tag": tag, "fw_compound": fw, "pte": pte,
            "comps": {w: round(res.window_stats[w].compound_pct, 2) for w in h.WINDOWS}}


def main():
    import json
    from pipeline.chimera_loader import ChimeraLoader

    def load4h(sym):
        g = ChimeraLoader().load(sym, cadence="4h"); d = g.to_dict(as_series=False)
        raw = np.asarray(d["date"]); dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
        return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float), "high": np.asarray(d["high"], float),
                             "low": np.asarray(d["low"], float), "close": np.asarray(d["close"], float)})

    print("=" * 92)
    print("POSITIVE CONTROL @4h (power check): a PLANTED within-gate timing edge MUST pass before any null is trusted")
    print("=" * 92)
    pc = run_one(make_positive_control_4h(), "POSITIVE_CONTROL_4h (synthetic planted within-gate timing edge)")
    power_ok = pc["pte"]["PASS_per_trade_expectancy"] and pc["fw_compound"]["beats_held"]
    print(f"\n  >>> HARNESS POWER @4h: {'CONFIRMED' if power_ok else '*** NOT CONFIRMED -- null untrustworthy ***'} "
          f"(pte_pass={pc['pte']['PASS_per_trade_expectancy']}, compound_beats_held={pc['fw_compound']['beats_held']})")

    print("\n" + "=" * 92)
    print("PRIMARY: BTC 4h ER-gated EMA8/21 p1 baseline vs REGIME-MATCHED random-entry null (per-trade expectancy)")
    print("=" * 92)
    btc = run_one(load4h("BTCUSDT"), "BTCUSDT 4h p1 baseline (EMA8/21 x-over, ER>=0.4 gate, signal_flip+maxhold42)")

    print("\n" + "=" * 92)
    print("BREADTH: re-test the 1d 0/69 failure mode across liquid assets @4h (regime-matched per-trade firewall)")
    print("=" * 92)
    universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]
    breadth = []
    n_pass = 0
    n_ok = 0
    for sym in universe:
        try:
            df = load4h(sym)
        except Exception as e:
            print(f"  {sym:10} SKIP ({type(e).__name__})")
            continue
        if len(df) < 800:
            print(f"  {sym:10} SKIP (only {len(df)} bars)")
            continue
        r = run_one(df, sym, verbose=False)
        c = r["pte"]["combined_held_out"]
        passed = r["pte"]["PASS_per_trade_expectancy"]
        n_ok += 1
        n_pass += int(passed)
        breadth.append({"sym": sym, "pass": passed,
                        "held_real_exp_pct": (c["real_exp_pct"] if c else None),
                        "held_null_mean_pct": (c["null_mean_pct"] if c else None),
                        "held_pctile_rank": (c["pctile_rank"] if c else None),
                        "compound_beats_held": r["fw_compound"]["beats_held"]})
        cc = c or {}
        print(f"  {sym:10} PASS={str(passed):5} held_real_exp={str(cc.get('real_exp_pct')):>9} "
              f"null_mean={str(cc.get('null_mean_pct')):>9} pctile={str(cc.get('held_pctile_rank', cc.get('pctile_rank'))):>6} "
              f"compound_beats_held={r['fw_compound']['beats_held']}")

    print("\n" + "=" * 92)
    print(f"VERDICT: positive_control@4h power={'CONFIRMED' if power_ok else 'NOT CONFIRMED'}")
    print(f"         BTC p1 baseline PER-TRADE-EXPECTANCY PASS = {btc['pte']['PASS_per_trade_expectancy']}")
    print(f"         BREADTH: {n_pass}/{n_ok} liquid assets beat the regime-matched per-trade null on held-out")
    print("=" * 92)

    out = {"positive_control_4h_power": bool(power_ok),
           "btc": {"pte_pass": btc["pte"]["PASS_per_trade_expectancy"],
                   "combined_held_out": btc["pte"]["combined_held_out"],
                   "per_window": btc["pte"]["per_window"],
                   "compound_firewall_verdict": btc["fw_compound"]["verdict"],
                   "compound_beats_held": btc["fw_compound"]["beats_held"]},
           "breadth": {"n_pass": n_pass, "n_ok": n_ok, "rows": breadth},
           "config": {"er_win": ER_WIN, "er_thr": ER_THR, "ema": [EMA_FAST, EMA_SLOW],
                      "max_hold": MAX_HOLD, "cost_rt": COST_RT, "n_books": N_BOOKS,
                      "windows": vars(WIN)}}
    outpath = Path(__file__).resolve().parent / "firewall_4h_regime_matched_result.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[wrote] {outpath}")
    return out


if __name__ == "__main__":
    main()
