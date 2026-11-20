"""src/strat/firewall.py -- LD-4: the cost-matched random-ENTRY null firewall (the PRIMARY gate).

PROVENANCE: ported 2026-06-05 from runs/staging/random_entry_null_2026_06_04.py. Hardened against the
2026-06-05 apparatus red-audit (docs/APPARATUS_AUDIT_2026_06_05.md):
  - F2 (CRITICAL->HIGH, FIXED): the `beats_held` guard used `... if x is not None`, which silently
    DROPPED a zero-trade window from the all() check -- a strategy that goes silent in OOS could pass
    on UNSEEN alone. Now zero-trade / None windows count as a FAIL (`is True`). (Note: the final
    verdict was already protected by `pos_held`, which requires real>0 on every held-out window; this
    is defense-in-depth so the `beats_held` flag is honest standalone.)
  - F4 (HIGH, DOCUMENTED-NOT-CHANGED): the null exit `xf = ef + d` can drift into the next window. This
    is INTENTIONAL parity: the REAL harness also holds trades across window boundaries (a trade entered
    near the end of TRAIN exits in VAL). Clipping only the null would break the apples-to-apples
    comparison and make the firewall inconsistent with the engine it audits. Left as-is by design.

Principle: a candidate's per-window compound must beat a null of the SAME number of trades entered at
RANDOM bars in that window, held for durations sampled from the candidate's OWN holding distribution,
at the SAME cost. If it does not beat random entries on the held-out windows, the timing adds nothing
-> BETA-IN-DISGUISE. This wraps the harness (runs it once); it does NOT modify the simulate/cost path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compound(nets):
    return float((np.prod(1.0 + np.asarray(nets)) - 1.0) * 100) if len(nets) else 0.0


_FILTER_OPS = {
    "gt": lambda x, v: x > v, "ge": lambda x, v: x >= v,
    "lt": lambda x, v: x < v, "le": lambda x, v: x <= v,
    "eq": lambda x, v: x == v, "ne": lambda x, v: x != v,
}


def _gate_on_mask(harness, n: int):
    """Boolean per-bar mask of where the candidate's GATE (filter_col op filter_val) is ON. Used by the
    regime-matched null so random entries are drawn only from bars the strategy WOULD consider -- isolating
    within-gate TIMING from gate/regime SELECTION. Returns None if there is no filter (no regime to match)."""
    s = harness.spec
    col = getattr(s, "filter_col", None)
    if not col or col not in harness.df.columns:
        return None
    op = _FILTER_OPS.get(getattr(s, "filter_op", "gt"), _FILTER_OPS["gt"])
    vals = harness.df[col].to_numpy(float)
    with np.errstate(invalid="ignore"):
        mask = op(vals, float(getattr(s, "filter_val", 0.0)))
    return np.asarray(mask, bool)


def random_entry_null(harness, n_books: int = 300, seed: int = 7, regime_matched: bool = False,
                      membership_matched: bool = False, move_radius_mult: float = 1.0) -> dict:
    """Run the real harness, then a cost-matched random-entry null distribution per window.

    regime_matched (G-2, 2026-06-05): for a GATED strategy the plain null draws from ALL window bars,
    including gate-OFF bars the strategy would never enter -- which can unfairly reward gate/regime
    SELECTION as if it were timing. When True, the null draws entries ONLY from gate-ON bars (filter_col
    op filter_val), so it isolates whether the WITHIN-gate entry timing beats random gate-ON entries.
    Falls back to the plain null (with a note) if there is no filter to match.

    membership_matched (M-1, 2026-06-07): the plain/regime null draws from the WHOLE window, so a real
    setup is rewarded for BOTH (a) selecting WHICH multi-candle move to be present in (window-/move-
    SELECTION value) AND (b) the precise TRIGGER TIMING within that move. To isolate (b) from (a), this
    mode draws each null entry from WITHIN the SAME multi-candle MOVE WINDOW the corresponding real setup
    fires in: a band [entry_idx - r, entry_idx + r] with r = round(move_radius_mult * duration_bars) of
    that trade. The null thus shares the move-MEMBERSHIP of the real setups (it is always present in the
    same moves, never in dead/flat stretches the setup avoided) and differs ONLY in trigger timing inside
    the move. Beating this null on held-out windows => the TRIGGER TIMING itself adds compound value over
    being randomly placed inside the same moves -- i.e. genuine trigger-timing-selection value, not merely
    move-selection value. Composable with regime_matched (bands are then further restricted to gate-ON
    bars). Falls back to per-trade entry bar if a band has no eligible bars."""
    rng = np.random.default_rng(seed)
    real = harness.run()
    df = harness.df
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    cost = float(harness.spec.cost_rt)
    windows = list(harness.WINDOWS)

    # per-bar window label (reuse the harness's own labeller for exact parity)
    wlab = np.array([harness._window_label(pd.Timestamp(dates.iloc[i])) for i in range(n)])

    # real trades grouped by window: count + holding durations + entry bar indices (for move bands)
    real_n = {w: 0 for w in windows}
    real_durs = {w: [] for w in windows}
    real_entries = {w: [] for w in windows}
    for t in real.trades:
        w = t["window"]
        real_n[w] += 1
        real_durs[w].append(max(1, int(t["duration_bars"])))
        real_entries[w].append(int(t["entry_idx"]))
    real_comp = {w: real.window_stats[w].compound_pct for w in windows}

    # eligible entry bars per window (need room for entry fill i+1 and an exit)
    gate_mask = _gate_on_mask(harness, n) if regime_matched else None
    regime_mode = "regime_matched_gate_on" if gate_mask is not None else ("plain_all_bars" if not regime_matched else "plain_all_bars(no_filter_to_match)")
    eligible = {w: np.array([i for i in range(1, n - 2)
                             if wlab[i] == w and (gate_mask is None or gate_mask[i])]) for w in windows}

    # M-1: per-real-trade move-window bands (membership-matched mode). One band per real setup; the null
    # draws that trade's entry uniformly from within its own move band, so null entries share the moves.
    bands = {w: [] for w in windows}
    if membership_matched:
        last_valid = n - 3  # need room for fill (e+1) and a >0-bar exit
        for w in windows:
            for e, d in zip(real_entries[w], real_durs[w]):
                r = max(1, int(round(move_radius_mult * d)))
                lo, hi = max(1, e - r), min(last_valid, e + r)
                band = np.arange(lo, hi + 1) if hi >= lo else np.array([], dtype=int)
                if gate_mask is not None and band.size:
                    band = band[gate_mask[band]]
                if band.size == 0:  # fallback: the real trigger bar itself (clipped to valid range)
                    band = np.array([min(max(e, 1), last_valid)])
                bands[w].append(band)
    membership_mode = ("membership_matched_move_window(radius=%.2gx_dur)" % move_radius_mult
                       if membership_matched else "off")

    out = {}
    for w in windows:
        nw = real_n[w]
        no_pool = (not membership_matched) and len(eligible[w]) == 0
        if nw == 0 or no_pool:
            out[w] = {"real": round(real_comp[w], 2), "null_p50": None, "null_p95": None,
                      "beats_null": None, "n_trades": nw}
            continue
        durs = np.array(real_durs[w]) if real_durs[w] else np.array([3])
        band_w = bands[w] if membership_matched else None
        null_comps = []
        for _ in range(n_books):
            if membership_matched:
                entries = np.array([rng.choice(b) for b in band_w])  # one draw per trade's move band
            else:
                entries = rng.choice(eligible[w], size=nw, replace=True)
            dsamp = rng.choice(durs, size=nw, replace=True)
            nets = []
            for e, d in zip(entries, dsamp):
                ef = e + 1
                xf = min(ef + int(d), n - 1)  # F4: cross-window exit allowed (parity with real engine)
                if xf <= ef:
                    continue
                nets.append(opens[xf] / opens[ef] - 1.0 - cost)
            null_comps.append(_compound(nets))
        nc = np.array(null_comps)
        p50, p95 = float(np.percentile(nc, 50)), float(np.percentile(nc, 95))
        out[w] = {"real": round(real_comp[w], 2), "null_p50": round(p50, 2),
                  "null_p95": round(p95, 2), "beats_null": bool(real_comp[w] > p95),
                  "n_trades": nw}

    # verdict: must beat null on held-out (OOS+UNSEEN) AND be absolute-positive there.
    # F2 FIX: a zero-trade / None window is a FAIL (no timing evidence), NOT silently skipped.
    held = ["OOS", "UNSEEN"]
    beats_held = all(out[w].get("beats_null") is True for w in held)
    pos_held = all((out[w]["real"] or 0) > 0 for w in held)
    verdict = ("REAL ENTRY-TIMING EDGE (beats cost-matched random-entry null on held-out AND positive)"
               if (beats_held and pos_held) else
               "BETA-IN-DISGUISE / no timing edge (does not beat random entries on held-out)")
    return {"per_window": out, "verdict": verdict, "beats_held": beats_held, "pos_held": pos_held,
            "n_books": n_books, "cost_rt": cost, "regime_mode": regime_mode,
            "membership_mode": membership_mode}


# ---------------------------------------------------------------------------
def _rwyb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.chimera_loader import ChimeraLoader
    from wealth_bot.harness import CanonicalHarness

    def to_pandas(loaded):
        df = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
        df["date"] = pd.to_datetime(df["date"], unit="ms") if np.issubdtype(df["date"].dtype, np.number) else pd.to_datetime(df["date"])
        return df
    print("[firewall RWYB] BTC 1d R12 (WMA whale-gated) vs cost-matched random-entry null ...")
    df = to_pandas(ChimeraLoader().load("BTCUSDT", cadence="1d"))
    h = CanonicalHarness.from_r12_defaults(df, chimera_path="firewall_rwyb")
    for label, rm in [("PLAIN null (all window bars)", False), ("REGIME-MATCHED null (gate-ON bars only)", True)]:
        res = random_entry_null(h, n_books=300, seed=7, regime_matched=rm)
        print(f"\n  --- {label}  [mode={res['regime_mode']}] ---")
        for w, r in res["per_window"].items():
            print(f"  {w:8} real={r['real']:>+8}%  null_p50={r['null_p50']}  null_p95={r['null_p95']}  "
                  f"beats_null={r['beats_null']}  n={r['n_trades']}")
        print(f"  beats_held={res['beats_held']}  pos_held={res['pos_held']}  VERDICT: {res['verdict']}")
    print("\n[firewall RWYB] regime-matched draws null entries ONLY from gate-ON (whale>0) bars -> isolates "
          "within-gate timing from gate selection. For a regime-gated avenue this is the fairer firewall.")


if __name__ == "__main__":
    _rwyb()
