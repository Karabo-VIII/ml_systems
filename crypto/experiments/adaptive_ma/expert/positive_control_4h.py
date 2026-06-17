"""experiments/adaptive_ma/expert/positive_control_4h.py -- POWER check for the regime-matched (ER>0.4)
firewall at 4h, using the EXACT apparatus the falsifier uses (er_gate_4h.run_asset).

WHY: the falsifier (er_gate_4h.py) finds the ER-gated fixed-MA does NOT beat the regime-matched null on
held-out -> "beta-in-disguise". That conclusion is only trustworthy if the regime-matched firewall HAS
POWER at 4h: it must FLAG a genuine within-regime entry-TIMING edge when one exists. A firewall that
rejects everything is useless. This builds such a genuine edge and confirms the firewall detects it.

CONSTRUCTION (deterministic, seeded; synthetic -> no market claim): a 4h price built from clean,
alternating UP and DOWN trend segments (both LOW-noise => HIGH efficiency ratio => both inside the ER>0.4
gate), separated by occasional CHOP segments (HIGH-noise => ER<0.4 => gated OUT). A LONG-ONLY 8/21-EMA
state (fast>slow & er>0.4) goes LONG during the UP segments and FLAT during the DOWN segments -> it
captures only the up-drift. A REGIME-MATCHED random entry draws from ALL ER>0.4 bars (UP and DOWN alike)
-> it catches the down-drift too -> ~0 / negative. So the MA's WITHIN-REGIME timing genuinely beats the
gated-random null, and the firewall MUST say so. If it does, the apparatus has power and the falsifier's
NULL result on real data is a real refutation, not a dead firewall.

RWYB:  python experiments/adaptive_ma/expert/positive_control_4h.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import er_gate_4h as EG  # noqa: E402  (reuse the EXACT falsifier apparatus: build_cols + run_asset)


def make_regime_edge_4h(seed: int = 11, n: int = 14000, up_len: int = 40, down_len: int = 40,
                        chop_len: int = 20, up_drift: float = 0.005, down_drift: float = -0.005,
                        trend_noise: float = 0.0018, chop_noise: float = 0.012,
                        start: str = "2020-01-08") -> pd.DataFrame:
    """4h OHLC with a GENUINE within-ER>0.4 long-only timing edge.

    Cycle: UP (trend) -> DOWN (trend) -> CHOP. UP & DOWN are low-noise => high ER => both pass the gate;
    a long-only MA is long in UP, flat in DOWN. Random ER>0.4 entries hit UP and DOWN equally => the MA
    timing genuinely beats the gated-random null. CHOP segments are high-noise (ER<0.4) -> gated out."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq="4h")
    rets = np.empty(n)
    t = 0
    phase = 0  # 0=UP, 1=DOWN, 2=CHOP
    while t < n:
        if phase == 0:
            L, d, nz = up_len, up_drift, trend_noise
        elif phase == 1:
            L, d, nz = down_len, down_drift, trend_noise
        else:
            L, d, nz = chop_len, 0.0, chop_noise
        seg = min(L, n - t)
        rets[t:t + seg] = d + rng.normal(0.0, nz, seg)
        t += seg
        phase = (phase + 1) % 3
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0015, n)))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def main():
    print("=" * 78)
    print("[positive_control_4h] genuine within-ER>0.4 long-only timing edge through the EXACT falsifier")
    print("  apparatus (er_gate_4h.run_asset, regime-matched ER>0.4 firewall). Synthetic -> no market claim.")
    print("=" * 78)
    df = make_regime_edge_4h()
    cols = EG.build_cols(df, entry_style="state")
    print(f"  bars={len(df)}  ER-gated(state) entries={int(cols['entry'].sum())}  "
          f"er>0.4 bars={int((cols['er'] > EG.ER_THRESH).sum())}  er median={cols['er'].median():.3f}")

    rec = EG.run_asset(df, n_books=300, seed=7, entry_style="state")
    print(json.dumps(rec, indent=2, default=str))

    # POWER VERDICT: the firewall must (a) say the real strategy BEATS the regime-matched null on held-out,
    # and (b) the real strategy is POSITIVE on held-out (a genuine up-capture). Both => HAS POWER at 4h.
    has_power = bool(rec["beats_held"] and rec["pos_held"])
    # diagnostic: held-out real must clear the null p95 (the firewall's actual accept condition)
    oos = rec["firewall"]["OOS"]; uns = rec["firewall"]["UNSEEN"]
    print("\n" + "-" * 78)
    print(f"  OOS    real={oos['real']}%  null_p95={oos['null_p95']}  beats_null={oos['beats_null']}")
    print(f"  UNSEEN real={uns['real']}%  null_p95={uns['null_p95']}  beats_null={uns['beats_null']}")
    print(f"  beats_held={rec['beats_held']}  pos_held={rec['pos_held']}")
    print(f"\n[positive_control_4h] {'PASS -- HAS POWER' if has_power else '*** CHECK -- LOW POWER ***'}: "
          f"the regime-matched ER>0.4 firewall {'DETECTS' if has_power else 'FAILS TO DETECT'} a genuine "
          f"within-regime long-only timing edge at 4h.")
    if not has_power:
        print("  -> if this fails, the falsifier's NULL on real data could be a dead-firewall artifact; inspect.")
    out = {"has_power": has_power, "rec": rec}
    outpath = Path(__file__).resolve().parent / "positive_control_4h.json"
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}")
    return has_power


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
