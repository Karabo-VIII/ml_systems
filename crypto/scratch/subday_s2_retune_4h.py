"""Retune S2 (BTC cascade lag) thresholds at 4h cadence.

At default (k=2, btc_trigger=0.01) S2 gave n=665, mean_net=-0.47%, t=-2.81 at 4h.
At 1d it gave n=265, mean_net=+2.34%, t=+2.73 (shipped).

Grid search (k=[1,2,3] x btc_trigger=[0.005, 0.01, 0.015, 0.02, 0.03])
to find 4h sweet spot. If any cell shows n>=100, mean_net>0.5%, t>2.0,
signal survives. Else confirm CONCEDE.
"""
import sys, math, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "strategy"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.cadence_loader import load_asset_data  # type: ignore
from engine_subday_btc_cascade_lag import BTCCascadeLagEngine  # type: ignore

TAKER_RT = 0.0016
UNIVERSE_10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def main():
    t0 = time.time()
    names, _, asset_data = load_asset_data(UNIVERSE_10, cadence="4h", min_bars=100)
    n_buckets = len(next(iter(asset_data.values()))["close"])
    print(f"[info] loaded {len(names)} assets, {n_buckets} 4h buckets in {time.time()-t0:.1f}s")

    print(f"\n{'k':>3} {'btc_trig':>10} {'asset_lag':>10} {'h':>4} {'n':>5} {'mean_net':>10} {'t':>6} {'hit':>6}")
    print("=" * 70)
    ship = []

    for k in [1, 2, 3, 6]:
        for btc_trig in [0.005, 0.008, 0.012, 0.02, 0.03]:
            for h in [3, 6]:  # 12h or 24h forward
                eng = BTCCascadeLagEngine(k=k, btc_trigger=btc_trig, horizon_bars=h)
                fires = []
                for t in range(max(30, k+1), n_buckets - h):
                    try:
                        sigs = eng.compute_signals(asset_data, t)
                    except Exception:
                        continue
                    for asset, conv in sigs.items():
                        if conv <= 0:
                            continue
                        close = asset_data[asset].get("close")
                        if close is None: continue
                        exit_idx = t + h
                        if exit_idx >= len(close): continue
                        p0, p1 = close[t], close[exit_idx]
                        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0: continue
                        fires.append((p1/p0) - 1.0)

                if len(fires) < 5:
                    continue
                rets = [r - TAKER_RT for r in fires]
                n = len(rets)
                mean_n = sum(rets) / n
                var = sum((r - mean_n) ** 2 for r in rets) / max(n-1, 1)
                std = math.sqrt(var) if var > 0 else 1e-9
                t_stat = (mean_n * math.sqrt(n)) / std
                hit = sum(1 for r in rets if r > 0) / n
                flag = "  <<" if (n >= 100 and mean_n > 0.005 and t_stat > 2.0) else ""
                print(f"{k:>3} {btc_trig:>10.3f} {0.003:>10.3f} {h:>3}b {n:>5} "
                      f"{mean_n*100:>+8.2f}% {t_stat:>5.2f} {hit*100:>5.1f}%{flag}")
                if flag:
                    ship.append((k, btc_trig, h, n, mean_n, t_stat))

    print()
    if ship:
        print(f"[SHIP] {len(ship)} S2 configs at 4h pass ship criteria:")
        for k, bt, h, n, m, t in ship:
            print(f"  k={k} btc_trig={bt} h={h}b  n={n} mean_net={m*100:+.2f}% t={t:.2f}")
    else:
        print("[CONCEDE] S2 cannot be retuned at 4h. Alpha confirmed daily-only.")


if __name__ == "__main__":
    main()
