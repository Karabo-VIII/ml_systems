"""ENTRY-ZONE engine -- past-only candidate "where a move might start" zones + an OBSERVATION of whether each zone
precedes bigger forward moves than random. Descriptive/observational research -- NOT a deployed signal.

The premise (user, 2026-06-09): the true start of a move is only knowable in hindsight; in real time we can only
compute a ZONE (from recent price + volatility) where an entry is favourable. This builds several causal zone
definitions and OBSERVES their forward edge (forward max-favourable-excursion + forward return) vs the unconditional
(random-entry) baseline. A zone "closest to the oracle" is one whose forward MFE lift >> 1.

ZONES (all computed with data up to bar t only):
  near_long_ma : |close/SMA(ma_len) - 1| <= band            -- the user's "price near the long-term MA" zone
  donchian_low : close in the lower `q` of the N-bar range  -- pullback to recent support
  keltner_low  : close <= EMA(20) - k*ATR(14)               -- stretched below trend (mean-revert zone)
  bb_low       : Bollinger %B <= q                          -- near/below the lower band
  squeeze      : BB bandwidth in its lowest pctile recently -- volatility compression (breakout imminent)

OBSERVE: forward MFE_K = max_{1..K}(high - close_t)/close_t ; forward ret_K = (close_{t+K}-close_t)/close_t.
Lift = E[metric | in-zone] / E[metric | all-bars]. The all-bars mean IS the random-entry baseline.
No look-ahead in the ZONE; the forward metric is the OUTCOME we measure (never fed back into the zone).

Run:
  python -m mining.entry_zones --asset BTC --cadence 4h --observe
  python -m mining.entry_zones --observe-u10 --cadence 4h --kfwd 12
  python -m mining.entry_zones --selftest
No emoji (cp1252).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path: sys.path.insert(0, str(ROOT / "src"))
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

U10 = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","LTCUSDT"]


def compute_zones(close, high, low, *, ma_len=200, band=0.03, don_n=50, don_q=0.2,
                  kelt_ema=20, kelt_atr=14, kelt_k=1.5, bb_len=20, bb_std=2.0, bb_q=0.1,
                  sq_lookback=100, sq_q=0.2) -> dict:
    c = pd.Series(close); h = pd.Series(high); l = pd.Series(low)
    sma = c.rolling(ma_len, min_periods=ma_len//2).mean()
    near_long_ma = (np.abs(c / sma - 1.0) <= band).to_numpy()
    dl = c.rolling(don_n, min_periods=don_n//2).min(); dh = c.rolling(don_n, min_periods=don_n//2).max()
    rng = (dh - dl)
    donchian_low = ((c - dl) <= don_q * rng).to_numpy() & (rng > 0).to_numpy()
    ema = c.ewm(span=kelt_ema, adjust=False).mean()
    atr = ta.atr(h, l, c, length=kelt_atr)
    atr = atr if atr is not None else pd.Series(np.nan, index=c.index)
    keltner_low = (c <= (ema - kelt_k * atr)).to_numpy()
    bb = ta.bbands(c, length=bb_len, std=bb_std)
    if bb is not None and bb.shape[1] >= 3:
        lower = bb.iloc[:, 0].to_numpy(); upper = bb.iloc[:, 2].to_numpy()
        width = (upper - lower); pctb = np.where(width > 0, (close - lower) / width, np.nan)
        bb_low = pctb <= bb_q
        bw = pd.Series(np.where(c.to_numpy() != 0, width / c.to_numpy(), np.nan))
        bw_thr = bw.rolling(sq_lookback, min_periods=sq_lookback//2).quantile(sq_q)
        squeeze = (bw <= bw_thr).to_numpy()
    else:
        bb_low = np.zeros(len(close), bool); squeeze = np.zeros(len(close), bool)
    return {"near_long_ma": _clean(near_long_ma), "donchian_low": _clean(donchian_low),
            "keltner_low": _clean(keltner_low), "bb_low": _clean(bb_low), "squeeze": _clean(squeeze)}


def _clean(b):
    b = np.asarray(b); b[~np.isfinite(b.astype(float))] = False; return b.astype(bool)


def forward_metrics(close, high, low, K):
    """forward MFE_K (best long excursion), MAE_K (worst drawdown), ret_K over next K bars. NaN at the tail.
    MFE/|MAE| is the reward-to-risk ASYMMETRY -- the thing that actually makes a zone a good long ENTRY
    (not just 'precedes a big move'). NaN at the tail where the forward window is incomplete."""
    n = len(close); mfe = np.full(n, np.nan); mae = np.full(n, np.nan); ret = np.full(n, np.nan)
    for t in range(n - 1):
        end = min(t + K, n - 1)
        if end <= t: continue
        mfe[t] = (np.max(high[t+1:end+1]) - close[t]) / close[t]
        mae[t] = (np.min(low[t+1:end+1]) - close[t]) / close[t]   # <= 0, the post-entry drawdown
        ret[t] = (close[end] - close[t]) / close[t]
    return mfe, mae, ret


def observe(syms, cadence, K, **zone_kw):
    agg = {}
    for sym in syms:
        try:
            df = ChimeraLoader().load(sym if sym.endswith("USDT") else sym+"USDT", cadence=cadence).sort("date")
        except Exception:
            continue
        close = df["close"].to_numpy().astype(float)
        high = df["high"].to_numpy().astype(float) if "high" in df.columns else close
        low = df["low"].to_numpy().astype(float) if "low" in df.columns else close
        zones = compute_zones(close, high, low, **zone_kw)
        mfe, mae, ret = forward_metrics(close, high, low, K)
        valid = np.isfinite(mfe)
        base_mfe = np.nanmean(mfe[valid]); base_mae = np.nanmean(mae[valid]); base_ret = np.nanmean(ret[valid])
        for z, mask in zones.items():
            m = mask & valid
            agg.setdefault(z, {"mfe": [], "mae": [], "ret": [], "base_mfe": [], "base_mae": [],
                               "base_ret": [], "freq": [], "n": 0})
            if m.sum() >= 20:
                agg[z]["mfe"].append(np.nanmean(mfe[m])); agg[z]["mae"].append(np.nanmean(mae[m]))
                agg[z]["ret"].append(np.nanmean(ret[m]))
                agg[z]["base_mfe"].append(base_mfe); agg[z]["base_mae"].append(base_mae)
                agg[z]["base_ret"].append(base_ret)
                agg[z]["freq"].append(m.sum() / valid.sum()); agg[z]["n"] += int(m.sum())
    rows = []
    for z, d in agg.items():
        if not d["mfe"]: continue
        mfe = np.mean(d["mfe"]); mae = np.mean(d["mae"]); ret = np.mean(d["ret"])
        bm = np.mean(d["base_mfe"]); ba = np.mean(d["base_mae"]); br = np.mean(d["base_ret"])
        asym = mfe / abs(mae) if abs(mae) > 1e-9 else np.nan        # zone reward:risk
        basym = bm / abs(ba) if abs(ba) > 1e-9 else np.nan          # random-entry reward:risk
        rows.append({"zone": z, "n": d["n"], "freq": np.mean(d["freq"]),
                     "fwd_MFE": mfe, "MFE_lift": mfe/bm if bm else np.nan,
                     "fwd_MAE": mae, "asym": asym, "asym_lift": asym/basym if basym and np.isfinite(basym) else np.nan,
                     "fwd_ret": ret, "ret_lift": ret/br if abs(br) > 1e-9 else np.nan,
                     "base_MFE": bm, "base_MAE": ba, "base_ret": br, "base_asym": basym})
    return sorted(rows, key=lambda r: -(r["asym_lift"] if np.isfinite(r["asym_lift"]) else 0))


def selftest():
    rng = np.random.RandomState(3); n = 3000
    # construct a series where price dips near an SMA then rises -> near_long_ma should have forward edge
    t = np.arange(n); base = 100 + 20*np.sin(t/200) + np.cumsum(rng.randn(n)*0.2)
    close = base; high = base*1.003; low = base*0.997
    z = compute_zones(close, high, low); mfe, mae, ret = forward_metrics(close, high, low, 20)
    ok = all(z[k].dtype == bool and z[k].shape[0] == n for k in z) and np.isfinite(mfe[100:-100]).any()
    print(f"  [{'PASS' if ok else 'FAIL'}] zones computed (bool, aligned) + forward metrics finite")
    fired = {k: int(v.sum()) for k, v in z.items()}
    print(f"  zone fire counts on synthetic: {fired}")
    print(f"SELFTEST: {'PASS' if ok and sum(fired.values())>0 else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mining.entry_zones")
    ap.add_argument("--asset"); ap.add_argument("--cadence", default="4h")
    ap.add_argument("--kfwd", type=int, default=12, help="forward window in bars for the observation")
    ap.add_argument("--observe", action="store_true"); ap.add_argument("--observe-u10", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    syms = U10 if a.observe_u10 else ([a.asset] if a.asset else None)
    if not syms: ap.error("give --asset, --observe-u10, or --selftest")
    rows = observe(syms, a.cadence, a.kfwd)
    b = rows[0]
    print(f"## ENTRY-ZONE observation -- {'u10' if a.observe_u10 else a.asset} -- {a.cadence} -- forward {a.kfwd} bars")
    print(f"   baseline (random entry): MFE={b['base_MFE']*100:.2f}%  MAE={b['base_MAE']*100:.2f}%  "
          f"R:R={b['base_asym']:.2f}  ret={b['base_ret']*100:+.2f}%")
    print( "   asym = fwdMFE/|fwdMAE| (reward:risk after entry); *_lift = zone/random (>1 = zone is BETTER than random)")
    print(f"   {'zone':14} {'freq%':>6} {'n':>8} {'fwdMFE%':>8} {'fwdMAE%':>8} {'R:R':>5} {'asymLft':>7} {'MFElft':>6} {'retLft':>6}")
    for r in rows:
        print(f"   {r['zone']:14} {100*r['freq']:6.1f} {r['n']:>8} {100*r['fwd_MFE']:8.2f} {100*r['fwd_MAE']:8.2f} "
              f"{r['asym']:5.2f} {r['asym_lift']:7.2f} {r['MFE_lift']:6.2f} {r['ret_lift']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
