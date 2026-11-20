"""Alpha turn-010: R4 Signal-of-Signal (6-week supply-delta momentum) probe.

Tests whether MOMENTUM of stable-flow / ETF-flow signals (z-score their own
6-week rate of change) produces a better sizing gate than the instantaneous
z-score that Bravo tested in turn 006.

Hypothesis: when supply-delta is ACCELERATING (not just positive), risk-on
regime is confirmed -- pure sizing gate was too noisy. SoS should be cleaner.

If this rescues R4: ship the gate on the 4-sleeve blend.
If it also fails: 4th independent confirmation of "blend is regime-orthogonal."
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "r4_signal_of_signal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_signals() -> pd.DataFrame:
    # Try to pull stable + ETF features paths; fallback to reading frontier features
    paths = {
        "stable": ROOT / "data" / "frontier" / "defillama" / "stable_flow_features.parquet",
        "etf": ROOT / "data" / "frontier" / "etf" / "etf_flow_features.parquet",
    }
    dfs = []
    for name, p in paths.items():
        if not p.exists():
            print(f"[WARN] signal source missing: {p}")
            continue
        d = pd.read_parquet(p)
        # Normalize schema
        date_col = "date" if "date" in d.columns else d.columns[0]
        d["date"] = pd.to_datetime(d[date_col])
        # Pick a numeric signal column (first non-date float)
        sig_candidates = [c for c in d.columns if c != date_col and d[c].dtype.kind in "fi"]
        if not sig_candidates:
            continue
        pick = sig_candidates[0]
        d = d[["date", pick]].rename(columns={pick: f"{name}_sig"})
        dfs.append(d)
    if not dfs:
        raise SystemExit("no signals loaded")
    sig = dfs[0]
    for d in dfs[1:]:
        sig = sig.merge(d, on="date", how="outer")
    return sig.sort_values("date").reset_index(drop=True)


def compute_sos(sig: pd.DataFrame, lookback: int = 42) -> pd.DataFrame:
    """Signal-of-signal: z-score of the 42d rate of change of each signal."""
    out = sig.copy()
    for col in [c for c in sig.columns if c.endswith("_sig")]:
        # 42d rate-of-change
        roc = out[col].diff(lookback)
        # z-score of RoC over trailing 90d
        z = (roc - roc.rolling(90, min_periods=30).mean()) / roc.rolling(90, min_periods=30).std()
        out[f"{col}_sos_z"] = z
    # Combined SoS: equal-weight of z-scored RoCs
    sos_cols = [c for c in out.columns if c.endswith("_sos_z")]
    out["combined_sos"] = out[sos_cols].mean(axis=1)
    return out


def replay(signals: pd.DataFrame) -> dict:
    bp = pd.read_csv(
        ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_alpha_stack_daily.csv",
        parse_dates=["date"],
    ).sort_values("date").set_index("date")
    bp["daily_ret_pct"] = bp["portfolio_equity"].pct_change().fillna(0.0) * 100.0

    sig = signals.set_index("date")
    df = bp.join(sig, how="left").ffill()

    # Test multiple sizing schemes based on combined_sos
    #   scheme A: combined > +1 -> 1.0  (risk-on), < -1 -> 0.7, else 1.0
    #   scheme B: combined > +1 -> 1.0  (risk-on), < -1 -> 0.5, else 1.0  (stronger de-risk on accel-down)
    #   scheme C: continuous  mult = clip(1 + 0.2*combined_sos, 0.5, 1.0)
    results = {}
    for name, fn in [
        ("schemeA", lambda s: np.where(s < -1.0, 0.7, 1.0)),
        ("schemeB", lambda s: np.where(s < -1.0, 0.5, 1.0)),
        ("schemeC", lambda s: np.clip(1.0 + 0.2 * s.fillna(0), 0.5, 1.0)),
    ]:
        mult = pd.Series(fn(df["combined_sos"]), index=df.index).fillna(1.0)
        gated = df["daily_ret_pct"] * mult
        r = gated.dropna() / 100.0
        mean = r.mean(); std = r.std()
        sh = (mean / std) * (365 ** 0.5) if std > 0 else 0.0
        eq = (1.0 + r).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        cagr = (eq.iloc[-1]) ** (365 / len(r)) - 1.0
        results[name] = {
            "n_days": int(len(r)),
            "cagr_pct": float(cagr * 100),
            "sharpe": float(sh),
            "max_dd_pct": float(dd * 100),
            "avg_multiplier": float(mult.mean()),
        }

    # Baseline flat
    r0 = df["daily_ret_pct"].dropna() / 100.0
    eq0 = (1.0 + r0).cumprod()
    dd0 = float((eq0 / eq0.cummax() - 1.0).min())
    cagr0 = (eq0.iloc[-1]) ** (365 / len(r0)) - 1.0
    results["flat"] = {
        "n_days": int(len(r0)),
        "cagr_pct": float(cagr0 * 100),
        "sharpe": float((r0.mean() / r0.std()) * (365 ** 0.5)),
        "max_dd_pct": float(dd0 * 100),
        "avg_multiplier": 1.0,
    }
    return results


def main() -> None:
    sig = build_signals()
    print(f"[SIG] loaded: {sig.shape}, {sig['date'].min().date()} -> {sig['date'].max().date()}")
    sos = compute_sos(sig)
    print(f"[SoS] computed RoC + z-score for {len([c for c in sos.columns if c.endswith('_sos_z')])} signals")
    print()
    res = replay(sos)
    print(f"{'variant':<12} {'CAGR':>10} {'SHARPE':>10} {'DD':>8} {'avg_mult':>10}")
    for name, m in res.items():
        print(f"{name:<12} {m['cagr_pct']:>+10.2f} {m['sharpe']:>+10.4f} "
              f"{m['max_dd_pct']:>+8.2f} {m['avg_multiplier']:>10.4f}")
    with open(OUT_DIR / "r4_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[SAVE] {OUT_DIR / 'r4_result.json'}")


if __name__ == "__main__":
    main()
