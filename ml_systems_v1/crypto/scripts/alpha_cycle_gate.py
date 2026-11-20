"""Alpha turn-007: Cycle Sizing Gate prototype + historical replay.

STEP 1 — Design loose-threshold regime rule on BTC daily (2017→today):
  - euphoria_gate: 365d_return > 150% AND ATH_drawdown < -5% AND close > 1.5*365d_SMA
  - accumulation_gate: 365d_return < -30% AND ATH_drawdown > -60%
  - normal elsewhere

STEP 2 — Replay on existing 4-sleeve blend 2025-01-01 → 2026-04-19:
  - multiplier map: euphoria=0.3 (de-risk), accumulation=1.0 (can't leverage up),
    normal=1.0
  - compute gated CAGR / Sharpe / DD / Calmar vs baseline

STEP 3 — Honest report: window is 474d inside a single cycle fragment; multi-
  cycle efficacy is design-time evidence only. Limited in-sample lift expected.

NO NEW INFRA — uses Binance public klines REST (anonymous). Falls back to
existing data/raw/BTCUSDT aggTrades rollup if network unavailable.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "cycle_gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "btc_daily_klines.parquet"

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_klines(symbol: str = "BTCUSDT", start: str = "2017-08-17") -> pd.DataFrame:
    """Fetch daily klines from Binance. 1000-row chunks."""
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"{BINANCE_URL}?symbol={symbol}&interval=1d&startTime={cursor}"
               f"&endTime={end_ms}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                rows = json.loads(r.read().decode())
        except Exception as e:
            print(f"[WARN] fetch error at cursor={cursor}: {e}", file=sys.stderr)
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 86_400_000
        if len(rows) < 1000:
            break
    cols = ["open_ts", "open", "high", "low", "close", "volume",
            "close_ts", "qav", "n_trades", "tb_bav", "tb_qav", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_ts"], unit="ms").dt.tz_localize(None).dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date").reset_index(drop=True)


def load_btc_daily() -> pd.DataFrame:
    if CACHE.exists() and (dt.date.today() - dt.date.fromtimestamp(CACHE.stat().st_mtime)).days < 1:
        print(f"[CACHE] loading {CACHE}")
        return pd.read_parquet(CACHE)
    print(f"[FETCH] Binance klines 2017-08-17 -> today")
    df = fetch_binance_klines()
    df.to_parquet(CACHE)
    print(f"[CACHE] wrote {CACHE} ({len(df)} rows)")
    return df


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime columns."""
    out = df.sort_values("date").reset_index(drop=True).copy()
    close = out["close"]
    out["sma365"] = close.rolling(365, min_periods=60).mean()
    out["sma111"] = close.rolling(111, min_periods=20).mean()
    out["sma700"] = close.rolling(700, min_periods=120).mean()  # 2x 350d
    out["ret_365d"] = (close / close.shift(365) - 1.0)
    # ATH drawdown (peak-to-current, rolling all-time)
    out["ath"] = close.cummax()
    out["dd_from_ath"] = (close / out["ath"] - 1.0)
    # Pi-cycle top indicator: sma111 > 2*sma350 (bearish signal historically)
    out["sma350"] = close.rolling(350, min_periods=60).mean()
    out["pi_cycle_top"] = (out["sma111"] > 2.0 * out["sma350"]).astype(int)
    # Close vs 365d-SMA (MVRV-proxy; we lack on-chain)
    out["close_over_sma365"] = close / out["sma365"]

    # Euphoria gate: overextended + still near ATH
    euphoria = (
        (out["ret_365d"] > 1.5)
        & (out["dd_from_ath"] > -0.05)
        & (out["close_over_sma365"] > 1.5)
    )
    # Accumulation gate: beaten down + off ATH
    accumulation = (
        (out["ret_365d"] < -0.30)
        & (out["dd_from_ath"] < -0.50)
    )
    # Pi-cycle reinforcement: if pi_cycle triggered within last 30d, add to euphoria
    pi_recent = out["pi_cycle_top"].rolling(30, min_periods=1).max().fillna(0).astype(int)
    euphoria = euphoria | (pi_recent == 1)

    out["regime"] = "NORMAL"
    out.loc[accumulation, "regime"] = "ACCUMULATION"
    out.loc[euphoria, "regime"] = "EUPHORIA"

    # Multiplier map (no leverage so accumulation = 1.0 cap)
    MULT = {"EUPHORIA": 0.3, "NORMAL": 1.0, "ACCUMULATION": 1.0}
    out["multiplier"] = out["regime"].map(MULT).astype(float)
    return out


def replay_on_blend(gate: pd.DataFrame, sleeves_csv: Path) -> dict:
    sleeves = pd.read_csv(sleeves_csv, parse_dates=["date"]).set_index("date").sort_index()
    # Daily blend return — use weighted blend from summary if possible,
    # else equal-weight fallback.
    blend_cols = [c for c in sleeves.columns if c != "blend_EW"]
    sleeves["blend_EW"] = sleeves[blend_cols].mean(axis=1)
    # Reconstruct the weighted blend from the portfolio_aggregator CSV:
    bp = pd.read_csv(
        ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_alpha_stack_daily.csv",
        parse_dates=["date"],
    ).set_index("date").sort_index()
    bp["daily_ret"] = bp["portfolio_equity"].pct_change().fillna(0.0) * 100.0

    # Join cycle-gate regime + multiplier onto blend dates
    g = gate[["date", "regime", "multiplier"]].set_index("date")
    df = bp.join(g, how="left").ffill()
    # Gated daily return
    df["gated_ret"] = df["daily_ret"] * df["multiplier"].fillna(1.0)
    # Equity curves
    df["equity_flat"] = (1.0 + df["daily_ret"] / 100.0).cumprod() * 10000.0
    df["equity_gated"] = (1.0 + df["gated_ret"] / 100.0).cumprod() * 10000.0

    def metrics(rets_pct: pd.Series, label: str) -> dict:
        r = rets_pct.dropna() / 100.0
        if len(r) == 0:
            return {}
        mean = r.mean()
        std = r.std()
        sharpe = (mean / std) * (365 ** 0.5) if std > 0 else 0.0
        downside = r[r < 0]
        sortino = (mean / downside.std()) * (365 ** 0.5) if len(downside) > 0 and downside.std() > 0 else 0.0
        eq = (1.0 + r).cumprod()
        peak = eq.cummax()
        dd = (eq / peak - 1.0).min()
        n = len(r)
        total_ret = eq.iloc[-1] - 1.0
        cagr = (1.0 + total_ret) ** (365 / n) - 1.0
        return {
            "label": label,
            "n_days": int(n),
            "cagr_pct": float(cagr * 100),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_dd_pct": float(dd * 100),
            "calmar": float((cagr / -dd) if dd < 0 else float("inf")),
        }

    res = {
        "flat": metrics(df["daily_ret"], "flat"),
        "gated": metrics(df["gated_ret"], "cycle_gated"),
        "regime_day_counts": df["regime"].value_counts().to_dict(),
    }
    df.reset_index().to_csv(OUT_DIR / "cycle_gate_replay.csv", index=False)
    return res


def main() -> None:
    btc = load_btc_daily()
    print(f"[BTC] {len(btc)} days,  {btc['date'].min().date()} -> {btc['date'].max().date()}")
    gate = compute_regime(btc)

    # Regime distribution over all history
    print("\n[REGIME] distribution (all history):")
    print(gate["regime"].value_counts())
    print("\n[REGIME] dated examples (euphoria only, transitions):")
    eu = gate[gate["regime"] == "EUPHORIA"][["date", "close", "ret_365d", "dd_from_ath"]]
    if len(eu) > 0:
        eu["date"] = eu["date"].dt.strftime("%Y-%m-%d")
        print(eu.iloc[::30].head(30).to_string(index=False))

    # Persist
    gate.to_parquet(OUT_DIR / "btc_regime_panel.parquet")
    print(f"\n[SAVE] btc_regime_panel.parquet ({len(gate)} rows)")

    # Replay
    sleeves_csv = ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_per_sleeve_returns.csv"
    print(f"\n[REPLAY] 4-sleeve blend 2025-01-01 -> 2026-04-19")
    res = replay_on_blend(gate, sleeves_csv)
    print(f"  regime day counts in blend window: {res['regime_day_counts']}")
    print()
    print(f"  {'METRIC':<12} {'FLAT':>14} {'GATED':>14}")
    for k in ("n_days", "cagr_pct", "sharpe", "sortino", "max_dd_pct", "calmar"):
        flat_v = res["flat"].get(k, float("nan"))
        gated_v = res["gated"].get(k, float("nan"))
        fmt = ".4f" if k in ("sharpe", "sortino", "calmar") else ".2f"
        print(f"  {k:<12} {flat_v:>14{fmt}} {gated_v:>14{fmt}}")

    with open(OUT_DIR / "cycle_gate_result.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n[SAVE] cycle_gate_result.json")


if __name__ == "__main__":
    main()
