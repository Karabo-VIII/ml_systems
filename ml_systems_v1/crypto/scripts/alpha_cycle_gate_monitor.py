"""Alpha turn-010: Cycle-gate monitor CLI.

Reports current BTC cycle regime (EUPHORIA / ACCUMULATION / NORMAL) based on
the rule designed in alpha_cycle_gate.py. Intended to be called daily (cron)
or ad-hoc. Emits a concise stdout report + JSON side-artifact.

When rule flips from NORMAL -> EUPHORIA, human attention is needed: this
signals "reduce blend allocation to ~30% at next cycle top". The alpha stack
does NOT auto-throttle (user has not approved auto-sizing). This is monitor
only -- advisory output.

Usage:
    python scripts/alpha_cycle_gate_monitor.py
    python scripts/alpha_cycle_gate_monitor.py --history 30   # last 30 days
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "logs" / "frontier" / "cycle_gate" / "btc_regime_panel.parquet"
OUT_JSON = ROOT / "logs" / "frontier" / "cycle_gate" / "cycle_gate_monitor_latest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, default=14, help="days of regime history to print")
    ap.add_argument("--refresh", action="store_true",
                    help="re-run alpha_cycle_gate.py to refresh the panel first")
    args = ap.parse_args()

    if args.refresh or not PANEL.exists():
        print("[REFRESH] Running alpha_cycle_gate.py to rebuild panel...")
        import subprocess
        subprocess.check_call(["python", str(ROOT / "scripts" / "alpha_cycle_gate.py")])

    df = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    regime = str(last["regime"])
    multiplier = float(last["multiplier"])

    # Human-readable summary
    print("=" * 60)
    print(f"CYCLE GATE MONITOR -- as of {last['date'].date()}")
    print("=" * 60)
    print(f"BTC close:           ${last['close']:>12,.2f}")
    print(f"365d return:         {last['ret_365d']*100:>+11.2f}%")
    print(f"DD from ATH:         {last['dd_from_ath']*100:>+11.2f}%")
    print(f"close / 365d-SMA:    {last['close_over_sma365']:>12.4f}")
    print(f"Pi-cycle (top):      {int(last['pi_cycle_top']):>12d}")
    print()
    print(f"REGIME:              {regime}")
    print(f"Recommended mult:    {multiplier:>12.2f}  "
          f"(blend sizing advisory)")
    print()

    # Historical context
    hist = df.tail(args.history)[["date", "regime", "close", "ret_365d", "dd_from_ath"]].copy()
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    # Detect regime transition
    transitions = []
    regimes = df["regime"].tolist()
    dates = df["date"].tolist()
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append({"date": dates[i].strftime("%Y-%m-%d"),
                                "from": regimes[i - 1], "to": regimes[i]})
    last_transition = transitions[-1] if transitions else None

    print(f"Last regime transition: {last_transition}")
    print()
    print(f"Last {args.history} days regime history:")
    print(hist.to_string(index=False))

    # JSON artifact for downstream tooling
    artifact = {
        "as_of": str(last["date"].date()),
        "btc_close": float(last["close"]),
        "regime": regime,
        "multiplier_advisory": multiplier,
        "ret_365d_pct": float(last["ret_365d"] * 100),
        "dd_from_ath_pct": float(last["dd_from_ath"] * 100),
        "close_over_sma365": float(last["close_over_sma365"]),
        "pi_cycle_top": int(last["pi_cycle_top"]),
        "last_transition": last_transition,
        "alert_if_euphoria_transition": (
            last_transition is not None
            and last_transition["to"] == "EUPHORIA"
            and (pd.Timestamp(last["date"]) - pd.Timestamp(last_transition["date"])).days <= 7
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(artifact, f, indent=2)
    print()
    print(f"[SAVE] {OUT_JSON}")


if __name__ == "__main__":
    main()
