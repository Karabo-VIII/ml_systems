"""K-parameter sweep for xsec_K5_5_FULL_dneut on U50 universe.

Tests K in {1, 3, 5, 7, 10} for symmetric long/short + K long-only variants.
Reuses the xsec script's train step by invoking with an env flag that selects
the K. Snapshots saved to pt_xsec_K{K}_{K}_U50/daily_snapshot.csv.

Why: current baseline uses K=5+5 delta-neutral. Is there a better K at U50?
Multi-universe report showed U50 is the sweet spot; now sweep K within it.

Implementation: monkey-patch the variants list in xsec_variants_daily_equity.py
via a shim that imports + overrides.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = ROOT / "logs" / "paper_trader_v2" / "seeds"

# K values to test -- both symmetric (long=short) and long-only
K_VALUES = [
    # (K_long, K_short, variant_suffix)
    (1, 1, "K1_1"),
    (3, 3, "K3_3"),
    (5, 5, "K5_5"),   # baseline
    (7, 7, "K7_7"),
    (10, 10, "K10_10"),
    (3, 0, "K3_long"),
    (5, 0, "K5_long"),
    (10, 0, "K10_long"),
]


def make_sweep_script(variants: list[tuple[int, int, str]]) -> str:
    """Generate a one-off version of xsec_variants_daily_equity.py with the K list overridden."""
    src = (ROOT / "scratch" / "xsec_variants_daily_equity.py").read_text(encoding="utf-8")
    # Replace the variants list with our K sweep
    variant_code = "variants = [\n"
    for K_long, K_short, suffix in variants:
        name = f"xsec_{suffix}_FULL_dneut_sweep"
        variant_code += (
            f"    ({name!r},  {{'K_long': {K_long}, 'K_short': {K_short}, 'stop': 0.10,\n"
            f"                                 'regime_gate': True, 'meta_gate': True, 'meta_thresh': 0.45}}),\n"
        )
    variant_code += "]\n"

    import re
    # Replace the "variants = [...]" block
    pattern = re.compile(r"variants = \[.*?\]\s*\n", re.DOTALL)
    patched = pattern.sub(variant_code, src, count=1)
    return patched


def main():
    tmp_script = ROOT / "scratch" / "xsec_k_sweep_shim.py"
    tmp_script.write_text(make_sweep_script(K_VALUES), encoding="utf-8")
    print(f"[sweep] wrote shim: {tmp_script}")

    # Run with U50 env (UNIVERSE_50 from src/strategy/universe.py)
    sys.path.insert(0, str(ROOT / "src" / "strategy"))
    from universe import UNIVERSE_50
    u50 = ",".join(a.replace("USDT", "") for a in UNIVERSE_50)

    env = os.environ.copy()
    env["PT_UNIVERSE"] = u50

    print(f"[sweep] running {len(K_VALUES)} variants on U50 (n_assets={len(UNIVERSE_50)})...")
    t0 = time.time()
    out = subprocess.run([PY, str(tmp_script)], env=env,
                         capture_output=True, text=True, timeout=1800,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dt = time.time() - t0
    print(f"[sweep] rc={out.returncode}, dt={dt:.1f}s")

    # Extract variant summary lines
    print("\n" + "=" * 100)
    print("K SWEEP RESULTS (U50)")
    print("=" * 100)
    for line in (out.stdout or "").splitlines():
        if "FULL_dneut_sweep" in line or "variant" in line.lower():
            print(line.strip())

    # Write a clean summary
    import re
    summary_rows = []
    for line in (out.stdout or "").splitlines():
        m = re.match(r"\s*(xsec_K\d+_\d+_FULL_dneut_sweep)\s+n_days=(\d+)\s+total=([+-][\d\.]+)%\s+CAGR=([+-][\d\.]+)%\s+Sharpe=([+-][\d\.]+)\s+DD=([+-][\d\.]+)%", line)
        if m:
            summary_rows.append({
                "variant": m.group(1),
                "n_days": int(m.group(2)),
                "total_ret_pct": float(m.group(3)),
                "cagr_pct": float(m.group(4)),
                "sharpe": float(m.group(5)),
                "max_dd_pct": float(m.group(6)),
            })

    import json
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    out_dir = ROOT / "logs" / "deployment" / str(today)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "universe": "U50",
        "n_assets": len(UNIVERSE_50),
        "k_values_tested": [{"K_long": kl, "K_short": ks, "suffix": s} for kl, ks, s in K_VALUES],
        "elapsed_s": round(dt, 1),
        "results": summary_rows,
    }
    out_file = out_dir / "xsec_k_sweep.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[sweep] saved: {out_file}")

    if summary_rows:
        # Rank by Sharpe
        sorted_rows = sorted(summary_rows, key=lambda r: -r["sharpe"])
        print("\nRanked by Sharpe (U50):")
        print(f"{'variant':<40} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
        print("-" * 70)
        for r in sorted_rows:
            print(f"{r['variant']:<40} {r['cagr_pct']:>+7.2f} {r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f}")


if __name__ == "__main__":
    main()
