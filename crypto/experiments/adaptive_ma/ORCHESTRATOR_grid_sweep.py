#!/usr/bin/env python3
"""ORCHESTRATOR grid sweep (robust v2) -- run the loop's oracle-DNA falsifier across {cadences x assets} as
SUBPROCESSES with a per-run timeout, parsing its deterministic stdout. Produces the per-cadence realizable-ceiling
MAP fast + safely (one hanging run cannot stall the sweep). No emoji.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # experiments/adaptive_ma
ROOT = os.path.dirname(os.path.dirname(HERE))              # repo root
FALS = os.path.join(HERE, "sol", "oracle_dna_shuffled_falsifier.py")
PY = sys.executable

CADENCES = ["1d", "4h", "1h", "dollar", "range"]
ASSETS = ["BTC", "ETH", "SOL", "BNB"]
TIMEOUT = 30


def _parse(out: str) -> dict:
    def f(pat, cast=float):
        m = re.search(pat, out)
        try:
            return cast(m.group(1)) if m else None
        except Exception:
            return None
    return {
        "oracle_hold_bars": f(r"exit_H=(\d+)\s*bars", int),
        "oracle_entries_frac": f(r"oracle_entries=\d+\s*\(([\d.]+)\)"),
        "auc_heldout": f(r"HELD_OUT\s+AUC=([\d.]+)"),
        "dna_capture_plain_pct": f(r"DNA capture plain\s*=\s*(-?[\d.]+)%"),
        "null_p95_pct": f(r"null_plain\s+p95=([\d.]+)%"),
        "beats_null": (None if "beats=" not in out else ("beats=True" in out.split("REGIME-MATCHED")[-1] if "REGIME-MATCHED" in out else None)),
        "dna_genuine": (None if "DNA_GENUINE_SIGNAL" not in out else bool(re.search(r"DNA_GENUINE_SIGNAL[^\n]*=\s*True", out))),
        "apparatus_sound": (None if "APPARATUS_SOUND" not in out else bool(re.search(r"APPARATUS_SOUND[^\n]*=\s*True", out))),
    }


def main():
    rows = []
    for cad in CADENCES:
        for a in ASSETS:
            rec = {"cadence": cad, "asset": a}
            try:
                r = subprocess.run([PY, FALS, "--asset", a, "--cadence", cad, "--n-shuffle", "12"],
                                   cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
                rec.update(_parse(r.stdout or ""))
                rec["ok"] = rec.get("auc_heldout") is not None
                if not rec["ok"]:
                    rec["error"] = (r.stderr or r.stdout or "")[-90:].replace("\n", " ")
            except subprocess.TimeoutExpired:
                rec["ok"] = False; rec["error"] = f"TIMEOUT {TIMEOUT}s"
            except Exception as e:
                rec["ok"] = False; rec["error"] = str(e)[:80]
            rows.append(rec)
            print(f"  {cad:>7} {a:<4} AUC={rec.get('auc_heldout')} hold={rec.get('oracle_hold_bars')}b "
                  f"cap={rec.get('dna_capture_plain_pct')}% genuine={rec.get('dna_genuine')} ok={rec.get('ok')} "
                  f"{rec.get('error','')}", flush=True)

    import statistics as st
    agg = {}
    for cad in CADENCES:
        sub = [r for r in rows if r["cadence"] == cad and r.get("ok")]
        aucs = [r["auc_heldout"] for r in sub if isinstance(r.get("auc_heldout"), (int, float))]
        holds = [r["oracle_hold_bars"] for r in sub if isinstance(r.get("oracle_hold_bars"), (int, float))]
        caps = [r["dna_capture_plain_pct"] for r in sub if isinstance(r.get("dna_capture_plain_pct"), (int, float))]
        agg[cad] = {"n_ok": len(sub), "n_total": len([r for r in rows if r["cadence"] == cad]),
                    "mean_auc_heldout": round(st.mean(aucs), 3) if aucs else None,
                    "median_oracle_hold_bars": st.median(holds) if holds else None,
                    "mean_dna_capture_pct": round(st.mean(caps), 1) if caps else None,
                    "n_genuine": sum(1 for r in sub if r.get("dna_genuine") is True)}
    json.dump({"per_cadence": agg, "rows": rows}, open(os.path.join(HERE, "PER_CADENCE_MAP.json"), "w"), indent=2, default=str)
    print("\n=== PER-CADENCE REALIZABLE-CEILING MAP ===", flush=True)
    for cad, m in agg.items():
        print(f"  {cad:>7}: AUC={m['mean_auc_heldout']} oracle_hold={m['median_oracle_hold_bars']}b "
              f"DNA_capture={m['mean_dna_capture_pct']}% genuine={m['n_genuine']}/{m['n_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
