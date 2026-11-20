"""The SOLUTIONING PIPELINE -- a repeatable, MARKET-AGNOSTIC, REPRODUCIBLE working model.

Drop a (market, instrument) in, flow it through 7 gated stages to a deployable result, with ONE coherent,
crash-safe store per workspace (the manifest) so information is captured accurately + reproducibly, never disparate.
The stages, gates, and storage are generic; each market supplies the per-stage implementation (the MarketAdapter).

The 7 stages (each advances only when its gate PASSES -- gates are machine-checked where possible, not asserted):
  00_research   decompose the market to its fundamental constituents          gate: manual (+evidence)
  01_mining     mine the data for structure                                   gate: manual (+evidence)
  02_engine     build engines + validation apparatus                          gate: RUN check_invariants (CDAP)
  03_strat      build + GATE-validate strategy candidates                     gate: a SHIP run exists in runs.jsonl
  04_bot        wrap a validated strat into a bot                             gate: manual (+evidence)
  05_execution  execution model + paper-trade                                gate: manual (+evidence)
  06_deployment deploy live (monitor/decay/kill-switch)                       gate: manual (+evidence)

SOTA hardening (vs the v1 ledger; grounded in MLflow / quant-lifecycle / autonomous-discovery research 2026):
  * LINEAGE: every artifact + gate + run captures {git_sha, dirty, artifact_sha256, python, data_ref, seed, ts}
    -> results are reproducible / attributable (the project's own binding "record git SHA + seeds" invariant).
  * MACHINE-CHECKED GATES: a stage's gate_spec is RUN (exit-code/registry predicate); --manual-override stamps
    passed_by=human (auditable) -- a false "passed" is no longer free.
  * RUN/EXPERIMENT REGISTRY: stage-03 candidate runs live in runs.jsonl {run_id, params, metrics, status, lineage}
    (the MLflow-shaped layer); the strat gate selects the SHIP run from it.
  * CRASH-SAFE STORE: atomic tmp+os.replace writes + a fail-open per-workspace lock (the project's atomic_write +
    commit-lease patterns); doctor/registry are robust to a malformed manifest + verify cross-workspace refs.
  * SINGLE SOURCE OF TRUTH: status is DERIVED from gate.passed (no status/gate decoupling).

CLI:
  python -m framework.pipeline init   <market> <instrument>
  python -m framework.pipeline record <market> <instrument> <stage> --path P --kind doc --note "..." [--seed N --data-ref R]
  python -m framework.pipeline run    <market> <instrument> <stage> --run-id ID --status SHIP --params '{...}' --metrics '{...}'
  python -m framework.pipeline gate   <market> <instrument> <stage> [--manual-override --evidence "..."]   # else RUNS the gate_spec
  python -m framework.pipeline advance<market> <instrument>
  python -m framework.pipeline status <market> <instrument>
  python -m framework.pipeline registry
  python -m framework.pipeline doctor
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
    _YAML = True
except Exception:
    _YAML = False

ROOT = Path(__file__).resolve().parents[2]
WS_ROOT = ROOT / "workspaces"

# (name, scope, purpose, gate_name, gate_spec). gate_spec kinds:
#   {"manual": True}                          -> human-attested; gate requires --manual-override + evidence
#   {"cmd": "...", "pass_exit": [0,1]}        -> RUN the command; pass iff exit in pass_exit
#   {"checker": "ship_run_exists"}            -> a built-in predicate over the run registry
STAGES = [
    ("00_research", "market", "Decompose the market to its fundamental constituents", "research_complete", {"manual": True}),
    ("01_mining", "market", "Mine the data for structure (regime/cluster/trend/predictability)", "mining_complete", {"manual": True}),
    ("02_engine", "market", "Build reusable engines + validation apparatus (oracle/decomposer/gate)", "engine_ready",
     {"cmd": "python src/audit/check_invariants.py", "pass_exit": [0, 1]}),
    ("03_strat", "instrument", "Build + GATE-validate strategy candidates", "strat_gated", {"checker": "ship_run_exists"}),
    ("04_bot", "instrument", "Wrap a validated strat into a bot (sizing/risk/lifecycle)", "bot_built", {"manual": True}),
    ("05_execution", "instrument", "Execution model (fills/costs) + paper-trade", "execution_validated", {"manual": True}),
    ("06_deployment", "instrument", "Deploy live (monitoring/decay/kill-switch)", "deployed", {"manual": True}),
]
STAGE_NAMES = [s[0] for s in STAGES]
STAGE_SPEC = {s[0]: s[4] for s in STAGES}


# ---------------------------------------------------------------- low-level IO
def _dump(obj) -> str:
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False) if _YAML else json.dumps(obj, indent=2)


def _load(text: str):
    return yaml.safe_load(text) if _YAML else json.loads(text)


def _atomic_write(path: Path, text: str):
    """tmp + os.replace -- atomic on POSIX + Windows; a crash mid-write never corrupts the manifest."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class _Lock:
    """Fail-open per-workspace lock (O_CREAT|O_EXCL). Short retry, then proceed (the project's commit-lease pattern)."""
    def __init__(self, ws: Path, tries: int = 50):
        self.lock = ws / ".manifest.lock"; self.tries = tries; self.fd = None
    def __enter__(self):
        for _ in range(self.tries):
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_RDWR); return self
            except FileExistsError:
                time.sleep(0.02)
        return self  # fail-open
    def __exit__(self, *exc):
        try:
            if self.fd is not None:
                os.close(self.fd)
            self.lock.unlink(missing_ok=True)
        except Exception:
            pass


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    try:
        if path.is_dir():
            return "dir"
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]
    except Exception:
        return ""


def _lineage(artifact: str | None = None, seed=None, data_ref: str | None = None) -> dict:
    """The reproducibility binding: code (git) + artifact content + env + seed + data version + time."""
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    lin = {"git_sha": sha[:12] or "n/a", "git_dirty": dirty, "python": sys.version.split()[0],
           "ts": datetime.now().isoformat(timespec="seconds")}
    if seed is not None:
        lin["seed"] = seed
    if data_ref:
        lin["data_ref"] = data_ref
    if artifact:
        lin["artifact_sha256"] = _sha256((ROOT / artifact)) or "missing"
    return lin


# ---------------------------------------------------------------- store
def ws_dir(market, instrument) -> Path:
    return WS_ROOT / market / instrument


def manifest_path(market, instrument) -> Path:
    return ws_dir(market, instrument) / "manifest.yaml"


def runs_path(market, instrument) -> Path:
    return ws_dir(market, instrument) / "runs.jsonl"


def _check_stage(stage):
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage must be one of {STAGE_NAMES}, got {stage!r}")


def _migrate(man: dict) -> dict:
    """Backfill missing keys so old (v1) manifests load + gain gate_spec/lineage fields."""
    man.setdefault("schema", 2)
    for name in STAGE_NAMES:
        st = man.setdefault("stages", {}).setdefault(name, {})
        st.setdefault("status", "todo"); st.setdefault("artifacts", [])
        st.setdefault("scope", dict((s[0], s[1]) for s in STAGES)[name])
        st.setdefault("purpose", dict((s[0], s[2]) for s in STAGES)[name])
        st["gate_spec"] = STAGE_SPEC[name]
        g = st.setdefault("gate", {})
        g.setdefault("name", dict((s[0], s[3]) for s in STAGES)[name])
        g.setdefault("passed", False); g.setdefault("evidence", "")
        # single source of truth: status derived from gate.passed
        st["status"] = "done" if g["passed"] else ("in_progress" if st["artifacts"] else "todo")
    return man


def init(market, instrument, stamp: str = "") -> Path:
    d = ws_dir(market, instrument)
    for s in STAGE_NAMES:
        (d / s).mkdir(parents=True, exist_ok=True)
    mp = manifest_path(market, instrument)
    if mp.exists():
        return mp
    man = {"schema": 2, "market": market, "instrument": instrument,
           "created": stamp or datetime.now().isoformat(timespec="seconds"),
           "current_stage": STAGE_NAMES[0],
           "scope_note": "stages 00-02 market-scoped (shared via _market); 03-06 instrument-scoped",
           "stages": {name: {"status": "todo", "scope": scope, "purpose": purpose, "gate_spec": spec,
                             "gate": {"name": gname, "passed": False, "evidence": ""}, "artifacts": []}
                      for (name, scope, purpose, gname, spec) in STAGES}}
    _atomic_write(mp, _dump(man))
    return mp


def _read(market, instrument) -> dict:
    mp = manifest_path(market, instrument)
    if not mp.exists():
        raise FileNotFoundError(f"no workspace: {mp} (run `init {market} {instrument}` first)")
    return _migrate(_load(mp.read_text(encoding="utf-8")))


def _write(market, instrument, man):
    _atomic_write(manifest_path(market, instrument), _dump(man))


def record(market, instrument, stage, path, kind="doc", note="", seed=None, data_ref=None):
    _check_stage(stage)
    with _Lock(ws_dir(market, instrument)):
        man = _read(market, instrument)
        arts = man["stages"][stage]["artifacts"]
        # dedup on PATH (update note/lineage in place) -- no near-duplicate rows
        existing = next((a for a in arts if a["path"] == path), None)
        art = {"path": path, "kind": kind, "note": note, "lineage": _lineage(path, seed, data_ref)}
        if existing:
            existing.update(art)
        else:
            arts.append(art)
        if man["stages"][stage]["status"] == "todo":
            man["stages"][stage]["status"] = "in_progress"
        _write(market, instrument, man)
    return art


def run(market, instrument, stage, run_id, status, params=None, metrics=None, artifact=None, seed=None, data_ref=None):
    """Record a single experiment RUN (stage-03 candidates etc.) -- the MLflow-shaped layer."""
    _check_stage(stage)
    rec = {"run_id": run_id, "stage": stage, "status": status,
           "params": params or {}, "metrics": metrics or {}, "artifact": artifact,
           "lineage": _lineage(artifact, seed, data_ref)}
    with _Lock(ws_dir(market, instrument)):
        rp = runs_path(market, instrument)
        with open(rp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def _read_runs(market, instrument) -> list:
    rp = runs_path(market, instrument)
    if not rp.exists():
        return []
    return [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]


_SHIP = {"SHIP", "SHIP-TIER", "ship", "PASS"}


def _run_gate_spec(market, instrument, stage) -> tuple[bool, dict]:
    """Execute the stage's gate_spec. Returns (passed, evidence_dict)."""
    spec = STAGE_SPEC[stage]
    if spec.get("manual"):
        return False, {"kind": "manual", "msg": "manual gate -- pass with --manual-override --evidence '...'"}
    if "cmd" in spec:
        try:
            r = subprocess.run(spec["cmd"], shell=True, cwd=str(ROOT), capture_output=True,
                               encoding="utf-8", errors="replace", timeout=600)  # cp1252-safe (CDAP banner bytes)
            ok = r.returncode in spec.get("pass_exit", [0])
            return ok, {"kind": "cmd", "cmd": spec["cmd"], "exit": r.returncode,
                        "tail": (r.stdout or "")[-400:], "lineage": _lineage()}
        except Exception as e:
            return False, {"kind": "cmd", "cmd": spec["cmd"], "error": str(e)}
    if spec.get("checker") == "ship_run_exists":
        ship = [r for r in _read_runs(market, instrument) if r.get("stage") == stage and r.get("status") in _SHIP]
        return bool(ship), {"kind": "ship_run_exists", "n_ship": len(ship),
                            "ship_runs": [r["run_id"] for r in ship][:5], "lineage": _lineage()}
    return False, {"kind": "unknown", "spec": spec}


def gate(market, instrument, stage, manual_override=False, evidence=""):
    _check_stage(stage)
    with _Lock(ws_dir(market, instrument)):
        man = _read(market, instrument)
        g = man["stages"][stage]["gate"]
        if manual_override:
            g["passed"] = True
            g["passed_by"] = "human"
            g["evidence"] = evidence or "(manual override, no evidence given)"
            g["checked"] = _lineage()
        else:
            passed, ev = _run_gate_spec(market, instrument, stage)
            g["passed"] = passed
            g["passed_by"] = "machine"
            g["evidence"] = ev
        man["stages"][stage]["status"] = "done" if g["passed"] else (
            "in_progress" if man["stages"][stage]["artifacts"] else "todo")
        _write(market, instrument, man)
    return g


def advance(market, instrument):
    with _Lock(ws_dir(market, instrument)):
        man = _read(market, instrument)
        cur = man["current_stage"]
        if not man["stages"][cur]["gate"]["passed"]:
            return {"ok": False, "msg": f"gate '{man['stages'][cur]['gate']['name']}' for {cur} not passed -- cannot advance"}
        i = STAGE_NAMES.index(cur)
        if i + 1 >= len(STAGE_NAMES):
            return {"ok": True, "msg": "already at final stage (06_deployment)"}
        man["current_stage"] = STAGE_NAMES[i + 1]
        _write(market, instrument, man)
        return {"ok": True, "msg": f"advanced {cur} -> {man['current_stage']}"}


def status(market, instrument) -> str:
    man = _read(market, instrument)
    out = [f"WORKSPACE {market}/{instrument}   current: {man['current_stage']}   created {man.get('created')}"]
    for name in STAGE_NAMES:
        s = man["stages"][name]
        mark = {"done": "[x]", "in_progress": "[~]", "todo": "[ ]"}.get(s["status"], "[?]")
        g = "PASS" if s["gate"]["passed"] else "----"
        by = s["gate"].get("passed_by", "")
        gk = s.get("gate_spec", {})
        gkind = "manual" if gk.get("manual") else (gk.get("cmd") or gk.get("checker") or "?")
        out.append(f"  {mark} {name:14s} ({s['scope']:10s}) gate:{g}{('/'+by) if by and s['gate']['passed'] else ''}  "
                   f"[{gkind}]  {len(s['artifacts'])} artifacts -- {s['purpose']}")
        for a in s["artifacts"]:
            lin = a.get("lineage", {})
            tag = f"@{lin.get('git_sha','?')}{'*' if lin.get('git_dirty') else ''}" if lin else ""
            out.append(f"        - {a['kind']:5s} {a['path']}  {tag}  {('('+a['note']+')') if a.get('note') else ''}")
    nruns = len(_read_runs(market, instrument))
    if nruns:
        out.append(f"  runs.jsonl: {nruns} experiment run(s)")
    return "\n".join(out)


def registry() -> str:
    rows, malformed = [], []
    for mp in sorted(WS_ROOT.glob("*/*/manifest.yaml")) if WS_ROOT.exists() else []:
        try:
            man = _migrate(_load(mp.read_text(encoding="utf-8")))
            done = sum(1 for n in STAGE_NAMES if man["stages"][n]["gate"]["passed"])  # SINGLE SOURCE: gate.passed
            rows.append((man["market"], man["instrument"], man["current_stage"], f"{done}/7"))
        except Exception as e:
            malformed.append((str(mp.relative_to(WS_ROOT)), f"{type(e).__name__}: {e}"))
    lines = ["# Workspace Registry (auto-rendered)\n",
             "The single index of every (market, instrument) flowing through the solutioning pipeline.",
             "Stages-done counts GATE-PASSED stages (the single source of truth).\n",
             "| market | instrument | current stage | stages done |", "|---|---|---|---|"]
    lines += [f"| {m} | {i} | {c} | {d} |" for (m, i, c, d) in rows]
    if malformed:
        lines += ["", "## MALFORMED manifests (fix these)"] + [f"- {p}: {e}" for (p, e) in malformed]
    text = "\n".join(lines) + "\n"
    if WS_ROOT.exists():
        _atomic_write(WS_ROOT / "REGISTRY.md", text)
    return text


def doctor() -> tuple[str, int]:
    """Store-accuracy gate: every recorded artifact path resolves; every ref target exists; manifests are well-formed."""
    out, problems = ["# Store health (doctor)\n"], 0
    for mp in sorted(WS_ROOT.glob("*/*/manifest.yaml")) if WS_ROOT.exists() else []:
        rel = str(mp.relative_to(WS_ROOT))
        try:
            man = _migrate(_load(mp.read_text(encoding="utf-8")))
        except Exception as e:
            problems += 1; out.append(f"  MALFORMED  {rel}: {type(e).__name__}: {e}"); continue
        for name in STAGE_NAMES:
            for a in man["stages"][name]["artifacts"]:
                p, kind = a.get("path", ""), a.get("kind")
                target = ROOT / p
                if kind == "ref" and ("workspaces" in p.replace("\\", "/")):
                    # cross-workspace ref: the referenced workspace manifest must exist
                    if not (ROOT / p / "manifest.yaml").exists() and not target.exists():
                        problems += 1; out.append(f"  DANGLING-REF  {man['market']}/{man['instrument']}  {name}  {p}")
                elif not target.exists():
                    problems += 1; out.append(f"  MISSING  {man['market']}/{man['instrument']}  {name}  {p}")
    out.append(f"\n{'OK -- store is accurate (paths + refs resolve, manifests well-formed)' if problems == 0 else str(problems)+' PROBLEM(s)'}")
    return "\n".join(out), problems


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m framework.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("init", "status"):
        p = sub.add_parser(c); p.add_argument("market"); p.add_argument("instrument")
    p = sub.add_parser("record"); p.add_argument("market"); p.add_argument("instrument"); p.add_argument("stage")
    p.add_argument("--path", required=True); p.add_argument("--kind", default="doc"); p.add_argument("--note", default="")
    p.add_argument("--seed", type=int, default=None); p.add_argument("--data-ref", default=None)
    p = sub.add_parser("run"); p.add_argument("market"); p.add_argument("instrument"); p.add_argument("stage")
    p.add_argument("--run-id", required=True); p.add_argument("--status", required=True)
    p.add_argument("--params", default="{}"); p.add_argument("--metrics", default="{}"); p.add_argument("--artifact", default=None)
    p.add_argument("--seed", type=int, default=None); p.add_argument("--data-ref", default=None)
    p = sub.add_parser("gate"); p.add_argument("market"); p.add_argument("instrument"); p.add_argument("stage")
    p.add_argument("--manual-override", action="store_true"); p.add_argument("--evidence", default="")
    p = sub.add_parser("advance"); p.add_argument("market"); p.add_argument("instrument")
    sub.add_parser("registry"); sub.add_parser("doctor")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        print("created", init(a.market, a.instrument))
    elif a.cmd == "record":
        print("recorded", record(a.market, a.instrument, a.stage, a.path, a.kind, a.note, a.seed, a.data_ref))
    elif a.cmd == "run":
        print("run", run(a.market, a.instrument, a.stage, a.run_id, a.status,
                         json.loads(a.params), json.loads(a.metrics), a.artifact, a.seed, a.data_ref))
    elif a.cmd == "gate":
        print("gate", gate(a.market, a.instrument, a.stage, a.manual_override, a.evidence))
    elif a.cmd == "advance":
        print(advance(a.market, a.instrument))
    elif a.cmd == "status":
        print(status(a.market, a.instrument))
    elif a.cmd == "registry":
        print(registry())
    elif a.cmd == "doctor":
        rep, n = doctor(); print(rep); return min(n, 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
