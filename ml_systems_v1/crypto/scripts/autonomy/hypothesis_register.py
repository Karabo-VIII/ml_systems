#!/usr/bin/env python3
"""hypothesis_register.py -- the DISCOVERY FACTORY's connective tissue: a persistent, queryable register of strategy
HYPOTHESES (candidate specs) with their EV, status, and verdict -- plus a monotonic DEAD-CATALOG so a refuted vein is
NEVER re-mined. The firm can already rigorously KILL one hand-built candidate (src/strat/candidate_gate); this is the
apparatus that lets it TRACK and rank a candidate POPULATION without re-paying for lessons.

WHY (design run wq3u9dvq1, gap #5, ev 0.85): "no persistent queryable hypothesis_register, runs/discovery/ absent, no
mechanical failure/dead-catalog (the only one is archived + dead-cited in CLAUDE.md L69); lessons get re-paid for
because there is no monotonic memory of what was refuted." This closes that -- the strategy-agnostic HARNESS piece
(the candidate_gate that produces verdicts, and the actual MA/strategy work, are out of scope here).

Store: runs/discovery/hypotheses.json. A Hypothesis is keyed by a SPEC HASH (instrument+indicator+cadence+approach+...)
so dedupe + dead-check are mechanical. Composes with: candidate_gate (verdict source), skill_library (reusable assets),
rolling_ledger (session memory). No emoji (cp1252).

Usage:
  python scripts/autonomy/hypothesis_register.py register --spec '{"instrument":"SOL","indicator":"MA","cadence":"4h","approach":"breakout"}' --ev 0.6 --note "pullback-into-trend"
  python scripts/autonomy/hypothesis_register.py verdict --id <id> --status refuted --detail "0/77 beat-null, beta-confound"
  python scripts/autonomy/hypothesis_register.py list            # OPEN hypotheses, EV-ranked
  python scripts/autonomy/hypothesis_register.py dead            # the dead-catalog (do NOT re-mine)
  python scripts/autonomy/hypothesis_register.py check --spec '{...}'   # is this vein already refuted?
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, "runs", "discovery", "hypotheses.json")
STATUSES = ["proposed", "testing", "shipped", "refuted", "inconclusive"]
DEAD = {"refuted"}  # statuses that put a vein in the dead-catalog (do not re-mine)


def _spec_hash(spec: dict) -> str:
    """Canonical hash of the candidate SPEC, so the same vein (same instrument/indicator/cadence/approach) dedupes."""
    norm = json.dumps(spec, sort_keys=True, separators=(",", ":")).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _load() -> dict:
    if not os.path.exists(STORE):
        return {"hypotheses": {}}
    try:
        return json.load(open(STORE, encoding="utf-8"))
    except Exception:
        return {"hypotheses": {}}


def _save(db: dict):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    json.dump(db, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, STORE)  # atomic


def register(spec: dict, ev: float = 0.5, note: str = "", ts_ms: int | None = None) -> dict:
    """Add a hypothesis. Dedupes by spec hash; REFUSES a spec already in the dead-catalog (anti-re-mine guard)."""
    db = _load()
    h = _spec_hash(spec)
    existing = db["hypotheses"].get(h)
    if existing and existing.get("status") in DEAD:
        return {"ok": False, "id": h, "reason": f"DEAD vein -- already refuted ({existing.get('detail','')[:80]}); do NOT re-mine"}
    if existing:
        return {"ok": False, "id": h, "reason": f"already registered (status={existing.get('status')})"}
    rec = {"id": h, "spec": spec, "ev": float(ev), "status": "proposed", "note": note,
           "verdict_detail": "", "ts": ts_ms if ts_ms is not None else int(time.time() * 1000)}
    db["hypotheses"][h] = rec
    _save(db)
    return {"ok": True, "id": h, "reason": "registered"}


def record_verdict(hid: str, status: str, detail: str = "") -> dict:
    db = _load()
    rec = db["hypotheses"].get(hid)
    if not rec:
        return {"ok": False, "reason": f"no hypothesis {hid}"}
    if status not in STATUSES:
        return {"ok": False, "reason": f"status must be one of {STATUSES}"}
    rec["status"] = status
    rec["verdict_detail"] = detail
    rec["verdict_ts"] = int(time.time() * 1000)
    _save(db)
    return {"ok": True, "id": hid, "status": status}


def is_dead(spec: dict) -> bool:
    """True if this exact vein is already refuted (the discovery loop calls this BEFORE materializing a candidate)."""
    rec = _load()["hypotheses"].get(_spec_hash(spec))
    return bool(rec and rec.get("status") in DEAD)


def open_ranked() -> list:
    hs = [r for r in _load()["hypotheses"].values() if r.get("status") in ("proposed", "testing")]
    return sorted(hs, key=lambda r: r.get("ev", 0), reverse=True)


def dead_catalog() -> list:
    return [r for r in _load()["hypotheses"].values() if r.get("status") in DEAD]


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "list"

    def _opt(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    if cmd == "register":
        spec = json.loads(_opt("--spec", "{}"))
        r = register(spec, float(_opt("--ev", "0.5")), _opt("--note", ""))
        print(f"{'OK' if r['ok'] else 'REJECT'} [{r['id']}] {r['reason']}")
        return 0 if r["ok"] else 1
    if cmd == "verdict":
        r = record_verdict(_opt("--id", ""), _opt("--status", ""), _opt("--detail", ""))
        print(f"{'OK' if r['ok'] else 'ERR'}: {r.get('reason', r)}")
        return 0 if r["ok"] else 1
    if cmd == "check":
        spec = json.loads(_opt("--spec", "{}"))
        dead = is_dead(spec)
        print(f"{'DEAD (refuted -- do NOT re-mine)' if dead else 'OPEN (not refuted)'} :: spec_hash={_spec_hash(spec)}")
        return 0
    if cmd == "dead":
        d = dead_catalog()
        print(f"=== DEAD-CATALOG ({len(d)} refuted veins -- do NOT re-mine) ===")
        for r in d:
            print(f"  [{r['id']}] {json.dumps(r['spec'])} :: {r.get('verdict_detail','')[:90]}")
        return 0
    # list (default): open hypotheses EV-ranked
    o = open_ranked()
    print(f"=== OPEN HYPOTHESES ({len(o)}, EV-ranked) ===")
    for r in o:
        print(f"  [ev={r.get('ev')}|{r['id']}] {json.dumps(r['spec'])} {('-- ' + r['note']) if r.get('note') else ''}")
    if not o:
        print("  (none -- register candidate specs to seed the discovery factory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
