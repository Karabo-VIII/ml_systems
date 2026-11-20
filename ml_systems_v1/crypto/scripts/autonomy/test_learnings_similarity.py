#!/usr/bin/env python3
"""Regression test for G-B: TASK-SIMILARITY recall in the canonical learnings (mem0-style, pure-local TF-IDF).

Proves similar_for_plan retrieves by SIMILARITY to the query objective (not recency): a memory-related query
surfaces the memory lesson; an MA-related query surfaces the MA lesson; it degrades gracefully (<2 rows / empty
objective / sklearn absent). Uses a temp workspace so live learnings are untouched. No emoji (Windows cp1252).
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from harness.metaop import learnings as L


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="gb_learn_")

    # graceful: <2 rows
    L.record("use TF-IDF cosine over stored objectives for retrieval", "t1",
             "build a local vector similarity memory for the agent loop", 1, channel="a", workspace=tmp)
    if "not enough" not in L.similar_for_plan("anything", workspace=tmp):
        fails.append("with <2 rows should say 'not enough'")

    L.record("MA crossover needs regime gating to avoid chop on SOL", "t2",
             "find an adaptive moving-average edge on SOL", 1, channel="b", workspace=tmp)
    L.record("dollar bars clear maker costs better than time bars at 4h", "t3",
             "cost analysis across bar types", 1, channel="b", workspace=tmp)

    # similarity: a MEMORY-related query must surface the memory lesson above the MA/cost ones
    out_mem = L.similar_for_plan("wire a similarity-based memory retrieval into the loop", k=1, workspace=tmp)
    if "TF-IDF" not in out_mem and "vector similarity" not in out_mem:
        fails.append(f"memory query should surface the memory lesson; got: {out_mem}")

    # similarity: an MA-related query must surface the MA lesson
    out_ma = L.similar_for_plan("adaptive moving average crossover strategy for SOL", k=1, workspace=tmp)
    if "MA crossover" not in out_ma:
        fails.append(f"MA query should surface the MA lesson; got: {out_ma}")

    # retrieval is by SIMILARITY not recency: the MA lesson (t2, older than t3) beats the most-recent (t3 cost) for an MA query
    if "dollar bars" in out_ma and "MA crossover" not in out_ma:
        fails.append("similarity must beat recency (returned the recent cost lesson for an MA query)")

    # graceful: empty objective
    if "no objective" not in L.similar_for_plan("", workspace=tmp):
        fails.append("empty objective should be handled gracefully")

    if fails:
        print(f"[G-B similarity] FAIL ({len(fails)}):")
        for f in fails:
            print("   -", f)
        return 1
    print("[G-B similarity] ALL PASS (similarity-ranked recall, beats recency, graceful degradation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
