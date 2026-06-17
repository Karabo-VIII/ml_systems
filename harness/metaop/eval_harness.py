"""Harness EVAL / FITNESS HARNESS -- turns "does the engine solve problems?" into a single NUMBER.

This is the KEYSTONE metric. A BENCHMARK is a list of self-contained BOUNDED tasks; each is run through the REAL
metaop graph (graph.build's plan->dispatch->judge->reflect->route loop) and SCORED by the MECHANICAL VERIFIER
(graph._run_verify: exit 0 == ground-truth PASS, already trust-hardened by _screen_verify_cmd). The fitness signal
is solve_rate in [0,1] -- the fraction of tasks the engine mechanically solved -- plus a per-task breakdown.

WHY MECHANICAL (the metric cannot be faked): the score is NOT the brain's self-report. A task counts as solved ONLY
when its verify_cmd (a NON-trivial assertion the harness authors, NOT the worker) exits 0. The brain claiming
success is irrelevant; a deliberately-wrong artifact scores FAIL. This is what later unlocks DSPy (optimize the
planner prompt against solve_rate) + OpenEvolve (evolve the engine against solve_rate) on an honest objective.

TWO MODES (the keystone distinction, N9):
  - PRE-SEEDED (default, run_eval): the harness PRE-SEEDS ONE build node per task (carrying the harness-authored
    verify_cmd) -> the graph's plan node returns {} early and the loop works exactly OUR node. This measures the
    WORKER + VERIFIER, NOT the planner's decomposition. solve_rate here is "given a perfectly-decomposed plan, can
    the engine execute+verify it".
  - PLANNER (run_planner_eval / run_eval(planner_mode=True)): the harness PASSES ONLY THE OBJECTIVE (frontier left
    EMPTY) -> the graph's `plan` node calls brain.decide("plan", ...) and the BRAIN must DECOMPOSE the objective into
    the right multi-step frontier itself; the loop then executes that plan. Scoring is STILL the harness's own
    mechanical verify_cmd on the FINAL composed artifact (independent of the loop's bookkeeping). The score now
    reflects PLANNER QUALITY: if the brain plans the wrong/incomplete decomposition, the composed artifact fails the
    verifier and the task scores FAIL. THIS is the honest objective a future DSPy pass must beat (it optimizes the
    planner prompt _PLAN_INSTRUCTION against planner-mode solve_rate). The PLANNER_BENCHMARK tasks below REQUIRE a
    correct multi-file/multi-step decomposition (a missing or mis-ordered sub-task makes the end-to-end verify FAIL).

DESIGN:
  - Each task runs in its OWN isolated temp BUILD DIR (no cross-contamination; cleaned up after).
  - The graph's worker writes artifacts to that build dir (cwd=build_dir); the mechanical verifier runs there too.
  - Per-task wall-clock is BOUNDED (timeout) so a hung task can't wedge the whole eval.
  - The scorecard is written to runs/autonomy/eval/<label>.json (the caller stamps the timestamp).

PROJECT-AGNOSTIC: nothing here is crypto-specific. The benchmark tasks are tiny, deterministic, offline pure-Python
exercises (fib / is_prime / reverse_string / csv row count / a calculator package / a 2-stage transform / a state
machine). No network, no heavy deps. No emoji (cp1252).

TRAIN / HELD-OUT SPLIT (anti-overfit): every task carries a "split" field ("train" | "test"; a task missing it
DEFAULTS to "train"). benchmark_split(tasks, which="all"|"train"|"test") filters by split, and run_eval /
run_planner_eval take a split="all" param (default = current behavior). A future DSPy / OpenEvolve pass MUST optimize
the planner prompt against split="train" and REPORT the honest generalization number against split="test" -- tuning
against the held-out set (or against the same ~4 tasks with no split) would OVERFIT and corrupt every fitness number.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import graph as _graph
from .brain import MockBrain


# --------------------------------------------------------------------------- the BENCHMARK
# A task = {id, split, objective, verify_cmd(build_dir)->str, oracle_artifact:(filename, content)}.
#
#  - split        : "train" | "test" (HELD-OUT). Tuning/optimizing runs against "train"; the honest generalization
#                   number is reported against "test". A task missing this key DEFAULTS to "train" (task_split).
#  - objective    : the natural-language task the engine's worker must accomplish (write a file to the build dir).
#  - verify_cmd   : a callable build_dir -> shell-command STRING. It is a NON-trivial, non-destructive assertion
#                   authored BY THE HARNESS (never the worker) that exits 0 IFF the artifact is correct, and exits
#                   non-zero otherwise. We use the absolute venv interpreter + run from the build dir (cwd) so the
#                   artifact is importable as a top-level module. Each assertion checks KNOWN outputs the worker
#                   cannot fake (e.g. fib(10)==55 AND fib(20)==6765), so a wrong artifact REFUTES.
#  - oracle_artifact  : (filename, content) the KNOWN-CORRECT artifact. Used ONLY by OracleMockBrain (the plumbing
#                   brain) -- it proves the scorer + isolation + timeout WORK end-to-end with no credentials. A real
#                   brain (Claude / ollama) never sees this; it must produce its own artifact.
#  - oracle_artifacts : OPTIONAL list[(filename, content)] for MULTI-FILE / build-then-use tasks (the planning
#                   exercises). When present, OracleMockBrain writes EVERY listed file (so a 2-file composed task --
#                   e.g. a lib module + a consumer that imports it -- can be plumbed). The verify_cmd asserts the
#                   COMPOSED result (the consumer's output), so a wrong or missing part REFUTES. A real brain must
#                   produce all files itself; the planner is what should decompose a multi-step task into nodes.
#
# verify_cmd uses sys.path.insert(0,'.') so a module written to the build dir imports as a top-level name even
# though the verifier's own cwd is the build dir. Single-quoted -c body avoids Windows cmd double-quote nesting.

_PY = f'"{sys.executable}"'  # absolute interpreter -> works regardless of shell / PATH


def _vc(body: str):
    """Build a verify_cmd factory: returns a callable build_dir -> a python -c assertion run from the build dir.
    `body` is the assertion python source (single-quoted on the cmd line). It runs with cwd=build_dir so the
    artifact file is importable; we prepend the cwd to sys.path explicitly for robustness across shells."""
    inner = "import sys; sys.path.insert(0, '.'); " + body
    return lambda _build_dir: f"{_PY} -c \"{inner}\""


BENCHMARK = [
    {
        "id": "fib",
        "split": "train",
        "objective": (
            "Write a file named fib.py in the current working directory containing an ITERATIVE function "
            "fib(n) that returns the n-th Fibonacci number (fib(0)=0, fib(1)=1). Do not write anything else "
            "that prints on import."
        ),
        "verify_cmd": _vc("from fib import fib; assert fib(10) == 55, fib(10); assert fib(20) == 6765, fib(20); "
                          "assert fib(0) == 0 and fib(1) == 1; print('FIB_OK')"),
        "oracle_artifact": ("fib.py",
                            "def fib(n):\n"
                            "    a, b = 0, 1\n"
                            "    for _ in range(n):\n"
                            "        a, b = b, a + b\n"
                            "    return a\n"),
    },
    {
        "id": "is_prime",
        "split": "train",
        "objective": (
            "Write a file named is_prime.py in the current working directory containing a function "
            "is_prime(n) that returns True iff n is a prime number (n < 2 returns False). No import-time printing."
        ),
        "verify_cmd": _vc("from is_prime import is_prime; "
                          "assert is_prime(2) and is_prime(3) and is_prime(13) and is_prime(97); "
                          "assert not is_prime(1) and not is_prime(0) and not is_prime(9) and not is_prime(100); "
                          "print('PRIME_OK')"),
        "oracle_artifact": ("is_prime.py",
                            "def is_prime(n):\n"
                            "    if n < 2:\n"
                            "        return False\n"
                            "    i = 2\n"
                            "    while i * i <= n:\n"
                            "        if n % i == 0:\n"
                            "            return False\n"
                            "        i += 1\n"
                            "    return True\n"),
    },
    {
        "id": "reverse_string",
        "split": "train",
        "objective": (
            "Write a file named reverse_string.py in the current working directory containing a function "
            "reverse_string(s) that returns the input string reversed. No import-time printing."
        ),
        "verify_cmd": _vc("from reverse_string import reverse_string; "
                          "assert reverse_string('hello') == 'olleh'; "
                          "assert reverse_string('') == ''; assert reverse_string('a') == 'a'; "
                          "assert reverse_string('racecar') == 'racecar'; print('REV_OK')"),
        "oracle_artifact": ("reverse_string.py",
                            "def reverse_string(s):\n"
                            "    return s[::-1]\n"),
    },
    {
        "id": "csv_row_count",
        "split": "test",
        "objective": (
            "Write a file named count_rows.py in the current working directory containing a function "
            "count_rows(path) that returns the number of DATA rows in a CSV file at `path`, EXCLUDING the header "
            "row. Use the standard csv module. No import-time printing."
        ),
        # the verifier writes its own tiny CSV (4 data rows + 1 header) then asserts count_rows == 4.
        "verify_cmd": _vc("from count_rows import count_rows; "
                          "open('_probe.csv','w').write('a,b\\n1,2\\n3,4\\n5,6\\n7,8\\n'); "
                          "n = count_rows('_probe.csv'); assert n == 4, n; print('CSV_OK')"),
        "oracle_artifact": ("count_rows.py",
                            "import csv\n"
                            "def count_rows(path):\n"
                            "    with open(path, newline='') as fh:\n"
                            "        rows = list(csv.reader(fh))\n"
                            "    return max(0, len(rows) - 1)\n"),
    },
    {
        "id": "gcd",
        "split": "train",
        "objective": (
            "Write a file named gcd.py in the current working directory containing a function gcd(a, b) that "
            "returns the greatest common divisor of two non-negative integers. No import-time printing."
        ),
        "verify_cmd": _vc("from gcd import gcd; "
                          "assert gcd(48, 18) == 6, gcd(48, 18); assert gcd(17, 5) == 1; "
                          "assert gcd(100, 10) == 10; assert gcd(0, 7) == 7; print('GCD_OK')"),
        "oracle_artifact": ("gcd.py",
                            "def gcd(a, b):\n"
                            "    while b:\n"
                            "        a, b = b, a % b\n"
                            "    return a\n"),
    },
    {
        "id": "word_count",
        "split": "test",
        "objective": (
            "Write a file named word_count.py in the current working directory containing a function "
            "word_count(s) that returns the number of whitespace-separated words in the string s "
            "(word_count('') == 0). No import-time printing."
        ),
        "verify_cmd": _vc("from word_count import word_count; "
                          "assert word_count('hello world') == 2; assert word_count('') == 0; "
                          "assert word_count('  one   two  three ') == 3; assert word_count('solo') == 1; "
                          "print('WC_OK')"),
        "oracle_artifact": ("word_count.py",
                            "def word_count(s):\n"
                            "    return len(s.split())\n"),
    },
    # ----------------------------------------------------------------------- MULTI-STEP / PLANNING tasks (N7)
    # These need TWO files where the SECOND imports/uses the FIRST -- a genuine build-then-use that a single trivial
    # node cannot satisfy by accident. The verify_cmd asserts the COMPOSED result (the consumer's output), so a
    # missing/wrong part of EITHER file REFUTES. They give the eval harness something that exercises decomposition.
    {
        "id": "compose_mathlib",
        "split": "train",
        "objective": (
            "Build a small two-file library. (1) Write mathlib.py in the current working directory defining "
            "add(a, b) (returns a+b) and square(x) (returns x*x). (2) Write compute.py in the current working "
            "directory that imports from mathlib and defines solve(n) returning square(add(n, 1)). "
            "No import-time printing in either file."
        ),
        # compose: solve(3) = square(add(3,1)) = square(4) = 16 ; solve(0) = square(1) = 1
        "verify_cmd": _vc("from compute import solve; "
                          "assert solve(3) == 16, solve(3); assert solve(0) == 1, solve(0); "
                          "assert solve(5) == 36, solve(5); print('COMPOSE_OK')"),
        "oracle_artifacts": [
            ("mathlib.py", "def add(a, b):\n    return a + b\n\n\ndef square(x):\n    return x * x\n"),
            ("compute.py", "from mathlib import add, square\n\n\ndef solve(n):\n    return square(add(n, 1))\n"),
        ],
    },
    {
        "id": "config_scaler",
        "split": "test",
        "objective": (
            "Build a two-file config-driven scaler. (1) Write settings_cfg.py in the current working directory "
            "defining a module-level constant SCALE = 10. (2) Write scaler.py that imports SCALE from settings_cfg "
            "and defines scale(x) returning x * SCALE. No import-time printing in either file."
        ),
        # compose: scale(5) = 5*10 = 50 ; scale(0) = 0 ; scale(-3) = -30
        "verify_cmd": _vc("from scaler import scale; "
                          "assert scale(5) == 50, scale(5); assert scale(0) == 0, scale(0); "
                          "assert scale(-3) == -30, scale(-3); print('SCALER_OK')"),
        "oracle_artifacts": [
            ("settings_cfg.py", "SCALE = 10\n"),
            ("scaler.py", "from settings_cfg import SCALE\n\n\ndef scale(x):\n    return x * SCALE\n"),
        ],
    },
    # ----------------------------------------------------------------------- HELD-OUT (test) single-node tasks
    # Added with the split work: these are pre-seeded BENCHMARK tasks reserved for the held-out set so that scoring
    # split="test" on the pre-seeded mode is non-degenerate. Same schema/style as the train tasks above.
    {
        "id": "factorial",
        "split": "test",
        "objective": (
            "Write a file named factorial.py in the current working directory containing a function "
            "factorial(n) that returns n! (factorial(0) == 1). No import-time printing."
        ),
        "verify_cmd": _vc("from factorial import factorial; "
                          "assert factorial(0) == 1, factorial(0); assert factorial(1) == 1; "
                          "assert factorial(5) == 120, factorial(5); assert factorial(7) == 5040; print('FACT_OK')"),
        "oracle_artifact": ("factorial.py",
                            "def factorial(n):\n"
                            "    result = 1\n"
                            "    for i in range(2, n + 1):\n"
                            "        result *= i\n"
                            "    return result\n"),
    },
    {
        "id": "is_palindrome",
        "split": "test",
        "objective": (
            "Write a file named is_palindrome.py in the current working directory containing a function "
            "is_palindrome(s) that returns True iff the string s reads the same forwards and backwards "
            "(is_palindrome('') == True). No import-time printing."
        ),
        "verify_cmd": _vc("from is_palindrome import is_palindrome; "
                          "assert is_palindrome('racecar') is True; assert is_palindrome('') is True; "
                          "assert is_palindrome('a') is True; assert is_palindrome('abc') is False; "
                          "assert is_palindrome('abba') is True; print('PAL_OK')"),
        "oracle_artifact": ("is_palindrome.py",
                            "def is_palindrome(s):\n"
                            "    return s == s[::-1]\n"),
    },
]


# --------------------------------------------------------------------------- the PLANNER BENCHMARK (N9)
# These tasks exist to STRESS DECOMPOSITION. In PLANNER mode the harness pre-seeds NOTHING -- the brain's `plan` node
# must turn the objective into the correct multi-step frontier and the loop must execute EVERY part. Each verify_cmd
# is a single COMPOSITE assertion on the FINAL multi-file artifact: it imports the top-of-chain consumer and checks
# the END-TO-END composed behavior, so if ANY sub-task is MISSING or MIS-ORDERED (e.g. the CLI written but the ops
# module it imports never created; the aggregator built before the parser it depends on; a state-machine transition
# left out), the composite import/result FAILS and the task scores FAIL. The objectives spell out every file + the
# exact composed contract (so a CAPABLE planner can succeed) while still requiring the planner to SEQUENCE the build.
#
# Each task carries oracle_artifacts (the full correct file set) so the PlannerOracleMockBrain can plumb them, AND a
# verify_cmd that is non-trivial + non-destructive. Bounded / offline / deterministic (pure stdlib, no network).
PLANNER_BENCHMARK = [
    {
        # 1) MINI CALCULATOR PACKAGE: an ops module + a CLI that COMPOSES the ops. The CLI parses "a OP b" from argv
        #    and prints the result. Decomposition required: (a) write calc_ops.py with the four ops; (b) write
        #    calc_cli.py importing them + a main() dispatching on the operator; the verify runs the CLI as a
        #    subprocess for THREE cases (+ - *) -> a missing op or a CLI that doesn't import the ops module fails.
        "id": "calc_package",
        "split": "train",
        "objective": (
            "Build a small two-file calculator package in the current working directory. "
            "(1) Write calc_ops.py defining four functions: add(a, b), sub(a, b), mul(a, b), div(a, b) "
            "(div is integer floor division a // b). "
            "(2) Write calc_cli.py that IMPORTS those functions from calc_ops and defines main(argv) which takes "
            "argv = [a, op, b] (a and b are integer strings; op is one of '+', '-', '*', '/'), computes the result "
            "using the matching calc_ops function, and PRINTS only the integer result. calc_cli.py must run main "
            "with sys.argv[1:] when executed as a script (if __name__ == '__main__'). "
            "No import-time printing in calc_ops.py. Example: `python calc_cli.py 6 + 4` prints 10; "
            "`python calc_cli.py 6 '*' 4` prints 24; `python calc_cli.py 9 - 4` prints 5."
        ),
        # composite: run the CLI as a subprocess for +, -, * and assert each printed result. Uses sys.executable so
        # it is interpreter-correct; checks the COMPOSED behavior (CLI -> ops), not either file alone.
        "verify_cmd": _vc(
            "import subprocess, sys; "
            "r=lambda *a: subprocess.run([sys.executable,'calc_cli.py',*a],capture_output=True,text=True); "
            "o1=r('6','+','4'); assert o1.returncode==0, o1.stderr; assert o1.stdout.strip()=='10', o1.stdout; "
            "o2=r('9','-','4'); assert o2.stdout.strip()=='5', o2.stdout; "
            "o3=r('6','*','4'); assert o3.stdout.strip()=='24', o3.stdout; "
            "print('CALC_OK')"),
        "oracle_artifacts": [
            ("calc_ops.py",
             "def add(a, b):\n    return a + b\n\n\n"
             "def sub(a, b):\n    return a - b\n\n\n"
             "def mul(a, b):\n    return a * b\n\n\n"
             "def div(a, b):\n    return a // b\n"),
            ("calc_cli.py",
             "import sys\n"
             "from calc_ops import add, sub, mul, div\n\n"
             "_OPS = {'+': add, '-': sub, '*': mul, '/': div}\n\n\n"
             "def main(argv):\n"
             "    a, op, b = int(argv[0]), argv[1], int(argv[2])\n"
             "    print(_OPS[op](a, b))\n\n\n"
             "if __name__ == '__main__':\n"
             "    main(sys.argv[1:])\n"),
        ],
    },
    {
        # 2) TWO-STAGE DATA TRANSFORM: parse -> aggregate. parse.py turns a CSV-ish text block into records; agg.py
        #    imports the parser + sums a column. Decomposition required: the aggregator is USELESS without the parser
        #    it imports; building agg.py alone (or parse.py alone) fails the composite verify.
        "id": "parse_aggregate",
        "split": "train",
        "objective": (
            "Build a two-stage data pipeline (parse then aggregate) in the current working directory. "
            "(1) Write parse_stage.py defining parse(text) that takes a string of newline-separated rows where each "
            "row is 'name,amount' (amount is an integer), ignores blank lines, and returns a list of (name, amount) "
            "tuples with amount as int. "
            "(2) Write agg_stage.py that IMPORTS parse from parse_stage and defines total(text) which parses the text "
            "and returns the SUM of all amounts, and top(text) which returns the name with the LARGEST amount. "
            "No import-time printing in either file. "
            "Example: for text 'a,3\\nb,10\\nc,7', total(text) == 20 and top(text) == 'b'."
        ),
        # composite: import the aggregator (top of chain) and check both reductions over a known block -> a missing
        # parser, a wrong int-cast, or a missing reduction all REFUTE.
        "verify_cmd": _vc(
            "from agg_stage import total, top; "
            "txt='a,3\\nb,10\\n\\nc,7\\n'; "
            "assert total(txt)==20, total(txt); assert top(txt)=='b', top(txt); "
            "assert total('x,5')==5 and top('x,5')=='x'; print('AGG_OK')"),
        "oracle_artifacts": [
            ("parse_stage.py",
             "def parse(text):\n"
             "    rows = []\n"
             "    for line in text.splitlines():\n"
             "        line = line.strip()\n"
             "        if not line:\n"
             "            continue\n"
             "        name, amount = line.split(',')\n"
             "        rows.append((name, int(amount)))\n"
             "    return rows\n"),
            ("agg_stage.py",
             "from parse_stage import parse\n\n\n"
             "def total(text):\n"
             "    return sum(amount for _name, amount in parse(text))\n\n\n"
             "def top(text):\n"
             "    rows = parse(text)\n"
             "    return max(rows, key=lambda r: r[1])[0]\n"),
        ],
    },
    {
        # 3) SMALL STATE MACHINE: a turnstile (LOCKED/UNLOCKED) split across a transitions table + a machine that
        #    consumes events. Decomposition required: the machine drives transitions from the table; one without the
        #    other fails. The verify feeds an event SEQUENCE and checks the final state + a rejected-event guard.
        "id": "turnstile_fsm",
        "split": "train",
        "objective": (
            "Build a two-file finite state machine for a turnstile in the current working directory. "
            "(1) Write fsm_table.py defining TRANSITIONS, a dict mapping (state, event) -> next_state for a turnstile "
            "with states 'LOCKED' and 'UNLOCKED' and events 'coin' and 'push': "
            "('LOCKED','coin')->'UNLOCKED', ('LOCKED','push')->'LOCKED', ('UNLOCKED','push')->'LOCKED', "
            "('UNLOCKED','coin')->'UNLOCKED'. "
            "(2) Write fsm_machine.py that IMPORTS TRANSITIONS from fsm_table and defines run(events, start='LOCKED') "
            "which applies each event in the list `events` in order using TRANSITIONS and returns the FINAL state. "
            "If a (state, event) pair is not in TRANSITIONS, run must raise ValueError. No import-time printing in "
            "either file. Example: run(['coin','push']) == 'LOCKED'; run(['coin']) == 'UNLOCKED'."
        ),
        # composite: drive an event SEQUENCE through the machine + assert the final state, AND assert an unknown event
        # raises -> a missing transition, a wrong table, or no guard all REFUTE.
        "verify_cmd": _vc(
            "from fsm_machine import run; "
            "assert run(['coin','push'])=='LOCKED', run(['coin','push']); "
            "assert run(['coin'])=='UNLOCKED', run(['coin']); "
            "assert run(['coin','coin','push'])=='LOCKED'; "
            "assert run([])=='LOCKED'; "
            "import sys; "
            "raised=False;\n"
            "try:\n"
            "    run(['fly'])\n"
            "except ValueError:\n"
            "    raised=True\n"
            "assert raised, 'unknown event must raise ValueError'; print('FSM_OK')"),
        "oracle_artifacts": [
            ("fsm_table.py",
             "TRANSITIONS = {\n"
             "    ('LOCKED', 'coin'): 'UNLOCKED',\n"
             "    ('LOCKED', 'push'): 'LOCKED',\n"
             "    ('UNLOCKED', 'push'): 'LOCKED',\n"
             "    ('UNLOCKED', 'coin'): 'UNLOCKED',\n"
             "}\n"),
            ("fsm_machine.py",
             "from fsm_table import TRANSITIONS\n\n\n"
             "def run(events, start='LOCKED'):\n"
             "    state = start\n"
             "    for ev in events:\n"
             "        if (state, ev) not in TRANSITIONS:\n"
             "            raise ValueError(f'no transition for ({state!r}, {ev!r})')\n"
             "        state = TRANSITIONS[(state, ev)]\n"
             "    return state\n"),
        ],
    },
    {
        # 4) THREE-FILE COMPOSITION: constant -> transform -> pipeline runner. Stresses a 3-step chain (deeper
        #    decomposition than the 2-file tasks). normalize.py imports the BASE constant; pipeline.py imports the
        #    normalizer + runs it over a list. A missing middle file breaks the chain -> composite verify FAILS.
        "id": "scale_pipeline",
        "split": "train",
        "objective": (
            "Build a THREE-file numeric pipeline in the current working directory. "
            "(1) Write base_const.py defining BASE = 100. "
            "(2) Write normalize.py that imports BASE from base_const and defines normalize(x) returning "
            "round(x / BASE, 4). "
            "(3) Write run_pipeline.py that imports normalize from normalize and defines process(values) returning a "
            "list with normalize applied to each value, and grand_total(values) returning the SUM of the normalized "
            "values. No import-time printing in any file. "
            "Example: process([100, 50]) == [1.0, 0.5]; grand_total([100, 50, 50]) == 2.0."
        ),
        # composite: import the top-of-chain runner, check both the elementwise map AND the reduction over a known
        # input -> any broken link in the 3-file chain REFUTES.
        "verify_cmd": _vc(
            "from run_pipeline import process, grand_total; "
            "assert process([100,50])==[1.0,0.5], process([100,50]); "
            "assert grand_total([100,50,50])==2.0, grand_total([100,50,50]); "
            "assert process([])==[] and grand_total([])==0; print('PIPE_OK')"),
        "oracle_artifacts": [
            ("base_const.py", "BASE = 100\n"),
            ("normalize.py",
             "from base_const import BASE\n\n\n"
             "def normalize(x):\n"
             "    return round(x / BASE, 4)\n"),
            ("run_pipeline.py",
             "from normalize import normalize\n\n\n"
             "def process(values):\n"
             "    return [normalize(v) for v in values]\n\n\n"
             "def grand_total(values):\n"
             "    return sum(process(values))\n"),
        ],
    },
    # ======================================================================= HELD-OUT (test-split) PLANNER tasks
    # Added with the split work so the planner-mode held-out set is non-degenerate (~4 train / ~4 test). Each REQUIRES
    # a correct multi-file/multi-step decomposition: the consumer is useless without the dependency it imports, so a
    # missing OR mis-ordered sub-task leaves the composite import/result FAILING the verifier. Same schema + style as
    # the train tasks above (2- and 3-file build-then-use chains; pure stdlib; offline; deterministic).
    {
        # 5) RPN STACK EVALUATOR (2-file): an ops table + a stack evaluator that IMPORTS it and folds a token list.
        #    Decomposition required: the evaluator dispatches binary ops via the imported table; building either file
        #    alone fails the composite verify (evaluator without OPS = ImportError; OPS without evaluator = no run).
        "id": "rpn_eval",
        "split": "test",
        "objective": (
            "Build a two-file Reverse-Polish-Notation (RPN) calculator in the current working directory. "
            "(1) Write rpn_ops.py defining OPS, a dict mapping operator strings to two-argument functions: "
            "'+' -> add, '-' -> subtract, '*' -> multiply, '/' -> integer floor division (a // b). "
            "(2) Write rpn_eval.py that IMPORTS OPS from rpn_ops and defines evaluate(tokens) which evaluates a list "
            "of string tokens in RPN order using a stack: a numeric token (an integer string) is pushed; an operator "
            "token pops the top two values (second-popped is the left operand), applies the matching OPS function, and "
            "pushes the result; evaluate returns the single remaining integer on the stack. No import-time printing in "
            "either file. Example: evaluate(['3', '4', '+']) == 7; evaluate(['5', '1', '2', '+', '*']) == 15."
        ),
        # composite: import the evaluator (top of chain) + check several RPN expressions -> a missing op, a wrong
        # operand order (matters for - and /), or a missing import all REFUTE.
        "verify_cmd": _vc(
            "from rpn_eval import evaluate; "
            "assert evaluate(['3','4','+'])==7, evaluate(['3','4','+']); "
            "assert evaluate(['5','1','2','+','*'])==15, evaluate(['5','1','2','+','*']); "
            "assert evaluate(['10','2','-'])==8, evaluate(['10','2','-']); "
            "assert evaluate(['20','5','/'])==4, evaluate(['20','5','/']); "
            "assert evaluate(['7'])==7; print('RPN_OK')"),
        "oracle_artifacts": [
            ("rpn_ops.py",
             "OPS = {\n"
             "    '+': lambda a, b: a + b,\n"
             "    '-': lambda a, b: a - b,\n"
             "    '*': lambda a, b: a * b,\n"
             "    '/': lambda a, b: a // b,\n"
             "}\n"),
            ("rpn_eval.py",
             "from rpn_ops import OPS\n\n\n"
             "def evaluate(tokens):\n"
             "    stack = []\n"
             "    for tok in tokens:\n"
             "        if tok in OPS:\n"
             "            right = stack.pop()\n"
             "            left = stack.pop()\n"
             "            stack.append(OPS[tok](left, right))\n"
             "        else:\n"
             "            stack.append(int(tok))\n"
             "    return stack[-1]\n"),
        ],
    },
    {
        # 6) TOKENIZE -> TALLY (2-file): a tokenizer + a tally that IMPORTS it and counts token frequencies.
        #    Decomposition required: the tally is built on the tokenizer's output contract; tally alone (no tokenizer)
        #    or tokenizer alone fails the composite verify.
        "id": "tokenize_tally",
        "split": "test",
        "objective": (
            "Build a two-stage word-frequency tool (tokenize then tally) in the current working directory. "
            "(1) Write tokenizer.py defining tokenize(text) that lowercases the text, splits it on whitespace, and "
            "returns the list of word tokens (an empty string yields an empty list). "
            "(2) Write tally.py that IMPORTS tokenize from tokenizer and defines counts(text) returning a dict mapping "
            "each token to its frequency, and most_common(text) returning the token with the HIGHEST frequency (on a "
            "tie, the token that appears first in the text wins). No import-time printing in either file. "
            "Example: for text 'a b a A', counts(text) == {'a': 3, 'b': 1} and most_common(text) == 'a'."
        ),
        # composite: import the tally (top of chain), check the frequency dict AND the argmax over a known block ->
        # a missing tokenizer, a no-lowercase bug, or a wrong tie-break all REFUTE.
        "verify_cmd": _vc(
            "from tally import counts, most_common; "
            "assert counts('a b a A')=={'a':3,'b':1}, counts('a b a A'); "
            "assert most_common('a b a A')=='a', most_common('a b a A'); "
            "assert counts('')=={}; assert most_common('solo word solo')=='solo'; print('TALLY_OK')"),
        "oracle_artifacts": [
            ("tokenizer.py",
             "def tokenize(text):\n"
             "    return text.lower().split()\n"),
            ("tally.py",
             "from tokenizer import tokenize\n\n\n"
             "def counts(text):\n"
             "    out = {}\n"
             "    for tok in tokenize(text):\n"
             "        out[tok] = out.get(tok, 0) + 1\n"
             "    return out\n\n\n"
             "def most_common(text):\n"
             "    c = counts(text)\n"
             "    best = None\n"
             "    for tok in tokenize(text):\n"
             "        if best is None or c[tok] > c[best]:\n"
             "            best = tok\n"
             "    return best\n"),
        ],
    },
    {
        # 7) THREE-FILE TEMPERATURE PIPELINE: constant -> converter -> reporter. Stresses a 3-step chain; a missing
        #    middle file breaks the chain so the composite verify FAILS.
        "id": "temp_pipeline",
        "split": "test",
        "objective": (
            "Build a THREE-file temperature pipeline in the current working directory. "
            "(1) Write temp_const.py defining FREEZING_F = 32 and DEG_RATIO = 1.8. "
            "(2) Write convert.py that imports FREEZING_F and DEG_RATIO from temp_const and defines "
            "c_to_f(c) returning round(c * DEG_RATIO + FREEZING_F, 2). "
            "(3) Write report.py that imports c_to_f from convert and defines convert_all(celsius_list) returning the "
            "list of Fahrenheit values, and hottest(celsius_list) returning the MAXIMUM Fahrenheit value. "
            "No import-time printing in any file. "
            "Example: c_to_f(0) == 32.0; c_to_f(100) == 212.0; convert_all([0, 100]) == [32.0, 212.0]; "
            "hottest([0, 37, 100]) == 212.0."
        ),
        # composite: import the top-of-chain report, check the elementwise convert AND the reduction -> any broken
        # link in the 3-file chain REFUTES.
        "verify_cmd": _vc(
            "from report import convert_all, hottest; "
            "assert convert_all([0,100])==[32.0,212.0], convert_all([0,100]); "
            "assert hottest([0,37,100])==212.0, hottest([0,37,100]); "
            "assert convert_all([])==[] and hottest([0])==32.0; print('TEMP_OK')"),
        "oracle_artifacts": [
            ("temp_const.py", "FREEZING_F = 32\nDEG_RATIO = 1.8\n"),
            ("convert.py",
             "from temp_const import FREEZING_F, DEG_RATIO\n\n\n"
             "def c_to_f(c):\n"
             "    return round(c * DEG_RATIO + FREEZING_F, 2)\n"),
            ("report.py",
             "from convert import c_to_f\n\n\n"
             "def convert_all(celsius_list):\n"
             "    return [c_to_f(c) for c in celsius_list]\n\n\n"
             "def hottest(celsius_list):\n"
             "    return max(convert_all(celsius_list))\n"),
        ],
    },
    {
        # 8) THREE-FILE GRADE PIPELINE: thresholds -> classifier -> summarizer (a parse/classify/aggregate chain).
        #    The summarizer depends on the classifier which depends on the thresholds; a missing or mis-ordered file
        #    breaks the chain so the composite verify FAILS.
        "id": "grade_pipeline",
        "split": "test",
        "objective": (
            "Build a THREE-file grading pipeline in the current working directory. "
            "(1) Write grade_thresholds.py defining PASS_MARK = 50. "
            "(2) Write classify.py that imports PASS_MARK from grade_thresholds and defines classify(score) returning "
            "the string 'PASS' if score >= PASS_MARK else 'FAIL'. "
            "(3) Write summary.py that imports classify from classify and defines pass_count(scores) returning the "
            "number of scores that classify as 'PASS', and pass_rate(scores) returning the fraction of passing scores "
            "as a float rounded to 2 decimals (an empty list yields 0.0). No import-time printing in any file. "
            "Example: classify(50) == 'PASS'; classify(49) == 'FAIL'; pass_count([40, 50, 60]) == 2; "
            "pass_rate([40, 50, 60, 70]) == 0.75."
        ),
        # composite: import the top-of-chain summary, check both reductions over a known list AND the boundary case ->
        # a missing classifier, a wrong threshold/boundary, or a missing reduction all REFUTE.
        "verify_cmd": _vc(
            "from summary import pass_count, pass_rate; "
            "assert pass_count([40,50,60])==2, pass_count([40,50,60]); "
            "assert pass_rate([40,50,60,70])==0.75, pass_rate([40,50,60,70]); "
            "assert pass_count([])==0 and pass_rate([])==0.0; "
            "assert pass_count([49,50])==1, pass_count([49,50]); print('GRADE_OK')"),
        "oracle_artifacts": [
            ("grade_thresholds.py", "PASS_MARK = 50\n"),
            ("classify.py",
             "from grade_thresholds import PASS_MARK\n\n\n"
             "def classify(score):\n"
             "    return 'PASS' if score >= PASS_MARK else 'FAIL'\n"),
            ("summary.py",
             "from classify import classify\n\n\n"
             "def pass_count(scores):\n"
             "    return sum(1 for s in scores if classify(s) == 'PASS')\n\n\n"
             "def pass_rate(scores):\n"
             "    if not scores:\n"
             "        return 0.0\n"
             "    return round(pass_count(scores) / len(scores), 2)\n"),
        ],
    },
]


def benchmark_ids(tasks=None) -> list:
    return [t["id"] for t in (tasks if tasks is not None else BENCHMARK)]


def task_split(task: dict) -> str:
    """The TRAIN/TEST split label of a task. Tasks may omit the "split" key (legacy / hand-built tasks); those
    DEFAULT to "train" so nothing breaks and so a held-out optimizer never accidentally tunes on a test task it
    forgot to label. Returns "train" or "test"."""
    return task.get("split", "train")


def benchmark_split(tasks, which: str = "all") -> list:
    """Filter a benchmark by TRAIN/HELD-OUT split. This is the anti-overfit primitive: optimize against
    which="train", report the honest generalization number against which="test".

    Args:
      tasks : a benchmark list (BENCHMARK / PLANNER_BENCHMARK / any subset).
      which : "all"  -> every task unchanged (the legacy default; nothing is filtered out).
              "train"-> only tasks whose split is "train" (the optimizer/tuning set).
              "test" -> only tasks whose split is "test"  (the HELD-OUT set; never tuned on).

    A task with NO "split" key counts as "train" (see task_split). Returns a NEW list (does not mutate input)."""
    which = (which or "all").lower()
    if which == "all":
        return list(tasks)
    if which not in ("train", "test"):
        raise ValueError(f"benchmark_split: which must be 'all'|'train'|'test', got {which!r}")
    return [t for t in tasks if task_split(t) == which]


# --------------------------------------------------------------------------- the PLUMBING brain (honest plumbing)
class OracleMockBrain(MockBrain):
    """Deterministic PLUMBING brain: writes the KNOWN-CORRECT oracle_artifact for each built-in task to the build
    cwd. This is option (a) from the spec -- it lets the MockBrain plumbing test yield solve_rate=1.0, proving the
    SCORER + per-task ISOLATION + TIMEOUT all WORK end-to-end with no credentials.

    HONESTY: it does NOT fake the score. It writes a real file; the harness then runs the SAME mechanical verifier
    (graph._run_verify) against it. If the oracle artifact were wrong, the verifier would still REFUTE it. The mock
    only stands in for "a worker that produces a correct artifact"; the PASS still comes from the mechanical check.

    It maps a task by matching the verify_cmd's expected filename(s) against its known oracle table, so it works for
    any task whose objective names a file present in `artifacts`. MULTI-FILE tasks (oracle_artifacts) are written in
    FULL (every listed file). For an UNKNOWN task it falls back to the plain MockBrain behavior (which will NOT
    satisfy a real verify_cmd -> that task scores FAIL, honestly)."""
    name = "OracleMockBrain"

    def __init__(self, cwd: str, artifacts: dict | None = None, domain: str = "eval plumbing"):
        super().__init__(domain)
        self.cwd = cwd
        # artifacts: {key: list[(filename, content)]}. Every task's files are normalized to a LIST so single-file
        # (oracle_artifact) and multi-file (oracle_artifacts, the build-then-use planning tasks) share one code path.
        if artifacts is not None:
            self.artifacts = {k: (v if isinstance(v, list) else [v]) for k, v in artifacts.items()}
        else:
            # default oracle table covers BOTH the pre-seeded BENCHMARK and the PLANNER_BENCHMARK so the same mock can
            # plumb either mode (the planner-mode mock subclasses this to ALSO emit a real multi-node plan).
            self.artifacts = {}
            for t in (BENCHMARK + PLANNER_BENCHMARK):
                if "oracle_artifacts" in t:
                    self.artifacts[t["id"]] = list(t["oracle_artifacts"])
                elif "oracle_artifact" in t:
                    self.artifacts[t["id"]] = [t["oracle_artifact"]]

    def _match(self, task: str):
        """Pick the oracle file SET whose target filename(s) appear in the task text (the objective names the files).
        Returns a list[(filename, content)] for the matched task, or None when no known file is named."""
        for _key, files in self.artifacts.items():
            if any(fname in task for (fname, _content) in files):
                return files
        return None

    def work(self, task: str, persona: str = "") -> dict:
        hit = self._match(task)
        if hit is None:
            return super().work(task, persona=persona)  # unknown task -> plain mock (will not pass a real verifier)
        try:
            written = []
            for fname, content in hit:  # multi-file tasks get EVERY part written (build-then-use composes)
                (Path(self.cwd) / fname).write_text(content, encoding="utf-8")
                written.append(fname)
            return {"ok": True, "result": f"[oracle-mock] wrote correct artifact(s) {written}"}
        except Exception as e:
            return {"ok": False, "result": f"[oracle-mock] write failed: {type(e).__name__}: {e}"}


class PlannerOracleMockBrain(OracleMockBrain):
    """PLANNER-MODE plumbing brain. Unlike OracleMockBrain (which only WORKS -- it relies on the harness pre-seeding a
    node), this brain ALSO PLANS: in planner mode the harness pre-seeds NOTHING, so the graph's `plan` node calls
    decide('plan',...). A trivial empty plan would leave the loop with no work and every task would FAIL. So this mock
    DECOMPOSES the objective into a REAL multi-node frontier -- ONE build node per file the task's oracle artifacts
    name (so the plan genuinely reflects the decomposition) -- and its inherited work() then writes the matched oracle
    files. This proves the planner-mode PLUMBING end-to-end (plan -> decompose -> work -> the harness's OWN mechanical
    verify scores the composed artifact) with NO credentials.

    HONESTY: it does NOT fake the score. It emits a plan + writes real files; the harness independently re-runs its
    OWN composite verify_cmd on the final artifact (graph._run_verify). The plan-and-write here only stands in for
    'a brain that decomposed correctly + a worker that built each part'; the PASS still comes from the mechanical
    check on the composed result. For an UNKNOWN objective it falls back to the base MockBrain plan (n+-k generic),
    which will NOT satisfy a real composite verify_cmd -> that task scores FAIL, honestly."""
    name = "PlannerOracleMockBrain"

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        if role == "plan":
            obj = payload.get("objective", "") or ""
            files = self._match(obj)  # the oracle file set for THIS objective (matched by filename mentions)
            if files:
                # ONE build node per file, in the LISTED (dependency) order -- this IS the decomposition. Each node's
                # task names the file it must produce; the inherited work() matches the WHOLE objective text and writes
                # every oracle file, so executing ANY node materializes the composed artifact (the verify then passes).
                fr = []
                for i, (fname, _content) in enumerate(files):
                    fr.append({"id": f"p{i+1}_{fname.replace('.', '_')}",
                               "task": (f"Toward the objective, build the file {fname}. Full objective for context: "
                                        f"{obj}"),
                               "ev": round(0.9 - 0.05 * i, 3), "kind": "build", "status": "open"})
                # n+-k breadth (mirrors the real planner contract): a falsifier + a generalization node. These are
                # LLM-judged (no verify_cmd); they don't gate the harness's mechanical score but keep the plan honest.
                fr.append({"id": "p_falsifier", "kind": "verify", "status": "open", "ev": 0.5,
                           "task": f"falsifier (-k): confirm the composed files import + run end-to-end for: {obj[:120]}"})
                fr.append({"id": "p_generalize", "kind": "diverge", "status": "open", "ev": 0.4,
                           "task": f"generalization (+k): does the design extend to an adjacent case for: {obj[:120]}"})
                return {"frontier": fr}
            # unknown objective -> the base MockBrain n+-k plan (will not satisfy a composite verify -> honest FAIL)
            return super().decide(role, payload, persona=persona)
        return super().decide(role, payload, persona=persona)


# --------------------------------------------------------------------------- the scorer
def _seed_state(task: dict, verify_cmd: str, budget: int, run_id: str) -> dict:
    """OpState seeded with ONE build node carrying the harness-authored verify_cmd. Pre-seeding `frontier` makes the
    graph's plan node return {} early -> the loop works exactly OUR node (no LLM planning), fully bounded."""
    node = {"id": task["id"], "task": task["objective"], "ev": 0.95, "kind": "build",
            "status": "open", "verify_cmd": verify_cmd, "verify_retries": 2}
    return {"objective": task["objective"],
            "success_criteria": "the artifact exists AND its verify_cmd exits 0 (mechanical ground truth)",
            "frontier": [node], "ledger": [], "budget": budget, "cycle": 0, "status": "running",
            "parallel": 1, "run_id": run_id, "awaiting_approval": []}


def _run_one(brain, task: dict, build_dir: str, budget: int, timeout: int) -> dict:
    """Run a single task through the REAL graph in an ISOLATED build dir; SCORE the resulting artifact with the
    MECHANICAL VERIFIER. Returns a per-task record. Bounded by `timeout` wall-clock seconds; on timeout/exception
    the task scores FAIL with the reason in error_tail.

    HONESTY (the keystone): the PASS decision is the harness INDEPENDENTLY re-running its OWN trust-hardened
    verify_cmd (graph._run_verify, exit 0 == PASS) against the final artifact -- it does NOT trust the loop's `done`
    bookkeeping. This matters because the loop CAN declare a node 'done' without the artifact being correct (e.g. a
    REPLAN re-lists a node WITHOUT its verify_cmd -> it then gets LLM-judged 'pass'; an alt node passes; etc). The
    fitness signal must be the mechanical ground truth on the artifact, full stop -- so a wrong artifact scores FAIL
    regardless of what the brain/loop claims. We still REPORT the loop's view (loop_node_status / loop_verdict) for
    diagnostics, but they never decide the score."""
    verify_cmd = task["verify_cmd"](build_dir)
    run_id = f"eval-{task['id']}-{int(time.time() * 1000)}"
    t0 = time.time()
    passed = False
    node_status = "?"
    loop_verdict = "?"
    error_tail = ""
    timed_out = False
    try:
        # build the graph with cwd + workspace pinned to THIS task's isolated dir (no cross-contamination).
        app = _graph.build(brain, parallel=1, judges=1, taper=1, max_replans=1,
                           workspace=build_dir, cwd=build_dir)
        cfg = {"configurable": {"thread_id": run_id}}
        last = None
        for step in app.stream(_seed_state(task, verify_cmd, budget, run_id), cfg, stream_mode="values"):
            last = step
            if time.time() - t0 > timeout:
                error_tail = f"TIMEOUT after {timeout}s (loop did not converge)"
                timed_out = True
                break
        if last is not None:
            node = next((n for n in last.get("frontier", []) if n.get("id") == task["id"]), {})
            node_status = node.get("status", "?")
            loop_verdict = node.get("verdict", "?")
        # INDEPENDENT mechanical SCORE -- the harness re-runs its OWN verify_cmd on the artifact (ground truth).
        # This is the ONLY thing that decides PASS; the loop's done/verdict above is diagnostic only.
        code, tail = _graph._run_verify(verify_cmd, build_dir)
        passed = (code == 0) and not timed_out
        if not passed and not error_tail:
            error_tail = (f"mechanical verify_exit={code} (loop said status={node_status}/verdict={loop_verdict}) :: "
                          + (tail or "")).strip()[-1500:]
    except Exception as e:
        error_tail = f"EXCEPTION: {type(e).__name__}: {e}"
    return {"id": task["id"], "passed": passed, "seconds": round(time.time() - t0, 2),
            "node_status": node_status, "loop_verdict": loop_verdict, "error_tail": (error_tail or "")[-1500:]}


# --------------------------------------------------------------------------- PLANNER MODE (N9): measure decomposition
def _seed_state_planner(task: dict, budget: int, run_id: str) -> dict:
    """OpState with an EMPTY frontier -> the graph's `plan` node calls brain.decide('plan',...) and the BRAIN must
    DECOMPOSE the objective into the multi-step frontier itself (no pre-seeded node, no harness-supplied verify_cmd on
    the nodes). The loop then executes whatever plan the brain produced. Scoring is done AFTER by the harness's own
    composite verify_cmd on the final artifact (NOT by the loop), so the score reflects PLANNER QUALITY: a wrong /
    incomplete decomposition leaves the composed artifact failing the verifier -> FAIL.

    NOTE: success_criteria describes the composed end-to-end contract so a capable planner knows what 'done' means;
    it deliberately does NOT hand the brain the verify_cmd (the brain cannot author a trusted external check)."""
    return {"objective": task["objective"],
            "success_criteria": ("decompose the objective into the necessary files/steps, build EVERY part, and ensure "
                                 "the FINAL composed artifact behaves end-to-end as the objective specifies"),
            "frontier": [], "ledger": [], "budget": budget, "cycle": 0, "status": "running",
            "parallel": 1, "run_id": run_id, "awaiting_approval": []}


def _run_one_planner(brain, task: dict, build_dir: str, budget: int, timeout: int) -> dict:
    """PLANNER-MODE single-task run: like _run_one but the frontier starts EMPTY so the brain PLANS the decomposition
    (the graph `plan` node runs the brain's planner). The loop executes the planned nodes; the harness then SCORES the
    final composed artifact with its OWN mechanical composite verify_cmd (graph._run_verify) -- the only thing that
    decides PASS. Reports the planned-frontier shape (planned_nodes / planned_kinds) for diagnostics so a low score
    can be attributed to a BAD plan vs a bad execution. Bounded by `timeout`."""
    verify_cmd = task["verify_cmd"](build_dir)  # the harness's composite check (NOT given to the brain)
    run_id = f"planner-{task['id']}-{int(time.time() * 1000)}"
    t0 = time.time()
    passed = False
    planned_nodes = 0
    planned_kinds: dict = {}
    done_nodes = 0
    error_tail = ""
    timed_out = False
    try:
        # plan_critique left ON (it is part of the planner under test); replans bounded so a hung plan can't wedge.
        app = _graph.build(brain, parallel=1, judges=1, taper=1, max_replans=1,
                           workspace=build_dir, cwd=build_dir)
        cfg = {"configurable": {"thread_id": run_id}}
        last = None
        for step in app.stream(_seed_state_planner(task, budget, run_id), cfg, stream_mode="values"):
            last = step
            if time.time() - t0 > timeout:
                error_tail = f"TIMEOUT after {timeout}s (planner/loop did not converge)"
                timed_out = True
                break
        if last is not None:
            fr = [n for n in last.get("frontier", []) if isinstance(n, dict)]
            planned_nodes = len(fr)
            for n in fr:
                k = (n.get("kind") or "?")
                planned_kinds[k] = planned_kinds.get(k, 0) + 1
            done_nodes = sum(1 for n in fr if n.get("status") == "done")
        # INDEPENDENT mechanical SCORE on the composed artifact -- ground truth, decided by the HARNESS not the loop.
        code, tail = _graph._run_verify(verify_cmd, build_dir)
        passed = (code == 0) and not timed_out
        if not passed and not error_tail:
            error_tail = (f"mechanical verify_exit={code} (planner emitted {planned_nodes} node(s) "
                          f"{planned_kinds}, {done_nodes} done) :: " + (tail or "")).strip()[-1500:]
    except Exception as e:
        error_tail = f"EXCEPTION: {type(e).__name__}: {e}"
    return {"id": task["id"], "passed": passed, "seconds": round(time.time() - t0, 2),
            "planned_nodes": planned_nodes, "planned_kinds": planned_kinds, "done_nodes": done_nodes,
            "error_tail": (error_tail or "")[-1500:]}


def run_eval(brain, tasks=None, build_cwd_base: str | None = None, budget: int = 4, timeout: int = 240,
             brain_label: str | None = None, oracle_mock: bool = False, planner_mode: bool = False,
             split: str = "all") -> dict:
    """Run a benchmark through the engine and return a SCORECARD dict (the fitness signal).

    Args:
      brain        : a Brain (mock / litellm->ollama / cli / ...). IGNORED when oracle_mock=True (a per-task
                     OracleMockBrain / PlannerOracleMockBrain is built instead, since it must know each task's cwd).
      tasks        : the benchmark (default = BENCHMARK pre-seeded; PLANNER_BENCHMARK when planner_mode=True).
      build_cwd_base : parent dir for the per-task isolated temp dirs (default = a fresh system temp dir).
      budget       : max graph cycles per task (bounded).
      timeout      : per-task wall-clock cap in seconds (a hung task can't wedge the eval).
      brain_label  : how to label the brain in the scorecard (default = brain.name, or the oracle-mock name).
      oracle_mock  : True -> use the deterministic oracle plumbing brain (honest PLUMBING path; the score still comes
                     from the mechanical verifier, proving the scorer/isolation/timeout work).
      planner_mode : True -> PLANNER mode (N9). The frontier is NOT pre-seeded: the brain's `plan` node must DECOMPOSE
                     the objective into the multi-step frontier and the loop executes it; the harness still scores the
                     FINAL composed artifact with its OWN composite verify_cmd. The score reflects PLANNER QUALITY.
                     Default tasks become PLANNER_BENCHMARK; the oracle-mock brain becomes PlannerOracleMockBrain.
      split        : "all" (DEFAULT, current behavior -- run every task) | "train" (tuning set) | "test" (HELD-OUT).
                     The anti-overfit knob: optimize against split="train", report the honest generalization number
                     against split="test". Filtering is applied AFTER `tasks` is resolved (so it works on the default
                     benchmark OR an explicit subset). split="all" preserves the exact pre-split behavior.

    Returns (pre-seeded): {brain, mode, split, solve_rate, n, n_passed, budget, timeout, split_breakdown,
                           per_task:[{id,passed,seconds,split,node_status,loop_verdict,error_tail}]}
    Returns (planner)   : {brain, mode, split, solve_rate, n, n_passed, budget, timeout, split_breakdown,
                           per_task:[{id,passed,seconds,split,planned_nodes,planned_kinds,done_nodes,error_tail}]}
    `split` echoes the requested filter; `split_breakdown` is the per-split {train,test} solve_rate within the tasks
    actually run (so a split="all" run still reports train-vs-test separately).
    (timestamp-less; the CLI/caller stamps `ts`). Each task runs in its own isolated temp dir, cleaned up after."""
    default_tasks = PLANNER_BENCHMARK if planner_mode else BENCHMARK
    tasks = list(tasks if tasks is not None else default_tasks)
    tasks = benchmark_split(tasks, split)  # split="all" is a no-op (legacy behavior); train/test filter the set
    # remember each task's split so the per-task record + the per-split breakdown can be computed even on split="all".
    _split_of = {t["id"]: task_split(t) for t in tasks}
    base = Path(build_cwd_base) if build_cwd_base else Path(tempfile.mkdtemp(prefix="metaop_eval_"))
    base.mkdir(parents=True, exist_ok=True)
    oracle_name = "PlannerOracleMockBrain" if planner_mode else "OracleMockBrain"
    label = brain_label or (oracle_name if oracle_mock else getattr(brain, "name", "Brain"))
    per_task = []
    for task in tasks:
        build_dir = tempfile.mkdtemp(prefix=f"task_{task['id']}_", dir=str(base))
        try:
            if oracle_mock:
                b = PlannerOracleMockBrain(cwd=build_dir) if planner_mode else OracleMockBrain(cwd=build_dir)
            else:
                b = brain
            # if a real (non-oracle) brain carries a writable cwd, point it at THIS task's dir so its worker writes
            # where the verifier runs (litellm/ollama/sdk brains expose `.cwd`).
            if not oracle_mock and hasattr(b, "cwd"):
                try:
                    b.cwd = build_dir
                except Exception:
                    pass
            rec = (_run_one_planner(b, task, build_dir, budget=budget, timeout=timeout) if planner_mode
                   else _run_one(b, task, build_dir, budget=budget, timeout=timeout))
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)
        rec["split"] = _split_of.get(rec["id"], "train")  # annotate so split_breakdown + downstream can group
        per_task.append(rec)
        if planner_mode:
            diag = f"plan={rec['planned_nodes']}n {rec['planned_kinds']} done={rec['done_nodes']}"
        else:
            diag = f"node={rec['node_status']}"
        print(f"  [{rec['id']:<16}] {'PASS' if rec['passed'] else 'FAIL'}  {rec['seconds']:>6.1f}s  {diag}"
              + (f"  :: {rec['error_tail'][:90]}" if not rec['passed'] else ""))
    if build_cwd_base is None:
        shutil.rmtree(base, ignore_errors=True)
    n = len(per_task)
    n_passed = sum(1 for r in per_task if r["passed"])
    # per-split breakdown over the tasks that actually ran (useful even on split="all": train vs test side-by-side).
    split_breakdown = {}
    for sp in ("train", "test"):
        rows = [r for r in per_task if r.get("split") == sp]
        if rows:
            npass = sum(1 for r in rows if r["passed"])
            split_breakdown[sp] = {"n": len(rows), "n_passed": npass,
                                   "solve_rate": round(npass / len(rows), 4)}
    return {"brain": label, "mode": "planner" if planner_mode else "pre-seeded", "split": split,
            "solve_rate": round(n_passed / n, 4) if n else 0.0, "n": n, "n_passed": n_passed,
            "budget": budget, "timeout": timeout, "split_breakdown": split_breakdown, "per_task": per_task}


def run_planner_eval(brain, tasks=None, build_cwd_base: str | None = None, budget: int = 4, timeout: int = 240,
                     brain_label: str | None = None, oracle_mock: bool = False, split: str = "all") -> dict:
    """Convenience wrapper = run_eval(..., planner_mode=True). The PLANNER-quality fitness signal: the brain must
    DECOMPOSE the objective itself (no pre-seeded node) and the loop must execute the plan; the harness scores the
    final COMPOSED artifact mechanically. This is the honest objective a future DSPy pass optimizes the planner prompt
    against. Default tasks = PLANNER_BENCHMARK; oracle_mock=True -> the PlannerOracleMockBrain plumbing path.

    split : "all" (DEFAULT, current behavior) | "train" (tune against this) | "test" (HELD-OUT generalization). A
            future DSPy pass MUST optimize against split="train" and REPORT split="test" to avoid overfitting the
            planner prompt to the very tasks it tuned on."""
    return run_eval(brain, tasks=tasks, build_cwd_base=build_cwd_base, budget=budget, timeout=timeout,
                    brain_label=brain_label, oracle_mock=oracle_mock, planner_mode=True, split=split)


def write_scorecard(scorecard: dict, label: str, workspace: str | None = None) -> Path:
    """Persist the scorecard (the fitness signal) to <workspace>/eval/<label>.json (stamps `ts`). The default
    workspace is the harness workspace (config.workspace_root); the crypto CLI points it at runs/autonomy."""
    from .config import workspace_root
    out_dir = workspace_root(workspace) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = dict(scorecard)
    sc["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = out_dir / f"{label}.json"
    out.write_text(json.dumps(sc, indent=2, default=str), encoding="utf-8")
    return out


def _selftest_splits() -> int:
    """RWYB self-test: run the oracle plumbing on the TRAIN and HELD-OUT (test) splits in BOTH modes and assert
    solve_rate == 1.0 on each. This proves every task's oracle artifact(s) satisfy its OWN verify_cmd AND that the
    split filter is wired correctly (a task whose oracle fails its own check is a BROKEN task -> this catches it).
    Returns 0 on success, 1 on any split scoring < 1.0 (so it can gate CI)."""
    ok = True
    plans = [("PRE-SEEDED", False), ("PLANNER", True)]
    for mode_label, planner in plans:
        for sp in ("train", "test"):
            card = run_eval(None, oracle_mock=True, planner_mode=planner, split=sp)
            rate = card["solve_rate"]
            status = "OK" if rate == 1.0 else "FAIL"
            print(f"[selftest] {mode_label:<10} split={sp:<5} solve_rate={rate}  ({card['n_passed']}/{card['n']})  {status}")
            if rate != 1.0:
                ok = False
                for r in card["per_task"]:
                    if not r["passed"]:
                        print(f"    BROKEN TASK: {r['id']} :: {r['error_tail'][:200]}")
    print("[selftest] ALL SPLITS solve_rate=1.0" if ok else "[selftest] BROKEN: a split did NOT reach solve_rate=1.0")
    return 0 if ok else 1


if __name__ == "__main__":
    # If invoked with --selftest, run the split RWYB gate (both modes x train/test -> solve_rate must be 1.0 each).
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest_splits())
    # quick self-demo (plumbing): both modes via the deterministic oracle brains (split="all" = full benchmark).
    print("EVAL HARNESS self-demo -- PRE-SEEDED (OracleMockBrain plumbing)")
    card = run_eval(None, oracle_mock=True)
    print(json.dumps(card, indent=2))
    print("\nEVAL HARNESS self-demo -- PLANNER (PlannerOracleMockBrain plumbing)")
    pcard = run_eval(None, oracle_mock=True, planner_mode=True)
    print(json.dumps(pcard, indent=2))
