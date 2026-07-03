"""TRUST CORE: the mechanical verifier must be UNSPOOFABLE. A verify_cmd that can only ever exit 0 asserts nothing
and must be REJECTED (so a swapped weak brain can't fake a green PASS); a real assertion must be ALLOWED; a
destructive command must be refused. Pins the 2026-07-03 anti-spoof fix (compound / python -c / tautology bypasses)."""
from __future__ import annotations

import pytest

from metaop.graph import _screen_verify_cmd

TRIVIAL = [  # always exit 0, assert nothing -> MUST be rejected
    "true", ":", "exit 0", "exit", "echo ok", "echo done && true", "true && true", "true || true",
    ": ; echo ok", 'python -c "exit(0)"', "python -c 'exit(0)'", 'python -c "print(1)"', "python -c 'pass'",
    'python3 -c "sys.exit(0)"', "test 1 = 1", "[ 1 = 1 ]", "test foo = foo", "[ -n x ]", "true; true; true",
    # 2026-07-03 round-2: the re-audit's named residual bypasses (constant python -c bodies, subshells,
    # background &, cwd/root existence tautologies) must now ALSO be rejected -- positive rule, not a blocklist.
    'python -c "1==1"', 'python -c "len([])"', 'python -c "True"', 'python -c "0"', 'python -c "1 == 1"',
    'python -c "True and False"', 'python -c "None"', "python -c \"len('')\"", "( true )", "(:)", "true & true",
    "( echo ok )", "test -e .", "test -d .", "test -f /", "[ -e . ]",
]
LEGIT = [  # a genuine assertion that CAN fail -> MUST be allowed
    'python -c "import m; assert m.f()==5"', "pytest -q", "python -m pytest test_x.py", "python verify.py",
    'python -c "assert 1+1==2"', "grep -q PASS out.txt", "node test.js", 'python -c "raise SystemExit(1 if bad else 0)"',
    # these CAN fail (import error / missing file / conditional exit / real-artifact existence) -> must stay allowed
    'python -c "import adder"', "python -c \"open('out.txt')\"", 'python -c "sys.exit(0 if ok else 1)"',
    "test -e output.txt",
]
DESTRUCTIVE = [  # side-effecting / irreversible -> MUST be refused (a verifier is a read-only assertion)
    "rm -rf .", "rm -rf *", "rm -fr /", "rm -r build", "RM -RF /", "sudo rm x", "mkfs.ext4 /dev/sda",
    "dd if=/dev/zero of=/dev/sda", "g" "it push --force origin main",
]


@pytest.mark.parametrize("cmd", TRIVIAL)
def test_trivial_rejected(cmd):
    assert _screen_verify_cmd(cmd) is not None, f"trivial no-op should be rejected: {cmd!r}"


@pytest.mark.parametrize("cmd", LEGIT)
def test_legit_allowed(cmd):
    assert _screen_verify_cmd(cmd) is None, f"genuine assertion should be allowed: {cmd!r}"


@pytest.mark.parametrize("cmd", DESTRUCTIVE)
def test_destructive_rejected(cmd):
    assert _screen_verify_cmd(cmd) is not None, f"destructive command should be refused: {cmd!r}"
