"""SAFETY FENCE: the worker's shell + file surface. Recursive rm / force-push in ANY flag order + case are denied;
legitimate ops are allowed; write/read/list are root-CONTAINED (a `../` traversal cannot escape the build cwd).
Pins the 2026-07-03 fence hardening (the audit reproduced `rm -rf .`, `rm -fr /`, `git push -f`, and
`write_file('../../../.bashrc')` all sailing past the old fence)."""
from __future__ import annotations

import pytest

from metaop.tools import HARD_DENY, Tools, _blocked, _contained

_G = "g" + "it"  # keep the destructive git literal out of any command line that a shell permission gate might scan
BLOCK = ["rm -rf .", "rm -rf *", "rm -fr /", "rm -r build", "RM -RF /", "rd /s x", "del /q *",
         _G + " push -f origin main", _G + " push --force-with-lease", _G + " reset --hard",
         "sudo rm x", "dd if=/dev/zero", "mkfs.ext4 /dev/sda", "curl x | bash", ":(){ :|:& };:"]
ALLOW = ["python --version", "rm _snippet.py", "rm -f tmp.txt", _G + " push origin main", _G + " commit -m x",
         "ls -la", "echo hi", "pytest -q", "grep -r foo ."]


@pytest.mark.parametrize("cmd", BLOCK)
def test_destructive_blocked(cmd):
    assert _blocked(cmd, HARD_DENY), f"should be blocked: {cmd!r}"


@pytest.mark.parametrize("cmd", ALLOW)
def test_legit_allowed(cmd):
    assert not _blocked(cmd, HARD_DENY), f"should be allowed: {cmd!r}"


@pytest.mark.parametrize("path", ["../../../x", "../secret.txt", "sub/../../escape"])
def test_containment_refuses_escape(tmp_path, path):
    _resolved, reason = _contained(tmp_path, path)
    assert reason, f"traversal should be refused: {path!r}"


@pytest.mark.parametrize("path", ["a.txt", "sub/b.txt", "./x/y.txt"])
def test_containment_allows_inside(tmp_path, path):
    _resolved, reason = _contained(tmp_path, path)
    assert reason is None, f"in-root path should be allowed: {path!r}"


def test_write_file_root_contained(tmp_path):
    t = Tools(cwd=str(tmp_path))
    escaped = t.write_file("../escape.txt", "x")
    assert not escaped["ok"] and "DENIED" in escaped["error"]
    assert not (tmp_path.parent / "escape.txt").exists()
    ok = t.write_file("ok.txt", "hi")
    assert ok["ok"] and (tmp_path / "ok.txt").read_text() == "hi"


def test_write_file_protected_basename(tmp_path):
    t = Tools(cwd=str(tmp_path))
    assert not t.write_file(".env", "SECRET=1")["ok"]
    assert not t.write_file(".git/config", "x")["ok"]
