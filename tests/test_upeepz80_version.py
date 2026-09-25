"""uplm80 refuses an upeepz80 too old for PL/M-80's calling convention.

upeepz80 0.2.5 turned `push ... / call p / ret' into `push ... / jp p', and
p, which takes the words pushed for it off the stack, took its return
address for its first argument.  The compiler compares upeepz80.__version__
with UPEEPZ80_MIN, pyproject.toml's floor, before it optimizes; below it, it
accepts only a development tree, still numbered 0.2.5, that keeps the `call'
of a probe.  These tests stand in for other upeepz80s with a version string
and an optimizer that does or does not make that tail call.
"""

import re
from pathlib import Path

import pytest
import upeepz80

from uplm80 import compiler as C
from uplm80.compiler import Compiler

SRC = "t: do;\ndeclare x byte;\nx = 1;\nend t;\n"


class _TailCalling:
    """Makes `call x / ret' a `jp x' wherever it is, as 0.2.5 did."""

    def __init__(self) -> None:
        self.stats: dict[str, int] = {}

    def optimize(self, text: str) -> str:
        return re.sub(r"\tcall\t(\S+)\n\tret\n", r"\tjp\t\1\n", text)


class _Keeping:
    """Leaves the text as it is, so keeps the probe's `call', as 0.2.6 does."""

    def __init__(self) -> None:
        self.stats: dict[str, int] = {}

    def optimize(self, text: str) -> str:
        return text


@pytest.fixture
def fake_upeepz80(monkeypatch):
    """Install ``version`` and ``optimizer`` as upeepz80's, for one test."""
    def install(version: str, optimizer: type) -> None:
        monkeypatch.setattr(upeepz80, "__version__", version)
        monkeypatch.setattr(C, "PeepholeOptimizer", optimizer)
        C.upeepz80_problem.cache_clear()
    yield install
    C.upeepz80_problem.cache_clear()


def _compile(opt: int) -> tuple[str | None, list[str]]:
    compiler = Compiler(opt_level=opt)
    out = compiler.compile(SRC, "<test>")
    return out, [str(e) for e in compiler.errors.errors]


def test_the_floor_is_pyprojects():
    """UPEEPZ80_MIN is the version pyproject.toml's dependency requires."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    m = re.search(r'"upeepz80>=([^"]+)"', pyproject.read_text())
    assert m is not None
    assert m.group(1) == C.UPEEPZ80_MIN


def test_the_upeepz80_in_use_is_accepted():
    C.upeepz80_problem.cache_clear()
    assert C.upeepz80_problem() is None


def test_the_upeepz80_in_use_keeps_the_probes_call():
    """The probe is a call 0.2.6 keeps, so a tree with the fix passes it."""
    out = upeepz80.PeepholeOptimizer().optimize(C._TAIL_CALL_PROBE)
    assert re.search(r"^\s*call\s+P3\b", out, re.MULTILINE)


@pytest.mark.parametrize("version", ["0.2.5", "0.2.4", "0.1.9"])
def test_a_release_before_the_floor_is_refused(fake_upeepz80, version):
    fake_upeepz80(version, _TailCalling)
    out, errors = _compile(2)
    assert out is None
    assert len(errors) == 1
    assert f"upeepz80 {version} is too old" in errors[0]
    assert f"needs upeepz80 {C.UPEEPZ80_MIN} or later" in errors[0]
    assert "-O0" in errors[0]


def test_a_release_before_the_floor_is_refused_for_several_modules(
        fake_upeepz80, tmp_path, capsys):
    fake_upeepz80("0.2.5", _TailCalling)
    main = tmp_path / "main.plm"
    main.write_text("m: do;\nlib: procedure external; end lib;\ncall lib;\nend m;\n")
    lib = tmp_path / "lib.plm"
    lib.write_text("l: do;\nlib: procedure public; end lib;\nend l;\n")
    ok = Compiler(opt_level=2).compile_files([main, lib], tmp_path / "out.mac")
    assert not ok
    assert "upeepz80 0.2.5 is too old" in capsys.readouterr().err
    assert not (tmp_path / "out.mac").exists()


def test_minus_o0_does_not_use_upeepz80(fake_upeepz80):
    fake_upeepz80("0.2.5", _TailCalling)
    out, errors = _compile(0)
    assert out is not None, errors


def test_a_development_tree_with_the_fix_is_accepted(fake_upeepz80):
    """Numbered 0.2.5, but it keeps the probe's `call'."""
    fake_upeepz80("0.2.5", _Keeping)
    out, errors = _compile(2)
    assert out is not None, errors


@pytest.mark.parametrize("version", ["0.2.6", "0.2.10", "0.3.0", "1.0"])
def test_a_version_at_or_past_the_floor_is_taken_at_its_word(fake_upeepz80, version):
    """No probe: the version says the fix is there (0.2.10 is past 0.2.6)."""
    fake_upeepz80(version, _TailCalling)
    assert C.upeepz80_problem() is None
