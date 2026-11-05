"""`uplm80 -g` writes ;# source markers and leaves the code as it is.

The markers are whole-line comments of the form in um80_and_friends
docs/debug_info.md (cpmemu docs/debugger_plan.md, D2 and D3).  The tests
here are of three kinds:

* what the markers say (files, lines, routines, frames, inline calls);
* D2, the rule that matters most: a program compiled with -g and without it
  is the same .COM at every -O level, and the same text but for the marker
  lines;
* with um80 and ul80 --debug-info: each instruction of the program's own
  routines has a source line in prog.dbg.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import upeepz80

from uplm80 import compiler as C
from uplm80.compiler import Compiler

HERE = Path(__file__).parent
LEVELS = (0, 1, 2, 3)

# A marker is ;, optional blanks, #, a kind in letters; the writer is exact.
MARKER = re.compile(r"^;#(file|line|inline|endinline|func|endfunc)\b")
# The labels that -g adds for ixbegin= and ixend= and for the range= of a
# parameter (the peephole pass is over), and the variable markers.
IX_LABEL = re.compile(r"^\?\?(IX[BE]|VR)\d+:$")
VAR_MARKER = re.compile(r"^;#(type|var)\b")


def compile_g(src: str, opt: int = 2, name: str = "<test>", g: bool = True) -> str:
    out = Compiler(opt_level=opt, debug_info=g).compile(src, name)
    assert out is not None, "compilation failed"
    return out


def markers(asm: str) -> list[str]:
    return [ln for ln in asm.split("\n") if MARKER.match(ln)]


def strip_markers(asm: str) -> str:
    return "\n".join(ln for ln in asm.split("\n")
                     if not MARKER.match(ln) and not VAR_MARKER.match(ln)
                     and not IX_LABEL.match(ln))


PROGRAM = """0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare n address;

three: procedure (a, b, c) address;
    declare (a, b, c) address;
    return a + b + c;
end three;

five: procedure (a, b, c, d, e) address;
    declare (a, b, c, d, e) address;
    return a + e;
end five;

four: procedure (a, b, c, d) address reentrant;
    declare (a, b, c, d) address;
    declare x address;
    x = a + b;
    x = x + c + d;
    return x;
end four;

zero: procedure;
    n = n + 1;
end zero;

n = three(1, 2, 3);
n = five(1, 2, 3, 4, 5);
n = four(1, 2, 3, 4);
call zero;
call mon1(2, low(n));
end t;
"""


def line_of(text: str, src: str = PROGRAM) -> int:
    """The line of ``src`` (1-based) that has ``text`` first."""
    return next(i for i, ln in enumerate(src.split("\n"), 1) if text in ln)


# upeepz80 0.2.7, the release on PyPI on 2026-11-04, lets a ;# marker hide
# a dead store, and compile() refuses -g at -O1 and above with it.  Those
# tests then skip with the reason (fail under UPLM80_REQUIRE_TOOLS=1) until
# a newer upeepz80 is released; todo.txt has the release item.
_NO_UPEEPZ80 = ("test_markers_off_by_default",
                "test_an_upeepz80_that_lets_a_marker_hide_a_dead_store_is_refused")


@pytest.fixture(autouse=True)
def _upeepz80_keeps_markers(request):
    if request.node.originalname in _NO_UPEEPZ80:
        return
    callspec = getattr(request.node, "callspec", None)
    opt = callspec.params.get("opt", 2) if callspec else 2
    if opt == 0:
        return
    C.upeepz80_markers_problem.cache_clear()
    problem = C.upeepz80_markers_problem()
    if problem:
        _missing(problem)


def test_markers_off_by_default():
    assert markers(compile_g(PROGRAM, g=False)) == []


@pytest.mark.parametrize("opt", LEVELS)
def test_every_marker_has_the_exact_form(opt):
    asm = compile_g(PROGRAM, opt)
    ms = markers(asm)
    assert ms
    for m in ms:
        assert re.fullmatch(
            r";#(file [1-9]\d* \"[^\"]*\""
            r"|line \d+ \d+"
            r"|inline [1-9]\d* \d+ \S+"
            r"|endinline"
            r"|func \S+ \d+ \d+ frame=static pops=\d+"
            r"|func \S+ \d+ \d+ frame=ix pops=\d+ ixbegin=\S+ ixend=\S+"
            r"|func \S+ \d+ \d+ frame=none pops=0"
            r"|endfunc \S+)", m), m
    # whole lines, in column 1: never a trailing comment
    assert not [ln for ln in asm.split("\n") if ";#" in ln and not ln.startswith(";#")]


def test_a_routine_has_func_endfunc_frame_and_pops():
    asm = compile_g(PROGRAM)
    funcs = {m.split()[1]: m for m in markers(asm) if m.startswith(";#func")}
    for name, text, frame, pops in (
            ("THREE", "three: procedure", "static", 2),
            ("FIVE", "five: procedure", "static", 6),     # the 3rd to 5th are on the stack
            ("FOUR", "four: procedure", "ix", 4),
            ("ZERO", "zero: procedure", "static", 0)):
        want = f";#func {name} 1 {line_of(text)} frame={frame} pops={pops}"
        assert funcs[name].startswith(want)
        assert (" ixbegin=" in funcs[name]) == (frame == "ix")
    # the module's own statements are a routine named for the module
    assert funcs["T"] == f";#func T 1 {line_of('n = three(1, 2, 3);')} frame=none pops=0"
    ends = [m.split()[1] for m in markers(asm) if m.startswith(";#endfunc")]
    assert sorted(ends) == sorted(funcs)
    # a func comes right after its label
    lines = asm.split("\n")
    assert lines[lines.index("THREE:") + 1].startswith(";#func THREE")


def test_each_statement_has_the_line_it_is_on():
    asm = compile_g(PROGRAM)
    got = {int(m.split()[2]) for m in markers(asm) if m.startswith(";#line")}
    for text in ("return a + b + c;", "x = a + b;", "x = x + c + d;", "return x;",
                 "n = n + 1;", "n = three(1, 2, 3);", "n = four(1, 2, 3, 4);",
                 "call zero;", "call mon1(2, low(n));"):
        assert line_of(text) in got, text


def test_the_epilogue_is_the_line_of_the_end():
    asm = compile_g(PROGRAM)
    after = asm[asm.index("ZERO:"):]
    # `zero' has no RETURN: its `ret' is on the line of `end zero;'
    assert re.search(rf";#line 1 {line_of('end zero;')}\n\tret\n;#endfunc ZERO", after)


def test_the_jump_table_of_a_big_case_has_no_line():
    src = ("0100H:\nt: do;\ndeclare (i, r) byte;\ndo case i;\n"
           + "".join(f"  r = {k};\n" for k in range(10)) + "end;\nend t;\n")
    asm = compile_g(src)
    before, after = asm.split("\tdw\t", 1)
    assert ";#line 0 0" in before.rsplit("jp\t(hl)", 1)[1]
    # each case, after the table, has its line again
    assert f";#line 1 {line_of('r = 0;', src)}" in after


@pytest.mark.parametrize("opt", (1, 2, 3))
def test_the_lines_of_an_include_and_a_literally_are_the_source_s(opt, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib.inc").write_text(
        "twice: procedure (v) byte;\n  declare v byte;\n  return v + v;\nend twice;\n")
    (tmp_path / "main.plm").write_text(
        "m: do;\ndeclare r byte;\n"
        "declare big literally 'twice(3) +\n   1';\n"
        "$include (lib.inc)\n"
        "r = big;\n"
        "r = r + twice(r);\n"
        "end m;\n")
    out = tmp_path / "main.mac"
    assert Compiler(opt_level=opt, debug_info=True).compile_file(tmp_path / "main.plm", out)
    asm = out.read_text()
    ms = markers(asm)
    assert ';#file 1 "' + str(tmp_path / "main.plm") + '"' in ms or any(
        m.startswith(";#file 1 ") and m.endswith('main.plm"') for m in ms)
    inc = next(m for m in ms if m.startswith(";#file 2 "))
    assert inc.endswith('lib.inc"')
    assert ";#func TWICE 2 1 frame=static pops=0" in ms
    assert ";#line 2 3" in ms                      # the RETURN, in the include file
    # statements of main.plm are on their own lines, not the preprocessed text's
    assert ";#line 1 6" in ms and ";#line 1 7" in ms


def test_an_inlined_call_has_inline_and_endinline_at_O3():
    src = ("0100H:\nt: do;\ndeclare total address;\ndeclare i byte;\n"
           "bump: procedure;\n  total = total + 1;\nend bump;\n"
           "do i = 1 to 3;\n  call bump;\n  total = total + i;\nend;\nend t;\n")
    asm = compile_g(src, 3)
    ms = markers(asm)
    assert ";#inline 1 9 BUMP" in ms
    at = ms.index(";#inline 1 9 BUMP")
    assert ms[at + 1] == ";#line 1 6" and ms[at + 2] == ";#endinline"
    # the call is inlined at -O3 only, and the other levels have no record
    assert not [m for m in markers(compile_g(src, 2)) if m.startswith(";#inline")]


@pytest.mark.parametrize("opt", LEVELS)
def test_a_reentrant_routine_has_the_range_where_ix_holds_the_frame(opt):
    asm = compile_g(PROGRAM, opt)
    func = next(m for m in markers(asm) if m.startswith(";#func FOUR "))
    begin = re.search(r" ixbegin=(\S+)", func).group(1)
    end = re.search(r" ixend=(\S+)", func).group(1)
    lines = asm.split("\n")
    # ixbegin is right after `add ix,sp'; ixend is right before the last `pop ix'
    assert lines[lines.index(begin + ":") - 1].split() == ["add", "ix,sp"]
    assert lines[lines.index(end + ":") + 1].split() == ["pop", "ix"]
    assert lines.index(begin + ":") < lines.index(end + ":") < lines.index(";#endfunc FOUR")
    # a routine that is not REENTRANT has none
    assert " ixbegin=" not in next(m for m in markers(asm) if m.startswith(";#func THREE "))


OPTIMIZED = """0100H:
t: do;
declare (a, b, c, r, s, u, v) address;
declare i byte;
r = (a + b) * c;
s = (a + b) * 3 + (a + b);
u = 2 * 8 + 1;
do i = 0 to 1;
  r = r + 1;
  v = r + b;
end;
c = 1 + 2;
end t;
"""


@pytest.mark.parametrize("opt", (2, 3))
def test_the_optimizers_keep_the_line_of_a_statement_they_rewrite(opt):
    """CSE (line 6 and 7), a folded constant (8 and 12) and, at -O3, an
    unrolled loop (9 to 11): each statement has its own ;#line, with its
    own code after it, and none has the previous statement's line."""
    asm = compile_g(OPTIMIZED, opt)
    chunks: dict[int, list[str]] = {}
    here = None
    for ln in asm[asm.index(";#func T "):asm.index(";#endfunc T")].split("\n"):
        m = re.fullmatch(r";#line 1 (\d+)", ln)
        if m:
            here = int(m.group(1))
            chunks.setdefault(here, [])
        elif here is not None:
            chunks[here].append(ln)
    for text, stored in (("r = (a + b) * c;", "(@R),hl"), ("s = (a + b) * 3", "(S),hl"),
                         ("u = 2 * 8 + 1;", "(U),hl"), ("c = 1 + 2;", "(@C),hl")):
        code = "\n".join(chunks[line_of(text, OPTIMIZED)])
        assert stored in code, (text, code)
        assert "ld\thl,11H" in "\n".join(chunks[line_of("u = 2 * 8 + 1;", OPTIMIZED)])
    assert "ld\thl,3" in "\n".join(chunks[line_of("c = 1 + 2;", OPTIMIZED)])
    if opt == 3:
        order = [int(m.split()[2]) for m in markers(asm) if m.startswith(";#line")]
        l_do, l_r, l_v = (line_of(t, OPTIMIZED) for t in ("do i = 0", "r = r + 1;", "v = r + b;"))
        # two copies of the body, each after the `i = value' of the DO
        assert order.count(l_r) == 2 and order.count(l_v) == 2
        assert order.count(l_do) == 3
        k = order.index(l_r)
        assert order[k - 1] == l_do and order[k + 1] == l_v


@pytest.mark.parametrize("opt", LEVELS)
def test_the_text_is_the_same_but_for_the_markers(opt):
    assert strip_markers(compile_g(PROGRAM, opt)) == compile_g(PROGRAM, opt, g=False)


# --- the .COM is the same (D2) ----------------------------------------------

def _programs() -> list[Path]:
    return sorted(p for p in HERE.glob("*.plm") if (HERE / f"{p.stem}_expected.txt").exists())


def _build(plm: Path, opt: int, g: bool, where: Path) -> bytes:
    """The .COM (or the .REL when the program is not complete) of ``plm``."""
    where.mkdir(parents=True, exist_ok=True)
    mac, rel, com = where / "p.mac", where / "p.rel", where / "p.com"
    opts = ["-D", "TEST_DEFINED"] if plm.stem == "test_conditional_compile" else []
    r = subprocess.run(
        [*_cmd(), "-O", str(opt), *opts, *(["-g"] if g else []), str(plm), "-o", str(mac)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    for argv in (["um80", "-o", str(rel), str(mac)], ["ul80", "-o", str(com), str(rel)]):
        r = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=where)
        assert r.returncode == 0, f"{argv[0]}: {r.stdout}{r.stderr}"
    return com.read_bytes()


def _cmd() -> list[str]:
    from ._toolchain import compile_cmd
    return compile_cmd()


def _missing(what: str) -> None:
    """Skip, or fail when UPLM80_REQUIRE_TOOLS=1: a skip exits 0, and then
    the byte test D2 has not run."""
    if os.environ.get("UPLM80_REQUIRE_TOOLS") == "1":
        pytest.fail(f"{what} (UPLM80_REQUIRE_TOOLS=1)")
    pytest.skip(what)


@pytest.fixture(scope="module")
def tools():
    from ._toolchain import compiler_env  # noqa: F401  (the checkout the subprocess runs)
    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            _missing(f"{tool} not installed")


@pytest.mark.parametrize("plm", _programs(), ids=lambda p: p.stem)
def test_the_com_is_the_same_with_and_without_g(plm, tools, tmp_path, monkeypatch):
    """D2: -g never changes the code, at any -O level."""
    from ._toolchain import compiler_env
    for k, v in compiler_env().items():
        monkeypatch.setenv(k, v)
    for opt in LEVELS:
        plain = _build(plm, opt, False, tmp_path / f"{opt}" / "plain")
        marked = _build(plm, opt, True, tmp_path / f"{opt}" / "marked")
        assert marked == plain, f"{plm.name} at -O{opt}: the .COM differs with -g"


# --- upeepz80 ---------------------------------------------------------------

def test_the_upeepz80_in_use_keeps_the_code_the_same_with_markers():
    C.upeepz80_markers_problem.cache_clear()
    assert C.upeepz80_markers_problem() is None


def test_an_upeepz80_that_lets_a_marker_hide_a_dead_store_is_refused(monkeypatch):
    class Old:
        """upeepz80 0.2.7: keeps the markers, and a store behind one."""
        def optimize(self, text):
            return text

    monkeypatch.setattr(C, "PeepholeOptimizer", Old)
    C.upeepz80_markers_problem.cache_clear()
    try:
        problem = C.upeepz80_markers_problem()
        assert problem and "-g" in problem and "dead store" in problem
        with pytest.raises(C.CompilerError):
            C.require_upeepz80_markers(2)
        C.require_upeepz80_markers(0)           # -O0 uses no upeepz80
    finally:
        C.upeepz80_markers_problem.cache_clear()
    assert upeepz80.PeepholeOptimizer is not Old


# --- prog.dbg ---------------------------------------------------------------

def _um80_has_debug_info() -> bool:
    if shutil.which("um80") is None or shutil.which("ul80") is None:
        return False
    r = subprocess.run(["um80", "--help"], capture_output=True, text=True, check=False)
    return "--debug-info" in r.stdout + r.stderr


@pytest.mark.parametrize("opt", (0, 2, 3))
def test_each_instruction_of_the_routines_has_a_source_line(opt, tmp_path, monkeypatch):
    """Through um80 and ul80 --debug-info: no instruction byte between a
    ;#func and its ;#endfunc lacks a source line (a REENTRANT procedure
    and a recursive one included)."""
    if not _um80_has_debug_info():
        _missing("um80/ul80 with --debug-info not installed")
    from ._toolchain import compiler_env
    for k, v in compiler_env().items():
        monkeypatch.setenv(k, v)
    shutil.copy(HERE / "test_reentrant.plm", tmp_path / "p.plm")
    run = lambda *a: subprocess.run(a, capture_output=True, text=True, cwd=tmp_path, check=False)
    r = run(*_cmd(), "-g", "-O", str(opt), "p.plm", "-o", "p.mac")
    assert r.returncode == 0, r.stderr
    assert run("um80", "--debug-info", "-o", "p.rel", "p.mac").returncode == 0
    r = run("ul80", "--debug-info", "-o", "p.com", "p.rel")
    assert r.returncode == 0, r.stdout + r.stderr
    dbg = json.loads((tmp_path / "p.dbg").read_text())
    ranges = [(a, b) for a, b, groups, *_ in dbg["lines"] if groups]
    funcs = [f for f in dbg["funcs"] if f["end"] is not None]
    assert {f["frame"] for f in funcs} >= {"ix", "static"}
    for f in funcs:                     # IX holds the frame inside the range
        if f["frame"] == "ix":
            lo, hi = f["ix"]
            assert f["start"] < lo < hi <= f["end"], f
    total = missing = 0
    for row in dbg["asm"]:
        start, end, kind = row[0], row[1], row[2]
        if kind != "i" or not any(f["start"] <= start < f["end"] for f in funcs):
            continue
        total += end - start
        if not any(a <= start and end <= b for a, b in ranges):
            missing += end - start
    assert total > 100
    assert missing == 0


# --- more than one module ---------------------------------------------------

def test_two_modules_have_one_file_table_and_their_own_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.plm").write_text(
        "a: do;\ndeclare x byte;\nbump: procedure (v) byte;\n  declare v byte;\n  return v + 1;\nend bump;\nx = bump(1);\nend a;\n")
    (tmp_path / "b.plm").write_text(
        "b: do;\ndeclare y byte;\nbump2: procedure (v) byte;\n  declare v byte;\n  y = v;\n  return y + 1;\nend bump2;\nend b;\n")
    out = tmp_path / "ab.mac"
    assert Compiler(opt_level=2, debug_info=True).compile_files(
        [tmp_path / "a.plm", tmp_path / "b.plm"], out)
    ms = markers(out.read_text())
    files = [m for m in ms if m.startswith(";#file ")]
    assert len(files) == 2
    assert files[0].endswith('a.plm"') and files[1].endswith('b.plm"')
    n_a = int(files[0].split()[1])
    n_b = int(files[1].split()[1])
    assert n_a != n_b
    assert f";#line {n_a} 5" in ms and f";#line {n_a} 7" in ms
    assert f";#line {n_b} 5" in ms and f";#line {n_b} 6" in ms
    assert sum(m.startswith(";#func") for m in ms) == sum(m.startswith(";#endfunc") for m in ms) == 3
