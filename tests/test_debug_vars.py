"""`uplm80 -g` writes `;#var` and `;#type` markers (phase 8 of the debugger plan).

The markers are in um80_and_friends docs/debug_info.md (Variables).  The
tests here are of four kinds:

* what the markers say, for each storage kind: a label (module level, INITIAL,
  DATA, address taken, PUBLIC, AT), a slot of the shared area ??AUTO, an IX
  frame (REENTRANT), a BASED variable, the parameters, the DO blocks;
* the pass that writes them (`debug_vars.add_vars`) on a text made by hand:
  `none` with a reason, the ranges of the parameters;
* D2: the text, but for the marker lines, is the text of a compile without
  -g, at the four -O levels (the .COM test is in test_debug_info.py);
* with um80, ul80, cpmemu and udb80: `print`, `info locals`, `info args`
  and `frame 1` at known values.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from uplm80 import debug_vars
from uplm80.compiler import Compiler

from .test_debug_info import LEVELS, _cmd, _missing, strip_markers

HERE = Path(__file__).parent

# One program with each storage kind.  The comments say where each is.
PROGRAM = """0100H:
vars: do;
declare lit literally '10';
declare gb byte initial (7);
declare ga address initial (1234h);
declare garr (4) byte initial (1,2,3,4);
declare gwarr (3) address initial (100,200,300);
declare tab (3) byte data (5, 6, 7);
declare pt structure (x address, y address, tag byte);
declare pts (2) structure (a byte, b address);
declare buf (8) byte;
declare bp address;
declare bb based bp byte;
declare bw based bp address;
declare ptr address;
declare pub byte public initial (9);
declare ext byte external;
declare (fa, fb) address initial (21, 22);
declare atv byte at (0080H);

leaf: procedure (p, q) address;
    declare (p, q) address;
    declare t address;
    declare keep byte initial (5);
    t = p + q;
    keep = keep + 1;
    return t + keep;
end leaf;

mid: procedure (a, b, c) address;
    declare (a, b, c) address;
    declare m address;
    declare marr (3) byte;
    marr(0) = low(a); marr(1) = low(b); marr(2) = low(c);
    m = leaf(a, b) + marr(1);
    return m + c;
end mid;

fact: procedure (n) address reentrant;
    declare n address;
    declare r address;
    declare ra (2) byte;
    ra(0) = 3;
    if n <= 1 then return 1;
    r = n * fact(n - 1);
    return r + ra(0) - 3;
end fact;

bump: procedure (k, flag) byte reentrant;
    declare k byte, flag address;
    declare tmp byte;
    tmp = k + 1;
    if k > 3 then return tmp;
    return bump(tmp, flag + 1);
end bump;

declare i byte;
declare res address;
gb = gb + 1;
pt.x = 11; pt.y = 22; pt.tag = 3;
pts(1).a = 4; pts(1).b = 5;
do i = 0 to 7; buf(i) = i + 10; end;
bp = .buf;
ptr = .pt;
atv = 3;
res = mid(1, 2, 3);
do;
    declare blk address;
    blk = res + 1;
    res = res + fact(4) + bump(1, blk);
end;
res = res + bw + bb + garr(2) + gwarr(1) + tab(1) + fb;
end vars;
"""


def line_of(text: str, src: str = PROGRAM) -> int:
    return next(i for i, ln in enumerate(src.split("\n"), 1) if text in ln)


def compile_g(src=PROGRAM, opt=2, name="vars.plm", g=True) -> str:
    out = Compiler(opt_level=opt, debug_info=g).compile(src, name)
    assert out is not None, "compilation failed"
    return out


def var_lines(asm: str) -> list[str]:
    return [ln for ln in asm.split("\n") if ln.startswith(";#var ")]


def var_of(asm: str, name: str, routine: str | None = None) -> str:
    """The `;#var` line of `name`: of `routine` (a `;#func` label), or the
    one before the first `;#func`."""
    lines = asm.split("\n")
    lo, hi = 0, len(lines)
    if routine is not None:
        lo = next(i for i, ln in enumerate(lines) if ln.startswith(f";#func {routine} "))
        hi = next(i for i in range(lo, len(lines)) if lines[i].startswith(";#endfunc"))
    else:
        hi = next(i for i, ln in enumerate(lines) if ln.startswith(";#func "))
    found = [ln for ln in lines[lo:hi] if re.match(rf";#var {name} ", ln)]
    assert len(found) == 1, (name, routine, found)
    return found[0]


@pytest.fixture(autouse=True)
def _upeepz80_keeps_markers(request):
    callspec = getattr(request.node, "callspec", None)
    opt = callspec.params.get("opt", 2) if callspec else 2
    if opt == 0:
        return
    from uplm80 import compiler as C
    C.upeepz80_markers_problem.cache_clear()
    problem = C.upeepz80_markers_problem()
    if problem:
        _missing(problem)


# --- what the markers say ----------------------------------------------------

VAR = re.compile(
    r";#var [A-Za-z][A-Za-z0-9_.]* \S+ kind=(global|static|local|param)"
    r"( at=[A-Za-z?@_$.][\w?@$.]*([+-]\d+)?| ix=-?\d+| none)"
    r"( why=[a-z-]+)?"
    r"( range=\?\?VR\d+:\?\?VR\d+)*"
    r"( scope=\d+:\d+-\d+)?( decl=\d+:\d+)?")


@pytest.mark.parametrize("opt", LEVELS)
def test_every_var_marker_has_the_exact_form(opt):
    asm = compile_g(opt=opt)
    lines = var_lines(asm)
    assert len(lines) >= 30
    for ln in lines:
        assert VAR.fullmatch(ln), ln
    # whole lines in column 1, never trailing
    assert not [ln for ln in asm.split("\n") if ";#" in ln and not ln.startswith(";#")]
    for ln in asm.split("\n"):
        if ln.startswith(";#type "):
            assert re.fullmatch(
                r";#type (typedef:\w+ target=\w+:\d|struct:[\w.]+ size=\d+ members=\S+)", ln), ln


def test_no_variable_markers_without_g():
    asm = compile_g(g=False)
    assert ";#var" not in asm and ";#type" not in asm and "??VR" not in asm


@pytest.mark.parametrize("opt", LEVELS)
def test_module_level_variables_are_at_their_labels(opt):
    asm = compile_g(opt=opt)
    d = lambda text: f"decl=1:{line_of(text)}"
    assert var_of(asm, "gb") == f";#var gb typedef:BYTE kind=static at=GB {d('declare gb')}"
    assert var_of(asm, "ga").startswith(";#var ga typedef:ADDRESS kind=static at=GA ")
    assert " [4]typedef:BYTE " in var_of(asm, "garr")
    assert " [3]typedef:ADDRESS " in var_of(asm, "gwarr")
    assert " [3]typedef:BYTE kind=static at=TAB " in var_of(asm, "tab")     # DATA
    assert " struct:pt kind=static at=PT " in var_of(asm, "pt")
    assert " [2]struct:pts kind=static at=PTS " in var_of(asm, "pts")
    assert " [8]typedef:BYTE kind=static at=BUF " in var_of(asm, "buf")
    # a name that is a register name is @-prefixed by the compiler
    assert " typedef:BYTE kind=static at=@I " in var_of(asm, "i")
    # PUBLIC is global
    assert " kind=global at=PUB " in var_of(asm, "pub")
    # AT is an EQU: the name is the symbol; um80 resolves it
    assert " typedef:BYTE kind=static at=ATV " in var_of(asm, "atv")
    # the second name of a factored INITIAL is an EQU of its own
    assert " kind=static at=FB " in var_of(asm, "fb")


def test_types_have_their_layout_and_the_names_are_the_source_s():
    asm = compile_g()
    types_ = [ln for ln in asm.split("\n") if ln.startswith(";#type ")]
    assert ";#type typedef:BYTE target=ushort:1" in types_
    assert ";#type typedef:ADDRESS target=uint:2" in types_
    assert ";#type struct:pt size=5 members=x:typedef:ADDRESS@0,y:typedef:ADDRESS@2,tag:typedef:BYTE@4" in types_
    assert ";#type struct:pts size=3 members=a:typedef:BYTE@0,b:typedef:ADDRESS@1" in types_
    # EXTERNAL data, LITERALLY, a procedure: not variables
    names = {ln.split()[1] for ln in var_lines(asm)}
    assert not names & {"ext", "lit", "leaf", "mid", "fact", "memory", "stackptr"}


def test_a_based_variable_is_a_pointer_at_its_base():
    asm = compile_g()
    # the debugger prints bb as a pointer; *bb is the byte
    assert " *typedef:BYTE kind=static at=BP " in var_of(asm, "bb")
    assert " *typedef:ADDRESS kind=static at=BP " in var_of(asm, "bw")


def test_based_on_a_member_is_at_the_member():
    src = ("0100H:\nm: do;\ndeclare ctl structure (lo address, ptr address, n byte);\n"
           "declare cp based ctl.ptr byte;\ndeclare (qa, qb) based ctl.ptr address;\n"
           "ctl.ptr = .ctl; ctl.n = cp;\nend m;\n")
    asm = compile_g(src)
    assert " *typedef:BYTE kind=static at=CTL+2 " in var_of(asm, "cp")
    assert " *typedef:ADDRESS kind=static at=CTL+2 " in var_of(asm, "qa")
    assert " *typedef:ADDRESS kind=static at=CTL+2 " in var_of(asm, "qb")


@pytest.mark.parametrize("opt", LEVELS)
def test_a_static_procedure_keeps_its_locals_in_auto_or_at_a_label(opt):
    asm = compile_g(opt=opt)
    p = var_of(asm, "p", "LEAF")
    q = var_of(asm, "q", "LEAF")
    t = var_of(asm, "t", "LEAF")
    assert re.fullmatch(r";#var p typedef:ADDRESS kind=param at=\?\?AUTO(\+\d+)? "
                        r"range=\?\?VR\d+:\?\?VR\d+ scope=1:\d+-\d+ decl=1:\d+", p), p
    assert "kind=param" in q and "range=" in q
    assert re.search(r" kind=local at=\?\?AUTO\+\d+ range=\?\?VR\d+:\?\?VR\d+ scope=", t), t
    # INITIAL keeps the local at a label of its own
    keep = var_of(asm, "keep", "LEAF")
    assert " typedef:BYTE kind=local at=@LEAF$KEEP " in keep
    # the scope is the procedure: from its line to the line of its END
    a, b = line_of("leaf: procedure"), line_of("end leaf;")
    assert f"scope=1:{a}-{b}" in t and f"decl=1:{line_of('declare t address')}" in t


@pytest.mark.parametrize("opt", LEVELS)
def test_a_parameter_range_starts_after_the_stores_of_the_entry_code(opt):
    asm = compile_g(opt=opt)
    lines = asm.split("\n")
    m = re.search(r" range=(\S+):(\S+)", var_of(asm, "a", "MID"))
    begin, end = (lines.index(f"{x}:") for x in m.groups())
    func = lines.index(next(ln for ln in lines if ln.startswith(";#func MID ")))
    endfunc = lines.index(";#endfunc MID")
    assert func < begin < end < endfunc
    # the range of each parameter begins right after a store to its slot
    for name in ("a", "b", "c"):
        line = var_of(asm, name, "MID")
        m = re.search(r" at=\?\?AUTO(\+\d+)? range=(\S+):(\S+)", line)
        start = lines.index(m.group(2) + ":")
        slot = "??AUTO" + (m.group(1) or "+0")
        assert re.fullmatch(rf"\tld\t\({re.escape(slot)}\),\w+", lines[start - 1]), (name, lines[start - 1])
    assert lines[end + 1] == ";#endfunc MID"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_reentrant_procedure_has_its_variables_in_the_ix_frame(opt):
    asm = compile_g(opt=opt)
    assert re.search(r"kind=param ix=4 ", var_of(asm, "n", "FACT"))
    r, ra = var_of(asm, "r", "FACT"), var_of(asm, "ra", "FACT")
    assert re.search(r"kind=local ix=-\d+ ", r) and " [2]typedef:BYTE " in ra
    # two parameters, both words on the stack: the first is the deeper
    assert "kind=param ix=6 " in var_of(asm, "k", "BUMP")
    assert " typedef:BYTE kind=param ix=6 " in var_of(asm, "k", "BUMP")
    assert " typedef:ADDRESS kind=param ix=4 " in var_of(asm, "flag", "BUMP")
    assert re.search(r" typedef:BYTE kind=local ix=-1 ", var_of(asm, "tmp", "BUMP"))
    assert "range=" not in var_of(asm, "n", "FACT")


def test_the_variables_of_a_do_block_of_the_main_program_are_the_main_program_s():
    asm = compile_g()
    first = line_of("declare blk address;") - 1          # the DO
    last = line_of("res = res + fact(4)") + 1            # the END
    assert re.fullmatch(
        rf";#var blk typedef:ADDRESS kind=local at=@B\d+\$BLK scope=1:{first}-{last} "
        rf"decl=1:{first + 1}", var_of(asm, "blk", "VARS"))


def test_nested_procedures_and_blocks_in_a_procedure():
    src = """0100H:
m: do;
declare r address;
outer: procedure (x, flag) address;
    declare x address, flag byte;
    declare loc (3) address;
    declare p address;
    declare pp based p byte;
    inner: procedure (y) address;
        declare y address;
        declare z address;
        z = y + loc(1);
        return z + x;
    end inner;
    p = .loc;
    loc(0) = x; loc(1) = x + 1; loc(2) = x + 2;
    do;
        declare blk byte;
        blk = flag;
        loc(2) = loc(2) + blk;
    end;
    return inner(loc(0)) + pp;
end outer;
r = outer(5, 3);
end m;
"""
    asm = compile_g(src)
    # address taken: static, at a label of the procedure
    assert " [3]typedef:ADDRESS kind=local at=@OUTER$LOC " in var_of(asm, "loc", "OUTER")
    assert " *typedef:BYTE kind=local at=@OUTER$P " in var_of(asm, "pp", "OUTER")
    first = line_of("declare blk byte;", src) - 1
    last = line_of("loc(2) = loc(2) + blk;", src) + 1
    assert re.search(rf" typedef:BYTE kind=local at=@OUTER\$B\d+\$BLK (range=\S+ )?scope=1:{first}-{last} ",
                     var_of(asm, "blk", "OUTER"))
    # a nested procedure is a routine of its own, named for its label
    assert " kind=param at=@OUTER$INNER$Y range=" in var_of(asm, "y", "@OUTER$INNER")
    assert " kind=local at=@OUTER$INNER$Z " in var_of(asm, "z", "@OUTER$INNER")
    # the nested procedure sees the variables of the outer one, as locals
    # that are valid in the whole routine, and sees its own y, not another
    x = var_of(asm, "x", "@OUTER$INNER")
    assert re.fullmatch(r";#var x typedef:ADDRESS kind=local at=\?\?AUTO scope=1:\d+-\d+ decl=1:\d+", x), x
    assert " [3]typedef:ADDRESS kind=local at=@OUTER$LOC " in var_of(asm, "loc", "@OUTER$INNER")
    assert " kind=param " in var_of(asm, "x", "OUTER")      # the outer one's own record stays
    assert "range=" not in x


def test_the_name_is_spelled_as_the_source_spells_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib.inc").write_text("declare Inc$Var address public;\n")
    (tmp_path / "m.plm").write_text(
        "m: do;\n$include (lib.inc)\ndeclare MyVar$x (2) byte;\n"
        "declare Pt structure (Hd byte, TL address);\nPt.Hd = 1;\nend m;\n")
    out = tmp_path / "m.mac"
    assert Compiler(opt_level=2, debug_info=True).compile_file(tmp_path / "m.plm", out)
    asm = out.read_text()
    # $ is dropped, the case is kept; a member and a tag too; the include's
    # variable has its own file number and line
    inc = re.search(r'^;#file (\d+) "lib.inc"$', asm, re.M).group(1)
    main = re.search(r'^;#file (\d+) "[^"]*m.plm"$', asm, re.M).group(1)
    assert re.search(rf"^;#var IncVar typedef:ADDRESS kind=global at=INCVAR decl={inc}:1$",
                     asm, re.M), asm
    assert re.search(rf"^;#var MyVarx \[2\]typedef:BYTE kind=static at=MYVARX decl={main}:3$",
                     asm, re.M)
    assert ";#type struct:Pt size=3 members=Hd:typedef:BYTE@0,TL:typedef:ADDRESS@1" in asm


def test_a_name_is_found_near_the_column_of_the_token():
    text = "declare  abc  byte;\nDCL (Xy, ab) ADDR;\n"
    src = debug_vars._Sources({"a.plm": 1}, {"a.plm": text})
    assert src.spell((1, 1, 10, 1), "ABC") == "abc"
    assert src.spell((1, 1, 9, 1), "ABC") == "abc"          # the column is a little off
    # the parser read DCL as DECLARE: four columns more than the source has
    assert src.spell((1, 2, 9, 2), "XY") == "Xy"
    assert src.spell((1, 2, 13, 2), "AB") == "ab"
    assert src.spell((1, 1, 10, 1), "ABD") == "ABD"         # no such name on the line
    assert src.spell((1, 5, 1, 5), "ABC") == "ABC"          # no such line
    assert src.spell((2, 1, 1, 1), "ABC") == "ABC"          # no such file
    assert src.spell(None, "ABC") == "ABC"


def test_two_structures_of_one_name_get_two_tags():
    src = ("0100H:\nm: do;\n"
           "p1: procedure; declare s structure (a byte, b byte); s.a = 1; end p1;\n"
           "p2: procedure; declare s structure (a address); s.a = 1; end p2;\n"
           "call p1; call p2;\nend m;\n")
    asm = compile_g(src)
    assert ";#type struct:s size=2 members=a:typedef:BYTE@0,b:typedef:BYTE@1" in asm
    assert ";#type struct:s.2 size=2 members=a:typedef:ADDRESS@0" in asm
    assert " struct:s kind=local " in var_of(asm, "s", "P1")
    assert " struct:s.2 kind=local " in var_of(asm, "s", "P2")


def test_a_multi_module_compile_has_the_variables_of_both(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.plm").write_text(
        "a: do;\ndeclare shared address public;\ndeclare ea address external;\n"
        "f: procedure (v) address;\n  declare v address;\n  return v + 1;\nend f;\n"
        "shared = f(1);\nend a;\n")
    (tmp_path / "b.plm").write_text(
        "b: do;\ndeclare other byte;\ng: procedure (w) address public;\n"
        "  declare w address;\n  declare loc address;\n  loc = w;\n  return loc;\nend g;\nend b;\n")
    out = tmp_path / "ab.mac"
    assert Compiler(opt_level=2, debug_info=True).compile_files(
        [tmp_path / "a.plm", tmp_path / "b.plm"], out)
    asm = out.read_text()
    assert " kind=global at=SHARED " in var_of(asm, "shared")
    assert " kind=static at=B?OTHER " in var_of(asm, "other")     # the module's name qualifies it
    assert "kind=param" in var_of(asm, "v", "A?F")
    assert "kind=param" in var_of(asm, "w", "G")
    assert not [ln for ln in var_lines(asm) if ln.startswith(";#var ea ")]


@pytest.mark.parametrize("mode", ("bare", "mpm"))
def test_the_other_modes_have_the_variables(mode):
    from uplm80.codegen import Mode
    asm = Compiler(mode={"bare": Mode.BARE, "mpm": Mode.MPM}[mode], opt_level=2,
                   debug_info=True).compile(PROGRAM, "vars.plm")
    assert asm and " kind=param at=" in var_of(asm, "p", "LEAF")


# --- the pass that writes them ---------------------------------------------------

def _rec(name, owner, place, *, kind="local", size=2, why=None, rng=False, shown=None):
    return {"name": name, "shown": shown or name.lower(), "type": "typedef:ADDRESS",
            "kind": kind, "owner": owner, "scope": (1, 3, 9), "decl": (1, 3),
            "place": place, "why": why, "sym": types.SimpleNamespace(size=size),
            "based": False, "range": rng}


TEXT = """;
;#file 1 "t.plm"
	cseg
F:
;#func F 1 3 frame=static pops=0
;#line 1 3
	ld	(??AUTO+0),hl
;#line 1 5
	ld	hl,(??AUTO+0)
	ld	(??AUTO+4),hl
	ret
;#endfunc F
G:
;#func G 1 7 frame=ix pops=0
;#line 1 7
	push	ix
	ret
;#endfunc G
"""


def test_a_slot_that_no_instruction_names_is_not_available():
    info = {"funcs": {"F": [
        _rec("USED", "F", ("at", "??AUTO", 0)),
        _rec("HIGH", "F", ("at", "??AUTO", 3), size=2),    # bytes 3 and 4: ??AUTO+4 names one
        _rec("GONE", "F", ("at", "??AUTO", 2)),            # bytes 2 and 3: nothing names them
        _rec("NOPLACE", "F", None, why="based-on-unknown"),
    ]}, "globals": [], "types": {}}
    out = debug_vars.add_vars(TEXT, info).split("\n")
    got = {ln.split()[1]: ln for ln in out if ln.startswith(";#var ")}
    assert " at=??AUTO " in got["used"] + " "
    assert "at=??AUTO+3" in got["high"]
    assert got["gone"].split()[3:5] == ["none", "why=no-storage"] or \
        "none why=no-storage" in got["gone"]
    assert "none why=based-on-unknown" in got["noplace"]
    # after the func marker, before its first line marker
    i = out.index(next(ln for ln in out if ln.startswith(";#func F ")))
    assert out[i + 1].startswith(";#var ") and out[i + 5].startswith(";#line")


def test_an_ix_variable_in_a_routine_with_no_ix_range_is_not_available():
    info = {"funcs": {"G": [_rec("V", "G", ("ix", -2))]}, "globals": [], "types": {}}
    out = debug_vars.add_vars(TEXT, info)
    assert "none why=no-frame" in out
    # with the range it is at IX
    text = TEXT.replace("frame=ix pops=0", "frame=ix pops=0 ixbegin=A ixend=B")
    assert "kind=local ix=-2 " in debug_vars.add_vars(text, info)


def test_the_range_of_a_parameter_starts_after_its_first_store():
    info = {"funcs": {"F": [
        _rec("P", "F", ("at", "??AUTO", 0), kind="param", rng=True),
        _rec("Q", "F", ("at", "??AUTO", 4), kind="param", rng=True),
        _rec("X", "F", ("at", "@F$X", 0), kind="param", rng=True),     # never stored
    ]}, "globals": [], "types": {}}
    out = debug_vars.add_vars(TEXT, info)
    lines = out.split("\n")
    got = {ln.split()[1]: ln for ln in lines if ln.startswith(";#var ")}
    # P is stored in the entry code, Q in the second statement
    assert "range=??VR1:??VR2 " in got["p"] and "range=??VR3:??VR2 " in got["q"]
    assert lines[lines.index("??VR1:") - 1] == "\tld\t(??AUTO+0),hl"
    assert lines[lines.index("??VR3:") - 1] == "\tld\t(??AUTO+4),hl"
    assert lines[lines.index("??VR2:") + 1] == ";#endfunc F"
    # no store to find: the whole routine
    assert "range=" not in got["x"]


def test_a_routine_with_no_marker_loses_its_variables():
    info = {"funcs": {"NOWHERE": [_rec("V", "NOWHERE", ("at", "V", 0))]},
            "globals": [_rec("g", None, ("at", "G", 0), kind="static")], "types": {}}
    out = debug_vars.add_vars(TEXT, info)
    assert ";#var v " not in out and ";#var g typedef:ADDRESS kind=static at=G " in out
    # the globals come after the file table
    assert out.index(";#var g ") > out.index(';#file 1 "t.plm"')


def test_a_fault_in_the_variable_records_does_not_stop_the_build(monkeypatch, capsys):
    monkeypatch.delenv("UPLM80_DEBUG_VARS_STRICT", raising=False)
    monkeypatch.setattr(debug_vars, "add_vars", lambda *a: 1 / 0)
    from uplm80 import compiler as C
    monkeypatch.setattr(C, "add_vars", debug_vars.add_vars)
    asm = compile_g()
    assert ";#func LEAF" in asm and ";#var" not in asm
    assert "no variable records for -g" in capsys.readouterr().err


# --- D2: the code does not change --------------------------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_the_text_is_the_same_but_for_the_markers(opt):
    assert strip_markers(compile_g(opt=opt)) == compile_g(opt=opt, g=False)


@pytest.mark.parametrize("opt", LEVELS)
def test_the_com_is_the_same_with_the_variable_markers(opt, tmp_path, monkeypatch):
    from ._toolchain import compiler_env
    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            _missing(f"{tool} not installed")
    for k, v in compiler_env().items():
        monkeypatch.setenv(k, v)
    (tmp_path / "p.plm").write_text(PROGRAM)
    coms = []
    for g in (False, True):
        where = tmp_path / ("g" if g else "plain")
        where.mkdir()
        run = lambda *a: subprocess.run(a, capture_output=True, text=True, cwd=where, check=False)
        r = run(*_cmd(), "-O", str(opt), *(["-g"] if g else []), str(tmp_path / "p.plm"),
                "-o", "p.mac")
        assert r.returncode == 0, r.stderr
        di = ["--debug-info"] if g else []
        assert run("um80", *di, "-o", "p.rel", "p.mac").returncode == 0
        r = run("ul80", *di, "-o", "p.com", "p.rel")
        assert r.returncode == 0, r.stdout + r.stderr
        coms.append((where / "p.com").read_bytes())
    assert coms[0] == coms[1]


# --- live: udb80 against cpmemu --gdb ------------------------------------------

def _live_tools():
    from ._toolchain import cpmemu
    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            _missing(f"{tool} not installed")
    emu = cpmemu()
    if emu is None:
        _missing("cpmemu not installed")
    try:
        import um80.udb80  # noqa: F401
        import um80.vardata  # noqa: F401
    except ImportError:
        _missing("um80 with udb80 variables (um80_and_friends d39debc) not installed")
    return emu


def _udb80(tmp_path, monkeypatch, opt, commands, errors=False, src=PROGRAM):
    emu = _live_tools()
    from ._toolchain import compiler_env
    for k, v in compiler_env().items():
        monkeypatch.setenv(k, v)
    (tmp_path / "vars.plm").write_text(src)
    run = lambda *a: subprocess.run(a, capture_output=True, text=True, cwd=tmp_path, check=False)
    r = run(*_cmd(), "-g", "-O", str(opt), "vars.plm", "-o", "vars.mac")
    assert r.returncode == 0, r.stderr
    assert run("um80", "--debug-info", "-o", "vars.rel", "vars.mac").returncode == 0
    r = run("ul80", "--debug-info", "-o", "vars.com", "vars.rel")
    assert r.returncode == 0, r.stdout + r.stderr
    (tmp_path / "cmds.txt").write_text("\n".join(commands) + "\n")
    r = subprocess.run(
        [sys.executable, "-m", "um80.udb80", "vars.dbg", "--run", f"{emu} vars.com",
         "--batch", "-x", "cmds.txt"],
        capture_output=True, text=True, cwd=tmp_path, check=False, timeout=120)
    assert r.returncode == (1 if errors else 0), r.stdout + r.stderr
    return r.stdout + r.stderr


@pytest.mark.parametrize("opt", LEVELS)
def test_udb80_prints_each_storage_kind(opt, tmp_path, monkeypatch):
    out = _udb80(tmp_path, monkeypatch, opt, [
        f"break vars.plm:{line_of('keep = keep + 1;')}",
        f"break vars.plm:{line_of('res = res + fact(4)')}",
        f"break vars.plm:{line_of('r = n * fact(n - 1);')}",
        f"break vars.plm:{line_of('tmp = k + 1;')}",
        "continue",
        # the static procedure LEAF: ??AUTO (p, q, t) and a label (keep)
        "info args", "info locals",
        # frame 1 is MID: its parameters and locals, one of them an array
        "frame 1", "info args", "info locals", "print a + b + c", "print marr",
        # globals and statics, each kind
        "print gb", "print ga", "print/x ga", "print garr", "print gwarr", "print tab",
        "print pt", "print pts", "print buf", "print bp", "print pub", "print i",
        "print atv", "print fa", "print fb",
        # a BASED variable is a pointer; the pointee is the value
        "print bb", "print *bb", "print *bw", "print pt.y", "print pts[1].b",
        "whatis bb", "whatis pt", "whatis buf", "ptype pt",
        # the DO block of the main program
        "continue", "info locals", "print blk", "print res",
        # a REENTRANT procedure: IX frame, the third time round (n = 2)
        "continue", "continue", "continue", "info args", "info locals", "bt",
        "frame 1", "info args", "print n", "print r",
        # the byte and the word parameter of a REENTRANT procedure
        "continue", "info args", "info locals", "continue", "info args",
    ])
    text = out
    # LEAF (breakpoint 1): p = 1, q = 2, t = 3, keep = 5
    assert re.search(r"Breakpoint 1, LEAF .*\n.*keep = keep \+ 1;\np = 1\nq = 2\n", text), text
    assert re.search(r"\nkeep = 5\nt = 3\n", text) or re.search(r"\nt = 3\nkeep = 5\n", text), text
    # frame 1 is MID
    assert re.search(r"#1 +0x[0-9a-f]+ in MID \(\).*\n.*\na = 1\nb = 2\nc = 3\n", text), text
    # m is not stored yet at that call: it is not available, not 0 or old bytes
    assert re.search(r"\nmarr = \{1, 2, 3\}\nm = <not available>\n", text) or \
        re.search(r"\nm = <not available>\nmarr = \{1, 2, 3\}\n", text), text
    assert "= 6\n" in text                                 # a + b + c
    assert "= 8\n" in text                                 # gb was 7, and one was added
    assert "= 4660\n" in text and "= 0x1234\n" in text    # ga
    assert "= {1, 2, 3, 4}\n" in text and "= {100, 200, 300}\n" in text
    assert "= {5, 6, 7}\n" in text                         # DATA
    assert "= {x = 11, y = 22, tag = 3}\n" in text
    assert "= {{a = 0, b = 0}, {a = 4, b = 5}}\n" in text
    assert "= {10, 11, 12, 13, 14, 15, 16, 17}\n" in text
    assert "= 9\n" in text                                 # pub
    assert re.search(r"= 3\n", text) and "= 21\n" in text and "= 22\n" in text   # atv, fa, fb
    assert re.search(r"= \(BYTE \*\) 0x[0-9a-f]+ <buf>\n", text)     # bb is a pointer
    assert "= 10\n" in text                                # *bb: buf(0)
    assert "= 2826\n" in text                              # *bw: 0B0AH
    assert "type = BYTE *\n" in text and "type = struct pt\n" in text
    assert "type = BYTE [8]\n" in text
    assert "    ADDRESS x;\n    ADDRESS y;\n    BYTE tag;\n" in text
    # the main program's block: MID(1, 2, 3) is 14, and blk is one more
    assert re.search(r"\nblk = 15\n", text), text
    assert re.search(r"\n\$\d+ = 15\n\$\d+ = 14\n", text), text
    # FACT: n = 2 at the third stop, one frame up n = 3, and the chain
    assert re.search(r"Breakpoint 3, FACT .*\n.*\n(Breakpoint 3, FACT .*\n.*\n){2}n = 2\n", text), text
    assert re.search(r"\n#1 +0x[0-9a-f]+ in FACT .*\n.*\nn = 3\n", text), text
    assert re.search(r"#3 +0x[0-9a-f]+ in VARS \(\)", text)
    # BUMP: the byte k and the word flag, in the IX frame
    assert re.search(r"Breakpoint 4, BUMP .*\n.*\nk = 1\nflag = 15\ntmp = ", text), text
    assert re.search(r"\nk = 2\nflag = 16\n", text), text


def test_udb80_does_not_show_a_variable_of_another_procedure(tmp_path, monkeypatch):
    out = _udb80(tmp_path, monkeypatch, 2, [
        f"break vars.plm:{line_of('keep = keep + 1;')}", "continue",
        "print m", "print marr", "print n", "print t"], errors=True)
    assert 'No symbol "m"' in out and 'No symbol "n"' in out
    assert re.search(r"\$\d+ = 3\n", out)       # t is LEAF's


ADV = """adv: do;
declare (Aa, aB,
   Cc) byte public;
First: procedure(Alpha, Beta) address;
  declare (Alpha, Beta, Tmp$One) address;
  Tmp$One = Alpha + Beta;
  return Tmp$One;
end First;
Aa = 1;
Cc = First(10, 20) mod 256;
end adv;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_local_of_a_procedure_is_valid_from_its_first_store(opt):
    asm = compile_g(ADV, opt=opt, name="adv.plm")
    lines = asm.split("\n")
    line = var_of(asm, "TmpOne", "FIRST")
    m = re.search(r" at=(\?\?AUTO(?:\+\d+)?) range=(\S+):(\S+)", line)
    assert m, line
    start = lines.index(m.group(2) + ":")
    assert re.fullmatch(rf"\tld\t\({re.escape(m.group(1))}\),\w+", lines[start - 1]), lines[start - 1]
    assert " range=" not in var_of(asm, "Aa")      # a global has no range


def test_a_local_with_an_initial_value_or_in_main_has_no_range():
    asm = compile_g()
    assert " range=" not in var_of(asm, "keep", "LEAF")
    assert " range=" not in var_of(asm, "r", "FACT")      # an IX frame


def test_a_name_on_a_continuation_line_has_its_own_spelling_and_line():
    asm = compile_g(ADV, name="adv.plm")
    assert re.search(r"^;#var Cc typedef:BYTE kind=global at=CC decl=1:3$", asm, re.M)
    assert re.search(r"^;#var aB typedef:BYTE kind=global at=AB decl=1:2$", asm, re.M)


@pytest.mark.parametrize("opt", LEVELS)
def test_udb80_never_shows_a_local_before_its_store(opt, tmp_path, monkeypatch):
    # at the RETURN the local is 30 at -O0; where the optimizer moved the
    # store past the line marker it is not available - never 0 or old bytes
    out = _udb80(tmp_path, monkeypatch, opt, [
        f"break vars.plm:{line_of('return Tmp$One;', ADV)}",
        "continue", "print TmpOne", "print Alpha + Beta",
    ], src=ADV)
    want = "$1 = 30\n" if opt == 0 else "$1 = <not available>\n"
    assert want in out and "$2 = 30\n" in out, out
