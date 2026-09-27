"""uplm80 against Intel's own PL/M-80 V3.1 (scripts/intel_oracle.py).

A few programs of tests/plm_intel.py's generator, and one of the corpus,
built with both compilers and run: every -O level must print what Intel's
build prints.  The generator leaves out the differences already known and
documented (README, Testing against Intel's PL/M-80), so what fails here is
new.

Part of the suite, and skipped - at once, building nothing - unless
Intel's binaries are found ($PLM80_TOOLS, or DRI's work disk at
~/src/mpm2/mpm2_external/mpm2src/PLM_WORK), with tools/isis/isis (built
here, when they are, if make and a C++ compiler can) or romwbw_emu's
tools/romwbw-plm80 to run them, and um80, ul80 and cpmemu.  The oracle's
own pieces that need no tools - the source normalisation, the halt patch,
the skip itself - are checked always.
"""

import importlib.util
import os
import re
import sys

import pytest

from tests.plm_difftest import Bin, Num, Var
from tests.plm_intel import generate, zero_dividend
from tests.test_calls_and_loops import END_LABELS, MEMORY_RUNS_BACK, MEMORY_RUNS_ON
from tests.test_expression_types import (_PRELUDE, BYTE_OPERANDS, BYTE_SHIFTS, EMBEDDED_TARGET,
                                         QUALIFIED_SIZES)
from tests.test_names import V31_CALLS, V31_RECURSION
from tests.test_names import PRELUDE as NAMES_PRELUDE
from tests.test_names import (NO_FIXED_ADDRESS, PAST_THE_SPACE, V31_ALLOWS, V31_BASES,
                              V31_BLOCK_LOCATIONS, V31_DECLARED_BUILTINS, V31_DECLS,
                              V31_FACTORED_BASED, V31_LOCATIONS, V31_REJECTS, V31_RESTRICTED,
                              V31_WARNS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "intel_oracle", os.path.join(ROOT, "scripts", "intel_oracle.py"))
oracle = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("intel_oracle", oracle)   # its dataclasses look themselves up there
_spec.loader.exec_module(oracle)

# The differences the README lists; each is a generator feature it can
# leave out.
KNOWN = frozenset({"shift9", "wide-limit", "sub-zero", "zero-dividend", "neg-widened"})
SEEDS = [3, 8, 10]
# The programs of a release's tests whose expected output is what Intel's
# build prints, transcribed there: each body follows tests/
# test_expression_types.py's _PRELUDE.
RELEASE_PROGRAMS = {
    "qualified-sizes": QUALIFIED_SIZES,          # 0.4.2, LENGTH(st.z), SIZE(sa(2))
    "memory-runs-on": MEMORY_RUNS_ON,            # 0.4.2, memory(5) = 20
    **{f"memory-runs-back-{k}": v for k, v in MEMORY_RUNS_BACK.items()},
    "end-labels": END_LABELS,                    # 0.4.2, out: end p;
    "byte-shifts": BYTE_SHIFTS,                  # 0.4.3, shl(b, 4) of a BYTE
    "v31-calls": V31_CALLS,                      # 0.4.4, call sg.g(3, 4), call qb(5, 6)
    "byte-operands": BYTE_OPERANDS,              # 0.4.4, shr(s.k, 3), b * 32 + v
    "v31-recursion": V31_RECURSION,              # 0.4.4, fact(5) REENTRANT, length(memory)
    "v31-allows": V31_ALLOWS,                    # 0.4.3, size(ab(b + 1)), forward REENTRANT
    "v31-restricted": V31_RESTRICTED,            # 0.4.4, data (.memory), .(1 + 2)
    "v31-locations": V31_LOCATIONS,              # 0.4.4, data (.arr(2)), data (.s.k)
    **{f"declared-builtins-{k}": v[0] for k, v in V31_DECLARED_BUILTINS.items()},
    **{f"block-locations-{k}": v[0] for k, v in V31_BLOCK_LOCATIONS.items()},
    "factored-based": V31_FACTORED_BASED,        # 0.4.4, (a based s.p) byte
    **{f"bases-{k}": v[0] for k, v in V31_BASES.items()},     # 0.4.4, a based q in p
}
# V3.1's bugs the README's Known differences has and the generator does not
# leave out: a program, after _PRELUDE; how the README writes it; what
# V3.1's build prints; and what uplm80's prints at every level, the
# manual's value.
V31_BUGS = {
    # 0.4.3's campaign, seed 50252: V3.1 adds the 16-bit copy of b in ew.
    "embedded-byte-sum": ("""
declare b byte, (ew, w) address;
b = 0ech; w = (ew := b) + b; call ph(w);
""", "`b = 0ECH; w = (ew := b) + b;`", "01D8 ", "00D8 "),
    # 0.4.3's release check, seed 80353: V3.1 takes w2's value for an address.
    "embedded-target": (EMBEDDED_TARGET, "`w2 = 0FFFEH; w2, w3 = (ew := w2);`",
                        "FFFE 0000 FFFE FFFE 0032 00FE ", "FFFE FFFE FFFE FFFE FFFE 00FE "),
}


@pytest.fixture(scope="module")
def tools():
    t = oracle.Tools.find()
    why = t.missing()
    if why:
        pytest.skip(why)
    return t


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_program_prints_what_intels_build_prints(tools, seed):
    text = generate(seed, n_stmts=20, avoid=KNOWN).render()
    res = oracle.check_text(tools, text, f"seed{seed}")
    assert res.verdict == "same", oracle.format_result(res)


@pytest.mark.parametrize("name", sorted(RELEASE_PROGRAMS))
def test_a_release_program_prints_what_intels_build_prints(tools, name):
    text = _PRELUDE + RELEASE_PROGRAMS[name] + "\nend t;\n"
    res = oracle.check_text(tools, text, name)
    assert res.verdict == "same", oracle.format_result(res)


@pytest.mark.parametrize("name", sorted(V31_REJECTS) + sorted(V31_WARNS))
def test_v31_rejects_what_uplm80_rejects_or_warns_of(tools, name):
    """Each program tests/test_names.py holds uplm80 to: V3.1 rejects it
    with the errors uplm80's message names, those and no other, and
    uplm80 rejects it too, or, where programs written for it rely on it,
    compiles it."""
    stmts, errors, _ = {**V31_REJECTS, **V31_WARNS}[name]
    text = oracle.prepare_text(NAMES_PRELUDE + V31_DECLS + stmts + "end t;\n")
    res = oracle.check_text(tools, text, name, levels=(0,))
    assert res.verdict == "intel-rejects", oracle.format_result(res)
    given = {int(n) for e in res.intel_errors for n in re.findall(r"ERROR #(\d+),", e)}
    assert given == set(errors), res.intel_errors
    assert ("uplm80 rejects it too" in res.detail) == (name in V31_REJECTS), res.detail


@pytest.mark.parametrize("name", sorted(PAST_THE_SPACE))
def test_v31_gives_209_alone_past_a_declarations_space(tools, name):
    """0.4.3's and 0.4.4's Known issues: V3.1 rejects more values than a
    declaration holds (#209), and a number past its space it does not
    hold to a byte (no #210); uplm80 lays the values out after it."""
    text = oracle.prepare_text(NAMES_PRELUDE + V31_DECLS + PAST_THE_SPACE[name] + "end t;\n")
    res = oracle.check_text(tools, text, name, levels=(0,))
    assert res.verdict == "intel-rejects", oracle.format_result(res)
    given = {int(n) for e in res.intel_errors for n in re.findall(r"ERROR #(\d+),", e)}
    assert given == {209}, res.intel_errors
    assert "uplm80 rejects it too" not in res.detail, res.detail


@pytest.mark.parametrize("name", ["stackptr", "shl", "double"])
def test_v31_takes_the_location_of_a_built_in_in_a_list(tools, name):
    """0.4.4's Known issues: V3.1 takes `data (.stackptr)' for an address
    of its own, which uplm80 has none to give, and refuses."""
    text = _PRELUDE + f"declare d address data (.{name});\ncall ph(d);\nend t;\n"
    res = oracle.check_text(tools, text, name, levels=(0,))
    assert res.verdict == "uplm80-rejects", oracle.format_result(res)


@pytest.mark.parametrize("name", sorted(k for k, v in NO_FIXED_ADDRESS.items() if v[3]))
def test_v31_takes_a_location_uplm80_gives_no_fixed_address(tools, name):
    """0.4.4's Known issues: V3.1 takes the location of a REENTRANT
    procedure's local, and of a BASED variable in a list, for an address;
    uplm80, which has the local on the stack and no address of a BASED
    variable to give, refuses both."""
    stmts = NO_FIXED_ADDRESS[name][0]
    text = oracle.prepare_text(NAMES_PRELUDE + V31_DECLS + stmts + "end t;\n")
    res = oracle.check_text(tools, text, name, levels=(0,))
    assert res.verdict == "uplm80-rejects", oracle.format_result(res)


def test_corpus_program_prints_what_intels_build_prints(tools):
    path = os.path.join(ROOT, "tests", "test_based_proc.plm")
    res = oracle.check_text(tools, oracle.prepare_source(path), "test_based_proc")
    assert res.verdict == "same", oracle.format_result(res)


def test_a_difference_is_seen(tools):
    """A BYTE SHL by 9: Intel's V3.1 shifts by 1, and uplm80 by 9, which
    leaves 0 of a BYTE (11.1.4)."""
    text = """t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare b byte;
b = 7;
call mon1(2, 40h + shl(b, 9));
end t;
"""
    res = oracle.check_text(tools, text, "shl9", levels=(0,))
    assert res.verdict == "differs"
    assert res.intel_out == "N"            # 40H + 0EH
    assert res.levels[0]["out"] == "@"


@pytest.mark.parametrize("name", sorted(V31_BUGS))
def test_a_v31_bug_the_readme_has_is_seen(tools, name):
    """V3.1's build of each program prints what it did, and uplm80's the
    manual's value at every level."""
    body, _, intel, ours = V31_BUGS[name]
    res = oracle.check_text(tools, _PRELUDE + body + "\nend t;\n", name)
    assert res.verdict == "differs", oracle.format_result(res)
    assert res.intel_out == intel, oracle.format_result(res)
    assert all(r.get("out") == ours for r in res.levels.values()), oracle.format_result(res)


def test_intel_rejects_what_v31_does_not_take(tools):
    text = """t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
call mon1(9, .'hello$');
end t;
"""
    res = oracle.check_text(tools, text, "dotstring", levels=(0,))
    assert res.verdict == "intel-rejects"
    assert "#101" in res.detail


# ---- no tools needed ----------------------------------------------------------

def test_normalize_makes_uplm80_only_spellings_intel():
    text, changes = oracle.normalize_text(
        "declare hex data ('0123456789ABCDEF');\n"
        "call mon1(9, .'hi$');\n")
    assert "declare hex(*) BYTE data ('0123456789ABCDEF');" in text
    assert "call mon1(9, .('hi$'));" in text
    assert text.startswith("ORACLE$MODULE: DO;\n")
    assert text.rstrip().endswith("END ORACLE$MODULE;")
    assert len(changes) == 3


def test_normalize_leaves_intel_source_alone():
    src = "t: do;\ndeclare x address data (5);\ncall p(.('a$'));\nend t;\n"
    assert oracle.normalize_text(src) == (src, [])


def test_prepare_drops_an_origin_line_and_reads_includes(tmp_path):
    (tmp_path / "defs.lit").write_text("declare k literally '3';\n")
    (tmp_path / "main.plm").write_text(
        "0100H: /* origin */\nt: do;\n$include (:F1:DEFS.LIT)\nend t;\n\x1a")
    assert oracle.prepare_source(str(tmp_path / "main.plm")) == \
        "t: do;\ndeclare k literally '3';\nend t;\n"


def test_without_intels_binaries_the_oracle_checks_nothing(monkeypatch, tmp_path, capsys):
    """Where Intel's binaries are not found - as on CI - the oracle says so,
    builds no ISIS emulator, and exits 0 (2 with --strict); the tests that
    need them skip."""
    monkeypatch.delenv("PLM80_TOOLS", raising=False)
    monkeypatch.setattr(oracle, "DEFAULT_TOOLS", str(tmp_path / "none"))
    built = []
    monkeypatch.setattr(oracle, "_try_build_isis", lambda: built.append(True))
    monkeypatch.setattr(oracle, "ISIS_DIR", str(tmp_path))       # no isis there
    t = oracle.Tools.find()
    assert t.missing().startswith("Intel's PL/M-80 V3.1 not found")
    assert not built
    assert oracle.main(["--random", "1"]) == 0
    assert oracle.main(["--random", "1", "--strict"]) == 2
    assert "intel_oracle: skipped - Intel's PL/M-80 V3.1 not found" in capsys.readouterr().err


def _omf(rtype: int, body: bytes) -> bytes:
    n = len(body) + 1
    rec = bytes([rtype, n & 0xFF, n >> 8]) + body
    return rec + bytes([-sum(rec) & 0xFF])


def test_patch_halt_makes_the_last_statements_hlt_a_warm_boot():
    # LINES for the CODE segment: statement 1 at 0000H, statement 2 (the
    # module's END) at 0003H.
    obj = _omf(0x08, bytes([1, 0, 0, 1, 0, 3, 0, 2, 0])) + _omf(0x0E, b"")
    com = bytes([0x31, 0, 0, 0xFB, 0x76, 0xC9])
    patched, note = oracle.patch_halt(com, obj)
    assert note == ""
    assert patched == bytes([0x31, 0, 0, 0xC7, 0x76, 0xC9])
    # located at 0103H behind OBJCPM's jump, everything is 3 bytes later
    patched, note = oracle.patch_halt(bytes([0xC3, 0, 1]) + com, obj, 0x103)
    assert patched[6] == 0xC7
    # anything else there is left alone
    assert oracle.patch_halt(bytes(6), obj)[1].startswith("no EI; HLT")


def test_a_label_on_the_modules_end_is_where_the_halt_is():
    """`fin: end t;' where a procedure's GOTO reaches FIN: V3.1 sets SP
    there again, as at any such label, and its EI; HLT follows the LXI SP
    the LINES record points at.  The HLT was left alone, and the build ran
    on past it: `timeout' (0.4.2's Known issues)."""
    obj = _omf(0x08, bytes([1, 0, 0, 1, 0, 3, 0, 2, 0])) + _omf(0x0E, b"")
    com = bytes([0x31, 0, 0, 0x31, 0x34, 0x12, 0xFB, 0x76, 0xC9])
    patched, note = oracle.patch_halt(com, obj)
    assert note == ""
    assert patched == bytes([0x31, 0, 0, 0x31, 0x34, 0x12, 0xC7, 0x76, 0xC9])


FIN_BY_GOTO = """t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare b byte;
bail: procedure;
  goto fin;
end bail;
b = 0;
call mon1(2, 41h);
if b = 0 then call bail;
call mon1(2, 58h);
fin: end t;
"""


def test_a_program_that_ends_at_a_label_a_procedure_jumps_to(tools):
    res = oracle.check_text(tools, FIN_BY_GOTO, "fin", timeout=5)
    assert res.verdict == "same", oracle.format_result(res)
    assert res.intel_out == "A"


def test_zero_dividend_leaves_out_a_dividend_that_folds_to_0():
    """`--avoid zero-dividend' left out `0 / x' but not a dividend that
    folds to 0 by what folds to 0 too, `(8 / 0FF00H) / (0F82AH <= 1)',
    which V3.1 folds to 0 and uplm80 divides, 0FFFFH (0.4.2's Known
    issues), nor a product with 0, `((k0 * 0) * x) / y' (seed 50094)."""
    assert zero_dividend(Bin("/", Num(8), Num(0xFF00)), Bin("<=", Num(0xF82A), Num(1)))
    assert zero_dividend(Num(0), Var("b1"))
    assert zero_dividend(Bin("*", Bin("*", Num(65534), Num(0)), Var("b1")), Var("w1"))
    assert not zero_dividend(Num(0), Num(3))
    assert not zero_dividend(Num(8), Var("b1"))
    avoid = KNOWN | {"zero-dividend"}
    assert "(high(144) / (7 = last(ms)))" not in generate(115, n_stmts=20, avoid=avoid).render()


def test_zero_dividend_leaves_out_a_product_that_overflows_to_0():
    """A constant product that overflows to 0 is a zero dividend too,
    `256 * 256 / b1'.  Seed 96044 of 0.4.3's release check divides
    `('N' * 08000H)' by `HIGH(AB(0))', which is 0: V3.1 prints 0000 and
    uplm80 0FFFFH.  0.4.2's generator left it out; 56fad40 looked only at
    whether a factor folds to 0, and `--avoid zero-dividend' let it
    through."""
    assert zero_dividend(Bin("*", Num(256), Num(256)), Var("b1"))
    assert zero_dividend(Bin("*", Num(0x4E, "str"), Num(0x8000)), Var("b1"))
    assert zero_dividend(Bin("*", Bin("*", Num(256), Num(256)), Var("b1")), Var("w1"))
    assert not zero_dividend(Bin("*", Num(256), Num(256)), Num(3))
    assert not zero_dividend(Bin("*", Num(255), Num(257)), Var("b1"))
    text = generate(96044, avoid=KNOWN).render()
    assert "(('N' * 08000h) / high(ab(0)))" not in text
    assert "(('N' * 08000h) / (high(ab(0)) or 1))" in text


def test_the_readme_has_each_v31_bug():
    """Each V3.1 bug the tests know is a row of the README's Known
    differences with no `--avoid' name."""
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        text = f.read()
    known = text.split("### Known differences from Intel's PL/M-80 V3.1")[1].split("\n## ")[0]
    rows = [line for line in known.splitlines() if line.startswith("| | ")]
    for name, (_, spelling, _, _) in V31_BUGS.items():
        assert any(row.startswith(f"| | {spelling} ") for row in rows), (name, spelling)


def test_generator_is_deterministic_and_reducible():
    a, b = generate(5), generate(5)
    assert a.render() == b.render()
    chunks = a.chunks()
    assert chunks
    smaller = a.render(frozenset(c.id for c in chunks if c.kind == "stmt"))
    assert len(smaller) < len(a.render())
    assert all(len(line) <= 110 for line in a.render().splitlines())
