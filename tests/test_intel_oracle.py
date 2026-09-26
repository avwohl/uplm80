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
import sys

import pytest

from tests.plm_intel import generate
from tests.test_expression_types import _PRELUDE, QUALIFIED_SIZES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "intel_oracle", os.path.join(ROOT, "scripts", "intel_oracle.py"))
oracle = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("intel_oracle", oracle)   # its dataclasses look themselves up there
_spec.loader.exec_module(oracle)

# The differences the README lists; each is a generator feature it can
# leave out.
KNOWN = frozenset({"shl-byte", "shift9", "wide-limit", "sub-zero", "zero-dividend",
                   "neg-widened"})
SEEDS = [3, 8, 10]
# The programs of a release's tests whose expected output is what Intel's
# build prints, transcribed there: each body follows tests/
# test_expression_types.py's _PRELUDE.
RELEASE_PROGRAMS = {
    "qualified-sizes": QUALIFIED_SIZES,          # 0.4.2, LENGTH(st.z), SIZE(sa(2))
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


def test_corpus_program_prints_what_intels_build_prints(tools):
    path = os.path.join(ROOT, "tests", "test_based_proc.plm")
    res = oracle.check_text(tools, oracle.prepare_source(path), "test_based_proc")
    assert res.verdict == "same", oracle.format_result(res)


def test_a_difference_is_seen(tools):
    """A BYTE SHL by 9: Intel's V3.1 shifts by 1, uplm80 by 9 (in 16 bits)."""
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


def test_generator_is_deterministic_and_reducible():
    a, b = generate(5), generate(5)
    assert a.render() == b.render()
    chunks = a.chunks()
    assert chunks
    smaller = a.render(frozenset(c.id for c in chunks if c.kind == "stmt"))
    assert len(smaller) < len(a.render())
    assert all(len(line) <= 110 for line in a.render().splitlines())
