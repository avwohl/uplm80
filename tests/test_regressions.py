"""Regression tests for reported compiler bugs.

* issue #7 — ``_check_impossible_comparison`` crashed (``AttributeError:
  'BinaryOpKind' object has no attribute 'kind'``) when the migration to
  the uplox typed AST left it dispatching on the wrong enum, and
  ``_get_expr_type`` typed every STRUCTURE member as BYTE — so an ADDRESS
  member compared to a 16-bit constant (``rec.len <= 1025``) both crashed
  and, once that was fixed, looked like an impossible BYTE comparison.

* issue #6 — PL/M-80 v4.0 conditional compilation is a left-margin
  control line (``$IF`` …), not a ``/** **/`` comment marker, and needs
  no enabling ``$COND``.
"""

import os

from uplm80.compiler import Compiler
from uplm80.preprocess import preprocess

_HERE = os.path.dirname(__file__)


def _compile(src: str) -> str:
    asm = Compiler().compile(src, "<test>")
    assert asm is not None, "compilation failed/raised"
    return asm


def _compile_file(name: str) -> str:
    path = os.path.join(_HERE, name)
    with open(path, encoding="latin-1") as fh:
        src = fh.read()
    asm = Compiler().compile(src, path)  # real path -> $include resolves
    assert asm is not None, f"compilation of {name} failed/raised"
    return asm


class TestAddressStructMemberComparison:
    """issue #7 — ADDRESS struct members must not be typed as BYTE."""

    SRC = """
b: do;
declare p address;
declare rec based p structure (type byte, len address, rec(1) byte);
foo: procedure;
    if rec.len <= 1025 then
        call bar;
end foo;
bar: procedure external; end;
end b;
"""

    def test_compiles_without_crash_or_spurious_error(self) -> None:
        # Before the fix this raised AttributeError, then (with only the
        # crash patched) a spurious "comparison BYTE <= 1025 is always
        # true" CodeGenError.
        asm = _compile(self.SRC)
        assert "always true" not in asm

    def test_address_member_uses_16bit_compare(self) -> None:
        # The ADDRESS member must be compared 16-bit (??subde), not via a
        # truncating 8-bit ``cp``.
        asm = _compile(self.SRC)
        assert "??subde" in asm

    def test_ogden_link1a_repro(self) -> None:
        # The exact file from issue #7 (Mark Ogden's intel80tools
        # link_3.0). Previously crashed in _check_impossible_comparison on
        # the nested ``IF inRecord.len <= 1025``; must now compile.
        asm = _compile_file("link1a.plm")
        assert "GETRECORD" in asm.upper()
        assert "??subde" in asm  # rec.len (ADDRESS) compared 16-bit


class TestLineStartConditionals:
    """issue #6 — left-margin $ directives, no $COND needed."""

    def test_if_without_cond_selects_true_branch(self) -> None:
        out = preprocess("d: do;\n$set (FOO)\n$if FOO\nINCL\n$else\nEXCL\n$endif\nend d;\n")
        assert "INCL" in out
        assert "EXCL" not in out

    def test_if_undefined_selects_else(self) -> None:
        out = preprocess("d: do;\n$if NOPE\nINCL\n$else\nEXCL\n$endif\nend d;\n")
        assert "EXCL" in out
        assert "INCL" not in out

    def test_elseif_first_branch(self) -> None:
        # First $ELSEIF matches.
        src = ("d: do;\n$set (BAR)\n$if FOO\nA\n$elseif BAR\nB\n"
               "$elseif BAZ\nC\n$else\nD\n$endif\nend d;\n")
        out = preprocess(src)
        assert "B" in out
        for tok in ("\nA\n", "\nC\n", "\nD\n"):
            assert tok not in out

    def test_elseif_later_branch_after_false_elseif(self) -> None:
        # A *false* $ELSEIF must not poison the chain: a later true
        # $ELSEIF still has to match. (Regression for the arm-selection
        # bug the adversarial review caught.)
        src = ("d: do;\n$set (BAZ)\n$if FOO\nA\n$elseif BAR\nB\n"
               "$elseif BAZ\nC\n$else\nD\n$endif\nend d;\n")
        out = preprocess(src)
        assert "C" in out
        for tok in ("\nA\n", "\nB\n", "\nD\n"):
            assert tok not in out

    def test_else_after_all_false_elseifs(self) -> None:
        # All arms false -> $ELSE must win even with $ELSEIFs present.
        src = ("d: do;\n$if FOO\nA\n$elseif BAR\nB\n$elseif BAZ\nC\n"
               "$else\nD\n$endif\nend d;\n")
        out = preprocess(src)
        assert "D" in out
        for tok in ("\nA\n", "\nB\n", "\nC\n"):
            assert tok not in out

    def test_nested_dead_outer_kills_inner(self) -> None:
        src = ("d: do;\n$if OUTER\n$if INNER\nII\n$else\nIO\n$endif\n"
               "$else\nOO\n$endif\nend d;\n")
        out = preprocess(src)
        assert "OO" in out
        assert "II" not in out and "IO" not in out

    def test_cond_accepted_as_noop(self) -> None:
        # $COND / $NOCOND are listing controls; presence must not change
        # which branch compiles.
        out = preprocess("d: do;\n$nocond\n$set (X)\n$if X\nYES\n$else\nNO\n$endif\nend d;\n")
        assert "YES" in out and "NO" not in out


class TestDirectiveInComment:
    """issue #6 — a $-directive only counts at the left margin, so an
    indented one inside a comment stays disabled (MP/M GENSYS.PLM ships
    ``/* $include (copyrt.lit) */``)."""

    def test_indented_include_in_comment_not_processed(self) -> None:
        # If this were processed, preprocess would raise trying to read
        # the missing include file. (preprocess preserves case; folding
        # happens later in macro_pass.)
        out = preprocess("d: do;\n/* $include (does_not_exist.lit) */\ndeclare x byte;\nend d;\n")
        assert "does_not_exist" in out  # left intact as comment text
        assert "byte" in out
