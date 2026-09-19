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


def _instructions(asm: str) -> list[str]:
    """The asm as a list of non-blank, comment-free lines."""
    out = []
    for line in asm.splitlines():
        line = line.split(';')[0].strip()
        if line:
            out.append(line)
    return out


def _clobbered_before_use(instrs, park, consumers, writes):
    """
    Indices where `park` leaves a value in a register and something in `writes`
    destroys the value before any of `consumers` reads it.  A label ends the
    window: control flow joins there, so the run is no longer straight-line.
    """
    bad = []
    for i, ins in enumerate(instrs):
        if ins != park:
            continue
        for j in range(i + 1, len(instrs)):
            nxt = instrs[j]
            if nxt.endswith(':'):
                break
            if any(nxt.startswith(c) for c in consumers):
                break
            if any(nxt.startswith(w) for w in writes):
                bad.append(j)
                break
    return bad


# The ??subde and ??mul16 helpers take DE as an operand, so a call to one reads
# DE rather than destroying an operand parked there.
_DE_CONSUMERS = ("add\thl,de", "adc\thl,de", "sbc\thl,de", "ex\tde,hl",
                 "push\tde", "call\t??")
_DE_WRITES = ("ld\tde,", "ld\td,", "ld\te,", "pop\tde")

# Nothing passes an argument in B, and the helpers are free to use B, so any
# call is a hazard for a value parked there.
_B_CONSUMERS = ("and\tb", "or\tb", "xor\tb", "sub\tb", "add\ta,b", "cp\tb")
_B_WRITES = ("ld\tb,", "pop\tbc", "call\t")


class TestSubscriptBaseSurvivesTheIndex:
    """
    The base address of a subscripted store must survive generation of the
    index expression.

    `_gen_subscript_addr` used to park the base in DE and then generate the
    index.  An index that is itself a 16-bit expression emits `ld de,nn`, which
    destroyed the base: `PRNT(I + 629) = I` stored to `(I + 629) * 2 + 629`
    instead of `PRNT + (I + 629) * 2`.  Every element address in the CrLZH
    decoder of the 80un unpacker came out wrong, so the Huffman parent table
    never got initialised.
    """

    SRC = """
        t: do;
        declare prnt (1300) address;
        declare i address;
        p: procedure;
            i = 0;
            do while i < 629;
                prnt(i + 629) = i;
                i = i + 1;
            end;
        end p;
        call p;
        end t;
    """

    def test_base_is_not_held_in_de_across_the_index(self) -> None:
        instrs = _instructions(_compile(self.SRC))
        bad = _clobbered_before_use(instrs, "ex\tde,hl", _DE_CONSUMERS, _DE_WRITES)
        assert not bad, (
            "a value parked in DE is destroyed before any read, by: "
            f"{[instrs[j] for j in bad]}"
        )

    def test_the_base_reaches_the_final_add(self) -> None:
        """The base must come back off the stack, not be re-derived from thin air."""
        instrs = _instructions(_compile(self.SRC))
        store = next(i for i, ins in enumerate(instrs) if ins == "ld\thl,PRNT")
        window = instrs[store:store + 12]
        assert "push\thl" in window, f"the base is not spilled: {window}"
        # The base comes back off the stack immediately before the add that
        # applies it.  An earlier `add hl,de` in the window belongs to the index
        # expression itself, which is why the last one is the one that matters.
        restore = window.index("pop\tde")
        assert window[restore + 1] == "add\thl,de", (
            f"the base is not applied straight after being restored: {window}"
        )


class TestByteOperandSurvivesTheOtherOperand:
    """
    A byte AND/OR/XOR/SUB must not hold one operand in B while generating the
    other.  B is not callee-saved, a nested byte comparison uses `ld b,a` as its
    own scratch move, and a procedure call clobbers B.
    """

    def test_b_is_not_held_across_the_other_operand(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (a, b, c, d) byte;
            f: procedure byte; return 1; end f;
            p: procedure;
                d = (a > 0) and (b < c);
                d = f and (b < c);
            end p;
            call p;
            end t;
        """))
        bad = _clobbered_before_use(instrs, "ld\tb,a", _B_CONSUMERS, _B_WRITES)
        assert not bad, (
            "a byte operand parked in B is destroyed before it is read, by: "
            f"{[instrs[j] for j in bad]}"
        )


class TestByteGreaterThanAsAValue:
    """
    A BYTE `>` used as a value must yield 0 when false.  The GT arm used to jump
    past the `xor a` to the merge label, leaving the compared operand in the
    accumulator, so `Y = X > 32` with `X = 6` assigned 6.
    """

    def test_false_path_loads_zero(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (x, y) byte;
            p: procedure;
                y = x > 32;
            end p;
            call p;
            end t;
        """))
        jumped_to = {
            ins.split(',')[-1].strip()
            for ins in instrs
            if ('jp' in ins.split()[0] or 'jr' in ins.split()[0]) and ',' in ins
        }
        zeroed = [i for i, ins in enumerate(instrs) if ins == "xor\ta"]
        assert zeroed, "no `xor a` was emitted for the false path at all"
        for i in zeroed:
            label = instrs[i - 1]
            if label.endswith(':'):
                assert label[:-1] in jumped_to, (
                    f"the `xor a` at {i} sits behind {label}, which nothing "
                    "jumps to, so the false path never loads zero"
                )
