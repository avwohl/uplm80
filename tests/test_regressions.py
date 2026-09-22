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


_TRANSFERS = ("ret", "jp", "jr")


def _labels(instrs) -> set:
    return {ins[:-1] for ins in instrs if ins.endswith(':')}


def _called(instrs) -> set:
    return {
        ins.split(None, 1)[1].strip()
        for ins in instrs
        if ins.split(None, 1)[0] == "call" and len(ins.split(None, 1)) == 2
    }


class TestProcedureDeclaredInADoBlock:
    """
    PL/M-80 lets a PROCEDURE be declared at the head of any ``DO ... END``
    block, not only a procedure body — MP/M II's ``ED.PLM`` declares
    ``DIGIT`` / ``NUMBER`` / ``RELDISTANCE`` that way, inside a ``DO`` block in
    the middle of an IF/ELSE chain.

    Such a procedure was emitted where the block sits, with nothing jumping
    over it, so the enclosing code ran straight into the procedure body and
    took its ``RET``; everything after the block was unreachable.  The label
    also carried the block scope (``@B24$DIGIT``) while the call sites did
    not, because the collection passes never descended into blocks, so the
    call named a symbol that was never defined.
    """

    MODULE_LEVEL = """
        t: do;
        declare (ch, result) byte;
        do;
            declare n byte;
            digit: procedure byte;
                return (n := ch - '0') <= 9;
                end digit;
            if digit then result = n;
        end;
        result = result + 1;
        end t;
    """

    IN_PROCEDURE = """
        t: do;
        declare (ch, result) byte;
        outer: procedure;
            do;
                declare n byte;
                mark: procedure;
                    result = n;
                    end mark;
                n = ch;
                call mark;
            end;
            result = result + 1;
            end outer;
        call outer;
        end t;
    """

    def test_module_level_block_procedure_is_not_fallen_into(self) -> None:
        self._assert_out_of_line(self.MODULE_LEVEL, "DIGIT")

    def test_procedure_level_block_procedure_is_not_fallen_into(self) -> None:
        self._assert_out_of_line(self.IN_PROCEDURE, "@OUTER$MARK")

    def test_module_level_call_names_a_defined_label(self) -> None:
        self._assert_calls_resolve(self.MODULE_LEVEL)

    def test_procedure_level_call_names_a_defined_label(self) -> None:
        self._assert_calls_resolve(self.IN_PROCEDURE)

    def test_block_local_is_still_reachable_from_the_hoisted_body(self) -> None:
        # The procedure is emitted after the block has been left, but it
        # still reads the block's own local, so the load must name the
        # block-scoped storage rather than an undefined bare symbol.
        instrs = _instructions(_compile(self.IN_PROCEDURE))
        loads = [ins for ins in instrs if ins.startswith("ld\ta,(") and "N" in ins]
        assert loads, "the hoisted procedure never loads the block local at all"
        for ins in loads:
            assert ins != "ld\ta,(N)", (
                "the hoisted procedure reads a bare `N`, which no label "
                f"defines: {ins}"
            )
        # The storage line is `@OUTER$B1$N:\tds\t1`, so match the label
        # token rather than a bare label line.
        assert any(ins.split(':')[0].endswith("$N") for ins in instrs if ':' in ins), (
            "the block local has no block-scoped storage label"
        )

    def _assert_out_of_line(self, src: str, label: str) -> None:
        instrs = _instructions(_compile(src))
        assert f"{label}:" in instrs, f"{label} is never defined"
        i = instrs.index(f"{label}:")
        preceding = [ins for ins in instrs[:i] if not ins.endswith(':')]
        assert preceding, f"{label} is the first thing emitted"
        last = preceding[-1].split()[0]
        assert last in _TRANSFERS, (
            f"control reaches {label} by falling through `{preceding[-1]}`: the "
            "procedure body is emitted inline in the enclosing code"
        )

    def _assert_calls_resolve(self, src: str) -> None:
        """Every symbol the code names must be one some label defines.

        Not just call targets: when the block-scoped procedure name fails
        to resolve, a typed procedure used as a value degrades into
        `ld hl,(NAME)` — a load from a symbol nothing defines.
        """
        instrs = _instructions(_compile(src))
        defined = {ins.split(':')[0] for ins in instrs if ':' in ins}
        referenced = set()
        for ins in instrs:
            if ins.endswith(':'):
                continue
            parts = ins.split(None, 1)
            if len(parts) != 2:
                continue
            op, operand = parts[0], parts[1].strip()
            if op == "call":
                referenced.add(operand)
            elif op == "ld" and operand.endswith(")") and "(" in operand:
                inner = operand[operand.index("(") + 1:-1]
                if inner[:1].isalpha() or inner.startswith("@"):
                    referenced.add(inner)
        for target in referenced:
            if target.startswith("??") or target in ("hl", "de", "bc", "sp"):
                continue  # runtime helper / register indirect
            assert target in defined, (
                f"`{target}` is referenced but no label defines it"
            )


class TestConditionTestsBitZero:
    """
    PL/M-80 tests bit 0 of a condition value, not whether it is non-zero.
    DRI's binaries settle it: PIP.PRL's code segment holds 70 `RAR;JNC` and
    14 `RAR;JC` truth tests against a single `ORA A;JZ`. The two rules agree
    on a relational (0FFH / 00H) and part company on `NOT` -- `NOT 1` is
    0FEH -- and on the ROL/ROR bit-extraction idiom DRI writes throughout.
    """

    def test_byte_condition_tests_bit_zero(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (f, r) byte;
            p: procedure;
                if rol(f,1) then r = 1;
            end p;
            call p;
            end t;
        """))
        assert "bit\t0,a" in instrs, (
            "a BYTE condition is not tested with a bit-0 test: " f"{instrs}"
        )
        assert "or\ta" not in instrs, "the non-zero truth test is still emitted"

    def test_address_condition_tests_bit_zero_of_l(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (v, r) address;
            f: procedure address; return 4; end f;
            p: procedure;
                if f then r = 1;
            end p;
            call p;
            end t;
        """))
        assert "bit\t0,l" in instrs, (
            f"a 16-bit condition is not tested with a bit-0 test: {instrs}"
        )

    def test_not_of_a_one_flag_is_false(self) -> None:
        # `do while f$i$adr <> 0 and not found;` with `true literally '1'`
        # is SDIR's hash-chain scan (DSE.PLM:340). Under a non-zero test
        # 0FFH AND (NOT 1) = 0FEH reads as true and the loop never ends.
        instrs = _instructions(_compile("""
            t: do;
            declare true literally '1';
            declare (adr) address;
            declare found byte;
            p: procedure;
                do while adr <> 0 and not found;
                    found = true;
                end;
            end p;
            call p;
            end t;
        """))
        assert any(i.startswith("bit\t0,") for i in instrs), (
            "the loop condition is not a bit-0 test, so NOT 1 reads as true"
        )


class TestBasedAddressDoesNotPreserveDE:
    """
    A BASED ADDRESS load is `ld hl,(base) / ld e,(hl) / inc hl / ld d,(hl) /
    ex de,hl`: it writes DE and leaves base+1 there. `_expr_preserves_de`
    claimed otherwise, so `baccum = baccum + bpb` in MP/M II's SHOW.PLM
    computed `baccum + (ab+1)`.
    """

    SRC = """
        t: do;
        declare ab address;
        declare bpb address;
        declare baccum based ab address;
        p: procedure;
            baccum = baccum + bpb;
        end p;
        call p;
        end t;
    """

    def test_the_other_operand_is_not_parked_in_de(self) -> None:
        instrs = _instructions(_compile(self.SRC))
        based = [i for i, ins in enumerate(instrs) if ins == "ld\te,(hl)"]
        loads_bpb = [i for i, ins in enumerate(instrs) if ins == "ld\thl,(BPB)"]
        adds = [i for i, ins in enumerate(instrs) if ins == "add\thl,de"]
        assert based and loads_bpb and adds, f"unexpected shape: {instrs}"
        # BPB is the operand that has to reach `add hl,de`. It must be
        # fetched AFTER the BASED load, which writes E and D.
        assert max(loads_bpb) > max(based), (
            "BPB is fetched before the BASED ADDRESS load destroys DE, so "
            f"the add uses the base pointer instead: {instrs}"
        )


class TestAssignmentTargetIsNotConstantFolded:
    """
    `-O 3` ran assignment targets through the value optimizer, so `a = 5`
    with `a` known-constant became a store *through address 5* -- on CP/M,
    into the BDOS entry vector at 0005H.
    """

    def test_o3_stores_to_the_variable_not_through_it(self) -> None:
        asm = Compiler(opt_level=3).compile("""
            t: do;
            declare (a, b) byte;
            p: procedure;
                a = 5;
                b = a + 1;
            end p;
            call p;
            end t;
        """, "<test>")
        assert asm is not None
        instrs = _instructions(asm)
        assert "ld\t(@A),a" in instrs, f"the store lost its target: {instrs}"
        assert "ld\t(hl),e" not in instrs, (
            f"the assignment became a store through an address: {instrs}"
        )


class TestSideEffectingOperandIsNotDropped:
    """
    PL/M-80 evaluates both operands of every operator, so `x AND 0` must
    still evaluate `x`. The algebraic identities discarded it.
    """

    def test_and_zero_still_calls_the_operand(self) -> None:
        asm = _compile("""
            t: do;
            declare r byte;
            bump: procedure byte; return 7; end bump;
            p: procedure;
                r = bump and 0;
            end p;
            call p;
            end t;
        """)
        assert "call\tBUMP" in _instructions(asm), (
            "the AND-with-zero identity optimised the call away"
        )


class TestFoldedRelationalMatchesTheRuntimeValue:
    """
    A folded relational has to agree with the one the generator computes at
    runtime, under every operator that consumes it. The folder is untyped
    and masks to 16 bits, so the folded value must be a fixed point of
    NOT / unary minus / +1 the way the runtime's 0FFH is: an attempt to
    narrow it to 0FFH made `NOT (1 = 1)` fold to 0FF00H, where -O 0 and
    PL/M-80 both give 0.
    """

    # `w = NOT (1 = 1)` is zero: NOT of true is false. -O 0 computes it and
    # gets 0; the folder must reach the same answer.
    SRC = """
        t: do;
        declare w address;
        p: procedure;
            w = not (1 = 1);
        end p;
        call p;
        end t;
    """

    def test_every_level_computes_it_the_same_way(self) -> None:
        # Folding a relational is only safe where just bit 0 is observable,
        # so as a value it is left to the generator at every level. All four
        # then emit the same opcode sequence, which is what guarantees they
        # agree on the value.
        shapes = {}
        for level in (0, 1, 2, 3):
            asm = Compiler(opt_level=level).compile(self.SRC, "<test>")
            assert asm is not None
            instrs = _instructions(asm)
            body = instrs[instrs.index("P:"):]
            body = body[:body.index("ret") + 1]
            # Opcodes only: label names and jp/jr selection are not the point.
            ops = [i.split()[0].replace("jr", "jp") for i in body
                   if not i.endswith(":")]
            shapes[level] = ops
        assert len(set(map(tuple, shapes.values()))) == 1, (
            f"the levels disagree on how NOT (1 = 1) is evaluated: {shapes}"
        )
        assert "cpl" in shapes[2], (
            f"-O 2 did not compute the NOT at all: {shapes[2]}"
        )


class TestBdosFunctionNumberSurvivesTheArgument:
    """
    MON1/MON2 loaded the BDOS function number into C and then generated the
    argument. C is not callee-saved, so an argument containing a call left a
    different function number in C.
    """

    def test_c_is_loaded_after_the_argument(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            mon1: procedure (f,a) external; declare (f,a) address; end mon1;
            mon2: procedure (f,a) byte external; declare (f,a) address; end mon2;
            getc: procedure byte; return mon2(1,0); end getc;
            p: procedure;
                call mon1(2, getc);
            end p;
            call p;
            end t;
        """))
        # Look only inside P: GETC's own body legitimately loads C first.
        # P is emitted last and tail-calls the BDOS, so it runs to the end.
        body = instrs[instrs.index("P:"):]
        c_load = [i for i, ins in enumerate(body) if ins.startswith("ld\tc,")]
        calls = [i for i, ins in enumerate(body)
                 if ins.startswith("call\t") and "GETC" in ins]
        assert c_load and calls, f"expected both a `ld c,` and a call: {body}"
        assert min(c_load) > max(calls), (
            f"the function number is parked in C across a call: {body}"
        )


class TestConditionByteCompareOperandSurvives:
    """
    The 0.3.3 spill was applied to the byte binary/comparison VALUE paths but
    not to the condition paths, so `IF f > x` still parked x in B across the
    call to f. With f = 100 and x = 200 the comparison read true.
    """

    def test_b_is_not_held_across_the_other_operand_in_a_condition(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (x, y, z, r) byte;
            f: procedure byte; return y and z; end f;
            p: procedure;
                if f > x then r = 1;
                if x > f then r = 2;
            end p;
            call p;
            end t;
        """))
        bad = _clobbered_before_use(instrs, "ld\tb,a", _B_CONSUMERS, _B_WRITES)
        assert not bad, (
            "a byte operand parked in B for a condition is destroyed before "
            f"the compare reads it, by: {[instrs[j] for j in bad]}"
        )


class TestByteReturnTruncatesAnAddress:
    """
    PL/M-80 narrows ADDRESS to BYTE by truncation, like LOW(). A BYTE
    procedure returning an ADDRESS expression normalised it to a 0FFH/00H
    boolean instead, so `P: PROCEDURE BYTE; RETURN N + 1; END P;` with
    N = 64 returned 0FFH rather than 65.
    """

    def test_return_takes_the_low_byte(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare n address;
            declare r byte;
            p: procedure byte;
                return n + 1;
            end p;
            q: procedure;
                r = p;
            end q;
            call q;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "ld\ta,l" in body, f"the return does not take the low byte: {body}"
        assert "ld\ta,0ffh" not in body, (
            f"the return still normalises to a boolean: {body}"
        )


class TestConstantConditionsUseBitZero:
    """
    The bit-0 truth rule has to hold for a constant condition too, or the
    same source gets different answers at different -O levels. `IF NOT TRUE`
    with `TRUE LITERALLY '1'` folds to 0FEH, which is false.
    """

    SRC = """
        t: do;
        declare true literally '1';
        declare r byte;
        if not true then r = 1; else r = 2;
        end t;
    """

    def test_every_level_agrees(self) -> None:
        seen = set()
        for level in (0, 1, 2, 3):
            asm = Compiler(opt_level=level).compile(self.SRC, "<test>")
            assert asm is not None
            instrs = _instructions(asm)
            # r = 2 (the ELSE arm) is the correct answer.
            took_else = "ld\ta,2" in instrs
            seen.add(took_else)
            assert took_else, f"-O {level} took the THEN arm: {instrs}"
        assert seen == {True}

    def test_do_while_even_constant_never_runs(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare n byte;
            do while 2;
                n = n + 1;
            end;
            end t;
        """))
        assert "inc\ta" not in instrs and "add\ta,1" not in instrs, (
            f"DO WHILE 2 emitted a loop body; bit 0 of 2 is clear: {instrs}"
        )


class TestByteOperandsReachA:
    """
    A byte operand generated with `_gen_expr` lands in HL when it is a
    NumberLiteral (`ld hl,n`), so `IF 5 > X` compared an undefined A and
    `SHL(DOUBLE(hi),8) OR 5` destroyed the high byte with `ld hl,5`.
    """

    def test_constant_left_operand_reaches_a(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (x, r) byte;
            p: procedure;
                if 5 > x then r = 1;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "ld\thl,5" not in body, (
            f"the constant operand loaded into HL, not A: {body}"
        )

    def test_shl_or_with_a_constant_keeps_the_high_byte(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare hi byte, r address;
            p: procedure;
                r = shl(double(hi),8) or 5;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "ld\thl,5" not in body, (
            f"`ld hl,5` destroys the high byte parked in H: {body}"
        )
        assert "ld\th,a" in body and "ld\tl,a" in body, body


class TestIterativeDoBlockProcedure:
    """An iterative DO is a block: a PROCEDURE declared in it must be emitted."""

    def test_the_procedure_is_emitted(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (i, r) byte;
            p: procedure;
                do i = 1 to 3;
                    declare n byte;
                    bump: procedure; r = r + 1; end bump;
                    n = i;
                    call bump;
                end;
            end p;
            call p;
            end t;
        """))
        defined = {ins.split(':')[0] for ins in instrs if ':' in ins}
        called = {ins.split(None, 1)[1].strip() for ins in instrs
                  if ins.split(None, 1)[0] == "call"}
        for target in called:
            if not target.startswith("??"):
                assert target in defined, f"`call {target}` has no label: {instrs}"


class TestByteStoreToAStructureMember:
    """`rec.f = ch` stored from L while the BYTE value was in A."""

    def test_the_value_comes_from_a(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare rec structure (f byte, g address);
            declare ch byte;
            p: procedure;
                rec.f = ch;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "ld\t(hl),a" in body, f"the member store does not use A: {body}"
        assert "ld\ta,l" not in body, (
            f"the member store still reads the value out of L: {body}"
        )


class TestMoveCountReachesBC:
    """A non-constant BYTE count landed in A, so BC took the source address."""

    def test_count_is_widened_before_going_to_bc(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare src(4) byte, dst(4) byte, n byte;
            p: procedure;
                call move(n, .src, .dst);
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        i = body.index("ld\tb,h")
        # The instruction feeding HL just before must be the widened count,
        # not a pointer load.
        assert "ld\th,0" in body[:i], (
            f"the BYTE count was never widened into HL: {body}"
        )


class TestOptimizerStateDoesNotLeakBetweenProcedures:
    """`constants` / `copies` / `cse_cache` were never reset, so one
    procedure's constant was folded into another's body."""

    def test_a_read_is_not_folded_to_another_procedures_constant(self) -> None:
        asm = Compiler(opt_level=3).compile("""
            t: do;
            declare (v, r) byte;
            one: procedure;
                v = 1;
            end one;
            two: procedure;
                r = v;
            end two;
            call one;
            call two;
            end t;
        """, "<test>")
        assert asm is not None
        instrs = _instructions(asm)
        body = instrs[instrs.index("TWO:"):]
        assert "ld\ta,(V)" in body, (
            f"TWO folded V to ONE's constant instead of loading it: {body}"
        )


class TestCarrySensitiveScanNumeric:
    """
    The `scan$numeric` shape that MP/M II's SHOW, MSCHD and TOD all share:

        b = shl(b,3) + shl(b,1);   /* b * 10, may carry */
        if carry then call terminate;
        b = b + (chr - '0');
        if carry then call terminate;

    It exercises the byte-add carry, the CARRY built-in and the condition
    truth test together. `bit 0,x` leaves carry alone where `or a` cleared
    it, so the sequence between the add and the CARRY read has to stay free
    of anything that disturbs the flag.
    """

    def test_nothing_disturbs_carry_between_the_add_and_the_read(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (b, d, r) byte;
            p: procedure;
                b = shl(b,3) + shl(b,1);
                if carry then r = 1;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        # The CARRY built-in reads the flag with `sbc a,a`. Between the add
        # that sets carry and that read, nothing may write flags. `ld a,0`
        # would be safe in itself, but the peephole rewrites it to `xor a`,
        # which clears carry -- so the generator must not emit it here.
        reads = [i for i, ins in enumerate(body) if ins == "sbc\ta,a"]
        assert reads, f"the CARRY built-in did not emit a flag read: {body}"
        read = min(reads)
        adds = [i for i, ins in enumerate(body[:read]) if ins.startswith("add\t")]
        assert adds, f"the add that sets carry was optimised away: {body}"
        between = [ins for ins in body[max(adds) + 1:read] if not ins.endswith(':')]
        clobbers = [ins for ins in between
                    if ins.split()[0] in ("or", "and", "xor", "sub", "cp",
                                          "inc", "dec", "add", "adc", "sbc",
                                          "rlca", "rrca", "rla", "rra", "scf", "ccf")]
        assert not clobbers, (
            f"carry is destroyed between the add and the CARRY read by {clobbers}"
        )

    def test_the_add_survives_constant_folding(self) -> None:
        # At -O 3 constant propagation folded `a + b` to a literal, so the
        # `add` whose carry the next statement reads no longer existed.
        for level in (0, 1, 2, 3):
            asm = Compiler(opt_level=level).compile("""
                t: do;
                declare (a, b, s, r) byte;
                p: procedure;
                    a = 200; b = 100;
                    s = a + b;
                    if carry then r = 1;
                end p;
                call p;
                end t;
            """, "<test>")
            assert asm is not None
            body = _instructions(asm)
            assert any(i.startswith("add\t") for i in body), (
                f"-O {level} folded the carry-setting add away: {body}"
            )


class TestRemainingRegisterParkingFixes:
    """
    The rest of the "park a value in a register, then generate arbitrary
    code, then read it" family. Each of these had a fix but no test of its
    own, so reverting one left the suite green.
    """

    def test_iterative_do_index_survives_the_bound(self) -> None:
        # The 16-bit iterative DO parked the index in DE with `ex de,hl`
        # and then generated a bound free to emit `ld de,nn` or call out.
        instrs = _instructions(_compile("""
            t: do;
            declare (i, n) address;
            f: procedure address; return 10; end f;
            p: procedure;
                do i = 1 to f;
                    n = n + 1;
                end;
            end p;
            call p;
            end t;
        """))
        bad = _clobbered_before_use(instrs, "ex\tde,hl",
                                    ("call\t??subde",), ("ld de,", "ld\tde,", "call\t"))
        assert not bad, (
            f"the loop index parked in DE is destroyed by the bound: "
            f"{[instrs[j] for j in bad]}"
        )

    def test_embedded_assignment_value_survives_the_store(self) -> None:
        # The value was parked in B across _gen_store, which generates the
        # index expression of a subscripted target.
        instrs = _instructions(_compile("""
            t: do;
            declare arr(8) byte, i byte, v byte;
            f: procedure byte; return 3; end f;
            p: procedure;
                arr(f), v = i;
            end p;
            call p;
            end t;
        """))
        bad = _clobbered_before_use(instrs, "ld\tb,a", _B_CONSUMERS, _B_WRITES)
        assert not bad, (
            f"a multi-target byte value is destroyed before the second store: "
            f"{[instrs[j] for j in bad]}"
        )

    def test_shl_or_high_byte_survives_a_call(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare hi byte, r address;
            f: procedure byte; return 7; end f;
            p: procedure;
                r = shl(double(hi),8) or f;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        # `ld h,a` must not sit before a call that can return in HL.
        for i, ins in enumerate(body):
            if ins != "ld\th,a":
                continue
            rest = body[i + 1:]
            upto = rest[:rest.index("ld\tl,a")] if "ld\tl,a" in rest else rest
            assert not [x for x in upto if x.startswith("call\t")], (
                f"the high byte is parked in H across a call: {body}"
            )

    def test_not_of_a_parenthesised_comparison_uses_the_compare(self) -> None:
        # `IF NOT (a = b)` must reach the optimised compare rather than
        # materialising a value and testing it.
        instrs = _instructions(_compile("""
            t: do;
            declare (a, b, r) byte;
            p: procedure;
                if not (a = b) then r = 1;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "sub\tb" in body, f"the comparison was not used directly: {body}"
        assert "ld\ta,0ffh" not in body, (
            f"the comparison was materialised as a value first: {body}"
        )


class TestConstantPropagationIsFlowSensitive:
    """
    Constant and copy propagation were flow-insensitive, so a fact
    established on one path was reused on another that cannot reach it.
    At `-O 3` this miscompiled the most ordinary loop there is:

        n = 0;
        do while n < 3; call pc('0' + n); n = n + 1; end;

    folded the condition to always-true and pinned `n` at 0 through the
    body, so the loop printed `0` for ever. The same flow-insensitivity
    reached a GOTO-formed loop, an IF arm and a DO CASE arm.
    """

    def _o3(self, body: str) -> list:
        asm = Compiler(opt_level=3).compile(f"""
            t: do;
            declare (n, k, x, r) byte;
            rd: procedure byte; return 1; end rd;
            p: procedure;
                {body}
            end p;
            call p;
            end t;
        """, "<test>")
        assert asm is not None
        instrs = _instructions(asm)
        return instrs[instrs.index("P:"):]

    def test_do_while_condition_is_not_folded_from_before_the_loop(self) -> None:
        body = self._o3("n = 0; do while n < 3; r = n; n = n + 1; end;")
        assert any(i == "ld\ta,(N)" for i in body), (
            f"the loop never reloads n, so the condition was pinned: {body}"
        )

    def test_a_label_ends_the_region(self) -> None:
        # A GOTO can close a loop through any label.
        body = self._o3(
            "n = 0;"
            "lp: if n >= 3 then go to fin;"
            "    r = n; n = n + 1; go to lp;"
            "fin: r = 0;"
        )
        # With the label treated as fall-through, `n` stayed pinned at 0
        # and the whole `n >= 3` test folded away, leaving a bare backward
        # jump -- an infinite loop.
        assert "cp\t3" in body, (
            f"the loop test was folded away using a pre-label constant: {body}"
        )

    def test_a_then_arm_constant_does_not_survive_the_join(self) -> None:
        body = self._o3("n = 0; k = rd; if k = 1 then n = 5; r = n;")
        i = max(idx for idx, ins in enumerate(body) if ins == "ld\ta,5")
        after = body[i + 1:]
        assert "ld\ta,(N)" in after, (
            f"n was folded to the THEN arm's 5 after the join: {body}"
        )

    def test_a_case_arm_constant_does_not_reach_the_next_case(self) -> None:
        body = self._o3(
            "n = 0; k = rd;"
            "do case k;"
            "  do; n = 7; r = n; end;"
            "  do; r = n; end;"
            "end;"
        )
        # Case 1 may legitimately use the value n had BEFORE the DO CASE
        # (0); what it must not use is the 7 that case 0 assigns.
        labels = [i for i, ins in enumerate(body) if ins.startswith("??CASE1")]
        assert labels, f"no second case was emitted: {body}"
        second = body[labels[0]:]
        upto = second[:second.index("??CASEND0001:")] if "??CASEND0001:" in second else second
        assert "ld\ta,7" not in upto, (
            f"the second case reused the first case's constant: {body}"
        )


class TestCopyPropagationDoesNotDuplicateACall:
    """
    `k = rd;` recorded a copy of the identifier `rd`, and propagating `k`
    into a later use turned it into a second call. A PL/M parameterless
    procedure reference is a call, not a variable read.
    """

    def test_the_procedure_is_called_once(self) -> None:
        asm = Compiler(opt_level=3).compile("""
            t: do;
            declare (k, n) byte;
            rd: procedure byte; n = n + 1; return 1; end rd;
            p: procedure;
                k = rd;
                if k = 1 then n = 2;
            end p;
            call p;
            end t;
        """, "<test>")
        assert asm is not None
        instrs = _instructions(asm)
        body = instrs[instrs.index("P:"):]
        calls = [i for i in body if i == "call\tRD"]
        assert len(calls) == 1, (
            f"RD is called {len(calls)} times, not once: {body}"
        )


class TestByteValueOperandsReachA:
    """
    The value-producing twin of the condition-path repair.
    `_gen_byte_comparison` still opened with `_gen_expr(left)`, so a
    NumberLiteral left operand loaded as `ld hl,n` and the `sub b` compared
    an undefined A: `r = 5 > x` with x = 3 gave 0, not 0FFH.
    """

    def test_constant_left_operand_of_a_value_comparison(self) -> None:
        instrs = _instructions(_compile("""
            t: do;
            declare (x, r) byte;
            p: procedure;
                r = 5 > x;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "ld\thl,5" not in body, (
            f"the constant operand loaded into HL, not A: {body}"
        )
        assert "ld\ta,5" in body, f"the constant never reached A: {body}"


class TestCachedConstantIsNarrowedToItsDeclaredWidth:
    """
    `b = 300` stores 44 in a BYTE, but the optimizer cached 300, so at
    `-O 3` the following `IF b = 44` folded to false.
    """

    def test_byte_variable_caches_the_truncated_value(self) -> None:
        asm = Compiler(opt_level=3).compile("""
            t: do;
            declare (b, r) byte;
            p: procedure;
                b = 300;
                if b = 44 then r = 1; else r = 2;
            end p;
            call p;
            end t;
        """, "<test>")
        assert asm is not None
        body = _instructions(asm)
        body = body[body.index("P:"):]
        assert "ld\ta,1" in body, (
            f"`b = 300; if b = 44` took the false arm: {body}"
        )
        assert "ld\ta,2" not in body, f"the true arm was not selected: {body}"


class TestEmbeddedAssignmentWidensOnTheActualType:
    """
    `IF a1 > (ar(i) := b1)` — an embedded assignment into an ADDRESS element
    is typed BYTE by `_get_expr_type` but produced in HL, and the branch that
    parks the right operand in DE widened on the static type, splicing in a
    stale A. Both comparisons came out false.
    """

    def test_the_right_operand_is_widened_from_the_register_it_is_in(self) -> None:
        # `ar` is an ADDRESS array, so `_get_expr_type` types the embedded
        # assignment ADDRESS, but the assigned value is a BYTE and stays in
        # A. Widening on the static type emitted `ex de,hl`, swapping in
        # whatever HL held; both comparisons then came out false.
        instrs = _instructions(_compile("""
            t: do;
            declare a1 address;
            declare ar(8) address;
            declare (b1, i, r) byte;
            p: procedure;
                b1 = 7; i = 2;
                a1 = 300;
                if a1 > (ar(i) := b1) then r = 1; else r = 2;
            end p;
            call p;
            end t;
        """))
        body = instrs[instrs.index("P:"):]
        assert "pop\taf" in body, f"the byte value was not spilled: {body}"
        i = body.index("pop\taf")
        nxt = [x for x in body[i + 1:] if not x.endswith(":")][0]
        assert nxt != "ex\tde,hl", (
            "a byte value restored into A was widened with `ex de,hl`, "
            f"which takes DE from HL instead: {body}"
        )
        assert nxt == "ld\te,a", f"unexpected widening of the operand: {body}"
