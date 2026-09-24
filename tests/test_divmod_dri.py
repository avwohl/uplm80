"""Division and MOD give what Digital Research's PL/M-80 gives, for every operand.

PL/M-80 has one divide routine and routes every `/` and `MOD` through it: its
runtime library PLM80.LIB (MP/M II's PLM_WORK) holds a single divide, module
@P0029, and no 8-bit one. A BYTE operand is zero-extended into a register pair
first (SDIR loads a BYTE dividend with LHLD / MVI H,00H before CALL 39B0H), so
a quotient or remainder of BYTEs is an ADDRESS like any other.

@P0029 takes the dividend in DE and the divisor in HL and runs sixteen
shift-and-subtract steps with no test for a zero divisor. For a divisor that
is not zero it is exact: the partial remainder after k steps is below 2**k, so
the 16-bit shift of it never overflows. For a zero divisor every trial
subtraction succeeds, so the quotient is 0FFFFH and the remainder is the
dividend, untouched. uplm80 returned 0 for `x MOD 0`, and MP/M II's SDIR,
whose page length defaults to 0, reprinted its heading before every line
because UTIL7/DSH.PLM asks `cur$line mod page$len = 0`.

The oracle here executes @P0029's own bytes on a small 8080 interpreter, and
the ways uplm80 computes a quotient or a remainder are checked against it.
"""

import os

import pytest

from uplm80.ast_optimizer import ASTOptimizer
from uplm80.ast_view import BinaryOpKind
from uplm80.compiler import Compiler
from uplm80.runtime import plm_div, plm_mod

from ._toolchain import run_asm, tools_missing

# PLM80.LIB module @P0029, as linked at 0000H. Entry @P0029 is offset 0
# (divisor in HL); entry @P0030 is offset 2 (divisor already in BC).
DRI_P0029 = bytes.fromhex(
    "444d2100003e10f529eb9729eb8d916f7c986713d21900091bf13dc20700c9")

_MPM2_SRC = os.path.expanduser("~/src/mpm2/mpm2_external/mpm2src")


def dri_divide(dividend: int, divisor: int) -> tuple[int, int]:
    """Run DRI's @P0029 on an 8080 and return (quotient, remainder).

    Only the opcodes the routine contains are implemented; anything else is a
    failure of the oracle, not a result.
    """
    mem = bytearray(DRI_P0029) + bytes(4)
    b = c = a = 0
    d, e = dividend >> 8, dividend & 0xFF
    h, l = divisor >> 8, divisor & 0xFF
    cy = z = False
    stack = []
    pc = 0
    for _ in range(2000):
        op = mem[pc]
        if op == 0x44:                                  # MOV B,H
            b, pc = h, pc + 1
        elif op == 0x4D:                                # MOV C,L
            c, pc = l, pc + 1
        elif op == 0x21:                                # LXI H,nn
            l, h, pc = mem[pc + 1], mem[pc + 2], pc + 3
        elif op == 0x3E:                                # MVI A,n
            a, pc = mem[pc + 1], pc + 2
        elif op == 0xF5:                                # PUSH PSW
            stack.append((a, cy, z))
            pc += 1
        elif op == 0xF1:                                # POP PSW
            a, cy, z = stack.pop()
            pc += 1
        elif op in (0x29, 0x09):                        # DAD H / DAD B
            hl = (h << 8) | l
            hl += hl if op == 0x29 else (b << 8) | c
            cy = hl > 0xFFFF
            h, l, pc = (hl >> 8) & 0xFF, hl & 0xFF, pc + 1
        elif op == 0xEB:                                # XCHG
            d, e, h, l, pc = h, l, d, e, pc + 1
        elif op == 0x97:                                # SUB A
            a, cy, z, pc = 0, False, True, pc + 1
        elif op in (0x8D, 0x91, 0x98):                  # ADC L / SUB C / SBB B
            if op == 0x8D:
                r = a + l + cy
                cy = r > 0xFF
            else:
                r = a - (c if op == 0x91 else b) - (cy if op == 0x98 else 0)
                cy = r < 0
            a = r & 0xFF
            z = a == 0
            pc += 1
        elif op == 0x6F:                                # MOV L,A
            l, pc = a, pc + 1
        elif op == 0x7C:                                # MOV A,H
            a, pc = h, pc + 1
        elif op == 0x67:                                # MOV H,A
            h, pc = a, pc + 1
        elif op in (0x13, 0x1B):                        # INX D / DCX D
            de = (((d << 8) | e) + (1 if op == 0x13 else -1)) & 0xFFFF
            d, e, pc = de >> 8, de & 0xFF, pc + 1
        elif op == 0x3D:                                # DCR A (carry kept)
            a = (a - 1) & 0xFF
            z = a == 0
            pc += 1
        elif op in (0xD2, 0xC2):                        # JNC / JNZ
            taken = not cy if op == 0xD2 else not z
            pc = (mem[pc + 1] | (mem[pc + 2] << 8)) if taken else pc + 3
        elif op == 0xC9:                                # RET
            return (d << 8) | e, (h << 8) | l
        else:
            raise AssertionError(f"@P0029 oracle: opcode {op:02X} at {pc:04X}")
    raise AssertionError("@P0029 oracle did not return")


# Operand tables. BYTE values stay below 256; the constant divisors cover 0,
# 1, every strength-reduced shape (powers of two, 256) and ordinary values.
A_VALS = [0, 1, 2, 3, 7, 10, 255, 256, 257, 1000, 0x1000, 0x7FFF, 0x8000,
          0x8001, 0xFFFE, 0xFFFF]
B_VALS = [0, 1, 2, 3, 7, 10, 16, 128, 254, 255]
K_DIVISORS = [0, 1, 2, 3, 4, 7, 8, 10, 16, 128, 255, 256, 512, 1000, 0x1000,
              0x8000, 0xFFFF]
K_DIVIDENDS = [0, 1, 2, 7, 255, 256, 1000, 0xFFFF]


def test_oracle_bytes_are_dris_divide():
    """The bytes the oracle runs are PLM80.LIB's @P0029 and SDIR's copy of it."""
    lib = os.path.join(_MPM2_SRC, "PLM_WORK", "PLM80.LIB")
    sdir = os.path.join(_MPM2_SRC, "UTIL7", "SDIR.PRL")
    if not (os.path.exists(lib) and os.path.exists(sdir)):
        pytest.skip("MP/M II sources not present")
    with open(lib, "rb") as f:
        assert DRI_P0029 in f.read()
    # SDIR.PRL: a 100H header, then the image linked at 100H, so a file
    # offset is the address. The routine sits at 39B0H with its two jump
    # targets relocated by that much.
    with open(sdir, "rb") as f:
        image = f.read()
    relocated = bytearray(DRI_P0029)
    for at in (0x15, 0x1C):
        target = relocated[at] | (relocated[at + 1] << 8)
        target += 0x39B0
        relocated[at], relocated[at + 1] = target & 0xFF, target >> 8
    assert image[0x39B0:0x39B0 + len(DRI_P0029)] == relocated


def test_oracle_is_exact_division_except_by_zero():
    """What the bytes compute, stated in closed form: exact unless b = 0."""
    values = sorted(set(A_VALS + B_VALS + K_DIVISORS + K_DIVIDENDS
                        + list(range(0, 0x10000, 0x0FFF))))
    for dividend in values:
        for divisor in values:
            expect = ((dividend // divisor, dividend % divisor) if divisor
                      else (0xFFFF, dividend))
            assert dri_divide(dividend, divisor) == expect, (dividend, divisor)


def test_plm_div_and_mod_follow_the_oracle():
    for dividend in A_VALS + K_DIVIDENDS:
        for divisor in A_VALS + K_DIVISORS:
            q, r = dri_divide(dividend, divisor)
            assert plm_div(dividend, divisor) == q
            assert plm_mod(dividend, divisor) == r


def _asm(src: str, opt: int) -> str:
    out = Compiler(opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


# --- Compile time: the folder and the rewrites.

def test_constant_folder_follows_the_oracle():
    """`7 MOD 0' folds to 7 and `7 / 0' to 0FFFFH instead of being left alone."""
    opt = ASTOptimizer(2)
    for dividend in K_DIVIDENDS + A_VALS:
        for divisor in K_DIVISORS + A_VALS:
            q, r = dri_divide(dividend, divisor)
            assert opt._eval_binary_const(BinaryOpKind.DIV, dividend, divisor) == q
            assert opt._eval_binary_const(BinaryOpKind.MOD, dividend, divisor) == r


def test_zero_over_a_variable_is_not_folded_to_zero():
    """`0 / x' is 0FFFFH when x is 0, so it is not the constant 0."""
    for opt in (1, 2, 3):
        asm = _asm("""
t: do;
declare (x, r) address;
r = 0 / x;
end t;
""", opt)
        assert "??div16" in asm, opt


def test_an_identity_does_not_drop_a_call():
    """`f MOD 1', `0 MOD f' and `f / 1' still call f."""
    for opt in (1, 2, 3):
        for expr in ("f mod 1", "0 mod f", "f / 1"):
            asm = _asm(f"""
t: do;
declare r address;
f: procedure address external; end f;
r = {expr};
end t;
""", opt)
            assert "call\tF" in asm, (opt, expr)


@pytest.mark.parametrize("use", [
    "y = {e};",
    "if ({e}) = 3 then call pb(1);",
    "if ({e}) < y then call pb(1);",
    "call pb({e});",
    "buf({e}) = 0;",
    "if {e} then call pb(1);",
    "y = ({e}) + y;",
])
def test_a_byte_remainder_costs_nothing_where_its_width_does_not_show(use):
    """`x MOD 8' is an ADDRESS, so it becomes DOUBLE(x AND 7); where only its
    low byte is read, or it is compared with a BYTE, that must compile to
    exactly what `x AND 7' does."""
    def compile_with(expr):
        asm = _asm(f"""
t: do;
declare (x, y) byte, buf(16) byte;
pb: procedure (c) external; declare c byte; end pb;
{use.format(e=expr)}
end t;
""", 2)
        return [line for line in asm.splitlines() if not line.startswith(";")]
    assert compile_with("x mod 8") == compile_with("x and 7")


# --- End to end: compile a table of divisions, run it, compare with the oracle.

def _run(src: str, opt: int) -> list[int]:
    """Compile, assemble, link and run ``src``; the hex words it printed."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    r = run_asm(_asm(src, opt), timeout=120)
    assert "Program exit" in r.stderr, r.stderr[-500:]
    return [int(w, 16) for w in r.stdout.replace("\r", "").split()]


# A program is a procedure `run' (uplm80 emits module-level DATA inline at the
# head of the code, where the entry point would run into it) and `ph', which
# prints an ADDRESS as four hex digits.
_PRELUDE = [
    "t: do;",
    "mon1: procedure (f, p) external; declare f byte, p address; end mon1;",
    "hexd: procedure (d);",
    "  declare d byte;",
    "  d = d and 0fh;",
    "  if d < 10 then call mon1(2, d + '0');",
    "  else call mon1(2, d + 37h);",
    "end hexd;",
    "ph: procedure (v);",
    "  declare v address;",
    "  call hexd(shr(high(v), 4)); call hexd(high(v));",
    "  call hexd(shr(low(v), 4)); call hexd(low(v));",
    "  call mon1(2, ' ');",
    "end ph;",
    "run: procedure;",
]
_POSTLUDE = ["end run;", "call run;", "end t;"]


def _hex(v: int) -> str:
    return f"0{v:x}h" if v > 9 else str(v)


def _compare(expect: list[tuple[str, int]], got: list[int]) -> None:
    assert len(got) == len(expect), (len(got), len(expect))
    wrong = [f"{label}: got {g:04X}, DRI {v:04X}"
             for (label, v), g in zip(expect, got) if g != v]
    assert not wrong, f"{len(wrong)} of {len(expect)} differ:\n" + "\n".join(wrong[:40])


def _probe_stmts(expr: str) -> list[str]:
    """A quotient or remainder is an ADDRESS: adding 0FFH carries into the
    high byte and subtracting 1 from 0 borrows from it, where BYTE arithmetic
    would wrap. Each case is printed plain and both ways."""
    return [f"call ph({expr});",
            f"call ph(({expr}) + 0ffh);",
            f"call ph(({expr}) - 1);"]


def _program(everything: bool) -> tuple[str, list[tuple[str, int]]]:
    """PL/M source that prints every case, and the values it must print.

    Without ``everything``, only the cases the runtime routines compute:
    both operands variables, and constants that only -O3 propagates.
    """
    decls: list[str] = []
    body: list[str] = []
    expect: list[tuple[str, int]] = []

    def check(label: str, expr: str, value: int) -> None:
        for suffix, v in (("", value), (" + 0FFH", value + 0xFF), (" - 1", value - 1)):
            expect.append((f"{label}: {expr}{suffix}", v & 0xFFFF))

    tables = (("A", A_VALS, "av"), ("B", B_VALS, "bv"))

    # Both operands variables: the runtime routines, for every width mix.
    for dname, dvals, dtab in tables:
        for sname, svals, stab in tables:
            dvar = "a" if dname == "A" else "x"
            svar = "b" if sname == "A" else "y"
            ops = (f"{dvar} / {svar}", f"{dvar} mod {svar}")
            body.append(f"do i = 0 to {len(dvals) - 1};")
            body.append(f"  do j = 0 to {len(svals) - 1};")
            body.append(f"  {dvar} = {dtab}(i); {svar} = {stab}(j);")
            for expr in ops:
                body.extend(_probe_stmts(expr))
            body.append("  end;")
            body.append("end;")
            for p in dvals:
                for q in svals:
                    for expr, v in zip(ops, dri_divide(p, q)):
                        check(f"{dname} {p:#x}, {sname} {q:#x}", expr, v)

    # Constants reached through variables (-O3 propagates them): SDIR's case.
    body.append("a = 7; b = 0; x = 7; y = 0;")
    for expr, v in (("a mod b", 7), ("a / b", 0xFFFF),
                    ("x mod y", 7), ("x / y", 0xFFFF)):
        body.extend(_probe_stmts(expr))
        check("propagated", expr, v)

    if everything:
        _compile_time_cases(tables, decls, body, check, expect)

    src = "\n".join([
        *_PRELUDE,
        "declare (a, b) address;",
        "declare (x, y, i, j) byte;",
        "declare av(*) address data (" + ", ".join(_hex(v) for v in A_VALS) + ");",
        "declare bv(*) byte data (" + ", ".join(_hex(v) for v in B_VALS) + ");",
        *decls,
        *body,
        *_POSTLUDE,
    ])
    return src, expect


def _compile_time_cases(tables, decls, body, check, expect) -> None:
    """Cases with a constant operand: folding, strength reduction, DATA."""
    # Constant divisor, variable dividend: strength reduction and folding.
    for dname, dvals, dtab in tables:
        dvar = "a" if dname == "A" else "x"
        body.append(f"do i = 0 to {len(dvals) - 1};")
        body.append(f"  {dvar} = {dtab}(i);")
        for k in K_DIVISORS:
            for expr in (f"{dvar} / {_hex(k)}", f"{dvar} mod {_hex(k)}"):
                body.extend(_probe_stmts(expr))
        body.append("end;")
        for p in dvals:
            for k in K_DIVISORS:
                ops = (f"{dvar} / {_hex(k)}", f"{dvar} mod {_hex(k)}")
                for expr, v in zip(ops, dri_divide(p, k)):
                    check(f"{dname} {p:#x}", expr, v)

    # Constant dividend, variable divisor.
    for sname, svals, stab in tables:
        svar = "b" if sname == "A" else "y"
        body.append(f"do j = 0 to {len(svals) - 1};")
        body.append(f"  {svar} = {stab}(j);")
        for k in K_DIVIDENDS:
            for expr in (f"{_hex(k)} / {svar}", f"{_hex(k)} mod {svar}"):
                body.extend(_probe_stmts(expr))
        body.append("end;")
        for q in svals:
            for k in K_DIVIDENDS:
                ops = (f"{_hex(k)} / {svar}", f"{_hex(k)} mod {svar}")
                for expr, v in zip(ops, dri_divide(k, q)):
                    check(f"{sname} {q:#x}", expr, v)

    # Both constant: folded at -O1 and above, the runtime at -O0.
    for k in K_DIVIDENDS:
        for m in K_DIVISORS:
            ops = (f"{_hex(k)} / {_hex(m)}", f"{_hex(k)} mod {_hex(m)}")
            for expr, v in zip(ops, dri_divide(k, m)):
                body.extend(_probe_stmts(expr))
                check("constants", expr, v)

    # DATA and INITIAL values are constant expressions too.
    dk = [(7, "mod", 3), (7, "/", 0), (7, "mod", 0), (0, "/", 0), (0, "mod", 0),
          (0xFFFF, "mod", 0), (1000, "mod", 7), (1000, "/", 7), (256, "/", 256)]
    bk = [(7, "mod", 3), (7, "mod", 0), (0, "mod", 0), (250, "/", 7),
          (200, "mod", 0), (255, "mod", 16)]

    def text(items):
        return ", ".join(f"{_hex(p)} {op} {_hex(q)}" for p, op, q in items)

    decls.append(f"declare dk(*) address data ({text(dk)});")
    decls.append(f"declare bk(*) byte data ({text(bk)});")
    decls.append(f"declare ik({len(dk)}) address initial ({text(dk)});")
    for name, items in (("dk", dk), ("bk", bk), ("ik", dk)):
        body.append(f"do i = 0 to last({name}); call ph({name}(i)); end;")
        for p, op, q in items:
            expect.append((f"{name}: {p} {op} {q}", dri_divide(p, q)[op != "/"]))


@pytest.mark.parametrize("opt", [0, 1, 2, 3])
def test_compiled_divisions_match_dri(opt):
    """Every quotient and remainder the program prints is DRI's, at every -O."""
    src, expect = _program(everything=True)
    _compare(expect, _run(src, opt))


@pytest.mark.parametrize("opt", [0, 2])
def test_shr_by_seven_keeps_bit_15(opt):
    """`a / 128' is SHR(a, 7), whose result has nine bits."""
    values = [0x7F, 0x80, 0x7FFF, 0x8000, 0x8001, 0xC07F, 0xFFFF]
    src = "\n".join([
        *_PRELUDE,
        "declare (a, i) address;",
        "declare av(*) address data (" + ", ".join(_hex(v) for v in values) + ");",
        f"do i = 0 to {len(values) - 1};",
        "  a = av(i); call ph(shr(a, 7)); call ph(a / 128);",
        "end;",
        *_POSTLUDE,
    ])
    expect = [(f"{label}({v:#x})", v >> 7) for v in values for label in ("shr", "/128")]
    _compare(expect, _run(src, opt))
