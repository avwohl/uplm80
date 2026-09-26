"""How a call passes its arguments: PL/M-80's convention (uplm80 0.4.0).

Intel's PL/M-80 passes the last argument in DE (E for a BYTE parameter), the
one before it in BC (C), a single one in BC (C), and any earlier ones pushed
left to right, one word each, which the callee takes off the stack.  A BYTE
result comes back in A, an ADDRESS one in HL.  uplm80 calls that way, so its
code links with assembly written for PL/M-80 (DRI's X0100.ASM, LDMONX.ASM)
and with what PL/M-80 compiled.  The one exception is a procedure nothing
outside the compile can reach with one parameter, which takes it in A or HL.

These tests pin the sequences at -O0; tests/test_calling_convention_run.py
runs them against assembly written to the convention.
"""

import pytest

from uplm80 import runtime
from uplm80.codegen import AsmLine, CodeGenerator, Mode
from uplm80.compiler import Compiler

DECLS = """
declare (b1, b2, b3, b4, b5, f) byte, (a1, a2, a3, a4, q) address;
p0: procedure external; end p0;
p1b: procedure (x) external; declare x byte; end p1b;
p1a: procedure (x) external; declare x address; end p1a;
p2bb: procedure (x, y) external; declare (x, y) byte; end p2bb;
p2aa: procedure (x, y) external; declare (x, y) address; end p2aa;
p2ab: procedure (x, y) external; declare x address, y byte; end p2ab;
p2ba: procedure (x, y) external; declare x byte, y address; end p2ba;
p3aba: procedure (x, y, z) external; declare (x, z) address, y byte; end p3aba;
p3bab: procedure (x, y, z) external; declare (x, z) byte, y address; end p3bab;
p4: procedure (w, x, y, z) external; declare (w, y) byte, (x, z) address; end p4;
p5: procedure (v, w, x, y, z) external; declare (v, x, z) byte, (w, y) address; end p5;
fb: procedure byte external; end fb;
fa: procedure address external; end fa;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
mon2: procedure (f, a) byte external; declare f byte, a address; end mon2;
"""


def _asm(src: str, opt: int = 0, mode: Mode = Mode.CPM) -> str:
    compiler = Compiler(mode=mode, opt_level=opt)
    out = compiler.compile(src, "<test>")
    assert out is not None, [str(e) for e in compiler.errors.errors]
    return out


def _lines(asm: str) -> list[str]:
    return [line.strip() for line in asm.splitlines()]


def _main(stmts: str, decls: str = DECLS, mode: Mode = Mode.CPM) -> list[str]:
    """The main program's code for ``stmts``, between setting SP and `jp 0'."""
    lines = _lines(_asm("t: do;\n" + decls + stmts + "\nend t;\n", mode=mode))
    start = next(i for i, l in enumerate(lines) if l.startswith("ld\tsp,")) + 1
    return lines[start:next(i for i in range(start, len(lines))
                            if lines[i] in ("jp\t0", "jp\t??BOOT"))]


def _proc(asm: str, name: str) -> list[str]:
    """Procedure ``name``'s code, from its label to the next procedure."""
    lines = _lines(asm)
    start = lines.index(f"{name}:") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith(
        ("; Procedure", "public\t", "extrn\t", "dseg", "jp\t??RTEND"))), len(lines))
    return [l for l in lines[start:end] if l and not l.startswith(";")]


def _errors(src: str) -> list[str]:
    compiler = Compiler(opt_level=0)
    assert compiler.compile(src, "<test>") is None, "compiled a program PL/M-80 rejects"
    return [str(e) for e in compiler.errors.errors]


# ---- Where each argument goes ------------------------------------------------

@pytest.mark.parametrize("stmt, code", [
    ("call p0;", []),
    # one: BC, or C for a BYTE
    ("call p1b(5);", ["ld\tc,5"]),
    ("call p1b(b1);", ["ld\ta,(B1)", "ld\tc,a"]),
    ("call p1b(b1 + 1);", ["ld\ta,(B1)", "add\ta,1", "ld\tc,a"]),
    ("call p1a(5);", ["ld\tbc,5"]),
    ("call p1a(a1);", ["ld\tbc,(A1)"]),
    ("call p1a(a1 + 1);", ["ld\thl,(A1)", "inc\thl", "ld\tb,h", "ld\tc,l"]),
    # two: BC then DE, in each order of types
    ("call p2bb(b1, b2);", ["ld\ta,(B1)", "ld\tc,a", "ld\ta,(B2)", "ld\te,a"]),
    ("call p2aa(1, 2);", ["ld\tbc,1", "ld\tde,2"]),
    ("call p2aa(a1, a2);", ["ld\tbc,(A1)", "ld\thl,(A2)", "ex\tde,hl"]),
    ("call p2aa(a1, a2 + 1);", ["ld\tbc,(A1)", "ld\thl,(A2)", "inc\thl", "ex\tde,hl"]),
    ("call p2ab(a1, b2);", ["ld\tbc,(A1)", "ld\ta,(B2)", "ld\te,a"]),
    ("call p2ab(a1, 5);", ["ld\tbc,(A1)", "ld\te,5"]),
    ("call p2ba(b1, a2);", ["ld\ta,(B1)", "ld\tc,a", "ld\thl,(A2)", "ex\tde,hl"]),
    # three and more: the first ones pushed, left to right
    ("call p3aba(a1, b2, a3);",
     ["ld\thl,(A1)", "push\thl", "ld\ta,(B2)", "ld\tc,a", "ld\thl,(A3)", "ex\tde,hl"]),
    ("call p3bab(b1, a2, b3);",
     ["ld\ta,(B1)", "ld\tl,a", "push\thl", "ld\tbc,(A2)", "ld\ta,(B3)", "ld\te,a"]),
    ("call p4(b1, a2, b3, a4);",
     ["ld\ta,(B1)", "ld\tl,a", "push\thl", "ld\thl,(A2)", "push\thl",
      "ld\ta,(B3)", "ld\tc,a", "ld\thl,(A4)", "ex\tde,hl"]),
    ("call p5(b1, a2, b3, a4, b5);",
     ["ld\ta,(B1)", "ld\tl,a", "push\thl", "ld\thl,(A2)", "push\thl",
      "ld\ta,(B3)", "ld\tl,a", "push\thl", "ld\tbc,(A4)", "ld\ta,(B5)", "ld\te,a"]),
    ("call p5(7, 1234H, b1 + b2, a1 + a2, 9);",
     ["ld\thl,7", "push\thl", "ld\thl,1234H", "push\thl",
      "ld\ta,(B1)", "push\taf", "ld\ta,(B2)", "ld\tb,a", "pop\taf", "add\ta,b",
      "ld\tl,a", "push\thl",
      "ld\thl,(A2)", "ex\tde,hl", "ld\thl,(A1)", "add\thl,de", "ld\tb,h", "ld\tc,l",
      "ld\te,9"]),
])
def test_arguments_are_placed_as_plm80_places_them(stmt, code):
    callee = stmt.split()[1].split("(")[0].rstrip(";").upper()
    # And nothing is popped or added to SP after the call: the callee takes
    # the pushed arguments off the stack.
    assert _main(stmt) == code + [f"call\t{callee}"]


@pytest.mark.parametrize("stmt, code", [
    # BYTE to ADDRESS: zero-extended, in B, D or H
    ("call p1a(b1);", ["ld\ta,(B1)", "ld\tc,a", "ld\tb,0"]),
    ("call p2aa(a1, b2);", ["ld\tbc,(A1)", "ld\ta,(B2)", "ld\te,a", "ld\td,0"]),
    ("call p3aba(b1, b2, a3);",
     ["ld\ta,(B1)", "ld\tl,a", "ld\th,0", "push\thl", "ld\ta,(B2)", "ld\tc,a", "ld\thl,(A3)", "ex\tde,hl"]),
    # ADDRESS to BYTE: the low byte
    ("call p1b(300);", ["ld\tc,2CH"]),
    ("call p1b(a1);", ["ld\thl,(A1)", "ld\ta,l", "ld\tc,a"]),
    ("call p2bb(b1, 300);", ["ld\ta,(B1)", "ld\tc,a", "ld\te,2CH"]),
    ("call p3bab(300, 300, b3);",
     ["ld\thl,2CH", "push\thl", "ld\tbc,012CH", "ld\ta,(B3)", "ld\te,a"]),
])
def test_an_argument_is_converted_to_its_parameters_type(stmt, code):
    callee = stmt.split()[1].split("(")[0].upper()
    assert _main(stmt) == code + [f"call\t{callee}"]


@pytest.mark.parametrize("stmt, saved", [
    ("call p2aa(a1, fa);", True),
    ("call p2bb(b1, fb);", True),
    ("call p2aa(a1, a2);", False),
    ("call p2ab(a1, 5);", False),
    ("call p2aa(a1, a2 + a3);", False),
    ("call p4(b1, a2, b3, fa);", True),
    # runtime routines: ??subde leaves BC, the others write B or C
    ("call p2aa(a1, a2 - a3);", False),
    ("call p4(b1, a2, b3, a4 - a1);", False),
    ("call p2aa(a1, a2 * a3);", True),
    ("call p2aa(a1, a2 / a3);", True),
    ("call p2aa(a1, a2 mod a3);", True),
    ("call p2ab(a1, input(b2));", True),
])
def test_the_next_to_last_argument_is_kept_while_the_last_is_evaluated(stmt, saved):
    """The last argument's code runs with the one before it in BC: where it
    may write B or C - it calls a procedure, or a runtime routine that
    writes them - BC is saved round it."""
    code = _main(stmt)
    i = code.index("ld\tbc,(A1)") if "ld\tbc,(A1)" in code else \
        max(i for i, l in enumerate(code) if l.startswith("ld\tc,"))
    rest = code[i + 1:]
    if saved:
        assert rest[0] == "push\tbc" and rest[-2] == "pop\tbc", code
    else:
        assert "push\tbc" not in rest and "pop\tbc" not in rest, code


def _routines() -> dict[str, list[AsmLine]]:
    """Each routine of runtime.py, by name, as lines of code."""
    routines = {}
    for const, text in vars(runtime).items():
        if not const.startswith("RUNTIME_"):
            continue
        name = text.split(":", 1)[0]
        code = []
        for line in text.splitlines():
            line = line.split(";", 1)[0].strip()
            if line and not line.endswith(":"):
                op, _, operands = line.partition("\t")
                code.append(AsmLine(opcode=op, operands=operands.strip()))
        routines[name] = code
    return routines


def test_a_runtime_routine_is_taken_to_write_what_its_code_writes():
    """BC is kept round the last argument's code where that may write B or C.
    A call of a runtime routine writes what the routine's code writes of B,
    C, D and E, a call in it counting for what its callee writes; a call of
    anything else - another routine, a procedure - may write all four."""
    routines = _routines()
    table = CodeGenerator._RUNTIME_WRITES
    assert set(table) <= set(routines)
    for name, written in table.items():
        assert {r for r in "bcde" if CodeGenerator._writes(routines[name], {r})} == written, name
    for name in sorted(set(routines) - set(table)) + ["P", "5"]:
        assert CodeGenerator._writes([AsmLine(opcode="call", operands=name)], {"b"}), name
    assert not CodeGenerator._writes([AsmLine(opcode="call", operands="nz,??subde")], set("bcde"))
    assert CodeGenerator._writes([AsmLine(opcode="call", operands="nz,P")], {"c"})


def test_a_call_in_an_argument_is_made_before_anything_is_placed_after_it():
    """Arguments are evaluated left to right, each call before the next
    argument is placed: nothing is left in a register across one."""
    code = _main("call p3aba(fa, fb, fa);")
    assert code == ["call\tFA", "push\thl",
                    "call\tFB", "ld\tc,a",
                    "push\tbc", "call\tFA", "ex\tde,hl", "pop\tbc",
                    "call\tP3ABA"]


def test_a_function_result_comes_back_in_a_or_hl():
    code = _main("b1 = fb; a1 = fa; a2 = fb;")
    assert code == ["call\tFB", "ld\t(B1),a", "call\tFA", "ld\t(A1),hl",
                    "call\tFB", "ld\tl,a", "ld\th,0", "ld\t(A2),hl"]


# ---- Entry -------------------------------------------------------------------

ENTRIES = """t: do;
declare (g1, g2, g3, g4, g5) address;
e1b: procedure (x) public; declare x byte; g1 = x; end e1b;
e1a: procedure (x) public; declare x address; g1 = x; end e1a;
e2ba: procedure (x, y) public; declare x byte, y address; g1 = x; g2 = y; end e2ba;
e2ab: procedure (x, y) public; declare x address, y byte; g1 = x; g2 = y; end e2ab;
e3aba: procedure (x, y, z) public; declare (x, z) address, y byte;
  g1 = x; g2 = y; g3 = z; end e3aba;
e4: procedure (w, x, y, z) public; declare (w, y) byte, (x, z) address;
  g1 = w; g2 = x; g3 = y; g4 = z; end e4;
e5: procedure (v, w, x, y, z) public; declare (v, x, z) byte, (w, y) address;
  g1 = v; g2 = w; g3 = x; g4 = y; g5 = z; end e5;
end t;
"""


@pytest.mark.parametrize("name, entry", [
    ("E1B", ["ld\ta,c", "ld\t(??AUTO+0),a"]),
    ("E1A", ["ld\th,b", "ld\tl,c", "ld\t(??AUTO+0),hl"]),
    ("E2BA", ["ld\t(??AUTO+1),de", "ld\ta,c", "ld\t(??AUTO+0),a"]),
    ("E2AB", ["ld\ta,e", "ld\t(??AUTO+2),a", "ld\th,b", "ld\tl,c", "ld\t(??AUTO+0),hl"]),
    # Three and more: the last two from DE and BC, then the return address
    # off the stack, the pushed ones after it (the last pushed first), and
    # the first by exchanging it with the return address.
    ("E3ABA", ["ld\t(??AUTO+3),de", "ld\ta,c", "ld\t(??AUTO+2),a",
               "pop\thl", "ex\t(sp),hl", "ld\t(??AUTO+0),hl"]),
    ("E4", ["ld\t(??AUTO+4),de", "ld\ta,c", "ld\t(??AUTO+3),a",
            "pop\thl", "pop\tde", "ld\t(??AUTO+1),de",
            "ex\t(sp),hl", "ld\ta,l", "ld\t(??AUTO+0),a"]),
    ("E5", ["ld\ta,e", "ld\t(??AUTO+6),a", "ld\t(??AUTO+4),bc",
            "pop\thl", "pop\tde", "ld\ta,e", "ld\t(??AUTO+3),a",
            "pop\tde", "ld\t(??AUTO+1),de",
            "ex\t(sp),hl", "ld\ta,l", "ld\t(??AUTO+0),a"]),
])
def test_a_procedure_takes_its_arguments_at_entry(name, entry):
    code = _proc(_asm(ENTRIES), name)
    assert code[:len(entry)] == entry, code
    assert code.count("ret") == 1 and code[-1] == "ret", code
    if name[1] in "12":
        # One or two parameters: BC and DE are left as they came, so a body
        # can pass them on (DRI's `MON1: ... GO TO BDOS').
        for line in entry:
            assert not line.startswith(("ld\tb,", "ld\tc,", "ld\td,", "ld\te,", "pop", "ex")), line


# ---- The one exception: A or HL ----------------------------------------------

PRIVATE = """t: do;
declare (g, w) address, v byte;
pb: procedure (x); declare x byte; v = x; end pb;
pa: procedure (x); declare x address; w = x; end pa;
{extra}
call pb(v); call pa(w);
end t;
"""


def test_a_private_procedure_with_one_parameter_takes_it_in_a_or_hl():
    asm = _asm(PRIVATE.format(extra=""))
    lines = _lines(asm)
    i = lines.index("call\tPB")
    assert lines[i - 1:i + 3] == ["ld\ta,(V)", "call\tPB", "ld\thl,(W)", "call\tPA"], asm
    assert _proc(asm, "PB")[0].endswith("),a")
    assert _proc(asm, "PA")[0].endswith("),hl")


@pytest.mark.parametrize("extra", [
    "g = .pb;",                                         # in a statement
    "declare t (2) address data (.pb, 0);",             # in DATA
    "declare i address initial (.pb);",                 # in INITIAL
])
def test_a_procedure_whose_address_is_taken_uses_bc(extra):
    """A CALL through an address, which cannot know which procedure it
    calls, passes a single argument in BC, and so must every call of one
    whose address is taken."""
    asm = _asm(PRIVATE.format(extra=extra))
    assert _proc(asm, "PB")[0] == "ld\ta,c", _proc(asm, "PB")
    lines = _lines(asm)
    i = lines.index("call\tPB")
    assert lines[i - 2:i] == ["ld\ta,(V)", "ld\tc,a"], lines[i - 3:i + 1]
    # pa's address is not taken.
    assert _proc(asm, "PA")[0].endswith("),hl")


@pytest.mark.parametrize("attrs", ["public", "reentrant"])
def test_a_public_or_reentrant_procedure_with_one_parameter_uses_bc(attrs):
    decls = f"declare v byte;\np: procedure (x) {attrs}; declare x byte; v = x; end p;\n"
    assert _main("call p(v);", decls=decls) == ["ld\ta,(V)", "ld\tc,a", "call\tP"]


# ---- REENTRANT ---------------------------------------------------------------

REENTRANT = """t: do;
declare r address;
r0: procedure address reentrant; return 1; end r0;
r1: procedure (x) byte reentrant; declare x byte; return x; end r1;
r2: procedure (x, y) address reentrant; declare (x, y) address;
  declare i byte;
  do i = 0 to 3; if x = y then return x; x = x + 1; end;
  return 0;
end r2;
r3: procedure (x, y, z) address reentrant; declare (x, y, z) address; return x + y + z; end r3;
r9: procedure (a, b, c, d, e, f, g, h, i) address reentrant;
  declare (a, b, c, d, e, f, g, h, i) address; return a; end r9;
r = r0 + r1(1) + r2(1, 2) + r3(1, 2, 3) + r9(1, 2, 3, 4, 5, 6, 7, 8, 9);
end t;
"""

_FRAME = ["push\tix", "ld\tix,0", "add\tix,sp"]


def _exit(n: int) -> list[str]:
    return ["ld\tsp,ix", "pop\tix"] + (
        ["ret"] if n == 0 else ["pop\tde"] + ["pop\tbc"] * n + ["push\tde", "ret"])


@pytest.mark.parametrize("name, n, entry", [
    ("R0", 0, []),
    ("R1", 1, ["pop\thl", "push\tbc", "push\thl"]),
    ("R2", 2, ["pop\thl", "push\tbc", "push\tde", "push\thl"]),
    ("R3", 3, ["pop\thl", "push\tbc", "push\tde", "push\thl"]),
])
def test_a_reentrant_procedure_pushes_bc_and_de_and_removes_every_argument(name, n, entry):
    """The arguments in BC and DE go on the stack under the return address,
    after any pushed ones, which is the frame 0.3.x had: parameter i of n is
    at IX + 4 + 2(n - i).  Every exit removes all n words, and leaves A and
    HL alone."""
    code = _proc(_asm(REENTRANT), name)
    assert code[:len(entry) + 3] == entry + _FRAME, code
    assert code[-len(_exit(n)):] == _exit(n), code


def test_a_return_from_a_counted_loop_in_a_reentrant_procedure():
    """The loop's count is on the stack; `ld sp,ix' drops it with the frame."""
    code = _proc(_asm(REENTRANT), "R2")
    i = code.index("pop\tbc")                 # the count, before the RETURN
    assert code[i + 1:i + 1 + len(_exit(2))] == _exit(2), code


def test_a_reentrant_procedure_with_many_arguments_moves_sp_once():
    code = _proc(_asm(REENTRANT), "R9")
    assert code[-10:] == ["ld\tsp,ix", "pop\tix", "pop\tbc", "ex\tde,hl", "ld\thl,18",
                          "add\thl,sp", "ld\tsp,hl", "ex\tde,hl", "push\tbc", "ret"], code


def test_a_call_of_a_reentrant_procedure_is_like_any_other():
    code = _main("q = r3(a1, 2, b1);",
                 decls="declare (q, a1) address, b1 byte;\n"
                       "r3: procedure (x, y, z) address reentrant; declare (x, y, z) address;\n"
                       "return x + y + z; end r3;\n")
    assert code == ["ld\thl,(A1)", "push\thl", "ld\tbc,2", "ld\ta,(B1)", "ld\te,a", "ld\td,0",
                    "call\tR3", "ld\t(Q),hl"]


# ---- CALL through an address -------------------------------------------------

@pytest.mark.parametrize("args, code", [
    ("", []),
    ("(b1)", ["ld\ta,(B1)", "ld\tc,a", "ld\tb,0"]),
    ("(a1, b1)", ["ld\tbc,(A1)", "ld\ta,(B1)", "ld\te,a", "ld\td,0"]),
    ("(1, 2, 3)", ["ld\thl,1", "push\thl", "ld\tbc,2", "ld\tde,3"]),
    ("(1, 2, 3, 4)", ["ld\thl,1", "push\thl", "ld\thl,2", "push\thl", "ld\tbc,3", "ld\tde,4"]),
])
def test_a_call_through_an_address_places_the_arguments_as_any_call(args, code):
    """Each argument widened to ADDRESS, then the address, then ??jphl."""
    assert _main(f"call q{args};") == code + ["ld\thl,(Q)", "call\t??jphl"]


def test_a_call_through_an_address_keeps_bc_and_de_while_the_address_is_found():
    """The address is evaluated after the arguments.  A variable's is loaded
    straight into HL; one that writes B, C, D or E has them saved round it."""
    assert "push\tbc" not in _main("call q(a1, a2);")
    decls = DECLS + "declare t (4) address;\n"
    code = _main("call t(b1)(a1, a2);", decls=decls)
    i = code.index("ld\thl,(A2)")
    assert code[i + 1:i + 4] == ["ex\tde,hl", "push\tbc", "push\tde"], code
    assert code[-3:] == ["pop\tde", "pop\tbc", "call\t??jphl"], code


def test_the_module_brings_jphl_and_no_jpde():
    lines = _lines(_asm("t: do;\n" + DECLS + "call q(1, 2);\nend t;\n"))
    i = lines.index("??jphl:")
    assert [l for l in lines[i + 1:i + 4] if l and not l.startswith(";")][0] == "jp\t(hl)"
    assert "??jpde:" not in lines


@pytest.mark.parametrize("n", [2, 3, 4])
def test_a_call_through_an_address_with_several_arguments_does_not_warn(n, capsys):
    args = ", ".join(str(i) for i in range(1, n + 1))
    _asm("t: do;\n" + DECLS + f"call q({args});\nend t;\n")
    assert "warning" not in capsys.readouterr().err


# ---- MON1 and MON2 -----------------------------------------------------------

def test_mon1_with_a_constant_function_calls_the_bdos():
    code = _main("call mon1(9, .a1); call mon1(9, a1); b1 = mon2(11, 0);")
    assert code == ["ld\tde,A1", "ld\tc,9", "call\t5",
                    "ld\thl,(A1)", "ex\tde,hl", "ld\tc,9", "call\t5",
                    "ld\tde,0", "ld\tc,0BH", "call\t5", "ld\t(B1),a"]


def test_a_byte_argument_to_mon1_is_zero_extended():
    """E alone left D as it was; a function that reads DE whole (MP/M's 141,
    delay) got a garbage high byte.  PL/M-80 zero-extends it (MPMLDR at
    03D1: LHLD char / MVI H,0 / XCHG / MVI C,2)."""
    assert _main("call mon1(2, b1);") == ["ld\ta,(B1)", "ld\te,a", "ld\td,0",
                                          "ld\tc,2", "call\t5"]


def test_a_byte_parameter_of_mon1_takes_e_alone():
    decls = "declare b1 byte;\nmon1: procedure (f, a) external; declare (f, a) byte; end mon1;\n"
    assert _main("call mon1(2, b1);", decls=decls) == ["ld\ta,(B1)", "ld\te,a", "ld\tc,2",
                                                       "call\t5"]


def test_mon1_with_a_function_not_constant_is_called():
    assert _main("call mon1(f, a1);") == ["ld\ta,(F)", "ld\tc,a", "ld\thl,(A1)", "ex\tde,hl", "call\tMON1"]


def test_mon1_under_mpm_calls_bdos_by_name():
    assert _main("call mon1(2, b1);", mode=Mode.MPM) == [
        "ld\ta,(B1)", "ld\te,a", "ld\td,0", "ld\tc,2", "call\t??BDOS"]


# ---- Errors ------------------------------------------------------------------

@pytest.mark.parametrize("stmt, message", [
    ("call p2aa(1);",
     "invalid number of arguments in call of P2AA, too few: 1 for 2 parameters"),
    ("call p2aa(1, 2, 3);",
     "invalid number of arguments in call of P2AA, too many: 3 for 2 parameters"),
    ("call p0(5);",
     "invalid number of arguments in call of P0, too many: 1 for 0 parameters"),
    ("a1 = fa(1);",
     "invalid number of arguments in call of FA, too many: 1 for 0 parameters"),
    ("call p1a;",
     "invalid number of arguments in call of P1A, too few: 0 for 1 parameter"),
    ("a1 = p1a + 1;",
     "invalid number of arguments in call of P1A, too few: 0 for 1 parameter"),
])
def test_a_call_passes_as_many_arguments_as_there_are_parameters(stmt, message):
    """PL/M-80 V3.1: errors 153 and 154.  With the callee taking the pushed
    arguments off the stack, a call with the wrong number of them would
    return with the stack moved."""
    errors = _errors("t: do;\n" + DECLS + stmt + "\nend t;\n")
    assert any(message in e for e in errors), errors


def test_a_private_procedure_called_with_too_many_arguments_is_an_error():
    """0.3.x dropped the extra ones."""
    errors = _errors("t: do;\ndeclare v byte;\np: procedure (x); declare x byte; v = x; end p;\n"
                     "call p(1, 2);\nend t;\n")
    assert any("in call of P, too many: 2 for 1 parameter" in e for e in errors), errors


def test_a_call_through_an_address_is_not_counted():
    """8.2.1: "the compiler does not check the number of parameters"."""
    _asm("t: do;\n" + DECLS + "q = .p2aa; call q(1, 2, 3);\nend t;\n")


def test_an_interrupt_procedure_may_not_have_parameters():
    """8.1.6: an INTERRUPT procedure is untyped and has none.  Nothing calls
    it to pass them, and its entry could not take pushed ones off."""
    errors = _errors("t: do;\nih: procedure (x) interrupt 3; declare x byte; end ih;\nend t;\n")
    assert any("IH: an INTERRUPT procedure may not have parameters (8.1.6)" in e
               for e in errors), errors
