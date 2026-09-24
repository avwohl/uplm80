"""What a loop leaves on the stack across its body has to be gone at every exit.

`DO i = 0 TO n' whose body never reads i is compiled as a B-register count,
and B is pushed around the body because the body is free to use it.  A RETURN
inside that body returned with the count still on the stack, so the RET took
the count as its return address.

The static check below follows every path through the emitted procedure and
tracks how many words it has pushed; the run test executes the same shapes
under a CP/M emulator.
"""

import re

import pytest

from uplm80.compiler import Compiler

from ._toolchain import run_plm


def _asm(src: str, opt: int) -> str:
    out = Compiler(opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


def _instructions(asm: str) -> list[str]:
    out = []
    for line in asm.splitlines():
        line = line.split(";")[0].strip()
        if line:
            out.append(" ".join(line.split()))
    return out


_JUMP = re.compile(r"^(jp|jr|djnz)\s+(?:(nz|z|nc|c|po|pe|p|m),)?\s*([^\s,()]+)$", re.I)


def _stack_depth_at_returns(asm: str, proc: str) -> list[tuple[int, int]]:
    """(instruction index, words pushed) at every RET reachable in `proc'.

    Follows jumps, conditional jumps, DJNZ and fall-through from the
    procedure's label; a label that is reached with two different depths is
    reported as a RET at depth -1, since that is as wrong.
    """
    ins = _instructions(asm)
    labels = {x[:-1].upper(): i for i, x in enumerate(ins) if x.endswith(":")}
    start = labels[proc.upper()]
    seen: dict[int, int] = {}
    work = [(start, 0)]
    rets = []
    while work:
        i, depth = work.pop()
        while i < len(ins):
            if i in seen:
                if seen[i] != depth:
                    rets.append((i, -1))
                break
            seen[i] = depth
            op = ins[i].lower()
            if op.startswith("push "):
                depth += 1
            elif op.startswith("pop "):
                depth -= 1
            elif op == "ret" or op == "reti":
                rets.append((i, depth))
                break
            elif re.match(r"^ret\s+\w+$", op):
                rets.append((i, depth))
            elif op.startswith("jp (") or op.startswith("jp\t("):
                break
            else:
                m = _JUMP.match(op)
                if m:
                    target = labels.get(m.group(3).upper())
                    conditional = m.group(2) is not None or m.group(1) == "djnz"
                    if target is not None:
                        work.append((target, depth))
                    if not conditional:
                        break
            i += 1
    return rets


def _counted(asm: str) -> bool:
    """Whether the loop kept its count in B - pushed around the body."""
    return "push\tbc" in [l.strip() for l in asm.splitlines()]


# The reported case had j at module level.  A RETURN leaves such an index
# where the caller can read it, so that loop is no longer counted at all (see
# test_an_index_the_caller_can_read_after_a_return_is_not_counted); with j
# local to g the count is still the fastest form, and still has to be popped.
RETURN_IN_COUNTED_LOOP = """
t: do;
declare (c, n) byte;
f: procedure byte external; end f;
g: procedure byte;
    declare j byte;
    do j = 0 to n;
        if f = 0 then return 1;
        c = c + 1;
    end;
    return 0;
end g;
c = g;
end t;
"""


@pytest.mark.parametrize("opt", [0, 2, 3])
def test_a_return_inside_a_counted_loop_leaves_the_count_behind(opt):
    """The reported case: `push bc / call F / ... / ld a,1 / ret'."""
    asm = _asm(RETURN_IN_COUNTED_LOOP, opt)
    assert "push\tbc" in asm, "this no longer takes the counted-loop path"
    bad = [r for r in _stack_depth_at_returns(asm, "G") if r[1] != 0]
    assert not bad, f"-O{opt}: RET with words still pushed {bad}:\n{asm}"


SHAPES = """
t: do;
declare (c, n) byte, w address;
f: procedure byte external; end f;

/* nested counted loops, returning from the inner one */
g1: procedure byte;
    declare (j, k) byte;
    do j = 0 to 3;
        do k = 0 to n;
            if f = 0 then return 7;
        end;
    end;
    return 0;
end g1;

/* an ADDRESS result has to survive the pops */
g2: procedure address;
    declare j byte;
    do j = 0 to 5;
        w = w + 1;
        if f = 1 then return w + 1000;
    end;
    return 0;
end g2;

/* the RETURN sits in a CASE and a DO WHILE inside the counted body */
g3: procedure;
    declare j byte;
    do j = 0 to 9;
        do case f;
            c = 1;
            return;
            do while f;
                if c = 2 then return;
                c = c + 1;
            end;
        end;
    end;
end g3;

/* a loop that falls out normally, and an untyped RETURN after it */
g4: procedure;
    declare j byte;
    do j = 0 to 9;
        c = c + 1;
    end;
    return;
end g4;

c = g1;
w = g2;
call g3;
call g4;
end t;
"""


@pytest.mark.parametrize("opt", [0, 2, 3])
@pytest.mark.parametrize("proc", ["G1", "G2", "G3", "G4"])
def test_every_return_in_a_procedure_leaves_the_stack_as_it_found_it(opt, proc):
    asm = _asm(SHAPES, opt)
    rets = _stack_depth_at_returns(asm, proc)
    assert rets, f"no RET found in {proc}"
    bad = [r for r in rets if r[1] != 0]
    assert not bad, f"-O{opt} {proc}: RET with words still pushed {bad}:\n{asm}"


RUN_SRC = """
0100H:
t: do;
declare (c, n, calls) byte, w address;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;

putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;

/* returns 0 on its third call */
f: procedure byte;
    calls = calls + 1;
    if calls = 3 then return 0;
    return 1;
end f;

g1: procedure byte;
    declare j byte;
    do j = 0 to n;
        if f = 0 then return 'A';
        c = c + 1;
    end;
    return 'X';
end g1;

g2: procedure address;
    declare (j, k) byte;
    do j = 0 to 3;
        do k = 0 to 4;
            w = w + 1;
            if w = 7 then return 'B' + 256;
        end;
    end;
    return 0;
end g2;

n = 9;
call putc(g1);
call putc(low(g2));
call putc(high(g2) + '0');
call putc('.');
end t;
"""


@pytest.mark.parametrize("opt", [0, 2])
def test_a_return_from_inside_a_counted_loop_runs(opt):
    """Compile, assemble, link and run.  With the count left on the stack the
    first RET jumps to the count and the program prints nothing sensible."""
    assert _counted(_asm(RUN_SRC, opt)), "this no longer takes the counted-loop path"
    # g2 runs twice: w reaches 7 on the first call, and on the second it
    # goes past 7 and the loop runs out, returning 0.
    assert run_plm(RUN_SRC, opt).strip() == "AB0."


@pytest.mark.parametrize("body", [
    # read in the subscript of a member - UTIL2/SCBRS.PLM clears its table so
    "s(i).x = 0;",
    # written, to end the loop early - UTIL6/PIP.PLM and ED.PLM stop reading
    # a file that way at end of file
    "if c = 3 then i = 9;",
    # the index of an inner loop
    "do i = 0 to 1; end;",
])
def test_a_body_that_uses_its_index_is_not_a_counted_loop(body):
    """The count in B stands in for the index, which is never stored, so it
    is only right when the body neither reads nor writes the index."""
    asm = _asm(f"""
t: do;
declare (i, c) byte, s (10) structure (x byte, y byte);
do i = 0 to 9; c = c + 1; {body} end;
end t;
""", 2)
    assert not _counted(asm), asm


LOOP_RUN_SRC = """
0100H:
t: do;
declare (i, j, n, c) byte, w address;
declare s (4) structure (x byte, y byte);
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;

do i = 0 to 3; s(i).x = 'K'; end;
do i = 0 to 3; call putc(s(i).x); end;

c = 0;
do i = 0 to 9; c = c + 1; if c = 3 then i = 9; end;
call putc('0' + c);

n = 255; w = 0;
do j = 0 to n; w = w + 1; end;
call putc('0' + high(w));
call putc('0' + low(w));
end t;
"""


@pytest.mark.parametrize("opt", [0, 2])
def test_counted_loops_run_the_right_number_of_times(opt):
    """`DO j = 0 TO n' runs n+1 times, 256 when n is 255 (PL/M-80 manual,
    5.1.4: the loop ends when the index wraps).  The counted form skipped the
    loop entirely for n = 255, because a count of 256 is 0 in B - which is
    exactly what DJNZ counts down from.  It also cleared only s(0) and ran
    the early-exit loop all ten times."""
    assert run_plm(LOOP_RUN_SRC, opt).strip() == "KKKK310"


LOOP_INDEX_SRC = """
0100H:
t: do;
declare (i, j, n, c, calls) byte;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
show: procedure; call putc('0' + i); end show;
/* returns 0 on its third call */
f: procedure byte;
    calls = calls + 1;
    if calls = 3 then return 0;
    return 1;
end f;
/* j is the module's: whoever called g can read it */
g: procedure byte;
    do j = 0 to 9;
        if f = 0 then return 1;
    end;
    return 0;
end g;

/* an inner loop over the same index leaves it at 10; the outer one then
   steps it to 11, past its bound, and stops after one pass */
c = 0;
do i = 0 to 9; c = c + 1; do i = 0 to 9; end; end;
call putc('0' + c); call putc('0' + i);
call putc('.');

/* a procedure the body calls reads the index */
do i = 0 to 3; call show; end;
n = 3; do i = 0 to n; call show; end;
call putc('.');

/* the index after the loop is one past the bound */
do i = 0 to 3; c = c + 1; end; call putc('0' + i);
n = 5; do i = 0 to n; c = c + 1; end; call putc('0' + i);
call putc('.');

/* the bound is looked at every time round */
n = 4; c = 0; do i = 0 to n; c = c + 1; n = 0; end; call putc('0' + c);
call putc('.');

/* where a RETURN left the index */
c = g; call putc('0' + j);
call putc('.');
end t;
"""


@pytest.mark.parametrize("opt", [0, 2, 3])
def test_a_loop_index_is_where_pl_m_puts_it(opt):
    """Inside the loop, after it, and after a RETURN from it, the index has
    the value PL/M-80 gives it (manual, 5.1.4): the start, stepped once per
    pass, and one step past the bound when the loop ends.

    The counted form kept the count in B and never stored the index, so an
    inner loop over the same index, a procedure the body calls, the code
    after the loop, and a caller after a RETURN all saw a stale value; and
    it counted a bound once that PL/M-80 looks at every time round."""
    assert run_plm(LOOP_INDEX_SRC, opt).strip() == "1;.01230123.46.1.2.", opt


LOOP_255_SRC = """
0100H:
t: do;
declare (j, n) byte, w address;
declare arr (256) byte;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;

/* to 255 is 256 times, and the index wraps to 0 */
w = 0; do j = 0 to 255; w = w + 1; end;
call putc('0' + high(w)); call putc('0' + low(w)); call putc('0' + j);
w = 0; do j = 0 to 255; arr(j) = 1; w = w + 1; end;
call putc('0' + high(w)); call putc('0' + low(w)); call putc('0' + j);
n = 255; w = 0; do j = 0 to n; arr(j) = 1; w = w + 1; end;
call putc('0' + high(w)); call putc('0' + low(w)); call putc('0' + j);
n = 255; w = 0; do j = 0 to n; w = w + 1; end;
call putc('0' + high(w)); call putc('0' + low(w)); call putc('0' + j);
w = 0; do j = 250 to 255; w = w + j; end;
call putc('0' + high(w)); call putc('0' + j);
w = 0; do j = 1 to 255 by 2; w = w + 1; end;
call putc('0' + w - 120); call putc('0' + j);
call putc('.');
end t;
"""


@pytest.mark.parametrize("opt", [0, 2, 3])
def test_a_byte_loop_to_255_runs_256_times(opt):
    """PL/M-80 manual, 5.1.4: the loop ends when stepping the index carries
    out of the byte, which passes any bound, so `DO j = 0 TO 255' runs 256
    times and leaves j at 0.  The test was `j < bound + 1', and bound + 1 is
    0: a constant bound of 255, and a variable one that is 255 in a loop
    that reads its index, ran the loop no times at all.  At -O3 constant
    propagation turns `n = 255; DO j = 0 TO n' into the constant form.
    Stepping BY 2 from 255 carries too."""
    assert run_plm(LOOP_255_SRC, opt).strip() == "1001001001005081.", opt


def test_an_index_a_called_procedure_reads_is_not_counted():
    asm = _asm("""
t: do;
declare (i, c) byte;
show: procedure; c = c + i; end show;
do i = 0 to 9; call show; end;
end t;
""", 2)
    assert not _counted(asm), asm


def test_an_index_the_caller_can_read_after_a_return_is_not_counted():
    asm = _asm(RETURN_IN_COUNTED_LOOP.replace("    declare j byte;\n", "")
               .replace("declare (c, n) byte;", "declare (c, n, j) byte;"), 2)
    assert not _counted(asm), asm


@pytest.mark.parametrize("src", [
    # a local index, whatever the body calls
    "g: procedure; declare j byte; do j = 0 to 9; call h; end; end g;",
    # a module-level index no called procedure names
    "g: procedure; do k = 0 to 9; call h; end; end g;",
])
def test_an_index_nothing_else_can_see_is_still_counted(src):
    asm = _asm(f"""
t: do;
declare (c, k) byte;
h: procedure; c = c + 1; end h;
{src}
call g;
end t;
""", 2)
    assert _counted(asm), asm
