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


RETURN_IN_COUNTED_LOOP = """
t: do;
declare (c, n, j) byte;
f: procedure byte external; end f;
g: procedure byte;
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
declare (c, n, j, k) byte, w address;
f: procedure byte external; end f;

/* nested counted loops, returning from the inner one */
g1: procedure byte;
    do j = 0 to 3;
        do k = 0 to n;
            if f = 0 then return 7;
        end;
    end;
    return 0;
end g1;

/* an ADDRESS result has to survive the pops */
g2: procedure address;
    do j = 0 to 5;
        w = w + 1;
        if f = 1 then return w + 1000;
    end;
    return 0;
end g2;

/* the RETURN sits in a CASE and a DO WHILE inside the counted body */
g3: procedure;
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
declare (c, n, j, k, calls) byte, w address;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;

putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;

/* returns 0 on its third call */
f: procedure byte;
    calls = calls + 1;
    if calls = 3 then return 0;
    return 1;
end f;

g1: procedure byte;
    do j = 0 to n;
        if f = 0 then return 'A';
        c = c + 1;
    end;
    return 'X';
end g1;

g2: procedure address;
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
    # g2 runs twice: w reaches 7 on the first call, and on the second it
    # goes past 7 and the loop runs out, returning 0.
    assert run_plm(RUN_SRC, opt).strip() == "AB0."
