"""A GOTO out of a procedure leaves nothing on the stack.

PL/M-80 allows a GOTO out of a procedure to a label at the outer level of
the main program (Programming Manual 9800268B, 8.1.3 and 9.3).  The calls it
abandons have pushed their return addresses, and DRI's PL/M-80 sets SP
again at such a label: in MP/M II's PIP.PRL, procedure ERROR ends
`CALL CRLF / JMP RETRY', and RETRY: begins `LXI SP,2251', the stack
PIPENTRY-3 sets.  A label only the main program jumps to gets no `LXI SP'
(SIMPLECOM, ENDCOM).  uplm80 only jumped, so every such GOTO left on the
stack what the calls on the way had pushed: MP/M II's PIP, retrying after
an error, printed garbage and dropped back to the CLI after 86 errors in
one session, and a `--mode bare' program, with its 64-byte stack, failed
after about a dozen.
"""

import os
import subprocess
import tempfile

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

from ._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing

LEVELS = (0, 1, 2, 3)

PRELUDE = """0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
pc: procedure (c); declare c byte; call mon1(2, c); end pc;
pn: procedure (n); declare n address;
    call pc('0' + n / 10000); call pc('0' + n / 1000 mod 10);
    call pc('0' + n / 100 mod 10); call pc('0' + n / 10 mod 10); call pc('0' + n mod 10);
end pn;
"""


def _retries(times: int) -> str:
    """A GOTO out of two calls, `times' times, then the count and a '.'."""
    return PRELUDE + f"""
declare (n, done) address;
bail: procedure;
    n = n + 1;
    goto again;
end bail;
check: procedure;
    declare k address;
    k = n;
    if k < {times} then call bail;
    done = 1;
end check;
n = 0; done = 0;
again:
    if done = 0 then call check;
    call pn(n); call pc('.');
    call mon1(0, 0);
end t;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_out_of_a_procedure_in_bare_mode(opt):
    """1000 GOTOs, each out of two calls: 4000 bytes the 64-byte stack of
    `--mode bare' does not have."""
    assert run_plm(_retries(1000), opt, mode="bare") == "01000."


@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_out_of_a_procedure_in_cpm_mode(opt):
    """In CP/M mode the stack is all the memory under the BDOS, and 30000
    GOTOs out of two calls take twice that."""
    assert run_plm(_retries(30000), opt) == "30000."


SRC = PRELUDE + """
declare n byte;
bail: procedure; n = n + 1; goto again; end bail;
n = 0;
again:
    if n < 3 then call bail;
    if n = 3 then goto last;
    n = 9;
last:
    call pc('0' + n);
end t;
"""


def _lines(asm: str) -> list[str]:
    return [" ".join(x.split(";")[0].split()) for x in asm.splitlines()]


@pytest.mark.parametrize("mode, reload", [
    (Mode.BARE, ["ld sp,??STACK"]),
    (Mode.MPM, ["ld sp,??STACK"]),
    (Mode.CPM, ["ld hl,(6)", "ld sp,hl"]),
])
def test_the_label_a_procedure_jumps_to_sets_sp_and_no_other(mode, reload):
    """As DRI's PIP.PRL has it: AGAIN, where BAIL jumps, begins by setting SP
    the way the program's first statement does, with the stack of the mode;
    LAST, which only the main program jumps to, does not."""
    lines = _lines(Compiler(mode=mode, opt_level=0).compile(SRC, "<test>"))
    again = lines.index("AGAIN:")
    assert lines[again + 1:again + 1 + len(reload)] == reload, lines[again:again + 4]
    last = lines.index("LAST:")
    assert not lines[last + 1].startswith("ld sp") and lines[last + 1] != reload[0]
    assert sum(1 for x in lines if x in ("ld sp,??STACK", "ld sp,hl")) == 2


PUBLIC_SRC = PRELUDE + """
declare again label public;
declare n address;
bail: procedure external; end bail;
count: procedure public; n = n + 1; end count;
done: procedure byte public; return n >= 500; end done;
n = 0;
again:
    if not done then call bail;
    call pn(n); call pc('.');
    call mon1(0, 0);
end t;
"""

# BAIL, in another module: counts, and jumps to AGAIN from two calls down.
BAIL = """\t.z80
\tpublic\tBAIL
\textrn\tAGAIN, COUNT
\tcseg
BAIL:\tcall\tBAIL2
BAIL2:\tcall\tCOUNT
\tjp\tAGAIN
\tend
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_public_label_another_module_jumps_to(opt):
    """A PUBLIC label may be where a GOTO in another module's procedure goes,
    which the module defining it, compiled apart, cannot see."""
    assert run_plm(PUBLIC_SRC, opt, extra_asm=BAIL, mode="bare") == "00500."


MAIN = PRELUDE + """
declare again label public;
declare n address public;
bail: procedure external; end bail;
n = 0;
again:
    if n < 500 then call bail;
    call pn(n); call pc('.');
    call mon1(0, 0);
end t;
"""

LIB = """lib: do;
declare again label external;
declare n address external;
bail: procedure public; call deeper; end bail;
deeper: procedure; n = n + 1; goto again; end deeper;
end lib;
"""


@pytest.mark.parametrize("opt", (0, 3))
def test_a_goto_from_another_modules_procedure_in_one_compile(opt):
    """The same in a multi-file compile, where the GOTO in LIB's procedure
    names LIB's EXTERNAL declaration of MAIN's label."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, text in (("MAIN.PLM", MAIN), ("LIB.PLM", LIB)):
            paths.append(os.path.join(d, name))
            with open(paths[-1], "w") as fh:
                fh.write(text)
        mac = os.path.join(d, "T.MAC")
        r = subprocess.run(compile_cmd("-O", str(opt), "--mode", "bare", "-o", mac, *paths),
                           capture_output=True, text=True, timeout=60, env=compiler_env(),
                           check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            asm = fh.read()
    assert run_asm(asm).stdout.replace("\r", "") == "00500."
