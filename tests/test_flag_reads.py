"""Every -O level reads the flags -O0 reads.

CARRY, ZERO, SIGN and PARITY read the flags the operation before them left,
PLUS and MINUS its carry, SCL and SCR rotate through it, and DEC adjusts by
it and the half carry (Programming Manual 9800268B, 12.1 to 12.5).  The
optimizer folded `d OR 0', `d XOR 0' and `SHL(3, 2)' from -O1 on, and at
-O3 an operation of a variable whose value it knew, `c = z' of z = 0 to
`xor a', a loop it unrolled or a test it decided, so that a flag read after
one read the flags of what came before at some levels and the operation's
own at -O0 (found checking 0.4.4; 0.4.3 the same).  The optimizer leaves as
it is each operation whose flags a reader can read (uplm80/flag_flow.py),
in another module of the compilation too.
"""

import os
import subprocess
import tempfile

import pytest

from uplm80.compiler import Compiler

from ._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing

LEVELS = (0, 1, 2, 3)

_PH = """mon1: procedure (f, p) external; declare f byte, p address; end mon1;
hexd: procedure (d);
  declare d byte;
  d = d and 0fh;
  if d < 10 then call mon1(2, d + '0');
  else call mon1(2, d + 37h);
end hexd;
ph: procedure (v);
  declare v address;
  call hexd(shr(high(v), 4)); call hexd(high(v));
  call hexd(shr(low(v), 4)); call hexd(low(v));
  call mon1(2, ' ');
end ph;
"""

_X0100 = "\t.z80\nMON1\tequ\t5\n\tpublic\tMON1\n\tend\n"

# (case, statements after `b = shl(k, 1);' with k = 0FFH, d = 1, z = 0, o = 1,
# what -O0 prints and every level now prints)
_CASES = [
    ("or0", "c = d or 0; e = carry;", "0000"),
    ("xor0", "c = d xor 0; e = carry;", "0000"),
    ("shl32", "c = shl(3, 2); e = carry;", "0000"),
    ("orz", "c = d or z; e = carry;", "0000"),
    ("assign0", "c = z; e = carry;", "00FF"),
    ("if", "if o then c = 5; e = zero;", "0000"),
    ("loop", "do i = 1 to 2; c = 5; end; e = carry;", "0000"),
    ("while", "do while z; c = 5; end; e = zero;", "00FF"),
    ("case", "do case z; c = 5; c = 6; end; e = carry;", "0000"),
    ("dec", "c = dec(34h + 21h); e = c;", "0055"),
    ("inline", "call p; e = carry;", "0000"),
]


def _program() -> str:
    body = []
    for _, stmts, _ in _CASES:
        body.append("k = input(0) or 0ffh; d = (input(0) and 0) + 1; z = 0; o = 1;\n"
                    f"b = shl(k, 1); {stmts} call ph(e);")
    return ("t: do;\n" + _PH
            + "declare (k, b, c, d, e, z, o, i) byte;\n"
            + "p: procedure; c = d + z; end p;\n" + "\n".join(body) + "\nend t;\n")


@pytest.mark.parametrize("opt", LEVELS)
def test_every_level_reads_the_flags_o0_reads(opt):
    assert run_plm(_program(), opt).split() == [want for _, _, want in _CASES]


@pytest.mark.parametrize("opt", LEVELS)
def test_what_no_flag_reader_reads_is_still_folded(opt):
    """Where nothing reads the flags, `d OR 0' is d, as before."""
    src = ("t: do;\ndeclare (c, d) byte;\nc = d or 0;\nend t;\n")
    asm = Compiler(opt_level=opt).compile(src, "T.PLM")
    assert asm is not None
    lines = [line.strip() for line in asm.splitlines()]
    assert ("or\t0" in lines or "or\ta" in lines) == (opt == 0), asm


def test_a_reader_in_another_module_keeps_what_its_flags_are_of():
    """A procedure of another module of the compilation that reads the
    flags it is entered with: the operation before its call is left as it
    is, `d OR 0' before `call p' of a P whose CARRY reads its carry."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    a = ("a: do;\n" + _PH + "declare (k, b, c, d) byte, e byte public;\n"
         "p: procedure external; end p;\n"
         "k = input(0) or 0ffh; d = (input(0) and 0) + 1;\n"
         "b = shl(k, 1); c = d or 0; call p; call ph(e);\nend a;\n")
    b = "b: do;\ndeclare e byte external;\np: procedure public; e = carry; end p;\nend b;\n"
    outs = []
    for opt in LEVELS:
        with tempfile.TemporaryDirectory() as d:
            pa, pb, mac = (os.path.join(d, n) for n in ("A.PLM", "B.PLM", "AB.MAC"))
            for path, text in ((pa, a), (pb, b)):
                with open(path, "w") as fh:
                    fh.write(text)
            r = subprocess.run(compile_cmd("-O", str(opt), "-o", mac, pa, pb),
                               capture_output=True, text=True, timeout=60, env=compiler_env(),
                               check=False)
            assert r.returncode == 0, r.stderr
            with open(mac) as fh:
                outs.append(run_asm(fh.read(), _X0100).stdout.replace("\r", "").split())
    assert outs == [["0000"]] * len(LEVELS), outs
