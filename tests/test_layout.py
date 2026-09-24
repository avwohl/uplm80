"""A program is laid out as Intel's PL/M-80 lays it out: the variables last.

Code and every constant - strings, `.(...)' lists, DATA - are in the code
segment; the data segment holds ??AUTO (the procedures' shared locals), the
stack of the modes that carry one, and then the variables in the order the
source declares them.  The linker puts every data segment after all the code,
so nothing follows a program's last variable.

DRI's programs depend on that.  MP/M II's UTIL5/SUB.PLM builds SUBMIT's
command file from `.minimum$buffer', its last variable but one, up to MAXB,
and UTIL5/MSPL.PLM reads a file into `.dummy$buffer' up to MAXB.  uplm80 put
the strings, ??AUTO and the stack after the last variable, so a long command
file overwrote SUBMIT's messages and SPOOL read over its own.
"""

import os
import subprocess
import tempfile

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

from ._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing

LEVELS = (0, 1, 2, 3)


def _asm(src: str, mode: Mode = Mode.CPM, opt: int = 2) -> str:
    out = Compiler(mode=mode, opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


def _segments(asm: str) -> tuple[list[str], list[str]]:
    """The code and data segments' lines, stripped, without comments or
    directives (.z80, extrn, public)."""
    code: list[str] = []
    data: list[str] = []
    into = code
    for raw in asm.splitlines():
        line = raw.split(";")[0].strip()
        if line == "dseg":
            into = data
        elif line == "cseg":
            into = code
        elif line and not line.startswith((".", "extrn", "public")):
            into.append(line)
    return code, data


def _storage(lines: list[str]) -> list[str]:
    """The labels of what takes room, in order: every label followed by a
    db, dw or ds before the next label."""
    out = []
    label = None
    for line in lines:
        if line.endswith(":"):
            label = line[:-1]
        elif ":" in line and line.split(":")[1].strip().startswith(("ds", "db", "dw")):
            out.append(line.split(":")[0])
            label = None
        elif line.split()[0] in ("ds", "db", "dw") and label:
            out.append(label)
            label = None
    return out


SUB_SHAPE = """
t: do;
declare maxb address external;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare hello (*) byte data ('HELLO$');
putc: procedure (c); declare c byte; call mon1(2, c); end putc;
puts: procedure (s); declare s address, ch based s byte;
  do while ch <> '$'; call putc(ch); s = s + 1; end;
end puts;
count: procedure byte;
  declare n byte initial ('0');     /* static, declared before the buffer */
  n = n + 1;
  return n;
end count;
slash: procedure byte;
  declare tag (*) byte data ('/');  /* DATA inside a procedure */
  return tag(0);
end slash;
declare (p, q) address, b based p byte;
declare minimum$buffer (16) byte;
declare last$dseg$byte byte initial (0);

/* UTIL5/SUB.PLM's way: from its buffer up to MAXB is all its own */
p = .minimum$buffer;
do while p < maxb - 16;
  b = 0e5h; p = p + 1;
end;
call puts(.('OK$'));
call puts(.hello);
call putc(count); call putc(slash); call putc(count);
end t;
"""

_PAGE_ZERO = """
\t.z80
??BDOS\tequ\t5
??BOOT\tequ\t0
??MAXB\tequ\t6
MAXB\tequ\t6
\tpublic\t??BDOS, ??BOOT, ??MAXB, MAXB
\tcseg
MON1:
\tpublic\tMON1
\tld\thl,2
\tadd\thl,sp
\tld\te,(hl)
\tinc\thl
\tld\td,(hl)
\tinc\thl
\tld\tc,(hl)
\tjp\t5
\tend
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_program_may_use_everything_from_its_last_variable_up(opt):
    """MP/M mode, as SUBMIT is built: the stack is in the image, and the
    program writes from its last variable but one up to MAXB.  Its strings,
    DATA, statics, ??AUTO and stack all survive, because none of them comes
    after the variables."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    out = run_asm(_asm(SUB_SHAPE, Mode.MPM, opt), _PAGE_ZERO).stdout
    assert out.replace("\r", "").strip().endswith("OKHELLO1/2"), out


def test_the_last_variable_declared_is_the_last_thing_in_the_module():
    """Code, strings, `.(...)' constants and DATA - the module's own and a
    procedure's - are in the code segment; the data segment is ??AUTO, then
    the stack, then the variables in source order, a procedure's static in
    its place among them."""
    for mode in (Mode.CPM, Mode.MPM, Mode.BARE):
        code, data = _segments(_asm(SUB_SHAPE, mode))
        assert "HELLO:" in code and "@SLASH$TAG:" in code, (mode, code)
        assert not any(l.startswith("db\t'OK") for l in data), (mode, data)
        names = _storage(data)
        assert names[-2:] == ["MINIMUMBUFFER", "LASTDSEGBYTE"], (mode, names)
        assert names.index("@COUNT$N") < names.index("P"), (mode, names)
        if "??AUTO" in names:
            assert names.index("??AUTO") == 0, (mode, names)
        if mode != Mode.CPM:
            # the stack: `ds n' just before the ??STACK label
            assert data.index("??STACK:") < data.index("MINIMUMBUFFER:\tds\t16"), (mode, data)


def test_several_modules_compiled_together_end_with_the_last_ones_variables():
    """A multi-file compile lays out every module's variables, in order,
    after all the code."""
    with tempfile.TemporaryDirectory() as d:
        srcs = {
            "A.PLM": "a: do; declare x byte public; declare s (*) byte data ('A$');\n"
                     "f: procedure public; x = s(0); end f; call f; end a;",
            "B.PLM": "b: do; declare x byte external; f: procedure external; end f;\n"
                     "declare y (4) byte; declare tail byte; y(0) = x; tail = 1; end b;",
        }
        paths = []
        for name, text in srcs.items():
            paths.append(os.path.join(d, name))
            with open(paths[-1], "w") as fh:
                fh.write(text)
        mac = os.path.join(d, "AB.MAC")
        r = subprocess.run(compile_cmd("-o", mac, *paths), capture_output=True, text=True,
                           env=compiler_env(), check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            code, data = _segments(fh.read())
    assert _storage(data)[-3:] == ["X", "Y", "TAIL"], data
    assert any(l.startswith("S:") or l == "S:" for l in code), code


def test_memory_is_still_the_end_of_the_whole_program():
    """.MEMORY is the linker's __END__, after every segment: with the
    variables last, it is one past the last variable's last byte."""
    out = run_plm("""
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare s (*) byte data ('XY');
declare w address, last (3) byte;
w = .memory - .last;
call mon1(2, '0' + low(w));
call mon1(2, s(1));
end t;
""")
    assert out.strip().endswith("3Y"), out


MODULE_DATA_SRC = """
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare t (2) byte data (0c9h, 42h);
declare jump byte data (0c3h), jadr address data (.start-3);
start:
call mon1(2, t(1));
end t;
"""


@pytest.mark.parametrize("opt", [0, 2])
def test_a_cp_m_program_starts_with_its_entry_code_not_its_data(opt):
    """CP/M mode's program starts at 100H with its own entry code, which
    sets the stack from 0006H.  A module's DATA went ahead of it, at 100H,
    and ran: 0C9H is RET, and the program returned to CP/M before its first
    statement (the verification's f7_module_data_first). DRI's own
    `jump byte data (0c3h)' at the head, in a program compiled for CP/M,
    jumped into the middle of the four-byte entry code."""
    assert run_plm(MODULE_DATA_SRC, opt).strip().endswith("B")
    code, _ = _segments(_asm(MODULE_DATA_SRC))
    assert code[0:2] == ["ld\thl,(6)", "ld\tsp,hl"], code


def test_bare_and_mp_m_programs_keep_their_data_at_the_head():
    """DRI's layout, which BARE and MP/M modes keep: the program's first
    bytes are its DATA, the `jump byte data (0c3h)' it enters itself by."""
    for mode in (Mode.BARE, Mode.MPM):
        code, _ = _segments(_asm(MODULE_DATA_SRC, mode))
        assert code[0:3] == ["T:", "db\t0C9H", "db\t42H"], (mode, code)
