"""Random programs of two PL/M modules, compiled apart, and one of assembly,
that call one another's procedures by PL/M-80's calling convention.

A fuzz test of the convention across the module boundary, where nothing but
the convention ties a call to the procedure it calls.  Module A and main
module B each define some of the procedures - 0 to 6 parameters, BYTE and
ADDRESS, untyped or returning either type - and declare the other's PUBLIC
ones EXTERNAL.  Some are private to their module (one of those with one
parameter and its address never taken takes it in A or HL: README, Calling
Convention), some REENTRANT, some called through an address.  Some are reached through the
assembly module: callers call PN, a routine written to the convention by
hand that reads each argument where the convention puts it, passes them on
to the PL/M procedure IN with garbage in the high byte of every BYTE, and
returns with the pushed words off the stack and every register but the
result's destroyed.  Bodies end in calls (`call p(...)' last, `return
f(...)'), which upeepz80 must not turn into jumps when words are pushed.

There is no model of what a program prints: it is built at -O0, and every
other build must print the same, the modules at the same level and at
different ones (the convention does not depend on the level of either
side), apart or in one multi-file compile.  The last word it prints is SP
at the end less SP at the start, which must be 0.

:func:`generate` makes a program; :func:`build_and_run` builds and runs it.
``tests/test_abi_fuzz.py`` runs a few seeds and ``scripts/abifuzz.py`` more.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ._toolchain import ToolchainError, run_asm

PH = """hexd: procedure (d);
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
GLOBS = ["acc", "g1", "g2", "gb1", "gb2"]


@dataclass
class Proc:  # pylint: disable=too-many-instance-attributes
    """A procedure of the program."""
    idx: int
    types: list[str]        # its parameters' types
    ret: str | None         # BYTE, ADDRESS or None
    module: str             # "A" or "B": where its body is
    public: bool            # the other module may call it
    via_asm: bool           # callers call the assembly's PN, which calls the body, IN
    reentrant: bool

    @property
    def name(self) -> str:
        """What its callers call."""
        return f"p{self.idx}"

    def callable_from(self, module: str) -> bool:
        """Whether code in ``module`` may call it."""
        return self.public or self.module == module

    def signature(self, name: str, attrs: str) -> str:
        """`name: procedure (...) type attrs;' and the parameters' declarations."""
        params = [f"a{self.idx}x{j}" for j in range(len(self.types))]
        head = (f"{name}: procedure" + (f" ({', '.join(params)})" if params else "")
                + (f" {self.ret}" if self.ret else "") + attrs + ";\n")
        return head + "".join(f"  declare {p} {t};\n" for p, t in zip(params, self.types))


@dataclass
class Program:
    """Module A, main module B and the assembly module."""
    a: str
    b: str
    asm: str


class Generator:
    """The program of one seed."""

    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)
        self.procs: list[Proc] = []

    def expr(self, scope: list[str], module: str, depth: int = 0) -> str:
        """A value: a constant, a parameter, a global, an operation, a call."""
        r = self.r
        opts = ["lit", "var", "var", "glob"]
        funcs = [p for p in self.procs if p.ret and p.callable_from(module)]
        if depth < 2:
            opts += ["bin", "bin"]
            if funcs:
                opts.append("fcall")
        k = r.choice(opts)
        if k == "lit":
            return str(r.randrange(1, 300))
        if k == "var" and scope:
            return r.choice(scope)
        if k in ("var", "glob"):
            return r.choice(GLOBS)
        if k == "bin":
            op = r.choice(["+", "-", "xor", "and", "or"])
            left = self.expr(scope, module, depth + 1)
            return f"({left} {op} {self.expr(scope, module, depth + 1)})"
        return self.call(r.choice(funcs), scope, module, depth + 1)

    def call(self, p: Proc, scope: list[str], module: str, depth: int = 0) -> str:
        """``p`` applied to arguments."""
        args = [self.expr(scope, module, max(depth, 1)) for _ in p.types]
        return f"{p.name}({', '.join(args)})" if args else p.name

    def callstmt(self, p: Proc, scope: list[str], module: str, op: str = "+") -> str:
        """A statement that calls ``p``: CALL, through an address now and
        then, or an assignment of what it returns."""
        if p.ret:
            return f"acc = acc {op} {self.call(p, scope, module)};"
        if self.r.random() < .2:
            # Each procedure has its own variable: the address is evaluated
            # after the arguments (README, Calling Convention), and a call
            # in one of them may set the variable again, which must not then
            # hold another procedure's address, of another number of
            # parameters (8.2.1: the compiler does not check it).
            q = f"q{p.idx}"
            args = [self.expr(scope, module, 1) for _ in p.types]
            call = f"{q}({', '.join(args)})" if args else q
            return f"do; {q} = .{p.name}; call {call}; end;"
        return f"call {self.call(p, scope, module)};"

    def body(self, p: Proc, params: list[str]) -> list[str]:
        """The statements of ``p``, which may call the procedures made before it."""
        r = self.r
        lines = [f"  acc = acc * 3 + {p.idx};"]
        lines += [f"  acc = acc + {pn} * {j + 2};" for j, pn in enumerate(params)]
        callees = [q for q in self.procs if q.callable_from(p.module)]
        for _ in range(r.randrange(0, 3)):
            if callees and r.random() < .6:
                q1, q2 = r.choice(callees), r.choice(callees)
                if r.random() < .5:
                    lines.append(f"  if (acc and {1 << r.randrange(8)}) <> 0 then "
                                 f"{self.callstmt(q1, params, p.module)} "
                                 f"else {self.callstmt(q2, params, p.module, 'xor')}")
                else:
                    lines.append("  " + self.callstmt(q1, params, p.module))
            else:
                v = r.choice(GLOBS)
                lines.append(f"  {v} = {v} + {self.expr(params, p.module)} + 1;")
        if p.ret is None:
            subs = [q for q in callees if q.ret is None]
            if subs and r.random() < .3:
                call = self.call(r.choice(subs), params, p.module)
                lines.append(f"  if (acc and 4) = 0 then do; call {call}; return; end;")
            if callees and r.random() < .85:
                # The last statement a call: `call x / ret', which is a jump
                # only when nothing was pushed for x.
                lines.append("  " + self.callstmt(r.choice(callees), params, p.module, "-"))
        else:
            funcs = [q for q in callees if q.ret]
            if funcs and r.random() < .8:
                lines.append(f"  return {self.call(r.choice(funcs), params, p.module)};")
            else:
                lines.append(f"  return acc + {len(params)};")
        return lines

    def proc(self, idx: int, mods: dict[str, list[str]], asm: list[str]) -> None:
        """Make procedure ``idx``: its body in its module, its EXTERNAL
        declarations where it is called from outside it, and its routine in
        the assembly if it has one."""
        r = self.r
        n = r.choice([0, 1, 1, 2, 3, 3, 4, 4, 5, 6])
        p = Proc(idx=idx, types=[r.choice(["address", "address", "byte"]) for _ in range(n)],
                 ret=r.choice([None, None, "address", "byte"]), module=r.choice("AB"),
                 public=True, via_asm=False, reentrant=r.random() < .15)
        roll = r.random()
        if roll < .3:
            p.via_asm = True
        elif roll < .5:
            p.public = False
        params = [f"a{idx}x{j}" for j in range(n)]
        body = "\n".join(self.body(p, params))
        attrs = (" reentrant" if p.reentrant else "") + (" public" if p.public else "")
        own = "i" if p.via_asm else "p"
        mods[p.module].append(p.signature(f"{own}{idx}", attrs) + body + f"\nend {own}{idx};\n")
        ext = p.signature(p.name, " external") + f"end {p.name};\n"
        if p.via_asm:
            mods["A"].append(ext)
            mods["B"].append(ext)
            asm.append(forwarder(p))
        elif p.public:
            mods["B" if p.module == "A" else "A"].append(ext)
        self.procs.append(p)

    def program(self) -> Program:
        """The three modules."""
        mods: dict[str, list[str]] = {"A": [], "B": []}
        asm: list[str] = []
        for i in range(self.r.randrange(4, 12)):
            self.proc(i, mods, asm)
        main = ["sp0 = stackptr;", "acc = 1; g1 = 7; g2 = 300; gb1 = 5; gb2 = 250;"]
        callable_b = [p for p in self.procs if p.callable_from("B")]
        for _ in range(6):
            if callable_b:
                main.append(self.callstmt(self.r.choice(callable_b), [], "B"))
            main.append("call ph(acc);")
        main.append("call ph(stackptr - sp0);")
        qs = f"declare ({', '.join(f'q{p.idx}' for p in self.procs)}) address;\n"
        a = ("ma: do;\ndeclare (acc, g1, g2) address external;\ndeclare (gb1, gb2) byte external;\n"
             + qs + "".join(mods["A"]) + "end ma;\n")
        b = ("t: do;\nmon1: procedure (f, p) external; declare f byte, p address; end mon1;\n" + PH
             + "declare (acc, g1, g2) address public;\ndeclare (gb1, gb2) byte public;\n"
             + "declare sp0 address;\n" + qs + "".join(mods["B"]) + "\n".join(main) + "\nend t;\n")
        return Program(a, b, "\t.z80\n\tcseg\n" + "".join(asm) + "\tend\n")


def forwarder(p: Proc) -> str:
    """PN in assembly, written to PL/M-80's convention: it takes its
    arguments where the convention puts them, calls IN with them, the high
    byte of each BYTE made garbage, and returns IN's result with the words
    pushed for it off the stack and every other register destroyed.

    The pushed words, a1 to a(n-2), are at SP+2(n-2) down to SP+2 at entry;
    pushing a copy of each moves the next one to the same offset from SP.
    """
    n = len(p.types)
    k = max(n - 2, 0)
    lines = [f"\tpublic\tP{p.idx}", f"\textrn\tI{p.idx}", f"P{p.idx}:"]
    for j in range(k):
        lines += [f"\tld\thl,{2 * k}", "\tadd\thl,sp", "\tld\ta,(hl)", "\tinc\thl",
                  "\tld\th,(hl)", "\tld\tl,a"]
        if p.types[j] == "byte":
            lines.append("\tld\th,0A5h")
        lines.append("\tpush\thl")
    if n >= 1 and p.types[k] == "byte":
        lines.append("\tld\tb,0A5h")            # a(n-1), or the only one, in C
    if n >= 2 and p.types[-1] == "byte":
        lines.append("\tld\td,5Ah")             # an in E
    lines += ["\tld\thl,0C3C3h", "\tld\ta,3Ch", f"\tcall\tI{p.idx}", "\tpop\tbc"]
    lines += ["\tinc\tsp\n\tinc\tsp"] * k
    lines.append("\tpush\tbc")
    if p.ret != "address":
        lines.append("\tld\thl,0E1E1h")
    if p.ret != "byte":
        lines.append("\tld\ta,1Eh")
    lines += ["\tld\tbc,0B4B4h", "\tld\tde,4B4Bh", "\tret"]
    return "\n".join(lines) + "\n"


def generate(seed: int) -> Program:
    """The program of ``seed``."""
    return Generator(seed).program()


# ---- building and running ---------------------------------------------------

def _compile(sources: list[str], opt: int) -> str:
    """The assembly of ``sources`` compiled together at ``-O opt``; raises
    ToolchainError, saying why, if the compile fails."""
    from uplm80.compiler import Compiler  # pylint: disable=import-outside-toplevel
    compiler = Compiler(opt_level=opt)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, text in enumerate(sources):
            paths.append(Path(os.path.join(d, f"M{i}.PLM")))
            paths[-1].write_text(text, encoding="ascii")
        mac = Path(os.path.join(d, "OUT.MAC"))
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                ok = (compiler.compile_file(paths[0], mac) if len(paths) == 1
                      else compiler.compile_files(paths, mac))
        except Exception as exc:  # pylint: disable=broad-except
            raise ToolchainError(f"compile: {type(exc).__name__}: {exc}") from exc
        if not ok:
            raise ToolchainError("compile: " + "; ".join(str(e) for e in compiler.errors.errors))
        return mac.read_text(encoding="ascii")


def build_and_run(prog: Program, opt_a: int, opt_b: int | None = None,
                  together: bool = False) -> tuple[str, str | None]:
    """Build ``prog`` - module A at ``-O opt_a``, B at ``-O opt_b`` (the same
    if None), or both in one multi-file compile at ``-O opt_a`` if
    ``together`` - link it with the assembly and run it: (output, error)."""
    try:
        if together:
            main, extra = _compile([prog.b, prog.a], opt_a), [prog.asm]
        else:
            main = _compile([prog.b], opt_a if opt_b is None else opt_b)
            extra = [_compile([prog.a], opt_a), prog.asm]
        r = run_asm(main, extra, timeout=15)
    except ToolchainError as exc:
        return "", str(exc)
    if "Program exit" not in r.stderr:
        return "", f"run: {r.stderr[-300:]}"
    return r.stdout.replace("\r", "").replace("\0", ""), None


def builds(levels: tuple[int, ...] = (0, 1, 2, 3), mixed: bool = True,
           together: bool = True) -> list[tuple[str, dict]]:
    """The builds to compare with -O0's: (name, build_and_run's keywords)."""
    out = [(f"-O{o}", {"opt_a": o}) for o in levels if o]
    if mixed:
        hi = max(levels)
        out += [(f"A -O{hi} B -O0", {"opt_a": hi, "opt_b": 0}),
                (f"A -O0 B -O{hi}", {"opt_a": 0, "opt_b": hi})]
    if together:
        out.append((f"one compile -O{max(levels)}", {"opt_a": max(levels), "together": True}))
    return out


def check(seed: int, **kw) -> list[str]:
    """What is wrong with the program of ``seed``: its -O0 build does not
    build or run or leaves SP moved, or another build prints something else.
    ``kw`` goes to :func:`builds`."""
    prog = generate(seed)
    base, err = build_and_run(prog, 0)
    if err:
        return [f"-O0: {err.strip().splitlines()[-1][:300] if err.strip() else err}"]
    bad = []
    words = base.split()
    if not words or words[-1] != "0000":
        bad.append(f"-O0: SP moved by {words[-1] if words else '?'}: {base!r}")
    for name, args in builds(**kw):
        got, err = build_and_run(prog, **args)
        if err:
            bad.append(f"{name}: {err.strip().splitlines()[-1][:300] if err.strip() else err}")
        elif got != base:
            bad.append(f"{name}: printed {got!r}, -O0 {base!r}")
    return bad
