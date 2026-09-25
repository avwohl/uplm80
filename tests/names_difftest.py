"""Random PL/M-80 programs that reuse a few names at every depth, and what
they must print.

A differential test of name resolution.  Each program declares, from a pool
of ten names, variables, LITERALLYs, procedures and labels in the module, in
procedures nested three deep and in DO blocks, so that most names are
declared several times and each use means one of them by PL/M-80's rules of
scope (Programming Manual, chapter 9): a name is the innermost declaration
of it in a block around the use, from the start of that block, and may not
be used before its declaration.  The generator builds those scopes itself
and knows what every use means, and runs the program on a model of static
storage to say what it prints; the compiled program, run under cpmemu at
-O0 to -O3, has to agree.  The pool has names the assembler reads as
something else (A, P, Z, EQ, TYPE, NUL) and one that is the name of a DO
block's variables (B1), and every label is jumped to, over a statement or
out of a DO block.

:func:`generate_modules` makes a program of three modules compiled together,
each with its own declarations of the pool, two of them reached through a
PUBLIC procedure the main module declares EXTERNAL.

uplm80's macro pass puts a LITERALLY's text in place of its name even where
an inner block declares the name again (a known limitation), so no name an
enclosing block has as a LITERALLY is declared again.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import tempfile
from pathlib import Path

from ._toolchain import ToolchainError, run_asm

POOL = ["X", "Y", "Q", "P", "A", "EQ", "TYPE", "B1", "Z", "NUL"]

PRELUDE = """0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
ph: procedure (v){public};
  declare v byte, h byte;
  h = shr(v, 4) + '0'; if h > '9' then h = h + 7; call mon1(2, h);
  h = (v and 0fh) + '0'; if h > '9' then h = h + 7; call mon1(2, h);
  call mon1(2, ' ');
end ph;
"""


class _Cell:  # pylint: disable=too-few-public-methods
    """A variable's storage."""


class _Proc:  # pylint: disable=too-few-public-methods
    """A procedure: what it runs."""

    def __init__(self) -> None:
        self.body: tuple = ("block", [])
        self.ready = False          # its body is generated: callable


class _Scope:
    def __init__(self, parent: "_Scope | None") -> None:
        self.parent = parent
        # name -> ("var", cell) | ("lit", value) | ("proc", _Proc)
        #         | ("label", None) | ("pending", None): declared further on
        self.names: dict[str, tuple] = {}

    def lookup(self, name: str):
        """The innermost binding of ``name``."""
        s: _Scope | None = self
        while s is not None:
            if name in s.names:
                return s.names[name]
            s = s.parent
        return None

    def visible(self, kind: str) -> list:
        """(name, binding) for each name whose innermost binding is ``kind``."""
        seen: set[str] = set()
        out = []
        s: _Scope | None = self
        while s is not None:
            for n, d in s.names.items():
                if n not in seen:
                    seen.add(n)
                    if d[0] == kind:
                        out.append((n, d))
            s = s.parent
        return out


class _Gen:
    """One module.  Statements are (ir, lines): ir is ('set', cell, expr),
    ('print', expr), ('call', proc) or ('block', [ir...])."""

    DEPTH = 3

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.count = 0
        self.split = 0      # where the module's statements start

    def expr(self, scope: _Scope, depth: int = 0):
        """(model, text) of an expression over what is in scope."""
        r = self.rng.random()
        if depth < 2 and r < 0.3:
            a, ta = self.expr(scope, depth + 1)
            b, tb = self.expr(scope, depth + 1)
            return ("+", a, b), f"({ta} + {tb})"
        variables = scope.visible("var")
        literals = scope.visible("lit")
        if variables and r < 0.7:
            name, (_, cell) = self.rng.choice(variables)
            return ("v", cell), name
        if literals and r < 0.85:
            name, (_, value) = self.rng.choice(literals)
            return ("c", value), name
        value = self.rng.randrange(256)
        return ("c", value), str(value)

    def stmt(self, scope: _Scope, depth: int, proc):
        """(model, lines) of a statement: an assignment, a print, a call of a
        procedure in scope other than ``proc``, or a DO block."""
        self.count += 1
        r = self.rng.random()
        variables = scope.visible("var")
        procs = [(n, d) for n, d in scope.visible("proc") if d[1] is not proc and d[1].ready]
        if variables and r < 0.35:
            name, (_, cell) = self.rng.choice(variables)
            e, text = self.expr(scope)
            return ("set", cell, e), [f"{name} = {text};"]
        if procs and 0.55 <= r < 0.7:
            name, (_, p) = self.rng.choice(procs)
            return ("call", p), [f"call {name};"]
        if depth < self.DEPTH and 0.7 <= r < 0.85 and self.count < 400:
            return self.block(scope, depth + 1, proc, "do")
        e, text = self.expr(scope)
        return ("print", e), [f"call ph({text});"]

    def block(self, parent, depth: int, proc, kind: str):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """A module, a procedure's body or a DO block: its declarations, then
        its statements."""
        rng = self.rng
        scope = _Scope(parent)
        lines: list[str] = []
        free = [n for n in POOL
                if not (parent and parent.lookup(n) and parent.lookup(n)[0] == "lit")]
        rng.shuffle(free)
        labels = []
        if free and rng.random() < 0.5:
            labels.append(free.pop())
            scope.names[labels[-1]] = ("label", None)
        chosen = []
        for _ in range(rng.randint(2, 4) if kind == "module" else rng.randint(0, 3)):
            if free:
                chosen.append(free.pop())
                scope.names[chosen[-1]] = ("pending", None)
        inits = []
        for name in chosen:
            k = rng.random()
            if k < 0.5:
                cell = _Cell()
                scope.names[name] = ("var", cell)
                lines.append(f"declare {name} byte;")
                inits.append((name, cell))
            elif k < 0.7 or depth >= self.DEPTH:
                value = rng.randrange(256)
                scope.names[name] = ("lit", value)
                lines.append(f"declare {name} literally '{value}';")
            else:
                p = _Proc()
                scope.names[name] = ("proc", p)
                p.body, body = self.block(scope, depth + 1, p, "proc")
                p.ready = True
                lines += [f"{name}: procedure;"] + ["  " + l for l in body] + [f"end {name};"]
        if kind == "module":
            self.split = len(lines)
        ir: list = []
        for name, cell in inits:
            value = rng.randrange(256)
            ir.append(("set", cell, ("c", value)))
            lines.append(f"{name} = {value};")
        for _ in range(rng.randint(1, 4)):
            s, text = self.stmt(scope, depth, proc)
            ir.append(s)
            lines += text
        for label in labels:
            if rng.random() < 0.5 and depth < self.DEPTH:
                # out of a DO block to its enclosing block's label
                s1, l1 = self.stmt(scope, depth, proc)
                _, l2 = self.stmt(scope, depth, proc)
                _, l3 = self.stmt(_Scope(scope), depth + 1, proc)
                s4, l4 = self.stmt(scope, depth, proc)
                lines += (["do;"] + ["  " + l for l in l1] + [f"  goto {label};"]
                          + ["  " + l for l in l3] + ["end;"] + l2 + [f"{label}:"] + l4)
                ir += [s1, s4]
            else:
                _, l1 = self.stmt(scope, depth, proc)
                s2, l2 = self.stmt(scope, depth, proc)
                lines += [f"goto {label};"] + l1 + [f"{label}:"] + l2
                ir.append(s2)
        if kind == "do":
            return ("block", ir), ["do;"] + ["  " + l for l in lines] + ["end;"]
        return ("block", ir), lines


def _run(ir, mem: dict, out: list) -> None:
    kind = ir[0]
    if kind == "block":
        for s in ir[1]:
            _run(s, mem, out)
    elif kind == "set":
        mem[id(ir[1])] = _eval(ir[2], mem)
    elif kind == "print":
        out.append(_eval(ir[1], mem))
    elif kind == "call":
        _run(ir[1].body, mem, out)


def _eval(e, mem: dict) -> int:
    if e[0] == "c":
        return e[1] & 0xFF
    if e[0] == "v":
        return mem[id(e[1])]
    return (_eval(e[1], mem) + _eval(e[2], mem)) & 0xFF


def generate(seed: int) -> tuple[str, list[int]]:
    """(source, the bytes it prints)."""
    ir, lines = _Gen(seed).block(None, 0, None, "module")
    out: list[int] = []
    _run(ir, {}, out)
    return PRELUDE.format(public="") + "\n".join(lines) + "\nend t;\n", out


def generate_modules(seed: int, count: int = 3) -> tuple[list[str], list[int]]:
    """(the modules' sources, the bytes the program prints)."""
    rng = random.Random(seed)
    sources, bodies = [], []
    for k in range(1, count):
        gen = _Gen(seed * 100 + k)
        ir, lines = gen.block(None, 0, None, "module")
        name = f"mod{k}" if rng.random() < 0.7 else None
        text = [f"{name}: do;"] if name else []
        text += ["mon1: procedure (f, a) external; declare f byte, a address; end mon1;",
                 "ph: procedure (v) external; declare v byte; end ph;"]
        text += lines[:gen.split]
        text += [f"pub{k}: procedure public;"] + ["  " + l for l in lines[gen.split:]]
        text += [f"end pub{k};"] + ([f"end {name};"] if name else [])
        sources.append("\n".join(text) + "\n")
        bodies.append(ir)
    ir, lines = _Gen(seed).block(None, 0, None, "module")
    calls = list(range(1, count)) * 2
    rng.shuffle(calls)
    main = (PRELUDE.format(public=" public")
            + "".join(f"pub{k}: procedure external; end pub{k};\n" for k in range(1, count))
            + "\n".join(lines) + "\n"
            + "".join(f"call pub{k};\n" for k in calls) + "end t;\n")
    out: list[int] = []
    mem: dict = {}
    _run(ir, mem, out)
    for k in calls:
        _run(bodies[k - 1], mem, out)
    return [main] + sources, out


def build_and_run(sources: list[str], opt: int) -> tuple[list[int], str | None]:
    """Compile the modules together at ``-O opt``, link and run the program:
    (printed bytes, error)."""
    from uplm80.compiler import Compiler  # pylint: disable=import-outside-toplevel
    compiler = Compiler(opt_level=opt)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, text in enumerate(sources):
            paths.append(os.path.join(d, f"M{i}.PLM"))
            with open(paths[-1], "w", encoding="ascii") as fh:
                fh.write(text)
        mac = os.path.join(d, "OUT.MAC")
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                ok = compiler.compile_files([Path(p) for p in paths], Path(mac))
        except Exception as exc:  # pylint: disable=broad-except
            return [], f"compile: {type(exc).__name__}: {exc}"
        if not ok:
            return [], "compile: " + "; ".join(str(e) for e in compiler.errors.errors)
        with open(mac, encoding="ascii") as fh:
            asm = fh.read()
    try:
        r = run_asm(asm, timeout=15)
    except ToolchainError as exc:
        return [], str(exc)
    if "Program exit" not in r.stderr:
        return [], f"run: {r.stderr[-300:]}"
    try:
        return [int(w, 16) for w in r.stdout.replace("\r", "").split()], None
    except ValueError:
        return [], f"run: output {r.stdout[:200]!r}"
