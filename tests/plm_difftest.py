"""Random PL/M-80 programs, and what Intel's rules say they print.

A differential test of expression typing. Each generated program assigns,
computes and prints BYTE and ADDRESS values; a Python model of the PL/M-80
Programming Manual's rules computes what it must print, and the compiled
program, run under cpmemu at -O0 to -O3, has to agree with the model at every
level. The rules modelled are:

* A numeric constant up to 255 is a BYTE, a larger one an ADDRESS; a
  one-character string is a BYTE constant (4.1.1).
* ``+ - AND OR XOR`` of two BYTEs are 8-bit and wrap, and so are unary ``-``
  and ``NOT`` of a BYTE (4.2.1, 4.2.2, 4.3); with an ADDRESS operand the BYTE
  is zero-extended and the operation is 16-bit.
* ``* / MOD`` are always ADDRESS (4.2.3, 4.2.4); the quotient and remainder
  are DRI's (@P0029's, via :func:`tests.test_divmod_dri.dri_divide`), zero
  divisor included.
* A relation is the BYTE 0FFH or 00H (4.4); a condition is its bit 0.
* LOW, HIGH and ROL/ROR are BYTE; DOUBLE and SIZE are ADDRESS; LENGTH and
  LAST are BYTE when the value fits (11.1). SHL and SHR are modelled as
  uplm80 has them, 16-bit with an ADDRESS result even of a BYTE pattern,
  where the manual (11.1.4) and DRI's compiler shift a BYTE in 8 bits: see
  uplm80.plm_types.
* An embedded assignment has the value, and type, of its right half (4.6.3).
* A value is converted to the type it is stored in, passed as or returned as
  by truncation or zero extension (4.6.1, 8.x).
* ``DO i = s TO l BY k`` assigns ``s``, runs while ``i <= l``, and stops when
  adding the step carries out of the index's width (5.1.4; DRI's code tests
  the limit at the top and exits on the carry of the increment).

The generator keeps each statement's value independent of evaluation order,
which PL/M-80 leaves undefined. Every function call counts itself in ``nc``,
which no expression reads; ``hb`` and ``hw`` also change ``b3`` and ``w4``,
which later statements read, and a statement that calls one of them calls
only that one and does not otherwise read or assign its variable. An
embedded assignment targets a variable its own statement does not read. The
call count is printed, so a call that an optimization drops or duplicates
shows, and so is what the calls changed. PLUS and
MINUS are left out: their carry-in is whatever the preceding instruction
left, which the manual (12.1) says cannot be relied on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ._toolchain import ToolchainError, run_asm

B, A = "BYTE", "ADDRESS"
MASK = {B: 0xFF, A: 0xFFFF}


def lit_type(v: int) -> str:
    return B if v <= 0xFF else A


def conv(v: int, t: str) -> int:
    return v & MASK[t]


def _divide(dividend: int, divisor: int) -> tuple[int, int]:
    # Imported late so this module can be used without pytest's rootdir on
    # the path (scripts/difftest.py adds it).
    from tests.test_divmod_dri import dri_divide
    return dri_divide(dividend, divisor)


_DIV_CACHE: dict[tuple[int, int], tuple[int, int]] = {}


def divide(dividend: int, divisor: int) -> tuple[int, int]:
    key = (dividend, divisor)
    if key not in _DIV_CACHE:
        _DIV_CACHE[key] = _divide(dividend, divisor)
    return _DIV_CACHE[key]


@dataclass
class Env:
    """Run-time state of the modelled program."""
    vals: dict[str, int] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    arrays: dict[str, tuple[str, list[int]]] = field(default_factory=dict)

    def store(self, name: str, value: int) -> None:
        self.vals[name] = conv(value, self.types[name])


# ---- expressions -----------------------------------------------------------

class Expr:
    def text(self) -> str:
        raise NotImplementedError

    def ev(self, env: Env) -> tuple[int, str]:
        raise NotImplementedError

    def reads(self) -> set[str]:
        return set()

    def assigns(self) -> set[str]:
        return set()

    def is_const(self) -> bool:
        """No variable, element or call: a constant."""
        return False

    def foldable(self) -> bool:
        """A constant uplm80 folds at every level (LENGTH, LAST and SIZE it
        does not)."""
        return self.is_const()


def _hex(v: int) -> str:
    return f"0{v:x}h"


@dataclass
class Num(Expr):
    v: int
    style: str = "dec"

    def text(self) -> str:
        if self.style == "str":
            return f"'{chr(self.v)}'"
        return _hex(self.v) if self.style == "hex" else str(self.v)

    def ev(self, env):
        return self.v, lit_type(self.v)

    def is_const(self):
        return True


@dataclass
class Var(Expr):
    name: str

    def text(self):
        return self.name

    def ev(self, env):
        return env.vals[self.name], env.types[self.name]

    def reads(self):
        return {self.name}


@dataclass
class Elem(Expr):
    array: str
    index: Expr

    def text(self):
        return f"{self.array}(({self.index.text()}) and 7)"

    def ev(self, env):
        t, values = env.arrays[self.array]
        i, _ = self.index.ev(env)
        return values[i & 7], t

    def reads(self):
        return self.index.reads()

    def assigns(self):
        return self.index.assigns()


_REL = {"=": lambda a, b: a == b, "<>": lambda a, b: a != b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


@dataclass
class Bin(Expr):
    op: str
    left: Expr
    right: Expr

    def text(self):
        return f"({self.left.text()} {self.op} {self.right.text()})"

    def ev(self, env):
        lv, lt = self.left.ev(env)
        rv, rt = self.right.ev(env)
        op = self.op
        if op in _REL:
            return (0xFF if _REL[op](lv, rv) else 0), B
        if op == "*":
            return (lv * rv) & 0xFFFF, A
        if op in ("/", "mod"):
            q, r = divide(lv, rv)
            return (q if op == "/" else r), A
        t = B if lt == B and rt == B else A
        if op == "+":
            v = lv + rv
        elif op == "-":
            v = lv - rv
        elif op == "and":
            v = lv & rv
        elif op == "or":
            v = lv | rv
        else:
            v = lv ^ rv
        return conv(v, t), t

    def reads(self):
        return self.left.reads() | self.right.reads()

    def assigns(self):
        return self.left.assigns() | self.right.assigns()

    def is_const(self):
        return self.left.is_const() and self.right.is_const()

    def foldable(self):
        return self.left.foldable() and self.right.foldable()


@dataclass
class Un(Expr):
    op: str             # "-" or "not"
    operand: Expr

    def text(self):
        return f"({self.op} ({self.operand.text()}))" if self.op == "not" \
            else f"(-({self.operand.text()}))"

    def ev(self, env):
        v, t = self.operand.ev(env)
        return conv(-v if self.op == "-" else ~v, t), t

    def reads(self):
        return self.operand.reads()

    def assigns(self):
        return self.operand.assigns()

    def is_const(self):
        return self.operand.is_const()

    def foldable(self):
        return self.operand.foldable()


@dataclass
class Builtin(Expr):
    name: str
    args: list

    def text(self):
        return f"{self.name}({', '.join(a.text() for a in self.args)})"

    def ev(self, env):
        n = self.name
        if n in ("length", "last", "size"):
            t, values = env.arrays[self.args[0].name]
            v = {"length": len(values), "last": len(values) - 1,
                 "size": len(values) * (1 if t == B else 2)}[n]
            return v, (A if n == "size" else lit_type(v))
        x, xt = self.args[0].ev(env)
        if n == "low":
            return x & 0xFF, B
        if n == "high":
            return (x >> 8 if xt == A else 0), B
        if n == "double":
            return x, A
        c, _ = self.args[1].ev(env)
        c &= 0xFF
        if n == "shl":
            return (x << c) & 0xFFFF, A
        if n == "shr":
            return x >> c, A
        x &= 0xFF
        c &= 7
        if n == "rol":
            return ((x << c) | (x >> (8 - c))) & 0xFF, B
        return ((x >> c) | (x << (8 - c))) & 0xFF, B     # ror

    def reads(self):
        return set().union(*(a.reads() for a in self.args)) if self.name not in (
            "length", "last", "size") else set()

    def assigns(self):
        return set().union(*(a.assigns() for a in self.args))

    def is_const(self):
        if self.name in ("length", "last", "size"):
            return True
        return all(a.is_const() for a in self.args)

    def foldable(self):
        if self.name in ("length", "last", "size"):
            return False
        return all(a.foldable() for a in self.args)


# The typed procedures every program declares. Each counts its call in nc.
FUNCS = {
    "fb": (B, B, lambda p: p ^ 0x5A),
    "fw": (A, A, lambda p: (p + 0x1234) & 0xFFFF),
    "gb": (None, B, lambda p: 0xC3),
    "gw": (None, A, lambda p: 0x8421),
    "hb": (None, B, None),
    "hw": (None, A, None),
}
# Functions that also change a variable the program reads, and return its
# new value. A statement calls at most one of them, and then does not read
# or assign the variable itself, so its value does not depend on the order
# of evaluation.
SIDE_EFFECTS = {"hb": ("b3", 7), "hw": ("w4", 0x105)}


@dataclass
class FCall(Expr):
    name: str
    arg: Expr | None = None

    def text(self):
        return self.name if self.arg is None else f"{self.name}({self.arg.text()})"

    def ev(self, env):
        ptype, rtype, fn = FUNCS[self.name]
        p = 0
        if self.arg is not None:
            p = conv(self.arg.ev(env)[0], ptype)
        env.vals["nc"] = (env.vals["nc"] + 1) & 0xFFFF
        if self.name in SIDE_EFFECTS:
            var, delta = SIDE_EFFECTS[self.name]
            env.store(var, env.vals[var] + delta)
            return env.vals[var], rtype
        return fn(p), rtype

    def reads(self):
        return self.arg.reads() if self.arg is not None else set()

    def assigns(self):
        return self.arg.assigns() if self.arg is not None else set()


@dataclass
class EAsg(Expr):
    target: str
    value: Expr

    def text(self):
        return f"({self.target} := {self.value.text()})"

    def ev(self, env):
        v, t = self.value.ev(env)
        env.store(self.target, v)
        return v, t

    def reads(self):
        return self.value.reads()

    def assigns(self):
        return {self.target} | self.value.assigns()


def calls_in(e: Expr) -> list[str]:
    """The names of the functions ``e`` calls, once per call."""
    out: list[str] = []
    stack: list = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, FCall):
            out.append(n.name)
        if isinstance(n, list):
            stack.extend(n)
        elif isinstance(n, Expr):
            stack.extend(getattr(n, f) for f in n.__dataclass_fields__)
    return out


# ---- the generator ---------------------------------------------------------

INTERESTING = [0, 1, 2, 3, 5, 7, 8, 15, 16, 0x7F, 0x80, 0x81, 0xFE, 0xFF,
               0x100, 0x101, 0x1FF, 0x3E8, 0x7FFF, 0x8000, 0x8001, 0xFF00,
               0xFFFE, 0xFFFF]
BYTE_VARS = ["b1", "b2", "b3", "b4"]
ADDR_VARS = ["w1", "w2", "w3", "w4"]
LOCAL_VARS = {"lb": B, "lw": A}
ARRAYS = {"ab": B, "aw": A}
BIG = 300           # extent of `big', so LENGTH/LAST are ADDRESS


class Generator:
    """Builds one random program and the list of values it must print."""

    def __init__(self, seed: int, n_stmts: int = 60) -> None:
        self.rnd = random.Random(seed)
        self.n_stmts = n_stmts
        self.env = Env()
        for v in BYTE_VARS + ["rb", "eb", "ib"]:
            self.env.types[v] = B
        for v in ADDR_VARS + ["rw", "ew", "iw", "nc", "n", "acc"]:
            self.env.types[v] = A
        self.env.types.update(LOCAL_VARS)
        for v in self.env.types:
            self.env.vals[v] = 0
        self.env.arrays["ab"] = (B, [self._const_value(B) for _ in range(8)])
        self.env.arrays["aw"] = (A, [self._const_value(A) for _ in range(8)])
        self.env.arrays["big"] = (B, [0] * BIG)
        self.lines: list[str] = []
        self.expect: list[tuple[str, int]] = []

    # -- values and leaves

    def _const_value(self, t: str | None = None) -> int:
        r = self.rnd
        if r.random() < 0.7:
            v = r.choice(INTERESTING)
        else:
            v = r.randrange(0x10000) if r.random() < 0.5 else r.randrange(0x100)
        if t is not None:
            v = conv(v, t)
        return v

    def _num(self) -> Num:
        v = self._const_value()
        r = self.rnd.random()
        if v <= 0xFF and 0x41 <= v <= 0x5A and r < 0.3:
            return Num(v, "str")
        return Num(v, "hex" if r < 0.6 else "dec")

    def _leaf(self, allow_calls: bool) -> Expr:
        r = self.rnd.random()
        if r < 0.3:
            return self._num()
        if r < 0.75:
            return Var(self.rnd.choice(BYTE_VARS + ADDR_VARS + list(LOCAL_VARS)))
        if r < 0.85:
            return Elem(self.rnd.choice(list(ARRAYS)), self._leaf(False))
        if r < 0.9:
            name = self.rnd.choice(["length", "last", "size"])
            return Builtin(name, [Var(self.rnd.choice(["ab", "aw", "big"]))])
        if allow_calls:
            name = self.rnd.choice(list(FUNCS))
            if FUNCS[name][0] is None:
                return FCall(name)
            return FCall(name, self.expr(1, allow_calls=False))
        return self._num()

    # -- expressions

    def expr(self, depth: int, allow_calls: bool = True) -> Expr:
        r = self.rnd
        if depth <= 0 or r.random() < 0.2:
            return self._leaf(allow_calls)
        k = r.random()
        if k < 0.55:
            op = r.choice(["+", "-", "+", "-", "*", "/", "mod", "and", "or",
                           "xor", "=", "<>", "<", ">", "<=", ">="])
            left = self.expr(depth - 1, allow_calls)
            right = self.expr(depth - 1, allow_calls)
            return Bin(op, left, right)
        if k < 0.65:
            return Un(r.choice(["-", "not"]), self.expr(depth - 1, allow_calls))
        if k < 0.85:
            name = r.choice(["low", "high", "double", "shl", "shr", "rol", "ror"])
            x = self.expr(depth - 1, allow_calls)
            if name in ("low", "high", "double"):
                return Builtin(name, [x])
            limit = 15 if name in ("shl", "shr") else 7
            if r.random() < 0.7:
                count: Expr = Num(r.randint(1, limit))
            else:
                # A computed count, kept away from the undefined 0.
                count = Bin("+", Bin("and", Var(r.choice(BYTE_VARS)), Num(limit // 2)),
                            Num(1))
            return Builtin(name, [x, count])
        if k < 0.95:
            return self._leaf(allow_calls)
        # An operand the optimizer can see through: a constant it must type.
        return Bin(r.choice(["+", "-", "*", "and", "or", "xor", "mod", "/"]),
                   self._num(), self._num())

    def _typed_expr(self, depth: int) -> Expr:
        """An expression for a statement: calls allowed, at most one embedded
        assignment, and none to a variable the statement reads."""
        def make() -> Expr:
            e = self.expr(depth)
            if self.rnd.random() < 0.15:
                target = self.rnd.choice(["eb", "ew"])
                e = EAsg(target, e)
                if self.rnd.random() < 0.5:
                    e = Bin(self.rnd.choice(["+", "-", "and", "xor"]), e, self.expr(1, False))
            return e
        return self._order_free(make)

    def _order_free(self, make, targets: tuple[str, ...] = ()) -> Expr:
        """An expression from ``make`` whose value does not depend on the
        order its operands are evaluated in: it calls at most one function
        that changes a variable, and then neither reads the variable nor is
        stored into it."""
        for _ in range(50):
            e = make()
            changed = [SIDE_EFFECTS[c][0] for c in calls_in(e) if c in SIDE_EFFECTS]
            if not changed:
                return e
            if (len(changed) == 1 and changed[0] not in e.reads()
                    and changed[0] not in targets and changed[0] not in e.assigns()):
                return e
        return self._num()

    # -- statements

    def emit(self, line: str) -> None:
        self.lines.append("  " + line)

    def print_value(self, label: str, value: int, what: str) -> None:
        self.emit(f"call ph({what});")
        self.expect.append((label, value & 0xFFFF))

    def stmt(self) -> None:
        r = self.rnd.random()
        env = self.env
        depth = self.rnd.randint(1, 4)
        if r < 0.22:
            e = self._typed_expr(depth)
            v, _ = e.ev(env)
            env.store("rw", v)
            self.emit(f"rw = {e.text()};")
            self.print_value(f"rw = {e.text()}", env.vals["rw"], "rw")
        elif r < 0.34:
            e = self._typed_expr(depth)
            v, _ = e.ev(env)
            env.store("rb", v)
            self.emit(f"rb = {e.text()};")
            self.print_value(f"rb = {e.text()}", env.vals["rb"], "rb")
        elif r < 0.46:
            e = self._typed_expr(depth)
            v, _ = e.ev(env)
            self.print_value(f"ph({e.text()})", v, e.text())
        elif r < 0.66:
            # Assign a pool variable: a constant (which -O3 propagates) or an
            # expression.
            name = self.rnd.choice(BYTE_VARS + ADDR_VARS + list(LOCAL_VARS))
            if self.rnd.random() < 0.5:
                e: Expr = self._num()
            else:
                e = self._order_free(lambda: self.expr(depth, allow_calls=True), (name,))
            v, _ = e.ev(env)
            env.store(name, v)
            self.emit(f"{name} = {e.text()};")
        elif r < 0.74:
            e = self._typed_expr(depth)
            v, _ = e.ev(env)
            then = 1 if v & 1 else 2
            env.store("rb", then)
            self.emit(f"if {e.text()} then rb = 1; else rb = 2;")
            self.print_value(f"if {e.text()}", then, "rb")
        elif r < 0.76:
            self.emit("call bump;")
            env.store("w4", env.vals["w4"] + 3)
            env.store("b4", env.vals["b4"] + 1)
        elif r < 0.80:
            # A store through a BASED variable into a pool variable, which
            # constant propagation must not see past.
            name = self.rnd.choice(BYTE_VARS + ADDR_VARS)
            based = "bb" if env.types[name] == B else "bw"
            e = self.expr(depth, allow_calls=False)
            v, _ = e.ev(env)
            env.store(name, v)
            self.emit(f"pv = .{name}; {based} = {e.text()};")
            self.print_value(f"{name} stored through {based}", env.vals[name], name)
        elif r < 0.90:
            self.loop()
        else:
            e = self._order_free(lambda: self.expr(depth), ("b1", "w1"))
            v, _ = e.ev(env)
            env.store("b1", v)
            env.store("w1", v)
            self.emit(f"b1, w1 = {e.text()};")
            self.print_value("b1 of b1, w1 =", env.vals["b1"], "b1")
            self.print_value("w1 of b1, w1 =", env.vals["w1"], "w1")
        if self.rnd.random() < 0.15:
            for name in ("nc", "eb", "ew", "b3", "b4", "w4"):
                self.print_value(f"{name} so far", env.vals[name], name)

    def loop(self) -> None:
        """DO i = s TO l [BY k] as DRI compiles it: the limit is tested at
        the top, and the loop stops when the increment carries out."""
        r = self.rnd
        idx = r.choice(["ib", "iw"])
        t = self.env.types[idx]
        for _ in range(50):
            limit = r.choice([0, 1, 7, 0x7F, 0xFE, 0xFF, 0x100, 0x1FF, 0xFFFE,
                              0xFFFF, r.randrange(0x10000)])
            start = conv(limit, t) - r.randint(-3, 40)
            start &= 0xFFFF
            step = r.choice([None, None, 1, 2, 3, 5, 0x10, 0x80, 0xFF, 0xFFFF])
            n, acc, final = run_loop(start, limit, 1 if step is None else step, t)
            if 0 < conv(1 if step is None else step, t) and n <= 300:
                break
        else:
            return
        # Bounds and step are constants, pool variables holding them, or a
        # variable expression; the body must not change any of them.
        parts = []
        holders = (("lw", "lb"), ("w3", "b3"), ("w2", "b2"))
        for value, (wide, narrow) in zip((start, limit, step), holders):
            if value is None:
                parts.append(None)
            elif r.random() < 0.5:
                parts.append(Num(value, "hex"))
            else:
                var = wide if value > 0xFF or r.random() < 0.5 else narrow
                self.emit(f"{var} = {_hex(value)};")
                self.env.store(var, value)
                parts.append(Var(var))
        by = f" by {parts[2].text()}" if parts[2] is not None else ""
        self.emit("n = 0; acc = 0;")
        self.emit(f"do {idx} = {parts[0].text()} to {parts[1].text()}{by};")
        self.emit(f"  n = n + 1; acc = acc + {idx};")
        self.emit("end;")
        self.env.store("n", n)
        self.env.store("acc", acc)
        self.env.store(idx, final)
        label = f"do {idx} = {start:#x} to {limit:#x} by {step}"
        self.print_value(f"{label}: iterations", n, "n")
        self.print_value(f"{label}: sum", acc, "acc")
        self.print_value(f"{label}: final index", final, idx)

    def program(self) -> tuple[str, list[tuple[str, int]]]:
        r = self.rnd
        self.emit("nc = 0; eb = 0; ew = 0;")
        for name in BYTE_VARS + ADDR_VARS + list(LOCAL_VARS):
            if r.random() < 0.5:
                v = self._const_value(self.env.types[name])
                self.emit(f"{name} = {_hex(v)};")
                self.env.store(name, v)
            else:
                # Not a constant the optimizer can know.
                arr = "ab" if self.env.types[name] == B else "aw"
                i = r.randrange(8)
                self.emit(f"{name} = {arr}({i});")
                self.env.store(name, self.env.arrays[arr][1][i])
        for _ in range(self.n_stmts):
            self.stmt()
        self.print_value("calls made", self.env.vals["nc"], "nc")
        ab = ", ".join(_hex(v) for v in self.env.arrays["ab"][1])
        aw = ", ".join(_hex(v) for v in self.env.arrays["aw"][1])
        src = "\n".join([
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
            "declare (" + ", ".join(BYTE_VARS) + ", rb, eb, ib) byte;",
            "declare (" + ", ".join(ADDR_VARS) + ", rw, ew, iw, nc, n, acc, pv) address;",
            "declare bb based pv byte, bw based pv address;",
            "fb: procedure (p) byte; declare p byte;",
            "  nc = nc + 1; return p xor 5ah; end fb;",
            "fw: procedure (p) address; declare p address;",
            "  nc = nc + 1; return p + 1234h; end fw;",
            "gb: procedure byte; nc = nc + 1; return 0c3h; end gb;",
            "gw: procedure address; nc = nc + 1; return 8421h; end gw;",
            "hb: procedure byte; nc = nc + 1; b3 = b3 + 7; return b3; end hb;",
            "hw: procedure address; nc = nc + 1; w4 = w4 + 105h; return w4; end hw;",
            "bump: procedure; w4 = w4 + 3; b4 = b4 + 1; end bump;",
            "run: procedure;",
            f"  declare ab(*) byte data ({ab});",
            f"  declare aw(*) address data ({aw});",
            f"  declare big({BIG}) byte;",
            "  declare lb byte, lw address;",
            *self.lines,
            "end run;",
            "call run;",
            "end t;",
        ])
        return src, self.expect


def run_loop(start: int, limit: int, step: int, t: str) -> tuple[int, int, int]:
    """(iterations, sum of the index, final index) of DRI's DO loop."""
    i = conv(start, t)
    lim = conv(limit, t)
    k = conv(step, t)
    n = acc = 0
    while i <= lim:
        n += 1
        acc = (acc + i) & 0xFFFF
        i += k
        if i > MASK[t]:
            i &= MASK[t]
            break
        if n > 70000:
            break
    return n, acc, i


def generate(seed: int, n_stmts: int = 60) -> tuple[str, list[tuple[str, int]]]:
    """PL/M source for ``seed`` and the values it must print, in order."""
    return Generator(seed, n_stmts).program()


# ---- building and running ---------------------------------------------------

def build_and_run(src: str, opt: int) -> tuple[list[int], str | None]:
    """Compile ``src`` at ``-O opt``, link and run it: (printed words, error)."""
    from uplm80.compiler import Compiler
    compiler = Compiler(opt_level=opt)
    try:
        asm = compiler.compile(src, "<difftest>")
    except Exception as exc:  # pylint: disable=broad-except
        return [], f"compile: {type(exc).__name__}: {exc}"
    if asm is None:
        errors = getattr(compiler.errors, "errors", None) or ["failed"]
        return [], "compile: " + "; ".join(str(e) for e in errors)
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
