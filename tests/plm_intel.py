"""Random PL/M-80 programs in the dialect Intel's PL/M-80 V3.1 and uplm80 share.

For scripts/intel_oracle.py, which builds each with both compilers and
compares what they print: Intel's compiler is the oracle, so there is no
model of what a program prints, only rules that keep what it prints defined.

What a program covers: BYTE and ADDRESS arithmetic, relations and the
logical operators; LOW, HIGH, DOUBLE, SHL, SHR, ROL, ROR; * / MOD, with
division by zero (what @P0029, the divide routine of Intel's PLM80.LIB,
returns - uplm80 follows it); one- and two-character string constants;
LITERALLY; arrays with constant and computed subscripts; DATA and INITIAL;
structures and arrays of structures; BASED variables; MOVE, LENGTH, LAST and
SIZE; strings printed through MON1; IF, DO CASE, DO WHILE and iterative DO
(including limits and steps the body changes, which the manual (5.1.4)
defines); procedures nested two deep with 0 to 5 BYTE and ADDRESS
parameters, typed and untyped; REENTRANT recursion; and, unless left out,
PLUS, MINUS, SCL and SCR in the one form whose carry is defined by the
statement itself.

What it avoids, because PL/M-80 leaves it undefined (9800268B):
* evaluation order (4.5.1): a statement calls at most one procedure that
  changes a variable (other than the call counter NC, which only the dumps
  read), and then neither reads nor assigns that variable; an embedded
  assignment's target is read nowhere else in its statement, nor by a
  procedure the statement calls; arguments obey the same rule;
* a shift or rotate by 0 (11.1.4), an overlapping or zero-length MOVE
  (11.1.5), a DO CASE selector out of range, a subscript out of range;
* uninitialised reads: every variable is set before it is read, procedure
  locals in every activation; STACKPTR, MEMORY, INPUT/OUTPUT;
* printing an address: the two compilers lay memory out differently.
PLUS, MINUS, SCL and SCR use the CARRY flag, which the manual (12.1) says
cannot be relied on; they appear only as `(x + y) PLUS z', `(x - y) MINUS z',
`SCL(x + y, n)' and `SCR(x - y, n)' with x, y and z variables or constants,
store only into RC, which nothing else reads, and are printed at once, so a
difference there shows on its own line (``avoid={"carry"}`` leaves them out).

Each printing statement prints one line, headed by its number, which the
source carries as a comment (/* S1F */), so the first line that differs
names the statement.  :func:`generate` returns a :class:`Program`, whose
:meth:`Program.render` leaves out any chunks (statements, procedures) it is
told to - the reducer in scripts/intel_oracle.py drops chunks while the
difference persists.

Features ``avoid`` can leave out - the first six are differences from V3.1
the README lists:
* shl-byte: SHL or SHR of a BYTE, which uplm80 shifts in 16 bits
  (uplm80/plm_types.py);
* shift9: SHL or SHR of a BYTE by more than 8, which V3.1 shifts by the
  count mod 8;
* wide-limit: a BYTE index counted to a limit above 255, which V3.1
  compares in 16 bits where 5.1.4 makes the limit a BYTE;
* sub-zero: a BYTE less an ADDRESS that folds to 0, which V3.1 leaves a
  BYTE;
* zero-dividend: a constant 0 divided by a variable, which V3.1 folds to 0
  even when the divisor is 0, where uplm80 divides (0FFFFH);
* neg-widened: `-b' or `0 - b' of a BYTE the expression also uses
  elsewhere, which V3.1 may negate in 16 bits;
* qualsize (LENGTH, LAST and SIZE of a structure member, which uplm80
  refused before 0.4.2), carry, div0, strings, based, struct, move, case,
  while, loops, procs, reentrant, embedded, nested, str2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .plm_difftest import (INTERESTING, Bin, Builtin, EAsg, Expr, Num, Un, Var,
                           _hex, run_loop)

B, A = "BYTE", "ADDRESS"
MASK = {B: 0xFF, A: 0xFFFF}

# Variables the statements of the main program read and assign.
POOL = {"b1": B, "b2": B, "b3": B, "b4": B, "w1": A, "w2": A, "w3": A, "w4": A}
# Written by procedures: a procedure that assigns one is "side-effecting".
SIDE = {"sb": B, "sw": A}
# The call counter: every procedure adds 1; nothing but the dumps reads it,
# so calls may change it in any order.
COUNTER = "nc"
# Result variables: assigned and printed at once.
RESULTS = {"rb": B, "rw": A, "rc": A}
# Loop variables of the main program.
LOOPS = {"ix": B, "jx": B, "kx": A, "cnt": A, "acc": A, "lim": A, "stp": A}
# Writable arrays: (type, length).
ARRAYS = {"ab": (B, 8), "aw": (A, 8), "tb": (B, 8), "tw": (A, 4)}
# Read-only arrays (DATA): (type, length), filled in by the generator.
DATA_ARRAYS = {"cb": B, "cw": A}
MSG = "ms"


def mask_index(e: Expr, bound: int) -> str:
    """``e`` made a subscript below ``bound``."""
    if isinstance(e, Num):
        return str(e.v % bound)
    if bound & (bound - 1) == 0:
        return f"({e.text()}) and {bound - 1}"
    return f"({e.text()}) mod {bound}"


# ---- expressions beyond plm_difftest's -----------------------------------------

@dataclass
class Ref(Expr):
    """A variable reference: `ab(i)', `st.z(i)', `sa(i).y', `bb'."""
    parts: list            # [(name, index Expr | None, bound)]
    root: str              # the variable the reference reads
    typ: str
    extra: frozenset = frozenset()     # other variables it reads (a BASED pointer)

    def text(self):
        out = []
        for name, idx, bound in self.parts:
            out.append(name if idx is None else f"{name}({mask_index(idx, bound)})")
        return ".".join(out)

    def reads(self):
        r = {self.root} | set(self.extra)
        for _, idx, _ in self.parts:
            if idx is not None:
                r |= idx.reads()
        return r

    def assigns(self):
        r = set()
        for _, idx, _ in self.parts:
            if idx is not None:
                r |= idx.assigns()
        return r


@dataclass
class Call(Expr):
    """A call of a generated procedure (a function reference, or the
    procedure of a CALL statement)."""
    proc: "Proc"
    args: list

    def text(self):
        if not self.args:
            return self.proc.name
        return f"{self.proc.name}({', '.join(a.text() for a in self.args)})"

    def reads(self):
        return set().union(*(a.reads() for a in self.args)) if self.args else set()

    def assigns(self):
        return set().union(*(a.assigns() for a in self.args)) if self.args else set()


@dataclass
class Raw(Expr):
    """Text the generator writes itself: a constant it names, a carry form."""
    txt: str
    typ: str
    reads_: frozenset = frozenset()
    const: bool = False

    def text(self):
        return self.txt

    def reads(self):
        return set(self.reads_)

    def is_const(self):
        return self.const


def nodes(e):
    """Every expression node in ``e``."""
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(n)
            continue
        if not isinstance(n, Expr):
            continue
        yield n
        if isinstance(n, Ref):
            stack.extend(idx for _, idx, _ in n.parts if idx is not None)
        elif isinstance(n, Call):
            stack.extend(n.args)
        elif not isinstance(n, Raw):
            stack.extend(getattr(n, f) for f in n.__dataclass_fields__)


def typ(e: Expr, env: dict) -> str:
    """The PL/M-80 type of ``e`` (4.x, 11.x, 12.x)."""
    if isinstance(e, Num):
        if e.style == "str":
            return B
        return B if e.v <= 0xFF else A
    if isinstance(e, Var):
        return env.get(e.name, A)
    if isinstance(e, (Ref, Raw)):
        return e.typ
    if isinstance(e, Call):
        return e.proc.rtype or A
    if isinstance(e, EAsg):
        return typ(e.value, env)
    if isinstance(e, Un):
        return typ(e.operand, env)
    if isinstance(e, Bin):
        if e.op in ("=", "<>", "<", ">", "<=", ">="):
            return B
        if e.op in ("*", "/", "mod"):
            return A
        return B if typ(e.left, env) == B and typ(e.right, env) == B else A
    if isinstance(e, Builtin):
        n = e.name
        if n in ("low", "high", "rol", "ror"):
            return B
        if n in ("double", "size"):
            return A
        if n in ("shl", "shr"):
            return typ(e.args[0], env)
        return B       # length, last of the arrays here
    return A


# LENGTH and element width of what every program declares, for const_value;
# the generator adds the DATA arrays, whose lengths it draws.
EXTENTS = {"ab": (8, 1), "aw": (8, 2), "tb": (8, 1), "tw": (4, 2), "st": (1, 7),
           "sa": (4, 5), "st.z": (4, 1), "sa.z": (2, 1), "sa(1).z": (2, 1), "st.y": (1, 2)}


class _NotConst(Exception):
    pass


def const_value(e: Expr, extents: dict | None = None, lits: dict | None = None) -> int | None:
    """The value of ``e`` when it is a constant expression - LENGTH, LAST
    and SIZE of the arrays in ``extents``, and the LITERALLY names in
    ``lits``, included - else None."""
    try:
        return _cv(e, dict(EXTENTS, **(extents or {})), lits or {})[0]
    except (_NotConst, KeyError):
        return None


def _cv(e: Expr, ext: dict, lits: dict) -> tuple[int, str]:  # pylint: disable=too-many-return-statements
    if isinstance(e, Num):
        return e.v, B if e.style == "str" or e.v <= 0xFF else A
    if isinstance(e, Raw) and e.const:
        if e.txt in lits:
            return lits[e.txt], B if lits[e.txt] <= 0xFF else A
        if len(e.txt) == 4 and e.txt[0] == e.txt[3] == "'":
            return ord(e.txt[1]) << 8 | ord(e.txt[2]), A
        raise _NotConst
    if isinstance(e, Builtin) and e.name in ("length", "last", "size"):
        n, width = ext[e.args[0].name]
        v = {"length": n, "last": n - 1, "size": n * width}[e.name]
        return v, A if e.name == "size" or v > 0xFF else B
    if isinstance(e, Builtin):
        x, xt = _cv(e.args[0], ext, lits)
        if e.name == "low":
            return x & 0xFF, B
        if e.name == "high":
            return (x >> 8 if xt == A else 0), B
        if e.name == "double":
            return x, A
        c = _cv(e.args[1], ext, lits)[0] & 0xFF
        if e.name in ("shl", "shr"):
            v = (x << c) if e.name == "shl" else (x >> c)
            return v & MASK[xt], xt
        x &= 0xFF
        c &= 7
        v = (x << c | x >> (8 - c)) if e.name == "rol" else (x >> c | x << (8 - c))
        return v & 0xFF, B
    if isinstance(e, Un):
        v, t = _cv(e.operand, ext, lits)
        return (-v if e.op == "-" else ~v) & MASK[t], t
    if isinstance(e, Bin):
        (lv, lt), (rv, rt) = _cv(e.left, ext, lits), _cv(e.right, ext, lits)
        if e.op in ("=", "<>", "<", ">", "<=", ">="):
            ok = {"=": lv == rv, "<>": lv != rv, "<": lv < rv, ">": lv > rv,
                  "<=": lv <= rv, ">=": lv >= rv}[e.op]
            return (0xFF if ok else 0), B
        if e.op == "*":
            return (lv * rv) & 0xFFFF, A
        if e.op in ("/", "mod"):
            from .plm_difftest import divide   # pylint: disable=import-outside-toplevel
            q, r = divide(lv, rv)
            return (q if e.op == "/" else r), A
        t = B if lt == B and rt == B else A
        v = {"+": lv + rv, "-": lv - rv, "and": lv & rv, "or": lv | rv, "xor": lv ^ rv}[e.op]
        return v & MASK[t], t
    raise _NotConst


def neg_widened(e: Expr, env: dict) -> bool:
    """Whether ``e`` negates a BYTE variable - `-b' or `0 - b' - that it
    also uses elsewhere, which V3.1 may then negate in 16 bits."""
    negated = set()
    for n in nodes(e):
        if isinstance(n, Un) and n.op == "-" and isinstance(n.operand, Var):
            negated.add(n.operand.name)
        elif isinstance(n, Bin) and n.op == "-" and isinstance(n.left, Num) \
                and n.left.v == 0 and isinstance(n.right, Var):
            negated.add(n.right.name)
    negated = {v for v in negated if env.get(v) == B}
    if not negated:
        return False
    uses = [n.name for n in nodes(e) if isinstance(n, Var)]
    return any(uses.count(v) > 1 for v in negated)


def folds_to_address_zero(e: Expr, env: dict, extents=None, lits=None) -> bool:
    """Whether ``e`` is an ADDRESS a compiler folds to 0: a constant
    expression of value 0, or a product with a literal 0."""
    if typ(e, env) != A:
        return False
    if isinstance(e, Bin) and e.op == "*" and any(
            isinstance(x, Num) and x.v == 0 for x in (e.left, e.right)):
        return True
    return const_value(e, extents, lits) == 0


# ---- procedures ---------------------------------------------------------------------

@dataclass
class Proc:  # pylint: disable=too-many-instance-attributes
    """A generated procedure."""
    name: str
    params: list            # [(name, type)]
    rtype: str | None
    reentrant: bool = False
    parent: "Proc | None" = None
    locals: list = field(default_factory=list)      # [(name, type)]
    reads: set = field(default_factory=set)         # outside variables, callees' too
    writes: set = field(default_factory=set)
    children: list = field(default_factory=list)
    depth: int = 0
    features: set = field(default_factory=set)      # what its body uses


# ---- the program ------------------------------------------------------------------------

@dataclass
class Chunk:
    """A piece of the program the reducer may leave out; ``parts`` are lines
    and nested chunks."""
    id: int
    parts: list
    kind: str = "stmt"


@dataclass
class Program:
    """A generated program: fixed text and chunks, in order."""
    seed: int
    parts: list
    features: dict = field(default_factory=dict)    # statement tag -> features

    def chunks(self) -> list[Chunk]:
        out = []

        def walk(parts):
            for p in parts:
                if isinstance(p, Chunk):
                    out.append(p)
                    walk(p.parts)
        walk(self.parts)
        return out

    def render(self, drop=frozenset()) -> str:
        """The source, without the chunks in ``drop``."""
        lines: list[str] = []

        def walk(parts):
            for p in parts:
                if isinstance(p, Chunk):
                    if p.id not in drop:
                        walk(p.parts)
                else:
                    lines.extend(wrap(p))
        walk(self.parts)
        return "\n".join(lines) + "\n"


def wrap(line: str, width: int = 100) -> list[str]:
    """``line`` broken at spaces outside strings into lines of at most about
    ``width`` characters: PL/M-80 V3.1 limits a source line (ERROR 86)."""
    if len(line) <= width:
        return [line]
    indent = line[:len(line) - len(line.lstrip())]
    out, cur, quoted, last_space = [], "", False, -1
    for ch in line:
        cur += ch
        if ch == "'":
            quoted = not quoted
        elif ch == " " and not quoted:
            last_space = len(cur) - 1
        if len(cur) > width and last_space > len(indent) + 4:
            out.append(cur[:last_space].rstrip())
            cur = indent + "    " + cur[last_space + 1:]
            last_space = -1
    out.append(cur)
    return out


PRINTERS = """hexd: procedure (d);
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
pt: procedure (n);
  declare n byte;
  call hexd(shr(n, 4)); call hexd(n); call mon1(2, ':'); call mon1(2, ' ');
end pt;
nl: procedure;
  call mon1(2, 0dh); call mon1(2, 0ah);
end nl;
pa: procedure (p, n);
  declare (p, n) address, c based p byte;
  do while n > 0;
    call hexd(shr(c, 4)); call hexd(c);
    p = p + 1; n = n - 1;
  end;
  call mon1(2, ' ');
end pa;"""


def _split_procs(text: str) -> list[str]:
    """``text``, a run of procedures, one string each."""
    out, cur = [], []
    for line in text.split("\n"):
        cur.append(line)
        if line.startswith("end "):
            out.append("\n".join(cur))
            cur = []
    return out


class Generator:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Builds one random program."""

    def __init__(self, seed: int, n_stmts: int = 50, avoid=frozenset()) -> None:
        self.seed = seed
        self.rnd = random.Random(seed)
        self.n_stmts = n_stmts
        self.avoid = frozenset(avoid)
        self.next_id = 0
        self.tag = 0
        self.features: dict[int, list[str]] = {}
        self.types: dict[str, str] = {}
        self.types.update(POOL)
        self.types.update(SIDE)
        self.types.update(RESULTS)
        self.types.update(LOOPS)
        self.types[COUNTER] = A
        self.procs: list[Proc] = []
        self.lits: dict[str, int] = {}
        self.data: dict[str, list[int]] = {}
        self.msg = ""
        self.extents: dict[str, tuple[int, int]] = {}   # the DATA arrays' LENGTH, width
        # The statement being generated: features it uses.
        self.cur: list[str] = []

    # -- small helpers

    def chunk(self, parts, kind="stmt") -> Chunk:
        self.next_id += 1
        return Chunk(self.next_id, parts, kind)

    def use(self, feature: str) -> bool:
        """Whether ``feature`` may be used; records it when it is."""
        if feature in self.avoid:
            return False
        if feature not in self.cur:
            self.cur.append(feature)
        return True

    def chance(self, p: float) -> bool:
        return self.rnd.random() < p

    def const_value(self, t: str | None = None) -> int:
        r = self.rnd
        if r.random() < 0.7:
            v = r.choice(INTERESTING)
        else:
            v = r.randrange(0x10000) if r.random() < 0.5 else r.randrange(0x100)
        if t is not None:
            v &= MASK[t]
        return v

    def num(self) -> Expr:
        v = self.const_value()
        r = self.rnd.random()
        if v <= 0xFF and 0x41 <= v <= 0x5A and r < 0.3:
            return Num(v, "str")
        if r < 0.05 and self.lits:
            name = self.rnd.choice(sorted(self.lits))
            return Raw(name, B if self.lits[name] <= 0xFF else A, const=True)
        if r < 0.08 and self.use("str2"):
            a, b = self.rnd.choice("ABCXYZ09"), self.rnd.choice("ABCXYZ09")
            return Raw(f"'{a}{b}'", A, const=True)
        return Num(v, "hex" if r < 0.6 else "dec")

    # -- references

    def elem(self, ctx: "Ctx", writable: bool = False) -> Ref | None:
        """An array element, a structure member, a character of the message
        or a BASED variable: to read, or in the main program to assign."""
        if writable and ctx.proc is not None:
            return None          # procedures assign their own variables only
        opts = ["array", "array"]
        if not writable:
            opts.append("data")
            if self.msg and "strings" not in self.avoid:
                opts.append("msg")
        if "struct" not in self.avoid:
            opts += ["struct"]
        if ctx.proc is None and "based" not in self.avoid:
            opts.append("based")
        kind = self.rnd.choice(opts)
        idx = self.index_expr(ctx)
        if kind == "array":
            name = self.rnd.choice(sorted(ARRAYS))
            t, n = ARRAYS[name]
            return Ref([(name, idx, n)], name, t)
        if kind == "data":
            name = self.rnd.choice(sorted(DATA_ARRAYS))
            return Ref([(name, idx, len(self.data[name]))], name, DATA_ARRAYS[name])
        if kind == "msg":
            self.use("strings")
            return Ref([(MSG, idx, len(self.msg))], MSG, B)
        if kind == "struct":
            self.use("struct")
            member = self.rnd.choice(["x", "y", "z"])
            mt = B if member != "y" else A
            if self.chance(0.4):
                if member == "z":
                    return Ref([("st", None, 0), ("z", idx, 4)], "st", B)
                return Ref([("st", None, 0), (member, None, 0)], "st", mt)
            if member == "z":
                return Ref([("sa", idx, 4), ("z", self.index_expr(ctx), 2)], "sa", B)
            return Ref([("sa", idx, 4), (member, None, 0)], "sa", mt)
        self.use("based")
        if self.chance(0.5):
            return Ref([("bb", None, 0)], "tb", B, frozenset({"pb"}))
        return Ref([("bw", None, 0)], "tw", A, frozenset({"pw"}))

    def index_expr(self, ctx: "Ctx") -> Expr:
        if self.chance(0.4):
            return Num(self.rnd.randrange(8))
        return self.expr(ctx, self.rnd.randint(0, 1), calls=False)

    # -- expressions

    def leaf(self, ctx: "Ctx", calls: bool) -> Expr:
        r = self.rnd.random()
        if r < 0.28:
            return self.num()
        if r < 0.7:
            return Var(self.rnd.choice(ctx.readable))
        if r < 0.82:
            return self.elem(ctx)
        if r < 0.87:
            which = self.rnd.random()
            fn = self.rnd.choice(["length", "last", "size"])
            if which < 0.5:
                name = self.rnd.choice(["ab", "aw", "tb", "tw", "cb", "cw"])
            elif which < 0.8 and self.use("struct"):
                name = self.rnd.choice(["st", "sa", "sa.z", "st.z", "sa(1).z", "st.y"])
                if "." in name and not self.use("qualsize"):
                    name = name.split("(")[0].split(".")[0]
                if name == "st.y":
                    fn = "size"
            else:
                name = MSG if self.msg and self.use("strings") else "ab"
            if name in ("st", "st.y"):
                fn = "size"
            return Builtin(fn, [Var(name)])
        if calls:
            c = self.call(ctx, typed=True)
            if c is not None:
                return c
        return self.num()

    def expr(self, ctx: "Ctx", depth: int, calls: bool = True) -> Expr:
        r = self.rnd
        if depth <= 0 or r.random() < 0.2:
            return self.leaf(ctx, calls)
        k = r.random()
        if k < 0.5:
            ops = ["+", "-", "+", "-", "*", "/", "mod", "and", "or", "xor",
                   "=", "<>", "<", ">", "<=", ">="]
            op = r.choice(ops)
            left = self.expr(ctx, depth - 1, calls)
            right = self.expr(ctx, depth - 1, calls)
            if op in ("/", "mod") and "div0" in self.avoid:
                right = Bin("or", right, Num(1))
            if op == "-" and folds_to_address_zero(right, self.types, self.extents, self.lits) \
                    and not self.use("sub-zero"):
                right = Bin("or", right, Num(1))
            if op == "/" and const_value(left, self.extents, self.lits) == 0 \
                    and const_value(right, self.extents, self.lits) is None \
                    and not self.use("zero-dividend"):
                right = Bin("or", right, Num(1))
            return Bin(op, left, right)
        if k < 0.6:
            return Un(r.choice(["-", "not"]), self.expr(ctx, depth - 1, calls))
        if k < 0.82:
            name = r.choice(["low", "high", "double", "shl", "shr", "rol", "ror"])
            x = self.expr(ctx, depth - 1, calls)
            if name in ("low", "high", "double"):
                return Builtin(name, [x])
            byte_shift = name in ("shl", "shr") and typ(x, self.types) == B
            if byte_shift and not self.use("shl-byte"):
                x = Builtin("double", [x])
                byte_shift = False
            limit = 17 if name in ("shl", "shr") else 9
            if byte_shift and "shift9" in self.avoid:
                limit = 8
            if r.random() < 0.7:
                count: Expr = Num(r.randint(1, limit))
                if byte_shift and count.v > 8:
                    self.use("shift9")
            else:
                # Computed, never 0 (11.1.4 leaves a count of 0 undefined).
                count = Bin("+", Bin("and", self.expr(ctx, 0, False), Num(7)), Num(1))
            return Builtin(name, [x, count])
        if k < 0.9:
            return self.leaf(ctx, calls)
        # An operand the optimizer can fold.
        return Bin(r.choice(["+", "-", "*", "and", "or", "xor", "mod", "/"]),
                   self.num(), self.num())

    def cond(self, ctx: "Ctx", calls: bool = True) -> Expr:
        """A condition: a relation, a combination, or any expression (bit 0)."""
        r = self.rnd.random()
        if r < 0.5:
            op = self.rnd.choice(["=", "<>", "<", ">", "<=", ">="])
            return Bin(op, self.expr(ctx, 2, calls), self.expr(ctx, 1, calls))
        if r < 0.75:
            a = Bin(self.rnd.choice(["<", ">=", "<>"]), self.expr(ctx, 1, calls),
                    self.expr(ctx, 1, False))
            b = Bin(self.rnd.choice(["=", ">", "<="]), self.expr(ctx, 1, False),
                    self.expr(ctx, 1, False))
            e = Bin(self.rnd.choice(["and", "or", "xor"]), a, b)
            return Un("not", e) if self.chance(0.2) else e
        return self.expr(ctx, 2, calls)

    def call(self, ctx: "Ctx", typed: bool) -> Call | None:
        cands = [p for p in ctx.callable if (p.rtype is not None) == typed]
        if not cands or not self.use("procs"):
            return None
        p = self.rnd.choice(cands)
        for f in sorted(p.features):          # what its body uses, this statement uses
            self.use(f)
        args = []
        for i in range(len(p.params)):
            if p.reentrant and i == 0:
                # the depth: small
                args.append(Bin("and", self.expr(ctx, 1, False), Num(7)))
            else:
                args.append(self.expr(ctx, self.rnd.randint(0, 2), calls=self.chance(0.3)))
        return Call(p, args)

    # -- evaluation order

    def order_free(self, exprs, targets=(), target_reads=frozenset()) -> bool:
        """Whether a statement of ``exprs`` stored into ``targets`` has one
        value whatever order its operands are evaluated in."""
        calls = [n for e in exprs for n in nodes(e) if isinstance(n, Call)]
        easg = [n.target for e in exprs for n in nodes(e) if isinstance(n, EAsg)]
        reads = set(target_reads)
        for e in exprs:
            reads |= e.reads()
        targets = set(targets)
        if len(set(easg)) != len(easg):
            return False
        for t in easg:
            if t in reads or t in targets:
                return False
        for i, c in enumerate(calls):
            w = c.proc.writes - {COUNTER}
            if w & (reads | targets | set(easg)):
                return False
            if set(easg) & c.proc.reads:
                return False
            for j, d in enumerate(calls):
                if i != j and w & (d.proc.reads | d.proc.writes):
                    return False
        return True

    def pick(self, make, targets=(), target_reads=frozenset(), tries: int = 40):
        """An expression from ``make`` that :meth:`order_free` accepts."""
        saved = list(self.cur)
        for _ in range(tries):
            self.cur = list(saved)
            e = make()
            if neg_widened(e, self.types) and not self.use("neg-widened"):
                continue
            if self.order_free([e], targets, target_reads):
                return e
        self.cur = saved
        return self.num()

    def typed_expr(self, ctx: "Ctx", targets=(), target_reads=frozenset()) -> Expr:
        """The right side of an assignment: calls allowed, sometimes one
        embedded assignment."""
        depth = self.rnd.randint(1, 4)

        def make():
            e = self.expr(ctx, depth)
            if ctx.proc is None and self.chance(0.12) and self.use("embedded"):
                e = EAsg(self.rnd.choice(["eb", "ew"]), e)
                if self.chance(0.5):
                    e = Bin(self.rnd.choice(["+", "-", "and", "xor"]), e,
                            self.expr(ctx, 1, False))
            return e
        return self.pick(make, targets, target_reads)

    # -- main-program statements

    def begin(self) -> None:
        self.cur = []

    def printed(self, lines: list[str], values: list[str]) -> list[str]:
        """``lines``, then a line printing ``values`` headed by a tag."""
        self.tag += 1
        tag = self.tag & 0xFF
        self.features[tag] = list(self.cur)
        out = [f"/* S{tag:02X} */"] + lines
        out.append(f"call pt({tag}); " + " ".join(f"call ph({v});" for v in values)
                   + " call nl;")
        return out

    def stmt(self, ctx: "Ctx") -> list[str]:
        self.begin()
        r = self.rnd.random()
        if r < 0.2:
            return self.s_assign(ctx)
        if r < 0.3:
            return self.s_elem(ctx)
        if r < 0.36:
            return self.s_multi(ctx)
        if r < 0.46:
            return self.s_if(ctx)
        if r < 0.52 and self.use("case"):
            return self.s_case(ctx)
        if r < 0.58 and self.use("while"):
            return self.s_while(ctx)
        if r < 0.7 and self.use("loops"):
            return self.s_loop(ctx)
        if r < 0.77:
            return self.s_call(ctx)
        if r < 0.81 and self.use("based"):
            return self.s_based(ctx)
        if r < 0.85 and self.use("move"):
            return self.s_move(ctx)
        if r < 0.88 and self.use("strings"):
            return self.s_string(ctx)
        if r < 0.93 and "carry" not in self.avoid:
            return self.s_carry(ctx)
        return self.s_assign(ctx)

    def s_assign(self, ctx) -> list[str]:
        name = self.rnd.choice(sorted(POOL) + ["rb", "rw"])
        e = self.typed_expr(ctx, (name,))
        return self.printed([f"{name} = {e.text()};"], [name])

    def s_elem(self, ctx) -> list[str]:
        ref = self.elem(ctx, writable=True)
        if ref is None:
            return self.s_assign(ctx)
        e = self.typed_expr(ctx, (ref.root,), frozenset(ref.reads() - {ref.root}))
        # read it back through a constant reference where there is one
        return self.printed([f"{ref.text()} = {e.text()};"], [self.readback(ref)])

    def readback(self, ref: Ref) -> str:
        """Something that prints what was just stored through ``ref``: the
        same reference, unless its subscript reads what was stored."""
        idx_reads = set()
        for _, idx, _ in ref.parts:
            if idx is not None:
                idx_reads |= idx.reads()
        if ref.root in idx_reads or ref.extra & idx_reads:
            return "0"
        return ref.text()

    def s_multi(self, ctx) -> list[str]:
        a, b = self.rnd.sample(sorted(POOL), 2)
        e = self.typed_expr(ctx, (a, b))
        return self.printed([f"{a}, {b} = {e.text()};"], [a, b])

    def s_if(self, ctx) -> list[str]:
        c = self.pick(lambda: self.cond(ctx))
        a, b = self.rnd.choice(sorted(POOL)), self.rnd.choice(sorted(POOL))
        e1 = self.typed_expr(ctx, (a,))
        e2 = self.typed_expr(ctx, (b,))
        lines = [f"if {c.text()} then"]
        if self.chance(0.4):
            c2 = self.pick(lambda: self.cond(ctx, calls=False))
            e3 = self.typed_expr(ctx, ("rb",))
            lines += ["  do;", f"    {a} = {e1.text()};",
                      f"    if {c2.text()} then rb = {e3.text()}; else rb = 5;", "  end;"]
        else:
            lines += [f"  {a} = {e1.text()};"]
        if self.chance(0.7):
            lines += [f"else {b} = {e2.text()};"]
        return self.printed(["rb = 0;"] + lines, [a, b, "rb"])

    def s_case(self, ctx) -> list[str]:
        n = self.rnd.randint(2, 6)
        sel = self.pick(lambda: self.expr(ctx, 2))
        sel_txt = mask_index(sel, n) if not isinstance(sel, Num) else str(sel.v % n)
        lines = ["rw = 0;", f"do case {sel_txt};"]
        for i in range(n):
            k = self.rnd.random()
            if k < 0.15:
                lines.append("  ;")
            elif k < 0.3:
                e = self.typed_expr(ctx, ("rw",))
                lines += ["  do;", f"    rw = {e.text()};", f"    rb = {i};", "  end;"]
            else:
                e = self.typed_expr(ctx, ("rw",))
                lines.append(f"  rw = {e.text()} + {i};")
        lines.append("end;")
        return self.printed(lines, ["rw", "rb"])

    def s_while(self, ctx) -> list[str]:
        n = self.rnd.randint(0, 12)
        body = self.typed_expr(ctx, ("acc",))
        if self.chance(0.5):
            head = f"do while (kx := kx + 1) <= {n};"
        else:
            c = self.pick(lambda: self.cond(ctx, calls=False), ("kx", "acc"))
            if "kx" in c.reads() or "acc" in c.reads():
                c = Num(1)
            head = f"do while kx < {n} and ({c.text()});"
        lines = ["kx = 0; acc = 0;", head, f"  acc = acc + ({body.text()});"]
        if "(kx :=" not in head:
            lines.append("  kx = kx + 1;")
        lines.append("end;")
        return self.printed(lines, ["kx", "acc"])

    def s_loop(self, ctx) -> list[str]:
        k = self.rnd.random()
        if k < 0.45:
            return self.loop_edges(ctx)
        if k < 0.75:
            return self.loop_computed(ctx)
        return self.loop_moving(ctx)

    def loop_edges(self, ctx) -> list[str]:
        """A counted DO at the edges of its index's range (plm_difftest's)."""
        r = self.rnd
        idx = r.choice(["ix", "kx"])
        t = LOOPS[idx]
        for _ in range(50):
            limit = r.choice([0, 1, 7, 0x7F, 0xFE, 0xFF, 0x100, 0x1FF, 0xFFFE,
                              0xFFFF, r.randrange(0x10000)])
            start = ((limit & MASK[t]) - r.randint(-3, 40)) & 0xFFFF
            step = r.choice([None, None, 1, 2, 3, 5, 0x10, 0x80, 0xFF, 0xFFFF])
            if t == B and limit > 0xFF and not self.use("wide-limit"):
                limit &= 0xFF
            n, _, _ = run_loop(start, limit, 1 if step is None else step, t)
            if (1 if step is None else step) & MASK[t] and n <= 300:
                break
        else:
            return self.s_assign(ctx)
        by = f" by {_hex(step)}" if step is not None else ""
        extra = []
        if r.random() < 0.4:
            e = self.pick(lambda: self.expr(ctx, 2, calls=True), ("w2",))
            if idx not in e.reads():
                extra = [f"  w2 = w2 + ({e.text()});"]
        lines = ["cnt = 0; acc = 0;", f"do {idx} = {_hex(start)} to {_hex(limit)}{by};",
                 f"  cnt = cnt + 1; acc = acc + {idx};", *extra, "end;"]
        return self.printed(lines, ["cnt", "acc", idx] + (["w2"] if extra else []))

    def loop_computed(self, ctx) -> list[str]:
        """DO with computed bounds and step, and a nested loop."""
        idx = self.rnd.choice(["ix", "kx"])
        lo = self.pick(lambda: self.expr(ctx, 2, calls=False))
        hi = self.pick(lambda: self.expr(ctx, 2, calls=False))
        lines = ["cnt = 0; acc = 0;",
                 f"do {idx} = ({lo.text()}) and 7 to ({hi.text()}) and 0fh"
                 + (f" by (({self.expr(ctx, 1, False).text()}) and 3) + 1;"
                    if self.chance(0.4) else ";")]
        body = self.pick(lambda: self.expr(ctx, 2, calls=True), ("acc",))
        if idx in body.assigns():
            body = Num(1)
        lines.append(f"  acc = acc + ({body.text()}) + {idx};")
        if self.chance(0.4):
            inner = self.pick(lambda: self.expr(ctx, 1, calls=False))
            lines += [f"  do jx = 1 to ({inner.text()}) and 3;",
                      "    cnt = cnt + jx;", "  end;"]
        lines += ["  cnt = cnt + 1;", "end;"]
        return self.printed(lines, ["cnt", "acc", idx])

    def loop_moving(self, ctx) -> list[str]:
        """A DO whose limit and step the body changes (5.1.4: both are
        evaluated again each time)."""
        idx = self.rnd.choice(["ix", "kx"])
        lim0 = self.rnd.randint(0, 20)
        stp0 = self.rnd.randint(1, 3)
        change_lim = self.rnd.choice(["lim = lim - 1;", "lim = lim + 1;", ""])
        change_stp = self.rnd.choice(["stp = stp + 1;", ""])
        lines = [f"cnt = 0; acc = 0; lim = {lim0}; stp = {stp0};",
                 f"do {idx} = 1 to lim by stp;",
                 f"  cnt = cnt + 1; acc = acc + {idx};"]
        if change_lim == "lim = lim + 1;" and not change_stp:
            change_stp = "stp = stp + 1;"
        lines += [f"  {x}" for x in (change_lim, change_stp) if x]
        lines.append("end;")
        return self.printed(lines, ["cnt", "acc", idx, "lim"])

    def s_call(self, ctx) -> list[str]:
        def make():
            c = self.call(ctx, typed=False)
            return c if c is not None else Num(0)
        c = self.pick(make)
        if not isinstance(c, Call):
            return self.s_assign(ctx)
        return self.printed([f"call {c.text()};"], ["sb", "sw"])

    def s_based(self, ctx) -> list[str]:
        if self.chance(0.5):
            i = self.rnd.randrange(8)
            e = self.typed_expr(ctx, ("tb",), frozenset({"pb"}))
            lines = [f"pb = .tb({i});", f"bb = {e.text()};"]
            return self.printed(lines, ["bb", f"tb({i})"])
        i = self.rnd.randrange(4)
        e = self.typed_expr(ctx, ("tw",), frozenset({"pw"}))
        lines = [f"pw = .tw({i});", f"bw = {e.text()};"]
        return self.printed(lines, ["bw", f"tw({i})"])

    def s_move(self, ctx) -> list[str]:
        """MOVE between two arrays (never overlapping, never 0 bytes)."""
        src, dst = self.rnd.choice([("ab", "tb"), ("cb", "tb"), ("tb", "ab"), ("ms", "tb"),
                                    ("aw", "tw"), ("cw", "tw"), ("ab", "st"), ("cb", "ab"),
                                    ("tb", "sa")])
        size = {"ab": 8, "tb": 8, "cb": len(self.data["cb"]), "ms": len(self.msg),
                "aw": 16, "tw": 8, "cw": 2 * len(self.data["cw"]), "st": 7, "sa": 20}
        byte_arrays = ("ab", "tb", "cb", "ms")
        s_off = self.rnd.randrange(size[src]) if src in byte_arrays else 0
        d_off = self.rnd.randrange(size[dst]) if dst in ("ab", "tb") else 0
        room = min(size[src] - s_off, size[dst] - d_off)
        if self.chance(0.5) or room == 1:
            count = str(self.rnd.randint(1, room))
        else:
            m = 1
            while m * 2 <= room:
                m *= 2
            e = self.pick(lambda: self.expr(ctx, 1, calls=False))
            count = f"(({e.text()}) and {m - 1}) + 1"
        s = f".{src}({s_off})" if s_off else f".{src}"
        d = f".{dst}({d_off})" if d_off else f".{dst}"
        self.tag += 1
        tag = self.tag & 0xFF
        self.features[tag] = list(self.cur)
        return [f"/* S{tag:02X} */", f"call move({count}, {s}, {d});",
                f"call pt({tag}); call pa(.{dst}, size({dst})); call nl;"]

    def s_string(self, ctx) -> list[str]:
        k = self.rnd.random()
        if k < 0.4:
            lines = ["call mon1(9, .ms);"]
        elif k < 0.7:
            words = self.rnd.choice(["HELLO", "World", "pl/m-80", "12 + 34", "a,b;c"])
            lines = [f"call mon1(9, .('{words}$'));"]
        else:
            lines = ["do ix = 0 to last(ms) - 1;", "  call mon1(2, ms(ix));", "end;"]
        self.tag += 1
        tag = self.tag & 0xFF
        self.features[tag] = list(self.cur)
        return [f"/* S{tag:02X} */", f"call pt({tag});"] + lines + ["call nl;"]

    def s_carry(self, ctx) -> list[str]:
        """PLUS, MINUS, SCL or SCR, with the carry made by the statement."""
        self.use("carry")

        def operand():
            if self.chance(0.4):
                return self.num().text()
            return self.rnd.choice(sorted(POOL))
        # x and y are variables: a folded x + y would leave no carry.
        x, y = self.rnd.choice(sorted(POOL)), self.rnd.choice(sorted(POOL))
        z = operand()
        k = self.rnd.random()
        if k < 0.35:
            e = f"({x} + {y}) plus {z}"
        elif k < 0.7:
            e = f"({x} - {y}) minus {z}"
        elif k < 0.85:
            e = f"scl({x} + {y}, {self.rnd.randint(1, 3)})"
        else:
            e = f"scr({x} - {y}, {self.rnd.randint(1, 3)})"
        return self.printed([f"rc = {e};"], ["rc"])

    # -- procedures

    def make_proc(self, idx: int, parent: Proc | None, visible: list[Proc],
                  depth: int) -> tuple[Proc, list]:
        """A procedure, and its text as parts (lines and chunks)."""
        r = self.rnd
        name = f"p{idx}"
        outer_cur, self.cur = self.cur, []
        reent = parent is None and r.random() < 0.2 and self.use("reentrant")
        nparams = r.randint(1, 3) if reent else r.randint(0, 5)
        params = [(f"{name}{chr(97 + i)}", r.choice([B, A])) for i in range(nparams)]
        if reent:
            params[0] = (params[0][0], B)       # the depth
        rtype = r.choice([None, B, A, A]) if not reent else A
        p = Proc(name, params, rtype, reent, parent, depth=depth)
        nloc = r.randint(0, 3)
        p.locals = [(f"{name}l{i}", r.choice([B, A])) for i in range(nloc)]
        own = dict(p.params + p.locals)
        self.types.update(own)
        head = f"{name}: procedure" + (f" ({', '.join(n for n, _ in params)})" if params else "")
        if rtype:
            head += f" {rtype.lower()}"
        if reent:
            head += " reentrant"
        parts: list = [head + ";"]
        for n, t in params + p.locals:
            parts.append(f"  declare {n} {t.lower()};")
        parts.append(f"  declare {name}i byte;")        # a loop index
        self.types[f"{name}i"] = B
        # Readable: the pool, the arrays, this procedure's own, and the
        # parameters of the procedures it is nested in.
        outer = []
        q = parent
        while q is not None:
            outer += [n for n, _ in q.params]
            q = q.parent
        readable = sorted(POOL) + [n for n, _ in params] + outer
        callable_ = [c for c in visible if not c.reentrant or c is p]
        # Nested procedures, which see this one's parameters.
        if depth < 1 and not reent and r.random() < 0.35 and self.use("nested"):
            for k in range(r.randint(1, 2)):
                child, cparts = self.make_proc(idx * 10 + k + 1, p, callable_ + p.children,
                                               depth + 1)
                p.children.append(child)
                for f in sorted(child.features):
                    self.use(f)
                parts.append(self.chunk(["  " + x if isinstance(x, str) else x
                                         for x in cparts], "proc"))
        ctx = Ctx(proc=p, readable=readable, callable=callable_ + p.children)
        body = []
        # Every local is set before anything reads it.
        for n, t in p.locals:
            e = self.pick(lambda: self.expr(ctx, r.randint(1, 3)), (n,))
            body.append(f"  {n} = {e.text()};")
            ctx.readable.append(n)
            self.note_reads(p, e)
        for _ in range(r.randint(0, 3)):
            body += self.proc_stmt(ctx)
        if r.random() < 0.3 and not reent:
            side = r.choice(sorted(SIDE))
            e = self.pick(lambda: self.expr(ctx, 2, calls=False), (side,))
            body.append(f"  {side} = {side} + ({e.text()});")
            p.writes.add(side)
            p.reads.add(side)
            self.note_reads(p, e)
        body.append(f"  {COUNTER} = {COUNTER} + 1;")
        p.writes.add(COUNTER)
        if reent:
            n0 = params[0][0]
            rest = [self.expr(ctx, 1, calls=False) for _ in params[1:]]
            e = self.expr(ctx, 2, calls=False)
            args = ", ".join([f"{n0} - 1"] + [x.text() for x in rest])
            body.append(f"  if {n0} = 0 then return {e.text()};")
            tail = self.expr(ctx, 1, calls=False)
            body.append(f"  return {name}({args}) + ({tail.text()});")
            for x in rest + [e, tail]:
                self.note_reads(p, x)
        elif rtype:
            e = self.pick(lambda: self.expr(ctx, r.randint(1, 3)))
            self.note_reads(p, e)
            body.append(f"  return {e.text()};")
        parts += body
        parts.append(f"end {name};")
        for c in p.children:
            p.reads |= {v for v in c.reads if v not in own}
            p.writes |= {v for v in c.writes if v not in own}
        p.features = set(self.cur) | {"procs"}
        self.cur = outer_cur
        return p, parts

    def note_reads(self, p: Proc, e: Expr) -> None:
        own = {n for n, _ in p.params + p.locals}
        p.reads |= {v for v in e.reads() if v not in own}
        for n in nodes(e):
            if isinstance(n, Call) and n.proc is not p:
                p.reads |= n.proc.reads
                p.writes |= n.proc.writes

    def proc_stmt(self, ctx: "Ctx") -> list[str]:
        """A statement in a procedure body: assigns its locals and
        parameters only."""
        p = ctx.proc
        own = [n for n, _ in p.locals + p.params if not (p.reentrant and n == p.params[0][0])]
        if not own:
            return []
        r = self.rnd.random()
        a = self.rnd.choice(own)
        if r < 0.5:
            e = self.pick(lambda: self.expr(ctx, self.rnd.randint(1, 3)), (a,))
            self.note_reads(p, e)
            return [f"  {a} = {e.text()};"]
        if r < 0.75:
            c = self.pick(lambda: self.cond(ctx))
            e = self.pick(lambda: self.expr(ctx, 2), (a,))
            self.note_reads(p, c)
            self.note_reads(p, e)
            return [f"  if {c.text()} then {a} = {e.text()};", f"  else {a} = {a} + 1;"]
        if r < 0.9 and len(own) >= 1:
            e = self.pick(lambda: self.expr(ctx, 2, calls=False), (a,))
            self.note_reads(p, e)
            n = self.rnd.randint(1, 5)
            return [f"  do case ({e.text()}) mod {n};"] + \
                [f"    {a} = {a} + {i * 3 + 1};" for i in range(n)] + ["  end;"]
        e = self.pick(lambda: self.expr(ctx, 1, calls=False), (a,))
        self.note_reads(p, e)
        # a loop on an index of its own
        lv = f"{p.name}i"
        return [f"  do {lv} = 1 to ({e.text()}) and 7;", f"    {a} = {a} + {lv};", "  end;"]

    # -- the whole program

    def program(self) -> Program:  # pylint: disable=too-many-locals,too-many-statements
        r = self.rnd
        head = [f"/* seed {self.seed}: a random program for scripts/intel_oracle.py */",
                "t: do;"]
        # LITERALLY: constants and a type.
        for i in range(r.randint(1, 3)):
            v = self.const_value()
            self.lits[f"k{i}"] = v
        lit_decl = ["declare " + ", ".join(f"{n} literally '{v}'" for n, v in
                                           sorted(self.lits.items()))
                    + ", word literally 'address';"]
        decls = [
            "mon1: procedure (f, a) external; declare f byte, a address; end mon1;",
            "declare (" + ", ".join(n for n, t in POOL.items() if t == B) + ") byte;",
            "declare (" + ", ".join(n for n, t in POOL.items() if t == A) + ") word;",
            "declare (sb, eb, rb, ix, jx) byte, (sw, ew, rw, rc, nc, kx, cnt, acc, lim, stp) "
            "address;",
            "declare (pb, pw) address, bb based pb byte, bw based pw address;",
        ]
        ab = [self.const_value(B) for _ in range(8)]
        aw = [self.const_value(A) for _ in range(8)]
        decls.append("declare ab(8) byte initial (" + ", ".join(_hex(v) for v in ab) + ");")
        decls.append("declare aw(8) address initial (" + ", ".join(_hex(v) for v in aw) + ");")
        decls.append("declare tb(8) byte, tw(4) address;")
        self.data["cb"] = [self.const_value(B) for _ in range(r.randint(1, 9))]
        self.data["cw"] = [self.const_value(A) for _ in range(r.randint(1, 5))]
        decls.append("declare cb(*) byte data (" + ", ".join(_hex(v) for v in self.data["cb"])
                     + ");")
        decls.append("declare cw(*) address data (" + ", ".join(_hex(v) for v in self.data["cw"])
                     + ");")
        self.msg = r.choice(["HELLO, WORLD$", "PL/M-80 V3.1$", "abc$", "x=1; y=2$"])
        self.extents = {"cb": (len(self.data["cb"]), 1), "cw": (len(self.data["cw"]), 2),
                        MSG: (len(self.msg), 1)}
        decls.append(f"declare ms(*) byte data ('{self.msg}');")
        decls.append("declare st structure (x byte, y address, z(4) byte);")
        decls.append("declare sa(4) structure (x byte, y address, z(2) byte);")
        parts: list = head + [self.chunk([x], "decl") for x in lit_decl + decls]
        parts += [self.chunk(block.split("\n"), "printer") for block in _split_procs(PRINTERS)]
        # dump: everything the statements change
        dump = ["dump: procedure;",
                "  call ph(b1); call ph(b2); call ph(b3); call ph(b4);",
                "  call ph(w1); call ph(w2); call ph(w3); call ph(w4);",
                "  call ph(sb); call ph(sw); call ph(eb); call ph(ew); call ph(nc);",
                "  call pa(.ab, size(ab)); call pa(.aw, size(aw));",
                "  call pa(.tb, size(tb)); call pa(.tw, size(tw));",
                "  call pa(.st, size(st)); call pa(.sa, size(sa));",
                "  call nl;",
                "end dump;"]
        parts.append(self.chunk(dump, "printer"))
        # procedures
        if "procs" not in self.avoid:
            for i in range(r.randint(2, 7)):
                p, pparts = self.make_proc(i + 1, None, list(self.procs), 0)
                self.procs.append(p)
                parts.append(self.chunk(pparts, "proc"))
        # the main program: set everything, then the statements
        init = ["nc = 0; eb = 0; ew = 0; rb = 0; rw = 0; rc = 0; sb = 0; sw = 0;",
                "ix = 0; jx = 0; kx = 0; cnt = 0; acc = 0; lim = 0; stp = 1;",
                "pb = .tb; pw = .tw;"]
        for name, t in POOL.items():
            if r.random() < 0.5:
                init.append(f"{name} = {_hex(self.const_value(t))};")
            else:
                arr = "ab" if t == B else "aw"
                init.append(f"{name} = {arr}({r.randrange(8)});")
        init.append("do ix = 0 to 7; tb(ix) = ab(7 - ix) xor 5ah; end;")
        init.append("do ix = 0 to 3; tw(ix) = aw(ix) + 1234h; end;")
        init.append("st.x = 1; st.y = 2345h; do ix = 0 to 3; st.z(ix) = ix * 3; end;")
        init.append("do ix = 0 to 3; sa(ix).x = ix; sa(ix).y = aw(ix + 4);"
                    " sa(ix).z(0) = ab(ix); sa(ix).z(1) = tb(ix); end;")
        parts += init
        ctx = Ctx(proc=None, readable=sorted(POOL), callable=list(self.procs))
        for i in range(self.n_stmts):
            parts.append(self.chunk(self.stmt(ctx)))
            if r.random() < 0.08:
                self.tag += 1
                parts.append(self.chunk([f"call pt({self.tag & 0xFF}); call dump;"], "dump"))
        self.tag += 1
        parts.append(f"call pt({self.tag & 0xFF}); call dump;")
        parts.append("end t;")
        return Program(self.seed, parts, self.features)


@dataclass
class Ctx:
    """Where a statement is: the procedure (None: the main program), what it
    may read and call."""
    proc: Proc | None
    readable: list
    callable: list


def generate(seed: int, n_stmts: int = 50, avoid=frozenset()) -> Program:
    """The program for ``seed``."""
    return Generator(seed, n_stmts, avoid).program()
