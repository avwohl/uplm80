"""A procedure's local variables keep their values from one call to the next.

PL/M-80 allocates a procedure's variables statically (Programming Manual,
8.1.7).  uplm80 overlays the locals of procedures that are never active
together in one block, ??AUTO, and used to put every uninitialised local
there, so a local a program counted on to keep its value - a first-time
flag, a running count, a position - came back with whatever another
procedure had left in its place.  Now a local shares ??AUTO only when every
call assigns it before anything reads it (uplm80/local_storage.py); every
other local is static, as the manual has it.

Each program here prints something else at every -O with 0.3.6, except
test_pointers_from_a_local_reach_the_locals_declared_after_it, which holds
what the new allocation must not break.  The rest check that the memory
saving is kept where it is safe.
"""

import re

import pytest

from uplm80.ast_optimizer import ASTOptimizer
from uplm80.codegen import CodeGenerator, Mode
from uplm80.compiler import Compiler
from uplm80.frontend import parse_source

from ._toolchain import run_asm, run_plm

LEVELS = (0, 1, 2, 3)

_PRELUDE = """0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
/* Never active with the procedures under test, so its locals are overlaid
   on theirs: it fills them with 0FFH, and prints a dot. */
other: procedure;
    declare (a, b, c) address;
    a = 0ffffh; b = a; c = a;
    call mon1(2, low(c) and '.');
end other;
"""


def _run(body: str, opt: int) -> str:
    return run_plm(_PRELUDE + body + "\nend t;\n", opt).replace("\0", "").strip()


def _asm(body: str, opt: int = 2) -> str:
    out = Compiler(opt_level=opt).compile(_PRELUDE + body + "\nend t;\n", "<test>")
    assert out is not None, "compilation failed"
    return out


def _static(asm: str) -> set[str]:
    """The procedure locals with storage of their own: `@PROC$NAME: ds n'."""
    return set(re.findall(r"^@(\w+\$\w+):\s+ds\b", asm, re.MULTILINE))


def _auto_size(asm: str) -> int:
    m = re.search(r"^\?\?AUTO:\s*\n\s*ds\s+(\d+)", asm, re.MULTILINE)
    return int(m.group(1)) if m else 0


# ---- locals that carry a value from one call to the next -------------------

COUNTER = """
tick: procedure address;
    declare (count, seen) address;
    if seen <> 1234h then do;
        seen = 1234h;
        count = 0;
    end;
    count = count + 1;
    return count;
end tick;
declare (i, n) address;
do i = 1 to 5;
    call other;
    n = tick;
end;
call mon1(2, '0' + n);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_count_kept_in_uninitialised_locals_survives_between_calls(opt):
    """The CHANGELOG's example: `tick' returned 1 every time."""
    assert _run(COUNTER, opt) == ".....5"


NESTED = """
p: procedure (v) byte;
    declare v byte;
    declare prev byte;
    get: procedure byte; return prev; end get;
    declare r byte;
    r = get;
    prev = v;
    return r;
end p;
declare (i, r) byte;
r = p('A');
do i = 'B' to 'E';
    call other;
    call mon1(2, p(i));
end;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_nested_procedure_reading_an_outer_local_reads_it_at_the_call(opt):
    """`p' assigns `prev' before it names it, but `get', called first,
    reads it: each call returns what the call before was given."""
    assert _run(NESTED, opt) == ".A.B.C.D"


GOTO = """
q: procedure (skip) byte;
    declare skip byte;
    declare v byte;
    if skip then go to use;
    v = 'A';
use:
    v = v + 1;
    return v;
end q;
declare i byte;
call mon1(2, q(0));
do i = 1 to 3;
    call other;
    call mon1(2, q(1));
end;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_past_the_assignment_reaches_the_read(opt):
    """`v = v + 1' follows `v = 'A'' in the text, but the GOTO skips it."""
    assert _run(GOTO, opt) == "B.C.D.E"


ADDRESS = """
show: procedure (p);
    declare p address, c based p byte;
    call mon1(2, c);
end show;
r: procedure (v);
    declare v byte;
    declare x byte;
    call show(.x);
    x = v;
end r;
declare i byte;
call r('A');
do i = 'B' to 'D';
    call other;
    call r(i);
end;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_local_read_through_its_address_is_static(opt):
    """`x' is assigned before it is read by name, and read through its
    address before that."""
    assert _run(ADDRESS, opt) == ".A.B.C"


BASED = """
declare msg (*) byte data ('XYZ');
peek: procedure (reset) byte;
    declare reset byte;
    declare p address, c based p byte;
    if reset then p = .msg(1);
    return c;
end peek;
declare i byte;
call mon1(2, peek(1));
do i = 1 to 3;
    call other;
    call mon1(2, peek(0));
end;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_based_variable_reads_its_base(opt):
    """`return c' reads `p', which only the first call assigns."""
    assert _run(BASED, opt) == "Y.Y.Y.Y"


WHILE = """
w: procedure (n) byte;
    declare n byte;
    declare (i, last) byte;
    i = 0;
    do while i < n;
        last = 'A' + i;
        i = i + 1;
    end;
    return last;
end w;
call mon1(2, w(3));
call other;
call mon1(2, w(0));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_loop_that_may_not_run_assigns_nothing(opt):
    assert _run(WHILE, opt) == "C.C"


CASE = """
k: procedure (sel) byte;
    declare sel byte;
    declare mode byte;
    do case sel;
        mode = 'X';
        mode = 'Y';
        ;
    end;
    return mode;
end k;
call mon1(2, k(1));
call other;
call mon1(2, k(2));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_case_that_does_not_assign_leaves_the_last_value(opt):
    assert _run(CASE, opt) == "Y.Y"


LAYOUT = """
fill2: procedure (p, v);
    declare p address, v byte, b based p (2) byte;
    b(0) = v; b(1) = v + 1;
end fill2;
f: procedure;
    declare (a, b) byte;
    declare c byte;
    declare d byte;
    b = '?'; d = '?';
    call fill2(.a, 'A');
    call fill2(.c, 'C');
    call mon1(2, b);
    call mon1(2, d);
end f;
call f;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_pointers_from_a_local_reach_the_locals_declared_after_it(opt):
    """`b' and `d' are assigned before they are read, but `.a' and `.c'
    reach them: a factored declaration is contiguous (6.2.4), and code that
    runs a pointer on from a variable expects what DRI declares after it.
    Taking only `a' and `c' out of ??AUTO printed `??'."""
    assert _run(LAYOUT, opt) == "BD"


# ---- declaration order ------------------------------------------------------

FOLDED_RUN = """
k2: procedure byte;
    declare arr (2) byte, nxt byte;
    arr(0) = 1; arr(1) = 2;
    arr(1 + 1) = 'S';
    return nxt;
end k2;
call mon1(2, k2);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_constant_subscript_expression_is_a_constant_at_every_level(opt):
    """`arr(1 + 1)' is `nxt', declared after `arr' (the gate's a4 k2).  -O1
    and up fold it before the analysis sees it and found it outside
    `arr'; at -O0 it was a variable subscript, `arr' stayed in ??AUTO, and
    the store went into ??AUTO, not into `nxt'."""
    assert _run(FOLDED_RUN, opt) == "S"


FOLDED = """
k7: procedure byte;
    declare arr (2) byte, nxt byte;
    arr(0) = 1; arr(2 - 1) = 2;
    nxt = arr(0) + arr(1);
    return nxt;
end k7;
k8: procedure byte;
    declare arr (3) byte, nxt byte;
    arr(0) = 1; arr(1) = 2; arr(last(arr)) = 3;
    nxt = arr(0) + arr(1) + arr(2);
    return nxt;
end k8;
k9: procedure byte;
    declare arr (3) byte, nxt byte;
    arr(0) = 1; arr(1) = 2; arr(-1) = 3;
    nxt = arr(0) + arr(1);
    return nxt;
end k9;
call mon1(2, k7 + k8 + k9);
"""


def test_every_level_decides_the_same_for_a_constant_subscript():
    """`arr(2 - 1)' and `arr(last(arr))' are elements, so the arrays are
    assigned whole and may share ??AUTO; `arr(-1)' is `arr(255)', past the
    end.  -O0 took the first two for variable subscripts and the last for
    one inside the array; LAST was not folded at any level."""
    for opt in LEVELS:
        asm = _asm(FOLDED, opt)
        assert _static(asm) == {"K9$ARR", "K9$NXT"}, (opt, asm)


RUN_ON = """
k1: procedure (n) byte;
    declare n byte;
    declare arr (2) byte, nxt (2) byte;
    declare i byte;
    arr(0) = 1; arr(1) = 2;
    do i = 0 to n; arr(i) = 'R'; end;
    return nxt(1);
end k1;
call mon1(2, k1(3));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_variable_subscript_runs_on_into_the_local_declared_after(opt):
    """DRI lays a procedure's locals out in declaration order, and so did
    0.3.6, in its frame in ??AUTO: `arr(3)' is `nxt(1)' (the gate's a4, k1).
    `nxt' is read before it is assigned, so it is static; `arr' was left in
    ??AUTO, and the stores went there."""
    assert _run(RUN_ON, opt) == "R"


RUN_ON_READ = """
rd: procedure (n) byte;
    declare n byte;
    declare arr (2) byte, nxt byte;
    declare r byte;
    arr(0) = 1; arr(1) = 2;
    r = arr(n);
    nxt = 'X';
    return r;
end rd;
call mon1(2, rd(0) + 'A' - 1);
call other;
call mon1(2, rd(2));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_read_through_a_variable_subscript_reads_what_it_runs_on_into(opt):
    """`arr(2)' is `nxt', which the call before left 'X'.  Every local is
    assigned before it is read by name, so all three were overlaid, and
    `other' filled them with 0FFH."""
    assert _run(RUN_ON_READ, opt) == "A.X"


RUN_ON_AUTO = """
w: procedure byte;
    declare arr (2) byte, nxt byte, i byte;
    nxt = '?';
    arr(0) = 0; arr(1) = 0;
    do i = 0 to 2; arr(i) = 'W'; end;
    return nxt;
end w;
call mon1(2, w);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_locals_run_on_into_may_share_auto_together(opt):
    """Nothing here is read before it is assigned, so `arr', `nxt' and `i'
    share ??AUTO - together, in declaration order."""
    assert _run(RUN_ON_AUTO, opt) == "W"
    asm = _asm(RUN_ON_AUTO, opt)
    assert _static(asm) == set(), asm
    # w's frame: arr at +0, nxt at +2, i at +3 (other's is overlaid on it).
    offsets = [int(m) for m in re.findall(r"\?\?AUTO\+(\d+)", asm)]
    assert {0, 2, 3} <= set(offsets), asm


RUN_ON_INITIAL = """
fx: procedure (n) byte;
    declare n byte;
    declare arr (2) byte initial (1, 2);
    declare nxt byte;
    nxt = '?';
    arr(n) = 'F';
    return nxt;
end fx;
call mon1(2, fx(2));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_an_initialised_array_runs_on_into_the_local_declared_after(opt):
    """`arr' is static for its INITIAL, so `nxt', which `arr(2)' is, has to
    follow it there."""
    assert _run(RUN_ON_INITIAL, opt) == "F"


ADDRESS_OF_INITIAL = """
ax: procedure byte;
    declare v byte initial (0);
    declare nxt byte;
    declare pp address, c based pp byte;
    nxt = '?';
    pp = .v + 1;
    c = 'I';
    return nxt;
end ax;
call mon1(2, ax);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_pointer_from_an_initialised_local_reaches_the_next_one(opt):
    """A pointer run on from a local whose address is taken reaches what
    is declared after it; `v' is static for its INITIAL, and was not
    counted."""
    assert _run(ADDRESS_OF_INITIAL, opt) == "I"


STATIC_ORDER = """
so: procedure (n) byte;
    declare n byte;
    declare first byte;
    declare arr (2) byte;
    declare mid byte initial (7);
    declare (x2, y2) byte;
    declare last2 address;
    arr(n) = 1;
    return first + arr(0) + mid + x2 + y2 + low(last2);
end so;
call mon1(2, so(1));
"""


def test_the_static_locals_keep_declaration_order():
    """From `arr' on, everything is static and contiguous, in the order it
    is declared, the INITIAL `mid' among the rest."""
    for opt in LEVELS:
        asm = _asm(STATIC_ORDER, opt)
        data = asm[asm.index("dseg"):]
        labels = re.findall(r"^(@SO\$\w+):", data, re.MULTILINE)
        assert labels == ["@SO$FIRST", "@SO$ARR", "@SO$MID", "@SO$X2", "@SO$Y2",
                          "@SO$LAST2"], (opt, asm)
        block = data[data.index("@SO$FIRST:"):data.index("@SO$LAST2:")]
        assert re.findall(r"^([@?\w$]+):", block, re.MULTILINE) == labels[:-1], (opt, asm)


# ---- parameters -------------------------------------------------------------

PARAM_KEPT = """
declare keep address, kept based keep byte;
sv: procedure (v);
    declare v byte;
    keep = .v;
end sv;
call sv('P');
call other;
call mon1(2, kept);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_pointer_to_a_parameter_kept_after_the_call(opt):
    """PL/M-80 allocates a parameter statically, like any other local, so
    `.v' still points at 'P' after `sv' has returned; in ??AUTO, `other'
    wrote 0FFH over it."""
    assert _run(PARAM_KEPT, opt) == ".P"


PARAM_RUN_ON = """
sq: procedure (v) byte;
    declare v byte;
    declare nxt byte;
    declare pp address, c based pp byte;
    pp = .v + 1;
    c = 'Q';
    return nxt;
end sq;
call mon1(2, sq(0));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_pointer_run_on_from_a_parameter_reaches_the_first_local(opt):
    """The parameters come first, then the locals: `.v + 1' is `nxt'.  `v'
    stayed in ??AUTO while `nxt', read before it is assigned, was static,
    and the store went into `pp'."""
    assert _run(PARAM_RUN_ON, opt) == "Q"


NESTED_PARAM = """
outer: procedure (w) byte;
    declare w byte;
    inner: procedure (v) byte;
        declare v byte;
        declare pp address, c based pp byte;
        pp = .v;
        return c;
    end inner;
    return inner(w + 1);
end outer;
pub: procedure (v) byte public;
    declare v byte;
    declare pp address, c based pp byte;
    pp = .v;
    return c + 1;
end pub;
call mon1(2, outer('M'));
call mon1(2, pub('N'));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_static_parameter_of_a_nested_and_of_a_public_procedure(opt):
    """A static parameter is where the caller stores it: the nested
    procedure's label has the procedures it is in, and a PUBLIC one takes
    its arguments off the stack into it."""
    assert _run(NESTED_PARAM, opt) == "NO"
    asm = _asm(NESTED_PARAM, opt)
    assert re.search(r"^@OUTER\$INNER\$V:\s+ds\s+1$", asm, re.MULTILINE), asm
    assert "PUB$V" in _static(asm), asm


def test_a_procedure_whose_address_is_taken_keeps_the_parameters_it_names():
    """`show' can be called through its address after `p' has returned,
    and reads `v', which is `p''s parameter."""
    asm = _asm("""
p: procedure (v) address;
    declare v byte;
    show: procedure; call mon1(2, v); end show;
    return .show;
end p;
declare f address;
f = p('A');
call other;
call f;
""")
    assert _static(asm) == {"P$V"}, asm


# ---- which frames ??AUTO may overlay ---------------------------------------

def _generator(body: str, opt: int = 2) -> CodeGenerator:
    """The code generator, once it has compiled ``body``."""
    ast = parse_source(_PRELUDE + body + "\nend t;\n", "<test>")
    if opt:
        ast = ASTOptimizer(opt).optimize(ast)
    cg = CodeGenerator(Mode.CPM)
    cg.generate(ast)
    return cg


def _frames(body: str, opt: int = 2) -> dict[str, tuple[int, int]]:
    """Each procedure's frame in ??AUTO, as [start, end)."""
    cg = _generator(body, opt)
    out = {}
    for proc, storage in cg.proc_storage.items():
        size = sum(entry[1] for entry in storage)
        if size:
            out[proc] = (cg.storage_offsets[proc], cg.storage_offsets[proc] + size)
    return out


def _overlap(frames: dict, a: str, b: str) -> bool:
    (s1, e1), (s2, e2) = frames[a], frames[b]
    return s1 < e2 and s2 < e1


INTERRUPT = """
helper: procedure;
    declare h address;
    h = 5;
    call mon1(2, low(h));
end helper;
ih: procedure interrupt 7;
    declare (x, y) byte;
    x = 1; y = x + 1;
    call helper;
end ih;
work: procedure byte;
    declare (w1, w2, w3) byte;
    w1 = 1; w2 = 2; w3 = w1 + w2;
    return w3;
end work;
call mon1(2, work);
call other;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_an_interrupt_procedure_overlays_no_other_frame(opt):
    """An INTERRUPT procedure runs when the interrupt comes, while `work'
    or `other' may be active, and so does `helper', which it calls
    (8.1.7).  Nothing calls `ih', so its frame, and `helper''s, were put
    over theirs."""
    frames = _frames(INTERRUPT, opt)
    for anytime in ("IH", "HELPER"):
        for proc in frames:
            if proc != anytime:
                assert not _overlap(frames, anytime, proc), (anytime, proc, frames)


CALLBACK_ASM = """
    .z80
    cseg
    public EXT, ICALL
    extrn CB
; ext: code outside the module that calls its PUBLIC procedure back.
EXT:    jp CB
; icall(a): calls the procedure at a.
ICALL:  ld hl,2
        add hl,sp
        ld a,(hl)
        inc hl
        ld h,(hl)
        ld l,a
        jp (hl)
    end
"""

CALLBACK = """
ext: procedure external; end ext;
icall: procedure (a) external; declare a address; end icall;
cb: procedure public;
    declare (u, v, w) address;
    u = 0ffffh; v = u; w = v;
end cb;
cb2: procedure;
    declare (u, v, w) address;
    u = 0ffffh; v = u; w = v;
end cb2;
declare f address;
back: procedure byte;
    declare (k, m) byte;
    k = 'E'; m = 'X';
    call ext;
    return k;
end back;
through: procedure byte;
    declare (k, m) byte;
    k = 'F'; m = 'Y';
    call icall(f);
    return k;
end through;
f = .cb2;
call mon1(2, back);
call mon1(2, through);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_code_outside_the_module_may_call_back_into_it(opt):
    """`ext' calls the PUBLIC `cb' back, and `icall' calls `cb2', whose
    address it was given.  Neither call is in the text, so `cb' and `cb2'
    were never active with `back' and `through', and their frames were
    put over those, whose `k' came back 0FFH."""
    asm = Compiler(opt_level=opt).compile(_PRELUDE + CALLBACK + "\nend t;\n", "<test>")
    assert asm is not None
    out = run_asm(asm, CALLBACK_ASM).stdout.replace("\0", "").strip()
    assert out == "EF", asm


def test_a_call_through_an_address_may_call_any_procedure_whose_address_is_taken():
    """`CALL f' calls the procedure at the address in `f' (8.2.1): `tgt',
    whose address is taken, may be active with `caller' and whatever calls
    it; `lone', whose address is not, need not be."""
    body = """
declare f address;
tgt: procedure;
    declare (u, v, w) address;
    u = 1; v = u; w = v;
end tgt;
lone: procedure;
    declare (u, v, w) address;
    u = 2; v = u; w = v;
end lone;
caller: procedure byte;
    declare (k, m) byte;
    k = 'G'; m = 'Z';
    call f;
    return k;
end caller;
outer: procedure byte;
    declare (n1, n2) byte;
    n1 = 1; n2 = caller;
    return n1 + n2;
end outer;
f = .tgt;
call lone;
call mon1(2, outer);
"""
    frames = _frames(body)
    assert not _overlap(frames, "TGT", "CALLER"), frames
    assert not _overlap(frames, "TGT", "OUTER"), frames
    together = _generator(body).can_be_active_together
    assert "LONE" not in together["CALLER"] | together["OUTER"], together


def test_a_bdos_call_is_no_call_back():
    """MON1 and MON2 with a constant function are the BDOS itself, which
    calls nothing back: a PUBLIC procedure may still share ??AUTO with a
    procedure that makes such a call."""
    frames = _frames("""
cb: procedure public;
    declare (u, v, w) address;
    u = 0ffffh; v = u; w = v;
end cb;
p: procedure;
    declare (k, m) byte;
    k = 'E'; m = k;
    call mon1(2, m);
end p;
call p;
call cb;
""")
    assert _overlap(frames, "CB", "P"), frames


# ---- the memory saving, where it is safe -----------------------------------

def test_locals_assigned_before_they_are_read_still_share_auto():
    """`other' assigns all three of its locals first, so they stay in
    ??AUTO; `tick' keeps `count' and `seen' to itself."""
    asm = _asm(COUNTER)
    assert _static(asm) == {"TICK$COUNT", "TICK$SEEN"}, asm
    assert _auto_size(asm) == 6, asm


def test_a_nested_reader_called_after_the_assignment_is_no_read_before_it():
    asm = _asm("""
p: procedure;
    declare x byte;
    show: procedure; call mon1(2, x); end show;
    x = 'A';
    call show;
end p;
call p;
""")
    assert _static(asm) == set(), asm


def test_parameters_always_share_auto():
    """Every call assigns them."""
    asm = _asm("""
p: procedure (a, b) byte;
    declare (a, b) byte;
    return a + b;
end p;
call mon1(2, p(1, 2));
""")
    assert _static(asm) == set(), asm
    assert _auto_size(asm) == 6, asm


ARRAYS = """
p: procedure (i) byte;
    declare i byte;
    declare part (3) byte;
    declare whole (3) byte;
    declare s structure (x byte, y (2) address);
    declare (j, sum) byte;
    part(0) = 1; part(2) = 3;
    whole(0) = 1; whole(1) = 2; whole(2) = 3;
    s.x = 1; s.y(0) = 2; s.y(1) = 3;
    j = 0; sum = 0;
    sum = whole(i) + part(0) + part(2) + s.x + low(s.y(i));
    sum = sum + part(1) + j;
    return sum;
end p;
q: procedure (i) byte;
    declare i byte;
    declare loop (3) byte, k byte;
    do k = 0 to 2; loop(k) = k; end;
    return loop(i);
end q;
call mon1(2, p(1) + q(1));
"""


def test_an_array_is_assigned_once_every_element_has_been():
    """An array or structure assigned element by element through constant
    subscripts is assigned when the last element is; a read of an element
    through a constant subscript needs only that element, and `part(1)'
    reads the one that was not assigned.  A read through a variable
    subscript may run on into the locals declared after the array, and
    they are all assigned by then.  A store through a variable subscript
    assigns nothing, so `loop' is static, and `k', which an overrun of
    `loop' reaches, with it."""
    asm = _asm(ARRAYS)
    assert _static(asm) == {"P$PART", "Q$LOOP", "Q$K"}, asm


def test_a_factored_declaration_stays_together():
    """`y' is fine, but `x' is not, and the two are one declaration."""
    asm = _asm("""
p: procedure byte;
    declare (x, y) byte;
    declare z byte;
    y = 1; z = 2;
    x = x + y + z;
    return x;
end p;
call mon1(2, p);
""")
    assert _static(asm) == {"P$X", "P$Y"}, asm


def test_a_procedure_whose_address_is_taken_keeps_what_it_names_static():
    """A call through the address can come at any time: here after `p' has
    returned, when `saved' would be someone else's."""
    asm = _asm("""
p: procedure (v) address;
    declare v byte;
    declare saved byte;
    show: procedure; call mon1(2, saved); end show;
    saved = v;
    return .show;
end p;
declare f address;
f = p('A');
call other;
call f;
""")
    assert _static(asm) == {"P$SAVED"}, asm


ICALL_ASM = """
    .z80
    cseg
    public ICALL
; icall(a): calls the procedure at a.
ICALL:  ld hl,2
        add hl,sp
        ld a,(hl)
        inc hl
        ld h,(hl)
        ld l,a
        jp (hl)
    end
"""

CALLED_THROUGH_ADDRESS = """
icall: procedure (a) external; declare a address; end icall;
lc: procedure;
    declare (n, m) byte;
    rd: procedure byte; return n; end rd;
    n = 'L';
    m = rd;
    call mon1(2, m);
end lc;
declare f address;
f = .lc;
call icall(f);
call other;
call icall(f);
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_called_through_its_address_keeps_its_own_locals_in_auto(opt):
    """`rd' names `n', but `rd' runs only while `lc' is active, however
    `lc' was called, and `lc' assigns `n' before it calls `rd': only the
    locals of the procedures `lc' is nested in, which a call through its
    address may find inactive, have to be static.  (MP/M II's LOAD takes
    the address of LOADCOM, and four of RELOC's locals went static.)"""
    asm = _asm(CALLED_THROUGH_ADDRESS, opt)
    assert _static(asm) == set(), asm
    out = run_asm(asm, ICALL_ASM).stdout.replace("\0", "").strip()
    assert out == "L.L", asm


def test_storage_that_is_not_in_auto_gets_no_slot_there():
    """INITIAL, BASED, AT and LABEL declarations, and a REENTRANT procedure's
    parameters and locals, live elsewhere; each used to be given a slot in
    ??AUTO as well, which it never used."""
    asm = _asm("""
declare buf (4) byte;
p: procedure;
    declare k byte initial (5);
    declare q address, z based q byte;
    declare y byte at (.buf(1));
    declare lab label;
    q = .buf;
    z = k + y;
lab: ;
end p;
r: procedure (n) byte reentrant;
    declare n byte;
    declare m byte;
    m = n;
    return m;
end r;
call p;
call mon1(2, r(1));
""")
    assert _auto_size(asm) == 6, asm   # other's (a, b, c) and nothing else
