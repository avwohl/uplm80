"""Which declaration each name in a program means, and what the assembler
calls it.

PL/M-80 gives every declared name - variable, procedure, label, macro - the
block that declares it for its scope, less any block nested in it that
declares the name again (Programming Manual 9800268B, chapter 9); every DO
block is a block, and so is every procedure.  Code generation names what it
emits the simple way: a module-level name as it is, a procedure's own
names and labels as ``@proc$name``, a DO block's variables as
``@proc$Bn$name``, a LITERALLY as an EQU of its name; and it finds a name
by looking it up as a procedure nested in each enclosing procedure, then
through its scopes, which hold everything but procedures.  That gives one
name, and the right one, per declaration only as long as no two
declarations it treats alike are in scope at once, and PL/M-80 lets them
be:

* a label in each of two DO blocks of a procedure (both ``@proc$L``), or
  of the main program (both ``L``);
* a procedure declared in each of two DO blocks of one procedure (both
  ``P$Q``), or in a DO block, named like a variable declared elsewhere;
* a LITERALLY in each of two procedures (two EQUs of one name);
* in a multi-file compile, a private name in each of two modules, which
  the modules' separate name spaces keep apart when they are compiled
  one at a time.

:func:`resolve_names` finds every declaration and binds every name to the
one PL/M-80 means, renames a declaration code generation would confuse
with another - ``L`` to ``L?2``; no PL/M-80 identifier contains a ``?``,
so a new name cannot meet one the program has - and qualifies, in a
multi-file compile, each module's private module-level names with the
module's own (``LIB?HELPER``).  A program in which no two declarations
meet is left exactly as it was.  It also checks every GOTO against the
rules of 9.3, tells code generation the label each one jumps to, and
rejects a name declared twice in one block.

:func:`data_name` and :func:`fix_symbols` keep names away from what um80
reads as a register, a condition or an operator.  um80 0.3.51 reads a name
spelled like an operator or a condition, and one ending in an operator's
letters before a + or a -, as the symbol, as M80 does; the renames and
rewrites for those are kept for older um80 releases, and cost nothing.  A
register name is still not a symbol to it in Z80 code.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from . import _plm_parser as P
from .ast_view import (
    BinaryOpKind,
    DataType,
    UnaryOpKind,
    binop_kind,
    decl_attrs,
    decl_item_struct_members,
    decl_item_type,
    dotted_ident_parts,
    expr_text,
    ident_text,
    is_end_of_block,
    literally_value,
    parse_plm_number,
    proc_attrs,
    proc_return_type,
    string_value,
    struct_member_dim,
    struct_member_names,
    struct_member_type,
    unop_kind,
    unwrap_paren,
)
from .errors import CodeGenError
from .frontend import source_location

# Names um80 reads as something other than a symbol.  A register, as an
# operand (`ld hl,A' is "Register 'A' used as value", `ld a,(IX)' is
# `ld a,(ix+0)'), and, before 0.3.51, an operator of its expressions: `call
# EQ' called 0FFFFH and `ld hl,SHL' loaded 0, without a word.
REGISTER_NAMES = frozenset(
    {"A", "B", "C", "D", "E", "H", "L", "M", "SP", "PSW",
     "AF", "BC", "DE", "HL", "IX", "IY", "I", "R"})
OPERATOR_NAMES = frozenset({"EQ", "NE", "LT", "LE", "GT", "GE", "SHL", "SHR", "NUL"})


def data_name(name: str) -> str:
    """The assembler's name for a variable, a procedure, a label or a
    LITERALLY called ``name``."""
    if name.upper() in REGISTER_NAMES or name.upper() in OPERATOR_NAMES:
        return f"@{name}"
    return name


# As the target of a jump, a condition, before um80 0.3.51: `jp P' wanted an
# address after the P (and M80 3.44 takes `JP P' to a P defined further
# down for one byte in its first pass).  A procedure is jumped to as well as
# called (the peephole turns `call p / ret' into `jp P'), and a label is.
# `call P' and `ld hl,P' are the symbol, so such a name is not renamed; the
# jump is written `jp 0+P' (fix_symbols).
_JUMP_TO_CONDITION = re.compile(
    r"^(\s*(?:[\w?@$.]+:)?\s*(?:jp|jr)\s+)(Z|NZ|NC|PO|PE|P)(\s*(?:;.*)?)$",
    re.IGNORECASE | re.MULTILINE)


# And a symbol followed by a + or a - where the letters it ends in are one of
# um80's word operators: um80 before 0.3.51 took the + for the sign of that
# operator's operand, so `ld hl,TYPE+2' was TYPE(+2), the type of the
# expression +2, and `@P$NUL+2', `LIB?EQ+1' and `X1LOW-1' did not parse.
# Bare, such a symbol is read as a symbol (but for the operators themselves,
# which data_name renames), and MP/M II's PIP has a variable TYPE, so they
# are not renamed; an offset from one is written the other way round,
# `2+TYPE' (fix_symbols).
_UM80_WORDS = frozenset({"MOD", "SHL", "SHR", "AND", "OR", "XOR", "NOT", "EQ", "NE", "LT",
                         "LE", "GT", "GE", "HIGH", "LOW", "NUL", "TYPE"})
_WORD_THEN_SIGN = re.compile(r"(?:" + "|".join(sorted(_UM80_WORDS)) + r")[+-]", re.IGNORECASE)
_SYMBOL_OFFSETS = re.compile(r"(?<![\w?@$.])([A-Za-z_?@$.][\w?@$.]*)((?:[+-][\w?@$.]+)+)")


def fix_symbols(asm: str) -> str:
    """``asm`` with every `TYPE+n', `@P$NUL-n', ... written `n+TYPE',
    `-n+@P$NUL', and every `jp P', `jr Z' ... `jp 0+P', `jr 0+Z', which
    um80 reads as the symbol (and an offset)."""
    asm = _JUMP_TO_CONDITION.sub(r"\g<1>0+\2\3", asm)
    if not _WORD_THEN_SIGN.search(asm):
        return asm
    lines = asm.split("\n")
    for i, line in enumerate(lines):
        if _WORD_THEN_SIGN.search(line):
            lines[i] = _fix_line(line)
    return "\n".join(lines)


def _fix_line(line: str) -> str:
    """One line, left alone in its strings and its comment."""
    out: list[str] = []
    code = ""
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            out.append(_SYMBOL_OFFSETS.sub(_swap, code))
            code = ""
            out.append(ch)
            quote = ch
        elif ch == ";":
            out.append(_SYMBOL_OFFSETS.sub(_swap, code))
            code = ""
            out.append(ch)
            quote = "\n"       # the comment runs to the end of the line
        else:
            code += ch
    out.append(_SYMBOL_OFFSETS.sub(_swap, code))
    return "".join(out)


def _swap(m: re.Match) -> str:
    sym, offsets = m.groups()
    tail = re.search(r"[A-Za-z]+$", sym)
    if tail is None or tail.group(0).upper() not in _UM80_WORDS:
        return m.group(0)
    return f"{offsets.removeprefix('+')}+{sym}"


GOTO_RULE = ("PL/M-80 allows a GOTO out of a procedure only to a label at the "
             "outer level of the main program module (Programming Manual "
             "9800268B, 8.1.3 and 9.3)")
SCOPE_RULE = ("a GOTO reaches only a label in its own block or in a block "
              "enclosing it (Programming Manual 9800268B, 9.3)")
# A GOTO out of a procedure to a label in a DO block of the main program:
# the manual does not allow it (the outer level of a module is its
# exclusive extent, 10.1), but Intel's PL/M-80 V3.1 compiles it without a
# word, to a plain JMP, and uplm80 0.3.6 did the same.
DO_BLOCK_GOTO = ("PL/M-80 allows a GOTO out of a procedure only to a label at the "
                 "outer level of the main program module, which a DO block is not "
                 "(Programming Manual 9800268B, 9.3 and 10.1); compiled as Intel's "
                 "PL/M-80 compiles it, a plain jump that leaves on the stack what "
                 "the calls it abandons pushed")

# What every module can name without declaring it (symbols.SymbolTable).
_BUILTINS = frozenset(
    {"INPUT", "OUTPUT", "LOW", "HIGH", "DOUBLE", "LENGTH", "LAST", "SIZE", "SHL", "SHR",
     "ROL", "ROR", "SCL", "SCR", "MOVE", "TIME", "CARRY", "SIGN", "ZERO", "PARITY", "DEC",
     "MEMORY", "STACKPTR", "CPUTIME"})
# The built-ins with a type, which a CALL does not call (Intel's PL/M-80
# V3.1: ERROR 129, ILLEGAL 'CALL' WITH TYPED PROCEDURE).
_TYPED_BUILTINS = frozenset(
    {"INPUT", "LOW", "HIGH", "DOUBLE", "LENGTH", "LAST", "SIZE", "SHL", "SHR", "ROL", "ROR",
     "SCL", "SCR", "CARRY", "SIGN", "ZERO", "PARITY", "DEC", "STACKPTR"})
# How many arguments each built-in procedure takes (Intel's PL/M-80 V3.1:
# ERROR 154, INVALID NUMBER OF ARGUMENTS IN CALL, TOO FEW, and 153, TOO
# MANY); LENGTH, LAST and SIZE take a variable, and INPUT and OUTPUT a port.
_BUILTIN_ARGS = {"CARRY": 0, "ZERO": 0, "SIGN": 0, "PARITY": 0, "STACKPTR": 0, "LOW": 1,
                 "HIGH": 1, "DOUBLE": 1, "DEC": 1, "TIME": 1, "ROL": 2, "ROR": 2, "SHL": 2,
                 "SHR": 2, "SCL": 2, "SCR": 2, "MOVE": 3}
_HOW_MANY = ("no argument", "one argument", "two arguments", "three arguments")

_DO_BLOCKS = (P.DoBlock, P.DoWhileBlock, P.DoIterBlock, P.DoIterByBlock, P.DoCaseBlock)


@dataclass(eq=False)
class _Decl:  # pylint: disable=too-many-instance-attributes
    """One declaration.  ``sites`` are the (node, field) pairs that spell
    its name where it is declared; the references bound to it are kept
    beside it (:attr:`_Resolver.refs_of`)."""

    name: str
    kind: str                   # "var", "param", "proc", "label", "lit"
    block: "_Block"
    order: int
    sites: list = field(default_factory=list)
    public: bool = False
    external: bool = False
    storage: bool = True        # a variable with a label of its own
    literal: str | None = None
    defined: bool = False       # a label that labels a statement
    reentrant: bool = False     # a REENTRANT procedure
    from_proc: bool = False     # a label a GOTO in a procedure jumps to
    orig: str = ""              # the name as declared
    dim: int | None = None      # a variable's: None for a scalar
    members: dict | None = None     # a structure's: member -> its dimension
    node: object = None         # a procedure's declaration
    address: bool = False       # a variable's or a parameter's type is ADDRESS
    address_members: frozenset = frozenset()    # a structure's ADDRESS members
    item: object = None         # a variable's DeclItem, or DeclItemBasedGroup
    based: bool = False         # a BASED variable
    # A parameter's DeclItem, and how many declarations were made before it
    typed: tuple | None = None

    def __post_init__(self) -> None:
        self.orig = self.name

    @property
    def shared(self) -> bool:
        """PUBLIC or EXTERNAL: named by the linker, so never renamed."""
        return self.public or self.external


@dataclass(eq=False)
class _Module:
    index: int
    name: str
    main: bool = False          # has executable statements at module level
    qualifier: str = ""
    first: object = None        # its first item


def _file_name(module) -> str | None:
    """A module without a name of its own goes by its file's, made a name."""
    path = getattr(module, "uplm80_file", None)
    if not path:
        return None
    stem = re.sub(r"[^A-Z0-9_]", "_", os.path.splitext(os.path.basename(path))[0].upper())
    return stem if stem[:1].isalpha() else f"M{stem}"


@dataclass(eq=False)
class _Block:
    kind: str                   # "module", "proc", "do", "global"
    parent: "_Block | None"
    module: _Module | None
    proc: _Decl | None          # the procedure it is in (its own, for a body)
    decls: dict = field(default_factory=dict)


@dataclass(eq=False)
class _Ref:  # pylint: disable=too-many-instance-attributes
    block: _Block
    node: object
    attr: str
    goto: bool = False
    decl: _Decl | None = None
    dot: bool = False           # the operand of a dot, `.name'
    in_list: bool = False       # in a DATA or INITIAL list
    empty: bool = False         # followed by empty parentheses, `name()'
    in_at: bool = False         # in an AT clause
    # the member or the subscripted reference it starts that empty
    # parentheses follow, `s.m()', `a(1)()'
    after: object = None
    args: int | None = None     # how many subscripts or arguments follow it
    extent: object = None       # the call of LENGTH, LAST or SIZE it is the argument of
    call: object = None         # the subscripted reference or call it names
    called: bool = False        # a CALL's: `call q(1, 2)' of an ADDRESS calls through it
    member: object = None       # the member of what it names, `s.m', if any
    target: bool = False        # what an assignment stores to, `x = ...', `(x := ...)'
    # The base it starts, `p' or `s.m': (the BASED variable's declaration,
    # the base, how many declarations were made before its declaration's)
    base: tuple | None = None


class _MemberUse(NamedTuple):
    """A structure's member used, `s.m', and how, as a _Ref has it."""

    node: object                # the member access
    block: _Block
    dot: bool
    extent: object
    args: int | None            # how many subscripts or arguments follow it
    call: object                # the subscripted reference or call it names
    called: bool                # what a CALL statement calls


def _key(tok) -> str:
    return ident_text(tok).upper()


def _retext(tok, text: str):
    """``tok`` spelling ``text``."""
    if dataclasses.is_dataclass(tok):
        return dataclasses.replace(tok, text=text)
    return type(tok)(text, tok.name, tok.kind)


class _Resolver:  # pylint: disable=too-many-instance-attributes
    """The declarations of one compilation and what each name is bound to."""

    def __init__(self) -> None:
        self.decls: list[_Decl] = []
        self.refs: list[_Ref] = []
        self.warnings: list[tuple] = []     # (location, text)
        self.modules: list[_Module] = []
        self.globals = _Block("global", None, None, None)
        self.refs_of: dict[int, list[_Ref]] = {}
        # What Intel's PL/M-80 V3.1 rejects and uplm80 compiles, because a
        # program written for uplm80 relies on it: (location, text).
        self.intel_warnings: list[tuple] = []
        self.used: set[str] = set()     # every name a declaration has
        self._lists = 0                 # DATA and INITIAL lists being visited
        self._at = 0                    # AT clauses being visited
        # The restricted expressions (check_restricted): ("list", the
        # values of a DATA or INITIAL list), ("at", an AT address) or
        # ("consts", a constant list).
        self.restricted: list[tuple] = []
        # What each DATA or INITIAL value fills, by id: "byte", "word", or
        # "past" the space its declaration has; and the values that do not
        # fit in it (_value_slots).
        self._slots: dict[int, str] = {}
        self._past: set[int] = set()
        # Structure members used, as (member access, block, the reference's
        # _Ref-like flags): dot, the LENGTH/LAST/SIZE call, subscripted.
        self.member_uses: list[tuple] = []
        # The calls of LENGTH, LAST or SIZE (if the program does not
        # declare the name), by id: (call, the _Ref of its name).
        self.extent_calls: dict[int, tuple] = {}
        self._do_labels: dict[int, set[str]] = {}   # a DO block's labels, by id
        # Subscripts or arguments after subscripts or arguments, `a(1)(2)'.
        self.chains: list = []
        # Whether to hold the program to what Intel's PL/M-80 V3.1 takes
        # (intel): check_names does, on the parser's tree; resolve_names,
        # on the optimizer's, which may have emptied a procedure, does not.
        self.checking = False

    # ---- collecting ----------------------------------------------------

    def add_module(self, module: P.Module, index: int) -> None:
        """Collect the declarations and names of one module."""
        items = list(module.items)
        if items and isinstance(items[0], P.AddressLiteral):
            items = items[1:]
        name = _file_name(module) or f"M{index + 1}"
        # The module's label is outside the module (9.3): not declared.
        if len(items) == 1 and isinstance(items[0], P.LabeledStmt) and isinstance(
                items[0].stmt, P.DoBlock):
            name = _key(items[0].label)
            self._do_labels[id(items[0].stmt)] = {name}
            self._check_do(items[0].stmt)
            items = list(items[0].stmt.items)
        mod = _Module(index, name)
        mod.main = any(not isinstance(it, (P.ProcDecl, P.DeclareStmt)) for it in items)
        mod.first = next((it for it in items
                          if not isinstance(it, (P.ProcDecl, P.DeclareStmt))), None)
        self.modules.append(mod)
        block = _Block("module", self.globals, mod, None)
        self._visit(items, block)

    def _declare(self, block: _Block, name: str, kind: str, site, **kw) -> _Decl:
        old = block.decls.get(name)
        if old is not None:
            # A LITERALLY declared again as it was (an $INCLUDE file and
            # the file including it often both have TRUE and FALSE) is
            # harmless; anything else declared twice in one block is two
            # definitions of one name.
            if not (kind == "lit" and old.kind == "lit" and old.literal == kw.get("literal")):
                node, attr = site
                raise CodeGenError(
                    f"{ident_text(getattr(node, attr))} is declared twice in the same block",
                    source_location(node))
            old.sites.append(site)
            return old
        d = _Decl(name, kind, block, len(self.decls), [site], **kw)
        block.decls[name] = d
        self.decls.append(d)
        return d

    def _visit(self, n, block: _Block) -> None:  # pylint: disable=too-many-branches,too-many-statements
        if n is None or isinstance(n, (str, int)):
            return
        if isinstance(n, (list, tuple)):
            for x in n:
                self._visit(x, block)
            return
        if isinstance(n, P.ProcDecl):
            self._check_nesting(n, block)
            self._proc(n, block)
        elif isinstance(n, P.DeclareStmt):
            self._visit(n.declarations, block)
        elif isinstance(n, P.DeclItem):
            self._decl_item(n, block)
        elif isinstance(n, P.DeclItemBasedGroup):
            # V3.1 knows none of the names before their bases: `(a based
            # b, b based a) byte' is ERROR 54 twice.
            order = len(self.decls)
            for bd in n.based_decls or []:
                d = self._declare(block, _key(bd.name), "var", (bd, "name"), storage=False,
                                  dim=_dimension(n), members=_members(n), item=n, based=True,
                                  address=decl_item_type(n)[0] == DataType.ADDRESS,
                                  address_members=_address_members(n))
                self._base(bd.base, block, d, order)
            self._visit([n.array_size, n.tail], block)
        elif isinstance(n, P.LiterallyDecl):
            self._declare(block, _key(n.name), "lit", (n, "name"), literal=literally_value(n))
        elif isinstance(n, P.LabeledStmt):
            self._label(n, block)
            # The label of a DO block, which its END may name: the one next
            # to the DO, as Intel's PL/M-80 V3.1 has it, of several.
            if not isinstance(n.stmt, P.LabeledStmt):
                self._do_labels.setdefault(id(n.stmt), set()).add(_key(n.label))
            self._visit(n.stmt, block)
        elif isinstance(n, P.GotoStmt):
            self.refs.append(_Ref(block, n, "label", goto=True))
        elif isinstance(n, (P.AssignStmt, P.EmbeddedAssign)):
            for t in n.targets if isinstance(n, P.AssignStmt) else [n.target]:
                self._reference(t, block, target=True)
            self._visit(n.value, block)
        elif isinstance(n, _DO_BLOCKS):
            self._check_do(n)
            inner = _Block("do", block, block.module, block.proc)
            if isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
                self.refs.append(_Ref(inner, n, "index"))
            for f in ("condition", "start", "bound", "step", "selector", "items"):
                self._visit(getattr(n, f, None), inner)
        elif self._visit_use(n, block):
            pass
        elif isinstance(n, P.CallStmt):
            self._reference(unwrap_paren(n.callee), block, called=True)
        elif isinstance(n, (P.Call, P.MemberAccess)):
            self._reference(n, block)
        elif isinstance(n, P.DottedMember):
            self._visit(n.base, block)      # a member's name is not in scope
        elif isinstance(n, (P.StructMember, P.StructMemberUntyped)):
            self._visit(n.array_size, block)    # a member's name is not in scope
        elif isinstance(n, P.SizeIdent):
            # The macro pass has put in place of a LITERALLY's name its text;
            # a name left in a dimension is not a number.  Intel's PL/M-80
            # V3.1: ERROR 59, ILLEGAL DIMENSION ATTRIBUTE.
            text = ident_text(n.name)
            raise CodeGenError(
                f"({text}): the dimension of an array is a number, and {text} is not a "
                "LITERALLY declared before it whose text is one (Programming Manual "
                "9800268B, 6.2.5)", source_location(n))
        elif isinstance(n, P.EndLabel):
            pass
        elif isinstance(n, P.SizeNumber):
            if parse_plm_number(n.value.text) == 0:
                # Intel's PL/M-80 V3.1: ERROR 57, INVALID DIMENSION, ZERO
                # ILLEGAL, of an array and of a structure's member.
                raise CodeGenError(
                    f"({n.value.text}): an array has at least one element, and a "
                    "dimension of 0 gives it none; Intel's PL/M-80 V3.1 rejects it "
                    "(ERROR #57, INVALID DIMENSION, ZERO ILLEGAL)", source_location(n))
        else:
            fields = getattr(n, "__dataclass_fields__", None)
            if fields and not hasattr(n, "file_id"):
                for f in fields:
                    if f != "pos":
                        self._visit(getattr(n, f), block)

    def _visit_use(self, n, block: _Block) -> bool:
        """Visit ``n`` if it is a use of a name, or where a use says more
        than the name: after a dot, before empty parentheses, in an AT or
        in a DATA or INITIAL list.  Whether it was."""
        if isinstance(n, (P.Identifier, P.DottedIdent)):
            self._ref(block, n)
        elif isinstance(n, P.CallNoArgs) and isinstance(unwrap_paren(n.callee), P.Identifier):
            self._ref(block, unwrap_paren(n.callee), empty=True)
        elif isinstance(n, P.CallNoArgs):
            self._empty_after(unwrap_paren(n.callee), block)
        elif isinstance(n, P.LocationOf) and isinstance(
                unwrap_paren(n.operand), (P.Identifier, P.Call, P.MemberAccess)):
            # `.a', `.a(i)', `.output(3)', `.s.m': the dot is the reference's.
            self._reference(unwrap_paren(n.operand), block, dot=True)
        elif isinstance(n, P.AttrAt):
            self.restricted.append(("at", n.address))
            self._at += 1
            self._visit(n.address, block)
            self._at -= 1
        elif isinstance(n, P.LocationOfList):
            self.restricted.append(("consts", n))
            self._visit(n.values, block)
        elif isinstance(n, P.AttrInitial) or hasattr(n, "data_values"):
            # A DATA or an INITIAL list, where the address of a label may be.
            for f in n.__dataclass_fields__:
                if f == "pos":
                    continue
                inside = f in ("values", "data_values")
                if inside:
                    self.restricted.append(("list", list(getattr(n, f) or [])))
                self._lists += inside
                self._visit(getattr(n, f), block)
                self._lists -= inside
        else:
            return False
        return True

    def _empty_after(self, callee, block: _Block) -> None:
        """``callee()``, ``callee`` a member or a subscripted reference,
        which is never a procedure's name: visit it, and mark the use of the
        name it starts from, so that check_uses refuses the parentheses
        once it knows the name is declared."""
        start = len(self.refs)
        self._visit(callee, block)
        root = callee
        while isinstance(root, (P.MemberAccess, P.Call)):
            root = unwrap_paren(root.base if isinstance(root, P.MemberAccess) else root.callee)
        for r in self.refs[start:]:
            if r.node is root:
                r.after = callee
                return
        text = _reference_text(callee)
        raise CodeGenError(f"{text}(): PL/M-80 has neither an empty subscript nor an "
                           "empty argument list", source_location(callee))

    def _ref(self, block: _Block, node, **kw) -> _Ref:
        """A use of the name ``node`` spells, in ``block``."""
        r = _Ref(block, node, "name", in_list=self._lists > 0, in_at=self._at > 0, **kw)
        self.refs.append(r)
        return r

    def _reference(self, n, block: _Block, dot: bool = False, *, extent=None,  # pylint: disable=too-many-arguments
                   args: int | None = None, call=None, called: bool = False,
                   target: bool = False) -> None:
        """A reference: a name, a member of what a reference names, or
        either subscripted, or a call.  ``dot``: it is the operand of a dot;
        ``extent``: the argument of the call of LENGTH, LAST or SIZE given;
        ``args``: the subscripts or arguments of ``call`` that follow it;
        ``called``: it is what a CALL statement calls; ``target``: what an
        assignment stores to.  What is inside a subscript is visited on its
        own."""
        n = unwrap_paren(n)
        if isinstance(n, P.Identifier):
            self._ref(block, n, dot=dot, extent=extent, args=args, call=call, called=called,
                      target=target)
        elif isinstance(n, P.Call):
            callee = unwrap_paren(n.callee)
            name = _key(callee.name) if isinstance(callee, P.Identifier) else ""
            if isinstance(callee, P.Call):
                self.chains.append(n)
            start = len(self.refs)
            self._reference(callee, block, dot, extent=extent, args=len(n.args), call=n,
                            called=called, target=target)
            if name in ("LENGTH", "LAST", "SIZE") and len(n.args) == 1 and not dot:
                self.extent_calls[id(n)] = (n, self.refs[start])
                self._reference(n.args[0], block, extent=n)
                return
            self._visit(n.args, block)
        elif isinstance(n, P.MemberAccess):
            self.member_uses.append(_MemberUse(n, block, dot, extent, args, call, called))
            start = len(self.refs)
            self._reference(n.base, block, dot, extent=extent, target=target)
            if len(self.refs) > start and self.refs[start].member is None \
                    and unwrap_paren(n.base) is self.refs[start].node:
                self.refs[start].member = n
        else:
            self._visit(n, block)

    def _check_do(self, n) -> None:
        """A DO block's END names a label of the block, if any; a DO CASE
        has a case.  Intel's PL/M-80 V3.1: ERROR 20, MISMATCHED IDENTIFIER
        AT END OF BLOCK; ERROR 201, INVALID DO CASE BLOCK, AT LEAST ONE
        CASE REQUIRED."""
        end = n.end_label
        labels = self._do_labels.get(id(n), set())
        if end is not None and _key(end.name) not in labels:
            text = ident_text(end.name)
            which = (f"a block labelled {', '.join(sorted(labels))}" if labels
                     else "a block with no label")
            self.intel(end, f"END {text}: the END of {which} names {text}", 20)
        if isinstance(n, P.DoCaseBlock) and not any(
                not is_end_of_block(it) and not isinstance(it, P.DeclareStmt)
                for it in n.items):
            self.intel(n, "DO CASE: a DO CASE block has at least one case", 201)

    # V3.1's text of each of its errors a message here names.
    INTEL_ERRORS = {
        20: "MISMATCHED IDENTIFIER AT END OF BLOCK",
        32: "INVALID SYNTAX, TEXT IGNORED UNTIL ';'",
        39: "INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL",
        50: "INVALID ATTRIBUTES FOR BASE",
        52: "INVALID BASE, MEMBER OF BASED STRUCTURE OR ARRAY OF STRUCTURES",
        54: "UNDECLARED BASE",
        55: "UNDECLARED STRUCTURE MEMBER IN BASE",
        88: "INVALID PROCEDURE NESTING, ILLEGAL IN REENTRANT PROCEDURE",
        104: "ILLEGAL PROCEDURE INVOCATION WITH DOT OPERATOR",
        105: "UNDECLARED IDENTIFIER",
        108: "MISSING ')' AFTER INPUT/OUTPUT PORT NUMBER",
        109: "MISSING INPUT/OUTPUT PORT NUMBER",
        114: "INVALID SUBSCRIPT, MULTIPLE SUBSCRIPTS ILLEGAL",
        118: "INVALID INDIRECT CALL, IDENTIFIER NOT AN ADDRESS SCALAR",
        124: "MISSING ARGUMENTS FOR BUILT-IN PROCEDURE",
        125: "ILLEGAL ARGUMENT FOR BUILT-IN PROCEDURE",
        126: "MISSING ')' AFTER BUILT-IN PROCEDURE ARGUMENT LIST",
        127: "INVALID SUBSCRIPT ON NON-ARRAY",
        128: "INVALID LEFT-HAND OPERAND OF ASSIGNMENT",
        129: "ILLEGAL 'CALL' WITH TYPED PROCEDURE",
        131: "ILLEGAL REFERENCE TO UNTYPED PROCEDURE",
        133: "ILLEGAL REFERENCE TO UNSUBSCRIPTED ARRAY",
        134: "ILLEGAL REFERENCE TO UNSUBSCRIPTED MEMBER ARRAY",
        146: "MISSING ')' AFTER 'AT' RESTRICTED EXPRESSION",
        147: "MISSING IDENTIFIER FOLLOWING DOT OPERATOR",
        149: "INVALID SUBSCRIPTING IN RESTRICTED REFERENCE",
        150: "MISSING ')' AT END OF RESTRICTED SUBSCRIPT",
        151: "INVALID OPERAND IN RESTRICTED EXPRESSION",
        152: "MISSING ')' AFTER CONSTANT LIST",
        153: "INVALID NUMBER OF ARGUMENTS IN CALL, TOO MANY",
        154: "INVALID NUMBER OF ARGUMENTS IN CALL, TOO FEW",
        156: "MISSING RETURN STATEMENT IN TYPED PROCEDURE",
        169: "ILLEGAL FORWARD CALL",
        170: "ILLEGAL RECURSIVE CALL",
        172: "INVALID LABEL: UNDEFINED",
        174: "INVALID NULL PROCEDURE",
        201: "INVALID DO CASE BLOCK, AT LEAST ONE CASE REQUIRED",
        209: "ILLEGAL INITIALIZATION OF MORE SPACE THAN DECLARED",
        210: "ILLEGAL INITIALIZATION OF A BYTE TO A VALUE > 255",
        211: "INVALID IDENTIFIER IN 'AT' RESTRICTED REFERENCE",
        212: "INVALID RESTRICTED REFERENCE IN 'AT', BASE ILLEGAL",
    }

    @classmethod
    def _rejects(cls, numbers) -> str:
        """That Intel's PL/M-80 V3.1 rejects it, with the errors ``numbers``."""
        return "Intel's PL/M-80 V3.1 rejects it (ERROR " + ", and ".join(
            f"#{n}, {cls.INTEL_ERRORS[n]}" for n in numbers) + ")"

    def intel(self, node, text: str, *numbers: int, warn: bool = False) -> None:
        """What Intel's PL/M-80 V3.1 rejects, with the errors ``numbers``:
        an error, or, where programs written for uplm80 rely on it
        (``warn``), a warning; ``text`` then says how it is compiled."""
        if not self.checking:
            return
        why = self._rejects(numbers)
        if warn:
            self.intel_warnings.append((source_location(node), f"{text}; {why}"))
            return
        raise CodeGenError(f"{text}; {why}", source_location(node))

    def _check_nesting(self, p: P.ProcDecl, block: _Block) -> None:
        """A REENTRANT procedure is declared at the outer level of the
        module, not in a procedure or a DO block, and has no procedure
        declared in it (8.1.7).  Intel's PL/M-80 V3.1: ERROR 39, INVALID
        ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL; ERROR 88,
        INVALID PROCEDURE NESTING, ILLEGAL IN REENTRANT PROCEDURE."""
        text = ident_text(p.name)
        attrs = proc_attrs(p)
        if block.kind != "module" and (attrs.interrupt_num is not None or attrs.is_public
                                       or attrs.is_external):
            return                  # _below_module
        outer = block.kind != "module" and attrs.is_reentrant
        inner = block.proc is not None and block.proc.reentrant
        if outer:
            also = (f", and {block.proc.orig}, a REENTRANT procedure, has no procedure declared "
                    "in it" if inner else "")
            self.intel(p, f"{text}: a REENTRANT procedure must be declared at the outer level "
                       f"of the module, not in {self._where(block)}{also} (Programming Manual "
                       "9800268B, 8.1.7)", *((39, 88) if inner else (39,)))
        elif inner:
            self.intel(p, f"{text}: procedure {text} is declared in {block.proc.orig}, and a "
                       "REENTRANT procedure has no procedure declared in it (Programming "
                       "Manual 9800268B, 8.1.7)", 88)

    def _below_module(self, p: P.ProcDecl, attrs, block: _Block) -> None:
        """An INTERRUPT, PUBLIC or EXTERNAL procedure in a procedure or a
        DO block.  Intel's PL/M-80 V3.1: ERROR 39, INVALID
        ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL, in a procedure
        and in a DO block of the module alike; and ERROR 88, INVALID
        PROCEDURE NESTING, ILLEGAL IN REENTRANT PROCEDURE, in a REENTRANT
        one; and V3.1 then takes it for a procedure of its own, which, with
        no statements, as an EXTERNAL one has none, is ERROR 174, INVALID
        NULL PROCEDURE, and of a typed one ERROR 156, MISSING RETURN
        STATEMENT IN TYPED PROCEDURE."""
        what = ("INTERRUPT" if attrs.interrupt_num is not None else
                "PUBLIC" if attrs.is_public else "EXTERNAL")
        inner = block.proc is not None and block.proc.reentrant
        also = (f", and {block.proc.orig}, a REENTRANT procedure, has no procedure declared in "
                "it" if inner else "")
        manual = " (Programming Manual 9800268B, 8.1.6)" if what == "INTERRUPT" else ""
        numbers = (39, *((88,) if inner else ()), *(() if _has_statements(p) else _null(p)))
        raise CodeGenError(
            f"{ident_text(p.name)}: a{'n' if what[0] in 'EI' else ''} {what} procedure must be "
            f"declared at the outer level of the module, not in {self._where(block)}{also}"
            f"{manual}; {self._rejects(numbers)}", source_location(p))

    def _proc(self, p: P.ProcDecl, block: _Block) -> None:
        attrs = proc_attrs(p)
        if block.kind != "module" and (attrs.interrupt_num is not None or attrs.is_public
                                       or attrs.is_external):
            self._below_module(p, attrs, block)
        d = self._declare(block, _key(p.name), "proc", (p, "name"), public=attrs.is_public,
                          external=attrs.is_external, reentrant=attrs.is_reentrant, node=p)
        end = p.body.end_label
        if end is not None and _key(end.name) == d.name:
            d.sites.append((end, "name"))
        elif end is not None:
            # Intel's PL/M-80 V3.1: ERROR 20.
            text = ident_text(end.name)
            self.intel(end, f"END {text}: the END of procedure {d.orig} names {text}", 20)
        if not attrs.is_external and not _has_statements(p):
            # Intel's PL/M-80 V3.1: ERROR 174, a label on its END or not.
            self.intel(p, f"{d.orig}: a procedure has at least one statement, and {d.orig} "
                       "has none", *_null(p))
        body = _Block("proc", block, block.module, d)
        params = p.signature.params
        for n in (params.names or []) if params is not None else []:
            # Static, a parameter has a label as a local does (local_storage);
            # an EXTERNAL procedure's are in the module that defines it.
            self._declare(body, _key(n.name), "param", (n, "name"),
                          storage=not (attrs.is_reentrant or attrs.is_external))
        self._visit(p.body.items, body)
        for n in (params.names or []) if params is not None else []:
            if len(body.decls[_key(n.name)].sites) < 2:
                # Intel's PL/M-80 V3.1: ERROR 25, UNDECLARED PARAMETER.
                raise CodeGenError(
                    f"{ident_text(n.name)} is a parameter of {d.orig}, and no DECLARE of "
                    "the procedure declares it; a parameter is declared a BYTE or an "
                    "ADDRESS scalar, not BASED, by a DECLARE of its procedure "
                    "(Programming Manual 9800268B, 8.1.1)", source_location(n))

    def _decl_item(self, item: P.DeclItem, block: _Block) -> None:
        attrs = decl_attrs(item)
        dtype, _ = decl_item_type(item)
        names = item.names
        nodes = [names] if isinstance(names, P.DeclName) else list(names.names or [])
        # A REENTRANT procedure's locals are in its frame, with no label.
        reentrant = block.proc is not None and block.proc.reentrant
        storage = (item.based is None
                   and not (reentrant and attrs.at_location is None
                            and not attrs.data_values and not attrs.initial_values))
        for node in nodes:
            name = _key(node.name)
            old = block.decls.get(name)
            if old is not None and old.kind == "param":
                self._declare_param(old, item, node)
                old.typed = (item, len(self.decls))
                continue
            if block.kind != "module":
                self._below_module_level(node, attrs, block)
            kind = "label" if dtype == DataType.LABEL else "var"
            self._declare(block, name, kind, (node, "name"), public=attrs.is_public,
                          external=attrs.is_external, storage=storage and kind == "var",
                          dim=_dimension(item), members=_members(item), item=item,
                          based=item.based is not None, address=dtype == DataType.ADDRESS,
                          address_members=_address_members(item))
        if item.based is not None:
            self._base(item.based.base, block, block.decls[_key(nodes[0].name)])
        slots, past = _value_slots(item)
        self._slots.update(slots)
        self._past |= past
        self._visit([item.array_size, item.tail], block)

    def _base(self, base, block: _Block, based: _Decl, order: int | None = None) -> None:
        """The base of ``based``, a BASED variable, `p' or `s.m', whose
        declaration is the ``order``-th, by default ``based``'s own: visit
        it, and mark the use of the name it starts from (bind,
        _check_bases, _settle_bases)."""
        start = len(self.refs)
        self._visit(base, block)
        root = base
        while isinstance(root, P.DottedMember):
            root = root.base
        for r in self.refs[start:]:
            if r.node is root:
                r.base = (based, base, based.order if order is None else order)
                return

    @staticmethod
    def _where(block: _Block) -> str:
        """The block a declaration below module level is in, in words."""
        return "a DO block" if block.kind == "do" else f"procedure {block.proc.orig}"

    def _below_module_level(self, node, attrs, block: _Block) -> None:
        """A variable declared in a procedure or a DO block: PUBLIC and
        EXTERNAL are errors, INITIAL a warning (uplm80 initializes the
        variable once, when the program is loaded, and programs written
        for it rely on that).  Intel's PL/M-80 V3.1 rejects all three:
        ERROR 73, INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE
        LEVEL."""
        text = ident_text(node.name)
        why = ("Intel's PL/M-80 V3.1 rejects it (ERROR #73, INVALID ATTRIBUTE OR "
               "INITIALIZATION, NOT AT MODULE LEVEL)")
        for flag, what in ((attrs.is_public, "PUBLIC"), (attrs.is_external, "EXTERNAL")):
            if flag:
                raise CodeGenError(
                    f"{text}: a{'n' if what[0] == 'E' else ''} {what} variable must be declared "
                    f"at the outer level of the module, not in {self._where(block)}; {why}",
                    source_location(node))
        if attrs.initial_values is not None:
            self.intel_warnings.append((source_location(node), (
                f"{text}: INITIAL in {self._where(block)} initializes the variable once, "
                f"when the program is loaded, not at each entry; {why}")))

    @staticmethod
    def _declare_param(d: _Decl, item: P.DeclItem, node) -> None:
        """``node`` of ``item`` declares ``d``, a parameter of the procedure
        whose block it is: once, as a BYTE or an ADDRESS scalar, not BASED
        and with no other attribute (8.1.1)."""
        if len(d.sites) > 1:
            # Intel's PL/M-80 V3.1: ERROR 78, DUPLICATE DECLARATION.
            raise CodeGenError(f"{ident_text(node.name)} is declared twice in the same block",
                               source_location(node))
        attrs = decl_attrs(item)
        dtype, dim = decl_item_type(item)
        plain = not (attrs.is_public or attrs.is_external) and all(
            v is None for v in (attrs.at_location, attrs.initial_values, attrs.data_values))
        if (dtype not in (DataType.BYTE, DataType.ADDRESS) or dim is not None
                or item.based is not None or not plain):
            # Intel's PL/M-80 V3.1: ERROR 76, CONFLICTING ATTRIBUTE WITH
            # PARAMETER; 77, INVALID PARAMETER DECLARATION, BASE ILLEGAL;
            # 79, ILLEGAL PARAMETER TYPE, NOT BYTE OR ADDRESS.
            raise CodeGenError(
                f"{ident_text(node.name)} is a parameter of {d.block.proc.orig}, and a "
                "parameter is declared a BYTE or an ADDRESS scalar, not BASED and with "
                "no other attribute (Programming Manual 9800268B, 8.1.1)",
                source_location(node))
        d.sites.append((node, "name"))
        d.address = dtype == DataType.ADDRESS

    def _label(self, s: P.LabeledStmt, block: _Block) -> None:
        name = _key(s.label)
        old = block.decls.get(name)
        if old is not None and old.kind == "label" and not old.defined:
            old.sites.append((s, "label"))      # declared a LABEL, and here it is
            old.defined = True
            return
        if old is not None and old.kind == "label":
            raise CodeGenError(
                f"label {ident_text(s.label)} is defined twice in the same block",
                source_location(s))
        d = self._declare(block, name, "label", (s, "label"))
        d.defined = True

    # ---- binding -------------------------------------------------------

    def bind(self) -> None:
        """Bind every name to the declaration it means."""
        # What one module makes PUBLIC another can name without declaring
        # it EXTERNAL; a multi-file compile has always allowed that.
        for d in self.decls:
            if d.block.kind == "module" and d.shared:
                cur = self.globals.decls.get(d.name)
                if cur is None or (d.public and not cur.public):
                    self.globals.decls[d.name] = d
        for r in self.refs:
            name = _key(getattr(r.node, r.attr))
            r.decl = self.lookup(name, r.block)
            if r.base is not None:
                # A base is the declaration of its name made before the
                # variable BASED on it, as Intel's PL/M-80 V3.1 reads it,
                # which takes one further down the block for another name
                # (ERROR 54, UNDECLARED BASE: _check_bases), and an outer
                # block's of the name, if any, for the base.
                r.decl = self.lookup(name, r.block, before=r.base[2]) or r.decl
            if r.decl is not None:
                self.refs_of.setdefault(id(r.decl), []).append(r)

    @staticmethod
    def lookup(name: str, block: _Block | None, before: int | None = None) -> _Decl | None:
        """The declaration ``name`` means in ``block``: the innermost, or,
        ``before`` given, the innermost of those made before the
        ``before``-th, or another module's.  What another module makes
        PUBLIC does not hide a built-in: a module that does not declare
        SHL means the built-in, as it does compiled alone
        (CodeGenerator._kept_builtins)."""
        while block is not None:
            d = block.decls.get(name)
            if d is not None and (before is None or block.kind == "global" or d.order < before):
                return None if block.kind == "global" and name in _BUILTINS else d
            block = block.parent
        return None

    # ---- what code generation calls things -----------------------------

    @staticmethod
    def owner(d: _Decl) -> _Decl | None:
        """The procedure a declaration is made in, None at module level."""
        return d.block.proc

    def proc_key(self, d: _Decl) -> str:
        """The name code generation files a procedure under: its own, or,
        nested, its parent's and its own (CodeGenerator._register_procedure)."""
        parent = self.owner(d)
        if d.shared or parent is None:
            return d.name
        return f"{self.proc_key(parent)}${d.name}"

    def asm_name(self, d: _Decl) -> str | None:  # pylint: disable=too-many-return-statements
        """The label code generation gives a declaration, if it defines one
        that can meet another's: a DO block's variables are numbered apart
        (@proc$Bn$name).  A procedure's local or parameter has one if it is
        static, which local_storage decides later, so it is taken to have
        one; a REENTRANT procedure's are on its stack."""
        parent = self.owner(d)
        if d.kind == "proc":
            if d.shared or parent is None:
                return data_name(d.name)
            return f"@{self.proc_key(d)}"
        if d.kind == "label":
            if d.shared or parent is None:
                return data_name(d.name)
            return f"@{self.proc_key(parent)}${d.name}"
        if d.kind == "lit":
            try:
                parse_plm_number(d.literal or "")
            except ValueError:
                return None     # no EQU
            return data_name(d.name)
        if d.kind in ("var", "param") and d.storage:
            if d.block.kind == "module":
                return data_name(d.name)
            if d.block.kind == "proc":
                return f"@{self.proc_key(parent)}${data_name(d.name)}"
        return None

    # ---- renaming ------------------------------------------------------

    def rename(self, d: _Decl, new: str) -> None:
        """Call ``d`` ``new`` from now on."""
        del d.block.decls[d.name]
        d.name = new
        d.block.decls[new] = d
        self.used.add(new)

    def fresh(self, name: str) -> str:
        """A name no declaration has: ``name?2``, ``name?3``, ..."""
        n = 2
        while f"{name}?{n}" in self.used:
            n += 1
        return f"{name}?{n}"

    def qualify(self) -> None:
        """Give each module's private module-level names its own name.

        Modules have separate name spaces for everything not PUBLIC or
        EXTERNAL (Programming Manual, 10.4); compiled one at a time they
        are kept apart by the linker, which sees only the PUBLIC names.  A
        multi-file compile makes one assembly of them, so a module-level
        name - and anything code generation names as one: a procedure or
        label in a DO block of the main program, and every LITERALLY's EQU
        - is qualified with the module's name: HELPER of module LIB is
        LIB?HELPER, its locals @LIB?HELPER$N.
        """
        taken: set[str] = set()
        for m in self.modules:
            q = m.name
            while q in taken:
                q = f"{q}{m.index + 1}"
            taken.add(q)
            m.qualifier = q
        for d in list(self.decls):
            if d.shared or "?" in d.name:
                continue
            if (d.block.kind == "module" or d.kind == "lit"
                    or (d.kind in ("proc", "label") and d.block.proc is None)):
                self.rename(d, f"{d.block.module.qualifier}?{d.name}")

    def check_private(self) -> None:
        """A name one module uses without declaring it, and another module
        declares without making it PUBLIC, is an error: compiled alone,
        the one module could not reach it."""
        private: dict[str, _Decl] = {}
        for d in self.decls:
            if d.block.kind == "module" and not d.shared:
                private.setdefault(d.orig, d)
        for r in self.refs:
            if r.decl is not None or r.goto:
                continue
            name = _key(getattr(r.node, r.attr))
            d = private.get(name)
            if d is None or name in _BUILTINS or d.block.module is r.block.module:
                continue
            text = ident_text(getattr(r.node, r.attr))
            raise CodeGenError(
                f"{text} is not declared in module {r.block.module.name}; module "
                f"{d.block.module.name} declares it but does not make it PUBLIC "
                "(declare it PUBLIC there and EXTERNAL here)", source_location(r.node))

    _KIND_WORDS = {"var": "a variable", "param": "a parameter", "label": "a label"}

    def check_uses(self) -> None:
        """Every name used as PL/M-80 allows: declared, or a built-in; not
        the address of a label, but in a DATA or an INITIAL list; and no
        empty parentheses after anything but a procedure."""
        for r in self.refs:
            d = r.decl
            if r.goto:
                continue
            text = ident_text(getattr(r.node, r.attr))
            if d is None and not (_key(getattr(r.node, r.attr)) in _BUILTINS or r.in_at):
                # Intel's PL/M-80 V3.1: ERROR 105, UNDECLARED IDENTIFIER.
                # (An AT names its own: _at_address.)
                raise CodeGenError(f"{text} is not declared (Programming Manual "
                                   "9800268B, 6.1)", source_location(r.node))
            if d is not None and d.kind == "lit":
                # The macro pass has put the LITERALLY's text in place of
                # its name everywhere after the declaration, in its scope;
                # a name left is before it.  Intel's PL/M-80 V3.1: ERROR
                # 105, UNDECLARED IDENTIFIER.
                raise CodeGenError(
                    f"{text} is not declared here: a LITERALLY declared after it puts its "
                    f"text in place of {text} only in the text that follows the declaration "
                    "(Programming Manual 9800268B, 6.4)", source_location(r.node))
            if r.after is not None:
                # Intel's PL/M-80 V3.1: ERROR 127, INVALID SUBSCRIPT ON
                # NON-ARRAY, or 102, MISSING PRIMARY OPERAND, after a member;
                # 32, INVALID SYNTAX, after a subscript.
                after = _reference_text(r.after)
                kind = _after_kind(r.after, d, _key(getattr(r.node, r.attr)))
                raise CodeGenError(
                    f"{after}(): {after} is {kind}, and PL/M-80 has neither an empty "
                    "subscript nor an empty argument list", source_location(r.node))
            if d is None:
                self._check_builtin_use(r, text)
                continue
            if r.empty and d.kind in self._KIND_WORDS:
                # Intel's PL/M-80 V3.1: ERROR 127, INVALID SUBSCRIPT ON
                # NON-ARRAY, and ERROR 102, MISSING PRIMARY OPERAND, for
                # scalars and arrays, in an expression and in a CALL.
                raise CodeGenError(
                    f"{text}(): {text} is {self._KIND_WORDS[d.kind]}, and PL/M-80 has "
                    "neither an empty subscript nor an empty argument list",
                    source_location(r.node))
            if r.empty and d.kind == "proc":
                # A procedure's `f()' is taken for `f', as it always has
                # been, and tests/test_implicit_calls.plm relies on it.
                self.intel_warnings.append((source_location(r.node), (
                    f"{text}(): PL/M-80 has no empty argument list, and this is taken for "
                    f"{text}, a call with no arguments; Intel's PL/M-80 V3.1 rejects it "
                    "(ERROR #102, MISSING PRIMARY OPERAND, and #153, INVALID NUMBER OF "
                    "ARGUMENTS IN CALL)")))
            if r.dot and d.kind == "label" and not r.in_list:
                # Intel's PL/M-80 V3.1: ERROR 158, INVALID DOT OPERAND,
                # LABEL ILLEGAL, whether or not the label is declared LABEL.
                raise CodeGenError(
                    f".{text}: {text} is a label, and the dot operator takes a variable "
                    "or a procedure (Programming Manual 9800268B, 4.1.3); the address "
                    "of a label may be given only in a DATA or an INITIAL list",
                    source_location(r.node))

    def check_declarations(self) -> None:
        """The base of each BASED variable, and each LABEL declared, as
        Intel's PL/M-80 V3.1 takes them (:meth:`_check_bases`,
        :meth:`_check_labels`)."""
        self._check_bases()
        self._check_labels()

    def _check_bases(self) -> None:
        """The base of each BASED variable is what Intel's PL/M-80 V3.1
        takes: an ADDRESS scalar, a variable or a parameter, or an ADDRESS
        scalar member of a structure that is neither BASED nor an array,
        declared before the variable BASED on it (bind); anything else is
        an error, with V3.1's error.  uplm80 did not check a base.  One
        that is BASED, or a member of what is, `declare s based sp
        structure (k byte, p address); declare a based s.p byte', has no
        address of its own to read the pointer from: um80's "Undefined
        symbol 'S'" (0.4.3 the same), in a factored declaration too, `(a
        based s.p) byte', since 0.4.4's b9c4a36 (0.4.3 took it for a
        variable of its own), and `a based a' recursed until Python gave
        up.  A member of an array of structures, or an array, was refused
        naming #133, V3.1's error in an expression; a BYTE, a structure, a
        member array, a procedure, a label, a member the structure does
        not have (its first word) and a name declared only further down
        were compiled."""
        for r in self.refs:
            if r.base is None:
                continue
            fault = self._base_fault(r)
            if fault is not None:
                based, base, _ = r.base
                why, number = fault
                self.intel(r.node, f"{based.orig} BASED {_base_text(base)}: {why}", number)

    def _base_fault(self, r: _Ref) -> tuple[str, int] | None:  # pylint: disable=too-many-return-statements,too-many-branches
        """What V3.1 does not take of the base ``r`` starts: (why, the error
        V3.1 gives), or None."""
        based, base, order = r.base
        parts = [p.upper() for p in dotted_ident_parts(base)]
        name, members = parts[0], parts[1:]
        d = r.decl
        if d is not None and d.block.kind != "global" and d.order >= order:
            # Declared only further down, where V3.1 does not know it yet.
            if name in _BUILTINS:
                d = None
            elif d is based:
                return "a variable is not its own base", 54
            else:
                return (f"{name} is declared after {based.orig}, and a base is declared "
                        "before the variable BASED on it"), 54
        if d is None:
            if name in _BUILTINS:
                return f"{name} is a built-in, and a base is a variable", 50
            return None     # declared nowhere: check_uses
        if d.kind == "lit" or len(members) > 1:
            return None     # a LITERALLY declared after it: check_uses
        if members and (d.kind != "var" or d.members is None):
            return f"{name} is not a structure", 55
        if d.kind in ("proc", "label"):
            what = "a procedure" if d.kind == "proc" else "a label"
            return f"{name} is {what}, and a base is a variable", 50
        if d.kind == "param":
            if d.typed is None or d.typed[1] > order:
                return (f"{name} is declared a BYTE or an ADDRESS only after "
                        f"{based.orig}, and a base is declared an ADDRESS before the "
                        "variable BASED on it"), 50
            if decl_item_type(d.typed[0])[0] != DataType.ADDRESS:
                return f"{name} is a BYTE, and a base is an ADDRESS", 50
            return None
        if members:
            member = members[0]
            if member not in d.members:
                return f"{name} has no member {member}", 55
            if d.members[member] is not None:
                return f"{name}.{member} is an array, and a base is a scalar", 50
            if _member_type(d.item, member) != DataType.ADDRESS:
                return f"{name}.{member} is a BYTE, and a base is an ADDRESS", 50
            if d.based:
                return f"{name} is BASED, and a base is not a member of what is BASED", 52
            if d.dim is not None:
                return (f"{name} is an array, and a base is not a member of an array of "
                        "structures"), 52
            return None
        if d.based:
            return f"{name} is BASED, and a base is not", 50
        if d.members is not None:
            return f"{name} is a structure, and a base is an ADDRESS scalar", 50
        if d.dim is not None:
            return f"{name} is an array, and a base is a scalar", 50
        if decl_item_type(d.item)[0] != DataType.ADDRESS:
            return f"{name} is a BYTE, and a base is an ADDRESS", 50
        return None

    def _check_labels(self) -> None:
        """A LABEL declared in a block, not PUBLIC nor EXTERNAL, labels a
        statement of the block (Programming Manual 9800268B, 9.3); a label
        of the name in a block nested in it is another label, that block's.
        Intel's PL/M-80 V3.1 rejects one that labels none, ERROR 172,
        INVALID LABEL: UNDEFINED, named or not, and ERROR 105, UNDECLARED
        IDENTIFIER, where the program names it.  uplm80 took it without a
        word; the location of it in a DATA list, `dw LB', was um80's
        "Undefined symbol", and a GOTO to it was refused only where no
        optimization had left the GOTO out (resolve_names).  A CALL of it is
        ERROR 118, INVALID INDIRECT CALL, IDENTIFIER NOT AN ADDRESS SCALAR,
        besides, as of a label that labels one (_check_called), and 32,
        INVALID SYNTAX, of what follows it in parentheses.  A PUBLIC one is
        check_public_labels'."""
        for d in self.decls:
            if d.kind != "label" or d.defined or d.shared:
                continue
            node, attr = d.sites[0]
            text = ident_text(getattr(node, attr))
            inner = next((x for x in self.decls if x.kind == "label" and x.orig == d.orig
                          and x.defined and x.block.module is d.block.module), None)
            also = (f" (the {text}: in {self._place(inner)} is another label, that block's)"
                    if inner is not None else "")
            uses = self.refs_of.get(id(d), [])
            if not uses:
                self.intel(node, f"{text} is declared a LABEL but labels no statement{also}", 172)
                continue
            r = uses[0]
            if r.called:
                what = expr_text(r.call) if r.call is not None else text
                self.intel(r.node, f"CALL {what}: {text} is declared a LABEL but labels no "
                           f"statement{also}, and a CALL calls a procedure, or through an "
                           "ADDRESS scalar (Programming Manual 9800268B, 8.2.1)",
                           *((105, 118, 32, 172) if r.call is not None else (105, 118, 172)))
                continue
            what = f"GOTO {text}" if r.goto else f".{text}" if r.dot else text
            self.intel(r.node, f"{what}: {text} is declared a LABEL but labels no statement"
                       f"{also}", 105, 172)

    def check_forms(self) -> None:
        """How each name is used, as Intel's PL/M-80 V3.1 allows (0.4.2's
        Known issues): no subscript on a scalar, and never more than one;
        an array, or a member array, without one only after a dot or in
        LENGTH, LAST and SIZE; a CALL of a procedure or through an ADDRESS
        scalar; not `.p(1)' of a procedure; no procedure used before its
        declaration; nothing in parentheses in a subscript of LENGTH, LAST
        or SIZE's argument."""
        for u in self.member_uses:
            if u.called:
                self._check_called_member(u)
        for r in self.refs:
            if not (r.goto or r.empty or r.after is not None):
                self._check_ref(r)
        for u in self.member_uses:
            if not u.called:
                self._check_member(u)
        for n in self.chains:
            # Intel's PL/M-80 V3.1: ERROR 32, INVALID SYNTAX.
            self.intel(n, f"{expr_text(n)}: a subscript or an argument list follows a name or "
                       "a member, not another", 32)
        for call, ref in self.extent_calls.values():
            if ref.decl is None:
                self._check_extent(call)

    def _check_ref(self, r: _Ref) -> None:
        """How a name is used, for what it names."""
        d = r.decl
        text = ident_text(getattr(r.node, r.attr))
        if r.target:
            self._check_target(r, d, text)
        if d is None:
            self._check_builtin_form(r, text)
        elif d.kind == "proc":
            if r.dot and r.args is not None:
                self.intel(r.node, f".{expr_text(r.call)}: {text} is a procedure, and the dot "
                           "operator takes the address of a procedure, not of a call of it "
                           "(Programming Manual 9800268B, 4.1.3)", 104)
            self._check_forward(r, d, text)
            self._check_recursive(r, d, text)
        elif d.kind in ("var", "param"):
            self._check_variable(r, d, text)
        elif d.kind == "label" and r.called:
            self._check_called(r, d, text)

    def _check_target(self, r: _Ref, d: _Decl | None, text: str) -> None:
        """What an assignment stores to: a variable, an element or a
        member, or MEMORY, OUTPUT or STACKPTR; not a procedure or another
        built-in, whose value uplm80 stored to a symbol of its name, which
        only the link found undefined (`input(1) = b', found checking
        0.4.4).  Intel's PL/M-80 V3.1: ERROR 128, INVALID LEFT-HAND OPERAND
        OF ASSIGNMENT, and 131, ILLEGAL REFERENCE TO UNTYPED PROCEDURE, of
        one without a type."""
        name = _key(getattr(r.node, r.attr))
        if d is None and name in _BUILTINS and name not in ("MEMORY", "OUTPUT", "STACKPTR",
                                                             "CPUTIME"):
            kind, untyped = "a built-in procedure", name in ("MOVE", "TIME")
        elif d is not None and d.kind == "proc":
            kind, untyped = "a procedure", proc_return_type(d.node) is None
        else:
            return
        what = expr_text(r.call) if r.call is not None else text
        self.intel(r.node, f"{what}: {text} is {kind}, and an assignment stores to a variable, "
                   "an element of an array or a structure's member, or to MEMORY, OUTPUT or "
                   "STACKPTR", *((131, 128) if untyped else (128,)))

    def _check_variable(self, r: _Ref, d: _Decl, text: str) -> None:
        """A variable or a parameter called through, subscripted, or named
        without a subscript.  Intel's PL/M-80 V3.1: ERROR 114, INVALID
        SUBSCRIPT, MULTIPLE SUBSCRIPTS ILLEGAL, of an array."""
        if r.called:
            self._check_called(r, d, text)
        elif r.args is not None and d.dim is None:
            self._check_scalar(r, text)
        elif r.args is not None and r.args > 1 and not self._extent(r.extent):
            self.intel(r.node, f"{expr_text(r.call)}: {text} is an array, and an array takes "
                       "one subscript", 114)
        elif r.args is None and d.dim is not None and not r.dot \
                and not self._extent(r.extent):
            self._check_unsubscripted(r, text)

    def _check_builtin_form(self, r: _Ref, text: str) -> None:
        """MEMORY, an array, with one subscript, but after a dot or in
        LENGTH, LAST or SIZE, and not called; INPUT and OUTPUT with one
        port.  Intel's PL/M-80 V3.1: ERROR 114, INVALID SUBSCRIPT,
        MULTIPLE SUBSCRIPTS ILLEGAL; 118, INVALID INDIRECT CALL, IDENTIFIER
        NOT AN ADDRESS SCALAR; 133, ILLEGAL REFERENCE TO UNSUBSCRIPTED
        ARRAY; 108, MISSING ')' AFTER INPUT/OUTPUT PORT NUMBER; 129, ILLEGAL
        'CALL' WITH TYPED PROCEDURE, of a built-in with a type."""
        name = _key(getattr(r.node, r.attr))
        if r.in_at or r.in_list:
            return
        if name == "MEMORY" and r.called:
            # And ERROR 32, INVALID SYNTAX, of what follows it in parentheses.
            what = expr_text(r.call) if r.call is not None else text
            self.intel(r.node, f"CALL {what}: MEMORY is an array, and a CALL calls a "
                       "procedure, or through an ADDRESS scalar (Programming Manual "
                       "9800268B, 8.2.1)", *((118, 32) if r.call is not None else (118,)))
            return
        if name in _TYPED_BUILTINS and r.called:
            # `call stackptr' called through the value of SP, and `call
            # carry' an undefined CARRY.  And ERROR 32, INVALID SYNTAX, of
            # what follows it in parentheses.
            what = expr_text(r.call) if r.call is not None else text
            self.intel(r.node, f"CALL {what}: {text} is a built-in with a type, and a CALL "
                       "calls a procedure without one, or through an ADDRESS scalar",
                       *((129, 32) if r.call is not None else (129,)))
            return
        if name == "MEMORY" and r.args is None and not r.dot and not self._extent(r.extent):
            self.intel(r.node, f"{text}: MEMORY is an array, and an array is named without a "
                       "subscript only as the operand of a dot or the argument of LENGTH, "
                       "LAST or SIZE (Programming Manual 9800268B, 3.6.2)", 133)
        if not r.dot:
            self._check_builtin_args(r, name, text)
        if r.args is None:
            return
        what = expr_text(r.call)
        if name == "MEMORY" and r.args > 1 and not self._extent(r.extent):
            self.intel(r.node, f"{what}: MEMORY is an array, and an array takes one "
                       "subscript", 114)
        elif name in ("INPUT", "OUTPUT") and r.args > 1:
            self.intel(r.node, f"{what}: {text} takes one port number", 108)

    def _check_builtin_args(self, r: _Ref, name: str, text: str) -> None:
        """A built-in procedure with as many arguments as it takes, LENGTH,
        LAST and SIZE with a variable, INPUT and OUTPUT with a port; where
        uplm80 stopped with a traceback, `call time;', `b = rol(b);', or
        compiled what it did not take.  Intel's PL/M-80 V3.1: ERROR 154,
        INVALID NUMBER OF ARGUMENTS IN CALL, TOO FEW, and 153, TOO MANY;
        124, MISSING ARGUMENTS FOR BUILT-IN PROCEDURE, and 126, MISSING ')'
        AFTER BUILT-IN PROCEDURE ARGUMENT LIST; 109, MISSING INPUT/OUTPUT
        PORT NUMBER."""
        n = r.args or 0
        what = expr_text(r.call) if r.call is not None else text
        if name in _BUILTIN_ARGS and n != _BUILTIN_ARGS[name]:
            self.intel(r.node, f"{what}: {text} takes {_HOW_MANY[_BUILTIN_ARGS[name]]}",
                       154 if n < _BUILTIN_ARGS[name] else 153)
        elif name in ("LENGTH", "LAST", "SIZE") and n != 1:
            self.intel(r.node, f"{what}: {text} takes one argument, a variable",
                       124 if n == 0 else 126)
        elif name in ("INPUT", "OUTPUT") and r.args is None:
            self.intel(r.node, f"{text}: {text} takes a port number, in parentheses", 109)

    def _check_called(self, r: _Ref, d: _Decl, text: str) -> None:
        """What a CALL statement calls through: an ADDRESS scalar, not an
        array, a structure, a BYTE or a label, with a subscript or without.
        Intel's PL/M-80 V3.1: ERROR 118, INVALID INDIRECT CALL, IDENTIFIER
        NOT AN ADDRESS SCALAR."""
        if d.kind != "label" and d.address and d.dim is None and d.members is None:
            return
        kind = ("a label" if d.kind == "label" else "an array" if d.dim is not None
                else "a structure" if d.members is not None else "a BYTE")
        what = expr_text(r.call) if r.call is not None else text
        # And ERROR 32, INVALID SYNTAX, of what follows it in parentheses.
        self.intel(r.node, f"CALL {what}: {text} is {kind}, and a CALL calls a procedure, "
                   "or through an ADDRESS scalar (Programming Manual 9800268B, 8.2.1)",
                   *((118, 32) if r.call is not None else (118,)))

    def _check_called_member(self, u: _MemberUse) -> None:
        """A CALL through a structure's member: an ADDRESS scalar member, of
        a structure, or of an array of structures named without a
        subscript, `call sa.g', which calls through the first element's,
        as V3.1 takes it.  Not of an element, `call sa(1).g', where V3.1
        takes SA for what is called and the rest for its arguments (ERROR
        118, and 32)."""
        d, member = self._member_decl(u)
        if d is None or member not in (d.members or {}):
            return
        element = isinstance(unwrap_paren(u.node.base), P.Call)
        if not element and d.members[member] is None and member in d.address_members:
            return
        name = ident_text(u.node.member)
        kind = (f"{d.orig} is {'an array' if d.dim is not None else 'a structure'}" if element
                else f"{name} is an array" if d.members[member] is not None
                else f"{name} is a BYTE")
        callee = u.call if u.call is not None else u.node
        self.intel(u.node, f"CALL {expr_text(callee)}: {kind}, and a CALL calls a procedure, "
                   "or through an ADDRESS scalar (Programming Manual 9800268B, 8.2.1)",
                   *((118, 32) if _has_parentheses(callee) else (118,)))

    def _check_scalar(self, r: _Ref, text: str) -> None:
        """A scalar with a subscript.  `x(1)' is taken for the element of
        x's type that far past x, as if x were an array, as programs
        written for uplm80 rely on (tests/test_optimizer_soundness.py);
        `shl(w, 3)' of a scalar SHL is an error.  Intel's PL/M-80 V3.1:
        ERROR 127, INVALID SUBSCRIPT ON NON-ARRAY, and 114."""
        what = expr_text(r.call)
        if r.args > 1:
            self.intel(r.node, f"{what}: {text} is not an array, and only an array takes a "
                       "subscript, and only one", 127, 114)
            return
        self.intel(r.node, f"{what}: {text} is not an array, and this is taken for the "
                   f"element that far past {text}, as if {text} were an array", 127, warn=True)

    def _check_unsubscripted(self, r: _Ref, text: str) -> None:
        """An array without a subscript, but after a dot or in LENGTH, LAST
        or SIZE (3.6.2).  A member of an array of structures, `s2.m(4)', is
        taken for `s2(0).m(4)', as tests/test_calls_and_loops.py relies
        on, and V3.1 takes it in a CALL, `call sa.g'; the rest is an
        error.  Intel's PL/M-80 V3.1: ERROR 133, ILLEGAL
        REFERENCE TO UNSUBSCRIPTED ARRAY."""
        if r.member is not None and any(u.called and u.node is r.member
                                        for u in self.member_uses):
            return          # `call sa.g', which V3.1 takes (_check_called_member)
        if r.member is not None:
            member = ident_text(r.member.member)
            self.intel(r.node, f"{text}.{member}: {text} is an array, and this is taken for "
                       f"{text}(0).{member}", 133, warn=True)
            return
        self.intel(r.node, f"{text}: {text} is an array, and an array is named without a "
                   "subscript only as the operand of a dot or the argument of LENGTH, LAST or "
                   "SIZE (Programming Manual 9800268B, 3.6.2)", 133)

    def _extent(self, call) -> bool:
        """Whether ``call`` is a call of the built-in LENGTH, LAST or SIZE."""
        return call is not None and id(call) in self.extent_calls \
            and self.extent_calls[id(call)][1].decl is None

    def _check_forward(self, r: _Ref, d: _Decl, text: str) -> None:
        """A procedure is called after its declaration, but by a REENTRANT
        procedure, if it is REENTRANT too (MP/M II's SN.PLM has them call
        one another); its address, `.p', may be taken before.  Intel's
        PL/M-80 V3.1: ERROR 169, ILLEGAL FORWARD CALL."""
        caller = r.block.proc
        if d.node is None or r.block.module is not d.block.module or r.dot:
            return
        if d.reentrant and caller is not None and caller.reentrant:
            return
        here = (r.node.pos.start_line, r.node.pos.start_column)
        if here < (d.node.pos.start_line, d.node.pos.start_column):
            self.intel(r.node, f"{text}: procedure {text} is declared after this call of "
                       "it, and a procedure is called only after its declaration, but by a "
                       "REENTRANT procedure if it is REENTRANT too", 169)

    def _member_decl(self, u: _MemberUse) -> tuple:
        """The declaration of the structure a member use is of, if the
        program declares it, and the member's name."""
        root = unwrap_paren(u.node.base)
        while isinstance(root, (P.Call, P.MemberAccess)):
            root = unwrap_paren(root.callee if isinstance(root, P.Call) else root.base)
        d = self.lookup(_key(root.name), u.block) if isinstance(root, P.Identifier) else None
        return d, _key(u.node.member)

    def _check_recursive(self, r: _Ref, d: _Decl, text: str) -> None:
        """A procedure is called from inside itself, directly or through a
        procedure declared in it, only if it is REENTRANT (8.1.7); a CALL
        through its address is not seen.  Intel's PL/M-80 V3.1: ERROR 170,
        ILLEGAL RECURSIVE CALL."""
        if d.reentrant or r.dot:
            return
        proc = r.block.proc
        while proc is not None and proc is not d:
            proc = proc.block.proc
        if proc is d:
            self.intel(r.node, f"{text}: procedure {text} is called from inside itself, and "
                       "only a REENTRANT procedure may be (Programming Manual 9800268B, "
                       "8.1.7)", 170)

    def _check_member(self, u: _MemberUse) -> None:
        """A member array without a subscript, but after a dot or in LENGTH,
        LAST or SIZE, or with more than one; a scalar member with any.
        Intel's PL/M-80 V3.1: ERROR 134, ILLEGAL REFERENCE TO UNSUBSCRIPTED
        MEMBER ARRAY; 114, INVALID SUBSCRIPT, MULTIPLE SUBSCRIPTS ILLEGAL;
        127, INVALID SUBSCRIPT ON NON-ARRAY."""
        d, member = self._member_decl(u)
        if d is None or member not in (d.members or {}):
            return
        n, extent = u.node, self._extent(u.extent)
        name = ident_text(n.member)
        if d.members[member] is None and u.args is not None:
            # And ERROR 32, INVALID SYNTAX, of the subscript.
            self.intel(n, f"{expr_text(u.call)}: {name} is not an array, and only an array "
                       "takes a subscript", 127, 32)
        elif d.members[member] is not None and u.args is not None and u.args > 1 \
                and not extent:
            self.intel(n, f"{expr_text(u.call)}: {name} is an array, and an array takes one "
                       "subscript", 114)
        elif d.members[member] is not None and u.args is None and not (u.dot or extent):
            self.intel(n, f"{expr_text(n)}: {name} is an array, and a member array is named "
                       "without a subscript only as the operand of a dot or the argument of "
                       "LENGTH, LAST or SIZE (Programming Manual 9800268B, 3.6.2)", 134)

    def _check_extent(self, call: P.Call) -> None:
        """The subscripts of the argument of LENGTH, LAST or SIZE, which
        are not evaluated: Intel's PL/M-80 V3.1 takes none with anything in
        parentheses, a call, a subscript or a parenthesized expression
        (ERROR 32, INVALID SYNTAX; and of LENGTH and LAST, ERROR 125,
        ILLEGAL ARGUMENT FOR BUILT-IN PROCEDURE, before it)."""
        ref = unwrap_paren(call.args[0])
        while isinstance(ref, (P.Call, P.MemberAccess)):
            if isinstance(ref, P.Call):
                if any(_has_parentheses(a) for a in ref.args):
                    name = ident_text(unwrap_paren(call.callee).name)
                    # LENGTH and LAST: and ERROR 125, ILLEGAL ARGUMENT FOR
                    # BUILT-IN PROCEDURE, of what V3.1 read before the text
                    # it ignores.
                    numbers = (32,) if _key(unwrap_paren(call.callee).name) == "SIZE" else (125, 32)
                    self.intel(call, f"{expr_text(call)}: the subscripts of {name}'s "
                               "argument are not evaluated, and none has anything in "
                               "parentheses in it, a call, a subscript or an expression", *numbers)
                    return
                ref = unwrap_paren(ref.callee)
            else:
                ref = unwrap_paren(ref.base)

    # What a restricted expression - a DATA or INITIAL value, an AT
    # address, a constant in a constant list `.(1, 'a')', or the subscript
    # of a location in one of them - does not take, and the errors Intel's
    # PL/M-80 V3.1 gives for it there, in the order of _CONTEXTS (each form
    # checked with scripts/intel_oracle.py, in each place and first, second
    # and alone in a constant list); what V3.1 gives of the list it is in,
    # #209, #210, #32 and #172, _check_value_list and _check_constant_list
    # add:
    _RESTRICTED = {
        "operator": ((152,), (146,), (152,), (150,)),           # 2 * 3, 1 < 2
        "not": ((151, 152), (146, 151), (151, 152), (150, 151)),
        "paren": ((151, 152), (146, 151), (151, 152), (150, 151)),         # (1), -(1)
        "call": ((151, 152), (146, 151), (151, 152), (150, 151)),          # a(1), shl(1, 2)
        "member": ((151, 152), (146, 151), (151, 152), (150, 151)),        # s.k
        "string": ((151, 152), (146, 151), (151, 152), (150, 151)),        # 1 + 'A'
        "string-first": ((152,), (146, 151), (152,), (150, 151)),          # 'A' + 1
        "location": ((151, 152), (146, 151), (151, 152), (150, 151)),      # 1 + .a, -.a
        "name": ((151,), (151,), (151,), (151,)),                          # x, memory
        "negation": ((151,), (151,), (151,), (151,)),                      # - -1
        "constants": ((147,), (147,), (147,), (147,)),                     # .(5)
        "text": ((147,), (147,), (147,), (147,)),                          # .'AB'
        "subscripts": ((152,), (146,), (152,), ()),                        # .a(1)(1)
        "subscripting": ((149,), (149,), (149,), ()),                      # .x(1), .p(1)
        "subscript-list": ((150,), (150,), (150,), ()),                    # .a(1, 2)
        "at-name": ((), (211,), (), ()),                   # at (.stackptr), at (.p), at (.lbl)
        "at-based": ((), (212,), (), ()),                                  # at (.bb)
        "constant-location": ((), (), (), ()),                             # .(.a)
    }
    _CONTEXTS = ("list", "at", "consts", "subscript")
    # What V3.1 reads past, going on to the rest of the value.
    _READ_PAST = frozenset({"name", "negation", "constant-location", "subscripting",
                            "subscript-list", "at-name", "at-based"})
    # The order V3.1 lists its errors in.
    _V31_ORDER = (149, 151, 152, 146, 150, 147, 32, 210, 209, 211, 212, 172)
    _RULES = {
        "list": ("a DATA or INITIAL value", "is a restricted expression, of constants and "
                 "locations only"),
        "at": ("an AT address", "is a restricted expression, a constant or a location plus "
               "or minus constants"),
        "consts": ("a constant list", "holds constants only"),
    }

    def check_restricted(self) -> None:
        """Each restricted expression - a DATA or INITIAL value, an AT
        address, a constant of a constant list `.(1, 'a')' - is what
        Intel's PL/M-80 V3.1 takes (Programming Manual 9800268B, 4.1.3,
        6.2.8, 6.2.9): numbers, added and subtracted, a minus sign before a
        number; in a DATA or INITIAL list a string alone, and a location,
        `.a', `.s.m(2)', `.memory', plus or minus numbers; in an AT a
        location; in a constant list a string alone, and numbers a byte
        holds.  A subscript of a location is numbers too.  Anything else
        is an error, with V3.1's errors: an operator but + and -, NOT,
        parentheses, a name not after a dot - a variable, a procedure, a
        built-in, `x', `a(1)', `s.k', `shl(1, 2)', `memory' - a string in a
        sum, a location anywhere but first, a constant list or `.'text''
        (V3.1 takes neither there), and in a constant list, or in a value
        that fills a BYTE, a location or what a byte does not hold,
        `.(300)', `byte data (300)'.  So is a location with two subscripts,
        `.a(1)(1)', or with one on what is not an array, `.x(1)', `.p(1)',
        and in an AT the location of a procedure, a label, a built-in but
        MEMORY or a BASED variable.  uplm80 took most of them, for the
        address of a name, the low byte of a number, or, at -O1 and up, a
        built-in folded (0.4.3's Known issues).  The message names every
        error V3.1 gives for the list the value is in, as V3.1 reads it
        (:meth:`_check_value_list`, :meth:`_check_constant_list`).  A list
        that names what the program does not declare, or a LITERALLY
        declared further on, is left to check_uses, which refuses it in its
        own words (0.4.4's Known issues): the message named the errors of
        the rest of the list, and in a constant list #210 of `.zz(1)' as of
        a location, where V3.1 gives #105 besides."""
        refs = {id(r.node): r for r in self.refs if r.attr == "name" and not r.goto}
        for where, node in self.restricted:
            if _names_undeclared(node, refs):
                continue
            if where == "consts":
                self._check_constant_list(node, refs)
            elif where == "list":
                self._check_value_list(node, refs)
            else:
                faults, numbers, _ = self._walk(node, "at", refs)
                if faults:
                    self._restricted_error(faults, "at", numbers, refs)

    def _walk(self, expr, where: str, refs: dict) -> tuple[list, set[int], object]:
        """The faults of the restricted expression ``expr`` in ``where``
        (:meth:`_restricted`), the errors V3.1 gives for them - those of
        each fault it reads, up to the first it does not read past - and
        the node it stops at there, None where it reads all of ``expr``."""
        faults: list = []
        self._restricted(expr, where, "top", refs, faults)
        numbers: set[int] = set()
        for kind, node, ctx in faults:
            numbers.update(self._RESTRICTED[kind][self._CONTEXTS.index(ctx)])
            if kind not in self._READ_PAST:
                if ctx == "subscript" and _two_subscripts(node):
                    # What V3.1 does not take in the subscript of a location
                    # has two subscripts itself, `.a(.a(1)(1))', `.a(b(1)(1))':
                    # V3.1 gives the second's error too, as in the value.
                    numbers.update(self._RESTRICTED["subscripts"][self._CONTEXTS.index(where)])
                return faults, numbers, node
        return faults, numbers, None

    def _check_value_list(self, values: list, refs: dict) -> None:
        """A DATA or INITIAL list, the errors of its values together, as
        Intel's PL/M-80 V3.1 reads them, up to the first it does not read
        to the end (:meth:`_walk`).  A value that fills a BYTE is #210 if
        it is more than 255 (:func:`_over_a_byte`); one that does not fit
        in the space its declaration has is #209, ILLEGAL INITIALIZATION OF
        MORE SPACE THAN DECLARED, and V3.1 does not check it further - a
        list uplm80 lays out after the declaration where nothing else in
        it is wrong (0.4.3's and 0.4.4's Known issues): `declare b (2)
        byte data (1, 2, x)' is #151 and #209, `declare b (3) byte data (x,
        300, 7)' #151 and #210, `declare b byte data (1, 300)' #209
        alone."""
        every: list = []
        errors: set[int] = set()
        for v in values:
            faults, numbers, stop = self._walk(v, "list", refs)
            errors |= numbers
            if self._slots.get(id(v)) == "byte" and _over_a_byte(v, stop):
                errors.add(210)
                if not faults:
                    faults = [("byte-location" if isinstance(_lead(v), P.LocationOf)
                               else "range", v, "list")]
            if id(v) in self._past:
                errors.add(209)
            every.extend(faults)
            if stop is not None:
                break
        if every:
            self._restricted_error(every, "list", errors, refs)

    # What makes a value of a constant list a word to V3.1: a name, or a
    # location first.
    _WORDS = frozenset({"name", "call", "member", "constant-location"})

    def _check_constant_list(self, node, refs: dict) -> None:
        """A constant list, `.(1, 'a')', which Intel's PL/M-80 V3.1 lays
        out as it lays out an untyped DATA list: the errors of its values
        and of the list together, as V3.1 gives them (each rule checked
        with scripts/intel_oracle.py).  V3.1 reads the values up to the
        first it does not read to the end (:meth:`_walk`), and from there
        looks for the list's `)': a `(' on the way is #32, INVALID SYNTAX,
        TEXT IGNORED UNTIL ';', `.((1), 7)', `.(-.a, (1))'.  A value with a
        name in it, or a location first, is a word to it, and a string, or
        a value that starts with one, after the last such value makes the
        list a list of bytes again.  Where it does not, the list has room
        for one byte: the first value is laid out, and the second and each
        after it is #209, ILLEGAL INITIALIZATION OF MORE SPACE THAN
        DECLARED, and not checked further, `.(x, 7)', `.(7, x)', `.(7, 300,
        x)'.  A value laid out that is more than 255 (:func:`_over_a_byte`)
        is #210, `.(.a)', `.(300 + x)', `.(x, '$', 300)'; and a list of one
        value, or cut short after its first, whose last name is a
        built-in's but MEMORY's, #172, INVALID LABEL: UNDEFINED,
        `.(stackptr)', `.(1 + time)', `.(.shl)', `.(shl(1, 2), 3)'."""
        values = list(node.values or [])
        read = []
        for v in values:
            faults, numbers, stop = self._walk(v, "consts", refs)
            read.append((v, faults, numbers, stop))
            if stop is not None:
                # V3.1 looks for the list's `)' from there, and a `(' on the
                # way makes it take another's: the rest of the statement is
                # #32, INVALID SYNTAX, TEXT IGNORED UNTIL ';'.
                skipped = [stop.right if isinstance(stop, P.BinaryOp) else stop]
                if _has_parentheses(skipped + values[len(read):]):
                    numbers.add(32)
                break
        bytes_only = True
        for v, faults, _, _ in read:
            if any(kind in self._WORDS for kind, _, _ in faults):
                bytes_only = False
            elif isinstance(_lead(v), P.StringLiteral):
                bytes_only = True
        every: list = []
        errors: set[int] = set()
        for i, (v, faults, numbers, stop) in enumerate(read):
            errors |= numbers
            if (bytes_only or i == 0) and _over_a_byte(v, stop):
                errors.add(210)
                if not faults:
                    faults = [("range", v, "consts")]
            every.extend(faults)
        if not bytes_only and len(read) > 1:
            errors.add(209)
        if len(read) == 1 and self._last_name_builtin(read[0][1], refs):
            errors.add(172)
        if every:
            self._restricted_error(every, "consts", errors, refs)

    @staticmethod
    def _last_name_builtin(faults: list, refs: dict) -> bool:
        """Whether the last name V3.1 reads of the faults ``faults``, up to
        the first it does not read past, is a built-in's but MEMORY's."""
        roots = []
        for kind, node, _ in faults:
            if kind in _Resolver._WORDS:
                root = node.operand if kind == "constant-location" else node
                root = unwrap_paren(root)
                while isinstance(root, (P.Call, P.MemberAccess)):
                    root = unwrap_paren(root.callee if isinstance(root, P.Call) else root.base)
                if isinstance(root, P.Identifier):
                    roots.append(root)
            if kind not in _Resolver._READ_PAST:
                break
        if not roots:
            return False
        last = max(roots, key=lambda n: (n.pos.start_line, n.pos.start_column))
        r = refs.get(id(last))
        return r is not None and r.decl is None and _key(last.name) in _BUILTINS - {"MEMORY"}

    def _restricted(self, e, ctx: str, role: str, refs: dict, faults: list) -> None:  # pylint: disable=too-many-return-statements,too-many-branches
        """The faults of ``e``, part of a restricted expression in ``ctx``,
        in the order V3.1 reads them: (kind, node, context).  ``role``:
        "top", the whole value; "left" or "right", an operand of + or -;
        "neg", the operand of a minus sign."""
        if isinstance(e, P.NumberLiteral):
            return
        if isinstance(e, P.StringLiteral):
            if ctx in ("at", "subscript") or role in ("right", "neg"):
                faults.append(("string", e, ctx))
            elif role == "left":
                faults.append(("string-first", e, ctx))
            return
        if isinstance(e, P.ParenExpr):
            faults.append(("paren", e, ctx))
        elif isinstance(e, P.UnaryOp):
            inner = e.operand
            if unop_kind(e) == UnaryOpKind.NOT:
                faults.append(("not", e, ctx))
            elif isinstance(inner, P.LocationOf):
                faults.append(("location", e, ctx))
            elif isinstance(inner, P.UnaryOp) and unop_kind(inner) == UnaryOpKind.NEG:
                faults.append(("negation", e, ctx))
                self._restricted(inner.operand, ctx, "neg", refs, faults)
            else:
                self._restricted(inner, ctx, "neg", refs, faults)
        elif isinstance(e, P.BinaryOp):
            self._restricted(e.left, ctx, "left", refs, faults)
            if faults and faults[-1][0] not in self._READ_PAST:
                return
            if binop_kind(e) not in (BinaryOpKind.ADD, BinaryOpKind.SUB):
                faults.append(("operator", e, ctx))
            elif isinstance(e.right, P.LocationOf):
                faults.append(("location", e.right, ctx))
            else:
                self._restricted(e.right, ctx, "right", refs, faults)
        elif isinstance(e, P.LocationOf):
            if role in ("right", "neg") or ctx == "subscript":
                faults.append(("location", e, ctx))
                return
            self._restricted_location(e.operand, ctx, refs, faults)
            if ctx == "consts":
                faults.append(("constant-location", e, ctx))
        elif isinstance(e, (P.LocationOfList, P.LocationOfString)) and ctx == "subscript":
            faults.append(("location", e, ctx))
        elif isinstance(e, P.LocationOfList):
            faults.append(("constants", e, ctx))
        elif isinstance(e, P.LocationOfString):
            faults.append(("text", e, ctx))
        elif isinstance(e, (P.Identifier, P.Call, P.MemberAccess)):
            root = e
            while isinstance(root, (P.Call, P.MemberAccess)):
                root = unwrap_paren(root.callee if isinstance(root, P.Call) else root.base)
            if not isinstance(root, P.Identifier) or not self._restricted_name(refs.get(id(root))):
                return      # declared nowhere, or a LITERALLY's: check_uses
            kind = ("name" if isinstance(e, P.Identifier) else
                    "call" if isinstance(e, P.Call) else "member")
            faults.append((kind, e, ctx))

    def _restricted_location(self, d, ctx: str, refs: dict, faults: list) -> None:
        """The faults of what a dot takes the location of, in ``ctx``, in
        the order V3.1 reads them: in an AT, a procedure's, a label's, a
        built-in's but MEMORY's (V3.1: ERROR 211) or a BASED variable's
        (212); a subscript on what is not an array (149); a second
        subscript, where V3.1 stops as at an operator it does not take
        (152, in an AT 146); the first subscript's faults; and more than
        one in its parentheses, `.a(1, 2)', where V3.1 misses the `)' after
        the first (150) and goes on after the one that ends the rest,
        which it does not read, `.a(1, x)' #150 alone - but of a built-in
        but MEMORY, which uplm80 refuses in its own words
        (:meth:`_check_builtin_use`): V3.1 gives #149 besides of
        `.input(1, 2)', and not of `.stackptr(1, 2)'."""
        d = unwrap_paren(d)
        if isinstance(d, P.Identifier):
            r = refs.get(id(d))
            kind = self._at_fault(d, r) if ctx == "at" and r is not None else None
            if kind is not None:
                faults.append((kind, d, ctx))
        elif isinstance(d, P.Call):
            callee = unwrap_paren(d.callee)
            self._restricted_location(callee, ctx, refs, faults)
            if faults and faults[-1][0] not in self._READ_PAST:
                return
            if isinstance(callee, P.Call):
                faults.append(("subscripts", d, ctx))
                return
            if not self._an_array(callee, refs):
                faults.append(("subscripting", d, ctx))
            for a in d.args[:1]:
                if faults and faults[-1][0] not in self._READ_PAST:
                    return
                self._restricted(a, "subscript", "top", refs, faults)
            if len(d.args) > 1 and not _a_builtin(callee, refs) \
                    and not (faults and faults[-1][0] not in self._READ_PAST):
                faults.append(("subscript-list", d, ctx))
        elif isinstance(d, P.MemberAccess):
            self._restricted_location(d.base, ctx, refs, faults)

    @staticmethod
    def _at_fault(d: P.Identifier, r: _Ref) -> str | None:
        """What V3.1 does not take of the location of ``d``, bound as
        ``r``, in an AT: "at-name", a procedure's, a label's or a built-in's
        but MEMORY's, "at-based", a BASED variable's, or None."""
        if r.decl is None:
            name = _key(d.name)
            return "at-name" if name in _BUILTINS and name != "MEMORY" else None
        if r.decl.kind in ("proc", "label"):
            return "at-name"
        return "at-based" if r.decl.based else None

    def _an_array(self, ref, refs: dict) -> bool:
        """Whether the reference ``ref`` a subscript follows in a location
        is of an array, or of what check_uses or code generation refuse
        anyway: a name declared nowhere, a member the structure does not
        have.  Of the built-ins MEMORY is an array; V3.1 takes the location
        of another, subscripted or not, `.stackptr(1)', for an address of
        its own, which uplm80 refuses (:meth:`_check_builtin_use`)."""
        if isinstance(ref, P.Identifier):
            r = refs.get(id(ref))
            d = r.decl if r is not None else None
            if d is None or d.kind == "lit":
                return True
            return d.kind == "var" and d.dim is not None
        if isinstance(ref, P.MemberAccess):
            root = unwrap_paren(ref.base)
            while isinstance(root, (P.Call, P.MemberAccess)):
                root = unwrap_paren(root.callee if isinstance(root, P.Call) else root.base)
            r = refs.get(id(root)) if isinstance(root, P.Identifier) else None
            d = r.decl if r is not None else None
            member = _key(ref.member)
            if d is None or not d.members or member not in d.members:
                return True
            return d.members[member] is not None
        return True

    @staticmethod
    def _restricted_name(r: _Ref | None) -> bool:
        """Whether the name ``r`` is of is one a restricted expression
        does not take: declared, but not a LITERALLY, or a built-in."""
        if r is None:
            return False
        if r.decl is None:
            return _key(getattr(r.node, r.attr)) in _BUILTINS
        return r.decl.kind != "lit"

    def _restricted_error(self, faults: list, where: str, numbers: set[int],
                          refs: dict) -> None:
        """Refuse a restricted expression in ``where`` for the first of its
        ``faults``, naming the errors V3.1 gives for it, ``numbers``."""
        kind, node, ctx = faults[0]
        self.intel(node, self._restricted_text(kind, node, ctx, where, refs),
                   *sorted(numbers, key=self._V31_ORDER.index))

    def _restricted_text(self, kind: str, node, ctx: str, where: str, refs: dict) -> str:  # pylint: disable=too-many-arguments,too-many-return-statements
        """What the message says of the fault ``kind`` at ``node``, in
        ``ctx``, of a value in ``where``."""
        what = expr_text(node)
        subject, rule = self._RULES[where]
        if kind in ("name", "call", "member"):
            root = node
            while isinstance(root, (P.Call, P.MemberAccess)):
                root = unwrap_paren(root.callee if isinstance(root, P.Call) else root.base)
            text = ident_text(root.name)
            if kind == "member":
                return f"{what}: {what} is a member of structure {text}, and {subject} {rule}"
            return f"{what}: {text} is {self._name_kind(refs.get(id(root)))}, and {subject} {rule}"
        if ctx == "subscript":
            subject = f"the subscript of a location in {subject}"
        if kind in ("operator", "not"):
            return f"{what}: of the operators, {subject} takes + and - only"
        if kind == "paren":
            return f"{what}: {subject} has nothing in parentheses"
        if kind == "negation":
            return f"{what}: in {subject} a minus sign goes before a number only"
        if kind in ("string", "string-first"):
            if ctx in ("at", "subscript"):
                return f"{what}: {subject} has no string in it"
            return (f"{what}: a string in {subject} is a value of its own, not added to or "
                    "subtracted from")
        if kind == "location":
            if ctx == "subscript":
                return f"{what}: {subject} is numbers only"
            if where == "consts":
                return f"{what}: {subject} {rule}, and a location is none"
            return (f"{what}: {subject} is a location plus or minus constants, the location "
                    "first, or constants alone")
        if kind == "constant-location":
            return f"{what}: {subject} {rule}, and {what} is a location"
        if kind in ("constants", "text"):
            return f"{what}: {subject} does not take the location of constants"
        if kind == "at-name":
            text = ident_text(node.name)
            return (f".{text}: {text} is {self._name_kind(refs.get(id(node)))}, and the location "
                    "in an AT address is a variable's, or MEMORY's")
        if kind == "at-based":
            text = ident_text(node.name)
            return f".{text}: {text} is BASED, and has no fixed address for an AT address to name"
        if kind == "subscripts":
            return f"{what}: a location takes one subscript, not two"
        if kind == "subscript-list":
            return f"{what}: a location takes one subscript"
        if kind == "subscripting":
            callee = unwrap_paren(node.callee)
            d = refs.get(id(callee)).decl if isinstance(callee, P.Identifier) else None
            name = expr_text(callee)
            if d is not None and d.kind in ("proc", "label"):
                return (f"{what}: {name} is {self._name_kind(refs.get(id(callee)))}, and only an "
                        "array's location takes a subscript")
            return f"{what}: {name} is not an array, and only an array's location takes a subscript"
        if kind == "byte-location":
            return f"{what}: a location is an address, and this value fills a BYTE"
        value = f"{_v31_read(node, None)[0][0]:X}H"
        value = "0" + value if value[0] in "ABCDEF" else value
        if where == "consts":
            return f"{what}: {subject} holds bytes, and {what} is {value}, more than 0FFH"
        return f"{what}: this value fills a BYTE, and {what} is {value}, more than 0FFH"

    @staticmethod
    def _name_kind(r: _Ref | None) -> str:
        """What the name ``r`` is of is, in a message."""
        d = r.decl if r is not None else None
        if d is None:
            return "a built-in"
        if d.kind == "var":
            return "an array" if d.dim is not None else "a variable"
        return {"param": "a parameter", "proc": "a procedure", "label": "a label"}.get(
            d.kind, "a name")

    @staticmethod
    def _check_builtin_use(r: _Ref, text: str) -> None:
        """A built-in's name after a dot, or before empty parentheses: of
        the built-ins only MEMORY has an address, and none takes an empty
        argument list.  Intel's PL/M-80 V3.1: ERROR 123, INVALID DOT
        OPERAND, BUILT-IN PROCEDURE ILLEGAL; ERROR 102, MISSING PRIMARY
        OPERAND (and 153, INVALID NUMBER OF ARGUMENTS IN CALL)."""
        if r.in_at:
            return
        if r.dot and _key(getattr(r.node, r.attr)) != "MEMORY" and r.in_list:
            # Intel's PL/M-80 V3.1 takes it there, for an address of its own
            # (057CH, 0103H in the programs checked), which uplm80 has no
            # counterpart of (0.4.4's Known issues).
            raise CodeGenError(
                f".{text}: {text} is a built-in, and of the built-ins only MEMORY has an "
                f"address; uplm80 has none to give {text} in a DATA or INITIAL list, where "
                f"Intel's PL/M-80 V3.1 takes .{text} for an address of its own",
                source_location(r.node))
        if r.dot and _key(getattr(r.node, r.attr)) != "MEMORY":
            raise CodeGenError(
                f".{text}: {text} is a built-in, and of the built-ins only MEMORY has an "
                "address; Intel's PL/M-80 V3.1 rejects it (ERROR #123, INVALID DOT "
                "OPERAND, BUILT-IN PROCEDURE ILLEGAL)", source_location(r.node))
        if r.empty:
            raise CodeGenError(
                f"{text}(): {text} is a built-in, and PL/M-80 has neither an empty "
                "subscript nor an empty argument list", source_location(r.node))

    # The kind of declaration renamed first when two meet in one assembler
    # name: a LITERALLY, whose EQU nothing uses (the macro pass has put its
    # text in place of every use), then a label, which nothing outside its
    # procedure names, then a procedure, and a variable last.
    _RENAME_FIRST = {"lit": 0, "label": 1, "proc": 2, "var": 3, "param": 3}

    def settle(self) -> None:
        """Rename declarations until code generation keeps them all apart."""
        self.used = {d.name for d in self.decls}
        for _ in range(len(self.decls) + 1):
            if not (self._settle_procedures() or self._settle_labels()
                    or self._settle_lookups() or self._settle_bases()):
                return

    def _settle_procedures(self) -> bool:
        """Code generation files every procedure under one table by its
        key, its enclosing procedures' names and its own (P$Q).  Two in
        sibling DO blocks of one procedure have one key and met there -
        and in the assembly, @P$Q "multiply defined"."""
        by_key: dict[str, list[_Decl]] = {}
        for d in self.decls:
            if d.kind == "proc":
                by_key.setdefault(self.proc_key(d).upper(), []).append(d)
        renamed = False
        for group in by_key.values():
            mine = [d for d in group if not d.shared]
            for d in mine[1:] if len(mine) == len(group) else mine:
                self.rename(d, self.fresh(d.name))
                renamed = True
        return renamed

    def _settle_labels(self) -> bool:
        """No two declarations defining one assembler label."""
        by_asm: dict[str, list[_Decl]] = {}
        for d in self.decls:
            name = self.asm_name(d)
            if name is not None:
                by_asm.setdefault(name.upper(), []).append(d)
        renamed = False
        for group in by_asm.values():
            if len(group) < 2:
                continue
            keep = [d for d in group if d.shared]
            if not keep:
                keep = [max(group, key=lambda d: (self._RENAME_FIRST[d.kind], -d.order))]
            for d in group:
                if d in keep or (d.kind == "lit" and all(
                        k.kind == "lit" and k.literal == d.literal for k in keep)):
                    continue
                self.rename(d, self.fresh(d.name))
                renamed = True
        return renamed

    def _settle_lookups(self) -> bool:
        """Every name found where code generation looks for it.

        It looks a name up as a procedure nested in each procedure around
        the use, innermost first, before anything else (CodeGenerator.
        _lookup_symbol): a procedure declared in a DO block of P, or in P
        itself, took the name from a variable declared in another of P's
        blocks, or in a procedure nested in P, or outside P - silently the
        wrong object.  Such a procedure is renamed.
        """
        procs: dict[str, _Decl] = {}
        for d in self.decls:
            if d.kind == "proc":
                procs.setdefault(self.proc_key(d).upper(), d)
        wrong: set[int] = set()
        for r in self.refs:
            if r.goto or r.decl is None:
                continue
            found = self._codegen_lookup(r.decl.name, r.block, procs)
            if found is r.decl:
                continue
            # The procedure goes: its name is the one code generation takes
            # for another's, or it is not found by its own.
            proc = found if found is not None and found.kind == "proc" else r.decl
            if proc.kind == "proc" and not proc.shared and id(proc) not in wrong:
                wrong.add(id(proc))
                self.rename(proc, self.fresh(proc.name))
        return bool(wrong)

    def _codegen_lookup(self, name: str, block: _Block, procs: dict) -> _Decl | None:
        """What code generation finds for ``name`` used in ``block``."""
        if block.proc is not None:
            parts = self.proc_key(block.proc).upper().split("$")
            for i in range(len(parts), 0, -1):
                d = procs.get("$".join(parts[:i]) + "$" + name)
                if d is not None:
                    return d
        # Then its scopes, which hold everything but procedures: those are
        # all in the outermost, under their keys.  A procedure a DO block
        # declares is behind every variable of its name in scope.
        b: _Block | None = block
        while b is not None:
            d = b.decls.get(name)
            if d is not None and d.kind != "proc":
                return d
            b = b.parent
        return procs.get(name)

    def _settle_bases(self) -> bool:
        """Every base found where it is looked up by its name: by code
        generation where the BASED variable is used, and by local storage
        where it is declared (local_storage._read_base), each taking the
        innermost declaration of the name there.  A block that declares the
        name hid the base: a procedure's `q' where a BASED variable of the
        module's `q' is used in it, `a = 77h', which stored through the
        procedure's; and a variable declared further down the block, which
        Intel's PL/M-80 V3.1 does not take for the base, but the one of an
        outer block (bind).  What hides the base is renamed, or the base,
        where what hides it is PUBLIC or EXTERNAL."""
        for r in self.refs:
            if r.base is None or r.decl is None:
                continue
            for use in [r] + self.refs_of.get(id(r.base[0]), []):
                found = self._scope_lookup(r.decl.name, use.block)
                if found is None or found is r.decl:
                    continue
                hider = r.decl if found.shared else found
                if not hider.shared:
                    self.rename(hider, self.fresh(hider.name))
                    return True
        return False

    @staticmethod
    def _scope_lookup(name: str, block: _Block) -> _Decl | None:
        """What code generation's scopes, which hold everything but
        procedures, give for ``name`` in ``block`` (SymbolTable.lookup)."""
        b: _Block | None = block
        while b is not None:
            d = b.decls.get(name)
            if d is not None and d.kind != "proc":
                return d
            b = b.parent
        return None

    # ---- GOTO ----------------------------------------------------------

    def check_gotos(self) -> None:
        """Hold every GOTO to 9.3, and note the label each jumps to."""
        for r in self.refs:
            if not r.goto:
                continue
            text = ident_text(r.node.label)
            where = source_location(r.node)
            d = r.decl
            if d is None:
                raise CodeGenError(self._undeclared(text, r), where)
            if d.kind != "label":
                raise CodeGenError(f"GOTO {text}: {text} is not a label", where)
            if not d.defined and not d.external:
                raise CodeGenError(f"GOTO {text}: {text} is declared a LABEL "
                                   "but labels no statement", where)
            here, there = r.block.proc, d.block.proc
            if here is not there and not (
                    d.external or (d.block.kind == "module" and d.block.module.main)):
                if there is not None:
                    raise CodeGenError(
                        f"GOTO {text} leaves procedure {here.orig} for a label in "
                        f"procedure {there.orig}; {GOTO_RULE}", where)
                self.warnings.append((where, (
                    f"GOTO {text} leaves procedure {here.orig} for a label in a DO "
                    f"block of the main program; {DO_BLOCK_GOTO}")))
            if here is not there:
                d.from_proc = True
            r.node.uplm80_asm = self.asm_name(d)

    def check_public_labels(self) -> None:
        """A PUBLIC label labels a statement at the outer level of the main
        program module (9.3).  One that labels none - a label of that name
        in a DO block is another label, the block's - is a PUBLIC name
        with no definition, which only the linker would find."""
        for d in self.decls:
            if d.kind != "label" or not d.public or d.defined:
                continue
            node, attr = d.sites[0]
            text = ident_text(getattr(node, attr))
            inner = next((x for x in self.decls if x.kind == "label" and x.orig == d.orig
                          and x.defined and x.block.module is d.block.module), None)
            also = (f"; the {text}: in {self._place(inner)} is another label, that block's"
                    if inner is not None else "")
            raise CodeGenError(
                f"{text} is declared a PUBLIC LABEL but labels no statement at the outer "
                "level of the main program module, where PL/M-80 requires a PUBLIC label "
                f"to be (Programming Manual 9800268B, 9.3){also}", source_location(node))

    @staticmethod
    def _place(d: _Decl) -> str:
        """The block a declaration is in, in words."""
        return ("a DO block" if d.block.kind == "do"
                else f"procedure {d.block.proc.orig}" if d.block.proc is not None
                else "the main program")

    def _undeclared(self, text: str, r: _Ref) -> str:
        """Why a GOTO names no label it can reach."""
        name = _key(r.node.label)
        for d in self.decls:
            if d.kind == "label" and d.orig == name and d.block.module is r.block.module:
                return (f"GOTO {text}: the label {text} is in {self._place(d)} this GOTO "
                        f"is not in; {SCOPE_RULE}")
        return f"GOTO {text}: no label {text} is declared here"

    def annotate(self) -> None:
        """Tell code generation what each label, and each other name for
        one, is called in the assembly, and which labels reload SP.

        A GOTO out of a procedure leaves on the stack the return addresses
        of the calls it abandons, and whatever they pushed; DRI's PL/M-80
        sets SP again, to the main program's stack, at a label such a GOTO
        can reach (MP/M II's PIP.PRL: procedure ERROR ends `JMP RETRY',
        and RETRY: begins `LXI SP' as PIPENTRY-3 does), and at no other.
        Such a label is at the outer level of the main program (the GOTO
        rule), where nothing else is on the stack.  A PUBLIC label may be
        reached by a GOTO in another module's procedure, compiled apart.

        And tell it which declaration a name in a DATA or INITIAL list or
        an AT address means, whatever its form (:meth:`_binding`):
        ``uplm80_decl``, or ``uplm80_builtin``, the built-in.  PL/M-80
        scopes a name to its whole block (9.1), and a DATA list or an AT
        comes before a declaration further down the block, which code
        generation has not reached there: it took a procedure's MEMORY
        declared after its DATA for the end of the program, `.arr(2)' for
        the module's array of the name, and, in a multi-file compile, the
        MEMORY another module makes PUBLIC for this one's, which does not
        declare it.  It looked up so each name it was not told of, a
        factored BASED declaration's among them: `declare (memory based
        bp) byte' further on, or at module level, where the module's DATA
        is laid out first, was the end of the program; a `b2' declared so
        further on in a procedure, the module's `b2'; and one no other
        block declares, um80's "Undefined symbol".
        """
        for d in self.decls:
            if d.kind != "label":
                continue
            name = self.asm_name(d)
            reload = d.block.kind == "module" and (d.from_proc or (d.public and d.defined))
            for node, _ in d.sites:
                if isinstance(node, P.LabeledStmt):
                    node.uplm80_asm = name
                    node.uplm80_reload_sp = reload
            for r in self.refs_of.get(id(d), []):
                if not r.goto:
                    r.node.uplm80_asm = name
        for r in self.refs:
            if r.goto or not (r.in_list or r.in_at):
                continue
            # A location in a DATA or INITIAL list or an AT address: the
            # declaration it names, which may be further down its block.
            d = r.decl
            if d is None and _key(getattr(r.node, r.attr)) in _BUILTINS:
                r.node.uplm80_builtin = True
            elif d is not None:
                r.node.uplm80_decl = self._binding(d)

    @staticmethod
    def _binding(d: _Decl) -> tuple[str, object, bool]:
        """What code generation resolves a name in a DATA or INITIAL list
        or an AT bound to ``d`` through: (its kind, the node that declares
        it, whether at module level).  A variable's is its DeclItem, or the
        factored BASED declaration `(a based p, b based p) byte' it is one
        of; a parameter's its procedure's declaration, a procedure's its
        own.  A label's is its name in the assembly too (``uplm80_asm``);
        a LITERALLY's name is not there, the macro pass having put its
        text in its place, but where it is declared further down, which
        check_uses refuses, as Intel's PL/M-80 V3.1 does (ERROR 105)."""
        module = d.block.kind == "module"
        if d.kind == "param":
            return "param", d.block.proc.node, False
        if d.kind == "proc":
            return "proc", d.node, module
        return d.kind, d.item, module

    def write_back(self) -> None:
        """Spell every renamed declaration's new name wherever it is named."""
        for d in self.decls:
            if d.name == d.orig:
                continue
            for node, attr in d.sites + [(r.node, r.attr) for r in self.refs_of.get(id(d), [])]:
                setattr(node, attr, _retext(getattr(node, attr), d.name))


def _null(p: P.ProcDecl) -> tuple:
    """The errors Intel's PL/M-80 V3.1 gives of a procedure with no
    statements: ERROR 174, INVALID NULL PROCEDURE, and of a typed one, which
    has no RETURN either, 156, MISSING RETURN STATEMENT IN TYPED
    PROCEDURE."""
    return (174, 156) if proc_return_type(p) is not None else (174,)


def _has_statements(p: P.ProcDecl) -> bool:
    """Whether the procedure ``p`` has a statement of its own."""
    return any(not isinstance(it, (P.DeclareStmt, P.ProcDecl)) and not is_end_of_block(it)
               for it in p.body.items)


def _lead(expr):
    """What a restricted expression starts with: the left operand of its
    sums and differences, all the way down."""
    while isinstance(expr, P.BinaryOp):
        expr = expr.left
    return expr


def _value_slots(item) -> tuple[dict[int, str], set[int]]:
    """What each value of the DATA or INITIAL list of ``item`` fills, by
    id - "byte", "word", or "past" the space the declaration has - and the
    values that do not fit in it, which Intel's PL/M-80 V3.1 rejects (ERROR
    209) and uplm80 lays out after it (0.4.3's and 0.4.4's Known issues).
    Each value fills the next of the scalars declared, in order: of a BYTE
    every one, of an ADDRESS none, of a STRUCTURE those that fill its BYTE
    members; a string fills one scalar to a character, or to two of an
    ADDRESS.  An untyped DATA, or an array of (*), is as long as its list,
    as uplm80 takes it (_dimension)."""
    attrs = decl_attrs(item)
    values = attrs.data_values or attrs.initial_values or []
    members = decl_item_struct_members(item)
    dtype, dim = decl_item_type(item)
    if members is None and (dtype is None or dim == -1):
        return {id(v): "byte" if dtype in (None, DataType.BYTE) else "word"
                for v in values}, set()
    if members is None:
        one = [1 if dtype == DataType.BYTE else 2]
    else:
        one = []
        for m in members:
            width = 1 if struct_member_type(m) == DataType.BYTE else 2
            one.extend([width] * ((struct_member_dim(m) or 1) * len(struct_member_names(m))))
    names = item.names
    count = 1 if isinstance(names, P.DeclName) else len(names.names or [])
    widths = one * (len(values) + 1 if dim == -1 else max(dim or 1, 1)) * count
    slots: dict[int, str] = {}
    past: set[int] = set()
    slot = 0
    for v in values:
        inner = unwrap_paren(v)
        slots[id(v)] = ("past" if slot >= len(widths) else
                        "byte" if widths[slot] == 1 else "word")
        if isinstance(inner, P.StringLiteral):
            used = 0
            while used < max(len(string_value(inner)), 1):
                used += widths[slot] if slot < len(widths) else 1
                slot += 1
        else:
            slot += 1
        if slot > len(widths):
            past.add(id(v))
    return slots, past


def _over_a_byte(expr, stop) -> bool:
    """Whether Intel's PL/M-80 V3.1 finds the restricted expression
    ``expr``, where a byte goes, more than 255 (ERROR 210, ILLEGAL
    INITIALIZATION OF A BYTE TO A VALUE > 255): a location first, which is
    an address, `.a(1) - 1', or a number above 0FFH of what it computes
    before it stops at ``stop`` (:func:`_v31_read`): `300', `299 + 1',
    `300 + x', `300 + (1)'."""
    if isinstance(_lead(expr), P.LocationOf):
        return True
    value, _ = _v31_read(expr, stop)
    return value is not None and value[0] > 0xFF


def _v31_read(expr, stop) -> tuple[tuple[int, bool] | None, bool]:  # pylint: disable=too-many-return-statements
    """What Intel's PL/M-80 V3.1 computes of the restricted expression
    ``expr``, numbers added and subtracted, before it stops at the node
    ``stop`` (None: it reads all of it), and whether it stopped: (value,
    whether a BYTE) or None where it has computed nothing.  A number below
    256 is a BYTE, and BYTE arithmetic is in eight bits, `-1' 0FFH, `-1 -
    1' 0FEH; ADDRESS arithmetic in sixteen, `0ffffh + 1' 0.  A name it reads
    past (ERROR 151) is a BYTE 0 to it, `300 + x' 300 and `x - 1' 0FFH; at
    an operator it does not take it has the operand before it, `300 * 2'
    300."""
    if isinstance(expr, P.BinaryOp) and expr is stop:
        return _v31_read(expr.left, None)[0], True
    if expr is stop:
        return None, True
    if isinstance(expr, P.NumberLiteral):
        value = parse_plm_number(expr.value.text)
        return (value & 0xFFFF, value < 0x100), False
    if isinstance(expr, P.StringLiteral):
        text = string_value(expr)
        return ((ord(text) & 0xFF, True) if len(text) == 1 else None), False
    if isinstance(expr, P.Identifier):
        return (0, True), False
    if isinstance(expr, P.UnaryOp) and unop_kind(expr) == UnaryOpKind.NEG:
        inner, stopped = _v31_read(expr.operand, stop)
        if inner is None:
            return None, stopped
        value, byte = inner
        return (-value & (0xFF if byte else 0xFFFF), byte), stopped
    if isinstance(expr, P.BinaryOp) and binop_kind(expr) in (BinaryOpKind.ADD, BinaryOpKind.SUB):
        left, stopped = _v31_read(expr.left, stop)
        if stopped or left is None:
            return left, True
        right, stopped = _v31_read(expr.right, stop)
        if right is None:
            return left, stopped
        byte = left[1] and right[1]
        value = left[0] + right[0] if binop_kind(expr) == BinaryOpKind.ADD else left[0] - right[0]
        return (value & (0xFF if byte else 0xFFFF), byte), stopped
    return None, True


def _names_undeclared(expr, refs: dict) -> bool:
    """Whether ``expr`` names what is declared nowhere, a built-in's name
    aside, or a LITERALLY declared further on, whose name the macro pass
    has left in place (check_uses)."""
    if isinstance(expr, (list, tuple)):
        return any(_names_undeclared(x, refs) for x in expr)
    if isinstance(expr, P.Identifier):
        r = refs.get(id(expr))
        if r is None:
            return False
        return r.decl.kind == "lit" if r.decl is not None else _key(expr.name) not in _BUILTINS
    return any(_names_undeclared(getattr(expr, f), refs)
               for f in getattr(expr, "__dataclass_fields__", ()) if f != "pos")


def _a_builtin(ref, refs: dict) -> bool:
    """Whether the reference ``ref`` is a name that means a built-in but
    MEMORY."""
    r = refs.get(id(ref)) if isinstance(ref, P.Identifier) else None
    return r is not None and r.decl is None and _key(ref.name) in _BUILTINS - {"MEMORY"}


def _two_subscripts(expr) -> bool:
    """Whether ``expr`` has a reference with two subscripts in it,
    `a(1)(1)'."""
    if isinstance(expr, P.Call) and isinstance(unwrap_paren(expr.callee), P.Call):
        return True
    if isinstance(expr, (list, tuple)):
        return any(_two_subscripts(x) for x in expr)
    return any(_two_subscripts(getattr(expr, f)) for f in getattr(expr, "__dataclass_fields__", ())
               if f != "pos")


def _has_parentheses(expr) -> bool:
    """Whether ``expr`` has anything in parentheses: a call, a subscript,
    `f()', `( )', `.(list)'."""
    if isinstance(expr, (list, tuple)):
        return any(_has_parentheses(x) for x in expr)
    if isinstance(expr, (P.ParenExpr, P.Call, P.CallNoArgs, P.LocationOfList)):
        return True
    return any(_has_parentheses(getattr(expr, f)) for f in getattr(expr, "__dataclass_fields__", ())
               if f != "pos")


def _dimension(item) -> int | None:
    """A variable's dimension, -1 for (*), None for a scalar: an untyped
    DATA of several values, or a string of several characters, is an array
    as long as they are, as uplm80 has always taken it (and as the
    oracle's --normalize writes it for Intel's PL/M-80)."""
    _, dim = decl_item_type(item)
    tail = getattr(item, "tail", None)
    if dim is None and isinstance(tail, P.DeclTailData):
        values = decl_attrs(item).data_values or []
        if len(values) > 1 or (len(values) == 1 and isinstance(values[0], P.StringLiteral)
                               and len(string_value(values[0])) > 1):
            return -1
    return dim


def _address_members(item) -> frozenset:
    """The names of a structure's members of type ADDRESS."""
    nodes = decl_item_struct_members(item) or []
    return frozenset(n.upper() for m in nodes
                     if isinstance(getattr(m, "type", None), (P.TypeAddress, P.TypeAddressSized))
                     for n in struct_member_names(m))


def _member_type(item, member: str) -> DataType | None:
    """The type of the structure member ``member`` of ``item``."""
    for m in decl_item_struct_members(item) or []:
        if member in (n.upper() for n in struct_member_names(m)):
            return struct_member_type(m)
    return None


def _base_text(base) -> str:
    """A BASED declaration's base as the source spells it, `S.P'."""
    return ".".join(dotted_ident_parts(base))


def _members(item) -> dict | None:
    """A structure's members, and the dimension of each (None, a scalar)."""
    nodes = decl_item_struct_members(item)
    if nodes is None:
        return None
    return {n.upper(): struct_member_dim(m) for m in nodes for n in struct_member_names(m)}


def _reference_text(expr) -> str:
    """A member or a subscripted reference as the source spells it, but
    for a subscript that is neither a name nor a number: ``SA(...).M``."""
    expr = unwrap_paren(expr)
    if isinstance(expr, P.Identifier):
        return ident_text(expr.name)
    if isinstance(expr, P.NumberLiteral):
        return ident_text(expr.value)
    if isinstance(expr, P.MemberAccess):
        return f"{_reference_text(expr.base)}.{ident_text(expr.member)}"
    if isinstance(expr, P.Call):
        args = ", ".join(_reference_text(a) for a in expr.args)
        return f"{_reference_text(expr.callee)}({args})"
    return "..."


def _after_kind(ref, decl: _Decl | None, name: str) -> str:
    """What the reference ``ref`` is, which starts from ``name``, declared
    by ``decl`` or, None, a built-in."""
    if isinstance(ref, P.MemberAccess):
        return "a structure member"
    if decl is None:
        return "a subscripted variable" if name in ("MEMORY", "OUTPUT") else "a call"
    return "a call" if decl.kind == "proc" else "a subscripted variable"


def check_names(modules: list, multi: bool = False) -> list[tuple]:
    """Hold the names of ``modules``, as the parser gave them, to what
    PL/M-80 allows, before the optimizer rewrites or drops any of them.

    Raises CodeGenError for the first thing that is not allowed, as
    :func:`resolve_names` does for the rest: a name declared nowhere (a
    built-in's aside), an INTERRUPT procedure, or a PUBLIC or EXTERNAL
    one or variable, anywhere but at the outer level of its module, a
    dimension of 0, the address of a label anywhere but in a DATA or an
    INITIAL list, the address of a built-in but MEMORY, empty
    parentheses after a variable or a built-in, a base Intel's PL/M-80
    V3.1 does not take and a LABEL that labels no statement.  ``multi``: the modules
    are compiled together (see resolve_names).  Returns warnings,
    (location, text) pairs, for what Intel's PL/M-80 V3.1 rejects and
    uplm80 compiles, as programs written for it rely on: a procedure's
    empty parentheses, `f()', and INITIAL below module level.
    """
    r = _Resolver()
    r.checking = True
    for i, m in enumerate(modules):
        r.add_module(m, i)
    r.bind()
    if multi:
        r.check_private()
    r.check_restricted()
    r.check_uses()
    r.check_declarations()
    r.check_forms()
    return r.intel_warnings


def resolve_names(modules: list, multi: bool = False) -> list[tuple]:
    """Bind the names of ``modules``, rename what code generation would
    confuse, and check the GOTOs (see the module docstring).  ``multi``:
    the modules are separate modules compiled together into one assembly
    (a multi-file compile), whose private names are qualified per module.
    Raises CodeGenError for what PL/M-80 does not allow: a GOTO out of a
    procedure to a label of another procedure, into a block or to what is
    not a label; a name declared twice in one block; a PUBLIC label that
    labels no statement at the main program's outer level; and,
    in a multi-file compile, a second main program module or another
    module's private name.  Returns the warnings, (location, text) pairs:
    a GOTO out of a procedure to a label in a DO block of the main
    program, which PL/M-80 does not allow either, but Intel's PL/M-80
    compiles."""
    r = _Resolver()
    for i, m in enumerate(modules):
        r.add_module(m, i)
    r.bind()
    if multi:
        mains = [m for m in r.modules if m.main]
        if len(mains) > 1:
            raise CodeGenError(
                f"modules {mains[0].name} and {mains[1].name} both have statements at their "
                "outer level; only the main program module may", source_location(mains[1].first))
        r.check_private()
        r.qualify()
    r.check_public_labels()
    r.settle()
    r.check_gotos()
    r.annotate()
    r.write_back()
    return r.warnings
