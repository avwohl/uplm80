"""Which declaration each name in a program means, and what the assembler
calls it.

PL/M-80 gives every declared name - variable, procedure, label, macro - the
block that declares it for its scope, less any block nested in it that
declares the name again (Programming Manual 9800268B, chapter 9); every DO
block is a block, and so is every procedure.  Code generation names what it
emits the simple way: a module-level name as it is, a procedure's own
names and labels as ``@proc$name``, a DO block's variables as
``@proc$Bn$name``, a LITERALLY as an EQU of its name.  That is one name per
declaration only as long as no two declarations it names alike are in
scope at once, and PL/M-80 lets them be:

* a label in each of two DO blocks of a procedure (both ``@proc$L``), or
  of the main program (both ``L``);
* a procedure declared in each of two DO blocks of one procedure;
* a LITERALLY in each of two procedures (two EQUs of one name), or one
  named like a variable somewhere else;
* in a multi-file compile, a private name in each of two modules, which
  the modules' separate name spaces keep apart when they are compiled
  one at a time.

:func:`resolve_names` finds every declaration and binds every name to the
one PL/M-80 means, renames a declaration code generation would confuse
with another - ``L`` to ``L?2``; no PL/M-80 identifier contains a ``?``,
so a new name cannot meet one the program has - and qualifies, in a
multi-file compile, each module's private module-level names with the
module's own (``LIB?HELPER``).  A program in which no two declarations
meet is left exactly as it was.

It also checks every GOTO against the rules of 9.3, and tells code
generation the label each one jumps to.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field

from . import _plm_parser as P
from .ast_view import (
    DataType,
    decl_attrs,
    decl_item_type,
    ident_text,
    literally_value,
    parse_plm_number,
    proc_attrs,
)
from .errors import CodeGenError
from .frontend import source_location

# Names um80 reads as something other than a symbol.  A register, as an
# operand (`ld hl,A' is "Register 'A' used as value", `ld a,(IX)' is
# `ld a,(ix+0)'), and an operator of its expressions: `call EQ' calls 0FFFFH
# and `ld hl,SHL' loads 0, without a word.
REGISTER_NAMES = frozenset(
    {"A", "B", "C", "D", "E", "H", "L", "M", "SP", "PSW",
     "AF", "BC", "DE", "HL", "IX", "IY", "I", "R"})
OPERATOR_NAMES = frozenset({"EQ", "NE", "LT", "LE", "GT", "GE", "SHL", "SHR", "NUL"})
# And as the target of a jump, a condition: `jp P' wants an address after
# the P.  A procedure is jumped to as well as called - the peephole turns
# `call x / ret' into `jp x'.
CONDITION_NAMES = frozenset({"Z", "NZ", "NC", "PO", "PE", "P"})


def data_name(name: str) -> str:
    """The assembler's name for a variable or a LITERALLY called ``name``."""
    if name.upper() in REGISTER_NAMES or name.upper() in OPERATOR_NAMES:
        return f"@{name}"
    return name


def code_name(name: str) -> str:
    """The assembler's name for a procedure or a label called ``name``."""
    if name.upper() in CONDITION_NAMES:
        return f"@{name}"
    return data_name(name)


GOTO_RULE = ("PL/M-80 allows a GOTO out of a procedure only to a label at the "
             "outer level of the main program module (Programming Manual "
             "9800268B, 8.1.3 and 9.3)")
SCOPE_RULE = ("a GOTO reaches only a label in its own block or in a block "
              "enclosing it (Programming Manual 9800268B, 9.3)")

# What every module can name without declaring it (symbols.SymbolTable).
_BUILTINS = frozenset(
    {"INPUT", "OUTPUT", "LOW", "HIGH", "DOUBLE", "LENGTH", "LAST", "SIZE", "SHL", "SHR",
     "ROL", "ROR", "SCL", "SCR", "MOVE", "TIME", "CARRY", "SIGN", "ZERO", "PARITY", "DEC",
     "MEMORY", "STACKPTR", "CPUTIME"})

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
    orig: str = ""              # the name as declared

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
    for it in module.items:
        origin = getattr(getattr(it, "pos", None), "origin", None)
        if origin:
            stem = os.path.splitext(os.path.basename(origin[0]))[0].upper()
            stem = re.sub(r"[^A-Z0-9_]", "_", stem)
            return stem if stem[:1].isalpha() else f"M{stem}"
    return None


@dataclass(eq=False)
class _Block:
    kind: str                   # "module", "proc", "do", "global"
    parent: "_Block | None"
    module: _Module | None
    proc: _Decl | None          # the procedure it is in (its own, for a body)
    decls: dict = field(default_factory=dict)


@dataclass(eq=False)
class _Ref:
    block: _Block
    node: object
    attr: str
    goto: bool = False
    decl: _Decl | None = None


def _key(tok) -> str:
    return ident_text(tok).upper()


def _retext(tok, text: str):
    """``tok`` spelling ``text``."""
    if dataclasses.is_dataclass(tok):
        return dataclasses.replace(tok, text=text)
    return type(tok)(text, tok.name, tok.kind)


class _Resolver:
    """The declarations of one compilation and what each name is bound to."""

    def __init__(self) -> None:
        self.decls: list[_Decl] = []
        self.refs: list[_Ref] = []
        self.modules: list[_Module] = []
        self.globals = _Block("global", None, None, None)
        self.refs_of: dict[int, list[_Ref]] = {}
        self.used: set[str] = set()     # every name a declaration has

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
            # A second declaration of a name in one block is PL/M-80's error;
            # the first stands, and the second is renamed with it.  A
            # parameter's type declaration is one of these.
            old.sites.append(site)
            return old
        d = _Decl(name, kind, block, len(self.decls), [site], **kw)
        block.decls[name] = d
        self.decls.append(d)
        return d

    def _visit(self, n, block: _Block) -> None:  # pylint: disable=too-many-branches
        if n is None or isinstance(n, (str, int)):
            return
        if isinstance(n, (list, tuple)):
            for x in n:
                self._visit(x, block)
            return
        if isinstance(n, P.ProcDecl):
            self._proc(n, block)
        elif isinstance(n, P.DeclareStmt):
            self._visit(n.declarations, block)
        elif isinstance(n, P.DeclItem):
            self._decl_item(n, block)
        elif isinstance(n, P.DeclItemBasedGroup):
            for bd in n.based_decls or []:
                self._declare(block, _key(bd.name), "var", (bd, "name"), storage=False)
                self._visit(bd.base, block)
            self._visit([n.array_size, n.tail], block)
        elif isinstance(n, P.LiterallyDecl):
            self._declare(block, _key(n.name), "lit", (n, "name"), literal=literally_value(n))
        elif isinstance(n, P.LabeledStmt):
            self._label(n, block)
            self._visit(n.stmt, block)
        elif isinstance(n, P.GotoStmt):
            self.refs.append(_Ref(block, n, "label", goto=True))
        elif isinstance(n, _DO_BLOCKS):
            inner = _Block("do", block, block.module, block.proc)
            if isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
                self.refs.append(_Ref(inner, n, "index"))
            for f in ("condition", "start", "bound", "step", "selector", "items"):
                self._visit(getattr(n, f, None), inner)
        elif isinstance(n, (P.Identifier, P.DottedIdent)):
            self.refs.append(_Ref(block, n, "name"))
        elif isinstance(n, (P.MemberAccess, P.DottedMember)):
            self._visit(n.base, block)      # a member's name is not in scope
        elif isinstance(n, (P.StructMember, P.StructMemberUntyped, P.EndLabel)):
            pass
        else:
            fields = getattr(n, "__dataclass_fields__", None)
            if fields and not hasattr(n, "file_id"):
                for f in fields:
                    if f != "pos":
                        self._visit(getattr(n, f), block)

    def _proc(self, p: P.ProcDecl, block: _Block) -> None:
        attrs = proc_attrs(p)
        d = self._declare(block, _key(p.name), "proc", (p, "name"), public=attrs.is_public,
                          external=attrs.is_external, reentrant=attrs.is_reentrant)
        end = p.body.end_label
        if end is not None and _key(end.name) == d.name:
            d.sites.append((end, "name"))
        body = _Block("proc", block, block.module, d)
        params = p.signature.params
        for n in (params.names or []) if params is not None else []:
            self._declare(body, _key(n.name), "param", (n, "name"), storage=False)
        self._visit(p.body.items, body)

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
                old.sites.append((node, "name"))
                continue
            kind = "label" if dtype == DataType.LABEL else "var"
            self._declare(block, name, kind, (node, "name"), public=attrs.is_public,
                          external=attrs.is_external, storage=storage and kind == "var")
        if item.based is not None:
            self._visit(item.based.base, block)
        self._visit([item.array_size, item.tail], block)

    def _label(self, s: P.LabeledStmt, block: _Block) -> None:
        name = _key(s.label)
        old = block.decls.get(name)
        if old is not None and old.kind == "label" and old.defined:
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
            r.decl = self.lookup(_key(getattr(r.node, r.attr)), r.block)
            if r.decl is not None:
                self.refs_of.setdefault(id(r.decl), []).append(r)

    @staticmethod
    def lookup(name: str, block: _Block | None) -> _Decl | None:
        """The declaration ``name`` means in ``block``: the innermost."""
        while block is not None:
            d = block.decls.get(name)
            if d is not None:
                return d
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
        (@proc$Bn$name) and a parameter lives in ??AUTO."""
        parent = self.owner(d)
        if d.kind == "proc":
            if d.shared or parent is None:
                return code_name(d.name)
            return f"@{self.proc_key(d)}"
        if d.kind == "label":
            if d.shared or parent is None:
                return code_name(d.name)
            return f"@{self.proc_key(parent)}${d.name}"
        if d.kind == "lit":
            try:
                parse_plm_number(d.literal or "")
            except ValueError:
                return None     # no EQU
            return data_name(d.name)
        if d.kind == "var" and d.storage:
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

    # The kind of declaration renamed first when two meet in one assembler
    # name: a LITERALLY, whose EQU nothing uses (the macro pass has put its
    # text in place of every use), then a label, which nothing outside its
    # procedure names, then a procedure, and a variable last.
    _RENAME_FIRST = {"lit": 0, "label": 1, "proc": 2, "var": 3, "param": 3}

    def settle(self) -> None:
        """Rename declarations until no two code generation would give one
        assembler name to are left."""
        self.used = {d.name for d in self.decls}
        by_asm: dict[str, list[_Decl]] = {}
        for d in self.decls:
            name = self.asm_name(d)
            if name is not None:
                by_asm.setdefault(name.upper(), []).append(d)
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
                if d.kind in ("label", "lit"):
                    self.rename(d, self.fresh(d.name))

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
                place = (f"procedure {there.orig}" if there is not None
                         else "a DO block of the main program")
                raise CodeGenError(
                    f"GOTO {text} leaves procedure {here.orig} for a label in {place}; "
                    f"{GOTO_RULE}", where)
            r.node.uplm80_asm = self.asm_name(d)

    def _undeclared(self, text: str, r: _Ref) -> str:
        """Why a GOTO names no label it can reach."""
        name = _key(r.node.label)
        for d in self.decls:
            if d.kind == "label" and d.orig == name and d.block.module is r.block.module:
                place = ("a DO block" if d.block.kind == "do"
                         else f"procedure {d.block.proc.orig}" if d.block.proc is not None
                         else "the main program")
                return (f"GOTO {text}: the label {text} is in {place} this GOTO is not in; "
                        f"{SCOPE_RULE}")
        return f"GOTO {text}: no label {text} is declared here"

    def annotate(self) -> None:
        """Tell code generation what each label, and each other name for
        one, is called in the assembly."""
        for d in self.decls:
            if d.kind != "label":
                continue
            name = self.asm_name(d)
            for node, _ in d.sites:
                if isinstance(node, P.LabeledStmt):
                    node.uplm80_asm = name
            for r in self.refs_of.get(id(d), []):
                if not r.goto:
                    r.node.uplm80_asm = name

    def write_back(self) -> None:
        """Spell every renamed declaration's new name wherever it is named."""
        for d in self.decls:
            if d.name == d.orig:
                continue
            for node, attr in d.sites + [(r.node, r.attr) for r in self.refs_of.get(id(d), [])]:
                setattr(node, attr, _retext(getattr(node, attr), d.name))


def resolve_names(modules: list, multi: bool = False) -> None:
    """Bind the names of ``modules``, rename what code generation would
    confuse, and check the GOTOs (see the module docstring).  ``multi``:
    the modules are separate modules compiled together into one assembly
    (a multi-file compile), whose private names are qualified per module.
    Raises CodeGenError for a GOTO PL/M-80 does not allow."""
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
    r.settle()
    r.check_gotos()
    r.annotate()
    r.write_back()
