"""``-g``: the variables of a program (phase 8 of the debugger plan).

``DebugVarsMixin`` is a part of the code generator.  It notes each variable
while the generator hands out the storage: the name, the PL/M type, the
kind, the routine, the block that scopes the name, and the line of the
declaration.  Nothing in the mixin changes the text of the code.

``add_vars`` writes the ``;#type`` and ``;#var`` markers
(docs/debug_info.md of um80_and_friends) after the last text pass, because
only the final text says

- whether a slot in ``??AUTO`` is still in use (the optimizer removes the
  store to a variable that nothing reads), and
- where a parameter of a static procedure is valid: after the stores of
  the entry code (``range=``, written as labels after the last pass, so no
  pass sees a label).

A label has no bytes and no pass runs after ``add_vars``, so the code is
the same with and without ``-g`` (D2).  ``docs/debug_info.md`` of uplm80
describes the records.
"""

from __future__ import annotations

import os
import re
import sys

from .ast_nodes import DataType
from .ast_view import decl_item_struct_members

_BYTE = "typedef:BYTE"
_ADDRESS = "typedef:ADDRESS"
_BASE_TYPES = {
    _BYTE: {"target": "ushort:1"},
    _ADDRESS: {"target": "uint:2"},
}

_SYMBOL = re.compile(r"^([A-Za-z_@?$.][\w@?$.]*)((?:[+-]\d+)*)$")
_TAG_CHARS = re.compile(r"[^A-Za-z0-9_.$~]")


class DebugVarsMixin:
    """Collects the variable records of ``-g`` (see the module text)."""

    # -- state ---------------------------------------------------------

    def _dbgv_reset(self) -> None:
        self._dbgv_records: list[dict] = []
        self._dbgv_types: dict[str, dict] = {}
        self._dbgv_tags: dict[str, tuple] = {}
        self._dbgv_funcs: list[str] = []         # labels of the routines open
        self._dbgv_main: str | None = None       # the main program's routine
        self._dbgv_scopes: list = []             # nodes: the blocks open
        self._dbgv_warned = False
        self._dbgv_text: _Sources | None = None

    # -- routines and blocks --------------------------------------------

    def dbgv_enter_proc(self, label: str, decl) -> None:
        if self.debug_info:
            self._dbgv_funcs.append(label)
            self._dbgv_scopes.append(decl)

    def dbgv_leave_proc(self) -> None:
        if self.debug_info and self._dbgv_funcs:
            self._dbgv_funcs.pop()
            self._dbgv_scopes.pop()

    def dbgv_enter_main(self, name: str) -> None:
        self._dbgv_main = name

    def dbgv_leave_main(self, name: str) -> None:
        if self._dbgv_main == name:
            self._dbgv_main = None

    def dbgv_enter_block(self, node) -> None:
        if self.debug_info:
            self._dbgv_scopes.append(node)

    def dbgv_leave_block(self) -> None:
        if self.debug_info and self._dbgv_scopes:
            self._dbgv_scopes.pop()

    def _dbgv_owner(self) -> str | None:
        if self._dbgv_funcs:
            return self._dbgv_funcs[-1]
        return self._dbgv_main

    def _dbgv_scope(self):
        """(file, first line, last line) of the innermost block, or None."""
        if not self._dbgv_scopes:
            return None
        node = self._dbgv_scopes[-1]
        a = self._dbg_where(node)
        b = self._dbg_where(node, end=True)
        if a is None or b is None or a[0] != b[0]:
            return None
        return (a[0], a[1], max(a[1], b[1]))

    # -- types -----------------------------------------------------------

    def _dbgv_scalar(self, data_type) -> str | None:
        if data_type == DataType.BYTE:
            self._dbgv_types.setdefault(_BYTE, _BASE_TYPES[_BYTE])
            return _BYTE
        if data_type == DataType.ADDRESS:
            self._dbgv_types.setdefault(_ADDRESS, _BASE_TYPES[_ADDRESS])
            return _ADDRESS
        return None

    @staticmethod
    def _dbgv_member_layout(struct_members):
        """[(name, data type, dimension, offset, size)] of a structure."""
        out = []
        offset = 0
        for m in struct_members:
            size = (m.dimension or 1) * (1 if m.data_type == DataType.BYTE else 2)
            out.append((m.name, m.data_type, m.dimension, offset, size))
            offset += size
        return out, offset

    def _dbgv_array(self, element: str, dimension) -> str:
        if dimension is not None and dimension > 0:
            return f"[{dimension}]{element}"
        return element

    def _dbgv_struct(self, name: str, struct_members, shown=None) -> str:
        """``struct:TAG`` for a structure; the definition goes in the table.
        PL/M structures have no tag.  The tag is the variable's name, and
        ``NAME.2`` for another structure of the same name."""
        layout, size = self._dbgv_member_layout(struct_members)
        members = []
        for mname, mtype, mdim, off, _ in layout:
            scalar = self._dbgv_scalar(mtype) or f"raw:{1 if mtype == DataType.BYTE else 2}"
            members.append(((shown or {}).get(mname, mname),
                            self._dbgv_array(scalar, mdim), off))
        key = (size, tuple(members))
        base = _TAG_CHARS.sub("_", name) or "S"
        tag, k = base, 1
        while tag in self._dbgv_tags and self._dbgv_tags[tag] != key:
            k += 1
            tag = f"{base}.{k}"
        self._dbgv_tags[tag] = key
        self._dbgv_types[f"struct:{tag}"] = {
            "kind": "struct", "size": size, "members": members}
        return f"struct:{tag}"

    def _dbgv_type(self, name, data_type, dimension, struct_members, size,
                   shown=None) -> str:
        if struct_members:
            element = self._dbgv_struct(name, struct_members, shown)
        else:
            element = self._dbgv_scalar(data_type) or f"raw:{size}"
        return self._dbgv_array(element, dimension)

    # -- one variable ----------------------------------------------------

    @staticmethod
    def _dbgv_plain(name: str) -> str:
        """``name`` as the programmer wrote it: a multi-file compile
        qualifies a module-level name (``LIB?NAME``) and numbers a name
        that another declaration has (``NAME?2``)."""
        if "?" not in name:
            return name
        parts = [p for p in name.split("?")]
        while len(parts) > 1 and parts[-1].isdigit():
            parts.pop()
        return parts[-1] if len(parts) > 1 else parts[0]

    def _dbgv_token_where(self, token, pos):
        """(file, line) of a name token.  ``pos`` is the position of the
        whole declaration: a name on a continuation line is that many lines
        after the first line of the declaration."""
        where = self._dbg_where_pos(pos)
        first = getattr(pos, "start_line", 0) or 0
        line = getattr(token, "line", 0) or 0
        if where is not None and first and line > first:
            where = (where[0], where[1] + line - first)
        return where

    def _dbgv_spell(self, token, pos, canonical: str) -> str:
        """The name as the source spells it (without ``$``), if the source
        line has it at the column of the token, else the upper-case name
        that the compiler uses."""
        canonical = self._dbgv_plain(canonical)
        where = self._dbgv_token_where(token, pos)
        if where is None or token is None:
            return canonical
        if self._dbgv_text is None:
            self._dbgv_text = _Sources(self._dbg_files, getattr(self, "dbg_sources", {}))
        return self._dbgv_text.spell(
            (where[0], where[1], getattr(token, "column", 0),
             getattr(token, "line", 0)), canonical)

    @staticmethod
    def _dbgv_name_token(item, name):
        """The token that spells ``name`` in the DeclItem ``item``."""
        names = getattr(item, "names", None)
        nodes = []
        if names is not None:
            nodes = getattr(names, "names", None) or [names]
        nodes = nodes or list(getattr(item, "based_decls", None) or [])
        for node in nodes:
            tok = getattr(node, "name", None)
            if tok is not None and getattr(tok, "text", "").replace("$", "") == name:
                return tok
        return None

    def _dbgv_member_names(self, item) -> dict:
        """{upper-case member name: name as the source spells it}."""
        out = {}
        for m in (decl_item_struct_members(item) or []) if hasattr(item, "tail") else []:
            names = getattr(m, "names", None)
            for node in (getattr(names, "names", None) or [names]):
                tok = getattr(node, "name", None)
                if tok is None:
                    continue
                canon = tok.text.replace("$", "")
                out[canon] = self._dbgv_spell(tok, getattr(m, "pos", None), canon)
        return out

    @staticmethod
    def _dbgv_param_node(decl, index):
        params = decl.signature.params
        names = list(getattr(params, "names", None) or [])
        return names[index] if index < len(names) else None

    def dbgv_variable(self, name, sym, item, *, token, data_type, dimension,
                      struct_members, based_on, based_member, attrs) -> None:
        """A variable of a declaration (``_gen_one_var``)."""
        if not self.debug_info or attrs.is_external:
            return
        try:
            owner = self._dbgv_owner()
            base = None
            if based_on:
                base = self._lookup_scoped(based_on)
            kind = "local" if owner else ("global" if attrs.is_public else "static")
            shown = self._dbgv_spell(token, getattr(item, "pos", None), name)
            text = self._dbgv_type(shown, data_type, dimension, struct_members,
                                   sym.size, self._dbgv_member_names(item))
            self._dbgv_records.append({
                "name": name, "sym": sym, "type": text, "kind": kind,
                "owner": owner, "scope": self._dbgv_scope() if owner else None,
                "decl": self._dbgv_token_where(token, getattr(item, "pos", None)),
                "shown": shown,
                "based": bool(based_on), "base": base, "member": based_member,
                # a local of a procedure holds no value until its first
                # store (the ??AUTO slot holds another routine's bytes)
                "range": bool(self._dbgv_funcs) and not (
                    attrs.is_public or based_on or attrs.at_location
                    or attrs.data_values or attrs.initial_values)
                    and sym.stack_offset is None,
            })
        except Exception as e:           # noqa: BLE001
            self._dbgv_fault(e)

    def dbgv_param(self, name, sym, decl, ident, data_type, reentrant) -> None:
        """A parameter (``_gen_proc_decl``); ``ident`` is its name in the
        parameter list."""
        if not self.debug_info:
            return
        try:
            where = self._dbg_where(decl)
            token = getattr(ident, "name", None)
            pos = getattr(ident, "pos", None)
            self._dbgv_records.append({
                "name": name, "sym": sym,
                "type": self._dbgv_scalar(data_type) or "raw:2",
                "kind": "param", "owner": self._dbgv_owner(),
                "scope": self._dbgv_scope(), "decl": where,
                "shown": self._dbgv_spell(token, pos or decl.pos, name),
                "based": False, "base": None, "member": None,
                "range": not reentrant,
            })
        except Exception as e:           # noqa: BLE001
            self._dbgv_fault(e)

    def _dbgv_fault(self, e: Exception) -> None:
        """A fault in the variable records must not stop a build."""
        if os.environ.get("UPLM80_DEBUG_VARS_STRICT"):
            raise e
        if not self._dbgv_warned:
            self._dbgv_warned = True
            print(f"uplm80: warning: variable records for -g are incomplete "
                  f"({type(e).__name__}: {e})", file=sys.stderr)

    # -- the result ------------------------------------------------------

    @staticmethod
    def _dbgv_place(sym):
        """("at", label, offset), ("ix", n), or None for a symbol."""
        if sym is None:
            return None
        if sym.stack_offset is not None:
            return ("ix", sym.stack_offset)
        m = _SYMBOL.match(sym.asm_name or "")
        if m is None:
            return None
        offset = sum(int(x) for x in re.findall(r"[+-]\d+", m.group(2)))
        return ("at", m.group(1), offset)

    def _dbgv_member_offset(self, base, member) -> int | None:
        if not member:
            return 0
        if not base.struct_members:
            return None
        layout, _ = self._dbgv_member_layout(base.struct_members)
        for mname, _, _, off, _ in layout:
            if mname == member:
                return off
        return None

    def dbg_var_info(self) -> dict:
        """The records for ``add_vars``: ``funcs`` (routine label to the
        records of its variables), ``globals``, ``types``, and the file
        table.  Each record has the location the generator gave."""
        funcs: dict[str, list] = {}
        globals_: list = []
        for rec in self._dbgv_records:
            sym = rec["sym"]
            why = None
            if rec["based"]:
                # A BASED variable is at the address its base holds.  The
                # format has no such place, so the record says: a pointer
                # to the variable, at the place of the base.
                base = rec["base"]
                place = self._dbgv_place(base)
                off = self._dbgv_member_offset(base, rec["member"]) if base else None
                if place is None or off is None:
                    place, why = None, "based-on-unknown"
                else:
                    place = (place[0], place[1], place[2] + off) if place[0] == "at" \
                        else ("ix", place[1] + off)
                    rec = dict(rec, type="*" + rec["type"])
            else:
                place = self._dbgv_place(sym)
                if place is None:
                    why = "no-place"
            out = dict(rec, place=place, why=why)
            if rec["owner"]:
                funcs.setdefault(rec["owner"], []).append(out)
            else:
                globals_.append(out)
        return {"funcs": funcs, "globals": globals_, "types": self._dbgv_types,
                }


# -- the markers, after the last text pass -----------------------------------

_FILE = re.compile(r'^;#file\s+(\d+)\s+"((?:[^"\\]|\\.)*)"')
_FUNC = re.compile(r"^;#func\s+(\S+)")
_ENDFUNC = re.compile(r"^;#endfunc\b")
_FRAME = re.compile(r"\bframe=(\w+)")
_AUTO = re.compile(r"\?\?AUTO((?:\+\d+)*)", re.IGNORECASE)
_SYMBOL_AT = re.compile(r"^([A-Za-z_@?$.][\w@?$.]*)((?:[+-]\d+)*)$")


def _type_line(tag, d) -> str:
    out = f";#type {tag} size={d['size']}" if d.get("kind") else f";#type {tag}"
    if d.get("kind") == "struct":
        out += " members=" + ",".join(f"{n}:{t}@{o}" for n, t, o in d["members"])
    else:
        out += f" target={d['target']}"
    return out


_STORE = re.compile(r"^\s*ld\s+\(([^()]+)\)\s*,", re.IGNORECASE)


def _first_store(lines, first: int, last: int, label: str, offset: int):
    """The index of the first instruction in ``lines[first:last]`` that
    stores to the slot ``label+offset``, or None."""
    for k in range(first, last):
        m = _STORE.match(lines[k])
        if m is None:
            continue
        target = _SYMBOL_AT.match(m.group(1).replace(" ", "").upper())
        if target is None or target.group(1) != label.upper():
            continue
        if sum(int(x) for x in re.findall(r"[+-]\d+", target.group(2))) == offset:
            return k
    return None


def _slot_used(text_lines, base: int, size: int) -> bool:
    """Whether some instruction names a byte of the slot ``??AUTO+base``."""
    for ln in text_lines:
        code = ln.split(";", 1)[0]
        for m in _AUTO.finditer(code):
            k = sum(int(x) for x in re.findall(r"\d+", m.group(1)))
            if base <= k < base + max(size, 1):
                return True
    return False


class _Sources:
    """The lines of the source files, to show a name as the source spells it."""

    def __init__(self, files: dict, sources: dict) -> None:
        self.files = files
        self.sources = sources
        self.cache: dict[int, list] = {}

    def lines(self, number: int):
        if number not in self.cache:
            path = next((p for p, n in self.files.items() if n == number), None)
            text = self.sources.get(path)
            if text is None and path:
                try:
                    with open(path, encoding="latin-1") as f:
                        text = f.read()
                except OSError:
                    text = None
            self.cache[number] = text.replace("\r", "").split("\n") if text else []
        return self.cache[number]

    def spell(self, hint, canonical: str) -> str:
        """The name as written, without ``$``, if the source line has it
        near the column of the token, else the upper-case name.  The
        column is the one of the text the parser read, where the compiler
        has written a keyword in full (``DCL`` as ``DECLARE``), so the
        name can lie a few columns away."""
        if not hint:
            return canonical
        number, line, col, tline = hint
        rows = self.lines(number)
        best = None
        for ln in (line, tline):
            if not 0 < ln <= len(rows):
                continue
            for m in re.finditer(r"[A-Za-z][A-Za-z0-9$]*", rows[ln - 1]):
                written = m.group(0).replace("$", "")
                if written.upper() != canonical:
                    continue
                distance = abs(m.start() + 1 - col)
                if best is None or distance < best[0]:
                    best = (distance, written)
            if best is not None:
                break
        return best[1] if best else canonical


def add_vars(asm: str, info: dict) -> str:
    """``asm`` with the ``;#type`` and ``;#var`` markers of ``info``
    (``CodeGenerator.dbg_var_info``)."""
    if not info["funcs"] and not info["globals"]:
        return asm
    lines = asm.split("\n")
    last_file = max((i for i, ln in enumerate(lines) if _FILE.match(ln)), default=-1)
    if last_file < 0:
        return asm

    # the routines: label -> (index of ;#func, index of ;#endfunc)
    spans: dict[str, tuple[int, int]] = {}
    open_at = None
    for i, ln in enumerate(lines):
        m = _FUNC.match(ln)
        if m:
            open_at = (m.group(1), i)
        elif _ENDFUNC.match(ln) and open_at:
            spans.setdefault(open_at[0], (open_at[1], i))
            open_at = None

    counter = [0]
    labels: dict[int, list[str]] = {}

    def label_before(index: int) -> str:
        counter[0] += 1
        name = f"??VR{counter[0]}"
        labels.setdefault(index, []).append(f"{name}:")
        return name

    def line_for(rec, words, why=None) -> str:
        out = [f";#var {rec['shown']} {rec['type']}",
               f"kind={rec['kind']}"]
        out += words
        if why:
            out.append(f"why={why}")
        if rec["scope"]:
            out.append("scope=%d:%d-%d" % rec["scope"])
        if rec["decl"]:
            out.append("decl=%d:%d" % rec["decl"])
        return " ".join(out)

    # what each routine's own variables are: (record, words, why, store)
    decided: dict[str, list] = {}
    for owner, recs in info["funcs"].items():
        if owner not in spans:
            continue            # the routine has no marker: its records go
        first, last = spans[owner]
        frame = (_FRAME.search(lines[first]) or [None, "none"])[1]
        has_range = " ixbegin=" in lines[first]
        # the text that can name a slot of the routine's: its own and the
        # text of the routines nested in it
        text = lines[first:last]
        nested = "@" + owner.lstrip("@") + "$"
        for name, (a, b) in spans.items():
            if name.startswith(nested):
                text = text + lines[a:b]
        rows = []
        for rec in recs:
            place, why = rec["place"], rec["why"]
            if place is None:
                rows.append((rec, ["none"], why, None))
            elif place[0] == "ix":
                if frame != "ix" or not has_range:
                    rows.append((rec, ["none"], "no-frame", None))
                else:
                    rows.append((rec, [f"ix={place[1]}"], None, None))
            elif (place[1].upper() == "??AUTO" and not _slot_used(
                    text, place[2], 2 if rec["based"] else rec["sym"].size)):
                rows.append((rec, ["none"], "no-storage", None))
            else:
                label, offset = place[1], place[2]
                # a parameter is valid from the first store to its slot: the
                # entry code stores the argument, and the optimizer can move
                # the store a few instructions on, into the first statement
                store = (_first_store(lines, first, last, label, offset)
                         if rec["range"] else None)
                at = f"at={label}" + (f"+{offset}" if offset > 0 else
                                      f"-{-offset}" if offset < 0 else "")
                rows.append((rec, [at], None, store))
        decided[owner] = rows

    def ancestors(label: str) -> list[str]:
        """The routines this routine is nested in, the nearest first."""
        parts = label.lstrip("@").split("$")
        found = []
        for i in range(len(parts) - 1, 0, -1):
            prefix = "$".join(parts[:i])
            found.extend(c for c in (prefix, "@" + prefix) if c in decided)
        return found

    after_func: dict[int, list[str]] = {}
    for owner, (first, last) in spans.items():
        out = []
        span = None
        shown = set()
        for rec, words, why, store in decided.get(owner, ()):
            if store is not None:
                start = label_before(store + 1)
                if span is None:
                    span = label_before(last)
                words = words + [f"range={start}:{span}"]
            out.append(line_for(rec, words, why))
            shown.add(rec["shown"])
        # A nested procedure sees the variables of the procedures around
        # it.  They are static, and valid while the nested one runs, which
        # is only while the outer one is active.  A variable in the IX
        # frame of the outer one is not reached from here.
        for outer in ancestors(owner):
            for rec, words, why, _ in decided[outer]:
                if rec["shown"] in shown or words[0] == "none" or words[0].startswith("ix="):
                    continue
                shown.add(rec["shown"])
                out.append(line_for(dict(rec, kind="local"), words, why))
        if out:
            after_func[first] = out

    header = [_type_line(t, d) for t, d in info["types"].items()]
    for rec in info["globals"]:
        place = rec["place"]
        if place is None or place[0] != "at":
            header.append(line_for(rec, ["none"], rec["why"] or "no-place"))
        else:
            label, offset = place[1], place[2]
            header.append(line_for(rec, [f"at={label}" + (
                f"+{offset}" if offset > 0 else f"-{-offset}" if offset < 0 else "")]))

    out_lines = []
    for idx, ln in enumerate(lines):
        out_lines.extend(labels.get(idx, ()))
        out_lines.append(ln)
        if idx in after_func:
            out_lines.extend(after_func[idx])
        if idx == last_file:
            out_lines.extend(header)
    return "\n".join(out_lines)
