#!/usr/bin/env python3
"""Unpack an Intel ISIS ``.src`` pack and regenerate its per-module ``.ipx``
include files from the shared ``.pex`` master.

Intel's PL/M tool sources (e.g. ogdenpm/intel80tools ``src/link_3.0``) ship
as a single form-feed-delimited ``*_all.src`` pack, and each module does
``$include(<module>.ipx)`` for the externals it references. Those ``.ipx``
files are *generated* by the Intel build from one ``.pex`` master rather than
stored, so to compile a module standalone we have to reproduce them.

The ``.pex`` DSL:

* a header of ``NAME 'value'`` LITERALLY macros, type ``STRUCTURE`` literals,
  and ``NAME "TYPE"..base`` based-variable templates;
* ``$file(NAME)`` sections, each listing that module's PUBLIC symbols in
  shorthand: ``A``=ADDRESS, ``B``=BYTE, ``AS``/``BS``=address/byte array,
  ``P(A x,B y)``=procedure (``A``/``B`` params, optional ``A``/``B`` return),
  ``"TYPE"..base``=based var.

For each module we emit: every shared literal, every *other* module's publics
rendered as ``EXTERNAL`` declarations, and the based templates whose base
pointer is external to that module. This is a superset of the minimal include
the Intel cross-referencer would produce (we don't prune to referenced-only),
which is harmless for compilation.

CLI::

    python3 genipx.py <pack.src> <outdir>

unpacks every member of the pack into ``outdir`` and writes a ``<module>.ipx``
next to each ``<module>.plm``.
"""
from __future__ import annotations

import os
import re
import sys


def unpack(src_path: str, outdir: str) -> list[str]:
    """Split a form-feed-delimited ISIS ``.src`` pack into its members.

    Each member is ``\\f<filename><newline><body>``. Returns the member
    filenames written into ``outdir``.
    """
    data = open(src_path, encoding="latin-1").read()
    os.makedirs(outdir, exist_ok=True)
    names: list[str] = []
    for chunk in data.split("\f")[1:]:
        nl_candidates = [x for x in (chunk.find("\r"), chunk.find("\n")) if x >= 0]
        nl = min(nl_candidates) if nl_candidates else len(chunk)
        name = chunk[:nl].strip()
        body = chunk[nl:].lstrip("\r\n")
        with open(os.path.join(outdir, name), "w", encoding="latin-1", newline="") as fh:
            fh.write(body)
        names.append(name)
    return names


# --- .pex parsing -----------------------------------------------------------

def parse_pex(path: str):
    """Return ``(header, files)`` where ``header`` is a list of ``(name, rest)``
    shared entries and ``files`` maps a ``$file`` name to its ``(name, rest)``
    public symbols."""
    text = open(path, encoding="latin-1").read().replace("\r\n", "\n")
    header: list[tuple[str, str]] = []
    files: dict[str, list[tuple[str, str]]] = {}
    cur: str | None = None
    for raw in text.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("/*"):
            continue
        m = re.match(r"\$file\(([^)]+)\)", raw.strip())
        if m:
            cur = m.group(1).strip()
            files.setdefault(cur, [])
            continue
        mm = re.match(r"(\S+)\s+(.*)", raw)
        if not mm:
            continue
        name = mm.group(1)
        rest = re.sub(r"\s*/\*.*?\*/\s*$", "", mm.group(2)).strip()
        (header if cur is None else files[cur]).append((name, rest))
    return header, files


def _is_literal(rest: str) -> bool:
    return rest.strip().startswith("'")


_SCALAR = {"A": "address", "B": "byte", "AS": "address", "BS": "byte"}


def _ptype(letter: str) -> str:
    return "byte" if letter == "B" else "address"


def _render_proc(name: str, spec: str) -> str:
    ret = ""
    m = re.match(r"P\((.*)\)([AB]?)$", spec)
    params = ""
    if m:
        params, retl = m.group(1), m.group(2)
        ret = f" {_ptype(retl)}" if retl else ""
    else:
        m2 = re.match(r"P([AB])$", spec)
        if m2:
            ret = f" {_ptype(m2.group(1))}"
    pieces = [p.strip() for p in params.split(",") if p.strip()]
    if pieces:
        names = [p.split()[1] for p in pieces]
        decls = [f"{p.split()[1]} {_ptype(p.split()[0])}" for p in pieces]
        return (f"{name}: procedure({', '.join(names)}){ret} external; "
                f"declare {', '.join(decls)}; end;")
    return f"{name}: procedure{ret} external; end;"


def _render_based(name: str, ts: str, base: str) -> str:
    ts = ts.strip()
    if ts.startswith('"'):
        m = re.match(r'"([^"]+)"(S?)$', ts)
        dim = "(1) " if m.group(2) == "S" else ""
        return f"declare {name} based {base} {dim}{m.group(1)};"
    dim = "(1) " if ts.endswith("S") else ""
    return f"declare {name} based {base} {dim}{_SCALAR.get(ts, 'byte')};"


def _render_typed(name: str, spec: str) -> str:
    spec = spec.strip()
    if ".." in spec:
        ts, base = spec.split("..", 1)
        return _render_based(name, ts.strip(), base.strip())
    if spec == "P" or spec.startswith("P(") or re.match(r"P[AB]$", spec):
        return _render_proc(name, spec)
    if spec in _SCALAR:
        dim = "(1)" if spec.endswith("S") else ""
        return f"declare {name}{dim} {_SCALAR[spec]} external;"
    return f"/* UNPARSED: {name} {spec} */"


def _owners(files) -> dict[str, str]:
    owner: dict[str, str] = {}
    for fname, syms in files.items():
        for name, rest in syms:
            if not _is_literal(rest) and ".." not in rest:
                owner.setdefault(name, fname)
    return owner


def gen_ipx(module: str, header, files, owner) -> str:
    """Render the ``.ipx`` text for ``module`` (a ``$file`` name like
    ``link1a.plm``)."""
    out = ["/* generated from link.pex by scripts/genipx.py - do not edit */"]
    seen: set[str] = set()

    def emit(name: str, line: str) -> None:
        if name not in seen:
            seen.add(name)
            out.append(line)

    for name, rest in header:                      # shared literals + structs
        if _is_literal(rest):
            emit(name, f"declare {name} literally {rest};")
    for fname, syms in files.items():              # other modules' publics
        if fname == module:
            continue
        for name, rest in syms:
            emit(name, f"declare {name} literally {rest};" if _is_literal(rest)
                 else _render_typed(name, rest))
    for name, rest in header:                       # based templates (external base)
        if _is_literal(rest) or ".." not in rest:
            continue
        ts, base = rest.split("..", 1)
        base = base.strip()
        if owner.get(base) not in (None, module):
            emit(name, _render_based(name, ts.strip(), base))
    return "\n".join(out) + "\n"


def generate_all(pex_path: str, outdir: str) -> list[str]:
    header, files = parse_pex(pex_path)
    owner = _owners(files)
    made = []
    for fname in files:
        if not fname.endswith(".plm"):
            continue
        ipx = gen_ipx(fname, header, files, owner)
        path = os.path.join(outdir, fname[:-4] + ".ipx")
        with open(path, "w", encoding="latin-1", newline="\n") as fh:
            fh.write(ipx)
        made.append(os.path.basename(path))
        for ln in ipx.splitlines():
            if "UNPARSED" in ln:
                print(f"  WARN {fname}: {ln}", file=sys.stderr)
    return made


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: genipx.py <pack.src> <outdir>", file=sys.stderr)
        return 2
    pack, outdir = argv
    members = unpack(pack, outdir)
    pex = next((m for m in members if m.endswith(".pex")), None)
    if pex is None:
        print("no .pex member in pack; cannot generate includes", file=sys.stderr)
        return 1
    made = generate_all(os.path.join(outdir, pex), outdir)
    print(f"unpacked {len(members)} members, generated {len(made)} .ipx files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
