#!/usr/bin/env python3
"""Differential test of uplm80 against Intel's own PL/M-80 V3.1.

    python3.14 scripts/intel_oracle.py PROGRAM.plm ...       # given programs
    python3.14 scripts/intel_oracle.py --random 300 [--first S]  # generated
    python3.14 scripts/intel_oracle.py --corpus                # tests/, sample_code/

Each program - a CP/M main program that prints through MON1/MON2 - is built
twice.  Intel's way, as DRI's P.SUB (mpm2src/PLM_WORK) built its programs:

    PLM80 T.PLM DEBUG PAGEWIDTH(80)
    LINK T.OBJ,X0100,PLM80.LIB TO T.MOD
    LOCATE T.MOD CODE(0100H) STACKSIZE(n)
    OBJCPM T                                  (a CP/M program: run on cpmemu)

on tools/isis/isis, a small ISIS-II emulator (`make -C tools/isis'), or when
that is not built, on romwbw_emu's tools/romwbw-plm80, which runs the same
recipe under DRI's ISX; and uplm80's way at every -O level (default 0 to 3).
Every build is run under cpmemu, and what each prints is compared with what
Intel's prints.  The verdict for each program is one of

    same            every level prints what Intel's build prints
    differs         some level prints something else (both outputs shown)
    intel-rejects   PL/M-80, LINK or LOCATE reports an error
    uplm80-rejects  uplm80, um80 or ul80 fails at some level
    timeout         a build or a run takes longer than --timeout

Two things the recipe leaves to the program, which DRI's programs did for
themselves.  LOCATE puts the constants first and starts the program behind
them, not at 0100H where CP/M enters it (DRI's programs begin with a DATA
jump): such a program is located again at 0103H, where OBJCPM puts a jump
to the start in front.  And a PL/M-80 main program ends in EI; HLT, which
cpmemu executes and runs past: the HLT at the module's last statement
(found through the LINES records DEBUG writes) becomes RST 0, a warm boot,
which is how uplm80's CP/M mode ends a program.

$INCLUDE files are read in and an old-style origin line such as `0100H:' is
dropped; both compilers get the same text.  --normalize also rewrites three
spellings only uplm80 takes (see normalize_text).  uplm80's build links an
X0100 of its own, the same equates as DRI's X0100.ASM.

--random N checks programs from tests/plm_intel.py's generator (--avoid
leaves out the features its docstring lists; the README's table says which
difference each stands for).  --reduce cuts every program that differs, or
that uplm80 rejects, down to what the difference needs: a generated one by
its statements first, then any program by blocks and lines (ddmin), keeping
the first pair of lines that differ; tidy the result by hand.

Intel's binaries are not part of this repository.  They are found in
--tools DIR, $PLM80_TOOLS, or DRI's work disk at
~/src/mpm2/mpm2_external/mpm2src/PLM_WORK (PLM80 and PLM80.OV0-4, PLM80.LIB,
LINK, LINK.OVL, LOCATE, X0100, OBJCPM.COM).  When they, or both ways to run
them, are missing the oracle says so and exits 0 without checking anything
(--strict: exit 2).  uplm80 is this checkout's, run as
`python -P -m uplm80.compiler' with the checkout first on PYTHONPATH;
um80, ul80 and cpmemu ($CPMEMU, PATH, ~/src/cpmemu/src/cpmemu) must be
installed.

Exit status: 0 every program checked is `same', 1 otherwise, 2 the tools
are missing and --strict was given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_TOOLS = "~/src/mpm2/mpm2_external/mpm2src/PLM_WORK"
DEFAULT_ROMWBW_PLM80 = "~/src/romwbw_emu/tools/romwbw-plm80"
INTEL_FILES = ["PLM80", "PLM80.OV0", "PLM80.OV1", "PLM80.OV2", "PLM80.OV3",
               "PLM80.OV4", "PLM80.LIB", "LINK", "LINK.OVL", "LOCATE", "X0100",
               "OBJCPM.COM"]
ISIS_DIR = os.path.join(REPO, "tools", "isis")
LEVELS = (0, 1, 2, 3)
# P.SUB says STACKSIZE(100).  That is plenty for DRI's programs, but not for
# the REENTRANT recursion the generator writes; the stack's size only moves
# the DATA segment, which a program that reads no uninitialised memory does
# not notice.
DEFAULT_STACK = 1024


# ---- finding the tools ------------------------------------------------------

def _executable(path: str | None) -> str | None:
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


@dataclass
class Tools:
    """Where everything the oracle runs is, and what is missing."""
    intel_dir: str | None = None
    intel_files: dict[str, str] = field(default_factory=dict)
    isis: str | None = None
    romwbw_plm80: str | None = None
    cpmemu: str | None = None
    um80: str | None = None
    ul80: str | None = None
    route: str = "auto"     # native, romwbw or auto

    @classmethod
    def find(cls, tools_dir: str | None = None, isis: str | None = None,
             route: str = "auto", build_isis: bool = True) -> "Tools":
        t = cls(route=route)
        for cand in (tools_dir, os.environ.get("PLM80_TOOLS"), DEFAULT_TOOLS):
            if not cand:
                continue
            d = os.path.expanduser(cand)
            if os.path.isdir(d):
                files = {fn.upper(): os.path.join(d, fn) for fn in os.listdir(d)}
                if all(n in files for n in INTEL_FILES):
                    t.intel_dir = d
                    t.intel_files = {n: files[n] for n in INTEL_FILES}
                    break
        t.isis = _executable(isis) or _executable(os.environ.get("ISIS_EMU"))
        if t.isis is None:
            own = os.path.join(ISIS_DIR, "isis")
            # Built only where there is something to run on it: without
            # Intel's binaries the oracle checks nothing, and the suite
            # skips at once.
            if not _executable(own) and build_isis and route != "romwbw" and t.intel_dir:
                _try_build_isis()
            t.isis = _executable(own)
        for cand in (os.environ.get("ROMWBW_PLM80"), os.path.expanduser(DEFAULT_ROMWBW_PLM80)):
            if cand and os.path.isfile(cand):
                t.romwbw_plm80 = cand
                break
        for cand in (os.environ.get("CPMEMU"), shutil.which("cpmemu"),
                     os.path.expanduser("~/src/cpmemu/src/cpmemu")):
            if _executable(cand):
                t.cpmemu = cand
                break
        t.um80 = shutil.which("um80")
        t.ul80 = shutil.which("ul80")
        return t

    def intel_route(self) -> str | None:
        """How Intel's tools will be run: 'native', 'romwbw' or None."""
        if self.route in ("auto", "native") and self.isis:
            return "native"
        if self.route in ("auto", "romwbw") and self.romwbw_plm80:
            return "romwbw"
        return None

    def missing(self) -> str | None:
        """Why the oracle cannot run here, or None."""
        if not self.intel_dir:
            return ("Intel's PL/M-80 V3.1 not found (set PLM80_TOOLS to a directory "
                    "holding " + ", ".join(INTEL_FILES) + ")")
        if self.intel_route() is None:
            return ("no way to run Intel's tools: build tools/isis (make -C tools/isis) "
                    "or set ROMWBW_PLM80 to romwbw_emu's tools/romwbw-plm80")
        for name in ("cpmemu", "um80", "ul80"):
            if getattr(self, name) is None:
                return f"{name} not installed"
        return None


def _try_build_isis() -> None:
    """Build tools/isis/isis if make and a compiler are there; quietly."""
    if not shutil.which("make") or not os.path.isfile(os.path.join(ISIS_DIR, "isis.cc")):
        return
    try:
        subprocess.run(["make", "-C", ISIS_DIR], capture_output=True, timeout=300,
                       check=False)
    except (OSError, subprocess.SubprocessError):
        pass


# ---- the source both compilers get ------------------------------------------

_INCLUDE = re.compile(r"^\$\s*include\s*\(\s*([^)\s]+)\s*\)", re.I)
_ORIGIN = re.compile(r"^\s*[0-9][0-9a-f]*h\s*:\s*(/\*.*?\*/\s*)*$", re.I)
_TOKEN = re.compile(r"/\*.*?\*/|'(?:[^']|'')*'|[A-Za-z$_][A-Za-z0-9$_]*|[0-9][0-9A-Za-z$]*"
                    r"|:=|<>|<=|>=|\S", re.S)
_TYPE_WORDS = {"BYTE", "ADDRESS", "STRUCTURE", "LABEL", "LITERALLY", "BASED", "PUBLIC",
               "EXTERNAL", "INITIAL", "DATA", "DECLARE"}


def _tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of each token of ``text``, comments included,
    skipping control lines (a `$' in column 1)."""
    out = []
    for m in _TOKEN.finditer(text):
        s = m.start()
        line_start = text.rfind("\n", 0, s) + 1
        if text.startswith("$", line_start) and not m.group().startswith("/*"):
            continue
        out.append((s, m.end(), m.group()))
    return out


def normalize_text(text: str) -> tuple[str, list[str]]:
    """``text`` with three uplm80-only spellings made Intel PL/M-80, and what
    was changed.

    * `.'string'' becomes `.('string')' - V3.1 takes the address of a string
      only in the list form (ERROR 101, INVALID ITEM FOLLOWS DOT OPERATOR);
    * an untyped `name DATA (...)' or `name(*) DATA (...)', which uplm80
      makes BYTE, becomes `name BYTE DATA (...)' (ERROR 61, MISSING TYPE);
    * a program that is not a module - no `name: DO;' first - is made one,
      `ORACLE$MODULE: DO; ... END ORACLE$MODULE;' (ERROR 89).
    Each has the meaning uplm80 gives the original.
    """
    toks = [t for t in _tokens(text) if not t[2].startswith("/*")]
    edits: list[tuple[int, int, str]] = []
    changes = set()
    for i, (s, e, tok) in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if tok == "." and nxt and nxt[2].startswith("'"):
            edits.append((nxt[0], nxt[1], "(" + nxt[2] + ")"))
            changes.add(".'string' -> .('string')")
        if tok.upper() == "DATA" and i >= 1:
            prev = toks[i - 1][2]
            j = i - 1
            if prev == ")" and i >= 4 and toks[i - 3][2] == "(":
                j = i - 4            # name ( dim ) DATA
            name = toks[j][2]
            if re.match(r"[A-Za-z$_]", name) and name.upper() not in _TYPE_WORDS \
                    and (j == i - 1 or toks[i - 2][2] in ("*",) or
                         re.match(r"[0-9A-Za-z$_]", toks[i - 2][2])):
                # uplm80 makes `name DATA (list)' a BYTE array as long as
                # the list; Intel wants the dimension written, as (*).
                dim = ""
                if j == i - 1 and nxt and nxt[2] == "(":
                    depth, k, many = 0, i + 1, False
                    while k < len(toks):
                        t = toks[k][2]
                        depth += (t == "(") - (t == ")")
                        if depth == 0:
                            break
                        if (depth == 1 and t == ",") or (t.startswith("'") and len(t) > 3):
                            many = True
                        k += 1
                    dim = "(*) " if many else ""
                if dim:
                    edits.append((toks[j][1], toks[j][1], dim.rstrip()))
                edits.append((s, s, "BYTE "))
                changes.add("untyped DATA -> BYTE DATA")
    if len(toks) >= 4 and not (re.match(r"[A-Za-z$_]", toks[0][2]) and toks[1][2] == ":"
                               and toks[2][2].upper() == "DO" and toks[3][2] == ";"):
        first = toks[0][0] if toks else len(text)
        edits.append((first, first, "ORACLE$MODULE: DO;\n"))
        end = len(text)
        if toks and toks[-1][2].upper() == "EOF":
            end = toks[-1][0]
        edits.append((end, end, "\nEND ORACLE$MODULE;\n"))
        changes.add("wrapped in a module")
    for s, e, rep in sorted(edits, reverse=True):
        text = text[:s] + rep + text[e:]
    return text, sorted(changes)


def _find_include(name: str, dirs: list[str]) -> str | None:
    name = re.sub(r"^:f\d:", "", name, flags=re.I)
    for d in dirs:
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for fn in entries:
            if fn.lower() == name.lower():
                return os.path.join(d, fn)
    return None


def prepare_text(text: str, where: str = ".", depth: int = 0) -> str:
    """``text`` with $INCLUDEs read in, origin lines dropped, ^Z and CRs gone."""
    text = text.replace("\r", "").split("\x1a", 1)[0]
    out = []
    for line in text.split("\n"):
        m = _INCLUDE.match(line)
        if m and depth < 8:
            path = _find_include(m.group(1), [where])
            if path is None:
                raise OSError(f"$include {m.group(1)}: not found beside the source")
            with open(path, encoding="latin-1") as fh:
                out.append(prepare_text(fh.read(), os.path.dirname(path), depth + 1).rstrip("\n"))
            continue
        if _ORIGIN.match(line):
            continue
        out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


def prepare_source(path: str) -> str:
    """The text of the PL/M program at ``path`` both compilers compile."""
    with open(path, encoding="latin-1") as fh:
        return prepare_text(fh.read(), os.path.dirname(os.path.abspath(path)))


# ---- building ----------------------------------------------------------------

@dataclass
class Build:
    """A build of the program: the .COM, or why there is none."""
    com: bytes | None = None
    error: str | None = None        # set when the build failed
    timeout: bool = False
    warnings: list[str] = field(default_factory=list)
    note: str = ""
    main: bool = True               # Intel's: a main module (it has statements)


def _run(argv: list[str], cwd: str, timeout: float, env: dict | None = None,
         stdin: bytes | None = b"") -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True,
                          timeout=timeout, env=env, check=False)


def _listing_messages(lst: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) of a PL/M-80 listing."""
    errors, warnings = [], []
    lines = lst.replace("\r", "").split("\n")
    for i, line in enumerate(lines):
        if line.startswith("***"):
            msg = line.strip()
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.strip().startswith("-"):      # the listing's continuation line
                msg += nxt.strip()[1:].strip()
            (warnings if "WARNING" in line else errors).append(msg)
    m = re.search(r"(\d+) PROGRAM ERROR", lst)
    if m and int(m.group(1)) and not errors:
        errors.append(f"{m.group(1)} PROGRAM ERROR(S)")
    return errors, warnings


def _some(errors: list[str], n: int = 3) -> str:
    """The first ``n`` of ``errors``, and how many more there are."""
    more = f"; ... {len(errors) - n} more" if len(errors) > n else ""
    return "; ".join(" ".join(e.split()) for e in errors[:n]) + more


def _last_statement_offset(obj: bytes) -> int | None:
    """Code offset of the module's last statement, from the LINES records."""
    p, best = 0, None
    while p + 3 <= len(obj):
        rtype, n = obj[p], obj[p + 1] | obj[p + 2] << 8
        rec = obj[p + 3:p + 3 + n - 1]
        if rtype == 0x08 and rec and rec[0] == 1:          # LINES, CODE segment
            for q in range(1, len(rec) - 3, 4):
                off, line = rec[q] | rec[q + 1] << 8, rec[q + 2] | rec[q + 3] << 8
                if best is None or line > best[0]:
                    best = (line, off)
        if rtype == 0x0E:
            break
        p += 3 + n
    return best[1] if best else None


def patch_halt(com: bytes, obj: bytes, base: int = 0x100) -> tuple[bytes, str]:
    """``com`` with the main program's final EI; HLT made RST 0; ``base`` is
    where the module's code segment was located."""
    off = _last_statement_offset(obj)
    if off is None:
        return com, "no LINES record: final HLT left alone"
    at = base - 0x100 + off
    if com[at:at + 2] != b"\xfb\x76":
        return com, f"no EI; HLT at {at + 0x100:04X}H: left alone"
    return com[:at] + b"\xc7" + com[at + 1:], ""


def omf_start(path: str) -> int | None:
    """The start address in the MODEND record of the object file at
    ``path``; None when it is not a main module (a module with no
    statements of its own, entered through a DATA jump, as DRI's LOAD)."""
    b = _read(path) or b""
    p = 0
    while p + 3 <= len(b):
        rtype, n = b[p], b[p + 1] | b[p + 2] << 8
        if rtype == 0x04 and n >= 5:
            return (b[p + 5] | b[p + 6] << 8) if b[p + 3] == 1 else None
        if rtype == 0x0E:
            break
        p += 3 + n
    return None


def intel_build(tools: Tools, text: str, work: str, stack: int = DEFAULT_STACK,
                timeout: float = 120) -> Build:
    """Build ``text`` with Intel's PL/M-80 by P.SUB's recipe.

    LOCATE puts the module's constants - DATA, and strings in `.(...)' -
    first in the code segment and starts the program behind them, so a
    program with any does not start at 0100H, where CP/M enters it.  DRI's
    programs begin with a DATA jump for that reason; here such a program is
    located again at 0103H, where OBJCPM puts a jump to the start in front.
    """
    os.makedirs(work, exist_ok=True)
    src = text.replace("\r", "").replace("\n", "\r\n").encode("latin-1")
    with open(os.path.join(work, "T.PLM"), "wb") as fh:
        fh.write(src)
    try:
        if tools.intel_route() == "native":
            b, obj, base = _intel_native(tools, work, stack, timeout)
        else:
            b, obj, base = _intel_romwbw(tools, work, stack, timeout)
    except subprocess.TimeoutExpired as exc:
        return Build(error=f"intel: timed out: {' '.join(map(str, exc.cmd))[:120]}",
                     timeout=True)
    if b.com is not None and obj is not None:
        b.com, note = patch_halt(b.com, obj, base)
        b.note = "; ".join(x for x in (b.note, note) if x)
    return b


def _read(path: str) -> bytes | None:
    """The file at ``path``, or one of the same name in another case (the
    CP/M and ISIS tools' files are upper case, cpmemu may write lower), or
    None."""
    d, name = os.path.split(path)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        pass
    try:
        for fn in os.listdir(d or "."):
            if fn.lower() == name.lower():
                with open(os.path.join(d, fn), "rb") as fh:
                    return fh.read()
    except OSError:
        pass
    return None


def _intel_native(tools: Tools, work: str, stack: int,
                  timeout: float) -> tuple[Build, bytes | None, int]:
    for name, path in tools.intel_files.items():
        shutil.copyfile(path, os.path.join(work, name))
    isis = tools.isis
    r = _run([isis, "PLM80", "T.PLM", "DEBUG", "PAGEWIDTH(80)"], work, timeout)
    lst = (_read(os.path.join(work, "T.LST")) or b"").decode("latin-1")
    errors, warnings = _listing_messages(lst)
    if not lst:
        errors = ["PLM80 wrote no listing: " + r.stdout.decode("latin-1")[-300:]]
    if errors or not os.path.exists(os.path.join(work, "T.OBJ")):
        return Build(error="PL/M-80: " + _some(errors or ["no object"]),
                     warnings=warnings), None, 0
    r = _run([isis, "LINK", "T.OBJ,X0100,PLM80.LIB", "TO", "T.MOD"], work, timeout)
    out = r.stdout.decode("latin-1")
    if re.search(r"UNRESOLVED|ERROR", out) or not os.path.exists(os.path.join(work, "T.MOD")):
        return Build(error="LINK: " + _link_messages(out), warnings=warnings), None, 0
    note = ""
    for base in (0x100, 0x103):
        if os.path.exists(os.path.join(work, "T")):
            os.remove(os.path.join(work, "T"))
        r = _run([isis, "LOCATE", "T.MOD", f"CODE({base:04X}H)", f"STACKSIZE({stack})",
                  "MAP", "PRINT(T.TRA)"], work, timeout)
        out = r.stdout.decode("latin-1") + (_read(os.path.join(work, "T.TRA"))
                                            or b"").decode("latin-1")
        if re.search(r"ERROR", out) or not os.path.exists(os.path.join(work, "T")):
            return Build(error="LOCATE: " + " ".join(out.split())[-300:],
                         warnings=warnings), None, 0
        start = omf_start(os.path.join(work, "T"))
        if start is None:
            note = "not a main module"
            break
        if start == base:
            break
        if base == 0x100:
            note = f"starts at {start:04X}H: located at 0103H behind OBJCPM's jump"
    r = _run([tools.cpmemu, "OBJCPM.COM", "T"], work, timeout)
    com = _read(os.path.join(work, "T.COM"))
    if com is None:
        return Build(error="OBJCPM: " + r.stdout.decode("latin-1")[-300:],
                     warnings=warnings), None, 0
    return Build(com=com, warnings=warnings, note=note, main=start is not None), \
        _read(os.path.join(work, "T.OBJ")), base


def _link_messages(out: str) -> str:
    """LINK's complaint, once: it prints each line to the console twice."""
    lines = []
    for line in out.replace("\r", "").split("\n"):
        line = " ".join(line.split())
        if line and "LINKER" not in line and not line.startswith("-LINK") and line not in lines:
            lines.append(line)
    return " ".join(lines)[-300:]


def _load_romwbw_plm80(path: str):
    import importlib.machinery  # pylint: disable=import-outside-toplevel
    import importlib.util  # pylint: disable=import-outside-toplevel
    loader = importlib.machinery.SourceFileLoader("romwbw_plm80", path)
    spec = importlib.util.spec_from_loader("romwbw_plm80", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _intel_romwbw(tools: Tools, work: str, stack: int,
                  timeout: float) -> tuple[Build, bytes | None, int]:
    """P.SUB's recipe on romwbw_emu, by tools/romwbw-plm80: as its `com'
    route, or, for a program that does not start at 0100H, with its command
    lines changed to locate at 0103H."""
    out = os.path.join(work, "out")
    r = _run([sys.executable, tools.romwbw_plm80, "com", "T.PLM", "--tools", tools.intel_dir,
              "--stack", str(stack), "-o", out, "-q"], work, max(timeout, 300))
    lst = (_read(os.path.join(out, "T.LST")) or b"").decode("latin-1")
    errors, warnings = _listing_messages(lst)
    com = _read(os.path.join(out, "T.COM"))
    if errors or com is None:
        why = _some(errors) or " ".join(r.stderr.decode("latin-1").split())[-300:]
        return Build(error="PL/M-80 (romwbw-plm80): " + why, warnings=warnings), None, 0
    tra = (_read(os.path.join(out, "T.TRA")) or b"").decode("latin-1")
    m = re.search(r"MODULE START ADDRESS\s+([0-9A-F]+)H", tra)
    if "NOT A MAIN MODULE" in tra:
        return Build(com=com, warnings=warnings, note="not a main module", main=False), \
            _read(os.path.join(out, "T.OBJ")), 0x100
    if not m or int(m.group(1), 16) == 0x100:
        return Build(com=com, warnings=warnings), _read(os.path.join(out, "T.OBJ")), 0x100
    start = int(m.group(1), 16)
    mod = _load_romwbw_plm80(tools.romwbw_plm80)
    rb = mod.rb
    args = mod.build_parser().parse_args(
        ["com", "T.PLM", "--tools", tools.intel_dir, "--stack", str(stack), "-o", out, "-q"])
    cmds, outs = mod.build_commands("com", ["T.PLM"], "T", args)
    cmds = [c.replace("CODE(0100H)", "CODE(0103H)") for c in cmds]
    t = mod.Tools([tools.intel_dir])
    b = rb.Batch(workdir=None, rom=args.rom, system=args.disk, emulator=args.emu,
                 offline=args.offline, isx=True, timeout=args.timeout, verbose=False,
                 max_instructions=args.max_instructions)
    try:
        for n in sorted(t.files):
            b.add(t.files[n], n, drive="B")
        b.add(os.path.join(work, "T.PLM"), None, "A", text=True)
        kept = mod.clear_outputs(b, outs)
        mod.build(b, cmds, outs, args, kept)
    finally:
        b.cleanup()
    com = _read(os.path.join(out, "T.COM"))
    if com is None:
        return Build(error="romwbw-plm80 at 0103H: no T.COM", warnings=warnings), None, 0
    note = f"starts at {start:04X}H: located at 0103H behind OBJCPM's jump"
    return Build(com=com, warnings=warnings, note=note), _read(os.path.join(out, "T.OBJ")), 0x103


# DRI's X0100.ASM (PLM_WORK), which P.SUB links into every program: the BDOS
# entry and page-zero locations as PUBLIC equates.  uplm80's build links the
# same, as INSTALL.md says a program that uses them must.
X0100_MAC = """; X0100 - DRI's PLM_WORK/X0100.ASM for um80
        public  mon1,mon2,mon2a,mon3
        public  cmdrv,fcb,pass0,len0
        public  fcb16,pass1,len1,tbuff
        public  bdisk,maxb,buff,boot
mon1    equ     0005h
mon2    equ     0005h
mon2a   equ     0005h
mon3    equ     0005h
cmdrv   equ     0050h
fcb     equ     005ch
pass0   equ     0051h
len0    equ     0053h
fcb16   equ     006ch
pass1   equ     0054h
len1    equ     0056h
tbuff   equ     0080h
bdisk   equ     0004h
maxb    equ     0006h
buff    equ     0080h
boot    equ     0000h
        end
"""


def uplm80_build(tools: Tools, text: str, opt: int, work: str, timeout: float = 300,
                 mode: str = "cpm") -> Build:
    """Build ``text`` with this checkout's uplm80 at -O ``opt``, um80 and ul80."""
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "T.PLM"), "w", encoding="latin-1") as fh:
        fh.write(text)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (REPO, env.get("PYTHONPATH")) if p)
    steps = [
        ("uplm80", [sys.executable, "-P", "-m", "uplm80.compiler", "-O", str(opt), "-m", mode,
                    "-o", "T.MAC", "T.PLM"]),
        ("um80", [tools.um80, "-o", "T.REL", "T.MAC"]),
        ("um80", [tools.um80, "-o", "X0100.REL", "X0100.MAC"]),
        ("ul80", [tools.ul80, "-o", "T.COM", "T.REL", "X0100.REL"]),
    ]
    with open(os.path.join(work, "X0100.MAC"), "w", encoding="ascii") as fh:
        fh.write(X0100_MAC)
    for name, argv in steps:
        try:
            r = _run(argv, work, timeout, env=env)
        except subprocess.TimeoutExpired:
            return Build(error=f"{name}: timed out", timeout=True)
        if r.returncode:
            msg = (r.stdout + r.stderr).decode("latin-1").strip().splitlines()
            return Build(error=f"{name}: " + " | ".join(msg[-4:])[:400])
    com = _read(os.path.join(work, "T.COM"))
    if com is None:
        return Build(error="ul80: no T.COM")
    return Build(com=com)


# ---- running -----------------------------------------------------------------

@dataclass
class Run:
    """What a program printed, and how it ended: exit, timeout or abnormal."""
    out: bytes = b""
    status: str = "exit"
    detail: str = ""


def run_com(tools: Tools, com: bytes, work: str, timeout: float = 10) -> Run:
    """Run ``com`` under cpmemu in its own directory, stdin empty."""
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "T.COM")
    with open(path, "wb") as fh:
        fh.write(com)
    try:
        r = subprocess.run([tools.cpmemu, "T.COM"], cwd=work, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return Run(out=exc.stdout or b"", status="timeout", detail=f"ran over {timeout}s")
    err = r.stderr.decode("latin-1")
    if r.returncode == 0 and re.search(r"^(Program exit|System reset)", err, re.M):
        return Run(out=r.stdout)
    tail = " | ".join(line for line in err.splitlines()
                      if not line.startswith(("CPU mode", "Loaded")))[-300:]
    return Run(out=r.stdout, status="abnormal", detail=f"exit {r.returncode}: {tail}")


# ---- checking one program ------------------------------------------------------

VERDICTS = ("same", "differs", "intel-rejects", "uplm80-rejects", "timeout")


@dataclass
class Result:  # pylint: disable=too-many-instance-attributes
    """The verdict on one program, and what it rests on."""
    name: str
    verdict: str = "same"
    note: str = ""                      # how the source was changed
    intel_error: str | None = None
    intel_note: str = ""
    intel_out: str | None = None
    intel_status: str | None = None
    levels: dict = field(default_factory=dict)  # level -> {out|error, status}
    detail: str = ""
    seconds: float = 0.0

    def outputs(self) -> dict[str, list[int]]:
        """Distinct uplm80 outputs (or errors) and the levels that gave them."""
        groups: dict[str, list[int]] = {}
        for lvl, r in sorted(self.levels.items()):
            key = r.get("error") or r.get("out") or ""
            if r.get("status") not in (None, "exit"):
                key = f"{key}\n[{r['status']}: {r.get('detail', '')}]"
            groups.setdefault(key, []).append(int(lvl))
        return groups


def _show(out: bytes) -> str:
    return out.decode("latin-1").replace("\r\n", "\n")


def check_text(tools: Tools, text: str, name: str = "T", levels=LEVELS, timeout: float = 10,
               work: str | None = None, stack: int = DEFAULT_STACK,
               keep: bool = False, note: str = "", mode: str = "auto") -> Result:
    """Build ``text`` with both compilers, run every build and compare.

    ``mode`` is uplm80's --mode; `auto' is `cpm', but `bare' for a module
    with no statements of its own (LOCATE: NOT A MAIN MODULE), which is
    entered through a DATA jump its constants put at 0100H - as DRI's CP/M
    2.0 LOAD is - and which only bare mode lays out as Intel does."""
    t0 = time.time()
    res = Result(name=name, note=note)
    top = work or tempfile.mkdtemp(prefix="oracle-")
    os.makedirs(top, exist_ok=True)
    try:
        ib = intel_build(tools, text, os.path.join(top, "intel"), stack)
        res.intel_note = ib.note
        intel_run = None
        if ib.com is None:
            res.intel_error = ib.error
        else:
            intel_run = run_com(tools, ib.com, os.path.join(top, "intel-run"), timeout)
            res.intel_out = _show(intel_run.out)
            res.intel_status = intel_run.status
        if mode == "auto":
            mode = "bare" if ib.com is not None and not ib.main else "cpm"
            if mode == "bare":
                res.note = "; ".join(x for x in (res.note, "uplm80 -m bare") if x)
        for lvl in levels:
            ub = uplm80_build(tools, text, lvl, os.path.join(top, f"u{lvl}"), mode=mode)
            if ub.com is None:
                res.levels[lvl] = {"error": ub.error, "status": "timeout" if ub.timeout else None}
                continue
            ur = run_com(tools, ub.com, os.path.join(top, f"u{lvl}-run"), timeout)
            res.levels[lvl] = {"out": _show(ur.out), "status": ur.status, "detail": ur.detail}
        _judge(res, ib, intel_run)
    finally:
        if not keep and work is None:
            shutil.rmtree(top, ignore_errors=True)
    res.seconds = round(time.time() - t0, 2)
    return res


def _judge(res: Result, ib: Build, intel_run: Run | None) -> None:
    rejected = [lvl for lvl, r in res.levels.items() if "error" in r and not r.get("status")]
    timed = [lvl for lvl, r in res.levels.items() if r.get("status") == "timeout"]
    if ib.com is None:
        res.verdict = "timeout" if ib.timeout else "intel-rejects"
        res.detail = ib.error or ""
        if rejected:
            res.detail += f" (uplm80 rejects it too at -O{','.join(map(str, rejected))})"
        return
    if rejected:
        res.verdict = "uplm80-rejects"
        res.detail = f"-O{','.join(map(str, rejected))}: " + res.levels[rejected[0]]["error"]
        return
    if intel_run is not None and intel_run.status == "timeout":
        timed = ["intel"] + timed
    if timed:
        res.verdict = "timeout"
        res.detail = "runs over the limit: " + ", ".join(
            "Intel" if t == "intel" else f"-O{t}" for t in timed)
        return
    differ = [lvl for lvl, r in res.levels.items()
              if r.get("out") != res.intel_out or r.get("status") != res.intel_status]
    if differ:
        res.verdict = "differs"
        res.detail = f"-O{','.join(map(str, differ))} differ from Intel"
        if res.intel_status != "exit":
            res.detail += f" (Intel's run: {intel_run.status} {intel_run.detail})"


def first_difference(a: str, b: str) -> str:
    """Where two outputs first part: the line, and the words there."""
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            wx, wy = x.split(" "), y.split(" ")
            j = next((k for k, (p, q) in enumerate(zip(wx, wy)) if p != q),
                     min(len(wx), len(wy)))
            return (f"line {i + 1}, word {j + 1}: Intel {' '.join(wx[j:j + 3])[:60]!r}, "
                    f"uplm80 {' '.join(wy[j:j + 3])[:60]!r}")
    if len(la) != len(lb):
        return f"Intel prints {len(la)} lines, uplm80 {len(lb)}"
    return "same"


def format_result(res: Result, full: bool = True, limit: int = 1200) -> str:
    """A report on ``res``: one line, and the outputs when it is not `same'."""
    head = f"{res.name}: {res.verdict}"
    if res.detail:
        head += f" - {res.detail}"
    if res.note:
        head += f" [{res.note}]"
    if res.verdict == "same":
        n = len((res.intel_out or "").splitlines())
        return f"{head} ({n} lines, {res.seconds}s)"
    lines = [head]
    if not full or res.verdict not in ("differs", "timeout"):
        return "\n".join(lines)
    if res.intel_note:
        lines.append(f"  note: {res.intel_note}")

    def block(label: str, text: str) -> None:
        body = text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"
        lines.append(f"  --- {label}")
        lines.extend("  | " + x for x in body.rstrip("\n").split("\n"))

    block(f"Intel ({res.intel_status})", res.intel_out or "")
    for text, lvls in res.outputs().items():
        label = "uplm80 -O" + ",".join(map(str, lvls))
        if text == res.intel_out:
            lines.append(f"  --- {label}: same as Intel")
            continue
        block(label, text)
        lines.append("  first difference: " + first_difference(res.intel_out or "", text))
    return "\n".join(lines)


# ---- the programs to check ---------------------------------------------------------

def corpus_files() -> list[str]:
    """Every PL/M program under tests/ and sample_code/."""
    out = []
    for top in ("tests", "sample_code"):
        for d, _, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                if fn.lower().endswith(".plm"):
                    out.append(os.path.join(d, fn))
    return sorted(out)


def generated(seed: int, **kw) -> tuple[str, str]:
    """(name, text) of the generator's program for ``seed``."""
    from tests.plm_intel import generate  # pylint: disable=import-outside-toplevel
    return f"seed{seed}", generate(seed, **kw).render()


# ---- reducing a difference ------------------------------------------------------

class Interesting:  # pylint: disable=too-few-public-methods
    """Whether a program still shows the difference being reduced: at level
    ``level``, Intel's build prints one thing and uplm80's another
    (mode `differs'), or Intel accepts it and uplm80 rejects it with an
    error containing ``needle'' (mode `rejects')."""

    def __init__(self, tools: Tools, level: int, mode: str = "differs", needle: str = "",
                 timeout: float = 5, stack: int = DEFAULT_STACK, main: bool = True) -> None:
        self.tools, self.level, self.mode, self.needle = tools, level, mode, needle
        self.timeout, self.stack = timeout, stack
        self.main = main        # Intel's build must end in the patched HLT
        self.status = "exit"    # how uplm80's run must end
        # The first lines where the outputs part, which a reduction keeps:
        # without it, a program cut down to garbage differs too.
        self.signature: tuple[str, str] | None = None
        self.tests = 0

    def __call__(self, text: str) -> bool:
        self.tests += 1
        top = tempfile.mkdtemp(prefix="reduce-")
        try:
            ib = intel_build(self.tools, text, os.path.join(top, "intel"), self.stack)
            if ib.com is None or (self.main and "left alone" in ib.note):
                return False
            ub = uplm80_build(self.tools, text, self.level, os.path.join(top, "u"),
                              mode="cpm" if ib.main else "bare")
            if self.mode == "rejects":
                return ub.com is None and not ub.timeout and self.needle in (ub.error or "")
            if ub.com is None:
                return False
            ir = run_com(self.tools, ib.com, os.path.join(top, "ir"), self.timeout)
            if ir.status != "exit":
                return False
            ur = run_com(self.tools, ub.com, os.path.join(top, "ur"), self.timeout)
            if ur.status != self.status or ir.out == ur.out:
                return False
            return self.signature is None or \
                first_lines(_show(ir.out), _show(ur.out)) == self.signature
        finally:
            shutil.rmtree(top, ignore_errors=True)


def ddmin(items: list, test, jobs: int = 1) -> list:
    """A 1-minimal sublist of ``items`` for which ``test`` holds (Zeller's
    ddmin, complements only, each round's candidates tried in parallel)."""
    n = 2
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while len(items) >= 2:
            size = -(-len(items) // n)
            cands = [items[:i] + items[i + size:] for i in range(0, len(items), size)]
            found = None
            for k in range(0, len(cands), max(1, jobs)):
                batch = cands[k:k + max(1, jobs)]
                for cand, ok in zip(batch, pool.map(test, batch)):
                    if ok:
                        found = cand
                        break
                if found is not None:
                    break
            if found is not None:
                items = found
                n = max(n - 1, 2)
            elif n >= len(items):
                break
            else:
                n = min(len(items), 2 * n)
    return items


def _blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line of each DO or PROCEDURE block that starts and ends
    on lines of its own, largest first."""
    opens: list[int] = []
    spans = []
    for i, line in enumerate(lines):
        words = [t.upper() for _, _, t in _tokens(line) if not t.startswith(("/*", "'"))]
        for w in words:
            if w in ("DO", "PROCEDURE"):
                opens.append(i)
            elif w == "END" and opens:
                j = opens.pop()
                if j != i:
                    spans.append((j, i))
    return sorted(set(spans), key=lambda sp: sp[0] - sp[1])


def reduce_lines(text: str, test, jobs: int = 1) -> str:
    """``text`` with every line, and every block, ``test`` does not need
    removed."""
    lines = text.rstrip("\n").split("\n")
    while True:
        before = len(lines)
        # whole blocks first: a procedure goes only with all its lines
        i = 0
        spans = _blocks(lines)
        while i < len(spans):
            a, b = spans[i]
            cand = lines[:a] + lines[b + 1:]
            if test("\n".join(cand) + "\n"):
                lines = cand
                spans = _blocks(lines)
                i = 0
                continue
            i += 1
        lines = ddmin(lines, lambda ls: test("\n".join(ls) + "\n"), jobs)
        if len(lines) == before:
            return "\n".join(lines) + "\n"


def reduce_program(prog, test, jobs: int = 1, tag: str | None = None) -> str:
    """A generated program with every chunk, then every line, ``test`` does
    not need removed.  ``tag`` names the statement (S1F) whose line differed
    first: the program with that statement alone is tried first."""
    ids = [c.id for c in prog.chunks()]
    if tag:
        mine = [c for c in prog.chunks() if f"/* S{tag} */" in c.parts]
        if mine:
            alone = [c.id for c in prog.chunks()
                     if c.kind not in ("stmt", "dump") or c is mine[0]]
            if test(prog.render(frozenset(ids) - set(alone))):
                ids_left = alone
                keep = set(ddmin(ids_left, lambda ks: test(prog.render(frozenset(ids) - set(ks))),
                                 jobs))
                return reduce_lines(prog.render(frozenset(ids) - keep), test, jobs)
    keep = set(ddmin(ids, lambda ks: test(prog.render(frozenset(ids) - set(ks))), jobs))
    return reduce_lines(prog.render(frozenset(ids) - keep), test, jobs)


def first_lines(intel: str, other: str) -> tuple[str, str]:
    """The first pair of lines that differ, without trailing blanks."""
    for a, b in zip(intel.split("\n") + [""], other.split("\n") + [""]):
        if a != b:
            return a.rstrip(), b.rstrip()
    return "", ""


def first_tag(intel: str, other: str) -> str | None:
    """The statement tag (`1F') of the first line where the outputs part."""
    for a, b in zip(intel.split("\n") + [""], other.split("\n") + [""]):
        if a != b:
            m = re.match(r"([0-9A-F]{2}):", a) or re.match(r"([0-9A-F]{2}):", b)
            return m.group(1) if m else None
    return None


# ---- command line ---------------------------------------------------------------

def reduce_result(tools: Tools, name: str, text: str, res: Result, args,
                  gen_kw: dict) -> str | None:
    """The reduced program for a result that differs or that uplm80 rejects,
    checked again, as a report; None for other results."""
    tag = None
    if res.verdict == "differs":
        bad = [lvl for lvl, r in sorted(res.levels.items())
               if r.get("out") != res.intel_out or r.get("status") != res.intel_status]
        tag = first_tag(res.intel_out or "", res.levels[bad[0]].get("out") or "")
        test = Interesting(tools, args.reduce_level if args.reduce_level is not None else bad[0],
                           timeout=min(args.timeout, 5), stack=args.stack,
                           main="left alone" not in res.intel_note)
        # A reduction keeps uplm80's run ending the way it did: a program
        # cut down to an endless recursion also differs, and says nothing.
        test.status = res.levels[test.level].get("status") or "exit"
        test.signature = first_lines(res.intel_out or "", res.levels[test.level].get("out") or "")
    elif res.verdict == "uplm80-rejects" and res.levels:
        bad = [lvl for lvl, r in sorted(res.levels.items()) if "error" in r]
        err = res.levels[bad[0]]["error"]
        needle = re.sub(r"^.*?error: ", "", err).split(" | ")[0][:60]
        test = Interesting(tools, bad[0], "rejects", needle, stack=args.stack)
    else:
        return None
    if not test(text):
        return f"{name}: reduce: the difference does not show again at -O{test.level}"
    m = re.fullmatch(r"seed(\d+)", name)
    if m:
        from tests.plm_intel import generate  # pylint: disable=import-outside-toplevel
        small = reduce_program(generate(int(m.group(1)), **gen_kw), test, args.jobs, tag)
    else:
        small = reduce_lines(text, test, args.jobs)
    again = check_text(tools, small, name + " reduced", LEVELS, args.timeout, stack=args.stack)
    out = [f"=== {name} reduced at -O{test.level} ({test.tests} tests):", small.rstrip("\n"),
           format_result(again)]
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
        base = os.path.join(args.keep, re.sub(r"[^\w.-]", "_", name) + "-reduced")
        with open(base + ".plm", "w", encoding="latin-1") as fh:
            fh.write(small)
        with open(base + ".txt", "w", encoding="latin-1") as fh:
            fh.write(format_result(again, limit=10 ** 7) + "\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    """Check the programs named on the command line; see the module doc."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("programs", nargs="*", metavar="PROGRAM.plm")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="check N generated programs (tests/plm_intel.py)")
    ap.add_argument("--first", type=int, default=1, metavar="SEED")
    ap.add_argument("--stmts", type=int, default=None, help="statements per generated program")
    ap.add_argument("--avoid", default="", metavar="F,F",
                    help="generator features to leave out (see tests/plm_intel.py)")
    ap.add_argument("--corpus", action="store_true",
                    help="check every .plm under tests/ and sample_code/")
    ap.add_argument("--normalize", action="store_true",
                    help="rewrite three uplm80-only spellings first (see normalize_text)")
    ap.add_argument("-O", dest="levels", default="0,1,2,3")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=float, default=10, help="seconds per run (10)")
    ap.add_argument("--mode", choices=["auto", "cpm", "bare"], default="auto",
                    help="uplm80's --mode (auto: bare for a module with no statements)")
    ap.add_argument("--stack", type=int, default=DEFAULT_STACK,
                    help=f"LOCATE's STACKSIZE ({DEFAULT_STACK}; P.SUB says 100)")
    ap.add_argument("--tools", help="directory of Intel's tools ($PLM80_TOOLS)")
    ap.add_argument("--isis", help="the ISIS emulator ($ISIS_EMU, tools/isis/isis)")
    ap.add_argument("--route", choices=["auto", "native", "romwbw"], default="auto",
                    help="run Intel's tools on tools/isis (native) or romwbw-plm80")
    ap.add_argument("--json", metavar="FILE", help="write every result to FILE")
    ap.add_argument("--keep", metavar="DIR",
                    help="save the source and outputs of every program that is not `same'")
    ap.add_argument("--work", metavar="DIR", help="build here and keep the builds")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 when the tools are missing, instead of 0")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only what is not `same'")
    ap.add_argument("--reduce", action="store_true",
                    help="reduce each program that differs (or that uplm80 rejects) to the "
                         "smallest that still does, at the first level that shows it")
    ap.add_argument("--reduce-level", type=int, default=None, metavar="L",
                    help="reduce at -O L instead")
    args = ap.parse_args(argv)

    tools = Tools.find(args.tools, args.isis, args.route)
    why = tools.missing()
    if why:
        print(f"intel_oracle: skipped - {why}", file=sys.stderr)
        return 2 if args.strict else 0
    levels = tuple(int(x) for x in args.levels.split(","))

    jobs: list[tuple[str, str, str]] = []     # (name, text, note)
    names = list(args.programs) + (corpus_files() if args.corpus else [])
    for path in names:
        name = os.path.relpath(path, REPO) if os.path.abspath(path).startswith(REPO) else path
        try:
            text = prepare_source(path)
        except (OSError, UnicodeError) as exc:
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            continue
        note = ""
        if args.normalize:
            text, changes = normalize_text(text)
            note = "; ".join(changes)
        jobs.append((name, text, note))
    gen_kw = {}
    if args.stmts:
        gen_kw["n_stmts"] = args.stmts
    if args.avoid:
        gen_kw["avoid"] = frozenset(a.strip() for a in args.avoid.split(",") if a.strip())
    for seed in range(args.first, args.first + args.random):
        jobs.append(generated(seed, **gen_kw) + ("",))
    if not jobs:
        ap.error("nothing to check: name programs, or give --random N or --corpus")

    print(f"intel_oracle: {len(jobs)} programs, -O{','.join(map(str, levels))}, "
          f"Intel's tools from {tools.intel_dir} via {tools.intel_route()}",
          file=sys.stderr)

    def one(job: tuple[str, str, str]) -> Result:
        name, text, note = job
        work = os.path.join(args.work, re.sub(r"[^\w.-]", "_", name)) if args.work else None
        try:
            return check_text(tools, text, name, levels, args.timeout, work, args.stack,
                              note=note, mode=args.mode)
        except Exception as exc:  # pylint: disable=broad-except
            return Result(name=name, verdict="uplm80-rejects",
                          detail=f"oracle error: {type(exc).__name__}: {exc}")

    results: list[Result] = []
    counts = {v: 0 for v in VERDICTS}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for (name, text, _), res in zip(jobs, pool.map(one, jobs)):
            results.append(res)
            counts[res.verdict] = counts.get(res.verdict, 0) + 1
            if not (args.quiet and res.verdict == "same"):
                print(format_result(res))
                sys.stdout.flush()
            if args.keep and res.verdict != "same":
                os.makedirs(args.keep, exist_ok=True)
                base = os.path.join(args.keep, re.sub(r"[^\w.-]", "_", name))
                with open(base + ".plm", "w", encoding="latin-1") as fh:
                    fh.write(text)
                with open(base + ".txt", "w", encoding="latin-1") as fh:
                    fh.write(format_result(res, limit=10 ** 7) + "\n")
    if args.reduce:
        for (name, text, _), res in zip(jobs, results):
            reduced = reduce_result(tools, name, text, res, args, gen_kw)
            if reduced:
                print(reduced)
                sys.stdout.flush()
    summary = ", ".join(f"{counts[v]} {v}" for v in VERDICTS if counts.get(v))
    print(f"intel_oracle: {len(results)} programs at -O{','.join(map(str, levels))}: "
          f"{summary} ({time.time() - t0:.0f}s)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=1)
    return 0 if counts["same"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
