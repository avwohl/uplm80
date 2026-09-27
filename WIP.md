# Work in progress — 2026-09-27

Every bug found in this round is fixed, merged and released.  Every repository
below is on `main`, has no other branches, no extra worktrees and no stash, and
nothing is left unpushed.  What remains is under **Still open**: each
repository's CHANGELOG lists it as Known issues, and none of it blocks anything.

## Released

| Repository         | Releases | What they brought |
|--------------------|----------|-------------------|
| `uplm80`           | 0.3.7, 0.4.0, 0.4.1, 0.4.2, 0.4.3, **0.4.4** | 0.3.7: procedure locals as PL/M-80 specifies them, overlaid in `??AUTO` only when provably safe; names and labels as PL/M-80's scopes give them. 0.4.0: **Intel PL/M-80's calling convention** across modules and through addresses. 0.4.1: the errors Intel's PL/M-80 V3.1 gives. 0.4.2: uplm80 checked against V3.1 itself (`scripts/intel_oracle.py`, in the suite); LENGTH/LAST/SIZE of qualified references; MEMORY and counted loops; a label on END. 0.4.3: **SHL and SHR of a BYTE are a BYTE**, as in V3.1, with a warning where the old 16-bit meaning differs; more of V3.1's errors. 0.4.4: DATA, INITIAL, AT and constant lists as V3.1 takes them - a name there is its block's variable (a program's own MEMORY or SIZE, and factored BASED names, were miscompiled); BASED bases and LABELs checked; a warning at every flag reader a shift of a BYTE's flags may reach, and no `-O` level drops an operation whose flags are read; more of V3.1's errors; less dead code after 8-bit shifts. |
| `upeepz80`         | 0.2.5, 0.2.6, **0.2.7** | Rewrites only where nothing reads what they change; dead-store elimination sound against address arithmetic and stack slots; no tail call over pushed arguments (needed by uplm80 0.4.0); patched instructions and `TABLE equ $` jump tables. |
| `um80_and_friends` | 0.3.49 - **0.3.52** | `--dri` reads DRI's MAC/RMAC sources (DRI's MP/M II nucleus assembles to the genuine RMAC 1.1 objects), and 0.3.52 follows MAC in a two-character string's byte order, IF's bit 0 and MACLIB; M80's reading exact against the genuine M80 3.44; ul80's LINK-80 `%Mult. Def. Global`. |
| `80un`             | **0.3.3** | Built with uplm80 0.4.3: BYTE shifts written `SHL(DOUBLE(x), n)` where 16 bits are wanted; members of 64K and more extracted to their end (all 18 of `test.arc`); Crunch V1; a CP/M name for every member; empty, broken and cut-short archives end cleanly. The first 80un on PyPI since 0.3.1 (0.3.2 was tagged, never published). |
| `mpm2`             | 0.3.6, **0.3.7** | MP/M II V2.1 from source as well as V2.0; DRI's PL/M and assembler sources build unmodified wherever only the toolchain needed a change; `verify_dri.py` compares more of DRI's files, whole; CI's source-built system tested for real. |
| `romwbw_emu`       | 1.48, **1.49** | `tools/romwbw-batch` and `tools/romwbw-plm80` (Intel PL/M-80 under DRI's ISX); console-idle and piped-input fixes; `--max-instructions`; 1.49: a SYSCONF autoboot countdown takes the seconds it says. |
| `cpmemu`           | **4.10.0** | The `.cfg` file alone decides text/binary and CR LF conversion (`default_mode` applies to opens too). |

`mbasic2025` (2d19520) is a test input of `um80_and_friends`: every historic
MBASIC variant builds byte for byte with um80/ul80 and with the genuine M80/L80.

### Toolchain on a fresh machine

Editable installs under Homebrew's Python 3.14:

```bash
for r in uplm80 upeepz80 um80_and_friends; do
  (cd ~/src/$r && /opt/homebrew/bin/python3.14 -m pip install --user --break-system-packages -e .)
done
make -C ~/src/cpmemu/src          # cpmemu; mpm2 also uses ~/src/cpmemu/util/cpm_disk.py
```

or from PyPI: `uplm80>=0.4.4` (pulls `upeepz80>=0.2.6`), `um80>=0.3.52`,
`80un>=0.3.3`.

### Checking uplm80 against Intel's PL/M-80 V3.1

`scripts/intel_oracle.py` builds a program with Intel's PLM80, LINK, LOCATE
and OBJCPM (run on `tools/isis`, `make -C tools/isis`) and with uplm80 at
`-O0` to `-O3`, and compares what the builds print.  Intel's binaries are not
in the repository; it finds them on DRI's MP/M II work disk
(`mpm2/mpm2_external/mpm2src/PLM_WORK`) or through `$PLM80_TOOLS`.
`--random N` checks generated programs, `--corpus` the programs of `tests/`
and `sample_code/`.  The README's Known differences lists V3.1's own bugs the
campaigns found.

### How MP/M II is checked

```bash
cd ~/src/mpm2
python3.14 tools/build.py && python3.14 tools/build.py --version 2.1
python3.14 tools/verify_dri.py
./scripts/build_all.sh --tree=src --version=2.1 && ./scripts/run_tests.sh all
```

## Still open

None of these is a regression; each is listed, with its examples, in the
CHANGELOG named.

- **uplm80** (CHANGELOG 0.4.4, Known issues): forms V3.1 rejects that uplm80
  still compiles or refuses in its own words (a typed procedure with no
  RETURN, #156; a label as a value, #132; a number above 0FFFFH, #94; an
  empty string in DATA; an untyped DATA list, #61; LENGTH/LAST/SIZE with two
  subscripts); messages that name fewer of V3.1's errors than V3.1 gives,
  for some combinations of two errors; the flags V3.1's own code leaves
  otherwise (`INR`/`DCR`, `ANI`+`RAR`), documented in the README; `??AUTO`'s
  layout is uplm80's, not DRI's.  The 7 SHL warnings and 4 flags warnings on
  DRI's MP/M II sources are expected: each is a place the old 16-bit shift
  and V3.1's differ, and the program is right as V3.1 compiles it.
- **um80_and_friends** (CHANGELOG 0.3.52): forms MAC, RMAC or M80 flag that
  um80 assembles without a word; M80's multi-line `.COMMENT`; `DS` with no
  operand; with `--dri`, a `!` after `DB`/`DW`, `LOCAL` in a `REPT`/`IRP`,
  a `MACLIB` inside a library, and RMAC's segment sizes.
- **upeepz80** (CHANGELOG 0.2.7): its stated assumptions about the stack -
  reached only through SP, and a jump out of the text taken to find its
  return address on top.
- **80un** (CHANGELOG 0.3.3): past about 100 names in a 64K CP/M 2.2, names
  are made but not kept, so two members made alike can land on one file;
  CrLZH decodes on past a cut-short member's end; `src/plm/archive/80un.plm`,
  the old one-file program, is not built and still relies on the 16-bit
  shift.
- **mpm2** (CHANGELOG 0.3.6 and 0.3.7): GENHEX differs from DRI's in 70 bytes
  no source sets; `--version 2.0` uses V2.1's banked BDOS, on purpose; SDIR
  built through ISX (romwbw_emu) differs from DRI's in uninitialised bytes,
  because DRI built on a 62K CP/M.
- **romwbw_emu** (`todo.txt`, `DECISIONS.md`): `romwbw-batch` and
  `romwbw-plm80` are not in the .deb/.rpm packages and need cpmemu's
  `cpm_disk.py`.

### Housekeeping

- `docs/git_stash_bug.md`: `git stash` / `git stash pop` in a clean worktree
  pops another worktree's entry.  The owner is reviewing it; no rule has been
  added to `CLAUDE.md`.  Two bug-report drafts about it are queued in
  `~/.claude/feedback/drafts` (`/feedback` sends or dismisses them).
- `uplm80/todo.txt`: `../uplox`'s `examples/plm_subset.uplox` still forbids a
  line break in a string; only uplox's own tests use it.
- `uplm80/mpm.sys` is an untracked local file, left as it was.
