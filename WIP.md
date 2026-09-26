# Work in progress — 2026-09-26

Everything from the 2026-09-23 handoff is finished, merged and released.
uplm80 0.4.1 is in progress on the branch `fix/0.4.1`, in the worktree
`~/src/uplm80-041`: not pushed, and its version not bumped.  Every other
repository below is on `main`, has no other branches and no extra worktrees, and
nothing is left unpushed.  What remains are the follow-ups listed under
**Still open**, none of which blocks anything.

## Released

| Repository         | Release | What it brought |
|--------------------|---------|-----------------|
| `uplm80`           | 0.3.7, **0.4.0** | 0.3.7: procedure locals stored as PL/M-80 specifies (overlaid in `??AUTO` only when provably safe), names and labels as PL/M-80's scopes give them, a GOTO out of a procedure reloads SP where DRI's compiler does. 0.4.0: calls across module boundaries, and through addresses, use **Intel PL/M-80's calling convention** (one argument in BC/C; two or more: next-to-last in BC/C, last in DE/E, earlier ones pushed, callee pops; BYTE result in A, ADDRESS in HL). |
| `upeepz80`         | 0.2.5, **0.2.6** | 0.2.5: rewrites made only where nothing reads what they change (an instruction model and liveness). 0.2.6: dead-store elimination sound against address arithmetic, stack slots and return addresses reachable through SP pointers, and no tail call where arguments are pushed under the return address (needed by uplm80 0.4.0). |
| `um80_and_friends` | 0.3.49, 0.3.50, **0.3.51** | 0.3.51: `--dri` reads DRI's MAC/RMAC sources (DRI's unmodified MP/M II nucleus assembles to the genuine RMAC 1.1 objects); M80's reading made exact against the genuine M80 3.44 (bit 7, column-1 words, operator words, `%`, macro-argument splitting); ul80's LINK-80 `%Mult. Def. Global` and `--fatal-mult-def`. |
| `cpmemu`           | **4.10.0** | The `.cfg` file alone decides text/binary and CR LF conversion (`default_mode` applies to opens too). |
| `romwbw_emu`       | **1.48** | `tools/romwbw-batch` (unattended CP/M runs on a disk image) and `tools/romwbw-plm80` (Intel PL/M-80 under DRI's ISX, as DRI built MP/M II); console-idle and piped-input fixes; `--max-instructions`. |
| `mpm2`             | **0.3.6** | MP/M II V2.1 from source as well as V2.0; DRI's PL/M and assembler sources build unmodified wherever only the toolchain needed a change; DRI's own X0100/BRSPBI/LDMONX linked; DRI's four RSPs in every system; the XIOS result race, the disk-bank bug and the SFTP/SUBMIT/SPOOL faults fixed. CI pins the toolchain release tags. |

`mbasic2025` (2d19520) is a test input of `um80_and_friends`: every historic
MBASIC variant builds byte for byte with um80/ul80 and with the genuine
M80/L80 in every mix (`tests/test_mbasic2025.py`, `tools/fourway_mbasic.py`).

### Toolchain on a fresh machine

Editable installs under Homebrew's Python 3.14 (plain `python3` on this machine
is Apple's 3.9.6):

```bash
for r in uplm80 upeepz80 um80_and_friends; do
  (cd ~/src/$r && /opt/homebrew/bin/python3.14 -m pip install --user --break-system-packages -e .)
done
make -C ~/src/cpmemu/src          # cpmemu; mpm2 also uses ~/src/cpmemu/util/cpm_disk.py
```

or from PyPI: `uplm80>=0.4.0` (pulls `upeepz80>=0.2.6`), `um80>=0.3.51`.
uplm80 0.4.0 refuses, at -O1 and up, an upeepz80 without the tail-call guard.

### How MP/M II is checked

```bash
cd ~/src/mpm2
python3.14 tools/build.py && python3.14 tools/build.py --version 2.1   # 44/44 each
python3.14 tools/verify_dri.py        # XDOS, BNKXDOS, RESBDOS, TMP, RDT, DDT identical
./scripts/build_all.sh --tree=src --version=2.1 && ./scripts/run_tests.sh all
```

`docs/mpm2_v21.md` is the V2.1 write-up; `tools/v21/` the research tools.

## Still open

### uplm80

* Differential testing against Intel's own compiler.  `romwbw_emu`'s
  `tools/romwbw-plm80` runs DRI's ISX with Intel's PLM80 V3.1 (from
  `mpm2/mpm2_external/mpm2src/PLM_WORK`) in about 1.5 CPU seconds per small
  compile; a 174-line test program already prints the same 66 lines from both
  compilers.  The next step is a `scripts/` oracle that feeds the random-program
  difftest through it.  0.4.1's items were each checked against V3.1 that way
  by hand, through a native ISIS emulator (`isis.cc`), again in a session
  scratch directory.  (A native ISIS emulator written during the 0.4.0
  research rebuilt 18 DRI programs byte for byte; it lived only in a session
  scratch directory.  Intel's binaries cannot be vendored into this repository.)
* 0.4.1 (`fix/0.4.1`, CHANGELOG `## 0.4.1 — unreleased`) settles the Known
  issues 0.3.7 listed and 0.4.0 carried, each against Intel's PL/M-80 V3.1's
  listing, diagnostics and linked build: names declared nowhere, empty
  parentheses, `.label`, nested INTERRUPT procedures, undeclared parameters,
  a LITERALLY used before its declaration, a dimension that is not a number
  and built-in names (in a multi-file compile too) are checked as V3.1
  checks them; a counted loop over a module-level or static index sees
  pointers and overruns, run on or back, of a structure's members too;
  PLUS/MINUS/SCL/SCR after `+ 4`; a REENTRANT procedure's factored parameter.
  Its own Known issues list what V3.1 still rejects and uplm80 compiles (`f()`
  of a procedure, a subscript on a scalar, an array without a subscript,
  INITIAL in a procedure, a procedure with no statements, a forward call),
  and a counted loop over a local in `??AUTO`, which a store from another
  frame or the module's variables does not end (counting none would cost ED,
  PIP and 80un 17 to 20 bytes each).  Left for the release: bump the
  version, rerun the Verified checks on the release commit, merge, tag, push.
* Argument evaluation order differs from Intel V3.1 where the last argument
  changes a variable passed next-to-last; the language leaves it undefined
  (9800268B 4.5.1), so it is documented, not changed.
* The workaround for um80 0.3.50's operator words (names.fix_symbols) is
  harmless and can go once uplm80 requires um80 0.3.51 or later.  Nothing
  requires it yet: pyproject.toml names only `upeepz80>=0.2.6`, and um80 is
  installed on its own (0.4.0 and 0.4.1 are checked with 0.3.51 and 0.3.52).
  0.4.1 drops the other, the `?` EQU for upeepz80 0.2.5's dead-store rule.

### um80_and_friends (0.3.52)

* `--dri` does not yet follow MAC in three things (CHANGELOG Known issues): a
  two-character string's byte order (`DW 'AB'` is 41 42 in MAC), `IF` truth (MAC
  tests bit 0 only), and a `MACLIB` library (MAC assembles none of its code).
  Also: `MACLIB NAME` should read `NAME.LIB`; `LOCAL` after a `!`; `END START`
  with START defined later; a label on an `ORG` line is placed at the old
  address (MAC: the new one), so mpm2's `genmod.py` still refuses that form.
* The documented choice that a `!` after a macro call's arguments is M80's
  quote accounts for all remaining silent differences from MAC in the fuzzers.
* Pre-existing M80 differences: `-1 SHR 8`, M80's signed division, `.XCREF` /
  `.CREF`.  Packaging: the sdist omits CHANGELOG.md; `package-data` lists a
  `py.typed` that does not exist.

### upeepz80

* Known issues in its CHANGELOG, none produced by uplm80: a tail call in code
  only address arithmetic reaches; a `pop` of the return address where the
  entry height is unknown; a public `jp` table entered by offset turned into
  `jr`s.

### mpm2

* `verify_dri.py` does not cover ABORT.RSP, DUMP.PRL or BNKBDOS.SPR (all match
  DRI's).  GENHEX and GENMOD differ from DRI's by 64 and 102 bytes (not
  investigated).  The .RSP program lengths are larger than DRI's (0050H/0095H
  against 0044H); only bytes DRI's DATA/INITIAL lists define are claimed.
* `run_tests.sh src` carries on after a failed build.  `tools/v21/where.py`
  hardcodes `/Users/wohl/src/mpm2`.  The MPMLDR.PLM override skips the serial
  check even under `--dri-exact`.  Some overrides drop DRI's trailing ^Z or
  change only a `$title`.
* LOAD.PRL (built from UTIL3/LOAD.PLM; DRI shipped LOAD.COM) addresses page
  zero absolutely, so it is right only in a segment based at 0000H — which
  every segment `gensys.sh` generates is.
* BNKBDOS has no V2.0 source; `--version 2.0` uses the V2.1 banked BDOS, on
  purpose (see `docs/mpm2_v21.md`).
* SDIR built through ISX (romwbw_emu) differs from DRI's in 526 uninitialised
  bytes, because DRI built on a 62K CP/M; RomWBW cannot provide that TPA.

### romwbw_emu

* `romwbw-batch` and `romwbw-plm80` are not in the .deb/.rpm packages and need
  cpmemu's `cpm_disk.py`; batch mode needs a CP/M 2.2 CCP and uses A:, B: and
  user 0 only.  A SYSCONF-set autoboot countdown runs about three times too fast
  (it stalled before 1.48).

### Housekeeping

* `uplm80/todo.txt`: `../uplox`'s `examples/plm_subset.uplox` still forbids a
  line break in a string; only uplox's own tests use it.
* `uplm80/mpm.sys` is an untracked local file, left as it was.
