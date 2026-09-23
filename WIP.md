# Work in progress — 2026-09-23

Saved from a session that started with a uplm80 code-generation bug report and
ended with MP/M II V2.1 reconstructed from Digital Research's binaries.  All
four repositories are clean, everything is committed and every branch is
pushed; **no branch has been merged to main.**

## Repository state

Everything is on a feature branch.  To pick up on another machine, clone or
fetch and check out the branch named here.

| Repository         | Branch                            | Ahead of `origin/main` |
|--------------------|-----------------------------------|------------------------|
| `uplm80`           | `fix/mpm-page-zero-relocation`    | 9 commits              |
| `um80_and_friends` | `fix/prl-origin-and-page-zero`    | 5 commits              |
| `upeepz80`         | `fix/dead-store-scope`            | 1 commit               |
| `mpm2`             | `fix/prl-relocation`              | 10 commits             |

```bash
for r in uplm80 um80_and_friends upeepz80 mpm2; do git -C ~/src/$r fetch --all; done
git -C ~/src/uplm80           checkout fix/mpm-page-zero-relocation
git -C ~/src/um80_and_friends checkout fix/prl-origin-and-page-zero
git -C ~/src/upeepz80         checkout fix/dead-store-scope
git -C ~/src/mpm2             checkout fix/prl-relocation
```

### What the toolchain needs on a fresh machine

On this machine `uplm80`, `um80`/`ul80`/`ud80` and `upeepz80` are editable
installs under Homebrew's Python 3.14, with the console scripts in
`~/Library/Python/3.14/bin`:

```
/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip list
  um80==0.3.46      -> ~/src/um80_and_friends/um80
  uplm80==0.3.5     -> ~/src/uplm80/uplm80
  upeepz80==0.2.3   -> ~/src/upeepz80/upeepz80
  uplox==3.3.1      (not editable; the parser-generator floor uplm80 declares)
```

so a fresh machine needs, in each of the three checkouts,

```bash
/opt/homebrew/bin/python3.14 -m pip install --user --break-system-packages -e .
```

plus `uplox>=3.3.1`.  Beware that the plain `python3` shim resolves to a
*different* site-packages that has an older non-editable `um80` and no
`upeepz80` at all — the CLI tools are what matter, and they carry the right
shebang.  `mpm2/tools/v21/where.py` sidesteps this by putting
`~/src/um80_and_friends` on `sys.path` itself.

`mpm2` additionally wants `cpmemu`'s `cpm_disk.py`, which
`scripts/build_hd1k.sh` looks for at `~/src/cpmemu/util/cpm_disk.py` (override
with `CPM_DISK=`), and the C++ emulator, which `scripts/build_asm.sh` builds —
see `mpm2/README.md`.

`uplm80` is at 0.3.5; `mpm2`'s CHANGELOG has an `[Unreleased]` section on top
of 0.3.6 covering the V2.1 work, so it wants a version bump before release.

## What was finished

### Compiler and toolchain fixes (earlier in the session)

`uplm80`, `um80_and_friends` and `upeepz80` carry a run of code-generation and
linker fixes found by building all of MP/M II from source and comparing the
result against DRI's binaries command by command.  Each has a regression test
that fails when its fix is reverted.  These are described in the individual
commit messages and CHANGELOGs; nothing is outstanding on them.

### MP/M II V2.1, recovered from the binaries

DRI published sources for V2.0 only.  `mpm2src.zip` also ships the **V2.0
binaries** the sources were cut from (`mpm2src/CONTROL`), and `mpm_ii.zip` the
V2.1 distribution (`mpm2dist`), which is what makes the difference recoverable
*and* checkable.  Both releases now build from one tree behind `IFDEF MPM21`
(assembler) and `$if MPM21` (PL/M):

```bash
cd ~/src/mpm2
./scripts/build_all.sh --tree=src --version=2.1     # or 2.0, the default
python3 tools/verify_dri.py                         # checks both against DRI
```

`verify_dri.py` reports `identical to DRI` for XDOS.SPR, BNKXDOS.SPR,
RESBDOS.SPR and TMP.SPR in **both** releases — program image and the relocation
bits that describe it.  (It compares nothing past the end of the program: DRI's
linker left stale bytes in the bitmap tail that no assembler can reproduce.)
The V2.1 system boots and reports `MP/M II V2.1 / Copyright (C) 1982, Digital
Research`; the V2.0 build is bit-for-bit what it was before the work started,
and `./scripts/run_tests.sh all` passes.

The full write-up, with the binary evidence behind every change, is
**`mpm2/docs/mpm2_v21.md`**.  Read that first when picking this up.

Two build switches were added for checking against DRI: `--serial dri` builds
in the serial from DRI's master instead of the `654321` placeholder, and
`--dri-exact` additionally drops the local fixes this repository carries on top
of DRI's code (at present only the 25 bytes of stack-pointer save/restore in
`TMPSUB.ASM`, guarded by `IFNDEF DRIEXACT`).

## What is still open

### 1. Four transients identified but not reconstructed

Offsets and evidence are in `docs/mpm2_v21.md`; none of these can be checked
byte for byte in any case, because `uplm80` is not DRI's PL/M-80.

* **GENSYS.COM** — the most valuable one.  8704 bytes in V2.0, 9472 in V2.1, so
  it is a recompile, not a patch, and a diff cannot recover it.  Its strings
  name what was added:

  ```
  *** Error Maximum Exceeded - 7 Assumed ***
  Enable Compatibility Attributes $
  ```

  `Enable Compatibility Attributes` is the missing half of the CLI change that
  *was* reconstructed: it is the GENSYS question whose answer lands in
  `system$data(96)`, which the patch area's `cliattr` tests before copying the
  command FCB's f1'..f4' bits into `pd(1dh)`.  **Until `GENSYS.PLM` is brought
  up to V2.1, a system generated here leaves that byte zero and the new
  attribute handling never fires.**  Source is `mpm2src/MPMLDR/GENSYS.PLM`.

* **PIP.PRL** — 64 bytes in seven places.  Two are legible: at `06E7`,
  `LDA 243AH`/`LXI H,2262H` became `LDA 2262H`/`LXI H,243AH`, i.e. `a = a or b`
  became `b = b or a`; at `0B12`, four calls are jumped over and the twelve
  bytes they occupied become a routine that zeroes `2270H` and `22CCH`.
  Whatever calls that routine is in the two unanalysed regions, `1FE1-1FF8` and
  `202E-203D`.  Source is `mpm2src/UTIL6`.

* **SDIR.PRL** — two bytes.  The PRL header's minimum-buffer field goes
  `0000`→`1000` (V2.1 asks MP/M for 4K more than its image), and at `23C1`
  `LHLD 3BADH` becomes `LHLD 3BB1H` inside
  `if mem16(3BB3H) >= mem16(3BADH) + 46`.  Three pointers sit at `3BAD`, `3BB1`
  and `3BB3`; the bounds check was reading the wrong one.  Locating them needs
  DRI's LOCATE map, which we do not have — the eight `UTIL7` modules were
  searched by hand without success.

* **SPOOL.BRS** — 37 bytes, the banked resident half of the spooler
  (`mpm2src/UTIL2/SPBRS.PLM`).  Blocked on item 2 below: this repository does
  not build `.BRS` files at all.

### 2. RSP/BRS build structure is wrong (pre-existing, not a V2.1 issue)

`build.py` builds `SCHED.RSP`, `SPOOL.RSP` and `MPMSTAT.RSP` from two modules
each (`*BRS.PLM` + `*RSP.PLM`).  DRI's `SCHED.SUB` builds:

* `*.RSP` from `*RSP.PLM` **alone**, and
* `*.BRS` — a separate banked-RSP file this repository never produces — from
  `*BRS.PLM` plus `BRSPBI.ASM` and `PLM80.LIB`.

Because the two are merged, the process descriptor does not land at the start
of the image where MP/M expects it: in our `SCHED.RSP` it ends up at offset
`0x5B2`.  `SCHED` reports *"Resident portion of scheduler is not in memory"* on
both releases because of this.  Fixing it means splitting those three targets
and adding `.BRS` as an output type.

### 3. BNKBDOS has no V2.0 source (deliberate non-goal)

`mpm2src/BNKBDOS/BNKBDOS.ASM` builds the **V2.1** `BNKBDOS.SPR` byte for byte —
DRI shipped the newer banked BDOS in the source release.  So `--version 2.0`
builds a V2.0 nucleus with the V2.1 banked BDOS, as this repository always has.
Going the other way would mean reconstructing the *older* code across 26
regions, including 271 bytes of patch routines that moved wholesale from
`05xx-08xx` to `22xx` — undoing bug fixes rather than recovering sources.  Not
done on purpose.

### 4. Housekeeping

* No branch merged to main (four repositories); all four are pushed.
* `mpm2/CHANGELOG.md` has an `[Unreleased]` section wanting a version bump.
* `uplm80/todo.txt` records one deliberate non-item: `../uplox`'s
  `examples/plm_subset.uplox` still declares a string grammar that forbids a
  line break, which real PL/M source uses.  Only uplox's own tests consume it;
  `plm_full.uplox`/`plm_pre.uplox` were fixed in uplox 3.3.1 and are what this
  repository vendors.

## Analysis tooling

`mpm2/tools/verify_dri.py` is part of the build and is the thing to run.  The
six scripts in `mpm2/tools/v21/` are the research tools the reconstruction was
done with; they are committed but are not part of the build, and
`tools/v21/README.md` repeats what is below.

| Script        | Use |
|---------------|-----|
| `spr.py A B`  | Diff two `.SPR`/`.PRL` images: header, differing runs, which bytes are flagged for relocation, and which bitmap bits differ. |
| `where.py T [off...]` | Map a linked-image offset back to `(module, source line, text)` for target `T` (`XDOS`, `BNKXDOS`, `RESBDOS`, `BNKBDOS`, `TMP`).  Assembles each module with `um80 -l`, gets the per-module CSEG/DSEG bases from `um80.ul80.Linker`, and tracks `cseg`/`dseg` per listing row. |
| `syms.py T [addr...]` | Full linked symbol table, via `um80 -g` + `ul80 -S`.  Answers "what is at `2081H`". |
| `annot.py T A B` | The workhorse: `spr.py` + `where.py`, i.e. a diff of two images with every run annotated with the source lines it covers. |
| `disasm.py f s e` | Disassemble one address range of a raw image at its real address, via `ud80`.  Needed because `ud80` stops at the first `RET` and dumps the rest as `DB`, so each routine must be disassembled from its own entry point. |
| `rawdiff.py A B [gap]` | Plain byte diff with ASCII, for files with no SPR header. |

Two environment notes for these:

* `V21=<dir>` sets where listings and `.rel` files are written (default
  `/tmp/v21`).
* `PRISTINE=1` makes `where.py` ignore `src/overrides` and use the untouched
  `mpm2_external` sources.  Wanted for `TMP`, whose override carries 25 bytes
  of local fix that shift every offset after `00CD`.  It does *not* work for
  `XDOS`: `mpm2_external`'s `MPM.ASM` will not assemble without the 6-character
  symbol aliases the `DATAPG.ASM` override adds (`Undefined symbol 'nmb$lst'`).
  For `XDOS` leave it unset — with no `-D MPM21` the overrides assemble to the
  V2.0 layout, which matches DRI's V2.0 image exactly.
* `ds` reserves space without emitting listing bytes, so a run that falls
  inside one cannot be mapped.  The only place that bites is `pdtbl` entry 0's
  `ds 36` in `DATAPG.ASM`; the `db 0ffh` in front of it identifies the spot.

Typical use when picking up PIP:

```bash
cd ~/src/mpm2
python3 tools/v21/rawdiff.py mpm2_external/mpm2src/CONTROL/PIP.PRL \
                             mpm2_external/mpm2dist/PIP.PRL
python3 tools/v21/disasm.py mpm2_external/mpm2dist/PIP.PRL 1fe1 2040
```

## Reference points

* `mpm2/docs/mpm2_v21.md` — the V2.1 write-up: where each reference binary
  tree is, what changed in each module, and what is outstanding.
* `mpm2_external/mpm2src/UTIL8/SYSDAT.LIT` — the system data page layout.  It
  is what decoded the patches: offset 3 `sys$call$stks`, 80–95 the system call
  user stacks, 96 *unassigned in V2.0* and the new compatibility-attributes
  flag in V2.1, 123 `system$drive`, 197 `nmb$printers`, 252 the MP/M data page
  address.
* Binary trees: `mpm2src/CONTROL` and `mpm2src/NUCLEUS` and `mpm2src/UTIL*` are
  **V2.0**; `mpm2dist`, `disk1_files` and `bin/dri` are **V2.1**.  `CONTROL` is
  serialized (`00 14 01 00 00 01`), `NUCLEUS`/`UTIL*` are not (`654321`), and
  the V2.1 serial is `00 14 01 00 09 2f`.
