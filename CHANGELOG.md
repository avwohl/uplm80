# Changelog

Notable changes to uplm80. Releases before 0.3.2 are described on the
[GitHub releases page](https://github.com/avwohl/uplm80/releases).

## 0.4.1 — unreleased

The Known issues of 0.3.7 and 0.4.0, each settled by the Programming
Manual (9800268B) and, where it leaves room, by what Intel's PL/M-80
V3.1 does with the same source: its listing, its diagnostics, and the
code it generates, linked with DRI's `X0100` and run.

### Fixed

- `tests/run_tests.sh`'s `test_byte_conditions` expected an IF and a DO
  WHILE to test for non-zero, as uplm80 did before 0.3.5, and failed.
  They test the least significant bit (5.1.2): 128, 10, 2 and 256 are
  false. The expected output is now what the program prints compiled by
  Intel's PL/M-80 V3.1, and by uplm80; all 22 programs pass.

## 0.4.0 — 2026-09-25

Procedures are now called the way Intel's PL/M-80 calls them, so code
uplm80 compiles links, unmodified, with assembly written for PL/M-80 -
Digital Research's `X0100.ASM` (`mon1 equ 0005h`), MP/M II's `LDMONX.ASM`
(`ldmon1 equ 0d06h`) and `BRSPBI.ASM` - and with objects PL/M-80
compiled. Up to 0.3.x no uplm80 module did: each of the three conventions
it had differed from Intel's, and every program linked with DRI's
interface modules needed a shim that turned one into the other.

The convention is PL/M-80 V3.1's as its own output shows it: DRI's
`PIP.PRL`, which PL/M-80 compiled, has `MOVE: PROCEDURE (S, D, N)` at
0AACH,

    LXI H,244EH / MOV M,E        ; N, the last argument, from E
    DCX H / MOV M,B / DCX H / MOV M,C    ; D, the one before, from BC
    DCX H / POP D                ; the return address
    POP B / MOV M,B / DCX H / MOV M,C    ; S, pushed by the caller
    PUSH D

and V3.1's listings of test modules, and byte-for-byte rebuilds of DRI's
utilities with Intel's compiler, agree.

### Incompatible: calling convention

**Rebuild every module.** A module built by 0.3.x does not work with one
built by 0.4.0, and nothing at link time says so: they export and import
the same names.

| Arguments | Where they are at the `call` |
|---|---|
| 0 | nothing |
| 1 | a1 in BC (C for a BYTE parameter) |
| 2 | a1 in BC (C), a2 in DE (E) |
| n ≥ 3 | a1 … a(n−2) pushed left to right, one word each; a(n−1) in BC (C); an in DE (E) |

- **The callee takes the pushed words off the stack.** The caller no
  longer pops anything after a call. At entry `[SP]` is the return
  address, `[SP+2]` a(n−2), and so on to `[SP+2(n−2)]`, a1.
- A BYTE argument in a register is in C or E, and B or D is undefined; a
  pushed BYTE is the low byte of its word.
- A BYTE result is in A and an ADDRESS one in HL, as before.
- A call keeps SP, IX and IY, and nothing else: A, the flags, BC, DE and
  HL are destroyed.
- This is so for every procedure - PUBLIC, EXTERNAL, nested, REENTRANT -
  but one exception. A procedure with one parameter that nothing outside
  the compile can reach - not PUBLIC, EXTERNAL or REENTRANT, and its
  address never taken - still takes it in A (a BYTE) or HL (an ADDRESS),
  where its body wants it. No other module, no assembly and no `CALL`
  through an address can tell.
- A `CALL` through an address places the arguments the same way, each
  widened to ADDRESS, with the address in HL, and calls `??jphl` (`jp
  (hl)`), which replaces `??jpde`.
- `MON1(f, a)` and `MON2(f, a)` with a constant `f` are still compiled as
  the BDOS call itself, `ld de,a / ld c,f / call 5`, and the argument is
  converted to the parameter's type like any other: a BYTE passed to
  MON1's ADDRESS parameter is now `ld e,a / ld d,0`, where D was left as it
  happened to be. PL/M-80 does the same (MP/M II's MPMLDR at 03D1H: `LHLD
  char / MVI H,0 / XCHG / MVI C,2`), and a function that reads DE whole,
  such as MP/M II's 141 (delay), needs it.
- Two new errors. A direct call must pass as many arguments as the
  procedure has parameters, as PL/M-80 V3.1 requires (153 and 154): with
  the callee removing the pushed words, a call with too many or too few
  would return with the stack moved. 0.3.x dropped extra arguments to a
  procedure private to its module without a word. A `CALL` through an
  address is not checked (8.2.1). And an INTERRUPT procedure may not have
  parameters (8.1.6).

      invalid number of arguments in call of P2, too few: 1 for 2 parameters
      IH: an INTERRUPT procedure may not have parameters (8.1.6)

- **upeepz80 0.2.6 or later is required.** 0.2.5 turned `push … / call p /
  ret` into `push … / jp p`, and p then took its return address for its
  first argument. At `-O1` and up the compiler now stops with an error
  naming both versions if upeepz80's is below 0.2.6, unless that
  upeepz80 keeps the `call` of such a routine: a development tree with the
  fix, still numbered 0.2.5, does, and the release 0.2.5 does not.

      upeepz80 0.2.5 is too old: uplm80 needs upeepz80 0.2.6 or later (…)

What 0.3.x did: a procedure private to its module had all but its last
argument written straight into its own storage by the caller and the last
in A or HL; a PUBLIC, EXTERNAL or REENTRANT one had every argument pushed,
left to right and each widened to 16 bits, and popped by the caller after
the call; a `CALL` through an address passed only one, except to a PUBLIC
or REENTRANT procedure.

An assembly routine written for 0.3.x changes like this:

| A 0.3.x assembly routine… | …becomes in 0.4.0 |
|---|---|
| read its one argument at SP+2; the caller popped it | reads BC (C for a BYTE); pops nothing |
| read two at SP+4 and SP+2 | reads BC and DE |
| read n ≥ 3 at SP+2n … SP+2 | pops the return address, pops the n−2 stacked words (last pushed first), takes BC = a(n−1) and DE = an, and puts the return address back |
| returned with its arguments still pushed | returns with the stacked words removed |

For example, the BDOS interface a CP/M program links with. MON1, MON2,
MON2A and MON3 are equates now, as in DRI's `X0100.ASM`, since a call
already has the function in C and the argument in DE:

```asm
; 0.3.x                                 ; 0.4.0
MON1:   ld      hl,2                    MON1    equ     5
        add     hl,sp                   MON2    equ     5
        ld      e,(hl)                  MON2A   equ     5
        inc     hl                      MON3    equ     5
        ld      d,(hl)                          public  MON1,MON2,MON2A,MON3
        inc     hl
        ld      c,(hl)
        jp      5
```

and a routine of three arguments, `cap3(a address, b byte, c address)`:

```asm
; 0.3.x: a, b, c pushed; the caller pops them
CAP3:   ld      hl,2
        add     hl,sp
        ld      e,(hl)          ; c
        inc     hl
        ld      d,(hl)
        ld      (VC),de
        inc     hl
        ld      a,(hl)          ; b
        ld      (VB),a
        inc     hl
        inc     hl
        ld      e,(hl)          ; a
        inc     hl
        ld      d,(hl)
        ld      (VA),de
        ret

; 0.4.0: a pushed, b in C, c in DE; CAP3 takes a off the stack
CAP3:   ld      (VC),de
        ld      a,c
        ld      (VB),a
        pop     hl              ; the return address
        ex      (sp),hl         ; a, and the return address back on top
        ld      (VA),hl
        ret
```

Assembly that calls a PL/M procedure does the same from the other side:
it pushes the first arguments, loads the last two into BC and DE, and
leaves the stack alone after the call.

### Changed

- **Smaller code.** Over MP/M II's PL/M (UTIL2 to UTIL7 and MPMLDR, 39
  modules, SDIR's eight included) and 80un's two programs, at `-O2` with
  upeepz80 0.2.6, the code is 111,219 bytes against 0.3.7's 113,528
  (−2,309), and no module is larger. A pushed argument costs one `push`
  where 0.3.x stored it into the callee's slot, the entry stores it once,
  and the A/HL exception saves the `ld c,a` or `ld b,h / ld c,l` every
  call of a one-parameter procedure would otherwise need (about 1,300
  bytes of it). The data is 36,526 bytes against 36,531: nothing is stored
  before a call, so the call graph no longer keeps a callee's frame apart
  from the procedures its arguments call.
- A procedure's entry stores its arguments (the last from DE, the one
  before from BC, the pushed ones popped). With one or two parameters it
  leaves BC and DE as they came, so DRI's CP/M 1.x `MON1: PROCEDURE (F,
  A); … GO TO BDOS; END MON1;` passes them on to the BDOS.
- A REENTRANT procedure pushes the arguments that came in BC and DE under
  its return address, which is the frame 0.3.x had, and every exit takes
  all of them off the stack, keeping A and HL.
- A GOTO out of a procedure can now abandon, besides the return addresses
  of the calls it leaves, the words pushed for a call whose arguments were
  being evaluated: the first arguments, and BC, kept round the last one.
  `call p3(1, 2, f)`, where `f` ends in `goto again`, leaves two. The
  label at the outer level of the main program that such a GOTO reaches
  sets SP again, as it does since 0.3.7, and that takes them off too: 1000
  such GOTOs run in `-m bare`'s 64-byte stack, out of an argument of a
  direct call, of a CALL through an address and of a REENTRANT
  procedure's callee (`tests/test_goto_stack.py`). A GOTO to a label in a
  DO block of the main program, which draws a warning, leaves them, as
  Intel's PL/M-80 does.
- `STACKPTR` in a call's arguments reads SP with the arguments pushed for
  the call so far, as in PL/M-80's code. After `sp0 = stackptr`, `call
  p5(1, 2, 3, 4, stackptr - sp0)` passes 0FFFAH, three words down, and
  `call p2(4, stackptr - sp0)` passes 0, as PL/M-80 V3.1's code does;
  0.3.7 passed 0 to both, p5 and p2 being private. BC, the next-to-last
  argument, is saved round the last one's code only where that may write
  B or C: round a call of a procedure, `??mul16`, `??div16`, `??mod16` or
  `??inp`, and not of `??subde`, which leaves BC as it is. The two
  compilers still differ in two cases. V3.1 loads a next-to-last argument
  that is a constant or a variable into BC after the last one's code, and
  so saves nothing round it: with `one` = 1, `call p2(4, (stackptr - sp0)
  * one)` passes 0 there and 0FFFEH here. And it pushes a next-to-last
  argument it has computed while it evaluates the last: `call p2(sp0 -
  stackptr, stackptr - sp0)` passes 0FFFEH there and 0 here.
- A call's arguments are evaluated from left to right. PL/M-80 V3.1 loads
  a next-to-last argument that is a variable into BC after it has
  evaluated the last one, so where the last argument's code changes that
  variable (`call p2(v, f)`, with `f` assigning `v`), its code passes the
  new value and uplm80's the old one. The language leaves this open: "PL/M
  does not guarantee the order of evaluation of operands", and where the
  order matters "the value of the expression is undefined" (9800268B,
  4.5.1).
- `??jphl` replaces `??jpde`.

### Fixed

- **A `CALL` through an address passes any number of arguments to any
  procedure.** 0.3.7 passed more than one only to a PUBLIC or REENTRANT
  procedure, and warned (Known issues, 0.3.7).
- A call with more arguments than a private procedure has parameters is
  an error; 0.3.x dropped the extra ones.

### Added

- `tests/test_calling_convention.py`: the sequences, at `-O0`, for 0 to 5
  arguments of either type in every order, conversions, BC kept while the
  last argument is evaluated where its code may write B or C (and what
  each runtime routine writes, read from its code), entries, the A/HL
  exception and what takes it away, REENTRANT entries and exits, calls
  through an address, MON1 and MON2, and the two errors.
- `tests/test_calling_convention_run.py`, at `-O0` to `-O3`: uplm80 code
  with assembly written to the convention, both ways; REENTRANT and
  recursive procedures across the boundary; calls through an address to
  every kind of procedure; PUBLIC procedures across separately compiled
  modules and in one multi-file compile; DRI's `MON1 … GO TO BDOS`;
  `STACKPTR` in a call's last argument, against what PL/M-80 V3.1's code
  passes; and Intel's own code, `tests/fixtures/plm80_v31`: a module
  compiled by PL/M-80 V3.1, its listing transcribed, with Intel's `MAIN`
  calling uplm80's procedures and uplm80's `MAIN` calling Intel's.
- `tests/test_upeepz80_version.py`: the upeepz80 floor, against
  pyproject.toml's, and what is refused and accepted below it.
- `tests/abi_fuzz.py`, a fuzz test of the convention across modules: random
  programs of two PL/M modules, each defining procedures of 0 to 6
  parameters that the other declares EXTERNAL, some private, some
  REENTRANT, some called through an address, and an assembly module that
  stands between some calls and the procedures they call, taking the
  arguments where the convention puts them and passing them on with
  garbage in the high byte of each BYTE. Bodies end in calls. Each program
  is built at `-O0`, which must leave SP where it found it, and at `-O1` to
  `-O3`, with its two modules at different levels and in one multi-file
  compile, and every build must print what `-O0`'s prints.
  `tests/test_abi_fuzz.py` runs three seeds, and checks that the test
  fails when `call x / ret` becomes `jp x`, as upeepz80 0.2.5 made it;
  `scripts/abifuzz.py --seeds N [--jobs J]` runs more.
- `tests/_toolchain.py`: `run_asm` links any number of assembly modules
  after the program.

### Verified

On 0.3.7 as released, with 0.3.7's last fixes underneath (the reload of SP
at a label a GOTO out of a procedure reaches, PUBLIC labels, the names of
an EXTERNAL procedure's parameters, -O3 and a subscripted scalar). The
release gate ran on it before BC stopped being saved round a call of
`??subde` (Changed, the `STACKPTR` entry); what was run again after that
says so.

- The test suite: 780 tests pass, after it. pylint rates the package
  9.72, as before.
- `scripts/abifuzz.py --seeds 500`, and after it `--seeds 200 --first
  9000`: every program prints, in each of its seven builds, what its
  `-O0` build prints, and leaves SP where it found it.
- `scripts/difftest.py --seeds 150 --first 14000`, and after it `--first
  30000`: all 150 programs as the model says at `-O0` to `-O3`.
  `scripts/namestest.py --seeds 100 --first 5000`, and with `--modules` 40
  seeds: all print what their scopes say.
- The 87 compiles of MP/M II's and 80un's PL/M (DRI's tree and mpm2's
  overrides, each in the mode `tools/build.py` uses, 80un's files one at
  a time and its two programs) at `-O2`: the same 77 assemble as with
  0.3.7, 165,312 bytes of code against 168,158, and each output sets SP
  again at the labels 0.3.7's does. The 41 programs of the size figures
  above compile and assemble at every level from `-O0` to `-O3`. Of the
  87, `??subde` changes MPMLDR, DRI's and mpm2's, at each level from
  `-O0` to `-O3` and nothing else: DISPLAYOS, whose two calls of
  PRINTITEMS end in a subtraction, loses a `push bc` and a `pop bc` round
  each, 4 bytes, and the same 77 assemble. The seven overrides that test
  MPM21 compile with `-D MPM21` at `-O2` as before it.
- 0.3.7's release-gate programs of GOTOs out of procedures - out of
  REENTRANT recursion, counted loops, calls through an address, nested
  procedures and another module's procedures, to PUBLIC labels and to a
  label in a DO block - print what 0.3.7 prints at `-O0` to `-O3`, in
  `-m bare` and CP/M mode, and the diagnostics among them say what 0.3.7
  says; but for a9 in `-m bare`, whose procedures read their locals before
  they set them, and which with 0.3.7 prints one thing at `-O0` and
  another at `-O1` to `-O3`. Run again after it: the same.
- 80un: `80un.com` and `80unbas.com` built at `-O0`, `-O2` and `-O3`
  write, on each of the 29 inputs in its tests, exactly the files and the
  output those built by 0.3.7 at the same level do. After `??subde` both
  compile to the same assembly as before at every level from `-O0` to
  `-O3`.
- MP/M II, V2.0 and V2.1, built from DRI's sources (`tools/build.py
  --tree=src`, 44 of 44 targets each): `scripts/run_tests.sh all` passes
  on both systems (DIR, STAT, STAT of a drive, the resident processes,
  HTTP, SFTP) and `run_tests.sh src` passes; the assembly-built files are
  DRI's byte for byte (`tools/verify_dri.py`; ASM.PRL but for 11 bytes no
  source sets); in a console session on each system the source-built DIR,
  SDIR, STAT, SHOW, PIP, ED, TYPE, ERA, REN, SET and USER, in 34 runs,
  print what DRI's own `.PRL`s print given the same commands; and
  GENSYS.COM built from source, given the same answers, makes the
  `MPM.SYS` and `SYSTEM.DAT` DRI's GENSYS.COM makes, but for the six
  bytes of the serial number at 0B5H. Of MP/M II, `??subde` changes only
  MPMLDR (above).
- `tests/run_tests.sh` prints what it prints with 0.3.7 for all 22
  programs (21 pass; `test_byte_conditions` fails with both).

Before those fixes were underneath:

- Every PL/M source of MP/M II (`mpm2src/*/*.PLM`), 80un and `sample_code`
  that 0.3.7 compiles compiles, without the argument-count or INTERRUPT
  error.

## 0.3.7 — 2026-09-25

Two sets of fixes, each checked against what Digital Research's own PL/M-80
does, and a program layout that is Intel's.

The first began with six defects found while rebuilding MP/M II's resident
system processes from DRI's sources, each worked around in those sources until
now; three more turned up in the same code, and an independent verification of
those fixes found the rest. DRI's binaries settle what is right: the
`SPOOL.RSP`, `SCHED.RSP` and `MPMSTAT.RSP` built from source now match DRI's in
layout and in every initialised byte, and each resident process's stack holds
what DRI's `SCHED.BRS` holds.

The second makes division, `MOD` and every other expression give what DRI's
PL/M-80 gives. Division and `MOD` agree with DRI's divide routine for every
operand pair, zero divisors included, on every path uplm80 computes them by:
the runtime routines, constant folding, strength reduction and DATA/INITIAL
values. `tests/test_divmod_dri.py` runs DRI's own routine on an 8080
interpreter as the reference and compares a compiled table of divisions
against it at `-O0` to `-O3`. Every expression now has the value and the type
Intel's PL/M-80 Programming Manual gives it, at every optimization level: a
constant up to 255 is a BYTE; `+ - AND OR XOR` of two BYTEs and `-` or `NOT`
of one wrap at eight bits; `* / MOD` are ADDRESS; a relation is the BYTE 0FFH
or 0. The optimizer folded constants as untyped 16-bit numbers that code
generation then typed by magnitude, so arithmetic next to a folded, propagated
or rewritten operand could change width. `uplm80/plm_types.py` states the
rules once, the optimizer and the code generator follow them, and
`tests/plm_difftest.py` checks random programs against a Python model of them.
DO loops now end the way DRI's end, when the increment carries out of the
index, and are laid out as DRI lays them out.

The layout puts a program's variables last, after its code, its constants,
the procedures' shared locals and the stack, as Intel's PL/M-80 does (see
Changed). DRI's programs use everything past their last variable as a
buffer, and MP/M II's SUBMIT and SPOOL now build from DRI's own sources and
behave as DRI's binaries do.

A procedure's variables are static, as the manual has them, its parameters
included, except where no program can tell: a local or a parameter shares
the procedures' overlaid storage only if every call assigns it before
anything reads it, and only where no pointer or overrun from another local
can tell it from DRI's layout (see Changed).

Every name means the declaration PL/M-80 gives it (`uplm80/names.py`):
labels, procedures and LITERALLYs of one name in different blocks, and in a
multi-file compile each module's private names, are kept apart; a GOTO that
PL/M-80 does not allow is an error; and a CALL through an address calls the
procedure. Every warning and error names the file and line it is about.

Each fix has a regression test that fails without it.

### Fixed

#### Procedure locals

- **A local without INITIAL did not keep its value from one call to the
  next.** PL/M-80 allocates a procedure's variables statically (Programming
  Manual, 8.1.7); uplm80 put every uninitialised local in `??AUTO`, overlaid
  on the locals of procedures never active at the same time, so a count, a
  first-time flag or a position kept in one came back as another procedure
  had left it: `tick: procedure address; declare (count, seen) address; if
  seen <> 1234h then do; seen = 1234h; count = 0; end; count = count + 1;
  return count; end tick;`, called in a loop with another procedure that
  has locals of its own, returned 1 every time, at every level (the
  integration verification's F6; 0.3.6 the same). Such a local is static
  now (see Changed). DRI's code counts on it: MP/M II's PIP, retrying a
  multi-file copy after an error, calls MULTCOPY again, which goes on from
  the directory entry and the count it keeps in NEXTDIR and NCOPIED
  (`if eretry = 0 then NEXTDIR, NCOPIED = 0`); both are static.
- **A pointer to a parameter, kept after the call, pointed into storage that
  other procedures overlay.** PL/M-80 allocates a procedure's parameters
  statically, like its other variables, but uplm80 kept every parameter in
  `??AUTO` because every call assigns it. After `sv: procedure (v); declare v
  byte; keep = .v; end sv;`, the sequence `call sv('P'); call other;` read
  0FFH (`other`'s locals) through `keep`, where DRI's code reads 'P' (0.3.6
  the same). A procedure whose address is taken also read a parameter of the
  procedure it is nested in out of `??AUTO` after that procedure had returned.
  A parameter is now static by the same rules as any other local (see
  Changed).
- **A pointer or an overrun from a local reached something other than what
  DRI's layout has there.** DRI lays out what a procedure's text declares in
  the order the text declares it (see Changed). uplm80 kept a frame's locals
  in `??AUTO` apart from three things: the parameters and locals of procedures
  nested among them, the variables of the procedure's DO blocks, and its
  INITIAL locals, which were static (0.3.6 the same). Examples:
  * with `declare arr (2) byte` before a nested `inner: procedure (v) byte`
    and `declare nxt byte` after it, `arr(2) = inner('I')` set `nxt` instead
    of `v`;
  * with `arr` INITIAL, `arr(n) = 'F'` did not reach the local declared after
    it, and neither did `.v + 1` with `v` INITIAL;
  * a read of `arr(2)` did not see what the previous call had left in `nxt`.

  These now reach what DRI's layout has there.
- **An INTERRUPT procedure's frame was put over the frames of the procedures
  it interrupts.** `??AUTO` overlays the frames of procedures that are never
  active at the same time, and the call graph decides which those are. Nothing
  calls an INTERRUPT procedure: the interrupt activates it (8.1.6), whatever
  is running. So `ih: procedure interrupt 7; declare (x, y) byte; ... call
  helper; end ih;` shared its frame with `work`, which `ih` may interrupt, and
  so did `helper` (0.3.6 the same). An INTERRUPT procedure and everything it
  calls are now active together with every procedure, and their frames overlap
  none.
- **A procedure called through an address, or called back from outside the
  module, shared its frame with its caller.** A `CALL` through an address
  (8.2.1) may call any procedure whose address is taken. An EXTERNAL procedure
  may call back any PUBLIC procedure, or any procedure whose address it was
  given. The call graph had none of those calls. `back` called an EXTERNAL
  `ext`, which called the PUBLIC `cb`; `cb`'s frame was `back`'s, and `back`'s
  local came back 0FFH at every level. The same happened to `through`, which
  passed the address of `cb2` to an EXTERNAL `icall` (0.3.6 the same). A
  procedure with a `CALL` through an address may now call every procedure
  whose address is taken, and an EXTERNAL procedure may call back every PUBLIC
  procedure and every procedure whose address is taken, whether the address is
  taken in the main program or in a procedure. A call of MON1 or MON2 with a
  constant function is a call of the BDOS, which calls nothing back. A `CALL`
  through an address now calls the procedure, too (see Names and labels):
  `CALL f` from `caller` to a procedure that fills its frame with 0FFH leaves
  `caller`'s locals as they were.
- **A parameter reached only through the address of the parameter before it
  lost its value at -O1 and up.** upeepz80 drops the store of the last
  argument at a procedure's entry when nothing else in the module names the
  parameter's storage. So `pq: procedure (a, b) byte; declare (a, b) byte; ...
  pp = .a + 1; return c;` (with `c` BASED on `pp`) returned 0 for `pq('A',
  'B')`. 0.3.6 did the same whenever no other procedure happened to name that
  slot in `??AUTO`. Each static parameter is now also named by an EQU, which
  costs nothing and which upeepz80 counts as a use. The rule itself belongs to
  upeepz80.
- **An embedded assignment was not stored when the procedure returned its
  target next.** RETURN took the value from A instead, even though the rest of
  the statement could change A and the target has to keep its value. MP/M II's
  LOAD reads a HEX file through `READCS: PROCEDURE BYTE; DECLARE B BYTE; CS =
  CS + (B := READBYTE); RETURN B;`, which returned CS + B, so LOAD built from
  source stopped with INVERTED LOAD ADDRESS on the first record of any file.
  With `v` static, `old = v; old = (v := old + 1); return v;` returned 1 on
  every call (0.3.6 the same for both). The store is now always made. LOAD
  built from source now produces, from a test HEX file, the same .COM as DRI's
  LOAD.COM: under MP/M II, and in BARE mode under cpmemu at -O0 to -O3. ED's
  GETSOURCE is 6 bytes longer.
- **A counted loop left its index with a value PL/M-80 would not give it.** A
  BYTE `DO i = 0 TO n` whose body does not name `i` counts its passes in B. It
  gives `i` its final value up front only if something may read `i`, and it
  allows a RETURN from the body when `i` is the procedure's own. Two things
  went wrong (0.3.6 the same):
  * a static `i` keeps its value until the next call, which read the final
    value (10) where a RETURN in the third pass leaves 2;
  * with `declare arr (2) byte, i byte`, `arr(2)` read 0 after the loop
    instead of 10.

  A loop is no longer counted when its index can be reached without naming it,
  or when the index is static and the body can RETURN.

#### Division and MOD

- **`x MOD 0` was 0; PL/M-80 gives `x`.** DRI's PL/M-80 sends every `/` and
  `MOD` through one routine, module @P0029 of its PLM80.LIB (the same 31 bytes
  sit at 39B0H in MP/M II's SDIR.PRL, and in DIR, ED, STAT, SHOW, TOD, SCHED,
  MPMSTAT, GENSYS, LINK and LIB). It has no zero test: sixteen
  shift-and-subtract steps, in which a zero divisor always fits, so the
  quotient comes out 0FFFFH and the remainder is the dividend. `??div16`
  tested for zero and returned a remainder of 0. SDIR's `page$len` defaults to
  0 and UTIL7/DSH.PLM asks `cur$line mod page$len = 0`, so every SDIR built
  from source reprinted its heading before every line of output. `??div16` and
  `??mod16` now run the same steps without the test, and are smaller: 31 bytes
  for the pair (50 before), 22 for `??mod16` alone. There is no BYTE divide to
  match: DRI zero-extends BYTE operands into the same routine (SDIR loads one
  with `LHLD` / `MVI H,00H` before `CALL 39B0H`), so a quotient or remainder
  of two BYTEs is an ADDRESS.
- **`SHR(x, 7)` lost bit 15,** and with it `x / 128`, which strength reduction
  turns into that shift: 8000H / 128 came out 0 instead of 100H. The result of
  a shift right by 7 has nine bits.
- **The compile-time forms of `/` and `MOD` disagreed with the runtime.**
  * Constant folding left `c / 0` and `c MOD 0` to the runtime; they now fold
    to 0FFFFH and `c`.
  * `0 / x` was folded to 0, but `0 / 0` is 0FFFFH, so the rule is gone.
    `x MOD 1` and `0 MOD x` (both 0) dropped a procedure call in the operand
    they discard; they now keep it.
  * A strength-reduced quotient or remainder of a BYTE became a BYTE:
    `x MOD 8` turned into `x AND 7` and `x / 1` into `x`, so
    `(x MOD 8) + 0FFH` wrapped at eight bits and `(x MOD 8) - 1` did not
    borrow. They stay ADDRESS now (as `DOUBLE(x AND 7)`), and code generation
    reads the byte itself wherever only the low byte is used or the value is
    compared with a BYTE, so assignments, arguments, subscripts and
    comparisons compile exactly as before.
  * A constant expression in DATA or INITIAL that the optimizer had not
    folded (any of them at `-O0`, and `c / 0` or `c MOD 0` at every level)
    went to the assembler, which rejects `7/0`, and `MOD` was written as `+`.
    Such an expression is now evaluated the way PL/M-80 evaluates it, and an
    operator the assembler cannot evaluate is an error rather than a `+`.

#### Expression types

- **A BYTE compared with a constant above 255 was rejected,** where the
  manual (4.4) compares the two as unsigned numbers and DRI's compiler
  accepts it: `IF b < 256 THEN ...` stopped with "comparison BYTE < 256 is
  always true", and so did `b <> 257`, `b = 300` and `(b + 1) < 256` (the
  integration verification's F5; 0.3.6 the same). It compiles, the BYTE
  zero-extended to meet the ADDRESS constant, and the message is a warning.
  The warning is given for a constant on either side - `IF 300 > b` said
  nothing - and at every level: the optimizer, moving a constant to the
  right of `=` or `<>`, marked it as one it had derived, which the check
  lets pass. The differential test generates such comparisons now; it kept
  a constant above 255 on the left, where it was not checked.
- **A folded constant was typed by its size, not by PL/M-80's rules.**
  `(8 MOD 0FFH) + b` with `b = 0FFH` gave 7 at `-O1` and above: the remainder
  is the ADDRESS 8, and adding 0FFH carries into the high byte (107H). The
  same folded 7 made `(7 MOD 0) > 1000H` fail to compile as "comparison BYTE >
  4096 is always false". Folding is typed now; an ADDRESS constant below 256
  is carried as `DOUBLE(n)`, and a constant the optimizer derived is not held
  against the program by the impossible-comparison check, nor is a relation
  between two constants.
- **`NOT` and unary `-` of a BYTE worked in sixteen bits,** even at `-O0`:
  `(NOT 7) MOD w` divided 0FFF8H. `NOT 7` is the BYTE 0F8H, `3 - 5` the BYTE
  0FEH and `-1` the BYTE 0FFH. The levels disagreed as well: `-O1` and up
  folded `-1` to 0FFFFH, while at `-O0` it was negated in HL and the
  BYTE-index path took A for the index, so `buf(-1)` was BUF-1 at one level
  and BUF+255 at another. It is `buf(255)` at every level now (see Changed),
  and the index path reads the register the index actually came out in.
- **BYTE `x * 2` became the BYTE add `x + x`,** so 200 * 2 was 144 at `-O2`.
  The product is an ADDRESS. Other rewrites kept the value but not the type
  and are fixed the same way: `b AND 0FFFFH`, `b XOR 0FFFFH`, `b + DOUBLE(0)`
  and `x * 1` are ADDRESS, `(b + 100) + 300` is not reassociated across the
  change of width, and a copy `w = b` is not propagated.
- **An element of a BYTE array member was typed ADDRESS.** Code for
  `s.m(i)` compared and combined it in sixteen bits, and a test against 0
  loaded the byte into A and then tested HL: SDIR (DSH.PLM:290, 294) printed
  every file's update and create stamps whether it had them or not.
- **`??mul16` took the carry of its own add into the product,** so any
  product that overflowed sixteen bits was wrong: 81H * 511 gave 817FH.
- **A nested subscript's release restored an outer claim's spill of DE,**
  and `aw(aw(aw(i) AND 7) AND 7)` added a stale DE in place of the base.
- **A BYTE value generated into A was read from HL** by a store through a
  BASED ADDRESS, by `MEMORY(b)` as a value and as a target, and by `MOVE`
  with a constant count and `TIME`; `OUTPUT(p) = b` replaced it with L.
  BYTE `1 - x` was `x XOR 1`, right only for 0 and 1.
- **`LOW` could take A for its operand after an embedded assignment.**
  `(b := w) + LOW(LAST(a))` added the low byte of `w` for 7, and
  `LOW((b := w) + 5)` - or `- 1`, `* 2`, `SHL`, `NOT` or unary minus in
  place of `+ 5` - took L of `w`: the flag that lets `LOW((b := w))` skip
  reloading A outlived the assignment. It is now used only when LOW's
  operand is the embedded assignment itself, as in ED's
  `LOW((N := SHR(NDEST,SECTSHF) - 1))`.
- **Shift and rotate counts of 129 or more shifted nothing:** the count loop
  tested the sign. Counts are unsigned BYTEs now, and a constant rotate is
  unrolled.
- **`LENGTH` and `LAST` were typed BYTE but generated into HL,** and an
  element of an untyped DATA array, `DECLARE hex DATA ('0123')`, was typed
  ADDRESS while being loaded as a BYTE. Code that goes by the type found
  nothing where it looked: `ab(LAST(sa))` read `ab(0FFH)`, and
  `hex(i) + 0FFH` added whatever HL held. A constant BYTE is now loaded
  straight into the register it is wanted in.
- **A 16-bit relation in an ADDRESS array's subscript** compared with a DE
  it had already restored for the subscript: `aw(w = 5)` read `aw(0)`.
- **An embedded assignment to a BASED BYTE** lost its value to the pointer
  the store loads into HL: `(x := w) + 1` added 1 to the pointer.
- **BYTE `0 PLUS x` and `0 MINUS x` cleared the carry they read:** `ld a,0`
  is `xor a` after the peephole. A constant left operand is now added to the
  other, by loads that leave the flags alone.

#### Calls

- **A BYTE argument to an ADDRESS parameter was stored as one byte,** unless
  it was the last argument, leaving the parameter's high byte from the call
  before. SHOW's and STAT's `pdecimal(getuser, 100, true)` printed the user
  number with the high byte of the previous number printed.
- **An argument that calls the procedure again,** `f(1, f(2, 3))`, stored over
  the arguments before it, which go into the procedure's own storage before
  the call. They now wait on the stack until it has run.
- **A `CALL` with five or more stacked arguments** (a REENTRANT, PUBLIC or
  EXTERNAL procedure) set SP to SP + HL instead of SP + 2n.
- **A call in the main program whose later argument calls a procedure** lost
  the earlier arguments: `CALL p2(5, g(1, 2))` passed 1 for 5. The earlier
  arguments are already in `p2`'s own storage, and the storage allocator,
  which keeps `g` out of it for such a call in a procedure, never looked at
  the main program's calls.
- **An array or structure local to a REENTRANT procedure did not
  assemble,** nor did a BASED variable whose pointer is one of its locals
  or parameters. The frame had room for them, but the code addressed them
  by a label that was never defined (`ld (V),a`, "Undefined symbol 'V'").
  An array or structure is now reached as IX plus its offset, and placed
  after the procedure's scalars, which `(ix+d)` has to reach, and a pointer
  is loaded with `ld l,(ix+d) / ld h,(ix+d+1)`; a scalar more than 128
  bytes into the frame is an error rather than a displacement the
  assembler rejects.
- **A local declared in a DO block of a REENTRANT procedure was below
  SP.** The frame was sized from the procedure's own declarations before
  the body was read, so the block's locals were outside it, and the next
  push or call wrote over them: a recursive `q` keeping `c` in a DO block
  returned 8 for 10. The frame is sized after the body now. (Both found by
  the integration verification or next to what it found; 0.3.6 did the
  same.)

#### DO loops

- **A BYTE loop to 255 ran no times.** DRI's PL/M-80 tests the limit before
  each pass and leaves the loop when the increment carries out of the index
  (GENSYS.COM's code for `do j = common$base to 0ffh` ends `INR A / JNZ top`;
  LOAD.COM's `BY 128` loop, `DAD D / JNC top`), so `DO j = 0 TO 255` runs 256
  times and leaves the index 0 (manual, 5.1.4). uplm80 tested
  `index < bound + 1`, and bound + 1 is 0, with a constant bound and with a
  variable one that is 255 - and at -O3 constant propagation turns the
  variable form into the constant one. BYTE and ADDRESS loops now end on the
  carry, so `DO w = 0FFF0H TO 0FFFFH` stops too; a step that would wrap stops
  the loop; and the limit, start and step are converted to the index's type
  (`DO b = 0 TO 300` runs to 44, and `BY -1` is `BY 0FFH`: PL/M-80 has no
  downward step). MPMLDR's `GENSYS.PLM` has two `TO 0FFH` loops that ran no
  times from 0. The wrap ends the loop even when the body put the index
  there: `DO i = 0 TO 0FEH` whose body adds 3 to `i` never stopped, since a
  constant limit and step that fit the index were taken to mean it could
  not wrap. The carry is now tested on every pass, as DRI's code tests it
  whatever the limit (LOAD.COM's `DO I = 0 TO 127` ends `INR M / JNZ`), so
  however the index got past the limit, the step that carries ends the loop.
- **A counted `DO` loop ignored its index being written or read in a
  target.** A loop whose body does not use its index counts in B, and never
  stores the index while it runs. `s(i).x = 0` and `i = n` in the body went
  unseen, so the index was never stored or its assignment ignored:
  `SCBRS.PLM` cleared one entry of its table four times, and `PIP.PLM` and
  `ED.PLM` (FILLSOURCE) kept reading past end of file. A variable bound of
  255 skipped the loop instead of running it 256 times: a count of 0 is 256
  passes to DJNZ, and the loop tested for 0 first.
- **A counted `DO` loop's index was stale to everything but its body.** An
  inner `DO` over the same index left it where it was, so
  `DO i = 0 TO 9; ...; DO i = 0 TO 9; END; END;` ran the outer body ten times
  instead of once; the code after the loop, a procedure the body calls, and
  the caller after a `RETURN` all read a stale index; and the bound was
  evaluated once, where PL/M-80 evaluates it at every test. The index now gets
  its final value before the loop starts, and a loop is counted only when
  nothing else can see its index - no procedure the body calls names it, no
  store reaches it through its address, no caller reads it after a `RETURN`
  from the body - and nothing can change its bound, by name or through a
  pointer (a bound BASED on `buf(3)` that the body sets through `buf(3)` was
  counted from its first value), and the bound does not read the index:
  `DO k = 0 TO k + 5` runs until `k + 5` wraps, 251 times, and was counted
  from `k`'s value before the loop. The final value is left out only where
  nothing can read the index afterwards.
- **A `RETURN` inside a counted `DO` loop left the count on the stack.** The
  count is pushed around the body, and the RET took it for its return
  address. A `RETURN` now pops what the loops around it pushed. Live in
  `SPBRS.PLM` (the spooler's stop request at the end of a line) and 80un's
  `lzh.plm` (a failed write).
- **A REENTRANT procedure's BYTE `DO` loop did not assemble** at `-O1` and
  above: upeepz80 turned the increment of `(ix+n)` into `ld hl,ix+n`. The
  index is now incremented in place, `inc (ix+n)`.
- **A BASED ADDRESS loop index stepped by 1 never wrapped:** the `inc hl`
  sets no flags, and the zero test that stands for it came after the store,
  which leaves the pointer in HL.

#### -O3

- **-O3 changed what programs do.** The inliner kept a `RETURN` that was not
  the last statement, which then returned from the caller - `ED.PLM`'s
  BACKSPACE, `PIP.PLM`'s and `SHOW.PLM`'s user checks and `TOD.PLM`'s
  COMPUTE$MONTH among them - defined a label once per call site, captured the
  caller's locals, and could inline another procedure of the same name. It now
  inlines only a small parameterless untyped procedure with no `RETURN` but a
  last one, no label, `GOTO` or declaration, whose names mean the same at the
  call as where it is declared. Nothing learned before a call was forgotten
  after it: `GENSYS.PLM` lost a whole `IF` after `get$response(.accept)`, and
  `cnt = 0; rw = f; call ph(cnt)`, where `f` increments `cnt`, printed 0. A
  call now ends everything the optimizer knows about variables, no fact is
  used in an expression that makes one, and a store through a BASED variable,
  a subscript or a member ends everything too, common subexpressions
  included. `.x` was folded like a value, so `c = 1; CALL setv(.c)` passed
  the address 1; a copy was propagated for a BASED variable, so 80un's
  `read16` returned its high byte twice; what a loop body sets was taken to
  be known after the loop; and an unrolled loop left its index at the last
  value.
- **-O3 unrolled a loop whose body could change its index,** through a call
  or a store through a pointer: `DO i = 0 TO 1; CALL bump; ...` with `bump`
  setting `i` ran twice. A loop is unrolled now only if its index is a
  plain variable, its body calls nothing, and it stores through no pointer
  when a pointer may reach the index. An index BASED on a pointer moves when
  the body sets the pointer, and one AT another variable changes when the
  body sets that variable by name: `DO x = 2 TO 0FEH BY 128` with `x BASED
  p` and `p = .buf(2)` in the body stored 2 and 82H through the moved
  pointer, and `DO y = 254 TO 0FEH BY 0FFH; g = 10; END` with `y AT (.g)`
  left `y` 0FDH, not 9 (0.3.6 was wrong at `-O3` too).
- **`-O3` turned `SIZE(b)` into `SIZE(5)`** after `b = 5`, which does not
  compile; SIZE, LENGTH and LAST name a variable, not its value.
- **`-O3` took a subscripted scalar's name for its value.** PL/M-80 lets a
  scalar be subscripted: `x(1)` is the byte after `x`, and DRI's code reads a
  following local that way. The optimizer put in place of the name the
  constant or the variable last assigned to `x`. After `x = 'x'`, `return
  x(1)` became a CALL through address 78H with the argument 1, and the
  program ran into page zero; after `w = 1234H`, `w(1)` was a CALL through
  1234H. After `x = y`, `x(1)` read the byte after `y`. 0.3.6 did this for a
  numeric constant and for some copies; once constants were typed, a
  character constant hit it too. The name of a subscripted variable is a
  place, like the target of an assignment, and is now left as it is.
- **`-O3` rejected a constant it had moved right of a relation:**
  `w = 'AB' <> b` was "comparison BYTE <> 16706 is always true" at `-O3`
  only. A constant the optimizer moves is marked as derived, like one it
  folds.

#### DATA, INITIAL and AT

- **`DATA` ignored member types and did not reserve the variable.** The 0.3.6
  fix that placed a STRUCTURE's `INITIAL` values member by member never
  reached `DATA`, which still emitted one byte per value and stopped at the
  last one; an array shorter than its dimension lost the rest too. `DATA` is
  `INITIAL` stored with the code (PL/M-80 Programming Manual, 6.2.9), and the
  two now share one emitter. `SPRSP.PLM`, `SCRSP.PLM` and `MSRSP.PLM` are the
  resident halves of the spooler, scheduler and status processes, nothing but a
  process descriptor and queues found by offset: SPOOL.RSP's queues were at 11H
  and 1CH, and are at 36H and 0CEH as in DRI's binary.
- **A string in a STRUCTURE initialiser filled one member.** A string fills one
  BYTE scalar per character and one ADDRESS scalar per two, and the width of
  each later value was taken from its position in the list instead. The queue
  control blocks in `SCRSP.PLM` and `MSRSP.PLM` got `msglen` and `nmbmsgs` as
  bytes.
- **A `LITERALLY` list stood for its first element inside `INITIAL`/`DATA`.**
  A special case added to match an earlier uplm80 cut the body at its first
  comma. MP/M's `restarts`, the case its comment cited, is nineteen `0C7C7H`
  words, and `SCBRS.PLM`, `MSBRS.PLM` and `SPBRS.PLM` build each process's
  stack as `initial (restarts,.entry)` with SP at `.stk+38`: the entry point
  landed in the second word and SP pointed at a zero.
- **An expression in `DATA` was always a word.** At -O0, where nothing folds it
  first, `x (4) BYTE DATA (68H+80H, k+1, 6)` took six bytes and moved
  everything after it, and a unary minus was not accepted: `SET.PLM` did not
  compile at -O0. An expression now fills its scalar at the scalar's width,
  evaluated as PL/M-80 evaluates a restricted expression: as plain 16-bit
  numbers, `/` and `MOD` as DRI's divide gives them.
- **`.(constant list)` in `DATA` or `INITIAL` was laid out in place.** It is the
  location of the constants (manual, 4.1.3), as it is in an expression, so
  `msgs (3) ADDRESS DATA (.('one$'), ...)` held characters, not pointers.
  `.'text'` in a list was not accepted at all.
- **Every name in a factored declaration got the first one's `AT` or values.**
  `DECLARE (A, B, C) BYTE AT (.BUF)` put all three at BUF, and
  `DECLARE (COUNTER, LIMIT, INCR) ADDRESS INITIAL (0, 1024, 2)` gave each name
  the whole list. Neither MP/M II nor 80un writes either form, but Intel's
  LINK does: `tests/link1a.plm`, from Mark Ogden's reconstruction, declares
  `(s, e) ADDRESS AT(.inRecord$p)` to reach the record pointer and the one
  after it, and `e = s + inRecord.len + 2` wrote over the record pointer.
- **`AT (.external +/- constant)` compiled to `EQU $`.** The catch-all at the
  end of `_emit_at_decl` was still there. `MSPL.PLM`'s
  `spool$msg (1) byte at (.tbuff-1)` sat on the queue control block after it.
  An AT address is now resolved as the manual defines it - a constant, or a
  location plus or minus constants - and anything else is an error. A location
  reference to a variable declared further down, which DRI's compiler
  accepted, is measured from that declaration instead of taken to be a byte.
- **A negative constant offset was written as `+65535`.** `.tbuff(-1)` in an
  `AT` was `TBUFF+65535`, whose relocation um80 0.3.48 drops, and a subscript
  of a variable AT an external was `EXT+c1+c2`, which it assembles as
  `EXT+c2`. Offsets from a symbol are written signed, and folded into one
  where the symbol is external. (An `AT`, `DATA` or `INITIAL` value is a
  restricted expression, evaluated as plain 16-bit numbers, so there `-1` is
  0FFFFH.)
- **An `AT` naming something declared further down was placed at 0.** An EQU
  is evaluated where it stands, and um80 0.3.48 takes a symbol it has not
  reached as zero; AT variables were defined in the data segment ahead of
  later declarations and of `??AUTO`. `SUB.PLM`'s `rbuff` at
  `.minimum$buffer` made SUBMIT build its command file at address 0. AT
  definitions now come after all storage.
- **An `AT` naming a later `AT` variable, or a later `EXTERNAL`, was still
  wrong.** `a1 AT (.b1 + 1)` above `b1 AT (.buf(2))` became `A1 EQU B1+1`
  ahead of B1's own EQU, so A1 was 0001H; the same for a later variable AT
  `.MEMORY`. A variable AT a later EXTERNAL was not aliased to it, and um80
  0.3.48 assembles `@A+2` with `@A EQU E1+1` as `E1+2`. A later declaration's
  own `AT` is now resolved down to its root, and a later EXTERNAL is known to
  be one. A circle of ATs is an error.
- **`AT` with `INITIAL` or `DATA` dropped the values without a word.** It is
  now an error.
- **In CP/M mode a module's DATA ran as code.** Module-level DATA was placed
  at the head of the program, where DRI's programs keep the jump they enter
  themselves by, but CP/M mode starts at 100H with its own entry code, and
  the DATA came first: `DECLARE t (2) BYTE DATA (0C9H, 42H)` returned to
  CP/M before the first statement. In CP/M mode it now follows the code;
  BARE and MP/M modes keep DRI's layout.

- **An element of an array BASED on a structure member, with a variable
  BYTE subscript, took its pointer from the start of the structure:**
  with `token BASED pcb.tok (4) BYTE`, `token(i)` read through
  `pcb.state`. 0.3.6 fixed the other paths for a variable BASED on a member
  (SDIR's `token BASED pcb.token$adr (12) byte`, which it subscripts only
  with constants) and missed this one.

#### Names and labels

- **A GOTO from a nested procedure to a label of the procedure around it
  did not assemble:** with `out:` in M1 and, nested in M1, `bail:
  procedure; goto out; end bail;`, the output jumped to `@M1$BAIL$OUT`,
  which nothing defines, at every level (0.3.6 assembled it at `-O3` only,
  by inlining BAIL). PL/M-80 does not allow it: "the label in the GOTO must
  be the label of a statement in the outermost level of the main program
  module" (Programming Manual 9800268B, 5.3.2; 8.1.3 and 9.3 say the same).
  It is a compile error now that cites the rule, and so are a GOTO into a
  block it is not in, a GOTO to a name that is not a label, and a label
  defined twice in one block. A GOTO out of a procedure to the main
  program's outer level, one within a procedure, and one out of a DO block
  to a label of a block around it work as before. A GOTO out of a
  procedure to a label in a DO block of the main program breaks the same
  rule, since the outer level is the module's exclusive extent (10.1), but
  Intel's PL/M-80 V3.1 compiles it without an error, to a plain `JMP` that
  leaves the procedure's return address on the stack, and 0.3.6 compiled
  it. It draws a warning that cites the rule and compiles to the same
  plain jump.
- **A GOTO in a procedure went to the main program's label of the same
  name,** not the procedure's own, silently: code generation found GOTO
  targets through its symbol table, which has the main program's labels and
  not a procedure's. `p: procedure; ... goto done; ... done: call pc('a');
  end p;` with a `done:` in the main program jumped out of P.
- **A GOTO from a procedure to a label at the outer level of the main
  program left the procedure's return addresses on the stack.** DRI's
  PL/M-80 reloads SP at such a label: MP/M II's PIP ends its ERROR
  procedure with `GO TO RETRY`, and in DRI's `PIP.PRL`, `RETRY:` begins
  with the same `LXI SP` as the program's entry. uplm80 only jumped, so each
  such GOTO leaked what the calls on the way had pushed. With `bail:
  procedure; n = n + 1; goto again; end bail;` and, in the main program,
  `again: if n < 1000 then call bail;`, `-m bare`'s 64-byte stack ran into
  the program after about a dozen rounds. MP/M II's PIP printed garbage and
  dropped back to the CLI after 86 errors in one interactive session. Such
  a label now reloads SP, as DRI's code does, and so does a PUBLIC label,
  which a procedure in another module can reach: `ld sp,??STACK` in bare
  and MP/M mode, and `ld hl,(6) / ld sp,hl` in CP/M mode. A label that
  only the main program jumps to is left as it was, as DRI leaves it. Every
  MP/M II program built from PL/M now loads SP as often as DRI's binary of
  it does: ED at five labels, GENSYS at two, and PIP, TOD, SCHED, PRLCOM
  and MPMLDR at one. Found by the release gate; 0.3.6 did the same.
- **A label in each of two DO blocks** of the main program or of one
  procedure was "multiply defined", both `LP:` or both `@P$LP:`, though
  each DO block has labels of its own (9.3). The second is `LP?2` now (no
  PL/M-80 identifier has a `?`). A `DECLARE l LABEL` in a procedure, and
  the address of a procedure's label, `.there` in a statement or in a DATA
  or INITIAL list, named the bare label.
- **A PUBLIC label that labels no statement at the outer level of the main
  program was left for the linker to find.** PL/M-80 requires a PUBLIC
  label to be attached to an executable statement there (9.3). With
  `declare again label public;` at module level and `again:` in a DO
  block, which is a label of the block's own, uplm80 emitted `public
  AGAIN` with nothing defining it: the module compiled alone did not link,
  and compiled with the module that jumps to it did not assemble
  ("Undefined symbol 'AGAIN'"). 0.3.6 took the DO block's label for the
  PUBLIC one. It is a compile error now that cites the rule and says where
  the other label is, as Intel's PL/M-80 V3.1 rejects the program (ERROR
  #172, INVALID LABEL: UNDEFINED). Found by the release gate.
- **`.show`, of a procedure nested in another, did not assemble:** it named
  `SHOW`, where the procedure is `@OUTER$SHOW`. A DATA or INITIAL list and
  an AT found it already; an expression does now.
- **`CALL q` through an ADDRESS variable (8.2.1) did not call the
  procedure** whose address q holds: it was `call Q`, which ran the bytes
  of Q itself, and `CALL s.p` was `jp (hl)` with no return address, so the
  procedure returned to its caller's caller. The address goes to DE and a
  new runtime routine, `??jpde`, jumps there from a CALL. The arguments are
  pushed, as a PUBLIC or REENTRANT procedure takes them, and the last is
  also left in HL and A, where a procedure private to its module takes its
  only one; more than one argument draws a warning, since such a procedure
  takes the others in storage the call cannot reach. The procedure called
  does not share its frame in `??AUTO` with the caller (see Procedure
  locals).
- **Procedures of one name in different blocks.** Code generation files a
  procedure under its enclosing procedures' names and its own, `P$Q`, and
  finds a name as a procedure nested in an enclosing procedure before
  anything else. So a procedure Q in each of two DO blocks of P was `@P$Q`
  twice ("multiply defined"); a procedure N in a DO block of the main
  program and a module variable N were both `N`; and, silently, a variable
  X of a procedure nested in P read as P's procedure X, and a variable V of
  P used outside the DO block of P that declares a procedure V as that
  procedure. With `do; declare q byte; ... do; declare e byte; q:
  procedure; ... end q; call q; end; ... end;` the procedure was generated
  under the variable's label. Such a procedure is renamed now (`Q?2`), and
  so is a procedure or a label that a DO block declares under the name of
  a parameter of the procedure around it, which is `@P$X` when it is
  static.
- **A DO block's variables are named after the block's number, `@B1$X`,**
  which is also procedure B1's X: the block's X took B1's place in
  `??AUTO` and the program printed B1's value for it. The number skips any
  that names a procedure.
- **LITERALLYs of one name.** One in each of two procedures, with
  different values, was `K EQU 1` and `K EQU 2`, and one named like a
  module variable or a main-program label met its label: "Symbol 'K'
  multiply defined". The later is `K?2` now. And code generation kept every
  LITERALLY in one table, so once procedure PA had declared `K LITERALLY
  '1'` a variable K of procedure PB read as 1 (silently, at `-O0` to
  `-O2`); a name is a LITERALLY there only where the declaration of it in
  scope is one.
- **A name declared twice in one block** was generated twice, and did not
  assemble; it is an error now. A LITERALLY declared again with the same
  text is let be (an $INCLUDE file and the file including it often both
  declare TRUE).
- **A declaration hides the built-in of its name when it is called or
  subscripted:** with `DECLARE size (4) BYTE`, `size(2)` was SIZE(2),
  "SIZE() needs a declared variable", and `high(1)` of an array HIGH was
  the high byte of 1. A variable already hid a condition flag read without
  parentheses, as MP/M II's STAT needs.
- **Names the assembler reads as something else.** um80 takes `A`, `HL` ...
  as registers and `EQ NE LT LE GT GE SHL SHR NUL` as operators: `call A`
  is "Register 'A' used as value", and `call EQ` calls 0FFFFH, `ld hl,SHL`
  loads 0, without a word. A procedure or label so named is `@A`, `@EQ`
  now, as a variable named like a register already was, and so is a
  variable named like an operator. It takes `Z NZ NC PO PE P` after `jp`
  for a condition ("JP with condition requires address"; the peephole
  makes `call p / ret` into `jp P`): the jump is written `jp 0+P`. And it
  takes a symbol whose letters end in one of its word operators (`MOD SHL
  SHR AND OR XOR NOT EQ NE LT LE GT GE HIGH LOW NUL TYPE`), followed by +
  or -, for that operator: `ld hl,TYPE+2` is TYPE(+2) and loads 0, and
  `X1EQ+2` or a procedure's `@Q$NUL+1` does not parse. Such an offset is
  written `2+TYPE`. (MP/M II's PIP has a variable TYPE, which it never
  offsets; its output is unchanged.)
- **`-O3` inlined the wrong one of two procedures of one name:** with a
  procedure NUL at module level and another in a DO block of P, a call of
  NUL in P after the block inlined the block's. And it inlined a procedure
  reading a variable Q into a block with a label Q, which its model of
  scope did not have (nor a scope for DO CASE): `qqqq` printed `q***`.
- **`INPUT(p)` and `OUTPUT(p) = v` with a port that is not a constant did
  not assemble:** they called `??inp` and `??outp`, which the runtime
  library did not have.

#### Diagnostics

- **A warning or an error named no file, or the wrong line.** `uplm80
  e.plm` printed `<unknown>:3:8: warning: comparison BYTE = 300 is always
  false`. The parser numbers the lines of the file with its $INCLUDE files
  spliced in and the lines a conditional skips taken out, so a warning in
  an included file came out at a line of the including one, and a syntax
  error there as `p.plm:1:1: error: ... at line 5, column 5`. Every
  diagnostic names the file and line it is about now - the included file
  for text from an $INCLUDE - and an error code generation raised without
  a location is placed at the statement or declaration it was generating.
  The warning for `IF 2` had no location at all, and an assignment was
  placed at its `=`.

#### Multi-file compiles

- **An EXTERNAL procedure that none of the files defined had no `extrn`.**
  `uplm80 A.PLM B.PLM` left out every EXTERNAL procedure declaration, on the
  grounds that one of the other files defines it, so a call to one that
  belongs to a third module did not assemble ("Undefined symbol"); A.PLM
  compiled alone had the `extrn`. Found by the integration verification;
  0.3.6 did the same.
- **Two modules with a private name in common did not assemble:** `uplm80
  a.plm b.plm`, with a procedure HELPER in each, gave "Symbol 'HELPER'
  multiply defined", and, since locals are static, `@HELPER$N` too; two
  module variables, DATA tables, labels or LITERALLYs of one name met the
  same way, and a module could use another's private name, which compiled
  alone it could not reach. PL/M-80 modules have separate name spaces for
  everything not PUBLIC or EXTERNAL (Programming Manual, 10.4). Each
  module's private names are now qualified with its name - `LIB?HELPER`,
  `@LIB?HELPER$N`; a module without a name goes by its file's - so each
  behaves as if compiled alone and linked. PUBLIC and EXTERNAL names bind
  across the modules as before, and a PUBLIC procedure can still be called
  from another module without an EXTERNAL declaration, as 80un's modules
  do. Using another module's private name is an error that says what to
  do.
- **A GOTO to a PUBLIC label of the main program, from a module that
  declares it EXTERNAL** (the third GOTO 9.3 allows), was "JR to 'AGAIN':
  its target is the external symbol AGAIN" at `-O1` and up: the EXTERNAL
  declaration was an EXTRN of a label the same assembly defines. No EXTRN
  is emitted for a name one of the modules makes PUBLIC.
- **A PUBLIC procedure's static parameter was `@KEEPIT$V?2`, not
  `@KEEPIT$V`,** when a module before the one that defines KEEPIT declared
  it EXTERNAL: the EXTERNAL declaration's parameter took the name, as if it
  were static in that module. The program ran as it should; only the names
  in the assembly differed from those of the module compiled alone, and now
  they do not. Found by the release gate (0.3.6 kept every parameter in
  `??AUTO`, with no name of its own).
- **Only the first of two modules with statements at their outer level was
  compiled;** the second's statements were dropped without a word. It is
  an error now: only the main program module may have them.

### Changed

- **A procedure's local shares `??AUTO` only if every call assigns it before
  anything reads it, and only where a program cannot tell the difference from
  DRI's layout.** `??AUTO` overlays the storage of procedures that are never
  active at the same time. What goes there now is the parameters and locals
  that a definite-assignment analysis (`uplm80/local_storage.py`) shows are
  assigned, on every path from the procedure's entry, before anything reads
  them. The analysis follows GOTOs and every form of DO. A call of a procedure
  nested in this one that names the local counts as a read at the call. A use
  of a BASED variable reads its base. A read through a subscript that is not a
  constant reads every local declared after the array. An array or structure
  counts as assigned once every element has been assigned through constant
  subscripts. A subscript is a constant when it folds to one by PL/M-80's
  rules (`a(1+1)`, `a(-1)`, which is `a(255)`, and `a(LAST(a))`), and every
  level now decides this the same way; before, -O0 took `a(1+1)` for a
  variable subscript and -O1 and up for a constant.

  Every other local and parameter is static, `@proc$name` among the variables:
  * one that may be read before it is assigned;
  * one whose address is taken (`.x`, in a statement, an AT or an INITIAL);
  * one reached outside its bounds: a scalar with any subscript but `(0)`, a
    constant subscript past the end, or a one-element array with any subscript
    but a constant 0;
  * one named by a procedure nested in its own procedure whose address is
    taken, or by anything that procedure calls, since it can run when the
    local's procedure is not active;
  * one declared with such a local in a factored declaration (6.2.4).

  DRI lays out what a procedure's text declares in the order the text declares
  it. The parameters come first: ERA's PRINT$FILE declares `k` before its
  parameter `fcbp`, and DRI's ERA.PRL has `fcbp` at 067AH and `k` at 067CH. A
  nested procedure's parameters and locals, and a DO block's variables, come
  where the text has them: SUBMIT's FILLRBUFF has `ssbp` at 0E7AH, the
  parameter of PUTRBUFF (declared next) at 0E7BH, and `reading` (declared
  after PUTRBUFF) at 0E7CH. That order is now kept wherever a program can
  tell:
  * everything declared after a local whose address is taken, or which is
    reached outside its bounds, is static too;
  * from the first array or structure subscripted by anything but a constant,
    a procedure's locals are either all static or all in `??AUTO`, and all
    static if any one of them is (an INITIAL one included). For example,
    `do i = 0 to n; a(i) = 'R'; end; return b(1);` with `declare a(2) byte,
    b(2) byte` has to reach `b`; the release gate's a4 printed 0 where 0.3.6
    printed 'R';
  * where a nested procedure with storage, or a DO block with variables, comes
    after either kind of local, what follows that local is static, and so is
    the nested procedure's storage.

  `??AUTO` also no longer keeps a slot for a declaration that never used one:
  an INITIAL, DATA, AT, BASED, PUBLIC, EXTERNAL or LABEL declaration in a
  procedure, and a REENTRANT procedure's parameters and locals, which are on
  its stack.

  In MP/M II (mpm2 at ef0a098, 87 compiles of which 77 assemble) and 80un,
  197 locals and parameters (4,687 bytes) are now static, the same at -O0,
  -O2 and -O3. The data comes to 45,095 bytes at each of those levels,
  against 46,754 before this change. The unused slots were 1,891 bytes. The
  static locals add back 216, since most of their bytes are buffers that also
  leave the frames that set the size of `??AUTO`, and the calls through an
  address and back from outside the module (see Fixed) 16. The code is the
  same except for two things: a parameter's store at a procedure's entry,
  which upeepz80 drops only when nothing else names its `??AUTO` address, and
  the embedded-assignment stores described under Fixed. Together they add 12
  bytes of code at -O0, 21 at -O2 and 15 at -O3.

- **Constants in expressions are typed as PL/M-80 types them, so some
  programs compute something else.** `w = -1` stores 00FFH, since the
  manual makes `-1` the BYTE `0 - 1` (write 0FFFFH for all ones), and for the
  same reason `buf(-1)` is `buf(255)`, not the element before `buf`
  (`buf(0FFFFH)` is); `NOT 0` is 0FFH. Comparing a BYTE with a constant from
  0FF00H up used to compare the low byte and is now the "always false/true"
  error, since the BYTE is zero-extended. An embedded assignment has the type
  of its right half (manual 4.6.3); BYTE `PLUS` and `MINUS` BYTE is a BYTE;
  `LENGTH` and `LAST` are BYTE when they fit; `CARRY` is 0FFH when set, as
  DRI's code has it; `SCL` and `SCR` have their pattern's type and rotate an
  ADDRESS in 17 bits; a two-character string is an ADDRESS constant, first
  character high.
- **`SHL` and `SHR` stay ADDRESS** even of a BYTE pattern, where the manual
  and DRI's compiler shift a BYTE in eight bits: programs written for uplm80
  rely on it (80un builds words with `lo + SHL(b, 8)`). A count of 0 leaves
  the pattern as it is; the manual leaves that undefined, and DRI's shift
  and rotate routines (SHOW.PRL's at 182BH to 1843H) have no zero test, so
  there a count of 0 shifts 256 times.
- **`DO` loops are laid out as DRI's are:** the limit is tested at the top,
  and the step jumps back only if it did not carry out, `jr nz` after an
  `INC`, `jr nc` after an `ADD`, so there is no jump to a test at the bottom
  and no separate wrap exit. A variable BYTE limit is compared in place,
  `ld hl,i / cp (hl)`. A limit, start or step that is a constant but not a
  literal (`LAST(x)`, `SIZE(x)`, `-1`) is used as one. With this layout and
  constants loaded straight into the register they are wanted in, MP/M II
  and 80un went from 225,929 bytes to 224,925 at `-O2`.

- **A program is laid out as Intel's PL/M-80 lays it out: the variables
  last.** The code and every constant - strings, `.(...)` lists, DATA
  declared in a procedure - are in the code segment (`cseg`); the data
  segment (`dseg`) holds `??AUTO`, then the stack of BARE and MP/M modes,
  then the variables in the order the source declares them, a procedure's
  static variables among them. ul80 puts every data segment after all the
  code, the runtime modules' included, so nothing follows a program's last
  variable, and `.MEMORY` is still the end of the whole program. DRI's
  programs use everything from their last variable up to MAXB, which works
  because DRI's layout puts nothing after it (MP/M II's `PIP.PRL` sets
  `LXI SP,2251H`: its stack is at 21EDH to 2250H and its variables start at
  2251H). `UTIL5/SUB.PLM` builds SUBMIT's command file at
  `.minimum$buffer`, and `UTIL5/MSPL.PLM` reads a file into `.dummy$buffer`;
  uplm80 put the strings, `??AUTO` and the stack after the last variable, so
  a command file of more than about 1.1K overwrote SUBMIT's messages, and
  SPOOL printed NULs for the first records of a file it spooled itself.
  In MP/M II and 80un the change only moves lines: every output holds the
  same instructions and data as before, in another order.
- **What PL/M-80 does not allow is an error,** where it compiled to
  something wrong or did not assemble: a GOTO out of a procedure except to
  the main program's outer level or an EXTERNAL label (to a DO block of the
  main program it is a warning, and compiles as Intel's PL/M-80 compiles
  it), a GOTO into a block or to a name that is not a label, a PUBLIC
  label that labels no statement at the main program's outer level, a name
  declared twice in one block,
  and, compiling several modules together, a second main program module or
  a name another module declares without PUBLIC.
- **Some names in the output are new.** In a multi-file compile a module's
  private names are qualified, `MODULE?NAME`; where two declarations would
  meet in one assembler name the later is `NAME?2`; and a procedure, label
  or variable named like a register or an um80 operator is `@NAME`, PUBLIC
  and EXTERNAL ones included, as a variable named like a register always
  was, so an assembly module that defines or uses one has to use that
  name. A static parameter is also named `?@proc$name`, by an EQU (see
  Fixed, Procedure locals). The new names change no single-file output of
  MP/M II and 80un, and the 80un multi-file programs' `.COM` files are byte
  for byte what they are without them.
- `docs/multi_file_compilation.md` said to call a PUBLIC procedure from
  another file without declaring it EXTERNAL, and that declaring it
  EXTERNAL would call it the wrong way; since a PUBLIC procedure takes its
  arguments on the stack, the EXTERNAL declaration is right, and it is what
  lets each module also be compiled alone.
- `CLAUDE.md` and the README's example of linking said to link a
  `runtime.rel`, and `CLAUDE.md`'s list of runtime routines left out
  `??jpde`, `??inp` and `??outp`. A module carries the runtime routines it
  uses at the end of its code; nothing is linked for them.
- **uplm80 requires upeepz80 0.2.5.** upeepz80 0.2.4 deleted register
  loads that were still needed, at `-O1` and above; the differential test
  found each, and each was in 0.3.6's output as well. `b, w = -(NOT b)` left
  `w` holding A's old value; after `b = 0FEH`, `w = LOW(LAST(big))`, with
  `big` 300 bytes long, got the low byte of the statement before instead of
  2BH (seed 1063 of `scripts/difftest.py`); `ld hl,0 / ld a,l / ld (sb),a /
  push hl / ld (w1),hl / pop hl / ld (w0),hl` lost its `ld hl,0`, since
  `ld (nn),hl` did not count as a read of HL; and `ld a,(ix+n) / inc a /
  ld (ix+n),a` became `ld hl,ix+n`, which is not a Z80 instruction. 0.2.5
  makes a rewrite only where nothing reads what it changes.

### Added

- **`tests/plm_difftest.py`**, a differential test: random programs over
  BYTE and ADDRESS variables, constants, every operator but `PLUS` and
  `MINUS` (whose carry-in depends on the code before them), built-ins, calls
  of procedures that change globals, BASED stores and DO loops, compiled at
  `-O0` to `-O3`, run under cpmemu and compared with a Python model of the
  manual's rules. The suite runs three programs; `scripts/difftest.py
  --seeds N` runs more.
- **`tests/names_difftest.py`**, a differential test of name resolution:
  random programs that declare a pool of ten names as variables,
  LITERALLYs, procedures and labels at every depth, with the output their
  scopes give, one module or three compiled together. The suite runs six;
  `scripts/namestest.py --seeds N [--modules]` runs more.

### Known issues

- **PLUS, MINUS, SCL and SCR after `+ 1` to `+ 4` or `- 1` to `- 4` of an
  ADDRESS** take a stale carry: those are `inc hl` and `dec hl`, which set
  none, so `(w - 1) MINUS z` with w = z = 0 gives 0FFFFH, not 0FFFEH. DRI's
  PL/M-80 increments with INX and INR too (PIP.PLM declares a variable
  `ONE = 1` so that `DEC(C1 + ONE)` gets an ADD and its carry), and the
  manual (12.1) warns that the flags cannot be relied on.
- **A REENTRANT procedure's parameter in a factored declaration with its
  locals,** `DECLARE (top, c) BYTE`, is taken for a local; declared on its own
  it is read from the stack as it should be.
- **A counted loop trusts that a pointer made from `.x` reaches only `x`, for
  a module-level `x`.** A BYTE `DO i = 0 TO n` whose body does not name `i`
  counts its passes in B. It is not used when anything can reach `i` another
  way: a procedure that names it, `.i` anywhere, `i` AT or BASED, and, for a
  procedure's local, an overrun of an array or a pointer from a local declared
  before it. For a module-level `i`, a pointer computed from the address of
  the variable declared before it still can reach it, and a store through that
  pointer does not end the loop. DRI's compiler never counts a loop.
- **An overrun or a pointer that runs backwards from a local, past a
  procedure's last local, or past a module-level variable** reaches what DRI's
  layout has there (the variable the text declares before or after it,
  possibly another procedure's) only where that variable is static. In
  `??AUTO` it reaches another frame, or nothing (0.3.6 the same). Keeping all
  of that in DRI's order would make every local static.
- **upeepz80 drops a store at a procedure's entry when nothing names the
  storage**, taking storage that nothing else names to be unreachable. uplm80
  names every static parameter with an EQU so that its store is kept. The rule
  itself should be fixed in upeepz80.
- **A LITERALLY's name declared again in an inner block** is a syntax error:
  the macro pass puts the LITERALLY's text in place of the name there too
  (`declare n literally '5'` and, in a procedure, `declare n byte` reads
  `declare 5 byte`), as it must for DRI's `mon1: procedure` with `mon1
  literally 'ldmon1'`.
- **A procedure named DOUBLE** is taken for the built-in where code generation
  folds constants: `double(30h)` is 30H however the procedure is written.
- A CALL through an address passes more than one argument only to a PUBLIC or
  REENTRANT procedure (a warning says so).
- **A name that is declared nowhere is not an error.** `y = nosuch + 1`
  compiles to `ld hl,(NOSUCH)`, and only um80 reports it, as an undefined
  symbol (0.3.6 the same).
- **`x()`, with empty parentheses, where x is a variable, is accepted
  without a diagnostic.** In an expression, `y = x() + 1` with x a BYTE
  compiles to a CALL through x's value (`ld a,(X) / ... / call ??jpde`);
  0.3.6 compiled it to `call X`. PL/M-80 has no empty argument list; `f()`
  of a procedure is taken as `f`.
- **`STACKPTR` read inside an expression can see a temporary the compiler
  pushed.** Where an operand evaluated before it is kept on the stack,
  STACKPTR reads 2 less than it does at the start of the statement. After
  `sp0 = stackptr`, `sp0 <> stackptr` is true at `-O0` to `-O2`; `-O3`
  evaluates `stackptr <> sp0` and `stackptr = sp0` in that order too, and
  all three find the two unequal. Intel's PL/M-80 V3.1 does the same, in
  other places: for `d = stackptr - sp0` it pushes a temporary and then
  reads SP. (0.3.6 folded all three comparisons at `-O3` to "equal".) The
  manual gives STACKPTR as the stack pointer register (11.2.3), not as it
  was when the statement began.
- **`.label`, the address of a label, is accepted in an expression.** The
  dot operator takes a variable or a procedure (Programming Manual, 4.1.3),
  and Intel's PL/M-80 V3.1 rejects `.label` in an expression (ERROR #158,
  INVALID DOT OPERAND, LABEL ILLEGAL), whether or not the label is declared
  LABEL; it accepts one in a DATA list. uplm80 compiles both to the label's
  address, with no diagnostic, as 0.3.6 did.
- **An INTERRUPT procedure nested in another procedure is accepted without
  a diagnostic.** PL/M-80 requires an INTERRUPT procedure to be at the outer
  level of the module; uplm80 compiles a nested one, and a local of the
  procedure around it that the interrupt reads may be in `??AUTO`, where
  another procedure's frame is (0.3.6 the same).

### Known issues — not this compiler

- **um80 0.3.50 reads a symbol whose letters end in one of its word
  operators, followed by + or -, as that operator** (`find_binary_addsub`
  takes the letters before the sign, `[A-Za-z]+$`, for a word operator even
  when they end a longer symbol), and `EQ`, `SHL`, `NUL` and the like on
  their own as operators, without an error: `ld hl,TYPE+2` loads 0, `call
  EQ` calls 0FFFFH. uplm80 renames or rewrites what it emits so as not to
  meet it; an assembly module written by hand can.
- **With `um80 -t` (PUBLIC and EXTERNAL names cut to six characters, as
  MACRO-80 does), ul80 links two PUBLIC names that agree in their first six
  characters as one** - PRINTCHAR and PRINTCRLF are both PRINTC - reporting
  "Multiply defined global" and linking anyway. uplm80's output keeps its
  names whole and is assembled without -t.

### Verified

- Every PL/M source in MP/M II (the 41 in DRI's tree and the 14 overrides,
  each in the mode `tools/build.py` uses) and in 80un (35 files one at a time,
  and `80un.com` and `80unbas.com` as their Makefile compiles them) compiles
  at -O0, -O2 and -O3, and the same 82 of the 92 outputs assemble with um80
  0.3.49 as with 0.3.6 (the rest are single modules of multi-module programs,
  and MSCMN.PLM, which is only ever included). At -O2 every output changes
  from 0.3.6, if only by the layout, and every change is one of the entries
  above: of 1,733 changed hunks (4,278 lines that only moved aside), 601 are
  the DO-loop layout and counted loops, 272 shift and rotate counts, 221
  constants loaded straight into the register wanted, 182 the `cseg`/`dseg`
  split, 151 BYTE operations kept in A, 136 the runtime routines, 91 ATs
  resolved to their root, 34 `SHR(x, 7)`, 23 DATA and INITIAL, 8 `CARRY`, 7
  a BYTE argument widened, 5 `jp`/`jr` distances and 2 the `extrn` below.
  The 82 come to 224,925 bytes, against 227,511 with 0.3.6, before the
  change to the allocation of locals (see Changed). The other fixes made
  after the integration verification (the entries that name it) change no
  output of MP/M II or 80un at -O0, -O2 or -O3 but the multi-file
  `80un.com` and `80unbas.com` compiles, which now declare `extrn MON1` and
  `extrn MON2`: 80un declares them EXTERNAL, and a CP/M-mode call goes to
  BDOS directly. `80un.com` extracts the same files, with the same console
  output, from all 17 sample archives as 0.3.6's, at -O2 and -O3, and
  `80unbas.com` detokenises `PALLOPS.BAS` the same.
- The allocation of locals (see Changed), measured with um80 0.3.50 and
  upeepz80's `fix/peephole-live-registers` against the commit before it:
  it changes 70 of the 82 outputs at each of -O0, -O2 and -O3, moving
  locals out of `??AUTO` into the variables, and with them the `??AUTO`
  offsets and the addresses of the variables that follow (the 82 go from
  224,744 bytes to 222,984 at -O2). `80un.com`
  and `80unbas.com` built from the new outputs, at -O2 and -O3, extract
  the same files with the same console output from all 17 sample archives,
  and detokenise `PALLOPS.BAS` and the four BASIC samples the same, as
  those built before it; GENSYS, one of the programs whose locals move,
  makes the same MPM.SYS and SYSTEM.DAT and prints the same, for V2.0 with
  the three sets of answers, at -O2. `scripts/difftest.py --seeds 200
  --first 9000`, with the generator as it was: all 200 programs as the
  model says at `-O0` to `-O3`; and with the generator that now compares a
  BYTE with a constant above 255 either way round, `--seeds 200 --first
  11000`: all 200.
- MP/M II built from source with this release - `tools/build.py` for V2.0
  and V2.1, 44 of 44 targets each - passes mpm2's `scripts/run_tests.sh all`
  on the V2.1 system and `scripts/run_tests.sh src`. SUBMIT and SPOOL were
  built from DRI's own `SUB.PLM` and `MSPL.PLM`, without the workarounds
  mpm2 carried for the old layout, and compared on the emulator, on the
  V2.0 system built from source, with DRI's V2.0 `SUBMIT.PRL` and
  `SPOOL.PRL`: SUBMIT runs two 300-line command files
  (3968 and 3328 bytes), a 250-line one of 15K, and one with parameters,
  printing exactly what DRI's does, where the release before the layout
  change printed its own messages over the 3968-byte file and ran none of
  it; SPOOL, printing files itself on a system without the spooler RSP,
  prints a 150-line file and a 7936-byte one as DRI's does, where before it
  printed 512 NULs in place of their first records. `stat usr:` now prints
  what DRI's STAT prints. GENSYS built from source, whose DATA now follows
  its code, prints what DRI's GENSYS prints under cpmemu, for V2.0 and
  V2.1 and with three sets of answers, and makes the same MPM.SYS and
  SYSTEM.DAT but for the six bytes of the serial number at 0B5H: the build
  leaves DRI's placeholder there, "654321", and the V2.0 GENSYS.COM in DRI's
  MPMLDR directory, which has the placeholder too, makes byte-identical
  files. Against DRI's serialised GENSYS - V2.0's in CONTROL, V2.1's on the
  distribution disk - the two files differ in those six bytes and nowhere
  else.
- The differential tests: `scripts/difftest.py --seeds 400 --first 1000`,
  all 400 random programs as the model says at `-O0` to `-O3` (seed 1063,
  which the second upeepz80 defect above broke, no longer meets it); and the
  integration verification's own generator, which covers DATA, INITIAL, AT,
  BASED, DO loops whose body moves the index or the bound, calls in
  arguments and module-level code: 497 programs at `-O0` to `-O3`, all as its
  model says. The independent verification of the release wrote a third,
  over typed expressions, calls that call back, nested and REENTRANT
  procedures, BASED, AT, structures, DATA, every form of DO, and programs
  of two modules built both separately and as one multi-file compile:
  2001 programs, 12,276 builds, whose only compiler defects were the two
  `-O3` unrolling defects above. Its seeds 1500 to 1599, 20480 to 20579 and
  30001 to 30100, taken again with this release, all run as its model says.
- The run tests compile with the checkout under test: they used to start the
  compiler with `python -P`, which found whatever uplm80 was installed. Every
  test that runs a program - the run tests, the differential test and the
  division oracle - now assembles, links and runs it with
  `tests/_toolchain.py`.
- MP/M II built from source with the fixes to procedure locals above
  (`build_all.sh --tree=src`, V2.0 and V2.1) passes mpm2's
  `scripts/run_tests.sh all` for both: DIR, STAT, STAT drive, the resident
  system processes, HTTP and SFTP.
- A 122-command session of DIR, STAT, SDIR, TYPE, PIP, SET, SHOW, ED,
  MPMSTAT and SCHED prints exactly what the same session prints with the
  tools built by 86d2720, apart from MPMSTAT's snapshot of which processes
  are delayed.
- `80un.com` and `80unbas.com` at -O2 and -O3 extract the same files from
  all 17 sample archives, and detokenise the BASIC samples the same, as
  86d2720's builds.
- The fixes to names and labels change none of that. Each of the 87
  compiles of the current MP/M II and 80un sources (77 of them assemble)
  gives the same `.mac` with them as without them, at -O0, -O2 and -O3,
  but for the two 80un multi-file programs, whose private names are
  qualified; those link to the same `80un.com` and `80unbas.com`, byte for
  byte.
- `scripts/difftest.py --seeds 200 --first 1000`: all 200 programs as the
  model says at -O0 to -O3. The release gate's generator of locals (static
  locals, nested readers, GOTOs, variable subscripts, pointers): 500
  programs at -O0 to -O3, all as its model says. `scripts/namestest.py`:
  1000 one-module and 480 three-module seeds before the fixes to locals
  were merged, and 300 and 100 after, all at -O0 to -O3, all print what
  their scopes say.
- The release gate's adversarial programs a1 to a14 print at -O0 to -O3
  what 86d2720's build printed, apart from a4, which prints `RST` at every
  level where 86d2720's printed `T` at -O0 and `ST` above, and a5 and a6,
  whose GOTO from a nested procedure to a label of its parent is a compile
  error now (86d2720's output did not assemble).
- The release gate's next round found one defect, `-O3` taking a subscripted
  scalar's name for its value (see Fixed, -O3). With the fix, its programs
  that subscript a scalar print at `-O3` what they print at -O0, and its
  other programs, b1 to b23, a1 to a14 and the multi-file sets, print at
  -O0 to -O3 what they printed without it. Each of the 87 compiles of MP/M
  II and 80un gives the same `.mac` with the fix as without it, at -O0, -O2
  and -O3. `scripts/difftest.py --seeds 200 --first 2000`: all 200 programs
  as the model says at -O0 to -O3; the gate's generator of locals, 100 seeds
  of each of its two versions: all 200 as its model says.
- The release gate's round after that found the GOTO out of a procedure
  that left its calls on the stack (see Fixed, Names and labels). With the
  fix, its program of sixty such GOTOs from two calls down prints sixty
  `e` and then `ad.` with `-m bare` at -O0 to -O3, where it stopped after
  11 to 18, and its program of twenty thousand prints `12.`, where it
  printed `.`. The MP/M II and 80un compiles change only by the new loads
  of SP, 52 bytes of code at each of -O0, -O2 and -O3, and GENSYS makes the
  same `MPM.SYS` for V2.0 and V2.1. MP/M II built from source for V2.0 and
  V2.1, 44 of 44 targets each, passes `run_tests.sh all` for both and
  `run_tests.sh src`, and `verify_dri.py` reports what it did. On the V2.0
  system built from source, one PIP session fed `t9.txt=nosuch.txt` 120
  times and then `con:=t1.txt` prints the 120 errors and types T1.TXT,
  exactly as DRI's `PIP.PRL` does. The 122-command session prints what it
  printed without the fix, apart from MPMSTAT's list of processes. The
  gate's other programs print what they printed before, at -O0 to -O3.
  `scripts/difftest.py --seeds 200 --first 6000`, the gate's generator of
  locals (100 seeds of each version) and `scripts/namestest.py` (100
  one-module and 40 three-module seeds): all as their models say.

## 0.3.6 — 2026-09-24

Found by building every PL/M program in Digital Research's MP/M II sources and
running each one next to DRI's own binary on the same disk, command by command,
until the two printed the same thing. Adds an MP/M runtime mode. Each fix has a
regression test that fails when it is reverted, and 80un's `80un.com` and
`80unbas.com` still rebuild byte-identical.

Requires upeepz80 0.2.4, whose dead-store elimination deleted the store in
`var = (a = b)`.

### Added

- **`-m mpm`, for MP/M II `.PRL`/`.RSP`/`.SPR` modules.** MP/M gives each
  process a memory segment and puts its page zero at the segment's base, so the
  BDOS entry, the stack-top pointer at 0006H and the warm-boot jump have to be
  relocated when the program loads. Only a resolved symbol reference reaches a
  `.PRL` relocation bitmap, so MP/M mode emits them as the externals `??BDOS`,
  `??MAXB` and `??BOOT` rather than literals (link with a module that defines
  them at 0005H, 0006H and 0000H, and with `ul80 --prl`, which relocates
  page-zero symbols). This is how DRI's PL/M-80 got the same effect:
  `PLM_WORK/X0100.ASM` and `X0200.ASM` publish the same names at different
  offsets and GENMOD diffed the two links. The names carry the compiler's `??`
  prefix because a PL/M identifier cannot contain `?`, and something collides
  otherwise: SDIR declares a variable called `bdos`. MP/M mode sets SP with the
  single three-byte `ld sp,??STACK` over a 512-byte stack in the image, because
  DRI's sources enter themselves by a jump to `.start-3`. CP/M and bare modes
  are unchanged.

### Fixed

- **A `PUBLIC` procedure took its arguments the way a private one does.** A
  procedure private to its module is called with the earlier arguments already
  written into its own storage and only the last in a register; a caller in
  another module cannot name that storage. A public procedure now takes all its
  arguments on the stack. SDIR's `pdecimal(v, prec, zerosup)` read two of its
  three arguments from slots nobody had written.
- **`AT(.MEMORY)` was not the end of the program.** It named a label at the
  end of the *module* — the middle of a multi-module program — and was emitted
  as an EQU, which reads as zero above its own declaration. It is now a label
  beside the linker's `__END__`, with the `EXTRN` ahead of the EQU that uses
  it. SDIR's 128-entry hash table first cleared page zero, then another
  module's strings.
- **`AT(...)` understood only a bare `NAME(<literal>)`.** A constant
  expression, `NAME(const)`, `STRUCT.MEMBER` or a chain of them fell through to
  `EQU $`, the assembler's location counter. STAT read a stray byte as its `$`
  parameter and set a file read-only instead of listing it; PIP
  (`DESTR ADDRESS AT(.DEST.FCB(33))`) and PRLCOM were miscompiled the same way.
  An `AT` that cannot be resolved is now an error. `AT(.name)` uses EQU rather
  than SET, and `AT(.external)` emits the EQU even when a reference comes
  first.
- **A STRUCTURE initialiser was emitted at one width.** It gives one value per
  member, each at the member's own width; whatever the list does not fill is
  now reserved. SDIR's ten-byte parser control block came out as five bytes of
  zero. A value the emitter could not place was dropped silently and is now an
  error; `.name(n)` is placeable.
- **`x BASED s.m` read its pointer from the start of `s`.** The member was
  parsed and dropped. SDIR matched every command-line argument against
  address 0 and answered "File Not Found."
- **`DECLARE x (*) BYTE DATA (...)` had no extent.** `LAST(x)` was -2, so PIP
  never searched its delimiter table and answered "INVALID FORMAT" to every
  command.
- **A nested procedure's return type was not known where it is used.** Its
  symbol is filed under its scoped name and the lookup searched only the top
  level, so a `BYTE` result was read out of `L` instead of `A`. `LENGTH`,
  `LAST` and `SIZE` used the same lookup; where they cannot answer they now
  raise instead of emitting zero.
- **A variable `BY` step was treated as `BY 1`.** Only a literal step was read.
  SDIR walks an FCB disk map `BY i`, and counted every block twice on a disk
  with word block numbers.
- **A declared variable did not shadow a condition-flag built-in.** `CARRY`,
  `ZERO`, `SIGN` and `PARITY` are ordinary words a program may declare; STAT's
  zero-suppression flag `zero` read the Z flag. `STACKPTR` deliberately stays a
  built-in, since assigning to it sets SP.
- **A callee's frame was reused while its own arguments were evaluated.** The
  overlay analysis let a procedure called from a later argument share storage
  with an earlier argument already stored, so `call f(7, g)` could destroy the
  7. Bites at the default `-O2`.
- **A procedure-local STRUCTURE was sized as two bytes.** The next procedure's
  frame was overlaid inside it. `-O2`.
- **The target of an embedded assignment was folded like a value.**
  `q = (k := 7)` after `k = 5` stored through the literal 5 and left the stale
  fact about `k` in place. `-O3` only.
- **An induction step hidden in a subscript did not count as modifying the
  variable,** so `arr(i := i + 1) = 9` could fold away a loop's exit test.
  `-O3` only.
- **Only the module that sets SP carries a stack buffer;** a program linked
  from eight modules was carrying eight.

## 0.3.5 — 2026-09-22

An audit prompted by the 80un report. The three defects 0.3.3 and 0.3.4 fixed
turned out to be members of a family that was never swept, and the audit also
found the language defect underneath the `AND`/`OR` story: PL/M-80 tests bit 0
of a condition, not whether the condition is non-zero. Thirty-two fixes, each with a
regression test; see Added for what that was and was not verified to mean.

### Fixed

- **A condition was tested for non-zero; PL/M-80 tests BIT 0.** Every `IF` and
  `DO WHILE` lowered to `or a` / `jp z` (and `ld a,l` / `or h` for a 16-bit
  value). DRI's binaries settle the rule: the code segment of `PIP.PRL` holds
  70 `RAR;JNC` and 14 `RAR;JC` truth tests against a single `ORA A;JZ`,
  `SDIR.PRL` 125 against none, and `ED.PRL` 77 against none. DRI's own hand
  translation of `bdos.plm`'s `IF NOT ROR(ROL(DLOG,1),CURDSK+1)` is
  `mov a,l! rar! rc` — a bit-0 test, and nothing else makes that source
  correct. A relational yields 0FFH or 00H, so both rules agree there; they
  part company on `NOT` and on any masked or rotated value.

  Two consequences, both live in the MP/M II corpus:

  * `NOT <0/1 flag>` inside an `AND`/`OR` read as true. `NOT 1` is 0FEH,
    non-zero. This is what commit 273c83a exposed: before it, `AND`/`OR`
    recursion reached the `NOT` handler, which compiled `NOT` by inverting the
    branch sense and so happened to be right. Eleven sites changed truth value.
    The worst is SDIR's hash-chain scan, `DSE.PLM:340`
    `do while f$i$adr <> 0 and not found;` with `true literally '1'` — the
    match arm sets `found` without advancing the pointer, so the loop never
    terminated and SDIR hung on any repeated name, which means every second
    extent of a file or an XFCB paired with its FCB. Also `DM.PLM:605`
    ("File Not Found." printed after every successful listing) and
    `PIP.PLM:1752` (ambiguous filenames no longer rejected).
  * The DRI bit-extraction idiom was broken everywhere, independently of the
    above. `PIP.PLM:1376-1381` writes every FCB attribute test as
    `if rol(source.fcb(n),1) then ...`, rotating bit 7 into bit 0; the
    non-zero test succeeded for essentially any filename character, so PIP
    stamped F1 — and, from `fcb(9)`/`fcb(10)`, R/O and SYS — onto files that
    did not have them. At least 19 sites across PIP, ED, SDIR and SET.

  Conditions now emit `bit 0,a` (BYTE) or `bit 0,l` (ADDRESS). The 16-bit form
  is the same size as before and no longer clobbers `A`. 273c83a itself was
  correct and stands: PL/M-80's `AND`/`OR` are bitwise, full-evaluation
  operators. The two `NOTE` comments it left behind said the result was tested
  for non-zero; they now state the bit-0 rule.

- **A PROCEDURE declared at the head of a `DO ... END` block was emitted
  inline.** PL/M-80 allows it anywhere a block begins, not only in a procedure
  body, and MP/M II's `ED.PLM` declares `DIGIT` / `NUMBER` / `RELDISTANCE` that
  way inside an IF/ELSE chain. Nothing jumped over the body, so the enclosing
  code ran straight into the procedure and took its `RET`; everything after the
  block was unreachable. Since the uplox front-end migration the label also
  carried the block scope (`@B24$DIGIT`) while the call sites did not, because
  no collection pass descended into blocks — which turned the silent
  miscompile into `Undefined symbol 'DIGIT'` and broke the build outright.
  `ED.PLM` was the last MP/M II target that would not compile. Such procedures
  are now hoisted out, named in the enclosing procedure's scope, and emitted
  after its body, and they keep the block's symbol scope so they still see the
  block's locals.

- **`_expr_preserves_de` claimed a BASED ADDRESS variable preserves `DE`.** Its
  load is `ld hl,(base) / ld e,(hl) / inc hl / ld d,(hl) / ex de,hl`, which
  writes `DE` and leaves base+1 in it. Four callers trusted the predicate, so
  `baccum = baccum + bpb` in `SHOW.PLM:959` computed `baccum + (ab+1)`; the
  same shape is in `DSE.PLM:250`, `STAT.PLM:1136`, and `DM.PLM:237`, where
  `vector = vector or 1` ORed in `v$adr+1`. A BASED BYTE is unaffected — it
  loads through `A`.

- **`-O 3` turned an assignment into a store through an absolute address.**
  Targets were run through the same optimizer as values, so constant
  propagation rewrote the `a` of `a = 5` into the literal `5` and the store
  landed on memory 0005H — CP/M's BDOS entry vector. Only an array element's
  subscript is a value, and only that is optimized now.

- **The byte-comparison operand was still parked in `B` on the condition
  paths.** The 0.3.3 spill was applied to `_gen_byte_binary` and
  `_gen_byte_comparison` but not to `_gen_condition_jump_false` /
  `_gen_condition_jump_true`, which 273c83a had just made the load-bearing
  path. `IF f > x` with `f` = 100 and `x` = 200 read true. The operand is now
  spilled through the stack when the expression generated in between can
  clobber `B`; when it cannot — a literal, a plain variable — the shorter
  sequence is kept, so the common `IF a > b` is unchanged.

- **A 16-bit iterative `DO` parked the loop index in `DE` across the bound
  expression.** A bound that is itself 16-bit emits `ld de,nn` or calls a
  runtime helper. The index goes on the stack unless the bound provably leaves
  `DE` alone.

- **MON1/MON2 parked the BDOS function number in `C` across the argument.**
  `C` is not callee-saved, so `CALL MON1(2, GETC)` reached the BDOS with
  `GETC`'s own function number. The function number is loaded last, which costs
  nothing.

- **A multi-target byte assignment parked the value in `B` across the store.**
  Storing to a subscripted or based target generates the index expression,
  which may call a procedure. `push af` / `pop af` is the same two bytes as
  `ld b,a` / `ld a,b`, and is what the ADDRESS path already did.

- **`SHL(DOUBLE(hi),8) OR lo` parked the high byte in `H` across the low
  operand.** Any low operand that computes in `HL` — a call, a subscript —
  destroyed it. The four-instruction form is kept when the low operand is a
  plain byte load and spilled otherwise.

- **A byte comparison used as a VALUE did not get its left operand into
  `A`.** The condition paths were repaired earlier in this release;
  `_gen_byte_comparison`, the value-producing twin, still opened with
  `_gen_expr(left)`, so a NumberLiteral loaded as `ld hl,n` and the closing
  `sub b` compared an undefined `A`. `r = 5 > x` with `x` = 3 gave 0 rather
  than 0FFH, at the default optimisation level.

- **A cached constant was not narrowed to its variable's declared width.**
  The store truncates — `b = 300` leaves 44 in a BYTE — but the optimizer
  remembered 300, so at `-O 3` the following `IF b = 44` folded to false.

- **A folded relational disagreed with the computed one.** A PL/M-80
  relational yields a BYTE 0FFH; the folder masks to 16 bits, so
  `w = (1 = 1)` stored 0FFFFH at `-O 2` and 00FFH at `-O 0`. Relationals are
  now folded only in a condition, where nothing but bit 0 is observable. As
  a value the comparison is left to the generator, and all four optimisation
  levels emit the same code. This closes the Known issue the previous
  release note carried.

- **The last statically-typed widening in the comparison code.** In the
  branch that parks a complex right operand in `DE`, the widening keyed off
  `_get_expr_type` rather than the type `_gen_expr` returned. They disagree
  for an embedded assignment: `ar(i) := b1` with `ar` an ADDRESS array is
  typed ADDRESS but leaves a BYTE in `A`, so `ex de,hl` took `DE` from
  whatever was in `HL`. Both arms of
  `IF a1 > (ar(i) := b1)` came out false.

- **`IF 2` was reported as "always true"** while the generator made it
  false. The diagnostic now follows the bit-0 rule like the code does.

- **Constant and copy propagation were flow-insensitive.** A fact
  established on one path was reused on another that cannot reach it, which
  at `-O 3` miscompiled the most ordinary loop there is:

  ```plm
  n = 0;
  do while n < 3; call pc('0' + n); n = n + 1; end;
  ```

  `n = 0` was still in scope when the condition was folded, so `n < 3` became
  always-true, `'0' + n` became the literal `'0'` and `n = n + 1` became
  `n = 1`. The loop printed `0` for ever. The same flow-insensitivity reached
  four shapes in all: a `DO WHILE`, an iterative `DO`, a loop closed by a
  backward `GOTO`, and the arms of an `IF` or `DO CASE`, where a value
  assigned in one arm was folded into code after the join that the other arm
  reaches. A loop now drops whatever its body can assign before its condition
  is touched — everything, if the body can call out — a label is treated as
  the join point it is, and each branch arm is optimized from the state at
  the branch rather than from whatever the previous arm left behind.

  Costs nothing at the default `-O 2`: the output is byte-identical. `-O 3`
  grows, because much of what it used to fold away it had no right to.

- **Copy propagation duplicated a procedure call.** `k = rd;` recorded a copy
  of the identifier `rd`, so a later use of `k` was rewritten back into `rd`
  — a second call. In PL/M a parameterless procedure reference is a call, not
  a variable read.

- **A dead IF arm was discarded along with any label inside it.** A `GOTO`
  elsewhere in the procedure still named the label, so codegen emitted a jump
  to a symbol nothing defined and the program failed to assemble at `-O 2`
  while building and running correctly at `-O 0`. The arm is now kept when it
  declares a label, and `DO WHILE` got the same guard. Reachable before this
  release only for `IF 0`; the bit-0 constant rule widened it to every even
  constant, which is how it was found.

- **`ZERO`, `SIGN` and `PARITY` declared a BYTE result but left it in `HL`**,
  so every byte consumer read the wrong register: `IF ZERO` tested `A`, and
  `x = ZERO` overwrote `L` with `A`. They now produce their value in `A`, via
  `ld a,0ffh` / `jp cc` / `inc a` — which, unlike loading zero, cannot be
  strength-reduced into something that writes `CARRY` before a later read.

- **Algebraic identities discarded side-effecting operands.** `x AND 0`,
  `x * 0`, `x OR 0FFFFH`, `x - x` and `x XOR x` dropped an operand that
  PL/M-80 requires to be evaluated — and a bare identifier naming a procedure
  is a parameterless call, not a variable read. The identities now apply only
  when the discarded operand is side-effect free.

- **`IF CARRY` always read false at `-O 1` and above.** The built-in emitted
  `ld a,0` / `rla` — correct in itself, because `ld a,0` does not touch the
  flags — but the peephole rewrites `ld a,0` into the one-byte `xor a`, which
  CLEARS the carry the `rla` is there to read. It now uses `sbc a,a`, which
  reads carry in one instruction and does not depend on `A`. MP/M II's
  `scan$numeric` — shared by `SHOW.PLM`, `MSCHD.PLM` and `TOD.PLM` — guards
  its `b * 10` and `b + digit` steps with `IF CARRY THEN`, so every overflow
  check in those three was dead.

- **Constant folding removed the operation whose carry was about to be
  read.** At `-O 3`, constant propagation folded `s = a + b` to a literal, so
  the `add` that set carry no longer existed and the following `IF CARRY` read
  a stale flag. Arithmetic is no longer folded inside a procedure or module
  body that reads `CARRY`, `ZERO`, `SIGN` or `PARITY`; elsewhere folding is
  unchanged.

- **A BYTE assignment of a constant above 255 emitted a 16-bit load.**
  `_gen_assign` took its byte path only for values that already fit, so a
  folded `200 + 100` went out as `ld hl,012CH` / `ld a,l` — which the peephole
  then collapsed into `ld a,012CH`, keeping all sixteen bits. PL/M-80 narrows
  to the target's width, so the constant is truncated in the generator.

- **The truth-rule fix initially missed the CONSTANT paths**, so the same
  source got different answers at different optimisation levels: `IF NOT TRUE`
  with `TRUE LITERALLY '1'` folds to 0FEH and was true at `-O 0` but false
  once `_optimize_if` folded it, `IF 4` and `IF (4)` disagreed in one
  compilation unit, and `DO WHILE 2` was an infinite loop where it should
  never run. All four constant sites — two in the generator, two in the AST
  optimizer — now test bit 0.

- **Byte operands were generated with `_gen_expr`, which leaves a
  NumberLiteral in `HL`.** `IF 5 > X` compared an undefined `A`, and
  `SHL(DOUBLE(hi),8) OR 5` emitted `ld hl,5` straight over the high byte
  parked in `H`. Byte operands now go through `_gen_expr_to_a`. The mirror
  problem — code that widened a byte result into `HL` keyed off the
  *statically inferred* type rather than the type `_gen_expr` actually
  returned — is fixed by a new `_gen_expr_to_hl`, which also repairs the
  16-bit iterative `DO` bound and the `MOVE` count below.

- **A BYTE store to a structure member read the value out of `L`.** The
  member path always saved and reloaded `HL`, but a BYTE value is in `A`, so
  `rec.f = ch` stored whatever happened to be in `L`.

- **`CALL MOVE` with a non-constant BYTE count took `BC` from the source
  address.** The count was generated into `A` and then moved with
  `ld b,h / ld c,l`.

- **An iterative `DO` discarded its declaration list**, so a `PROCEDURE`
  declared at the head of one was registered and never emitted — the call
  site named a label nothing defined.

- **`_boolean_simplify` was a sibling path the first pass missed**: it still
  produced 0FFFFH for a relational, and its `x REL x` and idempotent
  `(a AND b) AND b` rules still discarded operands that PL/M-80 requires to
  be evaluated.

- **`INPUT`, `OUTPUT`, `MOVE`, `TIME`, `SCL`/`SCR` and the flag built-ins
  were treated as side-effect free**, so `r = INPUT(5) AND 0` deleted the
  `in a,(port)`. The purity test now names them explicitly rather than
  inferring purity from "not a user procedure".

- **The AST optimizer never reset its flow-sensitive state.** `constants`,
  `copies`, `cse_cache`, `expr_vars` and `modified_vars` accumulated across
  procedure boundaries and across the five optimisation passes, so at `-O 3`
  one procedure's constant was folded into another's body: with `ONE` setting
  `V = 1`, the unrelated `TWO: PROCEDURE; R = V; END TWO;` compiled to
  `ld hl,1` and never read `V`. The state is now cleared per procedure and
  per pass.

- **A BYTE procedure returning an ADDRESS expression normalised it to a
  boolean.** PL/M-80 narrows ADDRESS to BYTE by truncation, like `LOW()`, so
  `P: PROCEDURE BYTE; RETURN N + 1; END P;` with `N` = 64 must return 65. The
  generator emitted `ld a,l / or h / jp z / ld a,0ffh`, returning 0FFH for
  every non-zero value. Found while sweeping for remaining non-zero tests
  after the truth-rule fix.

- **`NOT` in a condition did not strip parentheses**, so `IF NOT (a = b)` never
  reached the optimised compare and materialised a value instead. Generated
  code was correct; it was one instruction pair longer than it needed to be.

### Known issues

- In `-m bare`, a module body that runs off its end falls into the first
  procedure emitted after it rather than stopping. This is long-standing and
  unchanged here. The only bare-mode program in the corpora, MP/M II's
  `MPMLDR`, ends its body in a `di` / `halt` loop, so the fall-through is
  unreachable; giving the mode an explicit terminator would change semantics
  the mode documents as the program's own business, so it is left alone.

- Two sibling `DO` blocks in one procedure that each declare a procedure of
  the same name collide on one assembly label. The assembler rejects that
  outright, so it cannot go unnoticed, and no PL/M-80 source in the CP/M or
  MP/M II corpora writes it.

### Added

- Regression tests for the fixes above, written as invariants over the
  generated assembly rather than golden output. All of them fail against the
  baseline generator, and nine fixes were reverted individually to confirm the
  suite catches each on its own. The per-fix property has not been verified
  exhaustively for every fix in the list.
- String-literal tests, which settles the debt `todo.txt` recorded against
  0.3.4: a backslash is an ordinary character in PL/M-80, `''` is one quote,
  and a string may cross a line break. `uplm80/_plm_parser.py` is generated, so
  a regen against a grammar that brought the C escape rule back would otherwise
  have gone unnoticed.

### Verified

- 80un: `80un.com` and `80unbas.com` produce byte-identical output on all 21
  samples of the test corpus and on the BASIC detokeniser. The generated code
  changes, so the committed `.COM` files need a rebuild to stay reproducible
  against this release.
- MP/M II: all 41 targets build, up from 40 of 41. Thirty binaries change and
  the total is size-neutral (`ED` +128 because it is built at all, `SDIR` +128,
  `STOPSPLR` -128, `TYPE` -128). The source-built utilities boot and run on a
  DRI system — `DIR` and `STAT` both produce correct output under the
  emulator.

  No claim is made about mpm2's `scripts/run_tests.sh` pass rate. That
  harness is not deterministic: five runs against one unchanged disk image
  fail at three different points, and measured over five runs each, the
  binaries from this release and from the previous generator score the same
  within the noise. An earlier draft of this entry credited a fix with
  repairing the `STAT` test; that was a single lucky run and is withdrawn.

### Known issues — not this compiler

- MP/M II `--tree=src` fails system generation with `XIOS common base BF4BH
  below configured common base C000H`. `XDOS.SPR` grew from 8960 to 10112
  bytes, and XDOS is assembled entirely from `.ASM` — the growth reproduces
  with every uplm80 version tested and disappears with the January `um80`/
  `ul80`, so it belongs to the assembler and linker.
- MP/M II built `--tree=src` reaches the console banner but not a command
  prompt. This reproduces with the January toolchain, so it predates all of the
  above.

## 0.3.4 — 2026-09-19

The third code-generation defect of the same family, which is the one that made
the CrLZH decoder of the 80un unpacker come out wrong. That known issue is
closed, and all three defects now have regression tests.

### Fixed

- **The base address of a subscripted element was destroyed by the index
  expression.** `_gen_subscript_addr` parked the array base in `DE` and then
  generated the index. An index that is itself a 16-bit expression emits
  `ld de,nn`, which overwrote the base, and the closing `add hl,de` then added
  the index constant a second time instead of the base. `PRNT(I + LZH$T) = I`
  with `LZH$T` = 629 stored to `(I + 629) * 2 + 629` rather than to
  `PRNT + (I + 629) * 2`, so every element address was wrong and the store
  scribbled over unrelated memory.

  An index of `I` or `I + 1` was unaffected, because neither needs `DE` - a
  `+ 1` becomes `inc hl`. Only an index carrying a constant too large for
  `inc`, or any other 16-bit subexpression, triggered the defect, which is why
  so little broke so specifically: in 80un only `init$tree` and `update$tree`
  index that way, and the result was a CrLZH Huffman parent table that was
  never initialised.

  The base is now kept on the stack across generation of the index, which is
  what the generator did before bdb0f8a migrated subscripts onto the register
  allocator. The same defect was present in both member-subscript paths and is
  fixed there too.

### Added

- Regression tests for all three defects fixed in 0.3.3 and 0.3.4, written as
  register-liveness invariants over the generated assembly rather than as
  golden output: a value parked in `DE` or `B` must not be destroyed before it
  is read, and the false path of a BYTE `>` must reach the `xor a` that loads
  zero. Each test fails against a copy of the generator with the corresponding
  fix reverted.

### Fixed known issue

- CrLZH decoding in 80un is correct again. Built with this release, 80un scores
  15 of 15 byte-exact against the original CP/M UNCR24.COM and reproduces its
  Python decoders on 107 of 108 sample-corpus members, which is parity with the
  last generator known to be good, 01cfcc6. 80un no longer needs to pin an old
  compiler to ship a correct binary.

## 0.3.3 — 2026-09-19

Two code-generation defects that silently produced wrong Z80, plus the PL/M
string-literal fix that comes with the raised `uplox` floor. Both code
generation defects are present in every earlier release; commit 273c83a made
them reachable rather than introducing them.

### Fixed

- **A BYTE `>` comparison used as a value was non-zero when false.** In
  `_gen_byte_comparison_const` and `_gen_byte_comparison` the `GT` arm jumped
  to the join label that follows `ld a,0ffh`, not to the `xor a` false case,
  which left the `xor a` unreachable and the compared operand sitting in the
  accumulator. `Y = X > 32` with `X = 6` assigned 6 instead of 0. The arm now
  branches to a real false label.
- **Register `B` was clobbered while the other operand of a byte `AND`, `OR`,
  `XOR` or `SUB` was generated.** `_gen_byte_binary` and
  `_gen_byte_comparison` parked one operand in `B` with `ld b,a` and then
  generated the other operand. `B` is not callee-saved, a nested byte
  comparison uses `ld b,a` as its own scratch move, and a procedure call in
  the other operand overwrites `B`, so the closing `and b` masked against
  garbage: `D = (A > 0) AND (B < C)` with `A = 0` came out true. The operand
  is now spilled through the stack, and `B` is loaded only once the other
  operand has been generated. `pop bc` would be shorter than `ld b,a` plus
  `pop af`, but `pop bc` also overwrites `C`, where the CP/M call convention
  keeps a live argument - `CALL MON1(2, '0' + X)` became a call to BDOS
  function 68.

  Both defects only reach the generated code when a comparison is
  materialised as a value. Commit 273c83a, which correctly made PL/M-80's
  `AND` and `OR` bitwise rather than short-circuit, routed every `IF` and
  `DO WHILE` condition through that path, so the two defects went from latent
  to load-bearing. Standalone programs demonstrate both under 0.3.1 as well.

### Changed

- `uplox` floor raised to `>=3.3.1`, and `uplm80/_plm_parser.py` regenerated
  against that grammar. A backslash inside a PL/M character literal - MBASIC's
  integer-divide token, for one - no longer fails with
  `lexical error at byte 0x5c`. PL/M-80 has no backslash escape.

### Known issues

- Code generated for the CrLZH decoder of the 80un CP/M unpacker is still
  wrong. A bisect puts the first failure at bdb0f8a "Implement register
  tracking phases 3-5", where `lzh$get$byte` parked its accumulator in `DE`
  across a call to `lzh$get$bit`; that particular defect is gone from the
  current generator, whose `LZHGETBYTE` matches the pre-bdb0f8a output
  instruction for instruction. A later, separate defect in the same area
  remains, somewhere in the plox front-end migration range, and the commits
  in that range cannot be built against a current `uplox` to narrow it
  further. Until the defect is found, 80un builds its released binary with
  uplm80 01cfcc6.

## 0.3.2 — 2026-08-20

No change to the compiler since 0.3.1. This release raises one dependency
floor and corrects two documentation items.

### Changed

- `uplox` floor raised to `>=3.3.0`, so a fresh install resolves the parser
  runtime that uplm80 is actually developed against rather than 3.2.0.
  3.3.0 adds the classifier lookahead window and named LR-state sets and
  fixes an IELR backward-propagation bug that could trip a table-build
  assertion.
- `sample_code/CPM_source/1,1/origin.txt` now points at
  `https://www.icl1900.co.uk/...` — z80pack's sources moved off
  `autometer.de`, and the recorded provenance URL no longer resolved.
- The Related Projects section of the README was rewritten in Simplified
  Technical English.

The `upeepz80` floor stays at `>=0.2.3`; nothing in that package changed.
