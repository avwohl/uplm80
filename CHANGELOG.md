# Changelog

Notable changes to uplm80. Releases before 0.3.2 are described on the
[GitHub releases page](https://github.com/avwohl/uplm80/releases).

## Unreleased

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

Each fix has a regression test that fails without it.

### Fixed

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

### Changed

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

### Added

- **`tests/plm_difftest.py`**, a differential test: random programs over
  BYTE and ADDRESS variables, constants, every operator but `PLUS` and
  `MINUS` (whose carry-in depends on the code before them), built-ins, calls
  of procedures that change globals, BASED stores and DO loops, compiled at
  `-O0` to `-O3`, run under cpmemu and compared with a Python model of the
  manual's rules. The suite runs three programs; `scripts/difftest.py
  --seeds N` runs more.

### Known issues

- **upeepz80 0.2.4 deletes register loads that are still needed,** all found
  by the differential test and all in 0.3.6's output as well; the fixes
  belong in upeepz80.
  * It rewrites `ld a,(x) / cpl / cpl / inc a / push af / ld (x),a / pop af`
    into `ld hl,x / inc (hl)` although A is read next: `b, w = -(NOT b)` - a
    BYTE and an ADDRESS assigned together - leaves `w` holding A's old value
    at `-O1` and above (one of 700 random programs).
  * Right after a load of A, it rewrites `ld hl,nn / ld a,l / ld l,a /
    ld h,0` into `ld a,nn / ld h,0`, so L keeps whatever it held: after
    `b = 0FEH`, `w = LOW(LAST(big))`, with `big` 300 bytes long, got the low
    byte of the statement before instead of 2BH at `-O1` and above (seed 1063
    of `scripts/difftest.py`).
  * It drops an `ld hl,n` that a store of HL still reads: `ld hl,0 / ld a,l /
    ld (sb),a / push hl / ld (w1),hl / pop hl / ld (w0),hl` becomes `xor a /
    ld (sb),a / ld (w1),hl / ld (w0),hl`. Its check that HL is dead, before
    `ld hl,n / ld a,l` becomes `ld a,n`, does not count `ld (nn),hl` as a
    read. LOW and HIGH of a constant no longer generate either shape, but
    other code can.
  * Its `ld a,(x) / inc a / ld (x),a` to `ld hl,x / inc (hl)` takes `(ix+n)`
    for an address (`ld hl,ix+n`); DO loops no longer generate that.
- **PLUS, MINUS, SCL and SCR after `+ 1` to `+ 4` or `- 1` to `- 4` of an
  ADDRESS** take a stale carry: those are `inc hl` and `dec hl`, which set
  none, so `(w - 1) MINUS z` with w = z = 0 gives 0FFFFH, not 0FFFEH. DRI's
  PL/M-80 increments with INX and INR too (PIP.PLM declares a variable
  `ONE = 1` so that `DEC(C1 + ONE)` gets an ADD and its carry), and the
  manual (12.1) warns that the flags cannot be relied on.
- **A REENTRANT procedure's parameter in a factored declaration with its
  locals,** `DECLARE (top, c) BYTE`, is taken for a local; declared on its own
  it is read from the stack as it should be.
- **A procedure named like a register,** `H: PROCEDURE`, is not renamed as a
  variable of that name is, and the assembler rejects `call H`.
- **A counted loop trusts that a pointer made from `.x` reaches only `x`.**
  A BYTE `DO i = 0 TO n` whose body does not name `i` counts its passes in
  B, and is not used when anything can reach `i` another way: a procedure
  that names it, `.i` anywhere, `i` AT or BASED. A pointer computed from the
  address of the variable declared before `i` still can, and a store
  through it does not end the loop. DRI's compiler never counts a loop.
- `tests/test_byte_conditions` in `run_tests.sh` expects the pre-0.3.5
  non-zero truth test, and fails against 0.3.6 and this release alike.

### Verified

- Every PL/M source in MP/M II (the 41 in DRI's tree and the 14 overrides,
  each in the mode `tools/build.py` uses) and in 80un (35 files one at a time,
  and `80un.com` and `80unbas.com` as their Makefile compiles them) compiles
  at -O0, -O2 and -O3, and the same 82 of the 92 outputs assemble with um80
  0.3.49 as with 0.3.6 (the rest are single modules of multi-module programs,
  and MSCMN.PLM, which is only ever included). At -O2 every output changes
  from 0.3.6, if only by the layout, and every change is one of the entries
  above. The 82 come to 224,925 bytes, against 227,511 with 0.3.6. `80un.com`
  extracts the same files, with the same console output, from all 17 sample
  archives as 0.3.6's, at -O2 and -O3, and `80unbas.com` detokenises
  `PALLOPS.BAS` the same.
- MP/M II built from source with this release - `tools/build.py` for V2.0
  and V2.1, 44 of 44 targets each - passes mpm2's `scripts/run_tests.sh all`
  on the V2.1 system and `scripts/run_tests.sh src`. SUBMIT and SPOOL were
  built from DRI's own `SUB.PLM` and `MSPL.PLM`, without the workarounds
  mpm2 carried for the old layout, and compared on the emulator with DRI's
  V2.0 `SUBMIT.PRL` and `SPOOL.PRL`: SUBMIT runs two 300-line command files
  (3968 and 3328 bytes), a 250-line one of 15K, and one with parameters,
  printing exactly what DRI's does, where the release before the layout
  change printed its own messages over the 3968-byte file and ran none of
  it; SPOOL, printing files itself on a system without the spooler RSP,
  prints a 150-line file and a 7936-byte one as DRI's does, where before it
  printed 512 NULs in place of their first records. `stat usr:` now prints
  what DRI's STAT prints. GENSYS built from source, whose DATA now follows
  its code, makes the same MPM.SYS and SYSTEM.DAT under cpmemu as DRI's
  GENSYS, with the same console output (V2.1 differs in the six bytes of
  the serial number, which the build leaves as DRI's placeholder).
- The differential tests: `scripts/difftest.py --seeds 400 --first 1000`,
  all 400 random programs as the model says at `-O0` to `-O3` (seed 1063,
  which the second upeepz80 defect above broke, no longer meets it); and the
  integration verification's own generator, which covers DATA, INITIAL, AT,
  BASED, DO loops whose body moves the index or the bound, calls in
  arguments and module-level code: 497 programs at `-O0` to `-O3`, all as its
  model says.
- The run tests compile with the checkout under test: they used to start the
  compiler with `python -P`, which found whatever uplm80 was installed. Every
  test that runs a program - the run tests, the differential test and the
  division oracle - now assembles, links and runs it with
  `tests/_toolchain.py`.

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
