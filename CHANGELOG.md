# Changelog

Notable changes to uplm80. Releases before 0.3.2 are described on the
[GitHub releases page](https://github.com/avwohl/uplm80/releases).

## 0.3.5 — 2026-09-22

An audit prompted by the 80un report. The three defects 0.3.3 and 0.3.4 fixed
turned out to be members of a family that was never swept, and the audit also
found the language defect underneath the `AND`/`OR` story: PL/M-80 tests bit 0
of a condition, not whether the condition is non-zero. Twenty-four fixes, each with a
regression test that fails against the generator with that fix reverted.

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

- **A folded relational had the value 0FFFFH; the runtime paths give 0FFH.** A
  PL/M-80 relational yields a BYTE, so `x = 1 > 0` and `x = a > b` disagreed as
  values depending on whether the comparison folded.

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

### Added

- Regression tests for all twenty-four fixes, written as invariants over the
  generated assembly rather than golden output. Each fails against a copy of
  the generator with the corresponding fix reverted.
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
