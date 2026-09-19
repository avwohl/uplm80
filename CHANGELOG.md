# Changelog

Notable changes to uplm80. Releases before 0.3.2 are described on the
[GitHub releases page](https://github.com/avwohl/uplm80/releases).

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
