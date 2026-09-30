# Calling Convention, Runtime Library and Output Names

## Calling Convention

A call passes its arguments the way Intel's PL/M-80 does, so code uplm80
compiles links with assembly written for PL/M-80 - DRI's `X0100.ASM`
(`mon1 equ 0005h`), MP/M II's `LDMONX.ASM` and `BRSPBI.ASM` - and with what
PL/M-80 compiled.  (Up to 0.3.x it did not: see CHANGELOG.md, 0.4.0.)

| Arguments | Where they are at the `call` |
|---|---|
| 0 | nothing |
| 1 | a1 in **BC** (C for a BYTE parameter) |
| 2 | a1 in **BC** (C), a2 in **DE** (E) |
| n >= 3 | a1 ... a(n-2) **pushed left to right**, one word each; a(n-1) in **BC** (C); an in **DE** (E) |

- At entry `[SP]` is the return address, `[SP+2]` is a(n-2), and so on to
  `[SP+2(n-2)]`, which is a1.
- A BYTE argument in a register is in C or E; B or D is undefined.  A pushed
  BYTE is the low byte of its word; the high byte is undefined.
- **The callee takes the pushed words off the stack.**  The caller never
  adjusts SP after a call: when the callee returns, SP is what it was before
  the first push for the call.
- A BYTE result is returned in **A**, an ADDRESS one in **HL**.
- A call destroys A, the flags, BC, DE and HL.  It keeps SP, and **IX and
  IY**: a REENTRANT procedure keeps its frame in IX across its calls.
- The arguments are evaluated from left to right, each converted to its
  parameter's type.  An EXTERNAL declaration only gives those types; PUBLIC, EXTERNAL, nested and REENTRANT procedures are all called
  the same way, and a direct call must pass as many arguments as the
  procedure has parameters.
- A `CALL` through an address (8.2.1) places the arguments the same way,
  each widened to ADDRESS, and calls `??jphl` with the address in HL.  It
  may pass any number, to any procedure.
- `MON1(f, a)` and `MON2(f, a)` with a constant `f` are compiled as the
  BDOS call itself, `ld de,a / ld c,f / call 5` (`call ??BDOS` under
  `-m mpm`): the registers PL/M-80 sets for `mon1 equ 5`.
- An INTERRUPT procedure has no parameters, and is at the outer level of
  its module (8.1.6).

**One exception.**  A procedure with one parameter that nothing outside the
compile can reach - not PUBLIC, EXTERNAL or REENTRANT, and its address
never taken (`.p` anywhere, `INITIAL` and `DATA` included) - takes its
argument in **A** (BYTE) or **HL** (ADDRESS) instead of C or BC, where its
body usually wants it.  No other module, no assembly and no CALL through an
address can see the difference.

**Writing an assembly routine that PL/M calls.**  Take the arguments as the
table says; take the pushed words off the stack before you return; return a
BYTE in A and an ADDRESS in HL; and keep SP, IX and IY.  For

```plm
cap3: procedure (a, b, c) external; declare (a, c) address, b byte; end cap3;
```

```asm
        public  CAP3
CAP3:   ld      (VC),de         ; c, the last argument
        ld      a,c             ; b, in C
        ld      (VB),a
        pop     hl              ; the return address
        ex      (sp),hl         ; a, and the return address back on top
        ld      (VA),hl
        ret
```

With more pushed arguments, `pop` the return address, `pop` each pushed word
(the last argument pushed comes first), and `push` the return address
again.  Assembly that calls a PL/M procedure does the same from the other
side: push the first arguments, load the last two into BC and DE, call, and
leave the stack alone afterwards.

Code built with 0.4.0 needs upeepz80 0.2.6 or later: 0.2.5 turned `push ... /
call p / ret` into `push ... / jp p`, after which p takes its return address
for its first argument.  At `-O1` and up the compiler refuses an upeepz80
whose version is below 0.2.6, unless it keeps that `call` (a development
tree with the fix, numbered before 0.2.6 was released); `-O0` does not use
upeepz80.

## Runtime Library

A module carries the runtime routines it uses at the end of its code
(`uplm80/runtime.py`):

| Routine | Description |
|---------|-------------|
| `??mul16` | 16-bit multiply, HL = HL * DE |
| `??div16`, `??mod16` | 16-bit divide and remainder, as DRI's PL/M-80 computes them |
| `??subde` | 16-bit subtract, HL = HL - DE |
| `??jphl` | A CALL through an address (`CALL q`, Programming Manual 8.2.1): `jp (hl)` |
| `??inp`, `??outp` | INPUT and OUTPUT of a port that is not a constant |

## Names in the Output

A module-level name is its own name in the assembly, and a procedure's
names are `@proc$name`, but for the parameters and locals that share
storage in `??AUTO` (see [Procedure locals](runtime_modes.md#procedure-locals)).  Where
PL/M-80 would let two declarations meet in one assembler name - a label in
each of two DO blocks, a procedure in each of two blocks, a LITERALLY in
each of two procedures, a procedure or label in a DO block named like a
local or a parameter of the procedure around it - one is renamed `NAME?2`
(no PL/M-80 identifier has a `?`): a LITERALLY before a label, a label
before a procedure, a procedure before a variable.  A name the assembler
reads as a register or an operator (`A`, `HL`, `EQ`, `NUL`, ...) is
`@NAME`, and an offset from a name that ends in the letters of an operator
is written first (`2+TYPE`, not `TYPE+2`); um80 0.3.51 reads the operator
names and `TYPE+2` as M80 does, and needs neither, but older um80 releases
do.  `uplm80/names.py` binds every name to the declaration PL/M-80 means
and checks every GOTO against the Programming Manual (9.3): out of a
procedure only to a label at the outer level of the main program module.
