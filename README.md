# uplm80 - PL/M-80 Compiler

[![PyPI version](https://badge.fury.io/py/uplm80.svg)](https://pypi.org/project/uplm80/)
[![Tests](https://github.com/avwohl/uplm80/actions/workflows/pytest.yml/badge.svg)](https://github.com/avwohl/uplm80/actions/workflows/pytest.yml)
[![Pylint](https://github.com/avwohl/uplm80/actions/workflows/pylint.yml/badge.svg)](https://github.com/avwohl/uplm80/actions/workflows/pylint.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A modern PL/M-80 compiler targeting Zilog Z80 assembly language.

PL/M-80 was the primary systems programming language for CP/M and other 8080/Z80 operating systems. This compiler can rebuild original CP/M utilities from their PL/M source code.

**Repository:** https://github.com/avwohl/uplm80

## Features

- Full PL/M-80 language support
- Targets Z80 instruction
- Multi-file compilation with cross-module optimization
- Multiple optimization passes (AST optimizer, upeepz80 peephole optimizer)
- Generates relocatable object files compatible with standard CP/M linkers
- Calls procedures the way Intel's PL/M-80 does, so its code links with assembly and objects written for PL/M-80
- Produces code competitive with the original Digital Research compiler

## Code Quality

Compiled output is comparable to the original Digital Research PL/M-80 compiler:

| Program | DR PL/M-80 | uplm80 | Difference |
|---------|------------|--------|------------|
| PIP.COM | 7424 bytes | 7127 bytes | -4.0% |

## Installation

Quick install from PyPI:

```bash
pip install uplm80 um80 upeepz80
```

**Platform-specific guides:**
- **Raspberry Pi**: See [README_RASPBERRY_PI.md](README_RASPBERRY_PI.md)
- **General/Development**: See [INSTALL.md](INSTALL.md)

Or install from source:

```bash
git clone https://github.com/avwohl/uplm80.git
cd uplm80
pip install -e .
```

## Usage

### Compile PL/M-80 to Assembly

```bash
uplm80 input.plm -o output.mac
```

Or run as a module:

```bash
python -m uplm80.compiler input.plm -o output.mac
```

Options:
- `-m cpm`, `-m bare` or `-m mpm` - Runtime mode (default: cpm)
  - `cpm`: For new PL/M programs, maximum stack under BDOS
  - `bare`: Original Digital Research compatible (jump to start-3)
  - `mpm`: MP/M II relocatable modules (`.PRL`, `.RSP`, `.SPR`)
- `-o output.mac` - Output file name
- `-O 0|1|2|3` - Optimization level (default: 2)
- `-D SYMBOL` - Define conditional compilation symbol (can be repeated)

### Multi-File Compilation

Compile multiple source files together for optimal cross-module optimization:

```bash
uplm80 main.plm helper.plm library.plm -o output.mac
```

When multiple files are provided:
- All files are parsed together before code generation
- A unified call graph is built across all modules
- Procedures that are never active at the same time share storage for their parameters and for the locals that may share it (`??AUTO`, see [Procedure locals](#procedure-locals)), across module boundaries
- Calls between the modules are made as between modules compiled apart (see [Calling Convention](#calling-convention)), so a module can as well be compiled alone and linked with the others
- Each module keeps its own name space, as if it were compiled alone and linked: a name that is not PUBLIC or EXTERNAL is qualified with the module's name in the output (`LIB?HELPER`), and PUBLIC and EXTERNAL names bind across the modules (see [docs/multi_file_compilation.md](docs/multi_file_compilation.md))
- A single combined output file is generated

This produces better code than compiling files separately, as the compiler can share local variable storage between procedures in different modules that never call each other.

### Assemble and Link

Use your preferred Z80 assembler and linker. Example with um80/ul80:

```bash
um80 output.mac                              # Assemble to .rel
um80 x0100.asm                               # MON1 equ 5 and the rest (see CP/M Mode)
ul80 -o program.com output.rel x0100.rel     # Link to CP/M .com
```

The module carries the runtime routines it uses (see [Runtime
Library](#runtime-library)); link with it only what defines the names it
declares EXTERNAL, here `x0100.rel` for MON1.

## Language Reference

PL/M-80 is a typed systems programming language with:

- **Data types**: BYTE (8-bit), ADDRESS (16-bit)
- **Variables**: Scalars, arrays, structures, BASED variables (pointers)
- **Control flow**: DO/END, DO WHILE, DO CASE, IF/THEN/ELSE
- **Procedures**: With parameters, local variables, recursion
- **Built-in functions**: HIGH, LOW, DOUBLE, SHL, SHR, ROL, ROR, etc.
- **I/O**: INPUT, OUTPUT for port access

Example:

```plm
hello: DO;
    DECLARE message DATA ('Hello, World!$');
    DECLARE i BYTE;

    mon1: PROCEDURE(func, parm) EXTERNAL;   /* mon1 equ 5: see CP/M Mode */
        DECLARE func BYTE, parm ADDRESS;
    END mon1;

    print: PROCEDURE(addr) PUBLIC;
        DECLARE addr ADDRESS;
        /* CP/M BDOS print string */
        CALL mon1(9, addr);
    END print;

    CALL print(.message);
END hello;
```

See [examples/hellocpm.plm](examples/hellocpm.plm) for a complete working example.
A drop-in Makefile that drives the full `uplm80 → um80 → ul80` pipeline (with
optional `ud80`/`ux80` disassembly targets) is available at
[docs/example.Makefile](docs/example.Makefile) — contributed by Martin
Homuth-Rosemann ([@Ho-Ro](https://github.com/Ho-Ro), issue #5).

For more on CP/M BDOS usage, see [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md).

The compiler holds a program to these rules of the Programming Manual
(9800268B), as Intel's PL/M-80 V3.1 does, and says what it found and
where:

- Every name is declared, the built-ins aside (6.1); in a multi-file
  compile a module may name another's PUBLIC name without declaring it
  EXTERNAL, but for a built-in's name, which is the built-in in a module
  that does not declare it, as it is compiled alone.
- Each parameter is declared once, by a DECLARE of its procedure, as a
  BYTE or an ADDRESS scalar, not BASED and with no other attribute
  (8.1.1).
- A declaration hides the built-in of its name (9.2): a procedure DOUBLE
  or a variable OUTPUT, MEMORY or STACKPTR that a module declares is the
  module's there, and the compiler never takes it for the built-in.
- A LITERALLY's text takes its name's place throughout its scope, in the
  text after the declaration (6.4), a declaration of the name in an inner
  block included: after `DECLARE N LITERALLY '5'`, an inner `DECLARE N
  BYTE` is `DECLARE 5 BYTE`, an error.  The name used before the
  declaration is not declared there.
- A dimension is a number, or a LITERALLY declared before it whose text
  is one (6.2.5), and not 0.
- Empty parentheses are an error after a variable, `x()`, after a
  structure member, `s.m()`, after a subscript, `a(1)()`, and after a
  built-in, `carry()`; `f()` and `CALL g()` of a procedure are taken for
  `f` and `g`, with a warning, as programs written for uplm80 rely on
  them.
- The address of a label, `.label`, may be given in a DATA or an INITIAL
  list, not in an expression (4.1.3); of the built-ins, only MEMORY has
  an address.
- An INTERRUPT procedure is declared at the outer level of its module and
  has no parameters (8.1.6); so is a PUBLIC or EXTERNAL procedure or
  variable, not in a procedure or a DO block.  INITIAL there initializes
  the variable once, when the program is loaded, with a warning.
- A direct call passes as many arguments as the procedure has parameters.

## Conditional Compilation

PL/M-80 v4.0 added conditional compilation. The directives are **control lines** — a leading `$` at the left margin (column 1), exactly like `$INCLUDE` and `$TITLE` — so the same source can target different configurations (e.g., CP/M 2.2 vs CP/M 3, single-user vs MP/M). No enabling directive is required.

### Directives

| Directive | Description |
|-----------|-------------|
| `$SET (NAME)` | Define a symbol |
| `$RESET (NAME)` | Undefine a symbol |
| `$IF NAME` | Compile following code if NAME is defined |
| `$ELSEIF NAME` | Else-if branch |
| `$ELSE` | Else branch |
| `$ENDIF` | End conditional block |
| `$COND` / `$NOCOND` | Listing controls only (accepted as no-ops) |

A comment-wrapped form (`/** $if NAME **/`) is also accepted for CP/M-3-style sources that embed the same directives in comment syntax.

### Example

```plm
$set (CPM3)
DECLARE
$if CPM3
    VERSION LITERALLY '30H',
$else
    VERSION LITERALLY '22H',
$endif
    MAXFILES BYTE;
```

### Command Line

Symbols can also be defined from the command line:

```bash
uplm80 pip.plm -D CPM3 -D MPM -o pip.mac
```

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

## Names in the Output

A module-level name is its own name in the assembly, and a procedure's
names are `@proc$name`, but for the parameters and locals that share
storage in `??AUTO` (see [Procedure locals](#procedure-locals)).  Where
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

## Runtime Modes

The first 100H bytes of a CP/M program's memory are reserved by the operating
system (zero page, default FCB, default DMA buffer). All CP/M `.COM` programs
load at address 100H, so a CP/M binary's *contents* start at offset 0 of the
file — the linker takes care of relocating to 100H. **PL/M source files should
not declare `100H:` themselves**; doing so causes the assembler to emit a
`cseg org 100H`, which the linker then honors by padding the binary with 256
zero bytes from 0–FFH. See the `bare` mode notes below for the one situation
where a leading address constant is meaningful.

### CP/M Mode (default: `-m cpm`)

The mode to use for new PL/M-80 programs. The compiler emits a small entry
preamble that takes maximum stack space under BDOS and returns cleanly to CP/M:

- The compiler emits the entry code at the start of the `.com` image —
  **do not write `0100H:` in your source.** The linker (`ul80`) defaults to
  origin 100H, which is what CP/M wants.
- Entry preamble (auto-generated):

  ```asm
  ld   hl,(6)       ; load BDOS base from address 6
  ld   sp,hl        ; stack grows down from just below BDOS
  call MAIN         ; run your main procedure
  jp   0            ; warm-boot return to CP/M when MAIN returns
  ```
- Stack: maximum available — everything between program end and BDOS.
- The BDOS interface procedures a program declares EXTERNAL - `MON1`, `MON2`,
  `MON2A`, `MON3` - and `BOOT` are equates, as DRI's `X0100.ASM` defines
  them: a call of `MON1` already has the function in C and the argument in
  DE, which is what the BDOS at 5 takes.

  ```asm
          public  MON1, MON2, MON2A, MON3, BOOT
  MON1    equ     5
  MON2    equ     5
  MON2A   equ     5
  MON3    equ     5
  BOOT    equ     0
  ```
- System variables: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`.

### Bare Metal Mode (`-m bare`)

The mode required to rebuild original Digital Research utilities (PIP.PLM,
ED.PLM, etc.) byte-compatibly. These programs follow the Intel PL/M-80
convention of jumping to *start − 3* to skip over a local stack area:

- Entry preamble (auto-generated):

  ```asm
  ld   sp,??STACK   ; a 64-byte stack in the data segment (see Memory Layout)
  jp   MAIN         ; jump (not call) into MAIN
  ```
- The program controls its own exit — no automatic warm boot. Original DR
  utilities reboot or chain by writing to memory directly.
- Custom entry points: because the entry preamble lives in the first few bytes
  of the image, original programs sometimes prepend a `DECLARE … DATA(...)`
  block to forge a different jump (see PIP.PLM, which fakes a JMP table at
  page 1). In bare mode the leading address constant (e.g. `0100H:` or
  `0200H:`) *is* meaningful — it sets the assembler `org` for the bare image.
- Compatible with original Intel/DR PL/M-80 sources.

### MP/M Mode (`-m mpm`)

For MP/M II page-relocatable modules. MP/M gives each process a memory
segment and puts its page zero at the segment's base, so the page-zero
addresses a program uses have to be relocated when it loads — and only a
resolved *symbol* reference reaches a `.PRL` relocation bitmap.

- Page-zero references are emitted as externals: `??BDOS` (0005H), `??MAXB`
  (0006H) and `??BOOT` (0000H). Link with a small module that defines them at
  those addresses, using `ul80 --prl` (transient, linked at 100H) or
  `ul80 --spr` (system page, linked at 0); ul80 marks resolved page-zero symbol
  references for relocation.
- The program's own externals - `MON1`, `MON2`, `MON2A`, `MON3`, `FCB`,
  `TBUFF`, `BOOT` and the rest - can come from DRI's `PLM_WORK/X0100.ASM`,
  linked unmodified, as DRI linked it (see [Calling
  Convention](#calling-convention)); a banked resident process's from
  `UTIL2/BRSPBI.ASM`.  `??BDOS`, `??BOOT` and `??MAXB` remain the compiler's
  own.
- Entry preamble (auto-generated), one three-byte instruction as DRI's PL/M-80
  emitted — DRI's sources enter themselves by a jump to `.start-3`:

  ```asm
  ld   sp,??STACK   ; 512-byte stack carried in the image, before the variables
  ```
- `AT(.MEMORY)` storage lies past the image; give the `.PRL` the memory it
  needs with `ul80 --extra`, as DRI did with GENMOD's third argument.

### Memory Layout

A module is laid out the way Intel's PL/M-80 lays a program out, with the
variables last:

- **Code segment (`cseg`):** a BARE or MP/M module's own `DATA` first (DRI's
  programs begin with the `jump byte data (0c3h)` they enter themselves by),
  then the entry code, the procedures and the runtime routines, then every
  constant: string literals, `.(...)` lists, `DATA` declared in a procedure
  and, in CP/M mode, the module's own `DATA`, which there must not come
  before the entry code at 100H.
- **Data segment (`dseg`):** `??AUTO`, the procedures' shared locals; then the
  stack of BARE and MP/M modes (`??STACK`); then the variables, in the order
  the source declares them, a procedure's static locals among them.

`ul80` places every module's data segment after all the code segments,
including those of the runtime modules linked after it, so nothing follows a
program's last variable and `.MEMORY` (the linker's `__END__`) is one past it.
DRI's programs rely on that: MP/M II's `SUB.PLM` and `MSPL.PLM` use everything
from their last variable up to `MAXB` as a buffer.

### Procedure locals

PL/M-80 allocates a procedure's variables statically (Programming Manual,
8.1.7): a local keeps its value from one call to the next, and a program may
count on it - a first-time flag, a running count.  uplm80 saves memory by
overlaying the storage of procedures that are never active at the same time,
`??AUTO`, but only for what no call can see the old value of:

- **Parameters**, which every call assigns - the procedure's own entry
  stores the arguments it is passed (see [Calling
  Convention](#calling-convention)) - unless a rule below makes one static.
- **A local that every call assigns before anything reads it.**  The
  compiler follows each procedure's statements, GOTOs included, and counts
  a call of a nested procedure that names the local as a read of it at the
  call; a use of a BASED variable reads its base.  An array or structure is
  assigned once every element has been, through constant subscripts.

Every other local is static, `@proc$name` among the variables:

- one that may be read before it is assigned;
- one whose address is taken (`.x`, in a statement, an `AT` or an
  `INITIAL`) - a parameter too;
- one named by a procedure nested in its own whose address is taken, or
  by anything that procedure calls: a call through the address can come
  when the procedure the local belongs to is not active;
- one reached outside its bounds through a subscript: a scalar with any
  subscript but `(0)`, an array (or an array member of a structure) with a
  constant subscript past its end, and a one-element array with any
  subscript but a constant 0.  A subscript is a constant when it folds to
  one - `a(1+1)`, `a(-1)` (which is `a(255)`), `a(LAST(a))` - and it is
  decided the same way at every optimization level;
- one declared with a static local in a factored declaration (6.2.4 makes
  those contiguous).

A local with `INITIAL` or `PUBLIC` is static in any case; a `REENTRANT`
procedure's are on the stack.

DRI lays out what a procedure's text declares in the order it declares it:
its parameters and locals, and among them the parameters and locals of the
procedures nested in it and the variables of its `DO` blocks, where the
text has them (MP/M II's SUBMIT has FILLRBUFF's `ssbp` at 0E7AH, the
parameter of PUTRBUFF, declared next, at 0E7BH, and `reading`, which
FILLRBUFF declares after PUTRBUFF, at 0E7CH).  uplm80 keeps that order - the
static locals among the variables in the order of the text, the others in
the procedure's frame in `??AUTO`, the parameters first, as the
`PROCEDURE` statement lists them - wherever a program can tell:

- every local declared after one whose address is taken, or which is
  reached outside its bounds, is static too: a pointer run on from it
  reaches them;
- an array or structure subscripted by anything but a constant may run on
  into the locals declared after it, so from the first such array or
  structure on, a procedure's locals are all static or all in `??AUTO`:
  if any of them is static (an `INITIAL` one included), all of them are.
  A read through such a subscript counts as a read of every local it can
  run on into;
- a frame in `??AUTO` holds only its procedure's own locals, so where a
  procedure with parameters or locals, or a `DO` block with variables, is
  declared after the local that either of those starts from, what comes
  after that local is static, the nested procedure's storage too, in the
  order of the text.

A subscript or pointer that runs backwards, before the variable, is not
covered, nor is one that runs past a procedure's last local, or past a
module-level variable into the procedures declared after it.

The analysis is in `uplm80/local_storage.py`.

Two procedures' frames in `??AUTO` overlap only if the procedures are never
active at the same time, which the call graph decides.  It has the calls
the program's text does not name as well: a `CALL` through an address
(8.2.1) may call any procedure whose address is taken; an `EXTERNAL`
procedure may call back any `PUBLIC` procedure and any whose address is
taken (a call of `MON1` or `MON2` with a constant function is a call of the
BDOS, which calls nothing back); and an `INTERRUPT` procedure, and
everything it calls, may run while any procedure is active, so their
frames overlap no other.  A call's arguments need no edge: they are in
registers or on the stack until the callee's entry stores them, when its
caller is active too, so a procedure called while they are evaluated may
overlay the callee's frame.

## Testing against Intel's PL/M-80

`scripts/intel_oracle.py` is a differential test with Intel's own compiler as
the oracle.  It builds a program - a CP/M main program that prints through
MON1/MON2 - with Intel's PL/M-80 V3.1 the way Digital Research's `P.SUB`
built its CP/M programs:

```
PLM80 T.PLM DEBUG PAGEWIDTH(80)
LINK T.OBJ,X0100,PLM80.LIB TO T.MOD
LOCATE T.MOD CODE(0100H) STACKSIZE(1024)
OBJCPM T
```

and with uplm80 at `-O0` to `-O3` (linked with the same `X0100` equates),
runs every build under cpmemu with a time limit, and compares what each
prints with what Intel's build prints.  Each program is `same`, `differs`
(both outputs are shown, and where they first part), `intel-rejects`,
`uplm80-rejects` or `timeout`.

```bash
make -C tools/isis                                   # the ISIS-II emulator, once
python3.14 scripts/intel_oracle.py prog.plm          # programs of your own
python3.14 scripts/intel_oracle.py --random 300      # generated programs
python3.14 scripts/intel_oracle.py --corpus          # tests/ and sample_code/
python3.14 scripts/intel_oracle.py --random 50 --first 1000 --reduce --keep out/
python3.14 -m pytest tests/test_intel_oracle.py
```

`tests/test_intel_oracle.py` is part of the suite: `python3.14 -m pytest`
runs it with the rest where Intel's binaries are found (building
`tools/isis` first if it can), and skips it where they are not, as on CI,
building nothing; the oracle's own pieces that need no tools - the source
normalisation, the halt patch, the skip - are checked either way.  It
checks three generated programs and one of `tests/`, that a known
difference is seen and a V3.1 rejection is, and each program of a
release's tests whose output the release transcribed from Intel's build.

Intel's binaries are not part of this repository.  The oracle finds them
through `--tools DIR` or `$PLM80_TOOLS`, or on DRI's MP/M II work disk at
`~/src/mpm2/mpm2_external/mpm2src/PLM_WORK` (`PLM80`, `PLM80.OV0`-`OV4`,
`PLM80.LIB`, `LINK`, `LINK.OVL`, `LOCATE`, `X0100`, `OBJCPM.COM`), and when
they are missing it says so and checks nothing, as the pytest skips.  It runs
them on `tools/isis/isis`, a small ISIS-II emulator (`tools/isis/isis.cc`)
on cpmemu's qkz80 core, which `make -C tools/isis` finds the way romwbw_emu
does (pkg-config, else a `../cpmemu` checkout beside this one); without it,
on romwbw_emu's `tools/romwbw-plm80`, which runs the same recipe under DRI's
ISX, some ten times slower.  `OBJCPM` is a CP/M program and runs on cpmemu.

Two things the recipe needs that DRI's own programs did for themselves:

- LOCATE starts a program behind its constants (DATA, and strings in
  `.(...)`), not at 0100H, where CP/M enters it; DRI's programs begin with a
  DATA jump.  A program that does not start at 0100H is located again at
  0103H, where OBJCPM puts a jump to the start in front of it.
- A PL/M-80 main program ends in `EI; HLT`, which cpmemu runs past.  The
  HLT at the module's last statement (found through the LINES records
  `DEBUG` writes) becomes `RST 0`, a warm boot, as uplm80's CP/M mode ends.

A module with no statements of its own, entered through a DATA jump that
Intel's layout puts at 0100H - DRI's CP/M 2.0 `LOAD`, `STAT` and `SUBMIT` -
is built with uplm80's `-m bare`, which lays it out the same way (`--mode`
chooses otherwise).

`--normalize` rewrites three spellings only uplm80 takes into their Intel
equivalents before both compilers see the program: `.'string'` into
`.('string')`, an untyped `DATA` into `BYTE DATA`, and a program with no
`name: DO;` into a module.  `--reduce` cuts a program that differs, or that
uplm80 rejects, down to the statements, then the lines, the difference needs
(Zeller's ddmin); the result needs tidying by hand before it is a test.

The generator, `tests/plm_intel.py`, writes programs in the dialect both
compilers share, with no behaviour PL/M-80 leaves undefined but two, which
V3.1 defines and uplm80 follows: division by zero (what Intel's divide
routine returns) and the carry of PLUS, MINUS, SCL and SCR in the form where
the statement itself sets it.  It covers BYTE and ADDRESS arithmetic,
relations and logic, the built-ins, arrays, structures, BASED variables,
DATA/INITIAL, LITERALLY, strings, MOVE, LENGTH/LAST/SIZE, IF, DO CASE, DO
WHILE, iterative DO (with limits and steps the body changes), nested
procedures with 0 to 5 parameters, typed and untyped, and REENTRANT
recursion.  Evaluation order is kept out of what a program prints.  Each printed line starts with the number of the statement
that printed it, which the source carries as `/* S1F */`.

### Known differences from Intel's PL/M-80 V3.1

The generator can leave out those with a name (`--avoid NAME,...`), and the
pytest does.  In the first campaign - 1,200 generated programs and the 67
programs of `tests/` and `sample_code/`, at `-O0` to `-O3` - every difference
was one of these, and none depended on the optimization level; one more,
LENGTH, LAST and SIZE of a qualified reference, which uplm80 rejected, it
takes since 0.4.2 (`qualsize` no longer leaves them out).

| `--avoid` | Program | Intel V3.1 | uplm80 | |
|---|---|---|---|---|
| `shl-byte` | `b = 0F0H; w = SHL(b, 4);` | `0000` | `0F00H` | uplm80 shifts a BYTE in 16 bits, with an ADDRESS result (uplm80/plm_types.py); the manual (11.1.4) and Intel make it a BYTE |
| `shift9` | `b = 7; r = SHL(b, 9);` (r BYTE) | `0EH` | `0` | V3.1 shifts a BYTE by the count mod 8 (0 counts as 8), folded or not; the manual: 0 |
| `wide-limit` | `DO i = 250 TO 300;` (i BYTE) | 6 times | never | V3.1 compares the BYTE index with the 16-bit limit; the manual (5.1.4) converts the limit to the index's type |
| `sub-zero` | `w = (b - DOUBLE(0)) + 0F0H;` (b = 0F0H) | `00E0H` | `01E0H` | V3.1 drops `- 0` and with it the ADDRESS type (also `b - (c * 0)`); the manual (4.2.1): ADDRESS |
| `zero-dividend` | `z = 0; w = 0 / z;` | `0` | `0FFFFH` | V3.1 folds `0 / x` to 0; uplm80 divides, and a division by 0 gives what Intel's own divide routine gives (4.2.3: undefined) |
| `neg-widened` | `b = 0E7H; w = 0FFFEH; v = (b - w) + (-b);` | `0002` | `0102H` | V3.1 negates `b` in 16 bits when `b - w` has already widened it (the manual: `-b` is a BYTE, 19H) |
| | `w = 1; v = HIGH(0FH + w) >= (ROR(SIZE(aw), 2) AND w);` (aw(8) ADDRESS) | `0` | `0FFH` | V3.1 makes 10H from the 0FH in C with `MOV A,C` and `INX SP` (its listing: `INX PSW`) where it means `INR A`: the value is one short and SP one long.  `w1, b2 = b2;` as a program's first statement (b1-b4 BYTE, w1-w4 ADDRESS) is coded `INX H; INX SP; MOV M,A`, and the next CALL overwrites `b1` |
| | `CALL MOVE(0, .s, .d);` | moves 65536 bytes | moves none | Intel's MOVE counts down before it tests |

V3.1 also rejects what only uplm80 takes: `.'string'` (ERROR 101), an
untyped `DATA` (61), a program that is not a module (89), a declaration
after a statement (26), `NOT NOT x` (102), and a `CALL` of a typed
procedure (129); and what the CHANGELOG lists under Known issues: a
subscript on a scalar (127), an array without a subscript (133, 134), a
procedure with no statements (174), and a call of a procedure its block
declares after the call (169).  uplm80 compiles, with a warning that
names V3.1's error, `f()` and `CALL g()` of a procedure (102, 153) and
`INITIAL` in a procedure or a DO block (73), on which programs written
for it rely.  It rejects, as V3.1 does, a name declared nowhere, empty
parentheses after a variable or a built-in, `.label` in an expression, an
`INTERRUPT` procedure below module level, a parameter no `DECLARE`
declares, and a `LITERALLY` used before its declaration (since 0.4.1);
and a dimension of 0, the address of a built-in but MEMORY, and a PUBLIC
or EXTERNAL procedure or variable below module level (since 0.4.2).

A store through a pointer or an overrun in or from `??AUTO` reaches what
uplm80's layout puts there, not what DRI's does (CHANGELOG, Known issues),
which the generator never writes.  Since 0.4.2 uplm80 compiles a label on
an `END` statement, `out: end p;` (A.4.4.1), as V3.1 does, and a counted
loop over the last module-level variable sees a store through MEMORY that
reaches it.

## Project Structure

```
uplm80/
├── compiler.py      # Main compiler driver / CLI
├── frontend.py      # plox-driven lexer + LR parse → AST
├── preprocess.py    # PL/M preprocessor ($INCLUDE, $if, LITERALLY, ...)
├── ast_nodes.py     # AST definitions
├── ast_optimizer.py # AST-level optimizations
├── codegen.py       # Z80 code generator
├── local_storage.py # Which procedure locals may share ??AUTO
├── names.py         # Which declaration each name means; GOTO rules; assembler names
├── runtime.py       # Runtime helpers
├── symbols.py       # Symbol table
├── errors.py        # Diagnostic exception types
└── data/            # Pre-built plox grammar bundle (plm_full.json)
```

Peephole optimization is provided by the external [upeepz80](https://github.com/avwohl/upeepz80) package; the front-end is generated from [plox](https://github.com/avwohl/plox) grammars (`plm_pre` + `plm_full`) and loaded at import time from the JSON bundle in `data/`.

## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
## Related Projects

- [80un](https://github.com/avwohl/80un) - Unpacker for the CP/M archive and compression formats LBR, ARC, squeeze, crunch, and CrLZH.
- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android phones and tablets. It emulates the RomWBW HBIOS interface and a VT100 terminal.
- [cpmemu](https://github.com/avwohl/cpmemu) - Z80/CP/M emulator for Linux and Windows, with Z80 and 8080 CPU cores. It translates the BDOS and BIOS calls of CP/M 2.2 programs to the host file system.
- [ioscpm](https://github.com/avwohl/ioscpm) - Z80/CP/M emulator for iOS and macOS. It emulates the RomWBW HBIOS interface and runs CP/M 2.2 and CP/M 3.
- [learn-ada-z80](https://github.com/avwohl/learn-ada-z80) - Collection of more than 90 Ada example programs for uada80, the Ada compiler for the Z80 processor and CP/M.
- [mbasic](https://github.com/avwohl/mbasic) - Python interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. Two compiler backends compile the programs to CP/M .COM files or to JavaScript.
- [mbasic2025](https://github.com/avwohl/mbasic2025) - Reconstruction of the lost source code of MBASIC 5.21, the Microsoft BASIC-80 for CP/M. The MACRO-80 source code assembles to a binary that matches mbasic.com byte for byte.
- [mbasicc](https://github.com/avwohl/mbasicc) - C++17 interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. It runs on Linux and macOS.
- [mbasicc_web](https://github.com/avwohl/mbasicc_web) - Web browser interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. Emscripten compiles the mbasicc interpreter to WebAssembly.
- [mpm2](https://github.com/avwohl/mpm2) - Z80 emulator for MP/M II, the multi-user CP/M operating system. Users connect over SSH, and SFTP clients transfer files.
- [romwbw_emu](https://github.com/avwohl/romwbw_emu) - Hardware-level Z80/CP/M emulator for Linux and macOS. It emulates the RomWBW HBIOS interface and switches banks in 512 KB of ROM and 512 KB of RAM.
- [scelbal](https://github.com/avwohl/scelbal) - Floating-point BASIC interpreter for the 8080 processor and CP/M. A translator converts the original 8008 source code to 8080 source code.
- [uada80](https://github.com/avwohl/uada80) - Ada compiler for the Z80 processor and CP/M 2.2. It compiles a subset of Ada 2012 to CP/M .COM files.
- [uc80](https://github.com/avwohl/uc80) - C compiler for the Z80 processor and CP/M. It optimizes for small code size.
- [ucow](https://github.com/avwohl/ucow) - Cowgol compiler for the Z80 processor and CP/M. It runs on Linux in Python.
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Linux toolchain that is compatible with Microsoft MACRO-80. It has an assembler, a linker, a librarian, and a disassembler.
- [upeepz80](https://github.com/avwohl/upeepz80) - Peephole optimizer for Z80 compilers that write lowercase Z80 assembly language. It shortens jumps to jr, builds djnz loops, and removes dead stores.
- [uplox](https://github.com/avwohl/uplox) - LR(1) and GLR parser generator. It writes a lexer DFA, an LR parser, and a typed automatic AST, and uplm80 uses it to parse PL/M-80.
- [z80cpmw](https://github.com/avwohl/z80cpmw) - Z80/CP/M emulator for Windows. It emulates the RomWBW HBIOS interface and boots CP/M from disk images.

