# Runtime Modes, Memory Layout and Procedure Locals

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
  Convention](calling_convention.md)); a banked resident process's from
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
  Convention](calling_convention.md)) - unless a rule below makes one static.
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
