# PL/M-80 Compiler (uplm80)

A PL/M-80 compiler targeting Z80 assembly.

## Tools

### um80 - Universal Macro Assembler for 8080
Assembles .mac files to Intel .rel relocatable object format.

```bash
um80 input.mac                    # Creates input.rel
um80 input.mac -o output.rel      # Specify output name
```

### ul80 - Universal Linker for 8080
Links .rel files to create CP/M .com executables.

```bash
ul80 -o output.com main.rel lib.rel    # Link multiple .rel files
ul80 -p 100 -o output.com input.rel    # Set origin (default 0x100 for CP/M)
ul80 -s input.rel                      # Generate .sym symbol file
```

### cpmemu - CP/M 2.2 Emulator
Runs CP/M .com programs under Linux.

```bash
cpmemu program.com                # Run CP/M program
cpmemu program.com arg1 arg2      # Run with arguments
```

## Build Process

1. Compile PL/M-80 source to assembly:
   ```bash
   python -m uplm80.compiler input.plm -o output.mac           # CP/M mode (default)
   python -m uplm80.compiler input.plm -m bare -o output.mac   # Bare metal mode
   ```

   **Multi-file compilation** (for cross-module optimization):
   ```bash
   python -m uplm80.compiler main.plm helper.plm lib.plm -o output.mac
   ```
   All files are parsed together, building a unified call graph for optimal local variable storage allocation across module boundaries.

   Options:
   - `-m cpm` - CP/M mode (default): For new PL/M programs, maximum stack under BDOS
   - `-m bare` - Bare metal mode: Original Digital Research compatible (jump to start-3)
   - `-O 0|1|2|3` - Optimization level (default: 2)
   - `-D SYMBOL` - Define conditional compilation symbol (can be repeated)

2. Assemble to relocatable object:
   ```bash
   um80 output.mac
   ```

3. Link with runtime library:
   ```bash
   ul80 -o output.com output.rel runtime.rel
   ```

## Runtime Library

The compiler generates code that uses these runtime routines (must be provided in a runtime.rel):

- `??move` - Block memory move
- `??div16` - 16-bit unsigned division
- `??mul16` - 16-bit unsigned multiplication
- `??mod16` - 16-bit unsigned modulo
- `??subde` - 16-bit subtraction (HL = HL - DE)

## Runtime Modes

### CP/M Mode (`-m cpm`, default)

For new PL/M programs. Provides maximum stack space by using the area under BDOS:

**Entry Point Code:**
```asm
org 100h              ; CP/M TPA start
ld hl,(6)             ; Get BDOS address from location 6
ld sp,hl              ; Set stack to top of TPA (maximum stack)
call main             ; Call main procedure
jp 0                  ; Return to CP/M (warm boot)
```

**Features:**
- Maximum available stack (all memory between program end and BDOS)
- Clean return to CP/M on program exit
- Requires CP/M stubs: `mon1`, `mon2`, `mon3`, `boot`
- Requires memory locations: `bdisk`, `maxb`, `fcb`, `buff`, `iobyte`

### Bare Metal Mode (`-m bare`)

For original Digital Research PL/M-80 compatibility. Programs begin with a jump to start-3, which sets SP to a local stack block:

**Entry Point Code:**
```asm
org 100h              ; Or custom origin
jp ??start            ; Jump to initialization at start-3
ds 64                 ; 64-byte stack buffer
??stack:              ; Label at top of stack (SP set here)
??start:
ld sp,??stack         ; Set stack to local stack buffer
call main             ; Call main procedure
; (falls through - program controls exit behavior)
```

**Features:**
- Compatible with original Digital Research PL/M-80 programs (ED.PLM, PIP.PLM, etc.)
- Local 64-byte stack buffer embedded in program
- Entry point at start-3 (jump over stack area)
- Programs can define custom entry points via DATA declarations
- No automatic return to OS (program controls its own exit)

## Conditional Compilation

PL/M-80 v4.0 conditional compilation. The native form is a **control
line**: a leading `$` at the left margin (column 1), exactly like
`$INCLUDE` / `$TITLE` / `$LIST`. No enabling directive is required.

```plm
$set (MPM)
$if MPM
    /* Code for MP/M */
$elseif CPM3
    /* Code for CP/M 3 */
$else
    /* Code for single-user CP/M 2.2 */
$endif
```

Directives: `$SET (name)`, `$RESET (name)`, `$IF name`, `$ELSEIF name`,
`$ELSE`, `$ENDIF`. `$COND` / `$NOCOND` are *listing* controls only
(whether skipped lines appear in the listing) and are accepted as
no-ops.

A comment-wrapped form `/** $if NAME **/` is also honoured for
CP/M-3-style sources that embed the same directives in comment syntax.

Because the `$` must be at the left margin, a control line is disabled
simply by indenting it into a comment — e.g. MP/M's `GENSYS.PLM` ships
`/* $include (copyrt.lit) */`, which stays a comment.

Command line: `python -m uplm80.compiler input.plm -D MPM -D CPM3`

## CP/M Stubs

For CP/M programs (`-m cpm`), provide stubs for:
- `mon1` - BDOS call (void return)
- `mon2` - BDOS call (byte return)
- `mon3` - BDOS call (address return)
- `boot` - Warm boot
- Memory locations: bdisk, maxb, fcb, buff, iobyte

## Optimizations

### Compiler Optimizations (codegen.py)
- **SHL(DOUBLE(x),8) OR y pattern**: Combines two bytes into 16-bit address efficiently using `ld h,a; ld l,a` instead of 14+ instruction 16-bit OR
- **Z80-specific**: Uses Z80 instructions like `sbc hl,de`, `srl`, indexed addressing with IX

### Peephole Optimizer (upeepz80)
External peephole optimizer library that works on Z80 assembly:
- Register tracking, redundant load elimination, strength reduction
- DJNZ for loops, relative jumps (jr), block instructions
- Tail merging and other cross-procedure optimizations

## Reference Binaries

To compare against original Digital Research binaries, disassemble with ud80:

```bash
~/z80/RomWBW/Source/Images/d_cpm22/u0$ ud80 ED.COM -o ~/real_ed.mac
```
