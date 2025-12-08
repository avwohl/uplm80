# PL/M-80 Compiler (uplm80)

A PL/M-80 compiler targeting 8080/Z80 assembly.

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

   Options:
   - `-m cpm` - CP/M mode (default): For new PL/M programs, maximum stack under BDOS
   - `-m bare` - Bare metal mode: Original Digital Research compatible (jump to start-3)
   - `-t z80` or `-t 8080` - Target processor (default: z80)
   - `-O 0|1|2|3` - Optimization level (default: 2)
   - `-D SYMBOL` - Define conditional compilation symbol (can be repeated)

2. (Optional) Run post-assembly optimizer:
   ```bash
   python -m uplm80.postopt output.mac -o output_opt.mac
   ```

3. Assemble to relocatable object:
   ```bash
   um80 output.mac
   ```

4. Link with runtime library:
   ```bash
   ul80 -o output.com output.rel runtime.rel
   ```

## Runtime Library

The compiler generates code that uses these runtime routines (must be provided in a runtime.rel):

- `??MOVE` - Block memory move
- `??SHL` - 16-bit shift left
- `??SHR` - 16-bit shift right (logical)
- `??SHRS` - 16-bit shift right (arithmetic)
- `??DIV` - 16-bit unsigned division
- `??MUL` - 16-bit unsigned multiplication
- `??MOD` - 16-bit unsigned modulo

## Runtime Modes

### CP/M Mode (`-m cpm`, default)

For new PL/M programs. Provides maximum stack space by using the area under BDOS:

**Entry Point Code:**
```asm
ORG 100H              ; CP/M TPA start
LD HL,(6)             ; Get BDOS address from location 6
LD SP,HL              ; Set stack to top of TPA (maximum stack)
CALL MAIN             ; Call main procedure
JP 0                  ; Return to CP/M (warm boot)
```

**Features:**
- Maximum available stack (all memory between program end and BDOS)
- Clean return to CP/M on program exit
- Requires CP/M stubs: `MON1`, `MON2`, `MON3`, `BOOT`
- Requires memory locations: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`

### Bare Metal Mode (`-m bare`)

For original Digital Research PL/M-80 compatibility. Programs begin with a jump to start-3, which sets SP to a local stack block:

**Entry Point Code:**
```asm
ORG 100H              ; Or custom origin
JP ??START            ; Jump to initialization at start-3
DS 64                 ; 64-byte stack buffer
??STACK:              ; Label at top of stack (SP set here)
??START:
LXI SP,??STACK        ; Set stack to local stack buffer
CALL MAIN             ; Call main procedure
; (falls through - program controls exit behavior)
```

**Features:**
- Compatible with original Digital Research PL/M-80 programs (ED.PLM, PIP.PLM, etc.)
- Local 64-byte stack buffer embedded in program
- Entry point at start-3 (jump over stack area)
- Programs can define custom entry points via DATA declarations
- No automatic return to OS (program controls its own exit)

## Conditional Compilation

Supports conditional compilation directives embedded in comments (as used in CP/M 3 sources):

```plm
/** $set (MPM) **/      /* Define symbol */
/** $reset (CPM3) **/   /* Undefine symbol */
/** $cond **/           /* Enable conditional compilation */

/** $if MPM **/
    /* Code for MP/M */
/** $else **/
    /* Code for single-user CP/M */
/** $endif **/
```

Command line: `python -m uplm80.compiler input.plm -D MPM -D CPM3`

## CP/M Stubs

For CP/M programs (`-m cpm`), provide stubs for:
- `MON1` - BDOS call (void return)
- `MON2` - BDOS call (byte return)
- `MON3` - BDOS call (address return)
- `BOOT` - Warm boot
- Memory locations: BDISK, MAXB, FCB, BUFF, IOBYTE

## Optimizations

### Compiler Optimizations (codegen.py, peephole.py)
- **SHL(DOUBLE(x),8) OR y pattern**: Combines two bytes into 16-bit address efficiently using `LD H,A; LD L,A` instead of 14+ instruction 16-bit OR
- **Peephole optimizer**: Register tracking, redundant load elimination, strength reduction
- **Z80-specific**: DJNZ for loops, relative jumps, block instructions

### Post-Assembly Optimizer (postopt.py)
Multi-pass optimizer that works on generated assembly:
- **Tail merging**: Finds procedures with identical tail sequences and merges them using the DB 21H (LD HL,nn) skip trick to fall through to shared code
- **Skip trick**: Adjacent procedures ending with JP can skip 2-byte instructions using DB 21H

Example: If 4 procedures end with `LD DE,0; JP 5`, three are rewritten to fall through to the fourth, saving ~15 bytes.

## Reference Binaries

To compare against original Digital Research binaries, disassemble with ud80:

```bash
~/z80/RomWBW/Source/Images/d_cpm22/u0$ ud80 ED.COM -o ~/real_ed.mac

