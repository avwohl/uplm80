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
- Multiple optimization passes (peephole, post-assembly tail merging)
- Generates relocatable object files compatible with standard CP/M linkers
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
- `-m cpm` or `-m bare` - Runtime mode (default: cpm)
  - `cpm`: For new PL/M programs, maximum stack under BDOS
  - `bare`: Original Digital Research compatible (jump to start-3)
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
- Local variable storage (`??AUTO`) is optimally allocated based on which procedures can be active simultaneously across module boundaries
- A single combined output file is generated

This produces better code than compiling files separately, as the compiler can share local variable storage between procedures in different modules that never call each other.

### Assemble and Link

Use your preferred Z80 assembler and linker. Example with um80/ul80:

```bash
um80 output.mac                              # Assemble to .rel
ul80 -o program.com output.rel runtime.rel   # Link to CP/M .com
```

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

The compiler generates calls to these runtime routines (provide in a separate .rel file):

| Routine | Description |
|---------|-------------|
| `??MUL` | 16-bit unsigned multiply |
| `??DIV` | 16-bit unsigned divide |
| `??MOD` | 16-bit unsigned modulo |
| `??SHL` | 16-bit shift left |
| `??SHR` | 16-bit logical shift right |
| `??SHRS` | 16-bit arithmetic shift right |
| `??MOVE` | Block memory move |

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
- Requires CP/M stubs in your runtime: `MON1`, `MON2`, `MON3`, `BOOT`.
- System variables: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`.

### Bare Metal Mode (`-m bare`)

The mode required to rebuild original Digital Research utilities (PIP.PLM,
ED.PLM, etc.) byte-compatibly. These programs follow the Intel PL/M-80
convention of jumping to *start − 3* to skip over a local stack area:

- Entry preamble (auto-generated):

  ```asm
  jp   ??START      ; skip over the stack buffer
  ds   64           ; 64-byte local stack
  ??STACK:          ; top of stack
  ??START:
  ld   sp,??STACK   ; use the local stack
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

## Project Structure

```
uplm80/
├── compiler.py      # Main compiler driver / CLI
├── frontend.py      # plox-driven lexer + LR parse → AST
├── preprocess.py    # PL/M preprocessor ($INCLUDE, $if, LITERALLY, ...)
├── ast_nodes.py     # AST definitions
├── ast_optimizer.py # AST-level optimizations
├── codegen.py       # Z80 code generator
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

- [80un](https://github.com/avwohl/80un) - Unpacker for CP/M compression and archive formats (LBR, ARC, squeeze, crunch, CrLZH)
- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android with RomWBW HBIOS compatibility and VT100 terminal
- [cpmemu](https://github.com/avwohl/cpmemu) - CP/M 2.2 emulator with Z80/8080 CPU emulation and BDOS/BIOS translation to Unix filesystem
- [ioscpm](https://github.com/avwohl/ioscpm) - Z80/CP/M emulator for iOS and macOS with RomWBW HBIOS compatibility
- [learn-ada-z80](https://github.com/avwohl/learn-ada-z80) - Ada programming examples for the uada80 compiler targeting Z80/CP/M
- [mbasic](https://github.com/avwohl/mbasic) - Modern MBASIC 5.21 Interpreter & Compilers
- [mbasic2025](https://github.com/avwohl/mbasic2025) - MBASIC 5.21 source code reconstruction - byte-for-byte match with original binary
- [mbasicc](https://github.com/avwohl/mbasicc) - C++ implementation of MBASIC 5.21
- [mbasicc_web](https://github.com/avwohl/mbasicc_web) - WebAssembly MBASIC 5.21
- [mpm2](https://github.com/avwohl/mpm2) - MP/M II multi-user CP/M emulator with SSH terminal access and SFTP file transfer
- [romwbw_emu](https://github.com/avwohl/romwbw_emu) - Hardware-level Z80 emulator for RomWBW with 512KB ROM + 512KB RAM banking and HBIOS support
- [scelbal](https://github.com/avwohl/scelbal) - SCELBAL BASIC interpreter - 8008 to 8080 translation
- [uada80](https://github.com/avwohl/uada80) - Ada compiler targeting Z80 processor and CP/M 2.2 operating system
- [uc80](https://github.com/avwohl/uc80) - ANSI C compiler targeting Z80 processor and CP/M 2.2 operating system
- [ucow](https://github.com/avwohl/ucow) - Unix/Linux Cowgol to Z80 compiler
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Microsoft MACRO-80 compatible toolchain for Linux: assembler, linker, librarian, disassembler
- [upeepz80](https://github.com/avwohl/upeepz80) - Universal peephole optimizer for Z80 compilers
- [uplox](https://github.com/avwohl/uplox) - Compiler front-end generator (lexer DFA + LR parser + typed auto-AST) that uplm80 uses for PL/M-80 parsing
- [z80cpmw](https://github.com/avwohl/z80cpmw) - Z80 CP/M emulator for Windows (RomWBW)

