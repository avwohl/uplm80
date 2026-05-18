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

See [examples/hello_cpm.plm](examples/hello_cpm.plm) for a complete working example.

For more on CP/M BDOS usage, see [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md).

## Conditional Compilation

Later versions of PL/M-80 added conditional compilation directives embedded in comments. This allows the same source to be compiled for different configurations (e.g., CP/M 2.2 vs CP/M 3, single-user vs MP/M).

### Directives

| Directive | Description |
|-----------|-------------|
| `/** $set (NAME) **/` | Define a symbol |
| `/** $reset (NAME) **/` | Undefine a symbol |
| `/** $cond **/` | Enable conditional compilation |
| `/** $if NAME **/` | Include following code if NAME is defined |
| `/** $else **/` | Else branch |
| `/** $endif **/` | End conditional block |

### Example

```plm
/** $set (CPM3) **/
/** $cond **/

DECLARE
/** $if CPM3 **/
    VERSION LITERALLY '30H',
/** $else **/
    VERSION LITERALLY '22H',
/** $endif **/
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

### CP/M Mode (default: `-m cpm`)

For new PL/M programs. Provides maximum stack space by using the area under BDOS:

- Program starts with `ORG 100H` (CP/M TPA)
- Stack is set from BDOS address at location 0006H: `LD HL,(6)` / `LD SP,HL`
- Maximum available stack (all memory between program end and BDOS)
- Entry code calls main procedure with `CALL MAIN`
- Returns to CP/M with `JP 0` (warm boot) when main returns
- Requires CP/M stubs: `MON1`, `MON2`, `MON3`, `BOOT`
- System variables: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`

### Bare Metal Mode (`-m bare`)

For original Digital Research PL/M-80 compatibility. Programs begin with a jump to start-3:

- Entry begins with `JP ??START` to jump over local stack buffer
- Uses locally-defined stack (64 bytes in program image)
- Entry code at `??START` sets `SP` to `??STACK` label, then calls `MAIN`
- Compatible with original Digital Research programs (ED.PLM, PIP.PLM, etc.)
- Program can define custom entry point via DATA declarations
- No automatic OS return (program controls its own exit behavior)

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

