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

```bash
pip install uplm80 um80 upeepz80
```

From source: `git clone https://github.com/avwohl/uplm80.git && cd uplm80 && pip install -e .`
See [INSTALL.md](INSTALL.md) and [README_RASPBERRY_PI.md](README_RASPBERRY_PI.md) for more.

## Usage

```bash
uplm80 input.plm -o output.mac                  # compile (or: python -m uplm80.compiler)
uplm80 main.plm helper.plm lib.plm -o out.mac   # multi-file, cross-module optimization
um80 output.mac                                 # assemble to .rel
um80 x0100.asm                                  # MON1 equ 5 and the rest
ul80 -o program.com output.rel x0100.rel        # link to CP/M .com
```

Options:
- `-m cpm|bare|mpm` - runtime mode (default `cpm`); `bare` rebuilds DRI's utilities, `mpm` makes MP/M II `.PRL`/`.RSP`/`.SPR` modules
- `-O 0|1|2|3` - optimization level (default 2)
- `-D SYMBOL` - define a conditional compilation symbol (repeatable)

Each module carries the runtime routines it uses; link with it only what
defines its EXTERNALs. See [examples/hellocpm.plm](examples/hellocpm.plm)
and the drop-in [docs/example.Makefile](docs/example.Makefile) (contributed
by Martin Homuth-Rosemann, [@Ho-Ro](https://github.com/Ho-Ro)).

## Documentation

- [docs/language.md](docs/language.md) - rules the compiler checks, as Intel's V3.1 does; `$IF`/`$SET` conditional compilation
- [docs/calling_convention.md](docs/calling_convention.md) - calling convention, writing assembly PL/M calls, runtime routines, names in the output
- [docs/runtime_modes.md](docs/runtime_modes.md) - CP/M, bare and MP/M modes, memory layout, how procedure locals share `??AUTO`
- [docs/multi_file_compilation.md](docs/multi_file_compilation.md) - compiling several modules together
- [docs/intel_oracle.md](docs/intel_oracle.md) - differential testing against Intel's PL/M-80 V3.1, and the known differences
- [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md) - calling CP/M BDOS from PL/M-80
- [docs/project_structure.md](docs/project_structure.md) - source files and what each does
- [docs/PLM80_FEATURE_INDEX.md](docs/PLM80_FEATURE_INDEX.md) - language features and their tests
- [CHANGELOG.md](CHANGELOG.md) - releases and known issues

The front end is generated from [plox](https://github.com/avwohl/plox)
grammars; peephole optimization is [upeepz80](https://github.com/avwohl/upeepz80).

## License

GPL v3.0 or later - see [LICENSE](LICENSE). Issues and pull requests are welcome.

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

