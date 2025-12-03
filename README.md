# uplm80 - PL/M-80 Compiler

[![PyPI version](https://badge.fury.io/py/uplm80.svg)](https://pypi.org/project/uplm80/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A modern PL/M-80 compiler targeting Intel 8080 and Zilog Z80 assembly language.

PL/M-80 was the primary systems programming language for CP/M and other 8080/Z80 operating systems. This compiler can rebuild original CP/M utilities from their PL/M source code.

**Repository:** https://github.com/avwohl/uplm80

## Features

- Full PL/M-80 language support
- Targets both 8080 and Z80 instruction sets
- Multiple optimization passes (peephole, post-assembly tail merging)
- Generates relocatable object files compatible with standard CP/M linkers
- Produces code competitive with the original Digital Research compiler

## Code Quality

Compiled output is comparable to the original Digital Research PL/M-80 compiler:

| Program | DR PL/M-80 | uplm80 | Difference |
|---------|------------|--------|------------|
| PIP.COM | 7424 bytes | 7136 bytes | -3.9% |

## Installation

Install from PyPI:

```bash
pip install uplm80
```

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
- `-t 8080` or `-t z80` - Target CPU (default: Z80)
- `-o output.mac` - Output file name

### Post-Assembly Optimization (Optional)

```bash
python -m uplm80.postopt output.mac -o output_opt.mac
```

Performs multi-pass tail merging and skip trick optimizations.

### Assemble and Link

Use your preferred 8080/Z80 assembler and linker. Example with um80/ul80:

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

```
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

## CP/M Programs

For CP/M programs, provide stubs for:

- `MON1`, `MON2`, `MON3` - BDOS calls
- `BOOT` - Warm boot
- `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE` - System variables

## Project Structure

```
uplm80/
├── compiler.py    # Main compiler driver
├── lexer.py       # Tokenizer
├── parser.py      # PL/M-80 parser
├── ast_nodes.py   # AST definitions
├── codegen.py     # Code generator
├── peephole.py    # Peephole optimizer
├── postopt.py     # Post-assembly optimizer
└── symbols.py     # Symbol table
```

## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Digital Research for creating PL/M-80 and CP/M
- The CP/M source code preservation efforts that made the original PL/M sources available
