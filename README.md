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
| PIP.COM | 7424 bytes | 7127 bytes | -4.0% |

## Installation

Quick install from PyPI:

```bash
pip install uplm80 um80 upeep80
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
- `-t 8080` or `-t z80` - Target CPU (default: Z80)
- `-m cpm` or `-m bare` - Runtime mode (default: cpm)
  - `cpm`: CP/M program with stack from BDOS, returns to OS on exit
  - `bare`: Bare metal program with local stack, original Intel PL/M style
- `-o output.mac` - Output file name
- `-O 0|1|2|3` - Optimization level (default: 2)
- `-D SYMBOL` - Define conditional compilation symbol (can be repeated)

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

For CP/M programs that run under the CP/M operating system:

- Program starts with `ORG 100H` (CP/M TPA)
- Stack is set from BDOS address at location 0006H: `LD HL,(6)` / `LD SP,HL`
- Entry code calls main procedure with `CALL MAIN`
- Returns to CP/M with `JP 0` (warm boot) when main returns
- Requires CP/M stubs: `MON1`, `MON2`, `MON3`, `BOOT`
- System variables: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`

### Bare Metal Mode (`-m bare`)

For standalone programs or original Intel PL/M compatibility:

- Uses locally-defined stack (64 bytes by default)
- Entry code sets `SP` to `??STACK` label (top of local stack buffer)
- Compatible with original Intel PL/M-80 programs
- Program can define custom entry point via DATA declarations
- No OS return code (program may need explicit HALT or loop)

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

- Intel for creating PL/M-80
- Digital Research for creating CP/M
- The CP/M source code preservation efforts that made the original PL/M sources available
