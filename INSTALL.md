# Installation Guide

This guide covers installing uplm80 and the complete PL/M-80 development toolchain.

## Platform-Specific Guides

- **Raspberry Pi**: See [README_RASPBERRY_PI.md](README_RASPBERRY_PI.md) for Raspberry Pi specific instructions
- **Other Linux/macOS/Windows**: Follow the general instructions below

## Quick Install from PyPI

The easiest way to install uplm80 and the complete toolchain:

```bash
# Install from PyPI
pip install uplm80 um80 upeep80

# Verify installation
uplm80 --version
um80 --version
ul80 --version
```

## Install from Source (Development)

For development or to get the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/avwohl/uplm80.git
cd uplm80

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install associated tools
pip install um80 upeep80

# Verify installation
uplm80 --version
```

## Complete Installation with Virtual Environment

This is the recommended approach, especially for testing or development:

```bash
# Create a dedicated virtual environment
python3 -m venv myuplm80
cd myuplm80
source bin/activate  # On Windows: Scripts\activate

# Install all tools
pip install --upgrade pip
pip install uplm80 um80 upeep80

# Verify installation
uplm80 --version
um80 --version
ul80 --version
```

## Toolchain Components

The complete PL/M-80 development toolchain consists of:

### 1. uplm80 (PL/M-80 Compiler)
Compiles PL/M-80 source code to 8080/Z80 assembly language.

```bash
pip install uplm80
```

Features:
- Full PL/M-80 language support
- Targets both 8080 and Z80
- Multiple optimization passes
- CP/M and bare metal modes

### 2. um80 (Universal Macro Assembler)
Assembles .mac assembly files to Intel .rel relocatable object format.

```bash
pip install um80
```

Features:
- Supports 8080 and Z80 instruction sets
- MACRO-80 compatible
- Outputs Intel .rel format

### 3. ul80 (Universal Linker)
Included with um80. Links .rel object files to create CP/M .com executables.

Features:
- Links multiple .rel files
- Resolves external references
- Sets program origin
- Generates symbol files

### 4. upeep80 (Peephole Optimizer Library)
Automatically installed as a dependency of uplm80.

Features:
- Language-agnostic assembly optimizer
- Used internally by uplm80
- Can be used standalone

## Testing Your Installation

Create a simple test program to verify everything is working:

```bash
# Create hello_cpm.plm
cat > hello_cpm.plm << 'EOF'
/*
 * Hello World for CP/M
 */

0100H:

MON1: PROCEDURE(FUNC, PARM) EXTERNAL;
    DECLARE FUNC BYTE, PARM ADDRESS;
END MON1;

MAIN: PROCEDURE;
    DECLARE MESSAGE DATA ('Hello, World!', 13, 10, '$');
    CALL MON1(9, .MESSAGE);
END MAIN;

CALL MAIN;
EOF

# Compile PL/M to assembly
uplm80 hello_cpm.plm -o hello.mac

# Check output
ls -l hello.mac

# Assemble to relocatable object
um80 hello.mac

# Check output
ls -l hello.rel

# Link to CP/M executable
ul80 -o hello.com hello.rel

# Check final executable
ls -l hello.com
```

You should see three output files:
- `hello.mac` - Assembly language source
- `hello.rel` - Relocatable object file
- `hello.com` - CP/M executable

## Installing the CP/M Emulator (Optional)

To actually run and test your compiled programs, install the CP/M emulator:

### Linux (x86_64/ARM64)

```bash
# Download latest release for your architecture
# Visit: https://github.com/avwohl/cpmemu/releases/latest

# For x86_64:
curl -sL https://github.com/avwohl/cpmemu/releases/latest/download/cpmemu_*_amd64.deb -o cpmemu.deb
sudo dpkg -i cpmemu.deb

# For ARM64 (Raspberry Pi):
curl -sL https://github.com/avwohl/cpmemu/releases/latest/download/cpmemu_*_arm64.deb -o cpmemu.deb
sudo dpkg -i cpmemu.deb

# Verify
cpmemu --version
```

### macOS / Windows

See the cpmemu repository for platform-specific instructions:
https://github.com/avwohl/cpmemu

### Test with Emulator

If you installed cpmemu, run your compiled program:

```bash
cpmemu hello.com
```

Expected output:
```
Hello, World!
```

## Build Process Summary

The typical workflow for compiling PL/M-80 programs:

```bash
# 1. Compile PL/M to assembly
uplm80 input.plm -o output.mac

# 2. (Optional) Run post-assembly optimizer
python -m uplm80.postopt output.mac -o output_opt.mac

# 3. Assemble to relocatable object
um80 output.mac

# 4. Link with runtime library (if needed)
ul80 -o program.com output.rel runtime.rel

# 5. Run the program
cpmemu program.com
```

## Compiler Options

```bash
uplm80 [OPTIONS] input.plm
```

Common options:
- `-o FILE` - Output assembly filename (default: input.mac)
- `-m {cpm,bare}` - Runtime mode (default: cpm)
  - `cpm`: CP/M program with stack from BDOS, returns to OS
  - `bare`: Bare metal with local stack, Intel PL/M-80 compatible
- `-t {z80,8080}` - Target processor (default: z80)
- `-O {0,1,2,3}` - Optimization level (default: 2)
- `--debug` - Enable debug output
- `-v, --version` - Show version
- `-h, --help` - Show help

## Assembler Options

```bash
um80 [OPTIONS] input.mac
```

Options:
- `-o FILE` - Output filename (default: input.rel)

## Linker Options

```bash
ul80 [OPTIONS] file.rel [file2.rel ...]
```

Options:
- `-o FILE` - Output filename (default: a.com)
- `-p ADDR` - Set program origin (default: 0x100 for CP/M)
- `-s FILE` - Generate symbol file

## Examples

See the `examples/` directory for complete working programs:
- `hello_cpm.plm` - Simple hello world using BDOS

## Runtime Library

The compiler may generate calls to runtime routines for certain operations. You'll need to provide these in a `runtime.rel` file or link with a runtime library:

- `??MUL` - 16-bit unsigned multiplication
- `??DIV` - 16-bit unsigned division
- `??MOD` - 16-bit unsigned modulo
- `??SHL` - 16-bit shift left
- `??SHR` - 16-bit logical shift right
- `??SHRS` - 16-bit arithmetic shift right
- `??MOVE` - Block memory move

For CP/M programs, you'll also need stubs for:
- `MON1`, `MON2`, `MON3` - BDOS call interfaces
- `BOOT` - Warm boot
- System variables: `BDISK`, `MAXB`, `FCB`, `BUFF`, `IOBYTE`

See [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md) for details on CP/M BDOS usage.

## Troubleshooting

### ModuleNotFoundError: No module named 'uplm80'

Make sure you've activated your virtual environment:
```bash
source venv/bin/activate  # or myuplm80/bin/activate
```

### pip: command not found

Install pip or use Python's built-in pip:
```bash
python3 -m pip install uplm80
```

### Virtual environment creation fails

On some systems (particularly Raspberry Pi OS Debian 13), you need the full Python package:
```bash
sudo apt install python3-full python3-venv
```

### Permission errors when installing

Use a virtual environment instead of system-wide installation, or use `--user`:
```bash
pip install --user uplm80
```

## Updating

To update to the latest version:

```bash
# From PyPI
pip install --upgrade uplm80 um80 upeep80

# From source (in repository directory)
git pull
pip install --upgrade -e .
```

## Uninstalling

```bash
pip uninstall uplm80 um80 upeep80
```

## Getting Help

- **Documentation**: See [README.md](README.md) for language reference
- **BDOS Reference**: See [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md)
- **Examples**: Check the `examples/` directory
- **Issues**: Report bugs at https://github.com/avwohl/uplm80/issues
- **Discussions**: Ask questions on GitHub Discussions

## Additional Resources

- PL/M-80 Programming Manual: See `docs/external/`
- CP/M Documentation: Available online at http://www.cpm.z80.de/
- Example programs: Digital Research CP/M utilities source code
