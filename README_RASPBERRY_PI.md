# Installing uplm80 on Raspberry Pi

This guide covers installation of the uplm80 PL/M-80 compiler on Raspberry Pi, including setting up the CP/M emulator for testing.

## Raspberry Pi OS Debian 13 (Trixie) - Important Note

If you're running Raspberry Pi OS based on Debian 13 (Trixie), the default Python installation is minimal and doesn't include `venv` and other essential packages.

**You must install the full Python package:**

```bash
sudo apt update
sudo apt install python3-full
```

This provides the complete Python environment including `venv`, `pip`, and other necessary modules.

## Prerequisites

```bash
# Install required system packages
sudo apt update
sudo apt install python3-full curl
```

## Installation

### Option 1: Install from PyPI (Recommended)

Install uplm80 and related tools from PyPI:

```bash
# Create and activate a virtual environment
python3 -m venv myuplm80
cd myuplm80/
source bin/activate

# Install uplm80 and associated tools
pip install uplm80 um80 upeep80
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/avwohl/uplm80.git
cd uplm80

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Install associated tools
pip install um80 upeep80
```

## Installing the CP/M Emulator (Optional but Recommended)

The `cpmemu` emulator allows you to test compiled CP/M programs on your Raspberry Pi. It's highly recommended for development and testing.

### Quick Install Script

```bash
#!/bin/bash
set -e

# Download latest arm64 release
URL=$(curl -s https://api.github.com/repos/avwohl/cpmemu/releases/latest | \
  grep -o 'https://[^"]*arm64\.deb')

# Install the package
curl -sL "$URL" -o cpmemu.deb
sudo dpkg -i cpmemu.deb
rm cpmemu.deb

echo "cpmemu installed successfully"
cpmemu --version
```

Save this as `install_cpmemu.sh`, make it executable, and run:

```bash
chmod +x install_cpmemu.sh
./install_cpmemu.sh
```

### Manual Installation

Alternatively, download and install manually:

```bash
# Download the latest release for ARM64
# Visit: https://github.com/avwohl/cpmemu/releases/latest
# Download the *arm64.deb file

# Install with dpkg
sudo dpkg -i cpmemu_*_arm64.deb
```

## Quick Start: Compile and Run Hello World

Once installed, try compiling and running a simple program:

```bash
# Make sure you're in the virtual environment
source myuplm80/bin/activate  # or wherever you created your venv

# Create a simple hello world program
cat > hello.plm << 'EOF'
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
uplm80 hello.plm -o hello.mac

# Assemble to relocatable object
um80 hello.mac

# Link to CP/M executable
ul80 -o hello.com hello.rel

# Run in CP/M emulator (if installed)
cpmemu hello.com
```

You should see:
```
Hello, World!
```

## Example: Complete Test Program

Here's a more complete test program from the installation verification:

```bash
# Create test program (same as tests/test_simple.plm)
cat > print3.plm << 'EOF'
/*
 * SIMPLE TEST
 */

0100H:

MON1: PROCEDURE(FUNC, PARM) EXTERNAL;
    DECLARE FUNC BYTE, PARM ADDRESS;
END MON1;

/* PRINT SINGLE CHARACTER */
PRINT$CHAR: PROCEDURE(C);
    DECLARE C BYTE;
    CALL MON1(2, C);
END PRINT$CHAR;

/* PRINT DIGIT 0-9 */
PRINT$DIGIT: PROCEDURE(D);
    DECLARE D BYTE;
    CALL PRINT$CHAR(D + '0');
END PRINT$DIGIT;

/* SIMPLE MAIN */
MAIN: PROCEDURE;
    DECLARE X ADDRESS;

    X = 3;
    CALL PRINT$DIGIT(X);
    CALL PRINT$CHAR(13);
    CALL PRINT$CHAR(10);
END MAIN;

CALL MAIN;
EOF

# Compile, assemble, and link
uplm80 print3.plm
um80 print3.mac
ul80 -o print3.com print3.rel

# Run it
cpmemu print3.com
```

Output: `3` (followed by newline)

## Compiler Options

```bash
uplm80 --help
```

Common options:
- `-m cpm` - CP/M mode (default) - sets stack from BDOS, returns to OS
- `-m bare` - Bare metal mode - uses local stack, original Intel PL/M style
- `-t z80` or `-t 8080` - Target processor (default: Z80)
- `-O 0|1|2|3` - Optimization level (default: 2)
- `-o output.mac` - Output filename

## Toolchain Components

- **uplm80** - PL/M-80 compiler (PL/M source → assembly)
- **um80** - Universal Macro Assembler (assembly → relocatable object)
- **ul80** - Universal Linker (object files → CP/M executable)
- **upeep80** - Peephole optimizer library (used internally by uplm80)
- **cpmemu** - CP/M 2.2 emulator (runs CP/M programs on Linux/Raspberry Pi)

## Additional Resources

- Main README: [README.md](README.md)
- Installation Guide: [INSTALL.md](INSTALL.md)
- BDOS Reference: [docs/BDOS_REFERENCE.md](docs/BDOS_REFERENCE.md)
- Example Programs: [examples/](examples/)
- Project Repository: https://github.com/avwohl/uplm80

## Troubleshooting

### Python venv not found

If you get an error about `venv` not being found:
```bash
sudo apt install python3-full python3-venv
```

### pip not found

Make sure you're in an activated virtual environment:
```bash
source myuplm80/bin/activate
```

### cpmemu: architecture mismatch

Make sure you downloaded the ARM64 version of cpmemu for Raspberry Pi. Check with:
```bash
uname -m  # Should show: aarch64
```

### Permission denied running cpmemu

Make sure the package was installed correctly:
```bash
which cpmemu
cpmemu --version
```

## Getting Help

- Report issues: https://github.com/avwohl/uplm80/issues
- View examples: Check the `examples/` directory
- Read PL/M-80 docs: See `docs/external/` for language reference
