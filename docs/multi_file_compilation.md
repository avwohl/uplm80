# Multi-File Compilation with uplm80

## Overview

uplm80 supports compiling multiple PL/M source files together in a single compilation unit. This enables:
- Global optimizations across all modules
- Shared ??AUTO temporary storage
- Consistent calling conventions

## Usage

```bash
uplm80 file1.plm file2.plm file3.plm -o output.mac
```

Each file is one PL/M-80 module.  Only one of them may be the main
program module (the one with executable statements at its outer level).

## Names

Each module keeps its own name space, exactly as if it were compiled
alone and linked: a name that is not PUBLIC or EXTERNAL belongs to its
module (Programming Manual, 10.4), and two modules may each have a
procedure, variable, DATA table, label or LITERALLY of the same name
without meeting.  In the combined assembly such names are qualified with
their module's name: procedure `helper` of module `lib` is `LIB?HELPER`,
and its locals `@LIB?HELPER$N`.  (No PL/M-80 identifier contains `?`, so
a qualified name cannot meet one the program declares.)

PUBLIC and EXTERNAL names are not qualified: they are what the linker
binds, within the combined assembly as between separately compiled
modules.

A module that uses a name another module declares without PUBLIC is an
error, as it would be at link time if the modules were compiled
separately:

```
main.plm:17:18: error: T2 is not declared in module M; module LIB declares
it but does not make it PUBLIC (declare it PUBLIC there and EXTERNAL here)
```

## Procedures

Declare the procedure PUBLIC where it is defined, and EXTERNAL in each
module that calls it:

```plm
/* file1.plm - defines the procedure */
myproc: procedure(x, y) byte public;
    declare (x, y) address;
    /* ... */
end myproc;

/* file2.plm - calls it */
myproc: procedure(x, y) byte external;
    declare (x, y) address;
end myproc;

result = myproc(1, 2);
```

A PUBLIC procedure takes its arguments on the stack, which is how a call
to an EXTERNAL one passes them, so the two agree whether the modules are
compiled together or apart.  (A procedure private to its module is
called more cheaply: its arguments are stored straight into its own
storage, and the last passed in a register.)

When the modules are compiled together, a PUBLIC procedure can also be
called from another module without an EXTERNAL declaration there, as
older uplm80 multi-file programs do; the EXTERNAL declaration is what
lets each module also be compiled on its own.

## Variables

Variables use `external` and `public` to share storage across files:

```plm
/* file1.plm */
declare counter byte public;

/* file2.plm */
declare counter byte external;
counter = counter + 1;  /* Works correctly */
```

## Heap Allocation

The linker provides `__END__` at the end of all code/data. Since PL/M can't directly reference symbols with underscores, create a bridge in assembly:

```asm
; heap.asm
    .Z80
    PUBLIC  HEAPBASE
    EXTRN   __END__

HEAPBASE:
    DW  __END__

    END
```

Then in PL/M:

```plm
declare heap$base address external;

/* In initialization: */
buffer$ptr = heap$base;
```

Link with: `ul80 -o program.com program.rel heap.rel`

## Recommended File Order

1. **Startup module** (entry point, calls main)
2. **Common declarations** (globals, BDOS interface)
3. **Library modules** (I/O, utilities)
4. **Feature modules** (decompressors, etc.)
5. **Main module** (main procedure, program logic)

## Example Build

```makefile
SRCS = startup.plm common.plm io.plm feature.plm main.plm

$(TARGET): $(SRCS)
    uplm80 $(SRCS) -o program.mac
    um80 program.mac -o program.rel
    ul80 -o program.com program.rel
```
