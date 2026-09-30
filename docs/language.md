# PL/M-80 Language Rules and Conditional Compilation

## Overview

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

    mon1: PROCEDURE(func, parm) EXTERNAL;   /* mon1 equ 5: see runtime_modes.md */
        DECLARE func BYTE, parm ADDRESS;
    END mon1;

    print: PROCEDURE(addr) PUBLIC;
        DECLARE addr ADDRESS;
        /* CP/M BDOS print string */
        CALL mon1(9, addr);
    END print;

    CALL print(.message);
END hello;
```

See [examples/hellocpm.plm](../examples/hellocpm.plm) for a complete working example.

## Rules the compiler checks

The compiler holds a program to these rules of the Programming Manual
(9800268B), as Intel's PL/M-80 V3.1 does, and says what it found and
where:

- Every name is declared, the built-ins aside (6.1); in a multi-file
  compile a module may name another's PUBLIC name without declaring it
  EXTERNAL, but for a built-in's name, which is the built-in in a module
  that does not declare it, as it is compiled alone.
- Each parameter is declared once, by a DECLARE of its procedure, as a
  BYTE or an ADDRESS scalar, not BASED and with no other attribute
  (8.1.1).
- A BASED variable's base is an ADDRESS scalar, a variable or a parameter,
  or an ADDRESS scalar member of a structure that is neither BASED nor an
  array, declared before the BASED variable, in its block or one around
  it, as V3.1 takes it; the base is that declaration wherever the BASED
  variable is used.
- A LABEL declared in a block, not PUBLIC nor EXTERNAL, labels a statement
  of the block (9.3); a label of its name in a DO block of it is another
  label, the DO block's.
- A declaration hides the built-in of its name (9.2): a procedure DOUBLE
  or a variable OUTPUT, MEMORY or STACKPTR that a module declares is the
  module's there, and the compiler never takes it for the built-in.
- A LITERALLY's text takes its name's place throughout its scope, in the
  text after the declaration (6.4), a declaration of the name in an inner
  block included: after `DECLARE N LITERALLY '5'`, an inner `DECLARE N
  BYTE` is `DECLARE 5 BYTE`, an error.  The name used before the
  declaration is not declared there.
- A dimension is a number, or a LITERALLY declared before it whose text
  is one (6.2.5), and not 0.
- Empty parentheses are an error after a variable, `x()`, after a
  structure member, `s.m()`, after a subscript, `a(1)()`, and after a
  built-in, `carry()`; `f()` and `CALL g()` of a procedure are taken for
  `f` and `g`, with a warning, as programs written for uplm80 rely on
  them.
- The address of a label, `.label`, may be given in a DATA or an INITIAL
  list, not in an expression (4.1.3); of the built-ins, only MEMORY has
  an address.
- A DATA or INITIAL value, an AT address and a constant of a constant
  list `.(...)` are restricted expressions (4.1.3, 6.2.8, 6.2.9): numbers,
  added and subtracted, and a minus sign before a number; in a DATA or
  INITIAL list and a constant list a string alone; in a DATA or INITIAL
  list a location plus or minus numbers, `.s.m(1) + 2`, `.memory` among
  them, and in an AT one of a variable, not BASED, or of MEMORY; and in a
  constant list, or where the value fills a BYTE, what a byte holds.  A
  built-in, a name not after a dot, parentheses, an operator but + and -,
  a string in a sum or in an AT, a location anywhere but first, one with
  two subscripts, with more than one in its parentheses or with a
  subscript on what is not an array, a constant list in a DATA list, and
  in an AT the location of a procedure or a label are errors at every `-O`
  level, as V3.1 makes them: `data (shl(0f0h, 4))`, `data (x)`, `.(x, 7)`,
  `.(2 * 3)`, `.(300)`, `data (.a(1)(1))`, `data (.a(1, 2))`, `data
  (.x(1))`, `at (.p)`.  A location in a DATA or INITIAL list or an
  AT address is the variable the name means in its block (9.1), declared
  before it or after, there or in a block around it, whatever the form of
  its declaration: `.arr(2)` of an ADDRESS array is ARR+4, and a
  procedure's `.memory` its own MEMORY where it declares one further on.
- An INTERRUPT procedure is declared at the outer level of its module and
  has no parameters (8.1.6); so is a PUBLIC or EXTERNAL procedure or
  variable, not in a procedure or a DO block.  INITIAL there initializes
  the variable once, when the program is loaded, with a warning.
- A REENTRANT procedure is declared at the outer level of its module too,
  and has no procedure declared in it; a procedure calls itself, directly
  or from one declared in it, only if it is REENTRANT (8.1.7).
- A direct call passes as many arguments as the procedure has parameters,
  and follows the procedure's declaration, but a call by a REENTRANT
  procedure of one that is REENTRANT too; `.p(1)` of a procedure is no
  address.  `CALL q(1, 2)` of an ADDRESS q calls through it (8.2.1), as
  a CALL does through a structure's ADDRESS member or a BASED ADDRESS, and
  through the ADDRESS member of an array of structures named without a
  subscript, `CALL sa.g`, `sa(0).g`'s; not through an array, an element,
  `CALL sa(1).g`, a structure, a BYTE or a label, nor of a built-in with a
  type, `CALL STACKPTR`.
- An END that names a block names its own, the procedure or the label
  next to the DO; a DO CASE has a case, and a procedure a statement.
- An array, MEMORY or a member array, is named without a subscript only
  after a dot or as the argument of LENGTH, LAST or SIZE (3.6.2), whose
  subscripts have nothing in parentheses in them.  An array takes one
  subscript, and a scalar none: `x(1)`, the element that far past x, and
  `s2.m(1)` of an array of structures, `s2(0).m(1)`, are compiled with a
  warning, as before, since uplm80's own tests test them; `s.k(1)` of a
  scalar member is an error.  Nothing follows a subscript but a member,
  `a(1)(2)` is an error, and INPUT and OUTPUT take one port.
- SHL and SHR of a BYTE are a BYTE (11.1.4), and the bits shifted out of
  it are lost; `SHL(DOUBLE(b), n)` keeps them.  uplm80 before 0.4.3
  shifted a BYTE in 16 bits.  Where a SHL of a BYTE can shift a set bit
  out and the bits above its low byte are used - stored to an ADDRESS,
  compared, used as a subscript, ... - the compiler warns at the SHL; and
  it warns at every flag reader - PLUS, MINUS, CARRY, ZERO, SIGN, PARITY,
  SCL, SCR, DEC - the flags of a SHL or SHR of a BYTE, or of an 8-bit
  operation on one, may reach, which are an 8-bit operation's since 0.4.3
  (CHANGELOG, 0.4.3 and 0.4.4).
- A flag reader reads at every `-O` level the flags it reads at `-O0`:
  the optimizer leaves as it is any operation whose flags a reader can
  read, where it folded `x OR 0`, `x XOR 0` or `SHL(3, 2)` from `-O1` on,
  and at `-O3` an operation of a variable whose value it knew, a loop it
  unrolled, a test it decided or a store the next statement overwrites
  (0.4.4).

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
