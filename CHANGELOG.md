# Changelog

Notable changes to uplm80. Releases before 0.3.2 are described on the
[GitHub releases page](https://github.com/avwohl/uplm80/releases).

## 0.4.4 — 2026-09-27

What 0.4.3's final release check found and left for a later release, what
0.4.3's Known issues listed as compiled without a word, and what checking
0.4.4 against Intel's PL/M-80 V3.1 found.  More of the errors V3.1 gives,
V3.1's numbers and texts, at every level: more than one subscript, a CALL
through what is not an ADDRESS scalar or of a built-in with a type, a
built-in with too few or too many arguments, MEMORY without a subscript, a
procedure that calls itself, a REENTRANT procedure where V3.1 refuses one,
an assignment to a built-in or a procedure; what a restricted expression -
a DATA or INITIAL value, an AT address, a constant list - does not take: a
built-in, which `-O1` and up folded, a name, parentheses, an operator but
+ and -, a number a byte does not hold, a location with two subscripts or
with one on what is not an array, and in an AT the location of a
procedure, a label or a BASED variable among them; a base V3.1 does not
take - BASED or a member of what is, on which um80 failed, a BYTE, a name
declared further down; and a LABEL that labels no statement, on which in a
DATA list um80 failed.  Every flag reader the flags of a SHL or SHR of a
BYTE may reach is warned of, and every `-O` level reads the flags `-O0`
reads.  A location in a DATA or INITIAL list or an AT address is the
variable the name means, declared in its block or one around it, before
the list or after it, whatever the form of its declaration; that of a
REENTRANT procedure's local or a BASED variable in a list, of a factored
BASED declaration too, is uplm80's own refusal, as in an AT, not um80's
"Undefined symbol" or another block's variable.  A factored BASED
declaration on a member, `(a based s.p) byte`, is BASED, and a base is the
declaration of its name made before the variable BASED on it, wherever
that variable is used.  Smaller code after an 8-bit operation, and the
columns after a LITERALLY's text are the source's.

### Incompatible: more errors Intel's PL/M-80 gives

What uplm80 compiled and Intel's PL/M-80 V3.1 rejects is refused as V3.1
refuses it, with the number and text of each error V3.1 gives, at every
`-O` level (of some combinations of two errors, not every one: Known
issues).  No program of MP/M II (DRI's tree and mpm2's overrides), of
80un 0.3.3 or of `sample_code` has any of these, and none of the 88
compiles of their PL/M gives a new error; one test's program, which called
through an array's element to have its address found in DE, calls through
a BASED ADDRESS now.  V3.1 rejects each program of `tests/test_names.py`
with the errors the message names, and no other
(`tests/test_intel_oracle.py`), and builds the programs of what it takes
of the same forms to print what uplm80's builds print.  Three of 0.4.3's
messages name more of V3.1's errors: an INTERRUPT, PUBLIC or EXTERNAL
procedure with no statements in a procedure or a DO block #174, INVALID
NULL PROCEDURE, besides #39, as V3.1 then takes it for a procedure of its
own, and an EXTERNAL one has none, and a typed one #156, MISSING RETURN
STATEMENT IN TYPED PROCEDURE, too; a typed procedure with no statements
#156 besides #174; and LENGTH and LAST of a reference with anything in
parentheses in a subscript #125, ILLEGAL ARGUMENT FOR BUILT-IN PROCEDURE,
besides #32 (SIZE only #32).  And the message of `.lb` of a label in an
expression, which uplm80 refuses since 0.4.1, names V3.1's #158, INVALID
DOT OPERAND, LABEL ILLEGAL.

- **More than one subscript** (ERROR #114, INVALID SUBSCRIPT, MULTIPLE
  SUBSCRIPTS ILLEGAL), wherever a subscript goes: of an array, a BASED
  one, a member array, an array of structures and MEMORY, in an
  expression, as a target and after a dot - `b = a(1, 2)`, `a(1, 2) =
  w`, `.a(1, 2)`, `s.m(1, 2)`, `sa(1, 2).m(0)`, `sa(1).m(0, 1)`,
  `memory(1, 2)`.  uplm80 took the subscripts for arguments and called,
  at every level, through the value of `a(0)` (0.4.3's release check);
  0.4.3 made the same of a scalar an error.  In a DATA or INITIAL list,
  an AT address or a constant list, `data (.a(1, 2))`, it is V3.1's #150
  (Incompatible: V3.1's errors in a restricted expression).

      A(1, 2): A is an array, and an array takes one subscript; Intel's
      PL/M-80 V3.1 rejects it (ERROR #114, INVALID SUBSCRIPT, MULTIPLE
      SUBSCRIPTS ILLEGAL)

- **A subscript on a scalar member**, `s.k(1)`, which uplm80 took for
  the element that far past `s.k` (ERROR #127, INVALID SUBSCRIPT ON
  NON-ARRAY, and #32, INVALID SYNTAX, TEXT IGNORED UNTIL ';'); a
  subscript or an argument list after another, `a(1)(2)` and `h(1)(2)`,
  which uplm80 took for a call through the element or through what `h`
  returns (#32); and a second port, `input(1, 2)` and `output(1, 2) =
  b`, which uplm80 left out (ERROR #108, MISSING ')' AFTER INPUT/OUTPUT
  PORT NUMBER).
- **A CALL through what is not an ADDRESS scalar** (ERROR #118, INVALID
  INDIRECT CALL, IDENTIFIER NOT AN ADDRESS SCALAR, and #32 where
  parentheses follow): an array, `call aa(1)` and `call aa(1)(b, c)`,
  which uplm80 took for a call through `aa(0)`, with 1, and through
  `aa(1)`; a member array, a member of an element of an array of
  structures, `call sa(1).g`, where V3.1 takes SA for what is called and
  the rest for its arguments, MEMORY, a structure, and a BYTE - a
  variable, a parameter or a member - whose value uplm80 called; and a
  label, `call l`, which uplm80 called and the link found no L for.  A
  CALL through an ADDRESS, a structure's ADDRESS member or a BASED
  ADDRESS, with arguments or without, is compiled as before, as V3.1
  compiles it, and so is one through the ADDRESS member of an array of
  structures, BASED or not, named without a subscript, `call sa.g`, which
  calls through `sa(0).g`: V3.1 takes it in a CALL, and rejects it in an
  expression (ERROR #133, of which uplm80 warns: 0.4.3).

      CALL AA(1): AA is an array, and a CALL calls a procedure, or
      through an ADDRESS scalar (Programming Manual 9800268B, 8.2.1);
      Intel's PL/M-80 V3.1 rejects it (ERROR #118, INVALID INDIRECT CALL,
      IDENTIFIER NOT AN ADDRESS SCALAR, and #32, INVALID SYNTAX, TEXT
      IGNORED UNTIL ';')

- **A CALL of a built-in with a type** (ERROR #129, ILLEGAL 'CALL' WITH
  TYPED PROCEDURE, and #32 where parentheses follow): `call stackptr`,
  which uplm80 compiled to a call through the value of SP, `call carry`,
  to a call of an undefined CARRY, and `call rol(b, 1)` and the like, to
  the built-in's value, unused.  A CALL of a typed procedure the program
  declares is still taken (README).
- **A built-in with too few or too many arguments** (ERROR #154, INVALID
  NUMBER OF ARGUMENTS IN CALL, TOO FEW, and #153, TOO MANY): `call time;`,
  `call move(1, .b);`, `b = rol(b);` and `w = stackptr(1);`, which stopped
  uplm80 with a traceback, and `b = high;`, `b = low(b, 1)`, `b =
  carry(1)` and `call time(1, 2)`, which it compiled (found checking
  0.4.4; 0.4.3 the same); LENGTH, LAST or SIZE with no argument or more
  than one (#124, MISSING ARGUMENTS FOR BUILT-IN PROCEDURE; #126, MISSING
  ')' AFTER BUILT-IN PROCEDURE ARGUMENT LIST); and INPUT or OUTPUT without
  a port, `b = input;` and `output = b;` (#109, MISSING INPUT/OUTPUT PORT
  NUMBER).
- **MEMORY without a subscript**, `b = memory`, `memory = b`, `w = memory
  + 1` (ERROR #133, ILLEGAL REFERENCE TO UNSUBSCRIPTED ARRAY), which
  uplm80 took for `memory(0)`; `.memory` and LENGTH, LAST and SIZE of it
  are taken, as V3.1 takes them.
- **A procedure that calls itself**, not REENTRANT (ERROR #170, ILLEGAL
  RECURSIVE CALL): `r: procedure; ... call r; end r;`, `return h` in a
  typed H, and a call of R from a procedure declared in it (0.4.3's Known
  issues).  Its locals are static and a second activation overwrote the
  first's.  A REENTRANT procedure calls itself, and a CALL through the
  address of the procedure it is in is taken, as V3.1 takes them.

      R: procedure R is called from inside itself, and only a REENTRANT
      procedure may be (Programming Manual 9800268B, 8.1.7); Intel's
      PL/M-80 V3.1 rejects it (ERROR #170, ILLEGAL RECURSIVE CALL)

- **A REENTRANT procedure not at the outer level of its module**, in a
  procedure or in a DO block (ERROR #39, INVALID ATTRIBUTE OR
  INITIALIZATION, NOT AT MODULE LEVEL), and **a procedure declared in a
  REENTRANT one**, in it or in a DO block of it (ERROR #88, INVALID
  PROCEDURE NESTING, ILLEGAL IN REENTRANT PROCEDURE); both of a REENTRANT
  procedure in a REENTRANT one (0.4.3's Known issues), and of an
  INTERRUPT, PUBLIC or EXTERNAL one, which 0.4.3 refused with #39 alone;
  and of one with no statements #174, INVALID NULL PROCEDURE, besides,
  and of a typed one #156 too (found checking 0.4.4).
- **An assignment to a built-in or a procedure** (ERROR #128, INVALID
  LEFT-HAND OPERAND OF ASSIGNMENT, and #131, ILLEGAL REFERENCE TO UNTYPED
  PROCEDURE, of one without a type), as a statement's target or an
  embedded assignment's: `input(1) = b`, `low(w) = b`, `b, carry = 1`,
  `time(1) = b`, `f = b` of a procedure F, which uplm80 compiled to a
  store to a symbol of the name that only the link found undefined (found
  checking 0.4.4; 0.4.3 the same).  Of one without as many arguments as
  it takes, V3.1's error of its arguments besides: `time = 1`, `mon1 =
  1`, #154, INVALID NUMBER OF ARGUMENTS IN CALL, TOO FEW, `low(w, 2) =
  b` #153, TOO MANY, `input = b` #109 and `last = b` #124.  MEMORY,
  OUTPUT and STACKPTR are assigned as before.
- **An END that names the first of two labels on a DO**, `m: n: do; ...
  end m;` (ERROR #20, MISMATCHED IDENTIFIER AT END OF BLOCK): V3.1 takes
  only the label next to DO, `end n;` (0.4.3's Known issues).

### Incompatible: V3.1's errors in a restricted expression

What a restricted expression does not take - a DATA or INITIAL value, an
AT address, a constant of a constant list `.(...)` (Programming Manual
9800268B, 4.1.3, 6.2.8, 6.2.9) - is refused as V3.1 refuses it, with its
errors' numbers and texts, when the parser gives the program, at every
`-O` level; 0.4.3's Known issues had it for SHL and SHR.  (What is an
error anywhere, a member a structure does not have, is refused in uplm80's
own words, as it was: Known issues.)  V3.1 takes there numbers, added and
subtracted, and a minus sign before a number; in a DATA or INITIAL list
and a constant list a string alone; in a DATA or INITIAL list a location
plus or minus numbers, `.a(1 + 1) - 2`, `.s.m(1)`, `.memory`; and in an AT
a location plus or minus numbers, of a variable, not BASED, or of MEMORY.
It rejects the rest, which uplm80 compiled, at every level or some, or
refused in words of its own:

- a built-in (ERROR #151, INVALID OPERAND IN RESTRICTED EXPRESSION):
  `declare w address data (shl(0f0h, 4))`, `initial (size(a))`, `at
  (double(12h))`, `data (.a(low(1)))`, `data (memory)`, `p = .(shl(1, 2),
  3)`.  `-O1` and up folded SHL, SHR, ROL, ROR, LOW, HIGH and DOUBLE of
  constants there, SHL and SHR of a BYTE in 16 bits - `shl(0f0h, 4)` was
  0F00H, where the same SHL in a statement is 0 - and `-O0` refused them
  with messages of its own, but for `at (double(12h))`, 12H, and `data
  (.a + low(3))`, another address.  The other built-ins every level
  refused, or um80 did, and a constant list lost what was not folded,
  the next constant in its place: `.(shl(1, 2), 3)` at `-O0` and
  `.(memory, 7)` at every level were `.(3)` and `.(7)` (0.4.2 the same,
  but that its `.(shl(1, 2), 3)` was `.(3)` at every level).  The
  location of a built-in in an AT, `at (.stackptr)`, is #211, INVALID
  IDENTIFIER IN 'AT' RESTRICTED REFERENCE;
- a name not after a dot, a variable's, a procedure's, a label's
  (#151): `data (x)` and `initial (x)`, which uplm80 took for x's
  address; `x = 3; p = .(x, 7)` and `.(x + 1, 7)`, which `-O0` to `-O2`
  took for `.(7)`, the name left out and the next constant in its place,
  and `-O3` for `.(3, 7)` and `.(4, 7)` - a variable the program calls
  MEMORY too; `.(a(1), 7)` and `.(p, 7)` of a procedure, `.(7)` at every
  level; and `data (a)` and `.(a, 7)` of an array, which uplm80 refused
  as an array without its subscript, #133 (0.4.3's Known issues had
  `initial (a)`);
- parentheses (#151 and #152, MISSING ')' AFTER CONSTANT LIST, in an AT
  #146, MISSING ')' AFTER 'AT' RESTRICTED EXPRESSION, in a subscript
  #150, MISSING ')' AT END OF RESTRICTED SUBSCRIPT): `.((1 + 2), 7)` and
  `.(-(1), 7)`, `.(7)` at `-O0` and `.(3, 7)` and `.(0FFH, 7)` at `-O1`
  and up; `data ((1 + 2))`; `data (.a((1)))`, which `-O0` refused;
- an operator but + and - (#152, in an AT #146), and NOT (and #151):
  `.(2 * 3)`, left out at every level; `.(not 0f0h)`, left out at `-O0`
  and `.(0FH)` at `-O1` and up; `data (6 / 3)`, `initial (0ffh and 7)`,
  `at (2 * 3)`, taken for their values;
- a string in a sum, `data ('A' + 1)` (#152), taken for 'B', and a
  location anywhere but first, `at (3 + .buf(1))`, and `data (-1 +
  .a)`, which `-O0` refused (#151, and #152 or #146);
- a constant list or `.'text'` in a DATA or INITIAL list, `data (.(5))`,
  `data (.('one$'))` (#147, MISSING IDENTIFIER FOLLOWING DOT OPERATOR),
  which uplm80 took for the location of the constants;
- in a constant list a location, `.(.a, 7)`, left out at every level
  (#210, ILLEGAL INITIALIZATION OF A BYTE TO A VALUE > 255, and #209,
  ILLEGAL INITIALIZATION OF MORE SPACE THAN DECLARED), and a number a
  byte does not hold, `.(300, 7)`, `.(0ffffh, 7)`, `.(299 + 1, 7)` (#210),
  and the same in a DATA or INITIAL value that fills a BYTE, `declare b
  byte data (300)`, `byte initial (.w)`, of which uplm80 kept the low byte
  (and `.(299 + 1, 7)` was `.(7)` at `-O0`).  V3.1 computes such a number
  as it does a constant, a number below 256 a BYTE, BYTE arithmetic in
  eight bits: `.(-1)`, `.(1 - 2)`, `.(-255)` and `.(0ffffh + 1)` are
  taken, as they are in a BYTE;
- a location with two subscripts, `data (.a(1)(1))`, `initial
  (.s.m(1)(1))`, `at (.a(1)(1))` (#152, in an AT #146), where V3.1 stops
  as at an operator it does not take, which 0.4.3 took for an element
  further on, of an ADDRESS array .a+2 in a DATA list, .a+3 in an INITIAL
  one, .a+4 in an AT, and refused of a member in a DATA or INITIAL list,
  `.s.m(1)(1)`;
- more than one subscript in a location's parentheses, `data (.a(1,
  2))`, `at (.a(1, 2) + 1)`, `.(7, .a(1, 2))`, `data (.memory(1, 2))`
  (#150, MISSING ')' AT END OF RESTRICTED SUBSCRIPT), where V3.1 reads the
  first subscript and goes on after the `)` that ends the rest, which it
  does not read, `.a(1, x)` #150 alone; 0.4.3 refused it in a DATA or
  INITIAL list and an AT in words of its own ("Unsupported subscript in
  DATA location expression: Call(...)"), and took it in a constant list;
- a subscript on the location of what is not an array (#149, INVALID
  SUBSCRIPTING IN RESTRICTED REFERENCE): a scalar or a structure, `data
  (.x(1))`, `at (.s(1))`, which 0.4.3 took for the element that far past
  it, with a warning that named #127, V3.1's error in an expression
  (`.x(1)` of an ADDRESS x+1 in a DATA list, x+2 in an INITIAL list or an
  AT); a scalar member in an AT, `at (.s.k(1))`, and a label, `.lbl(1)`,
  which it took without a word, and a member in a DATA list, `data
  (.s.k(1))`, which it refused in its own words; and a procedure, `data
  (.p(1))`, which it refused naming #104, V3.1's error in an expression;
- in an AT the location of a procedure or a label, `at (.p)`, `at (.lbl)`
  (#211), which uplm80 took for the procedure's address and refused for
  the label in words of its own, as it did the location of a BASED
  variable, `at (.bb)` (#212, INVALID RESTRICTED REFERENCE IN 'AT', BASE
  ILLEGAL).

The message names every error V3.1 gives for the list the value is in, a
typed declaration's or a constant list, as V3.1 reads it (each rule
checked with `scripts/intel_oracle.py`, Verified).  It reads a list's
values up to the first it does not read to the end, and a name it reads
past is a BYTE 0 to it: `.(300 + x)` and `declare b byte data (300 + x)`
are #151 and #210.  A DATA or INITIAL list with more values than its
declaration holds is #209 besides, `declare b (2) byte data (1, 2, x)`,
and V3.1 checks nothing past the space; alone, it is 0.4.3's Known issue,
which uplm80 lays out after the declaration.  A constant list V3.1 lays
out as it does an untyped DATA list: a value with a name in it, or a
location first, is a word, and a string after the last such value makes
the list bytes again; where it does not, there is room for the first value
only, and each value after it is #209 and not checked further, `.(x, 7)`,
`.(7, x)`, `.(7, 300, x)`, where `.(x, '$')` is #151 alone and `.(x, '$',
300)` #151 and #210.  After what it does not take it looks for the list's
`)`, and one of a parenthesis further on leaves the rest of the statement
#32, INVALID SYNTAX, TEXT IGNORED UNTIL ';', `.((1), 7)`, `.(-.a, (1))`;
and a list of one value whose last name is a built-in's but MEMORY's is
#172, INVALID LABEL: UNDEFINED, `.(stackptr)`, `.(1 + time)`, `.(.shl)`,
`.(shl(1, 2), 3)`.  V3.1 reads past a subscript on what is not an array
and the location in an AT of a procedure, a label or a BASED variable:
`data (.x(1), y)` is #149 and #151; and a value in the subscript of a
location that has two subscripts itself, `.a(.a(1)(1))`, is the list's
#152 (in an AT #146) besides the subscript's #151 and #150.

No program of MP/M II (DRI's tree and mpm2's overrides), of 80un or of
`sample_code` has any of these but `.memory` in an AT (MP/M II's GENSYS,
PIP and DSE, CP/M's PIP), so none is kept with a warning.  Of `tests/`,
`tests/test_declaration_storage.py` had a constant list in a DATA list
(since 0.3.7), `3+.buf(1)` in an AT, and NOT and AND in a DATA or INITIAL
list and a location in a BYTE's, and `tests/test_divmod_dri.py` divisions
in DATA and INITIAL lists, and `tests/test_names.py`, in its program that
takes the address of every kind of procedure, the location of two
procedures in an AT; they have them no more, and the constant list in a
DATA list is tested as V3.1's error.

    SHL(0f0h, 4): SHL is a built-in, and a DATA or INITIAL value is a
    restricted expression, of constants and locations only; Intel's
    PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED
    EXPRESSION, and #152, MISSING ')' AFTER CONSTANT LIST)
    X: X is a variable, and a constant list holds constants only;
    Intel's PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN
    RESTRICTED EXPRESSION, and #209, ILLEGAL INITIALIZATION OF MORE
    SPACE THAN DECLARED)

### Incompatible: V3.1's errors for a base and a LABEL

A BASED variable's base that V3.1 does not take is refused as V3.1 refuses
it, with its error, when the parser gives the program, at every `-O`
level.  V3.1 takes an ADDRESS scalar, a variable or a parameter, or an
ADDRESS scalar member of a structure that is neither BASED nor an array,
declared before the variable BASED on it, in its block or one around it -
in a factored declaration before the declaration, whose names it does not
know yet.  uplm80 checked no base, and compiled the rest, or failed:

- BASED, or a member of what is BASED (#50, INVALID ATTRIBUTES FOR BASE,
  and of an ADDRESS scalar member #52, INVALID BASE, MEMBER OF BASED
  STRUCTURE OR ARRAY OF STRUCTURES): `declare s based sp structure (k
  byte, p address); declare a based s.p byte;`, and `a based q` of a BASED
  `q`, for which uplm80 read the pointer from `S+1` and `Q`, which nothing
  defines: um80's "Undefined symbol" at every level (0.4.3 the same), and
  since b9c4a36 of the factored `(a based s.p) byte` too, which 0.4.3 took
  for a variable of its own (Fixed).  `declare a based a address` (#54,
  UNDECLARED BASE), used, recursed until Python gave up;
- a member of an array of structures (#52), or an array (#50), `a based
  sa.p`, `a based q` of `q (2) address`, which uplm80 refused naming #133,
  V3.1's error in an expression;
- MEMORY, or another built-in (#50), `a based memory`, um80's "Undefined
  symbol 'MEMORY'";
- a BYTE, a structure, a member that is a BYTE or an array, a procedure or
  a label (#50), `a based b` of a BYTE, for which uplm80 took the word at
  its address for the pointer; a member the structure does not have, or a
  member of what is not a structure (#55, UNDECLARED STRUCTURE MEMBER IN
  BASE), `a based s.zz`, the structure's first word; and a name declared
  only further down the block, `declare a based q byte; declare q
  address;`, or in the same factored declaration, `(a based a2, a2 based
  w) byte` (#54), or a parameter declared an ADDRESS only after the
  variable BASED on it (#50), which uplm80 took for the base.

A base declared nowhere, #54 to V3.1, is refused as it was, as any name
declared nowhere is, in uplm80's own words.

A LABEL declared in a block, not PUBLIC nor EXTERNAL, that labels no
statement of the block - a label of its name on a statement of a DO block
in it is another label, the DO block's - is refused as V3.1 refuses it,
#172, INVALID LABEL: UNDEFINED, and where the program names it #105,
UNDECLARED IDENTIFIER, besides, and where a CALL calls it #118, INVALID
INDIRECT CALL, IDENTIFIER NOT AN ADDRESS SCALAR, as of a label that labels
one (Incompatible: more errors Intel's PL/M-80 gives), and #32 of what
follows it in parentheses.  uplm80 took it without a word: its
location in a DATA list, `declare lb label; declare w address data
(.lb);`, was `dw LB`, which um80 did not know, at every level (0.4.3 the
same); a GOTO to it was refused in words of its own, but not where
optimization left the GOTO out as dead code, `if 0 then goto lb` at `-O2`.

No program of MP/M II (DRI's tree and mpm2's overrides), of 80un, of
`sample_code` or of `tests/` has any of these: each compiles to the
assembly it did.

    A BASED S.P: S is BASED, and a base is not a member of what is BASED;
    Intel's PL/M-80 V3.1 rejects it (ERROR #52, INVALID BASE, MEMBER OF
    BASED STRUCTURE OR ARRAY OF STRUCTURES)
    .LB: LB is declared a LABEL but labels no statement; Intel's PL/M-80
    V3.1 rejects it (ERROR #105, UNDECLARED IDENTIFIER, and #172, INVALID
    LABEL: UNDEFINED)

### Changed

- **Smaller code after an 8-bit operation.**  A structure's BYTE member,
  and ROL and ROR of a BYTE, were left in HL as well as in A, `ld l,a /
  ld h,0`, before the 8-bit operation that reads A - a SHR of a BYTE, since
  0.4.3, and AND, OR, a comparison (0.4.3's Known issues: MP/M II's DA.PLM,
  `shr(b3.hbyte, 3)` and `ror(b3.hbyte, 3) and 11100000b`); and a SHL of
  an ADDRESS by a constant set DE to 0 after it, which nothing reads (80un's
  `lbr.plm`), as uplm80 0.2 made it do to look like the multiply routine a
  product by a power of 2 stands for.  What wants a BYTE in HL widens it,
  as before.  Of the 88 MP/M II and 80un compiles, those that have either
  are smaller: at `-O0` 39 (34 of MP/M II's, 822 bytes in all; 5 of
  80un's, 210), at `-O2` 36 (31, 786; 5, 213), at `-O3` 37 (32, 955; 5,
  213).  DP.PLM's at `-O2` and `-O3` spells two instructions otherwise,
  at the same size, and the rest is 0.4.3's.  80un's two programs built
  with it extract and detokenize its test files as 0.4.3's builds do.

### Fixed

- **Every flag reader the flags of a shift of a BYTE may reach is warned
  of** (0.4.3's release check, and checking 0.4.4).  A SHL or SHR of a
  BYTE is a shift of eight bits since 0.4.3, and it, and what is computed
  of it in eight bits where it was sixteen, sets the carry, ZERO, SIGN,
  PARITY and the half carry of eight bits, whatever the shift loses.  The
  compiler warns at each flag reader - PLUS, MINUS, CARRY, ZERO, SIGN,
  PARITY, SCL, SCR and DEC - those flags may reach, naming the shifts:
  `b = shl(k, 1); c = carry;` with k = 0FFH was 00 and is 0FFH, `c =
  shr(d, 7) plus 0` with d = 0FFH 01 and 02, `b = shl(k, 1); b = scl(1,
  1);` 02 and 03.  The flags are followed through the statements after
  the shift, round loops, through IF and DO CASE, from a GOTO to every
  label of its name, into a procedure called and back out of it - where
  it may set none, with the caller's flags - through a CALL through an
  address into every procedure whose address is taken, and from an
  EXTERNAL procedure's code into every PUBLIC one; an INTERRUPT procedure
  is entered with the flags of any.  One operation stops them: an
  addition, subtraction, AND, OR or XOR of two BYTEs that is the value of
  an assignment statement (or DEC of one, or a PLUS or MINUS of BYTEs
  whose left operand is one), which code generation makes an 8-bit `add`,
  `sub`, `and`, `or` or `xor` and every level keeps (the next entry); and
  so does a CALL of procedures each of which ends in one every way.
  Anything else passes the flags on - a store, a comparison, a MOVE, a
  16-bit operation, a shift by 0 - where 0.4.4 before this took some of
  them to set the flags, and gave no warning of a reader that read those
  of the shift before them (found checking 0.4.4: a MOVE of a constant
  count, `ldir`, a shift by 0 and an operation folded to a constant set
  none).  A reader may be warned of that in fact reads the flags of
  something else; none a shift's flags reach is left out, but in code the
  program does not have, an EXTERNAL procedure's or what a CALL through an
  address reaches outside it (uplm80/flag_flow.py).

      warning: CARRY may read the flags of SHL(K, 1) (line 12): since
      uplm80 0.4.3 a shift of a BYTE is one of 8 bits (Programming Manual
      9800268B, 11.1.4), and it, and an operation on it, set the flags of
      an 8-bit operation, where it was one of 16 bits; SHL(DOUBLE(K), 1)
      shifts in 16 bits

  Of the programs checked, MP/M II's SHOW (DRI's and mpm2's), MSCHD and
  TOD are warned of, once each: they read a number as `b = shl(b, 3) +
  shl(b, 1); if carry then ...`, and the sum carries out of eight bits
  from b = 26 on, where 0.4.2's did not out of sixteen (V3.1's build as
  0.4.4's); and the suite's `tests/test_byte_shifts.plm`, six times,
  whose SCL and SCR read the carry the calls before them leave.
- **Every level reads the flags `-O0` reads** (found checking 0.4.4;
  0.4.3 the same).  The optimizer folded `d or 0`, `d xor 0`, `shl(3,
  2)` and `3 + 4` from `-O1` on, and at `-O3` an operation of a variable
  whose value it knew, `c = z` of z = 0 to `xor a`, a loop it unrolled, a
  test it decided, or, in a procedure or a DO block, a store the next
  statement overwrites, which it dropped, so that a flag read after one
  read, at some levels, the flags of what came before it: `b = shl(k, 1);
  c = d or 0; e = carry;` with k = 0FFH left e 0 at `-O0` and 0FFH at
  `-O1` to `-O3`, and `do i = 1 to 2; c = 5; end; e = carry;` and `c = d
  and 1; c = 5; e = carry;` in a procedure 0 at `-O0` to `-O2` and 0FFH
  at `-O3`.  It leaves as it is each operation whose flags a reader can
  read, by the rule of the entry before (uplm80/flag_flow.py), and each
  reader: not folded, rewritten, unrolled, inlined or dropped, and of
  operands of the same kind - `w + one` is `add hl,de`, where `w + 1` is
  `inc hl` - but for an 8-bit +, -, AND, OR, XOR, PLUS or MINUS of BYTEs,
  whose flags are the same with a constant for one operand, in the other
  modules of a multi-file compile too.  The code of the 88 compiles of
  MP/M II's and 80un's PL/M, and of `sample_code`, is the same at every
  level; of the suite's programs, three that read flags are longer:
  `test_dec_bcd.plm` at `-O2` and `-O3`, whose `dec(34h + 21h)` is `ld
  a,34h / add a,21h / daa` at every level, `test_byte_shifts.plm` at
  `-O2` and `-O3`, and `test_plus_minus.plm` at `-O3`, where constants no
  longer go into the shifts, sums and PLUS and MINUS the SCL, SCR, PLUS
  and MINUS after them may read the flags of.  V3.1 folds some of these
  itself, and differs from every level as from `-O0` (README, Known
  differences).

- **A test that assigns the variable it bounds no longer bounds it**:
  `k = 1; if k < 4 and (k := 200) > 0 then w = shl(k, 6);` was 3200H and
  is 0, with no warning (0.4.3's release check).  What a condition
  assigns, by an embedded assignment or in a procedure it calls, keeps no
  bound from it.  **Nor does a variable an INTERRUPT procedure assigns**,
  itself or in what it calls, which may change it between any two
  statements.  **A procedure that CALLs through an address may assign**
  what any procedure whose address is taken assigns (found checking
  0.4.4): with p's `call q` and q = .setk, `k = 1; call p; w = shl(k,
  6);` was 3200H and is 0, with no warning, as in 0.4.3, and so was `k =
  1; if k < 4 and f > 0 then w = shl(k, 6);` of an f that calls through
  q; and after a CALL through a structure's member, `call s.g`, what the
  statements before it had left still held.  **Each name is what it
  names where it is used** (found checking 0.4.4): a parameter or a
  variable called through that has the name of a procedure was taken for
  a call of the procedure, and, as in 0.4.3, a variable a DO block
  declares for the one of its name after the block.  With v = .setk and a
  procedure P that assigns another variable, `k = 1; call r2(v); w =
  shl(k, 6);` of `r2: procedure (p); declare p address; call p; end r2;`
  was 3200H and is 0, with no warning, and so was `k = 1; call s2; w =
  shl(k, 6);` of `s2: procedure; do; declare k byte; k = 5; end; k =
  200; end s2;`, and the K of an INTERRUPT procedure written so kept its
  bound; and a procedure whose last statement calls through a parameter
  named like a procedure that does not return was taken not to return
  itself.  A test `(n and 0e0h) <> 0` bounds n by what the mask leaves
  (31), and a procedure that ends in a call of MON1 with the function 0,
  BDOS's system reset, or of such a procedure, and has no RETURN nor a
  label on its END, which a GOTO reaches past the call, does not return:
  MP/M II's SHOW, MSCHD and TOD read a number as `if (b and 1110$0000b)
  <> 0 then call terminate; b = shl(b, 3) + shl(b, 1); if carry then
  ...`, and b is below 32 at those SHLs, which lose nothing (the flags
  warning, above, is of their CARRY).

  What the warning of the value does not see, of the places where the two
  meanings can differ: a store that reaches the variable other than by
  its name - past the end of an array, through MEMORY or a BASED variable
  whose base is not its address (the layout is uplm80's, Known issues);
  an EXTERNAL MON1 that is not BDOS's entry, and returns from the function
  0 - MON1 is the name DRI's programs give BDOS's entry, and it is taken
  for that in every mode, `-m bare` too, where DRI's programs call BDOS
  through it as well; and arithmetic around a SHL that loses nothing,
  which can overflow eight bits in its value, as 0.4.3 has it, `shr(z,
  4) - 1` (of its flags the flags warning warns).  Of the programs
  checked, these warnings are 0.4.3's: MP/M II's 7, and of 80un 0.3.3
  none but the 20 of the old single-file source it keeps,
  `src/plm/archive/80un.plm`.

- **The columns after a LITERALLY's text are the source's.**  The text
  takes the place of the name, on one line however many it runs over,
  and every column after it on the line was the text's: with `lit` a
  text of three lines, `b = lit; b = zz;` was reported at 6:24, where
  `zz` is at column 14 (0.4.3's release check), and a name longer than
  its text moved them back.  What a message is about in the text itself
  is placed at the name.  A word declared LITERALLY 'LITERALLY' and a
  procedure's name given by a LITERALLY are counted as their texts.

- **A location in a DATA or INITIAL list or an AT address is the
  variable the name means**, at every level: PL/M-80 scopes a name to its
  whole block (9.1), and a list or an AT may name a variable its block
  declares further down, as V3.1 takes it.  The names were looked up
  among what code generation had laid out so far, and a module's DATA is
  laid out before its other variables, a procedure's or a DO block's DATA
  or AT before its later declarations.  The element of an array not yet
  laid out was taken for a byte: at module level `.arr(2)` of an ADDRESS
  array ARR+2, not ARR+4, and `.size(2)` of the program's `size (3) byte`,
  taken for the built-in's, SIZE+4.  In a procedure or a DO block a name
  the block declares further down was another block's: `.arr(2)` and
  `.size(2)` the module's ARR and SIZE of the other type, FFFF and FFFE
  where V3.1 prints 0004 and 0002, `at (.buf(3))` the module's BUF, and a
  name declared nowhere else, MEMORY among them, um80's "Undefined symbol"
  or "not declared".  At module level `at (.memory(3))` before `declare
  memory (4) byte` was the end of the program, 0007 where V3.1 prints
  0003.  A subscript steps by an element's width, an array of structures'
  by the structure's, `.sa(2)`, which took each for a word; a member,
  `.s.k`, `.s.m(1)`, `.sa(1).y`, which was refused ("Unsupported operand
  in DATA location expression: MemberAccess(...)"), is its offset, as in
  an AT; and a subscript is a constant expression at `-O0` too, `.a(1 +
  1)`, which `-O0` refused.  Code generation finds each such name through
  the declaration it means, whatever its form - a variable's, plain or
  factored, BASED or not, an array's, a structure's, a parameter's, a
  procedure's, a label's - and a built-in's name only where the program
  declares none.
- **`.memory` in a DATA or INITIAL list**, `declare p address data
  (.memory + 2)`, is the end of the program, `__END__`, as in an AT and an
  expression, and as V3.1 takes it.  It was `dw MEMORY`, which um80 did
  not know.  Where the block, or one around it, declares a MEMORY, before
  the list or after it, it is that variable, and one declared BASED,
  `declare (memory based bp) byte`, is refused as BASED (below).  In a
  multi-file compile a module that does not declare MEMORY means the
  built-in in its lists too, as in its code and its ATs; another module's
  PUBLIC MEMORY was `dw MEMORY` there.
- **An expression in a constant list at `-O0`**, `p = .(1 + 2, -1)`, is
  its value, as in a DATA list and at `-O1` and up.  It was left out, and
  the next constant took its place: `.(1 + 2, 7)` was `.(7)`.
- **The location of a REENTRANT procedure's local or parameter, or of a
  BASED variable, in a DATA or INITIAL list**, `p: procedure reentrant;
  declare x address; declare w address data (.x);`, is refused in uplm80's
  own words, as in an AT: uplm80 gives it no fixed address.  It was `dw
  X`, which um80 did not know ("Undefined symbol"), at every level (0.4.3
  the same), and so was the location in an AT of a REENTRANT procedure's
  array or structure, `at (.x(1))`, `EQU X+2`, which uplm80 refused of a
  scalar.  So is a name of a factored BASED declaration, `declare (b1
  based bp, b2 based bp) byte`, which code generation looked up among what
  it had laid out: `.b2` was `dw B2`, or `dw @P$B2` in a procedure P that
  declares it before the list, um80's "Undefined symbol", `.memory` of
  `(memory based bp) byte` `dw MEMORY`, and, in a procedure or a DO block
  that declares it further down, the module's variable of the name, `dw
  B2`, without a word (0.4.3 the same).  V3.1 compiles them (Known
  issues).  The message names the list, DATA or INITIAL; an INITIAL one's
  said DATA.
- **A factored BASED declaration on a member**, `declare (a based s.p)
  byte`, is BASED, as `declare a based s.p byte` is: each name is at the
  address the member holds.  It was a variable of its own, and `a = 5`
  stored there (0.4.3 the same); one on a variable, `(a based p) byte`,
  was BASED.  One on a member of a BASED structure, which V3.1 rejects, is
  refused, as the plain form is (Incompatible).
- **A base is the declaration of its name made before the variable BASED
  on it**, in its block or one around it, as V3.1 takes it, at every
  level, and wherever that variable is used.  Code generation looked the
  base up by its name where the variable was used: `declare q address;
  declare a based q byte;` used in a procedure that declares a `q` of its
  own, `a = 77h` stored through that `q` (0.4.3 the same).  And a
  declaration of the name further down the block was the base, where V3.1
  takes an outer block's: `p: procedure; declare a based q byte; declare q
  address;` of a module's `q` took the procedure's (0.4.3 the same, but
  that `(a based s.p) byte` was a variable of its own).  A declaration
  that hides the base where it is looked up is renamed, `Q?2`, as one code
  generation would take for another is.
- **`data (.stackptr)`**, `data (.shl)`, the location of a built-in but
  MEMORY in a DATA or INITIAL list, which uplm80 refuses since 0.4.2,
  says so in its own words: the message said V3.1 rejects it (#123), which
  V3.1 does in an expression only, and in a list takes it for an address
  of its own (Known issues).

### Known issues

0.4.3's Known issues stand, but for these, which are V3.1's errors now
(Incompatible): a call of a procedure from inside itself (#170), a
procedure declared in a REENTRANT one and a REENTRANT one in another
procedure (#88, #39), an END that names the first of two labels on a DO
(#20), `b = a(1, 2)` of an array (#114), and SHL and SHR in a DATA or
INITIAL list or an AT address; `declare p address initial (a)` of an
array, whose message names V3.1's #151 now; the `ld l,a / ld h,0` before
an 8-bit SHR (Changed); of the places where the SHL warning was not
given, all but a store past an array's end into the variable shifted,
which it still does not see (Fixed); and a subscript on a scalar, `x(1)`
(#127), which uplm80 still compiles with a warning, but for the location
of one in a restricted expression, `data (.x(1))`, V3.1's #149 now
(Incompatible).  More INITIAL or DATA values than a scalar holds
(#209) is so of an array's too, `declare b (2) byte data (1, 2, 3)`, which
uplm80 lays out after it.  And these, which checking 0.4.4 found, left for
a later release (0.4.3 the same):

- LENGTH, LAST and SIZE of a reference with more than one subscript,
  `size(a(1, 2))`, `size(s(1, 2).n)`, `length(s(1, 2).m)`, which V3.1
  takes, not evaluating the subscripts, are refused, with uplm80's own
  message (`SIZE() needs a variable, ...`), not V3.1's value.
- A typed procedure with no RETURN, `h: procedure byte; b = 1; end h;`,
  which V3.1 rejects (ERROR #156, MISSING RETURN STATEMENT IN TYPED
  PROCEDURE), is compiled, and returns what A or HL holds at its end.
- **V3.1's arithmetic in a restricted expression** is a constant's, a
  number below 256 a BYTE and BYTE arithmetic in eight bits, in a DATA or
  INITIAL value, an AT address and a subscript of a location, where
  uplm80 computes in sixteen: `declare w address data (200 + 100)` is
  002CH to V3.1 and 012CH to uplm80, `255 + 1` 0000H and 0100H, `1 - 2 +
  256` 01FFH and 00FFH, `initial (128 + 128 + 1)` 0001H and 0101H, `-1`
  00FFH and 0FFFFH, `data (.a(200 + 100))` .a+2CH and .a+12CH, `at (200 +
  100)` 002CH and 012CH.  In a statement uplm80 computes `w = 200 + 100`
  and `w = -1` as V3.1 does, 002CH and 00FFH.  No program of MP/M II,
  80un or `sample_code` has one.
- **A string that fills an ADDRESS in a DATA or INITIAL list** is a
  constant to V3.1, its first character in the high byte, as in an
  expression: `declare w (2) address data ('ABCD')` is 4142H, 4344H, and
  `'ABC'` 4142H, 0043H, where uplm80 lays the characters out in order,
  4241H, 4443H.  In a statement uplm80 computes `q = 'PQ'` as V3.1 does,
  5051H.  No program of MP/M II, 80un or `sample_code` has one.
- **The location of a built-in but MEMORY in a DATA or INITIAL list**,
  `data (.stackptr)`, `data (.shl)`, `data (.double)`, V3.1 takes for an
  address of its own, 057CH, 0103H and 057CH in the programs checked, and
  uplm80, which has none to give, refuses (since 0.4.2);
  `tests/test_intel_oracle.py` holds V3.1 to taking them.

      .STACKPTR: STACKPTR is a built-in, and of the built-ins only MEMORY
      has an address; uplm80 has none to give STACKPTR in a DATA or
      INITIAL list, where Intel's PL/M-80 V3.1 takes .STACKPTR for an
      address of its own

- **An AT that names a variable AT something further down**, `declare z
  byte at (.y(1)); declare y (2) address at (.buf(2));`, V3.1 rejects
  (#213, UNDEFINED RESTRICTED REFERENCE IN 'AT'); uplm80 puts z at
  `.buf(4)`, where V3.1 puts it with y declared first.
- **An untyped DATA list**, which V3.1 rejects (#61, MISSING TYPE) and
  uplm80 takes for an array as long as it is: its values are held to a
  BYTE array's rules, and a message for one names V3.1's errors of such
  an array, not #61, nor the #209 V3.1 gives of a word in it as it does
  of one in a constant list.  V3.1 takes such a declaration for a scalar,
  and the location of an element of it, `data (.hx(1))`, for a subscript
  on what is not an array besides (#149).
- **A constant list that names an array whose DATA or INITIAL list gives
  it more than one value**, `.(.a(1))`, `.(a(1))`, which both reject, V3.1
  gives #209 besides, on the array's declaration: `a (2) byte data (1,
  2)`, `a (3) byte data (1, 2)`, and of a factored declaration, where the
  list gives its last name more than one, `(a, b) (2) byte data (1, 2, 3,
  4)` whether the constant list names A or B, not `(1, 2, 3)`, whose B has
  one; in the programs checked, not of a structure, nor of an array of
  `(*)` declared after the constant list.  The message names the errors of
  the constant list only.  (0.4.3 compiled such a list, the location left
  out: Incompatible.)
- A label in a constant list, `.(lbl, 7)`, `.(7, .lbl)`, which uplm80
  refuses as it does a variable's name or location there, V3.1 fails on,
  writing no listing.
- **The location of a REENTRANT procedure's local or parameter** in a DATA
  list or an AT, and **of a BASED variable** in a DATA or INITIAL list, a
  factored BASED declaration's too, V3.1 compiles, taking each for an
  address: in the programs checked `z address at (.x)` of a REENTRANT
  procedure's local x is x, `w address data (.x)` 4 bytes past `.x` in its
  code, and the location of a BASED variable in a list, printed, 059BH to
  05AAH.  uplm80, which has the local on the stack and no address of a
  BASED variable to give, refuses them (Fixed);
  `tests/test_intel_oracle.py` holds V3.1 to taking them.

      DATA(.X): X has no fixed address (a REENTRANT local)

- **What a restricted expression does not take that is an error
  anywhere** - a member a structure does not have, `data (.s.zz)` (V3.1:
  #112), or of what is not a structure, `.a(1).k` (#148), empty
  parentheses, `.a()` (#151), a name declared nowhere (#105, and #149 of
  `.zz(1)`), or a LITERALLY declared after the list (#105) - uplm80
  refuses in words of its own, as it did, and the message names none of
  V3.1's errors, nor those of the rest of the list: `.(.zz(1))` is #105,
  #149 and #210 to V3.1, and to uplm80 a name not declared.
- A number above 0FFFFH, `declare d address data (70000)`, `w = 70000`,
  which V3.1 rejects (#94, ILLEGAL CONSTANT, VALUE > 65535), uplm80
  compiles to its low sixteen bits, 1170H.
- An empty string in a DATA list, `declare w (*) byte data ('', 7)`, which
  V3.1 rejects (#209), uplm80 takes for no bytes, `w(0)` 7.
- A label as a value or as an assignment's target, `lb: ...; w = lb;`,
  `lb = 1`, which V3.1 rejects (#132, ILLEGAL USE OF LABEL), uplm80
  compiles, to a read of the code at the label or a store into it (0.4.3
  the same); of a LABEL that labels no statement, `w = lb`, the message
  names #105 and #172, not #132 (found checking 0.4.4).
- **Some combinations of two errors**, of which the message names fewer of
  V3.1's errors than V3.1 gives, or another (0.4.4's release check): a
  procedure in a REENTRANT one whose location an AT or a constant list
  names, #88 without the list's #211 or #210; a LABEL that labels no
  statement in `at (.lb)` or `data (.lb(1))`, #211 or #149 without #105
  and #172; `b = f(1)(2)` of a procedure without parameters, #32 without
  #153; `length(w, b) = 1` of a scalar W, #126 and #128 without #157; and
  `output(a(1, 2)) = b`, #114, where V3.1, which takes no port that is not
  a constant (below), gives #107, #108, #116 and #32.
- **A port of INPUT or OUTPUT that is not a constant**, `input(b)`,
  `output(b) = 1`, which V3.1 rejects (#107, ILLEGAL INPUT/OUTPUT PORT
  NUMBER, NOT NUMERIC CONSTANT, and #108), uplm80 compiles, to a call of
  `??inp` or `??outp` (0.4.3 the same).
- These, which 0.4.3 has too (0.4.4's release check): `w = .memory.x`,
  which V3.1 rejects (#110, INVALID LEFT OPERAND OF QUALIFICATION, NOT A
  STRUCTURE, and #32), uplm80 compiles to a load from MEMORY's address,
  and um80 then fails, "Undefined symbol MEMORY"; `e: procedure reentrant
  external`, which V3.1 rejects (#41, CONFLICTING ATTRIBUTE, and #174),
  compiles; and `w = size()` and `w = last(aa(f))` (V3.1: #125, ILLEGAL
  ARGUMENT FOR BUILT-IN PROCEDURE), LENGTH or LAST of a scalar (#157,
  INVALID ARGUMENT, ARRAY REQUIRED FOR LENGTH OR LAST) and `b = f1(1, 2)`
  of a procedure of one parameter (#153) are refused in uplm80's own
  words, whose message names none of V3.1's errors.
- **Flags that V3.1's build leaves otherwise**, at every level, as in
  0.4.2 and 0.4.3: `x + 1` and `x - 1` of a BYTE are `INR` and `DCR` to
  V3.1, which set no carry; a SHR of a BYTE an `ANI` and then `RAR`s, so
  that SIGN, ZERO and PARITY after it are of the masked value before the
  shift; and after some comparisons and IF tests, a MOVE, TIME, a DO
  CASE's dispatch, subscript arithmetic, `x * 0`, `x / 1` and NOT of a
  comparison the flags are other ones too (README, Known differences).
- Found by 0.4.4's release check: some programs V3.1 rejects on two
  counts get a message that names fewer of V3.1's errors, or another one:
  `declare lb label; w = .lb;` names #158 (V3.1: #105, #158, #172);
  `lb: b = 1; w = .lb(1);` #158 (V3.1 adds #127); a typed procedure that
  assigns to its own name, `h = 1`, #128 (V3.1 adds #170); `data (.a((1),
  2))` #150 and #151 (V3.1 adds #209); `at (.stackptr(1, 2))` #211 (V3.1:
  #150, #211) and `at (.input(1, 2))` #211 (V3.1: #149, #150, #211); an
  empty procedure two levels inside a REENTRANT one, #88 (V3.1 adds #174),
  and of errors in two procedures, only the first's; a BASED variable on a
  base V3.1 refuses, named in an AT, #212 (V3.1: #50 only), and a list's
  #209, #210 or #211 without the base's #50, #52, #54 or #55, or without
  #105 and #172 of a LABEL that labels nothing. Each program is refused.
- `data (.stackptr(1, 2))` and `data (.input(1, 2))` are refused in
  uplm80's own words, whose message says V3.1 takes `.STACKPTR` (`.INPUT`)
  for an address of its own; V3.1 does of `.stackptr`, but rejects these
  subscripted forms (#150; #149 and #150).

### Verified

0.4.4 was made on two branches from 0.4.3 (4afe3c7), each verified on its
own: fix/0.4.4 - more of V3.1's errors (the first Incompatible section),
the flags, the smaller code and the LITERALLY columns - on 5ff45b5, and
fix/datafold - restricted expressions, bases and labels - on 91c282a, as
below; and then merged (ceb649e), and the merge checked as one release,
on 5afe194, by a release check whose second round found nothing to hold
it (the first found a store at `-O3` whose flags a reader read dropped as
dead, and 0.4.3's section of this file edited; 61d49cd and 6b5d9bf):

- The suite: 3411 tests pass with Intel's binaries, 3 skipping; without
  them 3074 pass and 340 skip. pylint 9.78, with no message kind 0.4.3
  did not have. `tests/run_tests.sh`: 22 of 22.
- `scripts/difftest.py` and `scripts/abifuzz.py`, 300 seeds each,
  `scripts/namestest.py` 200 and 40 `--modules`, and the storage fuzzer
  200: none fails.
- The flags: structural fuzzers of a shift of a BYTE and a reader, 4500
  cases and 200 programs at `-O0` to `-O3`, and 600 programs aimed at the
  `-O1` to `-O3` transforms: no program prints another value at another
  level, and none a value other than 0.4.2's without a warning. 190
  programs of a shift and a reader against V3.1: 169 print V3.1's values;
  the 21 others are a SHL by 6 or 7, or a SHR, of a BYTE read by ZERO,
  SIGN or PARITY, which V3.1 codes with `ANI` and `RAR` (Known issues).
- Restricted expressions, names in lists, bases and labels: the 4869
  programs of the fix/datafold matrix, 2783 earlier probes and 900 new
  ones, at `-O0` to `-O3`: the same at every level, no um80 failure and no
  wrong value; each either prints V3.1's value, is refused with V3.1's
  errors, or is one of the Known issues.
- `scripts/intel_oracle.py --random 600` (seeds 305001-305600, the
  documented quirks left out): 593 print what V3.1's build prints; the
  7 others are V3.1's `INX SP`/`DCX SP` bug (6) and its multiple
  assignment of an embedded target. `--random 100` with nothing left
  out: every difference is a documented one. `--corpus --normalize`: 38
  the same, 28 rejected by V3.1, `test_move_builtin.plm` as before.
- The 88 compiles of MP/M II's and 80un's PL/M at `-O0`, `-O2` and `-O3`:
  against 0.4.3, 39 files at `-O0` lose the dead `ld l,a / ld h,0` and
  `ld de,0` (Changed), and at `-O2` and `-O3` those and what the peephole
  optimizer then makes of them, and nothing else; the code at `-O2` is
  165,092 bytes, the data 45,264; the same 11 stop, with the same errors.
  The only new warnings are the flags warnings at SHOW (DRI's and
  mpm2's), MSCHD and TOD, each a CARRY read after `b = shl(b, 3) + shl(b,
  1)`, which from b = 26 is 1 under V3.1 and 0.4.4 and 2 under 0.4.2.
- 80un 0.3.3 built at `-O0`, `-O2` and `-O3` writes all 205 files of its
  tests byte for byte as 0.4.3's build does. MP/M II V2.0 and V2.1 built
  from source with 0.4.3 and with 0.4.4 (`tools/build.py`, `build_all.sh
  --tree=src`), `run_tests.sh all` and `src` passing with the same results
  under both, and `verify_dri.py` the same.

#### fix/0.4.4

On 5ff45b5, with upeepz80 0.2.7 and um80 0.3.52, against 0.4.3 (4afe3c7)
and c1bc8e9, 0.4.4 before its fourth round of checking.

- The suite: 2141 tests pass, where 0.4.3 had 1462.  Without Intel's
  binaries 135 of the oracle's tests that need them skip, as does
  `tests/test_divmod_dri.py`'s check of `PLM80.LIB`'s divide, and 2005
  pass.  pylint rates the package 9.77 (0.4.3: 9.75), with no message
  kind 0.4.3 did not have; `fixme` is gone with the TODO over `ld de,0`.
- `tests/run_tests.sh`: all 22 programs pass.
- `scripts/difftest.py`, `scripts/abifuzz.py` and `scripts/namestest.py`,
  200 seeds each (160000-160199, 161000-161199, 162000-162199; and
  namestest's `--modules`, 40, 162500-162539): every program prints what
  the model, its `-O0` build or its scopes say.
- `scripts/intel_oracle.py --random 300` (seeds 163001-163300), leaving
  out `shift9`, `wide-limit`, `sub-zero`, `zero-dividend` and
  `neg-widened`: 298 programs print at `-O0` to `-O3` what Intel's PL/M-80
  V3.1 build prints.  The two others are V3.1's (README, Known
  differences): its `DCX SP` bug in the pattern of a ROL in a DO's start,
  `rol(65535, 1)` (seed 163195), and its count of the stack gone below
  zero, a run of `POP PSW`, in `bw = double(((32767 or sa(1).z((shr(255,
  8)) and 1)) xor (00h and bb)));` (seed 163150); 0.4.3's builds and
  c1bc8e9's differ from V3.1's the same way.
- `scripts/intel_oracle.py --corpus --normalize`: of the 67 programs of
  `tests/` and `sample_code/`, 38 print what V3.1's build prints, V3.1
  rejects 28, and `tests/test_move_builtin.plm` differs where V3.1's MOVE
  of 0 bytes moves 65536, as with 0.4.3.
- The 88 compiles of MP/M II's and 80un's PL/M (0.4.3's 87, and 80un
  0.3.3's `names.plm`, one at a time), at `-O0`, `-O2` and `-O3`: the
  assembly is c1bc8e9's, byte for byte.  Against 0.4.3's, that of those
  with a widened BYTE or an `ld de,0` dead after it is smaller (Changed),
  and DP.PLM's at `-O2` and `-O3` spells two instructions otherwise; the
  rest is 0.4.3's.  The 11 that stop, 0.4.3's ten and `names.plm`, a
  module of 80un's that names the others', stop with 0.4.3's errors: no
  new error.  The warnings are 0.4.3's but for the CARRY of SHOW (DRI's
  and mpm2's), MSCHD and TOD (Fixed).  The code at `-O2` is 165,092 bytes
  (0.4.3: 166,091), the data 45,264.
- `sample_code/` and the programs of `tests/`, 67, against c1bc8e9: the
  same messages but the six warnings of `tests/test_byte_shifts.plm`
  (Fixed); the same code at `-O0`, and at `-O2` and `-O3` but for the
  three programs that read flags the entry before names.
- The flags warning and every level's flags: of 200 generated programs of
  12 cases each (seeds 164000-164199; and 120 more, 150000-150119, before
  the last change to the optimizer) - a SHL or SHR of a BYTE, what may come
  between, calls through an address and readers in a procedure among it,
  and a flag reader - built at `-O0` to `-O3`, every case that prints one
  value with 0.4.2's `-O0` build and another with 0.4.4's is warned of, of
  its value or at its reader, and every level prints what `-O0` prints.
  c1bc8e9's builds of 6 of the first 12 have unwarned differences, or
  levels that differ.
- 80un's two programs, built from 80un 0.3.3 at `-O0`, `-O2` and `-O3`,
  extract the 130 files of 80un's 21 test archives and compressed files,
  detokenize `PALLOPS.BAS` and MBASIC 5.21's tokenized copies of its four
  text `.bas` files, and refuse the text files, byte for byte as 0.4.3's
  `-O2` build does (cpmemu, binary mode); what they print differs only in
  the size cpmemu says it loaded.

#### fix/datafold

On 91c282a, with upeepz80 0.2.7 and um80 0.3.52, Intel's binaries found.

- The suite: 2473 tests pass, and 3 skip, the check that the message names
  V3.1's errors of `carry()`, `zero()` and `dec()`, whose messages name
  none (0.4.1).  pylint rates the package 9.77 (0.4.3: 9.75), with no
  message 0.4.3 has not.
- `tests/run_tests.sh`: all 22 programs pass.
- `scripts/difftest.py`, 100 seeds (53000-53099), and
  `scripts/namestest.py`, 100 (53200-53299) and 20 with `--modules`
  (53500-53519): every program prints what the model or its scopes say.
- The 88 compiles of MP/M II's and 80un's PL/M - DRI's tree and mpm2's
  overrides, 50 files, and 80un 0.3.3's 36 alone and its two programs - at
  `-O0`, `-O2` and `-O3`, and the 66 programs of `sample_code` and
  `tests/`, give 0.4.3's assembly byte for byte, and its errors where they
  stop (11 and 14 at each level): none has what a restricted expression
  does not take, nor a location in a list or an AT that moved, nor one
  uplm80 gives no fixed address, nor a factored BASED declaration on a
  member, nor a base V3.1 does not take or one a block hides, nor a LABEL
  that labels no statement.
- `scripts/intel_oracle.py`: V3.1 rejects each program of
  `tests/test_names.py` with the errors its message names and no other,
  compiles each of its programs of a location uplm80 gives no fixed
  address, and its builds of those of a factored BASED declaration and of
  bases (`V31_BASES`) print what uplm80's print.
- Each form of declaration - a scalar, a factored one, BASED on a variable
  or on a member, factored BASED on either, an array, one of `(*)`, an
  untyped DATA, a structure, an array of structures and a member of each,
  AT, DATA, INITIAL, PUBLIC and EXTERNAL variables, a LITERALLY, a label
  declared LABEL or not, a procedure, a parameter - named in a DATA list,
  an INITIAL list, an AT and a constant list, declared before the list and
  after it, at module level, in a procedure, in a DO block of either, in a
  procedure nested in either, in a REENTRANT procedure and a DO block of
  one, with a variable of its name around the block and without, called XV
  and MEMORY: 4869 programs, each built by V3.1 and by uplm80 at `-O0` to
  `-O3`, and run where it builds.  uplm80 does the same at every level:
  prints what V3.1's builds print (1289), refuses what V3.1 refuses with
  V3.1's errors (1237), or does what Known issues and the README have -
  compiles INITIAL below module level with a warning (#73, 705), refuses
  the location of a BASED variable or a REENTRANT local that V3.1 takes
  (664, and 480 more with INITIAL below module level), takes an untyped
  DATA (#61, 202), refuses a label in a constant list, on which V3.1
  writes no listing (54), compiles a procedure in a REENTRANT one (#88,
  56) and an AT naming a later AT (#213, 28), refuses a LITERALLY declared
  after the list in its own words (45), leaves out the #209 V3.1 puts on
  an initialized array's declaration (88), and stops at a PUBLIC or
  EXTERNAL variable below module level (#73) before the constant list V3.1
  gives #210 for too (21).  f452a39 did otherwise with 244 of them, each
  naming a factored BASED declaration's name: it compiled 173, 86 of them
  of a name BASED on a member, which it took for a variable of its own,
  failed in um80 on 45, and refused 26 for what they are not, 18 as naming
  a member the structure has not ("no member M") and 8 as a REENTRANT
  local; and 20 more it refused in the words it does now, but that it
  named an INITIAL list DATA.  On 91c282a the 1594 of them with a BASED or
  a LABEL declaration give at each level what they gave on 1bc31ab, and
  what changed since does not reach the rest.
- The programs of these checks' earlier rounds and of the release check's
  (1037 lists and declarations, 25 and 92 programs, the 836 of its second
  round and the 793 of its third) give on 1bc31ab, at each level checked,
  what they gave when those checks ran, but for 18: 11 names of a factored
  BASED declaration in a list, which uplm80 refuses as BASED now, and 7
  refusals of a location in an INITIAL list, which name the list INITIAL
  now.  So of the 1037 lists and declarations - the release check's 144
  constant lists and 155 DATA, INITIAL and AT values, 238 these checks
  wrote and 500 random ones - at `-O0` and `-O2`, uplm80 refuses those
  V3.1 refuses, with V3.1's errors, and compiles those V3.1 compiles to
  print what V3.1's builds print, but for 41 of the Known issues: 28 with
  more values than a declaration holds, 5 untyped DATA lists, 4
  `.stackptr`s in a DATA list and 4 labels in a constant list.  Of the 25,
  whose lists and ATs name what a procedure, a DO block or the module
  declares further down, or in a block around them, 24 print at `-O0` to
  `-O3` what V3.1's builds print, and V3.1 rejects the other, an AT naming
  a later AT (#213, Known issues).  Of the 92, of two subscripts, a
  subscript on what is not an array, a procedure, a label, a BASED
  variable or a REENTRANT local in a list or an AT, and a location in the
  subscript of one, uplm80 at `-O0` to `-O3` refuses those V3.1 refuses,
  with V3.1's errors, compiles those it compiles to print what its builds
  print, and refuses the REENTRANT locals, the BASED variables and the
  `.stackptr(1)` V3.1 takes, but for 3 of the Known issues (two subscripts
  on an array in an expression, twice, and a label in a constant list).
  Of the 836, at `-O0` and `-O2` uplm80 does what V3.1 does but for 52: 46
  of the Known issues and of the README's, 3 that print an address V3.1
  lays out elsewhere, and 3 that both reject, uplm80 first for another
  error or in its parser.  On 91c282a the 1378 of these with a BASED or a
  LABEL declaration give at each level what they gave on 1bc31ab, and what
  changed since does not reach the rest.
- The release check's fourth round's 1078 programs give on 91c282a what
  they gave on 98beb8c, but for 35, which now print what V3.1's builds
  print or are refused as V3.1 refuses them, with its errors: 7 of a base
  whose name its block declares again further down (Fixed), 20 of a base
  V3.1 does not take and 8 of a LABEL that labels no statement
  (Incompatible) - of those, one V3.1 gives #32 and #118 besides, of a
  `call w(1)` of an array, which uplm80 takes.
- 74 programs of a base - of each form Incompatible lists, of one hidden
  where the BASED variable is used or declared, plain and factored, at
  module level and in a procedure - with V3.1: uplm80 at `-O0` to `-O3`
  prints what V3.1's builds print (12) or refuses what V3.1 refuses with
  V3.1's errors (56), but for 5 - a base declared nowhere, twice, refused
  as not declared, where V3.1 gives #54; one of an untyped DATA (#61,
  Known issues); and two in a procedure with no statement, refused for
  that (#174) - and one that V3.1 builds but cannot link, of an EXTERNAL
  base nothing defines.

## 0.4.3 — 2026-09-26

SHL and SHR of a BYTE are a BYTE, as the manual and Intel's PL/M-80 V3.1
make them, with a warning where what a program written for uplm80
computes changes; 80un's sources are fixed for it.  The errors V3.1 gives
that 0.4.2 did not, its Known issues, are given, but for two forms
uplm80's own tests exist to test, which stay, with a warning.  Messages
name the source's line after a LITERALLY over several lines, and its
column after `out:end p;`.

### Incompatible: SHL and SHR of a BYTE are a BYTE

**SHL and SHR of a BYTE are a BYTE**, shifted in eight bits, as the manual
gives them the type of their pattern (11.1.4) and Intel's PL/M-80 V3.1
compiles them (0.4.2's Known issues).  uplm80 zero-extended a BYTE pattern
and shifted it in sixteen bits, with an ADDRESS result: `w = shl(b, 4)`
with b = 0F0H was 0F00H, and is 0000H.  The bits shifted out of a BYTE are
lost, a count of 8 or more leaves 0 (V3.1 shifts by the count mod 8:
README, Known differences), and the arithmetic around the shift is a
BYTE's: with c = 3 and z = 0, `w = shl(c, 7) + 0ffh` is 007FH and `w =
shr(z, 4) - 1` 00FFH, where they were 027FH and 0FFFFH.  A SHR of a BYTE
has the value it had.  `b * 16` is still `SHL(DOUBLE(b), 4)`, and
`SHL(DOUBLE(hi), 8) OR lo` still builds a word in two loads.  The program
in `tests/test_expression_types.py` prints at `-O0` to `-O3` what V3.1's
build of it prints.

It changes what a program written for uplm80 computes where it shifts a
BYTE and uses the bits shifted out: 80un built words with `lo + shl(b,
8)` and buffer addresses with `shl(i, 7)`, and built with SHL of a BYTE a
BYTE, its `80un test.arc` extracted one member of thirteen (0.4.2).  80un's
sources shift `SHL(DOUBLE(x), n)` where they want the bits kept, since its
branch fix/byte-shifts (80un 0.3.3, unreleased).  To keep the old meaning,
write `SHL(DOUBLE(x), n)`, which is right under any compiler.

The compiler warns where the two meanings can differ, but for the few
places Known issues lists (`uplm80/byte_shifts.py`): at a SHL of a BYTE that can shift a set bit out,
whose result, through what it is part of, reaches a place that uses the
bits above its low byte - a store to an ADDRESS, an ADDRESS argument or
RETURN, a subscript, a relation, `/` and `MOD`, SHR, SCL, SCR, HIGH and
DOUBLE of it, a DO CASE's selector, or an ADDRESS beside it, which such a
place then uses.  Intel's PL/M-80 gives no warning.

    warning: SHL(B, 4): SHL of a BYTE is a BYTE (Programming Manual
    9800268B, 11.1.4), and the bits shifted out of it, which this
    expression uses, are lost; SHL(DOUBLE(B), 4) keeps them. uplm80 before
    0.4.3 shifted a BYTE in 16 bits

A SHL whose pattern's largest value, shifted by its count's, fits in eight
bits loses nothing: `shl(dcnt and 11b, 5)`, `shl(3, 4)`, `shl(n, 2)` after
`if n > 32 then return;`.  A variable's largest value is what the program
assigns it anywhere, and at the SHL what the statements before it have left
it, the tests it passed and the case of a DO CASE on it included; a typed
procedure's is what its RETURNs give.  That is followed for a scalar whose
address is never taken, nor AT, BASED, PUBLIC or EXTERNAL, across calls of
procedures that do not assign it.  There is no warning for a SHR, nor for
BYTE arithmetic around a shift that loses nothing (`shr(z, 4) - 1` above).
Of the programs checked, only CP/M 2.0's STAT in `sample_code` has such
arithmetic whose value the compiler cannot bound, `.devr(shl(iobyte and
11b, 2) + j)`, whose j keeps the sum below 256.  Where it can, the values
are 0.4.2's: STAT's `.devr(shl(i, 4) + j)`, i at most 3 and j 12, and
MP/M II MPMLDR's `(shr(nmb$cns - 1, 2) + 1) * 256` and the like, DRI's
and mpm2's, at most 64 before the product.  The warnings, of the programs
checked:

- 80un's sources before the fix: the 29 places the fix changed, and no
  other; its sources after it: none.  `tests/bug_80un.plm`, 80un's old
  single-file source, which 80un keeps as `src/plm/archive/80un.plm`: 20,
  each of a kind 80un's fix changed.
- MP/M II, DRI's tree and mpm2's overrides: 7, in ERA, REN, SET (2), SHOW
  (DRI's and mpm2's) and STAT, each `shl(dcnt, 5) + .tbuff` or the like,
  of a directory code BDOS returns, 0 to 3 or 0FFH, which the program has
  tested against 0FFH, but for SHOW: its `readlbl` leaves 0FFH when the
  search does not find the label `getlbl` reported, and `shl(dcnt, 5)` is
  then 0E0H, as V3.1 computes it.  The compiler cannot know dcnt is below
  8, and the programs do what V3.1's builds of them do.
- `sample_code`, and the programs of `tests/` but those that test the
  warning and the generators' (Verified): none.

Of the 87 compiles of MP/M II's and 80un's PL/M, 24 of MP/M II's and 12
of 80un's are smaller, by 619 and 209 bytes in all at `-O2`.

### Incompatible: more errors Intel's PL/M-80 gives

What 0.4.2's Known issues listed as compiled by uplm80 and rejected by
Intel's PL/M-80 V3.1 is refused as V3.1 refuses it, with its error's
number and text, when the parser gives the program, at every `-O` level.
Two forms uplm80's own tests exist to test are compiled as before, with a
warning that names V3.1's error.  No program of MP/M II (DRI's tree and
mpm2's overrides), of 80un or of `sample_code` has any of these; the tests
whose programs had one in passing, a procedure with no statements or a
call of one declared further on, have it no more.  V3.1 rejects each
program of `tests/test_names.py` with the error the message names
(`tests/test_intel_oracle.py`), and compiles the program of the forms it
takes to print what uplm80's build prints.

- **An END that names another block** (ERROR #20, MISMATCHED IDENTIFIER
  AT END OF BLOCK): `p: procedure; ... end q;`, `out: end q;`, `l: do;
  ... end m;`, `do; ... end n;`.  uplm80 took no notice of the name.

      END Q: the END of procedure P names Q; Intel's PL/M-80 V3.1 rejects
      it (ERROR #20, MISMATCHED IDENTIFIER AT END OF BLOCK)

- **A DO CASE with no case**, `do case n; end;` or `do case n; l: end;`
  (ERROR #201, INVALID DO CASE BLOCK, AT LEAST ONE CASE REQUIRED).
- **The address of a call**, `.h(1)` of a procedure H (ERROR #104,
  ILLEGAL PROCEDURE INVOCATION WITH DOT OPERATOR); `.h` is its address.
- **Anything in parentheses in a subscript of the argument of LENGTH,
  LAST or SIZE** - a call, `size(ab(h(1)))`, a subscript, `size(ab(ab(1)))`,
  a built-in or a parenthesized expression, `size(ab((i)))` (ERROR #32,
  INVALID SYNTAX, TEXT IGNORED UNTIL ';'): V3.1 takes no such subscript,
  and uplm80, which does not evaluate the subscripts there, did not call
  `h`.  `size(ab(i + 1))`, `size(ab(f))` of a typed procedure and
  `size(sa(i).z)` are taken, as V3.1 takes them.
- **A procedure with no statements** (ERROR #174, INVALID NULL
  PROCEDURE), `g: procedure; declare k byte; end g;`, a label on its END
  or not; uplm80 made it return.
- **An array or a member array without a subscript**, but after a dot and
  as the argument of LENGTH, LAST or SIZE (3.6.2): `a = 3`, `x = a`,
  `call p(a)`, `sz(a)` (ERROR #133, ILLEGAL REFERENCE TO UNSUBSCRIPTED
  ARRAY) and `s.m = 4` (ERROR #134, ILLEGAL REFERENCE TO UNSUBSCRIPTED
  MEMBER ARRAY), which uplm80 took for `a(0)` and `s.m(0)`.  A member of an
  array of structures named without its subscript, `s2.m(4)`, is taken
  for `s2(0).m(4)` as before, with a warning:
  `tests/test_calls_and_loops.py` tests what a store through it reaches.

      A: A is an array, and an array is named without a subscript only as
      the operand of a dot or the argument of LENGTH, LAST or SIZE
      (Programming Manual 9800268B, 3.6.2); Intel's PL/M-80 V3.1 rejects
      it (ERROR #133, ILLEGAL REFERENCE TO UNSUBSCRIPTED ARRAY)
      warning: S2.M: S2 is an array, and this is taken for S2(0).M;
      Intel's PL/M-80 V3.1 rejects it (ERROR #133, ...)

- **A subscript on a scalar** (ERROR #127, INVALID SUBSCRIPT ON
  NON-ARRAY): `x(1)` is taken, as before, for the element that far past x,
  with a warning - `tests/test_optimizer_soundness.py` tests it - and
  `shl(w, 3)` of a program's ADDRESS SHL, two subscripts, which uplm80
  took for a call through SHL's value, is an error (#127, and #114,
  INVALID SUBSCRIPT, MULTIPLE SUBSCRIPTS ILLEGAL).  `call q(1, 2)` of an
  ADDRESS q calls through it (8.2.1), as V3.1 has it.
- **A call of a procedure declared further on** (ERROR #169, ILLEGAL
  FORWARD CALL): `p: procedure; call q; end p; q: procedure; ...`, and `y
  = f + 1` of a typed `f` declared after.  A REENTRANT procedure may call
  one declared after it that is REENTRANT too, as V3.1 allows and MP/M
  II's SN.PLM does, and the address of a procedure, `.q`, may be taken
  before its declaration, as the INITIAL lists of MP/M II's resident
  processes take theirs.

      Q: procedure Q is declared after this call of it, and a procedure is
      called only after its declaration, but by a REENTRANT procedure if
      it is REENTRANT too; Intel's PL/M-80 V3.1 rejects it (ERROR #169,
      ILLEGAL FORWARD CALL)

### Fixed

- **The columns after a label's colon against END**, `out:end p; b =
  zz;`, are the source's (0.4.2's Known issues).  The front end puts a
  null statement between the labels on an END and the END, and where no
  blank follows the colon for its `;` to take the place of, it put one in,
  and a message about what followed on the line gave a column one too far
  for each.  A second such label on a line was not marked the end of its
  block either: `do case k; ... x:end; y:end;` counted `y:` among its
  cases.
- **A LITERALLY whose text runs over several lines no longer moves the
  lines after its use.**  The macro pass put the text in with its line
  ends, so every message about a later line named a line too far on, by
  the text's line ends at each use: MP/M II's ERA, REN and SET use
  `PROCES.LIT`'s PROCESS$DESCRIPTOR once, whose text, with the texts it
  names, has 16 line ends, and were reported 16 lines on.  The text goes
  in on one line, as a line end in it is a blank; the 87 MP/M II and 80un
  compiles are unchanged.
- In the oracle, `scripts/intel_oracle.py` (0.4.2's Known issues):
  `--avoid zero-dividend` leaves out a dividend that folds to 0 divided by
  what folds to 0 too, `(8 / 0FF00H) / (0F82AH <= 1)`, which V3.1 folds to
  0 and uplm80 divides (the 0.4.2 release check's seed 20275), and a
  product with 0, `((k0 * 0) * x) / y` (0.4.3's seed 50094), as well as
  `0 / x` and a constant that overflows to 0, `256 * 256 / x` (the
  release check's seed 96044, which V3.1 prints 0000 and uplm80 0FFFFH);
  and where a label on the module's END is one a procedure's GOTO
  reaches, `fin: end t;`, the HLT made a warm boot is the one after the
  `LXI SP` V3.1 puts at the END, and Intel's build no longer runs on to
  `timeout`.

### Known issues

uplm80 still compiles these, which Intel's PL/M-80 V3.1 rejects, with a
warning that names V3.1's error (Incompatible):

- `f()` and `CALL g()` of a procedure, taken for `f` and `g` (ERROR #102,
  MISSING PRIMARY OPERAND, and #153, INVALID NUMBER OF ARGUMENTS IN
  CALL).
- A subscript on a scalar, `x(1)`, the element that far past x (#127).
- A member of an array of structures without its subscript, `s2.m(1)`,
  taken for `s2(0).m(1)` (#133).
- INITIAL in a procedure's declaration, or a DO block's, which
  initializes the variable once, when the program is loaded (#73, INVALID
  ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL).

And these, without a word, which checking 0.4.3 against V3.1 found:

- A call of a procedure, not REENTRANT, from inside itself, `r:
  procedure; ... call r; end r;` (#170, ILLEGAL RECURSIVE CALL).
- A procedure declared in a REENTRANT one (#88, INVALID PROCEDURE
  NESTING, ILLEGAL IN REENTRANT PROCEDURE), and a REENTRANT procedure
  declared in another procedure (#39, INVALID ATTRIBUTE OR
  INITIALIZATION, NOT AT MODULE LEVEL).
- More INITIAL or DATA values than a scalar holds, `declare y byte
  initial (1, 2)`, which fill the bytes after it (#209, ILLEGAL
  INITIALIZATION OF MORE SPACE THAN DECLARED).
- An END that names the first of two labels on a DO, `a: c: do; ... end
  a;` (#20, MISMATCHED IDENTIFIER AT END OF BLOCK): V3.1 takes only the
  label next to DO, `end c;`.
- SHL and SHR in a DATA or INITIAL list or an AT address, `declare w
  address data (shl(0f0h, 4))` (#151, INVALID OPERAND IN RESTRICTED
  EXPRESSION), which `-O0` refuses, and `-O1` and up fold with the BYTE
  shifted in 16 bits, 0F00H, where the same SHL in a statement is 0 (0.4.2
  the same).

And this V3.1 compiles to other code (0.4.2 the same):

- **What a store through a pointer or an overrun reaches in or from
  `??AUTO`** is uplm80's layout, not DRI's (Known issues, 0.3.7), and a
  counted loop over a local in `??AUTO` does not see all of it. `??AUTO`
  comes first in the data segment, before the module's variables, and
  holds the frames of procedures active together one after another: an
  overrun of a local in it can reach another frame or the module's first
  variables, and `.x - 1` of the first variable after it its last byte.
  A loop over a local in `??AUTO` ends where a pointer or an overrun from
  a local of its own procedure declared before the index sets it, and
  nowhere else. With q's `ql(2) byte, k byte` just before run's index `i`
  in `??AUTO`, `ql(3) = 20` from inside run's loop sets `i` and the loop
  still runs its count, 000B 0014 at `-O0` to `-O2`, where V3.1's build,
  whose layout has `i` there too, prints 0003 0015. Counting no such loop
  would cost ED and PIP 18 and 17 bytes at `-O2`, and 80un 20.

Found by 0.4.3's final release check, and left for a later release:

- `b = a(1, 2)` of an array (#114, INVALID SUBSCRIPT, MULTIPLE SUBSCRIPTS
  ILLEGAL) compiles, to a call through the value of `a(0)`; 0.4.3 makes
  the same of a scalar an error, not of an array (0.4.2 the same).
- `declare p address initial (a)` of an array is refused, but the message
  names V3.1's #133, where V3.1 gives #151.
- The SHL warning is not given where only these tell the two meanings
  apart; 0.4.3 computes V3.1's value at each: PLUS, MINUS or CARRY after
  an operation on a SHL of a BYTE (`b = (shl(k, 4) + 10h) plus 0` with k =
  0FFH was 00 and is 01; `b = shl(k, 1); c = carry;` was 00 and is 0FFH),
  the carry now coming from an 8-bit operation; a test that assigns the
  variable it bounds, `if k < 4 and (k := 200) > 0 then w = shl(k, 6);`;
  a store past an array's end into the variable shifted; and an INTERRUPT
  procedure's assignments.
- An 8-bit SHR can leave a `ld l,a / ld h,0` before it that nothing reads
  (DA.PLM): bytes, not wrong code.

### Verified

On 56fad40, with upeepz80 0.2.7 and um80 0.3.52; the suite, pylint,
`run_tests.sh`, the three fuzzers, the oracle's corpus and the release
check's campaign again on 5408683, whose `uplm80/` and `scripts/` are
56fad40's.

- The suite: 1462 tests pass.  Without Intel's binaries the oracle's 60
  tests that need them skip, as does `tests/test_divmod_dri.py`'s check of
  `PLM80.LIB`'s divide, and the oracle's 9 others pass.  pylint rates the
  package 9.75, with no message 0.4.2 did not have.
- `tests/run_tests.sh`: all 22 programs pass.
- `scripts/difftest.py`, `scripts/abifuzz.py` and `scripts/namestest.py`,
  200 seeds each (40000-40199, 41000-41199, 42000-42199; and namestest's
  `--modules`, 40): every program prints what the model, its `-O0` build
  or its scopes say.
- `scripts/intel_oracle.py --random 300` (seeds 50001-50300), leaving out
  `shift9`, `wide-limit`, `sub-zero`, `zero-dividend` and `neg-widened` -
  not `shl-byte`: 298 programs print at `-O0` to `-O3` what Intel's
  PL/M-80 V3.1 build prints.  The two others are V3.1's bugs (README,
  Known differences): seed 50265 its `INX SP` for `INR A`, and seed 50252
  a new one, `w = (ew := b) + b` of a BYTE b and an ADDRESS ew, which V3.1
  adds in 16 bits, 01D8H for b = 0ECH, where the embedded assignment is a
  BYTE (4.6.3) and the sum 00D8H.  The first run had a third, seed 50094,
  a product with 0 divided by 0, which `zero-dividend` now leaves out.
- The release check's `scripts/intel_oracle.py --random 600` (seeds
  80001-80600), leaving out the same: 599 programs print what V3.1's build
  prints.  Seed 80353 is another V3.1 bug (README, Known differences):
  `w2, w3 = (eb := w2)` of ADDRESS w2 and w3 and a BYTE eb, where V3.1
  takes w2's value, 0FFFEH, for an address, stores 0FFF5H at 0FFF5H and
  in w3, and leaves w2 unstored, where the manual makes each target
  0FFFEH (4.6.3).
- `scripts/intel_oracle.py --corpus --normalize`: of the 67 programs of
  `tests/` and `sample_code/`, 38 print what V3.1's build prints, V3.1
  rejects 28, and `tests/test_move_builtin.plm` differs where V3.1's MOVE
  of 0 bytes moves 65536, as with 0.4.2.
- The 87 compiles of MP/M II's and 80un's PL/M (DRI's tree and mpm2's
  overrides, each in the mode `tools/build.py` uses; 80un's files, from its
  fix/byte-shifts branch, one at a time and its two programs), against
  0.4.2: the assembly of those that shift a BYTE changes, to the shift in
  A, and each is smaller - at `-O0` 33 (24 of MP/M II's, 625 bytes in
  all; 9 of 80un's, 179), at `-O2` 36 (24, 619; 12, 209), at `-O3` 34 (24,
  609; 10, 202).  The rest compile to 0.4.2's assembly, and the ten that
  stop, stop with 0.4.2's errors.  The code at `-O2` is 164,354 bytes, the
  data 45,088.  No new error; the new warnings are the SHL warnings above,
  7 of MP/M II's and 20 of 80un's old source, and none of the new errors'
  warnings.
- 80un's two programs, built from fix/byte-shifts at `-O0`, `-O2` and
  `-O3`, extract the 136 files of the 23 archives and compressed files of
  80un's tests, detokenize its one tokenized BASIC file, `PALLOPS.BAS`,
  refuse its four text `.bas` files (`samples/bas/`: "Not a tokenized
  BASIC file"), and detokenize MBASIC 5.21's tokenized copies of those
  four (`SAVE` under cpmemu), `GOTO` line numbers, `&H` and `&O`
  included, all byte for byte as 0.4.2's build of 80un's own sources does
  (cpmemu, binary mode); what they print differs only in the size cpmemu
  says it loaded.
- The warnings of `tests/`' programs, the pytest's included, besides those
  above: the SHL of a BYTE warning in the programs that test it
  (`tests/test_shl_of_a_byte.py`, `tests/test_expression_types.py`'s
  byte shifts, `shl(b, 9)` of `tests/test_intel_oracle.py`) and in the
  generators' programs, which shift a BYTE by a computed count; the
  warning of `x(1)` in `tests/test_optimizer_soundness.py`, of `s2.m(4)`
  in `tests/test_calls_and_loops.py`, and of `size(i(1))` in
  `tests/test_expression_types.py`, whose program is an error anyway.
- The final release check, on 161ee2a (the release less its upeepz80
  floor commit, which is not in it, and with `uplm80/` and `scripts/` as
  56fad40's): the suite, 1463 tests, with Intel's binaries, and 1402
  with 61 skipped without them; pylint 9.75; `run_tests.sh` 22 of 22;
  difftest and abifuzz 300 seeds each, namestest 200 and 40 `--modules`,
  the storage fuzzer 200, none failing; `intel_oracle.py --random 600`
  (seeds 105001-105600, the same left out): 594 print what V3.1's build
  prints, 5 differ by V3.1's `INX SP`/`DCX SP` bug and 1 by its 50252 bug;
  `--random 100` with nothing left out (106001-106100): the 56 that
  differ are each a documented difference. The 87 MP/M II and 80un
  compiles as above, and 80un's programs extracting 201 files per build
  as 0.4.2's build of 80un's own sources does. MP/M II V2.0 and V2.1 built
  from source with 0.4.2 and with 0.4.3 (`tools/build.py`, `build_all.sh
  --tree=src`), `run_tests.sh all` and `src` passing at both versions
  with the same results, 44 and 30, and `verify_dri.py` the same; 16
  utilities differ, each one that shifts a BYTE.

## 0.4.2 — 2026-09-26

uplm80 checked against Intel's own PL/M-80 V3.1 as a matter of course:
`scripts/intel_oracle.py` builds a program with both compilers and
compares what the builds print, and the suite runs it where Intel's
binaries are found. What it and 0.4.1's Known issues found is settled:
LENGTH, LAST and SIZE of a qualified reference, which uplm80 refused; a
store through MEMORY that a counted loop did not see; a label on an END
statement; and the errors V3.1 gives and uplm80 did not, which stay
warnings where a program written for uplm80 relies on the form. SHL and
SHR of a BYTE stay an ADDRESS, as 80un relies on them (Known issues).

### Incompatible: errors Intel's PL/M-80 gives

More of what PL/M-80 does not allow, which uplm80 compiled (0.4.1's Known
issues), is refused as Intel's PL/M-80 V3.1 refuses it, when the parser
gives the program, at every `-O` level. What a program written for uplm80
relies on is compiled as before, with a warning that names V3.1's error:
of MP/M II's and 80un's sources, `sample_code` and the programs of
`tests/`, only the tests use any of these, and only `f()` of a procedure
(`tests/test_implicit_calls.plm`) and INITIAL in a procedure (the pytest's
programs; 80un, uplm80's own, may use it too). Every one of the 87 MP/M II
and 80un compiles, `sample_code` and the test programs compile to 0.4.1's
assembly.

- **A dimension of 0 is an error**, of an array or of a structure's
  member, `declare b (0) byte` (ERROR #57, INVALID DIMENSION, ZERO
  ILLEGAL). uplm80 took it for a scalar: one byte, and a SIZE of 1.

      (0): an array has at least one element, and a dimension of 0 gives it
      none; Intel's PL/M-80 V3.1 rejects it (ERROR #57, INVALID DIMENSION,
      ZERO ILLEGAL)

- **The address of a built-in is an error**, but MEMORY's: `.double`,
  `.stackptr`, `.move`, `.output(3)` (ERROR #123, INVALID DOT OPERAND,
  BUILT-IN PROCEDURE ILLEGAL). uplm80 compiled `.double` to the address
  of a symbol DOUBLE, which nothing defines, and `.output(3)` to a call
  of one.

      .DOUBLE: DOUBLE is a built-in, and of the built-ins only MEMORY has an
      address; Intel's PL/M-80 V3.1 rejects it (ERROR #123, INVALID DOT
      OPERAND, BUILT-IN PROCEDURE ILLEGAL)

- **Empty parentheses after a built-in are an error**, `carry()`,
  `zero()`, `dec()` (ERROR #102, MISSING PRIMARY OPERAND, and #153,
  INVALID NUMBER OF ARGUMENTS IN CALL), as after a variable (0.4.1).
  After a procedure, `f()` and `CALL g()`, they are still taken for `f`
  and `g` - `tests/test_implicit_calls.plm` has `result =
  callee$func();` - with a warning:

      CARRY(): CARRY is a built-in, and PL/M-80 has neither an empty
      subscript nor an empty argument list
      warning: F(): PL/M-80 has no empty argument list, and this is taken
      for F, a call with no arguments; Intel's PL/M-80 V3.1 rejects it
      (ERROR #102, MISSING PRIMARY OPERAND, and #153, INVALID NUMBER OF
      ARGUMENTS IN CALL)

- **A PUBLIC or EXTERNAL procedure or variable in a procedure or a DO
  block is an error** (ERROR #39 and #73, INVALID ATTRIBUTE OR
  INITIALIZATION, NOT AT MODULE LEVEL). INITIAL there initializes the
  variable once, when the program is loaded, as before, with a warning:
  the programs of the pytest have it, and 80un may.

      X: a PUBLIC variable must be declared at the outer level of the
      module, not in procedure P; Intel's PL/M-80 V3.1 rejects it (ERROR
      #73, INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL)
      warning: K: INITIAL in procedure P initializes the variable once,
      when the program is loaded, not at each entry; Intel's PL/M-80 V3.1
      rejects it (ERROR #73, INVALID ATTRIBUTE OR INITIALIZATION, NOT AT
      MODULE LEVEL)

  `tests/test_names.py` has each; V3.1 rejects each of its programs with
  the error the message names (`tests/test_intel_oracle.py`).

### Fixed

- **LENGTH, LAST and SIZE of a qualified reference**, as the manual
  allows them (11.1.2): of a structure's member, `LENGTH(st.z)`; of a
  member of an element of an array of structures, `LAST(sa(i).w)`, or of
  the array, `LENGTH(sa.z)` (partially qualified); and SIZE of an
  element, `SIZE(sa(2))`, `SIZE(ab(2))`, `SIZE(st.z(1))`. uplm80 took
  only a variable's name and stopped at each: "LENGTH() needs an array
  whose extent is known", "SIZE() needs a declared variable" (0.4.1 the
  same). The subscripts are not evaluated, and a LENGTH or LAST that fits
  is a BYTE, as of an array. Intel's PL/M-80 V3.1 compiles the program
  in `tests/test_expression_types.py` - members of structures at module
  level, in a procedure and in a REENTRANT one, and of BASED ones - to
  print what uplm80's build prints at `-O0` to `-O3`. LENGTH or LAST of
  what is not an array - an element, a scalar member, a structure - is
  still an error, as it is to V3.1 (ERROR #125, ILLEGAL ARGUMENT FOR
  BUILT-IN PROCEDURE, and #157, INVALID ARGUMENT, ARRAY REQUIRED FOR
  LENGTH OR LAST), and so is SIZE of a subscripted scalar (#127) or of a
  member no structure has (#112). The oracle's generator no longer leaves
  them out (README, Known differences).
- **A counted loop over the last variable ends where a store through
  MEMORY sets it.** MEMORY begins where the last variable ends, in V3.1's
  layout as in uplm80's, so `p = .memory - 1` with a BASED `b`,
  `memory(0ffffh)`, the subscript wrapping, and `memory(k)` with k =
  0FFFFH each reach the last module-level variable, or a procedure's
  static local laid out last, and a BYTE loop over it, counted in B, ran
  its count: 000B 0014 at `-O0` to `-O3`, where Intel's PL/M-80 V3.1
  build prints 0003 0015 (Known issues, 0.4.1). A store through MEMORY
  now makes every module-level variable and static local reachable, as
  the address of a variable does (0.4.1, Fixed), where `.memory` is
  taken or MEMORY is subscripted by what is not a constant, or by a
  constant of 8000H or more; `memory(5)` runs on from the end, and a
  loop beside it is still counted. The programs in
  `tests/test_calls_and_loops.py` print, at `-O0` to `-O3`, what Intel's
  build of each prints. The 87 MP/M II and 80un compiles, MP/M II's ED
  and STAT among them, which take `.memory` and subscript MEMORY by a
  variable, compile to 0.4.1's assembly at `-O0`, `-O2` and `-O3`.
- **A label on an END statement**, `out: end p;`, as a label may prefix
  any statement (Programming Manual A.4.4.1); it was a syntax error
  (Known issues, 0.4.1). The grammar takes a label only on a statement a
  block holds, so the front end puts a null statement between the labels
  and the END, in place of the blank after the colon where there is one,
  and a GOTO to it goes where Intel's PL/M-80 V3.1 goes: on to the next
  step of an iterative DO and the next test of a DO WHILE, and out of a
  DO, a DO CASE - whose cases it is not one of - and a procedure. The
  program in `tests/test_calls_and_loops.py` prints, at `-O0` to `-O3`,
  what V3.1's build of it prints. No program of MP/M II, 80un,
  `sample_code` or `tests/` has one; each compiles as before.

### Added

- **`scripts/intel_oracle.py`, a differential test with Intel's PL/M-80
  V3.1 as the oracle** (README, Testing against Intel's PL/M-80). It
  builds a program the way DRI's `P.SUB` built its CP/M programs - PLM80,
  LINK with `X0100` and `PLM80.LIB`, LOCATE, OBJCPM - and with uplm80 at
  `-O0` to `-O3`, runs every build under cpmemu, and says whether each
  level prints what Intel's build prints; `--random N` checks programs of
  `tests/plm_intel.py`, a generator of programs in the dialect both
  compilers share, `--corpus` the programs of `tests/` and `sample_code/`,
  and `--reduce` cuts a difference down to the lines it needs. Intel's
  binaries are not in the repository: the oracle finds them on DRI's MP/M
  II work disk, or through `$PLM80_TOOLS`, and runs them on
  `tools/isis`, a small ISIS-II emulator on cpmemu's qkz80 core (`make -C
  tools/isis`), or on romwbw_emu's `tools/romwbw-plm80`.
- `tests/test_intel_oracle.py`, in the suite: generated programs, a
  program of `tests/`, and the programs of this release's tests whose
  output is transcribed from Intel's build, each built by both compilers.
  Without Intel's binaries - as on CI - it skips, building nothing.

### Known issues

uplm80 still compiles these, which Intel's PL/M-80 V3.1 rejects (0.4.1
the same):

- `f()` and `CALL g()` of a procedure, taken for `f` and `g`, with a
  warning (ERROR #102, MISSING PRIMARY OPERAND, and #153, INVALID NUMBER
  OF ARGUMENTS IN CALL; Incompatible).
- A subscript on a scalar, `x(0)` or `x(1)`, the byte at X's address
  plus the subscript (#127, INVALID SUBSCRIPT ON NON-ARRAY); and `shl(w,
  3)` where the program declares SHL an ADDRESS, a call through SHL's
  value (#127, and #114, MULTIPLE SUBSCRIPTS ILLEGAL).
- An array, or an array member, without a subscript anywhere but in a
  location reference or LENGTH, LAST and SIZE (3.6.2): `a = 3` and `x =
  a` are `a(0)`, `s.m = 4` is `s.m(0)`, `s2.m(1)` of an array of
  structures is `s2(0).m(1)`, and `size(a)`, SIZE an array of the
  program's, is `size(a(0))` (#133, ILLEGAL REFERENCE TO UNSUBSCRIPTED
  ARRAY, and #134, ILLEGAL REFERENCE TO UNSUBSCRIPTED MEMBER ARRAY).
- INITIAL in a procedure's declaration, or a DO block's, which
  initializes the variable once, when the program is loaded, with a
  warning (#73, INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL;
  Incompatible).
- A procedure with no statements, `g: procedure; end g;`, which returns
  (#174, INVALID NULL PROCEDURE).
- A call of a procedure that its block declares after the call, `p:
  procedure; call q; end p; q: procedure; ... end q;`, and `y = f + 1`
  of a typed procedure `f` declared after it (#169, ILLEGAL FORWARD
  CALL).
- These four, which 0.4.2's release check found, each rejected by V3.1
  with the error named: an END that names another block, `p: procedure;
  ... end q;` or `out: end q;` (#20, MISMATCHED IDENTIFIER AT END OF
  BLOCK); a DO CASE with no case, `do case n; end;`, and, since a label
  on an END is taken, `do case n; l: end;` (#201, INVALID DO CASE BLOCK,
  AT LEAST ONE CASE REQUIRED); `.p(1)` of a procedure (#104, ILLEGAL
  PROCEDURE INVOCATION WITH DOT OPERATOR); and a subscript that calls a
  procedure inside SIZE, LENGTH or LAST, `size(ab(f(1)))` or
  `length(sa(f(1)).z)` (#32, INVALID SYNTAX), which uplm80 compiles
  without calling `f`.

And these V3.1 compiles to other code (0.4.1 the same):

- **SHL and SHR of a BYTE are an ADDRESS**, the BYTE zero-extended and
  shifted in sixteen bits (`uplm80/plm_types.py`): `w = shl(b, 4)` with
  b = 0F0H is 0F00H, where the manual gives SHL and SHR their pattern's
  type (11.1.4) and V3.1's build gives 0000H (the oracle's `shl-byte`).
  Programs written for uplm80 rely on it: 80un builds words with `lo +
  shl(b, 8)` (common.plm's READ16 and READWORD, the ARC and LBR headers'
  sizes), and compiled with SHL of a BYTE a BYTE, its `80un test.arc`
  extracts one member of the archive's thirteen. DRI's programs, written
  for Intel's compiler, shift a BYTE at 62 places in 18 of MP/M II's
  source files, and 24 of the MP/M II compiles would change. uplm80 keeps
  the 16-bit shift.
- **What a store through a pointer or an overrun reaches in or from
  `??AUTO`** is uplm80's layout, not DRI's (Known issues, 0.3.7), and a
  counted loop over a local in `??AUTO` does not see all of it. `??AUTO`
  comes first in the data segment, before the module's variables, and
  holds the frames of procedures active together one after another: an
  overrun of a local in it can reach another frame or the module's first
  variables, and `.x - 1` of the first variable after it its last byte.
  A loop over a local in `??AUTO` ends where a pointer or an overrun from
  a local of its own procedure declared before the index sets it, and
  nowhere else. With q's `ql(2) byte, k byte` just before run's index `i`
  in `??AUTO`, `ql(3) = 20` from inside run's loop sets `i` and the loop
  still runs its count, 000B 0014 at `-O0` to `-O2`, where V3.1's build,
  whose layout has `i` there too, prints 0003 0015. Counting no such loop
  would cost ED and PIP 18 and 17 bytes at `-O2`, and 80un 20.

Also: where a label's colon is followed at once by END, `out:end p;`,
a message about what follows on that line gives a column one too far,
for the null statement put before the END.

In the oracle, `scripts/intel_oracle.py`:

- `--avoid zero-dividend` leaves out `0 / x`, but not a dividend that
  folds to 0, `(8 / 0FF00H) / (0F82AH <= 1)`, which V3.1 folds to 0 and
  uplm80 divides (seed 20275; 4.2.3: undefined).
- Intel's build is stopped by a HLT patched in where the LINES record
  puts the module's END. For `fin: end t;` V3.1 puts an `LXI SP` there,
  before its `EI; HLT`, and the build runs past the HLT: the verdict is
  `timeout`, where uplm80's build prints what it should.

### Verified

On 830b234, with upeepz80 0.2.7 and um80 0.3.52.

- The suite: 1162 tests pass. Without Intel's binaries the oracle's 32
  tests that need them skip, and its 6 others pass. pylint rates the
  package 9.73, with no message 0.4.1 did not have.
- `tests/run_tests.sh`: all 22 programs pass.
- `scripts/difftest.py`, `scripts/abifuzz.py` and `scripts/namestest.py`,
  200 seeds each (7000-7199): every program prints what the model, its
  `-O0` build or its scopes say.
- `scripts/intel_oracle.py --random 300` (seeds 1-300), leaving out
  `shl-byte`, `shift9`, `wide-limit`, `sub-zero`, `zero-dividend` and
  `neg-widened`: 299 programs print at `-O0` to `-O3` what Intel's
  PL/M-80 V3.1 build prints. Seed 233 does not, at any level: V3.1 codes
  its `w1, b2 = b2;` as `INX H; INX SP; MOV M,A` and a later CALL
  overwrites `b1` (README, Known differences); with `w1 = b2;` in its
  place the program prints what V3.1's build prints. 0.4.1 gives the same
  on those seeds with `qualsize` left out as well.
- `scripts/intel_oracle.py --corpus --normalize`: of the 67 programs of
  `tests/` and `sample_code/`, 38 print what V3.1's build prints, V3.1
  rejects 28, and `tests/test_move_builtin.plm` differs where V3.1's MOVE
  of 0 bytes moves 65536, as with 0.4.1.
- The 87 compiles of MP/M II's and 80un's PL/M (DRI's tree and mpm2's
  overrides, each in the mode `tools/build.py` uses; 80un's files one at a
  time and its two programs) at `-O0`, `-O2` and `-O3` compile to 0.4.1's
  assembly, and the ten that stop, stop with 0.4.1's errors. The code at
  `-O2` is 165,182 bytes, the data 45,088 (with upeepz80 0.2.7). The
  `sample_code` programs and the 47 test programs compile to 0.4.1's
  assembly at `-O0`, `-O2` and `-O3`; `tests/test_implicit_calls.plm`
  now with the warning for its `callee$func()`.
- SHL and SHR of a BYTE made a BYTE, as the manual and V3.1 make them
  (Known issues), was built and checked, and not kept: 80un's `test.arc`
  extracts one member of thirteen. MP/M II's SHL sites store the result
  in a BYTE, or shift a value too small to lose a bit - `shl(dcnt and
  11b, 5)`, `shl(a, 4) or b` of a BCD digit - and its SHR sites give the
  same value in eight bits; 24 of the MP/M II compiles are smaller, 619
  bytes in all at `-O2`, SHOW.PLM the most, by 79.
- Checked again, on 75131f9, by a separate release check: the suite,
  with Intel's binaries and without; `run_tests.sh`; difftest,
  abifuzz, namestest and the storage fuzzer, 200 to 300 new seeds each,
  with no failure; the 87 MP/M II and 80un compiles and the test
  programs, 0.4.1's assembly; 80un's two programs extracting all 28 test
  inputs byte for byte as 0.4.1's build does; MP/M II V2.0 and V2.1 built
  from source, `run_tests.sh all` and `src` passing, and `verify_dri.py`
  finding XDOS, BNKXDOS, RESBDOS, TMP, BNKBDOS, RDT, DDT and GENMOD
  identical to DRI's. `intel_oracle.py --random 500`, seeds 20000-20499:
  496 print what V3.1's build prints, and the 4 others are V3.1's bugs
  (README, Known differences) or a division by 0.

## 0.4.1 — 2026-09-26

The Known issues 0.3.7 listed, and 0.4.0 carried, each settled by the
Programming Manual (9800268B) and, where it leaves room, by what Intel's
PL/M-80 V3.1 does with the same source: its listing, its diagnostics,
and the code it generates, linked with DRI's `X0100` and run. Four of
them were code the compiler got wrong, four were programs it should
have refused; one is PL/M-80's own rule, one a test's wrong
expectation, and the others were fixed by 0.4.0, upeepz80 0.2.6 and
um80 0.3.51. Checking them against V3.1 found three more kinds of
program to refuse - a parameter no DECLARE declares, a LITERALLY used
before its declaration, a dimension that is not a number - and two more
cases of two of the four: a store that runs back to a counted loop's
index, and a built-in's name another module of a multi-file compile
makes PUBLIC. What V3.1 rejects and uplm80 still compiles is listed
under Known issues.

### Incompatible: errors Intel's PL/M-80 gives

A program PL/M-80 does not allow, which uplm80 compiled, is now refused
as Intel's compiler refuses it. These rules of a program's names are
checked as the parser gives the program, before the optimizer rewrites
or drops anything, so every `-O` level finds the same errors; a
multi-file compile parses and checks every module before it optimizes
any.

- **A name declared nowhere is an error**, as it is to Intel's PL/M-80
  V3.1 (ERROR #105, UNDECLARED IDENTIFIER). `y = nosuch + 1` compiled to
  `ld hl,(NOSUCH)`, and only um80 reported it, as an undefined symbol, or
  nothing did where the optimizer dropped the use (Known issues, 0.3.7;
  0.3.6 the same). A built-in needs no declaration, and in a multi-file
  compile a module still names another's PUBLIC name without declaring it
  EXTERNAL, unless it is a built-in's name, which is the built-in (Fixed).

      NOSUCH is not declared (Programming Manual 9800268B, 6.1)

  Of the 87 compiles of MP/M II and 80un, the ten that did not assemble,
  for this reason, now stop here: MP/M II's `MSCMN.PLM`, which
  `MSBRS.PLM` and `MSRSP.PLM` include after declaring what it uses, and
  eight of 80un's modules compiled alone, which name each other's
  procedures (80un compiles them together).
- **A parameter that no DECLARE of its procedure declares, or that one
  declares as anything but a BYTE or an ADDRESS scalar, is an error**:
  "Each formal parameter must be declared as a non-based scalar variable
  in a DECLARE statement preceding the first executable statement in the
  procedure body" (8.1.1). A parameter that only the PROCEDURE statement
  names was an ADDRESS: `p: procedure (a, b); declare a byte; y = a + b;
  end p;` stored b with `ld (??AUTO+1),de` and added it as one, and with
  `a` declared only in a DO block of the procedure, that block's `y = a`
  read the block's own variable, which nothing set. A parameter declared
  an array, BASED, a LABEL or a structure, or with PUBLIC, EXTERNAL,
  INITIAL, DATA or AT, was compiled as declared, and a second declaration
  of one was taken without a word (0.4.0 the same). Intel's PL/M-80 V3.1
  rejects each: ERROR #25, UNDECLARED PARAMETER (and #105, UNDECLARED
  IDENTIFIER, at a use of it); #76, CONFLICTING ATTRIBUTE WITH
  PARAMETER; #77, INVALID PARAMETER DECLARATION, BASE ILLEGAL; #79,
  ILLEGAL PARAMETER TYPE, NOT BYTE OR ADDRESS; and #78, DUPLICATE
  DECLARATION. An EXTERNAL or a REENTRANT procedure's parameters are
  declared too, as V3.1 asks.

      B is a parameter of P, and no DECLARE of the procedure declares it; a
      parameter is declared a BYTE or an ADDRESS scalar, not BASED, by a
      DECLARE of its procedure (Programming Manual 9800268B, 8.1.1)
      A is a parameter of P, and a parameter is declared a BYTE or an
      ADDRESS scalar, not BASED and with no other attribute (Programming
      Manual 9800268B, 8.1.1)

  None of the 87 MP/M II and 80un compiles, `sample_code`, the test
  programs and intel80tools' 469 PL/M files has one.
- **A LITERALLY's name used before its declaration, and a dimension that
  is not a number, are errors.** A LITERALLY's text is "substituted for
  each occurrence of the identifier in subsequent text" (6.4), and uplm80
  put it in place of a use that comes before the declaration too: `p:
  procedure; y = lit; end p; declare lit literally '5';` compiled to `y =
  5`, and so did such a use in a DO CASE or in a procedure nested in the
  one declaring `lit`. A dimension "is a numeric constant in parentheses"
  (6.2.5), and `declare a (lit) byte` before `lit`'s declaration made `a`
  a scalar, as did a dimension that names a variable, a LITERALLY out of
  its scope or a name declared nowhere, of an array or of a structure's
  member (0.4.0 the same). Intel's PL/M-80 V3.1 rejects each: ERROR #105,
  UNDECLARED IDENTIFIER, and #59, ILLEGAL DIMENSION ATTRIBUTE. A variable
  that a procedure names and an enclosing block declares after it is
  still that variable, in both compilers.

      LIT is not declared here: a LITERALLY declared after it puts its text
      in place of LIT only in the text that follows the declaration
      (Programming Manual 9800268B, 6.4)
      (LIT): the dimension of an array is a number, and LIT is not a
      LITERALLY declared before it whose text is one (Programming Manual
      9800268B, 6.2.5)

  None of the 87 MP/M II and 80un compiles, `sample_code`, the test
  programs and intel80tools' 469 PL/M files has one.
- **Empty parentheses after a variable, `x()`, after a structure member,
  `s.m()`, and after a subscript, `a(1)()`, are an error.** PL/M-80 has
  no empty subscript or argument list. `y = x() + 1` with x a BYTE compiled
  to a CALL through x's value (0.3.6: `call X`), and so did an array, a
  BASED variable, a structure, a parameter and `CALL w()` of an ADDRESS
  (Known issues, 0.3.7); `y = s.m()` compiled to `ld a,(S) / ld l,a / ld
  h,0 / call ??jphl`, a call through the member's value, and `w =
  s.a()`, `y = sa(1).m()` and `CALL s.a()` the same, and `y = a(1)()` and
  `y = q(1)()` called through the element's value and through what `q(1)`
  returned (0.4.0 the same). Intel's PL/M-80 V3.1 rejects each: ERROR #127,
  INVALID SUBSCRIPT ON NON-ARRAY, and #102, MISSING PRIMARY OPERAND, in
  an expression after a scalar, a parameter or a BASED variable, and after
  a structure with #135, ILLEGAL REFERENCE TO AN UNQUALIFIED STRUCTURE;
  #102 alone after an array or an array member, and in `CALL w()` and
  `CALL s.a()`; #127 and #32, INVALID SYNTAX, after a scalar member in an
  expression; and #32 after a subscript (with #118 in `CALL a(1)()`, and
  #135 in `sa(1)()`).

      X(): X is a variable, and PL/M-80 has neither an empty subscript nor
      an empty argument list
      SA(1).M(): SA(1).M is a structure member, and PL/M-80 has neither an
      empty subscript nor an empty argument list

  A procedure's `f()` is still taken for `f`, as uplm80 always has; V3.1
  rejects that too (#102).
- **`.label`, the address of a label, in an expression is an error.** The
  dot operator takes a variable or a procedure (4.1.3), and Intel's
  PL/M-80 V3.1 rejects `.label` in an expression, ERROR #158, INVALID DOT
  OPERAND, LABEL ILLEGAL, whether or not the label is declared LABEL and
  whether it is defined before the expression or after; it accepts one in
  a DATA or an INITIAL list, and so does uplm80, as before (MP/M II's
  MPMLDR begins `DATA (0C3H, .start-3)`). uplm80 compiled the expression to
  the label's address (Known issues, 0.3.7; 0.3.6 the same).

      .HERE: HERE is a label, and the dot operator takes a variable or a
      procedure (Programming Manual 9800268B, 4.1.3); the address of a
      label may be given only in a DATA or an INITIAL list

- **An INTERRUPT procedure nested in a procedure, or declared in a DO
  block, is an error**, as it is to Intel's PL/M-80 V3.1 (ERROR #39,
  INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL): "it may only
  be used in a PROCEDURE statement at the outer level of a program
  module" (8.1.6). uplm80 compiled a nested one, and a local of the
  procedure around it that the interrupt read could be in `??AUTO`, in
  another procedure's frame (Known issues, 0.3.7; 0.3.6 the same).

      IH: an INTERRUPT procedure must be declared at the outer level of the
      module, not in procedure OUTER (Programming Manual 9800268B, 8.1.6)

### Fixed

- **A procedure or a variable named like a built-in is the program's,** as
  a declaration hides the built-in of its name (9.2), and as Intel's
  PL/M-80 V3.1 compiles it. The optimizer wrote an ADDRESS constant below
  256, and a value it widened to ADDRESS, as a call of DOUBLE, and every
  call of DOUBLE of a constant was taken for one: with a procedure DOUBLE
  of the program's, `double(30h)` was 30H at `-O1` and up however the
  procedure was written (Known issues, 0.3.7). The optimizer's DOUBLE is
  now called by a name no program can declare, `??DOUBLE`. And wherever
  the optimizer or code generation knew a built-in by its name alone, it
  now also checks that the program does not declare the name there. These
  were wrong too (0.3.6 the same): a DOUBLE or LOW of the program's in a
  condition or in a DO's bound was folded as the built-in, at every level;
  `w * 8` and `w / 2` became calls of the program's SHL and SHR at `-O2`
  and `-O3`; an array OUTPUT or MEMORY was stored to as the port, or as
  memory past the end of the program, and `.memory` of one was the end of
  the program, and a variable STACKPTR was SP, at every level. A program
  that declares no built-in's name compiles to the code 0.4.0 compiles it
  to: the 87 MP/M II and 80un compiles, and 300 random programs and the
  test programs, at `-O0` to `-O3`.

  In a multi-file compile, the other way round: a module that does not
  declare a built-in's name means the built-in, as it does compiled alone
  and linked, though another module makes a procedure or a variable of
  that name PUBLIC (10.4). With module MA's `shl: procedure (a, b) address
  public` and `double: procedure (a) address public`, module MB's `shl(w,
  4)` called MA's SHL at `-O0` to `-O2`, and `w * 8`, which the optimizer
  makes SHL(w, 3), at `-O2` and `-O3`; `double(30h)` was MA's DOUBLE at
  `-O0` and 30H from `-O1` up; and a MEMORY or a STACKPTR another module
  made PUBLIC was MB's `memory(0)`, `.memory` and `stackptr` (0.4.0 the
  same). The optimizer took each for the built-in, and code generation,
  which holds one PUBLIC name for every module, for MA's. Now MB's are the
  built-ins at every level, and a module that declares the name EXTERNAL
  gets MA's, as it did. The 80un programs compile as before.
- **PLUS, MINUS, SCL and SCR after `+ 4` or `- 4` of an ADDRESS read the
  carry that the addition or subtraction sets**, as in Intel's PL/M-80
  V3.1. uplm80 stepped an ADDRESS by 1 to 4 with `inc hl` and `dec hl`,
  which set no carry, so `(w + 4) PLUS z` with w = 0FFFEH gave 2, not 3
  (Known issues, 0.3.7). V3.1 steps it by 1 to 3 with INX and DCX, which
  leave the carry as it was, and from 4 on adds with DAD or subtracts
  with SUB and SBB:

      R = (W - 1) MINUS Z;   LHLD W / DCX H / XCHG / LHLD Z / CALL @P0074
      R = (W + 3) MINUS Z;   LHLD W / INX H / INX H / INX H / XCHG / ...
      R = (W + 4) PLUS Z;    LXI D,4H / LHLD W / DAD D / LXI D,Z / CALL @P0010
      R = (W - 4) PLUS Z;    MVI A,4H / LXI D,W / CALL @P0101
      D = (B + 1) PLUS C;    LDA B / INR A / ... / ADC M
      D = (B + 3) PLUS C;    LDA B / ADI 3H / ... / ADC M

  (@P0074 is `MOV A,E / SBB L / MOV L,A / MOV A,D / SBB H`, @P0010 `LDAX D
  / ADC L ...`, @P0101 `... SUB L ... SBB H`.) So after `+ 1` to `+ 3` or
  `- 1` to `- 3` of an ADDRESS, and `+ 1`, `+ 2`, `- 1` or `- 2` of a
  BYTE, V3.1's PLUS, MINUS, SCL and SCR read a stale carry too; `(w - 1)
  MINUS z` with w = z = 0 is 0FFFFH or 0FFFEH as the carry happens to
  be, in either compiler, and the manual warns that the flags are not to
  be relied on (12.1). uplm80 now does what V3.1 does for an ADDRESS:
  where a PLUS, MINUS, SCL or SCR reads the carry, `+ 4` and `- 4` are an
  addition or a subtraction that sets it, and 1 to 3 are still `inc hl`
  and `dec hl`; elsewhere 4 is still four `inc hl`. A BYTE plus or minus
  a constant was `add a,n` or `sub n` already, which set it. The program
  in `tests/test_expression_types.py` prints what V3.1's code prints at
  `-O0` to `-O3`. None of the 87 MP/M II and 80un compiles has such an
  expression.
- **A counted loop over a module-level index ends where a pointer or an
  overrun sets the index.** A BYTE `DO i = 0 TO n` whose body does not
  name `i` counts its passes in B, which is right only if nothing else can
  reach `i` while it runs. For a procedure's local that took in a pointer
  run on from a local declared before it and an overrun of an array
  declared before it; for a module-level `i` it did not, so with `p = .a +
  1` (`a` declared just before `i`) a store through `p` in the body did
  not end the loop (Known issues, 0.3.7), and neither did `buf(k) = 30`
  with `k` past the end of a `buf` declared before `i`, or with `p = .a2 +
  1` after `declare (a2, i2) byte`, or `s.m(2) = 20` of an `s structure
  (m(2) byte)` declared before `i`, or `s2(1).m(2)` or `s2(1).m(k)`, k =
  2, of an `s2 (2) structure (m(2) byte)`. Nor did a store that runs
  back to `i` from a variable laid out after it: through `p = .x - 1`, to
  a `z byte at (.x - 1)`, and `a(0ffffh) = 20` of an `a (2) byte`
  declared after `i`, the subscript wrapping, or `a(k) = 20` with `k` an
  ADDRESS set to 0FFFFH; nor, to a procedure's static local `i`, `.x - 1`
  of the local `x` declared after it. Now a module-level index, and a
  procedure's static local, which is laid out among the module's
  variables, is not counted when any variable - the module's, or a
  procedure's - has its address taken, or is subscripted, itself or a
  member of it, past the end of what is subscripted or by what is not a
  constant: a pointer or a subscript runs backwards through the variables
  as well as on. So is a procedure's local after `s2.m(4) = 20` of a
  local `s2 (2) structure (m(2) byte)`, which PL/M-80 does not allow and
  uplm80 compiles as `s2(0).m(4)` (Known issues): the overrun was taken
  to reach no further than `s2`. DRI's compiler never counts a loop, and
  the programs in `tests/test_calls_and_loops.py` that V3.1 accepts
  print, at `-O0` to `-O3`, what they print compiled by Intel's PL/M-80
  V3.1. Of the 87 MP/M II and 80un compiles, one loop changes, `DO jtab
  = 0 TO itab` in MSPL.PLM's LIST$BUF (DRI's and mpm2's), since `.pcb` is
  taken and PCB is declared before JTAB: 7 bytes more at `-O1` to `-O3`
  and 8 at `-O0`, in each of the two, and nothing else; ED's, PIP's and
  80un's counted loops are over locals in `??AUTO` (Known issues).
- **A REENTRANT procedure's parameter declared with its locals,**
  `DECLARE (top, c) BYTE`, was declared a second time, as a local in the
  frame, which nothing set, and every use of it read that: `rp(3)` of a
  recursive `rp` returned 1 where it returns 0AH (Known issues, 0.3.7;
  0.3.6 the same). It is the parameter, on the stack, as it was when
  declared on its own; the local beside it is the frame's first byte.
  Intel's PL/M-80 V3.1 compiles `tests/test_calls_and_loops.py`'s
  program to print what uplm80's prints at `-O0` to `-O3`.
- `tests/run_tests.sh`'s `test_byte_conditions` expected an IF and a DO
  WHILE to test for non-zero, as uplm80 did before 0.3.5, and failed.
  They test the least significant bit (5.1.2): 128, 10, 2 and 256 are
  false. The expected output is now what the program prints compiled by
  Intel's PL/M-80 V3.1, and by uplm80; all 22 programs pass.
- `scripts/genipx.py`, which makes the `.ipx` includes of an intel80tools
  pack from its `.pex`, put in each module's include another module's
  BASED variable whose base no module makes PUBLIC - lib_2.1's `declare
  arg based argChain ARG$T;` and `declare module based module$p
  MODULE$T;` - and nothing declared the base. With a name declared
  nowhere now an error, lib_2.1's ISIS1.PLM and ISIS2.PLM stopped at
  ARGCHAIN, as Intel's PL/M-80 V3.1 would (ERROR #54, UNDECLARED BASE).
  Such a variable is left out, as only its own module can use it; both
  compile to 0.4.0's assembly less its EQUs, and `scripts/
  test_intel80tools.sh` still compiles 12 of link_3.0's 15 modules.

### Changed

- **A LITERALLY's name declared again in an inner block** is PL/M-80's
  rule, not a defect (Known issues, 0.3.7): a LITERALLY's text is
  "substituted for each occurrence of the identifier in subsequent text"
  (6.4), throughout its scope, so after `declare n literally '5'` a
  procedure's `declare n byte` is `declare 5 byte`. Intel's PL/M-80 V3.1
  does the same, ERROR #48, ILLEGAL DECLARATION STATEMENT SYNTAX; with
  `m literally 'w'` an inner `declare m byte` declares a W of that block's
  own in both compilers, and MP/M II's MPMLDR needs `mon1 literally
  'ldmon1'` to make its `mon1: procedure external` LDMON1. The syntax
  error now says where the text came from:

      unexpected token 'NUMBER' '5'; expected one of: IDENT, LPAREN; that
      is the text of N, declared LITERALLY '5', which PL/M-80 puts in place
      of N wherever it occurs in the LITERALLY's scope (Programming Manual
      9800268B, 6.4)

  Where the text is a nested LITERALLY's, the note names that one, and
  the LITERALLY whose text names it: with `nn literally '5', n literally
  'nn'`, "that is the text of NN, declared LITERALLY '5', in the text of
  N, declared LITERALLY 'nn', ...". The syntax error is where it was.

- A static parameter is no longer named a second time by an EQU
  (`?@proc$name equ @proc$name`). upeepz80 before 0.2.6 dropped the store
  of an argument at a procedure's entry when nothing else named the
  parameter's storage, though a pointer from the parameter before it can
  reach it (Known issues, 0.3.7); 0.2.6, which 0.4.0 requires, keeps it.
  `pq: procedure (a, b) byte; declare (a, b) byte; ... pp = .a + 1; return
  c;`, `c` BASED on `pp`, still keeps `ld (@PQ$@B),a` and returns `B` at
  `-O1` to `-O3`. Over the 87 compiles of MP/M II and 80un, the assembly at
  `-O1`, `-O2` and `-O3` is 0.4.0's less its 25 such EQUs, line for line.
- The CHANGELOG's Known issues of 0.3.7 no longer list what 0.4.0 fixed:
  a CALL through an address passes any number of arguments to any
  procedure. um80 0.3.51 needs none of uplm80's renames and rewrites of
  names spelled like an operator (`@EQ`, `2+TYPE`, `jp 0+P`), which are
  kept, and cost nothing, for older um80 releases (README, Names in the
  Output).

### Added

- `tests/test_names.py`: each new error, at every level and where it is
  placed - a name declared nowhere, a parameter declared nowhere, twice or
  as anything but a scalar, a LITERALLY used before its declaration, a
  dimension that is not a number, empty parentheses after each kind of
  variable, after a member and after a subscript, the address of a label
  in an expression, an INTERRUPT procedure in a procedure and in a DO
  block, a LITERALLY's name declared again - and every built-in compiling
  undeclared.
- `tests/test_names.py`: a module compiled with one that makes SHL, SHR,
  DOUBLE, MEMORY and STACKPTR PUBLIC, declaring them EXTERNAL and not.
- `tests/test_expression_types.py`: a procedure named like each built-in
  that is one, and a variable named like each that can be one; PLUS,
  MINUS, SCL and SCR after `+ 4` and `- 4`, and 1 to 3 still `inc hl`.
- `tests/test_calls_and_loops.py`: REENTRANT procedures with their
  parameters factored with locals; counted loops whose index a pointer or
  an overrun sets, of an array and of a structure's member, run on or
  back to it.
- Each program in them that runs prints, at `-O0` to `-O3`, what it prints
  compiled by Intel's PL/M-80 V3.1 (DRI's `PLM_WORK` copy, under an ISIS
  emulator), linked with DRI's `X0100` and `PLM80.LIB` by Intel's LINK
  and LOCATE, and run under cpmemu. The expected output is transcribed:
  Intel's binaries are not in this repository.

### Known issues

uplm80 still compiles these, which Intel's PL/M-80 V3.1 rejects (0.4.0
the same):

- `f()` and `CALL g()` of a procedure, and `carry()` of a built-in,
  taken for `f`, `g` and `carry` (ERROR #102, MISSING PRIMARY OPERAND, and
  #153, INVALID NUMBER OF ARGUMENTS IN CALL).
- A subscript on a scalar, `x(0)` or `x(1)`, the byte at X's address
  plus the subscript (#127, INVALID SUBSCRIPT ON NON-ARRAY); and `shl(w,
  3)` where the program declares SHL an ADDRESS, a call through SHL's
  value (#127, and #114, MULTIPLE SUBSCRIPTS ILLEGAL).
- An array, or an array member, without a subscript anywhere but in a
  location reference or LENGTH, LAST and SIZE (3.6.2): `a = 3` and `x =
  a` are `a(0)`, `s.m = 4` is `s.m(0)`, `s2.m(1)` of an array of
  structures is `s2(0).m(1)`, and `size(a)`, SIZE an array of the
  program's, is `size(a(0))` (#133, ILLEGAL REFERENCE TO UNSUBSCRIPTED
  ARRAY, and #134, ILLEGAL REFERENCE TO UNSUBSCRIPTED MEMBER ARRAY).
- INITIAL in a procedure's declaration, which initializes the variable
  once, when the program is loaded (#73, INVALID ATTRIBUTE OR
  INITIALIZATION, NOT AT MODULE LEVEL).
- A procedure with no statements, `g: procedure; end g;`, which returns
  (#174, INVALID NULL PROCEDURE).
- A call of a procedure that its block declares after the call, `p:
  procedure; call q; end p; q: procedure; ... end q;`, and `y = f + 1`
  of a typed procedure `f` declared after it (#169, ILLEGAL FORWARD
  CALL).

And one that V3.1 compiles to other code (0.4.0 the same):

- **What a store through a pointer or an overrun reaches in or from
  `??AUTO`** is uplm80's layout, not DRI's (Known issues, 0.3.7), and a
  counted loop over a local in `??AUTO` does not see all of it. `??AUTO`
  comes first in the data segment, before the module's variables, and
  holds the frames of procedures active together one after another: an
  overrun of a local in it can reach another frame or the module's first
  variables, and `.x - 1` of the first variable after it its last byte.
  A loop over a local in `??AUTO` ends where a pointer or an overrun from
  a local of its own procedure declared before the index sets it, and
  nowhere else. With q's `ql(2) byte, k byte` just before run's index `i`
  in `??AUTO`, `ql(3) = 20` from inside run's loop sets `i` and the loop
  still runs its count, 000B 0014 at `-O0` to `-O2`, where V3.1's build,
  whose layout has `i` there too, prints 0003 0015. Counting no such loop
  would cost ED and PIP 18 and 17 bytes at `-O2`, and 80un 20.
- A loop over the last module-level variable is still counted where a
  store through MEMORY reaches it: MEMORY follows the last variable, in
  V3.1's layout as in uplm80's, so `p = .memory - 1` with a BASED `b`, or
  `memory(0ffffh) = 20`, sets that variable, and the loop runs its count
  (000B 0014 at `-O0` to `-O3`, where V3.1's build prints 0003 0015; 0.4.0
  the same).
- A label on an END statement, `out: end p;` (Programming Manual A.4.4.1),
  is a syntax error; V3.1 compiles it (0.4.0 the same).
- V3.1 rejects a zero dimension, `declare b (0) byte` (ERROR #57), and the
  address of a built-in, `.double`; uplm80 accepts both (0.4.0 the same).

### Verified

On 31444b1, with upeepz80 0.2.6 and um80 0.3.52 (84bea83 was checked
with um80 0.3.51).

- The suite: 1038 tests pass, and with upeepz80 0.2.7 too. pylint rates
  the package 9.73.
- `tests/run_tests.sh`: all 22 programs pass.
- `scripts/difftest.py --seeds 300`: every program prints what the model
  says at `-O0` to `-O3`. `scripts/abifuzz.py --seeds 300`: every program
  prints, in each of its builds, what its `-O0` build prints.
  `scripts/namestest.py --seeds 200`, and `--modules --seeds 40`: every
  program prints what its scopes say.
- The 87 compiles of MP/M II's and 80un's PL/M (DRI's tree and mpm2's
  overrides, each in the mode `tools/build.py` uses; 80un's files one at a
  time and its two programs), at each of `-O0` to `-O3`: 75 of the 77 that
  assemble compile to 0.4.0's assembly less its 25 EQUs, line for line -
  73 of them to 0.4.0's exactly, 80un's two programs among them, and
  LOAD.PLM and STAT.PLM less 21 and 4; MSPL.PLM, DRI's and mpm2's, is 7
  bytes larger at `-O1` to `-O3` and 8 at `-O0` (a loop no longer
  counted, Fixed); the other ten, which never assembled, stop at compile
  time with "is not declared". The code at `-O2` is 165,326 bytes against
  165,312, the data 45,088 as before.
- `sample_code`, at `-O0` to `-O3`: of the five programs that compile,
  CP/M 2.0's ED.PLM and PIP.PLM compile to 0.4.0's assembly, and its
  LOAD.PLM, STAT.PLM and SUBMIT.PLM to 0.4.0's less its 21, 14 and 1
  EQUs; the fourteen others stop where they stopped, five of them - CP/M
  1.1's CCP-ORIGINAL.PLM, CCP.PLM, HELLO.PLM and LOAD.PLM, and 1.3's
  BDOS.PLM (Zeidman) - now saying that the number they stopped at is a
  LITERALLY's text.
- 300 random programs, 150 each of `tests/plm_difftest.py` and
  `tests/names_difftest.py`, and the 47 test programs compile, at `-O0` to
  `-O3`, to 0.4.0's assembly, line for line.
- 600 random programs with module-level BYTE and ADDRESS scalars and
  arrays, structures and arrays of structures, and a counted loop over a
  module-level index whose body stores once through a subscript or a
  member - constant or not, inside its variable or past its end - and 300
  with those variables local to the procedure, print at `-O0` to `-O3`
  what each prints compiled by Intel's PL/M-80 V3.1; c71574d's `-O2`
  build of 11 of the 600 printed something else. 200 more such programs
  print what the generator's model of the layout says.
- Each program of the new tests that V3.1 accepts prints, compiled by it,
  what uplm80's build prints at `-O0` to `-O3`: the counted loops whose
  store runs back to the index, at module level and to a static local,
  and a LITERALLY used after its declaration beside a variable declared
  after the procedure that names it. V3.1 refuses each program the new
  errors refuse, with the errors named under Incompatible. The modules
  of the multi-file test print, compiled together, what they print
  compiled one at a time and linked.
- ogdenpm/intel80tools' 469 PL/M files, with the `.ipx` files
  `scripts/genipx.py` makes from the packs' `.pex`, at `-O2`: 176
  compile, 144 to 0.4.0's assembly and 32 to 0.4.0's less its EQUs;
  IXREF.PLM of ixref 1.2 and 1.3, which 0.4.0 compiled and um80 did not
  assemble, stop at WRITE, which nothing they include declares; the
  other 291 compile with neither.

## 0.4.0 — 2026-09-25

Procedures are now called the way Intel's PL/M-80 calls them, so code
uplm80 compiles links, unmodified, with assembly written for PL/M-80 -
Digital Research's `X0100.ASM` (`mon1 equ 0005h`), MP/M II's `LDMONX.ASM`
(`ldmon1 equ 0d06h`) and `BRSPBI.ASM` - and with objects PL/M-80
compiled. Up to 0.3.x no uplm80 module did: each of the three conventions
it had differed from Intel's, and every program linked with DRI's
interface modules needed a shim that turned one into the other.

The convention is PL/M-80 V3.1's as its own output shows it: DRI's
`PIP.PRL`, which PL/M-80 compiled, has `MOVE: PROCEDURE (S, D, N)` at
0AACH,

    LXI H,244EH / MOV M,E        ; N, the last argument, from E
    DCX H / MOV M,B / DCX H / MOV M,C    ; D, the one before, from BC
    DCX H / POP D                ; the return address
    POP B / MOV M,B / DCX H / MOV M,C    ; S, pushed by the caller
    PUSH D

and V3.1's listings of test modules, and byte-for-byte rebuilds of DRI's
utilities with Intel's compiler, agree.

### Incompatible: calling convention

**Rebuild every module.** A module built by 0.3.x does not work with one
built by 0.4.0, and nothing at link time says so: they export and import
the same names.

| Arguments | Where they are at the `call` |
|---|---|
| 0 | nothing |
| 1 | a1 in BC (C for a BYTE parameter) |
| 2 | a1 in BC (C), a2 in DE (E) |
| n ≥ 3 | a1 … a(n−2) pushed left to right, one word each; a(n−1) in BC (C); an in DE (E) |

- **The callee takes the pushed words off the stack.** The caller no
  longer pops anything after a call. At entry `[SP]` is the return
  address, `[SP+2]` a(n−2), and so on to `[SP+2(n−2)]`, a1.
- A BYTE argument in a register is in C or E, and B or D is undefined; a
  pushed BYTE is the low byte of its word.
- A BYTE result is in A and an ADDRESS one in HL, as before.
- A call keeps SP, IX and IY, and nothing else: A, the flags, BC, DE and
  HL are destroyed.
- This is so for every procedure - PUBLIC, EXTERNAL, nested, REENTRANT -
  but one exception. A procedure with one parameter that nothing outside
  the compile can reach - not PUBLIC, EXTERNAL or REENTRANT, and its
  address never taken - still takes it in A (a BYTE) or HL (an ADDRESS),
  where its body wants it. No other module, no assembly and no `CALL`
  through an address can tell.
- A `CALL` through an address places the arguments the same way, each
  widened to ADDRESS, with the address in HL, and calls `??jphl` (`jp
  (hl)`), which replaces `??jpde`.
- `MON1(f, a)` and `MON2(f, a)` with a constant `f` are still compiled as
  the BDOS call itself, `ld de,a / ld c,f / call 5`, and the argument is
  converted to the parameter's type like any other: a BYTE passed to
  MON1's ADDRESS parameter is now `ld e,a / ld d,0`, where D was left as it
  happened to be. PL/M-80 does the same (MP/M II's MPMLDR at 03D1H: `LHLD
  char / MVI H,0 / XCHG / MVI C,2`), and a function that reads DE whole,
  such as MP/M II's 141 (delay), needs it.
- Two new errors. A direct call must pass as many arguments as the
  procedure has parameters, as PL/M-80 V3.1 requires (153 and 154): with
  the callee removing the pushed words, a call with too many or too few
  would return with the stack moved. 0.3.x dropped extra arguments to a
  procedure private to its module without a word. A `CALL` through an
  address is not checked (8.2.1). And an INTERRUPT procedure may not have
  parameters (8.1.6).

      invalid number of arguments in call of P2, too few: 1 for 2 parameters
      IH: an INTERRUPT procedure may not have parameters (8.1.6)

- **upeepz80 0.2.6 or later is required.** 0.2.5 turned `push … / call p /
  ret` into `push … / jp p`, and p then took its return address for its
  first argument. At `-O1` and up the compiler now stops with an error
  naming both versions if upeepz80's is below 0.2.6, unless that
  upeepz80 keeps the `call` of such a routine: a development tree with the
  fix, still numbered 0.2.5, does, and the release 0.2.5 does not.

      upeepz80 0.2.5 is too old: uplm80 needs upeepz80 0.2.6 or later (…)

What 0.3.x did: a procedure private to its module had all but its last
argument written straight into its own storage by the caller and the last
in A or HL; a PUBLIC, EXTERNAL or REENTRANT one had every argument pushed,
left to right and each widened to 16 bits, and popped by the caller after
the call; a `CALL` through an address passed only one, except to a PUBLIC
or REENTRANT procedure.

An assembly routine written for 0.3.x changes like this:

| A 0.3.x assembly routine… | …becomes in 0.4.0 |
|---|---|
| read its one argument at SP+2; the caller popped it | reads BC (C for a BYTE); pops nothing |
| read two at SP+4 and SP+2 | reads BC and DE |
| read n ≥ 3 at SP+2n … SP+2 | pops the return address, pops the n−2 stacked words (last pushed first), takes BC = a(n−1) and DE = an, and puts the return address back |
| returned with its arguments still pushed | returns with the stacked words removed |

For example, the BDOS interface a CP/M program links with. MON1, MON2,
MON2A and MON3 are equates now, as in DRI's `X0100.ASM`, since a call
already has the function in C and the argument in DE:

```asm
; 0.3.x                                 ; 0.4.0
MON1:   ld      hl,2                    MON1    equ     5
        add     hl,sp                   MON2    equ     5
        ld      e,(hl)                  MON2A   equ     5
        inc     hl                      MON3    equ     5
        ld      d,(hl)                          public  MON1,MON2,MON2A,MON3
        inc     hl
        ld      c,(hl)
        jp      5
```

and a routine of three arguments, `cap3(a address, b byte, c address)`:

```asm
; 0.3.x: a, b, c pushed; the caller pops them
CAP3:   ld      hl,2
        add     hl,sp
        ld      e,(hl)          ; c
        inc     hl
        ld      d,(hl)
        ld      (VC),de
        inc     hl
        ld      a,(hl)          ; b
        ld      (VB),a
        inc     hl
        inc     hl
        ld      e,(hl)          ; a
        inc     hl
        ld      d,(hl)
        ld      (VA),de
        ret

; 0.4.0: a pushed, b in C, c in DE; CAP3 takes a off the stack
CAP3:   ld      (VC),de
        ld      a,c
        ld      (VB),a
        pop     hl              ; the return address
        ex      (sp),hl         ; a, and the return address back on top
        ld      (VA),hl
        ret
```

Assembly that calls a PL/M procedure does the same from the other side:
it pushes the first arguments, loads the last two into BC and DE, and
leaves the stack alone after the call.

### Changed

- **Smaller code.** Over MP/M II's PL/M (UTIL2 to UTIL7 and MPMLDR, 39
  modules, SDIR's eight included) and 80un's two programs, at `-O2` with
  upeepz80 0.2.6, the code is 111,219 bytes against 0.3.7's 113,528
  (−2,309), and no module is larger. A pushed argument costs one `push`
  where 0.3.x stored it into the callee's slot, the entry stores it once,
  and the A/HL exception saves the `ld c,a` or `ld b,h / ld c,l` every
  call of a one-parameter procedure would otherwise need (about 1,300
  bytes of it). The data is 36,526 bytes against 36,531: nothing is stored
  before a call, so the call graph no longer keeps a callee's frame apart
  from the procedures its arguments call.
- A procedure's entry stores its arguments (the last from DE, the one
  before from BC, the pushed ones popped). With one or two parameters it
  leaves BC and DE as they came, so DRI's CP/M 1.x `MON1: PROCEDURE (F,
  A); … GO TO BDOS; END MON1;` passes them on to the BDOS.
- A REENTRANT procedure pushes the arguments that came in BC and DE under
  its return address, which is the frame 0.3.x had, and every exit takes
  all of them off the stack, keeping A and HL.
- A GOTO out of a procedure can now abandon, besides the return addresses
  of the calls it leaves, the words pushed for a call whose arguments were
  being evaluated: the first arguments, and BC, kept round the last one.
  `call p3(1, 2, f)`, where `f` ends in `goto again`, leaves two. The
  label at the outer level of the main program that such a GOTO reaches
  sets SP again, as it does since 0.3.7, and that takes them off too: 1000
  such GOTOs run in `-m bare`'s 64-byte stack, out of an argument of a
  direct call, of a CALL through an address and of a REENTRANT
  procedure's callee (`tests/test_goto_stack.py`). A GOTO to a label in a
  DO block of the main program, which draws a warning, leaves them, as
  Intel's PL/M-80 does.
- `STACKPTR` in a call's arguments reads SP with the arguments pushed for
  the call so far, as in PL/M-80's code. After `sp0 = stackptr`, `call
  p5(1, 2, 3, 4, stackptr - sp0)` passes 0FFFAH, three words down, and
  `call p2(4, stackptr - sp0)` passes 0, as PL/M-80 V3.1's code does;
  0.3.7 passed 0 to both, p5 and p2 being private. BC, the next-to-last
  argument, is saved round the last one's code only where that may write
  B or C: round a call of a procedure, `??mul16`, `??div16`, `??mod16` or
  `??inp`, and not of `??subde`, which leaves BC as it is. The two
  compilers still differ in two cases. V3.1 loads a next-to-last argument
  that is a constant or a variable into BC after the last one's code, and
  so saves nothing round it: with `one` = 1, `call p2(4, (stackptr - sp0)
  * one)` passes 0 there and 0FFFEH here. And it pushes a next-to-last
  argument it has computed while it evaluates the last: `call p2(sp0 -
  stackptr, stackptr - sp0)` passes 0FFFEH there and 0 here.
- A call's arguments are evaluated from left to right. PL/M-80 V3.1 loads
  a next-to-last argument that is a variable into BC after it has
  evaluated the last one, so where the last argument's code changes that
  variable (`call p2(v, f)`, with `f` assigning `v`), its code passes the
  new value and uplm80's the old one. The language leaves this open: "PL/M
  does not guarantee the order of evaluation of operands", and where the
  order matters "the value of the expression is undefined" (9800268B,
  4.5.1).
- `??jphl` replaces `??jpde`.

### Fixed

- **A `CALL` through an address passes any number of arguments to any
  procedure.** 0.3.7 passed more than one only to a PUBLIC or REENTRANT
  procedure, and warned (Known issues, 0.3.7).
- A call with more arguments than a private procedure has parameters is
  an error; 0.3.x dropped the extra ones.

### Added

- `tests/test_calling_convention.py`: the sequences, at `-O0`, for 0 to 5
  arguments of either type in every order, conversions, BC kept while the
  last argument is evaluated where its code may write B or C (and what
  each runtime routine writes, read from its code), entries, the A/HL
  exception and what takes it away, REENTRANT entries and exits, calls
  through an address, MON1 and MON2, and the two errors.
- `tests/test_calling_convention_run.py`, at `-O0` to `-O3`: uplm80 code
  with assembly written to the convention, both ways; REENTRANT and
  recursive procedures across the boundary; calls through an address to
  every kind of procedure; PUBLIC procedures across separately compiled
  modules and in one multi-file compile; DRI's `MON1 … GO TO BDOS`;
  `STACKPTR` in a call's last argument, against what PL/M-80 V3.1's code
  passes; and Intel's own code, `tests/fixtures/plm80_v31`: a module
  compiled by PL/M-80 V3.1, its listing transcribed, with Intel's `MAIN`
  calling uplm80's procedures and uplm80's `MAIN` calling Intel's.
- `tests/test_upeepz80_version.py`: the upeepz80 floor, against
  pyproject.toml's, and what is refused and accepted below it.
- `tests/abi_fuzz.py`, a fuzz test of the convention across modules: random
  programs of two PL/M modules, each defining procedures of 0 to 6
  parameters that the other declares EXTERNAL, some private, some
  REENTRANT, some called through an address, and an assembly module that
  stands between some calls and the procedures they call, taking the
  arguments where the convention puts them and passing them on with
  garbage in the high byte of each BYTE. Bodies end in calls. Each program
  is built at `-O0`, which must leave SP where it found it, and at `-O1` to
  `-O3`, with its two modules at different levels and in one multi-file
  compile, and every build must print what `-O0`'s prints.
  `tests/test_abi_fuzz.py` runs three seeds, and checks that the test
  fails when `call x / ret` becomes `jp x`, as upeepz80 0.2.5 made it;
  `scripts/abifuzz.py --seeds N [--jobs J]` runs more.
- `tests/_toolchain.py`: `run_asm` links any number of assembly modules
  after the program.

### Verified

On 0.3.7 as released, with 0.3.7's last fixes underneath (the reload of SP
at a label a GOTO out of a procedure reaches, PUBLIC labels, the names of
an EXTERNAL procedure's parameters, -O3 and a subscripted scalar). The
release gate ran on it before BC stopped being saved round a call of
`??subde` (Changed, the `STACKPTR` entry); what was run again after that
says so.

- The test suite: 780 tests pass, after it. pylint rates the package
  9.72, as before.
- `scripts/abifuzz.py --seeds 500`, and after it `--seeds 200 --first
  9000`: every program prints, in each of its seven builds, what its
  `-O0` build prints, and leaves SP where it found it.
- `scripts/difftest.py --seeds 150 --first 14000`, and after it `--first
  30000`: all 150 programs as the model says at `-O0` to `-O3`.
  `scripts/namestest.py --seeds 100 --first 5000`, and with `--modules` 40
  seeds: all print what their scopes say.
- The 87 compiles of MP/M II's and 80un's PL/M (DRI's tree and mpm2's
  overrides, each in the mode `tools/build.py` uses, 80un's files one at
  a time and its two programs) at `-O2`: the same 77 assemble as with
  0.3.7, 165,312 bytes of code against 168,158, and each output sets SP
  again at the labels 0.3.7's does. The 41 programs of the size figures
  above compile and assemble at every level from `-O0` to `-O3`. Of the
  87, `??subde` changes MPMLDR, DRI's and mpm2's, at each level from
  `-O0` to `-O3` and nothing else: DISPLAYOS, whose two calls of
  PRINTITEMS end in a subtraction, loses a `push bc` and a `pop bc` round
  each, 4 bytes, and the same 77 assemble. The seven overrides that test
  MPM21 compile with `-D MPM21` at `-O2` as before it.
- 0.3.7's release-gate programs of GOTOs out of procedures - out of
  REENTRANT recursion, counted loops, calls through an address, nested
  procedures and another module's procedures, to PUBLIC labels and to a
  label in a DO block - print what 0.3.7 prints at `-O0` to `-O3`, in
  `-m bare` and CP/M mode, and the diagnostics among them say what 0.3.7
  says; but for a9 in `-m bare`, whose procedures read their locals before
  they set them, and which with 0.3.7 prints one thing at `-O0` and
  another at `-O1` to `-O3`. Run again after it: the same.
- 80un: `80un.com` and `80unbas.com` built at `-O0`, `-O2` and `-O3`
  write, on each of the 29 inputs in its tests, exactly the files and the
  output those built by 0.3.7 at the same level do. After `??subde` both
  compile to the same assembly as before at every level from `-O0` to
  `-O3`.
- MP/M II, V2.0 and V2.1, built from DRI's sources (`tools/build.py
  --tree=src`, 44 of 44 targets each): `scripts/run_tests.sh all` passes
  on both systems (DIR, STAT, STAT of a drive, the resident processes,
  HTTP, SFTP) and `run_tests.sh src` passes; the assembly-built files are
  DRI's byte for byte (`tools/verify_dri.py`; ASM.PRL but for 11 bytes no
  source sets); in a console session on each system the source-built DIR,
  SDIR, STAT, SHOW, PIP, ED, TYPE, ERA, REN, SET and USER, in 34 runs,
  print what DRI's own `.PRL`s print given the same commands; and
  GENSYS.COM built from source, given the same answers, makes the
  `MPM.SYS` and `SYSTEM.DAT` DRI's GENSYS.COM makes, but for the six
  bytes of the serial number at 0B5H. Of MP/M II, `??subde` changes only
  MPMLDR (above).
- `tests/run_tests.sh` prints what it prints with 0.3.7 for all 22
  programs (21 pass; `test_byte_conditions` fails with both).

Before those fixes were underneath:

- Every PL/M source of MP/M II (`mpm2src/*/*.PLM`), 80un and `sample_code`
  that 0.3.7 compiles compiles, without the argument-count or INTERRUPT
  error.

## 0.3.7 — 2026-09-25

Two sets of fixes, each checked against what Digital Research's own PL/M-80
does, and a program layout that is Intel's.

The first began with six defects found while rebuilding MP/M II's resident
system processes from DRI's sources, each worked around in those sources until
now; three more turned up in the same code, and an independent verification of
those fixes found the rest. DRI's binaries settle what is right: the
`SPOOL.RSP`, `SCHED.RSP` and `MPMSTAT.RSP` built from source now match DRI's in
layout and in every initialised byte, and each resident process's stack holds
what DRI's `SCHED.BRS` holds.

The second makes division, `MOD` and every other expression give what DRI's
PL/M-80 gives. Division and `MOD` agree with DRI's divide routine for every
operand pair, zero divisors included, on every path uplm80 computes them by:
the runtime routines, constant folding, strength reduction and DATA/INITIAL
values. `tests/test_divmod_dri.py` runs DRI's own routine on an 8080
interpreter as the reference and compares a compiled table of divisions
against it at `-O0` to `-O3`. Every expression now has the value and the type
Intel's PL/M-80 Programming Manual gives it, at every optimization level: a
constant up to 255 is a BYTE; `+ - AND OR XOR` of two BYTEs and `-` or `NOT`
of one wrap at eight bits; `* / MOD` are ADDRESS; a relation is the BYTE 0FFH
or 0. The optimizer folded constants as untyped 16-bit numbers that code
generation then typed by magnitude, so arithmetic next to a folded, propagated
or rewritten operand could change width. `uplm80/plm_types.py` states the
rules once, the optimizer and the code generator follow them, and
`tests/plm_difftest.py` checks random programs against a Python model of them.
DO loops now end the way DRI's end, when the increment carries out of the
index, and are laid out as DRI lays them out.

The layout puts a program's variables last, after its code, its constants,
the procedures' shared locals and the stack, as Intel's PL/M-80 does (see
Changed). DRI's programs use everything past their last variable as a
buffer, and MP/M II's SUBMIT and SPOOL now build from DRI's own sources and
behave as DRI's binaries do.

A procedure's variables are static, as the manual has them, its parameters
included, except where no program can tell: a local or a parameter shares
the procedures' overlaid storage only if every call assigns it before
anything reads it, and only where no pointer or overrun from another local
can tell it from DRI's layout (see Changed).

Every name means the declaration PL/M-80 gives it (`uplm80/names.py`):
labels, procedures and LITERALLYs of one name in different blocks, and in a
multi-file compile each module's private names, are kept apart; a GOTO that
PL/M-80 does not allow is an error; and a CALL through an address calls the
procedure. Every warning and error names the file and line it is about.

Each fix has a regression test that fails without it.

### Fixed

#### Procedure locals

- **A local without INITIAL did not keep its value from one call to the
  next.** PL/M-80 allocates a procedure's variables statically (Programming
  Manual, 8.1.7); uplm80 put every uninitialised local in `??AUTO`, overlaid
  on the locals of procedures never active at the same time, so a count, a
  first-time flag or a position kept in one came back as another procedure
  had left it: `tick: procedure address; declare (count, seen) address; if
  seen <> 1234h then do; seen = 1234h; count = 0; end; count = count + 1;
  return count; end tick;`, called in a loop with another procedure that
  has locals of its own, returned 1 every time, at every level (the
  integration verification's F6; 0.3.6 the same). Such a local is static
  now (see Changed). DRI's code counts on it: MP/M II's PIP, retrying a
  multi-file copy after an error, calls MULTCOPY again, which goes on from
  the directory entry and the count it keeps in NEXTDIR and NCOPIED
  (`if eretry = 0 then NEXTDIR, NCOPIED = 0`); both are static.
- **A pointer to a parameter, kept after the call, pointed into storage that
  other procedures overlay.** PL/M-80 allocates a procedure's parameters
  statically, like its other variables, but uplm80 kept every parameter in
  `??AUTO` because every call assigns it. After `sv: procedure (v); declare v
  byte; keep = .v; end sv;`, the sequence `call sv('P'); call other;` read
  0FFH (`other`'s locals) through `keep`, where DRI's code reads 'P' (0.3.6
  the same). A procedure whose address is taken also read a parameter of the
  procedure it is nested in out of `??AUTO` after that procedure had returned.
  A parameter is now static by the same rules as any other local (see
  Changed).
- **A pointer or an overrun from a local reached something other than what
  DRI's layout has there.** DRI lays out what a procedure's text declares in
  the order the text declares it (see Changed). uplm80 kept a frame's locals
  in `??AUTO` apart from three things: the parameters and locals of procedures
  nested among them, the variables of the procedure's DO blocks, and its
  INITIAL locals, which were static (0.3.6 the same). Examples:
  * with `declare arr (2) byte` before a nested `inner: procedure (v) byte`
    and `declare nxt byte` after it, `arr(2) = inner('I')` set `nxt` instead
    of `v`;
  * with `arr` INITIAL, `arr(n) = 'F'` did not reach the local declared after
    it, and neither did `.v + 1` with `v` INITIAL;
  * a read of `arr(2)` did not see what the previous call had left in `nxt`.

  These now reach what DRI's layout has there.
- **An INTERRUPT procedure's frame was put over the frames of the procedures
  it interrupts.** `??AUTO` overlays the frames of procedures that are never
  active at the same time, and the call graph decides which those are. Nothing
  calls an INTERRUPT procedure: the interrupt activates it (8.1.6), whatever
  is running. So `ih: procedure interrupt 7; declare (x, y) byte; ... call
  helper; end ih;` shared its frame with `work`, which `ih` may interrupt, and
  so did `helper` (0.3.6 the same). An INTERRUPT procedure and everything it
  calls are now active together with every procedure, and their frames overlap
  none.
- **A procedure called through an address, or called back from outside the
  module, shared its frame with its caller.** A `CALL` through an address
  (8.2.1) may call any procedure whose address is taken. An EXTERNAL procedure
  may call back any PUBLIC procedure, or any procedure whose address it was
  given. The call graph had none of those calls. `back` called an EXTERNAL
  `ext`, which called the PUBLIC `cb`; `cb`'s frame was `back`'s, and `back`'s
  local came back 0FFH at every level. The same happened to `through`, which
  passed the address of `cb2` to an EXTERNAL `icall` (0.3.6 the same). A
  procedure with a `CALL` through an address may now call every procedure
  whose address is taken, and an EXTERNAL procedure may call back every PUBLIC
  procedure and every procedure whose address is taken, whether the address is
  taken in the main program or in a procedure. A call of MON1 or MON2 with a
  constant function is a call of the BDOS, which calls nothing back. A `CALL`
  through an address now calls the procedure, too (see Names and labels):
  `CALL f` from `caller` to a procedure that fills its frame with 0FFH leaves
  `caller`'s locals as they were.
- **A parameter reached only through the address of the parameter before it
  lost its value at -O1 and up.** upeepz80 drops the store of the last
  argument at a procedure's entry when nothing else in the module names the
  parameter's storage. So `pq: procedure (a, b) byte; declare (a, b) byte; ...
  pp = .a + 1; return c;` (with `c` BASED on `pp`) returned 0 for `pq('A',
  'B')`. 0.3.6 did the same whenever no other procedure happened to name that
  slot in `??AUTO`. Each static parameter is now also named by an EQU, which
  costs nothing and which upeepz80 counts as a use. The rule itself belongs to
  upeepz80.
- **An embedded assignment was not stored when the procedure returned its
  target next.** RETURN took the value from A instead, even though the rest of
  the statement could change A and the target has to keep its value. MP/M II's
  LOAD reads a HEX file through `READCS: PROCEDURE BYTE; DECLARE B BYTE; CS =
  CS + (B := READBYTE); RETURN B;`, which returned CS + B, so LOAD built from
  source stopped with INVERTED LOAD ADDRESS on the first record of any file.
  With `v` static, `old = v; old = (v := old + 1); return v;` returned 1 on
  every call (0.3.6 the same for both). The store is now always made. LOAD
  built from source now produces, from a test HEX file, the same .COM as DRI's
  LOAD.COM: under MP/M II, and in BARE mode under cpmemu at -O0 to -O3. ED's
  GETSOURCE is 6 bytes longer.
- **A counted loop left its index with a value PL/M-80 would not give it.** A
  BYTE `DO i = 0 TO n` whose body does not name `i` counts its passes in B. It
  gives `i` its final value up front only if something may read `i`, and it
  allows a RETURN from the body when `i` is the procedure's own. Two things
  went wrong (0.3.6 the same):
  * a static `i` keeps its value until the next call, which read the final
    value (10) where a RETURN in the third pass leaves 2;
  * with `declare arr (2) byte, i byte`, `arr(2)` read 0 after the loop
    instead of 10.

  A loop is no longer counted when its index can be reached without naming it,
  or when the index is static and the body can RETURN.

#### Division and MOD

- **`x MOD 0` was 0; PL/M-80 gives `x`.** DRI's PL/M-80 sends every `/` and
  `MOD` through one routine, module @P0029 of its PLM80.LIB (the same 31 bytes
  sit at 39B0H in MP/M II's SDIR.PRL, and in DIR, ED, STAT, SHOW, TOD, SCHED,
  MPMSTAT, GENSYS, LINK and LIB). It has no zero test: sixteen
  shift-and-subtract steps, in which a zero divisor always fits, so the
  quotient comes out 0FFFFH and the remainder is the dividend. `??div16`
  tested for zero and returned a remainder of 0. SDIR's `page$len` defaults to
  0 and UTIL7/DSH.PLM asks `cur$line mod page$len = 0`, so every SDIR built
  from source reprinted its heading before every line of output. `??div16` and
  `??mod16` now run the same steps without the test, and are smaller: 31 bytes
  for the pair (50 before), 22 for `??mod16` alone. There is no BYTE divide to
  match: DRI zero-extends BYTE operands into the same routine (SDIR loads one
  with `LHLD` / `MVI H,00H` before `CALL 39B0H`), so a quotient or remainder
  of two BYTEs is an ADDRESS.
- **`SHR(x, 7)` lost bit 15,** and with it `x / 128`, which strength reduction
  turns into that shift: 8000H / 128 came out 0 instead of 100H. The result of
  a shift right by 7 has nine bits.
- **The compile-time forms of `/` and `MOD` disagreed with the runtime.**
  * Constant folding left `c / 0` and `c MOD 0` to the runtime; they now fold
    to 0FFFFH and `c`.
  * `0 / x` was folded to 0, but `0 / 0` is 0FFFFH, so the rule is gone.
    `x MOD 1` and `0 MOD x` (both 0) dropped a procedure call in the operand
    they discard; they now keep it.
  * A strength-reduced quotient or remainder of a BYTE became a BYTE:
    `x MOD 8` turned into `x AND 7` and `x / 1` into `x`, so
    `(x MOD 8) + 0FFH` wrapped at eight bits and `(x MOD 8) - 1` did not
    borrow. They stay ADDRESS now (as `DOUBLE(x AND 7)`), and code generation
    reads the byte itself wherever only the low byte is used or the value is
    compared with a BYTE, so assignments, arguments, subscripts and
    comparisons compile exactly as before.
  * A constant expression in DATA or INITIAL that the optimizer had not
    folded (any of them at `-O0`, and `c / 0` or `c MOD 0` at every level)
    went to the assembler, which rejects `7/0`, and `MOD` was written as `+`.
    Such an expression is now evaluated the way PL/M-80 evaluates it, and an
    operator the assembler cannot evaluate is an error rather than a `+`.

#### Expression types

- **A BYTE compared with a constant above 255 was rejected,** where the
  manual (4.4) compares the two as unsigned numbers and DRI's compiler
  accepts it: `IF b < 256 THEN ...` stopped with "comparison BYTE < 256 is
  always true", and so did `b <> 257`, `b = 300` and `(b + 1) < 256` (the
  integration verification's F5; 0.3.6 the same). It compiles, the BYTE
  zero-extended to meet the ADDRESS constant, and the message is a warning.
  The warning is given for a constant on either side - `IF 300 > b` said
  nothing - and at every level: the optimizer, moving a constant to the
  right of `=` or `<>`, marked it as one it had derived, which the check
  lets pass. The differential test generates such comparisons now; it kept
  a constant above 255 on the left, where it was not checked.
- **A folded constant was typed by its size, not by PL/M-80's rules.**
  `(8 MOD 0FFH) + b` with `b = 0FFH` gave 7 at `-O1` and above: the remainder
  is the ADDRESS 8, and adding 0FFH carries into the high byte (107H). The
  same folded 7 made `(7 MOD 0) > 1000H` fail to compile as "comparison BYTE >
  4096 is always false". Folding is typed now; an ADDRESS constant below 256
  is carried as `DOUBLE(n)`, and a constant the optimizer derived is not held
  against the program by the impossible-comparison check, nor is a relation
  between two constants.
- **`NOT` and unary `-` of a BYTE worked in sixteen bits,** even at `-O0`:
  `(NOT 7) MOD w` divided 0FFF8H. `NOT 7` is the BYTE 0F8H, `3 - 5` the BYTE
  0FEH and `-1` the BYTE 0FFH. The levels disagreed as well: `-O1` and up
  folded `-1` to 0FFFFH, while at `-O0` it was negated in HL and the
  BYTE-index path took A for the index, so `buf(-1)` was BUF-1 at one level
  and BUF+255 at another. It is `buf(255)` at every level now (see Changed),
  and the index path reads the register the index actually came out in.
- **BYTE `x * 2` became the BYTE add `x + x`,** so 200 * 2 was 144 at `-O2`.
  The product is an ADDRESS. Other rewrites kept the value but not the type
  and are fixed the same way: `b AND 0FFFFH`, `b XOR 0FFFFH`, `b + DOUBLE(0)`
  and `x * 1` are ADDRESS, `(b + 100) + 300` is not reassociated across the
  change of width, and a copy `w = b` is not propagated.
- **An element of a BYTE array member was typed ADDRESS.** Code for
  `s.m(i)` compared and combined it in sixteen bits, and a test against 0
  loaded the byte into A and then tested HL: SDIR (DSH.PLM:290, 294) printed
  every file's update and create stamps whether it had them or not.
- **`??mul16` took the carry of its own add into the product,** so any
  product that overflowed sixteen bits was wrong: 81H * 511 gave 817FH.
- **A nested subscript's release restored an outer claim's spill of DE,**
  and `aw(aw(aw(i) AND 7) AND 7)` added a stale DE in place of the base.
- **A BYTE value generated into A was read from HL** by a store through a
  BASED ADDRESS, by `MEMORY(b)` as a value and as a target, and by `MOVE`
  with a constant count and `TIME`; `OUTPUT(p) = b` replaced it with L.
  BYTE `1 - x` was `x XOR 1`, right only for 0 and 1.
- **`LOW` could take A for its operand after an embedded assignment.**
  `(b := w) + LOW(LAST(a))` added the low byte of `w` for 7, and
  `LOW((b := w) + 5)` - or `- 1`, `* 2`, `SHL`, `NOT` or unary minus in
  place of `+ 5` - took L of `w`: the flag that lets `LOW((b := w))` skip
  reloading A outlived the assignment. It is now used only when LOW's
  operand is the embedded assignment itself, as in ED's
  `LOW((N := SHR(NDEST,SECTSHF) - 1))`.
- **Shift and rotate counts of 129 or more shifted nothing:** the count loop
  tested the sign. Counts are unsigned BYTEs now, and a constant rotate is
  unrolled.
- **`LENGTH` and `LAST` were typed BYTE but generated into HL,** and an
  element of an untyped DATA array, `DECLARE hex DATA ('0123')`, was typed
  ADDRESS while being loaded as a BYTE. Code that goes by the type found
  nothing where it looked: `ab(LAST(sa))` read `ab(0FFH)`, and
  `hex(i) + 0FFH` added whatever HL held. A constant BYTE is now loaded
  straight into the register it is wanted in.
- **A 16-bit relation in an ADDRESS array's subscript** compared with a DE
  it had already restored for the subscript: `aw(w = 5)` read `aw(0)`.
- **An embedded assignment to a BASED BYTE** lost its value to the pointer
  the store loads into HL: `(x := w) + 1` added 1 to the pointer.
- **BYTE `0 PLUS x` and `0 MINUS x` cleared the carry they read:** `ld a,0`
  is `xor a` after the peephole. A constant left operand is now added to the
  other, by loads that leave the flags alone.

#### Calls

- **A BYTE argument to an ADDRESS parameter was stored as one byte,** unless
  it was the last argument, leaving the parameter's high byte from the call
  before. SHOW's and STAT's `pdecimal(getuser, 100, true)` printed the user
  number with the high byte of the previous number printed.
- **An argument that calls the procedure again,** `f(1, f(2, 3))`, stored over
  the arguments before it, which go into the procedure's own storage before
  the call. They now wait on the stack until it has run.
- **A `CALL` with five or more stacked arguments** (a REENTRANT, PUBLIC or
  EXTERNAL procedure) set SP to SP + HL instead of SP + 2n.
- **A call in the main program whose later argument calls a procedure** lost
  the earlier arguments: `CALL p2(5, g(1, 2))` passed 1 for 5. The earlier
  arguments are already in `p2`'s own storage, and the storage allocator,
  which keeps `g` out of it for such a call in a procedure, never looked at
  the main program's calls.
- **An array or structure local to a REENTRANT procedure did not
  assemble,** nor did a BASED variable whose pointer is one of its locals
  or parameters. The frame had room for them, but the code addressed them
  by a label that was never defined (`ld (V),a`, "Undefined symbol 'V'").
  An array or structure is now reached as IX plus its offset, and placed
  after the procedure's scalars, which `(ix+d)` has to reach, and a pointer
  is loaded with `ld l,(ix+d) / ld h,(ix+d+1)`; a scalar more than 128
  bytes into the frame is an error rather than a displacement the
  assembler rejects.
- **A local declared in a DO block of a REENTRANT procedure was below
  SP.** The frame was sized from the procedure's own declarations before
  the body was read, so the block's locals were outside it, and the next
  push or call wrote over them: a recursive `q` keeping `c` in a DO block
  returned 8 for 10. The frame is sized after the body now. (Both found by
  the integration verification or next to what it found; 0.3.6 did the
  same.)

#### DO loops

- **A BYTE loop to 255 ran no times.** DRI's PL/M-80 tests the limit before
  each pass and leaves the loop when the increment carries out of the index
  (GENSYS.COM's code for `do j = common$base to 0ffh` ends `INR A / JNZ top`;
  LOAD.COM's `BY 128` loop, `DAD D / JNC top`), so `DO j = 0 TO 255` runs 256
  times and leaves the index 0 (manual, 5.1.4). uplm80 tested
  `index < bound + 1`, and bound + 1 is 0, with a constant bound and with a
  variable one that is 255 - and at -O3 constant propagation turns the
  variable form into the constant one. BYTE and ADDRESS loops now end on the
  carry, so `DO w = 0FFF0H TO 0FFFFH` stops too; a step that would wrap stops
  the loop; and the limit, start and step are converted to the index's type
  (`DO b = 0 TO 300` runs to 44, and `BY -1` is `BY 0FFH`: PL/M-80 has no
  downward step). MPMLDR's `GENSYS.PLM` has two `TO 0FFH` loops that ran no
  times from 0. The wrap ends the loop even when the body put the index
  there: `DO i = 0 TO 0FEH` whose body adds 3 to `i` never stopped, since a
  constant limit and step that fit the index were taken to mean it could
  not wrap. The carry is now tested on every pass, as DRI's code tests it
  whatever the limit (LOAD.COM's `DO I = 0 TO 127` ends `INR M / JNZ`), so
  however the index got past the limit, the step that carries ends the loop.
- **A counted `DO` loop ignored its index being written or read in a
  target.** A loop whose body does not use its index counts in B, and never
  stores the index while it runs. `s(i).x = 0` and `i = n` in the body went
  unseen, so the index was never stored or its assignment ignored:
  `SCBRS.PLM` cleared one entry of its table four times, and `PIP.PLM` and
  `ED.PLM` (FILLSOURCE) kept reading past end of file. A variable bound of
  255 skipped the loop instead of running it 256 times: a count of 0 is 256
  passes to DJNZ, and the loop tested for 0 first.
- **A counted `DO` loop's index was stale to everything but its body.** An
  inner `DO` over the same index left it where it was, so
  `DO i = 0 TO 9; ...; DO i = 0 TO 9; END; END;` ran the outer body ten times
  instead of once; the code after the loop, a procedure the body calls, and
  the caller after a `RETURN` all read a stale index; and the bound was
  evaluated once, where PL/M-80 evaluates it at every test. The index now gets
  its final value before the loop starts, and a loop is counted only when
  nothing else can see its index - no procedure the body calls names it, no
  store reaches it through its address, no caller reads it after a `RETURN`
  from the body - and nothing can change its bound, by name or through a
  pointer (a bound BASED on `buf(3)` that the body sets through `buf(3)` was
  counted from its first value), and the bound does not read the index:
  `DO k = 0 TO k + 5` runs until `k + 5` wraps, 251 times, and was counted
  from `k`'s value before the loop. The final value is left out only where
  nothing can read the index afterwards.
- **A `RETURN` inside a counted `DO` loop left the count on the stack.** The
  count is pushed around the body, and the RET took it for its return
  address. A `RETURN` now pops what the loops around it pushed. Live in
  `SPBRS.PLM` (the spooler's stop request at the end of a line) and 80un's
  `lzh.plm` (a failed write).
- **A REENTRANT procedure's BYTE `DO` loop did not assemble** at `-O1` and
  above: upeepz80 turned the increment of `(ix+n)` into `ld hl,ix+n`. The
  index is now incremented in place, `inc (ix+n)`.
- **A BASED ADDRESS loop index stepped by 1 never wrapped:** the `inc hl`
  sets no flags, and the zero test that stands for it came after the store,
  which leaves the pointer in HL.

#### -O3

- **-O3 changed what programs do.** The inliner kept a `RETURN` that was not
  the last statement, which then returned from the caller - `ED.PLM`'s
  BACKSPACE, `PIP.PLM`'s and `SHOW.PLM`'s user checks and `TOD.PLM`'s
  COMPUTE$MONTH among them - defined a label once per call site, captured the
  caller's locals, and could inline another procedure of the same name. It now
  inlines only a small parameterless untyped procedure with no `RETURN` but a
  last one, no label, `GOTO` or declaration, whose names mean the same at the
  call as where it is declared. Nothing learned before a call was forgotten
  after it: `GENSYS.PLM` lost a whole `IF` after `get$response(.accept)`, and
  `cnt = 0; rw = f; call ph(cnt)`, where `f` increments `cnt`, printed 0. A
  call now ends everything the optimizer knows about variables, no fact is
  used in an expression that makes one, and a store through a BASED variable,
  a subscript or a member ends everything too, common subexpressions
  included. `.x` was folded like a value, so `c = 1; CALL setv(.c)` passed
  the address 1; a copy was propagated for a BASED variable, so 80un's
  `read16` returned its high byte twice; what a loop body sets was taken to
  be known after the loop; and an unrolled loop left its index at the last
  value.
- **-O3 unrolled a loop whose body could change its index,** through a call
  or a store through a pointer: `DO i = 0 TO 1; CALL bump; ...` with `bump`
  setting `i` ran twice. A loop is unrolled now only if its index is a
  plain variable, its body calls nothing, and it stores through no pointer
  when a pointer may reach the index. An index BASED on a pointer moves when
  the body sets the pointer, and one AT another variable changes when the
  body sets that variable by name: `DO x = 2 TO 0FEH BY 128` with `x BASED
  p` and `p = .buf(2)` in the body stored 2 and 82H through the moved
  pointer, and `DO y = 254 TO 0FEH BY 0FFH; g = 10; END` with `y AT (.g)`
  left `y` 0FDH, not 9 (0.3.6 was wrong at `-O3` too).
- **`-O3` turned `SIZE(b)` into `SIZE(5)`** after `b = 5`, which does not
  compile; SIZE, LENGTH and LAST name a variable, not its value.
- **`-O3` took a subscripted scalar's name for its value.** PL/M-80 lets a
  scalar be subscripted: `x(1)` is the byte after `x`, and DRI's code reads a
  following local that way. The optimizer put in place of the name the
  constant or the variable last assigned to `x`. After `x = 'x'`, `return
  x(1)` became a CALL through address 78H with the argument 1, and the
  program ran into page zero; after `w = 1234H`, `w(1)` was a CALL through
  1234H. After `x = y`, `x(1)` read the byte after `y`. 0.3.6 did this for a
  numeric constant and for some copies; once constants were typed, a
  character constant hit it too. The name of a subscripted variable is a
  place, like the target of an assignment, and is now left as it is.
- **`-O3` rejected a constant it had moved right of a relation:**
  `w = 'AB' <> b` was "comparison BYTE <> 16706 is always true" at `-O3`
  only. A constant the optimizer moves is marked as derived, like one it
  folds.

#### DATA, INITIAL and AT

- **`DATA` ignored member types and did not reserve the variable.** The 0.3.6
  fix that placed a STRUCTURE's `INITIAL` values member by member never
  reached `DATA`, which still emitted one byte per value and stopped at the
  last one; an array shorter than its dimension lost the rest too. `DATA` is
  `INITIAL` stored with the code (PL/M-80 Programming Manual, 6.2.9), and the
  two now share one emitter. `SPRSP.PLM`, `SCRSP.PLM` and `MSRSP.PLM` are the
  resident halves of the spooler, scheduler and status processes, nothing but a
  process descriptor and queues found by offset: SPOOL.RSP's queues were at 11H
  and 1CH, and are at 36H and 0CEH as in DRI's binary.
- **A string in a STRUCTURE initialiser filled one member.** A string fills one
  BYTE scalar per character and one ADDRESS scalar per two, and the width of
  each later value was taken from its position in the list instead. The queue
  control blocks in `SCRSP.PLM` and `MSRSP.PLM` got `msglen` and `nmbmsgs` as
  bytes.
- **A `LITERALLY` list stood for its first element inside `INITIAL`/`DATA`.**
  A special case added to match an earlier uplm80 cut the body at its first
  comma. MP/M's `restarts`, the case its comment cited, is nineteen `0C7C7H`
  words, and `SCBRS.PLM`, `MSBRS.PLM` and `SPBRS.PLM` build each process's
  stack as `initial (restarts,.entry)` with SP at `.stk+38`: the entry point
  landed in the second word and SP pointed at a zero.
- **An expression in `DATA` was always a word.** At -O0, where nothing folds it
  first, `x (4) BYTE DATA (68H+80H, k+1, 6)` took six bytes and moved
  everything after it, and a unary minus was not accepted: `SET.PLM` did not
  compile at -O0. An expression now fills its scalar at the scalar's width,
  evaluated as PL/M-80 evaluates a restricted expression: as plain 16-bit
  numbers, `/` and `MOD` as DRI's divide gives them.
- **`.(constant list)` in `DATA` or `INITIAL` was laid out in place.** It is the
  location of the constants (manual, 4.1.3), as it is in an expression, so
  `msgs (3) ADDRESS DATA (.('one$'), ...)` held characters, not pointers.
  `.'text'` in a list was not accepted at all.
- **Every name in a factored declaration got the first one's `AT` or values.**
  `DECLARE (A, B, C) BYTE AT (.BUF)` put all three at BUF, and
  `DECLARE (COUNTER, LIMIT, INCR) ADDRESS INITIAL (0, 1024, 2)` gave each name
  the whole list. Neither MP/M II nor 80un writes either form, but Intel's
  LINK does: `tests/link1a.plm`, from Mark Ogden's reconstruction, declares
  `(s, e) ADDRESS AT(.inRecord$p)` to reach the record pointer and the one
  after it, and `e = s + inRecord.len + 2` wrote over the record pointer.
- **`AT (.external +/- constant)` compiled to `EQU $`.** The catch-all at the
  end of `_emit_at_decl` was still there. `MSPL.PLM`'s
  `spool$msg (1) byte at (.tbuff-1)` sat on the queue control block after it.
  An AT address is now resolved as the manual defines it - a constant, or a
  location plus or minus constants - and anything else is an error. A location
  reference to a variable declared further down, which DRI's compiler
  accepted, is measured from that declaration instead of taken to be a byte.
- **A negative constant offset was written as `+65535`.** `.tbuff(-1)` in an
  `AT` was `TBUFF+65535`, whose relocation um80 0.3.48 drops, and a subscript
  of a variable AT an external was `EXT+c1+c2`, which it assembles as
  `EXT+c2`. Offsets from a symbol are written signed, and folded into one
  where the symbol is external. (An `AT`, `DATA` or `INITIAL` value is a
  restricted expression, evaluated as plain 16-bit numbers, so there `-1` is
  0FFFFH.)
- **An `AT` naming something declared further down was placed at 0.** An EQU
  is evaluated where it stands, and um80 0.3.48 takes a symbol it has not
  reached as zero; AT variables were defined in the data segment ahead of
  later declarations and of `??AUTO`. `SUB.PLM`'s `rbuff` at
  `.minimum$buffer` made SUBMIT build its command file at address 0. AT
  definitions now come after all storage.
- **An `AT` naming a later `AT` variable, or a later `EXTERNAL`, was still
  wrong.** `a1 AT (.b1 + 1)` above `b1 AT (.buf(2))` became `A1 EQU B1+1`
  ahead of B1's own EQU, so A1 was 0001H; the same for a later variable AT
  `.MEMORY`. A variable AT a later EXTERNAL was not aliased to it, and um80
  0.3.48 assembles `@A+2` with `@A EQU E1+1` as `E1+2`. A later declaration's
  own `AT` is now resolved down to its root, and a later EXTERNAL is known to
  be one. A circle of ATs is an error.
- **`AT` with `INITIAL` or `DATA` dropped the values without a word.** It is
  now an error.
- **In CP/M mode a module's DATA ran as code.** Module-level DATA was placed
  at the head of the program, where DRI's programs keep the jump they enter
  themselves by, but CP/M mode starts at 100H with its own entry code, and
  the DATA came first: `DECLARE t (2) BYTE DATA (0C9H, 42H)` returned to
  CP/M before the first statement. In CP/M mode it now follows the code;
  BARE and MP/M modes keep DRI's layout.

- **An element of an array BASED on a structure member, with a variable
  BYTE subscript, took its pointer from the start of the structure:**
  with `token BASED pcb.tok (4) BYTE`, `token(i)` read through
  `pcb.state`. 0.3.6 fixed the other paths for a variable BASED on a member
  (SDIR's `token BASED pcb.token$adr (12) byte`, which it subscripts only
  with constants) and missed this one.

#### Names and labels

- **A GOTO from a nested procedure to a label of the procedure around it
  did not assemble:** with `out:` in M1 and, nested in M1, `bail:
  procedure; goto out; end bail;`, the output jumped to `@M1$BAIL$OUT`,
  which nothing defines, at every level (0.3.6 assembled it at `-O3` only,
  by inlining BAIL). PL/M-80 does not allow it: "the label in the GOTO must
  be the label of a statement in the outermost level of the main program
  module" (Programming Manual 9800268B, 5.3.2; 8.1.3 and 9.3 say the same).
  It is a compile error now that cites the rule, and so are a GOTO into a
  block it is not in, a GOTO to a name that is not a label, and a label
  defined twice in one block. A GOTO out of a procedure to the main
  program's outer level, one within a procedure, and one out of a DO block
  to a label of a block around it work as before. A GOTO out of a
  procedure to a label in a DO block of the main program breaks the same
  rule, since the outer level is the module's exclusive extent (10.1), but
  Intel's PL/M-80 V3.1 compiles it without an error, to a plain `JMP` that
  leaves the procedure's return address on the stack, and 0.3.6 compiled
  it. It draws a warning that cites the rule and compiles to the same
  plain jump.
- **A GOTO in a procedure went to the main program's label of the same
  name,** not the procedure's own, silently: code generation found GOTO
  targets through its symbol table, which has the main program's labels and
  not a procedure's. `p: procedure; ... goto done; ... done: call pc('a');
  end p;` with a `done:` in the main program jumped out of P.
- **A GOTO from a procedure to a label at the outer level of the main
  program left the procedure's return addresses on the stack.** DRI's
  PL/M-80 reloads SP at such a label: MP/M II's PIP ends its ERROR
  procedure with `GO TO RETRY`, and in DRI's `PIP.PRL`, `RETRY:` begins
  with the same `LXI SP` as the program's entry. uplm80 only jumped, so each
  such GOTO leaked what the calls on the way had pushed. With `bail:
  procedure; n = n + 1; goto again; end bail;` and, in the main program,
  `again: if n < 1000 then call bail;`, `-m bare`'s 64-byte stack ran into
  the program after about a dozen rounds. MP/M II's PIP printed garbage and
  dropped back to the CLI after 86 errors in one interactive session. Such
  a label now reloads SP, as DRI's code does, and so does a PUBLIC label,
  which a procedure in another module can reach: `ld sp,??STACK` in bare
  and MP/M mode, and `ld hl,(6) / ld sp,hl` in CP/M mode. A label that
  only the main program jumps to is left as it was, as DRI leaves it. Every
  MP/M II program built from PL/M now loads SP as often as DRI's binary of
  it does: ED at five labels, GENSYS at two, and PIP, TOD, SCHED, PRLCOM
  and MPMLDR at one. Found by the release gate; 0.3.6 did the same.
- **A label in each of two DO blocks** of the main program or of one
  procedure was "multiply defined", both `LP:` or both `@P$LP:`, though
  each DO block has labels of its own (9.3). The second is `LP?2` now (no
  PL/M-80 identifier has a `?`). A `DECLARE l LABEL` in a procedure, and
  the address of a procedure's label, `.there` in a statement or in a DATA
  or INITIAL list, named the bare label.
- **A PUBLIC label that labels no statement at the outer level of the main
  program was left for the linker to find.** PL/M-80 requires a PUBLIC
  label to be attached to an executable statement there (9.3). With
  `declare again label public;` at module level and `again:` in a DO
  block, which is a label of the block's own, uplm80 emitted `public
  AGAIN` with nothing defining it: the module compiled alone did not link,
  and compiled with the module that jumps to it did not assemble
  ("Undefined symbol 'AGAIN'"). 0.3.6 took the DO block's label for the
  PUBLIC one. It is a compile error now that cites the rule and says where
  the other label is, as Intel's PL/M-80 V3.1 rejects the program (ERROR
  #172, INVALID LABEL: UNDEFINED). Found by the release gate.
- **`.show`, of a procedure nested in another, did not assemble:** it named
  `SHOW`, where the procedure is `@OUTER$SHOW`. A DATA or INITIAL list and
  an AT found it already; an expression does now.
- **`CALL q` through an ADDRESS variable (8.2.1) did not call the
  procedure** whose address q holds: it was `call Q`, which ran the bytes
  of Q itself, and `CALL s.p` was `jp (hl)` with no return address, so the
  procedure returned to its caller's caller. The address goes to DE and a
  new runtime routine, `??jpde`, jumps there from a CALL. The arguments are
  pushed, as a PUBLIC or REENTRANT procedure takes them, and the last is
  also left in HL and A, where a procedure private to its module takes its
  only one; more than one argument draws a warning, since such a procedure
  takes the others in storage the call cannot reach. The procedure called
  does not share its frame in `??AUTO` with the caller (see Procedure
  locals).
- **Procedures of one name in different blocks.** Code generation files a
  procedure under its enclosing procedures' names and its own, `P$Q`, and
  finds a name as a procedure nested in an enclosing procedure before
  anything else. So a procedure Q in each of two DO blocks of P was `@P$Q`
  twice ("multiply defined"); a procedure N in a DO block of the main
  program and a module variable N were both `N`; and, silently, a variable
  X of a procedure nested in P read as P's procedure X, and a variable V of
  P used outside the DO block of P that declares a procedure V as that
  procedure. With `do; declare q byte; ... do; declare e byte; q:
  procedure; ... end q; call q; end; ... end;` the procedure was generated
  under the variable's label. Such a procedure is renamed now (`Q?2`), and
  so is a procedure or a label that a DO block declares under the name of
  a parameter of the procedure around it, which is `@P$X` when it is
  static.
- **A DO block's variables are named after the block's number, `@B1$X`,**
  which is also procedure B1's X: the block's X took B1's place in
  `??AUTO` and the program printed B1's value for it. The number skips any
  that names a procedure.
- **LITERALLYs of one name.** One in each of two procedures, with
  different values, was `K EQU 1` and `K EQU 2`, and one named like a
  module variable or a main-program label met its label: "Symbol 'K'
  multiply defined". The later is `K?2` now. And code generation kept every
  LITERALLY in one table, so once procedure PA had declared `K LITERALLY
  '1'` a variable K of procedure PB read as 1 (silently, at `-O0` to
  `-O2`); a name is a LITERALLY there only where the declaration of it in
  scope is one.
- **A name declared twice in one block** was generated twice, and did not
  assemble; it is an error now. A LITERALLY declared again with the same
  text is let be (an $INCLUDE file and the file including it often both
  declare TRUE).
- **A declaration hides the built-in of its name when it is called or
  subscripted:** with `DECLARE size (4) BYTE`, `size(2)` was SIZE(2),
  "SIZE() needs a declared variable", and `high(1)` of an array HIGH was
  the high byte of 1. A variable already hid a condition flag read without
  parentheses, as MP/M II's STAT needs.
- **Names the assembler reads as something else.** um80 takes `A`, `HL` ...
  as registers and `EQ NE LT LE GT GE SHL SHR NUL` as operators: `call A`
  is "Register 'A' used as value", and `call EQ` calls 0FFFFH, `ld hl,SHL`
  loads 0, without a word. A procedure or label so named is `@A`, `@EQ`
  now, as a variable named like a register already was, and so is a
  variable named like an operator. It takes `Z NZ NC PO PE P` after `jp`
  for a condition ("JP with condition requires address"; the peephole
  makes `call p / ret` into `jp P`): the jump is written `jp 0+P`. And it
  takes a symbol whose letters end in one of its word operators (`MOD SHL
  SHR AND OR XOR NOT EQ NE LT LE GT GE HIGH LOW NUL TYPE`), followed by +
  or -, for that operator: `ld hl,TYPE+2` is TYPE(+2) and loads 0, and
  `X1EQ+2` or a procedure's `@Q$NUL+1` does not parse. Such an offset is
  written `2+TYPE`. (MP/M II's PIP has a variable TYPE, which it never
  offsets; its output is unchanged.)
- **`-O3` inlined the wrong one of two procedures of one name:** with a
  procedure NUL at module level and another in a DO block of P, a call of
  NUL in P after the block inlined the block's. And it inlined a procedure
  reading a variable Q into a block with a label Q, which its model of
  scope did not have (nor a scope for DO CASE): `qqqq` printed `q***`.
- **`INPUT(p)` and `OUTPUT(p) = v` with a port that is not a constant did
  not assemble:** they called `??inp` and `??outp`, which the runtime
  library did not have.

#### Diagnostics

- **A warning or an error named no file, or the wrong line.** `uplm80
  e.plm` printed `<unknown>:3:8: warning: comparison BYTE = 300 is always
  false`. The parser numbers the lines of the file with its $INCLUDE files
  spliced in and the lines a conditional skips taken out, so a warning in
  an included file came out at a line of the including one, and a syntax
  error there as `p.plm:1:1: error: ... at line 5, column 5`. Every
  diagnostic names the file and line it is about now - the included file
  for text from an $INCLUDE - and an error code generation raised without
  a location is placed at the statement or declaration it was generating.
  The warning for `IF 2` had no location at all, and an assignment was
  placed at its `=`.

#### Multi-file compiles

- **An EXTERNAL procedure that none of the files defined had no `extrn`.**
  `uplm80 A.PLM B.PLM` left out every EXTERNAL procedure declaration, on the
  grounds that one of the other files defines it, so a call to one that
  belongs to a third module did not assemble ("Undefined symbol"); A.PLM
  compiled alone had the `extrn`. Found by the integration verification;
  0.3.6 did the same.
- **Two modules with a private name in common did not assemble:** `uplm80
  a.plm b.plm`, with a procedure HELPER in each, gave "Symbol 'HELPER'
  multiply defined", and, since locals are static, `@HELPER$N` too; two
  module variables, DATA tables, labels or LITERALLYs of one name met the
  same way, and a module could use another's private name, which compiled
  alone it could not reach. PL/M-80 modules have separate name spaces for
  everything not PUBLIC or EXTERNAL (Programming Manual, 10.4). Each
  module's private names are now qualified with its name - `LIB?HELPER`,
  `@LIB?HELPER$N`; a module without a name goes by its file's - so each
  behaves as if compiled alone and linked. PUBLIC and EXTERNAL names bind
  across the modules as before, and a PUBLIC procedure can still be called
  from another module without an EXTERNAL declaration, as 80un's modules
  do. Using another module's private name is an error that says what to
  do.
- **A GOTO to a PUBLIC label of the main program, from a module that
  declares it EXTERNAL** (the third GOTO 9.3 allows), was "JR to 'AGAIN':
  its target is the external symbol AGAIN" at `-O1` and up: the EXTERNAL
  declaration was an EXTRN of a label the same assembly defines. No EXTRN
  is emitted for a name one of the modules makes PUBLIC.
- **A PUBLIC procedure's static parameter was `@KEEPIT$V?2`, not
  `@KEEPIT$V`,** when a module before the one that defines KEEPIT declared
  it EXTERNAL: the EXTERNAL declaration's parameter took the name, as if it
  were static in that module. The program ran as it should; only the names
  in the assembly differed from those of the module compiled alone, and now
  they do not. Found by the release gate (0.3.6 kept every parameter in
  `??AUTO`, with no name of its own).
- **Only the first of two modules with statements at their outer level was
  compiled;** the second's statements were dropped without a word. It is
  an error now: only the main program module may have them.

### Changed

- **A procedure's local shares `??AUTO` only if every call assigns it before
  anything reads it, and only where a program cannot tell the difference from
  DRI's layout.** `??AUTO` overlays the storage of procedures that are never
  active at the same time. What goes there now is the parameters and locals
  that a definite-assignment analysis (`uplm80/local_storage.py`) shows are
  assigned, on every path from the procedure's entry, before anything reads
  them. The analysis follows GOTOs and every form of DO. A call of a procedure
  nested in this one that names the local counts as a read at the call. A use
  of a BASED variable reads its base. A read through a subscript that is not a
  constant reads every local declared after the array. An array or structure
  counts as assigned once every element has been assigned through constant
  subscripts. A subscript is a constant when it folds to one by PL/M-80's
  rules (`a(1+1)`, `a(-1)`, which is `a(255)`, and `a(LAST(a))`), and every
  level now decides this the same way; before, -O0 took `a(1+1)` for a
  variable subscript and -O1 and up for a constant.

  Every other local and parameter is static, `@proc$name` among the variables:
  * one that may be read before it is assigned;
  * one whose address is taken (`.x`, in a statement, an AT or an INITIAL);
  * one reached outside its bounds: a scalar with any subscript but `(0)`, a
    constant subscript past the end, or a one-element array with any subscript
    but a constant 0;
  * one named by a procedure nested in its own procedure whose address is
    taken, or by anything that procedure calls, since it can run when the
    local's procedure is not active;
  * one declared with such a local in a factored declaration (6.2.4).

  DRI lays out what a procedure's text declares in the order the text declares
  it. The parameters come first: ERA's PRINT$FILE declares `k` before its
  parameter `fcbp`, and DRI's ERA.PRL has `fcbp` at 067AH and `k` at 067CH. A
  nested procedure's parameters and locals, and a DO block's variables, come
  where the text has them: SUBMIT's FILLRBUFF has `ssbp` at 0E7AH, the
  parameter of PUTRBUFF (declared next) at 0E7BH, and `reading` (declared
  after PUTRBUFF) at 0E7CH. That order is now kept wherever a program can
  tell:
  * everything declared after a local whose address is taken, or which is
    reached outside its bounds, is static too;
  * from the first array or structure subscripted by anything but a constant,
    a procedure's locals are either all static or all in `??AUTO`, and all
    static if any one of them is (an INITIAL one included). For example,
    `do i = 0 to n; a(i) = 'R'; end; return b(1);` with `declare a(2) byte,
    b(2) byte` has to reach `b`; the release gate's a4 printed 0 where 0.3.6
    printed 'R';
  * where a nested procedure with storage, or a DO block with variables, comes
    after either kind of local, what follows that local is static, and so is
    the nested procedure's storage.

  `??AUTO` also no longer keeps a slot for a declaration that never used one:
  an INITIAL, DATA, AT, BASED, PUBLIC, EXTERNAL or LABEL declaration in a
  procedure, and a REENTRANT procedure's parameters and locals, which are on
  its stack.

  In MP/M II (mpm2 at ef0a098, 87 compiles of which 77 assemble) and 80un,
  197 locals and parameters (4,687 bytes) are now static, the same at -O0,
  -O2 and -O3. The data comes to 45,095 bytes at each of those levels,
  against 46,754 before this change. The unused slots were 1,891 bytes. The
  static locals add back 216, since most of their bytes are buffers that also
  leave the frames that set the size of `??AUTO`, and the calls through an
  address and back from outside the module (see Fixed) 16. The code is the
  same except for two things: a parameter's store at a procedure's entry,
  which upeepz80 drops only when nothing else names its `??AUTO` address, and
  the embedded-assignment stores described under Fixed. Together they add 12
  bytes of code at -O0, 21 at -O2 and 15 at -O3.

- **Constants in expressions are typed as PL/M-80 types them, so some
  programs compute something else.** `w = -1` stores 00FFH, since the
  manual makes `-1` the BYTE `0 - 1` (write 0FFFFH for all ones), and for the
  same reason `buf(-1)` is `buf(255)`, not the element before `buf`
  (`buf(0FFFFH)` is); `NOT 0` is 0FFH. Comparing a BYTE with a constant from
  0FF00H up used to compare the low byte and is now the "always false/true"
  error, since the BYTE is zero-extended. An embedded assignment has the type
  of its right half (manual 4.6.3); BYTE `PLUS` and `MINUS` BYTE is a BYTE;
  `LENGTH` and `LAST` are BYTE when they fit; `CARRY` is 0FFH when set, as
  DRI's code has it; `SCL` and `SCR` have their pattern's type and rotate an
  ADDRESS in 17 bits; a two-character string is an ADDRESS constant, first
  character high.
- **`SHL` and `SHR` stay ADDRESS** even of a BYTE pattern, where the manual
  and DRI's compiler shift a BYTE in eight bits: programs written for uplm80
  rely on it (80un builds words with `lo + SHL(b, 8)`). A count of 0 leaves
  the pattern as it is; the manual leaves that undefined, and DRI's shift
  and rotate routines (SHOW.PRL's at 182BH to 1843H) have no zero test, so
  there a count of 0 shifts 256 times.
- **`DO` loops are laid out as DRI's are:** the limit is tested at the top,
  and the step jumps back only if it did not carry out, `jr nz` after an
  `INC`, `jr nc` after an `ADD`, so there is no jump to a test at the bottom
  and no separate wrap exit. A variable BYTE limit is compared in place,
  `ld hl,i / cp (hl)`. A limit, start or step that is a constant but not a
  literal (`LAST(x)`, `SIZE(x)`, `-1`) is used as one. With this layout and
  constants loaded straight into the register they are wanted in, MP/M II
  and 80un went from 225,929 bytes to 224,925 at `-O2`.

- **A program is laid out as Intel's PL/M-80 lays it out: the variables
  last.** The code and every constant - strings, `.(...)` lists, DATA
  declared in a procedure - are in the code segment (`cseg`); the data
  segment (`dseg`) holds `??AUTO`, then the stack of BARE and MP/M modes,
  then the variables in the order the source declares them, a procedure's
  static variables among them. ul80 puts every data segment after all the
  code, the runtime modules' included, so nothing follows a program's last
  variable, and `.MEMORY` is still the end of the whole program. DRI's
  programs use everything from their last variable up to MAXB, which works
  because DRI's layout puts nothing after it (MP/M II's `PIP.PRL` sets
  `LXI SP,2251H`: its stack is at 21EDH to 2250H and its variables start at
  2251H). `UTIL5/SUB.PLM` builds SUBMIT's command file at
  `.minimum$buffer`, and `UTIL5/MSPL.PLM` reads a file into `.dummy$buffer`;
  uplm80 put the strings, `??AUTO` and the stack after the last variable, so
  a command file of more than about 1.1K overwrote SUBMIT's messages, and
  SPOOL printed NULs for the first records of a file it spooled itself.
  In MP/M II and 80un the change only moves lines: every output holds the
  same instructions and data as before, in another order.
- **What PL/M-80 does not allow is an error,** where it compiled to
  something wrong or did not assemble: a GOTO out of a procedure except to
  the main program's outer level or an EXTERNAL label (to a DO block of the
  main program it is a warning, and compiles as Intel's PL/M-80 compiles
  it), a GOTO into a block or to a name that is not a label, a PUBLIC
  label that labels no statement at the main program's outer level, a name
  declared twice in one block,
  and, compiling several modules together, a second main program module or
  a name another module declares without PUBLIC.
- **Some names in the output are new.** In a multi-file compile a module's
  private names are qualified, `MODULE?NAME`; where two declarations would
  meet in one assembler name the later is `NAME?2`; and a procedure, label
  or variable named like a register or an um80 operator is `@NAME`, PUBLIC
  and EXTERNAL ones included, as a variable named like a register always
  was, so an assembly module that defines or uses one has to use that
  name. A static parameter is also named `?@proc$name`, by an EQU (see
  Fixed, Procedure locals). The new names change no single-file output of
  MP/M II and 80un, and the 80un multi-file programs' `.COM` files are byte
  for byte what they are without them.
- `docs/multi_file_compilation.md` said to call a PUBLIC procedure from
  another file without declaring it EXTERNAL, and that declaring it
  EXTERNAL would call it the wrong way; since a PUBLIC procedure takes its
  arguments on the stack, the EXTERNAL declaration is right, and it is what
  lets each module also be compiled alone.
- `CLAUDE.md` and the README's example of linking said to link a
  `runtime.rel`, and `CLAUDE.md`'s list of runtime routines left out
  `??jpde`, `??inp` and `??outp`. A module carries the runtime routines it
  uses at the end of its code; nothing is linked for them.
- **uplm80 requires upeepz80 0.2.5.** upeepz80 0.2.4 deleted register
  loads that were still needed, at `-O1` and above; the differential test
  found each, and each was in 0.3.6's output as well. `b, w = -(NOT b)` left
  `w` holding A's old value; after `b = 0FEH`, `w = LOW(LAST(big))`, with
  `big` 300 bytes long, got the low byte of the statement before instead of
  2BH (seed 1063 of `scripts/difftest.py`); `ld hl,0 / ld a,l / ld (sb),a /
  push hl / ld (w1),hl / pop hl / ld (w0),hl` lost its `ld hl,0`, since
  `ld (nn),hl` did not count as a read of HL; and `ld a,(ix+n) / inc a /
  ld (ix+n),a` became `ld hl,ix+n`, which is not a Z80 instruction. 0.2.5
  makes a rewrite only where nothing reads what it changes.

### Added

- **`tests/plm_difftest.py`**, a differential test: random programs over
  BYTE and ADDRESS variables, constants, every operator but `PLUS` and
  `MINUS` (whose carry-in depends on the code before them), built-ins, calls
  of procedures that change globals, BASED stores and DO loops, compiled at
  `-O0` to `-O3`, run under cpmemu and compared with a Python model of the
  manual's rules. The suite runs three programs; `scripts/difftest.py
  --seeds N` runs more.
- **`tests/names_difftest.py`**, a differential test of name resolution:
  random programs that declare a pool of ten names as variables,
  LITERALLYs, procedures and labels at every depth, with the output their
  scopes give, one module or three compiled together. The suite runs six;
  `scripts/namestest.py --seeds N [--modules]` runs more.

### Known issues

- **An overrun or a pointer that runs backwards from a local, past a
  procedure's last local, or past a module-level variable** reaches what DRI's
  layout has there (the variable the text declares before or after it,
  possibly another procedure's) only where that variable is static. In
  `??AUTO` it reaches another frame, or nothing (0.3.6 the same). Keeping all
  of that in DRI's order would make every local static.
- **`STACKPTR` read inside an expression can see a temporary the compiler
  pushed.** Where an operand evaluated before it is kept on the stack,
  STACKPTR reads 2 less than it does at the start of the statement. After
  `sp0 = stackptr`, `sp0 <> stackptr` is true at `-O0` to `-O2`; `-O3`
  evaluates `stackptr <> sp0` and `stackptr = sp0` in that order too, and
  all three find the two unequal. Intel's PL/M-80 V3.1 does the same, in
  other places: for `d = stackptr - sp0` it pushes a temporary and then
  reads SP. (0.3.6 folded all three comparisons at `-O3` to "equal".) The
  manual gives STACKPTR as the stack pointer register (11.2.3), not as it
  was when the statement began.

### Known issues — not this compiler

- **um80 0.3.50 reads a symbol whose letters end in one of its word
  operators, followed by + or -, as that operator** (`find_binary_addsub`
  takes the letters before the sign, `[A-Za-z]+$`, for a word operator even
  when they end a longer symbol), and `EQ`, `SHL`, `NUL` and the like on
  their own as operators, without an error: `ld hl,TYPE+2` loads 0, `call
  EQ` calls 0FFFFH. uplm80 renames or rewrites what it emits so as not to
  meet it; an assembly module written by hand can. um80 0.3.51 reads all of
  these as M80 does, as the symbols, and so does `jp P`; uplm80 keeps the
  renames and rewrites (`names.fix_symbols`, `data_name`) for older um80s.
- **With `um80 -t` (PUBLIC and EXTERNAL names cut to six characters, as
  MACRO-80 does), ul80 links two PUBLIC names that agree in their first six
  characters as one** - PRINTCHAR and PRINTCRLF are both PRINTC - reporting
  "Multiply defined global" and linking anyway. uplm80's output keeps its
  names whole and is assembled without -t.

### Verified

- Every PL/M source in MP/M II (the 41 in DRI's tree and the 14 overrides,
  each in the mode `tools/build.py` uses) and in 80un (35 files one at a time,
  and `80un.com` and `80unbas.com` as their Makefile compiles them) compiles
  at -O0, -O2 and -O3, and the same 82 of the 92 outputs assemble with um80
  0.3.49 as with 0.3.6 (the rest are single modules of multi-module programs,
  and MSCMN.PLM, which is only ever included). At -O2 every output changes
  from 0.3.6, if only by the layout, and every change is one of the entries
  above: of 1,733 changed hunks (4,278 lines that only moved aside), 601 are
  the DO-loop layout and counted loops, 272 shift and rotate counts, 221
  constants loaded straight into the register wanted, 182 the `cseg`/`dseg`
  split, 151 BYTE operations kept in A, 136 the runtime routines, 91 ATs
  resolved to their root, 34 `SHR(x, 7)`, 23 DATA and INITIAL, 8 `CARRY`, 7
  a BYTE argument widened, 5 `jp`/`jr` distances and 2 the `extrn` below.
  The 82 come to 224,925 bytes, against 227,511 with 0.3.6, before the
  change to the allocation of locals (see Changed). The other fixes made
  after the integration verification (the entries that name it) change no
  output of MP/M II or 80un at -O0, -O2 or -O3 but the multi-file
  `80un.com` and `80unbas.com` compiles, which now declare `extrn MON1` and
  `extrn MON2`: 80un declares them EXTERNAL, and a CP/M-mode call goes to
  BDOS directly. `80un.com` extracts the same files, with the same console
  output, from all 17 sample archives as 0.3.6's, at -O2 and -O3, and
  `80unbas.com` detokenises `PALLOPS.BAS` the same.
- The allocation of locals (see Changed), measured with um80 0.3.50 and
  upeepz80's `fix/peephole-live-registers` against the commit before it:
  it changes 70 of the 82 outputs at each of -O0, -O2 and -O3, moving
  locals out of `??AUTO` into the variables, and with them the `??AUTO`
  offsets and the addresses of the variables that follow (the 82 go from
  224,744 bytes to 222,984 at -O2). `80un.com`
  and `80unbas.com` built from the new outputs, at -O2 and -O3, extract
  the same files with the same console output from all 17 sample archives,
  and detokenise `PALLOPS.BAS` and the four BASIC samples the same, as
  those built before it; GENSYS, one of the programs whose locals move,
  makes the same MPM.SYS and SYSTEM.DAT and prints the same, for V2.0 with
  the three sets of answers, at -O2. `scripts/difftest.py --seeds 200
  --first 9000`, with the generator as it was: all 200 programs as the
  model says at `-O0` to `-O3`; and with the generator that now compares a
  BYTE with a constant above 255 either way round, `--seeds 200 --first
  11000`: all 200.
- MP/M II built from source with this release - `tools/build.py` for V2.0
  and V2.1, 44 of 44 targets each - passes mpm2's `scripts/run_tests.sh all`
  on the V2.1 system and `scripts/run_tests.sh src`. SUBMIT and SPOOL were
  built from DRI's own `SUB.PLM` and `MSPL.PLM`, without the workarounds
  mpm2 carried for the old layout, and compared on the emulator, on the
  V2.0 system built from source, with DRI's V2.0 `SUBMIT.PRL` and
  `SPOOL.PRL`: SUBMIT runs two 300-line command files
  (3968 and 3328 bytes), a 250-line one of 15K, and one with parameters,
  printing exactly what DRI's does, where the release before the layout
  change printed its own messages over the 3968-byte file and ran none of
  it; SPOOL, printing files itself on a system without the spooler RSP,
  prints a 150-line file and a 7936-byte one as DRI's does, where before it
  printed 512 NULs in place of their first records. `stat usr:` now prints
  what DRI's STAT prints. GENSYS built from source, whose DATA now follows
  its code, prints what DRI's GENSYS prints under cpmemu, for V2.0 and
  V2.1 and with three sets of answers, and makes the same MPM.SYS and
  SYSTEM.DAT but for the six bytes of the serial number at 0B5H: the build
  leaves DRI's placeholder there, "654321", and the V2.0 GENSYS.COM in DRI's
  MPMLDR directory, which has the placeholder too, makes byte-identical
  files. Against DRI's serialised GENSYS - V2.0's in CONTROL, V2.1's on the
  distribution disk - the two files differ in those six bytes and nowhere
  else.
- The differential tests: `scripts/difftest.py --seeds 400 --first 1000`,
  all 400 random programs as the model says at `-O0` to `-O3` (seed 1063,
  which the second upeepz80 defect above broke, no longer meets it); and the
  integration verification's own generator, which covers DATA, INITIAL, AT,
  BASED, DO loops whose body moves the index or the bound, calls in
  arguments and module-level code: 497 programs at `-O0` to `-O3`, all as its
  model says. The independent verification of the release wrote a third,
  over typed expressions, calls that call back, nested and REENTRANT
  procedures, BASED, AT, structures, DATA, every form of DO, and programs
  of two modules built both separately and as one multi-file compile:
  2001 programs, 12,276 builds, whose only compiler defects were the two
  `-O3` unrolling defects above. Its seeds 1500 to 1599, 20480 to 20579 and
  30001 to 30100, taken again with this release, all run as its model says.
- The run tests compile with the checkout under test: they used to start the
  compiler with `python -P`, which found whatever uplm80 was installed. Every
  test that runs a program - the run tests, the differential test and the
  division oracle - now assembles, links and runs it with
  `tests/_toolchain.py`.
- MP/M II built from source with the fixes to procedure locals above
  (`build_all.sh --tree=src`, V2.0 and V2.1) passes mpm2's
  `scripts/run_tests.sh all` for both: DIR, STAT, STAT drive, the resident
  system processes, HTTP and SFTP.
- A 122-command session of DIR, STAT, SDIR, TYPE, PIP, SET, SHOW, ED,
  MPMSTAT and SCHED prints exactly what the same session prints with the
  tools built by 86d2720, apart from MPMSTAT's snapshot of which processes
  are delayed.
- `80un.com` and `80unbas.com` at -O2 and -O3 extract the same files from
  all 17 sample archives, and detokenise the BASIC samples the same, as
  86d2720's builds.
- The fixes to names and labels change none of that. Each of the 87
  compiles of the current MP/M II and 80un sources (77 of them assemble)
  gives the same `.mac` with them as without them, at -O0, -O2 and -O3,
  but for the two 80un multi-file programs, whose private names are
  qualified; those link to the same `80un.com` and `80unbas.com`, byte for
  byte.
- `scripts/difftest.py --seeds 200 --first 1000`: all 200 programs as the
  model says at -O0 to -O3. The release gate's generator of locals (static
  locals, nested readers, GOTOs, variable subscripts, pointers): 500
  programs at -O0 to -O3, all as its model says. `scripts/namestest.py`:
  1000 one-module and 480 three-module seeds before the fixes to locals
  were merged, and 300 and 100 after, all at -O0 to -O3, all print what
  their scopes say.
- The release gate's adversarial programs a1 to a14 print at -O0 to -O3
  what 86d2720's build printed, apart from a4, which prints `RST` at every
  level where 86d2720's printed `T` at -O0 and `ST` above, and a5 and a6,
  whose GOTO from a nested procedure to a label of its parent is a compile
  error now (86d2720's output did not assemble).
- The release gate's next round found one defect, `-O3` taking a subscripted
  scalar's name for its value (see Fixed, -O3). With the fix, its programs
  that subscript a scalar print at `-O3` what they print at -O0, and its
  other programs, b1 to b23, a1 to a14 and the multi-file sets, print at
  -O0 to -O3 what they printed without it. Each of the 87 compiles of MP/M
  II and 80un gives the same `.mac` with the fix as without it, at -O0, -O2
  and -O3. `scripts/difftest.py --seeds 200 --first 2000`: all 200 programs
  as the model says at -O0 to -O3; the gate's generator of locals, 100 seeds
  of each of its two versions: all 200 as its model says.
- The release gate's round after that found the GOTO out of a procedure
  that left its calls on the stack (see Fixed, Names and labels). With the
  fix, its program of sixty such GOTOs from two calls down prints sixty
  `e` and then `ad.` with `-m bare` at -O0 to -O3, where it stopped after
  11 to 18, and its program of twenty thousand prints `12.`, where it
  printed `.`. The MP/M II and 80un compiles change only by the new loads
  of SP, 52 bytes of code at each of -O0, -O2 and -O3, and GENSYS makes the
  same `MPM.SYS` for V2.0 and V2.1. MP/M II built from source for V2.0 and
  V2.1, 44 of 44 targets each, passes `run_tests.sh all` for both and
  `run_tests.sh src`, and `verify_dri.py` reports what it did. On the V2.0
  system built from source, one PIP session fed `t9.txt=nosuch.txt` 120
  times and then `con:=t1.txt` prints the 120 errors and types T1.TXT,
  exactly as DRI's `PIP.PRL` does. The 122-command session prints what it
  printed without the fix, apart from MPMSTAT's list of processes. The
  gate's other programs print what they printed before, at -O0 to -O3.
  `scripts/difftest.py --seeds 200 --first 6000`, the gate's generator of
  locals (100 seeds of each version) and `scripts/namestest.py` (100
  one-module and 40 three-module seeds): all as their models say.

## 0.3.6 — 2026-09-24

Found by building every PL/M program in Digital Research's MP/M II sources and
running each one next to DRI's own binary on the same disk, command by command,
until the two printed the same thing. Adds an MP/M runtime mode. Each fix has a
regression test that fails when it is reverted, and 80un's `80un.com` and
`80unbas.com` still rebuild byte-identical.

Requires upeepz80 0.2.4, whose dead-store elimination deleted the store in
`var = (a = b)`.

### Added

- **`-m mpm`, for MP/M II `.PRL`/`.RSP`/`.SPR` modules.** MP/M gives each
  process a memory segment and puts its page zero at the segment's base, so the
  BDOS entry, the stack-top pointer at 0006H and the warm-boot jump have to be
  relocated when the program loads. Only a resolved symbol reference reaches a
  `.PRL` relocation bitmap, so MP/M mode emits them as the externals `??BDOS`,
  `??MAXB` and `??BOOT` rather than literals (link with a module that defines
  them at 0005H, 0006H and 0000H, and with `ul80 --prl`, which relocates
  page-zero symbols). This is how DRI's PL/M-80 got the same effect:
  `PLM_WORK/X0100.ASM` and `X0200.ASM` publish the same names at different
  offsets and GENMOD diffed the two links. The names carry the compiler's `??`
  prefix because a PL/M identifier cannot contain `?`, and something collides
  otherwise: SDIR declares a variable called `bdos`. MP/M mode sets SP with the
  single three-byte `ld sp,??STACK` over a 512-byte stack in the image, because
  DRI's sources enter themselves by a jump to `.start-3`. CP/M and bare modes
  are unchanged.

### Fixed

- **A `PUBLIC` procedure took its arguments the way a private one does.** A
  procedure private to its module is called with the earlier arguments already
  written into its own storage and only the last in a register; a caller in
  another module cannot name that storage. A public procedure now takes all its
  arguments on the stack. SDIR's `pdecimal(v, prec, zerosup)` read two of its
  three arguments from slots nobody had written.
- **`AT(.MEMORY)` was not the end of the program.** It named a label at the
  end of the *module* — the middle of a multi-module program — and was emitted
  as an EQU, which reads as zero above its own declaration. It is now a label
  beside the linker's `__END__`, with the `EXTRN` ahead of the EQU that uses
  it. SDIR's 128-entry hash table first cleared page zero, then another
  module's strings.
- **`AT(...)` understood only a bare `NAME(<literal>)`.** A constant
  expression, `NAME(const)`, `STRUCT.MEMBER` or a chain of them fell through to
  `EQU $`, the assembler's location counter. STAT read a stray byte as its `$`
  parameter and set a file read-only instead of listing it; PIP
  (`DESTR ADDRESS AT(.DEST.FCB(33))`) and PRLCOM were miscompiled the same way.
  An `AT` that cannot be resolved is now an error. `AT(.name)` uses EQU rather
  than SET, and `AT(.external)` emits the EQU even when a reference comes
  first.
- **A STRUCTURE initialiser was emitted at one width.** It gives one value per
  member, each at the member's own width; whatever the list does not fill is
  now reserved. SDIR's ten-byte parser control block came out as five bytes of
  zero. A value the emitter could not place was dropped silently and is now an
  error; `.name(n)` is placeable.
- **`x BASED s.m` read its pointer from the start of `s`.** The member was
  parsed and dropped. SDIR matched every command-line argument against
  address 0 and answered "File Not Found."
- **`DECLARE x (*) BYTE DATA (...)` had no extent.** `LAST(x)` was -2, so PIP
  never searched its delimiter table and answered "INVALID FORMAT" to every
  command.
- **A nested procedure's return type was not known where it is used.** Its
  symbol is filed under its scoped name and the lookup searched only the top
  level, so a `BYTE` result was read out of `L` instead of `A`. `LENGTH`,
  `LAST` and `SIZE` used the same lookup; where they cannot answer they now
  raise instead of emitting zero.
- **A variable `BY` step was treated as `BY 1`.** Only a literal step was read.
  SDIR walks an FCB disk map `BY i`, and counted every block twice on a disk
  with word block numbers.
- **A declared variable did not shadow a condition-flag built-in.** `CARRY`,
  `ZERO`, `SIGN` and `PARITY` are ordinary words a program may declare; STAT's
  zero-suppression flag `zero` read the Z flag. `STACKPTR` deliberately stays a
  built-in, since assigning to it sets SP.
- **A callee's frame was reused while its own arguments were evaluated.** The
  overlay analysis let a procedure called from a later argument share storage
  with an earlier argument already stored, so `call f(7, g)` could destroy the
  7. Bites at the default `-O2`.
- **A procedure-local STRUCTURE was sized as two bytes.** The next procedure's
  frame was overlaid inside it. `-O2`.
- **The target of an embedded assignment was folded like a value.**
  `q = (k := 7)` after `k = 5` stored through the literal 5 and left the stale
  fact about `k` in place. `-O3` only.
- **An induction step hidden in a subscript did not count as modifying the
  variable,** so `arr(i := i + 1) = 9` could fold away a loop's exit test.
  `-O3` only.
- **Only the module that sets SP carries a stack buffer;** a program linked
  from eight modules was carrying eight.

## 0.3.5 — 2026-09-22

An audit prompted by the 80un report. The three defects 0.3.3 and 0.3.4 fixed
turned out to be members of a family that was never swept, and the audit also
found the language defect underneath the `AND`/`OR` story: PL/M-80 tests bit 0
of a condition, not whether the condition is non-zero. Thirty-two fixes, each with a
regression test; see Added for what that was and was not verified to mean.

### Fixed

- **A condition was tested for non-zero; PL/M-80 tests BIT 0.** Every `IF` and
  `DO WHILE` lowered to `or a` / `jp z` (and `ld a,l` / `or h` for a 16-bit
  value). DRI's binaries settle the rule: the code segment of `PIP.PRL` holds
  70 `RAR;JNC` and 14 `RAR;JC` truth tests against a single `ORA A;JZ`,
  `SDIR.PRL` 125 against none, and `ED.PRL` 77 against none. DRI's own hand
  translation of `bdos.plm`'s `IF NOT ROR(ROL(DLOG,1),CURDSK+1)` is
  `mov a,l! rar! rc` — a bit-0 test, and nothing else makes that source
  correct. A relational yields 0FFH or 00H, so both rules agree there; they
  part company on `NOT` and on any masked or rotated value.

  Two consequences, both live in the MP/M II corpus:

  * `NOT <0/1 flag>` inside an `AND`/`OR` read as true. `NOT 1` is 0FEH,
    non-zero. This is what commit 273c83a exposed: before it, `AND`/`OR`
    recursion reached the `NOT` handler, which compiled `NOT` by inverting the
    branch sense and so happened to be right. Eleven sites changed truth value.
    The worst is SDIR's hash-chain scan, `DSE.PLM:340`
    `do while f$i$adr <> 0 and not found;` with `true literally '1'` — the
    match arm sets `found` without advancing the pointer, so the loop never
    terminated and SDIR hung on any repeated name, which means every second
    extent of a file or an XFCB paired with its FCB. Also `DM.PLM:605`
    ("File Not Found." printed after every successful listing) and
    `PIP.PLM:1752` (ambiguous filenames no longer rejected).
  * The DRI bit-extraction idiom was broken everywhere, independently of the
    above. `PIP.PLM:1376-1381` writes every FCB attribute test as
    `if rol(source.fcb(n),1) then ...`, rotating bit 7 into bit 0; the
    non-zero test succeeded for essentially any filename character, so PIP
    stamped F1 — and, from `fcb(9)`/`fcb(10)`, R/O and SYS — onto files that
    did not have them. At least 19 sites across PIP, ED, SDIR and SET.

  Conditions now emit `bit 0,a` (BYTE) or `bit 0,l` (ADDRESS). The 16-bit form
  is the same size as before and no longer clobbers `A`. 273c83a itself was
  correct and stands: PL/M-80's `AND`/`OR` are bitwise, full-evaluation
  operators. The two `NOTE` comments it left behind said the result was tested
  for non-zero; they now state the bit-0 rule.

- **A PROCEDURE declared at the head of a `DO ... END` block was emitted
  inline.** PL/M-80 allows it anywhere a block begins, not only in a procedure
  body, and MP/M II's `ED.PLM` declares `DIGIT` / `NUMBER` / `RELDISTANCE` that
  way inside an IF/ELSE chain. Nothing jumped over the body, so the enclosing
  code ran straight into the procedure and took its `RET`; everything after the
  block was unreachable. Since the uplox front-end migration the label also
  carried the block scope (`@B24$DIGIT`) while the call sites did not, because
  no collection pass descended into blocks — which turned the silent
  miscompile into `Undefined symbol 'DIGIT'` and broke the build outright.
  `ED.PLM` was the last MP/M II target that would not compile. Such procedures
  are now hoisted out, named in the enclosing procedure's scope, and emitted
  after its body, and they keep the block's symbol scope so they still see the
  block's locals.

- **`_expr_preserves_de` claimed a BASED ADDRESS variable preserves `DE`.** Its
  load is `ld hl,(base) / ld e,(hl) / inc hl / ld d,(hl) / ex de,hl`, which
  writes `DE` and leaves base+1 in it. Four callers trusted the predicate, so
  `baccum = baccum + bpb` in `SHOW.PLM:959` computed `baccum + (ab+1)`; the
  same shape is in `DSE.PLM:250`, `STAT.PLM:1136`, and `DM.PLM:237`, where
  `vector = vector or 1` ORed in `v$adr+1`. A BASED BYTE is unaffected — it
  loads through `A`.

- **`-O 3` turned an assignment into a store through an absolute address.**
  Targets were run through the same optimizer as values, so constant
  propagation rewrote the `a` of `a = 5` into the literal `5` and the store
  landed on memory 0005H — CP/M's BDOS entry vector. Only an array element's
  subscript is a value, and only that is optimized now.

- **The byte-comparison operand was still parked in `B` on the condition
  paths.** The 0.3.3 spill was applied to `_gen_byte_binary` and
  `_gen_byte_comparison` but not to `_gen_condition_jump_false` /
  `_gen_condition_jump_true`, which 273c83a had just made the load-bearing
  path. `IF f > x` with `f` = 100 and `x` = 200 read true. The operand is now
  spilled through the stack when the expression generated in between can
  clobber `B`; when it cannot — a literal, a plain variable — the shorter
  sequence is kept, so the common `IF a > b` is unchanged.

- **A 16-bit iterative `DO` parked the loop index in `DE` across the bound
  expression.** A bound that is itself 16-bit emits `ld de,nn` or calls a
  runtime helper. The index goes on the stack unless the bound provably leaves
  `DE` alone.

- **MON1/MON2 parked the BDOS function number in `C` across the argument.**
  `C` is not callee-saved, so `CALL MON1(2, GETC)` reached the BDOS with
  `GETC`'s own function number. The function number is loaded last, which costs
  nothing.

- **A multi-target byte assignment parked the value in `B` across the store.**
  Storing to a subscripted or based target generates the index expression,
  which may call a procedure. `push af` / `pop af` is the same two bytes as
  `ld b,a` / `ld a,b`, and is what the ADDRESS path already did.

- **`SHL(DOUBLE(hi),8) OR lo` parked the high byte in `H` across the low
  operand.** Any low operand that computes in `HL` — a call, a subscript —
  destroyed it. The four-instruction form is kept when the low operand is a
  plain byte load and spilled otherwise.

- **A byte comparison used as a VALUE did not get its left operand into
  `A`.** The condition paths were repaired earlier in this release;
  `_gen_byte_comparison`, the value-producing twin, still opened with
  `_gen_expr(left)`, so a NumberLiteral loaded as `ld hl,n` and the closing
  `sub b` compared an undefined `A`. `r = 5 > x` with `x` = 3 gave 0 rather
  than 0FFH, at the default optimisation level.

- **A cached constant was not narrowed to its variable's declared width.**
  The store truncates — `b = 300` leaves 44 in a BYTE — but the optimizer
  remembered 300, so at `-O 3` the following `IF b = 44` folded to false.

- **A folded relational disagreed with the computed one.** A PL/M-80
  relational yields a BYTE 0FFH; the folder masks to 16 bits, so
  `w = (1 = 1)` stored 0FFFFH at `-O 2` and 00FFH at `-O 0`. Relationals are
  now folded only in a condition, where nothing but bit 0 is observable. As
  a value the comparison is left to the generator, and all four optimisation
  levels emit the same code. This closes the Known issue the previous
  release note carried.

- **The last statically-typed widening in the comparison code.** In the
  branch that parks a complex right operand in `DE`, the widening keyed off
  `_get_expr_type` rather than the type `_gen_expr` returned. They disagree
  for an embedded assignment: `ar(i) := b1` with `ar` an ADDRESS array is
  typed ADDRESS but leaves a BYTE in `A`, so `ex de,hl` took `DE` from
  whatever was in `HL`. Both arms of
  `IF a1 > (ar(i) := b1)` came out false.

- **`IF 2` was reported as "always true"** while the generator made it
  false. The diagnostic now follows the bit-0 rule like the code does.

- **Constant and copy propagation were flow-insensitive.** A fact
  established on one path was reused on another that cannot reach it, which
  at `-O 3` miscompiled the most ordinary loop there is:

  ```plm
  n = 0;
  do while n < 3; call pc('0' + n); n = n + 1; end;
  ```

  `n = 0` was still in scope when the condition was folded, so `n < 3` became
  always-true, `'0' + n` became the literal `'0'` and `n = n + 1` became
  `n = 1`. The loop printed `0` for ever. The same flow-insensitivity reached
  four shapes in all: a `DO WHILE`, an iterative `DO`, a loop closed by a
  backward `GOTO`, and the arms of an `IF` or `DO CASE`, where a value
  assigned in one arm was folded into code after the join that the other arm
  reaches. A loop now drops whatever its body can assign before its condition
  is touched — everything, if the body can call out — a label is treated as
  the join point it is, and each branch arm is optimized from the state at
  the branch rather than from whatever the previous arm left behind.

  Costs nothing at the default `-O 2`: the output is byte-identical. `-O 3`
  grows, because much of what it used to fold away it had no right to.

- **Copy propagation duplicated a procedure call.** `k = rd;` recorded a copy
  of the identifier `rd`, so a later use of `k` was rewritten back into `rd`
  — a second call. In PL/M a parameterless procedure reference is a call, not
  a variable read.

- **A dead IF arm was discarded along with any label inside it.** A `GOTO`
  elsewhere in the procedure still named the label, so codegen emitted a jump
  to a symbol nothing defined and the program failed to assemble at `-O 2`
  while building and running correctly at `-O 0`. The arm is now kept when it
  declares a label, and `DO WHILE` got the same guard. Reachable before this
  release only for `IF 0`; the bit-0 constant rule widened it to every even
  constant, which is how it was found.

- **`ZERO`, `SIGN` and `PARITY` declared a BYTE result but left it in `HL`**,
  so every byte consumer read the wrong register: `IF ZERO` tested `A`, and
  `x = ZERO` overwrote `L` with `A`. They now produce their value in `A`, via
  `ld a,0ffh` / `jp cc` / `inc a` — which, unlike loading zero, cannot be
  strength-reduced into something that writes `CARRY` before a later read.

- **Algebraic identities discarded side-effecting operands.** `x AND 0`,
  `x * 0`, `x OR 0FFFFH`, `x - x` and `x XOR x` dropped an operand that
  PL/M-80 requires to be evaluated — and a bare identifier naming a procedure
  is a parameterless call, not a variable read. The identities now apply only
  when the discarded operand is side-effect free.

- **`IF CARRY` always read false at `-O 1` and above.** The built-in emitted
  `ld a,0` / `rla` — correct in itself, because `ld a,0` does not touch the
  flags — but the peephole rewrites `ld a,0` into the one-byte `xor a`, which
  CLEARS the carry the `rla` is there to read. It now uses `sbc a,a`, which
  reads carry in one instruction and does not depend on `A`. MP/M II's
  `scan$numeric` — shared by `SHOW.PLM`, `MSCHD.PLM` and `TOD.PLM` — guards
  its `b * 10` and `b + digit` steps with `IF CARRY THEN`, so every overflow
  check in those three was dead.

- **Constant folding removed the operation whose carry was about to be
  read.** At `-O 3`, constant propagation folded `s = a + b` to a literal, so
  the `add` that set carry no longer existed and the following `IF CARRY` read
  a stale flag. Arithmetic is no longer folded inside a procedure or module
  body that reads `CARRY`, `ZERO`, `SIGN` or `PARITY`; elsewhere folding is
  unchanged.

- **A BYTE assignment of a constant above 255 emitted a 16-bit load.**
  `_gen_assign` took its byte path only for values that already fit, so a
  folded `200 + 100` went out as `ld hl,012CH` / `ld a,l` — which the peephole
  then collapsed into `ld a,012CH`, keeping all sixteen bits. PL/M-80 narrows
  to the target's width, so the constant is truncated in the generator.

- **The truth-rule fix initially missed the CONSTANT paths**, so the same
  source got different answers at different optimisation levels: `IF NOT TRUE`
  with `TRUE LITERALLY '1'` folds to 0FEH and was true at `-O 0` but false
  once `_optimize_if` folded it, `IF 4` and `IF (4)` disagreed in one
  compilation unit, and `DO WHILE 2` was an infinite loop where it should
  never run. All four constant sites — two in the generator, two in the AST
  optimizer — now test bit 0.

- **Byte operands were generated with `_gen_expr`, which leaves a
  NumberLiteral in `HL`.** `IF 5 > X` compared an undefined `A`, and
  `SHL(DOUBLE(hi),8) OR 5` emitted `ld hl,5` straight over the high byte
  parked in `H`. Byte operands now go through `_gen_expr_to_a`. The mirror
  problem — code that widened a byte result into `HL` keyed off the
  *statically inferred* type rather than the type `_gen_expr` actually
  returned — is fixed by a new `_gen_expr_to_hl`, which also repairs the
  16-bit iterative `DO` bound and the `MOVE` count below.

- **A BYTE store to a structure member read the value out of `L`.** The
  member path always saved and reloaded `HL`, but a BYTE value is in `A`, so
  `rec.f = ch` stored whatever happened to be in `L`.

- **`CALL MOVE` with a non-constant BYTE count took `BC` from the source
  address.** The count was generated into `A` and then moved with
  `ld b,h / ld c,l`.

- **An iterative `DO` discarded its declaration list**, so a `PROCEDURE`
  declared at the head of one was registered and never emitted — the call
  site named a label nothing defined.

- **`_boolean_simplify` was a sibling path the first pass missed**: it still
  produced 0FFFFH for a relational, and its `x REL x` and idempotent
  `(a AND b) AND b` rules still discarded operands that PL/M-80 requires to
  be evaluated.

- **`INPUT`, `OUTPUT`, `MOVE`, `TIME`, `SCL`/`SCR` and the flag built-ins
  were treated as side-effect free**, so `r = INPUT(5) AND 0` deleted the
  `in a,(port)`. The purity test now names them explicitly rather than
  inferring purity from "not a user procedure".

- **The AST optimizer never reset its flow-sensitive state.** `constants`,
  `copies`, `cse_cache`, `expr_vars` and `modified_vars` accumulated across
  procedure boundaries and across the five optimisation passes, so at `-O 3`
  one procedure's constant was folded into another's body: with `ONE` setting
  `V = 1`, the unrelated `TWO: PROCEDURE; R = V; END TWO;` compiled to
  `ld hl,1` and never read `V`. The state is now cleared per procedure and
  per pass.

- **A BYTE procedure returning an ADDRESS expression normalised it to a
  boolean.** PL/M-80 narrows ADDRESS to BYTE by truncation, like `LOW()`, so
  `P: PROCEDURE BYTE; RETURN N + 1; END P;` with `N` = 64 must return 65. The
  generator emitted `ld a,l / or h / jp z / ld a,0ffh`, returning 0FFH for
  every non-zero value. Found while sweeping for remaining non-zero tests
  after the truth-rule fix.

- **`NOT` in a condition did not strip parentheses**, so `IF NOT (a = b)` never
  reached the optimised compare and materialised a value instead. Generated
  code was correct; it was one instruction pair longer than it needed to be.

### Known issues

- In `-m bare`, a module body that runs off its end falls into the first
  procedure emitted after it rather than stopping. This is long-standing and
  unchanged here. The only bare-mode program in the corpora, MP/M II's
  `MPMLDR`, ends its body in a `di` / `halt` loop, so the fall-through is
  unreachable; giving the mode an explicit terminator would change semantics
  the mode documents as the program's own business, so it is left alone.

- Two sibling `DO` blocks in one procedure that each declare a procedure of
  the same name collide on one assembly label. The assembler rejects that
  outright, so it cannot go unnoticed, and no PL/M-80 source in the CP/M or
  MP/M II corpora writes it.

### Added

- Regression tests for the fixes above, written as invariants over the
  generated assembly rather than golden output. All of them fail against the
  baseline generator, and nine fixes were reverted individually to confirm the
  suite catches each on its own. The per-fix property has not been verified
  exhaustively for every fix in the list.
- String-literal tests, which settles the debt `todo.txt` recorded against
  0.3.4: a backslash is an ordinary character in PL/M-80, `''` is one quote,
  and a string may cross a line break. `uplm80/_plm_parser.py` is generated, so
  a regen against a grammar that brought the C escape rule back would otherwise
  have gone unnoticed.

### Verified

- 80un: `80un.com` and `80unbas.com` produce byte-identical output on all 21
  samples of the test corpus and on the BASIC detokeniser. The generated code
  changes, so the committed `.COM` files need a rebuild to stay reproducible
  against this release.
- MP/M II: all 41 targets build, up from 40 of 41. Thirty binaries change and
  the total is size-neutral (`ED` +128 because it is built at all, `SDIR` +128,
  `STOPSPLR` -128, `TYPE` -128). The source-built utilities boot and run on a
  DRI system — `DIR` and `STAT` both produce correct output under the
  emulator.

  No claim is made about mpm2's `scripts/run_tests.sh` pass rate. That
  harness is not deterministic: five runs against one unchanged disk image
  fail at three different points, and measured over five runs each, the
  binaries from this release and from the previous generator score the same
  within the noise. An earlier draft of this entry credited a fix with
  repairing the `STAT` test; that was a single lucky run and is withdrawn.

### Known issues — not this compiler

- MP/M II `--tree=src` fails system generation with `XIOS common base BF4BH
  below configured common base C000H`. `XDOS.SPR` grew from 8960 to 10112
  bytes, and XDOS is assembled entirely from `.ASM` — the growth reproduces
  with every uplm80 version tested and disappears with the January `um80`/
  `ul80`, so it belongs to the assembler and linker.
- MP/M II built `--tree=src` reaches the console banner but not a command
  prompt. This reproduces with the January toolchain, so it predates all of the
  above.

## 0.3.4 — 2026-09-19

The third code-generation defect of the same family, which is the one that made
the CrLZH decoder of the 80un unpacker come out wrong. That known issue is
closed, and all three defects now have regression tests.

### Fixed

- **The base address of a subscripted element was destroyed by the index
  expression.** `_gen_subscript_addr` parked the array base in `DE` and then
  generated the index. An index that is itself a 16-bit expression emits
  `ld de,nn`, which overwrote the base, and the closing `add hl,de` then added
  the index constant a second time instead of the base. `PRNT(I + LZH$T) = I`
  with `LZH$T` = 629 stored to `(I + 629) * 2 + 629` rather than to
  `PRNT + (I + 629) * 2`, so every element address was wrong and the store
  scribbled over unrelated memory.

  An index of `I` or `I + 1` was unaffected, because neither needs `DE` - a
  `+ 1` becomes `inc hl`. Only an index carrying a constant too large for
  `inc`, or any other 16-bit subexpression, triggered the defect, which is why
  so little broke so specifically: in 80un only `init$tree` and `update$tree`
  index that way, and the result was a CrLZH Huffman parent table that was
  never initialised.

  The base is now kept on the stack across generation of the index, which is
  what the generator did before bdb0f8a migrated subscripts onto the register
  allocator. The same defect was present in both member-subscript paths and is
  fixed there too.

### Added

- Regression tests for all three defects fixed in 0.3.3 and 0.3.4, written as
  register-liveness invariants over the generated assembly rather than as
  golden output: a value parked in `DE` or `B` must not be destroyed before it
  is read, and the false path of a BYTE `>` must reach the `xor a` that loads
  zero. Each test fails against a copy of the generator with the corresponding
  fix reverted.

### Fixed known issue

- CrLZH decoding in 80un is correct again. Built with this release, 80un scores
  15 of 15 byte-exact against the original CP/M UNCR24.COM and reproduces its
  Python decoders on 107 of 108 sample-corpus members, which is parity with the
  last generator known to be good, 01cfcc6. 80un no longer needs to pin an old
  compiler to ship a correct binary.

## 0.3.3 — 2026-09-19

Two code-generation defects that silently produced wrong Z80, plus the PL/M
string-literal fix that comes with the raised `uplox` floor. Both code
generation defects are present in every earlier release; commit 273c83a made
them reachable rather than introducing them.

### Fixed

- **A BYTE `>` comparison used as a value was non-zero when false.** In
  `_gen_byte_comparison_const` and `_gen_byte_comparison` the `GT` arm jumped
  to the join label that follows `ld a,0ffh`, not to the `xor a` false case,
  which left the `xor a` unreachable and the compared operand sitting in the
  accumulator. `Y = X > 32` with `X = 6` assigned 6 instead of 0. The arm now
  branches to a real false label.
- **Register `B` was clobbered while the other operand of a byte `AND`, `OR`,
  `XOR` or `SUB` was generated.** `_gen_byte_binary` and
  `_gen_byte_comparison` parked one operand in `B` with `ld b,a` and then
  generated the other operand. `B` is not callee-saved, a nested byte
  comparison uses `ld b,a` as its own scratch move, and a procedure call in
  the other operand overwrites `B`, so the closing `and b` masked against
  garbage: `D = (A > 0) AND (B < C)` with `A = 0` came out true. The operand
  is now spilled through the stack, and `B` is loaded only once the other
  operand has been generated. `pop bc` would be shorter than `ld b,a` plus
  `pop af`, but `pop bc` also overwrites `C`, where the CP/M call convention
  keeps a live argument - `CALL MON1(2, '0' + X)` became a call to BDOS
  function 68.

  Both defects only reach the generated code when a comparison is
  materialised as a value. Commit 273c83a, which correctly made PL/M-80's
  `AND` and `OR` bitwise rather than short-circuit, routed every `IF` and
  `DO WHILE` condition through that path, so the two defects went from latent
  to load-bearing. Standalone programs demonstrate both under 0.3.1 as well.

### Changed

- `uplox` floor raised to `>=3.3.1`, and `uplm80/_plm_parser.py` regenerated
  against that grammar. A backslash inside a PL/M character literal - MBASIC's
  integer-divide token, for one - no longer fails with
  `lexical error at byte 0x5c`. PL/M-80 has no backslash escape.

### Known issues

- Code generated for the CrLZH decoder of the 80un CP/M unpacker is still
  wrong. A bisect puts the first failure at bdb0f8a "Implement register
  tracking phases 3-5", where `lzh$get$byte` parked its accumulator in `DE`
  across a call to `lzh$get$bit`; that particular defect is gone from the
  current generator, whose `LZHGETBYTE` matches the pre-bdb0f8a output
  instruction for instruction. A later, separate defect in the same area
  remains, somewhere in the plox front-end migration range, and the commits
  in that range cannot be built against a current `uplox` to narrow it
  further. Until the defect is found, 80un builds its released binary with
  uplm80 01cfcc6.

## 0.3.2 — 2026-08-20

No change to the compiler since 0.3.1. This release raises one dependency
floor and corrects two documentation items.

### Changed

- `uplox` floor raised to `>=3.3.0`, so a fresh install resolves the parser
  runtime that uplm80 is actually developed against rather than 3.2.0.
  3.3.0 adds the classifier lookahead window and named LR-state sets and
  fixes an IELR backward-propagation bug that could trip a table-build
  assertion.
- `sample_code/CPM_source/1,1/origin.txt` now points at
  `https://www.icl1900.co.uk/...` — z80pack's sources moved off
  `autometer.de`, and the recorded provenance URL no longer resolved.
- The Related Projects section of the README was rewritten in Simplified
  Technical English.

The `upeepz80` floor stays at `>=0.2.3`; nothing in that package changed.
