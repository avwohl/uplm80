# Testing against Intel's PL/M-80

`scripts/intel_oracle.py` is a differential test with Intel's own compiler as
the oracle.  It builds a program - a CP/M main program that prints through
MON1/MON2 - with Intel's PL/M-80 V3.1 the way Digital Research's `P.SUB`
built its CP/M programs:

```
PLM80 T.PLM DEBUG PAGEWIDTH(80)
LINK T.OBJ,X0100,PLM80.LIB TO T.MOD
LOCATE T.MOD CODE(0100H) STACKSIZE(1024)
OBJCPM T
```

and with uplm80 at `-O0` to `-O3` (linked with the same `X0100` equates),
runs every build under cpmemu with a time limit, and compares what each
prints with what Intel's build prints.  Each program is `same`, `differs`
(both outputs are shown, and where they first part), `intel-rejects`,
`uplm80-rejects` or `timeout`.

```bash
make -C tools/isis                                   # the ISIS-II emulator, once
python3.14 scripts/intel_oracle.py prog.plm          # programs of your own
python3.14 scripts/intel_oracle.py --random 300      # generated programs
python3.14 scripts/intel_oracle.py --corpus          # tests/ and sample_code/
python3.14 scripts/intel_oracle.py --random 50 --first 1000 --reduce --keep out/
python3.14 -m pytest tests/test_intel_oracle.py
```

`tests/test_intel_oracle.py` is part of the suite: `python3.14 -m pytest`
runs it with the rest where Intel's binaries are found (building
`tools/isis` first if it can), and skips it where they are not, as on CI,
building nothing; the oracle's own pieces that need no tools - the source
normalisation, the halt patch, the skip - are checked either way.  It
checks three generated programs and one of `tests/`, that a known
difference is seen and a V3.1 rejection is, that V3.1's builds of the
programs of the two V3.1 bugs 0.4.3's checks found print what they did
(and, needing no tools, that the table below has them), each program of
a release's tests whose output the release transcribed from Intel's
build, and, for each program `tests/test_names.py` expects uplm80 to
reject or warn of as V3.1 rejects it, that V3.1 does, with the errors
uplm80's message names and no other.

Intel's binaries are not part of this repository.  The oracle finds them
through `--tools DIR` or `$PLM80_TOOLS`, or on DRI's MP/M II work disk at
`~/src/mpm2/mpm2_external/mpm2src/PLM_WORK` (`PLM80`, `PLM80.OV0`-`OV4`,
`PLM80.LIB`, `LINK`, `LINK.OVL`, `LOCATE`, `X0100`, `OBJCPM.COM`), and when
they are missing it says so and checks nothing, as the pytest skips.  It runs
them on `tools/isis/isis`, a small ISIS-II emulator (`tools/isis/isis.cc`)
on cpmemu's qkz80 core, which `make -C tools/isis` finds the way romwbw_emu
does (pkg-config, else a `../cpmemu` checkout beside this one); without it,
on romwbw_emu's `tools/romwbw-plm80`, which runs the same recipe under DRI's
ISX, some ten times slower.  `OBJCPM` is a CP/M program and runs on cpmemu.

Two things the recipe needs that DRI's own programs did for themselves:

- LOCATE starts a program behind its constants (DATA, and strings in
  `.(...)`), not at 0100H, where CP/M enters it; DRI's programs begin with a
  DATA jump.  A program that does not start at 0100H is located again at
  0103H, where OBJCPM puts a jump to the start in front of it.
- A PL/M-80 main program ends in `EI; HLT`, which cpmemu runs past.  The
  HLT at the module's last statement (found through the LINES records
  `DEBUG` writes) becomes `RST 0`, a warm boot, as uplm80's CP/M mode ends;
  where a label on the module's END is one a procedure's GOTO reaches,
  `fin: end t;`, the `EI; HLT` follows the `LXI SP` V3.1 sets SP again
  with there.

A module with no statements of its own, entered through a DATA jump that
Intel's layout puts at 0100H - DRI's CP/M 2.0 `LOAD`, `STAT` and `SUBMIT` -
is built with uplm80's `-m bare`, which lays it out the same way (`--mode`
chooses otherwise).

`--normalize` rewrites three spellings only uplm80 takes into their Intel
equivalents before both compilers see the program: `.'string'` into
`.('string')`, an untyped `DATA` into `BYTE DATA`, and a program with no
`name: DO;` into a module.  `--reduce` cuts a program that differs, or that
uplm80 rejects, down to the statements, then the lines, the difference needs
(Zeller's ddmin); the result needs tidying by hand before it is a test.

The generator, `tests/plm_intel.py`, writes programs in the dialect both
compilers share, with no behaviour PL/M-80 leaves undefined but two, which
V3.1 defines and uplm80 follows: division by zero (what Intel's divide
routine returns) and the carry of PLUS, MINUS, SCL and SCR in the form where
the statement itself sets it.  It covers BYTE and ADDRESS arithmetic,
relations and logic, the built-ins, arrays, structures, BASED variables,
DATA/INITIAL, LITERALLY, strings, MOVE, LENGTH/LAST/SIZE, IF, DO CASE, DO
WHILE, iterative DO (with limits and steps the body changes), nested
procedures with 0 to 5 parameters, typed and untyped, and REENTRANT
recursion.  Evaluation order is kept out of what a program prints.  Each printed line starts with the number of the statement
that printed it, which the source carries as `/* S1F */`.

## Known differences from Intel's PL/M-80 V3.1

The generator can leave out those with a name (`--avoid NAME,...`), and the
pytest does.  In the first campaign - 1,200 generated programs and the 67
programs of `tests/` and `sample_code/`, at `-O0` to `-O3` - every difference
was one of these, and none depended on the optimization level.  Two more
are gone: LENGTH, LAST and SIZE of a qualified reference, which uplm80
rejected, it takes since 0.4.2 (`qualsize` no longer leaves them out); and
SHL and SHR of a BYTE, which uplm80 shifted in 16 bits, with an ADDRESS
result, are a BYTE since 0.4.3, as the manual (11.1.4) and V3.1 make them
(`shl-byte` is still taken, and leaves out nothing).

| `--avoid` | Program | Intel V3.1 | uplm80 | |
|---|---|---|---|---|
| `shift9` | `b = 7; r = SHL(b, 9);` (r BYTE) | `0EH` | `0` | V3.1 shifts a BYTE by the count mod 8 (0 counts as 8), folded or not; the manual: 0 |
| `wide-limit` | `DO i = 250 TO 300;` (i BYTE) | 6 times | never | V3.1 compares the BYTE index with the 16-bit limit; the manual (5.1.4) converts the limit to the index's type |
| `sub-zero` | `w = (b - DOUBLE(0)) + 0F0H;` (b = 0F0H) | `00E0H` | `01E0H` | V3.1 drops `- 0` and with it the ADDRESS type (also `b - (c * 0)`); the manual (4.2.1): ADDRESS |
| `zero-dividend` | `z = 0; w = 0 / z;` | `0` | `0FFFFH` | V3.1 folds `0 / x` to 0, a dividend that overflows to 0 too, `256 * 256 / z`, as it folds a product with 0, `(b * 0) / z`, and a dividend that folds to 0 by what folds to 0 too, `(8 / 0FF00H) / (0F82AH <= 1)`; uplm80 divides, and a division by 0 gives what Intel's own divide routine gives (4.2.3: undefined) |
| `neg-widened` | `b = 0E7H; w = 0FFFEH; v = (b - w) + (-b);` | `0002` | `0102H` | V3.1 negates `b` in 16 bits when `b - w` has already widened it (the manual: `-b` is a BYTE, 19H) |
| | `w = 1; v = HIGH(0FH + w) >= (ROR(SIZE(aw), 2) AND w);` (aw(8) ADDRESS) | `0` | `0FFH` | V3.1 makes 10H from the 0FH in C with `MOV A,C` and `INX SP` (its listing: `INX PSW`) where it means `INR A`: the value is one short and SP one long.  `w1, b2 = b2;` (b1-b4 BYTE, w1-w4 ADDRESS), as a program's first statement or later, is coded `INX H; INX SP; MOV M,A`, and a later CALL overwrites `b1` (the random campaign's seed 233).  The decrement form is the same: `MOV A,L; DCX SP` (`DCX PSW`) where it means `DCR A`, in `DO CASE ((-(LAST(sa.z))) / ROL(0FFFFH, 9)) MOD 5;` (seed 20226), and `INX SP` again in `DO CASE (ROR((07H AND 08001H), 1)) AND 1;` (0.4.3's seed 50265) |
| | `b = 0ECH; w = (ew := b) + b;` (ew, w ADDRESS) | `01D8H` | `00D8H` | V3.1 widens b to store it in ew and adds that 16-bit copy to itself, `SHLD EW; MOV D,H; MOV E,L; DAD D` (0.4.3's campaign, seed 50252); the manual (4.6.3): the embedded assignment is its right half, a BYTE, and BYTE + BYTE is a BYTE |
| | `w2 = 0FFFEH; w2, w3 = (ew := w2);` (w2, w3, ew ADDRESS) | w3 `0000` | w3 `0FFFEH` | Where a multiple assignment's value is an embedded assignment of one of its targets but the last, V3.1 takes that target's value for an address, `LHLD W2; SHLD EW; MOV D,H; MOV E,L; XCHG` and 52 `INX H`: it copies the word there, at 0032H, onto itself and into each target after w2, and does not store w2.  With a BYTE, `w2, w3 = (eb := w2);`, it stores that address, 0032H, at itself and in w3.  How far from the value it goes changes with the program: 54 bytes on in another, 9 back in 0.4.3's release check's seed 80353, which stored 0FFF5H.  `w3, w2 = (ew := w2);` is right; the manual: the embedded assignment is w2 (4.6.3), which each target gets |
| | `CALL MOVE(0, .s, .d);` | moves 65536 bytes | moves none | Intel's MOVE counts down before it tests |
| | `b = SHL(k, 1); c = d OR 0; e = CARRY;` (k = 0FFH, d = 1) | `0FFH` | `0` | V3.1 compiles no instruction for `x OR 0`, `x XOR 0`, `x + 0`, `x - 0`, `x AND 0FFH` and an operation of constants, `DEC(34H + 21H)`, and a flag read after one reads the flags of what came before; uplm80 computes each, as its `-O0` build always did and every level does since 0.4.4.  And the control code of a loop leaves other flags: `DO i = 1 TO 2; c = 5; END; e = CARRY;` is 0FFH with V3.1 and 0 with uplm80, `DO WHILE z; ...; END; e = ZERO;` of z = 0 the other way round.  Other operations leave other flags with V3.1 than with uplm80 too, as in 0.4.2 and 0.4.3: `x + 1` and `x - 1` of a BYTE are `INR` and `DCR`, which set no carry, `c = d + 1; e = CARRY;` after the SHL 0FFH with V3.1 and 0 with uplm80; a SHR of a BYTE, and a SHL V3.1 makes a rotation, is an `ANI` and then `RAR`s, so that SIGN, ZERO and PARITY after it are of the masked value before the shift, `b = SHR(k, 3); e = SIGN;` 0FFH with V3.1 and 0 with uplm80; and after some comparisons and IF tests, a MOVE (V3.1's leaves ZERO set), TIME, a DO CASE's dispatch, subscript arithmetic, `x * 0`, `x / 1` and NOT of a comparison.  Of 275 cases of 0.4.4's checks where a reader reads the flags of a shift, 256 print what V3.1's build prints and 19 are of these kinds.  The manual: the flags are not to be relied on (12.1) |
| | `rw = ((08B1FH - b3) XOR (-((b4 >= ms(3)) AND 0))) + 0;` (b3 = 20H, b4 = 0, `ms DATA('x=1; y=2$')`) | `0` | `8AFFH` | V3.1's count of what it has on the stack goes below zero (its listing counts 0, 255, 254, ...) and it pops what it never pushed, a run of `POP PSW` (seed 20090); 8AFFH is the manual's value |

V3.1 also rejects what only uplm80 takes: `.'string'` (ERROR 101), an
untyped `DATA` (61), a program that is not a module (89), a declaration
after a statement or among a DO CASE's cases (26; uplm80 refuses a DO CASE
of declarations alone as a DO CASE with no case, 201), `NOT NOT x` (102),
and a `CALL` of a typed procedure the program declares (129); and what the
CHANGELOG lists under Known issues: more `INITIAL` or `DATA` values than a
scalar or an array holds, or an empty string in a DATA list (209), an AT
that names a variable AT something further down (213), a typed procedure
with no RETURN (156), a label as a value or an assignment's target, `w =
lb` (132), a number above 0FFFFH (94), a port of INPUT or OUTPUT that is
not a constant, `INPUT(b)` (107), a REENTRANT EXTERNAL procedure (41,
174), and `.memory.x`, which uplm80 compiles to a MEMORY symbol that um80
finds undefined (32, 110).  uplm80
compiles, with a warning that names V3.1's error, `f()` and `CALL g()` of
a procedure (102, 153) and `INITIAL` in a procedure or a DO block (73), on
which programs written for it rely, and a subscript on a scalar, `x(1)`
(127), but for the location of one in a restricted expression, and a
member of an array of structures without its subscript, `s2.m(1)` (133),
which its own tests test.  It rejects, as V3.1 does, a name declared
nowhere, empty parentheses after a variable or a built-in, `.label` in an
expression (158), an `INTERRUPT` procedure below module level, a parameter no
`DECLARE` declares, and a `LITERALLY` used before its declaration (since
0.4.1); a dimension of 0, the address of a built-in but MEMORY, and a
PUBLIC or EXTERNAL procedure or variable below module level (since
0.4.2); and an `END` that names another block (20), a DO CASE with no
case (201), `.p(1)` of a procedure (104), anything in parentheses in a
subscript of the argument of SIZE, LENGTH or LAST, `size(ab(f(1)))` (32),
a procedure with no statements (174), two subscripts on a scalar (127,
114), an array or a member array without a subscript (133, 134), and a
call of a procedure declared further on (169) (since 0.4.3); and more
than one subscript (114), a subscript on a scalar member (127) or after a
subscript (32), a second port of INPUT or OUTPUT (108), a CALL through
what is not an ADDRESS scalar (118) or of a built-in with a type, `CALL
STACKPTR` (129), a built-in with too few or too many arguments (154, 153,
124, 126) and INPUT or OUTPUT without a port (109), MEMORY without a
subscript (133), a non-REENTRANT procedure's call of itself (170), a
procedure nested in a REENTRANT one or a REENTRANT one nested in another
(88, 39, and 174 of one with no statements), an `END` that names the
first of two labels on a DO, `a: c: do; ... end a;` (20), and an
assignment to a built-in or a procedure, `INPUT(1) = b` (128, 131 of one
without a type, and 154, 153, 109, 108, 124 or 126 where its arguments
are not as many as it takes, `TIME = 1`) (since 0.4.4); and in
a DATA or INITIAL list, an AT address or a constant list what a
restricted expression does not take, with the errors V3.1 gives for the
list (since 0.4.4): a built-in or a name, `data (shl(0f0h, 4))`, `data
(x)`, `.(x, 7)` (151); parentheses, `.((1 + 2), 7)`, an operator but + and
-, `.(2 * 3)`, `data (6 / 3)`, NOT, `.(not 0f0h)`, a string in a sum,
`data ('A' + 1)`, and a location anywhere but first, `at (3 + .buf(1))`
(152, 146, 151); a location with two subscripts, `data (.a(1)(1))` (152,
146), with more than one in its parentheses, `data (.a(1, 2))` (150), or
with a subscript on what is not an array, `data (.x(1))`, `data (.p(1))`
(149); the location in an AT of a procedure or a label, `at
(.p)` (211), or of a BASED variable (212); a constant list in a DATA
list, `data (.(5))` (147); and in a constant list, or where the value
fills a BYTE, a location or a number a byte does not hold, `.(300, 7)`,
`byte data (.w)` (210); and a base V3.1 does not take (since 0.4.4):
BASED, or a member of what is, `a based s.p` of a BASED `s` (50, 52), a
member of an array of structures (52), a BYTE, an array, a structure, a
built-in, a procedure or a label (50), a member its structure does not
have (55) or a name declared only further down (54); and a LABEL that
labels no statement (172, and 105 where the program names it, and 118
where a CALL calls it).  It refuses the location of a built-in but MEMORY
in a DATA or INITIAL list, `data (.stackptr)`, which V3.1 takes for an
address of its own: uplm80 has none to give it; and the location of a
REENTRANT procedure's local in a DATA or INITIAL list or an AT, and of a
BASED variable, a factored BASED declaration's too, in a DATA or INITIAL
list, which V3.1 gives an address (INITIAL in a procedure it rejects, 73):
uplm80 has the local on the stack, and no address of a BASED variable to
give.  And V3.1 computes
a restricted expression as a constant, a number below 256 a BYTE and BYTE
arithmetic in eight bits, `address data (200 + 100)` 002CH and `-1`
00FFH, where uplm80 computes in sixteen bits, 012CH and 0FFFFH, and a
string that fills an ADDRESS there as a constant, `address data ('AB')`
4142H, where uplm80 lays out 'A' and 'B' in order, 4241H (CHANGELOG,
Known issues).

A store through a pointer or an overrun in or from `??AUTO` reaches what
uplm80's layout puts there, not what DRI's does (CHANGELOG, Known issues),
which the generator never writes.  Since 0.4.2 uplm80 compiles a label on
an `END` statement, `out: end p;` (A.4.4.1), as V3.1 does, and a counted
loop over the last module-level variable sees a store through MEMORY that
reaches it.
