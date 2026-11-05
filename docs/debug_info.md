# Debug markers (-g)

`uplm80 -g` writes debug markers into the assembler text. A debugger uses the markers to find the PL/M-80 source line of each instruction. The design is in [debugger_plan.md](https://github.com/avwohl/cpmemu/blob/main/docs/debugger_plan.md) of cpmemu. The marker format is in `docs/debug_info.md` of um80_and_friends.

## Install

`-g` is in uplm80 0.5.0 and later. An older release has no `-g`. Install the compiler and the assembler tools from PyPI:

```
pip install "uplm80>=0.5.0" "um80>=0.4.0"
```

uplm80 0.5.0 requires upeepz80 0.2.8 or later, so pip installs the correct upeepz80. um80 0.4.0 has `um80` and `ul80` with `--debug-info`, and the debugger `udb80`. To run the program under a debugger, use `cpmemu --gdb` from cpmemu 4.11.0 or later. The cpmemu document `docs/gdb_stub.md` lists the parts for udb80 and for GDB.

A check on 2026-11-05 used only released packages: uplm80 0.5.0, um80 0.4.0 and upeepz80 0.2.8 from PyPI, and the cpmemu 4.11.0 macOS release. A REENTRANT recursive `FACT` was compiled with `-g -O2`. In `udb80`, `break fact.plm:5` stopped in the base case. `bt` showed 5 frames of `FACT` and the main program `DEMO`, and `p n` printed 0.

## Use

`uplm80` writes the `.mac` file only. You run `um80` and `ul80` yourself. Give both programs the option `--debug-info`.

```
uplm80 -g prog.plm -o prog.mac
um80 --debug-info prog.mac -o prog.rel
ul80 --debug-info prog.rel -o prog.com
```

`um80` writes `prog.dbm` next to `prog.rel`. `ul80` writes `prog.dbg` next to `prog.com`. The file `prog.dbg` lists the source line of each address, the routines and the labels. `udb80` reads the file.

The file names in the markers are the names that you give to `uplm80`. The names of the `$INCLUDE` files are relative to the directory where you run `uplm80`. Run `uplm80` in the directory of the source file, or give the debugger a source path.

## The code does not change

The `.com` file is the same, byte for byte, with and without `-g`, at each `-O` level. A marker is a whole-line comment. The assembler writes no byte for a comment. The `.mac` text is also the same when you remove the marker lines and the `??IXB` and `??IXE` labels.

Each marker is a whole line, never a trailing comment. The optimizer upeepz80 can remove a trailing comment.

At `-O1` and above, `-g` needs an upeepz80 that keeps the markers and gives the same code with them. upeepz80 0.2.8 is the first such release (the change of commit ab4de24). uplm80 0.5.0 requires upeepz80 0.2.8 or later.

The upeepz80 of 0.2.7 lets a marker hide the dead store at the start of a procedure. A `-g` build then has a store that the build without `-g` does not have. `uplm80 -g` tests the installed upeepz80 on a small text. `uplm80 -g` refuses an upeepz80 that fails the test. `-O0` does not use upeepz80.

The tests in `tests/test_debug_info.py` compile the programs of `tests/` with and without `-g` at the four `-O` levels. The tests compare the `.com` files.

The tests that run `um80` and `ul80` skip when the tools are not installed. A skip gives exit code 0. Set `UPLM80_REQUIRE_TOOLS=1` and the tests fail in place of the skip. The workflow `pytest.yml` does not install `um80`, so the tests skip there.

## What uplm80 writes

- `;#file N "PATH"` after the `cseg` line, one for each source file. An `$INCLUDE` file gets its own number.
- `;#func NAME F L frame=KIND pops=N` after the label of each procedure, and `;#endfunc NAME` after the last line of the procedure. A procedure with the attribute REENTRANT has `frame=ix`. Any other procedure has `frame=static`, because its locals are at fixed addresses. `pops` is the number of bytes that the procedure takes off the stack for its arguments. The first two arguments come in BC and DE, so `pops` is 2 for each argument after the second. NAME is the label of the procedure. A nested procedure has the name `@OUTER$INNER`.
- `ixbegin=LABEL ixend=LABEL` at the end of the `;#func` of a REENTRANT procedure. `ixbegin` is the label after the `add ix,sp` that sets the frame. `ixend` is the label in front of the last `pop ix` of the procedure. The labels have the names `??IXB1`, `??IXE1`, `??IXB2` and so on. uplm80 writes the labels after upeepz80 has run, so a label does not stop the peephole optimizer. A label is not code, so the `.com` file is the same.
- `;#func NAME F L frame=none pops=0` for the statements of the main program, which have no label. NAME is the name of the module. The routine starts at the stack set-up and ends after the `jp 0` of the CP/M mode.
- `;#line F L` in front of each statement. F and L are the file and the line where the statement starts in the source. For text from an `$INCLUDE` file, the file is the included file and the line is the line in that file. For text that a LITERALLY made, the line is the line of the name that the text replaced.
- The line of the procedure declaration, in front of the prologue. The line of the `END` of the procedure, in front of the implicit `RET` at the end.
- A second `;#line` with the line of the `DO`. The second `;#line` stands in front of the step of an iterative `DO` and in front of the jump back of a `DO WHILE`. The loop control then has the line of the `DO`, and not the line of the last statement of the body.
- `;#line 0 0` in front of the jump table of a `DO CASE` with more than eight cases. The table has no source line.
- `;#inline F L NAME` and `;#endinline` around the code of a procedure that `-O3` put into the caller. F and L are the position of the `CALL`. The `;#line` markers inside give the lines of the procedure body.

## Variables

`uplm80 -g` also writes a `;#var` marker for each variable and a `;#type` marker for each structure and for the two base types. The format is in `docs/debug_info.md` of um80_and_friends (Variables). `udb80` reads the records. `print`, `info locals`, `info args` and `frame 1` of `udb80` use the records. The compiler writes the records after the last text pass, so the code does not change.

```
;#type typedef:BYTE target=ushort:1
;#type typedef:ADDRESS target=uint:2
;#type struct:pt size=5 members=x:typedef:ADDRESS@0,y:typedef:ADDRESS@2,tag:typedef:BYTE@4
;#var pt struct:pt kind=static at=PT decl=1:8
;#func LEAF 1 18 frame=static pops=0
;#var p typedef:ADDRESS kind=param at=??AUTO+11 range=??VR1:??VR2 scope=1:18-25 decl=1:18
;#var t typedef:ADDRESS kind=local at=??AUTO+15 range=??VR3:??VR2 scope=1:18-25 decl=1:20
;#var keep typedef:BYTE kind=local at=@LEAF$KEEP scope=1:18-25 decl=1:21
```

The `;#type` markers and the markers of module-level variables stand after the `;#file` lines. The markers of a routine stand after its `;#func` marker. A routine that has no `;#func` marker takes its records away.

### Types

- `BYTE` is `typedef:BYTE`, with the target `ushort:1`. The base name is `ushort` and not `uchar`, so `udb80` shows a number, and shows an array of BYTE as a list of numbers. A `uchar` array would show as a C string, which ends at the first zero byte. `ptype` therefore shows `unsigned short` for a BYTE of one byte.
- `ADDRESS` is `typedef:ADDRESS`, with the target `uint:2`.
- An array is `[N]T`. N is the dimension of the declaration. `DECLARE X(4) BYTE` has four elements.
- A STRUCTURE is `struct:TAG`. The tag is the name of the variable. A second structure with the same tag and another layout gets the tag `NAME.2`. The members have byte offsets and no padding. A member can be an array.
- A BASED variable is a pointer to its type, at the place of its base. See [BASED variables](#based-variables).
- LITERALLY names, LABEL names, procedures, `MEMORY` and `STACKPTR` are not variables. An EXTERNAL variable has no record, because the PUBLIC record of the module that defines the variable names the place.

### Where each variable lives

- **Module level.** `at=LABEL`. A PUBLIC variable has `kind=global`. Another one has `kind=static`. The second and later names of a factored `INITIAL` or `AT` declaration are EQU symbols, and `um80` resolves them.
- **Static procedure.** A local with INITIAL or PUBLIC, or a local that has its address taken, has its own label (`at=@PROC$NAME`). Another local, and each parameter, has a slot in the shared area (`at=??AUTO+N`). The slot is correct while the procedure is active, because `local_storage.py` shares the area only between procedures that are never active together. The record is in the routine of the owner.
- **REENTRANT procedure.** `ix=N`. The first parameter is at the highest offset. All parameters take two bytes on the stack, a BYTE too. A local is at a negative offset. An array or structure comes after the scalars.
- **Main program.** A variable of a DO block has `kind=local` and the label `@B1$NAME`. The scope is the block. The records are in the routine of the module.
- **Nested procedure.** The records of its own variables are in the routine `@OUTER$INNER`. The routine also has a copy of each static variable of the procedures around it, as `kind=local`, with no `range=`. A copy of the same name as an own variable, or as a variable of a nearer procedure, is not written. A variable in the IX frame of the outer procedure has no copy.

A slot of `??AUTO` that no instruction of the routine names gets `none why=no-storage`. The text of the procedures nested in the routine counts as text of the routine. A routine with `frame=ix` and no `ixbegin` has lost its frame. Each `ix=` variable of the routine then gets `none why=no-frame`.

### Parameters and locals

The first instructions of the routine put each argument in the slot of the parameter. The record has `range=` from the instruction after the first store to the end of the routine. The compiler writes the two labels (`??VRn`) after the last text pass.

A local of a static procedure has the same `range=`, from its first store to the slot. The slot of `??AUTO` holds the bytes of other routines before that store. These have no `range=`, because they hold a value from the start: a local with INITIAL, DATA, AT or PUBLIC, a BASED variable, and a variable of the main program. An array or structure local has a `range=` only when an instruction stores to its first byte by name. A variable that no instruction stores to by name has no `range=`.

The optimizer can move a store into the first statement. A breakpoint on the first line then stops in front of the store. `udb80` shows `<not available>` for the parameter or local. Before the change, a local showed the old bytes of the slot in this case (`0` at `-O1` to `-O3` in a measured program). A REENTRANT procedure has no `range=` for a parameter, because the IX range of the routine applies.

### BASED variables

The format has no place that means "the address in another variable". A BASED variable is therefore a pointer: the record has the type `*T` and the place of the base. `print bb` shows the pointer, and `print *bb` shows the value. `DECLARE BB BASED BP BYTE` gives `;#var bb *typedef:BYTE kind=static at=BP`.

A variable BASED on a structure member, `BASED CTL.PTR`, has the place `at=CTL+2`. The debugger shows `bb` as a pointer, where the program sees a BYTE.

### Names

The compiler writes the name as the source spells it, without the `$`. `Print$Char` is `PrintChar`. `udb80` has no `$` in names and keeps the case. The compiler writes the upper-case name when the source line has no such name at the column of the token. A multi-file compile qualifies a module-level name (`LIB?NAME`), and the record has the plain name. The label in `at=` is the qualified one.

### Scope

`scope=` is the block of the declaration: the procedure, or the DO block. `decl=` is the line of the declaration. A copy of an outer variable in a nested routine keeps the scope of the outer variable. `udb80` therefore hides the copy of a DO block variable on the lines outside the block.

### A fault in the records

A fault in the variable code gives the warning `no variable records for -g`. The build then goes on with no `;#var` and no `;#type` marker. The code is the same. `UPLM80_DEBUG_VARS_STRICT=1` raises the fault.

## Limits

- A declaration with no code, a null statement and a `DECLARE` have no marker. Their line has no code.
- The optimizer folds constants and removes dead stores before code generation. A statement that the optimizer removed has no line. A debugger that sets a breakpoint on such a line uses the next line with code.
- `nostmt` is never written. The `-O2` and `-O3` optimizer keeps expressions of loops for common subexpression elimination, but it moves no statement out of a loop.
- A REENTRANT procedure with more than one exit has one range for IX. The `ret` after an earlier `pop ix` is inside the range, and IX there is not the frame.
- The jump over the `ELSE` part of an `IF` has the line of the last statement of the `THEN` part.
- The runtime routines (`??mul16`, `??div16` and the others) have no markers. `um80` gives each of their instructions a line of the `.mac` file (assembler level only).
- A call that the peephole optimizer made into a jump (`call x` and `ret` to `jp x`) removes one frame from a backtrace.
- A marker in a multi-file compile (`uplm80 a.plm b.plm`) names the file of each statement. The `.mac` file has one file table.
- The `outer` field is not written.
- A parameter or local of a static procedure is not available before the first store. See [Parameters and locals](#parameters-and-locals).
- A structure member from a LITERALLY has the upper-case name (8 of 1630 names). `udb80` finds names with the case as written. A case-insensitive lookup in `udb80` would remove the need to keep the case of the source.
- `;#func` of a nested or multi-module procedure has the assembler label (`@OUTER$INNER`, `A?F`), and `bt` shows that label.

## Measurements

Measured on 2026-11-04 with um80_and_friends 92196ca and 8c828b3, and upeepz80 0.2.7 plus the change of commit ab4de24.

- 68 programs came from `tests/*.plm`, `sample_code`, `examples` and one program of romwbw_emu. Each program was built with and without `-g` at the four `-O` levels. The builds total 272. In all 272 builds the `.com` file is the same. For 3 programs that need an external data symbol, the `.rel` file is the same. 14 of the 68 programs do not compile, with or without `-g`.
- With the upeepz80 of the tag v0.2.7, the `.mac` text without the markers differs in 15 of 156 compiles. Five programs at `-O1` to `-O3` are the cause. `uplm80 -g` refuses that upeepz80.
- 51 programs link without an undefined symbol. The routines of these programs hold 46172 instruction bytes at `-O2`. All 46172 bytes have a source line in `prog.dbg`. The share is 100.00 percent at `-O0` (50171 bytes), `-O1` (46220 bytes), `-O2` and `-O3` (45356 bytes).
- Outside the runtime, 46172 of 46182 instruction bytes have a source line at `-O2`. The 10 other bytes are the entry code (`ld hl,(6)`, `ld sp,hl`, `call` and `jp 0`) of `tests/test_array_bug.plm`, the one program with no statements in its main program.
- `tests/test_reentrant.plm` has REENTRANT procedures. All 909 instruction bytes of its routines have a source line at `-O2`.
- Variables, measured on 2026-11-04 at `-O2`: 55 programs compile and give 2848 `;#var` lines. 999 lines are in `??AUTO`. 1471 lines are at the label of a procedure or block variable. 331 lines are at a module label. 15 lines are in an IX frame, and 32 lines have no place. The 50 programs that link give 1416 rows in `prog.dbg`, and 28 of the rows have no place (all `no-storage`).
- The `.com` file is the same with and without `-g` for 55 programs at the four `-O` levels (220 builds). `um80 --debug-info` and `ul80 --debug-info` gave no warning.
- `udb80` against `cpmemu --gdb` printed each storage kind at the known values at `-O0` to `-O3`. `tests/test_debug_vars.py` has the session.
