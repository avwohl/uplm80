# Debug markers (-g)

`uplm80 -g` writes debug markers into the assembler text. A debugger uses the markers to find the PL/M-80 source line of each instruction. The design is in [debugger_plan.md](https://github.com/avwohl/cpmemu/blob/main/docs/debugger_plan.md) of cpmemu. The marker format is in `docs/debug_info.md` of um80_and_friends.

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

At `-O1` and above, `-g` needs an upeepz80 that keeps the markers and gives the same code with them. Use a version newer than 0.2.7 that has the change of commit ab4de24.

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

## Measurements

Measured on 2026-11-04 with um80_and_friends 92196ca and 8c828b3, and upeepz80 0.2.7 plus the change of commit ab4de24.

- 68 programs came from `tests/*.plm`, `sample_code`, `examples` and one program of romwbw_emu. Each program was built with and without `-g` at the four `-O` levels. The builds total 272. In all 272 builds the `.com` file is the same. For 3 programs that need an external data symbol, the `.rel` file is the same. 14 of the 68 programs do not compile, with or without `-g`.
- With the upeepz80 of the tag v0.2.7, the `.mac` text without the markers differs in 15 of 156 compiles. Five programs at `-O1` to `-O3` are the cause. `uplm80 -g` refuses that upeepz80.
- 51 programs link without an undefined symbol. The routines of these programs hold 46172 instruction bytes at `-O2`. All 46172 bytes have a source line in `prog.dbg`. The share is 100.00 percent at `-O0` (50171 bytes), `-O1` (46220 bytes), `-O2` and `-O3` (45356 bytes).
- Outside the runtime, 46172 of 46182 instruction bytes have a source line at `-O2`. The 10 other bytes are the entry code (`ld hl,(6)`, `ld sp,hl`, `call` and `jp 0`) of `tests/test_array_bug.plm`, the one program with no statements in its main program.
- `tests/test_reentrant.plm` has REENTRANT procedures. All 909 instruction bytes of its routines have a source line at `-O2`.
