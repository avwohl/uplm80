# Intel PL/M-80 V3.1 fixtures

What Intel's own compiler makes of a module, so that the tests can check
uplm80's calling convention against it (tests/test_calling_convention_run.py).

- `T2.PLM` is a small module written for the purpose: procedures with two to
  five parameters, in every mix of BYTE and ADDRESS, and a `MAIN` that calls
  each of them.
- `T2.LST` is Intel's listing of it, unedited: ISIS-II PL/M-80 V3.1 (`PLM80
  T2.PLM CODE`), the compiler Digital Research built MP/M II with, taken from
  the MP/M II source distribution's `PLM_WORK` directory and run under an
  ISIS-II emulator. The same compiler and emulator rebuild DRI's MP/M II
  utilities (DIR, STAT, PIP, ED, GENSYS and the others) byte for byte.
- `t2_procs.asm` is the listing's code for `P2BB` to `P5`, and `t2_main.asm`
  its code for `MAIN`, copied by hand, instruction for instruction, in 8080
  mnemonics that um80 assembles. Only the addresses are symbolic. The
  procedures' parameters are declared in the order PL/M-80 laid them out,
  one after another, because each entry stores them from the last down with
  `DCX H`.

ul80 cannot read Intel's object format (OMF-80), so the tests link these
transcriptions instead: Intel's `MAIN` with the procedures uplm80 compiles
from `T2.PLM`, and uplm80's `MAIN` with Intel's procedures.
