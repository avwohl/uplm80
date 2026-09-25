"""A few of tests/abi_fuzz.py's random programs of two PL/M modules compiled
apart and one of assembly, calling one another by PL/M-80's convention;
`scripts/abifuzz.py --seeds N' runs more.

Each is built at -O0, which must leave SP where it found it, and at -O1 to
-O3, with the modules at -O3 and -O0 each way round and in one multi-file
compile, and every build must print what -O0's prints.  With `call x / ret'
made `jp x', as upeepz80 0.2.5 made it, every one of these seeds fails:
the callee takes its caller's return address for the first word pushed.
"""

import re

import pytest

from uplm80 import compiler as C

from ._toolchain import tools_missing
from .abi_fuzz import check


@pytest.mark.parametrize("seed", (1, 5, 11))
def test_modules_calling_one_another(seed):
    """Seed 1: 6 procedures, three of them through the assembly, one
    REENTRANT, three with five or six parameters, calls through an address;
    5: a private procedure of one parameter among them; 11: 11 procedures,
    three REENTRANT."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    bad = check(seed)
    assert not bad, bad


def test_a_jump_to_a_procedure_with_words_pushed_is_caught(monkeypatch):
    """The test is not blind to what upeepz80 0.2.5 did: with every `call x
    / ret' after upeepz80 made `jp x', seed 5's -O1 build prints two wrong
    words of its seven and stops."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    real = C.PeepholeOptimizer

    class TailCalling(real):  # pylint: disable=too-few-public-methods
        """upeepz80, and then 0.2.5's tail calls."""
        def optimize(self, asm_text: str) -> str:
            return re.sub(r"^(\s*)call(\s+\S+)\n\s*ret\b", r"\1jp\2",
                          super().optimize(asm_text), flags=re.MULTILINE)

    monkeypatch.setattr(C, "PeepholeOptimizer", TailCalling)
    bad = check(5, levels=(0, 1), mixed=False, together=False)
    assert len(bad) == 1 and bad[0].startswith("-O1: printed"), bad
