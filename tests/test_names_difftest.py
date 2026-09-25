"""A few of tests/names_difftest.py's random programs of reused names, at
-O0 to -O3; `scripts/namestest.py --seeds N' runs more.  None of these
assembled with 86d2720 ("multiply defined", "Register 'A' used as
value", "JP with condition requires address"); seed 302 and the
three-module seed 3 found the -O3 inliner taking one of two procedures of
one name for the other."""

import pytest

from ._toolchain import tools_missing
from .names_difftest import build_and_run, generate, generate_modules


@pytest.mark.parametrize("opt", (0, 1, 2, 3))
@pytest.mark.parametrize("seed", (1, 3, 34, 302))
def test_a_program_of_reused_names(seed, opt):
    """One module."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    src, expect = generate(seed)
    got, err = build_and_run([src], opt)
    assert err is None, err
    assert got == expect


@pytest.mark.parametrize("opt", (0, 3))
@pytest.mark.parametrize("seed", (3, 5))
def test_modules_of_reused_names(seed, opt):
    """Three modules compiled together."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    sources, expect = generate_modules(seed)
    got, err = build_and_run(sources, opt)
    assert err is None, err
    assert got == expect
