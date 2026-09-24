"""Compile, assemble, link and run a PL/M program under a CP/M emulator.

For the tests that have to see what a program does, not only what it
assembles to.  They skip when um80, ul80 or cpmemu is not installed.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def cpmemu() -> str | None:
    """The emulator: $CPMEMU, cpmemu on PATH, or ~/src/cpmemu/src/cpmemu."""
    for cand in (os.environ.get("CPMEMU"), shutil.which("cpmemu"),
                 os.path.expanduser("~/src/cpmemu/src/cpmemu")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def run_plm(src: str, opt: int = 2) -> str:
    """What the program prints, carriage returns removed."""
    emu = cpmemu()
    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not installed")
    if emu is None:
        pytest.skip("cpmemu not installed")
    with tempfile.TemporaryDirectory() as d:
        plm, mac, rel, com = (os.path.join(d, n) for n in ("T.PLM", "T.MAC", "T.REL", "T.COM"))
        with open(plm, "w") as fh:
            fh.write(src)

        def run(*argv):
            return subprocess.run(argv, capture_output=True, text=True, timeout=60)

        # -P: the compiler under test, not one that happens to be in the cwd.
        r = run(sys.executable, "-P", "-m", "uplm80.compiler", "-O", str(opt), "-o", mac, plm)
        assert r.returncode == 0, r.stderr
        r = run("um80", "-o", rel, mac)
        assert r.returncode == 0, r.stderr
        r = run("ul80", "-o", com, rel)
        assert r.returncode == 0, r.stderr
        r = run(emu, com)
        return r.stdout.replace("\r", "")
