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

import uplm80

# The checkout the in-process tests import.  The compiler subprocess must run
# this one too, not whatever uplm80 happens to be installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(uplm80.__file__)))


def compiler_env() -> dict[str, str]:
    """os.environ with REPO_ROOT first on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p)
    return env


def compile_cmd(*args: str) -> list[str]:
    """argv that runs this checkout's compiler; pair it with compiler_env()."""
    # -P keeps the subprocess's cwd off sys.path; PYTHONPATH puts REPO_ROOT on it.
    return [sys.executable, "-P", "-m", "uplm80.compiler", *args]


def cpmemu() -> str | None:
    """The emulator: $CPMEMU, cpmemu on PATH, or ~/src/cpmemu/src/cpmemu."""
    for cand in (os.environ.get("CPMEMU"), shutil.which("cpmemu"),
                 os.path.expanduser("~/src/cpmemu/src/cpmemu")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def run_plm(src: str, opt: int = 2, extra_asm: str | None = None) -> str:
    """What the program prints, carriage returns removed.

    `extra_asm' is a second module, in assembly, linked after the program:
    somewhere to define what the program declares EXTERNAL.
    """
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

        def run(*argv, env=None):
            return subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)

        r = run(*compile_cmd("-O", str(opt), "-o", mac, plm), env=compiler_env())
        assert r.returncode == 0, r.stderr
        r = run("um80", "-o", rel, mac)
        assert r.returncode == 0, r.stderr
        rels = [rel]
        if extra_asm is not None:
            xmac, xrel = os.path.join(d, "X.MAC"), os.path.join(d, "X.REL")
            with open(xmac, "w") as fh:
                fh.write(extra_asm)
            r = run("um80", "-o", xrel, xmac)
            assert r.returncode == 0, r.stderr
            rels.append(xrel)
        r = run("ul80", "-o", com, *rels)
        assert r.returncode == 0, r.stderr
        r = run(emu, com)
        return r.stdout.replace("\r", "")
