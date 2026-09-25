"""Compile, assemble, link and run a PL/M program under a CP/M emulator.

For the tests that have to see what a program does, not only what it
assembles to.  They skip when um80, ul80 or cpmemu is not installed.
:func:`run_plm` compiles with this checkout's compiler as a command; the
differential test and the division oracle compile in-process and share
:func:`run_asm` for the rest.
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


def tools_missing() -> str | None:
    """Why a program cannot be assembled, linked and run here, or None."""
    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            return f"{tool} not installed"
    if cpmemu() is None:
        return "cpmemu not installed"
    return None


class ToolchainError(Exception):
    """A step of building or running a program failed; the message says which."""


def run_asm(asm: str, extra_asm: str | None = None,
            timeout: float = 60) -> subprocess.CompletedProcess:
    """Assemble `asm' with um80, link it with ul80 and run it under cpmemu.

    `extra_asm' is a second module, in assembly, linked after the program:
    somewhere to define what the program declares EXTERNAL.  Returns the
    emulator's CompletedProcess (text); raises ToolchainError, naming the
    step, if one fails or the program runs longer than `timeout' seconds.
    The test skips when um80, ul80 or cpmemu is not installed.
    """
    reason = tools_missing()
    if reason:
        pytest.skip(reason)

    def step(*argv: str) -> None:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
        if r.returncode:
            raise ToolchainError(f"{argv[0]}: {r.stdout}{r.stderr}")

    with tempfile.TemporaryDirectory() as d:
        rels = []
        for name, text in (("T", asm), ("X", extra_asm)):
            if text is None:
                continue
            mac, rel = os.path.join(d, name + ".MAC"), os.path.join(d, name + ".REL")
            with open(mac, "w") as fh:
                fh.write(text)
            step("um80", "-o", rel, mac)
            rels.append(rel)
        com = os.path.join(d, "T.COM")
        step("ul80", "-o", com, *rels)
        try:
            return subprocess.run([cpmemu(), com], capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ToolchainError("run: timed out") from exc


def run_plm(src: str, opt: int = 2, extra_asm: str | None = None,
            mode: str = "cpm") -> str:
    """What the program prints, carriage returns removed.

    It is compiled by this checkout's compiler, run as a command, in ``mode``
    (``--mode``), and built and run by :func:`run_asm`.  The test skips when
    um80, ul80 or cpmemu is not installed.
    """
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as d:
        plm, mac = os.path.join(d, "T.PLM"), os.path.join(d, "T.MAC")
        with open(plm, "w") as fh:
            fh.write(src)
        r = subprocess.run(compile_cmd("-O", str(opt), "--mode", mode, "-o", mac, plm),
                           capture_output=True, text=True, timeout=60, env=compiler_env(),
                           check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            asm = fh.read()
    return run_asm(asm, extra_asm).stdout.replace("\r", "")
