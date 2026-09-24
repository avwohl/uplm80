"""The run tests compile with this checkout, not an installed uplm80."""

import os
import subprocess
import sys

from ._toolchain import REPO_ROOT, compiler_env


def test_the_compiler_subprocess_imports_this_checkout():
    r = subprocess.run([sys.executable, "-P", "-c", "import uplm80; print(uplm80.__file__)"],
                       capture_output=True, text=True, env=compiler_env(), check=True)
    assert os.path.commonpath([REPO_ROOT, os.path.abspath(r.stdout.strip())]) == REPO_ROOT
