"""A warning or an error names the file and line it is about.

The parser sees one text: the file being compiled with its $INCLUDE files
spliced in, and the lines a conditional skips taken out.  Its line numbers
are not the file's past the first $INCLUDE or skipped line, and code
generation named no file at all: `uplm80 e.plm' printed
`<unknown>:3:8: warning: comparison BYTE = 300 is always false'.
"""

import os
import subprocess
import tempfile

from ._toolchain import compile_cmd, compiler_env


def _compile(files: dict[str, str], main: str, *args: str) -> subprocess.CompletedProcess:
    """Compile ``main`` from a directory holding ``files``, run from there so
    the diagnostics name the files as given."""
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as fh:
                fh.write(text)
        return subprocess.run(compile_cmd(*args, "-o", "OUT.MAC", main), cwd=d,
                              capture_output=True, text=True, timeout=60,
                              env=compiler_env(), check=False)


def test_a_warning_names_the_file_and_the_included_file():
    r = _compile({
        "e.plm": "t: do;\ndeclare b byte;\n$include (inc.plm)\nif b = 300 then b = 1;\nend t;\n",
        "inc.plm": "/* included */\ndeclare c byte;\nif c = 400 then c = 2;\n",
    }, "e.plm")
    assert r.returncode == 0, r.stderr
    assert "e.plm:4:8: warning: comparison BYTE = 300 is always false" in r.stderr, r.stderr
    assert "inc.plm:3:8: warning: comparison BYTE = 400 is always false" in r.stderr, r.stderr
    assert "<unknown>" not in r.stderr, r.stderr


def test_lines_a_conditional_skips_still_count():
    r = _compile({
        "c.plm": "t: do;\ndeclare b byte;\n$if NOPE\nzzz\nyyy\n$endif\nif b = 300 then b = 1;\nend t;\n",
    }, "c.plm")
    assert r.returncode == 0, r.stderr
    assert "c.plm:7:8: warning" in r.stderr, r.stderr


def test_an_if_on_a_constant_is_placed():
    """This warning carried no location at all."""
    r = _compile({"k.plm": "t: do;\ndeclare b byte;\n\nif 2 then b = 1;\nend t;\n"}, "k.plm", "-O0")
    assert r.returncode == 0, r.stderr
    assert "k.plm:4:4: warning: IF condition is always false" in r.stderr, r.stderr


def test_a_syntax_error_in_an_included_file_names_that_file():
    r = _compile({
        "p.plm": "t: do;\ndeclare b byte;\n$include (inc.plm)\nb = 1;\nend t;\n",
        "inc.plm": "/* included */\ndeclare c byte;\nc = = 2;\n",
    }, "p.plm")
    assert r.returncode != 0
    assert "inc.plm:3:5: error: unexpected token" in r.stderr, r.stderr
    assert "at line" not in r.stderr, r.stderr


def test_a_code_generation_error_names_the_file_and_line():
    """An error code generation raised without a location is placed at the
    declaration or statement it was generating."""
    r = _compile({
        "d.plm": "t: do;\ndeclare b byte;\n$include (inc.plm)\nb = 1;\nend t;\n",
        "inc.plm": "\n\ndeclare x (4) byte at (.nothere);\n",
    }, "d.plm")
    assert r.returncode != 0
    assert "inc.plm:3:9: error: AT(.NOTHERE): NOTHERE is not declared" in r.stderr, r.stderr
