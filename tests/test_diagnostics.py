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


def test_an_assignment_is_placed_at_its_target():
    """The grammar starts an assignment's span at its `=': the error was
    placed at column 6 here (and, before, nowhere)."""
    r = _compile({"sz.plm": "t: do;\ndeclare x address;\n\n   x = size(3);\nend t;\n"}, "sz.plm")
    assert r.returncode != 0
    assert "sz.plm:4:4: error: SIZE() needs a variable" in r.stderr, r.stderr


def test_a_literally_over_two_lines_does_not_move_the_lines_after_it():
    """The macro pass put a LITERALLY's text in with its line ends, and each
    use moved every line after it down: MP/M II's ERA.PLM, which uses a
    PROCESS$DESCRIPTOR of 16 lines once, was reported at line 357 for 341."""
    r = _compile({"m.plm": "t: do;\ndeclare two literally 'b byte,\n  c byte';\ndeclare two;\n"
                           "if b = 300 then b = 1;\nend t;\n"}, "m.plm")
    assert r.returncode == 0, r.stderr
    assert "m.plm:5:8: warning: comparison BYTE = 300 is always false" in r.stderr, r.stderr


def test_after_a_colon_against_an_end_the_columns_are_the_sources():
    """A label's colon against END, `out:end p;': the null statement put in
    before the END (frontend.label_the_ends) moved every column after it on
    the line on one, in every message (0.4.2's Known issues)."""
    r = _compile({"c.plm": "t: do;\ndeclare b byte;\np: procedure;\n  b = 1;\n"
                           "out:end p; b = zz;\nend t;\n"}, "c.plm")
    assert "c.plm:5:16: error: ZZ is not declared" in r.stderr, r.stderr
    r = _compile({"c.plm": "t: do;\ndeclare b byte;\np: procedure;\n"
                           "  do; x:end; do; y:end; b = zz;\nend p;\nend t;\n"}, "c.plm")
    assert "c.plm:4:29: error: ZZ is not declared" in r.stderr, r.stderr
    r = _compile({"c.plm": "t: do;\ndeclare b byte;\np: procedure;\n  b = 1;\n"
                           "out:end p; if b = 300 then b = 2;\nend t;\n"}, "c.plm")
    assert "c.plm:5:19: warning: comparison BYTE = 300 is always false" in r.stderr, r.stderr
    r = _compile({"c.plm": "t: do;\ndeclare b byte;\np: procedure;\n  b = 1;\n"
                           "out:end p; b = = 2;\nend t;\n"}, "c.plm")
    assert "c.plm:5:16: error: unexpected token 'EQ'" in r.stderr, r.stderr


LIT = "declare lit literally '1 +\n  2 +\n  3';\n"


def test_after_a_literallys_text_the_columns_are_the_sources():
    """A LITERALLY's text takes the place of its name, on one line
    however many it runs over, and moved every column after it on the line
    by how much longer it is than the name: `b = lit; b = zz;' was 6:24,
    where zz is at column 14 (0.4.3's release check), and a name longer
    than its text moved them back."""
    cases = [
        ("b = lit; b = zz;\n", "c.plm:6:14: error: ZZ is not declared"),
        ("b = lit; b = = 2;\n", "c.plm:6:14: error: unexpected token 'EQ'"),
        ("b = lit; if b = 300 then b = 2;\n",
         "c.plm:6:17: warning: comparison BYTE = 300 is always false"),
        ("declare longname literally '1';\nb = longname; b = zz;\n",
         "c.plm:7:19: error: ZZ is not declared"),
        # A text that names another: the outer name's length is what counts.
        ("declare two literally 'lit + lit';\nb = two; b = zz;\n",
         "c.plm:7:14: error: ZZ is not declared"),
        # With a colon against an END on the line too.
        ("p: procedure;\n  b = 1;\nout:end p; b = lit; b = zz;\n",
         "c.plm:8:25: error: ZZ is not declared"),
        # A word declared LITERALLY 'LITERALLY', and a procedure's name.
        ("declare lt literally 'literally';\ndeclare five lt '5';\nb = five; b = zz;\n",
         "c.plm:8:15: error: ZZ is not declared"),
        ("declare mon1 literally 'ldmon1x';\nmon1: procedure external; end mon1; b = zz;\n",
         "c.plm:7:41: error: ZZ is not declared"),
    ]
    for stmts, message in cases:
        r = _compile({"c.plm": "t: do;\ndeclare b byte;\n" + LIT + stmts + "end t;\n"}, "c.plm")
        assert message in r.stderr, (stmts, r.stderr)


def test_a_column_in_a_literallys_text_is_the_names():
    """What a message is about in a LITERALLY's text is placed at the
    name; the byte-shift check's warnings are placed as the rest."""
    r = _compile({"c.plm": "t: do;\ndeclare (b, k) byte, w address;\n" + LIT
                           + "k = input(1); b = lit; w = shl(k, 4) + 1;\nend t;\n"}, "c.plm")
    assert "c.plm:6:28: warning: SHL(K, 4): SHL of a BYTE is a BYTE" in r.stderr, r.stderr
    r = _compile({"c.plm": "t: do;\ndeclare b byte;\n" + LIT
                           + "declare bad literally 'zz + 1';\nb = lit + bad;\nend t;\n"}, "c.plm")
    assert "c.plm:7:11: error: ZZ is not declared" in r.stderr, r.stderr
