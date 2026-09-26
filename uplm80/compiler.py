"""
PL/M-80 Compiler Driver.

Main entry point for the uplm80 compiler.
"""

import argparse
import functools
import re
import sys
from pathlib import Path

# Import peephole optimizer from upeepz80 library
import upeepz80
from upeepz80 import PeepholeOptimizer

from . import __version__
from .frontend import parse_source
from .codegen import CodeGenerator, Mode
from .errors import CompilerError, ErrorCollector
from .names import check_names, fix_symbols

# Import AST optimizer (PL/M-80 specific)
from .ast_optimizer import ASTOptimizer

# The oldest upeepz80 whose code is right under PL/M-80's calling convention,
# as pyproject.toml's dependency has it (a test keeps the two the same).
UPEEPZ80_MIN = "0.2.6"

# A call of a procedure of three parameters, one of them pushed, which ends
# its routine.  upeepz80 0.2.6 leaves the `call'; 0.2.5 makes it `jp P3'.
_TAIL_CALL_PROBE = ("\textrn\tP3\nQ::\n\tld\thl,1111h\n\tpush\thl\n"
                    "\tld\tbc,2222h\n\tld\tde,3333h\n\tcall\tP3\n\tret\n")


def _release(version: str) -> tuple[int, ...]:
    """The numbers a version string starts with: "0.2.6rc1" is (0, 2, 6)."""
    m = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(n) for n in m.group().split(".")) if m else ()


@functools.cache
def upeepz80_problem() -> str | None:
    """Why the upeepz80 in use cannot optimize this compiler's code, or None.

    upeepz80 before 0.2.6 turns `push ... / call p / ret' into `push ... /
    jp p'.  Jumped to, p finds its caller's return address where the word
    pushed for it should be, and under PL/M-80's convention, in which the
    callee takes the pushed words off the stack, it takes the return address
    for its first argument and returns to where that argument points.

    A version before UPEEPZ80_MIN is refused unless it keeps the `call' of
    _TAIL_CALL_PROBE: a development tree with 0.2.6's fix that is still
    numbered 0.2.5 does, and the release 0.2.5 does not.
    """
    found = str(getattr(upeepz80, "__version__", "")) or "(no version)"
    if _release(found) >= _release(UPEEPZ80_MIN):
        return None
    probe = PeepholeOptimizer().optimize(_TAIL_CALL_PROBE)
    if re.search(r"^\s*call\s+P3\b", probe, re.MULTILINE | re.IGNORECASE):
        return None
    return (f"upeepz80 {found} is too old: uplm80 needs upeepz80 {UPEEPZ80_MIN} "
            f"or later (pip install --upgrade 'upeepz80>={UPEEPZ80_MIN}').  "
            f"upeepz80 {found} turns `push ... / call p / ret' into `push ... / "
            f"jp p', and p, which takes the words pushed for it off the stack, "
            f"then takes its return address for its first argument.  A "
            f"development upeepz80 numbered below {UPEEPZ80_MIN} is accepted "
            f"if it has {UPEEPZ80_MIN}'s fix, which the one in "
            f"{Path(upeepz80.__file__).parent} does not.  -O0 "
            f"does not use upeepz80.")


def require_upeepz80(opt_level: int) -> None:
    """Stop, with CompilerError, if a compile at ``opt_level`` would use
    upeepz80 and upeepz80_problem() finds a problem with it."""
    problem = upeepz80_problem() if opt_level > 0 else None
    if problem is not None:
        raise CompilerError(problem)


class Compiler:
    """
    PL/M-80 Compiler.

    Pipeline:
    1. Lexer: Source -> Tokens
    2. Parser: Tokens -> AST
    3. AST Optimizer: AST -> Optimized AST
    4. Code Generator: AST -> Assembly
    5. Peephole Optimizer: Assembly -> Optimized Assembly
    """

    def __init__(
        self,
        mode: Mode = Mode.CPM,
        opt_level: int = 2,
        debug: bool = False,
        defines: list[str] | None = None,
        include_paths: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.opt_level = opt_level
        self.debug = debug
        self.defines = defines or []  # Symbols to define for conditional compilation
        self.include_paths = include_paths or []  # Include search paths
        self.errors = ErrorCollector()

    def _clean_cpm_source(self, source: str) -> str:
        """Clean CP/M source files with high-bit characters.

        CP/M editors sometimes set the high bit (0x80) on characters for
        word-wrap hints or other metadata. Strip the high bit to recover
        the original ASCII character.

        This also fixes decorative banners like ' /ª...*/':
        - 'ª' (0xAA) becomes '*' (0x2A) when high bit is stripped
        - So ' /ª.../*' becomes ' /*...*/' - a valid comment
        """
        # Strip high bit from all characters
        return ''.join(chr(ord(c) & 0x7F) for c in source)

    def compile(self, source: str, filename: str = "<input>") -> str | None:
        """
        Compile PL/M-80 source code to assembly.

        Returns the assembly code string, or None if compilation failed.
        """
        try:
            require_upeepz80(self.opt_level)

            # Phases 1-2: preprocess + uplox plm_full LR parse + AST lowering.
            if self.debug:
                print(f"[DEBUG] Front-end (uplox plm_full) {filename}", file=sys.stderr)

            ast = parse_source(
                source,
                filename,
                defines=self.defines,
                include_paths=self.include_paths,
            )
            check_names([ast])

            if self.debug:
                print(f"[DEBUG] Parsed module: {ast.name}", file=sys.stderr)
                print(f"[DEBUG]   {len(ast.decls)} declarations", file=sys.stderr)
                print(f"[DEBUG]   {len(ast.stmts)} statements", file=sys.stderr)

            # Phase 3: AST Optimization
            if self.opt_level > 0:
                if self.debug:
                    print(
                        f"[DEBUG] Phase 3: AST Optimization (level {self.opt_level})",
                        file=sys.stderr,
                    )

                optimizer = ASTOptimizer(self.opt_level)
                ast = optimizer.optimize(ast)

                if self.debug:
                    print(f"[DEBUG]   Constants folded: {optimizer.stats.constants_folded}", file=sys.stderr)
                    print(f"[DEBUG]   Strength reductions: {optimizer.stats.strength_reductions}", file=sys.stderr)
                    print(f"[DEBUG]   Dead code eliminated: {optimizer.stats.dead_code_eliminated}", file=sys.stderr)

            # Phase 4: Code Generation
            if self.debug:
                print(
                    f"[DEBUG] Phase 4: Code Generation (mode: {self.mode.name})",
                    file=sys.stderr,
                )

            codegen = CodeGenerator(self.mode, reg_debug=self.debug)
            asm_code = codegen.generate(ast)

            # Print any warnings from code generation
            for warning in codegen.warnings:
                print(warning, file=sys.stderr)

            if self.debug:
                print(f"[DEBUG] Generated {len(asm_code.splitlines())} lines of assembly", file=sys.stderr)

            # Phase 5: Peephole Optimization
            if self.opt_level > 0:
                if self.debug:
                    print("[DEBUG] Phase 5: Peephole Optimization", file=sys.stderr)

                peephole = PeepholeOptimizer()
                asm_code = peephole.optimize(asm_code)

                if self.debug:
                    for pattern, count in peephole.stats.items():
                        print(f"[DEBUG]   {pattern}: {count} applied", file=sys.stderr)

            return fix_symbols(asm_code)

        except CompilerError as e:
            self.errors.add_error(e)
            return None

    def compile_file(self, input_path: Path, output_path: Path | None = None) -> bool:
        """
        Compile a PL/M-80 source file.

        Returns True on success, False on failure.
        """
        # Read source file (CP/M files may have high-bit characters)
        try:
            try:
                source = input_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                source = input_path.read_text(encoding='latin-1')
            # Handle CP/M decorative banners and high-bit characters
            source = self._clean_cpm_source(source)
        except OSError as e:
            print(f"Error reading {input_path}: {e}", file=sys.stderr)
            return False

        # Compile
        asm_code = self.compile(source, str(input_path))

        if asm_code is None:
            self.errors.report()
            return False

        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".mac")

        # Write output
        try:
            output_path.write_text(asm_code)
            print(f"Compiled {input_path} -> {output_path}")
        except OSError as e:
            print(f"Error writing {output_path}: {e}", file=sys.stderr)
            return False

        return True

    def _optimize_modules(self, modules: list, filenames: list[str]) -> list:
        """Phase 3, AST optimization, of each module of a multi-file compile."""
        if self.opt_level == 0:
            return modules
        out = []
        for ast, filename in zip(modules, filenames):
            if self.debug:
                print(f"[DEBUG] Phase 3: AST Optimization for {filename}", file=sys.stderr)
            ast = ASTOptimizer(self.opt_level).optimize(ast)
            ast.uplm80_file = filename  # a new Module; see names._file_name
            out.append(ast)
        return out

    def compile_files(self, input_paths: list[Path], output_path: Path | None = None) -> bool:
        """
        Compile multiple PL/M-80 source files together.

        All files are parsed first, then a unified call graph is built
        across all modules for optimal local variable storage allocation.

        Returns True on success, False on failure.
        """
        if len(input_paths) == 1:
            return self.compile_file(input_paths[0], output_path)

        try:
            require_upeepz80(self.opt_level)

            modules, filenames = [], []

            # Phase 1 & 2: Lex and parse all files
            for input_path in input_paths:
                try:
                    try:
                        source = input_path.read_text(encoding='utf-8')
                    except UnicodeDecodeError:
                        source = input_path.read_text(encoding='latin-1')
                    # Handle CP/M decorative banners and high-bit characters
                    source = self._clean_cpm_source(source)
                except OSError as e:
                    print(f"Error reading {input_path}: {e}", file=sys.stderr)
                    return False

                filename = str(input_path)
                filenames.append(filename)

                if self.debug:
                    print(f"[DEBUG] Front-end (uplox plm_full) {filename}", file=sys.stderr)

                ast = parse_source(
                    source,
                    filename,
                    defines=self.defines,
                    include_paths=self.include_paths,
                )
                modules.append(ast)

            # The names of every module, before any of them is optimized.
            check_names(modules, multi=True)
            modules = self._optimize_modules(modules, filenames)

            # Phase 4: Code Generation with unified call graph
            if self.debug:
                print(f"[DEBUG] Phase 4: Code Generation (multi-module, {len(modules)} files)", file=sys.stderr)

            codegen = CodeGenerator(self.mode, reg_debug=self.debug)
            asm_code = codegen.generate_multi(modules)

            # Print any warnings from code generation
            for warning in codegen.warnings:
                print(warning, file=sys.stderr)

            if self.debug:
                print(f"[DEBUG] Generated {len(asm_code.splitlines())} lines of assembly", file=sys.stderr)

            # Phase 5: Peephole Optimization
            if self.opt_level > 0:
                if self.debug:
                    print("[DEBUG] Phase 5: Peephole Optimization", file=sys.stderr)

                peephole = PeepholeOptimizer()
                asm_code = peephole.optimize(asm_code)
            asm_code = fix_symbols(asm_code)

            # Determine output path
            if output_path is None:
                output_path = input_paths[0].with_suffix(".mac")

            # Write output
            try:
                output_path.write_text(asm_code)
                files_str = ', '.join(str(p) for p in input_paths)
                print(f"Compiled {files_str} -> {output_path}")
            except OSError as e:
                print(f"Error writing {output_path}: {e}", file=sys.stderr)
                return False

            return True

        except CompilerError as e:
            self.errors.add_error(e)
            self.errors.report()
            return False


def main() -> None:
    """Main entry point for the uplm80 compiler."""
    parser = argparse.ArgumentParser(
        prog="uplm80",
        description="Highly optimizing PL/M-80 compiler targeting Z80",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs='+',
        help="Input PL/M-80 source file(s) (.plm)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output assembly file (.mac)",
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["cpm", "bare", "mpm"],
        default="cpm",
        help="Runtime mode: cpm=CP/M program, bare=bare metal, "
             "mpm=MP/M relocatable (.PRL/.SPR/.RSP) (default: cpm)",
    )

    parser.add_argument(
        "-O", "--optimize",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="Optimization level (default: 2)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    parser.add_argument(
        "-D", "--define",
        action="append",
        dest="defines",
        default=[],
        metavar="SYMBOL",
        help="Define conditional compilation symbol (can be repeated)",
    )

    parser.add_argument(
        "-I", "--include",
        action="append",
        dest="include_paths",
        default=[],
        metavar="PATH",
        help="Add include search path (can be repeated)",
    )

    args = parser.parse_args()

    # Select mode
    mode = {"cpm": Mode.CPM, "bare": Mode.BARE, "mpm": Mode.MPM}[args.mode]

    compiler = Compiler(
        mode=mode,
        opt_level=args.optimize,
        debug=args.debug,
        defines=args.defines,
        include_paths=args.include_paths,
    )

    # Compile (supports multiple input files)
    success = compiler.compile_files(args.input, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
