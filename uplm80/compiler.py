"""
PL/M-80 Compiler Driver.

Main entry point for the uplm80 compiler.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .lexer import Lexer
from .parser import Parser
from .ast_optimizer import ASTOptimizer
from .codegen import CodeGenerator, Target
from .peephole import PeepholeOptimizer
from .errors import CompilerError, ErrorCollector


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
        target: Target = Target.I8080,
        opt_level: int = 2,
        debug: bool = False,
    ) -> None:
        self.target = target
        self.opt_level = opt_level
        self.debug = debug
        self.errors = ErrorCollector()

    def compile(self, source: str, filename: str = "<input>") -> str | None:
        """
        Compile PL/M-80 source code to assembly.

        Returns the assembly code string, or None if compilation failed.
        """
        try:
            # Phase 1: Lexical Analysis
            if self.debug:
                print(f"[DEBUG] Phase 1: Lexing {filename}", file=sys.stderr)

            lexer = Lexer(source, filename)
            tokens = lexer.tokenize()

            if self.debug:
                print(f"[DEBUG] Produced {len(tokens)} tokens", file=sys.stderr)

            # Phase 2: Parsing
            if self.debug:
                print("[DEBUG] Phase 2: Parsing", file=sys.stderr)

            parser = Parser(tokens, filename)
            ast = parser.parse_module()

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
                    f"[DEBUG] Phase 4: Code Generation (target: {self.target.name})",
                    file=sys.stderr,
                )

            codegen = CodeGenerator(self.target)
            asm_code = codegen.generate(ast)

            if self.debug:
                print(f"[DEBUG] Generated {len(asm_code.splitlines())} lines of assembly", file=sys.stderr)

            # Phase 5: Peephole Optimization
            if self.opt_level > 0:
                if self.debug:
                    print("[DEBUG] Phase 5: Peephole Optimization", file=sys.stderr)

                peephole = PeepholeOptimizer(self.target)
                asm_code = peephole.optimize(asm_code)

                if self.debug:
                    for pattern, count in peephole.stats.items():
                        print(f"[DEBUG]   {pattern}: {count} applied", file=sys.stderr)

            return asm_code

        except CompilerError as e:
            self.errors.add_error(e)
            return None

    def compile_file(self, input_path: Path, output_path: Path | None = None) -> bool:
        """
        Compile a PL/M-80 source file.

        Returns True on success, False on failure.
        """
        # Read source file
        try:
            source = input_path.read_text()
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


def main() -> None:
    """Main entry point for the uplm80 compiler."""
    parser = argparse.ArgumentParser(
        prog="uplm80",
        description="Highly optimizing PL/M-80 compiler targeting 8080/Z80",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input PL/M-80 source file (.plm)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output assembly file (.mac)",
    )

    parser.add_argument(
        "-t", "--target",
        choices=["8080", "z80"],
        default="8080",
        help="Target processor (default: 8080)",
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

    args = parser.parse_args()

    # Select target
    target = Target.Z80 if args.target == "z80" else Target.I8080

    # Create compiler
    compiler = Compiler(
        target=target,
        opt_level=args.optimize,
        debug=args.debug,
    )

    # Compile
    success = compiler.compile_file(args.input, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
