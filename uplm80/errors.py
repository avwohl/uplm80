"""
Error handling for the PL/M-80 compiler.

Custom exceptions with source location tracking for better error messages.
"""

from dataclasses import dataclass, field


@dataclass
class SourceLocation:
    """Location in source code."""

    line: int
    column: int
    filename: str = "<unknown>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


class CompilerError(Exception):
    """Base class for all compiler errors."""

    def __init__(self, message: str, location: SourceLocation | None = None) -> None:
        self.message = message
        self.location = location
        super().__init__(self.format_message())

    def format_message(self) -> str:
        if self.location:
            return f"{self.location}: error: {self.message}"
        return f"error: {self.message}"


class LexerError(CompilerError):
    """Error during lexical analysis."""

    pass


class ParserError(CompilerError):
    """Error during parsing."""

    pass


class SemanticError(CompilerError):
    """Error during semantic analysis."""

    pass


class CodeGenError(CompilerError):
    """Error during code generation."""

    pass


@dataclass
class CompilerWarning:
    """A compiler warning."""

    message: str
    location: SourceLocation | None = None

    def format_message(self) -> str:
        if self.location:
            return f"{self.location}: warning: {self.message}"
        return f"warning: {self.message}"


@dataclass
class ErrorCollector:
    """Collects errors and warnings during compilation."""

    errors: list[CompilerError] = field(default_factory=list)
    warnings: list[CompilerWarning] = field(default_factory=list)

    def add_error(self, error: CompilerError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)

    def add_warning(self, warning: CompilerWarning) -> None:
        """Add a warning to the collection."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    def report(self) -> None:
        """Print all errors and warnings to stderr."""
        import sys

        for warning in self.warnings:
            print(warning.format_message(), file=sys.stderr)
        for error in self.errors:
            print(error.format_message(), file=sys.stderr)

    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
