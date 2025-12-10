"""
PL/M-80 Lexer (Tokenizer).

Converts PL/M-80 source code into a stream of tokens.
"""

from typing import Iterator
import re
from .tokens import Token, TokenType, RESERVED_WORDS
from .errors import LexerError, SourceLocation


class Lexer:
    """Tokenizer for PL/M-80 source code."""

    def __init__(self, source: str, filename: str = "<input>") -> None:
        self.source = source
        self.filename = filename
        self.pos = 0  # Current position in source
        self.line = 1  # Current line number (1-based)
        self.column = 1  # Current column (1-based)
        self.line_start = 0  # Position of start of current line

        # Conditional compilation state
        self._cond_symbols: set[str] = set()  # Defined symbols
        self._cond_stack: list[tuple[bool, bool]] = []  # (branch_taken, in_else)
        self._cond_enabled = False  # Whether $cond has been seen

    def define_symbol(self, symbol: str) -> None:
        """Define a conditional compilation symbol (like -D on command line)."""
        self._cond_symbols.add(symbol.upper())

    def _current_location(self) -> SourceLocation:
        """Get the current source location."""
        return SourceLocation(self.line, self.column, self.filename)

    def _is_skipping(self) -> bool:
        """Check if we're in a conditional branch that should be skipped."""
        if not self._cond_enabled or not self._cond_stack:
            return False
        # We're skipping if any level in the stack is in a false branch
        for branch_taken, in_else in self._cond_stack:
            # If branch_taken and not in_else: we took the $if branch, include code
            # If branch_taken and in_else: we're in $else after taking $if, skip
            # If not branch_taken and not in_else: we're in $if but condition false, skip
            # If not branch_taken and in_else: we're in $else, condition was false, include
            if branch_taken and in_else:
                return True
            if not branch_taken and not in_else:
                return True
        return False

    def _peek(self, offset: int = 0) -> str:
        """Peek at character at current position + offset."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return "\0"
        return self.source[pos]

    def _advance(self) -> str:
        """Advance position and return current character."""
        if self.pos >= len(self.source):
            return "\0"
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.line_start = self.pos
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace, comments, and conditionally-excluded code."""
        while self.pos < len(self.source):
            ch = self._peek()

            # Whitespace
            if ch in " \t\r\n":
                self._advance()
                continue

            # CP/M EOF marker (Ctrl-Z)
            if ch == "\x1a":
                self._advance()
                continue

            # Comment: /* ... */
            if ch == "/" and self._peek(1) == "*":
                self._skip_comment()
                continue

            # Control directive: $... (skip to end of line)
            # e.g., $Q=1, $INCLUDE, $PAGELENGTH
            if ch == "$" and self.column == 1:
                self._skip_control_directive()
                continue

            # If we're in a skipped conditional branch, skip non-comment characters
            if self._is_skipping():
                self._advance()
                continue

            break

    def _skip_comment(self) -> None:
        """Skip a /* ... */ comment, handling conditional directives."""
        start_loc = self._current_location()
        start_pos = self.pos
        self._advance()  # /
        self._advance()  # *

        # Check for directive comment: /** $... **/
        is_directive = self._peek() == "*" and self._peek(1) == " " and self._peek(2) == "$"

        while self.pos < len(self.source):
            if self._peek() == "*" and self._peek(1) == "/":
                end_pos = self.pos
                self._advance()  # *
                self._advance()  # /

                # Process directive if it matches /** $... **/
                if is_directive:
                    comment_text = self.source[start_pos:end_pos + 2]
                    self._process_directive(comment_text, start_loc)
                return
            self._advance()

        raise LexerError("Unterminated comment", start_loc)

    def _process_directive(self, comment: str, loc: SourceLocation) -> None:
        """Process a conditional compilation directive comment.

        Supported directives:
          /** $set (name) **/   - Define a symbol
          /** $reset (name) **/ - Undefine a symbol
          /** $cond **/         - Enable conditional compilation
          /** $if name **/      - Begin conditional block
          /** $else **/         - Else branch
          /** $endif **/        - End conditional block
        """
        # Extract the directive content between /** and **/
        match = re.match(r'/\*\*\s*\$(\w+)\s*(.*?)\s*\*\*/', comment, re.IGNORECASE)
        if not match:
            return  # Not a valid directive, treat as normal comment

        directive = match.group(1).lower()
        arg = match.group(2).strip()

        if directive == "set":
            # $set (name) - define symbol
            m = re.match(r'\((\w+)\)', arg)
            if m:
                self._cond_symbols.add(m.group(1).upper())
        elif directive == "reset":
            # $reset (name) - undefine symbol
            m = re.match(r'\((\w+)\)', arg)
            if m:
                self._cond_symbols.discard(m.group(1).upper())
        elif directive == "cond":
            # $cond - enable conditional compilation
            self._cond_enabled = True
        elif directive == "if":
            # $if name - begin conditional
            if self._cond_enabled:
                symbol = arg.upper()
                condition_true = symbol in self._cond_symbols
                self._cond_stack.append((condition_true, False))
        elif directive == "else":
            # $else - switch to else branch
            if self._cond_enabled and self._cond_stack:
                branch_taken, _ = self._cond_stack[-1]
                self._cond_stack[-1] = (branch_taken, True)
        elif directive == "endif":
            # $endif - end conditional
            if self._cond_enabled and self._cond_stack:
                self._cond_stack.pop()

    def _skip_control_directive(self) -> None:
        """Skip or process a $... control directive line.

        PL/M-80 uses $ at the start of a line for compiler controls like:
        $Q=1, $INCLUDE, $PAGELENGTH(60), etc.
        Most are skipped, but we process $DEFINE/$SET/$RESET for conditional compilation.
        """
        start_pos = self.pos
        # Collect the entire line
        while self.pos < len(self.source):
            ch = self._peek()
            if ch == "\n" or ch == "\r":
                break
            if ch == "\0":
                break
            self._advance()

        line = self.source[start_pos:self.pos].strip()

        # Skip the newline if present
        if self.pos < len(self.source) and self._peek() in "\r\n":
            self._advance()

        # Process conditional compilation directives
        # $DEFINE name or $SET(name) - define a symbol
        match = re.match(r'\$(?:DEFINE|SET)\s*\(?(\w+)\)?', line, re.IGNORECASE)
        if match:
            self._cond_symbols.add(match.group(1).upper())
            return

        # $RESET(name) - undefine a symbol
        match = re.match(r'\$RESET\s*\((\w+)\)', line, re.IGNORECASE)
        if match:
            self._cond_symbols.discard(match.group(1).upper())
            return

        # $COND - enable conditional compilation
        if re.match(r'\$COND\b', line, re.IGNORECASE):
            self._cond_enabled = True
            return

        # Other directives ($TITLE, $INCLUDE, etc.) are just skipped

    def _make_token(
        self, token_type: TokenType, value: object, lexeme: str, start_line: int, start_col: int
    ) -> Token:
        """Create a token with location info."""
        return Token(token_type, value, start_line, start_col, lexeme)

    def _scan_identifier(self) -> Token:
        """Scan an identifier or keyword."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        # First character already validated as letter
        self._advance()

        # Continue with letters, digits, or $
        while True:
            ch = self._peek()
            if ch.isalnum() or ch == "$" or ch == "_":
                self._advance()
            else:
                break

        lexeme = self.source[start_pos : self.pos]
        # Remove $ break characters (PL/M allows $ for readability in identifiers)
        upper_lexeme = lexeme.upper().replace("$", "")

        # Check if it's a reserved word
        if upper_lexeme in RESERVED_WORDS:
            return self._make_token(
                RESERVED_WORDS[upper_lexeme], upper_lexeme, lexeme, start_line, start_col
            )

        # It's an identifier
        return self._make_token(TokenType.IDENTIFIER, upper_lexeme, lexeme, start_line, start_col)

    def _scan_number(self) -> Token:
        """Scan a numeric constant (binary, octal, decimal, or hex)."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        # Collect all alphanumeric characters and $ (digit separator)
        while self._peek().isalnum() or self._peek() == "$":
            self._advance()

        lexeme = self.source[start_pos : self.pos]
        # Remove $ digit separators for parsing
        upper_lexeme = lexeme.upper().replace("$", "")

        # Determine the base and parse the number
        try:
            if upper_lexeme.endswith("H"):
                # Hexadecimal: must start with digit
                value = int(upper_lexeme[:-1], 16)
            elif upper_lexeme.endswith("B"):
                # Binary
                value = int(upper_lexeme[:-1], 2)
            elif upper_lexeme.endswith("O") or upper_lexeme.endswith("Q"):
                # Octal
                value = int(upper_lexeme[:-1], 8)
            elif upper_lexeme.endswith("D"):
                # Explicit decimal
                value = int(upper_lexeme[:-1], 10)
            else:
                # Plain decimal
                value = int(upper_lexeme, 10)
        except ValueError:
            raise LexerError(
                f"Invalid numeric constant: {lexeme}",
                SourceLocation(start_line, start_col, self.filename),
            )

        return self._make_token(TokenType.NUMBER, value, lexeme, start_line, start_col)

    def _scan_string(self) -> Token:
        """Scan a string literal."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        self._advance()  # Opening quote

        chars: list[str] = []
        while True:
            ch = self._peek()
            if ch == "\0" or ch == "\n":
                raise LexerError(
                    "Unterminated string literal",
                    SourceLocation(start_line, start_col, self.filename),
                )
            if ch == "'":
                self._advance()
                # Check for escaped quote ''
                if self._peek() == "'":
                    chars.append("'")
                    self._advance()
                else:
                    break
            else:
                chars.append(ch)
                self._advance()

        lexeme = self.source[start_pos : self.pos]
        value = "".join(chars)

        return self._make_token(TokenType.STRING, value, lexeme, start_line, start_col)

    def _scan_address_literal(self) -> Token:
        """Scan an address literal at beginning of line (e.g., 0FAH:)."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        # Collect all alphanumeric characters
        while self._peek().isalnum():
            self._advance()

        lexeme = self.source[start_pos : self.pos]
        upper_lexeme = lexeme.upper()

        # Parse as hex if it ends with H, otherwise decimal
        try:
            if upper_lexeme.endswith("H"):
                value = int(upper_lexeme[:-1], 16)
            else:
                value = int(upper_lexeme, 10)
        except ValueError:
            raise LexerError(
                f"Invalid address literal: {lexeme}",
                SourceLocation(start_line, start_col, self.filename),
            )

        return self._make_token(TokenType.NUMBER, value, lexeme, start_line, start_col)

    def next_token(self) -> Token:
        """Get the next token from the source."""
        self._skip_whitespace_and_comments()

        if self.pos >= len(self.source):
            return self._make_token(TokenType.EOF, None, "", self.line, self.column)

        start_line = self.line
        start_col = self.column
        ch = self._peek()

        # Identifier or keyword (starts with letter)
        if ch.isalpha():
            return self._scan_identifier()

        # Number (starts with digit)
        if ch.isdigit():
            return self._scan_number()

        # String literal
        if ch == "'":
            return self._scan_string()

        # Two-character operators
        if ch == "<":
            self._advance()
            if self._peek() == "=":
                self._advance()
                return self._make_token(TokenType.OP_LE, "<=", "<=", start_line, start_col)
            elif self._peek() == ">":
                self._advance()
                return self._make_token(TokenType.OP_NE, "<>", "<>", start_line, start_col)
            return self._make_token(TokenType.OP_LT, "<", "<", start_line, start_col)

        if ch == ">":
            self._advance()
            if self._peek() == "=":
                self._advance()
                return self._make_token(TokenType.OP_GE, ">=", ">=", start_line, start_col)
            return self._make_token(TokenType.OP_GT, ">", ">", start_line, start_col)

        # Single-character tokens
        single_char_tokens: dict[str, TokenType] = {
            "+": TokenType.OP_PLUS,
            "-": TokenType.OP_MINUS,
            "*": TokenType.OP_STAR,
            "/": TokenType.OP_SLASH,
            "=": TokenType.OP_EQ,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
            ":": TokenType.COLON,
            ".": TokenType.DOT,
            "$": TokenType.DOLLAR,
        }

        if ch in single_char_tokens:
            self._advance()
            return self._make_token(single_char_tokens[ch], ch, ch, start_line, start_col)

        # Unknown character
        raise LexerError(
            f"Unexpected character: {ch!r}",
            SourceLocation(start_line, start_col, self.filename),
        )

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source and return a list of tokens."""
        tokens: list[Token] = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens

    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens."""
        while True:
            token = self.next_token()
            yield token
            if token.type == TokenType.EOF:
                break


def tokenize(source: str, filename: str = "<input>") -> list[Token]:
    """Convenience function to tokenize source code."""
    return Lexer(source, filename).tokenize()
