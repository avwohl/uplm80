"""Tests for the PL/M-80 lexer."""

import pytest
from uplm80.lexer import Lexer, tokenize
from uplm80.tokens import TokenType


class TestLexer:
    """Test cases for the lexer."""

    def test_empty_source(self) -> None:
        """Test tokenizing empty source."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_identifiers(self) -> None:
        """Test identifier tokenization."""
        tokens = tokenize("foo BAR baz123 x$y")
        types = [t.type for t in tokens]
        assert types == [
            TokenType.IDENTIFIER,
            TokenType.IDENTIFIER,
            TokenType.IDENTIFIER,
            TokenType.IDENTIFIER,
            TokenType.EOF,
        ]
        # Identifiers are uppercased
        assert tokens[0].value == "FOO"
        assert tokens[1].value == "BAR"

    def test_keywords(self) -> None:
        """Test keyword recognition."""
        tokens = tokenize("DECLARE BYTE ADDRESS PROCEDURE IF THEN ELSE DO WHILE END")
        types = [t.type for t in tokens]
        assert types == [
            TokenType.DECLARE,
            TokenType.BYTE,
            TokenType.ADDRESS,
            TokenType.PROCEDURE,
            TokenType.IF,
            TokenType.THEN,
            TokenType.ELSE,
            TokenType.DO,
            TokenType.WHILE,
            TokenType.END,
            TokenType.EOF,
        ]

    def test_numbers_decimal(self) -> None:
        """Test decimal number tokenization."""
        tokens = tokenize("0 123 65535")
        assert tokens[0].value == 0
        assert tokens[1].value == 123
        assert tokens[2].value == 65535

    def test_numbers_hex(self) -> None:
        """Test hexadecimal number tokenization."""
        tokens = tokenize("0H 0FFH 1234H 0ABCDH")
        assert tokens[0].value == 0
        assert tokens[1].value == 0xFF
        assert tokens[2].value == 0x1234
        assert tokens[3].value == 0xABCD

    def test_numbers_binary(self) -> None:
        """Test binary number tokenization."""
        tokens = tokenize("0B 1B 1010B 11111111B")
        assert tokens[0].value == 0
        assert tokens[1].value == 1
        assert tokens[2].value == 0b1010
        assert tokens[3].value == 0b11111111

    def test_numbers_octal(self) -> None:
        """Test octal number tokenization."""
        tokens = tokenize("0O 77O 377Q")
        assert tokens[0].value == 0
        assert tokens[1].value == 0o77
        assert tokens[2].value == 0o377

    def test_strings(self) -> None:
        """Test string literal tokenization."""
        tokens = tokenize("'hello' 'world'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        assert tokens[1].value == "world"

    def test_string_escaped_quote(self) -> None:
        """Test escaped quote in string."""
        tokens = tokenize("'it''s'")
        assert tokens[0].value == "it's"

    def test_operators(self) -> None:
        """Test operator tokenization."""
        tokens = tokenize("+ - * / < > <= >= <> =")
        types = [t.type for t in tokens]
        assert types == [
            TokenType.OP_PLUS,
            TokenType.OP_MINUS,
            TokenType.OP_STAR,
            TokenType.OP_SLASH,
            TokenType.OP_LT,
            TokenType.OP_GT,
            TokenType.OP_LE,
            TokenType.OP_GE,
            TokenType.OP_NE,
            TokenType.OP_EQ,
            TokenType.EOF,
        ]

    def test_delimiters(self) -> None:
        """Test delimiter tokenization."""
        tokens = tokenize("( ) , ; : .")
        types = [t.type for t in tokens]
        assert types == [
            TokenType.LPAREN,
            TokenType.RPAREN,
            TokenType.COMMA,
            TokenType.SEMICOLON,
            TokenType.COLON,
            TokenType.DOT,
            TokenType.EOF,
        ]

    def test_comments(self) -> None:
        """Test comment skipping."""
        tokens = tokenize("foo /* this is a comment */ bar")
        assert len(tokens) == 3
        assert tokens[0].value == "FOO"
        assert tokens[1].value == "BAR"

    def test_multiline_comment(self) -> None:
        """Test multiline comment."""
        source = """foo /* this
        is a
        multiline comment */ bar"""
        tokens = tokenize(source)
        assert len(tokens) == 3
        assert tokens[0].value == "FOO"
        assert tokens[1].value == "BAR"

    def test_line_tracking(self) -> None:
        """Test line number tracking."""
        source = """line1
        line2
        line3"""
        tokens = tokenize(source)
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_hello_world(self) -> None:
        """Test tokenizing a simple hello world program."""
        source = """
        HELLO: PROCEDURE BYTE;
            DECLARE MSG DATA ('HELLO$');
            RETURN 0;
        END HELLO;
        """
        tokens = tokenize(source)
        # Should tokenize without error
        assert tokens[-1].type == TokenType.EOF

    def test_declare_statement(self) -> None:
        """Test tokenizing DECLARE statements."""
        source = "DECLARE X BYTE, Y ADDRESS, Z(10) BYTE;"
        tokens = tokenize(source)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.DECLARE in types
        assert TokenType.BYTE in types
        assert TokenType.ADDRESS in types
