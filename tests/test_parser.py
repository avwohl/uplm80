"""Tests for the PL/M-80 parser."""

import pytest
from uplm80.parser import parse
from uplm80.ast_nodes import (
    Module,
    VarDecl,
    ProcDecl,
    LiterallyDecl,
    AssignStmt,
    IfStmt,
    DoWhileBlock,
    DoIterBlock,
    CallStmt,
    ReturnStmt,
    NumberLiteral,
    Identifier,
    BinaryExpr,
    UnaryExpr,
    DataType,
    BinaryOp,
    UnaryOp,
)


class TestParser:
    """Test cases for the parser."""

    def test_empty_module(self) -> None:
        """Test parsing empty module."""
        ast = parse("")
        assert isinstance(ast, Module)

    def test_simple_declare(self) -> None:
        """Test parsing simple DECLARE."""
        ast = parse("DECLARE X BYTE;")
        assert len(ast.decls) == 1
        decl = ast.decls[0]
        assert isinstance(decl, VarDecl)
        assert decl.name == "X"
        assert decl.data_type == DataType.BYTE

    def test_declare_address(self) -> None:
        """Test parsing ADDRESS declaration."""
        ast = parse("DECLARE PTR ADDRESS;")
        decl = ast.decls[0]
        assert isinstance(decl, VarDecl)
        assert decl.data_type == DataType.ADDRESS

    def test_declare_array(self) -> None:
        """Test parsing array declaration."""
        ast = parse("DECLARE BUF(100) BYTE;")
        decl = ast.decls[0]
        assert isinstance(decl, VarDecl)
        assert decl.dimension == 100

    def test_declare_multiple(self) -> None:
        """Test parsing multiple declarations."""
        ast = parse("DECLARE (A, B, C) BYTE;")
        assert len(ast.decls) == 3
        for decl in ast.decls:
            assert isinstance(decl, VarDecl)
            assert decl.data_type == DataType.BYTE

    def test_declare_literally(self) -> None:
        """Test parsing LITERALLY declaration."""
        ast = parse("DECLARE CR LITERALLY '13';")
        decl = ast.decls[0]
        assert isinstance(decl, LiterallyDecl)
        assert decl.name == "CR"
        assert decl.value == "13"

    def test_declare_initial(self) -> None:
        """Test parsing initialization."""
        ast = parse("DECLARE X BYTE INITIAL(42);")
        decl = ast.decls[0]
        assert isinstance(decl, VarDecl)
        assert decl.initial_values is not None
        assert len(decl.initial_values) == 1

    def test_simple_procedure(self) -> None:
        """Test parsing simple procedure."""
        source = """
        FOO: PROCEDURE;
        END FOO;
        """
        ast = parse(source)
        assert len(ast.decls) == 1
        proc = ast.decls[0]
        assert isinstance(proc, ProcDecl)
        assert proc.name == "FOO"

    def test_procedure_with_params(self) -> None:
        """Test parsing procedure with parameters."""
        source = """
        ADD: PROCEDURE(A, B) ADDRESS;
            DECLARE (A, B) ADDRESS;
            RETURN A + B;
        END ADD;
        """
        ast = parse(source)
        proc = ast.decls[0]
        assert isinstance(proc, ProcDecl)
        assert proc.params == ["A", "B"]
        assert proc.return_type == DataType.ADDRESS

    def test_assignment(self) -> None:
        """Test parsing assignment statement."""
        source = """
        TEST: PROCEDURE;
            DECLARE X BYTE;
            X = 42;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        assert isinstance(proc, ProcDecl)
        assert len(proc.stmts) == 1
        stmt = proc.stmts[0]
        assert isinstance(stmt, AssignStmt)

    def test_if_statement(self) -> None:
        """Test parsing IF statement."""
        source = """
        TEST: PROCEDURE;
            DECLARE X BYTE;
            IF X > 0 THEN X = 0;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.condition, BinaryExpr)
        assert stmt.condition.op == BinaryOp.GT

    def test_if_else(self) -> None:
        """Test parsing IF-ELSE statement."""
        source = """
        TEST: PROCEDURE;
            DECLARE X BYTE;
            IF X > 0 THEN X = 1; ELSE X = 0;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_stmt is not None

    def test_do_while(self) -> None:
        """Test parsing DO WHILE block."""
        source = """
        TEST: PROCEDURE;
            DECLARE X BYTE;
            DO WHILE X > 0;
                X = X - 1;
            END;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, DoWhileBlock)

    def test_do_iter(self) -> None:
        """Test parsing iterative DO block."""
        source = """
        TEST: PROCEDURE;
            DECLARE I BYTE;
            DO I = 0 TO 10;
                CALL FOO;
            END;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, DoIterBlock)

    def test_call_statement(self) -> None:
        """Test parsing CALL statement."""
        source = """
        TEST: PROCEDURE;
            CALL FOO(1, 2, 3);
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, CallStmt)
        assert isinstance(stmt.callee, Identifier)
        assert stmt.callee.name == "FOO"
        assert len(stmt.args) == 3

    def test_return_value(self) -> None:
        """Test parsing RETURN with value."""
        source = """
        TEST: PROCEDURE BYTE;
            RETURN 42;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 42

    def test_expression_precedence(self) -> None:
        """Test expression precedence."""
        source = """
        TEST: PROCEDURE;
            DECLARE X ADDRESS;
            X = 1 + 2 * 3;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, AssignStmt)
        expr = stmt.value
        # Should be ADD(1, MUL(2, 3)) not MUL(ADD(1, 2), 3)
        assert isinstance(expr, BinaryExpr)
        assert expr.op == BinaryOp.ADD
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == BinaryOp.MUL

    def test_not_relational_precedence(self) -> None:
        """NOT must bind looser than relational operators (PL/M-80 spec).

        ``NOT A < B`` should parse as ``NOT (A < B)``, not ``(NOT A) < B``.
        """
        source = """
        TEST: PROCEDURE;
            DECLARE (A, B, C) BYTE;
            C = NOT A < B;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        assert isinstance(proc, ProcDecl)
        stmt = proc.stmts[0]
        assert isinstance(stmt, AssignStmt)
        # Top-level expression should be NOT, with a relational underneath.
        assert isinstance(stmt.value, UnaryExpr)
        assert stmt.value.op == UnaryOp.NOT
        assert isinstance(stmt.value.operand, BinaryExpr)
        assert stmt.value.operand.op == BinaryOp.LT

    def test_not_binds_tighter_than_and(self) -> None:
        """NOT must bind tighter than AND: ``NOT A AND B`` -> ``(NOT A) AND B``."""
        source = """
        TEST: PROCEDURE;
            DECLARE (A, B, C) BYTE;
            C = NOT A AND B;
        END TEST;
        """
        ast = parse(source)
        proc = ast.decls[0]
        stmt = proc.stmts[0]
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.value, BinaryExpr)
        assert stmt.value.op == BinaryOp.AND
        assert isinstance(stmt.value.left, UnaryExpr)
        assert stmt.value.left.op == UnaryOp.NOT

    def test_module_structure(self) -> None:
        """Test parsing module structure."""
        source = """
        0100H:
        HELLO: DO;
            DECLARE MSG DATA ('HELLO$');
        END HELLO;
        EOF
        """
        ast = parse(source)
        assert ast.origin == 0x100
        assert ast.name == "HELLO"
