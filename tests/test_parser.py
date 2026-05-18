"""Tests for the PL/M-80 front-end (preprocess + uplox parse → typed AST).

These tests assert on the uplox-generated typed AST in
``uplm80._plm_parser``. Shape unpacking goes through the helpers in
``uplm80.ast_view`` so the tests stay readable and stable across grammar
tweaks. Each test exercises one PL/M-80 surface form and corresponds to
a case in the legacy hand-built-AST test list.
"""

from uplm80 import ast_view as V
from uplm80._plm_parser import (
    AssignStmt,
    BinaryOp,
    Call,
    CallNoArgs,
    CallStmt,
    DeclItem,
    DoIterBlock,
    DoWhileBlock,
    Identifier,
    IfStmt,
    IfStmtElse,
    LiterallyDecl,
    NumberLiteral,
    ProcDecl,
    ReturnStmtValue,
    UnaryOp,
)
from uplm80.frontend import parse_source as parse


def _proc_stmts(proc: ProcDecl) -> list:
    """Statements inside a procedure body, with DECLAREs filtered out."""
    _, stmts = V.proc_local_decls_stmts(proc)
    return stmts


class TestParser:
    """Test cases for the uplox-driven PL/M-80 parser."""

    def test_simple_declare(self) -> None:
        """Top-level DECLARE produces a single typed DeclItem."""
        m = parse("DECLARE X BYTE;")
        shape = V.module_shape(m)
        assert len(shape.decls) == 1
        decl = shape.decls[0]
        assert isinstance(decl, DeclItem)
        assert V.decl_item_names(decl) == ["X"]
        dt, dim = V.decl_item_type(decl)
        assert dt == V.DataType.BYTE
        assert dim is None

    def test_declare_address(self) -> None:
        """DECLARE ... ADDRESS lifts to DataType.ADDRESS."""
        m = parse("DECLARE PTR ADDRESS;")
        decl = V.module_shape(m).decls[0]
        assert isinstance(decl, DeclItem)
        dt, _ = V.decl_item_type(decl)
        assert dt == V.DataType.ADDRESS

    def test_declare_array(self) -> None:
        """Array DECLARE carries its dimension through ArraySize/SizeNumber."""
        m = parse("DECLARE BUF(100) BYTE;")
        decl = V.module_shape(m).decls[0]
        assert isinstance(decl, DeclItem)
        dt, dim = V.decl_item_type(decl)
        assert dt == V.DataType.BYTE
        assert dim == 100

    def test_declare_multiple(self) -> None:
        """``DECLARE (A, B, C) BYTE;`` produces one DeclItem with three names.

        The typed AST keeps the multi-name group together (under
        DeclNames) rather than expanding into one DeclItem per name; the
        names live inside the single item and are recovered via
        :func:`decl_item_names`.
        """
        m = parse("DECLARE (A, B, C) BYTE;")
        shape = V.module_shape(m)
        assert len(shape.decls) == 1
        decl = shape.decls[0]
        assert isinstance(decl, DeclItem)
        assert V.decl_item_names(decl) == ["A", "B", "C"]
        dt, _ = V.decl_item_type(decl)
        assert dt == V.DataType.BYTE

    def test_declare_literally(self) -> None:
        """LITERALLY declarations land as LiterallyDecl nodes."""
        m = parse("DECLARE CR LITERALLY '13';")
        decl = V.module_shape(m).decls[0]
        assert isinstance(decl, LiterallyDecl)
        assert V.ident_text(decl.name) == "CR"
        assert V.literally_value(decl) == "13"

    def test_declare_initial(self) -> None:
        """INITIAL(...) is exposed via DeclAttrs.initial_values."""
        m = parse("DECLARE X BYTE INITIAL(42);")
        decl = V.module_shape(m).decls[0]
        assert isinstance(decl, DeclItem)
        attrs = V.decl_attrs(decl)
        assert attrs.initial_values is not None
        assert len(attrs.initial_values) == 1
        assert isinstance(attrs.initial_values[0], NumberLiteral)
        assert V.number_value(attrs.initial_values[0]) == 42

    def test_simple_procedure(self) -> None:
        """``FOO: PROCEDURE; END FOO;`` parses to a ProcDecl named FOO."""
        m = parse(
            """
            FOO: PROCEDURE;
            END FOO;
            """
        )
        shape = V.module_shape(m)
        assert len(shape.decls) == 1
        proc = shape.decls[0]
        assert isinstance(proc, ProcDecl)
        assert V.proc_name(proc) == "FOO"

    def test_procedure_with_params(self) -> None:
        """Param names + ADDRESS return type round-trip through ProcSignature."""
        m = parse(
            """
            ADD: PROCEDURE(A, B) ADDRESS;
                DECLARE (A, B) ADDRESS;
                RETURN A + B;
            END ADD;
            """
        )
        proc = V.module_shape(m).decls[0]
        assert isinstance(proc, ProcDecl)
        assert V.proc_param_names(proc) == ["A", "B"]
        assert V.proc_return_type(proc) == V.DataType.ADDRESS

    def test_assignment(self) -> None:
        """Bare ``X = 42;`` parses to AssignStmt with one target."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE X BYTE;
                X = 42;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmts = _proc_stmts(proc)
        assert len(stmts) == 1
        stmt = stmts[0]
        assert isinstance(stmt, AssignStmt)
        assert len(stmt.targets) == 1
        target = stmt.targets[0]
        assert isinstance(target, Identifier)
        assert V.ident_text(target.name) == "X"
        assert isinstance(stmt.value, NumberLiteral)
        assert V.number_value(stmt.value) == 42

    def test_if_statement(self) -> None:
        """IF without ELSE is the IfStmt variant; condition is a BinaryOp."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE X BYTE;
                IF X > 0 THEN X = 0;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, IfStmt)
        assert not isinstance(stmt, IfStmtElse)
        assert isinstance(stmt.condition, BinaryOp)
        assert V.binop_kind(stmt.condition) == V.BinaryOpKind.GT

    def test_if_else(self) -> None:
        """IF ... THEN ... ELSE ...; is the IfStmtElse variant."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE X BYTE;
                IF X > 0 THEN X = 1; ELSE X = 0;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, IfStmtElse)
        assert stmt.else_stmt is not None

    def test_do_while(self) -> None:
        """DO WHILE ...; END; parses to a DoWhileBlock."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE X BYTE;
                DO WHILE X > 0;
                    X = X - 1;
                END;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, DoWhileBlock)

    def test_do_iter(self) -> None:
        """``DO I = 0 TO 10`` is a DoIterBlock (no step)."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE I BYTE;
                DO I = 0 TO 10;
                    CALL FOO;
                END;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, DoIterBlock)
        assert V.ident_text(stmt.index) == "I"

    def test_call_statement(self) -> None:
        """``CALL FOO(1, 2, 3);`` parses to CallStmt whose callee is a Call."""
        m = parse(
            """
            TEST: PROCEDURE;
                CALL FOO(1, 2, 3);
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, CallStmt)
        callee = stmt.callee
        # Argful call wraps the target identifier in a Call node carrying args.
        assert isinstance(callee, Call)
        assert isinstance(callee.callee, Identifier)
        assert V.ident_text(callee.callee.name) == "FOO"
        assert len(callee.args) == 3
        assert [V.number_value(a) for a in callee.args] == [1, 2, 3]

    def test_call_no_args(self) -> None:
        """``CALL FOO;`` (no parens) leaves the callee as a bare Identifier.

        ``CALL FOO();`` (empty arg list) becomes a CallNoArgs wrapper.
        Both shapes are accepted by codegen.
        """
        m = parse(
            """
            TEST: PROCEDURE;
                CALL FOO;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, CallStmt)
        assert isinstance(stmt.callee, (Identifier, CallNoArgs))

    def test_return_value(self) -> None:
        """``RETURN 42;`` parses to ReturnStmtValue with a NumberLiteral."""
        m = parse(
            """
            TEST: PROCEDURE BYTE;
                RETURN 42;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, ReturnStmtValue)
        assert isinstance(stmt.value, NumberLiteral)
        assert V.number_value(stmt.value) == 42

    def test_expression_precedence(self) -> None:
        """``1 + 2 * 3`` parses as ADD(1, MUL(2, 3)) — * binds tighter than +."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE X ADDRESS;
                X = 1 + 2 * 3;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, AssignStmt)
        expr = stmt.value
        assert isinstance(expr, BinaryOp)
        assert V.binop_kind(expr) == V.BinaryOpKind.ADD
        assert isinstance(expr.right, BinaryOp)
        assert V.binop_kind(expr.right) == V.BinaryOpKind.MUL

    def test_not_relational_precedence(self) -> None:
        """NOT binds looser than relational: ``NOT A < B`` -> ``NOT (A < B)``."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE (A, B, C) BYTE;
                C = NOT A < B;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        assert isinstance(proc, ProcDecl)
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.value, UnaryOp)
        assert V.unop_kind(stmt.value) == V.UnaryOpKind.NOT
        assert isinstance(stmt.value.operand, BinaryOp)
        assert V.binop_kind(stmt.value.operand) == V.BinaryOpKind.LT

    def test_not_binds_tighter_than_and(self) -> None:
        """NOT binds tighter than AND: ``NOT A AND B`` -> ``(NOT A) AND B``."""
        m = parse(
            """
            TEST: PROCEDURE;
                DECLARE (A, B, C) BYTE;
                C = NOT A AND B;
            END TEST;
            """
        )
        proc = V.module_shape(m).decls[0]
        stmt = _proc_stmts(proc)[0]
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.value, BinaryOp)
        assert V.binop_kind(stmt.value) == V.BinaryOpKind.AND
        assert isinstance(stmt.value.left, UnaryOp)
        assert V.unop_kind(stmt.value.left) == V.UnaryOpKind.NOT

    def test_module_structure(self) -> None:
        """A leading origin literal and a labeled top-level DO surface as
        ``ModuleShape(origin=..., name=...)``."""
        source = """
        0100H:
        HELLO: DO;
            DECLARE MSG DATA ('HELLO$');
        END HELLO;
        EOF
        """
        m = parse(source)
        shape = V.module_shape(m)
        assert shape.origin == 0x100
        assert shape.name == "HELLO"
        assert len(shape.decls) == 1
        decl = shape.decls[0]
        assert isinstance(decl, DeclItem)
        assert V.decl_item_names(decl) == ["MSG"]
