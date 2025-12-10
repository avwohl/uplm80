"""
PL/M-80 Parser.

Recursive descent parser that converts tokens into an AST.
"""

from typing import Callable
from .tokens import Token, TokenType
from .lexer import Lexer
from .errors import ParserError, SourceLocation
from .ast_nodes import (
    ASTNode,
    SourceSpan,
    DataType,
    BinaryOp,
    UnaryOp,
    Expr,
    NumberLiteral,
    StringLiteral,
    Identifier,
    SubscriptExpr,
    MemberExpr,
    CallExpr,
    BinaryExpr,
    UnaryExpr,
    LocationExpr,
    ConstListExpr,
    EmbeddedAssignExpr,
    Stmt,
    AssignStmt,
    CallStmt,
    ReturnStmt,
    GotoStmt,
    HaltStmt,
    EnableStmt,
    DisableStmt,
    NullStmt,
    LabeledStmt,
    IfStmt,
    DoBlock,
    DoWhileBlock,
    DoIterBlock,
    DoCaseBlock,
    Declaration,
    StructMember,
    VarDecl,
    LabelDecl,
    LiterallyDecl,
    ProcDecl,
    DeclareStmt,
    Module,
)


class Parser:
    """Recursive descent parser for PL/M-80."""

    def __init__(self, tokens: list[Token], filename: str = "<input>") -> None:
        self.tokens = tokens
        self.filename = filename
        self.pos = 0
        self.current = tokens[0] if tokens else Token(TokenType.EOF, None, 1, 1, "")
        # Macro table for LITERALLY substitutions
        self.macros: dict[str, str] = {}

    @staticmethod
    def _parse_plm_number(s: str) -> int:
        """Parse a PL/M-style numeric literal (handles $ separators and B/H/O/Q/D suffixes)."""
        # Remove $ digit separators and convert to uppercase
        s = s.upper().replace("$", "")
        # Handle suffixes for different bases
        if s.endswith("H"):
            return int(s[:-1], 16)
        elif s.endswith("B"):
            return int(s[:-1], 2)
        elif s.endswith("O") or s.endswith("Q"):
            return int(s[:-1], 8)
        elif s.endswith("D"):
            return int(s[:-1], 10)
        else:
            # Default: try as Python literal (handles 0x, 0b, 0o prefixes)
            return int(s, 0)

    def _error(self, message: str) -> ParserError:
        """Create a parser error at current position."""
        return ParserError(
            message,
            SourceLocation(self.current.line, self.current.column, self.filename),
        )

    def _advance(self) -> Token:
        """Advance to next token and return the previous one."""
        prev = self.current
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        return prev

    def _peek(self, offset: int = 0) -> Token:
        """Peek at token at current position + offset."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def _check(self, *types: TokenType) -> bool:
        """Check if current token is one of the given types."""
        return self.current.type in types

    def _match(self, *types: TokenType) -> Token | None:
        """If current token matches, advance and return it."""
        if self._check(*types):
            return self._advance()
        return None

    def _expect(self, token_type: TokenType, message: str | None = None) -> Token:
        """Expect current token to be of given type, advance and return it."""
        if not self._check(token_type):
            msg = message or f"Expected {token_type.name}"
            raise self._error(msg)
        return self._advance()

    def _expect_number_or_macro(self, error_msg: str) -> int:
        """Expect a number literal or a LITERALLY macro that expands to a number."""
        if self._check(TokenType.NUMBER):
            return self._advance().value
        elif self._check(TokenType.IDENTIFIER):
            name = self._advance().value
            if name in self.macros:
                try:
                    return self._parse_plm_number(self.macros[name])
                except ValueError:
                    raise self._error(f"Macro '{name}' does not expand to a number")
            else:
                raise self._error(f"Unknown identifier '{name}' (expected number or macro)")
        else:
            raise self._error(error_msg)

    def _span_from(self, start: Token) -> SourceSpan:
        """Create a source span from start token to current position."""
        prev = self._peek(-1) if self.pos > 0 else start
        return SourceSpan(
            start.line, start.column, prev.line, prev.column + len(prev.lexeme), self.filename
        )

    def _is_literally_macro(self) -> bool:
        """Check if current identifier is a macro that expands to LITERALLY."""
        if self._check(TokenType.IDENTIFIER):
            name = self.current.value
            if name in self.macros:
                expansion = self.macros[name].strip().upper()
                if expansion == "LITERALLY":
                    self._advance()  # consume the macro identifier
                    return True
        return False

    def _is_dcl_abbreviation(self) -> bool:
        """Check if current identifier is DCL (abbreviation for DECLARE) or a LITERALLY macro expanding to 'DECLARE'."""
        if self._check(TokenType.IDENTIFIER):
            name = self.current.value
            if name == "DCL":
                return True
            # Also check if it's a LITERALLY macro that expands to 'DECLARE'
            if name in self.macros:
                expansion = self.macros[name].strip().strip("'").upper()
                if expansion == "DECLARE":
                    return True
        return False

    def _is_proc_abbreviation(self) -> bool:
        """Check if current identifier is a LITERALLY macro expanding to 'PROCEDURE'."""
        if self._check(TokenType.IDENTIFIER):
            name = self.current.value
            if name in self.macros:
                expansion = self.macros[name].strip().strip("'").upper()
                if expansion == "PROCEDURE":
                    return True
        return False

    def _is_lit_abbreviation(self) -> bool:
        """Check if current identifier is LIT (abbreviation for LITERALLY)."""
        if self._check(TokenType.IDENTIFIER) and self.current.value == "LIT":
            self._advance()  # consume LIT
            return True
        return False

    # ========================================================================
    # Expression Parsing (Precedence Climbing)
    # ========================================================================

    def _parse_primary(self) -> Expr:
        """Parse a primary expression."""
        token = self.current

        # Number literal
        if self._match(TokenType.NUMBER):
            return NumberLiteral(token.value, span=self._span_from(token))

        # String literal
        if self._match(TokenType.STRING):
            bytes_val = [ord(c) for c in token.value]
            return StringLiteral(token.value, bytes_val, span=self._span_from(token))

        # Parenthesized expression or embedded assignment
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            # Check for embedded assignment :=
            if self._match(TokenType.COLON):
                self._expect(TokenType.OP_EQ, "Expected '=' after ':' in embedded assignment")
                value = self._parse_expression()
                self._expect(TokenType.RPAREN, "Expected ')' after embedded assignment")
                return EmbeddedAssignExpr(expr, value, span=self._span_from(token))
            self._expect(TokenType.RPAREN, "Expected ')' after expression")
            return expr

        # Location reference: .variable or .(const, ...)
        if self._match(TokenType.DOT):
            if self._match(TokenType.LPAREN):
                # Constant list .(c1, c2, ...)
                values: list[Expr] = []
                if not self._check(TokenType.RPAREN):
                    values.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        values.append(self._parse_expression())
                self._expect(TokenType.RPAREN, "Expected ')' after constant list")
                return ConstListExpr(values, span=self._span_from(token))
            else:
                # Location of variable
                operand = self._parse_primary()
                return LocationExpr(operand, span=self._span_from(token))

        # Identifier (variable, procedure call, or builtin)
        if self._match(TokenType.IDENTIFIER):
            name = token.value

            # Check if this identifier is a LITERALLY macro that needs expansion
            if name in self.macros:
                macro_value = self.macros[name]
                # Check if the macro value is a simple numeric literal or complex expression
                try:
                    # Try parsing as a PL/M number (handles B/H/O/Q/D suffixes and $ separators)
                    val = self._parse_plm_number(macro_value)
                    # It's a simple number - just create number literal
                    return NumberLiteral(val, span=self._span_from(token))
                except ValueError:
                    # Not a simple number - try parsing as expression
                    # Re-lex and re-parse the macro value
                    lexer = Lexer(macro_value, f"<macro:{name}>")
                    macro_tokens = lexer.tokenize()
                    # Create a sub-parser with the same macro table
                    sub_parser = Parser(macro_tokens, f"<macro:{name}>")
                    sub_parser.macros = self.macros.copy()
                    try:
                        expanded_expr = sub_parser._parse_expression()
                        # If it's just an identifier, use it as the base and continue
                        # to parse subscripts/calls with the main parser
                        if isinstance(expanded_expr, Identifier):
                            # Use the expanded identifier name
                            name = expanded_expr.name
                            # Fall through to normal processing below
                        else:
                            # Complex expression - return it directly
                            return expanded_expr
                    except ParserError:
                        # If parsing fails, fall through to normal identifier handling
                        pass

            expr: Expr = Identifier(name, span=self._span_from(token))

            # Check for subscript or member access or call
            while True:
                if self._match(TokenType.LPAREN):
                    # Could be subscript or function call
                    args: list[Expr] = []
                    if not self._check(TokenType.RPAREN):
                        args.append(self._parse_argument_or_embedded_assign())
                        while self._match(TokenType.COMMA):
                            args.append(self._parse_argument_or_embedded_assign())
                    self._expect(TokenType.RPAREN, "Expected ')' after arguments")
                    # All parenthesized expressions become CallExpr
                    # The codegen will handle array subscripts vs procedure calls
                    # by looking up the symbol type
                    expr = CallExpr(expr, args, span=self._span_from(token))
                elif self._match(TokenType.DOT):
                    # Member access
                    member_tok = self._expect(TokenType.IDENTIFIER, "Expected member name after '.'")
                    member_name = member_tok.value
                    expr = MemberExpr(expr, member_name, span=self._span_from(token))
                else:
                    break

            return expr

        # Built-in functions that look like keywords
        builtins = {
            TokenType.PLUS,
            TokenType.MINUS,
        }
        # Handle these separately if needed

        raise self._error(f"Expected expression, got {self.current.type.name}")

    def _parse_unary(self) -> Expr:
        """Parse unary expression."""
        token = self.current

        # Unary minus
        if self._match(TokenType.OP_MINUS):
            operand = self._parse_unary()
            return UnaryExpr(UnaryOp.NEG, operand, span=self._span_from(token))

        # NOT
        if self._match(TokenType.NOT):
            operand = self._parse_unary()
            return UnaryExpr(UnaryOp.NOT, operand, span=self._span_from(token))

        return self._parse_primary()

    def _parse_multiplicative(self) -> Expr:
        """Parse multiplicative expression (* / MOD)."""
        left = self._parse_unary()

        while True:
            token = self.current
            if self._match(TokenType.OP_STAR):
                right = self._parse_unary()
                left = BinaryExpr(BinaryOp.MUL, left, right, span=self._span_from(token))
            elif self._match(TokenType.OP_SLASH):
                right = self._parse_unary()
                left = BinaryExpr(BinaryOp.DIV, left, right, span=self._span_from(token))
            elif self._match(TokenType.MOD):
                right = self._parse_unary()
                left = BinaryExpr(BinaryOp.MOD, left, right, span=self._span_from(token))
            else:
                break

        return left

    def _parse_additive(self) -> Expr:
        """Parse additive expression (+ - PLUS MINUS)."""
        left = self._parse_multiplicative()

        while True:
            token = self.current
            if self._match(TokenType.OP_PLUS):
                right = self._parse_multiplicative()
                left = BinaryExpr(BinaryOp.ADD, left, right, span=self._span_from(token))
            elif self._match(TokenType.OP_MINUS):
                right = self._parse_multiplicative()
                left = BinaryExpr(BinaryOp.SUB, left, right, span=self._span_from(token))
            elif self._match(TokenType.PLUS):
                right = self._parse_multiplicative()
                left = BinaryExpr(BinaryOp.PLUS, left, right, span=self._span_from(token))
            elif self._match(TokenType.MINUS):
                right = self._parse_multiplicative()
                left = BinaryExpr(BinaryOp.MINUS, left, right, span=self._span_from(token))
            else:
                break

        return left

    def _parse_relational(self) -> Expr:
        """Parse relational expression (< > <= >= = <>)."""
        left = self._parse_additive()

        token = self.current
        if self._match(TokenType.OP_LT):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.LT, left, right, span=self._span_from(token))
        elif self._match(TokenType.OP_GT):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.GT, left, right, span=self._span_from(token))
        elif self._match(TokenType.OP_LE):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.LE, left, right, span=self._span_from(token))
        elif self._match(TokenType.OP_GE):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.GE, left, right, span=self._span_from(token))
        elif self._match(TokenType.OP_EQ):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.EQ, left, right, span=self._span_from(token))
        elif self._match(TokenType.OP_NE):
            right = self._parse_additive()
            return BinaryExpr(BinaryOp.NE, left, right, span=self._span_from(token))

        return left

    def _parse_and(self) -> Expr:
        """Parse AND expression."""
        left = self._parse_relational()

        while True:
            token = self.current
            if self._match(TokenType.AND):
                right = self._parse_relational()
                left = BinaryExpr(BinaryOp.AND, left, right, span=self._span_from(token))
            else:
                break

        return left

    def _parse_or_xor(self) -> Expr:
        """Parse OR/XOR expression."""
        left = self._parse_and()

        while True:
            token = self.current
            if self._match(TokenType.OR):
                right = self._parse_and()
                left = BinaryExpr(BinaryOp.OR, left, right, span=self._span_from(token))
            elif self._match(TokenType.XOR):
                right = self._parse_and()
                left = BinaryExpr(BinaryOp.XOR, left, right, span=self._span_from(token))
            else:
                break

        return left

    def _parse_expression(self) -> Expr:
        """Parse a full expression."""
        return self._parse_or_xor()

    def _parse_expression_with_embedded_assign(self) -> Expr:
        """Parse an expression which may contain an embedded assignment at top level."""
        start_token = self.current
        expr = self._parse_expression()
        # Check for embedded assignment :=
        if self._check(TokenType.COLON) and self._peek(1).type == TokenType.OP_EQ:
            self._advance()  # consume :
            self._advance()  # consume =
            value = self._parse_expression_with_embedded_assign()  # Allow chaining
            return EmbeddedAssignExpr(expr, value, span=self._span_from(start_token))
        return expr

    def _parse_argument_or_embedded_assign(self) -> Expr:
        """Parse a function argument which may be an embedded assignment."""
        return self._parse_expression_with_embedded_assign()

    # ========================================================================
    # Statement Parsing
    # ========================================================================

    def _parse_statement(self) -> Stmt:
        """Parse a statement."""
        token = self.current

        # Check for label definition (IDENTIFIER:)
        if self._check(TokenType.IDENTIFIER) and self._peek(1).type == TokenType.COLON:
            # Check if this is a procedure definition
            # Either PROCEDURE keyword or a LITERALLY macro expanding to 'PROCEDURE' (e.g., PROC)
            peek2 = self._peek(2)
            is_proc_def = peek2.type == TokenType.PROCEDURE
            if not is_proc_def and peek2.type == TokenType.IDENTIFIER:
                macro_name = peek2.value
                if macro_name in self.macros:
                    expansion = self.macros[macro_name].strip().strip("'").upper()
                    if expansion == "PROCEDURE":
                        is_proc_def = True
            if is_proc_def:
                return self._parse_procedure_as_stmt()
            label_tok = self._advance()
            self._advance()  # consume colon
            stmt = self._parse_statement()
            return LabeledStmt(label_tok.value, stmt, span=self._span_from(label_tok))

        # IF statement
        if self._match(TokenType.IF):
            return self._parse_if()

        # DO block
        if self._match(TokenType.DO):
            return self._parse_do_block(token)

        # CALL statement
        if self._match(TokenType.CALL):
            return self._parse_call_stmt(token)

        # RETURN statement
        if self._match(TokenType.RETURN):
            return self._parse_return(token)

        # GOTO / GO TO statement
        if self._match(TokenType.GOTO):
            target = self._expect(TokenType.IDENTIFIER, "Expected label after GOTO")
            self._expect(TokenType.SEMICOLON, "Expected ';' after GOTO")
            return GotoStmt(target.value, span=self._span_from(token))
        if self._match(TokenType.GO):
            self._expect(TokenType.TO, "Expected 'TO' after 'GO'")
            target = self._expect(TokenType.IDENTIFIER, "Expected label after GO TO")
            self._expect(TokenType.SEMICOLON, "Expected ';' after GO TO")
            return GotoStmt(target.value, span=self._span_from(token))

        # HALT statement
        if self._match(TokenType.HALT):
            self._expect(TokenType.SEMICOLON, "Expected ';' after HALT")
            return HaltStmt(span=self._span_from(token))

        # ENABLE statement
        if self._match(TokenType.ENABLE):
            self._expect(TokenType.SEMICOLON, "Expected ';' after ENABLE")
            return EnableStmt(span=self._span_from(token))

        # DISABLE statement
        if self._match(TokenType.DISABLE):
            self._expect(TokenType.SEMICOLON, "Expected ';' after DISABLE")
            return DisableStmt(span=self._span_from(token))

        # Null statement (just semicolon)
        if self._match(TokenType.SEMICOLON):
            return NullStmt(span=self._span_from(token))

        # DECLARE statement (can appear in statement position)
        # Also handle DCL abbreviation
        if self._check(TokenType.DECLARE) or self._is_dcl_abbreviation():
            decls = self._parse_declare()
            return DeclareStmt(decls, span=self._span_from(token))

        # END statement (handled by block parsing)
        if self._check(TokenType.END):
            # This shouldn't be reached in normal flow
            raise self._error("Unexpected END statement")

        # Assignment statement (expression = expression)
        return self._parse_assignment_or_call(token)

    def _parse_lvalue(self) -> Expr:
        """Parse an lvalue (left side of assignment) - no relational operators."""
        # An lvalue is a variable reference, possibly with subscripts/members
        return self._parse_primary()

    def _parse_assignment_or_call(self, start_token: Token) -> Stmt:
        """Parse an assignment statement or expression statement."""
        # First, try to parse as assignment: target [, target]* = expr;
        # We parse the first lvalue, then check for = or ,

        # Parse first target (lvalue - no relational operators)
        first = self._parse_lvalue()
        targets: list[Expr] = [first]

        # Check for multiple assignment targets
        while self._match(TokenType.COMMA):
            targets.append(self._parse_lvalue())

        # Check for assignment
        if self._match(TokenType.OP_EQ):
            value = self._parse_expression()
            self._expect(TokenType.SEMICOLON, "Expected ';' after assignment")
            return AssignStmt(targets, value, span=self._span_from(start_token))

        # Otherwise it must be a call (procedure call as statement)
        if len(targets) == 1:
            target = targets[0]
            if isinstance(target, CallExpr):
                self._expect(TokenType.SEMICOLON, "Expected ';' after call")
                return CallStmt(target.callee, target.args, span=self._span_from(start_token))
            elif isinstance(target, SubscriptExpr):
                # Single arg call looks like subscript: FOO(arg)
                self._expect(TokenType.SEMICOLON, "Expected ';' after call")
                return CallStmt(target.base, [target.index], span=self._span_from(start_token))
            elif isinstance(target, Identifier):
                # Could be a no-arg procedure call
                self._expect(TokenType.SEMICOLON, "Expected ';' after call")
                return CallStmt(target, [], span=self._span_from(start_token))

        raise self._error("Expected assignment or call statement")

    def _parse_call_stmt(self, start_token: Token) -> Stmt:
        """Parse CALL statement."""
        callee = self._parse_expression()
        # Handle CALL name(args);
        args: list[Expr] = []
        if isinstance(callee, CallExpr):
            args = callee.args
            callee = callee.callee
        elif isinstance(callee, SubscriptExpr):
            # CALL name(single_arg) looks like subscript
            args = [callee.index]
            callee = callee.base
        self._expect(TokenType.SEMICOLON, "Expected ';' after CALL")
        return CallStmt(callee, args, span=self._span_from(start_token))

    def _parse_return(self, start_token: Token) -> Stmt:
        """Parse RETURN statement."""
        value: Expr | None = None
        if not self._check(TokenType.SEMICOLON):
            value = self._parse_expression_with_embedded_assign()
        self._expect(TokenType.SEMICOLON, "Expected ';' after RETURN")
        return ReturnStmt(value, span=self._span_from(start_token))

    def _parse_if(self) -> Stmt:
        """Parse IF statement."""
        start_token = self._peek(-1)
        condition = self._parse_expression()
        self._expect(TokenType.THEN, "Expected THEN after IF condition")

        then_stmt = self._parse_statement()

        else_stmt: Stmt | None = None
        if self._match(TokenType.ELSE):
            else_stmt = self._parse_statement()

        return IfStmt(condition, then_stmt, else_stmt, span=self._span_from(start_token))

    def _parse_do_block(self, start_token: Token) -> Stmt:
        """Parse DO block (simple, WHILE, iterative, or CASE)."""
        # DO WHILE
        if self._match(TokenType.WHILE):
            condition = self._parse_expression()
            self._expect(TokenType.SEMICOLON, "Expected ';' after DO WHILE condition")
            stmts = self._parse_block_body()
            end_label = self._parse_end()
            return DoWhileBlock(condition, stmts, end_label, span=self._span_from(start_token))

        # DO FOREVER (built-in, equivalent to DO WHILE TRUE)
        # FOREVER is not a reserved word, but when used after DO, it means WHILE TRUE
        if (
            self._check(TokenType.IDENTIFIER)
            and self.current.value == "FOREVER"
            and self._peek(1).type == TokenType.SEMICOLON
        ):
            self._advance()  # consume FOREVER
            self._expect(TokenType.SEMICOLON, "Expected ';' after DO FOREVER")
            condition = NumberLiteral(1, span=self._span_from(start_token))
            stmts = self._parse_block_body()
            end_label = self._parse_end()
            return DoWhileBlock(condition, stmts, end_label, span=self._span_from(start_token))

        # Check for LITERALLY macro that expands to WHILE TRUE (e.g., FOREVER)
        if self._check(TokenType.IDENTIFIER):
            macro_name = self.current.value
            if macro_name in self.macros:
                expansion = self.macros[macro_name].strip().upper()
                if expansion.startswith("WHILE"):
                    # Handle DO FOREVER -> DO WHILE TRUE
                    self._advance()  # consume the macro identifier
                    # Parse the rest of the expansion - condition comes from macro
                    # For 'WHILE TRUE', the condition is TRUE
                    if expansion == "WHILE TRUE":
                        condition = NumberLiteral(1, span=self._span_from(start_token))
                    else:
                        # Try to parse any expression after WHILE in the expansion
                        # This is a simplified approach
                        condition = NumberLiteral(1, span=self._span_from(start_token))
                    self._expect(TokenType.SEMICOLON, "Expected ';' after DO WHILE")
                    stmts = self._parse_block_body()
                    end_label = self._parse_end()
                    return DoWhileBlock(condition, stmts, end_label, span=self._span_from(start_token))

        # DO CASE
        if self._match(TokenType.CASE):
            selector = self._parse_expression()
            self._expect(TokenType.SEMICOLON, "Expected ';' after DO CASE selector")
            cases = self._parse_case_body()
            end_label = self._parse_end()
            return DoCaseBlock(selector, cases, end_label, span=self._span_from(start_token))

        # Check for iterative DO: DO var = start TO bound [BY step]
        if self._check(TokenType.IDENTIFIER) and self._peek(1).type == TokenType.OP_EQ:
            index_tok = self._advance()
            self._advance()  # consume =
            start = self._parse_expression()
            self._expect(TokenType.TO, "Expected TO in iterative DO")
            bound = self._parse_expression()
            step: Expr | None = None
            if self._match(TokenType.BY):
                step = self._parse_expression()
            self._expect(TokenType.SEMICOLON, "Expected ';' after iterative DO header")
            stmts = self._parse_block_body()
            end_label = self._parse_end()
            index_var = Identifier(index_tok.value, span=self._span_from(index_tok))
            return DoIterBlock(
                index_var, start, bound, step, stmts, end_label, span=self._span_from(start_token)
            )

        # Simple DO block
        self._expect(TokenType.SEMICOLON, "Expected ';' after DO")
        decls, stmts = self._parse_do_body()
        end_label = self._parse_end()
        return DoBlock(decls, stmts, end_label, span=self._span_from(start_token))

    def _parse_block_body(self) -> list[Stmt]:
        """Parse statements until END."""
        stmts: list[Stmt] = []
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            stmts.append(self._parse_statement())
        return stmts

    def _parse_do_body(self) -> tuple[list[Declaration], list[Stmt]]:
        """Parse declarations and statements in a DO block."""
        decls: list[Declaration] = []
        stmts: list[Stmt] = []

        # Parse declarations first
        while self._check(TokenType.DECLARE):
            decls.extend(self._parse_declare())

        # Then statements
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            stmts.append(self._parse_statement())

        return decls, stmts

    def _parse_case_body(self) -> list[list[Stmt]]:
        """Parse case alternatives in DO CASE."""
        cases: list[list[Stmt]] = []
        while not self._check(TokenType.END) and not self._check(TokenType.EOF):
            # Each case is a single statement (often a DO block)
            case_stmt = self._parse_statement()
            cases.append([case_stmt])
        return cases

    def _parse_end(self) -> str | None:
        """Parse END [label];"""
        self._expect(TokenType.END, "Expected END")
        label: str | None = None
        if self._match(TokenType.IDENTIFIER):
            label = self._peek(-1).value
        self._expect(TokenType.SEMICOLON, "Expected ';' after END")
        return label

    # ========================================================================
    # Declaration Parsing
    # ========================================================================

    def _parse_declare(self) -> list[Declaration]:
        """Parse DECLARE statement (also handles DCL abbreviation)."""
        if self._check(TokenType.DECLARE):
            self._advance()
        elif self._check(TokenType.IDENTIFIER) and self.current.value == "DCL":
            self._advance()  # consume DCL
        else:
            raise self._error("Expected DECLARE or DCL")
        decls: list[Declaration] = []

        # Parse declaration element list
        while True:
            decl = self._parse_declare_element()
            if isinstance(decl, list):
                decls.extend(decl)
            else:
                decls.append(decl)

            if not self._match(TokenType.COMMA):
                break

        self._expect(TokenType.SEMICOLON, "Expected ';' after DECLARE")
        return decls

    def _parse_declare_element(self) -> Declaration | list[Declaration]:
        """Parse a single declaration element."""
        # Check for factored declaration: (name [BASED x], name [BASED x], ...) type
        if self._match(TokenType.LPAREN):
            # Parse list of names, each potentially with BASED clause
            name_infos: list[tuple[str, str | None, str | None]] = []  # (name, based_on, based_member)

            while True:
                name = self._expect(TokenType.IDENTIFIER, "Expected identifier").value
                based_on: str | None = None
                based_member: str | None = None

                # Check for BASED clause on this name
                if self._match(TokenType.BASED):
                    based_tok = self._expect(TokenType.IDENTIFIER, "Expected identifier after BASED")
                    based_on = based_tok.value
                    if self._match(TokenType.DOT):
                        member_tok = self._expect(TokenType.IDENTIFIER, "Expected member name")
                        based_member = member_tok.value

                name_infos.append((name, based_on, based_member))

                if not self._match(TokenType.COMMA):
                    break

            self._expect(TokenType.RPAREN, "Expected ')' after identifier list")

            # Parse common attributes (dimension, type)
            dimension: int | None = None
            if self._match(TokenType.LPAREN):
                if self._match(TokenType.OP_STAR):
                    dimension = -1  # Implicit dimension
                elif self._check(TokenType.NUMBER):
                    dim_tok = self._advance()
                    dimension = dim_tok.value
                elif self._check(TokenType.IDENTIFIER):
                    # Could be a LITERALLY macro
                    dim_name = self._advance().value
                    if dim_name in self.macros:
                        try:
                            # Use PL/M number parser to handle H/B/O/Q/D suffixes
                            dimension = self._parse_plm_number(self.macros[dim_name])
                        except ValueError:
                            dimension = -2
                    else:
                        dimension = -2
                else:
                    raise self._error("Expected dimension")
                self._expect(TokenType.RPAREN, "Expected ')' after dimension")

            # Check for LABEL type in factored declaration
            if self._match(TokenType.LABEL):
                attrs = self._parse_var_attributes()
                decls: list[Declaration] = []
                for name, _, _ in name_infos:  # Ignore BASED for labels
                    decl = LabelDecl(
                        name=name,
                        is_public=attrs.get("public", False),
                        is_external=attrs.get("external", False),
                    )
                    decls.append(decl)
                return decls

            data_type = self._parse_type()
            attrs = self._parse_var_attributes()
            init_values = self._parse_initialization()

            # Create a declaration for each name
            decls = []
            for name, based_on, based_member in name_infos:
                decl = VarDecl(
                    name=name,
                    data_type=data_type,
                    based_on=based_on,
                    based_member=based_member,
                    dimension=dimension if dimension != -1 else None,
                    is_public=attrs.get("public", False),
                    is_external=attrs.get("external", False),
                    at_location=attrs.get("at"),
                    initial_values=init_values,
                )
                decls.append(decl)
            return decls

        # Single declaration
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected identifier")
        name = name_tok.value

        # Check for LITERALLY (macro) - also check for macro-defined LITERALLY aliases and LIT abbreviation
        if self._match(TokenType.LITERALLY) or self._is_literally_macro() or self._is_lit_abbreviation():
            value_tok = self._expect(TokenType.STRING, "Expected string after LITERALLY")
            # Register macro for expansion during parsing
            self.macros[name] = value_tok.value
            return LiterallyDecl(name, value_tok.value)

        # Check for LABEL
        if self._match(TokenType.LABEL):
            is_public = bool(self._match(TokenType.PUBLIC))
            is_external = bool(self._match(TokenType.EXTERNAL))
            return LabelDecl(name, is_public, is_external)

        # Check for BASED
        based_on: str | None = None
        based_member: str | None = None
        if self._match(TokenType.BASED):
            based_tok = self._expect(TokenType.IDENTIFIER, "Expected identifier after BASED")
            based_on = based_tok.value
            if self._match(TokenType.DOT):
                member_tok = self._expect(TokenType.IDENTIFIER, "Expected member name")
                based_member = member_tok.value

        # Parse dimension
        dimension: int | None = None
        if self._match(TokenType.LPAREN):
            if self._match(TokenType.OP_STAR):
                dimension = -1  # Implicit
            elif self._check(TokenType.NUMBER):
                dim_tok = self._advance()
                dimension = dim_tok.value
            elif self._check(TokenType.IDENTIFIER):
                # Could be a LITERALLY macro - expand if known
                dim_name = self._advance().value
                if dim_name in self.macros:
                    try:
                        # Use PL/M number parser to handle H/B/O/Q/D suffixes
                        dimension = self._parse_plm_number(self.macros[dim_name])
                    except ValueError:
                        dimension = -2  # Mark as macro reference
                else:
                    dimension = -2  # Unknown macro - will resolve at compile time
            else:
                raise self._error("Expected dimension")
            self._expect(TokenType.RPAREN, "Expected ')' after dimension")

        # Parse type
        struct_members: list[StructMember] | None = None
        data_type: DataType | None = None

        if self._match(TokenType.STRUCTURE):
            struct_members = self._parse_structure_type()
        elif self._match(TokenType.BYTE):
            data_type = DataType.BYTE
        elif self._match(TokenType.ADDRESS):
            data_type = DataType.ADDRESS
        elif self._check(TokenType.IDENTIFIER):
            # Check if this is a LITERALLY macro for a type
            type_name = self.current.value
            if type_name in self.macros:
                expansion = self.macros[type_name].strip().upper()
                if expansion == "BYTE":
                    self._advance()
                    data_type = DataType.BYTE
                elif expansion == "ADDRESS":
                    self._advance()
                    data_type = DataType.ADDRESS
        # Type might be implied for BASED variables

        # Parse attributes
        attrs = self._parse_var_attributes()

        # Parse initialization
        init_values = self._parse_initialization()

        # Parse DATA
        data_values: list[Expr] | None = None
        if self._match(TokenType.DATA):
            data_values = self._parse_data_values()

        return VarDecl(
            name=name,
            data_type=data_type,
            dimension=dimension if dimension and dimension != -1 else None,
            struct_members=struct_members,
            based_on=based_on,
            based_member=based_member,
            is_public=attrs.get("public", False),
            is_external=attrs.get("external", False),
            at_location=attrs.get("at"),
            initial_values=init_values,
            data_values=data_values,
        )

    def _parse_type(self) -> DataType | None:
        """Parse a type specifier."""
        if self._match(TokenType.BYTE):
            return DataType.BYTE
        elif self._match(TokenType.ADDRESS):
            return DataType.ADDRESS
        elif self._check(TokenType.IDENTIFIER):
            # Check for LITERALLY macro that expands to a type
            type_name = self.current.value
            if type_name in self.macros:
                expansion = self.macros[type_name].strip().upper()
                if expansion == "BYTE":
                    self._advance()
                    return DataType.BYTE
                elif expansion == "ADDRESS":
                    self._advance()
                    return DataType.ADDRESS
        return None

    def _parse_structure_type(self) -> list[StructMember]:
        """Parse STRUCTURE type definition."""
        self._expect(TokenType.LPAREN, "Expected '(' after STRUCTURE")
        members: list[StructMember] = []

        while True:
            name_tok = self._expect(TokenType.IDENTIFIER, "Expected member name")
            name = name_tok.value

            dimension: int | None = None
            if self._match(TokenType.LPAREN):
                dimension = self._expect_number_or_macro("Expected dimension")
                self._expect(TokenType.RPAREN, "Expected ')' after dimension")

            if self._match(TokenType.BYTE):
                dtype = DataType.BYTE
            elif self._match(TokenType.ADDRESS):
                dtype = DataType.ADDRESS
            else:
                raise self._error("Expected BYTE or ADDRESS in structure member")

            members.append(StructMember(name, dtype, dimension))

            if not self._match(TokenType.COMMA):
                break

        self._expect(TokenType.RPAREN, "Expected ')' after structure members")
        return members

    def _parse_var_attributes(self) -> dict[str, object]:
        """Parse variable attributes (PUBLIC, EXTERNAL, AT)."""
        attrs: dict[str, object] = {}

        while True:
            if self._match(TokenType.PUBLIC):
                attrs["public"] = True
            elif self._match(TokenType.EXTERNAL):
                attrs["external"] = True
            elif self._match(TokenType.AT):
                self._expect(TokenType.LPAREN, "Expected '(' after AT")
                attrs["at"] = self._parse_expression()
                self._expect(TokenType.RPAREN, "Expected ')' after AT expression")
            else:
                break

        return attrs

    def _parse_initialization(self) -> list[Expr] | None:
        """Parse INITIAL(...) clause."""
        if not self._match(TokenType.INITIAL):
            return None

        self._expect(TokenType.LPAREN, "Expected '(' after INITIAL")
        values: list[Expr] = []

        if not self._check(TokenType.RPAREN):
            values.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                values.append(self._parse_expression())

        self._expect(TokenType.RPAREN, "Expected ')' after INITIAL values")
        return values

    def _parse_data_values(self) -> list[Expr]:
        """Parse DATA(...) values."""
        self._expect(TokenType.LPAREN, "Expected '(' after DATA")
        values: list[Expr] = []

        if not self._check(TokenType.RPAREN):
            values.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                values.append(self._parse_expression())

        self._expect(TokenType.RPAREN, "Expected ')' after DATA values")
        return values

    # ========================================================================
    # Procedure Parsing
    # ========================================================================

    def _parse_procedure_as_stmt(self) -> Stmt:
        """Parse a procedure definition that appears as a statement."""
        proc = self._parse_procedure()
        return DeclareStmt([proc])

    def _parse_procedure(self) -> ProcDecl:
        """Parse a procedure declaration."""
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected procedure name")
        name = name_tok.value
        self._expect(TokenType.COLON, "Expected ':' after procedure name")
        # Accept PROCEDURE keyword or identifier that expands to 'PROCEDURE' via LITERALLY
        if not self._match(TokenType.PROCEDURE):
            if self._is_proc_abbreviation():
                self._advance()  # Consume the PROC macro identifier
            else:
                raise ParseError(f"Expected PROCEDURE", self._current_location())

        # Parse parameters
        params: list[str] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                params.append(
                    self._expect(TokenType.IDENTIFIER, "Expected parameter name").value
                )
                while self._match(TokenType.COMMA):
                    params.append(
                        self._expect(TokenType.IDENTIFIER, "Expected parameter name").value
                    )
            self._expect(TokenType.RPAREN, "Expected ')' after parameters")

        # Parse return type
        return_type: DataType | None = None
        if self._match(TokenType.BYTE):
            return_type = DataType.BYTE
        elif self._match(TokenType.ADDRESS):
            return_type = DataType.ADDRESS

        # Parse attributes
        is_public = False
        is_external = False
        is_reentrant = False
        interrupt_num: int | None = None

        while True:
            if self._match(TokenType.PUBLIC):
                is_public = True
            elif self._match(TokenType.EXTERNAL):
                is_external = True
            elif self._match(TokenType.REENTRANT):
                is_reentrant = True
            elif self._match(TokenType.INTERRUPT):
                interrupt_num = self._expect_number_or_macro("Expected interrupt number")
            else:
                break

        self._expect(TokenType.SEMICOLON, "Expected ';' after procedure header")

        # Parse body
        decls: list[Declaration] = []
        stmts: list[Stmt] = []

        # Parse declarations (allowed even for EXTERNAL to declare parameter types)
        while self._check(TokenType.DECLARE):
            decls.extend(self._parse_declare())

        if not is_external:
            # Parse statements (only for non-EXTERNAL procedures)
            while not self._check(TokenType.END) and not self._check(TokenType.EOF):
                stmts.append(self._parse_statement())

        # Parse END - even EXTERNAL procedures have END name;
        self._expect(TokenType.END, "Expected END")
        if self._match(TokenType.IDENTIFIER):
            end_name = self._peek(-1).value
            if end_name != name:
                # Warning: end label doesn't match
                pass
        self._expect(TokenType.SEMICOLON, "Expected ';' after END")

        return ProcDecl(
            name=name,
            params=params,
            return_type=return_type,
            is_public=is_public,
            is_external=is_external,
            is_reentrant=is_reentrant,
            interrupt_num=interrupt_num,
            decls=decls,
            stmts=stmts,
            span=self._span_from(name_tok),
        )

    # ========================================================================
    # Module Parsing
    # ========================================================================

    def parse_module(self) -> Module:
        """Parse a complete PL/M-80 module."""
        origin: int | None = None

        # Check for origin address at start (e.g., 0FAH:)
        if self._check(TokenType.NUMBER) and self._peek(1).type == TokenType.COLON:
            origin = self._advance().value
            self._advance()  # consume colon

        # Module should start with identifier: DO;
        # But we also handle the case where module wrapper is implicit
        name = "MODULE"
        decls: list[Declaration] = []
        stmts: list[Stmt] = []

        # Check for explicit module structure: name: DO;
        if self._check(TokenType.IDENTIFIER) and self._peek(1).type == TokenType.COLON:
            if self._peek(2).type == TokenType.DO:
                name_tok = self._advance()
                name = name_tok.value
                self._advance()  # colon
                self._advance()  # DO
                self._expect(TokenType.SEMICOLON, "Expected ';' after module DO")

                # Parse module body
                while self._check(TokenType.DECLARE):
                    decls.extend(self._parse_declare())

                while (
                    not self._check(TokenType.END)
                    and not self._check(TokenType.EOF)
                    and not self._check(TokenType.EOF_KW)
                ):
                    stmt = self._parse_statement()
                    if isinstance(stmt, DeclareStmt):
                        decls.extend(stmt.declarations)
                    else:
                        stmts.append(stmt)

                # Parse END
                if self._match(TokenType.END):
                    if self._match(TokenType.IDENTIFIER):
                        pass  # end label
                    self._expect(TokenType.SEMICOLON, "Expected ';' after END")
            elif self._peek(2).type == TokenType.PROCEDURE:
                # First item is a procedure - use its name as module name
                proc = self._parse_procedure()
                name = proc.name
                decls.append(proc)
                # Continue parsing the rest of the file
                while not self._check(TokenType.EOF) and not self._check(TokenType.EOF_KW):
                    if self._check(TokenType.DECLARE):
                        decls.extend(self._parse_declare())
                    else:
                        stmt = self._parse_statement()
                        if isinstance(stmt, DeclareStmt):
                            decls.extend(stmt.declarations)
                        else:
                            stmts.append(stmt)
            else:
                # Not a DO block and not a procedure - parse as implicit module
                while not self._check(TokenType.EOF) and not self._check(TokenType.EOF_KW):
                    if self._check(TokenType.DECLARE):
                        decls.extend(self._parse_declare())
                    else:
                        stmt = self._parse_statement()
                        if isinstance(stmt, DeclareStmt):
                            decls.extend(stmt.declarations)
                        else:
                            stmts.append(stmt)
        else:
            # Parse declarations and statements directly
            while not self._check(TokenType.EOF) and not self._check(TokenType.EOF_KW):
                if self._check(TokenType.DECLARE):
                    decls.extend(self._parse_declare())
                else:
                    stmt = self._parse_statement()
                    if isinstance(stmt, DeclareStmt):
                        decls.extend(stmt.declarations)
                    else:
                        stmts.append(stmt)

        # Skip EOF keyword if present
        self._match(TokenType.EOF_KW)

        return Module(name=name, origin=origin, decls=decls, stmts=stmts)


def parse(source: str, filename: str = "<input>") -> Module:
    """Convenience function to parse PL/M-80 source code."""
    from .lexer import Lexer

    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename)
    return parser.parse_module()
