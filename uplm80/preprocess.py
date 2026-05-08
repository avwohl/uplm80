"""PL/M-80 preprocessor for uplm80.

Runs ahead of plox's lexer + plm_full parser. Produces plain PL/M source
ready for the plm_full LR parser.

Per-compiler preprocessor — plox does not host one. uplm80's surface:

* High-bit / Ctrl-Z source cleaning (CP/M editor artifacts).
* Recursive ``$INCLUDE(name)`` expansion, with .LIT extension fallback.
* ``$cond`` / ``$if`` / ``$else`` / ``$endif`` / ``$set`` / ``$reset``
  conditional compilation, written either as a line-start directive
  (``$IF NAME``) or inside a ``/** $... **/`` comment marker.
* ``-D SYMBOL`` defines from the command line.
* Case folding to upper outside string literals + comments.
* Block-scoped ``LITERALLY`` macro substitution. PL/M-80 LITERALLYs are
  scoped to the enclosing ``DO``/``PROCEDURE`` block; PIP.PLM relies on
  this to keep an inner ``M LITERALLY '20'`` from clobbering a sibling
  block's variable named ``M``. The ``EQU`` and ``LIT``
  alias-bootstrap idioms (``LIT LITERALLY 'LITERALLY'`` then
  ``XYZZY LIT '5'``) are recognised and respect the same scoping.
* Stripping of harmless ``$``-directive lines (``$TITLE``, ``$Q=1``,
  ``$NOLIST``, …) so the LR parser doesn't trip over them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .errors import LexerError, SourceLocation


_DIRECTIVE_COMMENT_RE = re.compile(
    r"/\*\*\s*\$(\w+)\s*(.*?)\s*\*\*/", re.IGNORECASE | re.DOTALL
)
_LINE_DEFINE_RE = re.compile(r"\$(?:DEFINE|SET)\s*\(?(\w+)\)?", re.IGNORECASE)
_LINE_RESET_RE = re.compile(r"\$RESET\s*\((\w+)\)", re.IGNORECASE)
_LINE_COND_RE = re.compile(r"\$COND\b", re.IGNORECASE)
_LINE_IF_RE = re.compile(r"\$IF\s+(\w+)", re.IGNORECASE)
_LINE_ELSE_RE = re.compile(r"\$ELSE\b", re.IGNORECASE)
_LINE_ENDIF_RE = re.compile(r"\$ENDIF\b", re.IGNORECASE)
_LINE_INCLUDE_RE = re.compile(r"\$INCLUDE\s*\(([^)]+)\)", re.IGNORECASE)


@dataclass
class _State:
    """Conditional-compilation state shared across nested includes."""

    symbols: set[str] = field(default_factory=set)
    cond_enabled: bool = False
    # Stack of (branch_taken, in_else) — same encoding as the legacy lexer.
    stack: list[tuple[bool, bool]] = field(default_factory=list)

    def skipping(self) -> bool:
        if not self.cond_enabled or not self.stack:
            return False
        for branch_taken, in_else in self.stack:
            if branch_taken and in_else:
                return True
            if not branch_taken and not in_else:
                return True
        return False


def preprocess(
    source: str,
    filename: str = "<input>",
    defines: list[str] | None = None,
    include_paths: list[str] | None = None,
) -> str:
    """Run uplm80's preprocessor over ``source`` and return the
    transformed text. Handles include expansion and the $cond family;
    leaves LITERALLY / case folding / harmless ``$Q=1``-style directives
    for the downstream plox preprocess pass."""
    source = _strip_high_bits(source)

    state = _State()
    for sym in defines or []:
        state.symbols.add(sym.upper())

    paths = list(include_paths or [])
    if filename and filename != "<input>":
        d = os.path.dirname(os.path.abspath(filename))
        if d not in paths:
            paths.insert(0, d)

    return _process(source, filename, state, paths)


def _strip_high_bits(source: str) -> str:
    """Clean a freshly-read source file. PL/M-80 ends at the first
    Ctrl-Z (CP/M EOF) — bytes after it are sector-fill garbage and must
    be dropped, not parsed. CP/M editors also set the high bit on
    characters as word-wrap hints (so ``ª`` (0xAA) becomes ``*`` (0x2A)
    and decorative banners parse as comments); strip that across what
    remains."""
    eof = source.find("\x1a")
    if eof >= 0:
        source = source[:eof]
    return "".join(chr(ord(c) & 0x7F) for c in source)


def _process(source: str, filename: str, state: _State, paths: list[str]) -> str:
    """Walk ``source`` once, emitting active text into a buffer.

    The walk treats comments and strings as opaque so directives buried
    inside them aren't accidentally processed. ``/** $... **/`` comment
    markers are the exception — they are conditional-compilation
    directives even though they live in comment syntax — so we peek at
    each comment opener to decide.
    """
    out: list[str] = []
    i = 0
    n = len(source)
    line = 1
    col = 1
    at_line_start = True  # whitespace-only since last newline

    def loc() -> SourceLocation:
        return SourceLocation(line, col, filename)

    def emit(text: str) -> None:
        if not state.skipping():
            out.append(text)

    while i < n:
        ch = source[i]

        # Newline tracking + line-start signal reset
        if ch == "\n":
            emit(ch)
            i += 1
            line += 1
            col = 1
            at_line_start = True
            continue

        # Whitespace before any non-whitespace on this line preserves the
        # at_line_start flag so $-directives can be detected.
        if ch in " \t\r":
            emit(ch)
            i += 1
            col += 1
            continue

        # /** $... **/ directive comment — look for the directive form
        # before generic comment handling so we can act on it.
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end < 0:
                raise LexerError("Unterminated comment", loc())
            comment = source[i:end + 2]
            m = _DIRECTIVE_COMMENT_RE.fullmatch(comment.strip())
            if m:
                _apply_directive(m.group(1), m.group(2).strip(), state)
                # Drop the directive comment from the output entirely —
                # it's not meaningful to plm_full.
            else:
                emit(comment)
            # Update line/col for the consumed span.
            line += comment.count("\n")
            last_nl = comment.rfind("\n")
            col = (len(comment) - last_nl) if last_nl >= 0 else (col + len(comment))
            i = end + 2
            at_line_start = False
            continue

        # Line-start $-directive
        if ch == "$" and at_line_start:
            j = i
            while j < n and source[j] not in "\r\n":
                j += 1
            directive_line = source[i:j].strip()
            consumed = _apply_line_directive(
                directive_line, filename, state, paths, out, loc()
            )
            i = j
            col += (j - i if j > i else 0)
            # Don't reset at_line_start; the next char is \n or EOF.
            if not consumed and not state.skipping():
                # Unknown $-directive: leave it in the source so the
                # downstream plox preprocess can record it ($Q=1 etc).
                out.append(directive_line)
            continue

        # /* ... */ regular comment (already handled the directive form)
        # — handled above. Falls through here only on chars that aren't
        # comment openers.

        # String literal: pass through verbatim, multi-line allowed.
        if ch == "'":
            j = i + 1
            while j < n:
                if source[j] == "'":
                    if j + 1 < n and source[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                raise LexerError("Unterminated string literal", loc())
            text = source[i:j]
            emit(text)
            line += text.count("\n")
            last_nl = text.rfind("\n")
            col = (len(text) - last_nl) if last_nl >= 0 else (col + len(text))
            i = j
            at_line_start = False
            continue

        # Default: emit the character.
        emit(ch)
        i += 1
        col += 1
        at_line_start = False

    return "".join(out)


def _apply_directive(name: str, arg: str, state: _State) -> None:
    """Process a ``/** $... **/`` directive body."""
    name = name.lower()
    if name == "set":
        m = re.match(r"\((\w+)\)", arg)
        if m:
            state.symbols.add(m.group(1).upper())
    elif name == "reset":
        m = re.match(r"\((\w+)\)", arg)
        if m:
            state.symbols.discard(m.group(1).upper())
    elif name == "cond":
        state.cond_enabled = True
    elif name == "if":
        if state.cond_enabled:
            state.stack.append((arg.upper() in state.symbols, False))
    elif name == "else":
        if state.cond_enabled and state.stack:
            taken, _ = state.stack[-1]
            state.stack[-1] = (taken, True)
    elif name == "endif":
        if state.cond_enabled and state.stack:
            state.stack.pop()
    # Other directives (title, eject, list, ...) drop silently.


def _apply_line_directive(
    line: str,
    filename: str,
    state: _State,
    paths: list[str],
    out: list[str],
    loc: SourceLocation,
) -> bool:
    """Process a single line that starts with ``$``. Returns True if the
    directive was recognised and consumed; False if it should be passed
    through to the downstream preprocess."""
    if (m := _LINE_DEFINE_RE.match(line)):
        if not state.skipping():
            state.symbols.add(m.group(1).upper())
        return True
    if (m := _LINE_RESET_RE.match(line)):
        if not state.skipping():
            state.symbols.discard(m.group(1).upper())
        return True
    if _LINE_COND_RE.match(line):
        if not state.skipping():
            state.cond_enabled = True
        return True
    if (m := _LINE_IF_RE.match(line)):
        if state.cond_enabled:
            state.stack.append((m.group(1).upper() in state.symbols, False))
        return True
    if _LINE_ELSE_RE.match(line):
        if state.cond_enabled and state.stack:
            taken, _ = state.stack[-1]
            state.stack[-1] = (taken, True)
        return True
    if _LINE_ENDIF_RE.match(line):
        if state.cond_enabled and state.stack:
            state.stack.pop()
        return True
    if (m := _LINE_INCLUDE_RE.match(line)):
        if state.skipping():
            return True
        target = m.group(1).strip().strip("'").strip('"')
        content = _read_include(target, paths, loc)
        # Recursively expand. State is shared so a $set inside an
        # included file is visible to the outer file.
        included_paths = list(paths)
        d = os.path.dirname(os.path.abspath(target)) if os.path.isabs(target) else None
        if d and d not in included_paths:
            included_paths.insert(0, d)
        out.append(_process(content, target, state, included_paths))
        return True
    return False


def _read_include(name: str, paths: list[str], loc: SourceLocation) -> str:
    """Resolve and read an ``$INCLUDE`` target. Tries the name as given,
    upper-case, lower-case, plus ``.lit`` / ``.LIT`` extensions across
    every search path. CP/M devices like ``:F1:`` are stripped first
    since modern filesystems don't carry them."""
    bare = name
    if ":" in bare:
        bare = bare.split(":")[-1]
    candidates: list[str] = []
    for base in paths or [os.getcwd()]:
        for stem in (bare, bare.upper(), bare.lower()):
            candidates.append(os.path.join(base, stem))
            if not stem.upper().endswith(".LIT"):
                candidates.append(os.path.join(base, stem + ".lit"))
                candidates.append(os.path.join(base, stem + ".LIT"))
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    raw = f.read()
            except UnicodeDecodeError:
                with open(cand, "r", encoding="latin-1") as f:
                    raw = f.read()
            return _strip_high_bits(raw)
    raise LexerError(f"$INCLUDE file not found: {name}", loc)


# =====================================================================
# Macro pass: case-fold + scoped LITERALLY substitution + $-directive strip.
#
# Runs after the conditional/include pass above; produces the final
# source string handed to plox's lexer + plm_full parser.
# =====================================================================


_PLM_KEYWORDS_LOWER = frozenset(
    [
        "procedure", "do", "end",
        "literally", "declare", "dcl",  # dcl is the common abbreviation
        # The abbreviation `lit` and the `equ` idiom are *not* hard-coded
        # here — they're picked up by the alias bootstrap when the
        # source has e.g. `lit literally 'literally'`. PL/M-80 itself
        # doesn't reserve them.
    ]
)


def macro_pass(source: str) -> str:
    """Case-fold and apply block-scoped LITERALLY substitution.

    Tokenises the source with a small PL/M-aware scanner (the bare
    minimum needed to recognise IDENTs, STRINGs, comments, line-start
    ``$``-directives, and the ``DO`` / ``PROCEDURE`` / ``END`` /
    ``LITERALLY`` keywords). Walks tokens once with a scope stack; on
    each block-opener pushes a fresh macro frame, on ``END`` pops, and
    substitutes IDENT references against the stack from inner to outer.

    Re-substitutes macro bodies recursively so nested macros (e.g.
    ``process$descriptor`` → ``process$header, …, bdos$save`` where
    those are themselves macros) expand fully without an outer fixed-
    point loop.
    """
    # PL/M source files end at the first Ctrl-Z (0x1A) — that's the
    # CP/M end-of-file marker, and any bytes after it are garbage
    # (filler from the editor, padding to a 128-byte sector, etc.).
    eof = source.find("\x1a")
    if eof >= 0:
        source = source[:eof]
    tokens = _tokenize_for_macros(source)
    out: list[str] = []
    scope: list[dict[str, str]] = [{}]  # outermost frame
    aliases: list[set[str]] = [{"LITERALLY"}]  # words that act as KW LITERALLY

    def is_literally_kw(text: str) -> bool:
        for frame in reversed(aliases):
            if text in frame:
                return True
        return False

    def lookup(name: str) -> str | None:
        for frame in reversed(scope):
            if name in frame:
                return frame[name]
        return None

    def define(name: str, body: str) -> None:
        scope[-1][name] = body
        if body.strip().upper() == "LITERALLY":
            aliases[-1].add(name)

    # The walk uses an explicit cursor over the tokens list so we can
    # peek ahead for the `IDENT LITERALLY STRING` macro-definition
    # shape and consume the three tokens together without re-emitting
    # the partial spelling.
    i = 0
    n = len(tokens)
    pending: list[_Tok] = []  # putback queue for substituted-body tokens

    # Stack of `(` contexts: each entry is the keyword that opened the
    # outer parens (``"INITIAL"`` / ``"DATA"`` / ``""`` for any other
    # paren). Used to pick expression-level vs full-text substitution
    # for macro bodies — inside ``INITIAL(...)`` / ``DATA(...)`` only
    # the first top-comma chunk of a macro body is substituted, matching
    # legacy uplm80's "sub-parse macro body as a single expression"
    # semantics. Elsewhere the full body is substituted.
    paren_stack: list[str] = []
    last_significant: str = ""

    def next_tok() -> "_Tok | None":
        nonlocal i
        if pending:
            return pending.pop(0)
        if i >= n:
            return None
        tok = tokens[i]
        i += 1
        return tok

    while True:
        tok = next_tok()
        if tok is None:
            break

        if tok.kind == "WS" or tok.kind == "COMMENT":
            out.append(tok.text)
            continue
        if tok.kind == "DOLLAR_DIR":
            # $TITLE, $Q=1, $NOLIST, … — drop. ($INCLUDE / $cond /
            # $if etc. were handled earlier in the conditional pass.)
            continue
        if tok.kind == "STRING":
            out.append(tok.text)
            continue
        if tok.kind == "PUNCT":
            if tok.text == "(":
                opener = last_significant if last_significant in ("INITIAL", "DATA") else ""
                paren_stack.append(opener)
            elif tok.text == ")":
                if paren_stack:
                    paren_stack.pop()
            last_significant = tok.text
            out.append(tok.text)
            continue
        if tok.kind == "NUMBER":
            last_significant = tok.text
            out.append(tok.text)
            continue

        # IDENT (case-folded to upper). Check for keywords / aliases /
        # macro-def shape / macro-use, in that order.
        assert tok.kind == "IDENT"
        text = tok.text  # already upper-folded by the scanner
        last_significant = text

        # Block-scope tracking. PROCEDURE and DO open a scope; END
        # closes the innermost. Detection is purely lexical here —
        # PL/M-80 sources don't disguise these keywords.
        if text == "PROCEDURE" or text == "DO":
            out.append(text)
            scope.append({})
            aliases.append(set())
            continue
        if text == "END":
            out.append(text)
            if len(scope) > 1:
                scope.pop()
                aliases.pop()
            continue

        # Macro-definition: IDENT (KW_LITERALLY|alias) STRING. Peek
        # ahead past whitespace/comments to find the next two non-
        # trivia tokens.
        peek1, advance1 = _peek_significant(tokens, i, pending)
        if peek1 is not None and peek1.kind == "IDENT" and is_literally_kw(peek1.text):
            peek2, advance2 = _peek_significant(tokens, advance1, pending)
            if peek2 is not None and peek2.kind == "STRING":
                # Consume the IDENT-LITERALLY-STRING triple and emit it
                # verbatim (canonicalising the LITERALLY keyword to
                # the literal spelling so plm_full sees what it expects).
                out.append(text)
                # Whitespace between tok and peek1
                _emit_between(tokens, i, advance1 - 1, pending, out)
                out.append("LITERALLY")
                _emit_between(tokens, advance1, advance2 - 1, pending, out)
                out.append(peek2.text)
                # Body after stripping the surrounding quotes and any
                # PL/M `''` -> `'` doubling.
                body = peek2.text
                if len(body) >= 2 and body.startswith("'") and body.endswith("'"):
                    body = body[1:-1]
                body = body.replace("''", "'")
                define(text, body)
                # Advance the cursor past the consumed two extras + any
                # whitespace eaten by the peeks.
                i = advance2
                continue

        # Macro-use: substitute by re-tokenising the body and pushing
        # to the front of the pending queue. The substituted tokens
        # then go through the same scope / further-substitution
        # machinery, so nested macros expand to the bottom.
        body = lookup(text)
        if body is not None:
            # Inside INITIAL(...) / DATA(...) the legacy parser
            # sub-parses the macro body as a single expression and
            # silently discards anything past the first comma at the
            # body's top level. Mirror that here so init lists like
            # `initial(restarts, .status)` — where ``restarts`` is
            # ``'0C7C7H,0C7C7H,...,0C7C7H'`` — yield 2 init values, not
            # 19+1.
            in_init_ctx = bool(paren_stack) and paren_stack[-1] in ("INITIAL", "DATA")
            effective_body = _first_top_comma_chunk(body) if in_init_ctx else body
            sub_tokens = _tokenize_for_macros(effective_body)
            pending = list(sub_tokens) + pending
            continue

        # Plain identifier — emit as-is.
        out.append(text)

    return "".join(out)


def _first_top_comma_chunk(body: str) -> str:
    """Return everything up to the first top-level (paren-balanced)
    comma in ``body``. Mimics the legacy parser's behaviour of sub-
    parsing a macro body as a single expression in INITIAL/DATA arg
    lists — matters for definitions like
    ``restarts literally '0C7C7H,0C7C7H,...,0C7C7H'`` whose body has
    19 comma-separated values but is intended to act as one item when
    spliced into an INITIAL list."""
    depth = 0
    in_str = False
    for j, c in enumerate(body):
        if in_str:
            if c == "'":
                in_str = False
            continue
        if c == "'":
            in_str = True
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            if depth > 0:
                depth -= 1
        elif c == "," and depth == 0:
            return body[:j]
    return body


@dataclass
class _Tok:
    kind: str  # IDENT | NUMBER | STRING | PUNCT | WS | COMMENT | DOLLAR_DIR
    text: str
    line: int = 0
    col: int = 0


_PUNCT_CHARS = set("()[]{},.:;+-*/=<>")


def _tokenize_for_macros(source: str) -> list[_Tok]:
    """Tokenise PL/M-80 source for the macro pass.

    Output preserves whitespace + comments as separate tokens so the
    macro pass can reconstruct the original source layout. Identifiers
    are case-folded to upper. ``$<word>...`` lines are returned as a
    single ``DOLLAR_DIR`` token covering the whole line.
    """
    out: list[_Tok] = []
    n = len(source)
    i = 0
    line = 1
    col = 1
    at_line_start = True

    def advance(j: int) -> None:
        nonlocal line, col
        for k in range(j):
            if source[i + k] == "\n":
                line += 1
                col = 1
            else:
                col += 1

    while i < n:
        c = source[i]

        # Newline: emits as WS, resets line-start.
        if c == "\n":
            out.append(_Tok("WS", "\n", line, col))
            line += 1
            col = 1
            at_line_start = True
            i += 1
            continue

        if c in " \t\r":
            j = i
            while j < n and source[j] in " \t\r":
                j += 1
            out.append(_Tok("WS", source[i:j], line, col))
            col += j - i
            i = j
            continue

        # Comment /* ... */
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2)
            if j < 0:
                # Unterminated; emit rest as a comment so we don't lose
                # data. The downstream parser will catch the syntax issue.
                out.append(_Tok("COMMENT", source[i:], line, col))
                i = n
                continue
            text = source[i:j + 2]
            out.append(_Tok("COMMENT", text, line, col))
            advance(j + 2 - i)
            i = j + 2
            at_line_start = False
            continue

        # $-directive at line start
        if c == "$" and at_line_start:
            j = i
            while j < n and source[j] not in "\r\n":
                j += 1
            out.append(_Tok("DOLLAR_DIR", source[i:j], line, col))
            col += j - i
            i = j
            continue

        # String literal (multi-line OK, '' is the escape)
        if c == "'":
            j = i + 1
            while j < n:
                if source[j] == "'":
                    if j + 1 < n and source[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                # Unterminated; emit rest as STRING for downstream error.
                out.append(_Tok("STRING", source[i:], line, col))
                i = n
                continue
            text = source[i:j]
            out.append(_Tok("STRING", text, line, col))
            advance(j - i)
            i = j
            at_line_start = False
            continue

        # Identifier: [A-Za-z_$][A-Za-z0-9_$]* — but we strip the leading
        # `$` from being valid because that's reserved for $-directives;
        # mid-identifier `$` is fine and is the PL/M soft-hyphen.
        if c.isalpha() or c == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] in "_$"):
                j += 1
            # PL/M-80 ``$`` inside an identifier is a soft hyphen —
            # ``STACK$SIZ`` and ``STACKSIZ`` are the same name. The
            # legacy lexer normalised at tokenize time; we do the same
            # so macro lookups, keyword checks, and downstream emission
            # all see one canonical spelling per identifier.
            text = source[i:j].upper().replace("$", "")
            out.append(_Tok("IDENT", text, line, col))
            col += j - i
            i = j
            at_line_start = False
            continue

        # Number: [0-9][0-9A-Fa-f$]*[BbDdHhOoQq]?
        if c.isdigit():
            j = i
            while j < n and (source[j].isalnum() or source[j] == "$"):
                j += 1
            text = source[i:j]
            out.append(_Tok("NUMBER", text, line, col))
            col += j - i
            i = j
            at_line_start = False
            continue

        # Multi-char punctuation
        if c == "<" and i + 1 < n and source[i + 1] in "=>":
            out.append(_Tok("PUNCT", source[i:i + 2], line, col))
            col += 2
            i += 2
            at_line_start = False
            continue
        if c == ">" and i + 1 < n and source[i + 1] == "=":
            out.append(_Tok("PUNCT", source[i:i + 2], line, col))
            col += 2
            i += 2
            at_line_start = False
            continue
        if c == ":" and i + 1 < n and source[i + 1] == "=":
            out.append(_Tok("PUNCT", source[i:i + 2], line, col))
            col += 2
            i += 2
            at_line_start = False
            continue

        if c in _PUNCT_CHARS or c == "$":
            out.append(_Tok("PUNCT", c, line, col))
            col += 1
            i += 1
            at_line_start = False
            continue

        # Unknown — emit as a single-char punct so the parser can
        # surface a useful error later rather than silently dropping it.
        out.append(_Tok("PUNCT", c, line, col))
        col += 1
        i += 1
        at_line_start = False

    return out


def _peek_significant(
    tokens: list[_Tok], start: int, pending: list[_Tok]
) -> tuple[_Tok | None, int]:
    """Find the next non-trivia (non-WS, non-COMMENT) token from
    ``tokens[start:]``, ignoring any pending putback tokens which are
    irrelevant to the def-shape lookahead. Returns ``(tok, new_index)``
    where ``new_index`` is the position *after* the returned token."""
    j = start
    while j < len(tokens):
        if tokens[j].kind not in ("WS", "COMMENT"):
            return tokens[j], j + 1
        j += 1
    return None, j


def _emit_between(
    tokens: list[_Tok], start: int, end_inclusive: int, pending: list[_Tok], out: list[str]
) -> None:
    """Emit the WS/COMMENT trivia tokens between two significant tokens
    so the output preserves source layout. ``end_inclusive`` is the
    index of the *significant* end token; we emit everything strictly
    before it."""
    for j in range(start, end_inclusive):
        if 0 <= j < len(tokens):
            out.append(tokens[j].text)
