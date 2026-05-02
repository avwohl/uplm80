# Front-end migration to plox

`uplm80/lexer.py`, `uplm80/parser.py`, and `uplm80/tokens.py`
(~2000 hand-written lines) are slated to be retired in favour of
[plox](https://github.com/avwohl/plox)'s generated front-end.

## State of the world

The plox grammars are done and shipped:

- `examples/plm_pre.plox` — the macro layer (LITERALLY, EQU alias,
  `$`-directives, case folding).
- `examples/plm_full.plox` — the post-expansion PL/M-80 language.
- `plox.preprocess.plm.preprocess()` — the host driver that runs
  plm_pre and produces transformed PL/M source ready for plm_full.

Live acceptance, on the plox side:

- All 47 `.plm` files in `uplm80/tests/` plus `examples/hello_cpm.plm`
  parse end-to-end through the pipeline.
- 35/44 `.PLM` files in the bundled MP/M-II archive (`../mpm2`)
  parse. The remaining 9 split as 7 corrupted source files (truncated
  identifiers from old bit-rot), 1 needing `$INCLUDE` expansion, and
  1 with a macro/variable name collision the simple-text substitution
  can't disambiguate.

This means: the parse tree exists and is correct for the corpus we
care about. The remaining work is wiring uplm80's downstream stages
(AST optimizer, code generator) to consume it.

## The work

Two-phase, behind a flag for safe rollout.

### Phase 1: shim front-end

Add `uplm80/plox_frontend.py`. It runs plox's pipeline on a PL/M
source string and converts the resulting parse tree to uplm80's
existing AST (`uplm80/ast_nodes.py` — ~30 node classes). Downstream
stages (`ast_optimizer.py`, `codegen.py`) stay untouched.

The conversion is a tree walk, ~500–800 lines depending on how
faithful you want SourceSpan tracking to be. Each plm_full
production maps to one or more AST node constructors. The mapping
is mostly mechanical:

| plm_full rule                      | AST node               |
|------------------------------------|------------------------|
| `<module>`                         | `Module`               |
| `<proc_decl>`                      | `ProcDecl`             |
| `<declare_stmt>` → `<decl_item>`   | `VarDecl` / `LiterallyDecl` / `LabelDecl` |
| `<assignment_stmt>`                | `AssignStmt`           |
| `<call_stmt>`                      | `CallStmt`             |
| `<return_stmt>`                    | `ReturnStmt`           |
| `<if-stmt>` (matched/unmatched)    | `IfStmt`               |
| `<do_block>` variants              | `DoBlock` / `DoWhileBlock` / `DoIterBlock` / `DoCaseBlock` |
| `<expr>` ladder                    | `BinaryExpr` / `UnaryExpr` |
| `<primary>`                        | `NumberLiteral` / `StringLiteral` / `Identifier` / `LocationExpr` / `ConstListExpr` |
| `<qualname>`                       | `Identifier` / `MemberExpr` / `SubscriptExpr` / `CallExpr` |
| `<primary> ':=' <expr>`            | `EmbeddedAssignExpr`   |

The `matched_stmt` / `unmatched_stmt` split disappears at the AST
level — both collapse to plain `Stmt` subclasses. The `<other>`
non-terminal in `plm_pre` doesn't appear in plm_full's tree at all.

### Phase 2: switch the default

Add a flag (env var `UPLM80_FRONTEND=plox` or a CLI flag in
`compiler.py`) to choose between the old hand-written front-end and
the new plox-backed one. Default to the old one until equivalence
holds.

Equivalence test: for each `.plm` file in `tests/`, parse with both
front-ends and compare the resulting AST as text or via deep
equality. When all match, flip the default to plox and delete
`lexer.py`, `parser.py`, `tokens.py`. (`errors.py` and the
`SourceSpan` plumbing in `ast_nodes.py` stay.)

`pyproject.toml` will need a runtime dependency: `plox >= 2.0.0`.

## Things to know before starting

- **plox 2.0.0 is on PyPI / GitHub at <https://github.com/avwohl/plox>.**
  Public, GPL-3.0-or-later, same licence as uplm80.
- **The DSL had a breaking change at 2.0.0.** Single-quote token
  literals, `<name>` for non-terminals everywhere, `%keywords`
  shortcut. The `.plox` grammars in plox's `examples/` are written
  for the new DSL; nothing in uplm80 needs to know about that — it
  imports `plox.preprocess.plm.preprocess` and friends.
- **plox emits Python parsers** via `plox emit --target=py`, but the
  preprocessor uses plox's runtime directly (it builds the table at
  import time and caches it). Either approach works for the uplm80
  front-end. Building at import is simpler; emitting is slightly
  faster on cold start.
- **Macro substitution is single-pass.** Forward references to a
  LITERALLY definition that appears later in the source are not
  substituted. Real PL/M-80 compilers behave the same way.
- **plox's parse tree is left-recursive on every list rule.** Walking
  it with naive recursion blows the Python stack on real-size
  sources (bdos.plm and friends). The plm_pre walker in plox
  `src/plox/preprocess/plm.py` shows the iterative-stack pattern;
  copy it.

## Acceptance bar

The same one the plox-side work targeted: every `tests/*.plm` in
uplm80 produces equivalent codegen output before and after the
front-end swap. Once that holds, the swap is safe to commit.

## Why this isn't done yet

The plox side hit a natural commit point — grammars, preprocessor,
mpm2 acceptance, all shipped. The uplm80 side is a different repo
with a different test surface and is best done as a focused session
where the only context is "convert plox tree to uplm80 AST." That's
the next thing.
