# Project Structure

```
uplm80/
├── compiler.py      # Main compiler driver / CLI
├── frontend.py      # plox-driven lexer + LR parse → AST
├── preprocess.py    # PL/M preprocessor ($INCLUDE, $if, LITERALLY, ...)
├── ast_nodes.py     # AST definitions
├── ast_optimizer.py # AST-level optimizations
├── codegen.py       # Z80 code generator
├── local_storage.py # Which procedure locals may share ??AUTO
├── names.py         # Which declaration each name means; GOTO rules; assembler names
├── runtime.py       # Runtime helpers
├── symbols.py       # Symbol table
├── errors.py        # Diagnostic exception types
└── data/            # Pre-built plox grammar bundle (plm_full.json)
```

Peephole optimization is provided by the external [upeepz80](https://github.com/avwohl/upeepz80) package; the front-end is generated from [plox](https://github.com/avwohl/plox) grammars (`plm_pre` + `plm_full`) and loaded at import time from the JSON bundle in `data/`.
