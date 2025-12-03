# Work State

## Current Status
The PL/M-80 compiler successfully parses and compiles real-world CP/M source code.

## Recently Implemented Features

### Lexer Enhancements
- CP/M EOF marker (Ctrl-Z / 0x1A) handling
- `$` control directive lines (e.g., `$Q=1`, `$INCLUDE`) - skipped as compiler options
- `$` digit separators in numbers (e.g., `1$1111B` = `11111B`)

### Parser Enhancements
- `DO FOREVER` built-in support (equivalent to `DO WHILE TRUE`)
- `DCL` abbreviation for `DECLARE`
- `LIT` abbreviation for `LITERALLY` (context-sensitive, not reserved)
- LITERALLY macros expanding to type names (e.g., `COMSIZE LIT 'ADDRESS'`)
- BASED inside factored declarations: `(SA BASED A, SB BASED B) BYTE`
- Embedded assignment in RETURN: `RETURN CS := CS + READBYTE;`
- Factored LABEL declarations: `DCL (START, RESTART, ...) LABEL;`
- Origin address prefix: `2900H: DECLARE ...`
- EXTERNAL procedure END handling

### Code Generator Enhancements
- Nested procedure code placement (after parent procedure)
- LITERALLY macro expansion in GO TO targets
- LITERALLY expansion in DATA values
- Fixed double RET issue for procedures ending with RETURN

