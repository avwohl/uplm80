# Register Allocation Design for uplm80

## Background: Register Allocation Theory

Register allocation is a well-studied compiler problem. Common approaches include:

1. **Graph Coloring** (Chaitin): Build an interference graph where nodes are live ranges and edges connect simultaneously-live variables. Color the graph with k colors (k = number of registers). NP-complete but produces optimal results.

2. **Linear Scan**: Process live intervals in order, assigning registers greedily. O(n log n) complexity. Used by JIT compilers (V8, HotSpot, ART) where compile time matters.

3. **Sethi-Ullman**: Specifically for expression trees. Labels each node with minimum registers needed, then generates code processing the more expensive subtree first.

For uplm80, we need a simpler approach because:
- We generate code in a single pass (no separate allocation phase)
- Expression evaluation is stack-based with registers as a "register stack"
- The Z80 has very few registers with specific roles

## Z80 Register Conventions

| Register | Primary Use | Notes |
|----------|-------------|-------|
| A | BYTE values, accumulator | Most 8-bit ops require A; a BYTE result |
| HL | ADDRESS values, primary 16-bit | Memory access via (HL), arithmetic; an ADDRESS result |
| DE | Secondary 16-bit operand | Used for binary ops, block moves; a call's last argument |
| BC | Loop counter, tertiary 16-bit | DJNZ uses B, LDIR uses BC; a call's next-to-last (or only) argument |
| IX | Local variable frame pointer | Indexed addressing (IX+d); kept across calls |
| SP | Stack pointer | Implicit in PUSH/POP/CALL/RET |

A call keeps only SP, IX and IY: A, the flags, BC, DE and HL are all
destroyed (PL/M-80's calling convention; README, Calling Convention).
While a call's last argument is evaluated, the one before it is in BC;
the code generator saves BC round that code if it may write B or C.

## Current Problems

The current codegen uses ad-hoc register management:

```python
# Must manually check if expression will clobber DE
source_preserves_de = self._expr_preserves_de(args[1])
if source_preserves_de:
    # Safe to use DE
else:
    # Must push/pop around the expression
```

Problems:
1. **Fragile**: `_expr_preserves_de` must know internals of all expression types
2. **Error-prone**: Easy to miss cases (caused the MOVE bug)
3. **Hard to extend**: New expression types require updating multiple places
4. **No optimization**: Can't reorder to minimize spills

## Proposed Design: Register Descriptors with Demand-Driven Allocation

### Core Concept

Each register has a **descriptor** tracking:
- **State**: `free`, `busy`, or `spilled`
- **Contents**: What value it holds (for debugging/optimization)
- **Spill location**: Stack offset if spilled

When code needs a register:
1. **Request** a specific register (e.g., "need HL for result") or a class (e.g., "need any 16-bit")
2. If register is **free**: mark busy, return it
3. If register is **busy**: automatically **spill** to stack, mark as spilled-but-claimed
4. When done: **release** the register (restore from stack if spilled)

### Data Structures

```python
from enum import Enum, auto
from dataclasses import dataclass, field

class RegState(Enum):
    FREE = auto()      # Available for use
    BUSY = auto()      # Contains live value
    SPILLED = auto()   # Value saved to stack, register reused

class RegClass(Enum):
    """Register classes for allocation requests."""
    BYTE = auto()      # Need A
    ADDR = auto()      # Need HL
    ADDR_ALT = auto()  # Need DE or BC (secondary 16-bit)
    INDEX = auto()     # Need IX or IY

@dataclass
class RegDescriptor:
    state: RegState = RegState.FREE
    owner: str = ""           # Debug: what claimed this register
    spill_depth: int = 0      # Stack depth when spilled (for nested spills)
    contents: str = ""        # Debug: description of contents

@dataclass
class RegisterAllocator:
    """Tracks register state and manages allocation."""

    # Register descriptors
    a: RegDescriptor = field(default_factory=RegDescriptor)
    hl: RegDescriptor = field(default_factory=RegDescriptor)
    de: RegDescriptor = field(default_factory=RegDescriptor)
    bc: RegDescriptor = field(default_factory=RegDescriptor)
    ix: RegDescriptor = field(default_factory=RegDescriptor)

    # Stack tracking
    spill_stack: list[str] = field(default_factory=list)  # Order of spilled regs

    def get_reg(self, name: str) -> RegDescriptor:
        """Get descriptor by name."""
        return getattr(self, name.lower())
```

### Core Operations

#### `need_reg(reg_or_class, owner, emit_fn)` - Request a Register

```python
def need_reg(self, reg_or_class: str | RegClass, owner: str,
             emit_fn: Callable[[str, str], None]) -> str:
    """
    Request a register. Returns the register name.
    If busy, automatically spills it first.

    Args:
        reg_or_class: Specific register name ('hl', 'de') or RegClass
        owner: Debug string identifying the requester
        emit_fn: Callback to emit assembly (emit_fn('push', 'hl'))

    Returns:
        The allocated register name
    """
    # Resolve class to specific register
    if isinstance(reg_or_class, RegClass):
        reg = self._pick_reg_from_class(reg_or_class)
    else:
        reg = reg_or_class.lower()

    desc = self.get_reg(reg)

    if desc.state == RegState.BUSY:
        # Must spill - save current contents to stack
        emit_fn("push", reg)
        self.spill_stack.append(reg)
        desc.spill_depth = len(self.spill_stack)
        desc.state = RegState.SPILLED

    # Mark as busy with new owner
    desc.state = RegState.BUSY
    desc.owner = owner
    return reg

def _pick_reg_from_class(self, cls: RegClass) -> str:
    """Pick best register from class, preferring free ones."""
    candidates = {
        RegClass.BYTE: ['a'],
        RegClass.ADDR: ['hl'],
        RegClass.ADDR_ALT: ['de', 'bc'],
        RegClass.INDEX: ['ix'],
    }

    for reg in candidates[cls]:
        if self.get_reg(reg).state == RegState.FREE:
            return reg

    # All busy - return first (will be spilled)
    return candidates[cls][0]
```

#### `release_reg(reg, emit_fn)` - Release a Register

```python
def release_reg(self, reg: str, emit_fn: Callable[[str, str], None]) -> None:
    """
    Release a register. If it was spilled, restore it.

    Args:
        reg: Register name to release
        emit_fn: Callback to emit assembly
    """
    reg = reg.lower()
    desc = self.get_reg(reg)

    if desc.state == RegState.SPILLED and desc.spill_depth > 0:
        # Need to restore - but must pop in correct order
        # If this isn't top of spill stack, we have a problem
        if self.spill_stack and self.spill_stack[-1] == reg:
            emit_fn("pop", reg)
            self.spill_stack.pop()

    desc.state = RegState.FREE
    desc.owner = ""
    desc.spill_depth = 0
```

#### `with_reg(reg, owner)` - Context Manager for Scoped Use

```python
@contextmanager
def with_reg(self, reg: str, owner: str, emit_fn):
    """Context manager for scoped register use."""
    self.need_reg(reg, owner, emit_fn)
    try:
        yield reg
    finally:
        self.release_reg(reg, emit_fn)
```

### Expression Evaluation Model

The key insight is that `_gen_expr` returns results in a **known location**:
- BYTE expressions → result in **A**
- ADDRESS expressions → result in **HL**

When evaluating binary expressions like `left + right`:
1. Evaluate `left` → result in HL
2. **Claim DE** (may spill if busy)
3. `ex de,hl` → left now in DE
4. Evaluate `right` → result in HL (this may use/clobber other regs)
5. Perform `add hl,de`
6. **Release DE** (restores if spilled)

```python
def _gen_binary_expr_new(self, expr: BinaryExpr) -> DataType:
    """Generate binary expression with automatic register management."""

    # Evaluate left operand → result in HL (or A for BYTE)
    left_type = self._gen_expr(expr.left)

    if left_type == DataType.ADDRESS:
        # Need to preserve left in DE while evaluating right
        with self.regs.with_reg('de', 'binary_left', self._emit):
            self._emit("ex", "de,hl")  # DE = left
            right_type = self._gen_expr(expr.right)  # HL = right
            # Now: DE = left, HL = right
            # Perform operation...
            self._emit("add", "hl,de")  # Example: HL = left + right
    else:
        # BYTE operation - similar pattern with A and B
        ...

    return result_type
```

### Handling Nested Spills

Consider: `(a + b) + (c + d)` where we only have HL and DE:

1. Eval `a` → HL
2. Claim DE, `ex de,hl` → DE=a
3. Eval `b` → HL
4. `add hl,de` → HL = a+b
5. **Need DE again** for outer `+`, but we're releasing it
6. Claim DE, `ex de,hl` → DE = a+b
7. Eval `c+d`:
   - Eval `c` → HL
   - **Claim DE** - but it's busy! **Spill**: `push de`
   - `ex de,hl` → DE=c
   - Eval `d` → HL
   - `add hl,de` → HL = c+d
   - **Release DE** - was spilled, so `pop de` → DE = a+b restored
8. `add hl,de` → HL = (a+b) + (c+d)
9. Release DE

The spill stack ensures correct restore order.

### Sethi-Ullman Optimization (Future)

To minimize spills, we can label expression trees with register requirements:

```python
def _label_reg_need(self, expr: Expr) -> int:
    """Label expression with minimum registers needed (Sethi-Ullman)."""
    if isinstance(expr, (NumberLiteral, Identifier)):
        return 1  # Leaf: needs 1 register to hold result

    if isinstance(expr, BinaryExpr):
        left_need = self._label_reg_need(expr.left)
        right_need = self._label_reg_need(expr.right)

        if left_need == right_need:
            return left_need + 1  # Need extra reg to hold one side
        else:
            return max(left_need, right_need)  # Eval harder side first

    return 1
```

Then evaluate the subtree with **higher** register need **first** - this minimizes spills.

### Migration Strategy

#### Phase 1: Infrastructure
- Add `RegisterAllocator` class
- Add `self.regs` to `CodeGenerator.__init__`
- Keep all existing code working

#### Phase 2: Instrument Existing Code
- Add assertions to verify register state assumptions
- Log register operations to find patterns

#### Phase 3: Migrate Binary Expressions
- Update `_gen_binary_expr` to use `need_reg`/`release_reg`
- Remove `_expr_preserves_de` checks

#### Phase 4: Migrate All Expression Types
- CallExpr (function calls may clobber registers)
- SubscriptExpr (array access needs temp regs)
- MemberExpr (structure member access)

#### Phase 5: Optimize
- Implement Sethi-Ullman labeling
- Add peephole patterns to eliminate redundant push/pop

### Testing Strategy

1. **Regression tests**: All existing tests must pass
2. **Stack balance tests**: Verify SP is same before/after expressions
3. **Register state assertions**: Check registers are FREE at statement boundaries
4. **Stress tests**: Deeply nested expressions to test spill/restore

## References

- [Register Allocation - Wikipedia](https://en.wikipedia.org/wiki/Register_allocation)
- [Linear Scan Register Allocation (Poletto & Sarkar)](https://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf)
- [Register Allocation via Graph Coloring (Chaitin)](https://dl.acm.org/doi/10.1145/872726.806984)
- [CS701 Lecture Notes - Register Allocation](https://pages.cs.wisc.edu/~horwitz/CS701-NOTES/5.REGISTER-ALLOCATION.html)
- [Register Allocation Algorithms - GeeksforGeeks](https://www.geeksforgeeks.org/register-allocation-algorithms-in-compiler-design/)

---

## Implementation Status

### Phase 1: Infrastructure ✅ COMPLETE

**Date**: 2026-01-07

**Changes**:
- Added `RegState` enum: FREE, BUSY, SPILLED
- Added `RegClass` enum: BYTE, ADDR, ADDR_ALT, INDEX
- Added `RegDescriptor` dataclass for per-register state
- Added `RegisterAllocator` class with:
  - `need_reg(reg, owner, emit_fn)` - claim register, auto-spill if busy
  - `release_reg(reg, emit_fn)` - release register, auto-restore if spilled
  - `with_reg(reg, owner, emit_fn)` - context manager for scoped use
  - `mark_busy/mark_free` - for tracking existing code
  - `get_status()` - debug output
  - Statistics tracking (claims, spills, restores)
- Added `self.regs = RegisterAllocator()` to CodeGenerator

**Testing**:
- All 46 existing test files compile successfully
- Manual testing of spill/restore behavior verified

**Learnings**:
- The 'a' register needs special handling: push/pop uses 'af' (A + flags)
- Spill stack ordering is LIFO - must release in reverse order of claim
- Statistics are useful for understanding allocation patterns

### Phase 2: Instrument Existing Code ✅ COMPLETE

**Date**: 2026-01-07

**Changes**:
- Added `reg_debug` flag to CodeGenerator (enabled via --debug CLI flag)
- Added `_track_emit()` to monitor push/pop/ex operations
- Added `_check_regs_free()` for future boundary assertions
- Added `_reg_debug_log()` for debug output
- Statistics printed at end of compilation in debug mode

**Baseline Statistics** (across 46 test files):
- 501 `ex de,hl` operations
- 484 manual pop operations
- 481 manual push operations

**Learnings**:
- Push/pop are nearly balanced (3 difference from procedure prologue/epilogue)
- This baseline helps measure efficiency of allocator migration
- The `ex de,hl` pattern is very common - binary expr evaluation uses it heavily

### Phase 3: Migrate Binary Expressions ✅ COMPLETE

**Date**: 2026-01-07

**Changes**:
- Modified `_gen_binary()` to use RegisterAllocator
- Added `used_general_path` tracking to ensure `release_reg` is called after operations
- Simple path (leaf operands): Uses `mark_busy`/`mark_free` for explicit tracking
- General path (complex operands): Uses `need_reg`/`release_reg` with automatic spill/restore
- Fixed bug where `release_reg` was called before the operation, causing incorrect restores

**Statistics** (across 46 test files):
- Baseline (before): 501 ex_de_hl, 484 pops, 481 pushes
- After Phase 3: 564 ex_de_hl, 428 pops, 425 pushes
- Allocator usage: 63 claims, 2 spills, 2 restores

**Key fix**: For nested expressions like `(A+B) + (C+D)`:
- The inner expression's `release_reg` must happen AFTER its `add hl,de`, not before
- Added `used_general_path` flag to track which path was taken
- Release happens after all operations complete (ADD, SUB, MUL, etc.)

**Learnings**:
- The simple path optimization (evaluating right first when left is simple) must also check `is_free('de')` to avoid clobbering outer expression's DE value
- Spill/restore pairs must bracket the operation, not just the operand evaluation
- The comparison operations have an early return - must release DE before returning

### Phase 4: Migrate All Expression Types ✅ COMPLETE

**Date**: 2026-01-07

**Changes**:
- Migrated `_gen_subscript_addr()` to use RegisterAllocator for variable index expressions
- Migrated member array subscript in `_gen_call_expr()` to use allocator
- Migrated member array subscript in `_gen_operand_addr()` to use allocator
- All array indexing now uses `need_reg`/`release_reg` instead of manual `push hl`/`pop de`

**Statistics** (across 46 test files):
- Baseline (before): 501 ex_de_hl, 484 pops, 481 pushes
- After Phase 4: 603 ex_de_hl, 403 pops, 400 pushes
- Allocator usage: 102 claims, 9 spills, 9 restores

**Key insight**: The push/pop patterns in assignment code are safe because:
1. They save VALUE to stack (not DE) while computing target address
2. Target address computation uses allocator for subscript/binary operations
3. Assignments are statements (not sub-expressions) so no outer DE claim conflicts

**Patterns NOT migrated** (correctly):
- Interrupt handler save/restore (push/pop all registers)
- Carry chain operations (push/pop AF for carry flag preservation)
- MOVE builtin (specific pattern for source/dest pointers)
- Assignment value save (push hl to save value during address computation)

These patterns don't conflict with the allocator because they either:
- Are at statement boundaries (registers free before/after)
- Use stack for values, not for DE register saving
- Have specific semantics that require exact register usage

### Phase 5: Sethi-Ullman Optimization ✅ COMPLETE

**Date**: 2026-01-07

**Changes**:
- Added `_label_reg_need(expr)` method implementing Sethi-Ullman labeling algorithm
- Labels each expression node with minimum registers needed:
  - Leaves (literals, identifiers): 1 register
  - Unary expressions: same as operand
  - Binary expressions: max(left, right) if different, left+1 if equal
  - Subscripts, calls: 2 registers (conservative)
- Added `_lookup_symbol()` helper for consistent symbol lookup
- Modified `_gen_binary()` with three evaluation paths:
  1. **Simple path**: left preserves DE AND DE free → eval right first (no allocator)
  2. **Sethi-Ullman path**: right needs more registers → eval right first (with allocator)
  3. **General path**: left needs >= registers → eval left first (with allocator)

**How it works**:
For expression `A + ((B + C) + (D + E))`:
- Left (A): needs 1 register
- Right ((B+C) + (D+E)): needs 3 registers (each sub-expr needs 2, combined needs 3)
- Sethi-Ullman says: evaluate right first (needs more registers)
- This way, right uses 3 registers, then result goes to DE
- Left only needs 1 register, no spill needed

**Statistics** (across 46 test files):
- Same as Phase 4: 603 ex_de_hl, 403/400 pop/push, 102 claims, 9 spills
- The test suite expressions are relatively balanced, so Sethi-Ullman doesn't reduce spills
- Real benefit is for asymmetric expressions like `A + ((B + C) + (D + E))`

**Verification**:
- Created `/tmp/test_sethi_ullman.plm` with asymmetric expressions
- 5 complex expressions compiled with only 1 spill
- All 46 test files compile successfully
- DRI sources (ED, PIP) compile without spills

**Learnings**:
- Sethi-Ullman is most effective for expressions where one subtree is significantly more complex
- Real-world PL/M code tends to use simple expressions, so benefit is limited
- The simple path optimization (Phase 3) catches most practical cases
- Combined approach: simple path → Sethi-Ullman → general path
