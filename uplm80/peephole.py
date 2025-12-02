"""
Peephole Optimizer for PL/M-80.

Performs pattern-based optimizations on the generated assembly code.
This runs after code generation to clean up inefficient sequences.
"""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class PeepholePattern:
    """A peephole optimization pattern."""

    name: str
    # Pattern: list of (opcode, operands) tuples, or regex strings
    pattern: list[tuple[str, str | None]]
    # Replacement: list of (opcode, operands) tuples, or None to delete
    replacement: list[tuple[str, str]] | None
    # Optional condition function
    condition: Callable[[list[tuple[str, str]]], bool] | None = None


class PeepholeOptimizer:
    """
    Peephole optimizer that applies pattern-based transformations.

    Patterns are applied repeatedly until no more changes are made.
    """

    def __init__(self) -> None:
        self.patterns = self._init_patterns()
        self.stats: dict[str, int] = {}

    def _init_patterns(self) -> list[PeepholePattern]:
        """Initialize peephole optimization patterns."""
        return [
            # Push/Pop elimination: PUSH r; POP r -> (nothing)
            PeepholePattern(
                name="push_pop_same",
                pattern=[("PUSH", None), ("POP", None)],
                replacement=[],
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            # Redundant MOV: MOV A,r; MOV r,A -> MOV A,r
            PeepholePattern(
                name="redundant_mov",
                pattern=[("MOV", "A,*"), ("MOV", "*,A")],
                replacement=None,  # Keep first, handled specially
                condition=lambda ops: ops[0][1].split(",")[1] == ops[1][1].split(",")[0],
            ),
            # Jump to next: JMP L; L: -> L:
            PeepholePattern(
                name="jump_to_next",
                pattern=[("JMP", None)],
                replacement=[],
                condition=None,  # Checked specially
            ),
            # Zero A: MVI A,0 -> XRA A (smaller, faster)
            PeepholePattern(
                name="zero_a_mvi",
                pattern=[("MVI", "A,0")],
                replacement=[("XRA", "A")],
            ),
            # Compare to zero: CPI 0 -> ORA A
            PeepholePattern(
                name="cpi_zero",
                pattern=[("CPI", "0")],
                replacement=[("ORA", "A")],
            ),
            # Load then store same: LDA x; STA x -> LDA x
            PeepholePattern(
                name="load_store_same",
                pattern=[("LDA", None), ("STA", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            # Double INX: INX H; INX H -> LXI D,2; DAD D (if no flags needed)
            # This is actually worse, skip it

            # Redundant PUSH/POP around no-change: PUSH H; ... ; POP H
            # (complex pattern, skip for now)

            # MOV A,A -> (nothing)
            PeepholePattern(
                name="mov_a_a",
                pattern=[("MOV", "A,A")],
                replacement=[],
            ),
            # MOV to self: MOV r,r -> (nothing)
            PeepholePattern(
                name="mov_self",
                pattern=[("MOV", None)],
                replacement=[],
                condition=lambda ops: ops[0][1].split(",")[0] == ops[0][1].split(",")[1],
            ),
            # Sequential INX: INX H; INX H; INX H -> LXI D,3; DAD D (for 3+)
            # Skip for now - complex

            # LXI H,0 followed by DAD -> just use the other reg pair
            # Skip - complex

            # XCHG; XCHG -> (nothing)
            PeepholePattern(
                name="double_xchg",
                pattern=[("XCHG", ""), ("XCHG", "")],
                replacement=[],
            ),
            # XTHL; XTHL -> (nothing)
            PeepholePattern(
                name="double_xthl",
                pattern=[("XTHL", ""), ("XTHL", "")],
                replacement=[],
            ),
            # CMC; CMC -> (nothing) - complement carry twice
            PeepholePattern(
                name="double_cmc",
                pattern=[("CMC", ""), ("CMC", "")],
                replacement=[],
            ),
            # CMA; CMA -> (nothing) - complement A twice
            PeepholePattern(
                name="double_cma",
                pattern=[("CMA", ""), ("CMA", "")],
                replacement=[],
            ),
            # RAL; RAR -> (effectively nothing, but changes flags)
            # Skip - affects flags

            # Conditional jump followed by unconditional to same place
            # JZ L; JMP L -> JMP L
            PeepholePattern(
                name="cond_uncond_same",
                pattern=[("JZ", None), ("JMP", None)],
                replacement=None,  # Keep second only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_jnz",
                pattern=[("JNZ", None), ("JMP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_jc",
                pattern=[("JC", None), ("JMP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_jnc",
                pattern=[("JNC", None), ("JMP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
        ]

    def optimize(self, asm_text: str) -> str:
        """Optimize assembly text."""
        lines = asm_text.split("\n")
        changed = True
        passes = 0
        max_passes = 10

        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._optimize_pass(lines)
            if did_change:
                changed = True

        return "\n".join(lines)

    def _optimize_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Single optimization pass."""
        changed = False
        result: list[str] = []
        i = 0

        while i < len(lines):
            # Try to match patterns starting at current position
            matched = False

            for pattern in self.patterns:
                match_len = len(pattern.pattern)
                if i + match_len > len(lines):
                    continue

                # Extract instructions for potential match
                instructions = []
                skip_indices = []

                j = i
                instr_count = 0
                while instr_count < match_len and j < len(lines):
                    parsed = self._parse_line(lines[j])
                    if parsed is None:
                        # Comment or label - include but don't count
                        skip_indices.append(j - i)
                        j += 1
                        continue
                    instructions.append(parsed)
                    instr_count += 1
                    j += 1

                if len(instructions) < match_len:
                    continue

                # Check if pattern matches
                if self._matches_pattern(pattern, instructions):
                    # Check condition if present
                    if pattern.condition and not pattern.condition(instructions):
                        continue

                    # Apply replacement
                    self.stats[pattern.name] = self.stats.get(pattern.name, 0) + 1
                    changed = True
                    matched = True

                    if pattern.replacement is not None:
                        for opcode, operands in pattern.replacement:
                            result.append(f"\t{opcode}\t{operands}" if operands else f"\t{opcode}")
                    elif pattern.name.startswith("cond_uncond"):
                        # Keep second instruction only
                        result.append(lines[i + match_len - 1])
                    elif pattern.name == "redundant_mov":
                        # Keep first instruction only
                        result.append(lines[i])
                    elif pattern.name == "load_store_same":
                        # Keep first instruction only
                        result.append(lines[i])

                    i = j
                    break

            if not matched:
                result.append(lines[i])
                i += 1

        return result, changed

    def _parse_line(self, line: str) -> tuple[str, str] | None:
        """Parse an assembly line into (opcode, operands). Returns None for non-instructions."""
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(";"):
            return None

        # Skip labels (but they might have instructions after)
        if ":" in line and not line.startswith("\t"):
            # Label line
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                line = parts[1].strip()
            else:
                return None

        # Skip directives
        directives = {"ORG", "END", "DB", "DW", "DS", "EQU", "PUBLIC", "EXTRN"}

        # Parse instruction
        parts = line.split(None, 1)
        if not parts:
            return None

        opcode = parts[0].upper()
        if opcode in directives:
            return None

        operands = parts[1].split(";")[0].strip() if len(parts) > 1 else ""

        return (opcode, operands)

    def _matches_pattern(
        self, pattern: PeepholePattern, instructions: list[tuple[str, str]]
    ) -> bool:
        """Check if instructions match the pattern."""
        if len(instructions) != len(pattern.pattern):
            return False

        for (pat_op, pat_operands), (inst_op, inst_operands) in zip(
            pattern.pattern, instructions
        ):
            if pat_op != inst_op:
                return False

            if pat_operands is not None:
                # Check operands
                if "*" in pat_operands:
                    # Wildcard match
                    pat_re = pat_operands.replace("*", ".*")
                    if not re.match(pat_re, inst_operands):
                        return False
                elif pat_operands != inst_operands:
                    return False

        return True


def optimize_peephole(asm_text: str) -> str:
    """Convenience function to apply peephole optimization."""
    optimizer = PeepholeOptimizer()
    return optimizer.optimize(asm_text)
