"""
Peephole Optimizer for PL/M-80.

Performs pattern-based optimizations on the generated assembly code.
This runs after code generation to clean up inefficient sequences.

For Z80 target:
1. First applies universal patterns
2. Then translates 8080 mnemonics to Z80
3. Then applies Z80-specific patterns (JR relative jumps, etc.)
"""

import re
from dataclasses import dataclass
from typing import Callable

from .codegen import Target


# 8080 to Z80 mnemonic translations
Z80_TRANSLATIONS: dict[str, str] = {
    # Arithmetic
    "ADI": "ADD A,",
    "ACI": "ADC A,",
    "SUI": "SUB",
    "SBI": "SBC A,",
    "ANI": "AND",
    "ORI": "OR",
    "XRI": "XOR",
    "CPI": "CP",
    # Register operations
    "MOV": "LD",
    "MVI": "LD",
    "LXI": "LD",
    "LDA": "LD A,",
    "STA": "LD",
    "LHLD": "LD HL,",
    "SHLD": "LD",
    "LDAX": "LD A,",
    "STAX": "LD",
    # Arithmetic with accumulator
    "ADD": "ADD A,",
    "ADC": "ADC A,",
    "SUB": "SUB",
    "SBB": "SBC A,",
    "ANA": "AND",
    "ORA": "OR",
    "XRA": "XOR",
    "CMP": "CP",
    # Increment/Decrement
    "INR": "INC",
    "DCR": "DEC",
    "INX": "INC",
    "DCX": "DEC",
    "DAD": "ADD HL,",
    # Jumps and calls
    "JMP": "JP",
    "JZ": "JP Z,",
    "JNZ": "JP NZ,",
    "JC": "JP C,",
    "JNC": "JP NC,",
    "JP": "JP P,",
    "JM": "JP M,",
    "JPE": "JP PE,",
    "JPO": "JP PO,",
    "CZ": "CALL Z,",
    "CNZ": "CALL NZ,",
    "CC": "CALL C,",
    "CNC": "CALL NC,",
    "CP": "CALL P,",
    "CM": "CALL M,",
    "CPE": "CALL PE,",
    "CPO": "CALL PO,",
    "RZ": "RET Z",
    "RNZ": "RET NZ",
    "RC": "RET C",
    "RNC": "RET NC",
    "RP": "RET P",
    "RM": "RET M",
    "RPE": "RET PE",
    "RPO": "RET PO",
    # Stack
    "PUSH": "PUSH",
    "POP": "POP",
    "XTHL": "EX (SP),HL",
    "SPHL": "LD SP,HL",
    # Misc
    "XCHG": "EX DE,HL",
    "PCHL": "JP (HL)",
    "CMA": "CPL",
    "CMC": "CCF",
    "STC": "SCF",
    "RAL": "RLA",
    "RAR": "RRA",
    "RLC": "RLCA",
    "RRC": "RRCA",
    "DAA": "DAA",
    "NOP": "NOP",
    "HLT": "HALT",
    "DI": "DI",
    "EI": "EI",
    # I/O
    "IN": "IN A,",
    "OUT": "OUT",
    "INP": "IN A,(C)",  # Variable port
    "OUTP": "OUT (C),A",  # Variable port
}

# Z80 register pair translations
Z80_REG_PAIRS: dict[str, str] = {
    "B": "BC",
    "D": "DE",
    "H": "HL",
    "SP": "SP",
    "PSW": "AF",
}


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
    # Target-specific (None = both, Target.Z80 = Z80 only, Target.I8080 = 8080 only)
    target: Target | None = None


class PeepholeOptimizer:
    """
    Peephole optimizer that applies pattern-based transformations.

    Patterns are applied repeatedly until no more changes are made.
    """

    def __init__(self, target: Target = Target.Z80) -> None:
        self.target = target
        self.patterns = self._init_patterns()
        self.stats: dict[str, int] = {}
        # Track label positions for relative jump optimization
        self.label_positions: dict[str, int] = {}

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
            # POP r; PUSH r -> just peek, keep value on stack
            # Actually this is a "XTHL" for HL but simpler to just keep both
            # POP H; PUSH H copies TOS to HL - can we use XTHL? No, different semantics
            # Actually POP H; PUSH H loads TOS into HL AND keeps it on stack
            # Can't optimize this without more context... skip for now

            # Redundant MOV: MOV A,r; MOV r,A -> MOV A,r
            PeepholePattern(
                name="redundant_mov",
                pattern=[("MOV", "A,*"), ("MOV", "*,A")],
                replacement=None,  # Keep first, handled specially
                condition=lambda ops: ops[0][1].split(",")[1] == ops[1][1].split(",")[0],
            ),
            # Jump to next: JMP L; L: -> L:
            # This is handled specially in _optimize_pass, not as a standard pattern
            # Removed - was incorrectly deleting ALL JMP instructions!
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
            # Duplicate MOV: MOV X,Y; MOV X,Y -> MOV X,Y
            PeepholePattern(
                name="duplicate_mov",
                pattern=[("MOV", None), ("MOV", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            # Duplicate LD (Z80): LD X,Y; LD X,Y -> LD X,Y
            PeepholePattern(
                name="duplicate_ld",
                pattern=[("LD", None), ("LD", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),
            # Wasteful byte extension before byte op: LD L,A; LD H,0; SUB x -> SUB x
            # (Also for CP, AND, OR, XOR, ADD byte ops)
            PeepholePattern(
                name="useless_extend_before_sub",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("SUB", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,  # Always apply
            ),
            PeepholePattern(
                name="useless_extend_before_cp",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("CP", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,
            ),
            # Redundant byte extension: MOV L,A; MVI H,0; MOV L,A; MVI H,0 -> MOV L,A; MVI H,0
            PeepholePattern(
                name="double_byte_extend",
                pattern=[("MOV", "L,A"), ("MVI", "H,0"), ("MOV", "L,A"), ("MVI", "H,0")],
                replacement=[("MOV", "L,A"), ("MVI", "H,0")],
            ),
            # Redundant load to L: MOV L,A; MVI H,0; PUSH H; MOV L,A -> MOV L,A; MVI H,0; PUSH H
            # (The second MOV L,A before further ops is redundant if we just pushed)
            PeepholePattern(
                name="redundant_mov_l_after_push",
                pattern=[("MOV", "L,A"), ("MVI", "H,0"), ("PUSH", "H"), ("MOV", "L,A")],
                replacement=[("MOV", "L,A"), ("MVI", "H,0"), ("PUSH", "H")],
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
            # POP H; PUSH H; LXI H,x -> LXI H,x
            # The POP/PUSH just peeks at TOS (leaves stack unchanged), then HL is overwritten
            PeepholePattern(
                name="pop_push_lxi",
                pattern=[("POP", "H"), ("PUSH", "H"), ("LXI", None)],
                replacement=None,  # Handled specially - keep only LXI
                condition=lambda ops: ops[2][1].startswith("H,"),
            ),
            # PUSH PSW; STA addr; POP PSW -> STA addr
            # Saving/restoring A around a store of A is pointless
            PeepholePattern(
                name="push_sta_pop",
                pattern=[("PUSH", "PSW"), ("STA", None), ("POP", "PSW")],
                replacement=None,  # Keep only STA
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

            # LXI H,x; XCHG; POP H -> LXI D,x; POP H
            # (Moving constant to DE before popping can avoid XCHG)
            PeepholePattern(
                name="lxi_xchg_pop",
                pattern=[("LXI", None), ("XCHG", ""), ("POP", "H")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,x; XCHG; CALL y -> LXI D,x; CALL y
            # (Loading constant into DE directly saves the XCHG)
            PeepholePattern(
                name="lxi_xchg_call",
                pattern=[("LXI", None), ("XCHG", ""), ("CALL", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,x; XCHG; JMP y -> LXI D,x; JMP y
            PeepholePattern(
                name="lxi_xchg_jmp",
                pattern=[("LXI", None), ("XCHG", ""), ("JMP", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,x; XCHG; LDA y -> LXI D,x; LDA y
            PeepholePattern(
                name="lxi_xchg_lda",
                pattern=[("LXI", None), ("XCHG", ""), ("LDA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,x; XCHG; STA y -> LXI D,x; STA y
            PeepholePattern(
                name="lxi_xchg_sta",
                pattern=[("LXI", None), ("XCHG", ""), ("STA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,x; XCHG; LHLD y -> LXI D,x; LHLD y
            PeepholePattern(
                name="lxi_xchg_lhld",
                pattern=[("LXI", None), ("XCHG", ""), ("LHLD", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # PUSH H; LXI H,x; XCHG; POP H -> LXI D,x
            # (Constant goes to DE directly, PUSH/POP eliminated entirely)
            PeepholePattern(
                name="push_lxi_xchg_pop",
                pattern=[("PUSH", "H"), ("LXI", None), ("XCHG", ""), ("POP", "H")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[1][1].startswith("H,"),
            ),

            # MOV L,A; MVI H,0; XCHG; POP H -> MOV E,A; MVI D,0; POP H
            # (Putting byte value in DE directly, saves XCHG)
            PeepholePattern(
                name="mov_la_mvi_h0_xchg_pop",
                pattern=[("MOV", "L,A"), ("MVI", "H,0"), ("XCHG", ""), ("POP", "H")],
                replacement=[("MOV", "E,A"), ("MVI", "D,0"), ("POP", "H")],
            ),

            # MOV L,A; MVI H,0; DCX H; MOV A,L -> DCR A; MOV L,A
            # (16-bit decrement of byte value - just decrement A directly)
            PeepholePattern(
                name="mov_la_mvi_h0_dcx_mov_al",
                pattern=[("MOV", "L,A"), ("MVI", "H,0"), ("DCX", "H"), ("MOV", "A,L")],
                replacement=[("DCR", "A"), ("MOV", "L,A")],
            ),

            # LXI H,0; DAD SP -> LXI H,0; DAD SP (reading SP, can't optimize easily)

            # CALL x; POP D -> CALL x; POP D (can't optimize, stack cleanup)

            # LXI H,const; PUSH H; LXI H,const; PUSH H -> LXI H,const; PUSH H; PUSH H
            # (If same constant pushed twice)
            PeepholePattern(
                name="double_push_same_const",
                pattern=[("LXI", None), ("PUSH", "H"), ("LXI", None), ("PUSH", "H")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # LXI H,0FFFFH; MOV A,L; ORA H -> LXI H,0FFFFH (since 0xFFFF is always true)
            # The test is redundant
            PeepholePattern(
                name="test_true_const",
                pattern=[("LXI", "H,0FFFFH"), ("MOV", "A,L"), ("ORA", "H")],
                replacement=[("LXI", "H,0FFFFH"), ("ORA", "A")],  # Just set flags from A=L=FF
            ),

            # LXI H,1; MOV A,L; ORA H -> MVI A,1; ORA A (smaller, 1 is also true)
            PeepholePattern(
                name="test_true_const_1",
                pattern=[("LXI", "H,1"), ("MOV", "A,L"), ("ORA", "H")],
                replacement=[("MVI", "A,1"), ("ORA", "A")],
            ),

            # LXI H,1; MOV C,L -> MVI C,1 (for shift count)
            PeepholePattern(
                name="lxi_h1_mov_cl",
                pattern=[("LXI", "H,1"), ("MOV", "C,L")],
                replacement=[("MVI", "C,1")],
            ),

            # MOV A,L; MVI H,0; STA x -> MOV A,L; STA x (MVI H,0 is useless before STA)
            PeepholePattern(
                name="mov_al_mvi_h0_sta",
                pattern=[("MOV", "A,L"), ("MVI", "H,0"), ("STA", None)],
                replacement=None,  # Keep MOV A,L and STA, remove MVI H,0
                condition=lambda ops: True,
            ),

            # LXI H,0; MOV A,L; ORA H -> XRA A (sets Z, clears A)
            PeepholePattern(
                name="test_false_const",
                pattern=[("LXI", "H,0"), ("MOV", "A,L"), ("ORA", "H")],
                replacement=[("XRA", "A")],  # Sets Z flag and clears HL conceptually
            ),

            # PUSH H; SHLD x; POP H -> SHLD x (SHLD doesn't modify HL)
            PeepholePattern(
                name="push_shld_pop",
                pattern=[("PUSH", "H"), ("SHLD", None), ("POP", "H")],
                replacement=None,  # Handled specially - keep only SHLD
            ),

            # PUSH H; MVI A,x; MOV E,A; MVI D,0; POP H -> MVI E,x; MVI D,0
            # (HL not modified, PUSH/POP is wasteful)
            PeepholePattern(
                name="push_mvi_a_mov_e_mvi_d_pop",
                pattern=[("PUSH", "H"), ("MVI", None), ("MOV", "E,A"), ("MVI", "D,0"), ("POP", "H")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[1][1].startswith("A,"),
            ),

            # PUSH H; LDA x; MOV B,A; LDA y; SUB B; MOV E,A; MVI D,0; POP H
            # -> LDA x; MOV B,A; LDA y; SUB B; MOV E,A; MVI D,0
            # (HL not modified, PUSH/POP is wasteful)
            PeepholePattern(
                name="push_lda_sub_to_de_pop",
                pattern=[("PUSH", "H"), ("LDA", None), ("MOV", "B,A"), ("LDA", None),
                         ("SUB", "B"), ("MOV", "E,A"), ("MVI", "D,0"), ("POP", "H")],
                replacement=None,  # Handled specially - remove PUSH/POP
            ),

            # PUSH H; LHLD x; POP D -> XCHG; LHLD x; XCHG (Z80: use LD DE,(x) directly)
            # Gets (x) into HL and old HL into DE
            PeepholePattern(
                name="push_lhld_pop_d",
                pattern=[("PUSH", "H"), ("LHLD", None), ("POP", "D")],
                replacement=None,  # Handled specially
            ),

            # PUSH H; MOV E,A; MVI D,0; POP H -> MOV E,A; MVI D,0
            # (HL not modified, PUSH/POP is wasteful)
            PeepholePattern(
                name="push_mov_ea_mvi_d0_pop",
                pattern=[("PUSH", "H"), ("MOV", "E,A"), ("MVI", "D,0"), ("POP", "H")],
                replacement=[("MOV", "E,A"), ("MVI", "D,0")],
            ),

            # LHLD x; PUSH H; LDED y; LHLD z; CALL ??SUBDE; XCHG; POP H; CALL ??SUBDE
            # -> LDED y; LHLD z; CALL ??SUBDE; XCHG; LHLD x; CALL ??SUBDE
            # (Delay loading x until it's actually needed)
            PeepholePattern(
                name="early_load_push_subde",
                pattern=[("LHLD", None), ("PUSH", "H"), ("LDED", None), ("LHLD", None),
                         ("CALL", "??SUBDE"), ("XCHG", ""), ("POP", "H"), ("CALL", "??SUBDE")],
                replacement=None,  # Handled specially
            ),

            # ============================================================
            # Additional 8080/Z80 patterns
            # ============================================================

            # INX H; DCX H -> (nothing)
            PeepholePattern(
                name="inx_dcx_h",
                pattern=[("INX", "H"), ("DCX", "H")],
                replacement=[],
            ),
            # DCX H; INX H -> (nothing)
            PeepholePattern(
                name="dcx_inx_h",
                pattern=[("DCX", "H"), ("INX", "H")],
                replacement=[],
            ),
            # INR A; DCR A -> (nothing) - but affects flags differently
            # Skip - affects flags

            # LXI H,0; DAD SP -> LXI H,0; DAD SP (can't optimize, SP access)

            # PUSH PSW; POP PSW -> (nothing if no interrupt)
            PeepholePattern(
                name="push_pop_psw",
                pattern=[("PUSH", "PSW"), ("POP", "PSW")],
                replacement=[],
            ),

            # SHLD x; LHLD x -> SHLD x (store then load same = just store, keep HL)
            PeepholePattern(
                name="shld_lhld_same",
                pattern=[("SHLD", None), ("LHLD", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),

            # SHLD x; CALL y; LHLD x -> PUSH H; CALL y; POP H
            # (Save/restore HL around a call using stack instead of memory)
            PeepholePattern(
                name="shld_call_lhld_same",
                pattern=[("SHLD", None), ("CALL", None), ("LHLD", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # SHLD x; CALL y; LHLD x; JMP z -> PUSH H; CALL y; POP H; JMP z
            # (Same pattern but with tail optimization)
            PeepholePattern(
                name="shld_call_lhld_jmp_same",
                pattern=[("SHLD", None), ("CALL", None), ("LHLD", None), ("JMP", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # SHLD x; MVI r,n; LHLD x -> MVI r,n (MVI doesn't touch HL)
            PeepholePattern(
                name="shld_mvi_lhld_same",
                pattern=[("SHLD", None), ("MVI", None), ("LHLD", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # SHLD x; MVI r,n; LHLD x; XCHG -> MVI r,n; XCHG
            PeepholePattern(
                name="shld_mvi_lhld_xchg_same",
                pattern=[("SHLD", None), ("MVI", None), ("LHLD", None), ("XCHG", "")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # STA x; CALL y; LDA x -> PUSH PSW; CALL y; POP PSW
            # (Save/restore A around a call using stack instead of memory)
            PeepholePattern(
                name="sta_call_lda_same",
                pattern=[("STA", None), ("CALL", None), ("LDA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # STA x; MVI r,n; LDA x -> MVI r,n (MVI doesn't touch A, when r != A)
            PeepholePattern(
                name="sta_mvi_lda_same",
                pattern=[("STA", None), ("MVI", None), ("LDA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1] and not ops[1][1].startswith("A,"),
            ),

            # PUSH H; LXI H,const; MOV C,L; POP H -> MVI C,const
            # Loading a constant into C while preserving HL
            PeepholePattern(
                name="push_lxi_mov_cl_pop",
                pattern=[("PUSH", "H"), ("LXI", None), ("MOV", "C,L"), ("POP", "H")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[1][1].startswith("H,"),
            ),

            # PUSH H; MOV A,L; STA x; POP H; MVI H,0 -> MOV A,L; STA x; MVI H,0
            # The PUSH/POP is pointless since we're about to overwrite H anyway
            PeepholePattern(
                name="push_mov_sta_pop_mvi_h0",
                pattern=[("PUSH", "H"), ("MOV", "A,L"), ("STA", None), ("POP", "H"), ("MVI", "H,0")],
                replacement=None,  # Handled specially
            ),

            # MVI H,0; MVI L,x -> LXI H,x (smaller on Z80)
            # Complex - skip for now

            # ORA A; RZ -> RZ (ORA A sets Z based on A, RZ checks Z)
            # Only valid if we want to return if A==0
            # Skip - context dependent

            # ANI 0FFH -> ORA A (same effect, smaller)
            PeepholePattern(
                name="ani_ff",
                pattern=[("ANI", "0FFH")],
                replacement=[("ORA", "A")],
            ),

            # ORI 0 -> ORA A (same effect)
            PeepholePattern(
                name="ori_0",
                pattern=[("ORI", "0")],
                replacement=[("ORA", "A")],
            ),

            # XRI 0 -> ORA A (same effect, sets flags)
            PeepholePattern(
                name="xri_0",
                pattern=[("XRI", "0")],
                replacement=[("ORA", "A")],
            ),

            # ADI 0 -> ORA A (same effect on Z flag, but not on C)
            # Skip - different carry behavior

            # LDA x; ADI 1; STA x -> LXI H,x; INR M (in-place increment)
            # Saves 3 bytes when result not needed in A
            PeepholePattern(
                name="lda_adi1_sta_same",
                pattern=[("LDA", None), ("ADI", "1"), ("STA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),
            # LDA x; SUI 1; STA x -> LXI H,x; DCR M (in-place decrement)
            PeepholePattern(
                name="lda_sui1_sta_same",
                pattern=[("LDA", None), ("SUI", "1"), ("STA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1] == ops[2][1],
            ),

            # SUI 0 -> ORA A (same effect on Z flag, but not on C)
            # Skip - different carry behavior

            # DAD H -> DAD H (shift HL left, can't optimize)

            # CALL x; RET -> JMP x (tail call optimization)
            PeepholePattern(
                name="tail_call",
                pattern=[("CALL", None), ("RET", "")],
                replacement=None,  # Replaced specially
                condition=lambda ops: True,
            ),

            # RET; RET -> RET (unreachable code)
            PeepholePattern(
                name="double_ret",
                pattern=[("RET", ""), ("RET", "")],
                replacement=[("RET", "")],
            ),

            # LDA x; CPI y; JZ z; LDA x -> LDA x; CPI y; JZ z
            # (A unchanged after CPI/Jcond, so redundant reload)
            PeepholePattern(
                name="lda_cpi_jz_lda_same",
                pattern=[("LDA", None), ("CPI", None), ("JZ", None), ("LDA", None)],
                replacement=None,  # Keep first 3 only
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),
            PeepholePattern(
                name="lda_cpi_jnz_lda_same",
                pattern=[("LDA", None), ("CPI", None), ("JNZ", None), ("LDA", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),
            PeepholePattern(
                name="lda_cpi_jc_lda_same",
                pattern=[("LDA", None), ("CPI", None), ("JC", None), ("LDA", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),
            PeepholePattern(
                name="lda_cpi_jnc_lda_same",
                pattern=[("LDA", None), ("CPI", None), ("JNC", None), ("LDA", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),

            # LDA x; ORA A; JZ z; LDA x -> LDA x; ORA A; JZ z
            # (A unchanged after ORA A/Jcond)
            PeepholePattern(
                name="lda_ora_jz_lda_same",
                pattern=[("LDA", None), ("ORA", "A"), ("JZ", None), ("LDA", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),
            PeepholePattern(
                name="lda_ora_jnz_lda_same",
                pattern=[("LDA", None), ("ORA", "A"), ("JNZ", None), ("LDA", None)],
                replacement=None,
                condition=lambda ops: ops[0][1] == ops[3][1],
            ),

            # MOV B,A; MOV A,B -> MOV B,A
            PeepholePattern(
                name="mov_ba_ab",
                pattern=[("MOV", "B,A"), ("MOV", "A,B")],
                replacement=[("MOV", "B,A")],
            ),
            PeepholePattern(
                name="mov_ca_ac",
                pattern=[("MOV", "C,A"), ("MOV", "A,C")],
                replacement=[("MOV", "C,A")],
            ),
            PeepholePattern(
                name="mov_da_ad",
                pattern=[("MOV", "D,A"), ("MOV", "A,D")],
                replacement=[("MOV", "D,A")],
            ),
            PeepholePattern(
                name="mov_ea_ae",
                pattern=[("MOV", "E,A"), ("MOV", "A,E")],
                replacement=[("MOV", "E,A")],
            ),
            PeepholePattern(
                name="mov_ha_ah",
                pattern=[("MOV", "H,A"), ("MOV", "A,H")],
                replacement=[("MOV", "H,A")],
            ),
            PeepholePattern(
                name="mov_la_al",
                pattern=[("MOV", "L,A"), ("MOV", "A,L")],
                replacement=[("MOV", "L,A")],
            ),

            # MOV A,M; MOV E,A -> MOV E,M (load byte into E directly)
            PeepholePattern(
                name="mov_am_mov_ea",
                pattern=[("MOV", "A,M"), ("MOV", "E,A")],
                replacement=[("MOV", "E,M")],
            ),
            # MOV A,M; MOV D,A -> MOV D,M
            PeepholePattern(
                name="mov_am_mov_da",
                pattern=[("MOV", "A,M"), ("MOV", "D,A")],
                replacement=[("MOV", "D,M")],
            ),
            # MOV A,M; MOV C,A -> MOV C,M
            PeepholePattern(
                name="mov_am_mov_ca",
                pattern=[("MOV", "A,M"), ("MOV", "C,A")],
                replacement=[("MOV", "C,M")],
            ),
            # MOV A,M; MOV B,A -> MOV B,M
            PeepholePattern(
                name="mov_am_mov_ba",
                pattern=[("MOV", "A,M"), ("MOV", "B,A")],
                replacement=[("MOV", "B,M")],
            ),

            # LDA x; ORA A; JZ -> load and test combined
            # Skip - context dependent

            # PUSH H; XCHG; POP H -> MOV D,H; MOV E,L
            # The XCHG swaps HL<->DE, then POP restores HL, so DE = original HL
            PeepholePattern(
                name="push_xchg_pop",
                pattern=[("PUSH", "H"), ("XCHG", ""), ("POP", "H")],
                replacement=[("MOV", "D,H"), ("MOV", "E,L")],
            ),

            # MVI H,0; MOV D,H; MOV E,L -> MVI D,0; MOV E,L
            # D = H = 0, so just load D directly with 0
            PeepholePattern(
                name="mvi_h0_mov_dh_mov_el",
                pattern=[("MVI", "H,0"), ("MOV", "D,H"), ("MOV", "E,L")],
                replacement=[("MVI", "D,0"), ("MOV", "E,L")],
            ),

            # PUSH H; LXI D,x; POP H; DAD D -> LXI D,x; DAD D
            # The PUSH/POP is unnecessary since we're just adding D to H
            PeepholePattern(
                name="push_lxi_d_pop_dad",
                pattern=[("PUSH", "H"), ("LXI", None), ("POP", "H"), ("DAD", "D")],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[1][1].startswith("D,"),
            ),

            # MOV E,A; MVI D,0; POP H; XCHG; MOV M,E -> POP D; XCHG; MOV M,A
            # When storing a byte (in A) to a stacked address, we can use A directly
            # instead of copying to E then storing from E
            PeepholePattern(
                name="store_byte_via_stack",
                pattern=[("MOV", "E,A"), ("MVI", "D,0"), ("POP", "H"), ("XCHG", ""), ("MOV", "M,E")],
                replacement=[("POP", "D"), ("XCHG", ""), ("MOV", "M,A")],
            ),

            # MOV B,A; ... ; MOV A,B; SUB C -> remove MOV A,B if A==B already
            # This is context dependent, skip

            # LXI H,addr; MOV A,M -> LDA addr (direct memory access)
            PeepholePattern(
                name="lxi_mov_am_to_lda",
                pattern=[("LXI", None), ("MOV", "A,M")],
                replacement=None,  # Handled specially - convert to LDA
                condition=lambda ops: ops[0][1].startswith("H,") and not ops[0][1].startswith("H,0"),
            ),

            # LHLD x; MOV A,L; MVI H,0 -> LDA x; MOV L,A; MVI H,0
            # Only if we just need the low byte
            # Skip - complex pattern

            # STA x; LDA x -> STA x (redundant reload)
            PeepholePattern(
                name="sta_lda_same",
                pattern=[("STA", None), ("LDA", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1] == ops[1][1],
            ),

            # LXI H,const; MOV A,L; STA x -> MVI A,const; STA x
            # (When we only need the low byte of a constant)
            PeepholePattern(
                name="lxi_mov_al_sta",
                pattern=[("LXI", None), ("MOV", "A,L"), ("STA", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: ops[0][1].startswith("H,"),
            ),

            # LXI H,0; MOV L,A; MVI H,0 -> MOV L,A; MVI H,0
            # (LXI H,0 is redundant if we're about to set L from A)
            PeepholePattern(
                name="lxi_h0_mov_la",
                pattern=[("LXI", "H,0"), ("MOV", "L,A"), ("MVI", "H,0")],
                replacement=[("MOV", "L,A"), ("MVI", "H,0")],
            ),

            # MOV A,L; MVI H,0; MOV A,L -> MOV A,L; MVI H,0
            # (Second MOV A,L is redundant - A already has L)
            PeepholePattern(
                name="mov_al_mvi_h0_mov_al",
                pattern=[("MOV", "A,L"), ("MVI", "H,0"), ("MOV", "A,L")],
                replacement=[("MOV", "A,L"), ("MVI", "H,0")],
            ),

            # MOV L,A; MVI H,0; STA x -> STA x
            # (If we're just storing A, no need to extend to HL first)
            PeepholePattern(
                name="mov_la_mvi_h0_sta",
                pattern=[("MOV", "L,A"), ("MVI", "H,0"), ("STA", None)],
                replacement=None,  # Keep only STA
                condition=lambda ops: True,
            ),

            # MOV A,L; MVI H,0; ORA H -> MOV A,L; ORA A
            # (H is 0, so ORA H is same as ORA A but ORA A is 1 byte vs 2)
            # Actually: MVI H,0 then ORA H - since H was just set to 0, we can skip MVI and do ORA A
            PeepholePattern(
                name="mov_al_mvi_h0_ora_h",
                pattern=[("MOV", "A,L"), ("MVI", "H,0"), ("ORA", "H")],
                replacement=[("MOV", "A,L"), ("ORA", "A")],
            ),

            # MVI H,0; ORA H -> ORA A
            # (H is 0, so ORA H tests if A is 0 - same as ORA A)
            PeepholePattern(
                name="mvi_h0_ora_h",
                pattern=[("MVI", "H,0"), ("ORA", "H")],
                replacement=[("MVI", "H,0"), ("ORA", "A")],
            ),

            # SBB D; MOV H,A; ORA A; JM x -> SBB D; MOV H,A; JM x
            # (After 16-bit subtract, sign flag is set by SBB D, MOV doesn't affect flags)
            PeepholePattern(
                name="sbb_mov_ora_jm",
                pattern=[("SBB", "D"), ("MOV", "H,A"), ("ORA", "A"), ("JM", None)],
                replacement=None,  # Handled specially - remove ORA A
                condition=lambda ops: True,
            ),

            # SBB D; MOV H,A; ORA A; JP x -> SBB D; MOV H,A; JP x
            # (Same optimization for JP - checking if non-negative)
            PeepholePattern(
                name="sbb_mov_ora_jp",
                pattern=[("SBB", "D"), ("MOV", "H,A"), ("ORA", "A"), ("JP", None)],
                replacement=None,  # Handled specially - remove ORA A
                condition=lambda ops: True,
            ),

            # MOV B,A; STA x; MOV A,B -> STA x
            # (STA doesn't modify A, so A still equals B after the sequence)
            PeepholePattern(
                name="mov_ba_sta_mov_ab",
                pattern=[("MOV", "B,A"), ("STA", None), ("MOV", "A,B")],
                replacement=None,  # Keep only STA - handled specially
                condition=lambda ops: True,
            ),
            PeepholePattern(
                name="mov_ca_sta_mov_ac",
                pattern=[("MOV", "C,A"), ("STA", None), ("MOV", "A,C")],
                replacement=None,
                condition=lambda ops: True,
            ),
            PeepholePattern(
                name="mov_da_sta_mov_ad",
                pattern=[("MOV", "D,A"), ("STA", None), ("MOV", "A,D")],
                replacement=None,
                condition=lambda ops: True,
            ),
            PeepholePattern(
                name="mov_ea_sta_mov_ae",
                pattern=[("MOV", "E,A"), ("STA", None), ("MOV", "A,E")],
                replacement=None,
                condition=lambda ops: True,
            ),

            # SUB E; MOV L,A; MOV A,H; SBB D; MOV H,A; JM x -> SUB E; MOV A,H; SBB D; JM x
            # (When just checking sign, we don't need to store result in HL)
            PeepholePattern(
                name="sub_16bit_sign_jm",
                pattern=[("SUB", "E"), ("MOV", "L,A"), ("MOV", "A,H"), ("SBB", "D"), ("MOV", "H,A"), ("JM", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: True,
            ),
            # Same for JP (non-negative)
            PeepholePattern(
                name="sub_16bit_sign_jp",
                pattern=[("SUB", "E"), ("MOV", "L,A"), ("MOV", "A,H"), ("SBB", "D"), ("MOV", "H,A"), ("JP", None)],
                replacement=None,  # Handled specially
                condition=lambda ops: True,
            ),

            # MOV B,A; LHLD x; MOV A,B; MOV M,A -> MOV B,A; LHLD x; MOV M,B
            # (B already has the value, use it directly)
            PeepholePattern(
                name="mov_ba_lhld_mov_ab_mov_ma",
                pattern=[("MOV", "B,A"), ("LHLD", None), ("MOV", "A,B"), ("MOV", "M,A")],
                replacement=None,  # Handled specially
                condition=lambda ops: True,
            ),
            PeepholePattern(
                name="mov_ca_lhld_mov_ac_mov_ma",
                pattern=[("MOV", "C,A"), ("LHLD", None), ("MOV", "A,C"), ("MOV", "M,A")],
                replacement=None,
                condition=lambda ops: True,
            ),

            # ============================================================
            # Z80-specific patterns (applied after 8080->Z80 translation)
            # ============================================================

            # LD A,0 -> XOR A (smaller: 1 byte vs 2)
            PeepholePattern(
                name="z80_ld_a_0",
                pattern=[("LD", "A,0")],
                replacement=[("XOR", "A")],
                target=Target.Z80,
            ),

            # LD HL,0 -> LD HL,0 (can't improve, 3 bytes)

            # INC HL; INC HL -> LD DE,2; ADD HL,DE only if we can use DE
            # Skip - register pressure

            # JP Z,x; JP y where y is next instruction -> JP NZ,x
            # Complex - skip for now
        ]

    def optimize(self, asm_text: str) -> str:
        """Optimize assembly text."""
        lines = asm_text.split("\n")
        changed = True
        passes = 0
        max_passes = 10

        # Phase 1: Apply universal 8080 patterns
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._optimize_pass(lines)
            if did_change:
                changed = True

        # Phase 1.5: Register tracking optimization (eliminate redundant loads)
        lines, did_change = self._register_tracking_pass(lines)
        if did_change:
            # Run pattern matching again after register tracking
            changed = True
            passes = 0
            while changed and passes < max_passes:
                changed = False
                passes += 1
                lines, did_change = self._optimize_pass(lines)
                if did_change:
                    changed = True

        # Phase 1.6: Eliminate useless PUSH/POP pairs where register isn't modified
        lines, did_change = self._eliminate_useless_push_pop(lines)
        if did_change:
            # Run pattern matching again
            changed = True
            passes = 0
            while changed and passes < max_passes:
                changed = False
                passes += 1
                lines, did_change = self._optimize_pass(lines)
                if did_change:
                    changed = True

        # Phase 2: For Z80, translate to Z80 mnemonics
        if self.target == Target.Z80:
            lines = self._translate_to_z80(lines)

            # Phase 3: Apply Z80-specific patterns
            changed = True
            passes = 0
            while changed and passes < max_passes:
                changed = False
                passes += 1
                lines, did_change = self._optimize_z80_pass(lines)
                if did_change:
                    changed = True

            # Phase 4: Convert long jumps to relative jumps where possible
            lines = self._convert_to_relative_jumps(lines)

            # Phase 5: Apply Z80-specific patterns again (for DJNZ after JR conversion)
            lines, _ = self._optimize_z80_pass(lines)

        return "\n".join(lines)

    def _eliminate_useless_push_pop(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Eliminate PUSH H / POP H pairs when HL isn't modified between them.

        This handles the general case where we save HL, do operations that don't
        touch HL, then restore it - the save/restore is wasteful.
        """
        result: list[str] = []
        changed = False
        i = 0

        # Instructions that modify H or L (or HL as a pair)
        hl_modifying_opcodes = {
            'LHLD', 'LXI', 'INX', 'DCX', 'DAD', 'XTHL', 'SPHL', 'PCHL',
            'POP',  # POP H modifies HL
            'MOV',  # MOV H,x or MOV L,x modifies HL
            'MVI',  # MVI H,x or MVI L,x modifies HL
            'INR', 'DCR',  # INR H, DCR L etc
            'ADD', 'ADC', 'SUB', 'SBB', 'ANA', 'XRA', 'ORA', 'CMP',  # Don't modify HL but check operand
            'XCHG',  # Swaps HL with DE
            'CALL', 'RST',  # Could modify anything
        }

        def modifies_hl(line: str) -> bool:
            """Check if an instruction modifies H or L registers."""
            stripped = line.strip()
            if not stripped or stripped.startswith(';') or stripped.endswith(':'):
                return False

            parts = stripped.split(None, 1)
            if not parts:
                return False
            opcode = parts[0].upper()
            operands = parts[1] if len(parts) > 1 else ""

            # CALL modifies everything (conservatively)
            if opcode in ('CALL', 'RST'):
                return True

            # XCHG swaps HL with DE
            if opcode == 'XCHG':
                return True

            # These always modify HL
            if opcode in ('LHLD', 'XTHL', 'SPHL', 'PCHL'):
                return True

            # LXI H modifies HL
            if opcode == 'LXI' and operands.upper().startswith('H'):
                return True

            # INX H, DCX H modify HL
            if opcode in ('INX', 'DCX') and operands.upper() == 'H':
                return True

            # DAD modifies HL
            if opcode == 'DAD':
                return True

            # POP H modifies HL
            if opcode == 'POP' and operands.upper() == 'H':
                return True

            # MOV H,x or MOV L,x modifies H or L
            if opcode == 'MOV':
                dest = operands.split(',')[0].upper().strip() if ',' in operands else ""
                if dest in ('H', 'L', 'M'):  # M uses HL as pointer but doesn't modify it
                    if dest != 'M':
                        return True

            # MVI H,x or MVI L,x
            if opcode == 'MVI':
                dest = operands.split(',')[0].upper().strip() if ',' in operands else ""
                if dest in ('H', 'L'):
                    return True

            # INR H, DCR L, etc
            if opcode in ('INR', 'DCR'):
                if operands.upper().strip() in ('H', 'L'):
                    return True

            return False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            parts = stripped.split(None, 1)

            # Look for PUSH H
            if parts and parts[0].upper() == 'PUSH' and len(parts) > 1 and parts[1].upper() == 'H':
                # Found PUSH H - scan forward for matching POP H
                push_idx = i
                j = i + 1
                hl_modified = False
                pop_idx = -1

                while j < len(lines):
                    inner_line = lines[j]
                    inner_stripped = inner_line.strip()

                    # Skip empty lines and comments
                    if not inner_stripped or inner_stripped.startswith(';'):
                        j += 1
                        continue

                    # Check for label - can't optimize across labels
                    if inner_stripped.endswith(':'):
                        break

                    inner_parts = inner_stripped.split(None, 1)
                    if not inner_parts:
                        j += 1
                        continue

                    opcode = inner_parts[0].upper()
                    operands = inner_parts[1].upper() if len(inner_parts) > 1 else ""

                    # Found POP H?
                    if opcode == 'POP' and operands == 'H':
                        pop_idx = j
                        break

                    # Another PUSH H? Nested, can't optimize simply
                    if opcode == 'PUSH' and operands == 'H':
                        break

                    # Check if this instruction modifies HL
                    if modifies_hl(inner_line):
                        hl_modified = True
                        break

                    j += 1

                # If we found a matching POP H and HL wasn't modified, eliminate both
                if pop_idx > 0 and not hl_modified:
                    # Skip the PUSH H, copy everything in between, skip the POP H
                    for k in range(push_idx + 1, pop_idx):
                        result.append(lines[k])
                    i = pop_idx + 1
                    changed = True
                    continue

            result.append(line)
            i += 1

        return result, changed

    def _register_tracking_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Track register contents and eliminate redundant loads.

        Tracks what value is in each register and removes loads that
        would load the same value that's already there.
        """
        result: list[str] = []
        changed = False

        # Track register contents: reg -> value (string describing the value)
        # None means unknown, a string like "??AUTO+5" means that memory location
        regs: dict[str, str | None] = {
            'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'H': None, 'L': None
        }
        # Track memory locations loaded into HL as a pair
        hl_value: str | None = None

        def invalidate_all():
            nonlocal hl_value
            for r in regs:
                regs[r] = None
            hl_value = None

        def invalidate_reg(r: str):
            nonlocal hl_value
            regs[r] = None
            if r in ('H', 'L'):
                hl_value = None

        def invalidate_hl():
            nonlocal hl_value
            regs['H'] = None
            regs['L'] = None
            hl_value = None

        for line in lines:
            parsed = self._parse_line(line)

            # Labels and control flow invalidate tracking
            stripped = line.strip()
            if stripped and ':' in stripped and not stripped.startswith('\t'):
                # This is a label - invalidate all (could be jump target)
                invalidate_all()
                result.append(line)
                continue

            if parsed is None:
                result.append(line)
                continue

            opcode, operands = parsed

            # Control flow instructions invalidate tracking
            if opcode in ('JMP', 'JZ', 'JNZ', 'JC', 'JNC', 'JP', 'JM', 'JPE', 'JPO',
                          'CALL', 'CZ', 'CNZ', 'CC', 'CNC', 'RET', 'RZ', 'RNZ', 'RC', 'RNC',
                          'PCHL', 'RST'):
                invalidate_all()
                result.append(line)
                continue

            # Track LDA - A gets value from memory
            if opcode == 'LDA':
                addr = operands
                if regs['A'] == f"mem:{addr}":
                    # Already have this value in A - skip this instruction
                    changed = True
                    self.stats['redundant_load_eliminated'] = self.stats.get('redundant_load_eliminated', 0) + 1
                    continue
                regs['A'] = f"mem:{addr}"
                result.append(line)
                continue

            # Track STA - memory gets A, but A is unchanged
            if opcode == 'STA':
                # A is unchanged, memory now has A's value
                result.append(line)
                continue

            # Track LHLD - HL gets value from memory
            if opcode == 'LHLD':
                addr = operands
                if hl_value == f"mem:{addr}":
                    # Already have this value in HL - skip this instruction
                    changed = True
                    self.stats['redundant_load_eliminated'] = self.stats.get('redundant_load_eliminated', 0) + 1
                    continue
                hl_value = f"mem:{addr}"
                regs['H'] = None  # Individual regs unknown
                regs['L'] = None
                result.append(line)
                continue

            # Track SHLD - memory gets HL, but HL is unchanged
            if opcode == 'SHLD':
                result.append(line)
                continue

            # Track LXI H,const
            if opcode == 'LXI' and operands.startswith('H,'):
                const = operands[2:]
                if hl_value == f"const:{const}":
                    # Already have this constant in HL - skip
                    changed = True
                    self.stats['redundant_load_eliminated'] = self.stats.get('redundant_load_eliminated', 0) + 1
                    continue
                hl_value = f"const:{const}"
                regs['H'] = None
                regs['L'] = None
                result.append(line)
                continue

            # Track LXI for other register pairs
            if opcode == 'LXI':
                if operands.startswith('D,'):
                    regs['D'] = None
                    regs['E'] = None
                elif operands.startswith('B,'):
                    regs['B'] = None
                    regs['C'] = None
                result.append(line)
                continue

            # Track MVI
            if opcode == 'MVI':
                parts = operands.split(',')
                if len(parts) == 2:
                    reg, val = parts[0], parts[1]
                    if reg in regs:
                        if regs[reg] == f"const:{val}":
                            # Already have this constant - skip
                            changed = True
                            self.stats['redundant_load_eliminated'] = self.stats.get('redundant_load_eliminated', 0) + 1
                            continue
                        regs[reg] = f"const:{val}"
                        if reg in ('H', 'L'):
                            hl_value = None
                result.append(line)
                continue

            # Track MOV
            if opcode == 'MOV':
                parts = operands.split(',')
                if len(parts) == 2:
                    dst, src = parts[0], parts[1]
                    if dst in regs and src in regs:
                        if regs[dst] == regs[src] and regs[dst] is not None:
                            # Same value - skip
                            changed = True
                            self.stats['redundant_load_eliminated'] = self.stats.get('redundant_load_eliminated', 0) + 1
                            continue
                        regs[dst] = regs[src]
                        if dst in ('H', 'L'):
                            hl_value = None
                    elif dst in regs:
                        # Loading from memory via M or something else
                        invalidate_reg(dst)
                result.append(line)
                continue

            # Instructions that modify registers
            if opcode in ('ADD', 'ADC', 'SUB', 'SBB', 'ANA', 'ORA', 'XRA', 'CMP',
                          'ADI', 'ACI', 'SUI', 'SBI', 'ANI', 'ORI', 'XRI', 'CPI'):
                regs['A'] = None  # A is modified
                result.append(line)
                continue

            if opcode in ('INR', 'DCR'):
                if operands in regs:
                    invalidate_reg(operands)
                result.append(line)
                continue

            if opcode in ('INX', 'DCX'):
                if operands == 'H':
                    invalidate_hl()
                elif operands == 'D':
                    regs['D'] = None
                    regs['E'] = None
                elif operands == 'B':
                    regs['B'] = None
                    regs['C'] = None
                result.append(line)
                continue

            if opcode == 'DAD':
                invalidate_hl()
                result.append(line)
                continue

            if opcode == 'XCHG':
                # Swap HL and DE
                hl_value = None  # For simplicity, just invalidate
                regs['H'], regs['D'] = regs['D'], regs['H']
                regs['L'], regs['E'] = regs['E'], regs['L']
                result.append(line)
                continue

            if opcode in ('PUSH', 'POP'):
                if opcode == 'POP':
                    if operands == 'H':
                        invalidate_hl()
                    elif operands == 'D':
                        regs['D'] = None
                        regs['E'] = None
                    elif operands == 'B':
                        regs['B'] = None
                        regs['C'] = None
                    elif operands == 'PSW':
                        regs['A'] = None
                result.append(line)
                continue

            # Rotates modify A
            if opcode in ('RLC', 'RRC', 'RAL', 'RAR'):
                regs['A'] = None
                result.append(line)
                continue

            # Other instructions - be conservative and invalidate A
            if opcode in ('CMA', 'DAA'):
                regs['A'] = None

            result.append(line)

        return result, changed

    def _translate_to_z80(self, lines: list[str]) -> list[str]:
        """Translate 8080 mnemonics to Z80 equivalents."""
        result: list[str] = []
        for line in lines:
            translated = self._translate_line_to_z80(line)
            result.append(translated)
        return result

    def _translate_line_to_z80(self, line: str) -> str:
        """Translate a single line from 8080 to Z80 mnemonics."""
        stripped = line.strip()

        # Skip empty, comments, labels (without instructions)
        if not stripped or stripped.startswith(";"):
            return line

        # Handle labels with potential instruction after
        label_prefix = ""
        if ":" in stripped and not stripped.startswith("\t"):
            parts = stripped.split(":", 1)
            label_prefix = parts[0] + ":"
            if len(parts) > 1 and parts[1].strip():
                stripped = parts[1].strip()
            else:
                return line  # Just a label

        # Skip directives
        directives = {"ORG", "END", "DB", "DW", "DS", "EQU", "PUBLIC", "EXTRN"}
        parts = stripped.split(None, 1)
        if not parts:
            return line
        opcode = parts[0].upper()
        if opcode in directives:
            return line

        operands = parts[1].split(";")[0].strip() if len(parts) > 1 else ""
        comment = ""
        if ";" in line:
            comment = "\t;" + line.split(";", 1)[1]

        # Translate based on opcode
        z80_line = self._translate_instruction(opcode, operands)
        if z80_line:
            if label_prefix:
                return f"{label_prefix}\t{z80_line}{comment}"
            return f"\t{z80_line}{comment}"

        return line

    def _translate_instruction(self, opcode: str, operands: str) -> str | None:
        """Translate a single 8080 instruction to Z80."""
        # Special cases that need operand transformation

        # MOV r,r -> LD r,r
        if opcode == "MOV":
            return f"LD {operands}"

        # MVI r,n -> LD r,n
        if opcode == "MVI":
            return f"LD {operands}"

        # LXI rp,nn -> LD rp,nn (with register pair translation)
        if opcode == "LXI":
            parts = operands.split(",", 1)
            if len(parts) == 2:
                rp = Z80_REG_PAIRS.get(parts[0].upper(), parts[0])
                return f"LD {rp},{parts[1]}"

        # LDA addr -> LD A,(addr)
        if opcode == "LDA":
            return f"LD A,({operands})"

        # STA addr -> LD (addr),A
        if opcode == "STA":
            return f"LD ({operands}),A"

        # LHLD addr -> LD HL,(addr)
        if opcode == "LHLD":
            return f"LD HL,({operands})"

        # SHLD addr -> LD (addr),HL
        if opcode == "SHLD":
            return f"LD ({operands}),HL"

        # LDAX rp -> LD A,(rp)
        if opcode == "LDAX":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"LD A,({rp})"

        # STAX rp -> LD (rp),A
        if opcode == "STAX":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"LD ({rp}),A"

        # ADD r -> ADD A,r
        if opcode == "ADD" and not operands.startswith("A,"):
            return f"ADD A,{operands}"

        # ADC r -> ADC A,r
        if opcode == "ADC" and not operands.startswith("A,"):
            return f"ADC A,{operands}"

        # SUB r -> SUB r (no change needed, Z80 SUB doesn't use A prefix)
        if opcode == "SUB":
            return f"SUB {operands}"

        # SBB r -> SBC A,r
        if opcode == "SBB":
            return f"SBC A,{operands}"

        # ANA r -> AND r
        if opcode == "ANA":
            return f"AND {operands}"

        # ORA r -> OR r
        if opcode == "ORA":
            return f"OR {operands}"

        # XRA r -> XOR r
        if opcode == "XRA":
            return f"XOR {operands}"

        # CMP r -> CP r
        if opcode == "CMP":
            return f"CP {operands}"

        # INR r -> INC r
        if opcode == "INR":
            return f"INC {operands}"

        # DCR r -> DEC r
        if opcode == "DCR":
            return f"DEC {operands}"

        # INX rp -> INC rp
        if opcode == "INX":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"INC {rp}"

        # DCX rp -> DEC rp
        if opcode == "DCX":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"DEC {rp}"

        # DAD rp -> ADD HL,rp
        if opcode == "DAD":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"ADD HL,{rp}"

        # Immediate arithmetic
        if opcode == "ADI":
            return f"ADD A,{operands}"
        if opcode == "ACI":
            return f"ADC A,{operands}"
        if opcode == "SUI":
            return f"SUB {operands}"
        if opcode == "SBI":
            return f"SBC A,{operands}"
        if opcode == "ANI":
            return f"AND {operands}"
        if opcode == "ORI":
            return f"OR {operands}"
        if opcode == "XRI":
            return f"XOR {operands}"
        if opcode == "CPI":
            return f"CP {operands}"

        # Jumps
        if opcode == "JMP":
            return f"JP {operands}"
        if opcode == "JZ":
            return f"JP Z,{operands}"
        if opcode == "JNZ":
            return f"JP NZ,{operands}"
        if opcode == "JC":
            return f"JP C,{operands}"
        if opcode == "JNC":
            return f"JP NC,{operands}"
        if opcode == "JM":
            return f"JP M,{operands}"
        if opcode == "JPE":
            return f"JP PE,{operands}"
        if opcode == "JPO":
            return f"JP PO,{operands}"

        # Calls
        if opcode == "CZ":
            return f"CALL Z,{operands}"
        if opcode == "CNZ":
            return f"CALL NZ,{operands}"
        if opcode == "CC":
            return f"CALL C,{operands}"
        if opcode == "CNC":
            return f"CALL NC,{operands}"
        if opcode == "CM":
            return f"CALL M,{operands}"
        if opcode == "CPE":
            return f"CALL PE,{operands}"
        if opcode == "CPO":
            return f"CALL PO,{operands}"

        # Returns
        if opcode == "RZ":
            return "RET Z"
        if opcode == "RNZ":
            return "RET NZ"
        if opcode == "RC":
            return "RET C"
        if opcode == "RNC":
            return "RET NC"
        if opcode == "RM":
            return "RET M"
        if opcode == "RPE":
            return "RET PE"
        if opcode == "RPO":
            return "RET PO"

        # PUSH/POP with register pair translation
        if opcode == "PUSH":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"PUSH {rp}"
        if opcode == "POP":
            rp = Z80_REG_PAIRS.get(operands.upper(), operands)
            return f"POP {rp}"

        # Misc
        if opcode == "XTHL":
            return "EX (SP),HL"
        if opcode == "SPHL":
            return "LD SP,HL"
        if opcode == "XCHG":
            return "EX DE,HL"
        if opcode == "PCHL":
            return "JP (HL)"
        if opcode == "CMA":
            return "CPL"
        if opcode == "CMC":
            return "CCF"
        if opcode == "STC":
            return "SCF"
        if opcode == "RAL":
            return "RLA"
        if opcode == "RAR":
            return "RRA"
        if opcode == "RLC":
            return "RLCA"
        if opcode == "RRC":
            return "RRCA"
        if opcode == "HLT":
            return "HALT"

        # I/O
        if opcode == "IN":
            return f"IN A,({operands})"
        if opcode == "OUT":
            return f"OUT ({operands}),A"
        if opcode == "INP":
            return "IN A,(C)"
        if opcode == "OUTP":
            return "OUT (C),A"

        # Instructions that don't change
        if opcode in ("CALL", "RET", "DAA", "NOP", "DI", "EI", "RST"):
            if operands:
                return f"{opcode} {operands}"
            return opcode

        return None

    def _optimize_z80_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Apply Z80-specific optimizations."""
        changed = False
        result: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            parsed = self._parse_z80_line(line)

            if parsed:
                opcode, operands = parsed

                # LD A,0 -> XOR A (1 byte vs 2)
                if opcode == "LD" and operands == "A,0":
                    result.append("\tXOR A")
                    changed = True
                    self.stats["z80_xor_a"] = self.stats.get("z80_xor_a", 0) + 1
                    i += 1
                    continue

                # LD r,0 -> LD r,0 can sometimes use XOR for A
                # OR A -> OR A (can't improve)

                # EX DE,HL; EX DE,HL -> (nothing)
                if opcode == "EX" and operands == "DE,HL" and i + 1 < len(lines):
                    next_parsed = self._parse_z80_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "EX" and next_parsed[1] == "DE,HL":
                        changed = True
                        self.stats["z80_double_ex"] = self.stats.get("z80_double_ex", 0) + 1
                        i += 2
                        continue

                # INC HL; DEC HL -> (nothing)
                if opcode == "INC" and operands == "HL" and i + 1 < len(lines):
                    next_parsed = self._parse_z80_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "DEC" and next_parsed[1] == "HL":
                        changed = True
                        self.stats["z80_inc_dec_hl"] = self.stats.get("z80_inc_dec_hl", 0) + 1
                        i += 2
                        continue

                # DEC HL; INC HL -> (nothing)
                if opcode == "DEC" and operands == "HL" and i + 1 < len(lines):
                    next_parsed = self._parse_z80_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "INC" and next_parsed[1] == "HL":
                        changed = True
                        self.stats["z80_dec_inc_hl"] = self.stats.get("z80_dec_inc_hl", 0) + 1
                        i += 2
                        continue

                # LD (addr),HL; LD HL,(addr) -> LD (addr),HL (same address)
                if opcode == "LD" and operands.startswith("(") and operands.endswith("),HL"):
                    addr = operands[1:-4]
                    if i + 1 < len(lines):
                        next_parsed = self._parse_z80_line(lines[i + 1].strip())
                        if next_parsed and next_parsed[0] == "LD" and next_parsed[1] == f"HL,({addr})":
                            result.append(lines[i])
                            changed = True
                            self.stats["z80_ld_hl_same"] = self.stats.get("z80_ld_hl_same", 0) + 1
                            i += 2
                            continue

                # DEC B; JR NZ,label -> DJNZ label (saves 1 byte)
                if opcode == "DEC" and operands == "B" and i + 1 < len(lines):
                    next_parsed = self._parse_z80_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "JR" and next_parsed[1].startswith("NZ,"):
                        target = next_parsed[1][3:]  # Remove "NZ,"
                        result.append(f"\tDJNZ {target}")
                        changed = True
                        self.stats["z80_djnz"] = self.stats.get("z80_djnz", 0) + 1
                        i += 2
                        continue

                # CP 1; JP Z/NZ -> DEC A; JP Z/NZ (saves 1 byte: CP 1 is 2 bytes, DEC A is 1 byte)
                # Only valid when A is not needed afterward (comparison just sets flags)
                if opcode == "CP" and operands == "1" and i + 1 < len(lines):
                    next_parsed = self._parse_z80_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] in ("JP", "JR") and next_parsed[1].startswith(("Z,", "NZ,")):
                        result.append("\tDEC A")
                        changed = True
                        self.stats["z80_cp1_dec"] = self.stats.get("z80_cp1_dec", 0) + 1
                        i += 1
                        continue

                # Skip trick: JP label; LD A,0FFH; label: -> DB 21H; LD A,0FFH; label:
                # The 21H byte is the opcode for LD HL,nn which "eats" the next 2 bytes
                # This saves 2 bytes (JP is 3 bytes, DB 21H is 1 byte)
                if opcode == "JP" and "," not in operands and operands != "(HL)" and i + 2 < len(lines):
                    target = operands.strip()
                    next_line = lines[i + 1].strip()
                    next_parsed = self._parse_z80_line(next_line)
                    third_line = lines[i + 2].strip()
                    # Check if next instruction is LD A,0FFH (2 bytes) and third is the target label
                    if (next_parsed and next_parsed[0] == "LD" and next_parsed[1] == "A,0FFH" and
                        third_line.startswith(target + ":")):
                        result.append("\tDB 21H\t; skip next 2 bytes (LD HL,nn opcode)")
                        changed = True
                        self.stats["z80_skip_trick"] = self.stats.get("z80_skip_trick", 0) + 1
                        i += 1
                        continue

                # LD HL,(addr1); PUSH HL; LD HL,(addr2); EX DE,HL; POP HL
                # -> LD DE,(addr2); LD HL,(addr1)
                # Saves 3 bytes (10 -> 7) by using Z80's LD DE,(nn) instruction
                if opcode == "LD" and operands.startswith("HL,(") and operands.endswith(")"):
                    addr1 = operands[3:]  # Keep the (addr) part
                    if i + 4 < len(lines):
                        p1 = self._parse_z80_line(lines[i + 1].strip())
                        p2 = self._parse_z80_line(lines[i + 2].strip())
                        p3 = self._parse_z80_line(lines[i + 3].strip())
                        p4 = self._parse_z80_line(lines[i + 4].strip())
                        if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                            p2 and p2[0] == "LD" and p2[1].startswith("HL,(") and
                            p3 and p3[0] == "EX" and p3[1] == "DE,HL" and
                            p4 and p4[0] == "POP" and p4[1] == "HL"):
                            addr2 = p2[1][3:]  # Get (addr2)
                            result.append(f"\tLD DE,{addr2}")
                            result.append(f"\tLD HL,{addr1}")
                            changed = True
                            self.stats["z80_ld_de_nn"] = self.stats.get("z80_ld_de_nn", 0) + 1
                            i += 5
                            continue

                # LD HL,const; PUSH HL; LD HL,(addr); LD E,(HL); LD D,0; POP HL
                # -> LD HL,(addr); LD E,(HL); LD D,0; LD HL,const
                # Defer loading constant until after memory access, saves PUSH/POP (2 bytes)
                if opcode == "LD" and operands.startswith("HL,") and not operands.startswith("HL,("):
                    const_val = operands[3:]  # Get the constant
                    if i + 5 < len(lines):
                        p1 = self._parse_z80_line(lines[i + 1].strip())
                        p2 = self._parse_z80_line(lines[i + 2].strip())
                        p3 = self._parse_z80_line(lines[i + 3].strip())
                        p4 = self._parse_z80_line(lines[i + 4].strip())
                        p5 = self._parse_z80_line(lines[i + 5].strip())
                        if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                            p2 and p2[0] == "LD" and p2[1].startswith("HL,(") and
                            p3 and p3[0] == "LD" and p3[1] in ("E,(HL)", "E,M") and
                            p4 and p4[0] == "LD" and p4[1] in ("D,0", "D,00H") and
                            p5 and p5[0] == "POP" and p5[1] == "HL"):
                            addr = p2[1][3:]  # Get (addr)
                            result.append(f"\tLD HL,{addr}")
                            result.append("\tLD E,(HL)")
                            result.append("\tLD D,0")
                            result.append(f"\tLD HL,{const_val}")
                            changed = True
                            self.stats["z80_defer_const_load"] = self.stats.get("z80_defer_const_load", 0) + 1
                            i += 6
                            continue

                # LD HL,0; PUSH HL; LD A,L; LD (addr),A; POP HL
                # -> XOR A; LD (addr),A; LD HL,0
                # Using HL just to get 0 into A is wasteful. XOR A is 1 byte.
                # Saves PUSH/POP (2 bytes) and LXI->XOR saves 2 more bytes
                if opcode == "LD" and operands == "HL,0":
                    if i + 4 < len(lines):
                        p1 = self._parse_z80_line(lines[i + 1].strip())
                        p2 = self._parse_z80_line(lines[i + 2].strip())
                        p3 = self._parse_z80_line(lines[i + 3].strip())
                        p4 = self._parse_z80_line(lines[i + 4].strip())
                        if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                            p2 and p2[0] == "LD" and p2[1] == "A,L" and
                            p3 and p3[0] == "LD" and p3[1].startswith("(") and p3[1].endswith("),A") and
                            p4 and p4[0] == "POP" and p4[1] == "HL"):
                            addr = p3[1][:-2]  # Get (addr) part
                            result.append("\tXOR A")
                            result.append(f"\tLD {addr},A")
                            result.append("\tLD HL,0")
                            changed = True
                            self.stats["z80_xor_a_store"] = self.stats.get("z80_xor_a_store", 0) + 1
                            i += 5
                            continue

                # LD HL,0; LD A,L; LD (addr),A -> XOR A; LD (addr),A; LD HL,0
                # Loading 0 via HL just to get it into A is wasteful
                # XOR A is 1 byte vs LD A,L which is 1 byte, but we eliminate the dependency
                # Keep LD HL,0 at end in case subsequent code needs it
                if opcode == "LD" and operands == "HL,0":
                    if i + 2 < len(lines):
                        p1 = self._parse_z80_line(lines[i + 1].strip())
                        p2 = self._parse_z80_line(lines[i + 2].strip())
                        if (p1 and p1[0] == "LD" and p1[1] == "A,L" and
                            p2 and p2[0] == "LD" and p2[1].startswith("(") and p2[1].endswith("),A")):
                            addr = p2[1][:-2]  # Get (addr) part
                            result.append("\tXOR A")
                            result.append(f"\tLD {addr},A")
                            result.append("\tLD HL,0")
                            changed = True
                            self.stats["z80_xor_a_store_simple"] = self.stats.get("z80_xor_a_store_simple", 0) + 1
                            i += 3
                            continue

                # POP HL; PUSH HL; LD HL,x -> LD HL,x
                # POP/PUSH just peeks at TOS (no stack change), then HL is overwritten
                if opcode == "POP" and operands == "HL" and i + 2 < len(lines):
                    p1 = self._parse_z80_line(lines[i + 1].strip())
                    p2 = self._parse_z80_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                        p2 and p2[0] == "LD" and p2[1].startswith("HL,")):
                        result.append(lines[i + 2])  # Keep only LD HL,x
                        changed = True
                        self.stats["z80_pop_push_ld"] = self.stats.get("z80_pop_push_ld", 0) + 1
                        i += 3
                        continue

                # PUSH HL; LD HL,(addr); EX DE,HL; POP HL -> LD DE,(addr)
                # Z80 has direct LD DE,(addr) which 8080 doesn't have
                if opcode == "PUSH" and operands == "HL" and i + 3 < len(lines):
                    p1 = self._parse_z80_line(lines[i + 1].strip())
                    p2 = self._parse_z80_line(lines[i + 2].strip())
                    p3 = self._parse_z80_line(lines[i + 3].strip())
                    if (p1 and p1[0] == "LD" and p1[1].startswith("HL,(") and p1[1].endswith(")") and
                        p2 and p2[0] == "EX" and p2[1] == "DE,HL" and
                        p3 and p3[0] == "POP" and p3[1] == "HL"):
                        addr = p1[1][3:]  # Get (addr) including parens
                        result.append(f"\tLD DE,{addr}")
                        changed = True
                        self.stats["z80_ld_de_addr"] = self.stats.get("z80_ld_de_addr", 0) + 1
                        i += 4
                        continue

                # PUSH AF; LD (addr),A; POP AF -> LD (addr),A
                # Saving/restoring A around a store of A is pointless
                if opcode == "PUSH" and operands == "AF" and i + 2 < len(lines):
                    p1 = self._parse_z80_line(lines[i + 1].strip())
                    p2 = self._parse_z80_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "LD" and p1[1].startswith("(") and p1[1].endswith("),A") and
                        p2 and p2[0] == "POP" and p2[1] == "AF"):
                        result.append(lines[i + 1])  # Keep only LD (addr),A
                        changed = True
                        self.stats["z80_push_sta_pop"] = self.stats.get("z80_push_sta_pop", 0) + 1
                        i += 3
                        continue

                # LD HL,const; LD r,L -> LD r,const (when const fits in byte)
                # Loading constant via HL just to move low byte is wasteful
                if opcode == "LD" and operands.startswith("HL,") and not operands.startswith("HL,("):
                    const_val = operands[3:]
                    if i + 1 < len(lines):
                        p1 = self._parse_z80_line(lines[i + 1].strip())
                        if p1 and p1[0] == "LD" and p1[1].endswith(",L"):
                            dest_reg = p1[1][:-2]  # Get destination register
                            if dest_reg in ("A", "B", "C", "D", "E"):
                                result.append(f"\tLD {dest_reg},{const_val}")
                                changed = True
                                self.stats["z80_ld_via_hl"] = self.stats.get("z80_ld_via_hl", 0) + 1
                                i += 2
                                continue

                # PUSH HL; LD HL,const; <ops not using HL>; POP HL -> <ops not using HL>
                # If HL is saved, loaded with constant, then restored without using it
                if opcode == "PUSH" and operands == "HL" and i + 2 < len(lines):
                    p1 = self._parse_z80_line(lines[i + 1].strip())
                    # Check for LD HL,const (not memory load)
                    if p1 and p1[0] == "LD" and p1[1].startswith("HL,") and not p1[1].startswith("HL,("):
                        # Look for operations that don't use HL, followed by POP HL
                        j = i + 2
                        middle_ops = []
                        hl_used = False
                        while j < len(lines):
                            pj = self._parse_z80_line(lines[j].strip())
                            if not pj:
                                middle_ops.append(lines[j])
                                j += 1
                                continue
                            if pj[0] == "POP" and pj[1] == "HL":
                                # Found the matching POP - can eliminate PUSH/LD/POP
                                if not hl_used:
                                    for op in middle_ops:
                                        result.append(op)
                                    changed = True
                                    self.stats["z80_push_ld_pop_unused"] = self.stats.get("z80_push_ld_pop_unused", 0) + 1
                                    i = j + 1
                                break
                            # Check if this op uses HL
                            op_str = pj[1] if pj[1] else ""
                            if "HL" in pj[0] or "HL" in op_str or "(HL)" in op_str or "H" in op_str.split(",")[0] or "L" in op_str.split(",")[0]:
                                hl_used = True
                                break
                            middle_ops.append(lines[j])
                            j += 1
                            # Limit search depth
                            if len(middle_ops) > 5:
                                break
                        if not hl_used and j < len(lines) and changed:
                            continue

            result.append(lines[i])
            i += 1

        return result, changed

    def _parse_z80_line(self, line: str) -> tuple[str, str] | None:
        """Parse a Z80 assembly line."""
        if not line or line.startswith(";"):
            return None
        if ":" in line and not line.startswith("\t"):
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                line = parts[1].strip()
            else:
                return None

        parts = line.split(None, 1)
        if not parts:
            return None
        opcode = parts[0].upper()
        operands = parts[1].split(";")[0].strip() if len(parts) > 1 else ""
        return (opcode, operands)

    def _convert_to_relative_jumps(self, lines: list[str]) -> list[str]:
        """Convert JP to JR where the jump is within range (-126 to +129 bytes)."""
        # First pass: find all label positions (approximate by line number)
        # This is a simple approximation - actual byte distances would need
        # proper assembly
        label_lines: dict[str, int] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("\t"):
                label = stripped.split(":")[0].strip()
                label_lines[label] = i

        # Second pass: convert jumps where target is close
        result: list[str] = []
        for i, line in enumerate(lines):
            parsed = self._parse_z80_line(line.strip())

            if parsed:
                opcode, operands = parsed

                # Check for convertible jumps (JP, JP Z, JP NZ, JP C, JP NC)
                convert_map = {
                    "JP": ("JR", None),
                    "JP Z,": ("JR Z,", 5),
                    "JP NZ,": ("JR NZ,", 6),
                    "JP C,": ("JR C,", 5),
                    "JP NC,": ("JR NC,", 6),
                }

                for jp_prefix, (jr_prefix, prefix_len) in convert_map.items():
                    if prefix_len:
                        if opcode == "JP" and operands.startswith(jp_prefix[3:]):
                            # Conditional jump
                            target = operands[prefix_len - 3:].strip()
                            if target in label_lines:
                                distance = label_lines[target] - i
                                # Rough estimate: each line ~2-3 bytes on average
                                # JR range is -126 to +129 bytes
                                # Use conservative estimate of ~40 lines
                                if -40 < distance < 40:
                                    result.append(f"\t{jr_prefix}{target}")
                                    self.stats["z80_jr_convert"] = self.stats.get("z80_jr_convert", 0) + 1
                                    break
                    else:
                        if opcode == "JP" and "," not in operands and operands != "(HL)":
                            # Unconditional JP to label
                            target = operands.strip()
                            if target in label_lines:
                                distance = label_lines[target] - i
                                if -40 < distance < 40:
                                    result.append(f"\tJR {target}")
                                    self.stats["z80_jr_convert"] = self.stats.get("z80_jr_convert", 0) + 1
                                    break
                else:
                    result.append(line)
                    continue
                continue

            result.append(line)

        return result

    def _optimize_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Single optimization pass."""
        changed = False
        result: list[str] = []
        i = 0

        while i < len(lines):
            # Special case: JMP/JR to immediately following label
            parsed = self._parse_line(lines[i])
            if parsed and parsed[0] in ("JMP", "JR"):
                target = parsed[1]
                # Look ahead for the target label (skip comments/empty lines)
                j = i + 1
                found_target = False
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith(";"):
                        j += 1
                        continue
                    # Check if this is a label line
                    if ":" in next_line and not next_line.startswith("\t"):
                        label = next_line.split(":")[0].strip()
                        if label == target:
                            # JMP to next label - remove the JMP, keep going
                            self.stats["jump_to_next"] = self.stats.get("jump_to_next", 0) + 1
                            changed = True
                            found_target = True
                    break
                if found_target:
                    i += 1
                    continue

            # Special case: ORA A; JZ label where label: XRA A; RET -> ORA A; RZ
            # This handles the common IF condition THEN ... RETURN TRUE; RETURN FALSE pattern
            # Also handles: ORA A; JNZ label where label: MVI A,1; RET -> ORA A; RNZ
            if parsed and parsed[0] == "ORA" and parsed[1] == "A":
                # Check if next instruction is JZ or JNZ
                next_parsed = None
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith(";"):
                        j += 1
                        continue
                    next_parsed = self._parse_line(lines[j])
                    break

                if next_parsed and next_parsed[0] in ("JZ", "JNZ"):
                    is_jz = next_parsed[0] == "JZ"
                    target = next_parsed[1]
                    # Search for the target label and check pattern
                    for k in range(j + 1, min(j + 100, len(lines))):
                        line_k = lines[k].strip()
                        if line_k.startswith(target + ":"):
                            # Found the label, check next instructions
                            m = k + 1
                            while m < len(lines):
                                line_m = lines[m].strip()
                                if not line_m or line_m.startswith(";"):
                                    m += 1
                                    continue
                                parsed_m = self._parse_line(lines[m])

                                # For JZ: look for XRA A; RET (return 0)
                                if is_jz and parsed_m and parsed_m[0] == "XRA" and parsed_m[1] == "A":
                                    # Check for RET after XRA A
                                    n = m + 1
                                    while n < len(lines):
                                        line_n = lines[n].strip()
                                        if not line_n or line_n.startswith(";"):
                                            n += 1
                                            continue
                                        parsed_n = self._parse_line(lines[n])
                                        if parsed_n and parsed_n[0] == "RET" and parsed_n[1] == "":
                                            # Pattern matched! Replace ORA A; JZ label with ORA A; RZ
                                            result.append(lines[i])  # Keep ORA A
                                            result.append("\tRZ")    # Replace JZ with RZ
                                            self.stats["jz_xra_ret_to_rz"] = self.stats.get("jz_xra_ret_to_rz", 0) + 1
                                            changed = True
                                            i = j + 1  # Skip past JZ
                                            break
                                        break

                                # For JNZ: look for MVI A,1; RET (return 1/TRUE)
                                elif not is_jz and parsed_m and parsed_m[0] == "MVI" and parsed_m[1] == "A,1":
                                    # Check for RET after MVI A,1
                                    n = m + 1
                                    while n < len(lines):
                                        line_n = lines[n].strip()
                                        if not line_n or line_n.startswith(";"):
                                            n += 1
                                            continue
                                        parsed_n = self._parse_line(lines[n])
                                        if parsed_n and parsed_n[0] == "RET" and parsed_n[1] == "":
                                            # Pattern matched! Replace ORA A; JNZ label with ORA A; RNZ
                                            result.append(lines[i])  # Keep ORA A
                                            result.append("\tRNZ")   # Replace JNZ with RNZ
                                            self.stats["jnz_mvi_ret_to_rnz"] = self.stats.get("jnz_mvi_ret_to_rnz", 0) + 1
                                            changed = True
                                            i = j + 1  # Skip past JNZ
                                            break
                                        break
                                break
                            break

            # Try to match patterns starting at current position
            matched = False

            for pattern in self.patterns:
                match_len = len(pattern.pattern)
                if i + match_len > len(lines):
                    continue

                # Extract instructions for potential match
                instructions = []
                instruction_lines = []  # Line indices of actual instructions
                skip_indices = []

                j = i
                instr_count = 0
                while instr_count < match_len and j < len(lines):
                    line = lines[j].strip()
                    parsed = self._parse_line(lines[j])
                    if parsed is None:
                        # Check if this is a label (not just a comment)
                        # Labels break pattern matching - code can jump to labels
                        if line and ':' in line and not line.startswith(';'):
                            # This is a label - stop pattern matching here
                            break
                        # Comment or empty line - include but don't count
                        skip_indices.append(j - i)
                        j += 1
                        continue
                    instructions.append(parsed)
                    instruction_lines.append(j)
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

                    # First, preserve any labels/comments that were skipped during matching
                    # These should appear BEFORE any replacement instructions
                    for offset in skip_indices:
                        result.append(lines[i + offset])

                    # Then apply the replacement
                    if pattern.replacement is not None:
                        for opcode, operands in pattern.replacement:
                            result.append(f"\t{opcode}\t{operands}" if operands else f"\t{opcode}")
                    elif pattern.name.startswith("cond_uncond"):
                        # Keep second instruction only
                        result.append(lines[instruction_lines[-1]])
                    elif pattern.name in ("redundant_mov", "duplicate_mov", "duplicate_ld"):
                        # Keep first instruction only
                        result.append(lines[instruction_lines[0]])
                    elif pattern.name in ("load_store_same", "shld_lhld_same"):
                        # Keep first instruction only
                        result.append(lines[instruction_lines[0]])
                    elif pattern.name in ("useless_extend_before_sub", "useless_extend_before_cp"):
                        # Keep last instruction only (SUB or CP)
                        result.append(lines[instruction_lines[-1]])
                    elif pattern.name == "tail_call":
                        # CALL x; RET -> JMP x
                        call_target = instructions[0][1]
                        result.append(f"\tJMP\t{call_target}")
                    elif pattern.name == "pop_push_lxi":
                        # POP H; PUSH H; LXI H,x -> LXI H,x
                        # POP/PUSH peeks TOS (no stack change), then HL overwritten
                        result.append(lines[instruction_lines[2]])  # Keep only LXI H,x
                    elif pattern.name == "push_sta_pop":
                        # PUSH PSW; STA addr; POP PSW -> STA addr
                        result.append(lines[instruction_lines[1]])  # Keep only STA
                    elif pattern.name == "lxi_xchg_pop":
                        # LXI H,x; XCHG; POP H -> LXI D,x; POP H
                        operand = instructions[0][1][2:]  # Remove "H," prefix
                        result.append(f"\tLXI\tD,{operand}")
                        result.append("\tPOP\tH")
                    elif pattern.name == "lxi_xchg_call":
                        # LXI H,x; XCHG; CALL y -> LXI D,x; CALL y
                        operand = instructions[0][1][2:]  # Remove "H," prefix
                        result.append(f"\tLXI\tD,{operand}")
                        result.append(lines[instruction_lines[2]])  # CALL y
                    elif pattern.name == "lxi_xchg_jmp":
                        # LXI H,x; XCHG; JMP y -> LXI D,x; JMP y
                        operand = instructions[0][1][2:]  # Remove "H," prefix
                        result.append(f"\tLXI\tD,{operand}")
                        result.append(lines[instruction_lines[2]])  # JMP y
                    elif pattern.name in ("lxi_xchg_lda", "lxi_xchg_sta", "lxi_xchg_lhld"):
                        # LXI H,x; XCHG; LDA/STA/LHLD y -> LXI D,x; LDA/STA/LHLD y
                        operand = instructions[0][1][2:]  # Remove "H," prefix
                        result.append(f"\tLXI\tD,{operand}")
                        result.append(lines[instruction_lines[2]])  # LDA/STA/LHLD y
                    elif pattern.name == "push_lxi_xchg_pop":
                        # PUSH H; LXI H,x; XCHG; POP H -> LXI D,x
                        # The PUSH is no longer needed since we're not saving/restoring HL
                        operand = instructions[1][1][2:]  # Remove "H," prefix
                        result.append(f"\tLXI\tD,{operand}")
                    elif pattern.name == "double_push_same_const":
                        # LXI H,x; PUSH H; LXI H,x; PUSH H -> LXI H,x; PUSH H; PUSH H
                        result.append(lines[instruction_lines[0]])
                        result.append("\tPUSH\tH")
                        result.append("\tPUSH\tH")
                    elif pattern.name == "push_lxi_d_pop_dad":
                        # PUSH H; LXI D,x; POP H; DAD D -> LXI D,x; DAD D
                        result.append(lines[instruction_lines[1]])  # LXI D,x
                        result.append(lines[instruction_lines[3]])  # DAD D
                    elif pattern.name == "lxi_mov_am_to_lda":
                        # LXI H,addr; MOV A,M -> LDA addr
                        addr = instructions[0][1][2:]  # Remove "H," prefix
                        result.append(f"\tLDA\t{addr}")
                    elif pattern.name == "sta_lda_same":
                        # STA x; LDA x -> STA x
                        result.append(lines[instruction_lines[0]])

                    elif pattern.name in ("lda_cpi_jz_lda_same", "lda_cpi_jnz_lda_same",
                                          "lda_cpi_jc_lda_same", "lda_cpi_jnc_lda_same",
                                          "lda_ora_jz_lda_same", "lda_ora_jnz_lda_same"):
                        # LDA x; CPI/ORA; Jcond; LDA x -> LDA x; CPI/ORA; Jcond
                        # Keep first 3 instructions, drop the redundant reload
                        result.append(lines[instruction_lines[0]])  # LDA
                        result.append(lines[instruction_lines[1]])  # CPI/ORA
                        result.append(lines[instruction_lines[2]])  # Jcond

                    elif pattern.name == "lda_adi1_sta_same":
                        # LDA x; ADI 1; STA x -> LXI H,x; INR M
                        addr = instructions[0][1]
                        result.append(f"\tLXI\tH,{addr}")
                        result.append(f"\tINR\tM")
                    elif pattern.name == "lda_sui1_sta_same":
                        # LDA x; SUI 1; STA x -> LXI H,x; DCR M
                        addr = instructions[0][1]
                        result.append(f"\tLXI\tH,{addr}")
                        result.append(f"\tDCR\tM")

                    elif pattern.name == "lxi_mov_al_sta":
                        # LXI H,const; MOV A,L; STA x -> MVI A,const; STA x
                        const = instructions[0][1][2:]  # Remove "H," prefix
                        sta_addr = instructions[2][1]
                        result.append(f"\tMVI\tA,{const}")
                        result.append(f"\tSTA\t{sta_addr}")
                    elif pattern.name == "mov_la_mvi_h0_sta":
                        # MOV L,A; MVI H,0; STA x -> STA x
                        result.append(lines[instruction_lines[2]])

                    elif pattern.name == "mov_al_mvi_h0_sta":
                        # MOV A,L; MVI H,0; STA x -> MOV A,L; STA x
                        result.append(lines[instruction_lines[0]])  # MOV A,L
                        result.append(lines[instruction_lines[2]])  # STA x

                    elif pattern.name == "push_shld_pop":
                        # PUSH H; SHLD x; POP H -> SHLD x
                        # SHLD doesn't modify HL, so save/restore is unnecessary
                        result.append(lines[instruction_lines[1]])  # Keep only SHLD

                    elif pattern.name == "push_mvi_a_mov_e_mvi_d_pop":
                        # PUSH H; MVI A,x; MOV E,A; MVI D,0; POP H -> MVI E,x; MVI D,0
                        # HL not modified, PUSH/POP is wasteful
                        const = instructions[1][1][2:]  # Get constant from "A,x"
                        result.append(f"\tMVI\tE,{const}")
                        result.append(f"\tMVI\tD,0")

                    elif pattern.name == "push_lda_sub_to_de_pop":
                        # PUSH H; LDA x; MOV B,A; LDA y; SUB B; MOV E,A; MVI D,0; POP H
                        # -> just the middle instructions without PUSH/POP
                        # HL not modified, PUSH/POP is wasteful
                        for idx in range(1, 7):  # Instructions 1-6 (skip PUSH at 0 and POP at 7)
                            result.append(lines[instruction_lines[idx]])

                    elif pattern.name == "push_lhld_pop_d":
                        # PUSH H; LHLD x; POP D -> XCHG; LHLD x
                        # Old HL goes to DE, new value from (x) into HL
                        # This is 1 byte smaller (5 -> 4 bytes)
                        addr = instructions[1][1]  # Get address from LHLD
                        result.append(f"\tXCHG")
                        result.append(f"\tLHLD\t{addr}")

                    elif pattern.name == "early_load_push_subde":
                        # LHLD x; PUSH H; LDED y; LHLD z; CALL ??SUBDE; XCHG; POP H; CALL ??SUBDE
                        # -> LDED y; LHLD z; CALL ??SUBDE; XCHG; LHLD x; CALL ??SUBDE
                        # Delay loading x until needed, eliminate PUSH/POP
                        addr_x = instructions[0][1]  # First LHLD operand
                        addr_y = instructions[2][1]  # LDED operand
                        addr_z = instructions[3][1]  # Second LHLD operand
                        result.append(f"\tLDED\t{addr_y}")
                        result.append(f"\tLHLD\t{addr_z}")
                        result.append(f"\tCALL\t??SUBDE")
                        result.append(f"\tXCHG")
                        result.append(f"\tLHLD\t{addr_x}")
                        result.append(f"\tCALL\t??SUBDE")

                    elif pattern.name in ("mov_ba_sta_mov_ab", "mov_ca_sta_mov_ac",
                                          "mov_da_sta_mov_ad", "mov_ea_sta_mov_ae"):
                        # MOV x,A; STA y; MOV A,x -> STA y
                        # A is unchanged by STA, so the save/restore is unnecessary
                        result.append(lines[instruction_lines[1]])

                    elif pattern.name == "mov_ba_lhld_mov_ab_mov_ma":
                        # MOV B,A; LHLD x; MOV A,B; MOV M,A -> MOV B,A; LHLD x; MOV M,B
                        result.append(lines[instruction_lines[0]])  # MOV B,A
                        result.append(lines[instruction_lines[1]])  # LHLD x
                        result.append(f"\tMOV\tM,B")

                    elif pattern.name == "mov_ca_lhld_mov_ac_mov_ma":
                        # MOV C,A; LHLD x; MOV A,C; MOV M,A -> MOV C,A; LHLD x; MOV M,C
                        result.append(lines[instruction_lines[0]])  # MOV C,A
                        result.append(lines[instruction_lines[1]])  # LHLD x
                        result.append(f"\tMOV\tM,C")

                    elif pattern.name in ("sbb_mov_ora_jm", "sbb_mov_ora_jp"):
                        # SBB D; MOV H,A; ORA A; JM/JP x -> SBB D; MOV H,A; JM/JP x
                        # ORA A is redundant - sign flag already set by SBB D
                        result.append(lines[instruction_lines[0]])  # SBB D
                        result.append(lines[instruction_lines[1]])  # MOV H,A
                        result.append(lines[instruction_lines[3]])  # JM/JP x (skip ORA A)

                    elif pattern.name in ("sub_16bit_sign_jm", "sub_16bit_sign_jp"):
                        # SUB E; MOV L,A; MOV A,H; SBB D; MOV H,A; JM/JP x
                        # -> SUB E; MOV A,H; SBB D; JM/JP x
                        # When just checking sign, skip MOV L,A and MOV H,A
                        result.append(lines[instruction_lines[0]])  # SUB E
                        result.append(lines[instruction_lines[2]])  # MOV A,H
                        result.append(lines[instruction_lines[3]])  # SBB D
                        result.append(lines[instruction_lines[5]])  # JM/JP x

                    elif pattern.name == "shld_call_lhld_same":
                        # SHLD x; CALL y; LHLD x -> PUSH H; CALL y; POP H
                        call_target = instructions[1][1]
                        result.append("\tPUSH\tH")
                        result.append(f"\tCALL\t{call_target}")
                        result.append("\tPOP\tH")

                    elif pattern.name == "shld_call_lhld_jmp_same":
                        # SHLD x; CALL y; LHLD x; JMP z -> PUSH H; CALL y; POP H; JMP z
                        call_target = instructions[1][1]
                        jmp_target = instructions[3][1]
                        result.append("\tPUSH\tH")
                        result.append(f"\tCALL\t{call_target}")
                        result.append("\tPOP\tH")
                        result.append(f"\tJMP\t{jmp_target}")

                    elif pattern.name == "sta_call_lda_same":
                        # STA x; CALL y; LDA x -> PUSH PSW; CALL y; POP PSW
                        call_target = instructions[1][1]
                        result.append("\tPUSH\tPSW")
                        result.append(f"\tCALL\t{call_target}")
                        result.append("\tPOP\tPSW")

                    elif pattern.name == "shld_mvi_lhld_same":
                        # SHLD x; MVI r,n; LHLD x -> MVI r,n
                        # MVI doesn't touch HL, so save/restore is unnecessary
                        result.append(lines[instruction_lines[1]])  # MVI r,n

                    elif pattern.name == "sta_mvi_lda_same":
                        # STA x; MVI r,n; LDA x -> MVI r,n
                        # MVI doesn't touch A (when r != A), so save/restore is unnecessary
                        result.append(lines[instruction_lines[1]])  # MVI r,n

                    elif pattern.name == "push_lxi_mov_cl_pop":
                        # PUSH H; LXI H,const; MOV C,L; POP H -> MVI C,const
                        const = instructions[1][1][2:]  # Remove "H," prefix
                        result.append(f"\tMVI\tC,{const}")

                    elif pattern.name == "push_mov_sta_pop_mvi_h0":
                        # PUSH H; MOV A,L; STA x; POP H; MVI H,0 -> MOV A,L; STA x; MVI H,0
                        sta_addr = instructions[2][1]
                        result.append("\tMOV\tA,L")
                        result.append(f"\tSTA\t{sta_addr}")
                        result.append("\tMVI\tH,0")

                    elif pattern.name == "shld_mvi_lhld_xchg_same":
                        # SHLD x; MVI r,n; LHLD x; XCHG -> MVI r,n; XCHG
                        result.append(lines[instruction_lines[1]])  # MVI r,n
                        result.append("\tXCHG")

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
