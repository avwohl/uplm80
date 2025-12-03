"""
Runtime library for PL/M-80.

Contains assembly code for runtime support routines that are too complex
to generate inline (multiply, divide, etc.).
"""

# 16-bit unsigned multiply: HL = HL * DE
# Uses BC as temp
RUNTIME_MUL16 = """\
??MUL16:
	; 16-bit multiply: HL = HL * DE
	; Input: HL = multiplicand, DE = multiplier
	; Output: HL = product (low 16 bits)
	; Destroys: A, B, C, D, E
	MOV	B,H
	MOV	C,L		; BC = multiplicand
	LXI	H,0		; HL = result = 0
??MUL16L:
	MOV	A,E
	ORA	D		; DE == 0?
	RZ			; Yes, done
	MOV	A,E
	RAR			; LSB of multiplier into carry
	JNC	??MUL16S	; If bit 0 clear, skip add
	DAD	B		; HL = HL + BC
??MUL16S:
	; Shift multiplicand left
	MOV	A,C
	RAL
	MOV	C,A
	MOV	A,B
	RAL
	MOV	B,A
	; Shift multiplier right
	MOV	A,D
	RAR
	MOV	D,A
	MOV	A,E
	RAR
	MOV	E,A
	JMP	??MUL16L
"""

# 16-bit unsigned divide: HL = HL / DE, DE = HL % DE
RUNTIME_DIV16 = """\
??DIV16:
	; 16-bit divide: HL = HL / DE, remainder in BC
	; Input: HL = dividend, DE = divisor
	; Output: HL = quotient, BC = remainder
	; Destroys: A
	MOV	A,D
	ORA	E
	JZ	??DIV16Z	; Divide by zero
	PUSH	D		; Save divisor
	LXI	B,0		; BC = remainder = 0
	MVI	A,16		; 16 bits to process
??DIV16L:
	PUSH	PSW		; Save counter
	; Shift HL left, MSB into remainder
	DAD	H		; HL = HL * 2, carry = old H bit 7
	MOV	A,C
	RAL
	MOV	C,A		; C = C<<1 + carry (from HL)
	MOV	A,B
	RAL
	MOV	B,A		; B = B<<1 + carry (from C), BC shifted left with HL carry in
	; Shift carry into bit 0 of dividend (will be quotient)
	; Actually we need to track if remainder >= divisor
	; Compare BC with DE
	MOV	A,C
	SUB	E
	MOV	A,B
	SBB	D
	JC	??DIV16N	; BC < DE, don't subtract
	; BC >= DE, subtract and set quotient bit
	MOV	A,C
	SUB	E
	MOV	C,A
	MOV	A,B
	SBB	D
	MOV	B,A
	INX	H		; Set quotient bit
??DIV16N:
	POP	PSW		; Restore counter
	DCR	A
	JNZ	??DIV16L
	POP	D		; Restore divisor (not needed, but balance stack)
	RET
??DIV16Z:
	; Divide by zero - return FFFF
	LXI	H,0FFFFH
	LXI	B,0
	RET
"""

# 16-bit modulo: HL = HL MOD DE
RUNTIME_MOD16 = """\
??MOD16:
	; 16-bit modulo: HL = HL MOD DE
	; Input: HL = dividend, DE = divisor
	; Output: HL = remainder
	; Destroys: A, B, C
	CALL	??DIV16
	MOV	H,B
	MOV	L,C		; Move remainder to HL
	RET
"""

# 8-bit unsigned multiply: A = A * E
RUNTIME_MUL8 = """\
??MUL8:
	; 8-bit multiply: A = A * E (result in HL low byte)
	; Input: A = multiplicand, E = multiplier
	; Output: HL = product (16-bit)
	MOV	D,A
	MVI	A,0
	LXI	H,0
	MVI	B,8
??MUL8L:
	MOV	A,E
	RAR
	MOV	E,A
	JNC	??MUL8S
	MOV	A,L
	ADD	D
	MOV	L,A
	MOV	A,H
	ACI	0
	MOV	H,A
??MUL8S:
	MOV	A,D
	RAL
	MOV	D,A
	DCR	B
	JNZ	??MUL8L
	RET
"""

# Block move: MOVE(count, source, dest)
RUNTIME_MOVE = """\
??MOVE:
	; Block move: Move count bytes from source to dest
	; Stack: ret, dest, source, count
	; Destroys: A, B, C, D, E, H, L
	POP	H		; Return address
	POP	D		; Destination
	POP	B		; Source -> BC temporarily
	XTHL			; HL = count, ret addr on stack
	MOV	A,H
	ORA	L
	JZ	??MOVEX		; Count = 0, done
	PUSH	D		; Save dest
	MOV	D,B
	MOV	E,C		; DE = source
	POP	B		; BC = dest
??MOVEL:
	LDAX	D		; A = (source)
	STAX	B		; (dest) = A
	INX	D		; source++
	INX	B		; dest++
	DCX	H		; count--
	MOV	A,H
	ORA	L
	JNZ	??MOVEL
??MOVEX:
	RET
"""

# 16-bit subtract: HL = HL - DE (8080 version)
# Compact routine to save bytes when used frequently
RUNTIME_SUBDE = """\
??SUBDE:
	; 16-bit subtract: HL = HL - DE
	; Input: HL, DE
	; Output: HL = HL - DE, flags set
	; Destroys: A
	MOV	A,L
	SUB	E
	MOV	L,A
	MOV	A,H
	SBB	D
	MOV	H,A
	RET
"""

# 16-bit subtract: HL = HL - DE (Z80 version - 4 bytes shorter!)
RUNTIME_SUBDE_Z80 = """\
??SUBDE:
	; 16-bit subtract: HL = HL - DE (Z80)
	; Input: HL, DE
	; Output: HL = HL - DE, flags set
	OR	A		; Clear carry
	SBC	HL,DE
	RET
"""

# Compare strings for equality
RUNTIME_STRCMP = """\
??STRCMP:
	; Compare two strings
	; DE = string1, HL = string2, BC = length
	; Returns Z flag set if equal
??STRCML:
	MOV	A,B
	ORA	C
	RZ			; Length = 0, strings equal
	LDAX	D		; A = (string1)
	CMP	M		; Compare with (string2)
	RNZ			; Not equal
	INX	D
	INX	H
	DCX	B
	JMP	??STRCML
"""

def get_runtime_library(needed: set[str] | None = None, target_z80: bool = True) -> str:
    """Get the runtime library assembly code.

    Args:
        needed: Set of routine names that are needed (e.g., {"MUL16", "SUBDE"}).
                If None, includes all routines.
        target_z80: If True, use Z80-specific optimized routines. If False, use 8080.
    """
    # Map routine names to their code
    # Use Z80-optimized SUBDE if targeting Z80
    subde_routine = RUNTIME_SUBDE_Z80 if target_z80 else RUNTIME_SUBDE

    routines = {
        "MUL16": RUNTIME_MUL16,
        "DIV16": RUNTIME_DIV16,
        "MOD16": RUNTIME_MOD16,
        "MUL8": RUNTIME_MUL8,
        "MOVE": RUNTIME_MOVE,
        "SUBDE": subde_routine,
    }

    parts = ["; PL/M-80 Runtime Library", ""]

    if needed is None:
        # Include all
        for code in routines.values():
            parts.append(code)
    else:
        # Include only what's needed
        for name, code in routines.items():
            if name in needed:
                parts.append(code)

    return "\n".join(parts)


# Built-in function signatures for reference
BUILTIN_FUNCTIONS = {
    # (name, return_type, param_types, inline_capable)
    "INPUT": ("BYTE", ["BYTE"], True),
    "OUTPUT": ("BYTE", ["BYTE"], True),  # OUTPUT is special - used as lvalue
    "LOW": ("BYTE", ["ADDRESS"], True),
    "HIGH": ("BYTE", ["ADDRESS"], True),
    "DOUBLE": ("ADDRESS", ["BYTE"], True),
    "LENGTH": ("ADDRESS", ["ARRAY"], True),
    "LAST": ("ADDRESS", ["ARRAY"], True),
    "SIZE": ("ADDRESS", ["ARRAY"], True),
    "SHL": ("ADDRESS", ["ADDRESS", "BYTE"], True),
    "SHR": ("ADDRESS", ["ADDRESS", "BYTE"], True),
    "ROL": ("BYTE", ["BYTE", "BYTE"], True),
    "ROR": ("BYTE", ["BYTE", "BYTE"], True),
    "SCL": ("BYTE", ["BYTE", "BYTE"], True),
    "SCR": ("BYTE", ["BYTE", "BYTE"], True),
    "MOVE": (None, ["ADDRESS", "ADDRESS", "ADDRESS"], False),
    "TIME": (None, ["ADDRESS"], True),
    "CARRY": ("BYTE", [], True),
    "SIGN": ("BYTE", [], True),
    "ZERO": ("BYTE", [], True),
    "PARITY": ("BYTE", [], True),
    "DEC": ("BYTE", ["BYTE"], True),
    "STACKPTR": ("ADDRESS", [], True),  # Actually a variable, not function
}
