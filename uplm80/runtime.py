"""
Runtime library for PL/M-80.

Contains assembly code for runtime support routines that are too complex
to generate inline (multiply, divide, etc.).
"""


def plm_div(dividend: int, divisor: int) -> int:
    """PL/M-80's ``dividend / divisor`` on 16-bit unsigned operands.

    Digital Research's PL/M-80 divides with PLM80.LIB's @P0029, sixteen
    shift-and-subtract steps with no zero test: a zero divisor never makes a
    trial subtraction fail, so every quotient bit is set. The compile-time
    folder and ``??div16`` both have to give this.
    """
    dividend &= 0xFFFF
    divisor &= 0xFFFF
    return dividend // divisor if divisor else 0xFFFF


def plm_mod(dividend: int, divisor: int) -> int:
    """PL/M-80's ``dividend MOD divisor``: the dividend itself when divisor is 0."""
    dividend &= 0xFFFF
    divisor &= 0xFFFF
    return dividend % divisor if divisor else dividend


# 16-bit unsigned multiply: HL = HL * DE, the low sixteen bits of the product
# (DRI's PLM80.LIB @P0034 computes the same, most significant bit first).
# The shifts shift in zeros: the rotates through carry this used to do took in
# the carry of the add before them, and of the multiplicand's top bit, so a
# product that overflowed came out wrong -- 81H * 511 gave 817FH, not 017FH.
# DE is 0 on return.
RUNTIME_MUL16 = """\
??mul16:
	; 16-bit multiply: HL = HL * DE
	; Input: HL = multiplicand, DE = multiplier
	; Output: HL = product (low 16 bits), DE = 0
	; Destroys: A, B, C, D, E
	ld	b,h
	ld	c,l		; BC = multiplicand
	ld	hl,0		; HL = result = 0
??mul16l:
	ld	a,e
	or	d		; DE == 0?
	ret	z		; Yes, done
	srl	d
	rr	e		; multiplier >>= 1, its low bit into carry
	jp	nc,??mul16s	; If bit 0 clear, skip add
	add	hl,bc		; HL = HL + BC
??mul16s:
	sla	c
	rl	b		; multiplicand <<= 1
	jp	??mul16l
"""

# 16-bit unsigned divide and remainder, as PL/M-80 defines them.
#
# DRI's PL/M-80 routes every `/' and `MOD', BYTE operands zero-extended, through
# one routine, PLM80.LIB's @P0029: sixteen shift-and-subtract steps and no test
# for a zero divisor. With a zero divisor every trial subtraction fits, so the
# quotient comes out 0FFFFH and the remainder is the dividend; programs rely on
# that (MP/M II's SDIR tests `cur$line mod page$len = 0' with page$len 0).
# These routines do the same steps and so give the same results for every
# operand pair. ??mod16 is the loop itself, which leaves the remainder in HL;
# ??div16 moves the quotient into HL.
RUNTIME_MOD16 = """\
??mod16:
	; 16-bit divide: HL = HL MOD DE, quotient in A (high) and C (low)
	; Input: HL = dividend, DE = divisor
	; Output: HL = remainder, AC = quotient, B = 0
	; Preserves: DE
	; A zero divisor gives quotient 0FFFFH, remainder = dividend (as PL/M-80)
	ld	a,h
	ld	c,l		; AC = dividend; the quotient shifts in behind it
	ld	hl,0		; HL = partial remainder
	ld	b,16
??mod16l:
	sla	c
	rla			; next dividend bit into carry
	adc	hl,hl		; remainder = 2*remainder + bit; cannot carry out
	inc	c		; quotient bit = 1 (INC leaves carry clear)
	sbc	hl,de		; does the divisor fit?
	jr	nc,??mod16n	; yes: keep the difference and the bit
	add	hl,de		; no: restore the remainder
	dec	c		; and clear the bit
??mod16n:
	djnz	??mod16l
	ret
"""

# 16-bit unsigned divide: HL = HL / DE, remainder in BC
RUNTIME_DIV16 = """\
??div16:
	; 16-bit divide: HL = HL / DE, remainder in BC
	; Input: HL = dividend, DE = divisor
	; Output: HL = quotient, BC = remainder
	; Preserves: DE
	; A zero divisor gives quotient 0FFFFH, remainder = dividend (as PL/M-80)
	call	??mod16		; HL = remainder, AC = quotient
	ld	b,a
	push	bc
	ld	b,h
	ld	c,l		; BC = remainder
	pop	hl		; HL = quotient
	ret
"""

# 8-bit unsigned multiply: A = A * E
RUNTIME_MUL8 = """\
??mul8:
	; 8-bit multiply: A = A * E (result in HL low byte)
	; Input: A = multiplicand, E = multiplier
	; Output: HL = product (16-bit)
	ld	d,a
	ld	a,0
	ld	hl,0
	ld	b,8
??mul8l:
	ld	a,e
	rra
	ld	e,a
	jp	nc,??mul8s
	ld	a,l
	add	a,d
	ld	l,a
	ld	a,h
	adc	a,0
	ld	h,a
??mul8s:
	ld	a,d
	rla
	ld	d,a
	dec	b
	jp	nz,??mul8l
	ret
"""

# Block move: MOVE(count, source, dest)
RUNTIME_MOVE = """\
??move:
	; Block move: Move count bytes from source to dest
	; Stack: ret, dest, source, count
	; Destroys: A, B, C, D, E, H, L
	pop	hl		; Return address
	pop	de		; Destination
	pop	bc		; Source -> BC temporarily
	ex	(sp),hl		; HL = count, ret addr on stack
	ld	a,h
	or	l
	jp	z,??movex	; Count = 0, done
	push	de		; Save dest
	ld	d,b
	ld	e,c		; DE = source
	pop	bc		; BC = dest
??movel:
	ld	a,(de)		; A = (source)
	ld	(bc),a		; (dest) = A
	inc	de		; source++
	inc	bc		; dest++
	dec	hl		; count--
	ld	a,h
	or	l
	jp	nz,??movel
??movex:
	ret
"""

# 16-bit subtract: HL = HL - DE (Z80 version)
# Uses the Z80-specific SBC HL,DE instruction
RUNTIME_SUBDE = """\
??subde:
	; 16-bit subtract: HL = HL - DE (Z80)
	; Input: HL, DE
	; Output: HL = HL - DE, flags set
	or	a		; Clear carry
	sbc	hl,de
	ret
"""

# Call through an address: the caller's CALL pushes the return address,
# and this goes on to DE.  (The Z80 has no CALL (HL).)
RUNTIME_JPDE = """\
??jpde:
	; Jump to the address in DE, from a CALL
	push	de
	ret
"""

# INPUT and OUTPUT of a port that is not a constant: the Z80's IN and OUT
# take a variable port only in C.
RUNTIME_INP = """\
??inp:
	; INPUT(port): A = port, returns the byte read in A
	ld	c,a
	in	a,(c)
	ret
"""

RUNTIME_OUTP = """\
??outp:
	; OUTPUT(port) = value: C = port, A = value
	out	(c),a
	ret
"""

# Compare strings for equality
RUNTIME_STRCMP = """\
??strcmp:
	; Compare two strings
	; DE = string1, HL = string2, BC = length
	; Returns Z flag set if equal
??strcml:
	ld	a,b
	or	c
	ret	z		; Length = 0, strings equal
	ld	a,(de)		; A = (string1)
	cp	(hl)		; Compare with (string2)
	ret	nz		; Not equal
	inc	de
	inc	hl
	dec	bc
	jp	??strcml
"""

def get_runtime_library(needed: set[str] | None = None) -> str:
    """Get the runtime library assembly code.

    Args:
        needed: Set of routine names that are needed (e.g., {"mul16", "subde"}).
                If None, includes all routines.
    """
    routines = {
        "mul16": RUNTIME_MUL16,
        "div16": RUNTIME_DIV16,
        "mod16": RUNTIME_MOD16,
        "mul8": RUNTIME_MUL8,
        "move": RUNTIME_MOVE,
        "subde": RUNTIME_SUBDE,
        "jpde": RUNTIME_JPDE,
        "inp": RUNTIME_INP,
        "outp": RUNTIME_OUTP,
    }

    # Dependencies: some routines call others
    dependencies = {
        "div16": {"mod16"},  # div16 calls mod16
    }

    parts = ["; PL/M-80 Runtime Library", ""]

    if needed is None:
        # Include all
        for code in routines.values():
            parts.append(code)
    else:
        # Expand dependencies
        expanded = set(needed)
        for name in list(needed):
            if name in dependencies:
                expanded.update(dependencies[name])

        # Include only what's needed (in consistent order)
        for name, code in routines.items():
            if name in expanded:
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
