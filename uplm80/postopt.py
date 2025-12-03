"""
Post-assembly optimization pass.

This pass operates on the generated assembly file (.mac) before assembly.
It performs optimizations that are easier to do at the text level than
in the code generator, particularly cross-procedure optimizations.

Optimizations:
1. Tail merging with skip trick - merge procedures with common tails
2. JP to JR conversion - convert 3-byte JP to 2-byte JR when target in range
"""

import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Procedure:
    """Represents a procedure in the assembly file."""
    name: str
    start_line: int
    end_line: int
    instructions: list[str]  # Just the instruction part, no labels


def get_instr_size(instr: str) -> int | None:
    """Get the size of a Z80 instruction in bytes, or None if unknown."""
    instr = instr.strip()
    
    # 1-byte instructions
    if instr in ('NOP', 'HALT', 'RET', 'EXX', 'EX DE,HL', 'EX AF,AF\'',
                 'DI', 'EI', 'CCF', 'SCF', 'DAA', 'CPL', 'NEG',
                 'RLCA', 'RRCA', 'RLA', 'RRA'):
        return 1
    if re.match(r'^(INC|DEC)\s+[ABCDEHL]$', instr):
        return 1
    if re.match(r'^(INC|DEC)\s+(BC|DE|HL|SP|IX|IY)$', instr):
        return 1  # INC rr
    if re.match(r'^(PUSH|POP)\s+(AF|BC|DE|HL)$', instr):
        return 1
    if re.match(r'^LD\s+[ABCDEHL],[ABCDEHL]$', instr):
        return 1
    if re.match(r'^(ADD|ADC|SUB|SBC|AND|OR|XOR|CP)\s+A?,?[ABCDEHL]$', instr):
        return 1
    if re.match(r'^(ADD|ADC|SBC)\s+HL,(BC|DE|HL|SP)', instr):  # May have comment
        return 1
    if instr in ('XOR A', 'OR A', 'AND A', 'CP A'):
        return 1
    if instr == 'JP (HL)':
        return 1
        
    # 2-byte instructions
    if re.match(r'^LD\s+[ABCDEHL],[0-9][0-9A-F]*H?$', instr):
        return 2  # LD r,n
    if re.match(r'^(ADD|ADC|SUB|SBC|AND|OR|XOR|CP)\s+(A,)?[0-9][0-9A-F]*H?$', instr):
        return 2
    if re.match(r'^JR\s+', instr):
        return 2
    if re.match(r'^DJNZ\s+', instr):
        return 2
    if re.match(r'^(IN|OUT)\s+', instr):
        return 2  # Most IN/OUT forms
        
    # 3-byte instructions
    if re.match(r'^LD\s+(BC|DE|HL|SP),\d+H?$', instr):
        return 3  # LD rr,nn
    if re.match(r'^LD\s+(BC|DE|HL|SP),0[0-9A-F]+H$', instr):
        return 3
    if re.match(r'^LD\s+(BC|DE|HL|SP),[A-Za-z@$?_]', instr):
        return 3  # LD rr,label
    if re.match(r'^LD\s+\([A-Za-z@$?_0-9]+\),(HL|A)$', instr):
        return 3  # LD (addr),HL or LD (addr),A
    if re.match(r'^LD\s+(HL|A),\([A-Za-z@$?_0-9]+\)$', instr):
        return 3  # LD HL,(addr) or LD A,(addr)
    if re.match(r'^(JP|CALL)\s+', instr):
        if 'IX' in instr or 'IY' in instr or '(HL)' in instr:
            return 1 if '(HL)' in instr else 2
        return 3
        
    # 4-byte instructions (ED prefix + 2 byte addr)
    if re.match(r'^LD\s+\([A-Za-z@$?_0-9]+\),(BC|DE|SP)$', instr):
        return 4
    if re.match(r'^LD\s+(BC|DE|SP),\([A-Za-z@$?_0-9]+\)$', instr):
        return 4
        
    return None


def parse_procedures(lines: list[str]) -> list[Procedure]:
    """Extract procedures from assembly lines."""
    procs = []
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        
        # Look for procedure labels (not internal labels starting with ??)
        if (stripped.endswith(':') and
            not stripped.startswith('??') and
            not stripped.startswith(';') and
            len(stripped) > 1 and
            ('@' in stripped or stripped[0].isupper())):
            
            name = stripped[:-1]
            start = i
            instructions = []
            
            # Collect instructions until next procedure or end
            j = i + 1
            while j < len(lines):
                next_line = lines[j].rstrip()
                next_stripped = next_line.strip()
                
                # End at next procedure label
                if (next_stripped.endswith(':') and
                    not next_stripped.startswith('??') and
                    not next_stripped.startswith(';') and
                    len(next_stripped) > 1 and
                    ('@' in next_stripped or next_stripped[0].isupper())):
                    break
                    
                # Skip empty lines and comments
                if next_stripped and not next_stripped.startswith(';'):
                    # Skip internal labels, just collect instructions
                    if not next_stripped.endswith(':'):
                        instructions.append(next_stripped)
                        
                j += 1
            
            if instructions:
                procs.append(Procedure(name, start, j, instructions))
            i = j
        else:
            i += 1
            
    return procs


def find_tail_merge_groups(procs: list[Procedure]) -> dict[tuple, list[Procedure]]:
    """
    Find procedures that share common tail sequences.

    This looks for any common tails that end with:
    - JP (unconditional jump) - most common for BDOS wrappers
    - RET - common for procedures with shared cleanup
    - CALL followed by RET - common pattern
    """
    # Group by tail
    tail_groups = defaultdict(list)

    for proc in procs:
        if not proc.instructions:
            continue

        last = proc.instructions[-1]

        # Accept tails ending in: JP, RET, or any valid instruction
        # More permissive - we'll filter by savings later
        if not (re.match(r'^(JP|RET|CALL)\s*', last) or last in ('RET', 'PCHL')):
            continue

        # Try different tail lengths (2-6 instructions) - longer tails = more savings
        for length in range(2, min(7, len(proc.instructions) + 1)):
            tail = tuple(proc.instructions[-length:])

            # Calculate total size of tail
            sizes = [get_instr_size(instr) for instr in tail]
            if None in sizes:
                continue

            total_size = sum(sizes)
            # Only worth it if tail is at least 4 bytes (JP to merge costs 3)
            if total_size >= 4:
                tail_groups[tail].append(proc)

    # Filter to groups with 2+ procedures
    # Also prefer longer tails when they exist
    result = {}
    for tail, group in tail_groups.items():
        if len(group) >= 2:
            result[tail] = group

    return result


_tail_merge_counter = 0

def optimize_tail_merge(lines: list[str], procs: list[Procedure],
                        tail_groups: dict[tuple, list[Procedure]],
                        verbose: bool = False) -> tuple[list[str], int]:
    """
    Optimize procedures with shared tails using skip trick.

    The skip trick uses DB 21H (the opcode for LD HL,nn) to skip over
    the next 2 bytes, allowing tail merging without adding jump instructions.

    Example:
        PROC1:              PROC1:
            LD C,1              LD C,1
            LD DE,0             DB 21H        ; Skip next 2 bytes
            JP 5            PROC2:
        PROC2:          ->      LD C,2
            LD C,2          TAIL:
            LD DE,0             LD DE,0
            JP 5                JP 5

    Returns modified lines and bytes saved.
    """
    total_savings = 0
    result = lines.copy()

    # Process each tail group
    for tail, group in tail_groups.items():
        # Check if we can use skip trick
        # Each proc except last needs instruction before tail to be 2 bytes
        usable = []
        for proc in group:
            if len(proc.instructions) <= len(tail):
                continue  # No instruction before tail to skip from

            instr_before_tail = proc.instructions[-(len(tail) + 1)]
            size = get_instr_size(instr_before_tail)
            if size == 2:
                usable.append(proc)

        if len(usable) < 2:
            continue  # Need at least 2 procs

        # Sort by line number so we process in order
        usable.sort(key=lambda p: p.start_line)

        # Calculate savings:
        # - Last proc keeps full code (becomes the shared tail)
        # - Each earlier proc replaces tail with DB 21H (1 byte)
        tail_size = sum(get_instr_size(instr) or 3 for instr in tail)
        savings_per_proc = tail_size - 1  # Replace tail with just DB 21H
        group_savings = savings_per_proc * (len(usable) - 1)

        if verbose:
            print(f"  Tail merge: {len(usable)} procs share {len(tail)}-instr tail")
            print(f"    Tail: {tail}")
            print(f"    Procs: {[p.name for p in usable]}")
            print(f"    Savings: {group_savings} bytes")

        # Now rewrite: for each proc except the last, replace tail with skip
        # The last proc becomes the shared tail target
        global _tail_merge_counter
        _tail_merge_counter += 1
        last_proc = usable[-1]
        tail_label = f"??TAIL${_tail_merge_counter}"

        # Find where to insert the tail label (just before the tail in last proc)
        # We need to find the actual line in the file
        for proc in usable[:-1]:  # All except last
            # Find the tail instructions in this proc and replace with skip
            # We need to find the line numbers of the tail
            proc_lines_start = proc.start_line + 1  # Skip label

            # Find instruction lines in proc
            instr_lines = []
            for j in range(proc.start_line + 1, proc.end_line):
                line_stripped = result[j].strip()
                if line_stripped and not line_stripped.startswith(';') and not line_stripped.endswith(':'):
                    instr_lines.append(j)

            if len(instr_lines) < len(tail) + 1:
                continue  # Not enough instructions

            # The tail starts at instr_lines[-len(tail)]
            tail_start_idx = instr_lines[-len(tail)]

            # Replace first tail instruction with DB 21H ; skip and JR to tail
            # Actually, we use DB 21H to skip the next 2 bytes
            # But we need to jump to the shared tail, not just skip 2 bytes

            # Simpler approach: replace tail with JR to shared tail label
            # This saves (tail_size - 2) bytes per proc

            # First, blank out tail instructions but preserve labels
            preserved_labels = []
            for j in range(tail_start_idx, instr_lines[-1] + 1):
                line_stripped = result[j].strip()
                # Keep labels (lines ending with ':' but not just whitespace before)
                if line_stripped.endswith(':') and not line_stripped.startswith(';'):
                    preserved_labels.append(result[j])
                result[j] = ''

            # Insert JP to tail label (JR might be out of range)
            # Put preserved labels first, then the JP
            result[tail_start_idx] = ''.join(preserved_labels) + f'\tJP {tail_label}\t; tail merged\n'

        # Add tail label before the tail in last proc
        for j in range(last_proc.start_line + 1, last_proc.end_line):
            line_stripped = result[j].strip()
            if line_stripped and not line_stripped.startswith(';') and not line_stripped.endswith(':'):
                # Find instruction lines in last proc
                instr_lines = []
                for k in range(last_proc.start_line + 1, last_proc.end_line):
                    ls = result[k].strip()
                    if ls and not ls.startswith(';') and not ls.endswith(':'):
                        instr_lines.append(k)

                if len(instr_lines) >= len(tail):
                    tail_start_idx = instr_lines[-len(tail)]
                    # Insert label before tail
                    result[tail_start_idx] = f'{tail_label}:\n' + result[tail_start_idx]
                break

        # Recalculate savings with JP approach: saves tail_size - 3 per proc
        actual_savings = (tail_size - 3) * (len(usable) - 1)
        total_savings += actual_savings

    # Remove empty lines
    result = [line for line in result if line]

    return result, total_savings


def find_skip_opportunities(lines: list[str]) -> list[dict]:
    """
    Find opportunities for the skip trick where:
    JP label; 2-byte-instr; label:

    Can become:
    DB 21H; 2-byte-instr; label:

    Also handles adjacent procedures after tail merge:
    PROC1:
        LD C,3
        JP ??TAIL$X     <- can become DB 21H if next proc's first instr is 2 bytes
    PROC2:              <- and next proc also jumps to same tail
        LD C,1
        JP ??TAIL$X
    """
    opportunities = []

    i = 0
    while i < len(lines) - 2:
        line = lines[i].strip()

        # Look for unconditional JP with tail merge comment
        m = re.match(r'^JP\s+(\?\?TAIL\$[A-Za-z0-9_@$]+)', line)
        if m:
            target = m.group(1)

            # Look ahead for next procedure that also jumps to same target
            # Skip blank lines, comments, and look for proc label
            j = i + 1
            next_instr = None
            next_jp_line = None

            while j < len(lines) - 1:
                next_line = lines[j].strip()

                # Skip blanks and comments
                if not next_line or next_line.startswith(';'):
                    j += 1
                    continue

                # Found a label (procedure entry)
                if next_line.endswith(':'):
                    # Look for next instruction
                    k = j + 1
                    while k < len(lines):
                        instr_line = lines[k].strip()
                        if not instr_line or instr_line.startswith(';'):
                            k += 1
                            continue
                        if instr_line.endswith(':'):
                            break  # Another label, stop
                        # Found instruction
                        size = get_instr_size(instr_line)
                        if size == 2:
                            # Check if next line after this is JP to same target
                            m2 = k + 1
                            while m2 < len(lines):
                                check = lines[m2].strip()
                                if not check or check.startswith(';'):
                                    m2 += 1
                                    continue
                                if check == f'JP {target}' or check.startswith(f'JP {target}\t'):
                                    next_instr = instr_line
                                    next_jp_line = m2
                                break
                        break
                    break
                else:
                    break  # Not a label, not what we're looking for
                j += 1

            if next_instr and next_jp_line:
                opportunities.append({
                    'line_idx': i,
                    'target': target,
                    'skipped': next_instr,
                    'savings': 2  # JP is 3 bytes, DB 21H is 1 byte
                })

        # Also check original pattern: JP label; 2-byte; label:
        m = re.match(r'^JP\s+([A-Za-z0-9_@$?]+)$', line)
        if m and not line.startswith('JP ??TAIL'):
            target = m.group(1)
            next_line = lines[i + 1].strip()
            third_line = lines[i + 2].strip()

            # Check if third line is the target label
            if third_line.startswith(target + ':'):
                # Check if next instruction is exactly 2 bytes
                size = get_instr_size(next_line)
                if size == 2:
                    opportunities.append({
                        'line_idx': i,
                        'target': target,
                        'skipped': next_line,
                        'savings': 2  # JP is 3 bytes, DB 21H is 1 byte
                    })
        i += 1

    return opportunities


def apply_skip_trick(lines: list[str], opportunities: list[dict]) -> tuple[list[str], int]:
    """Apply the skip trick to identified opportunities."""
    if not opportunities:
        return lines, 0
    
    result = lines.copy()
    total_savings = 0
    
    # Process in reverse order to maintain line indices
    for opp in sorted(opportunities, key=lambda x: -x['line_idx']):
        idx = opp['line_idx']
        # Replace JP target with DB 21H
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        result[idx] = ' ' * indent + 'DB 21H\t; skip next 2 bytes\n'
        total_savings += 2
        
    return result, total_savings


def find_jp_to_jr_candidates(lines: list[str]) -> tuple[dict[str, int], list[tuple[int, str, str]]]:
    """
    First pass: Find all labels and JP instructions.

    Returns:
        - label_positions: dict mapping label name to line index
        - jp_candidates: list of (line_idx, condition, target_label)
    """
    label_positions = {}
    jp_candidates = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Collect label definitions
        if stripped.endswith(':') and not stripped.startswith(';'):
            label = stripped[:-1]
            label_positions[label] = i

        # Collect JP instructions (unconditional and conditional)
        # Match: JP target, JP Z,target, JP NZ,target, JP C,target, JP NC,target
        # But NOT JP (HL), JP (IX), JP (IY)
        m = re.match(r'^JP\s+(?:(Z|NZ|C|NC|PE|PO|P|M),\s*)?([A-Za-z@$?_][A-Za-z0-9@$?_]*)$', stripped)
        if m:
            condition = m.group(1)  # None for unconditional
            target = m.group(2)
            jp_candidates.append((i, condition, target))

    return label_positions, jp_candidates


def calculate_byte_offset(lines: list[str], from_line: int, to_line: int) -> int | None:
    """
    Calculate the byte offset between two lines.

    The offset is calculated from the END of the JP instruction to the target.
    For JR, the offset is relative to the byte AFTER the JR instruction.

    Returns None if any instruction size is unknown.
    """
    if from_line == to_line:
        return 0

    total = 0
    start, end = (from_line, to_line) if from_line < to_line else (to_line, from_line)

    for i in range(start, end):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith(';'):
            continue

        # Skip directives that don't generate code
        if line.startswith('.') or line.startswith('EXTRN') or line == 'END':
            continue

        # Skip pure labels (ending with : but no instruction after)
        if line.endswith(':'):
            continue

        # Handle label: instruction on same line
        if ':' in line and not line.endswith(':'):
            # Extract instruction part after the label
            colon_pos = line.index(':')
            line = line[colon_pos + 1:].strip()
            if not line:
                continue

        # Handle directives
        if line.startswith('DB ') or line.startswith('DB\t'):
            # Count bytes in DB directive - handle strings
            db_content = line[2:].strip()
            byte_count = 0
            in_string = False
            i = 0
            while i < len(db_content):
                c = db_content[i]
                if c == "'":
                    in_string = not in_string
                elif in_string:
                    byte_count += 1
                elif c == ',':
                    pass  # separator
                elif c.isalnum():
                    # Start of a number - count as 1 byte
                    byte_count += 1
                    # Skip rest of number
                    while i + 1 < len(db_content) and db_content[i+1].isalnum():
                        i += 1
                i += 1
            total += max(1, byte_count)
            continue
        if line.startswith('DS ') or line.startswith('DS\t'):
            # DS n reserves n bytes
            m = re.match(r'DS\s+(\d+)', line)
            if m:
                total += int(m.group(1))
            continue
        if line.startswith('DW ') or line.startswith('DW\t'):
            parts = line[3:].split(',')
            total += 2 * len(parts)
            continue
        if 'EQU' in line:
            continue  # EQU doesn't generate code

        # Strip trailing comments
        if ';' in line:
            line = line[:line.index(';')].strip()

        # Normalize whitespace (replace tabs with spaces)
        line = ' '.join(line.split())

        size = get_instr_size(line)
        if size is None:
            return None
        total += size

    # If jumping backward, offset is negative
    if from_line > to_line:
        total = -total

    return total


def apply_jp_to_jr(lines: list[str], label_positions: dict[str, int],
                   jp_candidates: list[tuple[int, str, str]],
                   verbose: bool = False) -> tuple[list[str], int]:
    """
    Convert JP to JR where target is within range.

    JR range is -126 to +129 bytes from the JP instruction.
    (The actual JR offset is from end of JR, which is 2 bytes shorter than JP)
    """
    result = lines.copy()
    total_savings = 0
    converted = []

    for line_idx, condition, target in jp_candidates:
        if target not in label_positions:
            continue

        # JR only supports Z, NZ, C, NC conditions (not P, M, PE, PO)
        if condition and condition not in ('Z', 'NZ', 'C', 'NC'):
            continue

        target_line = label_positions[target]

        # Calculate byte offset from JP to target
        # JP is 3 bytes, JR is 2 bytes
        # JR offset is from end of JR instruction
        offset = calculate_byte_offset(lines, line_idx, target_line)
        if offset is None:
            continue

        # Adjust: JP is at line_idx, we're measuring to target_line
        # For JR, offset is from the byte AFTER the JR instruction
        # If forward jump: offset needs to account for JP being 3 bytes, JR being 2
        # The offset as calculated is from start of JP to start of target
        # For JR: offset is from (JP_addr + 2) to target
        # Original offset is from JP_addr to target
        # So JR offset = original_offset - 2 (we start 2 bytes later with JR, not 3)
        # Actually, JR offset = target - (JR_addr + 2) = target - JP_addr - 2
        # Our offset = target - JP_addr (approximately)
        # So JR_offset = offset - 2 for forward, offset + 1 for backward

        # For forward jumps (offset > 0):
        # JP at addr, target at addr + offset
        # JR at addr, JR ends at addr + 2, target at addr + offset
        # JR offset = offset - 2 (but we save 1 byte, so target moves closer by 1 for each conversion)

        # For backward jumps (offset < 0):
        # Target before JP. JR offset = offset (target addr relative to JR+2)
        # Since target doesn't move and JR+2 is same position as JP+3-1,
        # JR offset = offset + 1

        # JR range is -128 to +127
        # But the actual displacement after JR+2:
        # Forward: need offset - 2 to be in 0..127, so offset in 2..129
        # Backward: need offset + 1 to be in -128..-1, so offset in -129..-2
        # But we only support -126..+129 to be conservative

        if offset > 0:
            jr_offset = offset - 2  # Forward jump offset adjustment
        else:
            jr_offset = offset + 1  # Backward jump offset adjustment

        # Check if in JR range
        if jr_offset < -128 or jr_offset > 127:
            continue

        # Convert JP to JR
        old_line = result[line_idx]
        indent = len(old_line) - len(old_line.lstrip())

        if condition:
            new_instr = f'JR {condition},{target}'
        else:
            new_instr = f'JR {target}'

        result[line_idx] = ' ' * indent + new_instr + '\n'
        total_savings += 1  # JP is 3 bytes, JR is 2 bytes
        converted.append((target, jr_offset))

    if verbose and converted:
        print(f"  JP->JR conversions: {len(converted)} ({total_savings} bytes)")

    return result, total_savings


def select_best_tails(tail_groups: dict[tuple, list[Procedure]]) -> list[tuple[tuple, list[Procedure]]]:
    """
    Select the best tail merges, avoiding conflicts where one procedure
    appears in multiple groups. Prefer longer tails with more procedures.
    """
    # Score each group: (tail_size - 3) * (num_procs - 1)
    scored = []
    for tail, group in tail_groups.items():
        sizes = [get_instr_size(instr) for instr in tail]
        if None in sizes:
            continue
        tail_size = sum(sizes)
        savings = (tail_size - 3) * (len(group) - 1)  # JP costs 3 bytes
        if savings > 0:
            scored.append((savings, tail, group))

    # Sort by savings descending
    scored.sort(reverse=True, key=lambda x: x[0])

    # Greedily select non-conflicting groups
    selected = []
    used_procs = set()

    for savings, tail, group in scored:
        # Check if any proc in this group is already used
        group_procs = set(p.name for p in group)
        if not group_procs & used_procs:
            selected.append((tail, group))
            used_procs.update(group_procs)

    return selected


def optimize_asm(asm_code: str, verbose: bool = False) -> tuple[str, int]:
    """
    Run post-assembly optimizations on assembly code string.

    Multi-pass optimization:
    1. Collect all common tails across all procedures
    2. Select best non-conflicting tail merges
    3. Apply tail merges
    4. Apply skip trick for adjacent procedures
    5. Repeat until no more savings
    6. Final pass: JP to JR conversion

    Returns (optimized_code, total_bytes_saved).
    """
    global _tail_merge_counter
    _tail_merge_counter = 0

    lines = asm_code.splitlines(keepends=True)
    # Ensure all lines have newlines
    lines = [line if line.endswith('\n') else line + '\n' for line in lines]

    total_savings = 0
    pass_num = 0
    max_passes = 5  # Prevent infinite loops

    while pass_num < max_passes:
        pass_num += 1
        pass_savings = 0

        # Parse procedures fresh each pass (line numbers change after edits)
        procs = parse_procedures(lines)

        # Find all common tails
        tail_groups = find_tail_merge_groups(procs)

        if tail_groups:
            # Select best non-conflicting tail merges
            selected = select_best_tails(tail_groups)

            if selected:
                if verbose:
                    print(f"  Pass {pass_num}: Found {len(selected)} tail merge groups")
                for tail, group in selected:
                    # Apply tail merge for this group
                    lines, savings = optimize_tail_merge(lines, procs, {tail: group}, verbose=verbose)
                    pass_savings += savings

                    # Re-parse procs since lines changed
                    procs = parse_procedures(lines)

        # Apply skip trick (can create new opportunities after tail merge)
        skip_opps = find_skip_opportunities(lines)
        if skip_opps:
            lines, savings = apply_skip_trick(lines, skip_opps)
            pass_savings += savings
            if verbose and savings > 0:
                print(f"  Pass {pass_num}: Skip trick saved {savings} bytes")

        total_savings += pass_savings

        # Stop if no savings this pass
        if pass_savings == 0:
            break

    # Final pass: JP to JR conversion
    # Do this after other optimizations stabilize since it depends on byte offsets
    label_positions, jp_candidates = find_jp_to_jr_candidates(lines)
    if jp_candidates:
        lines, savings = apply_jp_to_jr(lines, label_positions, jp_candidates, verbose=verbose)
        total_savings += savings

    return ''.join(lines), total_savings


def optimize(input_path: str, output_path: str | None = None) -> int:
    """
    Run post-assembly optimizations on a .mac file.

    Returns total bytes saved.
    """
    with open(input_path, 'r') as f:
        asm_code = f.read()

    optimized, savings = optimize_asm(asm_code, verbose=True)

    # Write output
    if output_path is None:
        output_path = input_path

    with open(output_path, 'w') as f:
        f.write(optimized)

    return savings


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m uplm80.postopt input.mac [output.mac]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Post-assembly optimization: {input_path}")
    savings = optimize(input_path, output_path)
    print(f"Total savings: {savings} bytes")
