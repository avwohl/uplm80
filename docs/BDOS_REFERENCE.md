# CP/M BDOS Reference for PL/M-80

This document provides a concise reference for calling CP/M BDOS (Basic Disk Operating System) functions from PL/M-80 programs.

## BDOS Interface Procedures

PL/M-80 programs typically declare these external procedures to access BDOS:

```plm
/* BDOS call with no return value */
MON1: PROCEDURE(FUNC, PARM) EXTERNAL;
    DECLARE FUNC BYTE, PARM ADDRESS;
END MON1;

/* BDOS call with byte return value (in register A) */
MON2: PROCEDURE(FUNC, PARM) BYTE EXTERNAL;
    DECLARE FUNC BYTE, PARM ADDRESS;
END MON2;

/* BDOS call with address return value (in register HL) */
MON3: PROCEDURE(FUNC, PARM) ADDRESS EXTERNAL;
    DECLARE FUNC BYTE, PARM ADDRESS;
END MON3;
```

## Common BDOS Functions

### Console I/O

| Function | Name | Parameters | Returns | Description |
|----------|------|------------|---------|-------------|
| 1 | C_READ | None (0) | Byte | Read character from console (waits for input) |
| 2 | C_WRITE | Byte (char) | - | Write character to console |
| 9 | C_WRITESTR | Address (string) | - | Write string to console ($ terminated) |
| 10 | C_READSTR | Address (buffer) | - | Read string from console into buffer |
| 11 | C_STAT | None (0) | Byte | Console status (0xFF if char ready, 0 if not) |

### File I/O

| Function | Name | Parameters | Returns | Description |
|----------|------|------------|---------|-------------|
| 15 | F_OPEN | Address (FCB) | Byte | Open file (0-3 = success, 0xFF = error) |
| 16 | F_CLOSE | Address (FCB) | Byte | Close file (0-3 = success, 0xFF = error) |
| 17 | F_SFIRST | Address (FCB) | Byte | Search for first match (0-3 = success, 0xFF = error) |
| 18 | F_SNEXT | None (0) | Byte | Search for next match (0-3 = success, 0xFF = error) |
| 19 | F_DELETE | Address (FCB) | Byte | Delete file (0-3 = success, 0xFF = error) |
| 20 | F_READ | Address (FCB) | Byte | Read sequential (0 = success, 1 = EOF, 0xFF = error) |
| 21 | F_WRITE | Address (FCB) | Byte | Write sequential (0 = success, 0xFF = error) |
| 22 | F_MAKE | Address (FCB) | Byte | Create file (0-3 = success, 0xFF = error) |

### System Functions

| Function | Name | Parameters | Returns | Description |
|----------|------|------------|---------|-------------|
| 0 | P_TERMCPM | None (0) | - | Terminate program, warm boot to CP/M |
| 12 | S_BDOSVER | None (0) | Address | Get CP/M version (H=major, L=minor) |
| 13 | DRV_ALLRESET | None (0) | - | Reset all disk drives |
| 14 | DRV_SET | Byte (drive) | - | Select disk drive (0=A, 1=B, etc.) |
| 25 | DRV_GET | None (0) | Byte | Get current drive (0=A, 1=B, etc.) |

## Usage Examples

### Example 1: Print String

```plm
PRINT$STRING: PROCEDURE(STR);
    DECLARE STR ADDRESS;
    CALL MON1(9, STR);  /* BDOS function 9: print string */
END PRINT$STRING;

DECLARE MSG DATA ('Hello, World!$');
CALL PRINT$STRING(.MSG);
```

### Example 2: Print Character

```plm
PRINT$CHAR: PROCEDURE(C);
    DECLARE C BYTE;
    CALL MON1(2, C);  /* BDOS function 2: write character */
END PRINT$CHAR;

CALL PRINT$CHAR('A');
CALL PRINT$CHAR(13);  /* CR */
CALL PRINT$CHAR(10);  /* LF */
```

### Example 3: Read Character

```plm
READ$CHAR: PROCEDURE BYTE;
    RETURN MON2(1, 0);  /* BDOS function 1: read character */
END READ$CHAR;

DECLARE CH BYTE;
CH = READ$CHAR();
```

### Example 4: Check Console Status

```plm
CONSOLE$READY: PROCEDURE BYTE;
    /* Returns 0xFF if character ready, 0 if not */
    RETURN MON2(11, 0);  /* BDOS function 11: console status */
END CONSOLE$READY;

IF CONSOLE$READY() <> 0 THEN
    CALL PROCESS$INPUT();
```

### Example 5: File Operations

```plm
DECLARE FCB DATA (
    0,                  /* Drive (0=default, 1=A, 2=B, etc.) */
    'TESTFILE',         /* Filename (8 chars) */
    'TXT',              /* Extension (3 chars) */
    0,0,0,0,            /* Current block, etc. */
    /* ... rest of FCB ... */
);

/* Open file */
IF MON2(15, .FCB) = 0FFH THEN
    CALL PRINT$STRING(.('File not found$'));
ELSE
    /* Read first record */
    IF MON2(20, .FCB) = 0 THEN
        /* Process data in DMA buffer */
        CALL PROCESS$RECORD();
    END;
    /* Close file */
    CALL MON1(16, .FCB);
END;
```

## File Control Block (FCB) Structure

```plm
DECLARE FCB STRUCTURE (
    DRIVE BYTE,         /* 0=default, 1=A:, 2=B:, etc. */
    NAME(8) BYTE,       /* Filename (space padded) */
    EXT(3) BYTE,        /* Extension (space padded) */
    EX BYTE,            /* Current extent */
    S1 BYTE,            /* Reserved */
    S2 BYTE,            /* Reserved */
    RC BYTE,            /* Record count in current extent */
    AL(16) BYTE,        /* Disk allocation map */
    CR BYTE,            /* Current record */
    R0 BYTE,            /* Random record number (low) */
    R1 BYTE,            /* Random record number (mid) */
    R2 BYTE             /* Random record number (high) */
);
```

## String Termination

BDOS function 9 (C_WRITESTR) requires strings terminated with '$':

```plm
DECLARE MESSAGE DATA ('Hello, World!', 13, 10, '$');
CALL MON1(9, .MESSAGE);
```

## DMA Buffer

CP/M uses a 128-byte DMA (Direct Memory Access) buffer at address 0080H by default for file I/O. You can set a custom DMA address with BDOS function 26.

## Additional Resources

- CP/M 2.2 Operating System Manual
- CP/M BDOS System Interface documentation
- See `examples/hello_cpm.plm` for a complete working example
