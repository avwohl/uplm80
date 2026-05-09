"""
uplm80 - Highly optimizing PL/M-80 compiler targeting Z80

This compiler implements the PL/M-80 language as specified in Intel's
PL/M-80 Programming Manual (9800268B, Jan 1980). It generates optimized
Zilog Z80 assembly.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("uplm80")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"  # Running from source without install
