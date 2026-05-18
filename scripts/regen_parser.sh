#!/usr/bin/env bash
# Regenerate uplm80/_plm_parser.py from ../uplox/examples/plm_full.uplox.
#
# The PL/M-80 grammar lives in the uplox repo; uplox emit-python produces a
# typed parser+AST module that this script vendors back into uplm80. Run
# this whenever the upstream grammar changes.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
UPLOX="${UPLOX:-$ROOT/../uplox}"
GRAMMAR="$UPLOX/examples/plm_full.uplox"
OUT="$ROOT/uplm80/_plm_parser.py"
TMP_BUNDLE="$(mktemp -t plm_full.XXXXXX.json)"
TMP_DIR="$(mktemp -d -t uplox_emit.XXXXXX)"
trap 'rm -f "$TMP_BUNDLE"; rm -rf "$TMP_DIR"' EXIT

if [[ ! -f "$GRAMMAR" ]]; then
    echo "grammar not found: $GRAMMAR" >&2
    echo "set UPLOX=/path/to/uplox checkout if it lives elsewhere" >&2
    exit 1
fi

PYTHONPATH="$UPLOX/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m uplox.cli.main build "$GRAMMAR" -o "$TMP_BUNDLE"
PYTHONPATH="$UPLOX/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m uplox.cli.main emit --target=py --out="$TMP_DIR" "$TMP_BUNDLE"
cp "$TMP_DIR/uplox_plm_full.py" "$OUT"
echo "wrote $OUT ($(wc -l <"$OUT") lines)"
