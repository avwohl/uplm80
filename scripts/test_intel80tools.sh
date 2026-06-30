#!/usr/bin/env bash
# Compile-test the Intel link 3.0 PL/M sources against uplm80.
#
# ogdenpm/intel80tools is a reconstruction of the original Intel ISIS-II
# development tools. Its src/link_3.0 tree is the source that surfaced the
# issue #7 codegen crash, so it makes a good real-world regression corpus.
#
# This harness unpacks the packed link_3.0 source, regenerates the per-module
# .ipx includes from link.pex (see scripts/genipx.py), then compiles every
# module with uplm80 and assembles the result with um80. All 14 modules are
# expected to compile and assemble cleanly; the harness exits non-zero if any
# module crashes the compiler, fails to compile, or fails to assemble.
#
# The corpus lives beside this repo as ../intel80tools (override with
# INTEL80TOOLS=/path). If it is missing the harness prints how to clone it and
# exits 0 (skipped), so it is safe to run in checkouts without the corpus.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
INTEL80="${INTEL80TOOLS:-$ROOT/../intel80tools}"
PACK="$INTEL80/src/link_3.0/link_3.0_all.src"

if [[ ! -f "$PACK" ]]; then
    echo "SKIP: intel80tools corpus not found at $INTEL80"
    echo "  clone it:  git clone --depth 1 https://github.com/ogdenpm/intel80tools.git \"$INTEL80\""
    echo "  or set:    INTEL80TOOLS=/path/to/intel80tools $0"
    exit 0
fi

have_um80=0
command -v um80 >/dev/null 2>&1 && have_um80=1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python3 "$HERE/genipx.py" "$PACK" "$WORK" || { echo "FAIL: could not prepare corpus"; exit 1; }

ok=0 err=0 crash=0 asmerr=0
for plm in "$WORK"/*.plm; do
    b="$(basename "$plm")"
    out="$(cd "$WORK" && python3 -m uplm80.compiler "$b" -o "$WORK/${b%.plm}.mac" 2>&1)"
    rc=$?
    if grep -q "Traceback (most recent call last)" <<<"$out"; then
        echo "CRASH  $b"
        grep -iE "error" <<<"$out" | tail -1 | sed 's/^/         /'
        crash=$((crash + 1))
        continue
    fi
    if [[ $rc -ne 0 ]]; then
        echo "ERR    $b"
        grep -iE "error" <<<"$out" | head -1 | sed 's/^/         /'
        err=$((err + 1))
        continue
    fi
    if [[ $have_um80 -eq 1 ]]; then
        if um80 "$WORK/${b%.plm}.mac" -o "$WORK/${b%.plm}.rel" >/dev/null 2>&1; then
            echo "OK     $b"
            ok=$((ok + 1))
        else
            echo "ASMERR $b"
            asmerr=$((asmerr + 1))
        fi
    else
        echo "OK*    $b  (compiled; um80 not on PATH, skipped assembly)"
        ok=$((ok + 1))
    fi
done

echo "----------------------------------------------"
echo "link_3.0:  $ok ok   $err err   $asmerr asmerr   $crash crash"
[[ $have_um80 -eq 1 ]] || echo "(um80 not found; assembly step skipped)"

if [[ $((err + crash + asmerr)) -ne 0 ]]; then
    exit 1
fi
echo "PASS"
