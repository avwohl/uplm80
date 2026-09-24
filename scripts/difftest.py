#!/usr/bin/env python3
"""Run the PL/M-80 expression-typing differential test at scale.

    python3 scripts/difftest.py [--seeds N] [--first S] [--stmts K] [-O 0,1,2,3]

Each seed is a random program from tests/plm_difftest.py; it is compiled at
every requested optimization level, run under cpmemu, and compared with what
the Python model of the PL/M-80 rules says it prints. A mismatch prints the
seed, the level and the first statements that differ; `--keep DIR' saves the
source of every failing program.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.plm_difftest import generate  # noqa: E402
from tests.plm_difftest import build_and_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first", type=int, default=1000)
    ap.add_argument("--stmts", type=int, default=60)
    ap.add_argument("-O", dest="levels", default="0,1,2,3")
    ap.add_argument("--keep", default=None, help="directory for failing sources")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    failures = 0
    for seed in range(args.first, args.first + args.seeds):
        src, expect = generate(seed, args.stmts)
        bad = []
        for opt in levels:
            got, err = build_and_run(src, opt)
            if err:
                bad.append(f"  -O{opt}: {err.strip().splitlines()[-1][:200]}")
                continue
            if len(got) != len(expect):
                bad.append(f"  -O{opt}: printed {len(got)} values, expected {len(expect)}")
            wrong = [(label, v, g) for (label, v), g in zip(expect, got) if v != g]
            for label, v, g in wrong[:3]:
                bad.append(f"  -O{opt}: {label[:150]}: got {g:04X}, expected {v:04X}")
            if len(wrong) > 3:
                bad.append(f"  -O{opt}: ... {len(wrong) - 3} more")
        if bad:
            failures += 1
            print(f"seed {seed}:")
            print("\n".join(bad))
            if args.keep:
                os.makedirs(args.keep, exist_ok=True)
                with open(os.path.join(args.keep, f"seed{seed}.plm"), "w") as f:
                    f.write(src)
            sys.stdout.flush()
    print(f"{failures} of {args.seeds} programs differ from the model")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
