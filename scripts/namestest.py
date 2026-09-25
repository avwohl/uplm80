#!/usr/bin/env python3
"""Run the name-resolution differential test at scale.

    python3 scripts/namestest.py [--seeds N] [--first S] [--modules] [-O 0,1,2,3]

Each seed is a random program from tests/names_difftest.py that reuses a
few names at every depth; it is compiled at every requested optimization
level, run under cpmemu, and compared with what the program's scopes say it
prints.  --modules makes each a program of three modules compiled together.
A mismatch prints the seed and the level; `--keep DIR' saves the sources.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.names_difftest import build_and_run, generate, generate_modules  # noqa: E402  pylint: disable=wrong-import-position


def main() -> int:
    """Run the seeds; 1 if any program differs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--modules", action="store_true")
    ap.add_argument("-O", dest="levels", default="0,1,2,3")
    ap.add_argument("--keep", default=None, help="directory for failing sources")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    failures = 0
    for seed in range(args.first, args.first + args.seeds):
        if args.modules:
            sources, expect = generate_modules(seed)
        else:
            src, expect = generate(seed)
            sources = [src]
        bad = []
        for opt in levels:
            got, err = build_and_run(sources, opt)
            if err:
                bad.append(f"  -O{opt}: {err.strip().splitlines()[-1][:200]}")
            elif got != expect:
                i = next((i for i, (g, e) in enumerate(zip(got, expect)) if g != e),
                         min(len(got), len(expect)))
                bad.append(f"  -O{opt}: printed {got[i:i + 3]} at {i}, expected {expect[i:i + 3]}")
        if bad:
            failures += 1
            print(f"seed {seed}:")
            print("\n".join(bad))
            if args.keep:
                os.makedirs(args.keep, exist_ok=True)
                for i, text in enumerate(sources):
                    with open(os.path.join(args.keep, f"seed{seed}_{i}.plm"), "w",
                              encoding="ascii") as f:
                        f.write(text)
            sys.stdout.flush()
    print(f"{failures} of {args.seeds} programs differ from their scopes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
