#!/usr/bin/env python3
"""Run the cross-module calling-convention fuzz test at scale.

    python3 scripts/abifuzz.py [--seeds N] [--first S] [-O 0,1,2,3] [--no-mixed]
                               [--no-together] [--jobs J] [--keep DIR]

Each seed is a random program from tests/abi_fuzz.py: two PL/M modules,
compiled apart, and one of assembly written to PL/M-80's calling convention,
calling one another's procedures.  It is built at -O0, which must leave SP
where it found it, and at every other requested level, with the two modules
at different levels (--no-mixed: not) and in one multi-file compile
(--no-together: not); every build must print what -O0's prints.  A mismatch
prints the seed and the builds; `--keep DIR' saves the program's sources.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.abi_fuzz import check, generate  # noqa: E402  pylint: disable=wrong-import-position


def _one(job: tuple) -> tuple[int, list[str]]:
    seed, kw = job
    return seed, check(seed, **kw)


def main() -> int:
    """Run the seeds; 1 if any program fails."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("-O", dest="levels", default="0,1,2,3")
    ap.add_argument("--no-mixed", dest="mixed", action="store_false")
    ap.add_argument("--no-together", dest="together", action="store_false")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--keep", default=None, help="directory for failing sources")
    args = ap.parse_args()
    kw = {"levels": tuple(int(x) for x in args.levels.split(",")),
          "mixed": args.mixed, "together": args.together}
    jobs = [(seed, kw) for seed in range(args.first, args.first + args.seeds)]
    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for seed, bad in pool.map(_one, jobs):
            if not bad:
                continue
            failures += 1
            print(f"seed {seed}:")
            print("\n".join("  " + b for b in bad))
            if args.keep:
                os.makedirs(args.keep, exist_ok=True)
                prog = generate(seed)
                for name, text in (("a.plm", prog.a), ("b.plm", prog.b), ("fwd.mac", prog.asm)):
                    with open(os.path.join(args.keep, f"seed{seed}_{name}"), "w",
                              encoding="ascii") as fh:
                        fh.write(text)
            sys.stdout.flush()
    print(f"{failures} of {args.seeds} programs fail")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
