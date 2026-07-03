"""
Verify RSPt input files before running.

Currently checks green.inp (and green.inp-* iteration variants) in a run
directory: structural lint for everything RSPt silently ignores, the
semantic stop conditions of green_init, and cross-file consistency against
data / symcof / spts / symt.inp when those are present.
"""

from argparse import ArgumentParser
import glob
import os
import sys

from pyRSPthon.read.greeninp import ERROR, INFO, WARNING
from pyRSPthon.run import report
from pyRSPthon.verify import verify_green

_LEVEL_COLOR = {ERROR: "red", WARNING: "yellow", INFO: "dim"}
_LEVEL_RANK = {ERROR: 0, WARNING: 1, INFO: 2}


def green_inp_files(directory):
    files = sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(directory, "green.inp*"))
    )
    return [f for f in files if f == "green.inp" or f.startswith("green.inp-")]


def main():
    parser = ArgumentParser(
        description="Verify RSPt input files (green.inp) in a run directory.",
        epilog="exit codes: 0 clean, 1 errors found, 2 warnings found (with --strict)",
    )
    parser.add_argument(
        "--directory", "-d", type=str, default=".", help="Run directory to verify"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero on warnings as well"
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet", "-q", action="store_const", const=ERROR, dest="min_level",
        help="Only report errors",
    )
    verbosity.add_argument(
        "--verbose", "-v", action="store_const", const=INFO, dest="min_level",
        help="Also report informational notes (skipped checks, ignored tokens)",
    )
    parser.set_defaults(min_level=WARNING)
    args = parser.parse_args()

    styled = report.detect_style()

    def emit(text, color=None):
        print(report.colorize(text, color) if styled else text)

    files = green_inp_files(args.directory)
    if not files:
        emit(f"No green.inp in {args.directory!r} - nothing to verify.")
        sys.exit(0)

    n_errors = 0
    n_warnings = 0
    max_rank = _LEVEL_RANK[args.min_level]
    for fname in files:
        findings = verify_green(args.directory, fname)
        findings.sort(key=lambda f: (f.line == 0, f.line, _LEVEL_RANK[f.level]))
        n_errors += sum(f.level == ERROR for f in findings)
        n_warnings += sum(f.level == WARNING for f in findings)
        shown = [f for f in findings if _LEVEL_RANK[f.level] <= max_rank]
        if not findings:
            emit(f"{fname}: clean", "green")
            continue
        counts = ", ".join(
            f"{n} {label}"
            for n, label in (
                (sum(f.level == ERROR for f in findings), "error(s)"),
                (sum(f.level == WARNING for f in findings), "warning(s)"),
                (sum(f.level == INFO for f in findings), "note(s)"),
            )
            if n
        )
        emit(f"{fname}: {counts}", "red" if n_errors else "yellow")
        for finding in shown:
            emit(f"  {finding.level:<7} {finding}", _LEVEL_COLOR[finding.level])

    if n_errors:
        sys.exit(1)
    if args.strict and n_warnings:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
