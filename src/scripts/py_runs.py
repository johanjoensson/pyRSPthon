from argparse import ArgumentParser
from pyRSPthon.run import runs
import shlex
import shutil
import sys


def run(runcommand: list[str], run_prefix: str, **kwargs):
    run_array = [val for part in runcommand for val in shlex.split(part)]
    run_prefix_array = shlex.split(run_prefix)
    rspt_executable = shutil.which(run_array[-1])
    run_prefix_array = run_prefix_array + run_array[:-1]
    run_prefix_array[0] = shutil.which(run_prefix_array[0])
    runs([rspt_executable], run_prefix_array, **kwargs)


def main():
    parser = ArgumentParser(description="Run SCF calculations with RSPt.")
    parser.add_argument("runcommand", nargs="+", type=str)
    parser.add_argument("--fsq_conv", "-f", type=float)
    parser.add_argument("--max_iter", "-i", type=int)
    parser.add_argument("--e_conv", "-e", type=float, default=float("inf"))
    parser.add_argument("--max_solver_it", type=int, default=1)
    parser.add_argument("--max_core_leakage", type=float, default=1e-3)
    parser.add_argument("--h_max", type=int, default=3)
    parser.add_argument("--max_boundary_mismatch", type=float, default=1e-3)
    parser.add_argument("--no-check_rspt", action="store_false", dest="check_rspt")
    parser.add_argument("--run-prefix", type=str, default="")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_solver_it", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        run(**vars(args))
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
