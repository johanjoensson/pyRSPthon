from argparse import ArgumentParser
from pyRSPthon.run import runs
import shlex
import shutil
import sys


def run(runcommand: list[str], run_prefix: str, **kwargs):
    run_array = [val for part in runcommand for val in shlex.split(part)]
    rspt_executable = shutil.which(run_array[-1])
    if rspt_executable is None:
        raise RuntimeError(
            f"Could not find the RSPt executable {run_array[-1]!r} on PATH."
        )
    prefix_array = shlex.split(run_prefix) + run_array[:-1]
    if prefix_array:
        launcher = shutil.which(prefix_array[0])
        if launcher is None:
            raise RuntimeError(
                f"Could not find the launcher {prefix_array[0]!r} on PATH."
            )
        prefix_array[0] = launcher
    return runs([rspt_executable], prefix_array, **kwargs)


def main():
    parser = ArgumentParser(description="Run SCF calculations with RSPt.")
    parser.add_argument(
        "runcommand",
        nargs="+",
        type=str,
        help="RSPt executable, optionally preceded by launcher arguments "
        '(ex. "rspt" or "-n 64 rspt" together with --run-prefix mpirun)',
    )
    parser.add_argument(
        "--fsq_conv",
        "-f",
        type=float,
        required=True,
        help="Convergence criterion for FSQ",
    )
    parser.add_argument(
        "--max_iter",
        "-i",
        type=int,
        required=True,
        help="Maximum number of SCF iterations to run",
    )
    parser.add_argument(
        "--e_conv",
        "-e",
        type=float,
        default=float("inf"),
        help="Convergence criterion for the total energy change",
    )
    parser.add_argument("--max_solver_it", type=int, default=1)
    parser.add_argument("--max_core_leakage", type=float, default=1e-3)
    parser.add_argument("--h_max", type=int, default=3)
    parser.add_argument("--max_boundary_mismatch", type=float, default=1e-3)
    parser.add_argument("--no-check_rspt", action="store_false", dest="check_rspt")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help='Launcher command (ex. "mpirun -n 64" or "srun -n 128 -c 2")',
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_solver_it", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        result = run(**vars(args))
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if result["converged"] else 2)


if __name__ == "__main__":
    main()
