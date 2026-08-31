from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyRSPthon.run import runs
from pyRSPthon.run import report
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
    parser = ArgumentParser(
        description="Run SCF calculations with RSPt.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog=(
            "exit codes: 0 converged, 1 error, 2 not converged (max iterations reached or diverged).\n"
            "Note on loops: --max_iter controls the outer SCF/charge self-consistency loop. "
            "For DMFT calculations, --max_solver_it controls the inner impurity solver loop."
        ),
    )

    run_group = parser.add_argument_group("Launcher Options")
    run_group.add_argument(
        "runcommand",
        nargs="+",
        type=str,
        help="RSPt executable, optionally preceded by launcher arguments "
        '(ex. "rspt", "mpirun -n 64 rspt" or "rspt" together with --run-prefix mpirun -n 64)',
    )
    run_group.add_argument(
        "--run-prefix",
        type=str,
        default="",
        metavar="CMD",
        help='Launcher command (ex. "mpirun -n 64" or "srun -n 128 -c 2")',
    )

    conv_group = parser.add_argument_group("Convergence Criteria")
    conv_group.add_argument(
        "--fsq_conv",
        "-f",
        type=float,
        required=True,
        metavar="FLOAT",
        help="Primary convergence criterion: Force square (fsq) convergence threshold in (Ry/Bohr)^2.",
    )
    conv_group.add_argument(
        "--e_conv",
        "-e",
        type=float,
        default=float("inf"),
        metavar="FLOAT",
        help="Energy convergence threshold in Rydbergs (Ry). Note: total energy convergence is not always monotonic during SCF.",
    )
    conv_group.add_argument(
        "--max_iter",
        "-i",
        type=int,
        required=True,
        metavar="INT",
        help="Maximum number of outer SCF iterations to run.",
    )

    basis_group = parser.add_argument_group("Basis & Physical Checks")
    basis_group.add_argument(
        "--max_core_leakage",
        type=float,
        default=1e-3,
        metavar="FLOAT",
        help="Basis sanity check: max allowed core state leakage beyond muffin-tin spheres.",
    )
    basis_group.add_argument(
        "--max_boundary_mismatch",
        type=float,
        default=1e-3,
        metavar="FLOAT",
        help="Basis sanity check: max allowed mismatch at muffin-tin boundaries.",
    )

    solver_group = parser.add_argument_group("DMFT / Solver Options")
    solver_group.add_argument(
        "--max_solver_it",
        type=int,
        default=1,
        metavar="INT",
        help="Maximum number of inner impurity solver iterations per SCF step.",
    )
    solver_group.add_argument(
        "--h_max",
        type=int,
        default=3,
        metavar="INT",
        help="Solver iterations parameter (h_max).",
    )
    solver_group.add_argument(
        "--save_solver_it",
        action="store_true",
        help="Save intermediate solver iterations.",
    )

    misc_group = parser.add_argument_group("Miscellaneous Options")
    misc_group.add_argument(
        "--no-check_rspt",
        action="store_false",
        dest="check_rspt",
        help="Disable checking RSPt executable.",
    )
    misc_group.add_argument(
        "--save", action="store_true", help="Save calculation state."
    )
    misc_group.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify_inputs",
        help="Skip the green.inp verification before the first iteration",
    )
    verbosity = misc_group.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_const",
        const=-1,
        dest="verbosity",
        help="Only print the final summary",
    )
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        const=1,
        dest="verbosity",
        help="Also print debug detail (solver status, timings, file operations)",
    )
    parser.set_defaults(verbosity=0)
    args = parser.parse_args()
    try:
        result = run(**vars(args))
    except RuntimeError as err:
        message = f"Error: {err}"
        if report.detect_style(sys.stderr):
            message = report.colorize(message, "red")
        print(message, file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if result["converged"] else 2)


if __name__ == "__main__":
    main()
