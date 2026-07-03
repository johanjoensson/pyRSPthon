import shutil
import os
import datetime
import glob
import logging
import sys
import time
from dataclasses import dataclass

from . import report
from . import run_rspt

logger = logging.getLogger("pyRSPthon.runs")

start_files = ["pot", "eparm"]
# Files for continuing an RSPt run
last_files = start_files + ["sig"]
# Files used for mixing
jacob_files = ["jacob1", "jacob2"]

# Same divergence guard as the original runs.c (FSQMAX)
FSQ_MAX = 1.0e8

# Abort files: "stopruns" is what the original runs.c reads, "stop" is kept
# for backwards compatibility with earlier versions of this driver.
STOP_FILES = ("stop", "stopruns")


@dataclass
class RunState:
    """Convergence state of an SCF run."""

    fsq: float = float("inf")
    etot: float = float("inf")
    it: int = 0
    ucvol: float = float("nan")
    efermi: float = float("nan")


def setup_logging(verbosity: int = 0):
    """
    Log to the console (stdout) and to runs.info. Safe to call more than once.
    verbosity: -1 = summary only, 0 = per-iteration progress, 1 = debug detail.
    The runs.info file always gets everything.
    """
    logger.setLevel(logging.DEBUG)
    styled = report.detect_style()
    report.set_style(styled)
    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        logfile = logging.FileHandler("runs.info")
        logfile.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(console)
        logger.addHandler(logfile)
    if verbosity < 0:
        console_level = report.SUMMARY
    elif verbosity == 0:
        console_level = logging.INFO
    else:
        console_level = logging.DEBUG
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(console_level)
            handler.setFormatter(report.ConsoleFormatter(styled))


def _emit(lines, level=logging.INFO):
    """Log a report block of (text, color) pairs."""
    for text, color in lines:
        logger.log(level, text, extra={"color": color})


def read_convergence():
    """
    Read data from the convergence file (written by RSPt each cycle;
    single line: fsq  iteration  etotal  unit-cell-volume  fermi-energy).
    Returns:
    ========
    RunState
    Raises:
    =======
    FileNotFoundError if the convergence file does not exist.
    """
    with open("convergence", "rt") as f:
        fields = f.readline().split()
    return RunState(
        fsq=float(fields[0]),
        it=int(fields[1]),
        etot=float(fields[2]),
        ucvol=float(fields[3]),
        efermi=float(fields[4]),
    )


def init_runs() -> RunState:
    """
    Prepare the working directory for an SCF run.

    If a convergence file exists this is a continuation: restore the state
    files (pot, eparm, sig) from their *_last backups where those exist, and
    keep the convergence data. Existing state is never deleted in this path.

    Without a convergence file, start fresh: remove leftover *_last, jacob*
    files, and require pot/eparm to be present either both or neither.

    Returns:
    ========
    RunState - state from the last iteration (defaults if starting fresh)
    """
    try:
        state = read_convergence()
    except FileNotFoundError:
        # Fresh start: clear leftovers from any previous run
        for file in last_files:
            if os.path.exists(f"{file}_last"):
                logger.info(f"Removing {file}_last")
                os.remove(f"{file}_last")
        for file in jacob_files:
            if os.path.exists(file):
                logger.info(f"Removing {file}")
                os.remove(file)
        # Ensure we have either both pot and eparm, or neither
        if not all(os.path.exists(file) for file in start_files) and any(
            os.path.exists(file) for file in start_files
        ):
            present = tuple(file for file in start_files if os.path.exists(file))
            raise RuntimeError(
                "To start a SCF calculation we need either BOTH pot and eparm, "
                "or NEITHER pot nor eparm nor sig.\n"
                f"The following files are present: {present}"
            )
        return RunState()

    # Continuation: restore state files from their _last backups where present.
    # sig_last only exists for DMFT runs; its absence is not an error.
    for file in last_files:
        if os.path.exists(f"{file}_last"):
            shutil.copy(f"{file}_last", file)
    return state


def setup_savedir():
    """
    Create folder to save data for each iteration.
    Moves already existing folders to avoid overwriting data.
    Returns:
    ========
    savedir: str - Path to the folder for saving data
    """
    savedir = f"outdir-{datetime.date.today()}"
    if os.path.exists(savedir):
        offset = len(glob.glob(f"{savedir}-*")) + 1
        shutil.move(savedir, f"{savedir}-{offset}")
    return savedir


def converged(fsq, delta_e, fsq_conv, e_conv):
    """
    Check if the DFT calculation is converged.
    Parameters:
    ===========
    fsq: FSQ from the last iteration
    delta_e: Change in total energy from last iteration
    fsq_conv: convergence limit for fsq
    e_conv: convergence limit for delta_e
    Returns:
    ========
    True  - if fsq is below fsq_conv and delta_e is below e_conv
    False - otherwise
    """
    if fsq_conv < float("inf"):
        logger.debug(f"FSQ    : {fsq:.3E} < {fsq_conv:.3E} ? {fsq < fsq_conv}")
    if e_conv < float("inf"):
        logger.debug(f"Delta E: {delta_e:.3E} < {e_conv:.3E} ? {delta_e < e_conv}")
    return fsq < fsq_conv and delta_e < e_conv


def stop_requested():
    """
    Check the abort files ("stopruns" as in the original runs, and "stop").
    A stop file requests an abort if it is empty or holds a positive integer.
    """
    for stop_file in STOP_FILES:
        if not os.path.exists(stop_file):
            continue
        with open(stop_file, "rt") as f:
            content = f.read().split()
        try:
            if not content or int(content[0]) > 0:
                return stop_file
        except ValueError:
            return stop_file
    return None


def solver_converged():
    """
    Check if the DMFT solver has converged.
    Returns:
    ========
    False - if the last iteration did not produce a new density
    True  - otherwise
    """
    real_conv = False
    sigdiff_real = float("inf")
    mats_conv = False
    sigdiff_mats = float("inf")
    with open("out", "rt") as f:
        for line in f:
            if "sigdiff (mats/real):" in line:
                tmp = line.split()
                sigdiff_mats = float(tmp[2])
                mats_conv = tmp[8] == "T"
                sigdiff_real = float(tmp[3])
                real_conv = tmp[9] == "T"

            if "Stop before density." in line:
                logger.debug(
                    f"    Sigdiff Matsubara: {sigdiff_mats:6.4f} {mats_conv}, Realaxis: {sigdiff_real:6.4f} {real_conv}"
                )
                return False
    return True


def save_solver(solver_it):
    """
    Save data produced by the DMFT solver.
    """
    solver_dir = f"solver-it-{solver_it}"
    os.makedirs(solver_dir, exist_ok=True)
    shutil.copy("out", f"{solver_dir}/")
    for outfile in glob.glob("*.out"):
        if "slurm" in outfile:
            continue
        shutil.copy(outfile, solver_dir)
    for outfile in glob.glob("cthyb-*.log"):
        shutil.copy(outfile, solver_dir)
    if os.path.exists("sig"):
        shutil.copy("sig", solver_dir)
    for datfile in glob.glob("*.dat"):
        shutil.copy(datfile, solver_dir)


def save_it(savedir, it):
    """
    Save data produced in this run of RSPt
    """
    savedir_it = f"{savedir}/outdir-{it}"
    if os.path.exists(savedir_it):
        offset = len(glob.glob(f"{savedir_it}-*")) + 1
        shutil.move(savedir_it, f"{savedir_it}-{offset}")
    os.makedirs(savedir_it)
    shutil.copy("out", savedir_it)
    shutil.copy("pot", savedir_it)
    shutil.copy("eparm", savedir_it)
    if os.path.exists("sig"):
        shutil.copy("sig", savedir_it)
    band_files = glob.glob("*band*.data") + glob.glob("*band*.gpi")
    if band_files:
        os.makedirs(f"{savedir_it}/band", exist_ok=True)
        for bandfile in band_files:
            shutil.move(bandfile, f"{savedir_it}/band")
    if len(glob.glob("*dos*.dat")) > 0:
        os.makedirs(f"{savedir_it}/dos", exist_ok=True)
        for dosfile in glob.glob("*dos*.dat"):
            shutil.move(dosfile, f"{savedir_it}/dos")
    if (len(glob.glob("real-*.dat")) + len(glob.glob("imag-*.dat"))) > 0:
        os.makedirs(f"{savedir_it}/dat", exist_ok=True)
        for datfile in glob.glob("real-*.dat") + glob.glob("imag-*.dat"):
            shutil.move(datfile, f"{savedir_it}/dat")
    for hdffile in glob.glob("*.h5"):
        shutil.move(hdffile, savedir_it)
    for solverdir in glob.glob("solver-it*"):
        shutil.move(solverdir, savedir_it)


def backup_state_files():
    """
    Copy the current state files to their *_last backups.
    """
    shutil.copy("out", "out_last")
    shutil.copy("pot", "pot_last")
    shutil.copy("eparm", "eparm_last")
    if os.path.exists("sig"):
        shutil.copy("sig", "sig_last")


def log_convergence_step(fsq, etot, delta_e):
    """
    Append one iteration to runsConvgeLog (same role as in the original runs).
    """
    with open("runsConvgeLog", "a") as f:
        f.write(f" {fsq:15.8e} {etot:.10f} {delta_e:15.8e}\n")


def verify_green_inputs():
    """
    Verify green.inp (and any green.inp-* iteration variants) before running.
    Findings are logged; errors raise RuntimeError so the run aborts before
    the first RSPt call. A failure of the verifier itself never blocks a run.
    """
    from ..read.greeninp import ERROR as V_ERROR, WARNING as V_WARNING
    from ..verify import verify_green

    files = sorted(
        fname
        for fname in glob.glob("green.inp*")
        if fname == "green.inp" or fname.startswith("green.inp-")
    )
    n_errors = 0
    for fname in files:
        try:
            findings = verify_green(".", fname)
        except Exception as exc:
            logger.warning(f"Could not verify {fname}: {exc}")
            continue
        for finding in findings:
            level = {V_ERROR: logging.ERROR, V_WARNING: logging.WARNING}.get(
                finding.level, logging.DEBUG
            )
            logger.log(level, f"{fname}: {finding}")
        n_errors += sum(finding.level == V_ERROR for finding in findings)
    if n_errors:
        raise RuntimeError(
            f"green.inp verification found {n_errors} error(s) that would "
            "stop or corrupt the run; fix them or rerun with --no-verify"
        )


def runs(
    rspt_binary: list[str],
    run_prefix: list[str],
    fsq_conv: float,
    e_conv: float,
    max_iter: int,
    max_solver_it: int,
    save: bool,
    save_solver_it: bool,
    verbosity: int = 0,
    verbose: bool = False,
    verify_inputs: bool = True,
    **kwargs,
):
    """
    Run RSPt iterations until convergence is achieved.
    Parameters:
    ===========
    rspt_binary: list[str] - The RSPt executable to run (usually ["rspt"])
    run_prefix: list[str]  - Launcher command (ex. ["mpirun", "-n", "64"])
    fsq_conv: float  - convergence criteria for FSQ
    e_conv: float    - convergence criteria for total energy
    max_iter: int    - Maximum number of SCF DFT iterations to run
    max_solver_it: int - Max number of attempts to converge the DMFT self energy (usually 1, but for real axis solvers this should be set higher)
    save: bool      - Save data at each SCF iteration
    save_solver_it  - Save data at each solver iteration
    verbosity: int  - Console detail: -1 summary only, 0 progress, 1 debug
    verbose: bool   - Deprecated alias for verbosity=1
    verify_inputs: bool - Verify green.inp before the first iteration
    **kwargs        - Arguments passed on to run_rspt

    Returns:
    ========
    dict with keys:
      converged: bool  - whether the SCF cycle converged
      diverged: bool   - whether fsq exceeded the divergence guard
      fsq: float       - final fsq
      delta_e: float   - final change in total energy
      etot: float      - final total energy
      it: int          - number of the last completed iteration
    """
    if verbose and verbosity == 0:
        verbosity = 1
    setup_logging(verbosity)
    t_start = time.perf_counter()
    save = save or save_solver_it
    savedir = None
    if save:
        savedir = setup_savedir()
    state = init_runs()
    run_command = " ".join(run_prefix + rspt_binary)
    mode = (
        "fresh start"
        if state.it == 0
        else f"continuation (from iteration {state.it})"
    )
    _emit(
        report.header_block(
            mode=mode,
            command=run_command,
            fsq_conv=fsq_conv,
            e_conv=e_conv,
            max_iter=max_iter,
            max_solver_it=max_solver_it,
            saving=savedir if save else "off",
            started=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    if state.it != 0:
        logger.debug(f"previous fsq = {state.fsq}")
        logger.debug(f"previous total energy = {state.etot}")
    settings = " ".join(f"{key} = {value}" for key, value in kwargs.items())
    logger.debug(f"Other settings: {settings}")
    with open("hist", "a") as f:
        f.write(f"run command: {run_command}\n")
    if verify_inputs:
        verify_green_inputs()

    fsq = state.fsq
    last_e = state.etot
    etot = state.etot
    # Always run at least one new iteration before declaring convergence;
    # the energy difference is unknown until we have a new total energy.
    delta_e = float("inf")
    diverged = False
    it = state.it
    performed = 0
    table_started = False
    while not converged(fsq, delta_e, fsq_conv, e_conv) and performed < max_iter:
        it += 1
        performed += 1
        t_iter = time.perf_counter()
        if os.path.exists(f"green.inp-{it}"):
            shutil.copy(f"green.inp-{it}", "green.inp")

        for i in range(1, max_solver_it + 1):
            stop_file = stop_requested()
            if stop_file is not None:
                _emit([report.stop_line(stop_file)], level=report.SUMMARY)
                raise SystemExit(f"Found a {stop_file} file. Therefore stopping.")
            if os.path.exists(f"green.inp-{it}-{i}"):
                shutil.copy(f"green.inp-{it}-{i}", "green.inp")
            t_rspt = time.perf_counter()
            run_rspt(rspt_binary, run_prefix, **kwargs)
            t_rspt = time.perf_counter() - t_rspt
            logger.debug(f"    RSPt took {t_rspt:5.3f} seconds")

            if solver_converged():
                break
            if save_solver_it:
                save_solver(i)
        backup_state_files()
        if save:
            save_it(savedir, it)
        t_iter = time.perf_counter() - t_iter
        if os.path.exists("convergence"):
            new_state = read_convergence()
            fsq = new_state.fsq
            etot = new_state.etot
            delta_e = abs(last_e - etot)
            last_e = etot
            log_convergence_step(fsq, etot, delta_e)
            if not table_started:
                _emit(report.table_header())
                table_started = True
            _emit([report.iteration_row(it, fsq, etot, delta_e, t_iter)])
        else:
            logger.warning(
                "RSPt did not write a convergence file; cannot check convergence."
            )
        if fsq > FSQ_MAX:
            diverged = True
            logger.error(
                f"SCF cycle diverged: fsq = {fsq:.3e} > {FSQ_MAX:.1e}. Stopping."
            )
            break

    is_converged = converged(fsq, delta_e, fsq_conv, e_conv)
    _emit(
        report.summary_block(
            converged=is_converged,
            diverged=diverged,
            fsq=fsq,
            delta_e=delta_e,
            etot=etot,
            fsq_conv=fsq_conv,
            e_conv=e_conv,
            it=it,
            performed=performed,
            max_iter=max_iter,
            elapsed=time.perf_counter() - t_start,
        ),
        level=report.SUMMARY,
    )
    logger.debug("Full log in runs.info, per-iteration convergence in runsConvgeLog")

    return dict(
        converged=is_converged,
        diverged=diverged,
        fsq=fsq,
        delta_e=delta_e,
        etot=etot,
        it=it,
    )
