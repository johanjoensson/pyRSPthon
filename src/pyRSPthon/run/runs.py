import subprocess
import shutil
import os
import datetime
import glob
import sys
import time

from . import run_rspt


start_files = ["pot", "eparm"]
# Files for continuing an RSPt run
last_files = start_files + ["sig"]
# Files used for mixing
jacob_files = ["jacob1", "jacob2"]


def init_runs():
    """
    Reads convergence data from the file "convergence", if it exists
    and sets fsq, total energy and iteration number accordingly.
    Also prepares the file pot, eparm and (potentially) sig, if the matching _last files exist.
    If convergence, pot_last  and eparm_last are present, continue an SCF run.
    Otherwise attempt to restart from pot, eparm, sig (if they exist)
    If that fails start from atomdens
    Return:
    =======
    fsq: float    - Value of fsq from last iteration (inf if no previous iteration)
    last_w: float - Total energy from last iteration (inf if no previous iteration)
    it: int       - Number of the last iteration (0 if no previous iteration)
    """
    # If we have a pot_last, eparm_last or sig_last file,
    # we should continue a previous SCF run
    try:
        fsq, last_e, it, _, _ = read_convergence()
        # Make backups of existing files
        for file in last_files:
            if os.path.exists(file):
                shutil.copy(file, f"{file}.bak")

        for file in last_files:
            shutil.copy(f"{file}_last", file)
        # The backups are no longer needed
        # because we were able to properly copy files from a previous SCF run
        for file in last_files:
            if os.path.exists(f"{file}.bak"):
                os.remove(f"{file}.bak")
    # Otherwise: start fresh.
    # Remove any existing (leftover) "*_last" files and "jacob*" files
    except FileNotFoundError:
        for file in last_files:
            if os.path.exists(f"{file}_last"):
                print(f"Removing {file}_last")
                os.remove(f"{file}_last")
        for file in jacob_files:
            if os.path.exists(file):
                print(f"Removing {file}")
                os.remove(file)
        if os.path.exists("convergence"):
            print("Removing convergence")
            os.remove("convergence")

        fsq = float("inf")
        last_e = float("inf")
        it = 0
        # Restore any files backed up above
        for file in last_files:
            if os.path.exists(f"{file}.bak"):
                print(f"restoring {file}")
                shutil.move(f"{file}.bak", file)
        # Ensure we have both pot and eparm
        if not all(os.path.exists(file) for file in start_files) and any(
            os.path.exists(file) for file in start_files
        ):
            raise RuntimeError(
                f"To start a SCF calculation we need either BOTH pot and eparm, or NEITHER pot nor eparm nor sig.\nThe following files are present {tuple(file for file in start_files if os.path.exists(file))}"
            )
    return fsq, last_e, it


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


def read_convergence():
    """
    Read data from convergence file
    Returns:
    ========
    fsq: float - FSQ from convergence file (inf if no convergence file)
    etot: float - Total energy from convergence file (inf if no convergence file)
    last_it: int - iteration number from convergence file (0 if no convergence file)
    unit_cell_volume: float - volume of the unit cell from the last run
    fermi_energy: float - Fermi energy from the last run
    Raises:
    =======
    FileNotFoundError if the convergence file does not exist.
    """
    with open("convergence", "rt") as f:
        line = f.read()
        fields = line.split()
        fsq = float(fields[0])
        last_iter = int(fields[1])
        etot = float(fields[2])
        unit_cell_volume = float(fields[3])
        fermi_energy = float(fields[4])

    return fsq, etot, last_iter, unit_cell_volume, fermi_energy


def converged(fsq, delta_e, fsq_conv, e_conv, verbose=False):
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
    if verbose:
        if fsq_conv < float("inf"):
            print(f"FSQ    : {fsq:.3E} < {fsq_conv:.3E} ? {fsq < fsq_conv}")
        if e_conv < float("inf"):
            print(f"Delta E: {delta_e:.3E} < {e_conv:.3E} ? {delta_e < e_conv}")
    return fsq < fsq_conv and delta_e < e_conv


def solver_converged(verbose: bool):
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
                if verbose:
                    print(
                        f"    Sigdiff Matsubara: {sigdiff_mats:6.4f} {mats_conv}, Realaxis: {sigdiff_real:6.4f} {real_conv}"
                    )
                return False
    return True


def save_solver(solver_it):
    """
    Save data produced by the DMFT solver.
    """
    solver_dir = f"solver-it-{solver_it}"
    os.makedirs(solver_dir)
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
    if len(glob.glob("*band*.data")) > 0:
        os.makedirs(f"{savedir_it}/band")
        for bandfile in glob.glob("*band*.{data,gpi}"):
            shutil.move(bandfile, f"{savedir_it}/band")
    if len(glob.glob("*dos*.dat")) > 0:
        os.makedirs(f"{savedir_it}/dos")
        for dosfile in glob.glob("*dos*.dat"):
            shutil.move(dosfile, f"{savedir_it}/dos")
    if (len(glob.glob("real-*.dat")) + len(glob.glob("imag-*.dat"))) > 0:
        os.makedirs(f"{savedir_it}/dat")
        for datfile in glob.glob("real-*.dat"):
            shutil.move(datfile, f"{savedir_it}/dat")
        for datfile in glob.glob("imag-*.dat"):
            shutil.move(datfile, f"{savedir_it}/dat")
    if len(glob.glob("*.h5")) > 0:
        for hdffile in glob.glob("*.h5"):
            shutil.move(hdffile, f"{savedir_it}")
    if len(glob.glob("solver-it*")) > 0:
        for solverdir in glob.glob("solver-it*"):
            shutil.move(solverdir, f"{savedir_it}")


def runs(
    rspt_binary: list[str],
    run_prefix: list[str],
    fsq_conv: float,
    e_conv: float,
    max_iter: int,
    max_solver_it: int,
    save: bool,
    save_solver_it: bool,
    verbose: bool = False,
    **kwargs,
):
    """
    Run RSPt iterations until convergence is achieved.
    Parameters:
    ===========
    rspt_binary: str - Name of the RSPt executable to run (usually "rspt")
    fsq_conv: float  - convergence criteria for FSQ
    e_conv: float    - convergence criteria for total energy
    max_iter: int    - Maximum number of SCF DFT iterations to run
    max_solver_it : int - Max number of attempts to converge the DMFT self energy (usually 1, but for real axis solvers this should be set higher)
    run_prefix: str - Commands used to launch the RSPt binary (ex. "mpirun -n 64" or "srun -n 128 -c 2")
    save: bool      - Save data at each SCF iteration
    save_solver_it  - Save data at each solver iteration
    **kwargs        - Arguments passed on to run_rspt

    Returns:
    ========
    True  - if the SCF cycle did converge
    False - otherwise
    """
    save = save or save_solver_it
    savedir = None
    if save:
        savedir = setup_savedir()
    fsq, last_e, it = init_runs()
    if it == 0:
        print(f"Starting new SCF run: {datetime.datetime.now()}")
        if verbose:
            print(f"run command: {' '.join(run_prefix + rspt_binary)}")
            print(
                f"Other settings: {' '.join([f'{key} = {value}' for key, value in kwargs.items()])}"
            )
            print()
        with open("runs.info", "w") as f:
            f.write(f"Starting new SCF run at {datetime.datetime.now()}\n")
            f.write(f"run command: {' '.join(run_prefix + rspt_binary)}\n")
            f.write(
                f"Other settings: {' '.join([f'{key} = {value}' for key, value in kwargs.items()])}\n"
            )
    else:
        print(f"Continuing SCF run from iteration {it}: {datetime.datetime.now()}")
        if verbose:
            print(f"run command: {' '.join(run_prefix + rspt_binary)}")
            print(
                f"Other settings: {' '.join([f'{key} = {value}' for key, value in kwargs.items()])}"
            )
            print(f"fsq read = {fsq}")
            print(f"last total energy read = {last_e}")
            print()

        with open("runs.info", "a") as f:
            f.write(
                f"Continuing SCF run from iteration {it}: {datetime.datetime.now()}\n"
            )
            f.write(f"run command: {' '.join(run_prefix + rspt_binary)}\n")
            f.write(
                f"Other settings: {' '.join([f'{key} = {value}' for key, value in kwargs.items()])}\n"
            )
            f.write(f"fsq read = {fsq}\n")
            f.write(f"last total energy read = {last_e}\n")
            it = 0
    with open("hist", "a") as f:
        f.write(f"run command: {' '.join(run_prefix + rspt_binary)}\n")
    it += 1
    # We don't store the difference in total energy, so use the last e_tot calculated.
    # This is just to have an initial values for delta_e, it will mean that we will run at least one
    # iteration before we think we are converged, regardless of fsq.
    delta_e = abs(last_e)
    etot = last_e
    while not converged(fsq, delta_e, fsq_conv, e_conv, verbose) and it < max_iter + 1:
        if os.path.exists(f"green.inp-{it}"):
            shutil.copy(f"green.inp-{it}", "green.inp")

        for i in range(1, max_solver_it + 1):
            if os.path.exists("stop"):
                sys.exit("Found a stop file. Therefore stopping.")
            if os.path.exists(f"green.inp-{it}-{i}"):
                shutil.copy(f"green.inp-{it}-{i}", "green.inp")
            t_rspt = time.perf_counter()
            run_rspt(rspt_binary, run_prefix, **kwargs)
            t_rspt = time.perf_counter() - t_rspt
            if verbose:
                print(f"    RSPt took {t_rspt:5.3f} seconds")

            if solver_converged(verbose):
                break
            if save_solver_it:
                save_solver(i)
        shutil.copy("out", "out_last")
        shutil.copy("pot", "pot_last")
        shutil.copy("eparm", "eparm_last")
        if os.path.exists("sig"):
            shutil.copy("sig", "sig_last")
        if save:
            save_it(savedir, it)
        if os.path.exists("convergence"):
            fsq, etot, _, _, _ = read_convergence()
            delta_e = abs(last_e - etot)
            last_e = etot
        it += 1
    print(f"Ending SCF run after {it-1} iterations: {datetime.datetime.now()}\n")
    with open("runs.info", "a") as f:
        f.write(f"Ending SCF run after {it} iterations\n")
        f.write(f"fsq = {fsq}\n")
        f.write(f"last total energy = {last_e}\n")

    return dict(
        converged=converged(fsq, delta_e, fsq_conv, e_conv),
        fsq=fsq,
        delta_e=delta_e,
        etot=etot,
        it=it - 1,
    )
