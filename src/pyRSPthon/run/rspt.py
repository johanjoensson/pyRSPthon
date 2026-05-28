"""
Collection of functions for running single RSPt steps and the calculations.
"""

import subprocess
import os
import signal
import time


def check_reset_fields(fields: list[str], t: int, e: int, l: int):
    """
    Check the node reset flag. Raise an exception if it is set.
    Parameters:
    ===========
    fields: list[str] - all fields for the energy parameters
    t: int            - type
    e: int            - energy set
    l: int            - l quantum number
    Returns:
    ========
    t: int            - type checked
    e: int            - energy set checked
    l: int            - l quantum number checked
    """
    if len(fields) == 11:
        t = int(fields[0])
        e = int(fields[1])
        l = int(fields[2])
    elif len(fields) == 10:
        e = int(fields[0])
        l = int(fields[1])
    elif len(fields) == 9:
        l = int(fields[0])
    if fields[-3] == "*":
        raise RuntimeError(f"Node reset flag for type {t}  energy set {e} and l {l}")
    return t, e, l


def check_fourier_arguments(fields: list[str]):
    """
    Check if the fourier argument is >= 6
    Parameters:
    ===========
    fields: list[str] - fields to check
    """
    if int(fields[-1]) <= 5:
        raise RuntimeError(
            f"Fourier mesh is not dense enough. Fourier argument is {fields[-1]}."
        )


def check_boundary_densities(
    fields: list[str], t, h, mt_density, h_max, max_boundary_mismatch
):
    """
    Check matching conditions at the boundaries.
    Parameters:
    ===========
    fields: list[str] - fields to check
    t: int  - type
    h: int  - harmonic
    density: float - current MT density
    Returns:
    ========
    t: int - Type checked
    h: int - Harmonics checked
    density: float - MT density to check
    """
    tol = max_boundary_mismatch if h <= h_max else 0.1
    if len(fields) == 5:
        t = int(fields[0])
        h = int(fields[1])
        mt_density = float(fields[2])
        assert fields[-1] == "mt"
    elif len(fields) == 4:
        h = int(fields[0])
        mt_density = float(fields[1])
        assert fields[-1] == "mt"
    elif len(fields) == 3:
        int_density = float(fields[0])
        assert fields[-1] == "int"
        if abs(mt_density - int_density) > tol:
            raise RuntimeError(
                f"Fourier mesh is not dense enough. Density mismatch for type {t} harmonic {h} is {abs(mt_density - int_density)}."
            )
        mt_density = 0

    return t, h, mt_density


def check_core_leakage(fields: list[str], t: int, max_core_leakage: float):
    """
    Check core state leakage
    Parameters:
    ===========
    fields: list[str] - Fields to check
    t: int - type to check
    max_core_leakage: float - maximum allowed core leakage
    """
    leakage = float(fields[-1])
    if leakage > max_core_leakage:
        raise RuntimeError(f"Core leakage for type {t} is large: {leakage} electrons.")


def check_rspt_run(h_max: int, max_boundary_mismatch: float, max_core_leakage: float):
    """
    Check an RSPt run for common errors.
    Checks energy reset parameter, look for stars in second column.
    Checks Fourier mesh, check arguments and density match at MT boundary.
    Check core leakage.
    Parameters:
    ===========
    h_max: int - Maximum harmonics to check for fine matching at the MT boundaries (usually 3 or 4)
    max_boundary_mismatch: float - Maximum allowed density mismatch at MT boundary (usually 1e3 or 1e4)
    max_core_leakage: float - Maximum allowed core leakage (usually 1e-6 or less)
    """
    check_overlapping_mt = False
    check_reset = False
    check_fourier = False
    check_boundary = False
    check_core = False
    t = 0
    e = 0
    l = -1
    h = 0
    density = 0.0
    with open("out", "rt") as f:
        for line in f:
            if "TIME: COUNTN" in line:
                check_overlapping_mt = False
            if (
                "Nearest neighbors: min_{b+R} |b+R-a|, |b+R-a| < 1.1 (min|R|=12.9767491)"
                in line
            ):
                check_overlapping_mt = True
                for _ in range(2):
                    line = next(f)
            if check_overlapping_mt:
                t = check_overlapping_muffin_tins(line.strip().split(), t)
            if "TIME: setene" in line:
                check_reset = False
            if "Energy parameters" in line:
                check_reset = True
                for _ in range(4):
                    line = next(f)
            if check_reset:
                t, e, l = check_reset_fields(line.strip().split(), t, e, l)
            if "Lm    m1    m2    m3" in line:
                check_fourier = False
            if "Fourier transform parameters" in line:
                check_fourier = True
                for _ in range(2):
                    line = next(f)
            if check_fourier:
                check_fourier_arguments(line.strip().split())
            if "TIME: conden" in line:
                check_boundary = False
            if "Charge densities at boundaries" in line:
                check_boundary = True
                for _ in range(2):
                    line = next(f)
            if check_boundary:
                t, h, density = check_boundary_densities(
                    line.strip().split(), t, h, density, h_max, max_boundary_mismatch
                )
            if "TIME: CORED" in line:
                check_core = False
            if "core states" in line:
                check_core = True
            if check_core and "type:" in line:
                t = int(line.strip().split()[-1])
            if check_core and "leakage" in line:
                check_core_leakage(line.strip().split(), t, max_core_leakage)


def check_overlapping_muffin_tins(fields: list[str], ta: int):
    """
    Check for overlapping muffin tins
    Arguments:
    ==========
    fields: list[str] - fields to check
    ta: int - type a
    """
    tb = None
    two_s_over_d = None
    if len(fields) == 8:
        ta = int(fields[0])
        tb = int(fields[1])
        two_s_over_d = float(fields[3])
    elif len(fields) == 7:
        tb = int(fields[0])
        two_s_over_d = float(fields[2])
    if two_s_over_d > 1.0:
        raise RuntimeError(
            f"Overlapping muffin tin spheres for type {ta} and type {tb}, 2S/d = {two_s_over_d}"
        )
    return ta


def check_early_fail():
    reset_checked = False
    t = 0
    e = 0
    l = -1
    # Wait until we have checked all the reset fields
    while not reset_checked:
        check_reset = False
        check_overlapping_mt = False
        with open("out", "rt") as f:
            try:
                for line in f:
                    if "TIME: COUNTN" in line:
                        check_overlapping_mt = False
                    if (
                        "Nearest neighbors: min_{b+R} |b+R-a|, |b+R-a| < 1.1 (min|R|=12.9767491)"
                        in line
                    ):
                        check_overlapping_mt = True
                        for _ in range(2):
                            line = next(f)
                    if check_overlapping_mt:
                        t = check_overlapping_muffin_tins(line.strip().split(), t)

                    if "TIME: setene" in line:
                        reset_checked = True
                        check_reset = False
                        break
                    if "Energy parameters" in line:
                        check_reset = True
                        for _ in range(4):
                            line = next(f)
                    if check_reset:
                        t, e, l = check_reset_fields(line.strip().split(), t, e, l)
            except StopIteration:
                check_reset = False
                check_overlapping_mt = False
                reset_checked = False
        time.sleep(0.5)


def run_rspt(rspt_binary: list[str], run_prefix: list[str], check_rspt: bool, **kwargs):
    """
    rspt_binary: str - Name of the RSPt executable to run (usually "rspt")
    run_prefix: str - Commands used to launch the RSPt binary (ex. "mpirun -n 64" or "srun -n 128 -c 2")
    **kwargs        - Arguments passed on to check_rspt_run
    """

    with subprocess.Popen(run_prefix + rspt_binary) as proc:
        # Handle SIGTERM signals by aborting all execution
        def handler(signum, _):
            signame = signal.Signals(signum).name
            print(f"Signal {signame} ({signum}) received.")
            print("Terminating execution.")
            proc.send_signal(signal.SIGINT)

        signal.signal(signal.SIGTERM, handler)
        try:
            # Wait until the out file is created
            while not os.path.exists("out"):
                time.sleep(0.5)

            # Early exit if reset fields are set
            check_early_fail()
        except Exception as err:
            proc.send_signal(signal.SIGINT)
            proc.kill()
            raise err  #  from None
        # proc.wait()
        err = proc.poll()
        if err:
            raise RuntimeError(
                f"RSPt command {run_prefix} {rspt_binary} failed! Returned vaues was {err}."
            )
    if check_rspt:
        check_rspt_run(**kwargs)
