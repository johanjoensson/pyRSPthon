"""
Convert any ASE-readable structure file into RSPt input.

Writes sym/symt.inp and can optionally drive the canonical RSPt setup chain
(symt -all, then stick) to produce a runnable calculation directory.
"""

from argparse import ArgumentParser
import os
import shutil
import subprocess
import sys

import ase.io
from numpy import array

from pyRSPthon.ase import write_symt
from pyRSPthon.ase.atoms import symmetrize_atoms


def find_rspt_binary(name, bin_dir=None):
    """
    Locate an RSPt helper binary: --rspt-bin-dir, then $RSPTHOME/bin, then PATH.
    """
    candidates = []
    if bin_dir:
        candidates.append(os.path.join(bin_dir, name))
    if "RSPTHOME" in os.environ:
        candidates.append(os.path.join(os.environ["RSPTHOME"], "bin", name))
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which(name)


def parse_moments(moment_args, atoms):
    """
    Apply --moments SYMBOL=VALUE settings to the initial magnetic moments.
    """
    magmoms = atoms.get_initial_magnetic_moments()
    for setting in moment_args:
        symbol, _, value = setting.partition("=")
        if not value:
            raise ValueError(f"--moments takes SYMBOL=VALUE, got {setting!r}")
        value = float(value)
        matched = False
        for i, s in enumerate(atoms.get_chemical_symbols()):
            if s == symbol:
                magmoms[i] = value
                matched = True
        if not matched:
            raise ValueError(f"No atoms with symbol {symbol!r} in the structure")
    atoms.set_initial_magnetic_moments(magmoms)


def make_all(rundir, bin_dir=None):
    """
    Run the RSPt setup chain in rundir: symt -all in rundir/sym, then
    stick form > dataForm (in dta/) and stick dataForm > data.
    """
    symt = find_rspt_binary("symt", bin_dir)
    stick = find_rspt_binary("stick", bin_dir)
    if symt is None or stick is None:
        raise RuntimeError(
            "Could not find the RSPt binaries 'symt' and 'stick'. "
            "Set $RSPTHOME or pass --rspt-bin-dir."
        )
    symdir = os.path.join(rundir, "sym")
    subprocess.run([symt, "-all"], cwd=symdir, check=True)

    # `data` is built by the Makefile symt generated (dta/Makefile sticks
    # panel_data and form into dataForm, then `stick dataForm > data`). It
    # needs a top-level length_scale file (runsl rewrites it per volume) and
    # stick on the PATH.
    dta_dir = os.path.join(rundir, "dta")
    length_scale = os.path.join(rundir, "length_scale")
    if not os.path.exists(length_scale) and os.path.exists(
        os.path.join(dta_dir, "length_scale")
    ):
        shutil.copy(os.path.join(dta_dir, "length_scale"), length_scale)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(stick) + os.pathsep + env.get("PATH", "")
    subprocess.run(["make", "data"], cwd=rundir, env=env, check=True)


def report_symmetry(atoms, rundir):
    """
    Cross-check the generated symcof against spglib, if available.
    """
    symcof_path = os.path.join(rundir, "symcof")
    if not os.path.exists(symcof_path):
        return
    try:
        from pyRSPthon.read.symcof import read_symcof, validate_against_spglib

        group = read_symcof(symcof_path)
        report = validate_against_spglib(atoms, group)
    except ImportError:
        print("spglib not installed; skipping symmetry validation.")
        return
    except Exception as err:
        print(f"Symmetry validation failed: {err}")
        return
    print(f"Space group (spglib): {report['spacegroup']}")
    print(
        f"Point-group operations: symt kept {report['symcof_order']}, "
        f"spglib finds {report['spglib_order']}"
    )
    if report["weeded_ops"]:
        print(
            f"  {len(report['weeded_ops'])} operations were weeded away by symt "
            "(spin axis / identity labels reduce the symmetry)."
        )
    if report["extra_ops"]:
        print(
            f"  WARNING: symcof contains {len(report['extra_ops'])} operations "
            "spglib does not find — check the structure!"
        )


def main():
    parser = ArgumentParser(
        description="Convert a structure file (CIF, POSCAR, xyz, ...) to "
        "RSPt input (sym/symt.inp), optionally running symt + stick."
    )
    parser.add_argument("structure", type=str, help="Structure file to convert")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        type=str,
        help="Calculation directory to set up (default: current directory)",
    )
    parser.add_argument("--spinpol", action="store_true", help="Spin polarized")
    parser.add_argument(
        "--fullrel", action="store_true", help="Fully relativistic (spin-orbit)"
    )
    parser.add_argument(
        "--spinaxis",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Spin quantization axis (lattice coordinates)",
    )
    parser.add_argument(
        "--mtradii",
        type=int,
        default=None,
        help="Muffin-tin radius algorithm (0-3, see the RSPt manual)",
    )
    parser.add_argument("--lmax", type=int, default=None, help="Maximum l (default 8)")
    parser.add_argument(
        "--moments",
        action="append",
        default=[],
        metavar="SYMBOL=VALUE",
        help="Initial magnetic moment per element, e.g. --moments Fe=2.2",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="Refine the cell with spglib before writing symt.inp",
    )
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument(
        "--make-all",
        action="store_true",
        help="Run 'symt -all' and 'stick' to produce a runnable data file "
        "(needs the RSPt binaries)",
    )
    parser.add_argument(
        "--rspt-bin-dir",
        type=str,
        default=None,
        help="Directory holding the RSPt binaries (default: $RSPTHOME/bin or PATH)",
    )
    args = parser.parse_args()

    atoms = ase.io.read(args.structure)
    if args.symmetrize:
        atoms = symmetrize_atoms(atoms, symprec=args.symprec)
    parse_moments(args.moments, atoms)

    if args.spinpol:
        atoms.info["spinpol"] = True
    if args.fullrel:
        atoms.info["fullrel"] = True
    if args.spinaxis is not None:
        atoms.info["spinaxis"] = array(args.spinaxis)
    if args.mtradii is not None:
        atoms.info["mtradii"] = args.mtradii
    if args.lmax is not None:
        atoms.info["lmax"] = args.lmax

    symdir = os.path.join(args.output_dir, "sym")
    os.makedirs(symdir, exist_ok=True)
    write_symt(atoms, prefix=symdir)
    print(f"Wrote {os.path.join(symdir, 'symt.inp')}")

    if args.make_all:
        make_all(args.output_dir, args.rspt_bin_dir)
        print(f"Ran symt -all and stick in {args.output_dir}")
        report_symmetry(atoms, args.output_dir)


if __name__ == "__main__":
    main()
