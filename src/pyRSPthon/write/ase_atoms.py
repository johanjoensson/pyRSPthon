from ase import Atoms
from ase.units import Bohr
from numpy import transpose
from datetime import date


def get_tags(atoms):
    magmoms = atoms.get_initial_magnetic_moments()
    tags = atoms.get_tags()
    tags = []
    for mm, tag in zip(magmoms, tags):
        if mm > 0:
            tags.append("up")
        elif mm < 0:
            tags.append("dn")
        else:
            tags.append(tag)

    return tags


def symmetrize(atoms, symprec, angle_tol):
    import spglib

    original_cell = (
        atoms.cell[:].tolist(),
        atoms.get_scaled_positions(),
        atoms.get_tags(),
        atoms.get_initial_magnetic_moments(),
    )
    (refined_cell, refined_basis, refined_z, refined_magmoms) = spglib.refine_cell(
        original_cell, symprec=symprec, angle_tolerance=angle_tol
    )

    return Atoms(
        cell=refined_cell,
        scaled_positions=refined_basis,
        numbers=refined_z,
        magmoms=refined_magmoms,
    )


def write_symt(
    atoms: Atoms,
    spinpol: bool,
    spinaxis: tuple[float, float, float, str],
    fullrel: bool,
    mtradii: int,
    symmetrize: bool,
    symprec: float,
    angle_tol: float,
):
    if symmetrize:
        atoms = symmetrize(atoms, symprec, angle_tol)
    magmoms = atoms.get_initial_magnetic_moments()
    ncl = len(magmoms.shape) > 1
    assert (
        not ncl
    ), "RSPt is not capable of doing non-collinear calculations. Initial moments need to be scalar."

    if any(abs(mm) > 0 for mm in atoms.get_initial_magnetic_moments()):
        spinpol = True

    # If fullrel and spinpol are true, we need to supply a spinaxis
    # by deault, along crystal z
    if fullrel and spinpol and spinaxis is None:
        spinaxis = (0.0, 0.0, 1.0, "c")
    cell = atoms.get_cell()
    lengthscale = atoms(cell[0])

    with open("symt.inp", "w") as f:
        f.write(f"# {atoms.symbols}\n")
        f.write(f"# {str(date.today())}\n")
        f.write("\n")
        f.write("lengthscale\n")
        f.write(f" {lengthscale/Bohr:.16f}\n")
        f.write("\n")
        f.write("latticevectors\n")
        [
            f.write(
                " " + "  ".join([f"{el/lengthscale: > 19.16f}" for el in row]) + "\n"
            )
            for row in transpose(cell)
        ]
        f.write("\n")
        if spinpol:
            f.write("spinpol\n")
        if spinaxis is not None:
            f.write("spinaxis\n")
            sx, sy, sz, c = spinaxis
            f.write(f" {sx} {sy} {sz} {c}\n")
        if fullrel:
            f.write("fullrel\n")
        f.write("\n")
        f.write("mtradii\n")
        f.write(f" {mtradii}\n")

        f.write("\n")
        f.write("atoms\n")
        f.write(f" {len(atoms)}\n")
        [
            f.write(
                "  "
                + "  ".join([f"{el:> 19.16f}" for el in pos])
                + f"  {Z} l {tag} ! {sym} {tag}\n"
            )
            for pos, Z, tag, sym in zip(
                atoms.get_scaled_positions(wrap=False),
                atoms.get_atomic_numbers(),
                get_tags(atoms),
                atoms.get_chemical_symbols(),
            )
        ]
