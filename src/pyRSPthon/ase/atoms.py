from ase import Atoms, Atom
from ase.units import Bohr
from numpy.linalg import norm
from numpy import array, empty, zeros, transpose, abs as np_abs, any as np_any
from numpy.linalg import solve
import datetime
import os


def extract_header(file_it):
    lines = []
    for line in file_it:
        line = line.strip()
        if len(line) == 0:
            break
        elif line[0] == "#":
            lines.append(line[1:].strip())
        break
    if len(lines) == 0:
        return None
    return "\n".join(lines)


def extract_float(file_it):
    line = next(file_it)
    return float(line.strip())


def extract_matrix(file_it):
    line = next(file_it)
    l0 = line.strip().split()
    line = next(file_it)
    l1 = line.strip().split()
    line = next(file_it)
    l2 = line.strip().split()
    return array(
        [[float(el) for el in l0], [float(el) for el in l1], [float(el) for el in l2]]
    )


def extract_int(file_it):
    line = next(file_it)
    return int(line.strip())


def extract_vector(file_it):
    line = next(file_it)
    tmp = line.strip().split()
    vector = array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    lattice_or_cartesian = tmp[3]
    return vector, lattice_or_cartesian


def extract_atoms(file_it):
    line = next(file_it)
    n_at = int(line.strip())

    Z = [0] * n_at
    labels = ["NO"] * n_at
    positions = empty((n_at, 3))
    magmoms = zeros((n_at, 3))
    lattice_or_cartesian = ["NO"] * n_at

    for i in range(n_at):
        line = next(file_it)
        tmp = line.strip().split()
        assert len(tmp) >= 6
        positions[i] = [float(tmp[0]), float(tmp[1]), float(tmp[2])]
        Z[i] = int(tmp[3])
        lattice_or_cartesian[i] = tmp[4]
        labels[i] = tmp[5]
        if tmp[5] == "up":
            magmoms[i] = 1
        elif tmp[5] == "dn":
            magmoms[i] = -1
    return Z, labels, positions, magmoms, lattice_or_cartesian


def vector_to_lattice(vec, l_or_cart, cell):
    if l_or_cart in "l":
        return vec
    return solve(cell.T, vec)


def read_symt(fname: str):
    """
    Read RSPt symt.inp file (in the new format) and return an ASE Atoms object.
    Additional RSPt info (such as mtradii, lmax, ...) is stored in the info member
    of the Atoms object.
    Arguments
    =========
    fname: string - file to read
    Returns
    =======
    Atoms - Structure read form file, with additional information in the info member.
    """
    lengthscale = 0
    cell = empty((3, 3))
    positions = empty((0, 3))
    lattice_or_cartesian = []
    spinpol = False
    fullrel = False
    lmax = 8
    mtradii = 0
    initial_moments = []
    labels = []
    Z = []
    spinaxis = None
    strainmatrix = None
    l_to_w = None
    offsets = None
    spinpol_atomdens = None
    header = None
    with open(fname, "r") as f:
        f_it = iter(f)
        # Initial commented lines are treated as a header
        header = extract_header(f_it)
        for line in f_it:
            line = line.strip()
            # Skip empty lines or comments
            if len(line) == 0:
                continue
            elif line[0] == "#":
                continue
            if "lengthscale" in line:
                lengthscale = extract_float(f_it)
            elif "latticevectors" in line:
                cell[:] = extract_matrix(f_it).T
            elif "spinpol" in line:
                spinpol = True
            elif "spinpol_atomdens" in line:
                spinpol_atomdens = True
            elif "spinaxis" in line:
                spinaxis = extract_vector(f_it)
            elif "fullrel" in line:
                fullrel = True
            elif "lmax" in line:
                fullrel = extract_int(f_it)
            elif "mtradii" in line:
                mtradii = extract_int(f_it)
            elif "atoms" in line:
                Z, labels, positions, initial_moments, lattice_or_cartesian = (
                    extract_atoms(f_it)
                )
            elif "strainmatrix" in line:
                strainmatrix = extract_matrix(f_it).T
            elif "l_to_w" in line:
                l_to_w = extract_matrix(f_it)
            elif "offsets" in line:
                offsets = empty(positions.shape)
                for i in offsets.shape[0]:
                    offsets[i] = extract_vector(f_it)
    cell = (lengthscale * Bohr) * cell
    for pos, l_or_c in zip(positions, lattice_or_cartesian):
        pos[:] = vector_to_lattice(pos, l_or_c, cell)

    spinpol = spinpol or "up" in labels or "dn" in labels

    additional_info = {
        "spinpol": spinpol,
        "fullrel": fullrel,
        "lmax": lmax,
        "mtradii": mtradii,
        "labels": labels,
    }
    unique_labels = set(labels)
    label_to_tag = dict(zip(unique_labels, range(len(unique_labels))))

    if header is not None:
        additional_info["header"] = header
    if spinaxis is not None:
        spinaxis, l_or_c = spinaxis
        additional_info["spinaxis"] = vector_to_lattice(spinaxis, l_or_c, cell)
    if strainmatrix is not None:
        additional_info["strainmatrix"] = strainmatrix
    if l_to_w is not None:
        additional_info["l_to_w"] = l_to_w
    if offsets is not None:
        additional_info["offsets"] = offsets
    if spinpol_atomdens is not None:
        additional_info["spinpol_atomdens"] = True
    return Atoms(
        symbols=[Atom(z).symbol for z in Z],
        scaled_positions=positions,
        cell=cell,
        magmoms=initial_moments,
        tags=[label_to_tag[label] for label in labels],
        info=additional_info,
    )


def moment_string(m, label):
    if m > 0:
        return "up"
    if m < 0:
        return "dn"
    if label is not None:
        return label
    return " a"


def vector_string(vec):
    return "  ".join([f"{el:> .15f}" for el in vec])


def matrix_string(mat):
    return "\n".join([vector_string(row) for row in mat])


def write_symt(geometry: Atoms, prefix=None):
    """
    Create symt input file, symt.inp from the ASE Atoms object geometry.
    symt-specific information, such as fullrel, lmax, etc. in the info member
    of geometry.
    Arguments:
    ==========
    geometry: ASE Atoms - The structure to generate input for.
    """
    fname = "symt.inp"
    if prefix is not None:
        fname = f"{prefix}/symt.inp"

    cell = geometry.get_cell() / Bohr
    lengthscale = norm(cell[0])
    cell /= lengthscale

    header = geometry.info.get("header", geometry.get_chemical_formula())
    labels = geometry.info.get("labels", [None] * len(geometry))
    spinpol = (
        geometry.info.get("spinpol", False)
        or np_any(np_abs(geometry.get_initial_magnetic_moments()) > 0)
        or ("up" in labels)
        or ("dn" in labels)
    )
    with open(fname, "w") as f:

        f.write("\n".join(f"# {line}" for line in header.split("\n")) + "\n")
        f.write(
            f"# Generated: {str(str(datetime.datetime.now().strftime('%A %d $B %Y at %H:%M %Z')))}\n"
        )
        f.write("\n")
        f.write("lengthscale\n")
        f.write(f" {lengthscale:.15f}\n")
        f.write("\n")
        f.write("latticevectors\n")
        f.write(matrix_string(transpose(cell)) + "\n")
        f.write("\n")
        if spinpol:
            f.write("spinpol\n")
            f.write("\n")
        if geometry.info.get("fullrel", False):
            f.write("fullrel\n")
            f.write("\n")
        if "spinaxis" in geometry.info:
            f.write("spinaxis\n")
            f.write(vector_string(geometry.info["spinaxis"]) + "  l" + "\n")
            f.write("\n")
        if "mtradii" in geometry.info:
            f.write("mtradii\n")
            f.write(f" {geometry.info['mtradii']}\n")
            f.write("\n")
        if "lmax" in geometry.info:
            f.write("lmax\n")
            f.write(f" {geometry.info['lmax']}\n")
            f.write("\n")
        if "spinpol_atomdens" in geometry.info:
            f.write("spinpol_atomdens\n")
            f.write("\n")

        f.write("atoms\n")
        f.write(f" {len(geometry)}\n")
        [
            f.write(
                vector_string(pos)
                + f"  {Z} l {moment_string(m, label)} # {at} {moment_string(m, label)}\n"
            )
            for pos, Z, m, label, at in zip(
                geometry.get_scaled_positions(wrap=False),
                geometry.get_atomic_numbers(),
                geometry.get_initial_magnetic_moments(),
                labels,
                geometry.get_chemical_symbols(),
            )
        ]
        if "strainmatrix" in geometry.info:
            f.write("strainmatrix\n")
            f.write(matrix_string(geometry.info["strainmatrix"]) + "\n")
            f.write("\n")
        if "l_to_w" in geometry.info:
            f.write("l_to_w\n")
            f.write(matrix_string(geometry.info["l_to_w"]) + "\n")
            f.write("\n")
        if "offsets" in geometry.info:
            f.write("offsets\n")
            f.write(matrix_string(geometry.info["offsets"]) + "\n")
            f.write("\n")
