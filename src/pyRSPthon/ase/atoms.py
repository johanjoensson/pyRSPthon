"""
Read and write RSPt structure input files (symt.inp) as ASE Atoms objects.

Both the new block format (lengthscale/latticevectors/atoms/... keywords) and
the legacy positional format (Bravais matrix, spin axis, natoms, atom lines,
trailing 3x3 matrix) are supported by read_symt. write_symt writes the new
block format.

Conventions (RSPt manual):
- In symt.inp the lattice vectors are given in COLUMN order: each column of
  the 3x3 block is one lattice vector.
- Atom/vector coordinate tags: 'l' = lattice, 'c'/'a' = cartesian (in units
  of the length scale), 'w' = Wyckoff (needs the l_to_w matrix).
- Identity labels: single letters a-z group/split atoms for symmetry
  purposes; the special labels 'up'/'dn' seed spin-polarized densities.
"""

from ase import Atoms, Atom
from ase.units import Bohr
from numpy.linalg import norm, solve
from numpy import array, empty, zeros, transpose, abs as np_abs, any as np_any, eye
import datetime
import warnings

_BLOCK_KEYWORDS = {
    "lengthscale",
    "latticevectors",
    "spinaxis",
    "atoms",
    "strainmatrix",
    "l_to_w",
    "offsets",
    "mtradii",
    "spinpol",
    "fullrel",
    "lmax",
    "spinpol_atomdens",
}


def extract_header(file_it):
    """
    Read initial comment lines ('#'-prefixed) as a header.
    Returns (header or None, first non-header line or None).
    """
    lines = []
    for line in file_it:
        stripped = line.strip()
        if stripped and stripped[0] == "#":
            lines.append(stripped[1:].strip())
            continue
        return ("\n".join(lines) if lines else None), line
    return ("\n".join(lines) if lines else None), None


def extract_float(file_it):
    line = next(file_it)
    return float(line.strip())


def extract_matrix(file_it):
    rows = []
    for _ in range(3):
        line = next(file_it)
        rows.append([float(el) for el in line.strip().split()[:3]])
    return array(rows)


def extract_int(file_it):
    line = next(file_it)
    return int(line.strip())


def extract_vector(file_it):
    line = next(file_it)
    tmp = line.strip().split()
    vector = array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    lattice_or_cartesian = tmp[3] if len(tmp) > 3 else "l"
    return vector, lattice_or_cartesian


def parse_atom_line(line):
    tmp = line.strip().split()
    if len(tmp) < 6:
        raise ValueError(f"Malformed atom line in symt.inp: {line!r}")
    position = array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    Z = int(tmp[3])
    lattice_or_cartesian = tmp[4]
    label = tmp[5]
    return position, Z, lattice_or_cartesian, label


def extract_atoms(file_it):
    line = next(file_it)
    n_at = int(line.strip())

    Z = [0] * n_at
    labels = ["a"] * n_at
    positions = empty((n_at, 3))
    magmoms = zeros(n_at)
    lattice_or_cartesian = ["l"] * n_at

    for i in range(n_at):
        positions[i], Z[i], lattice_or_cartesian[i], labels[i] = parse_atom_line(
            next(file_it)
        )
        if labels[i] == "up":
            magmoms[i] = 1
        elif labels[i] == "dn":
            magmoms[i] = -1
    return Z, labels, positions, magmoms, lattice_or_cartesian


def vector_to_lattice(vec, l_or_cart, cell):
    """
    Convert a vector to lattice (fractional) coordinates.
    cell rows are the lattice vectors, in the same length unit as vec
    (for symt.inp: units of the length scale).
    """
    if l_or_cart == "l":
        return vec
    if l_or_cart == "w":
        raise ValueError(
            "Wyckoff ('w') coordinates require the l_to_w matrix; not supported yet."
        )
    return solve(cell.T, vec)


def _read_new_format(f_it, first_line):
    data = {}
    line = first_line
    while line is not None:
        stripped = line.strip()
        if stripped and stripped[0] != "#":
            keyword = stripped.split()[0]
            if keyword == "lengthscale":
                data["lengthscale"] = extract_float(f_it)
            elif keyword == "latticevectors":
                # columns of the file block are the lattice vectors;
                # store rows-as-vectors
                data["cell"] = extract_matrix(f_it).T
            elif keyword == "spinpol_atomdens":
                data["spinpol_atomdens"] = True
            elif keyword == "spinpol":
                data["spinpol"] = True
            elif keyword == "spinaxis":
                data["spinaxis"] = extract_vector(f_it)
            elif keyword == "fullrel":
                data["fullrel"] = True
            elif keyword == "lmax":
                data["lmax"] = extract_int(f_it)
            elif keyword == "mtradii":
                data["mtradii"] = extract_int(f_it)
            elif keyword == "atoms":
                (
                    data["Z"],
                    data["labels"],
                    data["positions"],
                    data["magmoms"],
                    data["lattice_or_cartesian"],
                ) = extract_atoms(f_it)
            elif keyword == "strainmatrix":
                data["strainmatrix"] = extract_matrix(f_it).T
            elif keyword == "l_to_w":
                data["l_to_w"] = extract_matrix(f_it)
            elif keyword == "offsets":
                if "positions" not in data:
                    raise ValueError(
                        "The offsets block must come after the atoms block in symt.inp"
                    )
                offsets = empty(data["positions"].shape)
                for i in range(offsets.shape[0]):
                    offsets[i], _ = extract_vector(f_it)
                data["offsets"] = offsets
            else:
                warnings.warn(f"Unknown symt.inp keyword {keyword!r}, skipping")
        line = next(f_it, None)
    return data


def _read_legacy_format(f_it, first_line):
    """
    Legacy positional symt.inp: Bravais matrix (columns = lattice vectors),
    spin axis + tag, number of atoms, atom lines, optional trailing 3x3
    strain matrix. '#' lines are comments.
    """

    def data_lines():
        line = first_line
        while line is not None:
            stripped = line.strip()
            if stripped and stripped[0] != "#":
                yield stripped
            line = next(f_it, None)

    lines = data_lines()
    data = {}
    cell = array([[float(el) for el in next(lines).split()[:3]] for _ in range(3)])
    data["cell"] = cell.T
    tmp = next(lines).split()
    spinaxis = array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    axis_tag = tmp[3] if len(tmp) > 3 else "l"
    if norm(spinaxis) > 0:
        data["spinaxis"] = (spinaxis, axis_tag)
        # Legacy: a nonzero spin axis implies spin polarized + relativistic
        data["spinpol"] = True
        data["fullrel"] = True
    n_at = int(next(lines).split()[0])
    Z = [0] * n_at
    labels = ["a"] * n_at
    positions = empty((n_at, 3))
    magmoms = zeros(n_at)
    lattice_or_cartesian = ["l"] * n_at
    for i in range(n_at):
        positions[i], Z[i], lattice_or_cartesian[i], labels[i] = parse_atom_line(
            next(lines)
        )
        if labels[i] == "up":
            magmoms[i] = 1
        elif labels[i] == "dn":
            magmoms[i] = -1
    data["Z"] = Z
    data["labels"] = labels
    data["positions"] = positions
    data["magmoms"] = magmoms
    data["lattice_or_cartesian"] = lattice_or_cartesian
    try:
        strain = array(
            [[float(el) for el in next(lines).split()[:3]] for _ in range(3)]
        )
        if np_any(np_abs(strain - eye(3)) > 1e-12):
            data["strainmatrix"] = strain
    except (StopIteration, ValueError, IndexError):
        pass
    return data


def read_symt(fname: str, lengthscale: float = 1.0):
    """
    Read an RSPt symt.inp file (new block format or legacy positional format)
    and return an ASE Atoms object. Additional RSPt info (mtradii, lmax, ...)
    is stored in the info member of the Atoms object.
    Arguments
    =========
    fname: string - file to read
    lengthscale: float - length scale in Bohr, used if the file has no
                         lengthscale block (the legacy format never has one)
    Returns
    =======
    Atoms - Structure read from file, with additional information in info.
    """
    with open(fname, "r") as f:
        f_it = iter(f)
        header, line = extract_header(f_it)
        # Skip blank lines between the header and the first data line
        while line is not None and not line.strip():
            line = next(f_it, None)
        if line is None:
            raise ValueError(f"{fname} contains no structure data")
        first_token = line.strip().split()[0]
        if first_token in _BLOCK_KEYWORDS:
            data = _read_new_format(f_it, line)
        else:
            data = _read_legacy_format(f_it, line)

    if "positions" not in data:
        raise ValueError(f"{fname} contains no atoms block")

    lengthscale = data.get("lengthscale", lengthscale)
    cell = data["cell"]  # dimensionless, rows are lattice vectors
    positions = data["positions"]
    # Convert cartesian-tagged entries (given in units of the length scale)
    # to fractional coordinates using the dimensionless cell.
    for pos, l_or_c in zip(positions, data["lattice_or_cartesian"]):
        pos[:] = vector_to_lattice(pos, l_or_c, cell)
    if "offsets" in data:
        positions = positions + data["offsets"]

    cell_ang = (lengthscale * Bohr) * cell

    labels = data["labels"]
    spinpol = data.get("spinpol", False) or "up" in labels or "dn" in labels

    additional_info = {
        "spinpol": spinpol,
        "fullrel": data.get("fullrel", False),
        "lmax": data.get("lmax", 8),
        "mtradii": data.get("mtradii", 0),
        "labels": labels,
    }
    # Sort for deterministic tag numbering
    unique_labels = sorted(set(labels))
    label_to_tag = {label: tag for tag, label in enumerate(unique_labels)}

    if header is not None:
        additional_info["header"] = header
    if "spinaxis" in data:
        spinaxis, l_or_c = data["spinaxis"]
        additional_info["spinaxis"] = vector_to_lattice(spinaxis, l_or_c, cell)
    if "strainmatrix" in data:
        additional_info["strainmatrix"] = data["strainmatrix"]
    if "l_to_w" in data:
        additional_info["l_to_w"] = data["l_to_w"]
    if data.get("spinpol_atomdens", False):
        additional_info["spinpol_atomdens"] = True
    return Atoms(
        symbols=[Atom(z).symbol for z in data["Z"]],
        scaled_positions=positions,
        cell=cell_ang,
        pbc=True,
        magmoms=data["magmoms"],
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
    return "a"


def vector_string(vec):
    return "  ".join([f"{el:> .15f}" for el in vec])


def matrix_string(mat):
    return "\n".join([vector_string(row) for row in mat])


def symmetrize_atoms(atoms: Atoms, symprec: float = 1e-5, angle_tol: float = -1.0):
    """
    Return a symmetry-refined copy of atoms using spglib.refine_cell.
    Note: spglib works on the crystal structure only; initial magnetic
    moments, tags and info are not carried over (the refined cell may hold a
    different number of atoms).
    """
    import spglib

    spglib_cell = (
        atoms.cell[:],
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    refined = spglib.refine_cell(spglib_cell, symprec=symprec, angle_tolerance=angle_tol)
    if refined is None:
        raise RuntimeError("spglib.refine_cell failed to refine the structure")
    lattice, scaled_positions, numbers = refined
    return Atoms(
        cell=lattice,
        scaled_positions=scaled_positions,
        numbers=numbers,
        pbc=True,
    )


def write_symt(geometry: Atoms, prefix=None):
    """
    Create symt input file, symt.inp, from the ASE Atoms object geometry.
    symt-specific information, such as fullrel, lmax, etc. is taken from the
    info member of geometry.
    Arguments:
    ==========
    geometry: ASE Atoms - The structure to generate input for.
    prefix: str | None  - Directory to write symt.inp in.
    """
    fname = "symt.inp"
    if prefix is not None:
        fname = f"{prefix}/symt.inp"

    magmoms = geometry.get_initial_magnetic_moments()
    if magmoms.ndim > 1:
        raise ValueError(
            "RSPt only handles collinear magnetism; initial magnetic moments "
            "must be scalar."
        )

    cell = geometry.get_cell() / Bohr
    lengthscale = norm(cell[0])
    cell = cell / lengthscale

    scaled_positions = geometry.get_scaled_positions(wrap=False)
    if np_any(scaled_positions < 0) or np_any(scaled_positions >= 1):
        warnings.warn(
            "Some scaled positions lie outside [0, 1); RSPt may not fold them."
        )

    header = geometry.info.get("header", geometry.get_chemical_formula())
    labels = geometry.info.get("labels", [None] * len(geometry))
    spinpol = (
        geometry.info.get("spinpol", False)
        or np_any(np_abs(magmoms) > 0)
        or ("up" in labels)
        or ("dn" in labels)
    )
    fullrel = geometry.info.get("fullrel", False)
    spinaxis = geometry.info.get("spinaxis")
    # fullrel + spinpol needs a quantization axis; default to lattice z
    if fullrel and spinpol and spinaxis is None:
        spinaxis = array([0.0, 0.0, 1.0])

    with open(fname, "w") as f:
        f.write("\n".join(f"# {line}" for line in header.split("\n")) + "\n")
        f.write(
            f"# Generated: {datetime.datetime.now().strftime('%A %d %B %Y at %H:%M %Z')}\n"
        )
        f.write("\n")
        f.write("lengthscale\n")
        f.write(f" {lengthscale:.15f}\n")
        f.write("\n")
        f.write("latticevectors\n")
        # columns of the written block are the lattice vectors
        f.write(matrix_string(transpose(cell)) + "\n")
        f.write("\n")
        if spinpol:
            f.write("spinpol\n")
            f.write("\n")
        if fullrel:
            f.write("fullrel\n")
            f.write("\n")
        if spinaxis is not None:
            f.write("spinaxis\n")
            f.write(vector_string(spinaxis) + "  l" + "\n")
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
        for pos, Z, m, label, at in zip(
            scaled_positions,
            geometry.get_atomic_numbers(),
            magmoms,
            labels,
            geometry.get_chemical_symbols(),
        ):
            tag = moment_string(m, label)
            f.write(vector_string(pos) + f"  {Z} l {tag} # {at} {tag}\n")
        if "strainmatrix" in geometry.info:
            f.write("\n")
            f.write("strainmatrix\n")
            f.write(matrix_string(geometry.info["strainmatrix"]) + "\n")
        if "l_to_w" in geometry.info:
            f.write("\n")
            f.write("l_to_w\n")
            f.write(matrix_string(geometry.info["l_to_w"]) + "\n")
