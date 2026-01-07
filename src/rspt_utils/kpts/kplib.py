from .read import read_symt
from .write import write_spts
from kpLib.interface import get_kpoints
import ase


def get_kpts(atoms, min_distance)
    kpts = get_kpoints(
        atoms.get_cell()[:],
        atoms.get_scaled_positions(),  # Note: the atomic coordinates are in fractional format, not Cartesian.
        atoms.get_atomic_numbers(),
        min_distance=min_distance,
        include_gamma="auto",
    )
