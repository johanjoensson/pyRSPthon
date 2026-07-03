"""
K-point mesh and band-path generation for RSPt.

HARD INVARIANT: meshes are reduced only with the symmetry operations RSPt
knows (parsed from symcof, plus time reversal only on explicit request).
See pyRSPthon.kpts.grid for details.
"""

from .grid import get_kpts, generate_grid, reduce_grid, reciprocal_ops, fold_to_bz
from .spts import write_spts, write_spts_band, read_spts
from .path import write_band_path, spectrum_block
