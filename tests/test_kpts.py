import os

import numpy as np
import pytest

from pyRSPthon.kpts.grid import get_kpts, reciprocal_ops, reduce_grid, generate_grid
from pyRSPthon.kpts.path import interpolate_path, spectrum_block, write_band_path
from pyRSPthon.kpts.spts import format_spts_number, read_spts, write_spts
from pyRSPthon.read.symcof import read_symcof
from conftest import CUB_CELL, CUB_MAP


def _canonical_orbits(kpts, weights, ops, denominator):
    def canonical(k):
        ints = np.round(np.mod(k, 1.0) * denominator).astype(int) % denominator
        return min(tuple((H @ ints) % denominator) for H in ops)

    return sorted((canonical(k), int(w)) for k, w in zip(kpts, weights))


@pytest.mark.parametrize(
    "reference,nk,shift,shift_den,denominator",
    [
        ("cub.k.12", (12, 12, 12), (1, 1, 1), (2, 2, 2), 24),
        ("cub.k.48_no", (48, 48, 48), (0, 0, 0), (1, 1, 1), 48),
    ],
)
def test_cub_equivalence(fixtures, reference, nk, shift, shift_den, denominator):
    """The decisive correctness test: reproduce cub's irreducible set."""
    symcof = os.path.join(fixtures, "symcof_fcc")
    ops = reciprocal_ops(read_symcof(symcof).rotations, CUB_CELL)
    kpts, weights = get_kpts(
        CUB_CELL,
        nk,
        shift_num=shift,
        shift_den=shift_den,
        map_matrix=CUB_MAP,
        symcof_path=symcof,
    )
    ref_k, ref_w, ref_iwsum, _ = read_spts(os.path.join(fixtures, reference))
    assert len(kpts) == len(ref_k)
    assert int(weights.sum()) == ref_iwsum
    ours = _canonical_orbits(kpts, weights, ops, denominator)
    ref = _canonical_orbits(ref_k, ref_w, ops, denominator)
    assert ours == ref


def test_subgroup_reduction_is_safe(fixtures):
    """Reducing with fewer ops gives more points but the same total weight."""
    symcof = os.path.join(fixtures, "symcof_fcc")
    ops = reciprocal_ops(read_symcof(symcof).rotations, CUB_CELL)
    points, D, _ = generate_grid((6, 6, 6))
    full_reps, full_w, _ = reduce_grid(points, D, ops)
    sub_reps, sub_w, _ = reduce_grid(points, D, ops[:8])
    assert full_w.sum() == sub_w.sum() == 6**3
    assert len(sub_reps) >= len(full_reps)


def test_downfolding_multiplicity():
    # commensurate grid: |det M| = 4 points fold onto each distinct point
    pts, _, mult = generate_grid((12, 12, 12), map_matrix=CUB_MAP)
    assert mult == 4 and len(pts) == 12**3 // 4
    # grids cub would reject still fold consistently (just less/not at all)
    pts, _, mult = generate_grid((3, 5, 7), map_matrix=CUB_MAP)
    assert mult * len(pts) == 3 * 5 * 7


def test_spts_round_trip(tmp_path):
    kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [-0.5, 0.25, 0.125]])
    weights = np.array([1, 8, 27])
    write_spts(tmp_path / "spts", kpts, weights, grid=(4, 4, 4))
    back_k, back_w, iwsum, labels = read_spts(tmp_path / "spts")
    np.testing.assert_allclose(back_k, kpts, atol=1e-15)
    assert list(back_w) == [1, 8, 27]
    assert iwsum == 36
    assert labels == {}


def test_spts_number_format_matches_cub(fixtures):
    # spot-check against a line of the real cub output
    assert format_spts_number(1 / 24) == " .041666666666667"
    assert format_spts_number(-1 / 24) == "-.041666666666667"
    with open(os.path.join(fixtures, "cub.k.12")) as f:
        for _ in range(5):
            next(f)
        line = next(f)
    assert line.startswith(f" {format_spts_number(1 / 24)}")


def test_band_path(tmp_path):
    nodes = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    kpts, node_at = interpolate_path(nodes, 4)
    # kpath convention: sum(nk+1) + 1 points
    assert len(kpts) == (4 + 1) * 2 + 1
    assert set(node_at.values()) == {0, 1, 2}
    block = spectrum_block(["G", "X", "M"], 4)
    assert "spectrum" in block and "Band Pband eV" in block

    labels, nks = write_band_path(
        tmp_path / "spts.band", np.eye(3), "G,X,M", 4,
        special_points={"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0]},
        zero_weight_point=True,
    )
    back_k, back_w, _, back_labels = read_spts(tmp_path / "spts.band")
    assert len(back_k) == 12  # 11 path points + 1 zero-weight point
    assert back_w[-1] == 0 and all(back_w[:-1] == 1)
    assert back_labels[0] == "G" and back_labels[10] == "M"
