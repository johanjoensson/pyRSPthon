import numpy as np
import pytest

from pyRSPthon.orbitals import (
    OrbitalSelectionError,
    projected_orbital_count,
    resolve_layout,
    select_orbitals,
    spin_sum,
)

D = {"type": 1, "l": 2, "basis": 0}  # 5 orbitals per spin
P = {"type": 2, "l": 1, "basis": 0}  # 3 orbitals per spin
F_JJ = {"type": 1, "l": 3, "basis": 9}  # one 14-orbital j block


def test_projected_orbital_count():
    assert projected_orbital_count([D, P]) == 16
    assert projected_orbital_count([F_JJ]) == 14
    # crystal-field basis 1 keeps only the t2g of a d shell
    assert projected_orbital_count([{"type": 1, "l": 2, "basis": 1}]) == 6
    assert projected_orbital_count(None) is None


def test_resolve_layout_per_shell_spin_pairs():
    layout = resolve_layout([D, P], cfflag=False, fullrel=False, norb=16)
    down, up = layout.spin_pairs
    # per-shell layout: d↓=0..4, d↑=5..9, p↓=10..12, p↑=13..15
    assert down == (0, 1, 2, 3, 4, 10, 11, 12)
    assert up == (5, 6, 7, 8, 9, 13, 14, 15)
    assert layout.labels[0] == "t1 l2 m=-2 ↓"
    assert layout.labels[13] == "t2 l1 m=-1 ↑"


def test_resolve_layout_without_shells_splits_halves():
    layout = resolve_layout(None, cfflag=False, fullrel=None, norb=10)
    assert layout.spin_pairs == (tuple(range(5)), tuple(range(5, 10)))
    assert layout.labels[0] == r"d$_{xy}$ ↓"
    assert len(layout.labels) == 10


def test_spin_sum_pairs_columns():
    # orbital column i carries the value i
    weights = np.tile(np.arange(16, dtype=float), (3, 4, 1))
    layout = resolve_layout([D, P], cfflag=False, fullrel=False, norb=16)
    summed, labels = spin_sum(weights, layout)
    np.testing.assert_array_equal(summed[0, 0], [5, 7, 9, 11, 13, 23, 25, 27])
    assert labels[0] == "t1 l2 m=-2"
    assert len(labels) == 8
    # the input is not modified
    np.testing.assert_array_equal(weights[0, 0], np.arange(16))


@pytest.mark.parametrize(
    "shells, cfflag, fullrel, norb, reason",
    [
        ([F_JJ], False, True, 14, "JJ basis"),
        ([D], True, True, 10, "mixed spinors"),
        (None, False, False, 9, "cannot be split"),
    ],
)
def test_spin_sum_refused(shells, cfflag, fullrel, norb, reason):
    layout = resolve_layout(shells, cfflag, fullrel, norb)
    assert layout.spin_pairs is None
    with pytest.raises(OrbitalSelectionError, match=reason):
        spin_sum(np.zeros((2, 2, norb)), layout)


def test_cf_flag_without_fullrel_is_spin_split():
    # Cf states in a scalar-relativistic run keep their spin blocks
    layout = resolve_layout([D], cfflag=True, fullrel=False, norb=10)
    assert layout.spin_pairs is not None
    assert layout.labels[0] == "Cf state 1 ↓"


def test_select_orbitals_groups_and_overrides():
    weights = np.arange(2 * 5, dtype=float).reshape(2, 5)
    labels = ["a", "b", "c", "d", "e"]
    sel, sel_labels = select_orbitals(weights, labels, "0,1+2,3-4")
    np.testing.assert_array_equal(sel[1], [5, 13, 8, 9])
    assert sel_labels == ["a", "b + c", "d", "e"]
    _, sel_labels = select_orbitals(weights, labels, "0,1", ["first"])
    assert sel_labels == ["first", "b"]
    with pytest.raises(OrbitalSelectionError, match="out of range"):
        select_orbitals(weights, labels, "5")


def test_cf_flag_pairs_cluster_halves_across_shells():
    # a Cf cluster is one unit (all down, then all up), not per-shell blocks
    layout = resolve_layout([D, P], cfflag=True, fullrel=False, norb=16)
    assert layout.spin_pairs == (tuple(range(8)), tuple(range(8, 16)))
    assert layout.labels[0] == "Cf state 1 ↓"
    assert layout.labels[8] == "Cf state 1 ↑"
    weights = np.arange(16, dtype=float)[None, None, :]
    summed, labels = spin_sum(weights, layout)
    np.testing.assert_array_equal(summed[0, 0], np.arange(8) * 2 + 8)
    assert labels[0] == "Cf state 1"


def test_composite_uses_shared_scale():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    from pyRSPthon.plot.bands import plot_composite

    ne, nk = 4, 3
    weights = np.zeros((ne, nk, 2))
    weights[0, :, 0] = 1.0   # strong orbital in one energy row
    weights[2, :, 1] = 0.05  # weak orbital in another
    model = SimpleNamespace(kpath=SimpleNamespace(kdist=np.arange(nk, dtype=float)), energies=np.arange(ne, dtype=float))
    fig, ax = plt.subplots()
    plot_composite(ax, model, weights)
    img = np.asarray(ax.images[0].get_array())[::-1]
    # the weak orbital stays a faint tint instead of being scaled to full colour
    assert img[2, 0].min() > 0.9
    assert img[0, 0].min() < 0.5
    plt.close(fig)


def test_irrep_block_is_not_spin_summed():
    layout = resolve_layout([D], cfflag=True, fullrel=False, norb=4, irrep="0312")
    assert layout.spin_pairs is None
    assert layout.labels == tuple(f"irr0312 state {i}" for i in range(1, 5))
    with pytest.raises(OrbitalSelectionError, match="irrep block 0312"):
        spin_sum(np.zeros((2, 2, 4)), layout)
