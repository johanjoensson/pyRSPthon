from pyRSPthon.orbitals import OrbitalSelectionError
import subprocess
import sys

import numpy as np
import pytest

ENTRY_POINTS = [
    "pyRSPthon.cli.plot_dos",
    "pyRSPthon.cli.plot_pdos",
    "pyRSPthon.cli.plot_band",
    "pyRSPthon.cli.plot_pband",
    "pyRSPthon.cli.plot_dat",
    "pyRSPthon.cli.py_runs",
    "pyRSPthon.cli.rspt_structure",
    "pyRSPthon.cli.rspt_kpts",
    "pyRSPthon.cli.rspt_verify",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_plot_dos_output(dos_dir, tmp_path):
    out = tmp_path / "dos.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_dos",
            "-d",
            str(dos_dir),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_plot_band_output(band_dir, tmp_path):
    out = tmp_path / "band.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_band",
            "-d",
            str(band_dir),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def run_rspt_verify(directory, *flags):
    return subprocess.run(
        [sys.executable, "-m", "pyRSPthon.cli.rspt_verify", "-d", str(directory), *flags],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_rspt_verify_clean(fixtures, tmp_path):
    import shutil

    shutil.copy(f"{fixtures}/green_nio_ldau.inp", tmp_path / "green.inp")
    result = run_rspt_verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rspt_verify_broken(tmp_path):
    (tmp_path / "green.inp").write_text("Cluster\n1 IdX\n1 2 1 1 0 2.0\n2 2 0.5\n")
    result = run_rspt_verify(tmp_path)
    assert result.returncode == 0  # only warnings: the block is ignored
    assert "case-sensitive" in result.stdout
    (tmp_path / "green.inp").write_text("mixing\n5 0.15\n")
    result = run_rspt_verify(tmp_path)
    assert result.returncode == 1
    assert "unknown mix_method" in result.stdout


def test_rspt_verify_strict(tmp_path):
    (tmp_path / "green.inp").write_text("Cluster\n")
    result = run_rspt_verify(tmp_path, "--strict")
    assert result.returncode == 2


def test_plot_pband_composite_output(band_dir, tmp_path):
    out = tmp_path / "pband.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_pband",
            "Fe",
            "--composite",
            "--spin-sum",
            "-d",
            str(band_dir),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_parse_orbital_selection():
    from pyRSPthon.orbitals import parse_orbital_selection

    assert parse_orbital_selection(None, 3) == [[0], [1], [2]]
    assert parse_orbital_selection("0,2,4", 5) == [[0], [2], [4]]
    # a plain range plots each index separately
    assert parse_orbital_selection("1-3", 5) == [[1], [2], [3]]
    # '+' joins indices and ranges into one summed group
    assert parse_orbital_selection("0+1,2-4", 5) == [[0, 1], [2], [3], [4]]
    assert parse_orbital_selection("0-2+4", 5) == [[0, 1, 2, 4]]
    # a leading '+' sums a range into one group, same as 0+1+2+3+4
    assert parse_orbital_selection("+0-4", 5) == [[0, 1, 2, 3, 4]]
    # stray commas and whitespace are ignored
    assert parse_orbital_selection(" 0 , 1 ,", 5) == [[0], [1]]
    with pytest.raises(OrbitalSelectionError, match="out of range"):
        parse_orbital_selection("0,5", 5)
    with pytest.raises(OrbitalSelectionError, match="out of range"):
        parse_orbital_selection("-1", 5)
    for bad in ("5-2", "2-", "1-2-3", "a"):
        with pytest.raises(OrbitalSelectionError, match="Malformed"):
            parse_orbital_selection(bad, 5)


def run_plot_dat(directory, out, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_dat",
            "Fe",
            "hyb",
            "-d",
            str(directory),
            "-o",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_plot_dat_matrix_output(dos_dir, tmp_path):
    # indexmap present: 3D orbital matrix, block-diagonal fallback panels
    result = run_plot_dat(dos_dir, tmp_path / "hyb.png")
    assert result.returncode == 0, result.stderr
    result = run_plot_dat(dos_dir, tmp_path / "sel.png", "--orbitals", "0+1")
    assert result.returncode == 0, result.stderr


def test_plot_dat_columns_output(dos_dir, tmp_path):
    # no indexmap: 2D diagonal-only orbital columns, line plots
    import numpy as np

    w = np.linspace(-10, 10, 101)
    dat = np.column_stack([w, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    header = "  Energy      Orbitals"
    np.savetxt(dos_dir / "real-hyb-Fe.dat", dat, header=header, comments="#")
    np.savetxt(dos_dir / "imag-hyb-Fe.dat", dat * 0.5, header=header, comments="#")
    result = run_plot_dat(dos_dir, tmp_path / "hyb.png", "--orbitals", "0+1,2-3")
    assert result.returncode == 0, result.stderr


def test_find_cluster_shells(fixtures, tmp_path):
    import shutil

    from pyRSPthon.orbitals import find_cluster_shells

    shutil.copy(f"{fixtures}/green_nio_ldau.inp", tmp_path / "green.inp")
    # unlabeled clusters match RSPt's autogenerated t<T>.e<E>.l<L>... name
    shells, cf = find_cluster_shells("t2.e1.l2", str(tmp_path))
    assert shells == [{"type": 2, "l": 2, "basis": 0}]
    assert cf is False
    assert find_cluster_shells("nope", str(tmp_path)) == (None, False)
    assert find_cluster_shells("Fe", str(tmp_path / "missing")) == (None, False)


def test_plot_pdos_selection_output(dos_dir, tmp_path):
    # a labeled green.inp cluster gives the orbital columns real names
    (dos_dir / "green.inp").write_text(
        "cluster\n1 IdFe\n1 2 1 1 0 0.59 0.60 0.38\n2 2 1.0\n"
    )
    out = tmp_path / "pdos.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_pdos",
            "Fe",
            "-d",
            str(dos_dir),
            "-o",
            str(out),
            "--orbitals",
            "0-1,2+3+4",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_shell_labels():
    from pyRSPthon.orbitals import shell_labels

    d = {"type": 1, "l": 2, "basis": 0}
    p = {"type": 2, "l": 1, "basis": 0}
    # single shell: identical to orbital_labels, no prefix
    assert shell_labels([d], 5) == [f"m={m}" for m in range(-2, 3)]
    # multiple shells: concatenated in header order with a shell prefix
    labels = shell_labels([d, p], 8)
    assert labels[:5] == [f"t1 l2 m={m}" for m in range(-2, 3)]
    assert labels[5:] == [f"t2 l1 m={m}" for m in range(-1, 2)]
    # each shell holds its own spin-down block then spin-up block, shells
    # concatenated in header order (RSPt's per-shell layout): d(5)↓ d(5)↑
    # then p(3)↓ p(3)↑
    labels = shell_labels([d, p], 16, spin_split=True)
    assert len(labels) == 16
    assert labels[0] == "t1 l2 m=-2 ↓"
    assert labels[5] == "t1 l2 m=-2 ↑"
    assert labels[10] == "t2 l1 m=-1 ↓"
    assert labels[13] == "t2 l1 m=-1 ↑"
    # jj bases carry both j-manifolds already; spin_split is ignored
    jj = {"type": 1, "l": 3, "basis": 9}
    assert len(shell_labels([jj], 14, spin_split=True)) == 14
    # crystal-field states are not resolved per shell
    assert shell_labels([d, p], 4, cfflag=True) == [
        f"Cf state {i + 1}" for i in range(4)
    ]


def test_plot_pband_spin_sum_output(tmp_path):
    # A synthetic two-shell (d + p) cluster: pband binary + gpi + green.inp,
    # exercising plot_pband --spin-sum end to end.
    import numpy as np

    nk, ne, norb = 12, 8, 16
    ncol = 1 + norb
    A = np.zeros((nk, ne, ncol), dtype=np.float32)
    A[:, :, 0] = 1.0
    for o in range(norb):
        A[:, :, 1 + o] = (o + 1) / norb
    A.tofile(tmp_path / "pband-DP.data")
    gpi = (
        ' set ylabel "Energy (eV)"\n'
        f' set xtics ("{{/Symbol G}}" 0 ,"X" {nk - 1} )\n'
        f' set ytics ("-5.0" 0 ,"5.0" {ne - 1} )\n'
        f" ky(x) = (int(x)%{ne})\n"
        f" kx(x) = int(x)/{ne}\n"
        f" p [0:{nk - 1}] 'pband-DP.data' binary record={nk * ne}"
        ' format="%float%*16float" u (kx($0)):(ky($0)):1 w image\n'
    )
    (tmp_path / "pband-DP.gpi").write_text(gpi)
    (tmp_path / "green.inp").write_text(
        "cluster\n2 IdDP\n1 2 1 1 0 0.59 0.60 0.38\n"
        "2 1 1 1 0 0.59 0.60 0.38\n2 2 1.0\n"
    )
    out = tmp_path / "pband.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.plot_pband",
            "DP",
            "-d",
            str(tmp_path),
            "-o",
            str(out),
            "--spin-sum",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # total panel + 8 spin-summed orbital panels (one per d/p orbital)
    assert len(list(tmp_path.glob("pband-*.png"))) == 9


def test_spin_split_indices():
    from pyRSPthon.orbitals import spin_split_indices

    # SMO Valence: shells p, d(basis 3), p, p, p -> per-spin counts 3,5,3,3,3
    shells = [
        {"type": 1, "l": 1, "basis": 1},
        {"type": 2, "l": 2, "basis": 3},
        {"type": 3, "l": 1, "basis": 1},
        {"type": 3, "l": 1, "basis": 1},
        {"type": 3, "l": 1, "basis": 1},
    ]
    down_idx, up_idx = spin_split_indices(shells)
    assert down_idx == [0, 1, 2, 6, 7, 8, 9, 10, 16, 17, 18, 22, 23, 24, 28, 29, 30]
    assert up_idx == [3, 4, 5, 11, 12, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33]
    # every column is covered exactly once, 34 orbitals total
    assert sorted(down_idx + up_idx) == list(range(34))
    # jj bases are a single non-split block, and empty input is undefined
    assert spin_split_indices([{"type": 1, "l": 3, "basis": 9}]) is None
    assert spin_split_indices(None) is None


def test_orbital_labels_bases():
    from pyRSPthon.orbitals import orbital_labels

    # crystal-field bases keep a subset of the shell (lda_mlmsatomicqn)
    assert orbital_labels(3, basis_id=1, l=2) == [
        r"d$_{yz}$", r"d$_{xz}$", r"d$_{xy}$"
    ]
    assert orbital_labels(2, basis_id=2, l=2) == [r"d$_{z^2}$", r"d$_{x^2-y^2}$"]
    # basis 3 for an f shell: T1u then T2u
    assert orbital_labels(6, basis_id=3, l=3) == [
        r"f$_{x^3}$", r"f$_{y^3}$", r"f$_{z^3}$",
        r"f$_{x(y^2-z^2)}$", r"f$_{y(z^2-x^2)}$", r"f$_{z(x^2-y^2)}$",
    ]
    # unmatched bits fall through: basis 4-7 with a d shell = full eg+t2g
    assert orbital_labels(5, basis_id=4, l=2) == orbital_labels(5, basis_id=3, l=2)
    # only some (basis, l) combos reduce the shell (lda_id2mlmssize):
    # d is reduced only by basis 1 or 2, f by everything but 7
    assert [len(orbital_labels(5, basis_id=b, l=2)) for b in range(1, 8)] == [
        3, 2, 5, 5, 5, 5, 5
    ]
    assert [len(orbital_labels(7, basis_id=b, l=3)) for b in range(1, 8)] == [
        3, 3, 6, 1, 4, 4, 7
    ]
    # any basis 1-7 with a p shell gives py, px, pz
    assert orbital_labels(3, basis_id=6, l=1) == [r"p$_y$", r"p$_x$", r"p$_z$"]
    # basis 7 for an f shell: A2u, T1u, T2u
    assert orbital_labels(7, basis_id=7, l=3)[0] == r"f$_{xyz}$"
    # basis 8 is the identity basis (like 0), not JJ
    assert orbital_labels(5, basis_id=8, l=2) == [f"m={m}" for m in range(-2, 3)]
    # JJ bases are 9 (z), 10 (x), 11 (y): j=l-1/2 manifold then j=l+1/2
    labels = orbital_labels(10, basis_id=9, l=2)
    assert labels[0] == "j=3/2, m=-3/2"
    assert labels[4] == "j=5/2, m=-5/2"
    assert len(labels) == 10
    assert orbital_labels(10, basis_id=10, l=2)[0] == r"j=3/2, m$_x$=-3/2"
    assert orbital_labels(10, basis_id=11, l=2)[0] == r"j=3/2, m$_y$=-3/2"


def run_plot(module, directory, out, *extra):
    return subprocess.run(
        [sys.executable, "-m", f"pyRSPthon.cli.{module}", "-d", str(directory), "-o", str(out), *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )


def write_eigenvalues_file(directory, nk=8, nbands=5):
    lines = []
    for i, x in enumerate(np.linspace(0.0, 0.5, nk), start=1):
        lines += [f"# eigenvalues for k: {i:6d}", "# kx ky kz (* 2pi)", "# (e(i), i = 1, mx)"]
        lines.append(f" {x:17.9E} {0.0:17.9E} {0.0:17.9E}")
        ev = [0.3 + 0.1 * b + 0.05 * np.cos(4 * x) for b in range(nbands)]
        for j in range(0, nbands, 4):
            lines.append("".join(f" {e:17.9E}" for e in ev[j : j + 4]))
    (directory / "eigenvalues").write_text("\n".join(lines) + "\n")



def test_plot_band_bandfiles_with_labels(tmp_path):
    # -l applies to every source, one label per tick including the ends
    kd = np.linspace(0.0, 1.0, 20)
    np.savetxt(tmp_path / "bandfile_0", np.column_stack([kd, np.cos(3 * kd), np.sin(3 * kd)]))
    np.savetxt(tmp_path / "symline_1", [[0.5, -1000.0], [0.5, 1000.0]])
    result = run_plot("plot_band", tmp_path, tmp_path / "b.png", "-l", "G", "X", "M")
    assert result.returncode == 0, result.stderr
    result = run_plot("plot_band", tmp_path, tmp_path / "b.png", "-l", "G", "X")
    assert result.returncode == 1
    assert "2 tick labels given for 3 ticks" in result.stderr
    assert "Traceback" not in result.stderr


def test_plot_band_eigenvalues(tmp_path):
    write_eigenvalues_file(tmp_path)
    result = run_plot("plot_band", tmp_path, tmp_path / "e.png", "--efermi", "0.35", "--eV", "--nbands", "5")
    assert result.returncode == 0, result.stderr
    result = run_plot("plot_band", tmp_path, tmp_path / "e.png", "--nbands", "4")
    assert result.returncode == 1
    assert "not 4" in result.stderr


def test_plot_band_compare_four_dirs(band_dir, tmp_path):
    import shutil

    dirs = []
    for i in range(3):
        d = tmp_path / f"cmp{i}"
        d.mkdir()
        for f in ("band.data", "band.gpi"):
            shutil.copy(band_dir / f, d / f)
        dirs.append(str(d))
    result = run_plot("plot_band", band_dir, tmp_path / "cmp.png", "--compare", *dirs)
    assert result.returncode == 0, result.stderr


def test_plot_band_compare_mixed_kinds(band_dir, tmp_path):
    eig = tmp_path / "eig"
    eig.mkdir()
    write_eigenvalues_file(eig)
    result = run_plot("plot_band", band_dir, tmp_path / "mix.png", "--compare", str(eig))
    assert result.returncode == 1
    assert "same kind" in result.stderr


def test_plot_band_bad_nk_is_one_line_error(band_dir, tmp_path):
    result = run_plot("plot_band", band_dir, tmp_path / "bad.png", "-nk", "7", "-ne", "13")
    assert result.returncode == 1
    assert "not divisible" in result.stderr
    assert "Traceback" not in result.stderr


def test_plot_pband_spin_sum_refused_for_odd_columns(band_dir, tmp_path):
    # pband-Fe has 10 orbital columns; starting at column 2 leaves 9
    result = run_plot("plot_pband", band_dir, tmp_path / "p.png", "Fe", "--orb-start", "2", "--spin-sum")
    assert result.returncode == 1
    assert "cannot be split into spin pairs" in result.stderr


def test_plot_pband_spin_sum_orbital_groups(band_dir, tmp_path):
    result = run_plot(
        "plot_pband", band_dir, tmp_path / "p.png", "Fe", "--spin-sum", "--orbitals", "0+1,2-4", "--composite"
    )
    assert result.returncode == 0, result.stderr
    assert len(list(tmp_path.glob("p-*.png"))) == 2


def test_plot_band_dos_panel_and_ignored_flags(band_dir, dos_dir, tmp_path):
    # band_dir and dos_dir share tmp_path: band.data next to dos.dat
    result = run_plot("plot_band", band_dir, tmp_path / "bd.png", "--dos-panel", "--eV")
    assert result.returncode == 0, result.stderr
    assert "warning: --eV only apply to eigenvalue sources" in result.stderr


def test_plot_band_fatbands(tmp_path):
    kd = np.linspace(0.0, 1.0, 15)
    with open(tmp_path / "fatbands.t01.e1.l2", "w") as f:
        for band in range(3):
            for k in kd:
                f.write(f"{k:14.7f}{0.1 * band + 0.05 * np.cos(3 * k):14.7f}{abs(np.sin(k + band)):14.7f}\n")
            f.write("     \n")
    result = run_plot("plot_band", tmp_path, tmp_path / "fat.png", "--efermi", "0.1")
    assert result.returncode == 0, result.stderr


def test_apply_unit_conversion_returns_new_namedtuple(dos_dir):
    from pyRSPthon.cli._common import apply_unit_conversion
    from pyRSPthon.read import extract_dos, extract_pdos

    dos = extract_dos(prefix=str(dos_dir))
    conv = apply_unit_conversion(dos, 2.0, is_dos=True, data_type="dos")
    np.testing.assert_allclose(conv.w, dos.w * 2.0)
    np.testing.assert_allclose(conv.sum, dos.sum / 2.0)
    pdos = extract_pdos("Fe", prefix=str(dos_dir))
    conv = apply_unit_conversion(pdos, 2.0, is_dos=True, data_type="pdos")
    np.testing.assert_allclose(conv.orbitals, pdos.orbitals / 2.0)
    assert apply_unit_conversion(dos, 1.0) is dos


def test_plot_band_compare_different_line_sources(tmp_path):
    kd = np.linspace(0.0, 1.0, 8)
    bf = tmp_path / "bf"
    bf.mkdir()
    np.savetxt(bf / "bandfile_0", np.column_stack([kd, np.cos(kd)]))
    eig = tmp_path / "eig"
    eig.mkdir()
    write_eigenvalues_file(eig)
    result = run_plot("plot_band", bf, tmp_path / "c.png", "--compare", str(eig))
    assert result.returncode == 1
    assert "same source" in result.stderr


def test_plot_band_nk_zero_is_one_line_error(band_dir, tmp_path):
    result = run_plot("plot_band", band_dir, tmp_path / "z.png", "-nk", "0", "-ne", "30")
    assert result.returncode == 1
    assert "must be positive" in result.stderr
    assert "Traceback" not in result.stderr


def test_plot_pband_irrep_spin_sum_refused(tmp_path):
    nk, ne = 4, 3
    np.ones(nk * ne * 5, dtype=np.float32).tofile(tmp_path / "pband-Fe-irr0312.data")
    result = run_plot("plot_pband", tmp_path, tmp_path / "i.png", "Fe-irr0312", "-nk", "4", "-ne", "3", "--spin-sum")
    assert result.returncode == 1
    assert "irrep block 0312" in result.stderr
    result = run_plot("plot_pband", tmp_path, tmp_path / "i.png", "Fe-irr0312", "-nk", "4", "-ne", "3")
    assert result.returncode == 0, result.stderr


def test_orbitals_count_mismatch_is_one_line_error(dos_dir, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "pyRSPthon.cli.plot_pdos", "Fe", "-d", str(dos_dir),
         "-o", str(tmp_path / "x.png"), "--orbitals", "0", "1"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "one per cluster" in result.stderr
    assert "Traceback" not in result.stderr
