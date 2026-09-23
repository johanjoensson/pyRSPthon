import dataclasses
import struct

import numpy as np
import pytest

from pyRSPthon.read._band_binary import peek_band_header
from pyRSPthon.read.bands import (
    BandReadError,
    parse_band_gpi,
    read_bandfiles,
    read_bands,
    read_eigenvalues,
    read_fatbands,
    read_spectral_bands,
    to_unit,
    with_labels,
    with_reference,
)
from pyRSPthon.units import RY_TO_EV


def build_band_header(shape, emin, emax, nfermi, scale, shells=(), cfflag=False, ncorr=None):
    """
    Python mirror of build_band_header in RSPt's green_spectrum.F90: slot 13
    packs (cfflag, nshells, ncorr, 0). ncorr defaults to all shells.
    """
    nints = max(16, 13 + len(shells))
    hdr = bytearray(4 * nints)
    hdr[0:4] = b"RSPt"
    struct.pack_into("<ii", hdr, 4, 1, 4 * nints)
    struct.pack_into(f"<i{len(shape)}i", hdr, 12, len(shape), *shape)
    struct.pack_into("<ffif", hdr, 32, emin, emax, nfermi, scale)
    if shells:
        ncorr = len(shells) if ncorr is None else ncorr
        struct.pack_into("4B", hdr, 48, int(cfflag), len(shells), ncorr, 0)
        for n, (t, l, basis) in enumerate(shells):
            struct.pack_into("4B", hdr, 52 + 4 * n, t, l, basis, 0)
    return bytes(hdr)


# ---------------------------------------------------------------------------
# Band_header


def test_peek_band_header_unprojected(tmp_path):
    fname = tmp_path / "band.data"
    fname.write_bytes(build_band_header((11, 30, 40), -5.0, 5.0, 15, 13.6057))
    meta = peek_band_header(fname)
    assert meta["n_data"] == 11
    assert meta["ne"] == 30
    assert meta["nk"] == 40
    assert meta["fermi_index"] == 14
    np.testing.assert_allclose(meta["emin"], -5.0)
    np.testing.assert_allclose(meta["emax"], 5.0)
    np.testing.assert_allclose(meta["scale"], 13.6057, rtol=1e-6)
    assert "shells" not in meta


def test_peek_band_header_shells(tmp_path):
    fname = tmp_path / "pband-Fe.data"
    shells = [(1, 2, 3), (2, 1, 0), (1, 3, 8), (2, 2, 0)]
    # 4 shells, 2 of them correlated: the header grows past the 16-int
    # minimum (to 17 ints)
    fname.write_bytes(
        build_band_header((13, 30, 40), -0.4, 0.4, -15, 1.0, shells, ncorr=2)
    )
    meta = peek_band_header(fname)
    assert meta["cfflag"] is False
    assert meta["ncorr"] == 2
    assert meta["shells"] == [
        {"type": t, "l": l, "basis": b} for t, l, b in shells
    ]
    assert "fermi_index" not in meta  # negative nfermi means unknown
    np.testing.assert_allclose(meta["scale"], 1.0)


def test_peek_band_header_cf_flag(tmp_path):
    # the Cf flag makes RSPt treat the whole cluster as one correlated set
    fname = tmp_path / "pband-Ce.data"
    fname.write_bytes(
        build_band_header((15, 8, 6), -1, 1, 4, 1.0, [(1, 3, 0), (2, 2, 0)], cfflag=True, ncorr=1)
    )
    meta = peek_band_header(fname)
    assert meta["cfflag"] is True
    assert meta["ncorr"] == 1
    assert len(meta["shells"]) == 2


def test_peek_band_header_old_ncorr_byte(tmp_path):
    # headers written before RSPt stored ncorr hold 0 there: unknown
    fname = tmp_path / "pband-Fe.data"
    fname.write_bytes(build_band_header((11, 8, 6), -1, 1, 4, 1.0, [(1, 2, 0)], ncorr=0))
    meta = peek_band_header(fname)
    assert meta["ncorr"] is None
    assert meta["shells"] == [{"type": 1, "l": 2, "basis": 0}]


def test_peek_band_header_absent(tmp_path):
    fname = tmp_path / "band.data"
    fname.write_bytes(np.zeros(120, dtype=np.float32).tobytes())
    assert peek_band_header(fname) is None
    assert peek_band_header(tmp_path / "missing.data") is None
    rnd = tmp_path / "random.data"
    rnd.write_bytes(np.random.default_rng(1).standard_normal(256).astype("f").tobytes())
    assert peek_band_header(rnd) is None


def test_peek_band_header_truncated(tmp_path):
    fname = tmp_path / "band.data"
    fname.write_bytes(build_band_header((3, 4, 5), -1, 1, 2, 1.0)[:40])
    assert peek_band_header(fname) is None
    small = bytearray(build_band_header((3, 4, 5), -1, 1, 2, 1.0))
    struct.pack_into("<i", small, 8, 32)  # header size below the 64-byte minimum
    fname.write_bytes(bytes(small))
    assert peek_band_header(fname) is None


def test_read_spectral_bands_from_header(tmp_path):
    nk, ne, ncol = 12, 8, 3
    header = build_band_header((ncol, ne, nk), -2.0, 2.0, 4, 13.6057)
    payload = np.arange(nk * ne * ncol, dtype=np.float32)
    (tmp_path / "band.data").write_bytes(header + payload.tobytes())
    # no band.gpi and no green.inp: all metadata must come from the header
    bs = read_spectral_bands(prefix=str(tmp_path))
    assert bs.total.shape == (ne, nk)
    assert bs.extra.shape == (ne, nk, ncol - 1)
    assert bs.energy_unit == "eV"
    assert bs.fermi_index == 3
    np.testing.assert_allclose(bs.energies, np.linspace(-2.0, 2.0, ne), atol=1e-6)
    # payload is C order (nk, ne, ncol): energy is the fast axis
    assert bs.total[1, 0] == ncol
    assert bs.total[0, 1] == ne * ncol


def test_headered_equals_headerless(band_dir, tmp_path):
    ref = read_spectral_bands(prefix=str(band_dir))
    ne, nk = ref.total.shape
    raw = (band_dir / "band.data").read_bytes()
    out = tmp_path / "hdr"
    out.mkdir()
    header = build_band_header((11, ne, nk), ref.energies[0], ref.energies[-1], ne // 2 + 1, 13.6057)
    (out / "band.data").write_bytes(header + raw)
    got = read_spectral_bands(prefix=str(out))
    np.testing.assert_array_equal(got.total, ref.total)
    np.testing.assert_array_equal(got.extra, ref.extra)
    np.testing.assert_allclose(got.energies, ref.energies, atol=1e-5)


def test_read_spectral_bands_4d_header(tmp_path):
    (tmp_path / "band.data").write_bytes(build_band_header((3, 4, 5, 6), -1, 1, 2, 1.0))
    with pytest.raises(BandReadError, match="surface"):
        read_spectral_bands(prefix=str(tmp_path))


def test_pband_orbital_start_not_guessed_from_green_inp(tmp_path):
    # green.inp gives only the correlated shells, which undercounts the
    # projected sets, so it must not move orb_start (default column 1)
    nk, ne, ncol = 6, 5, 17
    np.ones(nk * ne * ncol, dtype=np.float32).tofile(tmp_path / "pband-Fe.data")
    (tmp_path / "green.inp").write_text("cluster\n1 IdFe\n1 2 1 1 0 0.59 0.60 0.38\n2 2 1.0\n")
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Fe", nk=nk, ne=ne)
    assert bs.projections.weights.shape == (ne, nk, 16)
    assert bs.projections.shells == ({"type": 1, "l": 2, "basis": 0},)


def test_pband_orbitals_from_header_shells(tmp_path):
    # no gpi: the orbital columns are the last 2*5 columns of a d shell,
    # after the total and six moment columns
    nk, ne, ncol = 6, 5, 17
    header = build_band_header((ncol, ne, nk), -1.0, 1.0, 3, 1.0, [(2, 2, 0)])
    payload = np.ones(nk * ne * ncol, dtype=np.float32)
    (tmp_path / "pband-Gd.data").write_bytes(header + payload.tobytes())
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Gd")
    assert bs.projections.weights.shape == (ne, nk, 10)
    assert bs.extra.shape == (ne, nk, 6)
    assert bs.projections.shells == ({"type": 2, "l": 2, "basis": 0},)
    assert bs.projections.irrep is None


# ---------------------------------------------------------------------------
# band.gpi


def test_parse_band_gpi(band_dir):
    meta = parse_band_gpi(band_dir / "band.gpi")
    assert meta["nk"] == 40
    assert meta["ne"] == 30
    assert meta["energy_unit"] == "eV"
    assert meta["tick_labels"] == ["Γ", "X", "L"]
    assert meta["ticks"] == [0, 20, 39]
    np.testing.assert_allclose(meta["energies"][0], -5.0, atol=0.2)
    np.testing.assert_allclose(meta["energies"][-1], 5.0, atol=0.2)


def test_parse_band_gpi_fractional_ticks_and_orbital_loop(tmp_path):
    gpi = (
        ' set xtics ("M" 0 ,"X" 19.5 ,"K" 39 )\n'
        ' set ytics ("E_F" 15 ,"-1.00" 0 ," 1.00" 30 )\n'
        " ky(x) = (int(x)%   31)\n"
        " p [0: 39] 'pband-A.data' binary record=  1240 format=\"%17float\" u 1\n"
        " do for [ie=     8:    17] {\n"
        "    set output sprintf('pband-A-proj-%02d.eps',ie-  7)\n"
        "    p [0: 39] 'pband-A.data' binary record=  1240 format=\"%17float\" u ie\n"
        " }\n"
    )
    (tmp_path / "pband-A.gpi").write_text(gpi)
    meta = parse_band_gpi(tmp_path / "pband-A.gpi")
    assert meta["ticks"] == [0.0, 19.5, 39.0]
    assert meta["nk"] == 40
    # the non-numeric "E_F" ytic is skipped, the mesh still comes out right
    np.testing.assert_allclose(meta["energies"][[0, 15, 30]], [-1.0, 0.0, 1.0], atol=1e-9)
    assert (meta["orb_start"], meta["orb_end"]) == (7, 17)


def test_read_spectral_bands_auto_metadata(band_dir):
    bs = read_spectral_bands(prefix=str(band_dir))
    assert bs.kind == "spectral"
    assert bs.total.shape == (30, 40)
    assert bs.extra.shape == (30, 40, 10)
    assert bs.energy_unit == "eV"
    assert bs.kpath.tick_labels == ("Γ", "X", "L")
    assert bs.shifted


def test_read_spectral_bands_explicit_nk_ne_override(band_dir):
    # 40*30 = 1200 = 60*20: explicit values win over the gpi
    bs = read_spectral_bands(prefix=str(band_dir), nk=60, ne=20)
    assert bs.total.shape == (20, 60)


def test_read_spectral_bands_size_mismatch(band_dir):
    with pytest.raises(BandReadError, match="not divisible"):
        read_spectral_bands(prefix=str(band_dir), nk=7, ne=13)


def test_spectral_arrays_are_writable(band_dir):
    bs = read_spectral_bands(prefix=str(band_dir), cluster="Fe")
    assert bs.total.flags.writeable
    assert bs.projections.weights.flags.writeable


# ---------------------------------------------------------------------------
# evconv bandfiles


def test_read_bandfiles(tmp_path):
    kd = np.linspace(0.0, 1.0, 50)
    b1, b2 = np.cos(kd * 6), np.sin(kd * 6)
    np.savetxt(tmp_path / "bandfile_0", np.column_stack([kd, b1, b2]))
    np.savetxt(tmp_path / "symline_1", [[0.5, -1000.0], [0.5, 1000.0]])
    bs = read_bandfiles(prefix=str(tmp_path))
    assert bs.kind == "lines"
    assert bs.bands.shape == (50, 2)
    assert bs.kpath.ticks == (0.0, 0.5, 1.0)
    assert bs.energy_unit == "eV"


def test_read_bandfiles_multiple_files_and_evconv_inp(tmp_path):
    kd = np.linspace(0.0, 1.0, 5)
    # evconv writes at most 10 band columns per file
    np.savetxt(tmp_path / "bandfile_0", np.column_stack([kd] + [kd * i for i in range(10)]))
    np.savetxt(tmp_path / "bandfile_1", np.column_stack([kd, kd * 10, kd * 11]))
    (tmp_path / "symline_1").write_text(" 0.25 -1000.\n")  # a single-row symline
    (tmp_path / "evconv.inp").write_text("0.51 1.0 -1.0 1.0\n")
    bs = read_bandfiles(prefix=str(tmp_path))
    assert bs.bands.shape == (5, 12)
    np.testing.assert_allclose(bs.bands[-1], np.arange(12))
    assert bs.kpath.ticks == (0.0, 0.25, 1.0)
    assert bs.energy_unit == "Ry"


# ---------------------------------------------------------------------------
# fatbands


def write_fatband(fname, kd, energies, weights):
    """bandplots.F90 layout: one block of nk rows per band, blank line after each."""
    with open(fname, "w") as f:
        for band in range(energies.shape[1]):
            for k in range(len(kd)):
                f.write(f"{kd[k]:14.7f}{energies[k, band]:14.7f}{weights[k, band]:14.7f}\n")
            f.write("     \n")


def test_read_fatbands(tmp_path):
    kd = np.linspace(0.0, 1.0, 30)
    energies = np.column_stack([np.cos(kd), np.cos(kd) - 1])
    weights = np.column_stack([abs(np.sin(kd)), abs(np.cos(kd))])
    write_fatband(tmp_path / "fatbands.t01.e1.l2.m-1", kd, energies, weights)
    bs = read_fatbands(prefix=str(tmp_path))
    assert bs.bands.shape == (30, 2)
    assert bs.projections.labels == ("t01.e1.l2.m-1",)
    np.testing.assert_allclose(bs.projections.weights[..., 0], weights, atol=1e-6)
    np.testing.assert_allclose(bs.bands, energies, atol=1e-6)
    assert bs.energy_unit == "Ry" and not bs.shifted


def test_read_fatbands_name_variants_and_repeated_k(tmp_path):
    # a repeated k-distance (joint between two lines) must not break the
    # band blocks apart
    kd = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    energies = np.column_stack([kd, kd + 1, kd + 2])
    names = ["fatbands.t01.s_up", "fatbands.t01.s_dn", "fatbands.s_up",
             "fatbands.t02.e1.l1.s_dn", "fatbands.t03.e2.l2.m01"]
    for i, name in enumerate(names):
        write_fatband(tmp_path / name, kd, energies, np.full_like(energies, i))
    (tmp_path / "fatbands.agr").write_text("not a fatband file\n")
    bs = read_fatbands(prefix=str(tmp_path))
    assert bs.bands.shape == (6, 3)
    assert sorted(bs.projections.labels) == sorted(n[len("fatbands."):] for n in names)
    assert bs.kpath.ticks == (0.0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# raw eigenvalues


def write_eigenvalues(fname, kvecs, eigs):
    """bandplots.F90 wrev layout: 3 comment lines, k line, 4 eigenvalues per line."""
    with open(fname, "w") as f:
        for i, (kv, ev) in enumerate(zip(kvecs, eigs), start=1):
            f.write(f"# eigenvalues for k: {i:6d}\n# kx ky kz (* 2pi)\n# (e(i), i = 1, mx)\n")
            f.write("".join(f" {c:17.9E}" for c in kv) + "\n")
            for j in range(0, len(ev), 4):
                f.write("".join(f" {e:17.9E}" for e in ev[j : j + 4]) + "\n")


def test_read_eigenvalues_band_count_from_records(tmp_path):
    # 12 k-points x 7 bands: 12 * (3 + 7) = 120 values, divisible by 4, which
    # fooled the old smallest-divisor guess into nbands = 1
    kvecs = [(x, 0.0, 0.0) for x in np.linspace(0, 0.5, 12)]
    eigs = [np.arange(7) * 0.1 + i for i in range(12)]
    write_eigenvalues(tmp_path / "eigenvalues", kvecs, eigs)
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    assert bs.bands.shape == (12, 7)
    np.testing.assert_allclose(bs.bands[3], eigs[3])
    assert bs.energy_unit == "Ry" and not bs.shifted
    with pytest.raises(BandReadError, match="not 5"):
        read_eigenvalues(tmp_path / "eigenvalues", nbands=5)


def test_read_eigenvalues_detects_corners(tmp_path):
    kvecs = [(x, 0.0, 0.0) for x in np.linspace(0, 0.5, 11)]
    kvecs += [(0.5, y, 0.0) for y in np.linspace(0.05, 0.5, 10)]
    write_eigenvalues(tmp_path / "eigenvalues", kvecs, [[kv[0], kv[1]] for kv in kvecs])
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    assert bs.bands.shape == (21, 2)
    np.testing.assert_allclose(bs.kpath.ticks, [0.0, 0.5, 1.0])


def test_read_eigenvalues_repeated_corner(tmp_path):
    # the corner X listed twice (end of one line, start of the next)
    kvecs = [(x, 0.0, 0.0) for x in np.linspace(0, 0.5, 6)]
    kvecs += [(0.5, y, 0.0) for y in np.linspace(0.0, 0.5, 6)]
    write_eigenvalues(tmp_path / "eigenvalues", kvecs, [[kv[0] + kv[1]] for kv in kvecs])
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    assert bs.bands.shape == (11, 1)  # duplicate dropped
    np.testing.assert_allclose(bs.kpath.ticks, [0.0, 0.5, 1.0])


def test_read_eigenvalues_kpath_tool_paths(tmp_path):
    # paths from the kpath tool (incl. single-step nk=0 segments and the
    # appended zero-weight point) give one tick per node and no gaps
    from pyRSPthon.kpts.path import interpolate_path

    nodes = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0, 0, 0]])
    kvecs, node_at = interpolate_path(nodes, [10, 0, 10, 10])
    kvecs = np.vstack([kvecs, kvecs[-1]])  # zero_weight_point duplicate
    write_eigenvalues(tmp_path / "eigenvalues", kvecs, [[float(i)] for i in range(len(kvecs))])
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    assert bs.bands.shape == (len(kvecs) - 1, 1)
    np.testing.assert_allclose(bs.kpath.ticks, bs.kpath.kdist[sorted(node_at)])
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    np.testing.assert_allclose(bs.kpath.kdist[-1], seg.sum())


def test_read_eigenvalues_rejects_garbage(tmp_path):
    (tmp_path / "eigenvalues").write_text("# c\n 0.0 0.0 0.0\n 1.0 two 3.0\n")
    with pytest.raises(BandReadError, match="cannot parse"):
        read_eigenvalues(tmp_path / "eigenvalues")


# ---------------------------------------------------------------------------
# read_bands and transforms


def test_read_bands_detects_source(band_dir, tmp_path):
    assert read_bands(str(band_dir)).kind == "spectral"
    eig_dir = tmp_path / "eig"  # band_dir is tmp_path itself
    eig_dir.mkdir()
    write_eigenvalues(eig_dir / "eigenvalues", [(0, 0, 0), (0.1, 0, 0)], [[1.0], [2.0]])
    assert read_bands(str(eig_dir)).kind == "lines"
    with pytest.raises(FileNotFoundError):
        read_bands(str(tmp_path / "empty"))


def test_transforms(tmp_path):
    write_eigenvalues(tmp_path / "eigenvalues", [(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0)], [[0.5], [0.6], [0.7]])
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    shifted = to_unit(with_reference(bs, 0.5), "eV")
    np.testing.assert_allclose(shifted.bands[:, 0], [0.0, 0.1 * RY_TO_EV, 0.2 * RY_TO_EV])
    assert shifted.shifted and shifted.energy_unit == "eV"
    # the original is untouched (frozen models, pure transforms)
    np.testing.assert_allclose(bs.bands[:, 0], [0.5, 0.6, 0.7])
    labelled = with_labels(bs, labels=["G", "X"])
    assert labelled.kpath.tick_labels == ("G", "X")
    with pytest.raises(BandReadError, match="3 tick labels given for 2 ticks"):
        with_labels(bs, labels=["G", "X", "L"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        bs.bands = None


def test_pband_padding_columns_are_not_orbitals(tmp_path):
    # RSPt pads every pband record to the largest cluster's orbital count:
    # a d cluster (10 orbitals) in a run with a 14-orbital cluster has four
    # trailing all-zero columns after its orbitals
    nk, ne, nmom, norb, npad = 6, 5, 6, 10, 4
    ncol = 1 + nmom + norb + npad
    rec = np.zeros(ncol, dtype=np.float32)
    rec[: 1 + nmom] = -1.0
    rec[1 + nmom : 1 + nmom + norb] = np.arange(1, norb + 1)
    payload = np.tile(rec, nk * ne)
    header = build_band_header((ncol, ne, nk), -1.0, 1.0, 3, 1.0, [(2, 2, 0)])
    (tmp_path / "pband-Fe.data").write_bytes(header + payload.tobytes())
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Fe")
    np.testing.assert_array_equal(bs.projections.weights[0, 0], np.arange(1, norb + 1))
    assert bs.extra.shape == (ne, nk, nmom)
    # without a header the default start (column 1) still stops at the padding
    (tmp_path / "pband-Co.data").write_bytes(payload.tobytes())
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Co", nk=nk, ne=ne)
    assert bs.projections.weights.shape == (ne, nk, nmom + norb)


def test_read_spectral_bands_rejects_nonpositive_nk(band_dir):
    with pytest.raises(BandReadError, match="must be positive"):
        read_spectral_bands(prefix=str(band_dir), nk=0, ne=30)


def test_read_eigenvalues_without_separators(tmp_path):
    # older/hand-made files: no '#' lines between k-points, so the band
    # count must be given
    rows = [[x, 0.0, 0.0, 0.1 + x, 0.2 + x] for x in np.linspace(0, 0.5, 11)]
    text = "kx ky kz e1 e2\n" + "\n".join(" ".join(f"{v:.6f}" for v in r) for r in rows)
    (tmp_path / "eigenvalues").write_text(text)
    with pytest.raises(BandReadError, match="--nbands"):
        read_eigenvalues(tmp_path / "eigenvalues")
    bs = read_eigenvalues(tmp_path / "eigenvalues", nbands=2)
    assert bs.bands.shape == (11, 2)
    np.testing.assert_allclose(bs.bands[:, 0], [r[3] for r in rows], atol=1e-6)
    with pytest.raises(BandReadError, match="not a whole number"):
        read_eigenvalues(tmp_path / "eigenvalues", nbands=3)


def test_line_sources_are_tagged(tmp_path):
    write_eigenvalues(tmp_path / "eigenvalues", [(0, 0, 0), (0.1, 0, 0)], [[1.0], [2.0]])
    assert read_eigenvalues(tmp_path / "eigenvalues").source == "eigenvalues"


def test_pband_irrep_file(tmp_path):
    # spectrum Cf flag: pband-<cluster>-irrNNab.data holds one irrep block;
    # its header describes the whole cluster (d: 10 orbitals), so the
    # orbital count must not be inferred from it
    nk, ne, nblock = 6, 5, 4
    ncol = 1 + nblock
    payload = np.tile(np.arange(1, ncol + 1, dtype=np.float32), nk * ne)
    header = build_band_header((ncol, ne, nk), -1.0, 1.0, 3, 1.0, [(2, 2, 0)], cfflag=True)
    (tmp_path / "pband-Fe-irr0312.data").write_bytes(header + payload.tobytes())
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Fe-irr0312")
    assert bs.projections.irrep == "0312"
    assert bs.projections.cfflag is True
    np.testing.assert_array_equal(bs.projections.weights[0, 0], [2, 3, 4, 5])


def test_pband_irrep_shells_from_base_cluster(tmp_path):
    nk, ne = 4, 3
    np.ones(nk * ne * 5, dtype=np.float32).tofile(tmp_path / "pband-Fe-irr0112.data")
    (tmp_path / "green.inp").write_text("cluster\n1 IdFe\n1 2 1 1 0 0.59 0.60 0.38\n2 2 1.0\n")
    bs = read_spectral_bands(prefix=str(tmp_path), cluster="Fe-irr0112", nk=nk, ne=ne)
    assert bs.projections.shells == ({"type": 1, "l": 2, "basis": 0},)
