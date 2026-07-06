import struct

import numpy as np
import pytest

from pyRSPthon.read.band import peek_band_header
from pyRSPthon.read.bands import (
    parse_band_gpi,
    read_bandfiles,
    read_eigenvalues,
    read_fatbands,
    read_spectral_bands,
)


def build_band_header(shape, emin, emax, nfermi, scale, shells=(), cfflag=False):
    """Python mirror of build_band_header in RSPt's green_spectrum.F90."""
    nints = max(16, 13 + len(shells))
    hdr = bytearray(4 * nints)
    hdr[0:4] = b"RSPt"
    struct.pack_into("<ii", hdr, 4, 1, 4 * nints)
    struct.pack_into(f"<i{len(shape)}i", hdr, 12, len(shape), *shape)
    struct.pack_into("<ffif", hdr, 32, emin, emax, nfermi, scale)
    if shells:
        struct.pack_into("4B", hdr, 48, int(cfflag), len(shells), 0, 0)
        for n, (t, l, basis) in enumerate(shells):
            struct.pack_into("4B", hdr, 52 + 4 * n, t, l, basis, 0)
    return bytes(hdr)


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
    # 4 shells: the header grows past the 16-int minimum (to 17 ints)
    fname.write_bytes(
        build_band_header((13, 30, 40), -0.4, 0.4, -15, 1.0, shells, cfflag=True)
    )
    meta = peek_band_header(fname)
    assert meta["cfflag"] is True
    assert meta["ncorr"] == 4
    assert meta["shells"] == [
        {"type": t, "l": l, "basis": b} for t, l, b in shells
    ]
    assert "fermi_index" not in meta  # negative nfermi means unknown
    np.testing.assert_allclose(meta["scale"], 1.0)


def test_peek_band_header_absent(tmp_path):
    fname = tmp_path / "band.data"
    fname.write_bytes(np.zeros(120, dtype=np.float32).tobytes())
    assert peek_band_header(fname) is None
    assert peek_band_header(tmp_path / "missing.data") is None


def test_read_spectral_bands_from_header(tmp_path):
    nk, ne, ncol = 12, 8, 3
    header = build_band_header((ncol, ne, nk), -2.0, 2.0, 4, 13.6057)
    payload = np.arange(nk * ne * ncol, dtype=np.float32)
    (tmp_path / "band.data").write_bytes(header + payload.tobytes())
    # no band.gpi and no green.inp: all metadata must come from the header
    bs = read_spectral_bands(prefix=str(tmp_path))
    assert bs.spectral.shape == (ne, nk, ncol)
    assert bs.energy_unit == "eV"
    assert bs.fermi_index == 3
    np.testing.assert_allclose(bs.energies, np.linspace(-2.0, 2.0, ne), atol=1e-6)


def test_parse_band_gpi(band_dir):
    meta = parse_band_gpi(band_dir / "band.gpi")
    assert meta["nk"] == 40
    assert meta["ne"] == 30
    assert meta["energy_unit"] == "eV"
    assert meta["tick_labels"] == ["Γ", "X", "L"]
    assert meta["ticks"] == [0, 20, 39]
    np.testing.assert_allclose(meta["energies"][0], -5.0, atol=0.2)
    np.testing.assert_allclose(meta["energies"][-1], 5.0, atol=0.2)


def test_read_spectral_bands_auto_metadata(band_dir):
    bs = read_spectral_bands(prefix=str(band_dir))
    assert bs.kind == "spectral"
    assert bs.spectral.shape == (30, 40, 11)
    assert bs.energy_unit == "eV"
    assert bs.tick_labels == ["Γ", "X", "L"]


def test_read_spectral_bands_size_mismatch(band_dir):
    with pytest.raises(ValueError, match="not divisible"):
        read_spectral_bands(prefix=str(band_dir), nk=7, ne=13)


def test_read_bandfiles(tmp_path):
    kd = np.linspace(0.0, 1.0, 50)
    b1, b2 = np.cos(kd * 6), np.sin(kd * 6)
    np.savetxt(tmp_path / "bandfile_0", np.column_stack([kd, b1, b2]))
    np.savetxt(tmp_path / "symline_1", [[0.5, -1000.0], [0.5, 1000.0]])
    bs = read_bandfiles(prefix=str(tmp_path))
    assert bs.kind == "lines"
    assert bs.bands.shape == (50, 2)
    assert bs.ticks == [0.5]


def test_read_fatbands(tmp_path):
    kd = np.linspace(0.0, 1.0, 30)
    rows = [[k, np.cos(k), abs(np.sin(k))] for k in kd]
    rows += [[k, np.cos(k) - 1, abs(np.cos(k))] for k in kd]
    np.savetxt(tmp_path / "fatbands.t01.e1.l2.m-1", np.array(rows))
    bs = read_fatbands(prefix=str(tmp_path))
    assert bs.bands.shape == (30, 2)
    assert list(bs.weights) == ["t01.e1.l2.m-1"]
    assert bs.weights["t01.e1.l2.m-1"].shape == (30, 2)


def test_read_eigenvalues_detects_breaks(tmp_path):
    lines = ["# eig", "# c", "# c", "kx ky kz"]
    kvecs = [(x, 0.0, 0.0) for x in np.linspace(0, 0.5, 11)]
    kvecs += [(0.5, y, 0.0) for y in np.linspace(0.05, 0.5, 10)]
    for kv in kvecs:
        lines.append(
            " ".join(f"{c:.6f}" for c in kv) + f" {kv[0]:.6f} {kv[1]:.6f}"
        )
    (tmp_path / "eigenvalues").write_text("\n".join(lines))
    bs = read_eigenvalues(tmp_path / "eigenvalues")
    assert bs.bands.shape == (21, 2)
    assert len(bs.ticks) == 3  # start, direction change, end
