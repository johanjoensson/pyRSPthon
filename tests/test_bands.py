import numpy as np
import pytest

from pyRSPthon.read.bands import (
    parse_band_gpi,
    read_bandfiles,
    read_eigenvalues,
    read_fatbands,
    read_spectral_bands,
)


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
