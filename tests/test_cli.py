import subprocess
import sys

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
