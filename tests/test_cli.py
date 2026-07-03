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
