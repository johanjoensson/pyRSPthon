# pyRSPthon

Utilities for setting up, managing, running and analyzing DFT (and DFT+DMFT)
calculations with the FP-LMTO code [RSPt](http://fplmto-rspt.org/).

Parts of this package reimplement helper tools that ship with RSPt
(`runs`, `cub`, `kpath`, `evconv`, plotting scripts) in Python, to make them
easier to use, script and modify.

## Installation

We recommend installing `pyRSPthon` inside a virtual environment (e.g. using `venv` or `uv`).

```sh
# Standard installation
pip install .

# Install with optional dependencies (note the quotes, which are required for zsh/macOS)
pip install ".[symmetry]"  # + spglib-based symmetry validation
pip install ".[kpoints]"   # + kpLib generalized k-point grids
pip install ".[symmetry,kpoints]"

# For development (installs in editable mode)
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Architecture & Design

`pyRSPthon` follows a layered architectural design separating business logic from the presentation layer:

1. **Layer 1 (Core)**: `orbitals.py` provides underlying data structures.
2. **Layer 2 (I/O & Adapters)**: `read/`, `write/`, and `ase/`. The `read()` functions return custom Data Transfer Objects (e.g. `NamedTuple` or `dataclass` representations of spectra). The `ase` module provides adapters converting core RSPt data to ASE `Atoms` objects.
3. **Layer 3 (Business Logic)**: `kpts/`, `verify/`, and `run/`. Contains the core algorithmic logic for k-point generation, green.inp verification, and SCF loop execution.
4. **Layer 4 (Presentation)**: `cli/`. Command-line entry points. 

**Strict CLI Rule**: CLI scripts in `src/pyRSPthon/cli/` are strictly thin wrappers that handle `argparse` and direct standard I/O. All heavy lifting and logic lives in the core library. This ensures that users can programmatically invoke all functionality by importing from `pyRSPthon.read`, `pyRSPthon.run`, etc.

## Command line tools

| Command | Purpose |
|---|---|
| `py_runs` | Drive RSPt SCF (and DMFT) cycles to convergence (replacement for `runs`) |
| `plot_dos` / `plot_pdos` | Plot total / cluster-projected density of states (`dos.dat`, `pdos-*.dat`) |
| `plot_band` / `plot_pband` | Plot (projected) band structures / spectral functions (`band.data`, `pband-*.data`, `bandfile_*`, `fatbands.*`) |
| `plot_dat` | Plot DMFT real/imag datasets (`real-*.dat`, `imag-*.dat`, hybridization, self-energies, ...) |
| `rspt_structure` | Convert any ASE-readable structure (CIF, POSCAR, ...) to `sym/symt.inp`, optionally driving `symt -all` + `stick` to a runnable calculation directory |
| `rspt_kpts` | Generate k-point meshes (`spts`) and band paths (`spts.band`) — a non-interactive replacement for `cub`/`kpath` |

All plot tools accept `--output FILE` for non-interactive figure export and
`-d DIR` to point at a calculation directory.

## Library

```python
from pyRSPthon.read import read            # auto-dispatch on filename
from pyRSPthon.read import extract_dos, extract_pdos, extract_dat
from pyRSPthon.ase import read_symt, write_symt   # ASE Atoms <-> symt.inp
from pyRSPthon.run import runs             # SCF driver, importable
```

**Important k-point invariant**: `rspt_kpts` reduces meshes using only the
symmetry operations RSPt itself knows (parsed from `symcof`); see
`pyRSPthon/kpts/` before changing that.

## License

GPL-3.0-or-later.
