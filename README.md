# pyRSPthon

Utilities for setting up, managing, running and analyzing DFT (and DFT+DMFT)
calculations with the FP-LMTO code [RSPt](http://fplmto-rspt.org/).

Parts of this package reimplement helper tools that ship with RSPt
(`runs`, `cub`, `kpath`, `evconv`, plotting scripts) in Python, to make them
easier to use, script and modify.

## Installation

```sh
pip install .            # core: readers, plotting, run driver
pip install .[symmetry]  # + spglib-based symmetry validation
pip install .[kpoints]   # + kpLib generalized k-point grids
```

Requires Python >= 3.10.

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
