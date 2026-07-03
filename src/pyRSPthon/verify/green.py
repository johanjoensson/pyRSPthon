"""
green.inp verification: structural lint (from the parser) plus semantic and
cross-file checks that RSPt itself only performs after an expensive startup,
or not at all. Source references (:NNN) point into rspt's green_init.F90;
the time-reversal check mirrors green_trunk_interface.F90:369-402.

Cross-file checks use the companion files of a run directory when present
(data, symcof, spts, sym/symt.inp, sig) and are skipped with an INFO note
when they are not - a missing companion never produces a false error.
"""

import os
from dataclasses import dataclass

import numpy as np

from ..read.greeninp import (
    ERROR,
    INFO,
    SOLVER_MAX,
    SOLVER_MIN,
    WARNING,
    Finding,
    parse_green_inp,
)

DMRG_SOLVER = 7
ED_REALAXIS_SOLVER = 9
DYNAMICAL_EXEMPT = (0, 2, -2)  # observer, LDA+U, carbon copy
CPA_SOLVER = -1
CARBONCOPY_SOLVER = -2


def _clname(cl, i):
    return f"cluster {cl.label!r}" if cl.label else f"cluster #{i + 1}"


def _orbital_key(cl):
    return tuple((o.t, o.l, o.e, o.site, o.basis) for o in cl.orbitals)


def check_intra(green):
    """Checks that need nothing beyond the parsed green.inp."""
    findings = []

    def add(level, line, message, suggestion=None):
        findings.append(Finding(level, line, message, suggestion))

    clusters = green.clusters
    solvers = [cl.solver_type for cl in clusters if cl.solver_type is not None]
    explicit_labels = {}
    auto_keys = {}
    for i, cl in enumerate(clusters):
        name = _clname(cl, i)
        if cl.solver_type is not None:
            if not SOLVER_MIN <= cl.solver_type <= SOLVER_MAX:
                add(
                    WARNING,
                    cl.line,
                    f"{name}: unknown solver_type {cl.solver_type}; RSPt "
                    "converts the cluster to an observer - its self-energy is "
                    "never updated (:405)",
                )
            elif cl.solver_type == 0 and cl.ncorr >= 1:
                add(
                    INFO,
                    cl.line,
                    f"{name}: solver_type 0 - a self-energy may be read from "
                    "file but is never updated in the DMFT loop (:391)",
                )
            if cl.sigma_mix is not None and cl.sigma_mix < -1e-10:
                add(ERROR, cl.line, f"{name}: negative sigma_mix (RSPt stops, :403)")
            if cl.solver_type == 0 and cl.double_counting:
                add(
                    WARNING,
                    cl.line,
                    f"{name}: double counting with solver_type 0 is forced to "
                    "0 by RSPt - the stored self-energy is already double-"
                    "counted (:416)",
                )
            corr = [o for o in cl.orbitals if o.correlated]
            if (
                SOLVER_MIN <= (cl.solver_type or 0) <= SOLVER_MAX
                and cl.solver_type
                and cl.double_counting
                and corr
                and all(abs(s) < 1e-8 for o in corr for s in o.slater)
            ):
                add(
                    WARNING,
                    cl.line,
                    f"{name}: all Slater parameters are zero but a double "
                    "counting is set; RSPt forces DC = 0 (:424)",
                )
            if cl.virtual and cl.solver_type == CPA_SOLVER:
                add(ERROR, cl.line, f"{name}: a virtual cluster may not use the CPA solver (RSPt stops, :470)")
            if cl.virtual and cl.solver_type == CARBONCOPY_SOLVER:
                add(ERROR, cl.line, f"{name}: a virtual cluster may not use the carbon-copy solver (RSPt stops, :476)")

        # Label collisions (RSPt stops on identical/empty labels, :344-364)
        if cl.label:
            if cl.label in explicit_labels:
                add(
                    ERROR,
                    cl.line,
                    f"duplicate cluster Id {cl.label!r} (also line "
                    f"{explicit_labels[cl.label]}); RSPt stops",
                )
            explicit_labels[cl.label] = cl.line
        elif cl.orbitals:
            key = (cl.orbitals[0].t, cl.orbitals[0].l, cl.orbitals[0].e, cl.orbitals[0].site, cl.ncorr > 0)
            if key in auto_keys:
                add(
                    ERROR,
                    cl.line,
                    "two clusters without Id and the same first orbital get "
                    "identical auto-generated labels; RSPt stops",
                    "add IdXXX to the cluster header lines",
                )
            auto_keys[key] = cl.line

    # Virtual clusters need a matching non-virtual cluster (:453-467)
    real_keys = {_orbital_key(cl) for cl in clusters if not cl.virtual}
    for i, cl in enumerate(clusters):
        if cl.virtual and _orbital_key(cl) not in real_keys:
            add(
                ERROR,
                cl.line,
                f"{_clname(cl, i)}: virtual cluster without a non-virtual "
                "cluster with exactly the same orbitals (RSPt stops, :462)",
            )

    # tensmom / modelexchange label references
    known = set(explicit_labels)
    has_auto = any(not cl.label for cl in clusters)
    for label, line in green.tensmom_labels:
        if label not in known:
            add(
                WARNING if has_auto else ERROR,
                line,
                f"tensmom: cluster Id {label!r} does not match any cluster "
                "Id in this file (RSPt stops if it does not resolve)",
            )
    by_label = {cl.label: cl for cl in clusters if cl.label}
    for label1, label2, _, line in green.modelexchange:
        missing = [lb for lb in (label1, label2) if lb not in known]
        if missing:
            add(
                WARNING if has_auto else ERROR,
                line,
                f"modelexchange: cluster Id(s) {missing} not found (RSPt stops, :644)",
            )
        elif label1 in by_label and label2 in by_label:
            sites1 = {(o.t, o.site) for o in by_label[label1].orbitals}
            sites2 = {(o.t, o.site) for o in by_label[label2].orbitals}
            if not sites1 & sites2:
                add(
                    ERROR,
                    line,
                    "modelexchange: the two orbital sets must belong to the "
                    "same atom (RSPt stops, :637)",
                )

    # Matsubara mesh (:487-552)
    if green.matsubara:
        (nmats, head, log, tail), line = green.matsubara
        if head < 0 or log < 0 or tail < 0:
            add(ERROR, line, "matsubara: negative head/log/tail mesh parameters (RSPt stops, :547)")
        dynamical = any(s not in DYNAMICAL_EXEMPT for s in solvers) or any(
            cl.virtual for cl in clusters
        )
        if nmats == 1 and dynamical:
            add(
                ERROR,
                line,
                "matsubara: nmats = 1 with a dynamical (or CPA/virtual) solver "
                "(RSPt stops: 'Set nmats > 1 ...', :521)",
            )
        if 1 not in solvers and nmats < head + log + tail:
            add(
                WARNING,
                line,
                f"matsubara: nmats = {nmats} < head+log+tail = "
                f"{head + log + tail}; RSPt silently raises nmats (:529)",
            )
        if any(cl.virtual for cl in clusters) and tail <= 1:
            add(ERROR, line, "matsubara: virtual/CPA clusters need nmats_tail > 2 (RSPt stops, :533)")
        if green.spectrum is not None and nmats < 100:
            add(
                WARNING,
                line,
                "matsubara: spectrum requested with nmats < 100 - check any "
                "Pade continuation carefully (:537)",
            )

    if green.energymesh:
        (size, emin, emax, eim), line = green.energymesh
        if size <= 0:
            add(ERROR, line, "energymesh: number of energy points must be positive")
        if emax <= emin:
            add(ERROR, line, f"energymesh: emax = {emax} <= emin = {emin}")
        if eim <= 0:
            add(
                WARNING,
                line,
                f"energymesh: imaginary broadening eim = {eim} <= 0 puts the "
                "mesh on the real axis",
            )

    if green.mixing:
        values, line = green.mixing
        if not 1 <= values[0] <= 3:
            add(ERROR, line, f"mixing: unknown mix_method {values[0]}, must be 1..3 (RSPt stops, :650)")

    sp = green.spectrum
    if sp is not None:
        solver_spectral = any(s in (DMRG_SOLVER, ED_REALAXIS_SOLVER) for s in solvers)
        if (sp.dos or sp.fermi or solver_spectral) and (sp.band or sp.surf):
            add(
                ERROR,
                sp.line,
                "spectrum: Band/Surf (k-path) and Dos/Fermi (dense mesh) are "
                "mutually exclusive (RSPt stops, :678)",
            )
        if sp.proj and sp.surf:
            add(ERROR, sp.line, "spectrum: Surf cannot project individual orbitals (Proj) (RSPt stops, :681)")
        if sp.pband or sp.psurf:
            if not clusters:
                add(
                    WARNING,
                    sp.line,
                    "spectrum: Pband/Psurf needs at least one cluster block - "
                    "RSPt silently disables the projection (:687)",
                )
            elif sp.obs and not any(cl.ncorr == 0 for cl in clusters):
                add(
                    WARNING,
                    sp.line,
                    "spectrum: Obs projection needs an observer cluster "
                    "(no correlated orbitals) - RSPt silently disables it (:692)",
                )
        if sp.cf and (sp.pband or sp.psurf) and clusters and not any(cl.cf for cl in clusters):
            add(
                WARNING,
                sp.line,
                "spectrum: Cf projection needs at least one cluster with the "
                "Cf flag - RSPt silently disables it (:703)",
            )
        if sp.miller is not None and all(m == 0 for m in sp.miller):
            add(ERROR, sp.line, "spectrum: Miller index [0 0 0] is unphysical (RSPt stops, :2476)")
        if sp.surf and (sp.surfnk is None or any(n <= 0 for n in sp.surfnk)):
            add(
                ERROR,
                sp.line,
                "spectrum: Surf needs the k-surface density nk(1:3) > 0 on "
                "the surface parameter line (RSPt stops, :2595)",
            )

    if green.carriers is not None and not (sp is not None and sp.dos):
        add(
            WARNING,
            0,
            "carriers: needs a DOS calculation (spectrum Dos) - RSPt silently "
            "disables the carrier analysis (:712)",
        )

    if green.jij and clusters and not any(cl.ncorr == 0 for cl in clusters):
        add(
            ERROR,
            0,
            "isoexch/susceptibility: a Jij calculation requires an observer "
            "cluster (no correlated orbitals) as center (RSPt stops, :723)",
        )

    return findings


@dataclass
class RunContext:
    directory: str
    data: object = None  # RSPtData
    atoms: object = None  # ase Atoms from symt.inp
    symcof: object = None  # SymcofGroup
    spts: tuple | None = None  # (kpoints, weights, iwsum, labels)
    has_sig: bool = False

    @property
    def fullrel(self):
        if self.data is not None and self.data.fullrel is not None:
            return self.data.fullrel
        if self.atoms is not None:
            return self.atoms.info.get("fullrel")
        return None

    @property
    def spinpol(self):
        if self.data is not None and self.data.spinpol is not None:
            return self.data.spinpol
        if self.atoms is not None:
            return self.atoms.info.get("spinpol")
        return None

    @property
    def spin_average(self):
        return self.data.spin_average if self.data is not None else False

    @property
    def cell(self):
        return self.atoms.cell[:] if self.atoms is not None else None


def load_context(directory):
    """Gather the companion files of a run directory (all optional)."""
    findings = []
    ctx = RunContext(directory=directory)

    def skip(name, exc):
        findings.append(
            Finding(INFO, 0, f"could not read {name} ({exc}); dependent cross-checks skipped")
        )

    path = os.path.join(directory, "data")
    if os.path.exists(path):
        try:
            from ..read.data import read_data

            ctx.data = read_data(path)
        except Exception as exc:  # defensive: data must never break verification
            skip("data", exc)

    for cand in (os.path.join("sym", "symt.inp"), "symt.inp"):
        path = os.path.join(directory, cand)
        if os.path.exists(path):
            try:
                from ..ase.atoms import read_symt

                ctx.atoms = read_symt(path)
            except Exception as exc:
                skip(cand, exc)
            break

    path = os.path.join(directory, "symcof")
    if os.path.exists(path):
        try:
            from ..read.symcof import read_symcof

            ctx.symcof = read_symcof(path)
        except Exception as exc:
            skip("symcof", exc)

    path = os.path.join(directory, "spts")
    if os.path.exists(path):
        try:
            from ..kpts.spts import read_spts

            ctx.spts = read_spts(path)
        except Exception as exc:
            skip("spts", exc)

    ctx.has_sig = os.path.exists(os.path.join(directory, "sig"))
    return ctx, findings


def _single_plane(kpoints, cell):
    """All k-points share one constant coordinate component (RSPt's escape
    for in-plane band paths, green_trunk_interface.F90:386-392)."""
    kpoints = np.asarray(kpoints, dtype=float)
    coords = kpoints
    if cell is not None:
        B = np.linalg.inv(np.asarray(cell, dtype=float).T).T
        coords = kpoints @ np.asarray(B).T
    return any(
        np.allclose(coords[:, j], coords[0, j], atol=1e-10) for j in range(3)
    )


def check_time_reversal(green, ctx):
    """
    Predict the spts time-reversal stop (green_trunk_interface.F90:369-402):
    fires for a fully-relativistic, non-spin-polarized run whose symcof group
    lacks space inversion, when the spts weights look like a mesh that was
    NOT reduced with time reversal (and no zero-weight escape point exists).
    """
    findings = []
    if ctx.symcof is None or ctx.spts is None:
        return findings
    if not ctx.fullrel or ctx.spinpol or ctx.spin_average:
        return findings
    rotations = ctx.symcof.rotations
    if any(np.allclose(rot, -np.eye(3), atol=1e-8) for rot in rotations):
        return findings  # space inversion present: RSPt skips the check
    kpoints, weights, _, _ = ctx.spts
    weights = np.asarray(weights)
    if not ((weights >= 2).any() and not (weights == 0).any()):
        return findings
    ngrp = len(rotations)
    if (weights > ngrp).any():
        return findings
    if (weights == ngrp).any() and _single_plane(kpoints, ctx.cell):
        return findings
    findings.append(
        Finding(
            ERROR,
            0,
            "RSPt will stop: 'The spts file was probably not generated to "
            "include time-reversal symmetry!' - this is a fully-relativistic, "
            "non-spin-polarized run without space inversion, so RSPt demands "
            "a time-reversal-reduced k-mesh",
            "regenerate the mesh with 'rspt_kpts grid ... --time-reversal', "
            "or append a k-point with weight 0 to spts to disable the check "
            "(only if the weights really do account for time reversal)",
        )
    )
    return findings


def check_cross(green, ctx):
    """Checks between green.inp and the companion files."""
    findings = []

    def add(level, line, message, suggestion=None):
        findings.append(Finding(level, line, message, suggestion))

    frel = ctx.fullrel
    sppol = ctx.spinpol
    clusters = green.clusters

    for i, cl in enumerate(clusters):
        name = _clname(cl, i)
        for orb in cl.orbitals:
            if orb.basis in (9, 10, 11) and frel is False:
                add(
                    ERROR,
                    orb.line,
                    f"{name}: the Jj basis ({orb.basis}) requires a fully "
                    "relativistic calculation (fulrel) (RSPt stops, :442)",
                )
            if orb.ewin_spin and sppol is False:
                add(
                    ERROR,
                    orb.line,
                    f"{name}: Ewinup/Ewindn need a spin-polarized calculation "
                    "(RSPt stops, :1781)",
                )

    # Cluster ids against the data file basis list
    if ctx.data is not None and ctx.data.types:
        types = ctx.data.types
        ntype = ctx.data.ntype or len(types)
        for i, cl in enumerate(clusters):
            name = _clname(cl, i)
            for orb in cl.orbitals:
                if not 1 <= orb.t <= ntype:
                    add(
                        ERROR,
                        orb.line,
                        f"{name}: type t = {orb.t} out of range 1..{ntype} "
                        "(RSPt stops: 'Invalid state', :1850)",
                    )
                    continue
                info = types[orb.t - 1]
                if info.natom is not None and not 1 <= orb.site <= info.natom:
                    add(
                        ERROR,
                        orb.line,
                        f"{name}: site = {orb.site} out of range 1.."
                        f"{info.natom} for type {orb.t} ({info.label})",
                    )
                if info.bases and (orb.l, orb.e) not in info.bases:
                    have = sorted(info.bases)
                    add(
                        ERROR,
                        orb.line,
                        f"{name}: no basis function with l = {orb.l}, energy "
                        f"set e = {orb.e} for type {orb.t} ({info.label}); "
                        f"available (l, e): {have} (RSPt stops: 'Invalid state', :1850)",
                    )

    if green.jij and sppol is False:
        add(
            ERROR,
            0,
            "isoexch/susceptibility: a Jij calculation requires a spin-"
            "polarized calculation (RSPt stops, :721)",
        )

    if (
        green.inputoutput is not None
        and green.inputoutput[0][0]
        and any(cl.ncorr >= 1 for cl in clusters)
        and not ctx.has_sig
    ):
        add(
            WARNING,
            green.inputoutput[1],
            "inputoutput: readsig = T but there is no sig file; RSPt starts "
            "from a zero self-energy",
        )

    sp = green.spectrum
    if sp is not None and sp.band and ctx.spts is not None:
        kpoints, weights, _, _ = ctx.spts
        weights = np.asarray(weights)
        if len(kpoints) > 1024:
            add(
                ERROR,
                sp.line,
                f"spectrum Band: spts holds {len(kpoints)} k-points, more "
                "than RSPt's limit of 1024 for band plots (RSPt stops, :540)",
            )
        if (weights > 1).any():
            add(
                WARNING,
                sp.line,
                "spectrum Band: the spts file has weights > 1 - it looks like "
                "an SCF integration mesh, not a k-point path; the band panels "
                "will not follow high-symmetry lines",
                "generate a path file with 'rspt_kpts path' and use it as spts",
            )
        elif sp.gp_nkp:
            expected = sum(n + 1 for n in sp.gp_nkp) + 1
            n_path = len(kpoints) - int((weights == 0).any())
            if n_path != expected:
                add(
                    WARNING,
                    sp.line,
                    f"spectrum: gp_nkp {sp.gp_nkp} implies {expected} path "
                    f"points but spts holds {n_path}; the band-plot symmetry-"
                    "point ticks will be misplaced",
                )

    findings.extend(check_time_reversal(green, ctx))
    return findings


def verify_green(directory=".", filename="green.inp"):
    """
    Verify a green.inp file in the context of its run directory.
    Returns a list of Findings (empty = clean).
    """
    green, findings = parse_green_inp(os.path.join(directory, filename))
    findings.extend(check_intra(green))
    ctx, ctx_findings = load_context(directory)
    findings.extend(ctx_findings)
    findings.extend(check_cross(green, ctx))
    return findings
