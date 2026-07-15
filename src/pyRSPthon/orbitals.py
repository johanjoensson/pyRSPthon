"""
Shell/orbital structure helpers shared by the read and CLI layers.

These describe the local bases RSPt uses for correlated shells (see
lda_mlmsatomicqn in green_trunk_interface.F90) and how their projected
columns are laid out in the *-<cluster>.dat / pband-<cluster>.data files:
per correlated shell, a spin-down block followed by that shell's spin-up
block, shells concatenated in header order.
"""

# Real-harmonic orbital names per l channel, in RSPt's m = -l..l order.
ORBITAL_NAMES = {
    1: ["s"],
    3: [r"p$_y$", r"p$_z$", r"p$_x$"],
    5: [r"d$_{xy}$", r"d$_{yz}$", r"d$_{z^2}$", r"d$_{xz}$", r"d$_{x^2-y^2}$"],
    7: [
        r"f$_{y(3x^2-y^2)}$",
        r"f$_{xyz}$",
        r"f$_{yz^2}$",
        r"f$_{z^3}$",
        r"f$_{xz^2}$",
        r"f$_{z(x^2-y^2)}$",
        r"f$_{x(x^2-3y^2)}$",
    ],
}


def _frac(val):
    return f"{int(round(val * 2))}/2"


# One spin block of RSPt's crystal-field bases (basis 1-7), by bit and l.
# The basis id is bit-coded: 4 = A2u (f only), 2 = Eg (d) / T1u (f),
# 1 = s / p / T2g (d) / T2u (f); blocks appear in the order 4, 2, 1.
_CF_BIT1 = {
    0: ["s"],
    1: [r"p$_y$", r"p$_x$", r"p$_z$"],
    2: [r"d$_{yz}$", r"d$_{xz}$", r"d$_{xy}$"],
    3: [r"f$_{x(y^2-z^2)}$", r"f$_{y(z^2-x^2)}$", r"f$_{z(x^2-y^2)}$"],
}
_CF_BIT2 = {
    2: [r"d$_{z^2}$", r"d$_{x^2-y^2}$"],
    3: [r"f$_{x^3}$", r"f$_{y^3}$", r"f$_{z^3}$"],
}
_CF_BIT4 = {3: [r"f$_{xyz}$"]}


def _cf_labels(basis_id, l):
    """
    One spin block of RSPt's crystal-field basis for one shell, following
    lda_mlmsatomicqn in green_trunk_interface.F90: a bit is consumed only
    when its l constraint matches, so e.g. basis 4-7 with a d shell all
    give the full eg+t2g set, and any basis 1-7 with a p shell gives
    py, px, pz.
    """
    i = basis_id
    half = []
    if i >= 4 and l == 3:
        i -= 4
        half += _CF_BIT4[l]
    if i >= 2 and l >= 2:
        i -= 2
        half += _CF_BIT2[l]
    if i >= 1:
        half += _CF_BIT1[l]
    return half


def orbital_labels(norb, spin_split=False, basis_id=None, l=None, cfflag=False):
    """
    Default labels for norb projected orbitals of one shell, matching the
    local bases defined by lda_mlmsatomicqn in RSPt's
    green_trunk_interface.F90. With spin_split, the columns hold two spin
    blocks (first half down, second half up). Note that some crystal-field
    bases (1-7) keep only a subset of the shell's orbitals (e.g. basis 1 =
    t2g), so the label count can be smaller than 2l+1; many combinations
    (any basis >= 3 for a d shell, basis 7 for f) still span the full
    shell. With the cluster Cf flag the given basis is only the starting
    point for the automatically generated crystal field states — a square
    rotation that never changes the orbital count — so the labels are
    generic.
    """
    if cfflag:
        n = norb // 2 if (spin_split and norb % 2 == 0) else norb
        half = [f"Cf state {i+1}" for i in range(n)]
        if spin_split and norb % 2 == 0:
            return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
        return half

    # JJ bases, quantized along the local z (9), x (10) or y (11) axis:
    # the j = l-1/2 manifold followed by j = l+1/2, mj ascending in each.
    if basis_id in (9, 10, 11) and l is not None:
        msym = {9: "m", 10: r"m$_x$", 11: r"m$_y$"}[basis_id]
        j1 = l - 0.5
        j2 = l + 0.5
        labels = []
        if j1 > 0:
            m = -j1
            while m <= j1 + 0.1:
                labels.append(f"j={_frac(j1)}, {msym}={_frac(m)}")
                m += 1.0
        m = -j2
        while m <= j2 + 0.1:
            labels.append(f"j={_frac(j2)}, {msym}={_frac(m)}")
            m += 1.0
        return labels

    if basis_id is not None and 1 <= basis_id <= 7 and l is not None and 0 <= l <= 3:
        half = _cf_labels(basis_id, l)
        if spin_split:
            return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
        return half

    # Everything else (0, 8, out-of-range ids) is the identity basis:
    # complex spherical harmonics in m = -l..l order.
    if basis_id is not None and l is not None:
        half = [f"m={m}" for m in range(-l, l + 1)]
        if spin_split:
            return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
        return half

    if spin_split and norb % 2 == 0:
        half = orbital_labels(norb // 2)
        return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
    if norb in ORBITAL_NAMES:
        return list(ORBITAL_NAMES[norb])
    return [f"orb {i}" for i in range(norb)]


def shell_orbital_count(basis, l):
    """
    Number of projected orbitals one shell contributes as a single block:
    the per-spin orbital count for ordinary bases (crystal-field bases
    1-7 may reduce it below 2l+1), or the full 4l+2 j-manifold block for
    the jj bases 9/10/11 (which are not spin split). Shares the crystal
    field reduction with orbital_labels so the two never diverge.
    """
    return len(orbital_labels(2 * l + 1, basis_id=basis, l=l))


def spin_split_indices(shells):
    """
    Column indices of the spin-down and spin-up orbitals of a cluster,
    given its correlated shells ([{'type', 'l', 'basis'}, ...] in header
    order). RSPt lays the projected orbitals out per shell as a spin-down
    block immediately followed by that shell's spin-up block. Returns
    (down_idx, up_idx) as 0-based indices into the orbital array, with
    down_idx[k] paired to up_idx[k] (the kth down orbital and its spin
    partner). Returns None when the split is undefined: no shells, or a jj
    basis (9/10/11) whose single block is not spin split.
    """
    if not shells:
        return None
    down_idx = []
    up_idx = []
    pos = 0
    for sh in shells:
        if sh["basis"] in (9, 10, 11):
            return None
        n = shell_orbital_count(sh["basis"], sh["l"])
        down_idx.extend(range(pos, pos + n))
        up_idx.extend(range(pos + n, pos + 2 * n))
        pos += 2 * n
    return down_idx, up_idx


def shell_labels(shells, norb, spin_split=False, cfflag=False):
    """
    Default labels for the projected columns of a cluster given its
    correlated shells ([{'type', 'l', 'basis'}, ...] in header order, e.g.
    from peek_band_header). The per-shell orbital_labels are concatenated,
    prefixed with the shell identity when there is more than one shell.
    With spin_split each shell holds its own spin-down block followed by
    its spin-up block (RSPt's per-shell layout), and the shells are
    concatenated in header order.
    """
    if cfflag or not shells:
        first = shells[0] if shells else {}
        return orbital_labels(
            norb,
            spin_split=spin_split,
            basis_id=first.get("basis"),
            l=first.get("l"),
            cfflag=cfflag,
        )
    out = []
    for sh in shells:
        labs = orbital_labels(2 * sh["l"] + 1, basis_id=sh["basis"], l=sh["l"])
        if len(shells) > 1:
            labs = [f"t{sh['type']} l{sh['l']} {lab}" for lab in labs]
        # jj bases carry both j-manifolds in one block; no down/up split applies
        if spin_split and sh["basis"] not in (9, 10, 11):
            labs = [f"{lab} ↓" for lab in labs] + [f"{lab} ↑" for lab in labs]
        out.extend(labs)
    return out


def find_cluster_shells(cluster, directory="."):
    """
    Look up the correlated shells of a named cluster in green.inp.
    Matches the explicit IdX label, the t.l.e.site.basis string of the first
    orbital, or (for unlabeled clusters) RSPt's autogenerated t<T>.e<E>.l<L>...
    name. Returns (shells, cfflag) shaped like the Band_header data
    ([{'type', 'l', 'basis'}, ...]), or (None, False) if nothing matches.
    """
    import os
    from pyRSPthon.read.greeninp import parse_green_inp

    try:
        green, _ = parse_green_inp(os.path.join(directory or ".", "green.inp"))
    except Exception:
        return None, False
    for cl in green.clusters:
        corr = [o for o in cl.orbitals if o.correlated] or cl.orbitals
        if not corr:
            continue
        orb = corr[0]
        names = (cl.label, f"{orb.t}{orb.l}{orb.e}{orb.site}{orb.basis}")
        autoprefix = f"t{orb.t}.e{orb.e}.l{orb.l}"
        if cluster in names or (not cl.label and cluster.startswith(autoprefix)):
            shells = [{"type": o.t, "l": o.l, "basis": o.basis} for o in corr]
            return shells, cl.cf
    return None, False
