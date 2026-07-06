"""
Shared band-structure plotting machinery for plot_band / plot_pband.
"""

import numpy as np

# Colorblind-safe categorical colors (Okabe-Ito), assigned to orbitals in
# fixed order.
ORBITAL_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple-pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

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

# Matches the palette RSPt itself puts in band.gpi (a YlGnBu-style ramp).
SPECTRAL_CMAP = "YlGnBu"


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


def shell_labels(shells, norb, spin_split=False, cfflag=False):
    """
    Default labels for the projected columns of a cluster given its
    correlated shells ([{'type', 'l', 'basis'}, ...] in header order, e.g.
    from peek_band_header). The per-shell orbital_labels are concatenated,
    prefixed with the shell identity when there is more than one shell;
    with spin_split the two spin blocks each hold all shells (first half
    down, second half up).
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
    half = []
    for sh in shells:
        labs = orbital_labels(2 * sh["l"] + 1, basis_id=sh["basis"], l=sh["l"])
        if len(shells) > 1:
            labs = [f"t{sh['type']} l{sh['l']} {lab}" for lab in labs]
        half.extend(labs)
    # jj bases carry both j-manifolds in one block; no down/up split applies
    if spin_split and not any(sh["basis"] in (9, 10, 11) for sh in shells):
        return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
    return half


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


def decorate_axes(ax, bs, emin=None, emax=None):
    """
    Common band-plot decorations: symmetry-point grid, Fermi line, labels.
    """
    if emin is None:
        emin = bs.energies[0] if bs.energies is not None else np.min(bs.bands)
    if emax is None:
        emax = bs.energies[-1] if bs.energies is not None else np.max(bs.bands)
    fermi = 0.0
    if bs.fermi_index is not None and bs.energies is not None:
        fermi = bs.energies[bs.fermi_index]
    ax.axhline(fermi, linestyle="dotted", color="gray", linewidth=1)
    for tick in bs.ticks:
        ax.axvline(tick, linestyle="dotted", color="gray", linewidth=1)
    if bs.ticks and any(bs.tick_labels):
        ax.set_xticks(bs.ticks, bs.tick_labels)
    elif bs.ticks:
        ax.set_xticks(bs.ticks)
        ax.set_xticklabels([""] * len(bs.ticks))
    ax.set_xlim(bs.kdist[0], bs.kdist[-1])
    ax.set_ylim(emin, emax)
    ax.set_ylabel(rf"E - E$_F$ ({bs.energy_unit})")


def plot_spectral(ax, bs, column=0, log=False, vmax=None, cmap=SPECTRAL_CMAP):
    """
    Intensity plot of one data column of a spectral BandStructure.
    """
    data = bs.spectral[:, :, column]
    if log:
        data = np.log(np.maximum(data, np.max(data) * 1e-6))
    im = ax.imshow(
        data[::-1],
        extent=(bs.kdist[0], bs.kdist[-1], bs.energies[0], bs.energies[-1]),
        aspect="auto",
        cmap=cmap,
        vmax=vmax,
    )
    return im


def plot_composite(ax, bs, columns, colors=None, gamma=1.0):
    """
    RGB-composite plot of several orbital columns of a spectral
    BandStructure: each column tints the image with its own color, alpha
    proportional to weight (the alpha-compositing of the old plot_pband).
    """
    from matplotlib.colors import to_rgb

    if colors is None:
        colors = ORBITAL_COLORS
    weights = bs.spectral[:, :, columns]
    weights = weights - np.min(weights)
    wmax = np.max(weights)
    if wmax > 0:
        weights = weights / wmax
    if gamma != 1.0:
        weights = weights**gamma

    ne, nk, nc = weights.shape
    # src-over compositing with premultiplied colors, then over white
    out = np.zeros((ne, nk, 4))
    for i in range(nc):
        rgb = np.array(to_rgb(colors[i % len(colors)]))
        alpha = weights[:, :, i]
        a_out = alpha + out[:, :, 3] * (1 - alpha)
        for c in range(3):
            out[:, :, c] = rgb[c] * alpha + out[:, :, c] * (1 - alpha)
        out[:, :, 3] = a_out
    img = out[:, :, :3] + (1.0 - out[:, :, 3:4])
    ax.imshow(
        img[::-1],
        extent=(bs.kdist[0], bs.kdist[-1], bs.energies[0], bs.energies[-1]),
        aspect="auto",
    )


def plot_lines(ax, bs, color="#0072B2", linewidth=1.2, label=None):
    """
    Eigenvalue-line plot of a lines BandStructure.
    """
    lines = ax.plot(bs.kdist, bs.bands, color=color, linewidth=linewidth)
    if label and lines:
        lines[0].set_label(label)
    return lines


def plot_fatbands(ax, bs, labels=None, scale=30.0, colors=None):
    """
    Scatter fatbands: marker area proportional to the projection weight.
    labels: which of bs.weights to draw (default: all).
    """
    if colors is None:
        colors = ORBITAL_COLORS
    if labels is None:
        labels = list(bs.weights)
    plot_lines(ax, bs, color="0.6", linewidth=0.6)
    wmax = max(np.max(np.abs(bs.weights[lab])) for lab in labels) or 1.0
    for i, lab in enumerate(labels):
        w = np.abs(bs.weights[lab]) / wmax
        for band in range(bs.bands.shape[1]):
            ax.scatter(
                bs.kdist,
                bs.bands[:, band],
                s=scale * w[:, band],
                color=colors[i % len(colors)],
                label=lab if band == 0 else None,
                alpha=0.7,
                edgecolors="none",
            )
    ax.legend(loc="upper right", fontsize="small")
