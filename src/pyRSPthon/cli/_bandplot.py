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


def orbital_labels(norb, spin_split=False):
    """
    Default labels for norb projected orbitals. With spin_split, the columns
    hold two spin blocks (first half down, second half up).
    """
    if spin_split and norb % 2 == 0:
        half = orbital_labels(norb // 2)
        return [f"{lab} ↓" for lab in half] + [f"{lab} ↑" for lab in half]
    if norb in ORBITAL_NAMES:
        return list(ORBITAL_NAMES[norb])
    return [f"orb {i}" for i in range(norb)]


def decorate_axes(ax, bs, emin=None, emax=None):
    """
    Common band-plot decorations: symmetry-point grid, Fermi line, labels.
    """
    if emin is None:
        emin = bs.energies[0] if bs.energies is not None else np.min(bs.bands)
    if emax is None:
        emax = bs.energies[-1] if bs.energies is not None else np.max(bs.bands)
    ax.axhline(0.0, linestyle="dotted", color="gray", linewidth=1)
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
