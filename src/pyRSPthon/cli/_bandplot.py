"""
Shared band-structure plotting machinery for plot_band / plot_pband.
"""

import numpy as np

# Shell/orbital structure helpers live in the shared pyRSPthon.orbitals
# module (also used by the read layer); re-exported here so the existing
# bp.orbital_labels / bp.shell_labels / bp.find_cluster_shells call sites
# keep working.
from pyRSPthon.orbitals import (  # noqa: F401
    ORBITAL_NAMES,
    orbital_labels,
    shell_labels,
    shell_orbital_count,
    spin_split_indices,
    find_cluster_shells,
)

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

# Matches the palette RSPt itself puts in band.gpi (a YlGnBu-style ramp).
SPECTRAL_CMAP = "YlGnBu"


def spin_sum_columns(spectral, orb_start, norb, shells):
    """
    Sum the spin-down and spin-up blocks of the orbital columns of a
    spectral array (column 0 is the total, the norb orbital columns start at
    orb_start). RSPt lays the orbitals out per correlated shell (a spin-down
    block followed by that shell's spin-up block), so pair them with
    spin_split_indices; fall back to a global first-half/second-half split
    when the shells are unknown or do not match the column count
    (single-shell case, where the two layouts coincide). Returns
    (new_spectral, n_summed) where new_spectral keeps the total column
    followed by the n_summed summed orbital columns.
    """
    split = spin_split_indices(shells)
    if split is not None and len(split[0]) + len(split[1]) == norb:
        down_idx, up_idx = split
    else:
        half = norb // 2
        down_idx = list(range(half))
        up_idx = list(range(half, norb))
    summed = (
        spectral[:, :, [orb_start + d for d in down_idx]]
        + spectral[:, :, [orb_start + u for u in up_idx]]
    )
    new_spectral = np.concatenate([spectral[:, :, :1], summed], axis=2)
    return new_spectral, summed.shape[2]


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
