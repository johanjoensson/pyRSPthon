"""
Band-structure plotting for the models of pyRSPthon.read.bands.

All functions draw onto a given matplotlib Axes and never modify the model.
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

# Matches the palette RSPt itself puts in band.gpi (a YlGnBu-style ramp).
SPECTRAL_CMAP = "YlGnBu"


def energy_label(model):
    """Y-axis label: relative to E_F only when the energies are shifted."""
    unit = model.energy_unit
    if model.shifted:
        return rf"E - E$_F$ ({unit})"
    return f"E ({unit})"


def decorate_axes(ax, model, emin=None, emax=None):
    """
    Common band-plot decorations: symmetry-point grid, Fermi line, labels.
    """
    kpath = model.kpath
    if model.kind == "spectral":
        lo, hi = model.energies[0], model.energies[-1]
    else:
        lo, hi = np.nanmin(model.bands), np.nanmax(model.bands)
    ax.set_ylim(lo if emin is None else emin, hi if emax is None else emax)
    fermi = model.fermi_level
    if fermi is not None:
        ax.axhline(fermi, linestyle="dotted", color="gray", linewidth=1)
    for tick in kpath.ticks:
        ax.axvline(tick, linestyle="dotted", color="gray", linewidth=1)
    if kpath.ticks:
        labels = kpath.tick_labels if any(kpath.tick_labels) else [""] * len(kpath.ticks)
        ax.set_xticks(kpath.ticks, labels)
    ax.set_xlim(kpath.kdist[0], kpath.kdist[-1])
    ax.set_ylabel(energy_label(model))


def _image_extent(model):
    """
    imshow extent that puts pixel centres on the (k, E) mesh points.
    """
    kd, e = model.kpath.kdist, model.energies
    dk = (kd[-1] - kd[0]) / (len(kd) - 1) / 2 if len(kd) > 1 else 0.5
    de = (e[-1] - e[0]) / (len(e) - 1) / 2 if len(e) > 1 else 0.5
    return (kd[0] - dk, kd[-1] + dk, e[0] - de, e[-1] + de)


def plot_spectral(ax, model, data=None, log=False, vmin=None, vmax=None, cmap=SPECTRAL_CMAP):
    """
    Intensity plot of a SpectralBands. data is an (ne, nk) array on the
    model's mesh (default: the total spectral weight).
    """
    if data is None:
        data = model.total
    if log:
        floor = max(float(np.max(data)) * 1e-6, np.finfo(float).tiny)
        data = np.log(np.maximum(data, floor))
    return ax.imshow(
        data[::-1],
        extent=_image_extent(model),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def plot_composite(ax, model, weights, colors=None, gamma=1.0):
    """
    RGB-composite plot of several (ne, nk) weight columns (weights has shape
    (ne, nk, ncol)): each column tints the image with its own color, alpha
    proportional to its weight on one scale shared by all columns, so the
    tints show the relative orbital character.
    """
    from matplotlib.colors import to_rgb

    if colors is None:
        colors = ORBITAL_COLORS
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
    ax.imshow(img[::-1], extent=_image_extent(model), aspect="auto")


def plot_lines(ax, model, color="#0072B2", linewidth=1.2, label=None):
    """
    Eigenvalue-line plot of a LineBands.
    """
    lines = ax.plot(model.kpath.kdist, model.bands, color=color, linewidth=linewidth)
    if label and lines:
        lines[0].set_label(label)
    return lines


def plot_fatbands(ax, model, columns=None, labels=None, scale=30.0, colors=None):
    """
    Scatter fatbands on top of the bands: marker area proportional to the
    projection weight. columns picks projection columns (default: all).
    """
    if colors is None:
        colors = ORBITAL_COLORS
    proj = model.projections
    if columns is None:
        columns = list(range(proj.weights.shape[-1]))
    if labels is None:
        labels = [proj.labels[c] if c < len(proj.labels) else f"proj {c}" for c in columns]
    plot_lines(ax, model, color="0.6", linewidth=0.6)
    if not columns:
        return
    w = np.abs(proj.weights[..., columns])
    wmax = float(np.nanmax(w)) or 1.0
    nk, nbands = model.bands.shape
    kd = np.repeat(model.kpath.kdist, nbands)
    energies = model.bands.ravel()
    for i, lab in enumerate(labels):
        ax.scatter(
            kd,
            energies,
            s=scale * w[..., i].ravel() / wmax,
            color=colors[i % len(colors)],
            label=lab,
            alpha=0.7,
            edgecolors="none",
        )
    ax.legend(loc="upper right", fontsize="small")
