"""
Shared plot colors and styles, chosen to survive colour-vision deficiency
and, with the redundant line styles and markers, grayscale printing.
"""

# Paul Tol's "medium contrast" scheme: of the common categorical sets it
# keeps the largest CVD-simulated and grayscale separation for the first
# 3-4 entries. Tol's first two are swapped so that the pale yellow, faint
# on white as a thin line, is not the second color. Beyond ~4 colors no
# palette stays apart in grayscale; pair colors with LINESTYLES/MARKERS.
CATEGORICAL = [
    "#004488",  # dark blue
    "#994455",  # dark red
    "#EECC66",  # light yellow
    "#6699CC",  # light blue
    "#997700",  # dark yellow
    "#EE99AA",  # light red
]

# Alpha composites mix the tints, and the mixtures stay distinguishable
# under CVD only for the first four colors.
COMPOSITE = CATEGORICAL[:4]

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Lightness falls monotonically from white (L* 99 -> 13), so intensities
# read correctly under CVD and in grayscale; it is also the ramp RSPt puts
# in band.gpi. cividis_r is more uniform but has a saturated yellow zero.
SPECTRAL_CMAP = "YlGnBu"


def color(i):
    return CATEGORICAL[i % len(CATEGORICAL)]


def linestyle(i):
    return LINESTYLES[i % len(LINESTYLES)]


def marker(i):
    return MARKERS[i % len(MARKERS)]
