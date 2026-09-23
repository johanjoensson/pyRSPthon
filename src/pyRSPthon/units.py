"""
Physical constants and unit conversions shared across pyRSPthon.
"""

# Rydberg -> eV, the value RSPt itself uses (green_config.F90), so converted
# numbers agree with RSPt's own eV output.
RY_TO_EV = 13.605693009


def energy_scale(from_unit, to_unit):
    """
    Factor that converts energies from from_unit to to_unit ("Ry" or "eV").
    """
    units = {"Ry": 1.0, "eV": RY_TO_EV}
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Unknown energy unit conversion {from_unit!r} -> {to_unit!r}")
    return units[to_unit] / units[from_unit]
