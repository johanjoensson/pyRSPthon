"""
Minimal reader for RSPt's `data` file: only the quantities that input
verification needs. The data file is machine-generated (symt -all + stick),
so the layout is stable; everything here is still parsed defensively and
left as None when a quantity cannot be found - dependent checks are then
skipped instead of misfiring.

Layout facts (rspt/rsptDir/src/readin.F):
- line 2 is the 72-character prnt flag string, read with "(/72l1)" (:82)
- the "lmax ntype ..." header line is followed by the values line whose
  last four t/f fields are win, wmt, f-rel, sp-po
- each type section starts "TYPE n: species: <label>"; the line after the
  "natom nharm" header starts with natom
- the "<nb> Bases" line is followed by nb lines "l ie tail" (:993-1000) -
  the exact basis list the cluster (t l e ...) ids are checked against
"""

from dataclasses import dataclass, field


@dataclass
class TypeInfo:
    label: str = ""
    natom: int | None = None
    bases: set = field(default_factory=set)  # of (l, energy_set)


@dataclass
class RSPtData:
    prnt: str = ""  # 72 't'/'f' characters ("" when not found)
    lmax: int | None = None
    ntype: int | None = None
    fullrel: bool | None = None
    spinpol: bool | None = None
    types: list = field(default_factory=list)

    @property
    def spin_average(self):
        """prnt(11): polarize the DMFT density of a non-spinpol run."""
        return len(self.prnt) >= 11 and self.prnt[10] == "t"


def read_data(fname="data"):
    """
    Extract verification-relevant quantities from an RSPt data file.
    Returns RSPtData; missing quantities stay None/empty.
    """
    with open(fname, "rt") as f:
        lines = f.read().splitlines()

    data = RSPtData()
    for line in lines[:4]:
        s = line.strip()
        if len(s) >= 40 and set(s) <= {"t", "f"}:
            data.prnt = s
            break

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        tokens = stripped.split()
        if "lmax" in line and "ntype" in line and data.ntype is None:
            values = lines[i + 1].split() if i + 1 < len(lines) else []
            try:
                data.lmax = int(values[0])
                data.ntype = int(values[1])
            except (IndexError, ValueError):
                pass
            logicals = [tok for tok in values if tok.lower() in ("t", "f")]
            if len(logicals) >= 4:
                data.fullrel = logicals[2].lower() == "t"
                data.spinpol = logicals[3].lower() == "t"
            i += 1
        elif stripped.startswith("TYPE") and "species" in line:
            info = TypeInfo(label=line.split("species:")[-1].strip())
            data.types.append(info)
            # the natom header follows; the value line comes after it
            if i + 2 < len(lines) and "natom" in lines[i + 1]:
                try:
                    info.natom = int(lines[i + 2].split()[0])
                except (IndexError, ValueError):
                    pass
        elif len(tokens) == 2 and tokens[1] == "Bases" and data.types:
            try:
                nb = int(tokens[0])
            except ValueError:
                nb = 0
            for j in range(i + 1, min(i + 1 + nb, len(lines))):
                values = lines[j].split()
                try:
                    l, ie = int(values[0]), int(values[1])
                except (IndexError, ValueError):
                    break
                data.types[-1].bases.add((l, ie))
            i += nb
        i += 1

    if data.ntype is None and data.types:
        data.ntype = len(data.types)
    return data
