"""
Parser for RSPt's green.inp, mirroring green_init.F90::dmft_readin.

RSPt's own parser has dangerous properties that this module makes visible:

- Block keywords are lowercase and substring-matched anywhere in a line;
  unknown, misspelled, or wrongly-cased keywords are SILENTLY ignored
  (nokeywords -> cycle, green_init.F90:190).
- Some readers consume the line following their block even when it belongs
  to another block: modelexchange swallows its terminating line, spectrum
  with Band/Surf consumes the next line while probing for gp_nkp, unitcell
  always consumes one data line.
- The file is scanned in multiple passes, so a DATA line that happens to
  contain a block keyword substring (e.g. a cluster label like "my_cluster"
  in a tensmom line) is re-dispatched as a block header in another pass.

parse_green_inp() returns both a structured GreenInp and a list of Findings
describing everything RSPt would silently drop or stop on.
"""

import difflib
import re
from dataclasses import dataclass, field

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

# The 18 block keywords (green_init.F90:3656-3678). Dispatch precedence
# follows the if-chain at green_init.F90:192-236.
KEYWORDS = (
    "matsubara",
    "energymesh",
    "inputoutput",
    "convergency",
    "projection",
    "simplemodels",
    "modelexchange",
    "carriers",
    "debug",
    "verbose",
    "observables",
    "mixing",
    "spectrum",
    "susceptibility",
    "isoexch",
    "cluster",
    "unitcell",
    "tensmom",
)

_DISPATCH_ORDER = (
    "matsubara",
    "energymesh",
    "inputoutput",
    "convergency",
    "projection",
    "simplemodels",
    "modelexchange",
    "carriers",
    "debug",
    "verbose",
    "observables",
    "mixing",
    "spectrum",
)

CLUSTER_FLAGS = ("eV", "UJ", "F2def", "Cf", "Virtual", "Cpa", "Pade")
SPECTRUM_FLAGS = (
    "Proj",
    "Hyb",
    "Dos",
    "Fermi",
    "Band",
    "Pband",
    "Surf",
    "Psurf",
    "Nospin",
    "Sproj",
    "Lproj",
    "Jproj",
    "Cartesian",
    "Bothaxes",
    "eV",
    "Eps",
    "Symm",
    "Parity",
    "Entropy",
    "Cf",
    "Obs",
)
TENSMOM_FLAGS = ("Active", "All", "Advanced", "Replace", "Symbrk", "Hamiltonian", "Dc")

# Solver types (green_config.F90:23-35)
SOLVER_MIN, SOLVER_MAX = -2, 10
DYNAMICAL_EXEMPT_SOLVERS = (0, 2, -2)  # observer, LDA+U, carbon copy
SPTF_SOLVER = 1


@dataclass
class Finding:
    level: str
    line: int  # 1-based; 0 = file-level
    message: str
    suggestion: str | None = None

    def __str__(self):
        where = f"line {self.line}: " if self.line else ""
        text = f"{where}{self.message}"
        if self.suggestion:
            text += f" ({self.suggestion})"
        return text


@dataclass
class Orbital:
    line: int
    t: int
    l: int
    e: int
    site: int
    basis: int
    slater: tuple = ()
    correlated: bool = False
    ewin: bool = False
    ewin_spin: bool = False  # Ewinup/Ewindn given (needs spin polarization)
    euler: bool = False


@dataclass
class Cluster:
    line: int  # line of the header (first data) line
    header: str = ""
    ntot: int = 0
    label: str = ""  # from IdXXX; "" means RSPt autogenerates one
    uj: bool = False
    f2def: bool = False
    ev: bool = False
    virtual: bool = False
    cf: bool = False
    npade: int = 0
    orbitals: list = field(default_factory=list)
    ncorr: int = 0
    solver_type: int | None = None
    double_counting: int | None = None
    sigma_mix: float | None = None
    tensmom_mag: float = 0.0
    solver_params: str = ""
    dc_params: str = ""


@dataclass
class Spectrum:
    line: int
    flags: str = ""
    proj: bool = False
    hyb: bool = False
    dos: bool = False
    fermi: bool = False
    band: bool = False
    pband: bool = False
    surf: bool = False
    psurf: bool = False
    ev: bool = False
    eps: bool = False
    symm: bool = False
    entropy: bool = False
    cf: bool = False
    obs: bool = False
    miller: tuple | None = None
    surfnk: tuple | None = None
    gp_nkp: list | None = None
    gp_labels: list | None = None


@dataclass
class GreenInp:
    matsubara: tuple | None = None  # (nmats, head, log, tail)
    energymesh: tuple | None = None  # (size, emin, emax, eim)
    inputoutput: tuple | None = None  # (readsig, cscdmft)
    convergency: tuple | None = None  # (ndelta, sigma_acc, maxiter, maxsolveriter)
    projection: int | None = None  # basis_type
    mixing: tuple | None = None  # (mix_method, ...)
    carriers: float | None = None  # doping
    spectrum: Spectrum | None = None
    clusters: list = field(default_factory=list)
    unitcells: int = 0
    modelexchange: list = field(default_factory=list)  # (label1, label2, idf)
    observables: list = field(default_factory=list)
    tensmom_flags: str | None = None
    tensmom_labels: list = field(default_factory=list)  # cluster labels referenced
    debug: str | None = None
    verbose: str | None = None
    n_suscept: int = 0  # susceptibility + isoexch blocks
    jij: bool = False  # an isoexch/susceptibility block is present


def dispatch_keyword(line):
    """The block keyword RSPt's if-chain would dispatch, or None."""
    for kw in _DISPATCH_ORDER:
        if kw in line:
            return kw
    if "isoexch" in line or "susceptibility" in line:
        return "susceptibility" if "susceptibility" in line else "isoexch"
    if "cluster" in line or "unitcell" in line:
        return "cluster" if "cluster" in line else "unitcell"
    if "tensmom" in line:
        return "tensmom"
    return None


def contained_keywords(line):
    return [kw for kw in KEYWORDS if kw in line]


def strip_comment(raw):
    """Mirror readline (green_init.F90:3591): '!' and '#' start comments."""
    i = raw.find("!")
    j = raw.find("#")
    if i >= 0 and j >= 0:
        raw = raw[: min(i, j)]
    elif i >= 0:
        raw = raw[:i]
    elif j >= 0:
        raw = raw[:j]
    return raw.strip()


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([edED][+-]?\d+)?$")


def fortran_int(tok):
    if not _INT_RE.match(tok):
        raise ValueError(tok)
    return int(tok)


def fortran_float(tok):
    if not _FLOAT_RE.match(tok):
        raise ValueError(tok)
    return float(re.sub("[dD]", "e", tok))


def fortran_logical(tok):
    s = tok.lstrip(".").lower()
    if not s or s[0] not in "tf":
        raise ValueError(tok)
    return s[0] == "t"


_CONVERT = {"i": fortran_int, "f": fortran_float, "l": fortran_logical, "s": str}


def tokenize(line):
    """List-directed tokens: whitespace or comma separated, '/' terminates."""
    tokens = []
    for tok in line.replace(",", " ").split():
        if tok.startswith("/"):
            break
        tokens.append(tok)
    return tokens


def read_values(line, spec):
    """
    Mimic a Fortran list-directed read of len(spec) values ('i'/'f'/'l'/'s').
    Returns (values, err): err = 0 ok, -1 too few values, +1 type mismatch
    (the same sign convention as IOSTAT in dmft_readin's readers). Like
    Fortran, a bad token is a type error even when the line is also short.
    """
    tokens = tokenize(line)
    values = []
    for k, kind in enumerate(spec):
        if k >= len(tokens):
            return None, -1
        try:
            values.append(_CONVERT[kind](tokens[k]))
        except ValueError:
            return None, 1
    return values, 0


class _Parser:
    def __init__(self, text):
        self.lines = [strip_comment(raw) for raw in text.splitlines()]
        self.pos = 0
        self.findings = []
        self.green = GreenInp()

    # --- line stream -------------------------------------------------
    def next(self):
        if self.pos >= len(self.lines):
            return None
        self.pos += 1
        return self.pos, self.lines[self.pos - 1]  # 1-based line number

    def backspace(self):
        self.pos -= 1

    def add(self, level, line, message, suggestion=None):
        self.findings.append(Finding(level, line, message, suggestion))

    def data_line(self, block):
        """
        Fetch the next line as block data (RSPt errors at EOF); also warn if
        a data line contains a keyword substring: RSPt's multi-pass reader
        would re-dispatch it as a block header in another pass.
        """
        item = self.next()
        if item is None:
            self.add(ERROR, len(self.lines), f"{block}: missing data line (RSPt stops: 'Too few arguments in line')")
            return None
        lineno, line = item
        hazards = contained_keywords(line)
        if hazards:
            self.add(
                WARNING,
                lineno,
                f"{block}: this data line contains the block keyword(s) "
                f"{', '.join(repr(k) for k in hazards)}; RSPt's multi-pass reader "
                "will misinterpret it as a new block header",
            )
        return lineno, line

    def report_parse(self, block, lineno, err):
        """Mirror reporterrors (green_init.F90:3630)."""
        if err < 0:
            self.add(ERROR, lineno, f"{block}: too few values on this line (RSPt stops)")
        elif err > 0:
            self.add(ERROR, lineno, f"{block}: wrong value type on this line (RSPt stops)")

    # --- lint helpers ------------------------------------------------
    def lint_header_extras(self, lineno, line, kw):
        extras = [tok for tok in tokenize(line) if kw not in tok]
        if extras:
            self.add(
                WARNING,
                lineno,
                f"extra text on the '{kw}' keyword line is ignored by RSPt "
                "(block data and flags go on the following line(s))",
            )

    def lint_ignored(self, lineno, line):
        tokens = tokenize(line)
        suggestion = None
        if tokens:
            tok = tokens[0]
            if tok.lower() in KEYWORDS:
                suggestion = f"keywords are case-sensitive lowercase: use {tok.lower()!r}"
            else:
                close = difflib.get_close_matches(tok.lower(), KEYWORDS, n=1, cutoff=0.75)
                if close:
                    suggestion = f"did you mean {close[0]!r}?"
        self.add(
            WARNING,
            lineno,
            "RSPt silently ignores this line (no recognized block keyword)",
            suggestion,
        )

    def lint_flags(self, lineno, line, known, block, skip=0):
        """
        Flags are substring-matched, so misspelled flags are silently
        inactive. Warn on near-misses, note other unrecognized tokens.
        skip: number of leading value tokens that are not flags.
        """
        for tok in tokenize(line)[skip:]:
            if any(flag in tok for flag in known):
                continue
            try:
                fortran_float(tok)
                continue  # numeric flag argument (e.g. Pade 100)
            except ValueError:
                pass
            close = difflib.get_close_matches(tok, known, n=1, cutoff=0.75)
            if not close:
                lowered = [f for f in known if f.lower() == tok.lower()]
                close = lowered
            if close:
                self.add(
                    WARNING,
                    lineno,
                    f"{block}: token {tok!r} is not a recognized flag and is "
                    "silently ignored by RSPt (flags are case-sensitive)",
                    f"did you mean {close[0]!r}?",
                )
            else:
                self.add(
                    INFO,
                    lineno,
                    f"{block}: token {tok!r} is not a recognized flag "
                    "(harmlessly ignored by RSPt)",
                )

    def duplicate(self, name, lineno, present):
        if present:
            self.add(
                WARNING,
                lineno,
                f"duplicate '{name}' block: RSPt keeps the values of the last one",
            )

    # --- block readers -------------------------------------------------
    def read_matsubara(self):
        item = self.data_line("matsubara")
        if item is None:
            return
        lineno, line = item
        self.duplicate("matsubara", lineno, self.green.matsubara is not None)
        values, err = read_values(line, "iiii")
        if err < 0:
            values, err = read_values(line, "iii")
            if err == 0:
                nmats, head, log = values
                values = [nmats, head, log, min(max(2, log // 15), head // 3)]
        if err:
            self.report_parse("matsubara", lineno, err)
            return
        self.green.matsubara = (tuple(values), lineno)

    def read_energymesh(self):
        item = self.data_line("energymesh")
        if item is None:
            return
        lineno, line = item
        self.duplicate("energymesh", lineno, self.green.energymesh is not None)
        values, err = read_values(line, "ifff")
        if err < 0:
            values, err = read_values(line, "iff")
            if err == 0:
                values = values + [1e-2]
        if err:
            self.report_parse("energymesh", lineno, err)
            return
        self.green.energymesh = (tuple(values), lineno)

    def read_inputoutput(self):
        item = self.data_line("inputoutput")
        if item is None:
            return
        lineno, line = item
        self.duplicate("inputoutput", lineno, self.green.inputoutput is not None)
        values, err = read_values(line, "ll")
        if err:
            self.report_parse("inputoutput", lineno, err)
            return
        self.green.inputoutput = (tuple(values), lineno)

    def read_convergency(self):
        item = self.data_line("convergency")
        if item is None:
            return
        lineno, line = item
        self.duplicate("convergency", lineno, self.green.convergency is not None)
        values, err = read_values(line, "ffii")
        if err == 0:
            ndelta, sigma_acc, maxiter, maxsolveriter = values
            maxiter = max(maxiter, 0)
            maxsolveriter = max(maxsolveriter, 0)
        elif err > 0:
            # RSPt only falls back when values are MISSING, not malformed
            self.report_parse("convergency", lineno, err)
            return
        else:
            values, err = read_values(line, "ff")
            if err:
                self.report_parse("convergency", lineno, err)
                return
            if len(tokenize(line)) == 3:
                self.add(
                    WARNING,
                    lineno,
                    "convergency: 3 values given but RSPt reads either 4 "
                    "(ndelta sigma_acc maxiter maxsolveriter) or 2 - the third "
                    "value is silently ignored",
                )
            ndelta, sigma_acc = values
            maxiter = None  # RSPt resolves the default later
            maxsolveriter = None
        self.green.convergency = ((ndelta, sigma_acc, maxiter, maxsolveriter), lineno)

    def read_projection(self):
        item = self.data_line("projection")
        if item is None:
            return
        lineno, line = item
        values, err = read_values(line, "i")
        if err:
            self.report_parse("projection", lineno, err)
            return
        self.green.projection = values[0]

    def read_simplemodels(self):
        item = self.data_line("simplemodels")
        if item is None:
            return
        lineno, line = item
        _, err = read_values(line, "ii")
        if err:
            self.report_parse("simplemodels", lineno, err)
        self.data_line("simplemodels (parameter string)")  # free text, consumed

    def read_modelexchange(self):
        entries = []
        while True:
            item = self.next()
            if item is None:
                break
            lineno, line = item
            values, err = read_values(line, "ssf")
            if err:
                # RSPt consumes this terminating line WITHOUT backspacing:
                # whatever it is, the main loop never sees it again.
                if contained_keywords(line):
                    self.add(
                        ERROR,
                        lineno,
                        "this block-keyword line is swallowed by RSPt's "
                        "modelexchange reader (it consumes its terminating line); "
                        "the block is lost or breaks the cluster bookkeeping",
                        "put a plain data/blank line between modelexchange and the next block",
                    )
                    # Model RSPt: the keyword line is gone.
                elif line:
                    self.add(
                        WARNING,
                        lineno,
                        "this line terminates the modelexchange block and is "
                        "consumed by RSPt without being interpreted",
                    )
                break
            entries.append((values[0], values[1], values[2], lineno))
        self.green.modelexchange.extend(entries)

    def read_carriers(self):
        item = self.data_line("carriers")
        if item is None:
            return
        lineno, line = item
        values, err = read_values(line, "f")
        if err:
            self.report_parse("carriers", lineno, err)
            return
        self.green.carriers = values[0]

    def read_debug(self):
        item = self.data_line("debug")
        if item is None:
            return
        _, line = item
        self.green.debug = line

    def read_verbose(self):
        item = self.data_line("verbose")
        if item is None:
            return
        _, line = item
        self.green.verbose = line

    def read_observables(self):
        item = self.data_line("observables")
        if item is None:
            return
        lineno, line = item
        names = tokenize(line)
        if not names:
            self.add(ERROR, lineno, "observables: empty word list (RSPt stops)")
            return
        self.green.observables = names
        if "Photon" in line:
            item = self.data_line("observables (photon)")
            if item is None:
                return
            lineno, line = item
            for spec in ("fffff", "fff"):
                values, err = read_values(line, spec)
                if err >= 0:
                    break
            if err > 0:
                self.add(
                    ERROR,
                    lineno,
                    "observables: Photon requires a following line with "
                    "k(1:3) [helicity]; RSPt stops trying to parse this line",
                )

    def read_mixing(self):
        item = self.data_line("mixing")
        if item is None:
            return
        lineno, line = item
        self.duplicate("mixing", lineno, self.green.mixing is not None)
        for spec in ("ifff", "iff", "if", "i"):
            values, err = read_values(line, spec)
            if err >= 0:
                break
        if err:
            self.report_parse("mixing", lineno, err)
            return
        self.green.mixing = (tuple(values), lineno)

    def read_spectrum(self):
        item = self.data_line("spectrum (flags)")
        if item is None:
            return
        lineno, line = item
        self.duplicate("spectrum", lineno, self.green.spectrum is not None)
        spec = Spectrum(line=lineno, flags=line)
        spec.proj = "Proj" in line
        spec.hyb = "Hyb" in line
        spec.dos = "Dos" in line
        spec.fermi = "Fermi" in line
        spec.band = "Band" in line
        spec.pband = "Pband" in line
        spec.surf = "Surf" in line
        spec.psurf = "Psurf" in line
        spec.ev = "eV" in line
        spec.eps = "Eps" in line
        spec.symm = "Symm" in line or "Parity" in line
        spec.entropy = "Entropy" in line
        spec.cf = "Cf" in line
        spec.obs = "Obs" in line
        if spec.pband:
            spec.band = True
        if spec.psurf:
            spec.surf = True
        if (spec.symm or spec.entropy) and not (spec.band or spec.surf):
            spec.dos = True
        if "Bothaxes" in line and (spec.band or spec.pband):
            self.add(
                ERROR,
                lineno,
                "spectrum: Bothaxes does not work for band structures (RSPt stops)",
            )
        self.lint_flags(lineno, line, SPECTRUM_FLAGS, "spectrum")

        if spec.surf:
            item = self.data_line("spectrum (surface)")
            if item is not None:
                lineno2, line2 = item
                values, err = read_values(line2, "iiifiii")
                if err == 0:
                    spec.miller = tuple(values[0:3])
                    spec.surfnk = tuple(values[4:7])
                elif err > 0:
                    self.add(
                        ERROR,
                        lineno2,
                        "spectrum: could not parse the surface parameters "
                        "(miller(1:3) kz/bz [nk(1:3)]); RSPt stops",
                    )
                else:
                    values, err = read_values(line2, "iiif")
                    if err == 0:
                        spec.miller = tuple(values[0:3])
                        if spec.psurf:
                            self.add(
                                ERROR,
                                lineno2,
                                "spectrum: Psurf requires nk(1:3) > 0 on the "
                                "surface parameter line (RSPt stops)",
                            )
                    else:
                        self.add(
                            ERROR,
                            lineno2,
                            "spectrum: could not parse the surface parameters "
                            "(miller(1:3) kz/bz [nk(1:3)]); RSPt stops",
                        )

        if spec.band or spec.surf:
            # RSPt consumes the next line probing for gp_nkp segment counts;
            # if it is not a list of integers it is LOST (no backspace).
            item = self.next()
            if item is not None:
                lineno2, line2 = item
                tokens = tokenize(line2)
                segments = None
                for i in range(min(len(tokens), 20), 0, -1):
                    values, err = read_values(line2, "i" * i)
                    if err == 0:
                        segments = values
                        break
                if segments is not None:
                    spec.gp_nkp = segments
                    item = self.data_line("spectrum (gp_labels)")
                    if item is None:
                        self.add(
                            ERROR,
                            lineno2,
                            "spectrum: gp_nkp given but the symmetry-point "
                            "label line is missing (RSPt stops)",
                        )
                    else:
                        lineno3, line3 = item
                        labels = tokenize(line3)
                        spec.gp_labels = labels
                        if len(labels) < len(segments) + 1:
                            self.add(
                                ERROR,
                                lineno3,
                                f"spectrum: {len(segments)} segment counts need "
                                f"{len(segments) + 1} symmetry-point labels, got "
                                f"{len(labels)} (RSPt stops)",
                            )
                        elif len(labels) > len(segments) + 1:
                            self.add(
                                WARNING,
                                lineno3,
                                f"spectrum: {len(labels)} labels given but only "
                                f"{len(segments) + 1} are used "
                                "(gp_nkp segments + 1)",
                            )
                elif contained_keywords(line2):
                    self.add(
                        ERROR,
                        lineno2,
                        "this block-keyword line is consumed by RSPt's spectrum "
                        "reader while probing for the gp_nkp line (Band/Surf); "
                        "the block is lost or breaks the cluster bookkeeping",
                        "put the gp_nkp + label lines (or a blank line) after the spectrum flags",
                    )
                elif line2:
                    self.add(
                        WARNING,
                        lineno2,
                        "spectrum: this line is consumed while probing for "
                        "gp_nkp and discarded; RSPt falls back to plain band output",
                    )
        self.green.spectrum = spec

    def read_unitcell(self, kwline):
        # readunitcell always consumes exactly one data line, even a keyword line.
        item = self.next()
        if item is None:
            return
        lineno, line = item
        if contained_keywords(line):
            self.add(
                ERROR,
                lineno,
                "this block-keyword line is consumed as the unitcell data line "
                "(nbasis [Cf]); the block is lost or breaks RSPt's cluster "
                "bookkeeping",
                "put an explicit basis line (e.g. '0') after 'unitcell'",
            )
        self.green.unitcells += 1

    def read_suscept(self, keyword):
        self.green.n_suscept += 1
        self.green.jij = True
        if self.green.n_suscept > 1:
            self.add(
                ERROR,
                self.pos,
                "two susceptibility/isoexch blocks: RSPt stops",
            )
        if keyword == "susceptibility":
            item = self.data_line("susceptibility")
            if item is None:
                return
            lineno, line = item
            values, err = read_values(line, "i")
            if err:
                self.report_parse("susceptibility", lineno, err)
                return
            for _ in range(values[0]):
                item = self.data_line("susceptibility (position)")
                if item is None:
                    return
                lineno, line = item
                _, err = read_values(line, "fff")
                if err:
                    self.add(
                        ERROR,
                        lineno,
                        "susceptibility: expected a position line (x y z [label])",
                    )
        else:  # isoexch: proj, center atom, radius, n neighbour types, neighbour list
            for name, spec in (
                ("isoexch (proj)", "i"),
                ("isoexch (center atom)", "i"),
                ("isoexch (radius)", "f"),
                ("isoexch (neighbour types)", "i"),
                ("isoexch (neighbour list)", "i"),
            ):
                item = self.data_line(name)
                if item is None:
                    return
                lineno, line = item
                _, err = read_values(line, spec)
                if err:
                    self.report_parse(name, lineno, err)

    def read_tensmom(self):
        item = self.data_line("tensmom (flags)")
        if item is None:
            return
        lineno, line = item
        self.green.tensmom_flags = line
        self.lint_flags(lineno, line, TENSMOM_FLAGS, "tensmom")
        advanced = "Advanced" in line
        while True:
            item = self.next()
            if item is None:
                break
            lineno, line = item
            if advanced:
                ok = False
                for spec in ("iiiisff", "iiiis", "iiii"):
                    values, err = read_values(line, spec)
                    if err == 0:
                        ok = True
                        if len(spec) >= 5:
                            self.green.tensmom_labels.append((values[4], lineno))
                        break
            else:
                ok = False
                for spec in ("iiiff", "iiif", "iii"):
                    values, err = read_values(line, spec)
                    if err == 0:
                        ok = True
                        break
            if not ok:
                self.backspace()  # readtensmom backspaces its terminating line
                break

    def read_cluster(self, kwline):
        item = self.data_line("cluster (header)")
        if item is None:
            return
        lineno, line = item
        cl = Cluster(line=lineno, header=line)
        cl.uj = "UJ" in line
        cl.f2def = "F2def" in line
        cl.ev = "eV" in line
        cl.virtual = "Virtual" in line or "Cpa" in line
        cl.cf = "Cf" in line
        tokens = tokenize(line)
        for i, tok in enumerate(tokens):
            if tok.startswith("Id"):
                cl.label = tok[2:]
            elif tok.startswith("Pade") or tok.startswith("Cpa"):
                if i + 1 >= len(tokens):
                    self.add(ERROR, lineno, f"cluster: {tok} needs a following value (RSPt stops)")
                elif tok.startswith("Pade"):
                    try:
                        cl.npade = fortran_int(tokens[i + 1])
                    except ValueError:
                        self.add(ERROR, lineno, "cluster: Pade needs an integer value (RSPt stops)")
        # Header numbers: ntot [udef [udefcalc]] (progressively optional)
        uflag = True
        ujflag = cl.uj
        values, err = read_values(line, "iii")
        if err == 0:
            ntot, udef, udefcalc = values
            ujflag = udef == 2 and udefcalc == 0
            uflag = udef >= 1
        else:
            values, err = read_values(line, "ii")
            if err == 0:
                ntot, udef = values
                ujflag = udef == 2
                uflag = udef >= 1
            else:
                values, err = read_values(line, "i")
                if err:
                    self.report_parse("cluster", lineno, err)
                    return
                ntot = values[0]
        self.lint_flags(lineno, line, CLUSTER_FLAGS + ("Id", "Irr"), "cluster", skip=0)
        if ntot <= 0:
            self.add(ERROR, lineno, f"cluster: ntot = {ntot}, need at least one atom (RSPt stops)")
            return
        cl.ntot = ntot
        cl.uj = cl.uj or ujflag

        # ntot orbital lines: t l e site basis [F0 [F2 [F4 [F6]]]] [Ewin...] [Euler...]
        for _ in range(ntot):
            item = self.data_line("cluster (orbital)")
            if item is None:
                return
            lineno, line = item
            ewin = "Ewin" in line
            ewin_spin = "Ewinup" in line or "Ewindn" in line
            euler = "Euler" in line
            if ("Ewinup" in line) != ("Ewindn" in line):
                self.add(ERROR, lineno, "cluster: Ewinup and Ewindn must be specified together (RSPt stops)")
            cut = len(line)
            for word in ("Euler", "Ewin"):
                i = line.find(word)
                if i > 0:
                    cut = min(cut, i)
                elif i == 0:
                    self.add(ERROR, lineno, f"cluster: orbital line starts with {word} - no orbital ids (RSPt stops)")
            data = line[:cut]
            slater = ()
            correlated = False
            slater_counts = (2, 1) if cl.uj else (4, 3, 2, 1)
            for nsl in slater_counts:
                values, err = read_values(data, "iiiii" + "f" * nsl)
                if err == 0:
                    slater = tuple(values[5:])
                    correlated = True
                    if nsl == 2 and not cl.f2def and not cl.uj:
                        self.add(
                            ERROR,
                            lineno,
                            "cluster: two Slater values are read as F0 and F2; "
                            "add the UJ flag (values are U J) or the F2def flag "
                            "(values really are F0 F2) - RSPt stops otherwise",
                        )
                    break
            if not correlated:
                values, err = read_values(data, "iiiii")
                if err:
                    self.report_parse("cluster (orbital)", lineno, err)
                    return
                uflag = False
            else:
                if not uflag:
                    self.add(
                        ERROR,
                        lineno,
                        "cluster: correlated orbital lines (with Slater/U values) "
                        "must come before uncorrelated ones (RSPt stops)",
                    )
                cl.ncorr += 1
            cl.orbitals.append(
                Orbital(
                    line=lineno,
                    t=values[0],
                    l=values[1],
                    e=values[2],
                    site=values[3],
                    basis=values[4],
                    slater=slater,
                    correlated=correlated,
                    ewin=ewin,
                    ewin_spin=ewin_spin,
                    euler=euler,
                )
            )

        if cl.ncorr >= 1:
            # solver line: solver_type double_counting sigma_mix [tensmom_mag]
            item = self.data_line("cluster (solver)")
            if item is None:
                self.green.clusters.append(cl)
                return
            lineno, line = item
            values, err = read_values(line, "iiff")
            if err:
                values, err = read_values(line, "iif")
            if err:
                self.add(
                    ERROR,
                    lineno,
                    "cluster: could not parse the solver line "
                    "(solver_type double_counting sigma_mix [tensmom_mag]) - RSPt stops",
                )
                self.green.clusters.append(cl)
                return
            cl.solver_type, cl.double_counting, cl.sigma_mix = values[0:3]
            cl.tensmom_mag = values[3] if len(values) > 3 else 0.0
            # optional solver / dc parameter lines: consumed only when they
            # do not contain a block keyword (green_init.F90:1948-1979)
            for attr in ("solver_params", "dc_params"):
                item = self.next()
                if item is None:
                    break
                _, line = item
                if contained_keywords(line):
                    self.backspace()
                    break
                setattr(cl, attr, line)
        self.green.clusters.append(cl)

    # --- main loop -----------------------------------------------------
    def parse(self):
        readers = {
            "matsubara": self.read_matsubara,
            "energymesh": self.read_energymesh,
            "inputoutput": self.read_inputoutput,
            "convergency": self.read_convergency,
            "projection": self.read_projection,
            "simplemodels": self.read_simplemodels,
            "modelexchange": self.read_modelexchange,
            "carriers": self.read_carriers,
            "debug": self.read_debug,
            "verbose": self.read_verbose,
            "observables": self.read_observables,
            "mixing": self.read_mixing,
            "spectrum": self.read_spectrum,
            "tensmom": self.read_tensmom,
        }
        while (item := self.next()) is not None:
            lineno, line = item
            if not line:
                continue
            kw = dispatch_keyword(line)
            if kw is None:
                self.lint_ignored(lineno, line)
                continue
            self.lint_header_extras(lineno, line, kw)
            if kw == "cluster":
                self.read_cluster(line)
            elif kw == "unitcell":
                self.read_unitcell(line)
            elif kw in ("susceptibility", "isoexch"):
                self.read_suscept(kw)
            else:
                readers[kw]()
        return self.green, self.findings


def parse_green_text(text):
    """Parse green.inp content. Returns (GreenInp, list[Finding])."""
    return _Parser(text).parse()


def parse_green_inp(path):
    """Parse a green.inp file. Returns (GreenInp, list[Finding])."""
    with open(path, "rt") as f:
        return parse_green_text(f.read())
