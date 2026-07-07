"""
Reader for the header of the binary 'sig' file written by RSPt
(selfenergy_write in green_selfenergy.F90).

The sig header is the authoritative source for the self-energy convergence
right after an RSPt run: the real-axis solvers run in the spectrum loop and
store their fresh sigdiff_real only here. The 'Sigdiff (mats/real)' lines in
dmft_hist are written during the DMFT mu-loop from the *previous* run's sig
header, so they lag one RSPt invocation behind.
"""

import struct
from dataclasses import dataclass

# Fortran unformatted sequential records: <int32 length> payload <int32 length>.
_MARKER = struct.Struct("<i")
# Version >= 7 header record (green_selfenergy.F90, selfenergy_readheader):
# line0, n0, t0, emeshsize0, nfermi0, emeshdiff0, green_mu, green_mu_old,
# hsig_updated, nel_old, nel, sigdiff_mats, sigdiff_real, sig_green_mu,
# sig_gave_lda_mats, sig_gave_lda_real
_HEADER_V7 = struct.Struct("<2id2i3di4d3i")


@dataclass
class SigHeader:
    version: int
    nlines: int  # number of clusters (line0)
    nmats: int  # matsubara mesh size (n0)
    temperature: float  # t0
    emeshsize: int
    nfermi: int
    emeshdiff: float
    green_mu: float
    green_mu_old: float
    hsig_updated: bool
    nel_old: float
    nel: float
    sigdiff_mats: float
    sigdiff_real: float
    sig_green_mu: bool
    sig_gave_lda_mats: bool
    sig_gave_lda_real: bool


def read_sig_header(path="sig"):
    """
    Read the header of an RSPt 'sig' file.

    Returns a SigHeader, or None if the file is missing, truncated, not a
    sig file, or older than version 7 (no sigdiff_real stored). Never raises.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(5 * _MARKER.size + _HEADER_V7.size)
    except OSError:
        return None
    # Record 1: the sig file version
    if len(data) < 2 * _MARKER.size + 4:
        return None
    (reclen,) = _MARKER.unpack_from(data, 0)
    if reclen != 4:
        return None
    (version,) = _MARKER.unpack_from(data, _MARKER.size)
    (endlen,) = _MARKER.unpack_from(data, 2 * _MARKER.size)
    if endlen != reclen or version < 7:
        return None
    # Record 2: the header fields
    offset = 3 * _MARKER.size
    if len(data) < offset + _MARKER.size:
        return None
    (reclen,) = _MARKER.unpack_from(data, offset)
    if reclen != _HEADER_V7.size:
        return None
    if len(data) < offset + 2 * _MARKER.size + reclen:
        return None
    (endlen,) = _MARKER.unpack_from(data, offset + _MARKER.size + reclen)
    if endlen != reclen:
        return None
    fields = _HEADER_V7.unpack_from(data, offset + _MARKER.size)
    (
        nlines,
        nmats,
        temperature,
        emeshsize,
        nfermi,
        emeshdiff,
        green_mu,
        green_mu_old,
        hsig_updated,
        nel_old,
        nel,
        sigdiff_mats,
        sigdiff_real,
        sig_green_mu,
        sig_gave_lda_mats,
        sig_gave_lda_real,
    ) = fields
    return SigHeader(
        version=version,
        nlines=nlines,
        nmats=nmats,
        temperature=temperature,
        emeshsize=emeshsize,
        nfermi=nfermi,
        emeshdiff=emeshdiff,
        green_mu=green_mu,
        green_mu_old=green_mu_old,
        hsig_updated=bool(hsig_updated),
        nel_old=nel_old,
        nel=nel,
        sigdiff_mats=sigdiff_mats,
        sigdiff_real=sigdiff_real,
        sig_green_mu=bool(sig_green_mu),
        sig_gave_lda_mats=bool(sig_gave_lda_mats),
        sig_gave_lda_real=bool(sig_gave_lda_real),
    )
