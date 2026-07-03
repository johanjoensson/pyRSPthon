import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyRSPthon.read.data import read_data
from pyRSPthon.read.greeninp import ERROR, INFO, WARNING, parse_green_inp, parse_green_text
from pyRSPthon.verify.green import RunContext, check_intra, check_time_reversal, verify_green


def errors(findings):
    return [f for f in findings if f.level == ERROR]


def warnings_(findings):
    return [f for f in findings if f.level == WARNING]


# --- the three real RSPt examples must parse without errors ------------


@pytest.mark.parametrize(
    "fixture", ["green_cefe2.inp", "green_nio_ldau.inp", "green_bccfe_sptf.inp"]
)
def test_real_files_parse_clean(fixtures, fixture):
    green, findings = parse_green_inp(os.path.join(fixtures, fixture))
    assert not errors(findings), [str(f) for f in errors(findings)]
    assert not errors(check_intra(green))


def test_cefe2_values(fixtures):
    green, _ = parse_green_inp(os.path.join(fixtures, "green_cefe2.inp"))
    assert green.matsubara[0] == (2048, 64, 128, 8)
    assert green.energymesh[0] == (2012, -1.0, 1.0, 0.01)
    assert green.mixing[0] == (3, 0.15, 0.2)
    assert green.projection == 1
    assert len(green.clusters) == 3
    for cl in green.clusters:
        assert cl.uj and cl.ev and cl.ncorr == 1
        assert cl.solver_type == 1 and cl.double_counting == -1
        assert cl.dc_params == ".false."


def test_nio_values(fixtures):
    green, _ = parse_green_inp(os.path.join(fixtures, "green_nio_ldau.inp"))
    assert green.convergency[0][:2] == (1e-6, 1e-4)
    assert [cl.solver_type for cl in green.clusters] == [2, 2]
    orb = green.clusters[0].orbitals[0]
    assert (orb.t, orb.l, orb.e, orb.site, orb.basis) == (1, 2, 1, 1, 0)
    assert orb.slater == (0.59, 0.60, 0.38)


def test_bccfe_values(fixtures):
    green, _ = parse_green_inp(os.path.join(fixtures, "green_bccfe_sptf.inp"))
    (cl,) = green.clusters
    assert cl.uj and cl.ev
    assert cl.solver_type == 1 and cl.sigma_mix == 0.6
    assert cl.solver_params == "4 100 1e-1 0.8d0"


# --- structural lint: what RSPt silently ignores ------------------------


def test_case_error_is_flagged():
    _, findings = parse_green_text("Matsubara\n2048 64 128 8\n")
    (f,) = [f for f in findings if f.suggestion and "case-sensitive" in f.suggestion]
    assert f.level == WARNING and f.line == 1


def test_misspelled_keyword_suggestion():
    _, findings = parse_green_text("matsubarra\n2048 64 128 8\n")
    assert any(f.suggestion == "did you mean 'matsubara'?" for f in findings)


def test_misspelled_flag_suggestion():
    _, findings = parse_green_text("spectrum\nDos Pbnad\n")
    assert any(f.suggestion == "did you mean 'Pband'?" for f in findings)


def test_extra_text_on_keyword_line():
    _, findings = parse_green_text("spectrum Band\nDos\n")
    assert any("keyword line is ignored" in f.message for f in findings)


def test_modelexchange_swallows_next_block():
    text = "modelexchange\nA B 0.5\nmatsubara\n512 16 40 10\n"
    green, findings = parse_green_text(text)
    assert green.matsubara is None  # exactly what RSPt would do
    assert any(
        f.level == ERROR and "swallowed" in f.message and f.line == 3 for f in findings
    )


def test_spectrum_band_consumes_next_line():
    text = "spectrum\nBand\nenergymesh\n501 -1.0 1.0\n"
    green, findings = parse_green_text(text)
    assert green.energymesh is None
    assert any(f.level == ERROR and "probing for the gp_nkp" in f.message for f in findings)


def test_data_line_keyword_hazard():
    # a debug string containing 'cluster' is re-dispatched in RSPt's pass 2
    _, findings = parse_green_text("debug\nOnesite cluster\n")
    assert any("multi-pass" in f.message for f in findings)


def test_gp_labels_count():
    text = "spectrum\nBand\n20 20 20\nG X W\n"
    _, findings = parse_green_text(text)
    assert any(f.level == ERROR and "symmetry-point labels" in f.message for f in findings)


# --- intra-file semantic checks -----------------------------------------


def intra(text):
    green, findings = parse_green_text(text)
    return green, findings + check_intra(green)


CLUSTER = "cluster\n1 IdX\n1 2 1 1 0 2.0\n{solver}\n"


def test_bad_mix_method():
    _, findings = intra("mixing\n5 0.15\n")
    assert any("unknown mix_method 5" in f.message for f in errors(findings))


def test_unknown_solver_becomes_observer():
    _, findings = intra(CLUSTER.format(solver="42 0 0.5"))
    assert any("unknown solver_type 42" in f.message for f in warnings_(findings))


def test_negative_sigma_mix():
    _, findings = intra(CLUSTER.format(solver="2 2 -0.5"))
    assert any("negative sigma_mix" in f.message for f in errors(findings))


def test_udef2_needs_uj_or_f2def():
    _, findings = intra("cluster\n1\n1 2 1 1 0 2.0 0.6\n2 2 0.5\n")
    assert any("UJ flag" in f.message for f in errors(findings))


def test_band_and_dos_exclusive():
    _, findings = intra("spectrum\nBand Dos\n\n\n")
    assert any("mutually exclusive" in f.message for f in errors(findings))


def test_duplicate_cluster_ids():
    text = (
        "cluster\n1 IdX\n1 2 1 1 0 2.0\n2 2 0.5\n\n"
        "cluster\n1 IdX\n2 2 1 1 0 2.0\n2 2 0.5\n"
    )
    _, findings = intra(text)
    assert any("duplicate cluster Id 'X'" in f.message for f in errors(findings))


def test_nmats_smaller_than_parts():
    _, findings = intra("matsubara\n32 16 40 10\n")
    assert any("head+log+tail" in f.message for f in warnings_(findings))


def test_nmats_one_with_dynamical_solver():
    text = "matsubara\n1 1 0 0\n" + CLUSTER.format(solver="4 1 0.5")
    _, findings = intra(text)
    assert any("nmats = 1" in f.message for f in errors(findings))


def test_correlated_lines_must_come_first():
    text = "cluster\n2 IdX\n1 2 1 1 0\n1 2 1 1 0 2.0\n2 2 0.5\n"
    _, findings = intra(text)
    assert any("must come before" in f.message for f in errors(findings))


# --- cross-file checks ---------------------------------------------------


def test_data_reader(fixtures):
    data = read_data(os.path.join(fixtures, "data_nio"))
    assert data.ntype == 3 and data.lmax == 8
    assert data.fullrel is False and data.spinpol is True
    assert [t.natom for t in data.types] == [1, 1, 2]
    assert (2, 1) in data.types[0].bases
    assert (2, 1) not in data.types[2].bases  # oxygen has no d basis


def test_cluster_ids_against_data(fixtures, tmp_path):
    (tmp_path / "data").write_text(
        (open(os.path.join(fixtures, "data_nio")).read())
    )
    # t out of range, e out of range, and l without a basis on oxygen
    (tmp_path / "green.inp").write_text(
        "cluster\n1 IdA\n5 2 1 1 0 2.0\n2 2 0.5\n\n"
        "cluster\n1 IdB\n1 2 2 1 0 2.0\n2 2 0.5\n\n"
        "cluster\n1 IdC\n3 2 1 1 0 2.0\n2 2 0.5\n"
    )
    findings = verify_green(str(tmp_path))
    messages = [f.message for f in errors(findings)]
    assert any("t = 5 out of range" in m for m in messages)
    assert any("energy set e = 2" in m for m in messages)
    assert any("l = 2" in m and "type 3" in m for m in messages)


def test_verify_without_companions(tmp_path):
    (tmp_path / "green.inp").write_text("cluster\n1 IdX\n1 2 1 1 0 2.0\n2 2 0.5\n")
    findings = verify_green(str(tmp_path))
    assert not errors(findings)


def test_band_with_scf_mesh_warns(tmp_path):
    from pyRSPthon.kpts.spts import write_spts

    (tmp_path / "green.inp").write_text("spectrum\nBand\n20 20\nG X W\n")
    write_spts(
        tmp_path / "spts",
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        np.array([1, 8]),
    )
    findings = verify_green(str(tmp_path))
    assert any("SCF integration mesh" in f.message for f in warnings_(findings))


def test_band_path_length_mismatch(tmp_path):
    from pyRSPthon.kpts.spts import write_spts_band

    (tmp_path / "green.inp").write_text("spectrum\nBand\n20 20\nG X W\n")
    kpts = np.zeros((10, 3))
    write_spts_band(tmp_path / "spts", kpts, labels={0: "G", 9: "W"})
    findings = verify_green(str(tmp_path))
    assert any("ticks will be misplaced" in f.message for f in warnings_(findings))


# --- the time-reversal predictor -----------------------------------------


def _tr_context(rotations, weights, fullrel=True, spinpol=False):
    data = SimpleNamespace(fullrel=fullrel, spinpol=spinpol, spin_average=False)
    symcof = SimpleNamespace(rotations=np.array(rotations, dtype=float))
    kpoints = np.random.default_rng(0).random((len(weights), 3))
    ctx = RunContext(directory=".")
    ctx.data = data
    ctx.symcof = symcof
    ctx.spts = (kpoints, np.array(weights), int(np.sum(weights)), {})
    return ctx


_C2Z = np.diag([-1.0, -1.0, 1.0])


def test_tr_check_fires_without_inversion():
    ctx = _tr_context([np.eye(3), _C2Z], weights=[2, 2, 1])
    findings = check_time_reversal(None, ctx)
    assert findings and findings[0].level == ERROR
    assert "time-reversal" in findings[0].message


def test_tr_check_passes_with_inversion():
    ctx = _tr_context([np.eye(3), -np.eye(3)], weights=[2, 2, 1])
    assert check_time_reversal(None, ctx) == []


def test_tr_check_zero_weight_escape():
    ctx = _tr_context([np.eye(3), _C2Z], weights=[2, 2, 0])
    assert check_time_reversal(None, ctx) == []


def test_tr_check_high_weight_escape():
    # weights above the group order can only come from time reversal
    ctx = _tr_context([np.eye(3), _C2Z], weights=[4, 2, 1])
    assert check_time_reversal(None, ctx) == []


def test_tr_check_needs_fulrel():
    ctx = _tr_context([np.eye(3), _C2Z], weights=[2, 2, 1], fullrel=False)
    assert check_time_reversal(None, ctx) == []
    ctx = _tr_context([np.eye(3), _C2Z], weights=[2, 2, 1], spinpol=True)
    assert check_time_reversal(None, ctx) == []
