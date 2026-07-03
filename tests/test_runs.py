import os
import re
import stat
import subprocess
import sys
import textwrap

import pytest

from pyRSPthon.run.report import fmt_duration, fmt_target, set_style
from pyRSPthon.run.runs import init_runs, read_convergence, runs


FAKE_RSPT = textwrap.dedent(
    """\
    #!/bin/bash
    export LC_ALL=C
    it=$( [ -f convergence ] && awk '{print $2}' convergence || echo 0 )
    fsq=$( [ -f convergence ] && awk '{print $1}' convergence || echo 1.0 )
    newfsq=$(python3 -c "print(float('$fsq')/10)")
    newit=$((it+1))
    printf 'starting\\nTIME: setene 0.1\\ndone\\n' > out
    echo "potdata iter $newit" > pot
    echo "eparmdata iter $newit" > eparm
    printf ' %15.7e %6d -1234.5678%s 100.0 0.5\\n' "$newfsq" "$newit" "$newit" > convergence
    """
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "fake_rspt.sh"
    script.write_text(FAKE_RSPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def run_driver(max_iter=10, fsq_conv=1e-3):
    return runs(
        ["./fake_rspt.sh"],
        [],
        fsq_conv=fsq_conv,
        e_conv=float("inf"),
        max_iter=max_iter,
        max_solver_it=1,
        save=False,
        save_solver_it=False,
        check_rspt=False,
    )


def test_fresh_run_converges(workdir):
    result = run_driver()
    assert result["converged"] and not result["diverged"]
    assert result["it"] == 4  # fsq: 0.1, 0.01, 1e-3, 1e-4 < 1e-3
    assert os.path.exists("pot_last") and os.path.exists("eparm_last")


def test_continuation_without_sig_keeps_state(workdir):
    """Regression: a DFT continuation without sig_last must not be wiped."""
    run_driver()
    state_before = read_convergence()
    result = run_driver(max_iter=2, fsq_conv=1e-12)
    assert os.path.exists("convergence")
    assert result["it"] == state_before.it + 2  # continued, not restarted


def test_fresh_start_requires_pot_and_eparm_together(workdir):
    (workdir / "pot").write_text("only pot")
    with pytest.raises(RuntimeError, match="BOTH pot and eparm"):
        init_runs()


def test_divergence_guard(workdir):
    script = workdir / "fake_rspt.sh"
    script.write_text(
        "#!/bin/bash\nexport LC_ALL=C\n"
        "printf 'x\\nTIME: setene 0.1\\n' > out\n"
        "echo p > pot; echo e > eparm\n"
        "printf ' %15.7e %6d -1.0 100.0 0.5\\n' 9.9e9 1 > convergence\n"
    )
    result = run_driver()
    assert result["diverged"] and not result["converged"]
    assert result["it"] == 1


def test_stopruns_aborts(workdir):
    (workdir / "stopruns").write_text("1")
    with pytest.raises(SystemExit, match="stopruns"):
        run_driver()


def test_failing_rspt_raises(workdir):
    (workdir / "fake_rspt.sh").write_text("#!/bin/bash\nexit 3\n")
    with pytest.raises(RuntimeError, match="Return value was 3"):
        run_driver()


def test_fmt_duration():
    assert fmt_duration(5.4) == "5s"
    assert fmt_duration(83) == "1m 23s"
    assert fmt_duration(7500) == "2h 05m"


def test_fmt_target():
    set_style(False)
    assert fmt_target(1e-9) == "1.0e-09"
    assert fmt_target(float("inf")) == "-"


ANSI_RE = re.compile("\x1b\\[")
ROW_RE = re.compile(r"^\s+\d+\s+\d\.\d{3}e[-+]\d\d\s")


def run_cli(workdir, *flags):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pyRSPthon.cli.py_runs",
            "./fake_rspt.sh",
            "-f",
            "1e-3",
            "-i",
            "10",
            "--no-check_rspt",
            *flags,
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_output_normal(workdir):
    result = run_cli(workdir)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # non-TTY stdout => plain ASCII, no escape codes
    assert not ANSI_RE.search(out)
    assert "SCF driver" in out
    assert "fresh start" in out
    rows = [line for line in out.splitlines() if ROW_RE.match(line)]
    assert len(rows) == 4  # fsq 0.1, 0.01, 1e-3, 1e-4 < 1e-3
    assert "CONVERGED after 4 iterations (4 this run)" in out
    info = (workdir / "runs.info").read_text()
    assert not ANSI_RE.search(info)
    assert "CONVERGED" in info


def test_cli_output_quiet(workdir):
    result = run_cli(workdir, "-q")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "SCF driver" not in out
    assert not any(ROW_RE.match(line) for line in out.splitlines())
    assert "CONVERGED" in out


def test_cli_output_verbose(workdir):
    result = run_cli(workdir, "-v")
    assert result.returncode == 0, result.stderr
    assert "RSPt took" in result.stdout
    assert "FSQ" in result.stdout  # convergence-check breakdown


BROKEN_GREEN = "mixing\n5 0.15\n"


def test_broken_green_inp_aborts_before_running(workdir):
    (workdir / "green.inp").write_text(BROKEN_GREEN)
    with pytest.raises(RuntimeError, match="verification found"):
        run_driver()
    assert not os.path.exists("convergence")  # rspt never ran


def test_no_verify_skips_green_check(workdir):
    (workdir / "green.inp").write_text(BROKEN_GREEN)
    result = run_cli(workdir, "--no-verify")
    assert result.returncode == 0, result.stderr


def test_cli_broken_green_inp_exits_1(workdir):
    (workdir / "green.inp").write_text(BROKEN_GREEN)
    result = run_cli(workdir)
    assert result.returncode == 1
    assert "unknown mix_method" in result.stdout + result.stderr


def test_cli_not_converged_summary(workdir):
    result = run_cli(workdir, "-f", "1e-30", "-i", "2")
    assert result.returncode == 2
    assert "NOT CONVERGED after 2 iterations (2 this run) [max iter = 2]" in (
        result.stdout
    )
