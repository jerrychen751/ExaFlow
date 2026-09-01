from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from exaflow.config import Case, InitialConditions, TimeControl, UniformValue
from exaflow.config.case_xml import write_case

pytestmark = pytest.mark.slow


@pytest.fixture
def run_exaflow(tmp_path: Path, build_case: Callable[..., Case]) -> Callable[..., subprocess.CompletedProcess[str]]:
    """
    Return a launcher that writes a two-step 2D case to `<tmp_path>/<name>.xml`, then runs the console entry point with the output root pointed at `<tmp_path>/runs`. Set `num_procs` above one to place that command under `mpiexec`.
    """

    def run(*arguments: str, name: str = "tiny", num_procs: int = 1) -> subprocess.CompletedProcess[str]:
        case = build_case(
            (6, 5),
            time=TimeControl(2, 0.25, 1),
            initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.0),))),
        )
        (tmp_path / f"{name}.xml").write_text(write_case(case), encoding="utf-8")
        command = [sys.executable, "-m", "exaflow.cli", *arguments]
        if num_procs > 1:
            command = ["mpiexec", "-n", str(num_procs), *command]
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            env=dict(os.environ, EXAFLOW_OUTPUT_ROOT=str(tmp_path / "runs")),
        )

    return run


def test_a_run_writes_one_directory_and_names_it_on_stdout(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"))

    assert completed.returncode == 0, completed.stderr[-2000:]
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert completed.stdout.strip() == f"Wrote {runs[0]}"
    assert "Final_Total.csv" in [entry.name for entry in runs[0].iterdir()]


def test_the_run_folder_takes_its_name_from_the_case_file(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "lid_cavity.xml"), name="lid_cavity")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert next((tmp_path / "runs").iterdir()).name.endswith("_lid_cavity")


def test_a_label_replaces_the_file_stem(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), "--label", "sweep 4")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert next((tmp_path / "runs").iterdir()).name.endswith("_sweep_4")


def test_input_xml_names_the_same_option_as_case(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """
    Existing external commands can still pass --input-xml, but current documentation uses --case.
    """

    completed = run_exaflow("run", "--input-xml", str(tmp_path / "tiny.xml"))

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len(list((tmp_path / "runs").iterdir())) == 1


@pytest.mark.mpi
def test_mpiexec_runs_the_standard_cli_entry_point(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), num_procs=2)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_a_command_line_with_no_subcommand_is_refused(
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow()

    assert completed.returncode != 0
    assert "the following arguments are required: command" in completed.stderr


def test_a_run_with_no_case_is_refused(
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run")

    assert completed.returncode != 0
    assert "--case" in completed.stderr


def test_a_case_file_that_is_not_there_stops_the_run(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "absent.xml"))

    assert completed.returncode != 0
    assert "absent.xml" in completed.stderr
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
