from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    FaceCondition,
    Fluid,
    Grid,
    InitialConditions,
    TimeControl,
    UniformValue,
)
from exaflow.solver import Solver, compute_time_step

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def build_case(num_steps: int = 5) -> Case:
    return Case(
        fluid=Fluid(1.225, 0.3),
        grid=Grid((12, 12, 12), (1.0, 1.0, 1.0), 1),
        time=TimeControl(num_steps, 0.25, 1),
        boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0))),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )


def test_time_step_takes_the_stricter_of_the_two_limits() -> None:
    case = build_case()
    spacing = case.grid.spacing[0]
    advective = case.time.cfl * spacing / 1.0
    viscous = 0.5 / (case.fluid.nu * 3.0 / spacing**2)
    assert compute_time_step(case, 1.0) == pytest.approx(min(advective, viscous))


def test_a_motionless_inviscid_case_has_no_stable_step() -> None:
    case = Case(fluid=Fluid(1.0, 0.0), grid=Grid((4, 4, 4), (1.0, 1.0, 1.0), 1), time=TimeControl(1, 0.25))
    with pytest.raises(ValueError, match="Cannot choose a time step"):
        compute_time_step(case, 0.0)


def test_a_serial_run_stays_finite_and_respects_the_inflow() -> None:
    solver = Solver(build_case())
    state = solver.run()
    assert np.all(np.isfinite(state.velocity))
    assert np.allclose(state.velocity[0][0, 1:-1, 1:-1], 2.0)


def test_writers_produce_the_expected_files() -> None:
    directory = tempfile.mkdtemp()
    try:
        solver = Solver(build_case(), output_directory=directory)
        solver.run()
        written = sorted(os.listdir(directory))
        assert "Final_Total.csv" in written
        assert "Final_0.csv" in written
        assert "Original_Total.csv" in written
        assert not any(name.endswith(".partial") for name in written)
    finally:
        shutil.rmtree(directory)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not on PATH")
@pytest.mark.parametrize("num_procs", [2, 4])
def test_the_answer_does_not_depend_on_the_rank_count(num_procs: int) -> None:
    """The property the old solver did not have. Its boundary operators could not tell a global
    domain face from an internal partition face, so the answer moved with the decomposition."""

    script = REPOSITORY_ROOT / "tests" / "_run_case.py"
    outputs = []
    try:
        for ranks in (1, num_procs):
            directory = tempfile.mkdtemp()
            outputs.append(directory)
            command = ["mpiexec", "-n", str(ranks), sys.executable, str(script), directory]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            assert completed.returncode == 0, completed.stderr[-2000:]
        serial = Path(outputs[0], "Final_Total.csv").read_bytes()
        parallel = Path(outputs[1], "Final_Total.csv").read_bytes()
        assert serial == parallel, f"1 rank and {num_procs} ranks disagree"
    finally:
        for directory in outputs:
            shutil.rmtree(directory, ignore_errors=True)
