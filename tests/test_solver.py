from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    OutputControl,
    TimeControl,
    UniformValue,
)
from exaflow.solver import Solver, compute_time_step


@pytest.fixture(scope="session")
def moving_case(build_case: Callable[..., Case]) -> Case:
    """
    A 3D case that marches five steps with an inflow on the left face and a uniform starting velocity, so every term of the right-hand side has something to act on.
    """

    return build_case(
        (12, 12, 12),
        time=TimeControl(5, 0.25, 1),
        boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0))),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )


def test_time_step_takes_the_stricter_of_the_two_limits(moving_case: Case) -> None:
    spacing = moving_case.grid.spacing[0]
    advective = moving_case.time.cfl * spacing / 1.0
    viscous = 0.5 / (moving_case.fluid.nu * 3.0 / spacing**2)
    assert compute_time_step(moving_case, 1.0) == pytest.approx(min(advective, viscous))


def test_an_inviscid_fluid_takes_the_advective_limit(build_case: Callable[..., Case]) -> None:
    case = build_case((8, 8, 8), fluid=Fluid(1.0, 0.0), time=TimeControl(1, 0.4))
    assert compute_time_step(case, 2.0) == pytest.approx(0.4 * case.grid.spacing[0] / 2.0)


def test_a_motionless_field_takes_the_viscous_limit(build_case: Callable[..., Case]) -> None:
    case = build_case((8, 8, 8), fluid=Fluid(1.0, 0.5))
    spacing = case.grid.spacing[0]
    assert compute_time_step(case, 0.0) == pytest.approx(0.5 / (0.5 * 3.0 / spacing**2))


def test_the_step_follows_the_shortest_spacing(build_case: Callable[..., Case]) -> None:
    """
    A stretched grid is stable only at the limit its thinnest cell allows, so the advective limit reads min(h) and not the spacing of the first axis.
    """

    case = build_case((5, 21), fluid=Fluid(1.0, 0.0), grid=Grid((5, 21), (1.0, 1.0), 1), time=TimeControl(1, 0.5))
    assert compute_time_step(case, 1.0) == pytest.approx(0.5 * min(case.grid.spacing))


def test_a_motionless_inviscid_case_has_no_stable_step(build_case: Callable[..., Case]) -> None:
    case = build_case((4, 4, 4), fluid=Fluid(1.0, 0.0), time=TimeControl(1, 0.25))
    with pytest.raises(ValueError, match="Cannot choose a time step"):
        compute_time_step(case, 0.0)


@pytest.mark.parametrize("max_velocity", [-1.0, float("nan"), float("inf")])
def test_a_velocity_that_is_not_a_speed_is_refused(moving_case: Case, max_velocity: float) -> None:
    with pytest.raises(ValueError, match="max_velocity must be finite and >= 0"):
        compute_time_step(moving_case, max_velocity)


def test_a_serial_run_stays_finite_and_respects_the_inflow(moving_case: Case) -> None:
    solver = Solver(moving_case)
    state = solver.run()
    assert np.all(np.isfinite(state.velocity))
    assert np.allclose(state.velocity[0][0, 1:-1, 1:-1], 2.0)


def test_a_solver_with_no_destination_writes_nothing(moving_case: Case) -> None:
    assert Solver(moving_case).writers == ()


def test_explicit_writers_replace_the_standard_set(tmp_path: Path, moving_case: Case) -> None:
    from exaflow.io.writers import RankCsvWriter

    solver = Solver(moving_case, output_directory=str(tmp_path), writers=())
    assert solver.writers == ()

    solver = Solver(moving_case, writers=(RankCsvWriter(str(tmp_path), Solver(moving_case).subdomain),))
    assert len(solver.writers) == 1


def test_writers_produce_the_expected_files(tmp_path: Path, moving_case: Case) -> None:
    Solver(moving_case, output_directory=str(tmp_path)).run()

    written = sorted(entry.name for entry in tmp_path.iterdir())
    assert "Final_Total.csv" in written
    assert "Final_0.csv" in written
    assert "Original_Total.csv" in written
    assert not any(name.endswith(".partial") for name in written)


def test_an_interval_writes_at_the_steps_it_selects(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    """
    The interval selects the zero-based step, so a frequency of 2 over five steps writes at 0, 2 and 4. The first and last state go through every writer whatever the interval, and they carry the labels Original and Final instead.
    """

    case = build_case(
        (6, 6, 6),
        time=TimeControl(5, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
        outputs=OutputControl(total_csv_frequency=2),
    )
    Solver(case, output_directory=str(tmp_path)).run()

    totals = sorted(entry.name for entry in tmp_path.iterdir() if entry.name.endswith("_Total.csv"))
    assert totals == ["0_Total.csv", "2_Total.csv", "4_Total.csv", "Final_Total.csv", "Original_Total.csv"]


def test_write_initial_false_leaves_out_the_starting_state(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    case = build_case(
        (6, 6, 6),
        time=TimeControl(2, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )
    Solver(case, output_directory=str(tmp_path)).run(write_initial=False)

    written = sorted(entry.name for entry in tmp_path.iterdir())
    assert not any(name.startswith("Original") for name in written)
    assert "Final_Total.csv" in written


@pytest.mark.gui
@pytest.mark.parametrize("shape,extent", [((8,), (2.0,)), ((8, 6), (2.0, 1.0)), ((8, 6, 4), (2.0, 1.0, 0.5))])
def test_vtk_output_pads_a_case_below_three_axes(
    tmp_path: Path,
    shape: tuple[int, ...],
    extent: tuple[float, ...],
) -> None:
    """
    pyevtk rejects an array that is not 3D, so a 1D or 2D case has to reach it padded to three axes.
    """

    pyvista = pytest.importorskip("pyvista")
    case = Case(
        fluid=Fluid(1.225, 0.3),
        grid=Grid(shape, extent, 1),
        time=TimeControl(1, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in shape)),
    )
    Solver(case, output_directory=str(tmp_path)).run()

    mesh = pyvista.read(str(tmp_path / "Final_Total.vtr"))
    assert mesh.dimensions == (*shape, *(1,) * (3 - len(shape)))
    assert mesh.bounds[: 2 * len(shape)] == pytest.approx([bound for span in extent for bound in (0.0, span)])
    assert sorted(mesh.point_data.keys()) == ["pressure", "velocity"]


@pytest.fixture(scope="session")
def serial_reference_output(
    tmp_path_factory: pytest.TempPathFactory,
    run_under_mpiexec: Callable[..., None],
) -> bytes:
    """
    The whole-domain CSV that `tests/_run_case.py` writes on one rank. Every rank count is compared against this one run, so mpiexec starts once for the reference and once for each count under test.
    """

    directory = tmp_path_factory.mktemp("serial")
    run_under_mpiexec("_run_case.py", 1, str(directory))
    return (directory / "Final_Total.csv").read_bytes()


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("num_procs", [2, 4, 8])
def test_the_answer_does_not_depend_on_the_rank_count(
    tmp_path: Path,
    run_under_mpiexec: Callable[..., None],
    serial_reference_output: bytes,
    num_procs: int,
) -> None:
    """
    The property the old solver did not have. Its boundary operators could not tell a global domain face from an internal partition face, so the answer moved with the decomposition.
    """

    run_under_mpiexec("_run_case.py", num_procs, str(tmp_path))
    written = (tmp_path / "Final_Total.csv").read_bytes()
    assert written == serial_reference_output, f"1 rank and {num_procs} ranks disagree"
