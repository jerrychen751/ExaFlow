from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from exaflow.config import Case, InitialConditions, OutputControl, TimeControl, UniformValue
from exaflow.config.case_xml import write_case
from exaflow.fields import TimeLevel
from exaflow.io.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    read_case_text,
    read_checkpoint,
    scatter_checkpoint,
    write_checkpoint,
)
from exaflow.mpi.subdomain import Subdomain
from exaflow.run import resume_case, run_case
from exaflow.session import SimulationSession


@pytest.fixture
def moving_case(build_case: Callable[..., Case]) -> Case:
    """
    A 2D case that marches six steps and saves a checkpoint every three, so a test can stop it half way and continue.
    """

    return build_case(
        (6, 5),
        time=TimeControl(6, 0.25, 1),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
        outputs=OutputControl(checkpoint_frequency=3),
    )


def build_checkpoint(case: Case) -> Checkpoint:
    velocity = np.arange(60.0).reshape(2, 6, 5)  # (60,) -> (2, 6, 5)
    return Checkpoint(velocity, velocity[0] * 0.5, TimeLevel(400, 0.8, 0.002), write_case(case))


def test_a_checkpoint_reads_back_as_the_record_that_was_written(tmp_path: Path, moving_case: Case) -> None:
    checkpoint = build_checkpoint(moving_case)

    write_checkpoint(str(tmp_path / "Checkpoint_400.npz"), checkpoint)
    read_back = read_checkpoint(str(tmp_path / "Checkpoint_400.npz"))

    assert np.array_equal(read_back.velocity, checkpoint.velocity)
    assert np.array_equal(read_back.pressure, checkpoint.pressure)
    assert read_back.level == TimeLevel(400, 0.8, 0.002)
    assert read_back.case_xml == checkpoint.case_xml


def test_a_written_checkpoint_leaves_no_partial_file(tmp_path: Path, moving_case: Case) -> None:
    """
    The file is renamed over its target only after the flush, so a reader never opens a half one.
    """

    write_checkpoint(str(tmp_path / "runs" / "Checkpoint_1.npz"), build_checkpoint(moving_case))

    assert sorted(entry.name for entry in (tmp_path / "runs").iterdir()) == ["Checkpoint_1.npz"]


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_1.npz"
    with open(path, "wb") as handle:
        np.savez(handle, velocity=np.zeros(3))

    with pytest.raises(ValueError, match="no format_version"):
        read_checkpoint(str(path))
    with pytest.raises(ValueError, match="no case_xml"):
        read_case_text(str(path))


def test_a_checkpoint_of_another_format_version_is_refused(tmp_path: Path, moving_case: Case) -> None:
    path = tmp_path / "Checkpoint_1.npz"
    checkpoint = build_checkpoint(moving_case)
    with open(path, "wb") as handle:
        np.savez(
            handle,
            format_version=CHECKPOINT_FORMAT_VERSION + 1,
            velocity=checkpoint.velocity,
            pressure=checkpoint.pressure,
            step_index=1,
            current_time=0.1,
            dt=0.1,
            case_xml=checkpoint.case_xml,
        )

    with pytest.raises(ValueError, match=f"version {CHECKPOINT_FORMAT_VERSION + 1}"):
        read_checkpoint(str(path))


def test_a_checkpoint_carries_the_case_that_produced_it(tmp_path: Path, moving_case: Case) -> None:
    path = tmp_path / "Checkpoint_400.npz"
    write_checkpoint(str(path), build_checkpoint(moving_case))

    assert read_case_text(str(path)) == write_case(moving_case)


def test_a_domain_of_another_shape_does_not_fit_the_case(
    tmp_path: Path,
    moving_case: Case,
    build_subdomain: Callable[..., Subdomain],
) -> None:
    path = tmp_path / "Checkpoint_400.npz"
    write_checkpoint(str(path), build_checkpoint(moving_case))
    wider = replace(moving_case, grid=replace(moving_case.grid, shape=(7, 5), extent=(1.0, 1.0)))

    with pytest.raises(ValueError, match="holds a domain of shape"):
        scatter_checkpoint(str(path), wider, build_subdomain(wider.grid))


def test_a_run_stopped_and_continued_reaches_the_state_the_whole_run_reaches(
    tmp_path: Path,
    moving_case: Case,
) -> None:
    """
    The test the whole format exists for. The restored state holds the interior of the stopped run, and the exchange and the boundary values fill the ghost layers again before the next step, so the two paths run the same arithmetic.
    """

    whole = run_case(moving_case, output_directory=str(tmp_path / "whole"))

    continued = resume_case(
        str(tmp_path / "whole" / "Checkpoint_3.npz"),
        output_directory=str(tmp_path / "continued"),
    )

    assert np.array_equal(continued.velocity, whole.velocity)
    assert np.array_equal(continued.pressure, whole.pressure)


def test_a_continued_run_writes_its_own_folder_and_carries_on_the_labels(
    tmp_path: Path,
    moving_case: Case,
) -> None:
    case = replace(moving_case, outputs=OutputControl(total_frequency=3, checkpoint_frequency=3))
    run_case(case, output_directory=str(tmp_path / "whole"))

    resume_case(
        str(tmp_path / "whole" / "Checkpoint_3.npz"),
        output_directory=str(tmp_path / "continued"),
    )

    written = sorted(entry.name for entry in (tmp_path / "continued").iterdir())
    assert "Resumed_Total.csv" in written
    assert "6_Total.csv" in written
    assert "3_Total.csv" not in written


def test_a_checkpointed_run_saves_one_file_at_the_interval_and_one_at_the_end(
    tmp_path: Path,
    moving_case: Case,
) -> None:
    run_case(moving_case, output_directory=str(tmp_path))

    saved = sorted(entry.name for entry in tmp_path.iterdir() if entry.name.startswith("Checkpoint"))
    assert saved == ["Checkpoint_3.npz", "Checkpoint_6.npz", "Checkpoint_Final.npz"]


def test_a_run_with_no_checkpoint_interval_saves_none(tmp_path: Path, build_case: Callable[..., Case]) -> None:
    case = build_case((6, 5), time=TimeControl(2, 0.25, 1))

    run_case(case, output_directory=str(tmp_path))

    assert not list(tmp_path.glob("*.npz"))


def test_a_replacement_case_carries_the_run_past_its_first_budget(
    tmp_path: Path,
    moving_case: Case,
) -> None:
    """
    The step budget counts from time zero, so a finished run continues only under a case that raises it.
    """

    run_case(moving_case, output_directory=str(tmp_path / "whole"))
    longer = replace(moving_case, time=TimeControl(9, 0.25, 1))

    session = SimulationSession(
        longer,
        output_directory=str(tmp_path / "longer"),
        checkpoint_path=str(tmp_path / "whole" / "Checkpoint_Final.npz"),
    )
    assert session.step_index == 6
    session.run_until_complete()

    assert session.step_index == 9


def test_a_replacement_case_of_another_shape_is_refused(tmp_path: Path, moving_case: Case) -> None:
    run_case(moving_case, output_directory=str(tmp_path / "whole"))
    wider = replace(moving_case, grid=replace(moving_case.grid, shape=(7, 5), extent=(1.0, 1.0)))

    with pytest.raises(ValueError, match="holds a domain of shape"):
        resume_case(
            str(tmp_path / "whole" / "Checkpoint_3.npz"),
            case=wider,
            output_directory=str(tmp_path / "continued"),
        )
