from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import pytest

from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    FaceCondition,
    Fluid,
    Grid,
    OutputControl,
    TimeControl,
)
from exaflow.config.case_xml import parse_case, read_case, write_case


def build_case(**overrides: object) -> Case:
    defaults = dict(
        fluid=Fluid(1.225, 0.3),
        grid=Grid((8, 8, 8), (1.0, 1.0, 1.0), 1),
        time=TimeControl(4, 0.25, 1),
    )
    defaults.update(overrides)
    return Case(**defaults)  # type: ignore[arg-type]


def test_grid_spacing_places_a_point_on_each_end() -> None:
    grid = Grid((5, 3), (1.0, 2.0), 1)
    assert grid.spacing == (0.25, 1.0)
    assert grid.dimension == 2


def test_case_is_frozen() -> None:
    case = build_case()
    with pytest.raises(Exception):
        case.grid = Grid((4, 4, 4), (1.0, 1.0, 1.0), 1)  # type: ignore[misc]


def test_grid_rejects_an_axis_with_one_point() -> None:
    with pytest.raises(ValueError, match="at least 2 grid points"):
        Grid((1, 4), (1.0, 1.0), 1)


def test_periodic_faces_must_be_paired() -> None:
    with pytest.raises(ValueError, match="Periodic faces must be paired"):
        Boundaries(left=FaceCondition(BoundaryCondition.PERIODIC))


def test_inflow_velocity_must_match_the_dimension() -> None:
    with pytest.raises(ValueError, match="velocity components"):
        build_case(
            grid=Grid((4, 4), (1.0, 1.0), 1),
            boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (1.0, 1.0, 1.0))),
        )


def test_output_control_rejects_a_zero_interval() -> None:
    with pytest.raises(ValueError, match="must be -1 or >= 1"):
        OutputControl(vtk_frequency=0)


def test_output_control_selects_the_right_steps() -> None:
    outputs = OutputControl(total_csv_frequency=10)
    assert outputs.is_due(outputs.total_csv_frequency, 0)
    assert outputs.is_due(outputs.total_csv_frequency, 20)
    assert not outputs.is_due(outputs.total_csv_frequency, 15)
    assert not outputs.is_due(outputs.vtk_frequency, 0)


def test_case_xml_round_trips() -> None:
    case = read_case("examples/input_template.xml")
    assert parse_case(ElementTree.fromstring(write_case(case))) == case


def test_reading_the_shipped_template_gives_the_documented_case() -> None:
    case = read_case("examples/input_template.xml")
    assert case.grid.shape == (100, 100, 50)
    assert case.time.num_steps == 1000
    assert case.fluid.nu == pytest.approx(1.81e-5)
