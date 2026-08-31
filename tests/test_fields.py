from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from exaflow.config import Case, Grid, InitialConditions, StepValue, UniformValue
from exaflow.fields import allocate_state, build_initial_state
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain

GRID = Grid((8,), (1.0,), 1)


def collect_global_velocity(case: Case, counts: tuple[int, ...]) -> np.ndarray:
    """
    The initial x velocity of the whole domain, assembled from the block each rank of a `counts` decomposition fills. Every rank evaluates the contributions over its own block, so this is what the domain would hold at that rank count.
    """

    process_grid = ProcessGrid(counts)
    assembled = np.zeros(case.grid.shape)
    for rank in range(process_grid.size):
        subdomain = Subdomain(case.grid, process_grid, rank)
        state = build_initial_state(case, subdomain)
        assembled[subdomain.global_slices] = state.velocity[0][subdomain.interior]
    return assembled


def test_an_allocated_state_is_zero_and_padded(build_subdomain: Callable[..., Subdomain]) -> None:
    subdomain = build_subdomain(Grid((6, 5), (1.0, 1.0), 2))
    state = allocate_state(subdomain, 2)
    assert state.velocity.shape == (2, 10, 9)
    assert state.pressure.shape == (10, 9)
    assert state.velocity.dtype == np.float64
    assert not state.velocity.any()
    assert state.dimension == 2
    assert state.padded_shape == (10, 9)


def test_collect_arrays_puts_the_velocity_components_before_pressure(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The ghost exchange indexes this tuple by position and the writers walk it in order, so the order is a contract and not a detail.
    """

    state = allocate_state(build_subdomain(Grid((4, 4, 4), (1.0, 1.0, 1.0), 1)), 3)
    for axis in range(3):
        state.velocity[axis][...] = float(axis)
    state.pressure[...] = 9.0

    arrays = state.collect_arrays()

    assert [float(array.flat[0]) for array in arrays] == [0.0, 1.0, 2.0, 9.0]
    arrays[1][...] = 7.0
    assert np.all(state.velocity[1] == 7.0)


def test_a_copy_shares_nothing_with_its_source(build_subdomain: Callable[..., Subdomain]) -> None:
    state = allocate_state(build_subdomain(GRID), 1)
    state.velocity[...] = 1.0
    state.pressure[...] = 2.0

    duplicate = state.copy()
    duplicate.velocity[...] = 9.0
    duplicate.pressure[...] = 9.0

    assert np.all(state.velocity == 1.0)
    assert np.all(state.pressure == 2.0)


def test_allocate_zeros_keeps_the_shape_and_drops_the_values(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    state = allocate_state(build_subdomain(GRID), 1)
    state.velocity[...] = 3.0
    empty = state.allocate_zeros()
    assert empty.velocity.shape == state.velocity.shape
    assert not empty.velocity.any()
    assert np.all(state.velocity == 3.0)


def test_set_sum_writes_base_plus_a_scaled_rate(build_subdomain: Callable[..., Subdomain]) -> None:
    subdomain = build_subdomain(GRID)
    base, rate, target = (allocate_state(subdomain, 1) for _ in range(3))
    base.velocity[...] = 2.0
    base.pressure[...] = 1.0
    rate.velocity[...] = 5.0
    rate.pressure[...] = 4.0

    target.set_sum(base, rate, 0.5)

    assert np.all(target.velocity == 4.5)
    assert np.all(target.pressure == 3.0)


def test_set_sum_accepts_the_rate_as_its_own_destination(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The docstring allows `rate` to be this same object and forbids `base`. The Runge-Kutta stages rely on that, so it is checked here rather than left to the integrator.
    """

    subdomain = build_subdomain(GRID)
    base, rate = allocate_state(subdomain, 1), allocate_state(subdomain, 1)
    base.velocity[...] = 2.0
    rate.velocity[...] = 5.0

    rate.set_sum(base, rate, 0.5)

    assert np.all(rate.velocity == 4.5)


def test_set_blend_weights_the_second_state(build_subdomain: Callable[..., Subdomain]) -> None:
    subdomain = build_subdomain(GRID)
    first, second, target = (allocate_state(subdomain, 1) for _ in range(3))
    first.velocity[...] = 1.0
    first.pressure[...] = 1.0
    second.velocity[...] = 5.0
    second.pressure[...] = 9.0

    target.set_blend(first, second, 0.25)

    assert np.all(target.velocity == 2.0)
    assert np.all(target.pressure == 3.0)


def test_set_blend_accepts_the_second_state_as_its_own_destination(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    subdomain = build_subdomain(GRID)
    first, second = allocate_state(subdomain, 1), allocate_state(subdomain, 1)
    first.velocity[...] = 1.0
    second.velocity[...] = 5.0

    second.set_blend(first, second, 0.25)

    assert np.all(second.velocity == 2.0)


def test_max_speed_is_the_magnitude_and_not_the_largest_component(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    A cell with u = v = w = 1 travels at sqrt(3). A CFL limit divided by the largest single component would be sqrt(3) too large, and the run would diverge while it reported a stable step.
    """

    state = allocate_state(build_subdomain(Grid((4, 4, 4), (1.0, 1.0, 1.0), 1)), 3)
    state.velocity[...] = 1.0
    assert state.compute_max_speed() == pytest.approx(np.sqrt(3.0))


def test_max_speed_reads_the_ghost_layers_too(build_subdomain: Callable[..., Subdomain]) -> None:
    """
    The value is reduced across ranks before it fixes the time step, and a ghost cell holds a real cell of a neighbor, so leaving the pad out would report a step the neighbor cannot take.
    """

    state = allocate_state(build_subdomain(GRID), 1)
    state.velocity[0][0] = 7.0
    assert state.compute_max_speed() == pytest.approx(7.0)


def test_a_state_with_no_velocity_component_reports_no_speed(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    assert allocate_state(build_subdomain(GRID), 0).compute_max_speed() == 0.0


def test_a_uniform_contribution_fills_every_real_cell_and_no_ghost_cell(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case(
        (6, 5),
        initial=InitialConditions(
            pressure=(UniformValue(1.5),),
            velocity=((UniformValue(2.0),), (UniformValue(-3.0),)),
        ),
    )
    subdomain = build_subdomain(case.grid)

    state = build_initial_state(case, subdomain)

    assert np.all(state.velocity[0][subdomain.interior] == 2.0)
    assert np.all(state.velocity[1][subdomain.interior] == -3.0)
    assert np.all(state.pressure[subdomain.interior] == 1.5)
    assert np.all(state.velocity[0][0] == 0.0)
    assert np.all(state.velocity[0][-1] == 0.0)


def test_contributions_are_summed_in_the_order_they_are_given(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case(
        (8,),
        initial=InitialConditions(
            velocity=((UniformValue(1.0), StepValue(10.0, (0.25,), (0.5,)), UniformValue(0.5)),),
        ),
    )
    subdomain = build_subdomain(case.grid)

    interior = build_initial_state(case, subdomain).velocity[0][subdomain.interior]

    assert interior.tolist() == [1.5, 1.5, 11.5, 11.5, 11.5, 1.5, 1.5, 1.5]


@pytest.mark.parametrize("counts", [(2,), (4,)])
def test_a_step_box_lands_on_the_same_cells_whatever_the_rank_count(
    build_case: Callable[..., Case],
    counts: tuple[int, ...],
) -> None:
    """
    Each rank intersects the global box with its own block, so a box that crosses a partition has to come out the same as it does on one rank. A wrong intersection is invisible in a serial run.
    """

    case = build_case((8,), initial=InitialConditions(velocity=((StepValue(5.0, (0.25,), (0.6,)),),)))
    assert collect_global_velocity(case, counts).tolist() == collect_global_velocity(case, (1,)).tolist()


def test_a_step_box_that_misses_a_rank_leaves_it_at_zero(build_case: Callable[..., Case]) -> None:
    case = build_case((8,), initial=InitialConditions(velocity=((StepValue(5.0, (0.0,), (0.2,)),),)))
    subdomain = Subdomain(case.grid, ProcessGrid((2,)), 1)
    assert not build_initial_state(case, subdomain).velocity[0].any()


def test_a_contribution_of_an_unknown_kind_is_refused(build_case: Callable[..., Case]) -> None:
    case = build_case((8,), initial=InitialConditions(pressure=("everywhere",)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported initial contribution"):
        build_initial_state(case, Subdomain(case.grid, ProcessGrid((1,)), 0))
