from __future__ import annotations

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
    SolverOptions,
    TimeControl,
)
from exaflow.fields import allocate_state
from exaflow.mpi.subdomain import Subdomain
from exaflow.numerics.convection import Convection
from exaflow.numerics.diffusion import Diffusion
from exaflow.numerics.operators import SpatialOperator, build_operators
from exaflow.numerics.time_step import TimeIntegrator

PERIODIC = FaceCondition(BoundaryCondition.PERIODIC)
ALL_PERIODIC = Boundaries(left=PERIODIC, right=PERIODIC, top=PERIODIC, bottom=PERIODIC, front=PERIODIC, back=PERIODIC)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_diffusion_reproduces_an_analytic_laplacian(
    build_subdomain: Callable[..., Subdomain],
    dimension: int,
) -> None:
    """
    The laplacian of x^2 is 2 whatever the dimension, so a term that reads the wrong axis or drops one shows up as a rate that is not 2.
    """

    grid = Grid((10,) * dimension, (1.0,) * dimension, 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, dimension), allocate_state(subdomain, dimension)
    coordinate = np.arange(subdomain.padded_shape[0]) * grid.spacing[0]  # (P,)
    shape = (-1, *(1,) * (dimension - 1))
    state.velocity[0][...] = (coordinate**2).reshape(shape)  # (P,) -> (P, 1, 1) broadcast over the block

    Diffusion(Fluid(1.0, 1.0), grid, subdomain).accumulate(state, rate)

    assert np.allclose(rate.velocity[0][subdomain.interior], 2.0)


def test_diffusion_writes_every_real_cell_including_the_block_edges(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The operator this replaced wrote its faces with a transverse span of 2:-2, so the twelve edges of every block were left at zero. That gap is what made the answer depend on the rank count.
    """

    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = np.random.default_rng(0).random((3, *subdomain.padded_shape))

    Diffusion(Fluid(1.0, 1.0), grid, subdomain).accumulate(state, rate)

    assert np.count_nonzero(rate.velocity[0][subdomain.interior] == 0.0) == 0


def test_an_inviscid_fluid_contributes_no_viscous_term(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = np.random.default_rng(1).random((3, *subdomain.padded_shape))

    Diffusion(Fluid(1.0, 0.0), grid, subdomain).accumulate(state, rate)

    assert not rate.velocity.any()


def test_convection_of_a_uniform_field_is_zero(build_subdomain: Callable[..., Subdomain]) -> None:
    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = 2.5

    Convection(grid, subdomain).accumulate(state, rate)

    assert np.allclose(rate.velocity[:, *subdomain.interior], 0.0)


def test_convection_matches_the_upwind_difference_of_a_ramp(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    grid = Grid((8, 4, 4), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    step = grid.spacing[0]
    state.velocity[0][...] = (np.arange(subdomain.padded_shape[0]) * step)[:, None, None]

    Convection(grid, subdomain).accumulate(state, rate)

    speed = state.velocity[0][subdomain.interior]
    assert np.allclose(rate.velocity[0][subdomain.interior], -speed * 1.0)


def test_convection_leans_into_the_flow_on_each_sign(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    A fixed backward difference is unconditionally unstable where the velocity is negative, and compute_time_step cannot see it because it only looks at |u|. A quadratic profile separates the two one-sided differences: forward gives 2x + h where backward gives 2x - h.
    """

    grid = Grid((8, 4, 4), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    step = grid.spacing[0]
    coordinate = (np.arange(subdomain.padded_shape[0]) * step)[:, None, None]

    for direction in (+1.0, -1.0):
        state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
        state.velocity[0][...] = direction
        state.velocity[1][...] = coordinate**2

        Convection(grid, subdomain).accumulate(state, rate)

        here = coordinate[subdomain.interior[0]]
        one_sided = 2.0 * here - direction * step
        assert np.allclose(rate.velocity[1][subdomain.interior], -direction * one_sided)


def test_one_term_adds_to_what_another_already_wrote(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The Operator protocol says accumulate adds and never clears. A term that assigned instead would silently drop every term built before it.
    """

    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state = allocate_state(subdomain, 3)
    state.velocity[...] = np.random.default_rng(2).random((3, *subdomain.padded_shape))
    convection, diffusion = Convection(grid, subdomain), Diffusion(Fluid(1.0, 0.5), grid, subdomain)

    alone = [allocate_state(subdomain, 3) for _ in range(2)]
    convection.accumulate(state, alone[0])
    diffusion.accumulate(state, alone[1])
    together = allocate_state(subdomain, 3)
    convection.accumulate(state, together)
    diffusion.accumulate(state, together)

    assert np.allclose(together.velocity, alone[0].velocity + alone[1].velocity)


@pytest.mark.parametrize(
    "options,expected",
    [
        (SolverOptions(), ["Convection", "Diffusion"]),
        (SolverOptions(include_convection=False), ["Diffusion"]),
        (SolverOptions(include_diffusion=False), ["Convection"]),
        (SolverOptions(include_convection=False, include_diffusion=False), []),
    ],
)
def test_build_operators_returns_the_terms_the_case_switches_on(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    options: SolverOptions,
    expected: list[str],
) -> None:
    case = build_case((6, 6, 6), solver=options)
    operators = build_operators(case, build_subdomain(case.grid))
    assert [type(operator).__name__ for operator in operators] == expected


def test_a_stage_overwrites_the_rate_it_is_handed(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The rate buffer is reused by every stage, so evaluate has to clear it first. A stage that added to the previous rate would compound the right-hand side step after step.
    """

    case = build_case((6, 6, 6), boundaries=ALL_PERIODIC)
    subdomain = build_subdomain(case.grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = np.random.default_rng(3).random((3, *subdomain.padded_shape))
    spatial = SpatialOperator(case, subdomain, None)

    spatial.evaluate(state, rate)
    first = rate.velocity.copy()
    spatial.evaluate(state, rate)

    assert np.array_equal(rate.velocity, first)


@pytest.mark.parametrize("order", [0, 4, -1])
def test_the_integrator_refuses_an_order_it_does_not_implement(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    order: int,
) -> None:
    case = build_case((6, 6, 6), boundaries=ALL_PERIODIC)
    subdomain = build_subdomain(case.grid)
    spatial = SpatialOperator(case, subdomain, None)
    with pytest.raises(ValueError, match="integration order must be 1, 2 or 3"):
        TimeIntegrator(spatial, order, allocate_state(subdomain, 3))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_step_returns_a_new_buffer_and_leaves_the_old_one_stale(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    order: int,
) -> None:
    """
    advance alternates between two buffers, so the caller must use the object it gets back. A caller that kept its own reference would read the level before last on every second step.
    """

    case = build_case(
        (6, 6, 6),
        boundaries=ALL_PERIODIC,
        solver=SolverOptions(include_convection=False, include_diffusion=False),
    )
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, 3)
    state.velocity[...] = 1.5
    integrator = TimeIntegrator(SpatialOperator(case, subdomain, None), order, state)

    advanced = integrator.advance(state, 0.1)

    assert advanced is not state
    assert np.all(advanced.velocity == 1.5)


@pytest.mark.parametrize("order,expected", [(1, 1.0), (2, 2.0), (3, 3.0)])
def test_each_scheme_converges_at_its_design_order(
    build_subdomain: Callable[..., Subdomain],
    order: int,
    expected: float,
) -> None:
    """
    u_j = sin(2*pi*j/NX) is an exact eigenvector of the discrete Laplacian over a wrapped block, so the exact answer is known and the only error left is the time-integration error. Every step stays inside the explicit diffusion limit, or the scheme would simply diverge.
    """

    points, viscosity = 8, 0.05
    grid = Grid((points, 3, 3), (1.0, 1.0, 1.0), 1)
    spacing = grid.spacing[0]
    mode = np.sin(2 * np.pi * np.arange(points) / points)
    decay = viscosity * (4.0 / spacing**2) * np.sin(np.pi / points) ** 2
    duration = 20 * 0.4 * spacing**2 / (2 * viscosity)

    errors = []
    for num_steps in (20, 40):
        case = Case(
            fluid=Fluid(1.0, viscosity),
            grid=grid,
            time=TimeControl(num_steps, 0.25, order),
            boundaries=ALL_PERIODIC,
            solver=SolverOptions(include_convection=False, include_diffusion=True),
        )
        subdomain = build_subdomain(grid)
        state = allocate_state(subdomain, 3)
        state.velocity[0][subdomain.interior] = mode[:, None, None]
        integrator = TimeIntegrator(SpatialOperator(case, subdomain, None), order, state)
        for _ in range(num_steps):
            state = integrator.advance(state, duration / num_steps)
        exact = np.exp(-decay * duration) * mode
        errors.append(float(np.abs(state.velocity[0][subdomain.interior][:, 1, 1] - exact).max()))

    assert np.log2(errors[0] / errors[1]) == pytest.approx(expected, abs=0.25)
