from __future__ import annotations

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
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain
from exaflow.numerics.convection import Convection
from exaflow.numerics.diffusion import Diffusion
from exaflow.numerics.operators import SpatialOperator
from exaflow.numerics.time_step import TimeIntegrator

PERIODIC = FaceCondition(BoundaryCondition.PERIODIC)
ALL_PERIODIC = Boundaries(left=PERIODIC, right=PERIODIC, top=PERIODIC, bottom=PERIODIC, front=PERIODIC, back=PERIODIC)


def build_subdomain(grid: Grid) -> Subdomain:
    return Subdomain(grid, ProcessGrid((1,) * grid.dimension), 0)


def test_diffusion_reproduces_an_analytic_laplacian() -> None:
    grid = Grid((10, 10, 10), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    coordinate = np.arange(subdomain.padded_shape[0]) * grid.spacing[0]  # (P,)
    state.velocity[0][...] = (coordinate**2)[:, None, None]  # (P,) -> (P, 1, 1) broadcast over the block

    Diffusion(Fluid(1.0, 1.0), grid, subdomain).accumulate(state, rate)

    assert np.allclose(rate.velocity[0][subdomain.interior], 2.0)


def test_diffusion_writes_every_real_cell_including_the_block_edges() -> None:
    """The operator this replaced wrote its faces with a transverse span of 2:-2, so the twelve
    edges of every block were left at zero. That gap is what made the answer depend on the rank count."""

    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = np.random.default_rng(0).random((3, *subdomain.padded_shape))

    Diffusion(Fluid(1.0, 1.0), grid, subdomain).accumulate(state, rate)

    assert np.count_nonzero(rate.velocity[0][subdomain.interior] == 0.0) == 0


def test_convection_of_a_uniform_field_is_zero() -> None:
    grid = Grid((6, 6, 6), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    state.velocity[...] = 2.5

    Convection(grid, subdomain).accumulate(state, rate)

    assert np.allclose(rate.velocity[:, *subdomain.interior], 0.0)


def test_convection_matches_the_upwind_difference_of_a_ramp() -> None:
    grid = Grid((8, 4, 4), (1.0, 1.0, 1.0), 1)
    subdomain = build_subdomain(grid)
    state, rate = allocate_state(subdomain, 3), allocate_state(subdomain, 3)
    step = grid.spacing[0]
    state.velocity[0][...] = (np.arange(subdomain.padded_shape[0]) * step)[:, None, None]

    Convection(grid, subdomain).accumulate(state, rate)

    speed = state.velocity[0][subdomain.interior]
    assert np.allclose(rate.velocity[0][subdomain.interior], -speed * 1.0)


@pytest.mark.parametrize("order,expected", [(1, 1.0), (2, 2.0), (3, 3.0)])
def test_each_scheme_converges_at_its_design_order(order: int, expected: float) -> None:
    """u_j = sin(2*pi*j/NX) is an exact eigenvector of the discrete Laplacian over a wrapped block,
    so the exact answer is known and the only error left is the time-integration error.
    Every step stays inside the explicit diffusion limit, or the scheme would simply diverge."""

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


def test_convection_leans_into_the_flow_on_each_sign() -> None:
    """A fixed backward difference is unconditionally unstable where the velocity is negative, and
    compute_time_step cannot see it because it only looks at |u|. A quadratic profile separates the
    two one-sided differences: forward gives 2x + h where backward gives 2x - h."""

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
