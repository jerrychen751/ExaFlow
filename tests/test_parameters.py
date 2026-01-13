from __future__ import annotations

import unittest

from ._bootstrap import ensure_src_on_path


ensure_src_on_path()

from navier_stokes_solver.boundary_conditions import BoundaryCondition
from navier_stokes_solver.parameters import SimulationParameters


class TestSimulationParameters(unittest.TestCase):
    def test_grid_spacing_3d_uses_physical_definition(self) -> None:
        sim = SimulationParameters(
            rho=1.0,
            nu=1.0,
            domain=(11, 21, 31),
            size=(1.0, 2.0, 3.0),
            nt=1,
            num_ghost_layers=1,
            cfl=0.25,
            comm=None,
            include_pressure=False,
        )
        self.assertAlmostEqual(sim.dx or 0.0, 1.0 / 10.0)
        self.assertAlmostEqual(sim.dy or 0.0, 2.0 / 20.0)
        self.assertAlmostEqual(sim.dz or 0.0, 3.0 / 30.0)

    def test_dt_is_min_of_advection_and_diffusion_constraints(self) -> None:
        sim = SimulationParameters(
            rho=1.0,
            nu=1.0,
            domain=(11, 11, 11),
            size=(1.0, 1.0, 1.0),
            nt=1,
            num_ghost_layers=1,
            cfl=0.25,
            comm=None,
            include_pressure=False,
        )
        dt = sim.compute_dt(max_velocity_magnitude=1.0)

        advection_dt = 0.25 * (1.0 / 10.0) / 1.0
        diffusion_dt = 0.5 / (1.0 * (3.0 * (10.0 * 10.0)))
        self.assertAlmostEqual(dt, min(advection_dt, diffusion_dt))

    def test_periodic_boundaries_must_be_paired(self) -> None:
        with self.assertRaises(ValueError):
            SimulationParameters(
                rho=1.0,
                nu=0.0,
                domain=(11, 11, 11),
                size=(1.0, 1.0, 1.0),
                nt=1,
                num_ghost_layers=1,
                cfl=0.25,
                comm=None,
                left_wall=BoundaryCondition.PERIODIC,
                right_wall=BoundaryCondition.NO_SLIP,
                include_pressure=False,
            )

