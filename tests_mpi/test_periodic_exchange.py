from __future__ import annotations

import unittest

import numpy as np
import mpi4py.MPI as mpi

from tests._bootstrap import ensure_src_on_path


ensure_src_on_path()

from navier_stokes_solver.boundary_conditions import BoundaryCondition
from navier_stokes_solver.mpi.domain import parallelize_domain
from navier_stokes_solver.mpi.ghost_layers import add_ghost_layers, recv_ghost_info, send_ghost_info
from navier_stokes_solver.parameters import SimulationParameters


class TestPeriodicGhostExchange(unittest.TestCase):
    def test_periodic_exchange_wraps_x_direction(self) -> None:
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size != 2:
            raise unittest.SkipTest("MPI smoke test expects exactly 2 ranks: run with `mpiexec -n 2`.")

        sim = SimulationParameters(
            rho=1.0,
            nu=0.0,
            domain=(8, 4, 4),
            size=(1.0, 1.0, 1.0),
            nt=1,
            num_ghost_layers=1,
            cfl=0.25,
            comm=comm,
            left_wall=BoundaryCondition.PERIODIC,
            right_wall=BoundaryCondition.PERIODIC,
            top_wall=BoundaryCondition.NO_SLIP,
            bottom_wall=BoundaryCondition.NO_SLIP,
            front_wall=BoundaryCondition.NO_SLIP,
            back_wall=BoundaryCondition.NO_SLIP,
            include_pressure=False,
        )

        u_global = np.empty(sim.domain, dtype=float)
        local, rank_coords = parallelize_domain(u_global, rank, sim)
        local = add_ghost_layers(local, sim.num_ghost_layers, fill_value=0.0)

        g = sim.num_ghost_layers
        local[g:-g, g:-g, g:-g] = float(rank + 1)

        req, buf = send_ghost_info(local, sim, rank_coords)
        recv_ghost_info(local, sim, rank_coords, req, buf)

        left_ghost = local[0:g, :, :]
        right_ghost = local[-g:, :, :]
        left_ghost_interior = left_ghost[:, g:-g, g:-g]
        right_ghost_interior = right_ghost[:, g:-g, g:-g]

        if rank == 0:
            self.assertTrue(np.allclose(left_ghost_interior, 2.0))
        if rank == 1:
            self.assertTrue(np.allclose(right_ghost_interior, 1.0))
