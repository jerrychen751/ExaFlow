"""Fill a periodic 1D grid with its global index, exchange, and check both ghost planes on every
rank. Exits non-zero on the first wrong plane. Used by the ghost exchange test, which runs this
under mpiexec at several rank counts."""

from __future__ import annotations

import sys

import numpy as np
import mpi4py.MPI as mpi

from exaflow.config import Boundaries, BoundaryCondition, FaceCondition, Grid
from exaflow.fields import allocate_state
from exaflow.mpi.ghost_exchange import post_ghost_exchange
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain

POINTS = 8

comm = mpi.COMM_WORLD
periodic = FaceCondition(BoundaryCondition.PERIODIC)
grid = Grid((POINTS,), (1.0,), 1)
subdomain = Subdomain(grid, ProcessGrid((comm.Get_size(),)), comm.Get_rank())

state = allocate_state(subdomain, 1)
start, stop = subdomain.bounds[0]
state.velocity[0][subdomain.interior()] = np.arange(start, stop, dtype=float)

post_ghost_exchange(state, subdomain, Boundaries(left=periodic, right=periodic), comm).complete()

for plane, expected in ((state.velocity[0][0], (start - 1) % POINTS), (state.velocity[0][-1], stop % POINTS)):
    if plane != expected:
        sys.exit(f"rank {comm.Get_rank()} read ghost cell {plane} where global index {expected} belongs")
