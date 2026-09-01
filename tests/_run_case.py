"""
Run a fixed case and write it to the directory named on the command line. Used by the rank-count independence test, which runs this under mpiexec at several rank counts.
"""

from __future__ import annotations

import sys

import mpi4py.MPI as mpi

from exaflow.config import (
    Boundaries, BoundaryCondition, Case, FaceCondition, Fluid, Grid,
    InitialConditions, TimeControl, UniformValue,
)
from exaflow.io.writers import TotalCsvWriter
from exaflow.session import SimulationSession

case = Case(
    fluid=Fluid(1.225, 0.3),
    grid=Grid((16, 16, 16), (1.0, 1.0, 1.0), 1),
    time=TimeControl(num_steps=12, cfl=0.25, integration_order=3),
    boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0))),
    initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
)
comm = mpi.COMM_WORLD
session = SimulationSession(case, comm, writers=None, output_directory=None)
session.writers = (TotalCsvWriter(sys.argv[1], session.subdomain, comm),)
session.run_until_complete(write_initial=False)
