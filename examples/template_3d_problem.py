"""
A 50x50x50 lid-inflow box run for 50 steps. This is the shortest complete driver: build a Case, hand it to a Solver, run.

Run it on one process with `uv run python examples/template_3d_problem.py`, or on four with `uv run mpiexec -n 4 python examples/template_3d_problem.py`. The result is the same either way.
"""

from __future__ import annotations

import mpi4py.MPI as mpi

from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    FaceCondition,
    Fluid,
    Grid,
    InitialConditions,
    TimeControl,
    UniformValue,
)
from exaflow.io.storage import create_run_directory
from exaflow.solver import Solver


def main() -> None:
    comm = mpi.COMM_WORLD
    run_directory = create_run_directory("template_3d", comm)

    case = Case(
        fluid=Fluid(rho=1.225, nu=0.3),
        grid=Grid(shape=(50, 50, 50), extent=(1.0, 1.0, 1.0), num_ghost_layers=1),
        time=TimeControl(num_steps=50, cfl=0.25, integration_order=1),
        boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, velocity=(2.0, 1.0, 1.0))),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )

    solver = Solver(case, comm, output_directory=run_directory)
    solver.run()
    if solver.rank == 0:
        print(f"Wrote {run_directory}", flush=True)


if __name__ == "__main__":
    main()
