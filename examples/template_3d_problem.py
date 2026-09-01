"""
A 50x50x50 lid-inflow box run for 50 steps. This example shows the typed Python API; `exaflow run --case` is the standard command-line entry point.
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
from exaflow.run import run_case


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

    run_case(case, comm, output_directory=run_directory)
    if comm.Get_rank() == 0:
        print(f"Wrote {run_directory}", flush=True)


if __name__ == "__main__":
    main()
