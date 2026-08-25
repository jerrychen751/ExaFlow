"""
Run a case given as an input XML file.

    uv run mpiexec -n 4 python examples/run_from_xml.py --input-xml examples/input_template.xml

The XML gives every parameter, including the output intervals. The rank arrangement comes from the communicator, not from the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mpi4py.MPI as mpi

from exaflow.config.case_xml import read_case
from exaflow.io.storage import create_run_directory
from exaflow.solver import Solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simulation from an input XML file.")
    parser.add_argument("--input-xml", required=True, help="Path to the input XML file.")
    arguments, _ = parser.parse_known_args()

    comm = mpi.COMM_WORLD
    case = read_case(arguments.input_xml)
    run_directory = create_run_directory(Path(arguments.input_xml).stem, comm)

    solver = Solver(case, comm, output_directory=run_directory)
    solver.run()
    if solver.rank == 0:
        print(f"Wrote {run_directory}", flush=True)


if __name__ == "__main__":
    main()
