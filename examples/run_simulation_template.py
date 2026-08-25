"""
The template a new simulation directory gets. `create_simulation_directories.py` copies this file to `<target>/build/run_simulation.py` and the input XML to `<target>/build/in/input.xml`.

    uv run mpiexec -n 4 python simulations/my_run/build/run_simulation.py

It reads `in/input.xml` next to itself unless --input-xml names another file, so the copy runs from wherever it sits and needs no repository root.
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
    default_xml = Path(__file__).resolve().parent / "in" / "input.xml"
    parser.add_argument("--input-xml", default=str(default_xml), help="Path to the input XML file.")
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
