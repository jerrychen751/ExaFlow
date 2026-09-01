"""
The `exaflow` console entry point.

    exaflow run --case examples/input_template.xml
    mpiexec -n 4 exaflow run --case examples/input_template.xml

This is the supported way to run a case without writing a driver, and it is what the GUI launches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config.case_xml import read_case
from .io.storage import create_run_directory
from .run import run_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exaflow", description="Run an ExaFlow case.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a case given as an input XML file.")
    run_parser.add_argument(
        "--case",
        "--input-xml",
        dest="case",
        required=True,
        help="Path to the input XML file.",
    )
    run_parser.add_argument(
        "--label",
        default=None,
        help="Name for the run folder; defaults to the file stem.",
    )
    arguments, _ = parser.parse_known_args(argv)

    try:
        import mpi4py.MPI as mpi
    except ImportError:
        comm = None
    else:
        comm = mpi.COMM_WORLD
    case = read_case(arguments.case)
    label = arguments.label or Path(arguments.case).stem
    run_directory = create_run_directory(label, comm)

    run_case(case, comm, output_directory=run_directory)
    rank = int(comm.Get_rank()) if comm is not None else 0
    if rank == 0:
        print(f"Wrote {run_directory}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
