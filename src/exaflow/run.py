from __future__ import annotations

from typing import Any

from .config import Case
from .fields import FlowState
from .solver import Solver


def run_case(
    case: Case,
    comm: Any | None = None,
    *,
    output_directory: str | None = None,
) -> FlowState:
    """
    Run one typed Case on this rank and return its final local state. Every MPI rank must call this function with the same Case and output directory.
    """

    return Solver(case, comm, output_directory=output_directory).run()
