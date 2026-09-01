from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Case
from .fields import FlowState
from .session import SimulationSession

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def run_case(
    case: Case,
    comm: Intracomm | None = None,
    *,
    output_directory: str | None = None,
) -> FlowState:
    """
    Run one typed Case on this rank and return its final local state. Every MPI rank must call this function with the same Case and output directory.
    """

    return SimulationSession(case, comm, output_directory=output_directory).run_until_complete()
