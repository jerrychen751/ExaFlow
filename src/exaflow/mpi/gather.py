from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .subdomain import Subdomain

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def gather_global_array(subdomain: Subdomain, comm: Intracomm | None, block: np.ndarray) -> np.ndarray | None:
    """
    Assemble the full-domain array from every rank's block on rank 0, and return None on the other ranks. `block` is this rank's interior, with the ghost layers already stripped.

    This is a collective call: every rank must reach it, or the ranks that do will wait forever.
    """

    if comm is None or subdomain.process_grid.size == 1:
        return np.ascontiguousarray(block)

    parts = comm.gather(np.ascontiguousarray(block), root=0)
    if parts is None:
        return None

    assembled = np.empty(subdomain.grid.shape, dtype=block.dtype)
    for rank, part in enumerate(parts):
        owner = Subdomain(subdomain.grid, subdomain.process_grid, rank)
        assembled[owner.global_slices] = part
    return assembled


def scatter_global_array(subdomain: Subdomain, comm: Intracomm | None, array: np.ndarray | None) -> np.ndarray:
    """
    Give this rank the block of a full-domain array that its subdomain owns, with no ghost layers, so the caller writes it straight into the interior of a state. Rank 0 passes the whole array and every other rank passes None.

    This is a collective call: every rank must reach it, or the ranks that do will wait forever.
    """

    if comm is None or subdomain.process_grid.size == 1:
        if array is None:
            raise ValueError("A serial scatter needs the whole-domain array, got None.")
        return np.ascontiguousarray(array)

    blocks = None
    if int(comm.Get_rank()) == 0:
        if array is None:
            raise ValueError("Rank 0 needs the whole-domain array to scatter, got None.")
        blocks = [
            np.ascontiguousarray(array[Subdomain(subdomain.grid, subdomain.process_grid, rank).global_slices])
            for rank in range(subdomain.process_grid.size)
        ]
    return np.ascontiguousarray(comm.scatter(blocks, root=0))
