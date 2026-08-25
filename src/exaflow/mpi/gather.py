from __future__ import annotations

from typing import Any

import numpy as np

from .subdomain import Subdomain


def gather_global_array(subdomain: Subdomain, comm: Any | None, block: np.ndarray) -> np.ndarray | None:
    """
    Assemble the full-domain array from every rank's block on rank 0, and return None on the other ranks. `block` is this rank's interior, with the ghost layers already stripped.

    This is a collective call: every rank must reach it, or the ranks that do will wait forever.
    """

    if comm is None or subdomain.process_grid.size == 1:
        return np.ascontiguousarray(block)

    parts = comm.gather(np.ascontiguousarray(block), root=0)
    if int(comm.Get_rank()) != 0:
        return None

    assembled = np.empty(subdomain.grid.shape, dtype=block.dtype)
    for rank, part in enumerate(parts):
        owner = Subdomain(subdomain.grid, subdomain.process_grid, rank)
        assembled[owner.global_slices()] = part
    return assembled
