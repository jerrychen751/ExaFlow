from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.boundaries import Boundaries, collect_faces
from ..fields import FlowState
from .subdomain import Subdomain


@dataclass(slots=True)
class GhostExchange:
    """
    A ghost exchange that has been posted and not yet completed. Between `post_ghost_exchange` and `complete` the caller may update cells whose stencil stays inside the real block; every cell whose stencil reaches a ghost layer must wait.

    The send buffers are held here on purpose. mpi4py does not keep a reference to the buffer passed to `Isend`, so releasing it before `Wait` would let MPI read freed memory.
    """

    _requests: list[Any] = field(default_factory=list)
    _outgoing: list[np.ndarray] = field(default_factory=list)
    _deliveries: list[tuple[np.ndarray, tuple[slice, ...], np.ndarray]] = field(default_factory=list)

    def complete(self) -> None:
        """
        Wait for every posted message, then copy each received plane into its ghost layer.
        """

        for request in self._requests:
            request.Wait()
        self._requests.clear()
        self._outgoing.clear()
        for array, destination, buffer in self._deliveries:
            array[destination] = buffer
        self._deliveries.clear()


def post_ghost_exchange(
    state: FlowState,
    subdomain: Subdomain,
    boundaries: Boundaries,
    comm: Any | None,
) -> GhostExchange:
    """
    Start a non-blocking exchange of the outermost real layer with every neighboring rank, for every field array in `state`.

    Only the interior transverse extent is exchanged, so the corner region of a ghost plane is left alone. No stencil in this solver reads a diagonal ghost cell; a stencil that did would need a second pass, one axis at a time.

    A face with no neighbor is skipped. A face whose neighbor is this rank itself, which is what a periodic axis carrying a single rank gives, is served by a local copy from the opposite face instead of a message, so a serial periodic run needs no communicator.

    Every message carries the tag of the face it leaves from, and a receive asks for the tag of the opposite face. Without the tag, a periodic axis carrying exactly two ranks makes each rank post two sends and two receives to the same peer, and MPI matches them in order, which fills the low ghost plane with the high one. Within one face the arrays are matched by MPI non-overtaking, so they are always walked in the order `state.arrays()` gives them.
    """

    exchange = GhostExchange()
    pad = subdomain.grid.num_ghost_layers
    transverse = subdomain.interior()

    for face in collect_faces(subdomain.grid.dimension):
        neighbor = subdomain.neighbor_rank(face, boundaries)
        if neighbor is None:
            continue
        if face.is_low:
            real_span, ghost_span = slice(pad, 2 * pad), slice(0, pad)
        else:
            real_span, ghost_span = slice(-2 * pad, -pad), slice(-pad, None)
        axis = face.axis
        source = tuple(real_span if i == axis else transverse[i] for i in range(len(transverse)))
        destination = tuple(ghost_span if i == axis else transverse[i] for i in range(len(transverse)))

        if neighbor == subdomain.rank:
            opposite = tuple(
                (slice(-2 * pad, -pad) if face.is_low else slice(pad, 2 * pad)) if i == axis else transverse[i]
                for i in range(len(transverse))
            )
            for array in state.arrays():
                exchange._deliveries.append((array, destination, array[opposite].copy()))
            continue

        if comm is None:
            raise ValueError(f"{face.name} has neighbor rank {neighbor} but no communicator was given.")

        send_tag = 2 * axis + (0 if face.is_low else 1)
        recv_tag = 2 * axis + (1 if face.is_low else 0)
        for array in state.arrays():
            outgoing = np.ascontiguousarray(array[source])
            incoming = np.empty_like(outgoing)
            exchange._requests.append(comm.Isend(outgoing, dest=neighbor, tag=send_tag))
            exchange._requests.append(comm.Irecv(incoming, source=neighbor, tag=recv_tag))
            exchange._outgoing.append(outgoing)
            exchange._deliveries.append((array, destination, incoming))

    return exchange
