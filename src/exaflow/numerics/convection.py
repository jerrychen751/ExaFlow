from __future__ import annotations

import numpy as np

from ..config.grid import Grid
from ..fields import FlowState
from ..mpi.subdomain import Subdomain


class Convection:
    """
    The explicit convection term, (u dot grad) u, as a first-order upwind difference.

    The rate written for velocity component `d` is -sum over axes a of vel[a] * d_a f_d, where d_a f_d is the backward difference (f_d[i] - f_d[i - 1 along a]) / h_a where vel[a] >= 0 and the forward difference (f_d[i + 1 along a] - f_d[i]) / h_a where vel[a] < 0. The difference has to lean into the flow: a backward difference used against a negative velocity grows without bound at every step, whatever the time step.

    Every real cell uses this one stencil, so a cell on a face or an edge of the block is treated exactly like a cell in the middle. Both neighbors of the outermost real cell are ghost cells, so the ghost exchange must complete before `accumulate` runs.

    Three scratch blocks are allocated here and reused by every call, so a stage allocates nothing.
    """

    def __init__(self, grid: Grid, subdomain: Subdomain) -> None:
        self._spacing = grid.spacing
        self._dimension = grid.dimension
        self._interior = subdomain.interior
        self._lower = tuple(subdomain.shift_interior(axis, -1) for axis in range(grid.dimension))
        self._upper = tuple(subdomain.shift_interior(axis, +1) for axis in range(grid.dimension))
        self._flux = np.empty(subdomain.shape, dtype=float)
        self._other = np.empty(subdomain.shape, dtype=float)
        self._weight = np.empty(subdomain.shape, dtype=float)

    def accumulate(self, state: FlowState, rate: FlowState) -> None:
        here = self._interior
        flux, other, weight = self._flux, self._other, self._weight
        for component in range(self._dimension):
            field = state.velocity[component]
            total = rate.velocity[component][here]
            for axis in range(self._dimension):
                speed = state.velocity[axis][here]

                np.subtract(field[here], field[self._lower[axis]], out=flux)
                np.maximum(speed, 0.0, out=weight)
                flux *= weight

                np.subtract(field[self._upper[axis]], field[here], out=other)
                np.minimum(speed, 0.0, out=weight)
                other *= weight

                flux += other
                flux *= 1.0 / self._spacing[axis]
                total -= flux
