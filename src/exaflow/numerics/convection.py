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
    """

    def __init__(self, grid: Grid, subdomain: Subdomain) -> None:
        self._spacing = grid.spacing
        self._dimension = grid.dimension
        self._interior = subdomain.interior
        self._lower = tuple(subdomain.shift_interior(axis, -1) for axis in range(grid.dimension))
        self._upper = tuple(subdomain.shift_interior(axis, +1) for axis in range(grid.dimension))

    def accumulate(self, state: FlowState, rate: FlowState) -> None:
        here = self._interior
        for component in range(self._dimension):
            field = state.velocity[component]
            total = rate.velocity[component][here]
            for axis in range(self._dimension):
                speed = state.velocity[axis][here]
                backward = (field[here] - field[self._lower[axis]]) / self._spacing[axis]
                forward = (field[self._upper[axis]] - field[here]) / self._spacing[axis]
                total -= np.maximum(speed, 0.0) * backward + np.minimum(speed, 0.0) * forward
