from __future__ import annotations

from ..config.fluid import Fluid
from ..config.grid import Grid
from ..fields import FlowState
from ..mpi.subdomain import Subdomain


class Diffusion:
    """
    The explicit viscous term, nu * laplacian(u), as a second-order central difference.

    The rate written for velocity component `d` is nu * sum over axes a of (f_d[i + 1] - 2 f_d[i] + f_d[i - 1]) / h_a^2. Every real cell uses this one stencil, so a cell on a face or an edge of the block is treated exactly like a cell in the middle. Both neighbors of the outermost real cell are ghost cells, so the ghost exchange must complete before `accumulate` runs.
    """

    def __init__(self, fluid: Fluid, grid: Grid, subdomain: Subdomain) -> None:
        self._nu = fluid.nu
        self._dimension = grid.dimension
        self._interior = subdomain.interior()
        self._inverse_square = tuple(1.0 / (step * step) for step in grid.spacing)
        self._lower = tuple(subdomain.shifted_interior(axis, -1) for axis in range(grid.dimension))
        self._upper = tuple(subdomain.shifted_interior(axis, +1) for axis in range(grid.dimension))

    def accumulate(self, state: FlowState, rate: FlowState) -> None:
        if self._nu == 0.0:
            return
        here = self._interior
        for component in range(self._dimension):
            field = state.velocity[component]
            middle = field[here]
            total = rate.velocity[component][here]
            for axis in range(self._dimension):
                curvature = field[self._upper[axis]] - 2.0 * middle + field[self._lower[axis]]
                total += self._nu * curvature * self._inverse_square[axis]
