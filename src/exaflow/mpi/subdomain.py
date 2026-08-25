from __future__ import annotations

from dataclasses import dataclass

from ..config.boundaries import Boundaries, Face
from ..config.grid import Grid
from .process_grid import ProcessGrid


@dataclass(frozen=True, slots=True)
class Subdomain:
    """
    The block of the global grid that one rank owns, and where that block sits. This is the single owner of the block-decomposition arithmetic: every consumer that needs local bounds, a padded shape, or a face test reads them here rather than recomputing them.

    Axis `a` is split into `process_grid.counts[a]` blocks. The first `n % parts` blocks take one extra point, so the blocks differ by at most one point and together cover the axis exactly.
    """

    grid: Grid
    process_grid: ProcessGrid
    rank: int

    def __post_init__(self) -> None:
        if self.grid.dimension != self.process_grid.dimension:
            raise ValueError(
                f"grid has {self.grid.dimension} axes but the process grid has {self.process_grid.dimension}."
            )
        pad = self.grid.num_ghost_layers
        for axis, (points, parts) in enumerate(zip(self.grid.shape, self.process_grid.counts)):
            if parts > points:
                raise ValueError(
                    f"axis {axis} has {points} grid points but {parts} ranks, so some rank would own nothing."
                )
            if points // parts < pad:
                raise ValueError(
                    f"axis {axis} splits {points} grid points over {parts} ranks, so the smallest block is "
                    f"{points // parts} points thick against {pad} ghost layers. The ghost exchange reads the "
                    f"outermost {pad} real layers, so a thinner block would send its own ghost cells as real data."
                )

    @property
    def coords(self) -> tuple[int, ...]:
        return self.process_grid.compute_coords(self.rank)

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        """
        The half-open range of global grid indices this rank owns, as (start, stop) per axis.
        """

        ranges = []
        for points, parts, index in zip(self.grid.shape, self.process_grid.counts, self.coords):
            block, remainder = divmod(points, parts)
            start = index * block + min(index, remainder)
            stop = start + block + (1 if index < remainder else 0)
            ranges.append((start, stop))
        return tuple(ranges)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The local shape without ghost layers.
        """

        return tuple(stop - start for start, stop in self.bounds)

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """
        The local shape with `grid.num_ghost_layers` added at each end of every axis. This is the shape of every array the solver actually works on.
        """

        pad = 2 * self.grid.num_ghost_layers
        return tuple(length + pad for length in self.shape)

    @property
    def global_slices(self) -> tuple[slice, ...]:
        """
        Where this rank's block lands in a full-domain array.
        """

        return tuple(slice(start, stop) for start, stop in self.bounds)

    @property
    def interior(self) -> tuple[slice, ...]:
        """
        The real cells inside a padded local array, with the ghost layers stripped off.
        """

        pad = self.grid.num_ghost_layers
        return tuple(slice(pad, -pad) for _ in self.grid.shape)

    def shift_interior(self, axis: int, offset: int) -> tuple[slice, ...]:
        """
        The interior slices with `axis` moved by `offset` grid points, which selects the neighbor of every interior cell along that axis. An offset of -1 reaches into the ghost layer at the low end of the axis and +1 reaches into the ghost layer at the high end, so the caller must complete the ghost exchange first.

        The offset must not exceed `grid.num_ghost_layers`, because nothing beyond the ghost layers is addressable.
        """

        if abs(offset) > self.grid.num_ghost_layers:
            raise ValueError(
                f"offset {offset} reaches past {self.grid.num_ghost_layers} ghost layers."
            )
        spans = list(self.interior)
        span = spans[axis]
        stop = span.stop + offset
        spans[axis] = slice(span.start + offset, stop if stop != 0 else None)
        return tuple(spans)

    def is_on_face(self, face: Face) -> bool:
        """
        Report whether this rank owns the given face of the global domain.

        False on an internal partition face, where the cells beyond the ghost layer belong to a neighboring rank. A one-sided boundary stencil and a boundary condition are correct only where this returns True; everywhere else the ghost layer already carries the neighbor's data.
        """

        if face.axis >= self.grid.dimension:
            return False
        index = self.coords[face.axis]
        if face.is_low:
            return index == 0
        return index == self.process_grid.counts[face.axis] - 1

    def find_neighbor_rank(self, face: Face, boundaries: Boundaries) -> int | None:
        """
        The rank holding the cells just beyond this face, or None when this face is a domain boundary that does not wrap. A periodic axis wraps, so the first and last ranks on it are neighbors. On a periodic axis carrying a single rank the answer is this rank itself, because the block wraps onto its own opposite face.
        """

        if face.axis >= self.grid.dimension:
            return None
        axis = face.axis
        parts = self.process_grid.counts[axis]
        index = self.coords[axis]
        step = -1 if face.is_low else 1
        target = index + step
        if not 0 <= target < parts:
            if not boundaries.is_periodic(axis):
                return None
            target %= parts
        coords = list(self.coords)
        coords[axis] = target
        return self.process_grid.compute_rank(tuple(coords))
