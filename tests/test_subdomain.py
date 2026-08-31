from __future__ import annotations

import math

import pytest

from exaflow.config import Boundaries, BoundaryCondition, Face, FaceCondition, Grid
from exaflow.mpi.process_grid import ProcessGrid, choose_process_grid
from exaflow.mpi.subdomain import Subdomain

GRID = Grid((9, 7, 8), (1.0, 1.0, 1.0), 1)


def build_subdomains(num_procs: int) -> list[Subdomain]:
    process_grid = ProcessGrid(choose_process_grid(num_procs, GRID.shape, GRID.num_ghost_layers))
    return [Subdomain(GRID, process_grid, rank) for rank in range(process_grid.size)]


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8])
def test_blocks_cover_the_domain_exactly_once(num_procs: int) -> None:
    seen: set[tuple[int, int, int]] = set()
    for subdomain in build_subdomains(num_procs):
        (x0, x1), (y0, y1), (z0, z1) = subdomain.bounds
        block = {(i, j, k) for i in range(x0, x1) for j in range(y0, y1) for k in range(z0, z1)}
        assert not (block & seen), "blocks overlap"
        seen |= block
    assert len(seen) == math.prod(GRID.shape)


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8])
def test_block_sizes_differ_by_at_most_one_point(num_procs: int) -> None:
    subdomains = build_subdomains(num_procs)
    for axis in range(3):
        sizes = {subdomain.shape[axis] for subdomain in subdomains}
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8])
def test_the_global_slices_of_a_block_match_its_bounds(num_procs: int) -> None:
    for subdomain in build_subdomains(num_procs):
        assert subdomain.global_slices == tuple(slice(start, stop) for start, stop in subdomain.bounds)
        assert tuple(span.stop - span.start for span in subdomain.global_slices) == subdomain.shape


@pytest.mark.parametrize("num_ghost_layers", [1, 2, 3])
def test_the_padded_shape_adds_the_ghost_layers_at_both_ends(num_ghost_layers: int) -> None:
    grid = Grid((9, 7, 8), (1.0, 1.0, 1.0), num_ghost_layers)
    subdomain = Subdomain(grid, ProcessGrid((1, 1, 1)), 0)
    assert subdomain.padded_shape == tuple(length + 2 * num_ghost_layers for length in subdomain.shape)
    assert subdomain.interior == tuple(slice(num_ghost_layers, -num_ghost_layers) for _ in grid.shape)


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8])
def test_every_global_face_is_owned_by_at_least_one_rank(num_procs: int) -> None:
    subdomains = build_subdomains(num_procs)
    for face in Face:
        assert any(subdomain.is_on_face(face) for subdomain in subdomains)


def test_an_internal_face_is_not_a_domain_face() -> None:
    subdomains = build_subdomains(4)
    process_grid = subdomains[0].process_grid
    axis = next(a for a, count in enumerate(process_grid.counts) if count > 1)
    high = next(f for f in Face if f.axis == axis and not f.is_low)
    low = next(f for f in Face if f.axis == axis and f.is_low)
    first = next(s for s in subdomains if s.coords[axis] == 0)
    assert first.is_on_face(low)
    assert not first.is_on_face(high)


def test_a_face_on_an_axis_the_case_does_not_have_belongs_to_nobody() -> None:
    grid = Grid((8, 8), (1.0, 1.0), 1)
    subdomain = Subdomain(grid, ProcessGrid((1, 1)), 0)
    assert not subdomain.is_on_face(Face.FRONT)
    assert subdomain.find_neighbor_rank(Face.BACK, Boundaries()) is None


def test_neighbors_are_symmetric() -> None:
    subdomains = build_subdomains(8)
    boundaries = Boundaries()
    for subdomain in subdomains:
        for face in Face:
            neighbor = subdomain.find_neighbor_rank(face, boundaries)
            if neighbor is None:
                continue
            back = subdomains[neighbor].find_neighbor_rank(face.opposite, boundaries)
            assert back == subdomain.rank


def test_a_periodic_axis_wraps_onto_itself_on_one_rank() -> None:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    boundaries = Boundaries(left=periodic, right=periodic)
    subdomain = build_subdomains(1)[0]
    assert subdomain.find_neighbor_rank(Face.LEFT, boundaries) == subdomain.rank
    assert subdomain.find_neighbor_rank(Face.TOP, boundaries) is None


def test_a_periodic_axis_joins_the_first_and_last_rank() -> None:
    grid = Grid((16,), (1.0,), 1)
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    boundaries = Boundaries(left=periodic, right=periodic)
    subdomains = [Subdomain(grid, ProcessGrid((4,)), rank) for rank in range(4)]
    assert subdomains[0].find_neighbor_rank(Face.LEFT, boundaries) == 3
    assert subdomains[3].find_neighbor_rank(Face.RIGHT, boundaries) == 0
    assert subdomains[0].find_neighbor_rank(Face.RIGHT, boundaries) == 1


def test_shifted_interior_reaches_one_ghost_layer_on_each_side() -> None:
    subdomain = build_subdomains(1)[0]
    assert subdomain.shift_interior(0, -1)[0] == slice(0, -2)
    assert subdomain.shift_interior(0, +1)[0] == slice(2, None)
    assert subdomain.shift_interior(0, +1)[1] == subdomain.interior[1]


def test_shifted_interior_cannot_reach_past_the_ghost_layers() -> None:
    subdomain = build_subdomains(1)[0]
    with pytest.raises(ValueError, match="reaches past"):
        subdomain.shift_interior(0, 2)


def test_a_split_thinner_than_the_ghost_layers_is_refused() -> None:
    grid = Grid((8, 8, 8), (1.0, 1.0, 1.0), 2)
    with pytest.raises(ValueError, match="ghost layers"):
        Subdomain(grid, ProcessGrid((8, 1, 1)), 0)


def test_more_ranks_than_points_on_an_axis_is_refused() -> None:
    grid = Grid((4, 8, 8), (1.0, 1.0, 1.0), 1)
    with pytest.raises(ValueError, match="so some rank would own nothing"):
        Subdomain(grid, ProcessGrid((8, 1, 1)), 0)


def test_a_process_grid_of_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="grid has 3 axes but the process grid has 2"):
        Subdomain(GRID, ProcessGrid((2, 1)), 0)
