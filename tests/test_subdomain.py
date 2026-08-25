from __future__ import annotations

import pytest

from exaflow.config import Boundaries, BoundaryCondition, FaceCondition, Face, Grid
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
    assert len(seen) == 9 * 7 * 8


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8])
def test_block_sizes_differ_by_at_most_one_point(num_procs: int) -> None:
    subdomains = build_subdomains(num_procs)
    for axis in range(3):
        sizes = {subdomain.shape[axis] for subdomain in subdomains}
        assert max(sizes) - min(sizes) <= 1


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


def test_neighbors_are_symmetric() -> None:
    subdomains = build_subdomains(8)
    boundaries = Boundaries()
    for subdomain in subdomains:
        for face in Face:
            neighbor = subdomain.neighbor_rank(face, boundaries)
            if neighbor is None:
                continue
            back = subdomains[neighbor].neighbor_rank(face.opposite, boundaries)
            assert back == subdomain.rank


def test_a_periodic_axis_wraps_onto_itself_on_one_rank() -> None:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    boundaries = Boundaries(left=periodic, right=periodic)
    subdomain = build_subdomains(1)[0]
    assert subdomain.neighbor_rank(Face.LEFT, boundaries) == subdomain.rank
    assert subdomain.neighbor_rank(Face.TOP, boundaries) is None


def test_shifted_interior_cannot_reach_past_the_ghost_layers() -> None:
    subdomain = build_subdomains(1)[0]
    with pytest.raises(ValueError, match="reaches past"):
        subdomain.shifted_interior(0, 2)


def test_an_elongated_grid_puts_every_rank_on_its_long_axis() -> None:
    """The factor search used to stop at the square or cube root of the rank count, so it returned
    a split with more ranks than points on a short axis and Subdomain refused it."""

    assert choose_process_grid(8, (100, 2), 1) == (8, 1)
    assert choose_process_grid(64, (1000, 2, 2), 1) == (64, 1, 1)


def test_a_split_thinner_than_the_ghost_layers_is_refused() -> None:
    grid = Grid((8, 8, 8), (1.0, 1.0, 1.0), 2)
    with pytest.raises(ValueError, match="ghost layers"):
        Subdomain(grid, ProcessGrid((8, 1, 1)), 0)
