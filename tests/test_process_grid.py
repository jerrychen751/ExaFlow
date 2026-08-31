from __future__ import annotations

import math

import pytest

from exaflow.mpi.process_grid import ProcessGrid, choose_process_grid


@pytest.mark.parametrize("num_procs", [1, 2, 3, 4, 6, 8, 12, 16, 64])
@pytest.mark.parametrize("shape", [(64,), (32, 16), (9, 7, 8)])
def test_the_chosen_counts_multiply_to_the_rank_count(num_procs: int, shape: tuple[int, ...]) -> None:
    counts = choose_process_grid(num_procs, shape, 1)
    assert len(counts) == len(shape)
    assert math.prod(counts) == num_procs


@pytest.mark.parametrize("num_procs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("shape", [(64,), (32, 16), (9, 7, 8)])
def test_no_axis_is_split_thinner_than_its_ghost_layers(num_procs: int, shape: tuple[int, ...]) -> None:
    num_ghost_layers = 2
    counts = choose_process_grid(num_procs, shape, num_ghost_layers)
    assert all(points // count >= num_ghost_layers for count, points in zip(counts, shape))


def test_an_elongated_grid_puts_every_rank_on_its_long_axis() -> None:
    """
    The factor search used to stop at the square or cube root of the rank count, so it returned a split with more ranks than points on a short axis and Subdomain refused it.
    """

    assert choose_process_grid(8, (100, 2), 1) == (8, 1)
    assert choose_process_grid(64, (1000, 2, 2), 1) == (64, 1, 1)


def test_a_cube_of_ranks_over_a_cube_of_points_splits_evenly() -> None:
    assert choose_process_grid(8, (16, 16, 16), 1) == (2, 2, 2)


def test_a_rank_count_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="num_procs must be >= 1"):
        choose_process_grid(0, (8, 8), 1)


def test_a_rank_count_no_factorization_can_carry_is_refused() -> None:
    """
    Each axis of an 8-point grid takes at most 4 ranks against 2 ghost layers, so 4 * 4 * 4 = 64 ranks is the ceiling. 128 ranks has no surviving split, and this raises rather than hand Subdomain a decomposition it would refuse.
    """

    with pytest.raises(ValueError, match="cannot be split over a grid"):
        choose_process_grid(128, (8, 8, 8), 2)


@pytest.mark.parametrize("counts", [(4,), (2, 3), (2, 3, 4)])
def test_every_rank_maps_to_one_position_and_back(counts: tuple[int, ...]) -> None:
    process_grid = ProcessGrid(counts)
    coords = [process_grid.compute_coords(rank) for rank in range(process_grid.size)]
    assert len(set(coords)) == process_grid.size
    assert all(process_grid.compute_rank(coord) == rank for rank, coord in enumerate(coords))


def test_rank_numbering_runs_with_x_fastest() -> None:
    process_grid = ProcessGrid((2, 3))
    assert process_grid.compute_coords(1) == (1, 0)
    assert process_grid.compute_coords(2) == (0, 1)
    assert process_grid.compute_rank((1, 2)) == 5


def test_size_and_dimension_follow_the_counts() -> None:
    process_grid = ProcessGrid((2, 3, 4))
    assert process_grid.size == 24
    assert process_grid.dimension == 3


@pytest.mark.parametrize("counts", [(), (0,), (2, -1)])
def test_an_axis_with_no_rank_is_refused(counts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least one rank"):
        ProcessGrid(counts)


@pytest.mark.parametrize("rank", [-1, 6])
def test_a_rank_outside_the_grid_is_refused(rank: int) -> None:
    with pytest.raises(ValueError, match=r"rank must lie in \[0, 6\)"):
        ProcessGrid((2, 3)).compute_coords(rank)


@pytest.mark.parametrize("coords", [(2, 0), (0, -1)])
def test_a_coordinate_outside_its_axis_is_refused(coords: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="is outside an axis of"):
        ProcessGrid((2, 3)).compute_rank(coords)
