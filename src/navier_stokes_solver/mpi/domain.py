from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

from ..parameters import SimulationParameters


RankCoords1D: TypeAlias = int
RankCoords2D: TypeAlias = tuple[int, int, int]
RankCoords3D: TypeAlias = tuple[int, int, int, int]
RankCoords: TypeAlias = RankCoords1D | RankCoords2D | RankCoords3D


def parallelize_domain(array: np.ndarray, rank: int, sim_params: SimulationParameters) -> tuple[np.ndarray, RankCoords]:
    if sim_params.dimension == 1:
        return _parallelize_domain_1d(array, rank, sim_params)
    if sim_params.dimension == 2:
        return _parallelize_domain_2d(array, rank, sim_params)
    if sim_params.dimension == 3:
        return _parallelize_domain_3d(array, rank, sim_params)
    raise ValueError(f"Invalid dimension: {sim_params.dimension}")


def _parallelize_domain_1d(array: np.ndarray, rank: int, sim_params: SimulationParameters) -> tuple[np.ndarray, RankCoords1D]:
    num_procs = sim_params.num_procs
    nx = int(array.shape[0])

    local_size = nx // num_procs
    remainder = nx % num_procs

    start = rank * local_size + min(rank, remainder)
    end = start + local_size + (1 if rank < remainder else 0)

    local_array = array[start:end]
    return local_array, rank


def _parallelize_domain_2d(array: np.ndarray, rank: int, sim_params: SimulationParameters) -> tuple[np.ndarray, RankCoords2D]:
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)

    rank_x = rank % num_procs_x
    rank_y = rank // num_procs_x

    nx, ny = map(int, array.shape)

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    local_y = ny // num_procs_y
    remainder_y = ny % num_procs_y

    start_x = rank_x * local_x + min(rank_x, remainder_x)
    start_y = rank_y * local_y + min(rank_y, remainder_y)

    end_x = start_x + local_x + (1 if rank_x < remainder_x else 0)
    end_y = start_y + local_y + (1 if rank_y < remainder_y else 0)

    local_array = array[start_x:end_x, start_y:end_y]
    return local_array, (rank_x, rank_y, rank)


def _parallelize_domain_3d(array: np.ndarray, rank: int, sim_params: SimulationParameters) -> tuple[np.ndarray, RankCoords3D]:
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)

    rank_x = rank % num_procs_x
    rank_y = (rank // num_procs_x) % num_procs_y
    rank_z = rank // (num_procs_x * num_procs_y)

    nx, ny, nz = map(int, array.shape)

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    local_y = ny // num_procs_y
    remainder_y = ny % num_procs_y
    local_z = nz // num_procs_z
    remainder_z = nz % num_procs_z

    start_x = rank_x * local_x + min(rank_x, remainder_x)
    start_y = rank_y * local_y + min(rank_y, remainder_y)
    start_z = rank_z * local_z + min(rank_z, remainder_z)

    end_x = start_x + local_x + (1 if rank_x < remainder_x else 0)
    end_y = start_y + local_y + (1 if rank_y < remainder_y else 0)
    end_z = start_z + local_z + (1 if rank_z < remainder_z else 0)

    local_array = array[start_x:end_x, start_y:end_y, start_z:end_z]
    return local_array, (rank_x, rank_y, rank_z, rank)


def rejoin_array(
    array_to_fill: np.ndarray,
    local_array: np.ndarray,
    sim_params: SimulationParameters,
    *,
    exclude_ghosts: bool = False,
) -> None:
    comm = sim_params.comm
    if comm is None:
        array_to_fill[...] = local_array
        return

    subdomain = np.ascontiguousarray(local_array)
    arrays = comm.allgather(subdomain)

    if sim_params.dimension == 1:
        for r, part in enumerate(arrays):
            _rejoin_array_1d(array_to_fill, part, r, sim_params, exclude_ghosts)
        return

    if sim_params.dimension == 2:
        for r, part in enumerate(arrays):
            _rejoin_array_2d(array_to_fill, part, r, sim_params, exclude_ghosts)
        return

    if sim_params.dimension == 3:
        for r, part in enumerate(arrays):
            _rejoin_array_3d(array_to_fill, part, r, sim_params, exclude_ghosts)
        return

    raise ValueError(f"Invalid dimension: {sim_params.dimension}")


def _rejoin_array_1d(
    array_to_fill: np.ndarray,
    part: np.ndarray,
    rank: int,
    sim_params: SimulationParameters,
    exclude_ghosts: bool,
) -> None:
    nx = int(sim_params.domain[0])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_ghosts = sim_params.num_ghost_layers

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    start_x = rank * local_x + min(rank, remainder_x)
    end_x = start_x + local_x + (1 if rank < remainder_x else 0)

    offset = num_ghosts if exclude_ghosts else 0
    array_to_fill[start_x:end_x] = part[offset : offset + (end_x - start_x)]


def _rejoin_array_2d(
    array_to_fill: np.ndarray,
    part: np.ndarray,
    rank: int,
    sim_params: SimulationParameters,
    exclude_ghosts: bool,
) -> None:
    nx, ny = map(int, sim_params.domain[:2])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_ghosts = sim_params.num_ghost_layers

    rank_x = rank % num_procs_x
    rank_y = (rank // num_procs_x) % num_procs_y

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    local_y = ny // num_procs_y
    remainder_y = ny % num_procs_y

    start_x = rank_x * local_x + min(rank_x, remainder_x)
    end_x = start_x + local_x + (1 if rank_x < remainder_x else 0)

    start_y = rank_y * local_y + min(rank_y, remainder_y)
    end_y = start_y + local_y + (1 if rank_y < remainder_y else 0)

    off = num_ghosts if exclude_ghosts else 0
    array_to_fill[start_x:end_x, start_y:end_y] = part[
        off : off + (end_x - start_x),
        off : off + (end_y - start_y),
    ]


def _rejoin_array_3d(
    array_to_fill: np.ndarray,
    part: np.ndarray,
    rank: int,
    sim_params: SimulationParameters,
    exclude_ghosts: bool,
) -> None:
    nx, ny, nz = map(int, sim_params.domain[:3])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)
    num_ghosts = sim_params.num_ghost_layers

    rank_x = rank % num_procs_x
    rank_y = (rank // num_procs_x) % num_procs_y
    rank_z = rank // (num_procs_x * num_procs_y)

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    local_y = ny // num_procs_y
    remainder_y = ny % num_procs_y
    local_z = nz // num_procs_z
    remainder_z = nz % num_procs_z

    start_x = rank_x * local_x + min(rank_x, remainder_x)
    end_x = start_x + local_x + (1 if rank_x < remainder_x else 0)

    start_y = rank_y * local_y + min(rank_y, remainder_y)
    end_y = start_y + local_y + (1 if rank_y < remainder_y else 0)

    start_z = rank_z * local_z + min(rank_z, remainder_z)
    end_z = start_z + local_z + (1 if rank_z < remainder_z else 0)

    off = num_ghosts if exclude_ghosts else 0
    array_to_fill[start_x:end_x, start_y:end_y, start_z:end_z] = part[
        off : off + (end_x - start_x),
        off : off + (end_y - start_y),
        off : off + (end_z - start_z),
    ]

