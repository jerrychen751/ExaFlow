from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..parameters import SimulationParameters


def write_csv(
    rank_coords: Any,
    sim_params: SimulationParameters,
    name: str,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    p: np.ndarray | None = None,
    with_ghosts: bool = False,
    out_dir: str = "out",
) -> None:
    if sim_params.dimension == 1:
        if u is None or p is None:
            raise ValueError("1D CSV output requires u and p arrays.")
        write_csv_1d(u, p, int(rank_coords), sim_params, name, with_ghosts=with_ghosts, out_dir=out_dir)
        return
    if sim_params.dimension == 2:
        if u is None or v is None or p is None:
            raise ValueError("2D CSV output requires u, v and p arrays.")
        write_csv_2d(u, v, p, rank_coords, sim_params, name, with_ghosts=with_ghosts, out_dir=out_dir)
        return
    if sim_params.dimension == 3:
        if u is None or v is None or w is None or p is None:
            raise ValueError("3D CSV output requires u, v, w and p arrays.")
        write_csv_3d(u, v, w, p, rank_coords, sim_params, name, with_ghosts=with_ghosts, out_dir=out_dir)
        return
    raise ValueError(f"Invalid dimension {sim_params.dimension}.")


def _atomic_write(path: str, contents: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.partial"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(contents)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_csv_1d(
    u: np.ndarray,
    p: np.ndarray,
    rank: int,
    sim_params: SimulationParameters,
    name: str,
    *,
    with_ghosts: bool = False,
    out_dir: str = "out",
) -> None:
    nx = int(sim_params.domain[0])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_ghosts = sim_params.num_ghost_layers

    local_size = nx // num_procs_x
    remainder = nx % num_procs_x

    start_x = rank * local_size + min(rank, remainder)
    end_x = start_x + local_size + (1 if rank < remainder else 0)

    lines = ["x,u,p\n"]
    if with_ghosts:
        for i in range(start_x - num_ghosts, end_x + num_ghosts):
            lines.append(f"{i}, {u[i - start_x + num_ghosts]}, {p[i - start_x + num_ghosts]}\n")
    else:
        for i in range(start_x, end_x):
            lines.append(f"{i}, {u[i - start_x]}, {p[i - start_x]}\n")

    out_path = os.path.join(out_dir, f"{name}_{rank}.csv")
    _atomic_write(out_path, "".join(lines))


def write_csv_2d(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rank_coords: Any,
    sim_params: SimulationParameters,
    name: str,
    *,
    with_ghosts: bool = False,
    out_dir: str = "out",
) -> None:
    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    rank = int(rank_coords[2])

    nx, ny = map(int, sim_params.domain[:2])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_ghosts = sim_params.num_ghost_layers

    local_x = nx // num_procs_x
    remainder_x = nx % num_procs_x
    local_y = ny // num_procs_y
    remainder_y = ny % num_procs_y

    start_x = rank_x * local_x + min(rank_x, remainder_x)
    start_y = rank_y * local_y + min(rank_y, remainder_y)

    end_x = start_x + local_x + (1 if rank_x < remainder_x else 0)
    end_y = start_y + local_y + (1 if rank_y < remainder_y else 0)

    lines = ["x,y,u,v,p\n"]
    if with_ghosts:
        for i in range(start_x - num_ghosts, end_x + num_ghosts):
            for j in range(start_y - num_ghosts, end_y + num_ghosts):
                ii = i - start_x + num_ghosts
                jj = j - start_y + num_ghosts
                lines.append(f"{i}, {j}, {u[ii, jj]}, {v[ii, jj]}, {p[ii, jj]}\n")
    else:
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                ii = i - start_x
                jj = j - start_y
                lines.append(f"{i}, {j}, {u[ii, jj]}, {v[ii, jj]}, {p[ii, jj]}\n")

    out_path = os.path.join(out_dir, f"{name}_{rank}.csv")
    _atomic_write(out_path, "".join(lines))


def write_csv_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    rank_coords: Any,
    sim_params: SimulationParameters,
    name: str,
    *,
    with_ghosts: bool = False,
    out_dir: str = "out",
) -> None:
    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    rank_z = int(rank_coords[2])
    rank = int(rank_coords[3])

    nx, ny, nz = map(int, sim_params.domain[:3])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)
    num_ghosts = sim_params.num_ghost_layers

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

    lines = ["x,y,z,u,v,w,p\n"]
    if with_ghosts:
        for i in range(start_x - num_ghosts, end_x + num_ghosts):
            for j in range(start_y - num_ghosts, end_y + num_ghosts):
                for k in range(start_z - num_ghosts, end_z + num_ghosts):
                    ii = i - start_x + num_ghosts
                    jj = j - start_y + num_ghosts
                    kk = k - start_z + num_ghosts
                    lines.append(f"{i}, {j}, {k}, {u[ii, jj, kk]}, {v[ii, jj, kk]}, {w[ii, jj, kk]}, {p[ii, jj, kk]}\n")
    else:
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                for k in range(start_z, end_z):
                    ii = i - start_x
                    jj = j - start_y
                    kk = k - start_z
                    lines.append(f"{i}, {j}, {k}, {u[ii, jj, kk]}, {v[ii, jj, kk]}, {w[ii, jj, kk]}, {p[ii, jj, kk]}\n")

    out_path = os.path.join(out_dir, f"{name}_{rank}.csv")
    _atomic_write(out_path, "".join(lines))


def write_total_array_to_csv(
    name: str,
    *,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    out_dir: str = "out",
) -> None:
    """
    Write a full-domain array to a single CSV file (rank 0 responsibility).
    """

    dimension = len(u.shape)
    lines: list[str] = []
    if dimension == 1:
        lines.append("x,u,p\n")
        nx = int(u.shape[0])
        for i in range(nx):
            lines.append(f"{i}, {u[i]}, {p[i]}\n")
    elif dimension == 2:
        if v is None:
            raise ValueError("2D total CSV output requires v.")
        lines.append("x,y,u,v,p\n")
        nx, ny = map(int, u.shape)
        for i in range(nx):
            for j in range(ny):
                lines.append(f"{i}, {j}, {u[i, j]}, {v[i, j]}, {p[i, j]}\n")
    elif dimension == 3:
        if v is None or w is None:
            raise ValueError("3D total CSV output requires v and w.")
        lines.append("x,y,z,u,v,w,p\n")
        nx, ny, nz = map(int, u.shape)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    lines.append(f"{i}, {j}, {k}, {u[i, j, k]}, {v[i, j, k]}, {w[i, j, k]}, {p[i, j, k]}\n")
    else:
        raise ValueError(f"Unsupported dimension {dimension}.")

    out_path = os.path.join(out_dir, f"{name}_Total.csv")
    _atomic_write(out_path, "".join(lines))

