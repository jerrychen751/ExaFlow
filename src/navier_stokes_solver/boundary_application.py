from __future__ import annotations

from typing import Any

import numpy as np

from .boundary_conditions import BoundaryCondition
from .parameters import SimulationParameters


def initialize_bc(
    rank_coords: Any,
    sim_params: SimulationParameters,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    p: np.ndarray | None = None,
) -> None:
    if sim_params.dimension == 1:
        if u is None or p is None:
            raise ValueError("1D boundary conditions require u and p arrays.")
        _initialize_bc_1d(u, p, int(rank_coords), sim_params)
        return
    if sim_params.dimension == 2:
        if u is None or v is None or p is None:
            raise ValueError("2D boundary conditions require u, v, and p arrays.")
        _initialize_bc_2d(u, v, p, rank_coords, sim_params)
        return
    if sim_params.dimension == 3:
        if u is None or v is None or w is None or p is None:
            raise ValueError("3D boundary conditions require u, v, w, and p arrays.")
        _initialize_bc_3d(u, v, w, p, rank_coords, sim_params)
        return
    raise ValueError(f"Unsupported dimension {sim_params.dimension}.")


def update_bc(
    rank_coords: Any,
    sim_params: SimulationParameters,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    p: np.ndarray | None = None,
) -> None:
    if sim_params.dimension == 1:
        if u is None or p is None:
            raise ValueError("1D boundary conditions require u and p arrays.")
        _update_bc_1d(u, p, int(rank_coords), sim_params)
        return
    if sim_params.dimension == 2:
        if u is None or v is None or p is None:
            raise ValueError("2D boundary conditions require u, v, and p arrays.")
        _update_bc_2d(u, v, p, rank_coords, sim_params)
        return
    if sim_params.dimension == 3:
        if u is None or v is None or w is None or p is None:
            raise ValueError("3D boundary conditions require u, v, w, and p arrays.")
        _update_bc_3d(u, v, w, p, rank_coords, sim_params)
        return
    raise ValueError(f"Unsupported dimension {sim_params.dimension}.")


def _initialize_bc_1d(u: np.ndarray, p: np.ndarray, rank_x: int, sim_params: SimulationParameters) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    num_ghosts = sim_params.num_ghost_layers
    num_procs_x = int(sim_params.num_procs_x or 1)

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP:
                u[0:num_ghosts] = 0
            case BoundaryCondition.INFLOW:
                u[0:num_ghosts] = float(sim_params.left_inflow.get("u", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[0:num_ghosts] = float(sim_params.left_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP:
                u[-num_ghosts:] = 0
            case BoundaryCondition.INFLOW:
                u[-num_ghosts:] = float(sim_params.right_inflow.get("u", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[-num_ghosts:] = float(sim_params.right_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")


def _update_bc_1d(u: np.ndarray, p: np.ndarray, rank_x: int, sim_params: SimulationParameters) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    num_ghosts = sim_params.num_ghost_layers
    num_procs_x = int(sim_params.num_procs_x or 1)

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[0:num_ghosts] = p[num_ghosts]
            case BoundaryCondition.OUTFLOW:
                u[0:num_ghosts] = u[num_ghosts]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[-num_ghosts:] = p[-num_ghosts - 1]
            case BoundaryCondition.OUTFLOW:
                u[-num_ghosts:] = u[-num_ghosts - 1]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")


def _initialize_bc_2d(u: np.ndarray, v: np.ndarray, p: np.ndarray, rank_coords: Any, sim_params: SimulationParameters) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    top = sim_params.top_wall
    bottom = sim_params.bottom_wall

    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_ghosts = sim_params.num_ghost_layers

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP:
                u[0:num_ghosts, :] = 0
                v[0:num_ghosts, :] = 0
            case BoundaryCondition.SLIP:
                u[0:num_ghosts, :] = 0
            case BoundaryCondition.INFLOW:
                u[0:num_ghosts, :] = float(sim_params.left_inflow.get("u", 0.0))
                v[0:num_ghosts, :] = float(sim_params.left_inflow.get("v", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[0:num_ghosts, :] = float(sim_params.left_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP:
                u[-num_ghosts:, :] = 0
                v[-num_ghosts:, :] = 0
            case BoundaryCondition.SLIP:
                u[-num_ghosts:, :] = 0
            case BoundaryCondition.INFLOW:
                u[-num_ghosts:, :] = float(sim_params.right_inflow.get("u", 0.0))
                v[-num_ghosts:, :] = float(sim_params.right_inflow.get("v", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[-num_ghosts:, :] = float(sim_params.right_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")

    if rank_y == 0:
        match top:
            case BoundaryCondition.NO_SLIP:
                u[:, 0:num_ghosts] = 0
                v[:, 0:num_ghosts] = 0
            case BoundaryCondition.SLIP:
                v[:, 0:num_ghosts] = 0
            case BoundaryCondition.INFLOW:
                u[:, 0:num_ghosts] = float(sim_params.top_inflow.get("u", 0.0))
                v[:, 0:num_ghosts] = float(sim_params.top_inflow.get("v", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, 0:num_ghosts] = float(sim_params.top_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {top}.")

    if rank_y == num_procs_y - 1:
        match bottom:
            case BoundaryCondition.NO_SLIP:
                u[:, -num_ghosts:] = 0
                v[:, -num_ghosts:] = 0
            case BoundaryCondition.SLIP:
                v[:, -num_ghosts:] = 0
            case BoundaryCondition.INFLOW:
                u[:, -num_ghosts:] = float(sim_params.bottom_inflow.get("u", 0.0))
                v[:, -num_ghosts:] = float(sim_params.bottom_inflow.get("v", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, -num_ghosts:] = float(sim_params.bottom_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {bottom}.")


def _update_bc_2d(u: np.ndarray, v: np.ndarray, p: np.ndarray, rank_coords: Any, sim_params: SimulationParameters) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    top = sim_params.top_wall
    bottom = sim_params.bottom_wall

    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_ghosts = sim_params.num_ghost_layers

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[0:num_ghosts, :] = np.tile(p[num_ghosts, :], (num_ghosts, 1))
            case BoundaryCondition.OUTFLOW:
                u[0:num_ghosts, :] = np.tile(u[num_ghosts, :], (num_ghosts, 1))
                v[0:num_ghosts, :] = np.tile(v[num_ghosts, :], (num_ghosts, 1))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[-num_ghosts:, :] = np.tile(p[-num_ghosts - 1, :], (num_ghosts, 1))
            case BoundaryCondition.OUTFLOW:
                u[-num_ghosts:, :] = np.tile(u[-num_ghosts - 1, :], (num_ghosts, 1))
                v[-num_ghosts:, :] = np.tile(v[-num_ghosts - 1, :], (num_ghosts, 1))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")

    if rank_y == 0:
        match top:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, 0:num_ghosts] = np.tile(p[:, num_ghosts][:, np.newaxis], (1, num_ghosts))
            case BoundaryCondition.OUTFLOW:
                u[:, 0:num_ghosts] = np.tile(u[:, num_ghosts][:, np.newaxis], (1, num_ghosts))
                v[:, 0:num_ghosts] = np.tile(v[:, num_ghosts][:, np.newaxis], (1, num_ghosts))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {top}.")

    if rank_y == num_procs_y - 1:
        match bottom:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, -num_ghosts:] = np.tile(p[:, -num_ghosts - 1][:, np.newaxis], (1, num_ghosts))
            case BoundaryCondition.OUTFLOW:
                u[:, -num_ghosts:] = np.tile(u[:, -num_ghosts - 1][:, np.newaxis], (1, num_ghosts))
                v[:, -num_ghosts:] = np.tile(v[:, -num_ghosts - 1][:, np.newaxis], (1, num_ghosts))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {bottom}.")


def _initialize_bc_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    rank_coords: Any,
    sim_params: SimulationParameters,
) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    top = sim_params.top_wall
    bottom = sim_params.bottom_wall
    front = sim_params.front_wall
    back = sim_params.back_wall

    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    rank_z = int(rank_coords[2])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)
    num_ghosts = sim_params.num_ghost_layers

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP:
                u[0:num_ghosts, :, :] = 0
                v[0:num_ghosts, :, :] = 0
                w[0:num_ghosts, :, :] = 0
            case BoundaryCondition.SLIP:
                u[0:num_ghosts, :, :] = 0
            case BoundaryCondition.INFLOW:
                u[0:num_ghosts, :, :] = float(sim_params.left_inflow.get("u", 0.0))
                v[0:num_ghosts, :, :] = float(sim_params.left_inflow.get("v", 0.0))
                w[0:num_ghosts, :, :] = float(sim_params.left_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[0:num_ghosts, :, :] = float(sim_params.left_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP:
                u[-num_ghosts:, :, :] = 0
                v[-num_ghosts:, :, :] = 0
                w[-num_ghosts:, :, :] = 0
            case BoundaryCondition.SLIP:
                u[-num_ghosts:, :, :] = 0
            case BoundaryCondition.INFLOW:
                u[-num_ghosts:, :, :] = float(sim_params.right_inflow.get("u", 0.0))
                v[-num_ghosts:, :, :] = float(sim_params.right_inflow.get("v", 0.0))
                w[-num_ghosts:, :, :] = float(sim_params.right_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[-num_ghosts:, :, :] = float(sim_params.right_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")

    if rank_y == 0:
        match top:
            case BoundaryCondition.NO_SLIP:
                u[:, 0:num_ghosts, :] = 0
                v[:, 0:num_ghosts, :] = 0
                w[:, 0:num_ghosts, :] = 0
            case BoundaryCondition.SLIP:
                v[:, 0:num_ghosts, :] = 0
            case BoundaryCondition.INFLOW:
                u[:, 0:num_ghosts, :] = float(sim_params.top_inflow.get("u", 0.0))
                v[:, 0:num_ghosts, :] = float(sim_params.top_inflow.get("v", 0.0))
                w[:, 0:num_ghosts, :] = float(sim_params.top_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, 0:num_ghosts, :] = float(sim_params.top_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {top}.")

    if rank_y == num_procs_y - 1:
        match bottom:
            case BoundaryCondition.NO_SLIP:
                u[:, -num_ghosts:, :] = 0
                v[:, -num_ghosts:, :] = 0
                w[:, -num_ghosts:, :] = 0
            case BoundaryCondition.SLIP:
                v[:, -num_ghosts:, :] = 0
            case BoundaryCondition.INFLOW:
                u[:, -num_ghosts:, :] = float(sim_params.bottom_inflow.get("u", 0.0))
                v[:, -num_ghosts:, :] = float(sim_params.bottom_inflow.get("v", 0.0))
                w[:, -num_ghosts:, :] = float(sim_params.bottom_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, -num_ghosts:, :] = float(sim_params.bottom_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {bottom}.")

    if rank_z == 0:
        match front:
            case BoundaryCondition.NO_SLIP:
                u[:, :, 0:num_ghosts] = 0
                v[:, :, 0:num_ghosts] = 0
                w[:, :, 0:num_ghosts] = 0
            case BoundaryCondition.SLIP:
                w[:, :, 0:num_ghosts] = 0
            case BoundaryCondition.INFLOW:
                u[:, :, 0:num_ghosts] = float(sim_params.front_inflow.get("u", 0.0))
                v[:, :, 0:num_ghosts] = float(sim_params.front_inflow.get("v", 0.0))
                w[:, :, 0:num_ghosts] = float(sim_params.front_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, :, 0:num_ghosts] = float(sim_params.front_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {front}.")

    if rank_z == num_procs_z - 1:
        match back:
            case BoundaryCondition.NO_SLIP:
                u[:, :, -num_ghosts:] = 0
                v[:, :, -num_ghosts:] = 0
                w[:, :, -num_ghosts:] = 0
            case BoundaryCondition.SLIP:
                w[:, :, -num_ghosts:] = 0
            case BoundaryCondition.INFLOW:
                u[:, :, -num_ghosts:] = float(sim_params.back_inflow.get("u", 0.0))
                v[:, :, -num_ghosts:] = float(sim_params.back_inflow.get("v", 0.0))
                w[:, :, -num_ghosts:] = float(sim_params.back_inflow.get("w", 0.0))
            case BoundaryCondition.OUTFLOW:
                p[:, :, -num_ghosts:] = float(sim_params.back_outflow.get("p", 0.0))
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {back}.")


def _update_bc_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    rank_coords: Any,
    sim_params: SimulationParameters,
) -> None:
    left = sim_params.left_wall
    right = sim_params.right_wall
    top = sim_params.top_wall
    bottom = sim_params.bottom_wall
    front = sim_params.front_wall
    back = sim_params.back_wall

    rank_x = int(rank_coords[0])
    rank_y = int(rank_coords[1])
    rank_z = int(rank_coords[2])
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)
    num_ghosts = sim_params.num_ghost_layers

    if rank_x == 0:
        match left:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[0:num_ghosts, :, :] = p[num_ghosts, :, :][np.newaxis, :, :]
            case BoundaryCondition.OUTFLOW:
                u[0:num_ghosts, :, :] = u[num_ghosts, :, :][np.newaxis, :, :]
                v[0:num_ghosts, :, :] = v[num_ghosts, :, :][np.newaxis, :, :]
                w[0:num_ghosts, :, :] = w[num_ghosts, :, :][np.newaxis, :, :]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {left}.")

    if rank_x == num_procs_x - 1:
        match right:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[-num_ghosts:, :, :] = p[-num_ghosts - 1, :, :][np.newaxis, :, :]
            case BoundaryCondition.OUTFLOW:
                u[-num_ghosts:, :, :] = u[-num_ghosts - 1, :, :][np.newaxis, :, :]
                v[-num_ghosts:, :, :] = v[-num_ghosts - 1, :, :][np.newaxis, :, :]
                w[-num_ghosts:, :, :] = w[-num_ghosts - 1, :, :][np.newaxis, :, :]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {right}.")

    if rank_y == 0:
        match top:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, 0:num_ghosts, :] = p[:, num_ghosts, :][:, np.newaxis, :]
            case BoundaryCondition.OUTFLOW:
                u[:, 0:num_ghosts, :] = u[:, num_ghosts, :][:, np.newaxis, :]
                v[:, 0:num_ghosts, :] = v[:, num_ghosts, :][:, np.newaxis, :]
                w[:, 0:num_ghosts, :] = w[:, num_ghosts, :][:, np.newaxis, :]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {top}.")

    if rank_y == num_procs_y - 1:
        match bottom:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, -num_ghosts:, :] = p[:, -num_ghosts - 1, :][:, np.newaxis, :]
            case BoundaryCondition.OUTFLOW:
                u[:, -num_ghosts:, :] = u[:, -num_ghosts - 1, :][:, np.newaxis, :]
                v[:, -num_ghosts:, :] = v[:, -num_ghosts - 1, :][:, np.newaxis, :]
                w[:, -num_ghosts:, :] = w[:, -num_ghosts - 1, :][:, np.newaxis, :]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {bottom}.")

    if rank_z == 0:
        match front:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, :, 0:num_ghosts] = p[:, :, num_ghosts][:, :, np.newaxis]
            case BoundaryCondition.OUTFLOW:
                u[:, :, 0:num_ghosts] = u[:, :, num_ghosts][:, :, np.newaxis]
                v[:, :, 0:num_ghosts] = v[:, :, num_ghosts][:, :, np.newaxis]
                w[:, :, 0:num_ghosts] = w[:, :, num_ghosts][:, :, np.newaxis]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {front}.")

    if rank_z == num_procs_z - 1:
        match back:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.SLIP | BoundaryCondition.INFLOW:
                p[:, :, -num_ghosts:] = p[:, :, -num_ghosts - 1][:, :, np.newaxis]
            case BoundaryCondition.OUTFLOW:
                u[:, :, -num_ghosts:] = u[:, :, -num_ghosts - 1][:, :, np.newaxis]
                v[:, :, -num_ghosts:] = v[:, :, -num_ghosts - 1][:, :, np.newaxis]
                w[:, :, -num_ghosts:] = w[:, :, -num_ghosts - 1][:, :, np.newaxis]
            case BoundaryCondition.PERIODIC:
                pass
            case BoundaryCondition.TIME_DEPENDENT:
                raise ValueError("Time-dependent BC not implemented.")
            case _:
                raise ValueError(f"Invalid boundary condition: {back}.")

