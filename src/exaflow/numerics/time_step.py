from __future__ import annotations

import numpy as np

from ..boundary_application import update_bc
from ..mpi.ghost_layers import recv_ghost_info, send_ghost_info
from ..parameters import RankCoords, SimulationParameters
from .convection_3d import convection_explicit_3d, convection_explicit_3d_boundary
from .diffusion_3d import diffusion_explicit_3d, diffusion_explicit_3d_boundary


def spatial_operator(
    *,
    sim_params: SimulationParameters,
    rank_coords: RankCoords,
    p_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    k_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RHS increments (k arrays) for the current timestep stage.

    Important invariant: k arrays are treated as scratch buffers and are fully
    overwritten/initialized here. Callers must not assume prior contents are preserved.
    """

    # k arrays are scratch; ensure deterministic behavior.
    k_u.fill(0.0)
    k_v.fill(0.0)
    k_w.fill(0.0)
    k_p.fill(0.0)

    # Ghost exchange for boundary calculations (interior stencils do not need remote data)
    req_u, buf_u = send_ghost_info(u_old, sim_params, rank_coords)
    req_v, buf_v = send_ghost_info(v_old, sim_params, rank_coords)
    req_w, buf_w = send_ghost_info(w_old, sim_params, rank_coords)
    req_p, buf_p = send_ghost_info(p_old, sim_params, rank_coords)

    if sim_params.include_convection:
        k_u, k_v, k_w = convection_explicit_3d(u_old, v_old, w_old, k_u, k_v, k_w, sim_params)
    if sim_params.include_diffusion:
        k_u, k_v, k_w = diffusion_explicit_3d(u_old, v_old, w_old, k_u, k_v, k_w, sim_params)

    recv_ghost_info(u_old, sim_params, rank_coords, req_u, buf_u)
    recv_ghost_info(v_old, sim_params, rank_coords, req_v, buf_v)
    recv_ghost_info(w_old, sim_params, rank_coords, req_w, buf_w)
    recv_ghost_info(p_old, sim_params, rank_coords, req_p, buf_p)

    update_bc(rank_coords, sim_params, u=u_old, v=v_old, w=w_old, p=p_old)

    if sim_params.include_convection:
        k_u, k_v, k_w = convection_explicit_3d_boundary(u_old, v_old, w_old, k_u, k_v, k_w, sim_params)
    if sim_params.include_diffusion:
        k_u, k_v, k_w = diffusion_explicit_3d_boundary(u_old, v_old, w_old, k_u, k_v, k_w, sim_params)

    return k_u, k_v, k_w, k_p


def advance_time_step_first_order(
    *,
    sim_params: SimulationParameters,
    rank_coords: RankCoords,
    p_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    p_new: np.ndarray,
    u_new: np.ndarray,
    v_new: np.ndarray,
    w_new: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    k_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler (first-order explicit).
    """

    k_u, k_v, k_w, k_p = spatial_operator(
        sim_params=sim_params,
        rank_coords=rank_coords,
        p_old=p_old,
        u_old=u_old,
        v_old=v_old,
        w_old=w_old,
        k_u=k_u,
        k_v=k_v,
        k_w=k_w,
        k_p=k_p,
    )

    u_new = u_old + k_u
    v_new = v_old + k_v
    w_new = w_old + k_w
    p_new = p_old + k_p
    return u_new, v_new, w_new, p_new


def advance_time_step_second_order(
    *,
    sim_params: SimulationParameters,
    rank_coords: RankCoords,
    p_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    p_new: np.ndarray,
    u_new: np.ndarray,
    v_new: np.ndarray,
    w_new: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    k_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Midpoint method (second-order explicit Runge–Kutta).
    """

    betas = [0.0, 0.5]
    for beta in betas:
        stage_u = u_old + beta * k_u
        stage_v = v_old + beta * k_v
        stage_w = w_old + beta * k_w
        stage_p = p_old + beta * k_p
        k_u, k_v, k_w, k_p = spatial_operator(
            sim_params=sim_params,
            rank_coords=rank_coords,
            p_old=stage_p,
            u_old=stage_u,
            v_old=stage_v,
            w_old=stage_w,
            k_u=k_u,
            k_v=k_v,
            k_w=k_w,
            k_p=k_p,
        )

    u_new = u_old + k_u
    v_new = v_old + k_v
    w_new = w_old + k_w
    p_new = p_old + k_p
    return u_new, v_new, w_new, p_new


def advance_time_step_third_order(
    *,
    sim_params: SimulationParameters,
    rank_coords: RankCoords,
    p_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    p_new: np.ndarray,
    u_new: np.ndarray,
    v_new: np.ndarray,
    w_new: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    k_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shu–Osher third-order TVD Runge–Kutta.
    """

    k_u, k_v, k_w, k_p = spatial_operator(
        sim_params=sim_params,
        rank_coords=rank_coords,
        p_old=p_old,
        u_old=u_old,
        v_old=v_old,
        w_old=w_old,
        k_u=k_u,
        k_v=k_v,
        k_w=k_w,
        k_p=k_p,
    )

    u_1 = u_old + k_u
    v_1 = v_old + k_v
    w_1 = w_old + k_w
    p_1 = p_old + k_p

    k_u, k_v, k_w, k_p = spatial_operator(
        sim_params=sim_params,
        rank_coords=rank_coords,
        p_old=p_1,
        u_old=u_1,
        v_old=v_1,
        w_old=w_1,
        k_u=k_u,
        k_v=k_v,
        k_w=k_w,
        k_p=k_p,
    )

    u_2 = 0.75 * u_old + 0.25 * (u_1 + k_u)
    v_2 = 0.75 * v_old + 0.25 * (v_1 + k_v)
    w_2 = 0.75 * w_old + 0.25 * (w_1 + k_w)
    p_2 = 0.75 * p_old + 0.25 * (p_1 + k_p)

    k_u, k_v, k_w, k_p = spatial_operator(
        sim_params=sim_params,
        rank_coords=rank_coords,
        p_old=p_2,
        u_old=u_2,
        v_old=v_2,
        w_old=w_2,
        k_u=k_u,
        k_v=k_v,
        k_w=k_w,
        k_p=k_p,
    )

    u_new = (1.0 / 3.0) * u_old + (2.0 / 3.0) * (u_2 + k_u)
    v_new = (1.0 / 3.0) * v_old + (2.0 / 3.0) * (v_2 + k_v)
    w_new = (1.0 / 3.0) * w_old + (2.0 / 3.0) * (w_2 + k_w)
    p_new = (1.0 / 3.0) * p_old + (2.0 / 3.0) * (p_2 + k_p)
    return u_new, v_new, w_new, p_new


def advance_time_step(
    *,
    sim_params: SimulationParameters,
    rank_coords: RankCoords,
    p_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    p_new: np.ndarray,
    u_new: np.ndarray,
    v_new: np.ndarray,
    w_new: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    k_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    match sim_params.time_integration_order:
        case 1:
            return advance_time_step_first_order(
                sim_params=sim_params,
                rank_coords=rank_coords,
                p_old=p_old,
                u_old=u_old,
                v_old=v_old,
                w_old=w_old,
                p_new=p_new,
                u_new=u_new,
                v_new=v_new,
                w_new=w_new,
                k_u=k_u,
                k_v=k_v,
                k_w=k_w,
                k_p=k_p,
            )
        case 2:
            return advance_time_step_second_order(
                sim_params=sim_params,
                rank_coords=rank_coords,
                p_old=p_old,
                u_old=u_old,
                v_old=v_old,
                w_old=w_old,
                p_new=p_new,
                u_new=u_new,
                v_new=v_new,
                w_new=w_new,
                k_u=k_u,
                k_v=k_v,
                k_w=k_w,
                k_p=k_p,
            )
        case 3:
            return advance_time_step_third_order(
                sim_params=sim_params,
                rank_coords=rank_coords,
                p_old=p_old,
                u_old=u_old,
                v_old=v_old,
                w_old=w_old,
                p_new=p_new,
                u_new=u_new,
                v_new=v_new,
                w_new=w_new,
                k_u=k_u,
                k_v=k_v,
                k_w=k_w,
                k_p=k_p,
            )
        case 4:
            raise NotImplementedError("Fourth-order (RK4) time integration is not implemented.")
        case _:
            raise ValueError(f"Unsupported time integration order: {sim_params.time_integration_order}.")

