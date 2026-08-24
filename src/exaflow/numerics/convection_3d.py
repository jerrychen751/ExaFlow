from __future__ import annotations

import numpy as np

from ..parameters import SimulationParameters


def convection_explicit_3d(
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    sim_params: SimulationParameters,
    order: int = 1,
    direction: str = "backward",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicit convection operator.
    """

    dx = sim_params.dx
    dy = sim_params.dy
    dz = sim_params.dz
    dt = sim_params.dt
    if dx is None or dy is None or dz is None or dt is None:
        raise ValueError("dx/dy/dz/dt must be computed before calling convection_explicit_3d.")

    match order:
        case 1:
            match direction:
                case "central":
                    k_u[2:-2, 2:-2, 2:-2] = -dt * (
                        u_old[2:-2, 2:-2, 2:-2] * (u_old[3:-1, 2:-2, 2:-2] - u_old[1:-3, 2:-2, 2:-2]) / (2 * dx)
                        + v_old[2:-2, 2:-2, 2:-2] * (u_old[2:-2, 3:-1, 2:-2] - u_old[2:-2, 1:-3, 2:-2]) / (2 * dy)
                        + w_old[2:-2, 2:-2, 2:-2] * (u_old[2:-2, 2:-2, 3:-1] - u_old[2:-2, 2:-2, 1:-3]) / (2 * dz)
                    )

                    k_v[2:-2, 2:-2, 2:-2] = -dt * (
                        u_old[2:-2, 2:-2, 2:-2] * (v_old[3:-1, 2:-2, 2:-2] - v_old[1:-3, 2:-2, 2:-2]) / (2 * dx)
                        + v_old[2:-2, 2:-2, 2:-2] * (v_old[2:-2, 3:-1, 2:-2] - v_old[2:-2, 1:-3, 2:-2]) / (2 * dy)
                        + w_old[2:-2, 2:-2, 2:-2] * (v_old[2:-2, 2:-2, 3:-1] - v_old[2:-2, 2:-2, 1:-3]) / (2 * dz)
                    )

                    k_w[2:-2, 2:-2, 2:-2] = -dt * (
                        u_old[2:-2, 2:-2, 2:-2] * (w_old[3:-1, 2:-2, 2:-2] - w_old[1:-3, 2:-2, 2:-2]) / (2 * dx)
                        + v_old[2:-2, 2:-2, 2:-2] * (w_old[2:-2, 3:-1, 2:-2] - w_old[2:-2, 1:-3, 2:-2]) / (2 * dy)
                        + w_old[2:-2, 2:-2, 2:-2] * (w_old[2:-2, 2:-2, 3:-1] - w_old[2:-2, 2:-2, 1:-3]) / (2 * dz)
                    )
                case "backward":
                    k_u[2:-1, 2:-1, 2:-1] = -dt * (
                        u_old[2:-1, 2:-1, 2:-1] * (u_old[2:-1, 2:-1, 2:-1] - u_old[1:-2, 2:-1, 2:-1]) / dx
                        + v_old[2:-1, 2:-1, 2:-1] * (u_old[2:-1, 2:-1, 2:-1] - u_old[2:-1, 1:-2, 2:-1]) / dy
                        + w_old[2:-1, 2:-1, 2:-1] * (u_old[2:-1, 2:-1, 2:-1] - u_old[2:-1, 2:-1, 1:-2]) / dz
                    )

                    k_v[2:-1, 2:-1, 2:-1] = -dt * (
                        u_old[2:-1, 2:-1, 2:-1] * (v_old[2:-1, 2:-1, 2:-1] - v_old[1:-2, 2:-1, 2:-1]) / dx
                        + v_old[2:-1, 2:-1, 2:-1] * (v_old[2:-1, 2:-1, 2:-1] - v_old[2:-1, 1:-2, 2:-1]) / dy
                        + w_old[2:-1, 2:-1, 2:-1] * (v_old[2:-1, 2:-1, 2:-1] - v_old[2:-1, 2:-1, 1:-2]) / dz
                    )

                    k_w[2:-1, 2:-1, 2:-1] = -dt * (
                        u_old[2:-1, 2:-1, 2:-1] * (w_old[2:-1, 2:-1, 2:-1] - w_old[1:-2, 2:-1, 2:-1]) / dx
                        + v_old[2:-1, 2:-1, 2:-1] * (w_old[2:-1, 2:-1, 2:-1] - w_old[2:-1, 1:-2, 2:-1]) / dy
                        + w_old[2:-1, 2:-1, 2:-1] * (w_old[2:-1, 2:-1, 2:-1] - w_old[2:-1, 2:-1, 1:-2]) / dz
                    )
                case "forward":
                    k_u[1:-1] = u_old[1:-1] * (u_old[2:] - u_old[1:-1]) / dx
                case _:
                    raise ValueError(f"Unsupported direction: {direction!r}")
        case 2:
            raise NotImplementedError("order=2 convection is not implemented.")
        case _:
            raise ValueError(f"Unsupported order: {order}")

    return k_u, k_v, k_w


def convection_explicit_3d_boundary(
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    sim_params: SimulationParameters,
    order: int = 1,
    direction: str = "backward",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boundary convection operator.
    """

    dx = sim_params.dx
    dy = sim_params.dy
    dz = sim_params.dz
    dt = sim_params.dt
    if dx is None or dy is None or dz is None or dt is None:
        raise ValueError("dx/dy/dz/dt must be computed before calling convection_explicit_3d_boundary.")

    match order:
        case 1:
            match direction:
                case "central":
                    # X-direction faces (i = 1 and i = -2)
                    k_u[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (u_old[2, 1:-1, 1:-1] - u_old[0, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[1, 1:-1, 1:-1] * (u_old[1, 2:, 1:-1] - u_old[1, :-2, 1:-1]) / (2 * dy)
                        + w_old[1, 1:-1, 1:-1] * (u_old[1, 1:-1, 2:] - u_old[1, 1:-1, :-2]) / (2 * dz)
                    )

                    k_u[-2, 1:-1, 1:-1] = -dt * (
                        u_old[-2, 1:-1, 1:-1] * (u_old[-1, 1:-1, 1:-1] - u_old[-3, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[-2, 1:-1, 1:-1] * (u_old[-2, 2:, 1:-1] - u_old[-2, :-2, 1:-1]) / (2 * dy)
                        + w_old[-2, 1:-1, 1:-1] * (u_old[-2, 1:-1, 2:] - u_old[-2, 1:-1, :-2]) / (2 * dz)
                    )

                    k_v[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (v_old[2, 1:-1, 1:-1] - v_old[0, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[1, 1:-1, 1:-1] * (v_old[1, 2:, 1:-1] - v_old[1, :-2, 1:-1]) / (2 * dy)
                        + w_old[1, 1:-1, 1:-1] * (v_old[1, 1:-1, 2:] - v_old[1, 1:-1, :-2]) / (2 * dz)
                    )

                    k_v[-2, 1:-1, 1:-1] = -dt * (
                        u_old[-2, 1:-1, 1:-1] * (v_old[-1, 1:-1, 1:-1] - v_old[-3, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[-2, 1:-1, 1:-1] * (v_old[-2, 2:, 1:-1] - v_old[-2, :-2, 1:-1]) / (2 * dy)
                        + w_old[-2, 1:-1, 1:-1] * (v_old[-2, 1:-1, 2:] - v_old[-2, 1:-1, :-2]) / (2 * dz)
                    )

                    k_w[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (w_old[2, 1:-1, 1:-1] - w_old[0, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[1, 1:-1, 1:-1] * (w_old[1, 2:, 1:-1] - w_old[1, :-2, 1:-1]) / (2 * dy)
                        + w_old[1, 1:-1, 1:-1] * (w_old[1, 1:-1, 2:] - w_old[1, 1:-1, :-2]) / (2 * dz)
                    )

                    k_w[-2, 1:-1, 1:-1] = -dt * (
                        u_old[-2, 1:-1, 1:-1] * (w_old[-1, 1:-1, 1:-1] - w_old[-3, 1:-1, 1:-1]) / (2 * dx)
                        + v_old[-2, 1:-1, 1:-1] * (w_old[-2, 2:, 1:-1] - w_old[-2, :-2, 1:-1]) / (2 * dy)
                        + w_old[-2, 1:-1, 1:-1] * (w_old[-2, 1:-1, 2:] - w_old[-2, 1:-1, :-2]) / (2 * dz)
                    )

                    # Y-direction faces (j = 1 and j = -2)
                    k_u[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (u_old[2:, 1, 1:-1] - u_old[:-2, 1, 1:-1]) / (2 * dx)
                        + v_old[1:-1, 1, 1:-1] * (u_old[1:-1, 2, 1:-1] - u_old[1:-1, 0, 1:-1]) / (2 * dy)
                        + w_old[1:-1, 1, 1:-1] * (u_old[1:-1, 1, 2:] - u_old[1:-1, 1, :-2]) / (2 * dz)
                    )

                    k_u[1:-1, -2, 1:-1] = -dt * (
                        u_old[1:-1, -2, 1:-1] * (u_old[2:, -2, 1:-1] - u_old[:-2, -2, 1:-1]) / (2 * dx)
                        + v_old[1:-1, -2, 1:-1] * (u_old[1:-1, -1, 1:-1] - u_old[1:-1, -3, 1:-1]) / (2 * dy)
                        + w_old[1:-1, -2, 1:-1] * (u_old[1:-1, -2, 2:] - u_old[1:-1, -2, :-2]) / (2 * dz)
                    )

                    k_v[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (v_old[2:, 1, 1:-1] - v_old[:-2, 1, 1:-1]) / (2 * dx)
                        + v_old[1:-1, 1, 1:-1] * (v_old[1:-1, 2, 1:-1] - v_old[1:-1, 0, 1:-1]) / (2 * dy)
                        + w_old[1:-1, 1, 1:-1] * (v_old[1:-1, 1, 2:] - v_old[1:-1, 1, :-2]) / (2 * dz)
                    )

                    k_v[1:-1, -2, 1:-1] = -dt * (
                        u_old[1:-1, -2, 1:-1] * (v_old[2:, -2, 1:-1] - v_old[:-2, -2, 1:-1]) / (2 * dx)
                        + v_old[1:-1, -2, 1:-1] * (v_old[1:-1, -1, 1:-1] - v_old[1:-1, -3, 1:-1]) / (2 * dy)
                        + w_old[1:-1, -2, 1:-1] * (v_old[1:-1, -2, 2:] - v_old[1:-1, -2, :-2]) / (2 * dz)
                    )

                    k_w[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (w_old[2:, 1, 1:-1] - w_old[:-2, 1, 1:-1]) / (2 * dx)
                        + v_old[1:-1, 1, 1:-1] * (w_old[1:-1, 2, 1:-1] - w_old[1:-1, 0, 1:-1]) / (2 * dy)
                        + w_old[1:-1, 1, 1:-1] * (w_old[1:-1, 1, 2:] - w_old[1:-1, 1, :-2]) / (2 * dz)
                    )

                    k_w[1:-1, -2, 1:-1] = -dt * (
                        u_old[1:-1, -2, 1:-1] * (w_old[2:, -2, 1:-1] - w_old[:-2, -2, 1:-1]) / (2 * dx)
                        + v_old[1:-1, -2, 1:-1] * (w_old[1:-1, -1, 1:-1] - w_old[1:-1, -3, 1:-1]) / (2 * dy)
                        + w_old[1:-1, -2, 1:-1] * (w_old[1:-1, -2, 2:] - w_old[1:-1, -2, :-2]) / (2 * dz)
                    )

                    # Z-direction faces (k = 1 and k = -2)
                    k_u[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (u_old[2:, 1:-1, 1] - u_old[:-2, 1:-1, 1]) / (2 * dx)
                        + v_old[1:-1, 1:-1, 1] * (u_old[1:-1, 2:, 1] - u_old[1:-1, :-2, 1]) / (2 * dy)
                        + w_old[1:-1, 1:-1, 1] * (u_old[1:-1, 1:-1, 2] - u_old[1:-1, 1:-1, 0]) / (2 * dz)
                    )

                    k_u[1:-1, 1:-1, -2] = -dt * (
                        u_old[1:-1, 1:-1, -2] * (u_old[2:, 1:-1, -2] - u_old[:-2, 1:-1, -2]) / (2 * dx)
                        + v_old[1:-1, 1:-1, -2] * (u_old[1:-1, 2:, -2] - u_old[1:-1, :-2, -2]) / (2 * dy)
                        + w_old[1:-1, 1:-1, -2] * (u_old[1:-1, 1:-1, -1] - u_old[1:-1, 1:-1, -3]) / (2 * dz)
                    )

                    k_v[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (v_old[2:, 1:-1, 1] - v_old[:-2, 1:-1, 1]) / (2 * dx)
                        + v_old[1:-1, 1:-1, 1] * (v_old[1:-1, 2:, 1] - v_old[1:-1, :-2, 1]) / (2 * dy)
                        + w_old[1:-1, 1:-1, 1] * (v_old[1:-1, 1:-1, 2] - v_old[1:-1, 1:-1, 0]) / (2 * dz)
                    )

                    k_v[1:-1, 1:-1, -2] = -dt * (
                        u_old[1:-1, 1:-1, -2] * (v_old[2:, 1:-1, -2] - v_old[:-2, 1:-1, -2]) / (2 * dx)
                        + v_old[1:-1, 1:-1, -2] * (v_old[1:-1, 2:, -2] - v_old[1:-1, :-2, -2]) / (2 * dy)
                        + w_old[1:-1, 1:-1, -2] * (v_old[1:-1, 1:-1, -1] - v_old[1:-1, 1:-1, -3]) / (2 * dz)
                    )

                    k_w[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (w_old[2:, 1:-1, 1] - w_old[:-2, 1:-1, 1]) / (2 * dx)
                        + v_old[1:-1, 1:-1, 1] * (w_old[1:-1, 2:, 1] - w_old[1:-1, :-2, 1]) / (2 * dy)
                        + w_old[1:-1, 1:-1, 1] * (w_old[1:-1, 1:-1, 2] - w_old[1:-1, 1:-1, 0]) / (2 * dz)
                    )

                    k_w[1:-1, 1:-1, -2] = -dt * (
                        u_old[1:-1, 1:-1, -2] * (w_old[2:, 1:-1, -2] - w_old[:-2, 1:-1, -2]) / (2 * dx)
                        + v_old[1:-1, 1:-1, -2] * (w_old[1:-1, 2:, -2] - w_old[1:-1, :-2, -2]) / (2 * dy)
                        + w_old[1:-1, 1:-1, -2] * (w_old[1:-1, 1:-1, -1] - w_old[1:-1, 1:-1, -3]) / (2 * dz)
                    )
                case "foward":
                    k_u[1:-1] = (
                        u_old[1:-1] * (u_old[2:] - u_old[1:-1]) / dx
                        + v_old[1:-1] * (u_old[2:] - u_old[1:-1]) / dy
                        + w_old[1:-1] * (u_old[2:] - u_old[1:-1]) / dz
                    )
                    k_v[1:-1] = (
                        u_old[1:-1] * (v_old[2:] - v_old[1:-1]) / dx
                        + v_old[1:-1] * (v_old[2:] - v_old[1:-1]) / dy
                        + w_old[1:-1] * (v_old[2:] - v_old[1:-1]) / dz
                    )
                    k_w[1:-1] = (
                        u_old[1:-1] * (w_old[2:] - w_old[1:-1]) / dx
                        + v_old[1:-1] * (w_old[2:] - w_old[1:-1]) / dy
                        + w_old[1:-1] * (w_old[2:] - w_old[1:-1]) / dz
                    )
                case "backward":
                    # I = 1
                    k_u[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (u_old[1, 1:-1, 1:-1] - u_old[0, 1:-1, 1:-1]) / dx
                        + v_old[1, 1:-1, 1:-1] * (u_old[1, 1:-1, 1:-1] - u_old[1, 0:-2, 1:-1]) / dy
                        + w_old[1, 1:-1, 1:-1] * (u_old[1, 1:-1, 1:-1] - u_old[1, 1:-1, 0:-2]) / dz
                    )

                    k_v[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (v_old[1, 1:-1, 1:-1] - v_old[0, 1:-1, 1:-1]) / dx
                        + v_old[1, 1:-1, 1:-1] * (v_old[1, 1:-1, 1:-1] - v_old[1, 0:-2, 1:-1]) / dy
                        + w_old[1, 1:-1, 1:-1] * (v_old[1, 1:-1, 1:-1] - v_old[1, 1:-1, 0:-2]) / dz
                    )

                    k_w[1, 1:-1, 1:-1] = -dt * (
                        u_old[1, 1:-1, 1:-1] * (w_old[1, 1:-1, 1:-1] - w_old[0, 1:-1, 1:-1]) / dx
                        + v_old[1, 1:-1, 1:-1] * (w_old[1, 1:-1, 1:-1] - w_old[1, 0:-2, 1:-1]) / dy
                        + w_old[1, 1:-1, 1:-1] * (w_old[1, 1:-1, 1:-1] - w_old[1, 1:-1, 0:-2]) / dz
                    )

                    # J = 1
                    k_u[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (u_old[1:-1, 1, 1:-1] - u_old[0:-2, 1, 1:-1]) / dx
                        + v_old[1:-1, 1, 1:-1] * (u_old[1:-1, 1, 1:-1] - u_old[1:-1, 0, 1:-1]) / dy
                        + w_old[1:-1, 1, 1:-1] * (u_old[1:-1, 1, 1:-1] - u_old[1:-1, 1, 0:-2]) / dz
                    )

                    k_v[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (v_old[1:-1, 1, 1:-1] - v_old[0:-2, 1, 1:-1]) / dx
                        + v_old[1:-1, 1, 1:-1] * (v_old[1:-1, 1, 1:-1] - v_old[1:-1, 0, 1:-1]) / dy
                        + w_old[1:-1, 1, 1:-1] * (v_old[1:-1, 1, 1:-1] - v_old[1:-1, 1, 0:-2]) / dz
                    )

                    k_w[1:-1, 1, 1:-1] = -dt * (
                        u_old[1:-1, 1, 1:-1] * (w_old[1:-1, 1, 1:-1] - w_old[0:-2, 1, 1:-1]) / dx
                        + v_old[1:-1, 1, 1:-1] * (w_old[1:-1, 1, 1:-1] - w_old[1:-1, 0, 1:-1]) / dy
                        + w_old[1:-1, 1, 1:-1] * (w_old[1:-1, 1, 1:-1] - w_old[1:-1, 1, 0:-2]) / dz
                    )

                    # K = 1
                    k_u[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (u_old[1:-1, 1:-1, 1] - u_old[0:-2, 1:-1, 1]) / dx
                        + v_old[1:-1, 1:-1, 1] * (u_old[1:-1, 1:-1, 1] - u_old[1:-1, 0:-2, 1]) / dy
                        + w_old[1:-1, 1:-1, 1] * (u_old[1:-1, 1:-1, 1] - u_old[1:-1, 1:-1, 0]) / dz
                    )

                    k_v[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (v_old[1:-1, 1:-1, 1] - v_old[0:-2, 1:-1, 1]) / dx
                        + v_old[1:-1, 1:-1, 1] * (v_old[1:-1, 1:-1, 1] - v_old[1:-1, 0:-2, 1]) / dy
                        + w_old[1:-1, 1:-1, 1] * (v_old[1:-1, 1:-1, 1] - v_old[1:-1, 1:-1, 0]) / dz
                    )

                    k_w[1:-1, 1:-1, 1] = -dt * (
                        u_old[1:-1, 1:-1, 1] * (w_old[1:-1, 1:-1, 1] - w_old[0:-2, 1:-1, 1]) / dx
                        + v_old[1:-1, 1:-1, 1] * (w_old[1:-1, 1:-1, 1] - w_old[1:-1, 0:-2, 1]) / dy
                        + w_old[1:-1, 1:-1, 1] * (w_old[1:-1, 1:-1, 1] - w_old[1:-1, 1:-1, 0]) / dz
                    )
                case _:
                    raise ValueError(f"Unsupported direction: {direction!r}")
        case 2:
            raise NotImplementedError("order=2 convection boundary is not implemented.")
        case _:
            raise ValueError(f"Unsupported order: {order}")
    return k_u, k_v, k_w


def convection_implicit_3d(u_old: np.ndarray, v_old: np.ndarray, w_old: np.ndarray, order: int = 1, direction: str = "central") -> None:
    raise NotImplementedError

