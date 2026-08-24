from __future__ import annotations

import numpy as np

from ..parameters import SimulationParameters


def diffusion_explicit_3d(
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    sim_params: SimulationParameters,
    order: int = 1,
    direction: str = "central",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = sim_params.dx
    dy = sim_params.dy
    dz = sim_params.dz
    dt = sim_params.dt
    nu = sim_params.nu
    if dx is None or dy is None or dz is None or dt is None:
        raise ValueError("dx/dy/dz/dt must be computed before calling diffusion_explicit_3d.")

    match order:
        case 1:
            match direction:
                case "central":
                    k_u[2:-2, 2:-2, 2:-2] += dt * nu * (
                        ((u_old[3:-1, 2:-2, 2:-2] - 2 * u_old[2:-2, 2:-2, 2:-2] + u_old[1:-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((u_old[2:-2, 3:-1, 2:-2] - 2 * u_old[2:-2, 2:-2, 2:-2] + u_old[2:-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((u_old[2:-2, 2:-2, 3:-1] - 2 * u_old[2:-2, 2:-2, 2:-2] + u_old[2:-2, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_v[2:-2, 2:-2, 2:-2] += dt * nu * (
                        ((v_old[3:-1, 2:-2, 2:-2] - 2 * v_old[2:-2, 2:-2, 2:-2] + v_old[1:-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((v_old[2:-2, 3:-1, 2:-2] - 2 * v_old[2:-2, 2:-2, 2:-2] + v_old[2:-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((v_old[2:-2, 2:-2, 3:-1] - 2 * v_old[2:-2, 2:-2, 2:-2] + v_old[2:-2, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_w[2:-2, 2:-2, 2:-2] += dt * nu * (
                        ((w_old[3:-1, 2:-2, 2:-2] - 2 * w_old[2:-2, 2:-2, 2:-2] + w_old[1:-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((w_old[2:-2, 3:-1, 2:-2] - 2 * w_old[2:-2, 2:-2, 2:-2] + w_old[2:-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((w_old[2:-2, 2:-2, 3:-1] - 2 * w_old[2:-2, 2:-2, 2:-2] + w_old[2:-2, 2:-2, 1:-3]) / (dz * dz))
                    )
                case "forward" | "backward":
                    raise NotImplementedError(f"Diffusion direction {direction!r} is not implemented.")
                case _:
                    raise ValueError(f"Unsupported direction: {direction!r}")
        case _:
            raise NotImplementedError(f"Diffusion order {order} is not implemented.")
    return k_u, k_v, k_w


def diffusion_explicit_3d_boundary(
    u_old: np.ndarray,
    v_old: np.ndarray,
    w_old: np.ndarray,
    k_u: np.ndarray,
    k_v: np.ndarray,
    k_w: np.ndarray,
    sim_params: SimulationParameters,
    order: int = 1,
    direction: str = "central",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = sim_params.dx
    dy = sim_params.dy
    dz = sim_params.dz
    dt = sim_params.dt
    nu = sim_params.nu
    if dx is None or dy is None or dz is None or dt is None:
        raise ValueError("dx/dy/dz/dt must be computed before calling diffusion_explicit_3d_boundary.")

    match order:
        case 1:
            match direction:
                case "central":
                    # Left face
                    k_u[1, 2:-2, 2:-2] += dt * nu * (
                        ((u_old[2, 2:-2, 2:-2] - 2 * u_old[1, 2:-2, 2:-2] + u_old[0, 2:-2, 2:-2]) / (dx * dx))
                        + ((u_old[1, 3:-1, 2:-2] - 2 * u_old[1, 2:-2, 2:-2] + u_old[1, 1:-3, 2:-2]) / (dy * dy))
                        + ((u_old[1, 2:-2, 3:-1] - 2 * u_old[1, 2:-2, 2:-2] + u_old[1, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_v[1, 2:-2, 2:-2] += dt * nu * (
                        ((v_old[2, 2:-2, 2:-2] - 2 * v_old[1, 2:-2, 2:-2] + v_old[0, 2:-2, 2:-2]) / (dx * dx))
                        + ((v_old[1, 3:-1, 2:-2] - 2 * v_old[1, 2:-2, 2:-2] + v_old[1, 1:-3, 2:-2]) / (dy * dy))
                        + ((v_old[1, 2:-2, 3:-1] - 2 * v_old[1, 2:-2, 2:-2] + v_old[1, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_w[1, 2:-2, 2:-2] += dt * nu * (
                        ((w_old[2, 2:-2, 2:-2] - 2 * w_old[1, 2:-2, 2:-2] + w_old[0, 2:-2, 2:-2]) / (dx * dx))
                        + ((w_old[1, 3:-1, 2:-2] - 2 * w_old[1, 2:-2, 2:-2] + w_old[1, 1:-3, 2:-2]) / (dy * dy))
                        + ((w_old[1, 2:-2, 3:-1] - 2 * w_old[1, 2:-2, 2:-2] + w_old[1, 2:-2, 1:-3]) / (dz * dz))
                    )

                    # Right face
                    k_u[-2, 2:-2, 2:-2] += dt * nu * (
                        ((u_old[-1, 2:-2, 2:-2] - 2 * u_old[-2, 2:-2, 2:-2] + u_old[-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((u_old[-2, 3:-1, 2:-2] - 2 * u_old[-2, 2:-2, 2:-2] + u_old[-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((u_old[-2, 2:-2, 3:-1] - 2 * u_old[-2, 2:-2, 2:-2] + u_old[-2, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_v[-2, 2:-2, 2:-2] += dt * nu * (
                        ((v_old[-1, 2:-2, 2:-2] - 2 * v_old[-2, 2:-2, 2:-2] + v_old[-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((v_old[-2, 3:-1, 2:-2] - 2 * v_old[-2, 2:-2, 2:-2] + v_old[-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((v_old[-2, 2:-2, 3:-1] - 2 * v_old[-2, 2:-2, 2:-2] + v_old[-2, 2:-2, 1:-3]) / (dz * dz))
                    )
                    k_w[-2, 2:-2, 2:-2] += dt * nu * (
                        ((w_old[-1, 2:-2, 2:-2] - 2 * w_old[-2, 2:-2, 2:-2] + w_old[-3, 2:-2, 2:-2]) / (dx * dx))
                        + ((w_old[-2, 3:-1, 2:-2] - 2 * w_old[-2, 2:-2, 2:-2] + w_old[-2, 1:-3, 2:-2]) / (dy * dy))
                        + ((w_old[-2, 2:-2, 3:-1] - 2 * w_old[-2, 2:-2, 2:-2] + w_old[-2, 2:-2, 1:-3]) / (dz * dz))
                    )

                    # Top face
                    k_u[2:-2, 1, 2:-2] += dt * nu * (
                        ((u_old[3:-1, 1, 2:-2] - 2 * u_old[2:-2, 1, 2:-2] + u_old[1:-3, 1, 2:-2]) / (dx * dx))
                        + ((u_old[2:-2, 2, 2:-2] - 2 * u_old[2:-2, 1, 2:-2] + u_old[2:-2, 0, 2:-2]) / (dy * dy))
                        + ((u_old[2:-2, 1, 3:-1] - 2 * u_old[2:-2, 1, 2:-2] + u_old[2:-2, 1, 1:-3]) / (dz * dz))
                    )
                    k_v[2:-2, 1, 2:-2] += dt * nu * (
                        ((v_old[3:-1, 1, 2:-2] - 2 * v_old[2:-2, 1, 2:-2] + v_old[1:-3, 1, 2:-2]) / (dx * dx))
                        + ((v_old[2:-2, 2, 2:-2] - 2 * v_old[2:-2, 1, 2:-2] + v_old[2:-2, 0, 2:-2]) / (dy * dy))
                        + ((v_old[2:-2, 1, 3:-1] - 2 * v_old[2:-2, 1, 2:-2] + v_old[2:-2, 1, 1:-3]) / (dz * dz))
                    )
                    k_w[2:-2, 1, 2:-2] += dt * nu * (
                        ((w_old[3:-1, 1, 2:-2] - 2 * w_old[2:-2, 1, 2:-2] + w_old[1:-3, 1, 2:-2]) / (dx * dx))
                        + ((w_old[2:-2, 2, 2:-2] - 2 * w_old[2:-2, 1, 2:-2] + w_old[2:-2, 0, 2:-2]) / (dy * dy))
                        + ((w_old[2:-2, 1, 3:-1] - 2 * w_old[2:-2, 1, 2:-2] + w_old[2:-2, 1, 1:-3]) / (dz * dz))
                    )

                    # Bottom face
                    k_u[2:-2, -2, 2:-2] += dt * nu * (
                        ((u_old[3:-1, -2, 2:-2] - 2 * u_old[2:-2, -2, 2:-2] + u_old[1:-3, -2, 2:-2]) / (dx * dx))
                        + ((u_old[2:-2, -1, 2:-2] - 2 * u_old[2:-2, -2, 2:-2] + u_old[2:-2, -3, 2:-2]) / (dy * dy))
                        + ((u_old[2:-2, -2, 3:-1] - 2 * u_old[2:-2, -2, 2:-2] + u_old[2:-2, -2, 1:-3]) / (dz * dz))
                    )
                    k_v[2:-2, -2, 2:-2] += dt * nu * (
                        ((v_old[3:-1, -2, 2:-2] - 2 * v_old[2:-2, -2, 2:-2] + v_old[1:-3, -2, 2:-2]) / (dx * dx))
                        + ((v_old[2:-2, -1, 2:-2] - 2 * v_old[2:-2, -2, 2:-2] + v_old[2:-2, -3, 2:-2]) / (dy * dy))
                        + ((v_old[2:-2, -2, 3:-1] - 2 * v_old[2:-2, -2, 2:-2] + v_old[2:-2, -2, 1:-3]) / (dz * dz))
                    )
                    k_w[2:-2, -2, 2:-2] += dt * nu * (
                        ((w_old[3:-1, -2, 2:-2] - 2 * w_old[2:-2, -2, 2:-2] + w_old[1:-3, -2, 2:-2]) / (dx * dx))
                        + ((w_old[2:-2, -1, 2:-2] - 2 * w_old[2:-2, -2, 2:-2] + w_old[2:-2, -3, 2:-2]) / (dy * dy))
                        + ((w_old[2:-2, -2, 3:-1] - 2 * w_old[2:-2, -2, 2:-2] + w_old[2:-2, -2, 1:-3]) / (dz * dz))
                    )

                    # Front face
                    k_u[2:-2, 2:-2, 1] += dt * nu * (
                        ((u_old[3:-1, 2:-2, 1] - 2 * u_old[2:-2, 2:-2, 1] + u_old[1:-3, 2:-2, 1]) / (dx * dx))
                        + ((u_old[2:-2, 3:-1, 1] - 2 * u_old[2:-2, 2:-2, 1] + u_old[2:-2, 1:-3, 1]) / (dy * dy))
                        + ((u_old[2:-2, 2:-2, 2] - 2 * u_old[2:-2, 2:-2, 1] + u_old[2:-2, 2:-2, 0]) / (dz * dz))
                    )
                    k_v[2:-2, 2:-2, 1] += dt * nu * (
                        ((v_old[3:-1, 2:-2, 1] - 2 * v_old[2:-2, 2:-2, 1] + v_old[1:-3, 2:-2, 1]) / (dx * dx))
                        + ((v_old[2:-2, 3:-1, 1] - 2 * v_old[2:-2, 2:-2, 1] + v_old[2:-2, 1:-3, 1]) / (dy * dy))
                        + ((v_old[2:-2, 2:-2, 2] - 2 * v_old[2:-2, 2:-2, 1] + v_old[2:-2, 2:-2, 0]) / (dz * dz))
                    )
                    k_w[2:-2, 2:-2, 1] += dt * nu * (
                        ((w_old[3:-1, 2:-2, 1] - 2 * w_old[2:-2, 2:-2, 1] + w_old[1:-3, 2:-2, 1]) / (dx * dx))
                        + ((w_old[2:-2, 3:-1, 1] - 2 * w_old[2:-2, 2:-2, 1] + w_old[2:-2, 1:-3, 1]) / (dy * dy))
                        + ((w_old[2:-2, 2:-2, 2] - 2 * w_old[2:-2, 2:-2, 1] + w_old[2:-2, 2:-2, 0]) / (dz * dz))
                    )

                    # Back face
                    k_u[2:-2, 2:-2, -2] += dt * nu * (
                        ((u_old[3:-1, 2:-2, -2] - 2 * u_old[2:-2, 2:-2, -2] + u_old[1:-3, 2:-2, -2]) / (dx * dx))
                        + ((u_old[2:-2, 3:-1, -2] - 2 * u_old[2:-2, 2:-2, -2] + u_old[2:-2, 1:-3, -2]) / (dy * dy))
                        + ((u_old[2:-2, 2:-2, -1] - 2 * u_old[2:-2, 2:-2, -2] + u_old[2:-2, 2:-2, -3]) / (dz * dz))
                    )
                    k_v[2:-2, 2:-2, -2] += dt * nu * (
                        ((v_old[3:-1, 2:-2, -2] - 2 * v_old[2:-2, 2:-2, -2] + v_old[1:-3, 2:-2, -2]) / (dx * dx))
                        + ((v_old[2:-2, 3:-1, -2] - 2 * v_old[2:-2, 2:-2, -2] + v_old[2:-2, 1:-3, -2]) / (dy * dy))
                        + ((v_old[2:-2, 2:-2, -1] - 2 * v_old[2:-2, 2:-2, -2] + v_old[2:-2, 2:-2, -3]) / (dz * dz))
                    )
                    k_w[2:-2, 2:-2, -2] += dt * nu * (
                        ((w_old[3:-1, 2:-2, -2] - 2 * w_old[2:-2, 2:-2, -2] + w_old[1:-3, 2:-2, -2]) / (dx * dx))
                        + ((w_old[2:-2, 3:-1, -2] - 2 * w_old[2:-2, 2:-2, -2] + w_old[2:-2, 1:-3, -2]) / (dy * dy))
                        + ((w_old[2:-2, 2:-2, -1] - 2 * w_old[2:-2, 2:-2, -2] + w_old[2:-2, 2:-2, -3]) / (dz * dz))
                    )
                case "forward" | "backward":
                    raise NotImplementedError(f"Diffusion direction {direction!r} is not implemented.")
                case _:
                    raise ValueError(f"Unsupported direction: {direction!r}")
        case _:
            raise NotImplementedError(f"Diffusion order {order} is not implemented.")
    return k_u, k_v, k_w


def diffusion_implicit_3d(u: np.ndarray, v: np.ndarray, w: np.ndarray, order: int = 1, direction: str = "central") -> None:
    raise NotImplementedError

