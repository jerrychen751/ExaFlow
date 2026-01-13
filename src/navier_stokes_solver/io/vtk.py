from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..parameters import SimulationParameters


def write_total_array_to_vtk(
    sim_params: SimulationParameters,
    name: str,
    *,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    out_dir: str = "out",
) -> None:
    """
    Write a full-domain dataset to VTK via pyevtk.

    This is intentionally a thin wrapper so that solver usage does not require
    pyevtk unless this function is called.
    """

    try:
        from pyevtk.hl import gridToVTK  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("write_total_array_to_vtk requires the 'pyevtk' extra.") from exc

    os.makedirs(out_dir, exist_ok=True)

    if sim_params.dimension == 2:
        length, width = sim_params.size
        nx, ny = sim_params.domain
        x = np.linspace(0.0, float(length), int(nx))
        y = np.linspace(0.0, float(width), int(ny))
        z = np.array([0.0], dtype=float)
        gridToVTK(
            os.path.join(out_dir, f"{name}_Total"),
            x,
            y,
            z,
            pointData={"pressure": p, "velocity": (u, v)},
        )
        return

    if sim_params.dimension == 3:
        if v is None or w is None:
            raise ValueError("3D VTK output requires v and w.")
        length, width, height = sim_params.size
        nx, ny, nz = sim_params.domain
        x = np.linspace(0.0, float(length), int(nx))
        y = np.linspace(0.0, float(width), int(ny))
        z = np.linspace(0.0, float(height), int(nz))
        gridToVTK(
            os.path.join(out_dir, f"{name}_Total"),
            x,
            y,
            z,
            pointData={"pressure": p, "velocity": (u, v, w)},
        )
        return

    raise ValueError(f"VTK output is only implemented for 2D and 3D, got dimension={sim_params.dimension}.")

