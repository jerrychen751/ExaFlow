from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

from ..config.grid import Grid
from ..fields import FlowState
from ..mpi.gather import gather_global_array
from ..mpi.subdomain import Subdomain
from .csv import format_field_csv, write_text_atomically


def gather_domain_fields(
    subdomain: Subdomain,
    comm: Any | None,
    state: FlowState,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """
    Assemble every field on rank 0 as (velocity components in axis order, pressure), with the ghost layers stripped, or None on every other rank.

    This is a collective call: every rank must reach it, because the gather reduces across all of them.
    """

    interior = subdomain.interior
    components = [
        gather_global_array(subdomain, comm, state.velocity[axis][interior])
        for axis in range(state.dimension)
    ]
    pressure = gather_global_array(subdomain, comm, state.pressure[interior])
    if pressure is None or any(part is None for part in components):
        return None
    return [part for part in components if part is not None], pressure


class Writer(Protocol):
    """
    One output format. `frequency` is the interval in steps, or -1 to write only when the solver asks directly. `write` is collective: every rank must call it, because a writer that assembles the full domain reduces across ranks.
    """

    frequency: int

    def write(self, label: str, state: FlowState) -> None: ...


class RankCsvWriter:
    """
    One CSV file per rank per label, named `<label>_<rank>.csv`, holding only the cells that rank owns. Writes no message between ranks, so it stays cheap at every interval.
    """

    def __init__(self, directory: str, subdomain: Subdomain, *, frequency: int = -1) -> None:
        self.frequency = frequency
        self._directory = directory
        self._subdomain = subdomain

    def write(self, label: str, state: FlowState) -> None:
        interior = self._subdomain.interior
        velocity = np.stack([state.velocity[axis][interior] for axis in range(state.dimension)])
        origin = tuple(start for start, _ in self._subdomain.bounds)
        text = format_field_csv(velocity, state.pressure[interior], origin)
        write_text_atomically(os.path.join(self._directory, f"{label}_{self._subdomain.rank}.csv"), text)


class TotalCsvWriter:
    """
    One CSV file per label holding the whole domain, named `<label>_Total.csv`. Rank 0 assembles the blocks and writes; the other ranks take part in the gather and write nothing.
    """

    def __init__(self, directory: str, subdomain: Subdomain, comm: Any | None, *, frequency: int = -1) -> None:
        self.frequency = frequency
        self._directory = directory
        self._subdomain = subdomain
        self._comm = comm

    def write(self, label: str, state: FlowState) -> None:
        assembled = gather_domain_fields(self._subdomain, self._comm, state)
        if assembled is None:
            return
        components, pressure = assembled
        text = format_field_csv(np.stack(components), pressure, (0,) * self._subdomain.grid.dimension)
        write_text_atomically(os.path.join(self._directory, f"{label}_Total.csv"), text)


class VtkWriter:
    """
    One VTK rectilinear grid per label holding the whole domain, named `<label>_Total.vtr`. Rank 0 assembles the blocks and writes; the other ranks take part in the gather and write nothing. Needs pyevtk, which is imported only when a write happens.
    """

    def __init__(
        self,
        directory: str,
        grid: Grid,
        subdomain: Subdomain,
        comm: Any | None,
        *,
        frequency: int = -1,
    ) -> None:
        self.frequency = frequency
        self._directory = directory
        self._grid = grid
        self._subdomain = subdomain
        self._comm = comm

    def write(self, label: str, state: FlowState) -> None:
        try:
            from pyevtk.hl import gridToVTK  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError("VTK output needs the 'pyevtk' package.") from error

        assembled = gather_domain_fields(self._subdomain, self._comm, state)
        if assembled is None:
            return
        components, pressure = assembled

        axes = [
            np.linspace(0.0, float(span), int(count))
            for span, count in zip(self._grid.extent, self._grid.shape)
        ]
        while len(axes) < 3:
            axes.append(np.array([0.0], dtype=float))
        while len(components) < 3:
            components.append(np.zeros_like(pressure))

        os.makedirs(self._directory, exist_ok=True)
        gridToVTK(
            os.path.join(self._directory, f"{label}_Total"),
            axes[0],
            axes[1],
            axes[2],
            pointData={"pressure": pressure, "velocity": tuple(components)},
        )


def build_writers(
    directory: str,
    grid: Grid,
    subdomain: Subdomain,
    comm: Any | None,
    total_csv_frequency: int = -1,
    partial_csv_frequency: int = -1,
    vtk_frequency: int = -1,
) -> tuple[Writer, ...]:
    """
    The standard writers for a run, each carrying the interval its format was given. A format switched off with -1 still gets a writer, because the solver writes the first and last state through every writer whatever its interval.
    """

    return (
        TotalCsvWriter(directory, subdomain, comm, frequency=total_csv_frequency),
        RankCsvWriter(directory, subdomain, frequency=partial_csv_frequency),
        VtkWriter(directory, grid, subdomain, comm, frequency=vtk_frequency),
    )
