from __future__ import annotations

import math
from typing import Any, Sequence

from .boundary_application import initialize_boundaries
from .config.case import Case
from .fields import FlowState, build_initial_state
from .io.writers import Writer, build_writers
from .mpi.process_grid import ProcessGrid, choose_process_grid
from .mpi.subdomain import Subdomain
from .numerics.operators import SpatialOperator
from .numerics.time_step import TimeIntegrator


def compute_time_step(case: Case, max_velocity: float) -> float:
    """
    The largest stable explicit step for this case, in seconds: the smaller of the advective limit cfl * min(h) / max|u| and the viscous limit 0.5 / (nu * sum(1 / h^2)).

    A motionless field has no advective limit and an inviscid fluid has no viscous limit, so a case with neither raises rather than return an infinite step.
    """

    if not math.isfinite(max_velocity) or max_velocity < 0.0:
        raise ValueError(f"max_velocity must be finite and >= 0, got {max_velocity}.")

    spacing = case.grid.spacing
    advective = math.inf if max_velocity == 0.0 else case.time.cfl * min(spacing) / max_velocity
    if case.fluid.nu == 0.0:
        viscous = math.inf
    else:
        viscous = 0.5 / (case.fluid.nu * sum(1.0 / (step * step) for step in spacing))

    step = min(advective, viscous)
    if not math.isfinite(step):
        raise ValueError(
            "Cannot choose a time step: the fluid is inviscid and the initial velocity is zero, "
            "so neither the advective nor the viscous limit applies."
        )
    return step


class Solver:
    """
    Runs one case to completion on this rank.

    The solver owns everything that belongs to a run rather than to the problem: the communicator, the decomposition, the time step, the stage buffers and the output schedule. The case itself stays frozen throughout.

    Every method is collective. Build the solver on every rank with the same case, and call `run` on every rank.
    """

    def __init__(
        self,
        case: Case,
        comm: Any | None = None,
        *,
        output_directory: str | None = None,
        writers: Sequence[Writer] | None = None,
    ) -> None:
        self.case = case
        self.comm = comm
        self.rank = int(comm.Get_rank()) if comm is not None else 0
        num_procs = int(comm.Get_size()) if comm is not None else 1

        self.process_grid = ProcessGrid(choose_process_grid(num_procs, case.grid.shape, case.grid.num_ghost_layers))
        self.subdomain = Subdomain(case.grid, self.process_grid, self.rank)

        if writers is not None:
            self.writers: tuple[Writer, ...] = tuple(writers)
        elif output_directory is not None:
            self.writers = build_writers(
                output_directory,
                case.grid,
                self.subdomain,
                comm,
                total_csv_frequency=case.outputs.total_csv_frequency,
                partial_csv_frequency=case.outputs.partial_csv_frequency,
                vtk_frequency=case.outputs.vtk_frequency,
            )
        else:
            self.writers = ()

    def build_initial_state(self) -> FlowState:
        """
        The starting fields for this rank, with the prescribed boundary values already written into the ghost layers of every domain face this rank owns.
        """

        state = build_initial_state(self.case, self.subdomain)
        initialize_boundaries(state, self.case, self.subdomain)
        return state

    def choose_time_step(self, state: FlowState) -> float:
        """
        The time step for the whole run, reduced over every rank so that all ranks march together.
        """

        local_max = state.compute_max_speed()
        if self.comm is not None:
            local_max = float(self.comm.allreduce(local_max, op=_max_op()))
        return compute_time_step(self.case, local_max)

    def run(self, *, write_initial: bool = True) -> FlowState:
        """
        March the case for `case.time.num_steps` steps and return the final state for this rank. Writers fire on the steps their interval selects, and every writer is called once more with the label "Final" at the end. `write_initial` also writes the starting state under the label "Original".
        """

        state = self.build_initial_state()
        dt = self.choose_time_step(state)
        spatial = SpatialOperator(self.case, self.subdomain, self.comm)
        integrator = TimeIntegrator(spatial, self.case.time.integration_order, state)

        if write_initial:
            self.write("Original", state)
        for step in range(self.case.time.num_steps):
            state = integrator.advance(state, dt)
            for writer in self.writers:
                if self.case.outputs.is_due(writer.frequency, step):
                    writer.write(str(step), state)
        self.write("Final", state)
        return state

    def write(self, label: str, state: FlowState) -> None:
        """
        Call every writer once with this label, whatever its interval. A frequency of -1 suppresses the interval writes only; the first and last state are written through every writer.
        """

        for writer in self.writers:
            writer.write(label, state)


def _max_op() -> Any:
    from mpi4py import MPI

    return MPI.MAX
