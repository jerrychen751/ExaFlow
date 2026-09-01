from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .boundary_application import initialize_boundaries
from .config.case import Case
from .fields import FlowState, build_initial_state
from .io.writers import Writer, build_writers
from .mpi.process_grid import ProcessGrid, choose_process_grid
from .mpi.subdomain import Subdomain
from .numerics.operators import SpatialOperator
from .numerics.time_step import TimeIntegrator, compute_time_step

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


class SimulationSession:
    """
    One run in progress on this rank. The session owns everything that belongs to a run rather than to the problem: the communicator, the decomposition, the fields, how far the run has gone, the stage buffers and the output schedule. The case itself stays frozen throughout.

    `state`, `step_index`, `current_time` and `dt` move together, so a caller that stops between steps can read where the run is. `advance_one_step` replaces `state`, so a reference kept across a step points at a stage buffer the next step overwrites.

    Every method is collective. Build the session on every rank with the same case, and call the same methods on every rank in the same order.
    """

    def __init__(
        self,
        case: Case,
        comm: Intracomm | None = None,
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
            self.writers = build_writers(output_directory, case.grid, self.subdomain, comm, case.outputs)
        else:
            self.writers = ()

        self.state = self.build_initial_state()
        self.step_index = 0
        self.current_time = 0.0
        self.dt = self.choose_time_step()

        self._integrator = TimeIntegrator(
            SpatialOperator(case, self.subdomain, comm),
            case.time.integration_order,
            self.state,
        )

    def build_initial_state(self) -> FlowState:
        """
        The starting fields for this rank, with the prescribed boundary values already written into the ghost layers of every domain face this rank owns.
        """

        state = build_initial_state(self.case, self.subdomain)
        initialize_boundaries(state, self.case, self.subdomain)
        return state

    def choose_time_step(self) -> float:
        """
        The largest stable step in seconds for the state this session holds now, reduced over every rank so that all ranks march together.
        """

        local_max = self.state.compute_max_speed()
        if self.comm is not None:
            from mpi4py import MPI

            local_max = float(self.comm.allreduce(local_max, op=MPI.MAX))
        return compute_time_step(self.case, local_max)

    def is_complete(self) -> bool:
        """
        Report whether the run has reached its target. `case.time.num_steps` is the step budget of the whole run counted from time zero.
        """

        return self.step_index >= self.case.time.num_steps

    def write(self, label: str) -> None:
        """
        Call every writer once with this label, whatever its interval. A frequency of -1 suppresses the interval writes only; the first and last state are written through every writer.
        """

        for writer in self.writers:
            writer.write(label, self.state)

    def advance_one_step(self) -> None:
        """
        Take one step, then write through every writer whose interval names the completed step count, so the label of a file and the step a reader counts are the same number.

        The caller tests `is_complete` first. This replaces `state`, so a reference the caller kept before the call is a stale buffer.
        """

        if self.is_complete():
            raise RuntimeError(f"The run is complete at step {self.step_index} of {self.case.time.num_steps}.")

        self.state = self._integrator.advance(self.state, self.dt)
        self.step_index += 1
        self.current_time += self.dt

        label = str(self.step_index)
        for writer in self.writers:
            if self.case.outputs.is_due(writer.frequency, self.step_index):
                writer.write(label, self.state)

    def run_until_complete(self, *, write_initial: bool = True) -> FlowState:
        """
        Advance until `is_complete` reports True, and return the final state for this rank. Every writer is called once more with the label "Final" at the end. `write_initial` writes the starting state first, under the label "Original".
        """

        if write_initial:
            self.write("Original")
        while not self.is_complete():
            self.advance_one_step()
        self.write("Final")
        return self.state
