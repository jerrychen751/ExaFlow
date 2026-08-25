from __future__ import annotations

from ..fields import FlowState
from .operators import SpatialOperator


class TimeIntegrator:
    """
    Explicit Runge-Kutta time integration. Order 1 is forward Euler, order 2 is the midpoint method and order 3 is the Shu-Osher TVD scheme.

    The stage buffers are allocated once at construction and reused, so a step allocates nothing. `advance` returns the state holding the new time level, which alternates between two buffers; the caller must use the returned object and must not keep a reference to the one it passed in.
    """

    def __init__(self, spatial: SpatialOperator, order: int, template: FlowState) -> None:
        if order not in (1, 2, 3):
            raise ValueError(f"integration order must be 1, 2 or 3, got {order}.")
        self._spatial = spatial
        self._order = order
        self._rate = template.allocate_zeros()
        self._next = template.allocate_zeros()
        self._stage = template.allocate_zeros() if order > 1 else None

    def advance(self, state: FlowState, dt: float) -> FlowState:
        """
        Take one step of size `dt` from `state` and return the new state.
        """

        match self._order:
            case 1:
                self._spatial.evaluate(state, self._rate)
                self._next.set_sum(state, self._rate, dt)
            case 2:
                stage = self._require_stage()
                self._spatial.evaluate(state, self._rate)
                stage.set_sum(state, self._rate, 0.5 * dt)
                self._spatial.evaluate(stage, self._rate)
                self._next.set_sum(state, self._rate, dt)
            case _:
                stage = self._require_stage()
                self._spatial.evaluate(state, self._rate)
                stage.set_sum(state, self._rate, dt)

                self._spatial.evaluate(stage, self._rate)
                self._next.set_sum(stage, self._rate, dt)
                stage.set_blend(state, self._next, 0.25)

                self._spatial.evaluate(stage, self._rate)
                self._next.set_sum(stage, self._rate, dt)
                self._next.set_blend(state, self._next, 2.0 / 3.0)

        state, self._next = self._next, state
        return state

    def _require_stage(self) -> FlowState:
        if self._stage is None:
            raise AssertionError("A scheme above order 1 always allocates a stage buffer.")
        return self._stage
