from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TimeControl:
    """
    How far the run marches and how each step is taken. `num_steps` is the count of time steps. `cfl` scales the advective step size limit. `integration_order` selects the explicit Runge-Kutta scheme: 1 is Euler, 2 is the midpoint method and 3 is the Shu-Osher TVD scheme.
    """

    num_steps: int
    cfl: float
    integration_order: int = 1

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}.")
        if not math.isfinite(self.cfl) or self.cfl <= 0.0:
            raise ValueError(f"cfl must be finite and > 0, got {self.cfl}.")
        if self.integration_order not in (1, 2, 3):
            raise ValueError(f"integration_order must be 1, 2 or 3, got {self.integration_order}.")


@dataclass(frozen=True, slots=True)
class OutputControl:
    """
    How often each output format is written during the march, counted in time steps. A value of -1 asks for no writes during the march; it does not turn the format off, because the solver writes the first and last state through every writer whatever its interval. Zero and negative values other than -1 are rejected, because a modulo against them cannot decide a step.
    """

    total_csv_frequency: int = -1
    partial_csv_frequency: int = -1
    vtk_frequency: int = -1

    def __post_init__(self) -> None:
        for name in ("total_csv_frequency", "partial_csv_frequency", "vtk_frequency"):
            frequency = getattr(self, name)
            if frequency != -1 and frequency < 1:
                raise ValueError(f"{name} must be -1 or >= 1, got {frequency}.")

    def is_due(self, frequency: int, step: int) -> bool:
        """
        Report whether a format whose interval is `frequency` writes at this zero-based step.
        """

        return frequency >= 1 and step % frequency == 0
