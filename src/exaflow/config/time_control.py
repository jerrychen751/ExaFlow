from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True, slots=True)
class TimeControl:
    """
    How far the run marches and how each step is taken. `num_steps` is the step budget of the whole run, counted from time zero, so a run restored at step 400 of 1000 takes 600 more steps. `cfl` scales the advective step size limit. `integration_order` selects the explicit Runge-Kutta scheme: 1 is Euler, 2 is the midpoint method and 3 is the Shu-Osher TVD scheme.

    `end_time` is the simulated time in seconds the run marches to, or None to march until the step budget runs out. With an end time set, `num_steps` becomes the cap that stops a run whose step size shrinks faster than the time left. `adaptive_time_step` recomputes the step size from the current state before every step instead of once from the initial state, which costs one reduction across the ranks per step.
    """

    num_steps: int
    cfl: float
    integration_order: int = 1
    end_time: float | None = None
    adaptive_time_step: bool = False

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}.")
        if not math.isfinite(self.cfl) or self.cfl <= 0.0:
            raise ValueError(f"cfl must be finite and > 0, got {self.cfl}.")
        if self.integration_order not in (1, 2, 3):
            raise ValueError(f"integration_order must be 1, 2 or 3, got {self.integration_order}.")
        if self.end_time is not None and (not math.isfinite(self.end_time) or self.end_time <= 0.0):
            raise ValueError(f"end_time must be finite and > 0, got {self.end_time}.")


class OutputFormat(str, Enum):
    """
    The file format a run writes. Values are kept as strings to preserve readable serialization (e.g., XML).
    """

    CSV = "CSV"
    VTK = "VTK"


def parse_output_format(value: str) -> OutputFormat:
    """
    Parse an output format from a string. This is intentionally strict: an unknown value raises a ValueError rather than silently accepting the string.
    """

    try:
        return OutputFormat(value)
    except ValueError as exc:
        allowed = ", ".join(fmt.value for fmt in OutputFormat)
        raise ValueError(f"Unknown output format {value!r}. Allowed: {allowed}.") from exc


@dataclass(frozen=True, slots=True)
class OutputControl:
    """
    Which format a run writes, and how often, counted in time steps. One run writes one format, so a run folder holds .vtr files or .csv files and never both.

    `total_frequency` is the interval of the file that holds the whole domain, and `partial_frequency` the interval of the per-rank files. Only CSV has a per-rank file, so VTK rejects a `partial_frequency` other than -1 rather than accept a value it would drop. A frequency of -1 asks for no writes during the march; it does not turn the format off, because the session writes the first and last state through every writer whatever its interval. Zero and negative values other than -1 are rejected, because a modulo against them cannot decide a step.
    """

    format: OutputFormat = OutputFormat.CSV
    total_frequency: int = -1
    partial_frequency: int = -1

    def __post_init__(self) -> None:
        for name in ("total_frequency", "partial_frequency"):
            frequency = getattr(self, name)
            if frequency != -1 and frequency < 1:
                raise ValueError(f"{name} must be -1 or >= 1, got {frequency}.")
        if self.format is not OutputFormat.CSV and self.partial_frequency != -1:
            raise ValueError(
                f"{self.format.value} writes no per-rank file, so partial_frequency must be -1, "
                f"got {self.partial_frequency}."
            )

    def is_due(self, frequency: int, step: int) -> bool:
        """
        Report whether a writer whose interval is `frequency` writes at this step. The step is the count of completed steps, so a frequency of 2 over five steps selects steps 2 and 4.
        """

        return frequency >= 1 and step % frequency == 0
