from __future__ import annotations

from enum import Enum


class BoundaryCondition(str, Enum):
    """
    Boundary condition identifiers for domain faces.

    Values are kept as strings to preserve readable serialization (e.g., XML).
    """

    NO_SLIP = "No Slip Wall"
    SLIP = "Slip Wall"
    INFLOW = "Inflow"
    OUTFLOW = "Outflow"
    PERIODIC = "Periodic"
    TIME_DEPENDENT = "Time Dependent"


def parse_boundary_condition(value: str) -> BoundaryCondition:
    """
    Parse a boundary condition from a string.

    This is intentionally strict: since the refactor is enums-only, unknown values
    raise a ValueError rather than silently accepting strings.
    """

    try:
        return BoundaryCondition(value)
    except ValueError as exc:
        allowed = ", ".join(bc.value for bc in BoundaryCondition)
        raise ValueError(f"Unknown boundary condition {value!r}. Allowed: {allowed}.") from exc

