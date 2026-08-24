from __future__ import annotations

import math
from typing import Any

import numpy as np

from .parameters import SimulationParameters, _parse_xml_bool


def initialize_fields(sim_params: SimulationParameters) -> tuple[np.ndarray, ...]:
    """
    Initialize fields (p, u, v, w) from `sim_params.initial_conditions`.

    Returns:
        1D: (p, u)
        2D: (p, u, v)
        3D: (p, u, v, w)
    """

    if sim_params.initial_conditions is None:
        raise ValueError("SimulationParameters.initial_conditions is required to initialize fields.")

    ic = sim_params.initial_conditions

    if _parse_xml_bool(ic.get("ReadFromVtrFile", "False")):
        raise NotImplementedError("Reading initial conditions from VTR is not implemented.")
    if _parse_xml_bool(ic.get("ReadFromCsvFile", "False")):
        raise NotImplementedError("Reading initial conditions from CSV is not implemented.")

    if not _parse_xml_bool(ic.get("SpecifyValues", "False")):
        raise ValueError("InitialConditions must set SpecifyValues=True (other modes not implemented).")

    p = _fill_from_specified_values(sim_params, "p")
    u = _fill_from_specified_values(sim_params, "u")
    if sim_params.dimension == 1:
        return (p, u)

    v = _fill_from_specified_values(sim_params, "v")
    if sim_params.dimension == 2:
        return (p, u, v)

    w = _fill_from_specified_values(sim_params, "w")
    return (p, u, v, w)


def _fill_from_specified_values(sim_params: SimulationParameters, quantity: str) -> np.ndarray:
    ic = sim_params.initial_conditions
    if ic is None:
        raise ValueError("SimulationParameters.initial_conditions is required.")

    specified = ic["SpecifiedValues"][quantity]
    array = np.zeros(sim_params.domain, dtype=float)

    if _parse_xml_bool(specified.get("UseUniform", "False")):
        constant = float(specified["UniformParameters"]["ConstantValue"])
        array += constant

    if _parse_xml_bool(specified.get("UseStep", "False")):
        step = specified["StepParameters"]
        magnitude = float(step["StepMagnitude"])

        slices = _step_slices(sim_params.domain, step)
        array[slices] += magnitude

    if _parse_xml_bool(specified.get("UseSinusoidal", "False")):
        raise NotImplementedError("Sinusoidal initial conditions are not implemented.")
    if _parse_xml_bool(specified.get("UsePolynomial", "False")):
        raise NotImplementedError("Polynomial initial conditions are not implemented.")
    if _parse_xml_bool(specified.get("UseGaussian", "False")):
        raise NotImplementedError("Gaussian initial conditions are not implemented.")

    return array


def _step_slices(domain: tuple[int, ...], step: dict[str, Any]) -> tuple[slice, ...]:
    """
    Convert step parameters (fractional start/end) to numpy slices. The condition is start <= i/n <= end, inclusive, expressed with the integer index bounds i_start = ceil(start*n) and i_end = floor(end*n).
    """

    def bounds(n: int, start_key: str, end_key: str) -> slice:
        start = float(step[start_key])
        end = float(step[end_key])
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError(f"Step bounds must be within [0, 1]. Got start={start}, end={end}.")
        if end < start:
            raise ValueError(f"Step end must be >= start. Got start={start}, end={end}.")

        i_start = int(math.ceil(start * n))
        i_end_inclusive = int(math.floor(end * n))
        if i_end_inclusive < i_start:
            return slice(0, 0)
        return slice(i_start, i_end_inclusive + 1)

    if len(domain) == 1:
        (nx,) = domain
        return (bounds(nx, "startX", "endX"),)
    if len(domain) == 2:
        nx, ny = domain
        return (bounds(nx, "startX", "endX"), bounds(ny, "startY", "endY"))
    nx, ny, nz = domain
    return (bounds(nx, "startX", "endX"), bounds(ny, "startY", "endY"), bounds(nz, "startZ", "endZ"))

