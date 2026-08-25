from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from xml.dom import minidom

from ..boundary_conditions import BoundaryCondition, parse_boundary_condition
from .boundaries import Boundaries, Face, FaceCondition
from .case import Case, SolverOptions
from .fluid import Fluid
from .grid import Grid
from .initial_conditions import FieldInitial, InitialConditions, StepValue, UniformValue
from .time_control import OutputControl, TimeControl

AXIS_LETTERS = ("X", "Y", "Z")
VELOCITY_NAMES = ("u", "v", "w")


def read_case(path: str) -> Case:
    """
    Build a Case from an input XML file. Raises ValueError on a missing or malformed element rather than filling in a default, because a silent default in an input file is a wrong answer nobody sees.

    ParallelizationProperties is ignored. The rank arrangement follows the communicator the solver is given, so it belongs to a run and not to the case.
    """

    return parse_case(ElementTree.parse(path).getroot())


def parse_case(root: ElementTree.Element) -> Case:
    """
    Build a Case from a parsed `<Simulation>` element.
    """

    if root.tag != "Simulation":
        raise ValueError(f"Expected a <Simulation> root element, got <{root.tag}>.")

    fluid_node = _child(root, "FluidProperties")
    fluid = Fluid(rho=_float(fluid_node, "Rho"), nu=_float(fluid_node, "Nu"))

    grid_node = _child(root, "GridProperties")
    domain_node = _child(grid_node, "Domain")
    size_node = _child(grid_node, "Size")
    counts = [_int(domain_node, name) for name in ("nx", "ny", "nz")]
    spans = [_float(size_node, name) for name in ("Length", "Width", "Height")]
    dimension = sum(1 for count in counts if count > 0)
    if dimension == 0:
        raise ValueError("Domain must give a positive nx.")
    grid = Grid(
        shape=tuple(counts[:dimension]),
        extent=tuple(spans[:dimension]),
        num_ghost_layers=_int(grid_node, "numGhosts"),
    )

    solver_node = _child(root, "SolverProperties")
    time = TimeControl(
        num_steps=_int(grid_node, "nt"),
        cfl=_float(grid_node, "CFL"),
        integration_order=_int(solver_node, "TimeIntegrationOrder"),
    )
    solver = SolverOptions(
        include_convection=_bool(solver_node, "IncludeConvectionEffects"),
        include_diffusion=_bool(solver_node, "IncludeViscousEffects"),
        include_pressure=_bool(solver_node, "IncludePressureEffects"),
        convection_scheme=_text(solver_node, "ConvectionScheme"),
        viscous_scheme=_text(solver_node, "ViscousScheme"),
    )

    output_node = _child(root, "OutputProperties")
    outputs = OutputControl(
        total_csv_frequency=_int(output_node, "WriteTotalCSVFrequency"),
        partial_csv_frequency=_int(output_node, "WritePartialCSVFrequency"),
        vtk_frequency=_int(output_node, "WriteTotalVTKFrequency"),
    )

    return Case(
        fluid=fluid,
        grid=grid,
        time=time,
        boundaries=_parse_boundaries(_child(root, "BoundaryConditions"), dimension),
        initial=parse_initial_conditions(_child(root, "InitialConditions"), dimension),
        solver=solver,
        outputs=outputs,
    )


def write_case(case: Case) -> str:
    """
    Render a Case as input XML text. Reading the result back gives a Case whose fields start at the same values, so the reader and this writer stay in step.

    The format holds one uniform value and one step box per field. Several uniform contributions on one field are written as their sum, which starts the field identically but reads back as a single UniformValue. Several step contributions on one field cannot be written at all and raise.

    ParallelizationProperties is written with -1 everywhere, which means the rank arrangement is derived from the communicator.
    """

    root = ElementTree.Element("Simulation")

    fluid_node = ElementTree.SubElement(root, "FluidProperties")
    _put(fluid_node, "Rho", case.fluid.rho)
    _put(fluid_node, "Nu", case.fluid.nu)

    grid_node = ElementTree.SubElement(root, "GridProperties")
    size_node = ElementTree.SubElement(grid_node, "Size")
    spans = list(case.grid.extent) + [0.0] * (3 - case.dimension)
    for name, span in zip(("Length", "Width", "Height"), spans):
        _put(size_node, name, span)
    domain_node = ElementTree.SubElement(grid_node, "Domain")
    counts = list(case.grid.shape) + [0] * (3 - case.dimension)
    for name, count in zip(("nx", "ny", "nz"), counts):
        _put(domain_node, name, count)
    _put(grid_node, "nt", case.time.num_steps)
    _put(grid_node, "numGhosts", case.grid.num_ghost_layers)
    _put(grid_node, "CFL", case.time.cfl)

    root.append(_write_initial(case.initial, case.dimension))
    root.append(_write_boundaries(case.boundaries, case.dimension))

    parallel_node = ElementTree.SubElement(root, "ParallelizationProperties")
    for name in ("numProcs", "numProcsX", "numProcsY", "numProcsZ"):
        _put(parallel_node, name, -1)

    solver_node = ElementTree.SubElement(root, "SolverProperties")
    _put(solver_node, "IncludeConvectionEffects", case.solver.include_convection)
    _put(solver_node, "ConvectionScheme", case.solver.convection_scheme)
    _put(solver_node, "IncludeViscousEffects", case.solver.include_diffusion)
    _put(solver_node, "ViscousScheme", case.solver.viscous_scheme)
    _put(solver_node, "TimeIntegrationOrder", case.time.integration_order)
    _put(solver_node, "IncludePressureEffects", case.solver.include_pressure)

    output_node = ElementTree.SubElement(root, "OutputProperties")
    _put(output_node, "WriteTotalVTKFrequency", case.outputs.vtk_frequency)
    _put(output_node, "WriteTotalCSVFrequency", case.outputs.total_csv_frequency)
    _put(output_node, "WritePartialCSVFrequency", case.outputs.partial_csv_frequency)

    raw = ElementTree.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _parse_boundaries(node: ElementTree.Element, dimension: int) -> Boundaries:
    conditions = {}
    for face in Face:
        name = face.name.capitalize()
        kind = parse_boundary_condition(_text(node, f"{name}Wall"))
        velocity: tuple[float, ...] = ()
        pressure = 0.0
        if kind == BoundaryCondition.INFLOW:
            inflow = _child(node, f"{name}Inflow")
            velocity = tuple(_float(inflow, key) for key in VELOCITY_NAMES[:dimension])
        if kind == BoundaryCondition.OUTFLOW:
            pressure = _float(_child(node, f"{name}Outflow"), "p")
        conditions[face.name.lower()] = FaceCondition(kind=kind, velocity=velocity, pressure=pressure)
    return Boundaries(**conditions)


def _write_boundaries(boundaries: Boundaries, dimension: int) -> ElementTree.Element:
    node = ElementTree.Element("BoundaryConditions")
    for face in Face:
        name = face.name.capitalize()
        condition = boundaries.face(face)
        _put(node, f"{name}Wall", condition.kind.value)
        inflow = ElementTree.SubElement(node, f"{name}Inflow")
        components = list(condition.velocity) + [0.0] * (3 - len(condition.velocity))
        for key, value in zip(VELOCITY_NAMES, components):
            _put(inflow, key, value)
        _put(ElementTree.SubElement(node, f"{name}Outflow"), "p", condition.pressure)
    return node


def parse_initial_conditions(node: ElementTree.Element, dimension: int) -> InitialConditions:
    """
    Build the initial conditions from an `<InitialConditions>` element. The GUI dialog keeps this block as XML text, so it parses it through here rather than keeping a second reader.
    """

    for flag, message in (
        ("ReadFromVtrFile", "Reading initial conditions from a VTR file is not implemented."),
        ("ReadFromCsvFile", "Reading initial conditions from a CSV file is not implemented."),
    ):
        if _bool(node, flag):
            raise NotImplementedError(message)
    if not _bool(node, "SpecifyValues"):
        raise ValueError("InitialConditions must set SpecifyValues to True; no other mode is implemented.")

    values = _child(node, "SpecifiedValues")
    pressure = _parse_field_initial(_child(values, "p"), dimension)
    velocity = tuple(_parse_field_initial(_child(values, name), dimension) for name in VELOCITY_NAMES[:dimension])
    return InitialConditions(pressure=pressure, velocity=velocity)


def _parse_field_initial(node: ElementTree.Element, dimension: int) -> FieldInitial:
    for flag, shape in (("UseSinusoidal", "Sinusoidal"), ("UsePolynomial", "Polynomial"), ("UseGaussian", "Gaussian")):
        if _bool(node, flag):
            raise NotImplementedError(f"{shape} initial conditions are not implemented.")

    contributions: list[UniformValue | StepValue] = []
    if _bool(node, "UseUniform"):
        contributions.append(UniformValue(_float(_child(node, "UniformParameters"), "ConstantValue")))
    if _bool(node, "UseStep"):
        step = _child(node, "StepParameters")
        contributions.append(
            StepValue(
                magnitude=_float(step, "StepMagnitude"),
                start=tuple(_float(step, f"start{letter}") for letter in AXIS_LETTERS[:dimension]),
                end=tuple(_float(step, f"end{letter}") for letter in AXIS_LETTERS[:dimension]),
            )
        )
    return tuple(contributions)


def _write_initial(initial: InitialConditions, dimension: int) -> ElementTree.Element:
    node = ElementTree.Element("InitialConditions")
    _put(node, "ReadFromVtrFile", False)
    _put(node, "ReadFromCsvFile", False)
    _put(node, "SpecifyValues", True)
    values = ElementTree.SubElement(node, "SpecifiedValues")
    fields = [("p", initial.pressure)]
    for index, name in enumerate(VELOCITY_NAMES[:dimension]):
        fields.append((name, initial.velocity[index] if initial.velocity else ()))
    for name, contributions in fields:
        _write_field_initial(ElementTree.SubElement(values, name), contributions, dimension)
    return node


def _write_field_initial(node: ElementTree.Element, contributions: FieldInitial, dimension: int) -> None:
    uniforms = [c for c in contributions if isinstance(c, UniformValue)]
    steps = [c for c in contributions if isinstance(c, StepValue)]
    if len(steps) > 1:
        raise ValueError(
            f"<{node.tag}> has {len(steps)} step contributions, but the XML format holds one "
            f"<StepParameters> per field, so the rest would be lost."
        )
    total = sum(c.value for c in uniforms)
    step = steps[0] if steps else None

    _put(node, "UseUniform", bool(uniforms))
    _put(ElementTree.SubElement(node, "UniformParameters"), "ConstantValue", total if uniforms else 0.0)
    _put(node, "UseStep", step is not None)
    step_node = ElementTree.SubElement(node, "StepParameters")
    _put(step_node, "StepMagnitude", step.magnitude if step else 0.0)
    for index, letter in enumerate(AXIS_LETTERS):
        if step is not None and index < dimension:
            low, high = step.start[index], step.end[index]
        else:
            low, high = 0.0, 0.0
        _put(step_node, f"start{letter}", low)
        _put(step_node, f"end{letter}", high)
    for flag in ("UseSinusoidal", "UsePolynomial", "UseGaussian"):
        _put(node, flag, False)


def _child(node: ElementTree.Element, tag: str) -> ElementTree.Element:
    found = node.find(tag)
    if found is None:
        raise ValueError(f"<{node.tag}> is missing the <{tag}> element.")
    return found


def _text(node: ElementTree.Element, tag: str) -> str:
    return (_child(node, tag).text or "").strip()


def _int(node: ElementTree.Element, tag: str) -> int:
    raw = _text(node, tag)
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"<{tag}> must be an integer, got {raw!r}.") from error


def _float(node: ElementTree.Element, tag: str) -> float:
    raw = _text(node, tag)
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"<{tag}> must be a number, got {raw!r}.") from error


def _bool(node: ElementTree.Element, tag: str) -> bool:
    raw = _text(node, tag)
    if raw in ("True", "true", "1"):
        return True
    if raw in ("False", "false", "0"):
        return False
    raise ValueError(f"<{tag}> must be True or False, got {raw!r}.")


def _put(parent: ElementTree.Element, tag: str, value: object) -> ElementTree.Element:
    element = ElementTree.SubElement(parent, tag)
    element.text = str(value)
    return element
