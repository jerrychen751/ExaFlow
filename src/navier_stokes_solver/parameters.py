from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, SupportsFloat, SupportsInt, TypeAlias, TypedDict

import xmltodict

from .boundary_conditions import BoundaryCondition, parse_boundary_condition


class Inflow(TypedDict, total=False):
    u: float
    v: float
    w: float


class Outflow(TypedDict, total=False):
    p: float


Domain: TypeAlias = tuple[int, ...]
Size: TypeAlias = tuple[float, ...]

# Rank Coords depend on whether it's 1D, 2D, or 3D
RankCoords: TypeAlias = int | tuple[int, ...]


def _parse_xml_bool(value: object) -> bool:
    """
    Parse the repo's XML booleans (historically "True"/"False" strings) to bool.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
    raise ValueError(f"Expected XML boolean 'True'/'False', got {value!r}.")


def _require_positive_int(value: str | SupportsInt, *, name: str) -> int:
    try:
        as_int = int(value)
    except Exception as exc:
        raise ValueError(f"Expected integer for {name}, got {value!r}.") from exc
    if as_int <= 0:
        raise ValueError(f"Expected {name} > 0, got {as_int}.")
    return as_int


def _require_non_negative_int(value: str | SupportsInt, *, name: str) -> int:
    try:
        as_int = int(value)
    except Exception as exc:
        raise ValueError(f"Expected integer for {name}, got {value!r}.") from exc
    if as_int < 0:
        raise ValueError(f"Expected {name} >= 0, got {as_int}.")
    return as_int


def _require_positive_float(value: str | SupportsFloat, *, name: str) -> float:
    try:
        as_float = float(value)
    except Exception as exc:
        raise ValueError(f"Expected float for {name}, got {value!r}.") from exc
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise ValueError(f"Expected finite {name} > 0, got {as_float}.")
    return as_float


def _require_non_negative_float(value: str | SupportsFloat, *, name: str) -> float:
    try:
        as_float = float(value)
    except Exception as exc:
        raise ValueError(f"Expected float for {name}, got {value!r}.") from exc
    if not math.isfinite(as_float) or as_float < 0.0:
        raise ValueError(f"Expected finite {name} >= 0, got {as_float}.")
    return as_float


@dataclass(slots=True)
class SimulationParameters:
    """
    Canonical simulation configuration.

    This class centralizes parameter validation and derived quantities like
    grid spacing (`dx`, `dy`, `dz`) and timestep size (`dt`).
    """

    # Fluid properties
    rho: float
    nu: float

    # Grid definition (physical domain)
    domain: Domain
    size: Size

    # Runtime
    nt: int
    num_ghost_layers: int
    cfl: float

    # Parallelization (topology may be inferred if not provided)
    num_procs_x: int | None = None
    num_procs_y: int | None = None
    num_procs_z: int | None = None
    comm: Any | None = None

    # Boundary conditions
    left_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    left_inflow: Inflow = field(default_factory=Inflow)
    left_outflow: Outflow = field(default_factory=Outflow)

    right_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    right_inflow: Inflow = field(default_factory=Inflow)
    right_outflow: Outflow = field(default_factory=Outflow)

    top_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    top_inflow: Inflow = field(default_factory=Inflow)
    top_outflow: Outflow = field(default_factory=Outflow)

    bottom_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    bottom_inflow: Inflow = field(default_factory=Inflow)
    bottom_outflow: Outflow = field(default_factory=Outflow)

    front_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    front_inflow: Inflow = field(default_factory=Inflow)
    front_outflow: Outflow = field(default_factory=Outflow)

    back_wall: BoundaryCondition = BoundaryCondition.NO_SLIP
    back_inflow: Inflow = field(default_factory=Inflow)
    back_outflow: Outflow = field(default_factory=Outflow)

    # Initial conditions (XML-driven structure; kept as parsed dict for now)
    initial_conditions: dict[str, Any] | None = None

    # Solver toggles and schemes
    include_convection: bool = True
    include_diffusion: bool = True
    include_pressure: bool = False
    convection_scheme: str = "Upwind"
    viscous_scheme: str = "Central"
    time_integration_order: int = 1

    # Output controls
    vtk_frequency: int = -1
    total_csv_frequency: int = -1
    partial_csv_frequency: int = -1

    # Derived quantities (set by methods)
    dx: float | None = field(init=False, default=None)
    dy: float | None = field(init=False, default=None)
    dz: float | None = field(init=False, default=None)
    dt: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if len(self.domain) not in (1, 2, 3):
            raise ValueError(f"domain must be 1D/2D/3D, got {self.domain!r}.")
        if len(self.size) != len(self.domain):
            raise ValueError(
                f"size must have same dimension as domain; got domain={self.domain!r}, size={self.size!r}."
            )
        if any(n < 2 for n in self.domain):
            raise ValueError(f"All domain sizes must be >= 2, got {self.domain!r}.")
        if self.nt <= 0:
            raise ValueError(f"nt must be > 0, got {self.nt}.")
        if self.num_ghost_layers <= 0:
            raise ValueError(f"num_ghost_layers must be > 0, got {self.num_ghost_layers}.")
        if self.cfl <= 0.0:
            raise ValueError(f"cfl must be > 0, got {self.cfl}.")
        if self.rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {self.rho}.")
        if self.nu < 0.0:
            raise ValueError(f"nu must be >= 0, got {self.nu}.")
        if self.time_integration_order not in (1, 2, 3, 4):
            raise ValueError(f"time_integration_order must be 1..4, got {self.time_integration_order}.")

        self._validate_periodic_pairs()
        self._finalize_process_topology()
        self.compute_grid_spacing()

        if self.include_pressure:
            raise NotImplementedError(
                "Pressure projection / Poisson solve is not implemented in the refactored solver yet. "
                "Set include_pressure=False."
            )

    @property
    def dimension(self) -> int:
        return len(self.domain)

    @property
    def num_procs(self) -> int:
        if self.comm is None:
            return 1
        return int(self.comm.Get_size())

    def _validate_periodic_pairs(self) -> None:
        pairs = [
            ("left_wall", "right_wall"),
            ("top_wall", "bottom_wall"),
            ("front_wall", "back_wall"),
        ]
        for a, b in pairs:
            va = getattr(self, a)
            vb = getattr(self, b)
            if (va == BoundaryCondition.PERIODIC) != (vb == BoundaryCondition.PERIODIC):
                raise ValueError(
                    f"Periodic boundaries must be paired; got {a}={va.value!r}, {b}={vb.value!r}."
                )

    def _finalize_process_topology(self) -> None:
        """
        Ensure (num_procs_x, num_procs_y, num_procs_z) are set consistently.

        If no MPI communicator is provided, defaults to a single process.
        """

        nprocs = self.num_procs
        if self.dimension == 1:
            self.num_procs_x = nprocs
            self.num_procs_y = None
            self.num_procs_z = None
            return

        if self.dimension == 2:
            if self.num_procs_x is None or self.num_procs_y is None or self.num_procs_x * self.num_procs_y != nprocs:
                self.num_procs_x, self.num_procs_y = self._best_factorization_2d(nprocs, self.domain[0], self.domain[1])
            self.num_procs_z = None
            return

        if self.dimension == 3:
            if (
                self.num_procs_x is None
                or self.num_procs_y is None
                or self.num_procs_z is None
                or (self.num_procs_x * self.num_procs_y * self.num_procs_z != nprocs)
            ):
                self.num_procs_x, self.num_procs_y, self.num_procs_z = self._best_factorization_3d(
                    nprocs, self.domain[0], self.domain[1], self.domain[2]
                )
            return

        raise AssertionError("Unreachable: dimension already validated.")

    @staticmethod
    def _best_factorization_2d(nprocs: int, nx: int, ny: int) -> tuple[int, int]:
        """
        Pick a (px, py) with px*py=nprocs that minimizes aspect distortion.
        """

        best: tuple[int, int] = (1, nprocs)
        best_error = float("inf")
        for px in range(1, int(math.sqrt(nprocs)) + 1):
            if nprocs % px != 0:
                continue
            py = nprocs // px
            error = abs((px / nx) - (py / ny))
            if error < best_error:
                best_error = error
                best = (px, py)
        return best

    @staticmethod
    def _best_factorization_3d(nprocs: int, nx: int, ny: int, nz: int) -> tuple[int, int, int]:
        """
        Pick a (px, py, pz) with px*py*pz=nprocs that minimizes aspect distortion.
        """

        best_triplet = (1, 1, nprocs)
        best_error = float("inf")

        for px in range(1, int(round(nprocs ** (1 / 3))) + 2):
            if nprocs % px != 0:
                continue
            remaining = nprocs // px
            for py in range(1, int(math.sqrt(remaining)) + 2):
                if remaining % py != 0:
                    continue
                pz = remaining // py
                error = abs((px / nx) - (py / ny)) + abs((px / nx) - (pz / nz))
                if error < best_error:
                    best_error = error
                    best_triplet = (px, py, pz)

        return best_triplet

    def compute_grid_spacing(self) -> None:
        """
        Compute physical grid spacing.

        Invariant (chosen in Q&A): dx = length/(nx-1), etc.
        """

        if self.dimension == 1:
            (nx,) = self.domain
            (length,) = self.size
            self.dx = length / float(nx - 1)
            return

        if self.dimension == 2:
            nx, ny = self.domain
            length, width = self.size
            self.dx = length / float(nx - 1)
            self.dy = width / float(ny - 1)
            return

        nx, ny, nz = self.domain
        length, width, height = self.size
        self.dx = length / float(nx - 1)
        self.dy = width / float(ny - 1)
        self.dz = height / float(nz - 1)

    def compute_dt(self, max_velocity_magnitude: float) -> float:
        """
        Compute dt from advection CFL and explicit diffusion stability constraints.
        """

        if self.dx is None:
            self.compute_grid_spacing()

        max_u = float(max_velocity_magnitude)
        if max_u < 0.0 or not math.isfinite(max_u):
            raise ValueError(f"max_velocity_magnitude must be finite and >= 0, got {max_u}.")

        spacings: list[float] = []
        if self.dx is not None:
            spacings.append(self.dx)
        if self.dy is not None:
            spacings.append(self.dy)
        if self.dz is not None:
            spacings.append(self.dz)

        min_h = min(spacings)
        if max_u == 0.0:
            advection_dt = float("inf")
        else:
            advection_dt = self.cfl * min_h / max_u

        if self.nu == 0.0:
            diffusion_dt = float("inf")
        else:
            inv_sum = 0.0
            inv_sum += 1.0 / (float(self.dx) * float(self.dx))  # type: ignore[arg-type]
            if self.dy is not None:
                inv_sum += 1.0 / (float(self.dy) * float(self.dy))
            if self.dz is not None:
                inv_sum += 1.0 / (float(self.dz) * float(self.dz))
            diffusion_dt = 0.5 / (self.nu * inv_sum)

        self.dt = min(advection_dt, diffusion_dt)
        return self.dt


def simulation_parameters_from_xml(file_path: str, comm: Any | None) -> SimulationParameters:
    with open(file_path, "r", encoding="utf-8") as file:
        sim_dict = xmltodict.parse(file.read())

    sim = sim_dict["Simulation"]

    fluid = sim["FluidProperties"]
    rho = _require_positive_float(fluid["Rho"], name="rho")
    nu = _require_non_negative_float(fluid["Nu"], name="nu")

    grid = sim["GridProperties"]
    domain_dict = grid["Domain"]
    nx = int(domain_dict.get("nx", 0))
    ny = int(domain_dict.get("ny", 0))
    nz = int(domain_dict.get("nz", 0))
    if nx > 0 and ny > 0 and nz > 0:
        domain: Domain = (nx, ny, nz)
    elif nx > 0 and ny > 0:
        domain = (nx, ny)
    elif nx > 0:
        domain = (nx,)
    else:
        raise ValueError(f"Invalid domain in XML: {domain_dict!r}")

    size_dict = grid["Size"]
    length = _require_positive_float(size_dict["Length"], name="length")
    width = _require_non_negative_float(size_dict.get("Width", 0.0), name="width")
    height = _require_non_negative_float(size_dict.get("Height", 0.0), name="height")
    if len(domain) == 3:
        size: Size = (length, width, height)
    elif len(domain) == 2:
        size = (length, width)
    else:
        size = (length,)

    nt = _require_positive_int(grid["nt"], name="nt")
    num_ghost_layers = _require_positive_int(grid["numGhosts"], name="numGhosts")
    cfl = _require_positive_float(grid["CFL"], name="CFL")

    parallel = sim["ParallelizationProperties"]
    requested_num_procs = _require_positive_int(parallel["numProcs"], name="numProcs")
    num_procs_x = int(parallel["numProcsX"])
    num_procs_y = int(parallel["numProcsY"])
    num_procs_z = int(parallel["numProcsZ"])
    px = None if num_procs_x <= 0 else num_procs_x
    py = None if num_procs_y <= 0 else num_procs_y
    pz = None if num_procs_z <= 0 else num_procs_z

    output = sim["OutputProperties"]
    vtk_frequency = int(output["WriteTotalVTKFrequency"])
    partial_csv_frequency = int(output["WritePartialCSVFrequency"])
    total_csv_frequency = int(output["WriteTotalCSVFrequency"])

    solver = sim["SolverProperties"]
    include_convection = _parse_xml_bool(solver["IncludeConvectionEffects"])
    convection_scheme = str(solver["ConvectionScheme"])
    include_diffusion = _parse_xml_bool(solver["IncludeViscousEffects"])
    viscous_scheme = str(solver["ViscousScheme"])
    include_pressure = _parse_xml_bool(solver["IncludePressureEffects"])
    time_integration_order = _require_non_negative_int(solver["TimeIntegrationOrder"], name="TimeIntegrationOrder")

    initial_conditions: dict[str, Any] | None = sim.get("InitialConditions")

    bc = sim["BoundaryConditions"]
    left_wall = parse_boundary_condition(str(bc["LeftWall"]))
    right_wall = parse_boundary_condition(str(bc["RightWall"]))
    top_wall = parse_boundary_condition(str(bc["TopWall"]))
    bottom_wall = parse_boundary_condition(str(bc["BottomWall"]))
    front_wall = parse_boundary_condition(str(bc["FrontWall"]))
    back_wall = parse_boundary_condition(str(bc["BackWall"]))

    def parse_inflow(name: str) -> Inflow:
        flow = bc[name]
        return {
            "u": float(flow["u"]),
            "v": float(flow["v"]),
            "w": float(flow["w"]),
        }

    def parse_outflow(name: str) -> Outflow:
        flow = bc[name]
        return {"p": float(flow["p"])}

    left_inflow = parse_inflow("LeftInflow")
    right_inflow = parse_inflow("RightInflow")
    top_inflow = parse_inflow("TopInflow")
    bottom_inflow = parse_inflow("BottomInflow")
    front_inflow = parse_inflow("FrontInflow")
    back_inflow = parse_inflow("BackInflow")

    left_outflow = parse_outflow("LeftOutflow")
    right_outflow = parse_outflow("RightOutflow")
    top_outflow = parse_outflow("TopOutflow")
    bottom_outflow = parse_outflow("BottomOutflow")
    front_outflow = parse_outflow("FrontOutflow")
    back_outflow = parse_outflow("BackOutflow")

    if comm is not None:
        actual_num_procs = int(comm.Get_size())
        if actual_num_procs != requested_num_procs:
            raise ValueError(
                f"XML requests numProcs={requested_num_procs}, but MPI communicator size is {actual_num_procs}."
            )

    return SimulationParameters(
        rho=rho,
        nu=nu,
        domain=domain,
        size=size,
        nt=nt,
        num_ghost_layers=num_ghost_layers,
        cfl=cfl,
        num_procs_x=px,
        num_procs_y=py,
        num_procs_z=pz,
        comm=comm,
        left_wall=left_wall,
        left_inflow=left_inflow,
        left_outflow=left_outflow,
        right_wall=right_wall,
        right_inflow=right_inflow,
        right_outflow=right_outflow,
        top_wall=top_wall,
        top_inflow=top_inflow,
        top_outflow=top_outflow,
        bottom_wall=bottom_wall,
        bottom_inflow=bottom_inflow,
        bottom_outflow=bottom_outflow,
        front_wall=front_wall,
        front_inflow=front_inflow,
        front_outflow=front_outflow,
        back_wall=back_wall,
        back_inflow=back_inflow,
        back_outflow=back_outflow,
        initial_conditions=initial_conditions,
        include_convection=include_convection,
        include_diffusion=include_diffusion,
        include_pressure=include_pressure,
        convection_scheme=convection_scheme,
        viscous_scheme=viscous_scheme,
        time_integration_order=time_integration_order,
        vtk_frequency=vtk_frequency,
        total_csv_frequency=total_csv_frequency,
        partial_csv_frequency=partial_csv_frequency,
    )

