# ExaFlow

A Python solver for the incompressible Navier-Stokes equations in 1D, 2D, and 3D with MPI parallelization.

It simulates how fluids (like air or water) move through space over time. The Navier-Stokes equations describe fluid motion through three physical effects:

- **Convection** - fluid carries momentum along with its flow
- **Diffusion** - viscosity smooths out velocity differences (think honey vs. water)
- **Pressure** - enforces that the fluid can't compress or expand (incompressibility)

The solver discretizes a physical domain into a grid, then marches forward in time using explicit Runge-Kutta integration (orders 1-3).

## Quick start

### 1. Install

The project uses [uv](https://docs.astral.sh/uv/). One command builds the whole environment:

```bash
uv sync
```

This creates `.venv`, installs Python 3.13 (the version in `.python-version`), installs every dependency, and installs ExaFlow itself in editable mode. Nothing else is needed. Open MPI arrives as a Python wheel, so you do not install MPI through Homebrew or conda. `uv.lock` pins every exact version.

Run any command in that environment with `uv run`:

```bash
uv run python -c "from mpi4py import MPI; print(MPI.Get_library_version().split(',')[0])"
```

### 2. Run your first simulation

```bash
uv run python examples/template_3d_problem.py
```

That runs a 50x50x50 domain for 50 steps on one process. It prints the step number and finishes in about one second. To use four processes:

```bash
uv run mpiexec -n 4 python examples/template_3d_problem.py
```

### 3. Find the output

Every run writes into its own folder under `~/Documents/ExaFlow`:

```
~/Documents/ExaFlow/
    2026-08-24_200239_template_3d/
        Original_Total.csv
        Final_Total.csv
        Final_0.csv
        Final_1.csv
```

`*_Total.csv` holds the full domain, joined on rank 0. `*_<n>.csv` holds the part that rank `n` owned. A folder name is the start time plus a label, so a new run never overwrites an old one.

Set `EXAFLOW_OUTPUT_ROOT` to write the run folders somewhere else:

```bash
EXAFLOW_OUTPUT_ROOT=/tmp/exaflow-runs uv run python examples/template_3d_problem.py
```

### 4. Open the GUI

```bash
uv run python run_gui.py
```

The window has a control column on the left and a 3D viewer on the right. Choose a script, set the number of MPI processes, and press Run. The viewer loads the newest result file from the output root while the run continues.

The **Slice** row cuts a 3D result on one axis and shows that plane by itself. Pick the axis, move the position slider, and the camera faces the plane and stops rotating. The position label states the unit: metres for a `.vtr` file, and cells for a CSV file, which carries the indices and no physical extent. A 1D or 2D result is already a cross-section, so the viewer shows it flat and the control stays disabled.

## Where to look

```
src/exaflow/
    config/ # frozen value types; no numpy, no MPI, no Qt
        case.py # Case: the whole problem definition, plus SolverOptions
        fluid.py # Fluid(rho, nu)
        grid.py # Grid(shape, extent, num_ghost_layers) -> spacing, dimension
        boundaries.py # Face, FaceCondition, Boundaries
        initial_conditions.py # UniformValue, StepValue as value types
        time_control.py # TimeControl, OutputControl
        case_xml.py # read_case and write_case: one field map, both directions
    fields.py # FlowState: velocity (dimension, *padded), pressure (*padded)
    mpi/
        process_grid.py # ProcessGrid: rank arrangement and the factorization
        subdomain.py # Subdomain: the block one rank owns and every index rule
        ghost_exchange.py # non-blocking exchange of the outermost real layer
        gather.py # assemble the full domain on rank 0
    numerics/
        operators.py # Operator protocol, build_operators, SpatialOperator
        convection.py # first-order upwind (u dot grad) u
        diffusion.py # second-order central nu * laplacian(u)
        time_step.py # TimeIntegrator, Runge-Kutta orders 1-3
        pressure_poisson.py # Chorin projection (single rank, not yet wired in)
        README.md # array shapes, ghost layer rules, how to write an operator
    boundary_application.py # writes boundary values into the ghost layers
    solver.py # Solver: the one time loop
    io/
        writers.py # Writer protocol, TotalCsvWriter, RankCsvWriter, VtkWriter
        csv.py # CSV formatting and the atomic write
        storage.py # picks the run folder under ~/Documents/ExaFlow
    gui/
        main_window.py # controls, run process, output watcher
        viewer.py # PyVista 3D view
        sim_parameters_dialog.py # builds an input XML from the form
        streaming/ # sends fields from a running job to the viewer
examples/ # runnable drivers and an input XML
simulations/ # scaffold script for a new simulation directory
tests/ # pytest suite, including a rank-count independence check
packaging/ # PyInstaller spec and the desktop app build script
```

Start with `examples/template_3d_problem.py`. It is the whole driver: build a `Case`, hand it to a `Solver`, call `run`. The loop, the decomposition, the ghost exchange and the output schedule all live in the library.

Read `src/exaflow/numerics/README.md` before you touch a stencil. It states the array shapes, the ghost layer index rules, and the rules an operator has to follow.

## Key concepts

### Grid and ghost layers

The domain is a structured grid with uniform spacing. Each dimension has `n` real cells plus extra **ghost layers** padded around the edges. Ghost cells allow finite difference stencils to work at boundaries without going out of bounds.

```
ghost | real cells              | ghost
  [g]   [0]  [1]  [2]  ...  [n-1]   [g]
```

Ghost cells are filled with either MPI neighbor data or boundary condition values.

### Boundary conditions

Each of the 6 domain faces (left, right, top, bottom, front, back) can be independently set to:

| Type | What it does |
|------|-------------|
| `NO_SLIP` | Velocity = 0 at the wall (fluid sticks) |
| `SLIP` | Normal velocity = 0, tangential is free |
| `INFLOW` | Prescribed velocity entering the domain |
| `OUTFLOW` | Prescribed pressure, velocity extrapolated |
| `PERIODIC` | Wraps around (must be set on both opposing faces) |

### Time stepping

The timestep `dt` is automatically chosen as the smaller of:

- **Advection CFL**: `dt_adv = CFL * min(dx, dy, dz) / max_velocity`
- **Diffusion stability**: `dt_diff = 0.5 / (nu * (1/dx^2 + 1/dy^2 + 1/dz^2))`

### MPI parallelization

The domain is split across MPI processes in a process grid. Each process works on its own block and exchanges ghost layer data with its neighbors using non-blocking sends and receives.

`Subdomain` owns every index rule that follows from that split: the block bounds, the padded shape, the interior slice, the neighbor ranks, and `is_on_face`, which reports whether this rank owns a face of the **global** domain. A boundary condition or a one-sided stencil is correct only where `is_on_face` is true; elsewhere the ghost layer already holds the neighbor's data.

**The answer does not depend on the rank count.** A run at 1, 2, 4 or 8 ranks writes byte-identical output. `tests/test_solver.py` checks this, and it is the first property to look at after touching an operator or the exchange.

## Running simulations

### From a Python script

Build a `Case` and give it to a `Solver`:

```python
from mpi4py import MPI
from exaflow.config import Case, Fluid, Grid, TimeControl
from exaflow.io.storage import create_run_directory
from exaflow.solver import Solver

comm = MPI.COMM_WORLD
case = Case(
    fluid=Fluid(rho=1.225, nu=0.3),
    grid=Grid(shape=(50, 50, 50), extent=(1.0, 1.0, 1.0), num_ghost_layers=1),
    time=TimeControl(num_steps=50, cfl=0.25, integration_order=1),
)
Solver(case, comm, output_directory=create_run_directory("my_case", comm)).run()
```

A `Case` is frozen, so it can be built once and compared. It holds no communicator and no output directory: those belong to a run, and the `Solver` takes them.

`create_run_directory` is a collective call. Rank 0 picks the folder name and broadcasts it, so every rank has to call it. A call on rank 0 alone makes the other ranks wait forever.

### From an XML file

```bash
uv run mpiexec -n 4 python examples/run_from_xml.py --input-xml examples/input_template.xml
```

`examples/input_template.xml` is a 100x100x50 domain for 1000 steps, so this run takes many minutes. Lower `nt` in the file for a quick check.

### As a new simulation directory

```bash
uv run python simulations/create_simulation_directories.py simulations/my_run
uv run mpiexec -n 4 python simulations/my_run/build/run_simulation.py
```

The scaffold writes `build/in/input.xml` and `build/run_simulation.py`. The run script reads the `input.xml` beside it, so the target directory can sit anywhere.

## Developing

The editable install means an edit under `src/` takes effect on the next run. You do not reinstall, and you do not set `PYTHONPATH`.

### Add a dependency

```bash
uv add <package>
```

This writes the package into `pyproject.toml`, resolves it into `uv.lock`, and installs it. The project keeps one flat dependency list, with no optional extras and no separate development group.

### Check types

```bash
uv run mypy src
uv run mypy examples packaging run_gui.py simulations
```

Both report no issues today. Keep it that way. When a new error appears, fix the cause rather than silencing it: correct a loose annotation first, use `typing.cast` when a dimension branch already fixed the type, and raise on a value that arrives from outside. Keep `# type: ignore[<code>]` for a third-party stub that is wrong or missing, always with the exact code the checker prints. `src/exaflow/gui/viewer.py` shows that last case, because pyvista and vtk ship no usable stubs for those calls.

### Verify a change

```bash
uv run pytest
```

The suite runs in about a second. It covers the configuration values and the XML round trip, the block decomposition, the operators against analytic solutions, the convergence order of each Runge-Kutta scheme, and the rank-count independence of a whole run under `mpiexec`.

For a numerical change, also compare output against a run made before it:

```bash
EXAFLOW_OUTPUT_ROOT=/tmp/before uv run mpiexec -n 4 python examples/template_3d_problem.py
# make the change
EXAFLOW_OUTPUT_ROOT=/tmp/after uv run mpiexec -n 4 python examples/template_3d_problem.py
cmp /tmp/before/*/Final_Total.csv /tmp/after/*/Final_Total.csv
```

A refactor that should not change the numbers gives byte-identical files. Run it at more than one rank count, because a decomposition fault only shows up in parallel.

## The desktop app

`packaging/build_app.py` builds a standalone `ExaFlow.app` that runs without this repository, without Python and without a separate MPI installation:

```bash
uv run python packaging/build_app.py
rm -rf /Applications/ExaFlow.app.new && cp -R dist/ExaFlow.app /Applications/ExaFlow.app.new && rm -rf /Applications/ExaFlow.app && mv /Applications/ExaFlow.app.new /Applications/ExaFlow.app
```

The build takes a few minutes and leaves 1.3 GB in `dist/`: the 568 MB `ExaFlow.app` bundle, and the 798 MB `ExaFlow` directory that PyInstaller copied it from. It does five things: it draws `packaging/ExaFlow.icns`, it runs PyInstaller with `packaging/ExaFlow.spec`, it copies Open MPI from `.venv` into `Contents/Resources/mpi`, it removes the local symbols from every `.dylib` and `.so` file in the bundle and signs the bundle again, and it starts the bundled app once to prove that scipy, VTK, PySide6 and MPI still import.

The build prints that second command when it finishes. It copies to a new name first and removes the old bundle only after that copy succeeds, so a copy that fails leaves the app that works in `/Applications`. A bare `cp -R` writes into the bundle that is already there, so the files of the older build stay and `codesign --verify` then rejects the whole app.

Two details keep MPI working inside the bundle:

- `packaging/entry.py` sets `OPAL_PREFIX`, `PRTE_PREFIX` and `MPI4PY_LIBMPI` to the copied Open MPI before anything imports mpi4py. It also removes `MPI4PY_MPIABI` from the environment, because mpi4py takes the ABI name from that variable and then dlopens no `MPI4PY_LIBMPI` at all.
- A frozen bundle holds no separate Python interpreter, so each rank re-runs the app itself: `mpirun -np 4 ExaFlow --run-script <script>`. `main_window.py` adds that argument when `sys.frozen` is set.

Pass `--keep-icon` to skip drawing the icon again.

## Current status

**Working:**
- Convection and diffusion operators, written for 1D, 2D and 3D from one stencil each
- Explicit RK time stepping (orders 1-3), each verified at its design order
- MPI domain decomposition and ghost exchange, verified rank-count independent
- All boundary condition types (except time-dependent)
- CSV and VTK output
- A pytest suite, and `uv run mypy src` reporting no issues

**In progress:**
- Pressure projection (Poisson solver for incompressibility)

Without pressure, the solver evolves momentum but does not enforce that the velocity field is divergence-free. `SolverOptions` raises `NotImplementedError` when you set `include_pressure=True`, so leave it at the default `False`. `pressure_poisson.py` builds a Laplacian over one rank's block and solves it with conjugate gradient; it has no ghost exchange, so it is correct on a single rank only and is not wired into the time loop.
