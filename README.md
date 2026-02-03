# Python 3D Navier-Stokes Solver

A Python solver for the incompressible Navier-Stokes equations in 1D, 2D, and 3D with MPI parallelization.

## What does this solver do?

It simulates how fluids (like air or water) move through space over time. The Navier-Stokes equations describe fluid motion through three physical effects:

- **Convection** - fluid carries momentum along with its flow
- **Diffusion** - viscosity smooths out velocity differences (think honey vs. water)
- **Pressure** - enforces that the fluid can't compress or expand (incompressibility)

The solver discretizes a physical domain into a grid, then marches forward in time using explicit Runge-Kutta integration (orders 1-3).

## Project structure

```
src/navier_stokes_solver/
    boundary_application.py   # Applies wall, inflow, outflow, periodic BCs
    boundary_conditions.py    # BoundaryCondition enum
    parameters.py             # SimulationParameters config + grid/timestep math
    initial_conditions.py     # Set up starting velocity/pressure fields
    numerics/
        time_step.py          # RK time integration + spatial operator
        convection_3d.py      # Advection stencils (u dot grad u)
        diffusion_3d.py       # Viscous stencils (nu * laplacian u)
        pressure_poisson.py   # Pressure Poisson solver (in progress)
    mpi/
        domain.py             # Domain decomposition across MPI ranks
        ghost_layers.py       # Ghost cell padding + MPI exchange
    io/
        csv.py                # CSV output
        vtk.py                # VTK output (via pyevtk)
    gui/                      # Optional desktop GUI for visualization
examples/                     # Ready-to-run simulation scripts
tests/                        # Unit tests
tests_mpi/                    # MPI integration tests
```

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

The domain is split across MPI processes in a Cartesian grid. Each process works on its local subdomain and exchanges ghost layer data with neighbors using non-blocking MPI sends/receives.

## Current status

**Working:**
- Convection and diffusion operators (1D/2D/3D)
- Explicit RK time stepping (orders 1-3)
- MPI domain decomposition + ghost exchange
- All boundary condition types (except time-dependent)
- CSV and VTK output

**In progress:**
- Pressure projection (Poisson solver for incompressibility)

Without pressure, the solver evolves momentum but does not enforce that the velocity field is divergence-free. Set `include_pressure=False` (the default) until the pressure solver is complete.

## Quick start

### Install

```bash
pip install -e ".[dev]"       # core + dev dependencies
pip install -e ".[gui]"       # add GUI dependencies
pip install -e ".[io]"        # add VTK output (pyevtk)
```

### Run an example

```bash
# Single process
PYTHONPATH=src python examples/template_3d_problem.py

# With MPI (4 processes)
PYTHONPATH=src mpiexec -n 4 python examples/template_3d_problem.py

# From an XML configuration file
PYTHONPATH=src mpiexec -n 4 python examples/run_from_xml.py --input-xml examples/input_template.xml
```

### Run the GUI

```bash
python run_gui.py
```

### Run tests

```bash
# Unit tests
PYTHONPATH=src python -m unittest -v

# MPI tests (macOS may need slot mapping)
PYTHONPATH=src mpiexec --host localhost:2 --map-by slot:OVERSUBSCRIBE -n 2 \
  python -m unittest -v tests_mpi.test_periodic_exchange
```

## Creating a new simulation

```bash
python Simulations/createNewSimulationDirectories.py Simulations/MyRun
cd Simulations/MyRun/Build
mpiexec -n 4 python runSimulation.py
```

This creates a directory with a template XML config and run script.
