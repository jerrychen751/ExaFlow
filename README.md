# Python 3D Navier–Stokes Solver (Refactored)

This repository contains a parallel (MPI) explicit time-stepping Navier–Stokes solver prototype and a companion desktop GUI for visualization.

The current refactor focuses on:
- making the codebase importable as a conventional Python package,
- fixing correctness bugs that materially affect results (grid spacing / timestep invariants, periodic ghost exchange),
- enforcing enums-only boundary conditions (no legacy string compatibility),
- adding a small but reliable test suite (unit + MPI smoke tests),
- documenting solver invariants clearly.

## Repo layout

- Core solver package: `src/navier_stokes_solver/`
- GUI package (optional deps): `src/navier_stokes_solver/gui/`
- Example scripts: `examples/`
- Tests: `tests/` and `tests/mpi/`

## Current capabilities (and what is *not* implemented)

Implemented:
- MPI domain decomposition in 1D/2D/3D and reassembly
- Ghost-layer exchange (including periodic wrap-around when the BCs are paired)
- Explicit convection and diffusion operators (legacy stencils preserved)
- Explicit RK time stepping orders 1–3 (order 4 is not implemented)

Not implemented (important):
- Pressure projection / Poisson solve for incompressibility

Because of that, you must run with `include_pressure=False`. The code raises a `NotImplementedError` if pressure is enabled so that “silent wrong physics” is avoided.

## Numerical invariants (documented and enforced)

Grid spacing is defined in physical units as:
- `dx = length / (nx - 1)` (and similarly `dy`, `dz`)

The timestep is chosen as the minimum of:
- advection CFL constraint:
  - `dt_adv = CFL * min(dx, dy, dz) / max_velocity_magnitude`
- explicit diffusion stability constraint:
  - `dt_diff = 0.5 / (nu * (1/dx^2 + 1/dy^2 + 1/dz^2))`

So:
- `dt = min(dt_adv, dt_diff)`

## Boundary conditions (enums only)

Boundary conditions are represented by `navier_stokes_solver.boundary_conditions.BoundaryCondition`.

This refactor intentionally does **not** accept legacy boundary-condition strings at runtime. XML parsing converts strings into enums, and invalid values raise a clear error.

## Quick start

### 1) Install (editable)

From the repo root:

```bash
pip install -e ".[dev]"
```

For GUI dependencies:

```bash
pip install -e ".[gui]"
```

For VTK output via pyevtk:

```bash
pip install -e ".[io]"
```

### 2) Run the GUI

```bash
python run_gui.py
```

### 3) Run an MPI example script

If you do not want to install the package, you can run scripts by pointing `PYTHONPATH` at `src/`:

```bash
PYTHONPATH=src mpiexec -n 4 python examples/template_3d_problem.py
```

To run from an XML file:

```bash
PYTHONPATH=src mpiexec -n 4 python examples/run_from_xml.py --input-xml examples/input_template.xml
```

## Simulation directory generator

The helper script `Simulations/createNewSimulationDirectories.py` creates a per-simulation directory with:
- `Build/in/Input.xml` (copied from `examples/input_template.xml`)
- `Build/runSimulation.py` (copied from `examples/run_simulation_template.py`)

Example:

```bash
python Simulations/createNewSimulationDirectories.py Simulations/MyRun
cd Simulations/MyRun/Build
mpiexec -n 4 python runSimulation.py
```

## Tests

Before running Python commands, activate your conda environment:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cfd
```

Unit tests (single-process, stdlib `unittest`):

```bash
PYTHONPATH=src python -m unittest -v
```

MPI smoke tests (OpenMPI on macOS often needs explicit slot mapping):

```bash
PYTHONPATH=src mpiexec --host localhost:2 --map-by slot:OVERSUBSCRIBE -n 2 \
  python -m unittest -v tests_mpi.test_periodic_exchange
```

## Issues / discussions / PRs

- Use Issues to track solver correctness tasks, missing features (especially pressure projection), and performance work.
- Use Discussions for design questions (numerical schemes, boundary semantics, GUI workflow).
- PRs should be focused and should include a short “what changed / why / how to verify” section.
