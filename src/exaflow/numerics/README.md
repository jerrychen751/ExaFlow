# Numerics

This package contains the finite difference operators and time integration for the solver.

## Array conventions

The solver uses four field arrays: `u`, `v`, `w` (velocity) and `p` (pressure). All are `np.ndarray`.

| Array | Physical meaning |
|-------|-----------------|
| `u` | Velocity in the x-direction |
| `v` | Velocity in the y-direction |
| `w` | Velocity in the z-direction |
| `p` | Pressure |

### Indexing: axis 0 = x, axis 1 = y, axis 2 = z

```
u[i, j, k]
  ^  ^  ^
  x  y  z
```

- `u[3, 7, 2]` is the x-velocity at grid point (x=3, y=7, z=2).
- `u[:, 0, :]` is a slice of x-velocities along the entire x-z plane at y=0.

### Ghost layers

Arrays include ghost cells padded around the real domain. With `ng` ghost layers:

```
axis length = ng + n_real + ng
interior    = array[ng:-ng, ...]    (strips ghost cells)
```

For example, a 50x50x50 domain with `ng=1` gives arrays of shape `(52, 52, 52)`.

### Dimensions

- **1D**: arrays have shape `(nx,)`, only `u` is active
- **2D**: arrays have shape `(nx, ny)`, `u` and `v` are active
- **3D**: arrays have shape `(nx, ny, nz)`, all three velocity components are active

Inactive velocity arrays still exist but remain zero.

## Modules

- `time_step.py` - Runge-Kutta time integration (orders 1-3) and the spatial operator that orchestrates convection, diffusion, and ghost exchanges
- `convection_3d.py` - Explicit advection operator (central and upwind schemes)
- `diffusion_3d.py` - Explicit viscous operator (central difference Laplacian)
- `pressure_poisson.py` - Pressure Poisson solver (in progress)
