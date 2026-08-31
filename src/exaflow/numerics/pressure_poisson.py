"""
For incompressible flow, fluid density is constant everywhere, so whatever flows into our box must flow out.
    - In other words, the net divergence is exactly 0 (grad u = 0).

This module implements Chorin's Projection Method, splitting each time step into 3 sub-steps:
    1. Predict a velocity u* using only convection/diffusion.
    2. Solve for a pressure field so that correcting u* using pressure results in gradient being 0.
    3. Fix the velocity.

The corrected velocity is: u = u* - (dt/ρ)∇p
Taking the divergence of both sides: ∇·u = ∇·u* - (dt/ρ)∇²p = 0
Rearranging: ∇²p = (ρ/dt) · ∇·u*


"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from ..config.case import Case
from ..mpi.subdomain import Subdomain


class PoissonSolver:

    def __init__(
        self,
        case: Case,
        subdomain: Subdomain,
    ) -> None:
        """
        Build the discrete Laplacian for the block `subdomain` owns.

        This solver is correct on a single rank only. It factors one dense-structured Laplacian over its own block and never exchanges with a neighbor, so on several ranks each block would be solved as if it were the whole domain. It is not wired into the time loop; `SolverOptions` rejects include_pressure until it is.
        """

        # Destructure simulation parameters
        spacing = case.grid.spacing
        self.dx = spacing[0]
        self.dy = spacing[1] if len(spacing) > 1 else None
        self.dz = spacing[2] if len(spacing) > 2 else None
        self.ng = case.grid.num_ghost_layers
        self.rho = case.fluid.rho
        self.ndims = case.dimension
        self.interior_shape = subdomain.shape

        # Build sparse laplacian matrix based on self.ndims
        laplacian = self._build_laplacian()

        # The pure Neumann Laplacian is singular (pressure is only defined up to a constant).
        self._operator = -laplacian

        self._p_flat_prev: np.ndarray | None = None

    def compute_divergence(
            self,
            u: np.ndarray,
            v: np.ndarray,
            w: np.ndarray,
        ) -> np.ndarray:
        """
        Compute the divergence of the velocity field using second-order central differencing.

        Uses the stencil (u[i+1] - u[i-1]) / (2 * dx) for each direction.
        Returns ∇·u*, which is a term in the RHS of the pressure Poisson equation.

        Args:
            u: Velocity field in the x-direction (includes ghost layers).
            v: Velocity field in the y-direction (includes ghost layers).
            w: Velocity field in the z-direction (includes ghost layers).

        Returns:
            Array of divergence values at interior grid points. The stencil reaches one point beyond each end, so the value at an outermost real point reads a ghost layer and is only as good as whatever last filled it.
        """
        # Need a None guard here because if self.ng == 1, hi = 0 which can result in an empty slice as it's the right bound
        hi = -self.ng + 1 if self.ng > 1 else None # right boundary of "shifted forward" slice
        lo = -(self.ng + 1) # right boundary of "shifted backward" slice

        if self.ndims == 1:
            if not self.dx:
                raise ValueError("Simulation parameters must contain dx")

            du_dx = (u[self.ng+1:hi] - u[self.ng-1:lo]) / (2 * self.dx)
            return du_dx
        elif self.ndims == 2:
            if not self.dx or not self.dy:
                raise ValueError("Simulation parameters must contain dx and dy")

            du_dx = (u[self.ng+1:hi, self.ng:-self.ng] - u[self.ng-1:lo, self.ng:-self.ng]) / (2 * self.dx)
            dv_dy = (v[self.ng:-self.ng, self.ng+1:hi] - v[self.ng:-self.ng, self.ng-1:lo]) / (2 * self.dy)
            return du_dx + dv_dy
        else:
            if not self.dx or not self.dy or not self.dz:
                raise ValueError("Simulation parameteres must contain dx, dy, and dz")
            
            du_dx = (u[self.ng+1:hi, self.ng:-self.ng, self.ng:-self.ng] - u[self.ng-1:lo, self.ng:-self.ng, self.ng:-self.ng]) / (2 * self.dx)
            dv_dy = (v[self.ng:-self.ng, self.ng+1:hi, self.ng:-self.ng] - v[self.ng:-self.ng, self.ng-1:lo, self.ng:-self.ng]) / (2 * self.dy)
            dw_dz = (w[self.ng:-self.ng, self.ng:-self.ng, self.ng+1:hi] - w[self.ng:-self.ng, self.ng:-self.ng, self.ng-1:lo]) / (2 * self.dz)
            return du_dx + dv_dy + dw_dz
        
    # The discrete Laplacian operator ∇²p = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²
    # Approximated via central differences: ∂²p/∂x² ≈ (p[i+1] - 2p[i] + p[i-1]) / dx²
    def _build_laplacian(self) -> sparse.csr_matrix:
        """
        Builds the LHS of ∇²p = (ρ/dt) · ∇·u*. We need to convert the PDE operator ∇²p into a matrix equation Lp = RHS where L is a sparse matrix representing the central differences, p is the pressure flattened into a vector, and RHS is (ρ/dt) · ∇·u*. Essentially, we return the matrix that is used in matrix-vector multiplication to apply central differences to each pressure point.
        """

        if self.ndims == 1:
            if not self.dx:
                raise ValueError("Simulation parameters must contain dx")

            (nx,) = self.interior_shape
            return self._build_axis_laplacian(nx, self.dx)

        elif self.ndims == 2:
            # 2D pressure grid -> 1D vector via row-major flattening (row by row ordering)
            # 2D Laplacian for a point is ∇²p₁₁ = p₀₁ + p₂₁ + p₁₀ + p₁₂ − 4·p₁₁
                # 4 Neighbors (l, r, t, b) - 4*self
            # We achieve this through matrix-vector multiplication using Kronecker products

            # D_xx ⊗ I_y: applies ∂²/∂x², connects points ny apart in the flat vector (x-neighbors)
            # I_x ⊗ D_yy: applies ∂²/∂y², connects adjacent entries in the flat vector (y-neighbors)
            if not self.dx or not self.dy:
                raise ValueError("Simulation parameters must contain dx and dy")
            
            (nx, ny) = self.interior_shape

            Dxx = self._build_axis_laplacian(nx, self.dx) # ∂²/∂x²
            Dyy = self._build_axis_laplacian(ny, self.dy) # ∂²/∂y²
            
            Ix = sparse.eye(nx, format='csr')
            Iy = sparse.eye(ny, format='csr')
            return sparse.kron(Dxx, Iy, format='csr') + sparse.kron(Ix, Dyy, format='csr')

        elif self.ndims == 3:
            # 3D pressure grid -> 1D vector via row-major flattening (x slowest, z fastest)
            # 3D Laplacian for a point is ∇²p = p_l + p_r + p_t + p_b + p_f + p_k − 6·p_c
            #     6 neighbors (left, right, top, bottom, front, back) - 6*self
            #
            # D_xx ⊗ I_y ⊗ I_z: applies ∂²/∂x², connects points ny*nz apart (x-neighbors)
            # I_x ⊗ D_yy ⊗ I_z: applies ∂²/∂y², connects points nz apart (y-neighbors)
            # I_x ⊗ I_y ⊗ D_zz: applies ∂²/∂z², connects adjacent entries (z-neighbors)
            if not self.dx or not self.dy or not self.dz:
                raise ValueError("Simulation parameters must contain dx, dy, and dz")

            (nx, ny, nz) = self.interior_shape

            Dxx = self._build_axis_laplacian(nx, self.dx) # ∂²/∂x²
            Dyy = self._build_axis_laplacian(ny, self.dy) # ∂²/∂y²
            Dzz = self._build_axis_laplacian(nz, self.dz) # ∂²/∂z²

            Ix = sparse.eye(nx, format='csr')
            Iy = sparse.eye(ny, format='csr')
            Iz = sparse.eye(nz, format='csr')

            return (
                sparse.kron(sparse.kron(Dxx, Iy, format='csr'), Iz, format='csr') +
                sparse.kron(sparse.kron(Ix, Dyy, format='csr'), Iz, format='csr') +
                sparse.kron(sparse.kron(Ix, Iy, format='csr'), Dzz, format='csr')
            )

        raise ValueError(f"Unsupported number of dimensions: {self.ndims}")

    def _build_axis_laplacian(self, points: int, spacing: float) -> sparse.csr_matrix:
        """
        The second-difference operator along one axis, with a zero normal pressure gradient at each end. A ghost value equal to the edge value turns the two end rows into -1 instead of -2, so every row sums to zero and the operator is the pure Neumann one. The matrix stays symmetric, which conjugate gradient needs.
        """

        band = sparse.diags(
            diagonals=[1.0, -2.0, 1.0],
            offsets=[-1, 0, 1],
            shape=(points, points),
            format='lil'
        )
        band[0, 0] = -1.0
        band[points - 1, points - 1] = -1.0
        return band.tocsr() / (spacing ** 2)

    def solve(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Solve ∇²p = (ρ/dt) · ∇·u* for the pressure field.

        The operator carries a zero normal gradient on every face, so it is singular: it fixes the pressure only up to a constant. This removes the mean of the right-hand side, which is the condition such a problem must satisfy to have a solution at all, and returns the one answer whose own mean is zero.

        Args:
            u, v, w: Predicted velocity fields (with ghost layers). Read, never modified.
            dt: Current timestep size.

        Returns:
            Pressure field at interior grid points (no ghost layers), shaped like self.interior_shape, with a mean of zero.
        """
        # Step 1: Compute RHS = (ρ/dt) · ∇·u*
        div = self.compute_divergence(u, v, w)
        rhs = (self.rho / dt) * div

        # Step 2: Flatten to 1D vector (row-major to match Kronecker product ordering)
        rhs_flat = rhs.ravel()

        # Step 3: A pure Neumann problem is solvable only where the right-hand side sums to zero, so remove its mean.
        rhs_flat = rhs_flat - rhs_flat.mean()

        # Step 4: Solve with conjugate gradient
        p_flat, info = cg(self._operator, -rhs_flat, x0=self._p_flat_prev)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient solver did not converge (info={info})")
        self._p_flat_prev = p_flat.copy()

        # Step 5: Reshape back to grid dimensions
        return (p_flat - p_flat.mean()).reshape(self.interior_shape)

    def correct_velocity(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        dt: float,
    ) -> None:
        """
        Apply pressure correction: u = u* - (dt/ρ) · ∇p.

        Modifies u, v, w in-place at every interior point, the outermost real point included. The pressure gradient is a central difference, ∂p/∂x ≈ (p[i+1] - p[i-1]) / (2·dx), taken over a pressure array padded by one layer that repeats the edge value. That padding states the same zero normal gradient the Laplacian carries, so the end points need no separate rule.

        The ghost layers of u, v and w keep the predicted velocity. The divergence at the outermost real point reads one of them, so the caller refreshes the ghost layers through the ghost exchange and the boundary conditions before that point means anything.

        The compact Laplacian is not the exact divergence of this gradient, because both differences span 2h while the operator spans h. The divergence left behind therefore falls as h^2 at a point whose stencil reads corrected values only, rather than to zero.

        Args:
            u, v, w: Predicted velocity fields (with ghost layers). Modified in-place.
            p: Pressure field at interior points (from solve()), no ghost layers.
            dt: Current timestep size.
        """
        coeff = dt / self.rho
        interior = tuple(slice(self.ng, -self.ng) for _ in range(self.ndims))
        padded = np.pad(p, 1, mode="edge")  # (*interior_shape,) -> each axis + 2

        # ∂p/∂x_a via central differences over the padded pressure
        for axis, field in enumerate((u, v, w)[: self.ndims]):
            spacing = (self.dx, self.dy, self.dz)[axis]
            if not spacing:
                raise ValueError(f"Simulation parameters must contain the spacing of axis {axis}")
            lower = tuple(slice(0, -2) if i == axis else slice(1, -1) for i in range(self.ndims))
            upper = tuple(slice(2, None) if i == axis else slice(1, -1) for i in range(self.ndims))
            field[interior] -= coeff * (padded[upper] - padded[lower]) / (2 * spacing)
