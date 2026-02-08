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

from ..parameters import SimulationParameters, RankCoords
from ..boundary_conditions import BoundaryCondition


class PoissonSolver:

    def __init__(
        self,
        sim_params: SimulationParameters,
        rank_coords: RankCoords
    ) -> None:
        # Destructure simulation parameters
        self.dx = sim_params.dx
        self.dy = sim_params.dy
        self.dz = sim_params.dz
        self.ng = sim_params.num_ghost_layers
        self.rho = sim_params.rho
        self.ndims = sim_params.dimension
        self.interior_shape = sim_params.domain

        # Build sparse laplacian matrix based on self.ndims
        self._laplacian = self._build_laplacian()

    def _compute_divergence(
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
            Array of divergence values at interior grid points.
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
        Builds the LHS of ∇²p = (ρ/dt) · ∇·u*.

        We need to convert the PDE operator ∇²p into a matrix equation Lp = RHS where L is a sparse matrix representing the central differences, p is the pressure flattened into a vector, and RHS is (ρ/dt) · ∇·u*.

        Essentially, we return the matrix that is used in matrix-vector multiplication to apply central differences to each pressure point.
        """

        if self.ndims == 1:
            if not self.dx:
                raise ValueError("Simulation parameters must contain dx")

            (nx,) = self.interior_shape
            L = sparse.diags(
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(nx, nx),
                format='csr'
            ) / (self.dx ** 2)
            return L.tocsr()

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

            Dxx = sparse.diags( # ∂²/∂x²
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(nx, nx),
                format='csr'
            ) / (self.dx ** 2)
            Dyy = sparse.diags( # ∂²/∂y²
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(ny, ny),
                format='csr'
            ) / (self.dy ** 2)
            
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

            Dxx = sparse.diags( # ∂²/∂x²
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(nx, nx),
                format='csr'
            ) / (self.dx ** 2)
            Dyy = sparse.diags( # ∂²/∂y²
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(ny, ny),
                format='csr'
            ) / (self.dy ** 2)
            Dzz = sparse.diags( # ∂²/∂z²
                diagonals=[1, -2, 1],
                offsets=[-1, 0, 1],
                shape=(nz, nz),
                format='csr'
            ) / (self.dz ** 2)

            Ix = sparse.eye(nx, format='csr')
            Iy = sparse.eye(ny, format='csr')
            Iz = sparse.eye(nz, format='csr')

            return (
                sparse.kron(sparse.kron(Dxx, Iy, format='csr'), Iz, format='csr') +
                sparse.kron(sparse.kron(Ix, Dyy, format='csr'), Iz, format='csr') +
                sparse.kron(sparse.kron(Ix, Iy, format='csr'), Dzz, format='csr')
            )

        raise ValueError(f"Unsupported number of dimensions: {self.ndims}")

    def solve(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Solve ∇²p = (ρ/dt) · ∇·u* for the pressure field.

        Args:
            u, v, w: Predicted velocity fields (with ghost layers).
            dt: Current timestep size.

        Returns:
            Pressure field at interior grid points (no ghost layers), shaped like self.interior_shape.
        """
        # Step 1: Compute RHS = (ρ/dt) · ∇·u*
        div = self._compute_divergence(u, v, w)
        rhs = (self.rho / dt) * div

        # Step 2: Flatten to 1D vector (row-major to match Kronecker product ordering)
        rhs_flat = rhs.ravel()

        # Step 3: Pin one pressure value to remove nullspace singularity.
        # The pure Neumann Laplacian is singular (pressure is only defined up to a constant).
        # Setting rhs[0] = 0 with the corresponding row zeroed + diagonal=1 fixes p[0] = 0.
        rhs_flat[0] = 0.0
        L = self._laplacian.tolil()
        L[0, :] = 0
        L[0, 0] = 1.0
        L = L.tocsr()

        # Step 4: Solve with conjugate gradient
        p_flat, info = cg(L, rhs_flat)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient solver did not converge (info={info})")

        # Step 5: Reshape back to grid dimensions
        return p_flat.reshape(self.interior_shape)

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

        Modifies u, v, w in-place at interior points.
        The pressure gradient is computed with central differences: ∂p/∂x ≈ (p[i+1] - p[i-1]) / (2·dx).

        Args:
            u, v, w: Predicted velocity fields (with ghost layers). Modified in-place.
            p: Pressure field at interior points (from solve()), no ghost layers.
            dt: Current timestep size.
        """
        coeff = dt / self.rho
        ng = self.ng

        if self.ndims >= 1 and self.dx:
            # ∂p/∂x via central differences
            dp_dx = (p[2:, ...] - p[:-2, ...]) / (2 * self.dx)
            u_slice = tuple(slice(ng + 1, -(ng + 1)) if i == 0 else slice(ng, -ng) for i in range(self.ndims))
            u[u_slice] -= coeff * dp_dx

        if self.ndims >= 2 and self.dy:
            # ∂p/∂y via central differences
            dp_dy = (p[:, 2:, ...] - p[:, :-2, ...]) / (2 * self.dy)
            v_slice = tuple(slice(ng + 1, -(ng + 1)) if i == 1 else slice(ng, -ng) for i in range(self.ndims))
            v[v_slice] -= coeff * dp_dy

        if self.ndims >= 3 and self.dz:
            # ∂p/∂z via central differences
            dp_dz = (p[:, :, 2:] - p[:, :, :-2]) / (2 * self.dz)
            w_slice = tuple(slice(ng + 1, -(ng + 1)) if i == 2 else slice(ng, -ng) for i in range(self.ndims))
            w[w_slice] -= coeff * dp_dz