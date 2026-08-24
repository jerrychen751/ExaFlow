from __future__ import annotations

import numpy as np
import mpi4py.MPI as mpi

from exaflow.boundary_conditions import BoundaryCondition
from exaflow.boundary_application import initialize_bc
from exaflow.mpi.domain import parallelize_domain, rejoin_array
from exaflow.mpi.ghost_layers import add_ghost_layers, remove_ghost_layers
from exaflow.numerics.time_step import advance_time_step
from exaflow.parameters import SimulationParameters
from exaflow.io.csv import write_csv, write_total_array_to_csv


def main() -> None:
    # Initialize MPI World
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()

    # Properties of air (example values)
    rho = 1.225
    nu = 0.3

    # Domain resolution
    nx, ny, nz = 50, 50, 50

    # Physical size
    length, width, height = 1.0, 1.0, 1.0

    # Runtime parameters
    num_ghost_layers = 1
    cfl = 0.25
    nt = 50

    # Boundary conditions (enums only)
    sim_params = SimulationParameters(
        rho=rho,
        nu=nu,
        domain=(nx, ny, nz),
        size=(length, width, height),
        nt=nt,
        num_ghost_layers=num_ghost_layers,
        cfl=cfl,
        comm=comm,
        left_wall=BoundaryCondition.INFLOW,
        left_inflow={"u": 2.0, "v": 1.0, "w": 1.0},
        right_wall=BoundaryCondition.NO_SLIP,
        top_wall=BoundaryCondition.NO_SLIP,
        bottom_wall=BoundaryCondition.NO_SLIP,
        front_wall=BoundaryCondition.NO_SLIP,
        back_wall=BoundaryCondition.NO_SLIP,
        include_convection=True,
        include_diffusion=True,
        include_pressure=False,
        time_integration_order=1,
        total_csv_frequency=-1,
        partial_csv_frequency=-1,
        vtk_frequency=-1,
    )

    # Initialize full-domain arrays
    u = np.ones(sim_params.domain, dtype=float)
    v = np.ones(sim_params.domain, dtype=float)
    w = np.ones(sim_params.domain, dtype=float)
    p = np.zeros(sim_params.domain, dtype=float)

    max_vel = float(np.max([np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w))]))
    sim_params.compute_dt(max_vel)

    u_original = u.copy()
    v_original = v.copy()
    w_original = w.copy()
    p_original = p.copy()

    # Split domain across ranks
    u_local, rank_coords = parallelize_domain(u, rank, sim_params)
    v_local, _ = parallelize_domain(v, rank, sim_params)
    w_local, _ = parallelize_domain(w, rank, sim_params)
    p_local, _ = parallelize_domain(p, rank, sim_params)

    # Add ghost layers
    u_local = add_ghost_layers(u_local, sim_params.num_ghost_layers)
    v_local = add_ghost_layers(v_local, sim_params.num_ghost_layers)
    w_local = add_ghost_layers(w_local, sim_params.num_ghost_layers)
    p_local = add_ghost_layers(p_local, sim_params.num_ghost_layers)

    initialize_bc(rank_coords, sim_params, u=u_local, v=v_local, w=w_local, p=p_local)
    comm.Barrier()

    # Allocate stage buffers
    u_new = u_local.copy()
    v_new = v_local.copy()
    w_new = w_local.copy()
    p_new = p_local.copy()

    u_old = u_local.copy()
    v_old = v_local.copy()
    w_old = w_local.copy()
    p_old = p_local.copy()

    k_u = np.empty_like(u_local)
    k_v = np.empty_like(v_local)
    k_w = np.empty_like(w_local)
    k_p = np.empty_like(p_local)

    for i in range(sim_params.nt):
        u_new, v_new, w_new, p_new = advance_time_step(
            sim_params=sim_params,
            rank_coords=rank_coords,
            p_old=p_old,
            u_old=u_old,
            v_old=v_old,
            w_old=w_old,
            p_new=p_new,
            u_new=u_new,
            v_new=v_new,
            w_new=w_new,
            k_u=k_u,
            k_v=k_v,
            k_w=k_w,
            k_p=k_p,
        )

        # Swap buffers
        u_new, u_old = u_old, u_new
        v_new, v_old = v_old, v_new
        w_new, w_old = w_old, w_new
        p_new, p_old = p_old, p_new

        if rank == 0:
            print(i, flush=True)
        comm.Barrier()

    # Remove ghost layers for final rejoin
    u_local = remove_ghost_layers(u_old, sim_params.num_ghost_layers)
    v_local = remove_ghost_layers(v_old, sim_params.num_ghost_layers)
    w_local = remove_ghost_layers(w_old, sim_params.num_ghost_layers)
    p_local = remove_ghost_layers(p_old, sim_params.num_ghost_layers)

    comm.Barrier()

    # Rejoin arrays on all ranks
    u_final = u_original.copy()
    v_final = v_original.copy()
    w_final = w_original.copy()
    p_final = p_original.copy()
    rejoin_array(u_final, u_local, sim_params)
    rejoin_array(v_final, v_local, sim_params)
    rejoin_array(w_final, w_local, sim_params)
    rejoin_array(p_final, p_local, sim_params)

    if rank == 0:
        write_total_array_to_csv("Original", p=p_original, u=u_original, v=v_original, w=w_original)
        write_total_array_to_csv("Final", p=p_final, u=u_final, v=v_final, w=w_final)

    write_csv(rank_coords, sim_params, "Final", u=u_old, v=v_old, w=w_old, p=p_old, with_ghosts=False)


if __name__ == "__main__":
    main()

