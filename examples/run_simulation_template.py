from __future__ import annotations

"""
Template runner intended to be copied into per-simulation Build/ directories.

This script assumes it lives at:
    <repo>/Simulations/<simulation_name>/Build/runSimulation.py

and therefore adds `<repo>/src` to `sys.path` to import the solver package
without requiring a global installation.
"""

import argparse
import sys
from pathlib import Path

import mpi4py.MPI as mpi


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    sys.path.insert(0, str(src_root))


def main() -> None:
    _ensure_repo_src_on_path()

    from navier_stokes_solver.parameters import simulation_parameters_from_xml
    from navier_stokes_solver.initial_conditions import initialize_fields
    from navier_stokes_solver.boundary_application import initialize_bc
    from navier_stokes_solver.mpi.domain import parallelize_domain, rejoin_array
    from navier_stokes_solver.mpi.ghost_layers import add_ghost_layers, remove_ghost_layers
    from navier_stokes_solver.numerics.time_step import advance_time_step
    from navier_stokes_solver.io.csv import write_total_array_to_csv

    import numpy as np

    parser = argparse.ArgumentParser(description="Run a simulation from Build/in/Input.xml")
    parser.add_argument(
        "--input-xml",
        default=str(Path(__file__).resolve().parent / "in" / "Input.xml"),
        help="Path to input XML (default: Build/in/Input.xml).",
    )
    args, _ = parser.parse_known_args()

    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()

    sim_params = simulation_parameters_from_xml(args.input_xml, comm)
    if sim_params.dimension != 3:
        raise ValueError("This template runner currently expects 3D inputs.")

    p, u, v, w = initialize_fields(sim_params)  # type: ignore[misc]

    max_vel = float(np.max([np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w))]))
    sim_params.compute_dt(max_vel)

    u_original = u.copy()
    v_original = v.copy()
    w_original = w.copy()
    p_original = p.copy()

    u_local, rank_coords = parallelize_domain(u, rank, sim_params)
    v_local, _ = parallelize_domain(v, rank, sim_params)
    w_local, _ = parallelize_domain(w, rank, sim_params)
    p_local, _ = parallelize_domain(p, rank, sim_params)

    u_local = add_ghost_layers(u_local, sim_params.num_ghost_layers)
    v_local = add_ghost_layers(v_local, sim_params.num_ghost_layers)
    w_local = add_ghost_layers(w_local, sim_params.num_ghost_layers)
    p_local = add_ghost_layers(p_local, sim_params.num_ghost_layers)

    initialize_bc(rank_coords, sim_params, u=u_local, v=v_local, w=w_local, p=p_local)
    comm.Barrier()

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

    for _ in range(sim_params.nt):
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

        u_new, u_old = u_old, u_new
        v_new, v_old = v_old, v_new
        w_new, w_old = w_old, w_new
        p_new, p_old = p_old, p_new
        comm.Barrier()

    u_local = remove_ghost_layers(u_old, sim_params.num_ghost_layers)
    v_local = remove_ghost_layers(v_old, sim_params.num_ghost_layers)
    w_local = remove_ghost_layers(w_old, sim_params.num_ghost_layers)
    p_local = remove_ghost_layers(p_old, sim_params.num_ghost_layers)
    comm.Barrier()

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


if __name__ == "__main__":
    main()

