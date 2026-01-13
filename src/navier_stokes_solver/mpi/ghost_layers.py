from __future__ import annotations

from typing import Any

import numpy as np

from ..boundary_conditions import BoundaryCondition
from ..parameters import SimulationParameters


def add_ghost_layers(array: np.ndarray, num_ghost_layers: int, *, fill_value: float = 1.0) -> np.ndarray:
    dimension = len(array.shape)
    if dimension == 1:
        left = np.full((num_ghost_layers,), fill_value, dtype=array.dtype)
        right = np.full((num_ghost_layers,), fill_value, dtype=array.dtype)
        return np.concatenate([left, array, right], axis=0)
    if dimension == 2:
        ghost_r = np.full((num_ghost_layers, array.shape[1]), fill_value, dtype=array.dtype)
        ghost_c = np.full((array.shape[0] + 2 * num_ghost_layers, num_ghost_layers), fill_value, dtype=array.dtype)
        out = np.vstack([ghost_r, array, ghost_r])
        out = np.hstack([ghost_c, out, ghost_c])
        return out
    if dimension == 3:
        ghost_r = np.full((num_ghost_layers, array.shape[1], array.shape[2]), fill_value, dtype=array.dtype)
        ghost_c = np.full((array.shape[0] + 2 * num_ghost_layers, num_ghost_layers, array.shape[2]), fill_value, dtype=array.dtype)
        ghost_d = np.full(
            (array.shape[0] + 2 * num_ghost_layers, array.shape[1] + 2 * num_ghost_layers, num_ghost_layers),
            fill_value,
            dtype=array.dtype,
        )
        out = np.vstack([ghost_r, array, ghost_r])
        out = np.hstack([ghost_c, out, ghost_c])
        out = np.dstack([ghost_d, out, ghost_d])
        return out
    raise ValueError(f"Unsupported dimension {dimension}.")


def remove_ghost_layers(array: np.ndarray, num_ghost_layers: int) -> np.ndarray:
    dimension = len(array.shape)
    if dimension == 1:
        return array[num_ghost_layers:-num_ghost_layers]
    if dimension == 2:
        return array[num_ghost_layers:-num_ghost_layers, num_ghost_layers:-num_ghost_layers]
    if dimension == 3:
        return array[
            num_ghost_layers:-num_ghost_layers,
            num_ghost_layers:-num_ghost_layers,
            num_ghost_layers:-num_ghost_layers,
        ]
    raise ValueError(f"Unsupported dimension {dimension}.")


def send_ghost_info(array: np.ndarray, sim_params: SimulationParameters, rank_coords: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    comm = sim_params.comm
    if comm is None:
        raise ValueError("send_ghost_info requires an MPI communicator (sim_params.comm).")

    num_ghosts = sim_params.num_ghost_layers
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)

    request: dict[str, Any] = {}
    buffers: dict[str, Any] = {}

    if sim_params.dimension == 1:
        left_data = np.ascontiguousarray(array[num_ghosts : 2 * num_ghosts])
        right_data = np.ascontiguousarray(array[-2 * num_ghosts : -num_ghosts])
        recv_left = np.empty(num_ghosts, dtype=array.dtype)
        recv_right = np.empty(num_ghosts, dtype=array.dtype)

        rank_x = int(rank_coords)
        rank = int(rank_coords)

        req_left_send = req_left_recv = req_right_send = req_right_recv = None

        if rank_x > 0:
            req_left_send = comm.Isend(left_data, dest=rank - 1)
            req_left_recv = comm.Irecv(recv_left, source=rank - 1)
        if rank_x < num_procs_x - 1:
            req_right_send = comm.Isend(right_data, dest=rank + 1)
            req_right_recv = comm.Irecv(recv_right, source=rank + 1)

        if sim_params.left_wall == BoundaryCondition.PERIODIC:
            if rank_x == 0:
                req_left_send = comm.Isend(left_data, dest=num_procs_x - 1)
                req_left_recv = comm.Irecv(recv_left, source=num_procs_x - 1)
            if rank_x == num_procs_x - 1:
                req_right_send = comm.Isend(right_data, dest=0)
                req_right_recv = comm.Irecv(recv_right, source=0)

        request["left_send"] = req_left_send
        request["left_recv"] = req_left_recv
        request["right_send"] = req_right_send
        request["right_recv"] = req_right_recv
        buffers["from_left"] = recv_left
        buffers["from_right"] = recv_right
        return request, buffers

    if sim_params.dimension == 2:
        left_data = np.ascontiguousarray(array[num_ghosts : 2 * num_ghosts, :])
        right_data = np.ascontiguousarray(array[-2 * num_ghosts : -num_ghosts, :])
        top_data = np.ascontiguousarray(array[:, num_ghosts : 2 * num_ghosts])
        bottom_data = np.ascontiguousarray(array[:, -2 * num_ghosts : -num_ghosts])

        recv_left = np.empty((num_ghosts, array.shape[1]), dtype=array.dtype)
        recv_right = np.empty((num_ghosts, array.shape[1]), dtype=array.dtype)
        recv_top = np.empty((array.shape[0], num_ghosts), dtype=array.dtype)
        recv_bottom = np.empty((array.shape[0], num_ghosts), dtype=array.dtype)

        rank_x = int(rank_coords[0])
        rank_y = int(rank_coords[1])
        rank = int(rank_coords[2])

        req_left_send = req_left_recv = req_right_send = req_right_recv = None
        req_top_send = req_top_recv = req_bottom_send = req_bottom_recv = None

        if rank_x > 0:
            req_left_send = comm.Isend(left_data, dest=rank - 1)
            req_left_recv = comm.Irecv(recv_left, source=rank - 1)
        if rank_x < num_procs_x - 1:
            req_right_send = comm.Isend(right_data, dest=rank + 1)
            req_right_recv = comm.Irecv(recv_right, source=rank + 1)

        if rank_y > 0:
            req_top_send = comm.Isend(top_data, dest=rank - num_procs_x)
            req_top_recv = comm.Irecv(recv_top, source=rank - num_procs_x)
        if rank_y < num_procs_y - 1:
            req_bottom_send = comm.Isend(bottom_data, dest=rank + num_procs_x)
            req_bottom_recv = comm.Irecv(recv_bottom, source=rank + num_procs_x)

        if sim_params.left_wall == BoundaryCondition.PERIODIC:
            if rank_x == 0:
                req_left_send = comm.Isend(left_data, dest=rank + (num_procs_x - 1))
                req_left_recv = comm.Irecv(recv_left, source=rank + (num_procs_x - 1))
            if rank_x == num_procs_x - 1:
                req_right_send = comm.Isend(right_data, dest=rank - (num_procs_x - 1))
                req_right_recv = comm.Irecv(recv_right, source=rank - (num_procs_x - 1))

        if sim_params.top_wall == BoundaryCondition.PERIODIC:
            if rank_y == 0:
                req_top_send = comm.Isend(top_data, dest=rank + (num_procs_y - 1) * num_procs_x)
                req_top_recv = comm.Irecv(recv_top, source=rank + (num_procs_y - 1) * num_procs_x)
            if rank_y == num_procs_y - 1:
                req_bottom_send = comm.Isend(bottom_data, dest=rank - (num_procs_y - 1) * num_procs_x)
                req_bottom_recv = comm.Irecv(recv_bottom, source=rank - (num_procs_y - 1) * num_procs_x)

        request.update(
            {
                "left_send": req_left_send,
                "left_recv": req_left_recv,
                "right_send": req_right_send,
                "right_recv": req_right_recv,
                "top_send": req_top_send,
                "top_recv": req_top_recv,
                "bottom_send": req_bottom_send,
                "bottom_recv": req_bottom_recv,
            }
        )
        buffers.update(
            {
                "from_left": recv_left,
                "from_right": recv_right,
                "from_top": recv_top,
                "from_bottom": recv_bottom,
            }
        )
        return request, buffers

    if sim_params.dimension == 3:
        left_data = np.ascontiguousarray(array[num_ghosts : 2 * num_ghosts, :, :])
        right_data = np.ascontiguousarray(array[-2 * num_ghosts : -num_ghosts, :, :])
        top_data = np.ascontiguousarray(array[:, num_ghosts : 2 * num_ghosts, :])
        bottom_data = np.ascontiguousarray(array[:, -2 * num_ghosts : -num_ghosts, :])
        front_data = np.ascontiguousarray(array[:, :, num_ghosts : 2 * num_ghosts])
        back_data = np.ascontiguousarray(array[:, :, -2 * num_ghosts : -num_ghosts])

        recv_front = np.empty((array.shape[0], array.shape[1], num_ghosts), dtype=array.dtype, order="C")
        recv_back = np.empty((array.shape[0], array.shape[1], num_ghosts), dtype=array.dtype, order="C")
        recv_top = np.empty((array.shape[0], num_ghosts, array.shape[2]), dtype=array.dtype, order="C")
        recv_bottom = np.empty((array.shape[0], num_ghosts, array.shape[2]), dtype=array.dtype, order="C")
        recv_left = np.empty((num_ghosts, array.shape[1], array.shape[2]), dtype=array.dtype, order="C")
        recv_right = np.empty((num_ghosts, array.shape[1], array.shape[2]), dtype=array.dtype, order="C")

        rank_x = int(rank_coords[0])
        rank_y = int(rank_coords[1])
        rank_z = int(rank_coords[2])
        rank = int(rank_coords[3])

        req_left_send = req_left_recv = req_right_send = req_right_recv = None
        req_top_send = req_top_recv = req_bottom_send = req_bottom_recv = None
        req_front_send = req_front_recv = req_back_send = req_back_recv = None

        if rank_x > 0:
            req_left_send = comm.Isend(left_data, dest=rank - 1)
            req_left_recv = comm.Irecv(recv_left, source=rank - 1)
        if rank_x < num_procs_x - 1:
            req_right_send = comm.Isend(right_data, dest=rank + 1)
            req_right_recv = comm.Irecv(recv_right, source=rank + 1)

        if rank_y > 0:
            req_top_send = comm.Isend(top_data, dest=rank - num_procs_x)
            req_top_recv = comm.Irecv(recv_top, source=rank - num_procs_x)
        if rank_y < num_procs_y - 1:
            req_bottom_send = comm.Isend(bottom_data, dest=rank + num_procs_x)
            req_bottom_recv = comm.Irecv(recv_bottom, source=rank + num_procs_x)

        if rank_z > 0:
            req_front_send = comm.Isend(front_data, dest=rank - (num_procs_x * num_procs_y))
            req_front_recv = comm.Irecv(recv_front, source=rank - (num_procs_x * num_procs_y))
        if rank_z < num_procs_z - 1:
            req_back_send = comm.Isend(back_data, dest=rank + (num_procs_x * num_procs_y))
            req_back_recv = comm.Irecv(recv_back, source=rank + (num_procs_x * num_procs_y))

        if sim_params.left_wall == BoundaryCondition.PERIODIC:
            if rank_x == 0:
                req_left_send = comm.Isend(left_data, dest=rank + (num_procs_x - 1))
                req_left_recv = comm.Irecv(recv_left, source=rank + (num_procs_x - 1))
            if rank_x == num_procs_x - 1:
                req_right_send = comm.Isend(right_data, dest=rank - (num_procs_x - 1))
                req_right_recv = comm.Irecv(recv_right, source=rank - (num_procs_x - 1))

        if sim_params.top_wall == BoundaryCondition.PERIODIC:
            if rank_y == 0:
                req_top_send = comm.Isend(top_data, dest=rank + (num_procs_y - 1) * num_procs_x)
                req_top_recv = comm.Irecv(recv_top, source=rank + (num_procs_y - 1) * num_procs_x)
            if rank_y == num_procs_y - 1:
                req_bottom_send = comm.Isend(bottom_data, dest=rank - (num_procs_y - 1) * num_procs_x)
                req_bottom_recv = comm.Irecv(recv_bottom, source=rank - (num_procs_y - 1) * num_procs_x)

        if sim_params.front_wall == BoundaryCondition.PERIODIC:
            z_offset = (num_procs_z - 1) * num_procs_y * num_procs_x
            if rank_z == 0:
                req_front_send = comm.Isend(front_data, dest=rank + z_offset)
                req_front_recv = comm.Irecv(recv_front, source=rank + z_offset)
            if rank_z == num_procs_z - 1:
                req_back_send = comm.Isend(back_data, dest=rank - z_offset)
                req_back_recv = comm.Irecv(recv_back, source=rank - z_offset)

        request.update(
            {
                "left_send": req_left_send,
                "left_recv": req_left_recv,
                "right_send": req_right_send,
                "right_recv": req_right_recv,
                "top_send": req_top_send,
                "top_recv": req_top_recv,
                "bottom_send": req_bottom_send,
                "bottom_recv": req_bottom_recv,
                "front_send": req_front_send,
                "front_recv": req_front_recv,
                "back_send": req_back_send,
                "back_recv": req_back_recv,
            }
        )
        buffers.update(
            {
                "from_left": recv_left,
                "from_right": recv_right,
                "from_top": recv_top,
                "from_bottom": recv_bottom,
                "from_front": recv_front,
                "from_back": recv_back,
            }
        )
        return request, buffers

    raise ValueError(f"Invalid dimension: {sim_params.dimension}")


def recv_ghost_info(
    array: np.ndarray,
    sim_params: SimulationParameters,
    rank_coords: Any,
    request: dict[str, Any],
    buffers: dict[str, Any],
) -> None:
    num_ghosts = sim_params.num_ghost_layers
    num_procs_x = int(sim_params.num_procs_x or 1)
    num_procs_y = int(sim_params.num_procs_y or 1)
    num_procs_z = int(sim_params.num_procs_z or 1)

    if sim_params.dimension == 1:
        rank_x = int(rank_coords)

        left_send = request["left_send"]
        right_send = request["right_send"]
        left_recv = request["left_recv"]
        right_recv = request["right_recv"]

        if rank_x > 0 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            left_send.Wait()
            left_recv.Wait()
            array[0:num_ghosts] = buffers["from_left"]
        if rank_x < num_procs_x - 1 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            right_send.Wait()
            right_recv.Wait()
            array[-num_ghosts:] = buffers["from_right"]
        return

    if sim_params.dimension == 2:
        rank_x = int(rank_coords[0])
        rank_y = int(rank_coords[1])

        if rank_x > 0 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            request["left_send"].Wait()
            request["left_recv"].Wait()
            array[0:num_ghosts, :] = buffers["from_left"]
        if rank_x < num_procs_x - 1 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            request["right_send"].Wait()
            request["right_recv"].Wait()
            array[-num_ghosts:, :] = buffers["from_right"]

        if rank_y > 0 or sim_params.top_wall == BoundaryCondition.PERIODIC:
            request["top_send"].Wait()
            request["top_recv"].Wait()
            array[:, 0:num_ghosts] = buffers["from_top"]
        if rank_y < num_procs_y - 1 or sim_params.top_wall == BoundaryCondition.PERIODIC:
            request["bottom_send"].Wait()
            request["bottom_recv"].Wait()
            array[:, -num_ghosts:] = buffers["from_bottom"]
        return

    if sim_params.dimension == 3:
        rank_x = int(rank_coords[0])
        rank_y = int(rank_coords[1])
        rank_z = int(rank_coords[2])

        if rank_x > 0 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            request["left_send"].Wait()
            request["left_recv"].Wait()
            array[0:num_ghosts, :, :] = buffers["from_left"]
        if rank_x < num_procs_x - 1 or sim_params.left_wall == BoundaryCondition.PERIODIC:
            request["right_send"].Wait()
            request["right_recv"].Wait()
            array[-num_ghosts:, :, :] = buffers["from_right"]

        if rank_y > 0 or sim_params.top_wall == BoundaryCondition.PERIODIC:
            request["top_send"].Wait()
            request["top_recv"].Wait()
            array[:, 0:num_ghosts, :] = buffers["from_top"]
        if rank_y < num_procs_y - 1 or sim_params.top_wall == BoundaryCondition.PERIODIC:
            request["bottom_send"].Wait()
            request["bottom_recv"].Wait()
            array[:, -num_ghosts:, :] = buffers["from_bottom"]

        if rank_z > 0 or sim_params.front_wall == BoundaryCondition.PERIODIC:
            request["front_send"].Wait()
            request["front_recv"].Wait()
            array[:, :, 0:num_ghosts] = buffers["from_front"]
        if rank_z < num_procs_z - 1 or sim_params.front_wall == BoundaryCondition.PERIODIC:
            request["back_send"].Wait()
            request["back_recv"].Wait()
            array[:, :, -num_ghosts:] = buffers["from_back"]
        return

    raise ValueError(f"Invalid dimension: {sim_params.dimension}")

