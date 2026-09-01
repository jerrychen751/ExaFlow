"""
The restart format. One checkpoint holds the whole domain plus the position of the run that wrote it, so a later process can continue from it at any rank count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config.case import Case
from ..fields import FlowState, TimeLevel, allocate_state
from ..mpi.gather import scatter_global_array
from ..mpi.subdomain import Subdomain

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """
    One run position and the whole domain that goes with it. `velocity` has shape (dimension, *grid.shape) and `pressure` has shape grid.shape, both float64 and both with the ghost layers stripped. `case_xml` is the text `write_case` produced for the running case, so the file needs no second file to be read.

    This is a whole-domain record. Rank 0 builds it and reads it; the other ranks never hold one.
    """

    velocity: np.ndarray
    pressure: np.ndarray
    level: TimeLevel
    case_xml: str


def write_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    """
    Write one checkpoint, and never a half one. Rank 0 calls this and no other rank does. The arrays go to `<path>.partial`, are flushed to disk, and the file is then renamed over the target, which is the rule `write_text_atomically` follows for a CSV. The file is opened here rather than named, because `np.savez` appends `.npz` to a path that does not end in it and would leave the partial file behind.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial_path = f"{path}.partial"
    with open(partial_path, "wb") as handle:
        np.savez(
            handle,
            format_version=CHECKPOINT_FORMAT_VERSION,
            velocity=checkpoint.velocity,
            pressure=checkpoint.pressure,
            step_index=checkpoint.level.step_index,
            current_time=checkpoint.level.current_time,
            dt=checkpoint.level.dt,
            case_xml=checkpoint.case_xml,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)


def read_checkpoint(path: str) -> Checkpoint:
    """
    Read one whole checkpoint on one rank. Raises ValueError when the file is not a checkpoint or was written by another format version, naming both version numbers, and raises whatever `np.load` raises for a file it cannot open at all.
    """

    with np.load(path) as data:
        if "format_version" not in data:
            raise ValueError(f"{path} is not an ExaFlow checkpoint: it has no format_version.")
        version = int(data["format_version"])
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format version {version}, and this build reads version "
                f"{CHECKPOINT_FORMAT_VERSION}."
            )
        return Checkpoint(
            velocity=data["velocity"],
            pressure=data["pressure"],
            level=TimeLevel(int(data["step_index"]), float(data["current_time"]), float(data["dt"])),
            case_xml=str(data["case_xml"]),
        )


def read_case_text(path: str, comm: Intracomm | None = None) -> str:
    """
    The input XML text a checkpoint carries, on every rank. Rank 0 reads that one member and broadcasts it, and an npz reads a member only when it is asked for, so the arrays stay on disk.

    This is a collective call: every rank must reach it. Raises ValueError when the file holds no case text. A rank 0 that cannot read the file at all broadcasts the failure and every rank raises it, because a raise on rank 0 alone would leave the others waiting at the broadcast forever.
    """

    is_parallel = comm is not None and int(comm.Get_size()) > 1
    text = ""
    failure = ""
    if comm is None or int(comm.Get_rank()) == 0:
        try:
            with np.load(path) as data:
                if "case_xml" not in data:
                    raise ValueError(f"{path} is not an ExaFlow checkpoint: it has no case_xml.")
                text = str(data["case_xml"])
        except Exception as error:
            if not is_parallel:
                raise
            failure = f"{type(error).__name__}: {error}"
    if is_parallel:
        assert comm is not None
        text, failure = comm.bcast((text, failure), root=0)
        if failure:
            raise ValueError(f"rank 0 could not read {path}: {failure}")
    return text


def scatter_checkpoint(
    path: str,
    case: Case,
    subdomain: Subdomain,
    comm: Intracomm | None = None,
) -> tuple[FlowState, TimeLevel]:
    """
    Spread one checkpoint over the ranks. Rank 0 reads the file and every rank gets the block of the domain its subdomain owns, plus the level rank 0 broadcasts.

    This is a collective call: every rank must reach it. The returned state holds the stored interior values and zero ghost layers, so the caller writes the prescribed boundary values before the first step. Raises ValueError when the stored domain has a different shape from the case, because the blocks would not fit. A rank 0 that cannot read the file broadcasts the failure and every rank raises it, because a raise on rank 0 alone would leave the others waiting at the broadcast forever.
    """

    is_parallel = comm is not None and int(comm.Get_size()) > 1
    checkpoint = None
    failure = ""
    if comm is None or int(comm.Get_rank()) == 0:
        try:
            checkpoint = read_checkpoint(path)
            if checkpoint.pressure.shape != tuple(case.grid.shape):
                raise ValueError(
                    f"{path} holds a domain of shape {checkpoint.pressure.shape}, and the case asks for "
                    f"{tuple(case.grid.shape)}."
                )
        except Exception as error:
            if not is_parallel:
                raise
            checkpoint = None
            failure = f"{type(error).__name__}: {error}"

    level = checkpoint.level if checkpoint is not None else None
    if is_parallel:
        assert comm is not None
        level, failure = comm.bcast((level, failure), root=0)
        if failure:
            raise ValueError(f"rank 0 could not read {path}: {failure}")
    if level is None:
        raise ValueError(f"{path} gave no run position to continue from.")

    state = allocate_state(subdomain, case.dimension)
    for axis in range(case.dimension):
        component = None if checkpoint is None else checkpoint.velocity[axis]  # (dimension, *shape) -> (*shape,)
        state.velocity[axis][subdomain.interior] = scatter_global_array(subdomain, comm, component)  # (dimension, *padded_shape) -> (*shape,)
    pressure = None if checkpoint is None else checkpoint.pressure
    state.pressure[subdomain.interior] = scatter_global_array(subdomain, comm, pressure)
    return state, level
