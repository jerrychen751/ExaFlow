from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def resolve_output_root() -> str:
    """
    Return the directory that holds every ExaFlow run, and create it if it is absent. The default is ~/Documents/ExaFlow. Set EXAFLOW_OUTPUT_ROOT to an absolute path to use a different directory.
    """

    root = os.environ.get("EXAFLOW_OUTPUT_ROOT", "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), "Documents", "ExaFlow")
    root = os.path.abspath(os.path.expanduser(root))
    os.makedirs(root, exist_ok=True)
    return root


def create_run_directory(label: str, comm: Intracomm | None = None) -> str:
    """
    Create one directory for this run under the output root and return its absolute path. The name is <YYYY-MM-DD_HHMMSS>_<label>, so a later run never overwrites an earlier one. Every character outside [A-Za-z0-9-_] in the label becomes an underscore. This is a collective call: rank 0 picks the name and broadcasts it, so every rank must call it, and a call on rank 0 alone makes the other ranks wait forever.
    """

    rank = int(comm.Get_rank()) if comm is not None else 0

    run_dir: str | None = None
    if rank == 0:
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label).strip("_")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = os.path.join(resolve_output_root(), f"{stamp}_{safe_label}" if safe_label else stamp)
        os.makedirs(run_dir, exist_ok=True)
    if comm is not None and int(comm.Get_size()) > 1:
        run_dir = comm.bcast(run_dir, root=0)
    return str(run_dir)
