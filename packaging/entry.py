from __future__ import annotations

import os
import runpy
import sys


def configure_bundled_mpi() -> None:
    """
    Point Open MPI and mpi4py at the copy of Open MPI in Contents/Resources/mpi, and overwrite any value the environment already holds, because the bundle runs its own Open MPI and no other. mpiexec reads OPAL_PREFIX, prterun reads PRTE_PREFIX, and mpi4py dlopens the file named in MPI4PY_LIBMPI. Without these three variables the bundled app finds no MPI library and every import of mpi4py fails. An inherited MPI4PY_MPIABI must go, because mpi4py names the ABI from that variable and then dlopens no MPI4PY_LIBMPI at all. This does nothing outside a frozen bundle, where the virtual environment already supplies Open MPI.
    """

    if not getattr(sys, "frozen", False):
        return

    contents_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    mpi_root = os.path.join(contents_dir, "Resources", "mpi")
    if not os.path.isdir(mpi_root):
        return

    os.environ["OPAL_PREFIX"] = mpi_root
    os.environ["PRTE_PREFIX"] = mpi_root
    os.environ.pop("MPI4PY_MPIABI", None)
    os.environ["MPI4PY_LIBMPI"] = os.path.join(mpi_root, "lib", "libmpi.40.dylib")
    os.environ["PATH"] = os.path.join(mpi_root, "bin") + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    """
    Start the ExaFlow window, or run one simulation script when the first argument is --run-script. mpirun starts the bundled app once for each rank in that second mode, because a frozen bundle holds no separate Python interpreter to start.
    """

    configure_bundled_mpi()

    if len(sys.argv) > 1 and sys.argv[1] == "--run-script":
        if len(sys.argv) < 3:
            print("--run-script needs the path of a Python script.", file=sys.stderr)
            return 2
        script_path = sys.argv[2]
        sys.argv = [script_path, *sys.argv[3:]]
        runpy.run_path(script_path, run_name="__main__")
        return 0

    from exaflow.gui.app import main as start_gui

    return start_gui()


if __name__ == "__main__":
    sys.exit(main())
