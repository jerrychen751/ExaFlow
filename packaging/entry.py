from __future__ import annotations

import os
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
    Start the ExaFlow window when no argument is present. Answer `--check-bundle` here, and route every other argument to the standard CLI, because each MPI rank restarts this bundled executable.
    """

    configure_bundled_mpi()

    if len(sys.argv) > 1 and sys.argv[1] == "--check-bundle":
        return _check_bundle()
    if len(sys.argv) > 1:
        from exaflow.cli import main as start_cli

        return start_cli(sys.argv[1:])

    from exaflow.gui.app import main as start_gui

    return start_gui()


def _check_bundle() -> int:
    import mpi4py

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    mpi4py.rc.initialize = False
    import PySide6.QtWidgets
    import exaflow.gui.app
    import exaflow.numerics.pressure_poisson
    import pyvista
    import pyvistaqt
    import scipy.sparse
    import vtkmodules.all
    from mpi4py import MPI

    __import__("vtk")
    PySide6.QtWidgets.QApplication([])
    print(f"The bundle imports scipy, VTK, PySide6 and {MPI.Get_library_version().split(',')[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
