"""
Launches one simulation under mpirun and reports what it prints.
"""

from __future__ import annotations

import os
import sys

from PySide6 import QtCore


class SimulationRunner(QtCore.QObject):
    """
    Owns the child process a run happens in. `start` builds the mpirun command line and the environment the child needs, and returns the command as one line for a log. Output arrives on `output` while the run continues, and `finished` fires once at the end.

    One runner drives one process at a time. A second `start` while a run is active replaces nothing and is the caller's mistake to avoid; ask `is_running` first.
    """

    output = QtCore.Signal(str)
    finished = QtCore.Signal(int, str)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        return self._process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(
        self,
        script_path: str,
        working_directory: str,
        num_procs: int,
        script_arguments: list[str],
        output_root: str,
    ) -> str:
        """
        Start `script_path` under mpirun at `num_procs` ranks and return the command line as one line of text. Emits `failed` when the process does not start within three seconds.

        The child gets the source root of this package on PYTHONPATH, so a script run from any directory imports it, and EXAFLOW_OUTPUT_ROOT so its results land where the window watches.
        """

        # Run the chosen script directly; scripts include a small import bootstrap for relative imports
        interpreter_arguments = [sys.executable, "--run-script"] if getattr(sys, "frozen", False) else [sys.executable]
        launch_arguments = ["-np", str(num_procs), *interpreter_arguments, script_path, *script_arguments]
        display_tail = f"{os.path.basename(script_path)} {' '.join(script_arguments)}".strip()
        display_command = f"mpirun -np {num_procs} {' '.join(interpreter_arguments)} {display_tail}"

        self._process.setWorkingDirectory(working_directory)
        # Ensure child Python can import the top-level package
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        # Prepend the src/ root to PYTHONPATH
        package_source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        existing_path = environment.value("PYTHONPATH", "")
        separator = ":" if existing_path else ""
        environment.insert("PYTHONPATH", f"{package_source_root}{separator}{existing_path}")
        environment.insert("EXAFLOW_OUTPUT_ROOT", output_root)
        self._process.setProcessEnvironment(environment)
        self._process.start("mpirun", launch_arguments)

        if not self._process.waitForStarted(3000):
            self.failed.emit(self._process.errorString())
        return display_command

    def stop(self) -> None:
        """
        Kill the run. Does nothing when no process is active, so a caller need not test first.
        """

        if self.is_running():
            self._process.kill()

    def _on_ready_read(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode(errors="ignore")
        if data:
            self.output.emit(data.rstrip("\n"))

    def _on_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self.finished.emit(int(code), str(status))
