"""
Launches one ExaFlow case under MPI and reports what it prints.
"""

from __future__ import annotations

import os
import shlex
import sys

from PySide6 import QtCore


class SimulationRunner(QtCore.QObject):
    """
    Owns the child process for one run. `start` executes `exaflow run --case` under MPI and returns the command for the log.

    Output arrives on `output`. `failed` reports a start fault. `finished` fires once at the end.
    """

    output = QtCore.Signal(str)
    finished = QtCore.Signal(int, str)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.errorOccurred.connect(self._on_error)
        self._process.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        return self._process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(
        self,
        case_path: str,
        num_procs: int,
        output_root: str,
    ) -> str:
        """
        Start the case under `mpiexec` with `num_procs` ranks. Return the command as one line of text.

        A source run uses `python -m exaflow.cli`. A frozen run uses the bundle executable. Both processes receive `EXAFLOW_OUTPUT_ROOT`.
        """

        # The executable is the absolute path to the binary running the current process
        # For Desktop app this is the ExaFlow application (sys.frozen is True), for MPI CLI this is Python executable
        entry_arguments = (
            [sys.executable]
            if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "exaflow.cli"]
        )
        launch_arguments = [
            "-n",
            str(num_procs),
            *entry_arguments,
            "run",
            "--case",
            os.path.abspath(case_path),
        ]
        display_command = shlex.join(["mpiexec", *launch_arguments])

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("EXAFLOW_OUTPUT_ROOT", output_root)
        self._process.setProcessEnvironment(environment)
        self._process.start("mpiexec", launch_arguments)
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

    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.failed.emit(self._process.errorString())

    def _on_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self.finished.emit(int(code), str(status))
