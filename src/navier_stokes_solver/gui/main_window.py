from __future__ import annotations

import os
import sys
import glob
import traceback
from datetime import datetime
from PySide6 import QtCore, QtWidgets
from typing import Optional, Any
from pathlib import Path

import pyvista as pv
from .viewer import PyVistaViewer
from .help_dialog import HelpDialog
from .csv_loader import load_total_csv_to_imagedata
from .streaming import StreamingServer, DEFAULT_STREAMING_PORT
from .sim_parameters_dialog import SimulationParametersDialog, simulation_values_to_xml
from .settings_dialog import SettingsDialog, SessionSettings


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Navier–Stokes – GUI")

        # Process used to run simulations
        self._simulation_process = QtCore.QProcess(self)
        self._simulation_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._simulation_process.readyReadStandardOutput.connect(self._on_process_output)
        self._simulation_process.finished.connect(self._on_process_finished)

        # State
        self._last_loaded_path: Optional[str] = None
        self._gui_parameters: Optional[dict[str, Any]] = None
        self._script_allows_gui_parameters: bool = False
        self._session_settings = SessionSettings()
        self._streaming_active: bool = False

        # UI
        self.viewer: Any = None
        self._build_ui()

        # Directory watcher
        self._autoload_timer = QtCore.QTimer(self)
        self._autoload_timer.setInterval(self._session_settings.autoload_interval_ms)
        self._autoload_timer.timeout.connect(self._maybe_autoload_latest)
        self._autoload_timer.start()
        
        # Run autoload immediately on startup
        self._maybe_autoload_latest()

        # Streaming server for real-time updates
        self._streaming_server: Optional[StreamingServer] = None
        try:
            self._streaming_server = StreamingServer(
                port=DEFAULT_STREAMING_PORT,
                parent=self,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Streaming Error",
                f"Unable to start streaming server:\n{exc}",
            )
        else:
            if not self._streaming_server.is_listening():
                QtWidgets.QMessageBox.critical(
                    self,
                    "Streaming Error",
                    "Unable to start streaming server.",
                )
                self._streaming_server = None
            else:
                # Signal is emitted only after full message is read and successfully unpickled (see server.py)
                self._streaming_server.data_received.connect(self._on_streaming_dataset)

    # ------------------------- UI -------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        # Left controls
        controls = QtWidgets.QWidget(central)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        form = QtWidgets.QFormLayout()

        # Script selection
        self._script_path_input = QtWidgets.QLineEdit(controls)
        self._script_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._script_browse_button.clicked.connect(self._browse_script)
        self._script_path_input.textChanged.connect(self._on_script_path_changed)
        script_input_row = QtWidgets.QHBoxLayout()
        script_input_row.addWidget(self._script_path_input)
        script_input_row.addWidget(self._script_browse_button)
        form.addRow("Script (.py)", script_input_row)

        self._params_button = QtWidgets.QPushButton("Simulation Params…", controls)
        self._params_button.clicked.connect(self._open_simulation_params_dialog)
        self._params_status = QtWidgets.QLabel("Unavailable")
        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(self._params_button)
        params_row.addWidget(self._params_status, 1)
        form.addRow("GUI parameters", params_row)

        # MPI processes
        self._mpi_processes_input = QtWidgets.QSpinBox(controls)
        self._mpi_processes_input.setRange(1, 512)
        self._mpi_processes_input.setValue(4)
        form.addRow("MPI processes", self._mpi_processes_input)

        # Working directory
        self._working_directory_input = QtWidgets.QLineEdit(controls)
        default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self._working_directory_input.setText(default_root)
        self._working_directory_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._working_directory_browse_button.clicked.connect(self._browse_working_directory)
        working_directory_row = QtWidgets.QHBoxLayout()
        working_directory_row.addWidget(self._working_directory_input)
        working_directory_row.addWidget(self._working_directory_browse_button)
        form.addRow("Working dir", working_directory_row)

        # Output directory (for watcher)
        self._output_directory_input = QtWidgets.QLineEdit(controls)
        self._output_directory_input.setText(default_root)
        self._output_directory_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._output_directory_browse_button.clicked.connect(self._browse_output_directory)
        output_directory_row = QtWidgets.QHBoxLayout()
        output_directory_row.addWidget(self._output_directory_input)
        output_directory_row.addWidget(self._output_directory_browse_button)
        form.addRow("Output dir", output_directory_row)

        # Auto-load latest
        controls_layout.addLayout(form)

        # Run controls
        run_controls_row = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run", controls)
        self._run_button.clicked.connect(self._run_simulation)
        self._stop_button = QtWidgets.QPushButton("Stop", controls)
        self._stop_button.clicked.connect(self._stop_simulation)
        self._open_file_button = QtWidgets.QPushButton("Open File…", controls)
        self._open_file_button.clicked.connect(self._open_file)
        self._help_button = QtWidgets.QPushButton("?", controls)
        self._help_button.clicked.connect(self._show_help)
        self._help_button.setMaximumWidth(30)
        self._settings_button = QtWidgets.QToolButton(controls)
        self._settings_button.setToolTip("Session settings")
        settings_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self._settings_button.setIcon(settings_icon)
        self._settings_button.clicked.connect(self._open_settings_dialog)
        run_controls_row.addWidget(self._run_button)
        run_controls_row.addWidget(self._stop_button)
        run_controls_row.addWidget(self._open_file_button)
        run_controls_row.addWidget(self._help_button)
        run_controls_row.addWidget(self._settings_button)
        controls_layout.addLayout(run_controls_row)

        # Log output
        self._log_output = QtWidgets.QPlainTextEdit(controls)
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumBlockCount(5000)
        controls_layout.addWidget(QtWidgets.QLabel("Run log"))
        controls_layout.addWidget(self._log_output, 1)

        # Right viewer
        viewer_container: QtWidgets.QWidget
        self._viewer: PyVistaViewer = PyVistaViewer(central)  # type: ignore

        # Build a right-side layout with a small toolbar for viewer controls
        right = QtWidgets.QWidget(central)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar row: overlays and vectors
        toolbar = QtWidgets.QHBoxLayout()

        self._axes_checkbox = QtWidgets.QCheckBox("Axes")
        self._axes_checkbox.setChecked(True)
        self._axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_axes(bool(v)))  # type: ignore
        toolbar.addWidget(self._axes_checkbox)

        self._outline_checkbox = QtWidgets.QCheckBox("Outline")
        self._outline_checkbox.setChecked(True)
        self._outline_checkbox.toggled.connect(lambda v: self._viewer.set_show_outline(bool(v)))  # type: ignore
        toolbar.addWidget(self._outline_checkbox)

        self._cube_axes_checkbox = QtWidgets.QCheckBox("Cube Axes")
        self._cube_axes_checkbox.setChecked(True)
        self._cube_axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_cube_axes(bool(v)))  # type: ignore
        toolbar.addWidget(self._cube_axes_checkbox)

        self._vectors_checkbox = QtWidgets.QCheckBox("Vectors")
        self._vectors_checkbox.setChecked(False)
        self._vectors_checkbox.toggled.connect(lambda v: self._viewer.set_show_vectors(bool(v)))  # type: ignore
        toolbar.addWidget(self._vectors_checkbox)

        toolbar.addWidget(QtWidgets.QLabel("Stride"))
        self._vector_stride_spinbox = QtWidgets.QSpinBox()
        self._vector_stride_spinbox.setRange(1, 50)
        self._vector_stride_spinbox.setValue(4)
        self._vector_stride_spinbox.valueChanged.connect(lambda n: self._viewer.set_vector_stride(int(n)))  # type: ignore
        toolbar.addWidget(self._vector_stride_spinbox)

        toolbar.addStretch(1)

        # Camera preset buttons
        btns = [
            ("+X", "view_pos_x"), ("-X", "view_neg_x"),
            ("+Y", "view_pos_y"), ("-Y", "view_neg_y"),
            ("+Z", "view_pos_z"), ("-Z", "view_neg_z"),
            ("ISO", "view_iso"),
        ]
        for text, meth in btns:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(getattr(self._viewer, meth))
            toolbar.addWidget(b)

        right_layout.addLayout(toolbar)
        right_layout.addWidget(self._viewer)
        viewer_container = right

        # Splitter layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        controls.setMinimumWidth(360)
        splitter.addWidget(controls)
        splitter.addWidget(viewer_container)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Defaults: try to prefill script path
        template_script = os.path.abspath(os.path.join(default_root, "examples", "template_3d_problem.py"))
        if os.path.exists(template_script):
            self._script_path_input.setText(template_script)
        else:
            self._on_script_path_changed(self._script_path_input.text())

    # ------------------------- Process handling -------------------------
    def _run_simulation(self) -> None:
        script_path = self._script_path_input.text().strip()
        if not script_path:
            QtWidgets.QMessageBox.warning(self, "Missing script", "Please choose a Python script to run.")
            return
        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.critical(self, "Script not found", script_path)
            return

        # Re-evaluate script capabilities in case file changed on disk.
        self._script_allows_gui_parameters = self._script_supports_gui_parameters(script_path)
        self._update_params_status()

        working_directory = self._working_directory_input.text().strip() or os.path.dirname(script_path)
        mpi_processes = int(self._mpi_processes_input.value())

        extra_script_args: list[str] = []
        gui_xml_path: Optional[str] = None
        if self._should_use_gui_parameters():
            try:
                gui_xml_path = self._write_gui_parameters_xml(script_path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Failed to write GUI parameters", str(exc))
                return
            extra_script_args.append(f"--input-xml={gui_xml_path}")
            self._append_log(f"[{self._now()}] GUI parameters saved to {gui_xml_path}")
        elif self._gui_parameters and not self._script_allows_gui_parameters:
            self._append_log("GUI parameters ignored: script defines its own SimulationParameters or loads XML.")

        self._log_output.clear()
        # Run the chosen script directly; scripts include a small import bootstrap for relative imports
        extra_str = " ".join(extra_script_args)
        display_tail = f"{os.path.basename(script_path)} {extra_str}".strip()
        display_command = f"mpirun -np {mpi_processes} {sys.executable} {display_tail}"
        launch_arguments = ["-np", str(mpi_processes), sys.executable, script_path]
        launch_arguments.extend(extra_script_args)
        self._append_log(f"[{self._now()}] Starting: {display_command}")

        self._run_button.setEnabled(False)

        # Start process (non-interactive)
        program = "mpirun"
        args = launch_arguments
        self._simulation_process.setWorkingDirectory(working_directory)
        # Ensure child Python can import the top-level package
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        # Prepend repo root to PYTHONPATH
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        existing_path = environment.value("PYTHONPATH", "")
        separator = ":" if existing_path else ""
        environment.insert("PYTHONPATH", f"{repo_root}{separator}{existing_path}")
        self._simulation_process.setProcessEnvironment(environment)
        self._simulation_process.start(program, args)

        if not self._simulation_process.waitForStarted(3000):
            self._run_button.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "Failed to start", self._simulation_process.errorString())

    def _stop_simulation(self) -> None:
        if self._simulation_process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        self._append_log(f"[{self._now()}] Stopping…")
        self._simulation_process.kill()
        self._append_log(f"[{self._now()}] Process stopped.")

    def _on_process_output(self) -> None:
        data = bytes(self._simulation_process.readAllStandardOutput()).decode(errors="ignore")
        if data:
            self._append_log(data.rstrip("\n"))

    def _on_process_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self._run_button.setEnabled(True)
        self._append_log(f"[{self._now()}] Finished with code {code} ({str(status)})")

    # ------------------------- File operations -------------------------
    def _open_file(self) -> None:
        if self._viewer is None:
            QtWidgets.QMessageBox.information(self, "Viewer unavailable", "Install 'pyvista' and 'pyvistaqt' to enable visualization.")
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open result file",
            self._output_directory_input.text().strip() or os.getcwd(),
            "VTK rectilinear (*.vtr);;CSV totals (*_Total.csv);;All files (*)",
        )
        if not file_path:
            return
        self._load_file_path(file_path)

    def _maybe_autoload_latest(self) -> None:
        if self._streaming_active:
            return
        if not self._session_settings.autoload_enabled or self._viewer is None:
            return
        output_directory = self._output_directory_input.text().strip()
        if not output_directory or not os.path.isdir(output_directory):
            return

        latest_file = self._find_latest_file(output_directory)
        if latest_file and latest_file != self._last_loaded_path:
            self._load_file_path(latest_file)

    def _find_latest_file(self, directory: str) -> str | None:
        vtr_files = glob.glob(os.path.join(directory, "**", "*.vtr"), recursive=True)
        csv_files = glob.glob(os.path.join(directory, "**", "*_Total.csv"), recursive=True)
        candidate_files = vtr_files + csv_files
        if not candidate_files:
            return None
        
        # Sort by modification time, but prefer .vtr files over .csv files
        # when timestamps are close (within 1 second)
        def sort_key(file_path: str) -> tuple[float, int]:
            mtime = os.path.getmtime(file_path)
            # Prefer .vtr files (priority 0) over .csv files (priority 1)
            priority = 0 if file_path.lower().endswith('.vtr') else 1
            return (-mtime, priority)  # Negative mtime for descending order
        
        candidate_files.sort(key=sort_key)
        return candidate_files[0]

    def _load_file_path(self, file_path: str) -> None:
        try:
            if file_path.lower().endswith(".vtr"):
                mesh = pv.read(file_path)
                self._viewer.load_mesh(mesh)  # type: ignore
                self._append_log(f"[{self._now()}] Loaded VTR: {file_path}")
            elif file_path.lower().endswith("_total.csv"):
                vtk_image_data = load_total_csv_to_imagedata(file_path)
                mesh = pv.wrap(vtk_image_data)
                self._viewer.load_mesh(mesh)  # type: ignore
                self._append_log(f"[{self._now()}] Loaded CSV: {file_path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported file", os.path.basename(file_path))
                return
            self._last_loaded_path = file_path
        except Exception:
            self._append_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Failed to load", os.path.basename(file_path))

    # ------------------------- Network operations -------------------------
    def _on_streaming_dataset(self, dataset: pv.DataSet) -> None:
        self._streaming_active = True
        if self._viewer is None:
            self._append_log("Streaming data received but viewer unavailable.")
            return
        try:
            self._viewer.load_mesh(dataset)
        except Exception:
            self._append_log(traceback.format_exc())

    # ------------------------- Helpers -------------------------
    def _browse_script(self) -> None:
        default_path = self._script_path_input.text().strip() or os.path.join(self._working_directory_input.text().strip(), "examples")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Python script", default_path, "Python (*.py)")
        if file_path:
            self._script_path_input.setText(file_path)

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self, self._session_settings)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._session_settings = dialog.settings()
            self._autoload_timer.setInterval(self._session_settings.autoload_interval_ms)
            if self._session_settings.autoload_enabled:
                if not self._autoload_timer.isActive():
                    self._autoload_timer.start()
                self._maybe_autoload_latest()
            else:
                self._autoload_timer.stop()

    def _open_simulation_params_dialog(self) -> None:
        dialog = SimulationParametersDialog(self, self._gui_parameters)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._gui_parameters = dialog.values()
            self._update_params_status()

    def _on_script_path_changed(self, _: str) -> None:
        self._script_allows_gui_parameters = self._script_supports_gui_parameters(self._script_path_input.text().strip())
        if not self._script_allows_gui_parameters:
            self._gui_parameters = None
        self._update_params_status()

    def _script_supports_gui_parameters(self, script_path: str) -> bool:
        if not script_path or not os.path.isfile(script_path):
            return False
        try:
            text = Path(script_path).read_text(encoding="utf-8")
        except Exception:
            return False
        lowered = text.lower()

        # Hard disable when the script manually instantiates SimulationParameters.
        if "simulationparameters(" in lowered:
            return False

        uses_cli_xml = "--input-xml" in text
        loads_xml = "simulationparametersfromxml(" in lowered or ".xml" in lowered

        # Allow scripts that expect the GUI to pass an XML path via --input-xml.
        if loads_xml and not uses_cli_xml:
            return False

        return True

    def _update_params_status(self) -> None:
        if not self._script_allows_gui_parameters:
            self._params_button.setEnabled(False)
            self._params_status.setText("Disabled: script defines its own parameters")
            return
        self._params_button.setEnabled(True)
        if self._gui_parameters is None:
            self._params_status.setText("Not configured")
        else:
            self._params_status.setText("Configured")

    def _should_use_gui_parameters(self) -> bool:
        return bool(self._script_allows_gui_parameters and self._gui_parameters)

    def _write_gui_parameters_xml(self, script_path: str) -> str:
        if self._gui_parameters is None:
            raise RuntimeError("GUI parameters not configured")
        script_file = Path(script_path).resolve()
        xml_path = script_file.with_name(f"{script_file.stem}_input_parameters.xml")
        xml_payload = simulation_values_to_xml(self._gui_parameters)
        xml_path.write_text(xml_payload, encoding="utf-8")
        return str(xml_path)

    def _browse_working_directory(self) -> None:
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose working directory", self._working_directory_input.text().strip() or os.getcwd())
        if directory_path:
            self._working_directory_input.setText(directory_path)

    def _browse_output_directory(self) -> None:
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory", self._output_directory_input.text().strip() or os.getcwd())
        if directory_path:
            self._output_directory_input.setText(directory_path)

    def _show_help(self) -> None:
        """Show the help dialog."""
        help_dialog = HelpDialog(self)
        help_dialog.exec()

    def _append_log(self, text: str) -> None:
        self._log_output.appendPlainText(text)
        self._log_output.verticalScrollBar().setValue(self._log_output.verticalScrollBar().maximum())

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _path_to_module(script_path: str, work_dir: str) -> str | None:
        """Convert a script path under work_dir to a dotted module path.

        Returns None if the path is not a .py file under work_dir.
        """
        try:
            abs_script = os.path.abspath(script_path)
            abs_root = os.path.abspath(work_dir)
            if not abs_script.endswith(".py"):
                return None
            # Ensure script is inside the working directory
            try:
                rel = os.path.relpath(abs_script, abs_root)
            except Exception:
                return None
            if rel.startswith(os.pardir):
                return None
            # Build dotted module path
            module = rel[:-3].replace(os.sep, ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            return module or None
        except Exception:
            return None

