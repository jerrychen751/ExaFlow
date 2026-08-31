from __future__ import annotations

import os
import traceback
from datetime import datetime
from PySide6 import QtCore, QtWidgets
from typing import Optional
from pathlib import Path

import pyvista as pv
from .viewer import PyVistaViewer
from .help_dialog import HelpDialog
from .streaming import StreamingServer, DEFAULT_STREAMING_PORT
from .sim_parameters_dialog import SimulationParametersDialog
from .settings_dialog import SettingsDialog, SessionSettings
from .simulation_runner import SimulationRunner
from .result_watcher import LatestResultWatcher
from .slice_controller import SliceController
from ..config import Case
from ..config.case_xml import write_case
from ..io.storage import resolve_output_root


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Navier–Stokes – GUI")

        # Process used to run simulations
        self._runner = SimulationRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._on_process_finished)
        self._runner.failed.connect(self._on_process_failed)

        # State
        self._gui_case: Optional[Case] = None
        self._script_allows_gui_parameters: bool = False
        self._session_settings = SessionSettings()

        # UI
        self._build_ui()

        # Directory watcher
        self._watcher = LatestResultWatcher(self._session_settings.autoload_interval_ms, self)
        self._watcher.found.connect(self._load_file_path)
        self._refresh_watcher()

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
        self._script_path_input.editingFinished.connect(self._on_script_path_changed)
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
        default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
        self._working_directory_input.setText(default_root)
        self._working_directory_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._working_directory_browse_button.clicked.connect(self._browse_working_directory)
        working_directory_row = QtWidgets.QHBoxLayout()
        working_directory_row.addWidget(self._working_directory_input)
        working_directory_row.addWidget(self._working_directory_browse_button)
        form.addRow("Working dir", working_directory_row)

        # Output directory (for watcher)
        self._output_directory_input = QtWidgets.QLineEdit(controls)
        self._output_directory_input.setText(resolve_output_root())
        self._output_directory_input.editingFinished.connect(self._refresh_watcher)
        self._output_directory_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._output_directory_browse_button.clicked.connect(self._browse_output_directory)
        output_directory_row = QtWidgets.QHBoxLayout()
        output_directory_row.addWidget(self._output_directory_input)
        output_directory_row.addWidget(self._output_directory_browse_button)
        form.addRow("Output root", output_directory_row)

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
        self._viewer = PyVistaViewer(central)
        self._viewer.render_failed.connect(self._append_log)

        # Build a right-side layout with a small toolbar for viewer controls
        right = QtWidgets.QWidget(central)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar row: overlays and vectors
        toolbar = QtWidgets.QHBoxLayout()

        self._axes_checkbox = QtWidgets.QCheckBox("Axes")
        self._axes_checkbox.setChecked(True)
        self._axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_axes(bool(v)))
        toolbar.addWidget(self._axes_checkbox)

        self._outline_checkbox = QtWidgets.QCheckBox("Outline")
        self._outline_checkbox.setChecked(True)
        self._outline_checkbox.toggled.connect(lambda v: self._viewer.set_show_outline(bool(v)))
        toolbar.addWidget(self._outline_checkbox)

        self._cube_axes_checkbox = QtWidgets.QCheckBox("Cube Axes")
        self._cube_axes_checkbox.setChecked(True)
        self._cube_axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_cube_axes(bool(v)))
        toolbar.addWidget(self._cube_axes_checkbox)

        self._vectors_checkbox = QtWidgets.QCheckBox("Vectors")
        self._vectors_checkbox.setChecked(False)
        self._vectors_checkbox.toggled.connect(lambda v: self._viewer.set_show_vectors(bool(v)))
        toolbar.addWidget(self._vectors_checkbox)

        toolbar.addWidget(QtWidgets.QLabel("Stride"))
        self._vector_stride_spinbox = QtWidgets.QSpinBox()
        self._vector_stride_spinbox.setRange(1, 50)
        self._vector_stride_spinbox.setValue(4)
        self._vector_stride_spinbox.valueChanged.connect(lambda n: self._viewer.set_vector_stride(int(n)))
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

        self._slice = SliceController(self._viewer, self)

        right_layout.addLayout(toolbar)
        right_layout.addLayout(self._slice.build_toolbar())
        right_layout.addWidget(self._viewer)

        # Splitter layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        controls.setMinimumWidth(360)
        splitter.addWidget(controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Defaults: try to prefill script path
        template_script = os.path.abspath(os.path.join(default_root, "examples", "template_3d_problem.py"))
        if os.path.exists(template_script):
            self._script_path_input.setText(template_script)
        self._on_script_path_changed()

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
            except OSError as exc:
                QtWidgets.QMessageBox.critical(self, "Failed to write GUI parameters", str(exc))
                return
            extra_script_args.append(f"--input-xml={gui_xml_path}")
            self._append_log(f"[{self._now()}] GUI parameters saved to {gui_xml_path}")
        elif self._gui_case and not self._script_allows_gui_parameters:
            self._append_log("GUI parameters ignored: the script accepts no --input-xml argument.")

        self._log_output.clear()
        self._run_button.setEnabled(False)
        display_command = self._runner.start(
            script_path,
            working_directory,
            mpi_processes,
            extra_script_args,
            self._output_directory_input.text().strip() or resolve_output_root(),
        )
        self._append_log(f"[{self._now()}] Starting: {display_command}")

    def _stop_simulation(self) -> None:
        if not self._runner.is_running():
            return
        self._append_log(f"[{self._now()}] Stopping…")
        self._runner.stop()
        self._append_log(f"[{self._now()}] Process stopped.")

    def _on_process_finished(self, code: int, status: str) -> None:
        self._run_button.setEnabled(True)
        self._append_log(f"[{self._now()}] Finished with code {code} ({status})")

    def _on_process_failed(self, message: str) -> None:
        self._run_button.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Failed to start", message)

    # ------------------------- File operations -------------------------
    def _open_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open result file",
            self._output_directory_input.text().strip() or os.getcwd(),
            "VTK rectilinear (*.vtr);;CSV totals (*_Total.csv);;All files (*)",
        )
        if not file_path:
            return
        self._load_file_path(file_path)

    def _refresh_watcher(self) -> None:
        self._watcher.set_directory(self._output_directory_input.text().strip())
        self._watcher.set_interval(self._session_settings.autoload_interval_ms)
        self._watcher.set_enabled(self._session_settings.autoload_enabled)
        if self._session_settings.autoload_enabled:
            self._watcher.poll()

    def _load_file_path(self, file_path: str) -> None:
        try:
            if file_path.lower().endswith(".vtr"):
                self._viewer.load_vtr(file_path)
                self._append_log(f"[{self._now()}] Loaded VTR: {file_path}")
            elif file_path.lower().endswith("_total.csv"):
                self._viewer.load_csv(file_path)
                self._append_log(f"[{self._now()}] Loaded CSV: {file_path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported file", os.path.basename(file_path))
                return
            self._watcher.note_loaded(file_path)
            self._slice.refresh()
        except Exception:
            self._watcher.note_loaded(file_path)
            self._append_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Failed to load", os.path.basename(file_path))

    # ------------------------- Network operations -------------------------
    def _on_streaming_dataset(self, dataset: pv.DataSet) -> None:
        # A live stream replaces the newest file on disk, so the watcher stops competing with it.
        self._watcher.set_enabled(False)
        try:
            self._viewer.load_mesh(dataset)
            self._slice.refresh()
        except Exception:
            self._append_log(traceback.format_exc())

    # ------------------------- Helpers -------------------------
    def _browse_script(self) -> None:
        default_path = self._script_path_input.text().strip() or os.path.join(self._working_directory_input.text().strip(), "examples")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Python script", default_path, "Python (*.py)")
        if file_path:
            self._script_path_input.setText(file_path)
            self._on_script_path_changed()

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self, self._session_settings)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._session_settings = dialog.settings()
            self._refresh_watcher()

    def _open_simulation_params_dialog(self) -> None:
        dialog = SimulationParametersDialog(self, self._gui_case)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._gui_case = dialog.case()
            self._update_params_status()

    def _on_script_path_changed(self) -> None:
        self._script_allows_gui_parameters = self._script_supports_gui_parameters(self._script_path_input.text().strip())
        if not self._script_allows_gui_parameters:
            self._gui_case = None
        self._update_params_status()

    def _script_supports_gui_parameters(self, script_path: str) -> bool:
        """
        Report whether the chosen script accepts a case file on the command line, which is the contract the GUI needs to pass its parameters through.

        This looks for the argparse flag the script declares, not for what the script means. A script that builds its own Case in code declares no flag and is left alone.
        """

        if not script_path or not os.path.isfile(script_path):
            return False
        try:
            text = Path(script_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        return "--input-xml" in text or "--case" in text

    def _update_params_status(self) -> None:
        if not self._script_allows_gui_parameters:
            self._params_button.setEnabled(False)
            self._params_status.setText("Disabled: script takes no --input-xml")
            return
        self._params_button.setEnabled(True)
        if self._gui_case is None:
            self._params_status.setText("Not configured")
        else:
            self._params_status.setText("Configured")

    def _should_use_gui_parameters(self) -> bool:
        return bool(self._script_allows_gui_parameters and self._gui_case)

    def _write_gui_parameters_xml(self, script_path: str) -> str:
        """
        Write the dialog's case beside the script as `<stem>_input_parameters.xml` and return that path. The rank count is not written; the arrangement follows the communicator the run is launched with.
        """

        if self._gui_case is None:
            raise RuntimeError("GUI parameters not configured")
        script_file = Path(script_path).resolve()
        xml_path = script_file.with_name(f"{script_file.stem}_input_parameters.xml")
        xml_path.write_text(write_case(self._gui_case), encoding="utf-8")
        return str(xml_path)

    def _browse_working_directory(self) -> None:
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose working directory", self._working_directory_input.text().strip() or os.getcwd())
        if directory_path:
            self._working_directory_input.setText(directory_path)

    def _browse_output_directory(self) -> None:
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory", self._output_directory_input.text().strip() or os.getcwd())
        if directory_path:
            self._output_directory_input.setText(directory_path)
            self._refresh_watcher()

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
