from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import xml.etree.ElementTree as ET
from xml.dom import minidom

from PySide6 import QtCore, QtWidgets


DEFAULT_INITIAL_CONDITIONS_XML = """<InitialConditions>
  <ReadFromVtrFile>False</ReadFromVtrFile>
  <ReadFromCsvFile>False</ReadFromCsvFile>
  <SpecifyValues>True</SpecifyValues>
  <SpecifiedValues>
    <p>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </p>
    <u>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </u>
    <v>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </v>
    <w>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </w>
  </SpecifiedValues>
</InitialConditions>
"""


def _default_values() -> Dict[str, Any]:
    return {
        "rho": 1.225,
        "nu": 1.81e-5,
        "domain": [100, 100, 50],
        "size": [6.28, 3.14, 2.0],
        "nt": 1000,
        "numGhostLayers": 1,
        "CFL": 0.5,
        "numProcs": 8,
        "numProcsX": -1,
        "numProcsY": -1,
        "numProcsZ": -1,
        "leftWall": "Inflow",
        "leftInflow": {"u": 2.0, "v": 1.0, "w": 1.0},
        "leftOutflow": {"p": 0.0},
        "rightWall": "Outflow",
        "rightInflow": {"u": -1.0, "v": -1.0, "w": -1.0},
        "rightOutflow": {"p": 0.0},
        "topWall": "No Slip Wall",
        "topInflow": {"u": -1.0, "v": -1.0, "w": -1.0},
        "topOutflow": {"p": 0.0},
        "bottomWall": "No Slip Wall",
        "bottomInflow": {"u": -1.0, "v": -1.0, "w": -1.0},
        "bottomOutflow": {"p": 0.0},
        "frontWall": "No Slip Wall",
        "frontInflow": {"u": -1.0, "v": -1.0, "w": -1.0},
        "frontOutflow": {"p": 0.0},
        "backWall": "No Slip Wall",
        "backInflow": {"u": -1.0, "v": -1.0, "w": -1.0},
        "backOutflow": {"p": 0.0},
        "includeConvection": True,
        "includeDiffusion": True,
        "includePressure": True,
        "convectionScheme": "Upwind",
        "viscousScheme": "CentralDifference",
        "timeIntegrationOrder": 1,
        "VTKfrequency": 100,
        "totalCSVfrequency": 100,
        "partialCSVfrequency": 250,
        "initialConditionsXML": DEFAULT_INITIAL_CONDITIONS_XML.strip(),
    }


@dataclass
class BoundaryWidgets:
    wall: QtWidgets.QLineEdit
    inflow: Dict[str, QtWidgets.QDoubleSpinBox]
    outflow: Dict[str, QtWidgets.QDoubleSpinBox]


class SimulationParametersDialog(QtWidgets.QDialog):
    """Dialog that exposes every field needed to build SimulationParameters."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, initial_values: Dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation Parameters")
        self.setModal(True)
        self.resize(640, 720)

        self._values: Dict[str, Any] = _default_values()
        if initial_values:
            self._values.update(initial_values)

        self._double_fields: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._int_fields: Dict[str, QtWidgets.QSpinBox] = {}
        self._text_fields: Dict[str, QtWidgets.QLineEdit] = {}
        self._bool_fields: Dict[str, QtWidgets.QCheckBox] = {}
        self._combo_fields: Dict[str, QtWidgets.QComboBox] = {}
        self._boundary_fields: Dict[str, BoundaryWidgets] = {}

        self._initial_conditions_editor = QtWidgets.QPlainTextEdit()

        self._build_ui()
        self._populate_from_values()

    # ------------------------ UI construction ------------------------ #
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs, 1)

        tabs.addTab(self._build_fluid_tab(), "Fluid & Grid")
        tabs.addTab(self._build_parallel_solver_tab(), "Parallel & Solver")
        tabs.addTab(self._build_boundaries_tab(), "Boundaries")
        tabs.addTab(self._build_output_tab(), "Output & Misc")
        tabs.addTab(self._build_initial_conditions_tab(), "Initial Conditions")

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_fluid_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._double_fields["rho"] = self._create_double_input(minimum=1e-9, maximum=1e6, decimals=6)
        form.addRow("Density (rho)", self._double_fields["rho"])

        self._double_fields["nu"] = self._create_double_input(minimum=1e-9, maximum=1.0, decimals=8, single_step=1e-6)
        form.addRow("Kinematic viscosity (nu)", self._double_fields["nu"])

        self._int_fields["domain_nx"] = self._create_int_input(1, 10000)
        self._int_fields["domain_ny"] = self._create_int_input(1, 10000)
        self._int_fields["domain_nz"] = self._create_int_input(1, 10000)
        form.addRow("Domain (nx, ny, nz)", self._merge_triplet(self._int_fields["domain_nx"], self._int_fields["domain_ny"], self._int_fields["domain_nz"]))

        self._double_fields["size_length"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        self._double_fields["size_width"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        self._double_fields["size_height"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        form.addRow("Size (L, W, H)", self._merge_triplet(self._double_fields["size_length"], self._double_fields["size_width"], self._double_fields["size_height"]))

        self._int_fields["nt"] = self._create_int_input(1, 10_000_000)
        form.addRow("Time steps (nt)", self._int_fields["nt"])

        self._int_fields["numGhostLayers"] = self._create_int_input(0, 10)
        form.addRow("Ghost layers", self._int_fields["numGhostLayers"])

        self._double_fields["CFL"] = self._create_double_input(1e-6, 10.0, 4, 0.05)
        form.addRow("CFL", self._double_fields["CFL"])

        return widget

    def _build_parallel_solver_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._int_fields["numProcs"] = self._create_int_input(1, 4096)
        form.addRow("MPI ranks", self._int_fields["numProcs"])

        self._int_fields["numProcsX"] = self._create_int_input(-1, 4096)
        self._int_fields["numProcsY"] = self._create_int_input(-1, 4096)
        self._int_fields["numProcsZ"] = self._create_int_input(-1, 4096)
        form.addRow("Decomposition (x, y, z)", self._merge_triplet(self._int_fields["numProcsX"], self._int_fields["numProcsY"], self._int_fields["numProcsZ"]))

        self._bool_fields["includeConvection"] = QtWidgets.QCheckBox("Include convection")
        form.addRow(self._bool_fields["includeConvection"])
        self._combo_fields["convectionScheme"] = self._create_scheme_combo(["Upwind", "CentralDifference", "Hybrid"])
        form.addRow("Convection scheme", self._combo_fields["convectionScheme"])

        self._bool_fields["includeDiffusion"] = QtWidgets.QCheckBox("Include diffusion")
        form.addRow(self._bool_fields["includeDiffusion"])
        self._combo_fields["viscousScheme"] = self._create_scheme_combo(["CentralDifference", "Upwind", "Hybrid"])
        form.addRow("Viscous scheme", self._combo_fields["viscousScheme"])

        self._bool_fields["includePressure"] = QtWidgets.QCheckBox("Include pressure")
        form.addRow(self._bool_fields["includePressure"])

        self._int_fields["timeIntegrationOrder"] = self._create_int_input(1, 10)
        form.addRow("Time integration order", self._int_fields["timeIntegrationOrder"])

        return widget

    def _build_boundaries_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)

        boundary_names = [
            ("left", "Left"),
            ("right", "Right"),
            ("top", "Top"),
            ("bottom", "Bottom"),
            ("front", "Front"),
            ("back", "Back"),
        ]

        for idx, (key, title) in enumerate(boundary_names):
            group = QtWidgets.QGroupBox(f"{title} Boundary")
            group_layout = QtWidgets.QFormLayout(group)

            wall_field = QtWidgets.QLineEdit()
            group_layout.addRow("Wall type", wall_field)

            inflow_u = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_v = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_w = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_layout = self._merge_triplet(inflow_u, inflow_v, inflow_w, labels=("u", "v", "w"))
            group_layout.addRow("Inflow (u,v,w)", inflow_layout)

            outflow_p = self._create_double_input(-1e6, 1e6, 4, 0.1)
            outflow_layout = QtWidgets.QHBoxLayout()
            outflow_layout.addWidget(QtWidgets.QLabel("p"))
            outflow_layout.addWidget(outflow_p)
            group_layout.addRow("Outflow", outflow_layout)

            row = idx // 2
            col = idx % 2
            grid.addWidget(group, row, col)

            self._boundary_fields[key] = BoundaryWidgets(
                wall=wall_field,
                inflow={"u": inflow_u, "v": inflow_v, "w": inflow_w},
                outflow={"p": outflow_p},
            )

        layout.addStretch(1)
        return widget

    def _build_output_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._int_fields["VTKfrequency"] = self._create_int_input(-1, 1_000_000)
        form.addRow("VTK frequency", self._int_fields["VTKfrequency"])

        self._int_fields["totalCSVfrequency"] = self._create_int_input(-1, 1_000_000)
        form.addRow("Total CSV frequency", self._int_fields["totalCSVfrequency"])

        self._int_fields["partialCSVfrequency"] = self._create_int_input(-1, 1_000_000)
        form.addRow("Partial CSV frequency", self._int_fields["partialCSVfrequency"])

        return widget

    def _build_initial_conditions_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel("Provide a full <InitialConditions> XML block:"))
        self._initial_conditions_editor.setPlaceholderText("<InitialConditions>…</InitialConditions>")
        self._initial_conditions_editor.setTabChangesFocus(True)
        layout.addWidget(self._initial_conditions_editor, 1)
        return widget

    # ------------------------ Helpers ------------------------ #
    def _create_double_input(
        self,
        minimum: float = -1e9,
        maximum: float = 1e9,
        decimals: int = 6,
        single_step: float = 0.01,
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(single_step)
        widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        return widget

    def _create_int_input(self, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        widget.setAccelerated(True)
        return widget

    @staticmethod
    def _merge_triplet(
        first: QtWidgets.QWidget,
        second: QtWidgets.QWidget,
        third: QtWidgets.QWidget,
        labels: tuple[str, str, str] | None = None,
    ) -> QtWidgets.QWidget:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        widgets: List[QtWidgets.QWidget] = [first, second, third]
        for idx, widget in enumerate(widgets):
            if labels:
                layout.addWidget(QtWidgets.QLabel(labels[idx]))
            layout.addWidget(widget)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    @staticmethod
    def _create_scheme_combo(options: List[str]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.addItems(options)
        return combo

    def _populate_from_values(self) -> None:
        v = self._values
        self._double_fields["rho"].setValue(float(v["rho"]))
        self._double_fields["nu"].setValue(float(v["nu"]))

        nx, ny, nz = v["domain"]
        self._int_fields["domain_nx"].setValue(int(nx))
        self._int_fields["domain_ny"].setValue(int(ny))
        self._int_fields["domain_nz"].setValue(int(nz))

        length, width, height = v["size"]
        self._double_fields["size_length"].setValue(float(length))
        self._double_fields["size_width"].setValue(float(width))
        self._double_fields["size_height"].setValue(float(height))

        self._int_fields["nt"].setValue(int(v["nt"]))
        self._int_fields["numGhostLayers"].setValue(int(v["numGhostLayers"]))
        self._double_fields["CFL"].setValue(float(v["CFL"]))

        self._int_fields["numProcs"].setValue(int(v["numProcs"]))
        self._int_fields["numProcsX"].setValue(int(v["numProcsX"]))
        self._int_fields["numProcsY"].setValue(int(v["numProcsY"]))
        self._int_fields["numProcsZ"].setValue(int(v["numProcsZ"]))

        self._bool_fields["includeConvection"].setChecked(bool(v["includeConvection"]))
        self._combo_fields["convectionScheme"].setCurrentText(str(v["convectionScheme"]))

        self._bool_fields["includeDiffusion"].setChecked(bool(v["includeDiffusion"]))
        self._combo_fields["viscousScheme"].setCurrentText(str(v["viscousScheme"]))

        self._bool_fields["includePressure"].setChecked(bool(v["includePressure"]))
        self._int_fields["timeIntegrationOrder"].setValue(int(v["timeIntegrationOrder"]))

        self._int_fields["VTKfrequency"].setValue(int(v["VTKfrequency"]))
        self._int_fields["totalCSVfrequency"].setValue(int(v["totalCSVfrequency"]))
        self._int_fields["partialCSVfrequency"].setValue(int(v["partialCSVfrequency"]))

        for key, widgets in self._boundary_fields.items():
            wall_key = f"{key}Wall"
            inflow_key = f"{key}Inflow"
            outflow_key = f"{key}Outflow"
            widgets.wall.setText(str(v[wall_key]))
            inflow = v[inflow_key]
            outflow = v[outflow_key]
            widgets.inflow["u"].setValue(float(inflow.get("u", 0.0)))
            widgets.inflow["v"].setValue(float(inflow.get("v", 0.0)))
            widgets.inflow["w"].setValue(float(inflow.get("w", 0.0)))
            widgets.outflow["p"].setValue(float(outflow.get("p", 0.0)))

        initial_text = v.get("initialConditionsXML", DEFAULT_INITIAL_CONDITIONS_XML).strip()
        self._initial_conditions_editor.setPlainText(initial_text)

    # ------------------------ Data extraction ------------------------ #
    def values(self) -> Dict[str, Any]:
        domain = [
            int(self._int_fields["domain_nx"].value()),
            int(self._int_fields["domain_ny"].value()),
            int(self._int_fields["domain_nz"].value()),
        ]
        size = [
            float(self._double_fields["size_length"].value()),
            float(self._double_fields["size_width"].value()),
            float(self._double_fields["size_height"].value()),
        ]

        data: Dict[str, Any] = {
            "rho": float(self._double_fields["rho"].value()),
            "nu": float(self._double_fields["nu"].value()),
            "domain": domain,
            "size": size,
            "nt": int(self._int_fields["nt"].value()),
            "numGhostLayers": int(self._int_fields["numGhostLayers"].value()),
            "CFL": float(self._double_fields["CFL"].value()),
            "numProcs": int(self._int_fields["numProcs"].value()),
            "numProcsX": int(self._int_fields["numProcsX"].value()),
            "numProcsY": int(self._int_fields["numProcsY"].value()),
            "numProcsZ": int(self._int_fields["numProcsZ"].value()),
            "includeConvection": bool(self._bool_fields["includeConvection"].isChecked()),
            "includeDiffusion": bool(self._bool_fields["includeDiffusion"].isChecked()),
            "includePressure": bool(self._bool_fields["includePressure"].isChecked()),
            "convectionScheme": self._combo_fields["convectionScheme"].currentText().strip(),
            "viscousScheme": self._combo_fields["viscousScheme"].currentText().strip(),
            "timeIntegrationOrder": int(self._int_fields["timeIntegrationOrder"].value()),
            "VTKfrequency": int(self._int_fields["VTKfrequency"].value()),
            "totalCSVfrequency": int(self._int_fields["totalCSVfrequency"].value()),
            "partialCSVfrequency": int(self._int_fields["partialCSVfrequency"].value()),
            "initialConditionsXML": self._initial_conditions_editor.toPlainText().strip() or DEFAULT_INITIAL_CONDITIONS_XML.strip(),
        }

        for key, widgets in self._boundary_fields.items():
            data[f"{key}Wall"] = widgets.wall.text().strip() or "Wall"
            data[f"{key}Inflow"] = {
                "u": float(widgets.inflow["u"].value()),
                "v": float(widgets.inflow["v"].value()),
                "w": float(widgets.inflow["w"].value()),
            }
            data[f"{key}Outflow"] = {
                "p": float(widgets.outflow["p"].value()),
            }

        return data

    # ------------------------ Validation ------------------------ #
    def _on_accept(self) -> None:
        domain = [self._int_fields["domain_nx"].value(), self._int_fields["domain_ny"].value(), self._int_fields["domain_nz"].value()]
        if not all(n > 0 for n in domain):
            QtWidgets.QMessageBox.warning(self, "Invalid domain", "All domain sizes must be positive integers.")
            return
        size = [self._double_fields["size_length"].value(), self._double_fields["size_width"].value(), self._double_fields["size_height"].value()]
        if not all(s > 0 for s in size):
            QtWidgets.QMessageBox.warning(self, "Invalid size", "All physical dimensions must be positive.")
            return
        xml_text = self._initial_conditions_editor.toPlainText().strip()
        if xml_text and not xml_text.startswith("<InitialConditions"):
            QtWidgets.QMessageBox.warning(self, "Invalid initial conditions", "The block must start with <InitialConditions>.")
            return
        self.accept()


def _add_text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    elem = ET.SubElement(parent, tag)
    elem.text = str(value)
    return elem


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _prettify(element: ET.Element) -> str:
    rough_string = ET.tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent="  ")


def simulation_values_to_xml(values: Dict[str, Any]) -> str:
    """Convert dialog values to an InputTemplate-compatible XML string."""
    root = ET.Element("Simulation")

    fluid = ET.SubElement(root, "FluidProperties")
    _add_text(fluid, "Rho", values["rho"])
    _add_text(fluid, "Nu", values["nu"])

    grid = ET.SubElement(root, "GridProperties")
    size = ET.SubElement(grid, "Size")
    _add_text(size, "Length", values["size"][0])
    _add_text(size, "Width", values["size"][1])
    _add_text(size, "Height", values["size"][2])

    domain = ET.SubElement(grid, "Domain")
    _add_text(domain, "nx", values["domain"][0])
    _add_text(domain, "ny", values["domain"][1])
    _add_text(domain, "nz", values["domain"][2])

    _add_text(grid, "nt", values["nt"])
    _add_text(grid, "numGhosts", values["numGhostLayers"])
    _add_text(grid, "CFL", values["CFL"])

    initial_xml = values.get("initialConditionsXML") or DEFAULT_INITIAL_CONDITIONS_XML
    try:
        initial_element = ET.fromstring(initial_xml)
    except ET.ParseError:
        initial_element = ET.fromstring(DEFAULT_INITIAL_CONDITIONS_XML)
    root.append(initial_element)

    boundary = ET.SubElement(root, "BoundaryConditions")
    names = [
        ("left", "Left"),
        ("right", "Right"),
        ("top", "Top"),
        ("bottom", "Bottom"),
        ("front", "Front"),
        ("back", "Back"),
    ]
    for key, title in names:
        _add_text(boundary, f"{title}Wall", values[f"{key}Wall"])
        inflow = ET.SubElement(boundary, f"{title}Inflow")
        inflow_values = values[f"{key}Inflow"]
        _add_text(inflow, "u", inflow_values["u"])
        _add_text(inflow, "v", inflow_values["v"])
        _add_text(inflow, "w", inflow_values["w"])
        outflow = ET.SubElement(boundary, f"{title}Outflow")
        outflow_values = values[f"{key}Outflow"]
        _add_text(outflow, "p", outflow_values["p"])

    parallel = ET.SubElement(root, "ParallelizationProperties")
    _add_text(parallel, "numProcs", values["numProcs"])
    _add_text(parallel, "numProcsX", values["numProcsX"])
    _add_text(parallel, "numProcsY", values["numProcsY"])
    _add_text(parallel, "numProcsZ", values["numProcsZ"])

    solver = ET.SubElement(root, "SolverProperties")
    _add_text(solver, "IncludeConvectionEffects", _bool_text(values["includeConvection"]))
    _add_text(solver, "ConvectionScheme", values["convectionScheme"])
    _add_text(solver, "IncludeViscousEffects", _bool_text(values["includeDiffusion"]))
    _add_text(solver, "ViscousScheme", values["viscousScheme"])
    _add_text(solver, "TimeIntegrationOrder", values["timeIntegrationOrder"])
    _add_text(solver, "IncludePressureEffects", _bool_text(values["includePressure"]))

    output = ET.SubElement(root, "OutputProperties")
    _add_text(output, "WriteTotalVTKFrequency", values["VTKfrequency"])
    _add_text(output, "WriteTotalCSVFrequency", values["totalCSVfrequency"])
    _add_text(output, "WritePartialCSVFrequency", values["partialCSVfrequency"])

    return _prettify(root)


__all__ = ["SimulationParametersDialog", "DEFAULT_INITIAL_CONDITIONS_XML", "_default_values", "simulation_values_to_xml"]
