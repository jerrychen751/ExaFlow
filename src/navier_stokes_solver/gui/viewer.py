from __future__ import annotations

import os
import math
from PySide6 import QtWidgets
from typing import Optional, Any

import pyvista as pv
from pyvistaqt import QtInteractor  # type: ignore

import numpy as np
import vtk

from .csv_loader import load_total_csv_to_imagedata


_VTK_SUPPORTS_BOOL_SCALAR_BAR_TOGGLES = hasattr(vtk.vtkScalarBarActor, "SetDrawTickMarks")


class PyVistaViewer(QtWidgets.QFrame):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Embedded PyVista plotter
        self._plotter = QtInteractor(self)  # type: ignore
        self._plotter.set_background("slategray")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)

        self._main_mesh_actor: Any = None  # main mesh actor
        self._scalar_bar: Any = None  # not strictly needed; kept for API symmetry

        # Cached dataset (PyVista dataset such as UniformGrid/RectilinearGrid)
        self._simulation_data: Any = None

        # Orientation widget (axes) and overlays
        self._coordinate_axes_actor: Any = None
        self._ori_widget: Any = True  # placeholder to keep previous attribute available

        # Outline and cube axes
        self._domain_outline_actor: Any = None
        self._cube_axes: Any = None  # flag/actor depending on PyVista version

        # Vector glyph pipeline
        self._velocity_arrows_actor: Any = None

        # Toggles and settings
        self._show_coordinate_axes: bool = True
        self._show_domain_outline: bool = True
        self._show_cube_axes: bool = True
        self._show_velocity_vectors: bool = False
        self._vector_sampling_stride: int = 4  # sampling ratio for vectors

        # Color map to mimic blue->red
        self._color_map: str = "coolwarm"

        # Build orientation widget (disabled/enabled via toggle)
        if self._show_coordinate_axes:
            self._coordinate_axes_actor = self._plotter.add_axes()

    # ------------------ Public API ------------------
    def clear(self) -> None:
        if self._main_mesh_actor is not None:
            self._plotter.remove_actor(self._main_mesh_actor)
            self._main_mesh_actor = None

        # Remove outline
        if self._domain_outline_actor is not None:
            self._plotter.remove_actor(self._domain_outline_actor)
            self._domain_outline_actor = None

        # Remove cube axes
        self._plotter.remove_bounds_axes()
        self._cube_axes = None

        # Remove vector glyphs
        if self._velocity_arrows_actor is not None:
            self._plotter.remove_actor(self._velocity_arrows_actor)
            self._velocity_arrows_actor = None

        # Orientation marker visibility per toggle
        if self._show_coordinate_axes and self._coordinate_axes_actor is None:
            self._coordinate_axes_actor = self._plotter.add_axes()
        if (not self._show_coordinate_axes) and (self._coordinate_axes_actor is not None):
            self._plotter.remove_actor(self._coordinate_axes_actor)
            self._coordinate_axes_actor = None

        self._simulation_data = None
        self._plotter.render()

    def load_vtr(self, path: str) -> None:
        abspath = os.path.abspath(path)
        mesh = pv.read(abspath)
        self.load_mesh(mesh)

    def load_csv(self, path: str) -> None:
        vtk_image_data = load_total_csv_to_imagedata(path)
        mesh = pv.wrap(vtk_image_data)
        self.load_mesh(mesh)

    def load_mesh(self, mesh: pv.DataSet) -> None:
        self.clear()
        self._load_mesh(mesh)

    def _load_mesh(self, mesh: pv.DataSet) -> None:
        self._simulation_data = mesh

        # Choose scalar if available
        scalar_name: Optional[str] = None
        if mesh.point_data and mesh.active_scalars_name:
            scalar_name = mesh.active_scalars_name
        elif "pressure" in mesh.point_data:
            scalar_name = "pressure"
        elif len(mesh.point_data.keys()) > 0:
            scalar_name = list(mesh.point_data.keys())[0]

        # Scalar bar formatting
        scalar_bar_format = "%.3g"
        if scalar_name is not None:
            data_range = mesh.get_data_range(scalar_name, "point")
            max_magnitude = max(abs(data_range[0]), abs(data_range[1]))
            if max_magnitude >= 1e4 or (0 < max_magnitude <= 1e-3):
                scalar_bar_format = "%.2e"

        # Scalar bar geometry
        bar_height = 0.34
        bar_width = 0.038
        margin_left = 0.045
        margin_top = 0.05
        bar_position_x = margin_left
        bar_position_y = 1.0 - bar_height - margin_top

        scalar_bar_args = dict(
            title=(scalar_name or "").capitalize(),
            n_labels=6,
            fmt=scalar_bar_format,
            position_x=bar_position_x,
            position_y=bar_position_y,
            width=bar_width,
            height=bar_height,
            vertical=True,
            label_font_size=11,
            title_font_size=13,
        )

        surface_mesh = mesh.extract_surface()

        self._main_mesh_actor = self._plotter.add_mesh(
            surface_mesh,
            scalars=scalar_name,
            cmap=self._color_map,
            scalar_bar_args=scalar_bar_args,
        )

        # Cache and tweak scalar bar appearance
        self._scalar_bar = None
        scalar_bars = getattr(self._plotter, "scalar_bars", {})
        if scalar_name and scalar_name in scalar_bars:
            self._scalar_bar = scalar_bars[scalar_name]
        elif getattr(self._plotter, "scalar_bar", None) is not None:
            self._scalar_bar = self._plotter.scalar_bar

        if self._scalar_bar is not None:
            text_color = (1.0, 1.0, 1.0)
            if _VTK_SUPPORTS_BOOL_SCALAR_BAR_TOGGLES:
                self._scalar_bar.SetDrawTickMarks(True)
                self._scalar_bar.SetDrawTickLabels(True)
            else:
                if hasattr(self._scalar_bar, "DrawTickMarksOn"):
                    self._scalar_bar.DrawTickMarksOn()
                if hasattr(self._scalar_bar, "DrawTickLabelsOn"):
                    self._scalar_bar.DrawTickLabelsOn()
            if hasattr(self._scalar_bar, "SetTickLength"):
                self._scalar_bar.SetTickLength(0.03)
            if hasattr(self._scalar_bar, "SetLabelTextPad"):
                self._scalar_bar.SetLabelTextPad(12)

            label_text_property = self._scalar_bar.GetLabelTextProperty()
            label_text_property.SetColor(*text_color)
            if hasattr(label_text_property, "SetJustificationToCentered"):
                label_text_property.SetJustificationToCentered()
            title_text_property = self._scalar_bar.GetTitleTextProperty()
            title_text_property.SetColor(*text_color)
            if hasattr(title_text_property, "SetJustificationToCentered"):
                title_text_property.SetJustificationToCentered()

        # Update overlays and vectors
        self._update_overlays_and_vectors()
        self._plotter.reset_camera()
        self._plotter.render()


    # ------------------ Overlays and vectors ------------------
    def _update_overlays_and_vectors(self) -> None:
        self._update_outline()
        self._update_cube_axes()
        self._update_vectors()

    def _update_outline(self) -> None:
        if self._simulation_data is None:
            return
        if self._show_domain_outline:
            if self._domain_outline_actor is None:
                domain_outline = self._simulation_data.outline()
                self._domain_outline_actor = self._plotter.add_mesh(
                    domain_outline,
                    color="black",
                    line_width=1,
                    style="wireframe",
                    show_scalar_bar=False,
                )
        else:
            if self._domain_outline_actor is not None:
                self._plotter.remove_actor(self._domain_outline_actor)
                self._domain_outline_actor = None

    def _update_cube_axes(self) -> None:
        if self._simulation_data is None:
            return
        if self._show_cube_axes:
            if self._cube_axes is None:
                # Show cube axes with white labels
                self._plotter.show_bounds(
                    bounds=self._simulation_data.bounds,
                    location="outer",
                    color="white",
                    show_xaxis=True,
                    show_yaxis=True,
                    show_zaxis=True,
                    grid=False,
                )
                self._cube_axes = True
            else:
                # Refresh bounds if dataset changed
                self._plotter.remove_bounds_axes()
                self._plotter.show_bounds(
                    bounds=self._simulation_data.bounds,
                    location="outer",
                    color="white",
                    show_xaxis=True,
                    show_yaxis=True,
                    show_zaxis=True,
                    grid=False,
                )
                self._cube_axes = True
        else:
            self._plotter.remove_bounds_axes()
            self._cube_axes = None

    def _find_vector_array_name(self) -> Optional[str]:
        if self._simulation_data is None:
            return None
        point_data = self._simulation_data.point_data
        if point_data is None:
            return None
        # Prefer explicitly named vectors
        for name in ("velocity", "Velocity", "U", "VEL", "vec"):
            if name in point_data:
                vector_array = point_data[name]
                if vector_array.ndim == 2 and vector_array.shape[1] == 3:
                    return name
        # Fallback: search any 3-comp array
        for name in point_data.keys():
            vector_array = point_data[name]
            if vector_array.ndim == 2 and vector_array.shape[1] == 3:
                return name
        return None

    def _update_vectors(self) -> None:
        # Remove if disabled or no dataset
        if (not self._show_velocity_vectors) or (self._simulation_data is None):
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None
            return

        vector_field_name = self._find_vector_array_name()
        if not vector_field_name:
            # No vectors available
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None
            return

        try:
            mesh = self._simulation_data
            mesh_points = np.asarray(mesh.points)
            velocity_vectors = np.asarray(mesh.point_data[vector_field_name])
            num_points = int(mesh_points.shape[0])

            sampling_stride = max(1, int(self._vector_sampling_stride))
            sampled_indices = np.arange(0, num_points, sampling_stride, dtype=int)
            if sampled_indices.size == 0:
                sampled_indices = np.array([0], dtype=int)

            arrow_centers = mesh_points[sampled_indices]
            arrow_directions = velocity_vectors[sampled_indices]

            # Scale arrows relative to dataset size
            mesh_bounds = mesh.bounds
            diagonal_length = math.sqrt((mesh_bounds[1]-mesh_bounds[0])**2 + (mesh_bounds[3]-mesh_bounds[2])**2 + (mesh_bounds[5]-mesh_bounds[4])**2)
            arrow_scale_factor = max(diagonal_length, 1e-6) / 50.0

            # Build glyphs using PolyData.glyph so we can color by scalars
            arrow_polydata = pv.PolyData(arrow_centers)
            arrow_polydata.point_data[vector_field_name] = arrow_directions
            # Optional coloring by pressure
            pressure_scalar_name: Optional[str] = None
            if "pressure" in mesh.point_data:
                arrow_polydata.point_data["pressure"] = np.asarray(mesh.point_data["pressure"])[sampled_indices]
                pressure_scalar_name = "pressure"

            arrow_glyphs = arrow_polydata.glyph(orient=vector_field_name, scale=False, factor=arrow_scale_factor)

            # Remove old glyph actor
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None

            if pressure_scalar_name is not None:
                self._velocity_arrows_actor = self._plotter.add_mesh(
                    arrow_glyphs,
                    scalars=pressure_scalar_name,
                    cmap=self._color_map,
                    show_scalar_bar=False,
                )
            else:
                self._velocity_arrows_actor = self._plotter.add_mesh(
                    arrow_glyphs,
                    color="white",
                    show_scalar_bar=False,
                )
        except Exception:
            # On any failure, remove glyphs
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None

    # ------------------ Public toggles ------------------
    def set_show_axes(self, show: bool) -> None:
        self._show_coordinate_axes = show
        if self._show_coordinate_axes:
            if self._coordinate_axes_actor is None:
                self._coordinate_axes_actor = self._plotter.add_axes()
        else:
            if self._coordinate_axes_actor is not None:
                self._plotter.remove_actor(self._coordinate_axes_actor)
                self._coordinate_axes_actor = None
        self._plotter.render()

    def set_show_outline(self, show: bool) -> None:
        self._show_domain_outline = show
        self._update_outline()
        self._plotter.render()

    def set_show_cube_axes(self, show: bool) -> None:
        self._show_cube_axes = show
        self._update_cube_axes()
        self._plotter.render()

    def set_show_vectors(self, show: bool) -> None:
        self._show_velocity_vectors = show
        self._update_vectors()
        self._plotter.render()

    def set_vector_stride(self, stride: int) -> None:
        self._vector_sampling_stride = max(1, int(stride))
        if self._show_velocity_vectors:
            self._update_vectors()
            self._plotter.render()

    # ------------------ Camera presets ------------------
    def _dataset_center(self) -> Optional[tuple[float, float, float]]:
        if self._simulation_data is None:
            return None
        bounds = self._simulation_data.bounds
        return ((bounds[0]+bounds[1])/2.0, (bounds[2]+bounds[3])/2.0, (bounds[4]+bounds[5])/2.0)

    def _dataset_size(self) -> float:
        if self._simulation_data is None:
            return 1.0
        bounds = self._simulation_data.bounds
        return math.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)

    def _set_camera(self, position: tuple[float, float, float], up_vector: tuple[float, float, float]) -> None:
        center = self._dataset_center()
        if center is None:
            return
        self._plotter.camera_position = [position, center, up_vector]
        self._plotter.reset_camera()
        self._plotter.render()

    def view_pos_x(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0] + distance, center[1], center[2]), (0.0, 0.0, 1.0))

    def view_neg_x(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0] - distance, center[1], center[2]), (0.0, 0.0, 1.0))

    def view_pos_y(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0], center[1] + distance, center[2]), (0.0, 0.0, 1.0))

    def view_neg_y(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0], center[1] - distance, center[2]), (0.0, 0.0, 1.0))

    def view_pos_z(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0], center[1], center[2] + distance), (0.0, 1.0, 0.0))

    def view_neg_z(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        self._set_camera((center[0], center[1], center[2] - distance), (0.0, 1.0, 0.0))

    def view_iso(self) -> None:
        center = self._dataset_center()
        if center is None:
            return
        distance = self._dataset_size()
        offset = distance / math.sqrt(3.0)
        self._set_camera((center[0] + offset, center[1] + offset, center[2] + offset), (0.0, 0.0, 1.0))

