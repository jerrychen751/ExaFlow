from __future__ import annotations

import csv
import os
from typing import Tuple, List

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData  # type: ignore
from vtkmodules.util.numpy_support import numpy_to_vtk  # type: ignore


def _infer_dimensions_from_indices(coordinate_rows: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    max_x_index = max(row[0] for row in coordinate_rows)
    max_y_index = max(row[1] for row in coordinate_rows)
    max_z_index = max(row[2] for row in coordinate_rows)
    return max_x_index + 1, max_y_index + 1, max_z_index + 1


def load_total_csv_to_imagedata(file_path: str) -> vtkImageData:
    """
    Load a *_Total.csv produced by write_total_array_to_csv (dimension==3) into vtkImageData. The CSV format is: x,y,z,u,v,w,p with integer indices and float values.
    """
    absolute_path = os.path.abspath(file_path)
    coordinate_indices: List[Tuple[int, int, int]] = []
    velocity_u_values: List[float] = []
    velocity_v_values: List[float] = []
    velocity_w_values: List[float] = []
    pressure_values: List[float] = []

    with open(absolute_path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        # Robust to headers with/without spaces
        for data_row in csv_reader:
            if not data_row or len(data_row) < 7:
                continue
            try:
                x_index = int(data_row[0])
                y_index = int(data_row[1])
                z_index = int(data_row[2])
                velocity_u = float(data_row[3])
                velocity_v = float(data_row[4])
                velocity_w = float(data_row[5])
                pressure = float(data_row[6])
            except Exception:
                # Some writers include trailing commas; try to strip empties and retry
                cleaned_row = [cell for cell in data_row if cell.strip() != ""]
                if len(cleaned_row) < 7:
                    continue
                x_index = int(cleaned_row[0])
                y_index = int(cleaned_row[1])
                z_index = int(cleaned_row[2])
                velocity_u = float(cleaned_row[3])
                velocity_v = float(cleaned_row[4])
                velocity_w = float(cleaned_row[5])
                pressure = float(cleaned_row[6])
            coordinate_indices.append((x_index, y_index, z_index))
            velocity_u_values.append(velocity_u)
            velocity_v_values.append(velocity_v)
            velocity_w_values.append(velocity_w)
            pressure_values.append(pressure)

    if not coordinate_indices:
        raise ValueError(f"No parsable data rows in {absolute_path}; expected rows of x,y,z,u,v,w,p.")

    nx, ny, nz = _infer_dimensions_from_indices(coordinate_indices)

    # Allocate arrays
    velocity_u_array = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_v_array = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_w_array = np.zeros((nx, ny, nz), dtype=np.float32)
    pressure_array = np.zeros((nx, ny, nz), dtype=np.float32)

    # Fill arrays (assumes rows are arbitrary order)
    for (x_idx, y_idx, z_idx), u_val, v_val, w_val, p_val in zip(coordinate_indices, velocity_u_values, velocity_v_values, velocity_w_values, pressure_values):
        velocity_u_array[x_idx, y_idx, z_idx] = u_val
        velocity_v_array[x_idx, y_idx, z_idx] = v_val
        velocity_w_array[x_idx, y_idx, z_idx] = w_val
        pressure_array[x_idx, y_idx, z_idx] = p_val

    # Convert to vtkImageData (index space)
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(nx, ny, nz)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)

    # VTK expects x-fastest ordering; flatten in Fortran order
    vtk_velocity_u = numpy_to_vtk(velocity_u_array.flatten(order="F"), deep=True)
    vtk_velocity_u.SetName("u")
    vtk_velocity_v = numpy_to_vtk(velocity_v_array.flatten(order="F"), deep=True)
    vtk_velocity_v.SetName("v")
    vtk_velocity_w = numpy_to_vtk(velocity_w_array.flatten(order="F"), deep=True)
    vtk_velocity_w.SetName("w")
    vtk_pressure = numpy_to_vtk(pressure_array.flatten(order="F"), deep=True)
    vtk_pressure.SetName("pressure")

    point_data = vtk_image.GetPointData()
    point_data.AddArray(vtk_velocity_u)
    point_data.AddArray(vtk_velocity_v)
    point_data.AddArray(vtk_velocity_w)
    point_data.AddArray(vtk_pressure)
    point_data.SetActiveScalars("pressure")

    # Also add a 3-component velocity vector for glyphs
    try:
        velocity_vector_field = np.stack([velocity_u_array, velocity_v_array, velocity_w_array], axis=-1)
        velocity_vector_field = velocity_vector_field.astype(np.float32, copy=False)
        vtk_velocity_vector = numpy_to_vtk(velocity_vector_field.reshape(-1, 3, order="F"), deep=True)
        vtk_velocity_vector.SetName("velocity")
        point_data.AddArray(vtk_velocity_vector)
        point_data.SetActiveVectors("velocity")
    except Exception:
        # If anything fails, continue without vectors
        pass

    return vtk_image


