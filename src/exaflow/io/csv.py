from __future__ import annotations

import os
from itertools import product

import numpy as np

VELOCITY_NAMES = ("u", "v", "w")
AXIS_NAMES = ("x", "y", "z")


def format_field_csv(velocity: np.ndarray, pressure: np.ndarray, origin: tuple[int, ...]) -> str:
    """
    Render one block as CSV text. `velocity` is (dimension, *shape) and `pressure` is (shape). `origin` is the global index of the block's first cell, so the index columns carry global indices even when the block is one rank's share.

    The column order is the axis indices, then the velocity components in axis order, then pressure. Rows run with the last axis varying fastest.
    """

    dimension = int(velocity.shape[0])
    header = ",".join((*AXIS_NAMES[:dimension], *VELOCITY_NAMES[:dimension], "p"))
    lines = [f"{header}\n"]
    for index in product(*(range(length) for length in pressure.shape)):
        labels = [str(start + offset) for start, offset in zip(origin, index)]
        values = [str(velocity[(component, *index)]) for component in range(dimension)]
        lines.append(", ".join([*labels, *values, str(pressure[index])]) + "\n")
    return "".join(lines)


def write_text_atomically(path: str, contents: str) -> None:
    """
    Write the file so that a reader never sees a half-written one. The text goes to a sibling `.partial` file, is flushed to disk, and is then renamed over the target.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial_path = f"{path}.partial"
    with open(partial_path, "w", encoding="utf-8") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)
