from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not on PATH")
@pytest.mark.parametrize("num_procs", [1, 2, 4])
def test_a_periodic_axis_wraps_onto_the_right_plane(num_procs: int) -> None:
    """Two ranks on a periodic axis are each other's neighbor on both faces, so each posts two
    sends and two receives to the same peer. Untagged, MPI matched them in order and filled the low
    ghost plane with the high one, which no rank-count independence test caught."""

    script = REPOSITORY_ROOT / "tests" / "_run_periodic_exchange.py"
    command = ["mpiexec", "-n", str(num_procs), sys.executable, str(script)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
