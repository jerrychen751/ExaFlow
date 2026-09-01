from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_scaffold_creates_only_the_case_file(tmp_path: Path) -> None:
    target = tmp_path / "simulation"
    script = Path(__file__).resolve().parents[1] / "Simulations" / "create_simulation_directories.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(target)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (target / "build" / "in" / "input.xml").is_file()
    assert not (target / "build" / "run_simulation.py").exists()
    assert "mpiexec -n 4 exaflow run --case build/in/input.xml" in (target / "metadata.txt").read_text()
    assert "Created directories" in completed.stdout
