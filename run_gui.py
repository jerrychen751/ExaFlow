from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """
    Allow `python run_gui.py` without requiring an editable install.

    The canonical code lives under `src/`, so we add it to sys.path if needed.
    """

    repo_root = Path(__file__).resolve().parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from exaflow.gui.app import main


if __name__ == "__main__":
    sys.exit(main())
