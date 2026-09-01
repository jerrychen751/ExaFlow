from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Create build directory with templates")
    parser.add_argument("target_dir", help="Folder where build/ will be created")
    args = parser.parse_args()

    simulations_dir = Path(__file__).resolve().parent
    examples_dir = simulations_dir.parent / "examples"
    src_xml = examples_dir / "input_template.xml"
    if not src_xml.exists():
        raise FileNotFoundError(f"Template not found: {src_xml}")

    target_root = Path(args.target_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    metadata_path = target_root / "metadata.txt"
    if not metadata_path.exists():
        metadata_path.write_text(
            "Do not modify this file\n"
            "Created by create_simulation_directories.py\n"
            "Contains build/in/input.xml\n"
            "in/ has input.xml where you specify simulation parameters\n"
            "Output goes to one run folder under ~/Documents/ExaFlow, in the one format <Format> selects\n"
            "Set EXAFLOW_OUTPUT_ROOT to write those run folders somewhere else\n"
            "Run with: mpiexec -n 4 exaflow run --case build/in/input.xml\n"
            f"Location: {target_root}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n",
            encoding="utf-8",
        )

    build_dir = target_root / "build"
    in_dir = build_dir / "in"

    in_dir.mkdir(parents=True, exist_ok=True)

    dst_xml = in_dir / "input.xml"
    if not dst_xml.exists():
        shutil.copyfile(src_xml, dst_xml)
        print(f"Copied input.xml to {dst_xml}")

    print(f"Created directories and files in {build_dir}")


if __name__ == "__main__":
    main()
