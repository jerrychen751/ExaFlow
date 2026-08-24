from __future__ import annotations

import argparse
from datetime import datetime
import os
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Build directory with templates")
    parser.add_argument("target_dir", help="Folder where Build/ will be created")
    args = parser.parse_args()

    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "examples")
    src_xml = os.path.join(examples_dir, "input_template.xml")
    src_py = os.path.join(examples_dir, "run_simulation_template.py")
    for template_path in (src_xml, src_py):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {os.path.normpath(template_path)}")

    target_root = os.path.abspath(args.target_dir)
    simulations_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(target_root) != simulations_dir:
        raise ValueError(f"target_dir must be directly inside {simulations_dir}, because run_simulation.py resolves the repo root as its fourth parent; got {target_root}.")

    os.makedirs(target_root, exist_ok=True)

    metadata_path = os.path.join(target_root, "metadata.txt")
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as meta_file:
            meta_file.write("Do not modify this file\n")
            meta_file.write("Created by create_simulation_directories.py\n")
            meta_file.write("Contains Build/ directory with in/, out/, and run_simulation.py\n")
            meta_file.write("in/ has Input.xml where you specify simulation parameters\n")
            meta_file.write("out/ is where output CSVs and VTK files will be written\n")
            meta_file.write("run_simulation.py is the script to run the simulation\n")
            meta_file.write(f"Location: {target_root}\n")
            meta_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")

    build_dir = os.path.join(target_root, "Build")
    in_dir = os.path.join(build_dir, "in")
    out_dir = os.path.join(build_dir, "out")

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    dst_xml = os.path.join(in_dir, "Input.xml")
    if not os.path.exists(dst_xml):
        shutil.copyfile(src_xml, dst_xml)
        print(f"Copied Input.xml to {dst_xml}")

    dst_py = os.path.join(build_dir, "run_simulation.py")
    if not os.path.exists(dst_py):
        shutil.copyfile(src_py, dst_py)
        print(f"Copied run_simulation.py to {dst_py}")

    print(f"Created directories and files in {build_dir}")


if __name__ == "__main__":
    main()
