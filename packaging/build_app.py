from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def render_icon_image(size: int):
    """
    Draw the ExaFlow icon at the given pixel size and return the RGBA image. The icon is a rounded square with a vertical blue gradient. The drawing runs at four times the requested size and is then reduced, so the rounded corners stay smooth.
    """

    from PIL import Image, ImageDraw

    supersample = 4
    canvas = size * supersample
    scale = canvas / 1024.0

    gradient = Image.new("RGBA", (1, 1024))
    for y in range(1024):
        t = y / 1023.0
        gradient.putpixel(
            (0, y),
            (
                int(round(30 + (56 - 30) * t)),
                int(round(86 + (170 - 86) * t)),
                int(round(184 + (214 - 184) * t)),
                255,
            ),
        )
    gradient = gradient.resize((canvas, canvas), Image.Resampling.BILINEAR)

    inset = 100.0 * scale
    mask = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (inset, inset, canvas - inset - 1, canvas - inset - 1),
        radius=184.0 * scale,
        fill=255,
    )
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    image.paste(gradient, (0, 0), mask)

    return image.resize((size, size), Image.Resampling.LANCZOS)


def write_icon_file(icon_path: Path) -> None:
    """
    Write a macOS .icns file. iconutil reads an .iconset directory, so the sizes are rendered into a temporary directory first.
    """

    with tempfile.TemporaryDirectory() as temporary_dir:
        iconset_dir = Path(temporary_dir) / "ExaFlow.iconset"
        iconset_dir.mkdir()
        for size in (16, 32, 64, 128, 256, 512, 1024):
            image = render_icon_image(size)
            image.save(iconset_dir / f"icon_{size}x{size}.png")
            if size <= 512:
                render_icon_image(size * 2).save(iconset_dir / f"icon_{size}x{size}@2x.png")
        subprocess.run(
            ["iconutil", "--convert", "icns", "--output", str(icon_path), str(iconset_dir)],
            check=True,
        )


def copy_open_mpi(venv_root: Path, app_path: Path) -> int:
    """
    Copy Open MPI into Contents/Resources/mpi and return the number of files copied. The file list comes from the openmpi wheel's own RECORD, and the bin/, lib/, etc/ and share/ layout is kept, because prterun finds its libraries through an @executable_path/../lib/ run path. packaging/entry.py points OPAL_PREFIX at the copied directory.
    """

    site_packages = next(venv_root.glob("lib/python3.*/site-packages"))
    record_files = sorted(site_packages.glob("openmpi-*.dist-info/RECORD"))
    if not record_files:
        raise FileNotFoundError(f"No openmpi wheel is installed in {venv_root}. Run `uv sync` first.")

    mpi_root = app_path / "Contents" / "Resources" / "mpi"
    real_venv_root = Path(os.path.realpath(venv_root))
    copied = 0
    for line in record_files[0].read_text().splitlines():
        relative_path = line.split(",")[0]
        if not relative_path.startswith(".."):
            continue
        source = Path(os.path.realpath(site_packages / relative_path))
        if not source.is_file():
            continue
        if not source.is_relative_to(real_venv_root):
            raise RuntimeError(f"{record_files[0]} lists {relative_path}, which resolves to {source} outside {real_venv_root}. Set UV_LINK_MODE to hardlink or copy, then run `uv sync --reinstall`.")
        destination = mpi_root / source.relative_to(real_venv_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    if not (mpi_root / "lib" / "libmpi.40.dylib").is_file():
        raise FileNotFoundError(f"{record_files[0]} put no lib/libmpi.40.dylib under {mpi_root}. packaging/entry.py sets MPI4PY_LIBMPI to that exact path, and mpi4py loads no MPI library without it.")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the standalone ExaFlow.app bundle.")
    parser.add_argument("--keep-icon", action="store_true", help="Reuse packaging/ExaFlow.icns instead of drawing it again.")
    args = parser.parse_args()

    packaging_dir = Path(__file__).resolve().parent
    repo_root = packaging_dir.parent
    venv_root = repo_root / ".venv"
    if not venv_root.is_dir():
        raise FileNotFoundError(f"No virtual environment at {venv_root}. Run `uv sync` first.")

    icon_path = packaging_dir / "ExaFlow.icns"
    if not args.keep_icon or not icon_path.exists():
        write_icon_file(icon_path)
        print(f"Wrote {icon_path}")

    build_dir = repo_root / "build"
    dist_dir = repo_root / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(build_dir),
            str(packaging_dir / "ExaFlow.spec"),
        ],
        cwd=repo_root,
        check=True,
    )

    app_path = dist_dir / "ExaFlow.app"
    if not app_path.is_dir():
        raise FileNotFoundError(f"PyInstaller wrote no bundle at {app_path}.")

    try:
        copied = copy_open_mpi(venv_root, app_path)
        print(f"Copied {copied} Open MPI files into {app_path.name}/Contents/Resources/mpi")

        library_paths = [path for path in app_path.rglob("*") if path.is_file() and not path.is_symlink() and path.suffix in (".dylib", ".so")]
        library_sizes = []
        expected_warnings = ("will invalidate the code signature", "replacing existing signature")
        for command in (["strip", "-x"], ["codesign", "--force", "--sign", "-"]):
            library_sizes.append(sum(path.stat().st_size for path in library_paths))
            result = subprocess.run([*command, *library_paths], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{command[0]} failed on the {len(library_paths)} libraries of the bundle: {result.stderr.strip()}")
            for line in result.stderr.splitlines():
                if not any(warning in line for warning in expected_warnings):
                    print(line)
        size_before_strip, size_after_strip = library_sizes
        print(f"Stripped {len(library_paths)} .dylib and .so files and removed {(size_before_strip - size_after_strip) / 1e6:.0f} MB")

        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app_path)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app_path)], check=True)
        print(f"Re-signed {app_path.name}; the Open MPI copy and the strip both invalidated the PyInstaller signature")

        build_dir.mkdir(parents=True, exist_ok=True)
        smoke_path = build_dir / "smoke_imports.py"
        smoke_path.write_text(
            "import os\n"
            "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
            "import mpi4py\n"
            "mpi4py.rc.initialize = False\n"
            "import exaflow.gui.app, exaflow.numerics.pressure_poisson\n"
            "import PySide6.QtWidgets, pyvista, pyvistaqt, scipy.sparse, vtk, vtkmodules.all\n"
            "from mpi4py import MPI\n"
            "PySide6.QtWidgets.QApplication([])\n"
            "print('The bundle imports scipy, VTK, PySide6 and', MPI.Get_library_version().split(',')[0])\n"
        )
        hostile_environment = {name: os.environ[name] for name in ("HOME", "TMPDIR", "USER", "LANG") if name in os.environ}
        hostile_environment["PATH"] = "/usr/bin:/bin"
        hostile_environment["MPI4PY_MPIABI"] = "mpich"
        subprocess.run([str(app_path / "Contents" / "MacOS" / "ExaFlow"), "--run-script", str(smoke_path)], check=True, timeout=600, env=hostile_environment)
    except BaseException:
        shutil.rmtree(app_path, ignore_errors=True)
        raise

    size_output = subprocess.run(["du", "-sh", str(app_path)], capture_output=True, text=True, check=True)
    print(f"Built {app_path} ({size_output.stdout.split()[0]})")
    print("Install it with: rm -rf /Applications/ExaFlow.app.new && cp -R dist/ExaFlow.app /Applications/ExaFlow.app.new && rm -rf /Applications/ExaFlow.app && mv /Applications/ExaFlow.app.new /Applications/ExaFlow.app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
