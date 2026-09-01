from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from exaflow.config import Case, Fluid, Grid, TimeControl
from exaflow.config.case_xml import write_case


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


def build_bundle(repo_root: Path, keep_icon: bool) -> Path:
    """
    Build dist/ExaFlow.app and return the path to it. Deletes the bundle and raises again when a step after PyInstaller fails, so a bundle that the checks reject never stays in dist/. Copies Open MPI out of .venv, so that environment has to exist.
    """

    packaging_dir = repo_root / "packaging"
    venv_root = repo_root / ".venv"
    if not venv_root.is_dir():
        raise FileNotFoundError(f"No virtual environment at {venv_root}. Run `uv sync` first.")

    icon_path = packaging_dir / "ExaFlow.icns"
    if not keep_icon or not icon_path.exists():
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

        hostile_environment = {name: os.environ[name] for name in ("HOME", "TMPDIR", "USER", "LANG") if name in os.environ}
        hostile_environment["PATH"] = "/usr/bin:/bin"
        hostile_environment["MPI4PY_MPIABI"] = "mpich"
        executable_path = app_path / "Contents" / "MacOS" / "ExaFlow"
        subprocess.run([str(executable_path), "--check-bundle"], check=True, timeout=600, env=hostile_environment)

        with tempfile.TemporaryDirectory(prefix="exaflow-bundle-smoke-") as smoke_directory:
            smoke_root = Path(smoke_directory)
            case_path = smoke_root / "case.xml"
            case_path.write_text(
                write_case(
                    Case(
                        fluid=Fluid(1.225, 0.3),
                        grid=Grid((4, 4, 4), (1.0, 1.0, 1.0), 1),
                        time=TimeControl(1, 0.25, 1),
                    )
                ),
                encoding="utf-8",
            )
            run_environment = dict(
                hostile_environment,
                EXAFLOW_OUTPUT_ROOT=str(smoke_root / "runs"),
            )
            subprocess.run(
                [str(executable_path), "run", "--case", str(case_path)],
                check=True,
                timeout=600,
                env=run_environment,
            )
    except BaseException:
        shutil.rmtree(app_path, ignore_errors=True)
        raise

    size_output = subprocess.run(["du", "-sh", str(app_path)], capture_output=True, text=True, check=True)
    print(f"Built {app_path} ({size_output.stdout.split()[0]})")
    return app_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ExaFlow.app and install it as /Applications/ExaFlow.app.")
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument("--build-only", action="store_true", help="Build the bundle into dist/ and install nothing.")
    stage_group.add_argument("--install-only", action="store_true", help="Install the bundle already in dist/ instead of building it again.")
    parser.add_argument("--keep-icon", action="store_true", help="Reuse packaging/ExaFlow.icns instead of drawing it again.")
    parser.add_argument("--quit-app", action="store_true", help="Quit the running ExaFlow, which ends the simulation it runs.")
    parser.add_argument("--destination", type=Path, default=Path("/Applications/ExaFlow.app"), help="Install the bundle here instead of /Applications/ExaFlow.app.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source_path = repo_root / "dist" / "ExaFlow.app"
    if not args.install_only:
        source_path = build_bundle(repo_root, args.keep_icon)
    if args.build_only:
        return 0
    if not source_path.is_dir():
        raise FileNotFoundError(f"No bundle at {source_path}. Run this script without --install-only to build one.")

    destination_path = args.destination.expanduser().resolve()
    executable_pattern = str(destination_path / "Contents" / "MacOS")
    running = subprocess.run(["pgrep", "-f", executable_pattern], capture_output=True, text=True).stdout.split()
    if running:
        if not args.quit_app:
            raise RuntimeError(f"{destination_path} runs as process {', '.join(running)}. Quit it, or pass --quit-app, because the install deletes the bundle that the running app reads its own Python code, Qt plugins and MPI libraries from.")
        subprocess.run(["osascript", "-e", f'tell application "{destination_path}" to quit'], check=True)
        deadline = time.monotonic() + 30.0
        while subprocess.run(["pgrep", "-f", executable_pattern], capture_output=True, text=True).stdout.split():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{destination_path} still runs 30 seconds after the quit. Close its windows, then run this script again.")
            time.sleep(0.5)

    staging_path = destination_path.with_name(f"{destination_path.name}.new")
    shutil.rmtree(staging_path, ignore_errors=True)
    subprocess.run(["ditto", str(source_path), str(staging_path)], check=True)
    shutil.rmtree(destination_path, ignore_errors=True)
    staging_path.rename(destination_path)

    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(destination_path)], check=True)
    size_output = subprocess.run(["du", "-sh", str(destination_path)], capture_output=True, text=True, check=True)
    print(f"Installed {destination_path} ({size_output.stdout.split()[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
