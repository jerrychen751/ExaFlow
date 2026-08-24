from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="exaflow",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.13",
    install_requires=[
        "numpy",
        "mpi4py",
        "xmltodict",
    ],
    extras_require={
        "gui": [
            "pyside6",
            "pyvista",
            "pyvistaqt",
            "vtk",
            "Pillow",
        ],
        "io": [
            "pyevtk",
        ],
    },
)
