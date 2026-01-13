from __future__ import annotations

import tempfile
import unittest

import numpy as np

from ._bootstrap import ensure_src_on_path


ensure_src_on_path()

from navier_stokes_solver.io.csv import write_csv, write_total_array_to_csv
from navier_stokes_solver.parameters import SimulationParameters


class TestCsvWriters(unittest.TestCase):
    def test_write_total_array_to_csv_1d(self) -> None:
        u = np.array([1.0, 2.0, 3.0], dtype=float)
        p = np.array([0.0, 0.0, 0.0], dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            write_total_array_to_csv("Test", p=p, u=u, out_dir=tmp)
            with open(f"{tmp}/Test_Total.csv", "r", encoding="utf-8") as f:
                text = f.read()
        self.assertTrue(text.startswith("x,u,p"))
        self.assertIn("2, 3.0, 0.0", text)

    def test_write_csv_1d_local(self) -> None:
        sim = SimulationParameters(
            rho=1.0,
            nu=0.0,
            domain=(5,),
            size=(1.0,),
            nt=1,
            num_ghost_layers=1,
            cfl=0.25,
            comm=None,
            include_pressure=False,
        )
        u_local = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
        p_local = np.zeros_like(u_local)

        with tempfile.TemporaryDirectory() as tmp:
            write_csv(0, sim, "Step0", u=u_local, p=p_local, out_dir=tmp)
            with open(f"{tmp}/Step0_0.csv", "r", encoding="utf-8") as f:
                text = f.read()
        self.assertTrue(text.startswith("x,u,p"))

