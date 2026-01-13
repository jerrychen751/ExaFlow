from __future__ import annotations

import unittest

import numpy as np

from ._bootstrap import ensure_src_on_path


ensure_src_on_path()

from navier_stokes_solver.initial_conditions import initialize_fields
from navier_stokes_solver.parameters import SimulationParameters


class TestInitialConditions(unittest.TestCase):
    def test_initialize_uniform_and_step_values(self) -> None:
        sim = SimulationParameters(
            rho=1.0,
            nu=0.0,
            domain=(10, 10, 10),
            size=(1.0, 1.0, 1.0),
            nt=1,
            num_ghost_layers=1,
            cfl=0.25,
            comm=None,
            include_pressure=False,
            initial_conditions={
                "ReadFromVtrFile": "False",
                "ReadFromCsvFile": "False",
                "SpecifyValues": "True",
                "SpecifiedValues": {
                    "p": {
                        "UseUniform": "True",
                        "UniformParameters": {"ConstantValue": "1"},
                        "UseStep": "False",
                        "UseSinusoidal": "False",
                        "UsePolynomial": "False",
                        "UseGaussian": "False",
                    },
                    "u": {
                        "UseUniform": "True",
                        "UniformParameters": {"ConstantValue": "2"},
                        "UseStep": "True",
                        "StepParameters": {
                            "StepMagnitude": "3",
                            "startX": "0.4",
                            "endX": "0.6",
                            "startY": "0.4",
                            "endY": "0.6",
                            "startZ": "0.4",
                            "endZ": "0.6",
                        },
                        "UseSinusoidal": "False",
                        "UsePolynomial": "False",
                        "UseGaussian": "False",
                    },
                    "v": {
                        "UseUniform": "True",
                        "UniformParameters": {"ConstantValue": "0"},
                        "UseStep": "False",
                        "UseSinusoidal": "False",
                        "UsePolynomial": "False",
                        "UseGaussian": "False",
                    },
                    "w": {
                        "UseUniform": "True",
                        "UniformParameters": {"ConstantValue": "0"},
                        "UseStep": "False",
                        "UseSinusoidal": "False",
                        "UsePolynomial": "False",
                        "UseGaussian": "False",
                    },
                },
            },
        )

        p, u, v, w = initialize_fields(sim)  # type: ignore[misc]
        self.assertEqual(p.shape, (10, 10, 10))
        self.assertTrue(np.isclose(p.min(), 1.0))
        self.assertTrue(np.isclose(p.max(), 1.0))
        self.assertTrue(np.isclose(u.min(), 2.0))
        self.assertGreater(float(u.max()), 2.0)

