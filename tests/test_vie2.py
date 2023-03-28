import unittest
import numpy as np
from numpy import sqrt, sin, sinh, cos, cosh
from numpy.testing import assert_array_almost_equal
from scipy.special import hyp0f1

from triqs.gf import MeshReTime

from tddt.vie2 import VIE2Solver


class test_vie2(unittest.TestCase):
    """Volterra integral equations of the 2nd kind"""

    def test_VIE2Solver_scalar(self):
        mesh = MeshReTime(0.0, 5.0, 501)

        solver = VIE2Solver(mesh, ())
        self.assertEqual(solver.N, 501)
        self.assertEqual(solver.solution_shape, ())
        self.assertEqual(solver.y_shape, (501,))
        self.assertEqual(solver.k_shape, (501, 501,))
        self.assertEqual(solver.startup_shape, (5,))

        t = np.array(list(map(float, mesh)))

        t1, t2 = np.meshgrid(t, t, indexing='ij')
        K = t1 ** 2 - t2 ** 2
        q = 1 + t ** 3

        y = solver(K, q)

        y_ref = 0.5 * (3 - hyp0f1(1 / 3, -2 * (t ** 3) / 9))
        assert_array_almost_equal(y, y_ref, decimal=10)

    def test_VIE2Solver_vector(self):
        mesh = MeshReTime(0.0, 5.0, 501)

        solver = VIE2Solver(mesh, (3,))
        self.assertEqual(solver.N, 501)
        self.assertEqual(solver.solution_shape, (3,))
        self.assertEqual(solver.y_shape, (501, 3))
        self.assertEqual(solver.k_shape, (501, 501, 3, 3))
        self.assertEqual(solver.startup_shape, (5, 3))

        t = np.array(list(map(float, mesh)))

        t1m, t2m, im, jm = np.meshgrid(t, t, range(3), range(3), indexing='ij')
        K = (im - jm) * (t1m - t2m)

        tm, im = np.meshgrid(t, range(3), indexing='ij')
        q = cos(im * tm)

        y = solver(K, q)

        # These solutions are obtained by means of the Laplace transform
        # (see doc/vie2.nb for details).
        x = 1.5 ** 0.25
        y_ref = np.array([
            (77 - 198 * cos(t) - 147 * cos(2 * t)
             + 730 * cos(x * t) * cosh(x * t)
             + 100 * sqrt(6) * sin(x * t) * sinh(x * t)) / 462,
            (-77 + 165 * cos(t) - 63 * cos(2 * t)
             + 206 * cos(x * t) * cosh(x * t)
             - 53 * sqrt(6) * sin(x * t) * sinh(x * t)) / 231,
            (77 - 66 * cos(t) + 357 * cos(2 * t)
             + 94 * cos(x * t) * cosh(x * t)
             - 312 * sqrt(6) * sin(x * t) * sinh(x * t)) / 462
        ])
        y_ref = np.moveaxis(y_ref, -1, 0)  # Move time axis to the front
        assert_array_almost_equal(y, y_ref, decimal=10)

    def test_VIE2Solver_matrix(self):
        mesh = MeshReTime(0.0, 5.0, 501)

        solver = VIE2Solver(mesh, (3, 2))
        self.assertEqual(solver.N, 501)
        self.assertEqual(solver.solution_shape, (3, 2))
        self.assertEqual(solver.y_shape, (501, 3, 2))
        self.assertEqual(solver.k_shape, (501, 501, 3, 2, 3, 2))
        self.assertEqual(solver.startup_shape, (5, 3, 2))

        t = np.array(list(map(float, mesh)))

        t1m, t2m, im, jm, km, lm = np.meshgrid(t, t,
                                               range(3), range(2),
                                               range(3), range(2),
                                               indexing='ij')
        K = (im - km) * (jm - lm) * (t1m - t2m)

        tm, im, jm = np.meshgrid(t, range(3), range(2), indexing='ij')
        q = cos((im - jm) * tm)

        y = solver(K, q)

        # These solutions are obtained by means of the Laplace transform
        # (see doc/vie2.nb for details).
        x = 6 ** 0.25
        y_ref = np.array([
            [(10 - 48 * cos(t) - 6 * cos(2 * t)
              + (52 + 21 * sqrt(6)) * cos(x * t)
              + (52 - 21 * sqrt(6)) * cosh(x * t)) / 60,
             (-5 + 6 * cos(t) - 12 * cos(2 * t)
              + (13 + sqrt(6)) * cos(x * t)
              - (-13 + sqrt(6)) * cosh(x * t)) / 15],
            [(-10 + 18 * cos(t) + 6 * cos(2 * t)
              - (-8 + sqrt(6)) * cos(x * t)
              + (8 + sqrt(6)) * cosh(x * t)) / 30,
             (20 - 24 * cos(t) - 12 * cos(2 * t)
              + (23 + 6 * sqrt(6)) * cos(x * t)
              + (23 - 6 * sqrt(6)) * cosh(x * t)) / 30],
            [(2 + 18 * cos(2 * t) - (4 + 5 * sqrt(6)) * cos(x * t)
              + (-4 + 5 * sqrt(6)) * cosh(x * t)) / 12,
             (-1 + (2 + sqrt(6)) * cos(x * t)
              - (-2 + sqrt(6)) * cosh(x * t)) / 3]
        ])
        y_ref = np.moveaxis(y_ref, -1, 0)  # Move time axis to the front
        assert_array_almost_equal(y, y_ref, decimal=10)
