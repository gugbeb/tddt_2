import unittest
import numpy as np
from numpy import sqrt, exp, sin, sinh, cos, cosh
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
        self.assertEqual(solver.shape, ())
        self.assertEqual(solver.f1_shape, (501,))
        self.assertEqual(solver.f2_shape, (501, 501,))
        self.assertEqual(solver._startup_shape(0), (5,))
        self.assertEqual(solver._startup_shape(1), (4,))

        t = np.fromiter(mesh, float)

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
        self.assertEqual(solver.shape, (3,))
        self.assertEqual(solver.f1_shape, (501, 3))
        self.assertEqual(solver.f2_shape, (501, 501, 3, 3))
        self.assertEqual(solver._startup_shape(0), (5, 3))
        self.assertEqual(solver._startup_shape(1), (4, 3))

        t = np.fromiter(mesh, float)

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
        self.assertEqual(solver.shape, (3, 2))
        self.assertEqual(solver.f1_shape, (501, 3, 2))
        self.assertEqual(solver.f2_shape, (501, 501, 3, 2, 3, 2))
        self.assertEqual(solver._startup_shape(0), (5, 3, 2))
        self.assertEqual(solver._startup_shape(1), (4, 3, 2))

        t = np.fromiter(mesh, float)

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

    def test_VIE2Solver_causal_scalar(self):
        mesh = MeshReTime(0.0, 5.0, 101)

        solver = VIE2Solver(mesh, ())

        t = np.fromiter(mesh, float)

        t1, t2 = np.meshgrid(t, t, indexing='ij')
        f = -1j * exp(-1j * (t1 - t2))
        q = -1j * exp(-2j * (t1 - t2))

        g = solver.solve_causal(f, q)

        g_ref = -1j * cos(t1 - t2) * exp(-1j * (t1 - t2))
        assert_array_almost_equal(g, g_ref, decimal=10)

    def test_VIE2Solver_causal_matrix(self):
        mesh = MeshReTime(0.0, 5.0, 101)

        solver = VIE2Solver(mesh, (3,))

        t = np.fromiter(mesh, float)
        t1, t2 = np.meshgrid(t, t, indexing='ij')

        H = np.array([[1, 0.5, 0],
                      [0.5, 2, 0.5],
                      [0, 0.5, 3]])
        E, U = np.linalg.eig(H)

        shape = (len(mesh), len(mesh), 3, 3)
        f = np.empty(shape, dtype=complex)
        q = np.empty(shape, dtype=complex)
        for i, j in np.ndindex(shape[:2]):
            f[i, j, ...] = -1j * U @ np.diag(exp(-1j * E * (t[i] - t[j]))) @ U.T
            q[i, j, ...] = -1j * U @ np.diag(exp(-2j * E * (t[i] - t[j]))) @ U.T

        g = solver.solve_causal(f, q)

        # These solutions are obtained by means of the Laplace transform
        # (see doc/vie2.nb for details).
        dt = t1 - t2
        s23 = sqrt(2 / 3)
        s32 = sqrt(3 / 2)
        g_ref = np.multiply.outer(1j / 18 * (exp(-1j * dt) + 2 * exp(-4j * dt)),
                                  np.array([[-1, -2, 1],
                                            [-2, -4, 2],
                                            [1, 2, -1]]))
        g_ref += np.multiply.outer(exp(-4j * dt) * sin(sqrt(6) * dt) * s23 / 30,
                                   np.array([[13, -7, -1],
                                             [-7, -2, -11],
                                             [-1, -11, -23]]))
        g_ref += np.multiply.outer(exp(-4j * dt) * cos(sqrt(6) * dt) * 1j / 30,
                                   np.array([[-11, 4, -3],
                                             [4, -6, -8],
                                             [-3, -8, -19]]))
        g_ref += np.multiply.outer(exp(-1j * dt) * sin(s32 * dt) * s23 / 30,
                                   np.array([[17, -8, 1],
                                             [-8, 2, -4],
                                             [1, -4, -7]]))
        g_ref += np.multiply.outer(exp(-1j * dt) * cos(s32 * dt) * 1j / 30,
                                   np.array([[-14, 6, -2],
                                             [6, -4, -2],
                                             [-2, -2, -6]]))

        # Move time axes to the front
        assert_array_almost_equal(g, g_ref, decimal=7)


if __name__ == '__main__':
    unittest.main()
