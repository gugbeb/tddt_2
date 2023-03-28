import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from triqs.gf import MeshReTime

from tddt.integration import (simpsons_weights,
                              gregory_coefficients,
                              GregoryIntegrator)


class test_integration(unittest.TestCase):
    """Integration tools"""

    mesh = MeshReTime(1.0, 3.4, 13)

    def test_simpsons_weights(self):
        w = simpsons_weights(self.mesh)
        w_ref = 0.2 * np.array([1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1]) / 3
        assert_array_almost_equal(w, w_ref)

    def test_gregory_coefficients(self):
        with self.assertRaises(AssertionError):
            gregory_coefficients(0)

        w = gregory_coefficients(6)
        w_ref = [1 / 2, -1 / 12, 1 / 24, -19 / 720, 3 / 160, -863 / 60480]
        assert_array_almost_equal(w, w_ref)

    def _make_gregory_w_ref(self, s_ref, Sigma_ref):
        size = len(self.mesh)
        b_size = s_ref.shape[0]

        omega = Sigma_ref[-1, :]
        b_ll = np.block([[Sigma_ref]] + [[omega]] * (size - 2 * b_size))
        b_lr = np.zeros((size - b_size, size - b_size))
        for i in range(size - b_size):
            for j in range(i + 1):
                b_lr[i, j] = omega[i - j] if (i - j <= b_size - 1) else 1

        return np.block([[s_ref, np.zeros((b_size, size - b_size))],
                         [b_ll, b_lr]]) * self.mesh.delta

    def test_GregoryIntegrator0(self):
        g = GregoryIntegrator(0)
        self.assertEqual(g.order, 0)

        s_ref = np.array([[0]])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([[0.5]])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator1(self):
        g = GregoryIntegrator(1)
        self.assertEqual(g.order, 1)

        s_ref = np.array([[0, 0],
                          [0.5, 0.5]])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([[5 / 12, 7 / 6],
                              [5 / 12, 13 / 12]])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator2(self):
        g = GregoryIntegrator(2)
        self.assertEqual(g.order, 2)

        s_ref = np.array([[0, 0, 0],
                          [5 / 12, 2 / 3, -1 / 12],
                          [1 / 3, 4 / 3, 1 / 3]])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([[3 / 8, 9 / 8, 9 / 8],
                              [3 / 8, 7 / 6, 11 / 12],
                              [3 / 8, 7 / 6, 23 / 24]])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator3(self):
        g = GregoryIntegrator(3)
        self.assertEqual(g.order, 3)

        s_ref = np.array([[0, 0, 0, 0],
                          [3 / 8, 19 / 24, -5 / 24, 1 / 24],
                          [1 / 3, 4 / 3, 1 / 3, 0],
                          [3 / 8, 9 / 8, 9 / 8, 3 / 8]])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([[251 / 720, 229 / 180, 91 / 120, 229 / 180],
                              [251 / 720, 299 / 240, 163 / 180, 163 / 180],
                              [251 / 720, 299 / 240, 211 / 240, 379 / 360],
                              [251 / 720, 299 / 240, 211 / 240, 739 / 720]])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator4(self):
        g = GregoryIntegrator(4)
        self.assertEqual(g.order, 4)

        s_ref = np.array([
            [0, 0, 0, 0, 0],
            [251 / 720, 323 / 360, -11 / 30, 53 / 360, -19 / 720],
            [29 / 90, 62 / 45, 4 / 15, 2 / 45, -1 / 90],
            [27 / 80, 51 / 40, 9 / 10, 21 / 40, -3 / 80],
            [14 / 45, 64 / 45, 8 / 15, 64 / 45, 14 / 45]
        ])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([
            [95 / 288, 125 / 96, 125 / 144, 125 / 144, 125 / 96],
            [95 / 288, 317 / 240, 359 / 480, 433 / 360, 359 / 480],
            [95 / 288, 317 / 240, 23 / 30, 1559 / 1440, 1559 / 1440],
            [95 / 288, 317 / 240, 23 / 30, 793 / 720, 77 / 80],
            [95 / 288, 317 / 240, 23 / 30, 793 / 720, 157 / 160]
        ])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator5(self):
        g = GregoryIntegrator(5)
        self.assertEqual(g.order, 5)

        s_ref = np.array([
            [0, 0, 0, 0, 0, 0],
            [95 / 288, 1427 / 1440, -133 / 240, 241 / 720, -173 / 1440,
             3 / 160],
            [14 / 45, 43 / 30, 7 / 45, 7 / 45, -1 / 15, 1 / 90],
            [51 / 160, 219 / 160, 57 / 80, 57 / 80, -21 / 160, 3 / 160],
            [14 / 45, 64 / 45, 8 / 15, 64 / 45, 14 / 45, 0],
            [95 / 288, 125 / 96, 125 / 144, 125 / 144, 125 / 96, 95 / 288]
        ])
        assert_array_almost_equal(g.s, s_ref)
        Sigma_ref = np.array([
            [19087 / 60480, 14177 / 10080, 10763 / 20160, 22501 / 15120,
             10763 / 20160, 14177 / 10080],
            [19087 / 60480, 84199 / 60480, 4289 / 6720, 69793 / 60480,
             69793 / 60480, 4289 / 6720],
            [19087 / 60480, 84199 / 60480, 18869 / 30240, 15221 / 12096,
             24791 / 30240, 15221 / 12096],
            [19087 / 60480, 84199 / 60480, 18869 / 30240, 37621 / 30240,
             27947 / 30240, 27947 / 30240],
            [19087 / 60480, 84199 / 60480, 18869 / 30240, 37621 / 30240,
             55031 / 60480, 31103 / 30240],
            [19087 / 60480, 84199 / 60480, 18869 / 30240, 37621 / 30240,
             55031 / 60480, 61343 / 60480]
        ])

        w_ref = self._make_gregory_w_ref(s_ref, Sigma_ref)
        w = g.weights(self.mesh)
        assert_array_almost_equal(w, w_ref)

        w_conv = g.weights_conv(self.mesh)
        assert_array_almost_equal(w_conv, w[-1, :])

    def test_GregoryIntegrator_small_meshes(self):
        g = GregoryIntegrator(2)

        mesh_min = MeshReTime(1.0, 2.0, 3)
        assert_array_almost_equal(g.weights(mesh_min),
                                  mesh_min.delta * g.s)
        assert_array_almost_equal(g.weights_conv(mesh_min),
                                  mesh_min.delta * g.s[-1, :])

        mesh_min_plus = MeshReTime(1.0, 2.0, 4)
        w_ref = np.array([[0, 0, 0, 0],
                          [5 / 12, 2 / 3, -1 / 12, 0],
                          [1 / 3, 4 / 3, 1 / 3, 0],
                          [3 / 8, 9 / 8, 9 / 8, 3 / 8]])
        w_ref *= mesh_min_plus.delta
        assert_array_almost_equal(g.weights(mesh_min_plus), w_ref)
        assert_array_almost_equal(g.weights_conv(mesh_min_plus), w_ref[-1, :])

        mesh_min2 = MeshReTime(1.0, 2.0, 6)
        w_ref = np.array([[0, 0, 0, 0, 0, 0],
                          [5 / 12, 2 / 3, -1 / 12, 0, 0, 0],
                          [1 / 3, 4 / 3, 1 / 3, 0, 0, 0],
                          [3 / 8, 9 / 8, 9 / 8, 3 / 8, 0, 0],
                          [3 / 8, 7 / 6, 11 / 12, 7 / 6, 3 / 8, 0],
                          [3 / 8, 7 / 6, 23 / 24, 23 / 24, 7 / 6, 3 / 8]])
        w_ref *= mesh_min2.delta
        assert_array_almost_equal(g.weights(mesh_min2), w_ref)
        assert_array_almost_equal(g.weights_conv(mesh_min2), w_ref[-1, :])


if __name__ == '__main__':
    unittest.main()
