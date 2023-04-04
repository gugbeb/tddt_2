import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshProduct, MeshBrillouinZone
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import Branch, KeldyshGF
from tddt.diagrams import polarization_2nd_order
from tddt.integration import GregoryIntegrator
from tddt.testing import assert_keldysh_gf_almost_equal

FW, BW = Branch.FORWARD, Branch.BACKWARD


class test_diagrams(unittest.TestCase):
    """Diagrams on Keldysh contour"""

    @classmethod
    def setUpClass(cls):
        cls.n_t = (3, 5, 7, 9)
        cls.t_ranges = [range(n) for n in cls.n_t]
        cls.t_mesh = [MeshReTime(0.0, 2.0, 3),
                      MeshReTime(1.0, 3.0, 5),
                      MeshReTime(2.0, 4.0, 7),
                      MeshReTime(0.0, 4.0, 9)]

        bl = BravaisLattice(units=[(1, 0, 0)])
        cls.n_k = 3
        cls.bz_mesh = MeshBrillouinZone(BrillouinZone(bl), cls.n_k)

        # Use a lower order quadrature rule so that is is compatible with small
        # time meshes.
        KeldyshGF.integrator = GregoryIntegrator(2)

    def _make_test_keldysh_gf(self, mesh, x, target_subshapes=None):
        g = KeldyshGF(mesh=mesh, target_subshapes=target_subshapes)
        for n, (b0, b1) in enumerate(product(Branch, repeat=2)):
            g_comp = g[b0, b1]
            s = g_comp.data.size
            g_comp.data[:] = x * np.arange(s).reshape(g_comp.data.shape) + n
        return g

    def _make_test_keldysh_vertex3(self, mesh, x, target_subshapes=None):
        Lambda = KeldyshGF(mesh=mesh, target_subshapes=target_subshapes)
        for n, (b0, b1, b2) in enumerate(product(Branch, repeat=3)):
            l_comp = Lambda[b0, b1, b2]
            s = l_comp.data.size
            l_comp.data[:] = x * np.arange(s).reshape(l_comp.data.shape) + n
        return Lambda

    def test_polarization_2nd_order_scalar(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(Lambda, g)

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1], self.t_mesh[1]))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n in product(*[self.t_ranges[1]] * 2,
                                            *[self.t_ranges[0]] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j] += -1j * sign * \
                    Lambda[b2, b3, b0].data[k, l, i] * g[b5, b2].data[n, k] * \
                    W[k] * W[l] * W[m] * W[n] * \
                    g[b3, b4].data[l, m] * Lambda[b4, b5, b1].data[m, n, j]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_matrix(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh,
                                                 1.0,
                                                 ((1,), (1,), (3,)))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, ((1,), (1,)))

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(Lambda, g)

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1]),
                           target_subshapes=((3,), (3,)))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, x, y, w1, w2, w3, w4 in product(
                    *[self.t_ranges[1]] * 2,
                    *[self.t_ranges[0]] * 4,
                    *[range(3)] * 2, *[range(1)] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, x, y] += -1j * sign *\
                    Lambda[b2, b3, b0].data[k, l, i, w1, w2, x] *\
                    g[b5, b2].data[n, k, w4, w1] *\
                    W[k] * W[l] * W[m] * W[n] *\
                    g[b3, b4].data[l, m, w2, w3] *\
                    Lambda[b4, b5, b1].data[m, n, j, w3, w4, y]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_scalar_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(Lambda, g)

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1],
                                            self.bz_mesh))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K in product(*[self.t_ranges[1]] * 2,
                                               *[self.t_ranges[0]] * 4,
                                               range(self.n_k)):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, K] += -1j * sign * \
                    Lambda[b2, b3, b0].data[k, l, i] * \
                    g[b5, b2].data[n, k, K] * \
                    W[k] * W[l] * W[m] * W[n] * \
                    g[b3, b4].data[l, m, K] * \
                    Lambda[b4, b5, b1].data[m, n, j]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_matrix_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh,
                                                 1.0,
                                                 ((1,), (1,), (3,)))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, ((1,), (1,)))

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(Lambda, g)

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1],
                                            self.bz_mesh),
                           target_subshapes=((3,), (3,)))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K, x, y, w1, w2, w3, w4 in product(
                    *[self.t_ranges[1]] * 2,
                    *[self.t_ranges[0]] * 4,
                    range(self.n_k),
                    *[range(3)] * 2, *[range(1)] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, x, y, K] += -1j * sign *\
                    Lambda[b2, b3, b0].data[k, l, i, w1, w2, x] *\
                    g[b5, b2].data[n, k, K, w4, w1] *\
                    W[k] * W[l] * W[m] * W[n] *\
                    g[b3, b4].data[l, m, K, w2, w3] *\
                    Lambda[b4, b5, b1].data[m, n, j, w3, w4, y]

        assert_keldysh_gf_almost_equal(pi, pi_ref)


if __name__ == '__main__':
    unittest.main()
