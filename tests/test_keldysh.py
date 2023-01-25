import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, GfReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.gf.descriptors import Function
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import (Branch,
                          ContourPoint,
                          contour_ordering2,
                          contour_ordering3,
                          KeldyshGF,
                          KeldyshVertex3)
from tddt.util import simpsons_weights

CP = ContourPoint
FW, BW = Branch.FORWARD, Branch.BACKWARD


class test_keldysh(unittest.TestCase):
    """Keldysh Green's functions and vertices"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 6.0
        cls.n_t = 7
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)
        cls.tt_mesh = MeshProduct(cls.t_mesh, cls.t_mesh)
        cls.ttt_mesh = MeshProduct(cls.t_mesh, cls.t_mesh, cls.t_mesh)
        cls.t_points = list(cls.t_mesh)

    def test_contour_ordering2(self):
        t1 = self.t_points[2]
        t2 = self.t_points[3]
        order = contour_ordering2

        # (+,+)
        self.assertEqual(order(CP(FW, t2), CP(FW, t1)), (0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t1)), (0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2)), (1, 0))
        # (-,-)
        self.assertEqual(order(CP(BW, t1), CP(BW, t2)), (0, 1))
        self.assertEqual(order(CP(BW, t1), CP(BW, t1)), (1, 0))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1)), (1, 0))
        # (-,+)
        self.assertEqual(order(CP(BW, t1), CP(FW, t2)), (0, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t1)), (0, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1)), (0, 1))
        # (+,-)
        self.assertEqual(order(CP(FW, t1), CP(BW, t2)), (1, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t1)), (1, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1)), (1, 0))

    def test_contour_ordering3(self):
        t1 = self.t_points[2]
        t2 = self.t_points[3]
        t3 = self.t_points[4]
        order = contour_ordering3

        # (+,+,+)
        self.assertEqual(order(CP(FW, t3), CP(FW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(FW, t3), CP(FW, t1), CP(FW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t3)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(FW, t3), CP(FW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(FW, t3), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t3)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t2), CP(FW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(FW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(FW, t1)), (0, 1, 2))
        # (-,-,-)
        self.assertEqual(order(CP(BW, t3), CP(BW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(BW, t3), CP(BW, t1), CP(BW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t3)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t3), CP(BW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(BW, t3), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t3)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(BW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(BW, t1)), (2, 1, 0))
        # (+,+,-)
        self.assertEqual(order(CP(FW, t3), CP(FW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t3), CP(FW, t1), CP(BW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t3)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(FW, t3), CP(BW, t2)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(FW, t3), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t3)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t2), CP(FW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(BW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t2)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(BW, t1)), (2, 0, 1))
        # (-,-,+)
        self.assertEqual(order(CP(BW, t3), CP(BW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t3), CP(BW, t1), CP(FW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t3)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t3), CP(FW, t2)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t3), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t3)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(FW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t2)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(FW, t1)), (1, 0, 2))
        # (-,+,+)
        self.assertEqual(order(CP(BW, t3), CP(FW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t3), CP(FW, t1), CP(FW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t3)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t3), CP(FW, t2)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(FW, t3), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t3)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(FW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t2)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t1)), (0, 1, 2))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(FW, t1)), (0, 1, 2))
        # (+,-,-)
        self.assertEqual(order(CP(FW, t3), CP(BW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t3), CP(BW, t1), CP(BW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t3)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t3), CP(BW, t2)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t3), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t3)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(BW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t2)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t1)), (2, 1, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(BW, t1)), (2, 1, 0))
        # (+,-,+)
        self.assertEqual(order(CP(FW, t3), CP(BW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t3), CP(BW, t1), CP(FW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t3)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t3), CP(FW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t3), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t3)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(FW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t2)), (1, 2, 0))
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t1)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t2)), (1, 0, 2))
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(FW, t1)), (1, 0, 2))
        # (-,+,-)
        self.assertEqual(order(CP(BW, t3), CP(FW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t3), CP(FW, t1), CP(BW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t3)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t3), CP(BW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t3), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t3)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(BW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t2)), (0, 2, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t1)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t2)), (2, 0, 1))
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(BW, t1)), (2, 0, 1))

    def _test_gf(self, g):
        # Check Aoki RMP Eq. (16)
        g11 = g[Branch.FORWARD, Branch.FORWARD]
        g12 = g[Branch.FORWARD, Branch.BACKWARD]
        g21 = g[Branch.BACKWARD, Branch.FORWARD]
        g22 = g[Branch.BACKWARD, Branch.BACKWARD]
        assert_array_almost_equal((g11 + g22).data, (g12 + g21).data)

        non_t_shape = tuple(len(m) for m in g.mesh.components[2:]) \
            + g.target_shape

        t = next(iter(self.t_mesh))

        if len(g.mesh.components) == 2:
            g[CP(BW, t), CP(FW, t)] = 3.0
        else:
            g[CP(BW, t), CP(FW, t)] = Function(lambda i: 3.0)
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           3.0 * np.ones(non_t_shape))

        g[BW, FW].data[:] = 2 * np.ones((self.n_t, self.n_t, *non_t_shape))
        assert_array_equal(g[BW, FW].data,
                           2 * np.ones((self.n_t, self.n_t, *non_t_shape)))

        # Multiplication by a scalar
        g *= 3
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           6.0 * np.ones(non_t_shape))

        # Addition
        g += g
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           12.0 * np.ones(non_t_shape))
        assert_array_equal((g + g)[CP(BW, t), CP(FW, t)].data,
                           24.0 * np.ones(non_t_shape))

        # Subtraction
        g -= 0.5 * g
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           6.0 * np.ones(non_t_shape))
        assert_array_equal((g - g)[CP(BW, t), CP(FW, t)].data,
                           0.0 * np.ones(non_t_shape))

        # Unary minus
        assert_array_equal((-g)[CP(BW, t), CP(FW, t)].data,
                           -6.0 * np.ones(non_t_shape))

    def test_keldysh_gf(self):
        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF
            g_l = Gf(mesh=self.tt_mesh, target_shape=target_shape)
            g_g = Gf(mesh=self.tt_mesh, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = KeldyshGF.from_g_l_g_g(g_l, g_g)
            self.assertEqual(len(g.data), 4)

            for i in range(4):
                self.assertEqual(g.data[i].data.shape,
                                 (self.n_t, self.n_t) + target_shape)

            self._test_gf(g)

    def test_keldysh_gf_bz(self):
        bl = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # Square lattice
        n_k = 10
        bz_mesh = MeshBrillouinZone(BrillouinZone(bl), n_k)

        mesh = MeshProduct(*self.tt_mesh.components, bz_mesh)

        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF with an extra
            # Brillouin zone mesh component
            g_l = Gf(mesh=mesh, target_shape=target_shape)
            g_g = Gf(mesh=mesh, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = KeldyshGF.from_g_l_g_g(g_l, g_g)
            self.assertEqual(len(g.data), 4)

            for i in range(4):
                self.assertEqual(g.data[i].data.shape,
                                 (self.n_t, self.n_t, n_k**2) + target_shape)

            self._test_gf(g)

    def _make_test_keldysh_gf(self, mesh, x, target_shape=()):
        g = KeldyshGF(mesh=mesh, target_shape=target_shape)
        s = g[FW, FW].data.size
        g[FW, FW].data[:] = x * np.arange(s).reshape(g[FW, FW].data.shape)
        g[FW, BW].data[:] = x * np.arange(s).reshape(g[FW, BW].data.shape) + 1
        g[BW, FW].data[:] = x * np.arange(s).reshape(g[BW, FW].data.shape) + 2
        g[BW, BW].data[:] = x * np.arange(s).reshape(g[BW, BW].data.shape) + 3
        return g

    def test_keldysh_gf_convolution(self):
        w = simpsons_weights(self.t_mesh)

        # Scalar-valued GF
        g1 = self._make_test_keldysh_gf(self.tt_mesh, 1)
        g2 = self._make_test_keldysh_gf(self.tt_mesh, 2)
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=self.tt_mesh)
        for i, k, j in product(range(self.n_t), repeat=3):
            conv_ref[FW, FW].data[i, j] += \
                g1[FW, FW].data[i, k] * w[k] * g2[FW, FW].data[k, j] - \
                g1[FW, BW].data[i, k] * w[k] * g2[BW, FW].data[k, j]
            conv_ref[FW, BW].data[i, j] += \
                g1[FW, FW].data[i, k] * w[k] * g2[FW, BW].data[k, j] - \
                g1[FW, BW].data[i, k] * w[k] * g2[BW, BW].data[k, j]
            conv_ref[BW, FW].data[i, j] += \
                g1[BW, FW].data[i, k] * w[k] * g2[FW, FW].data[k, j] - \
                g1[BW, BW].data[i, k] * w[k] * g2[BW, FW].data[k, j]
            conv_ref[BW, BW].data[i, j] += \
                g1[BW, FW].data[i, k] * w[k] * g2[FW, BW].data[k, j] - \
                g1[BW, BW].data[i, k] * w[k] * g2[BW, BW].data[k, j]

        for b0, b1 in product((FW, BW), repeat=2):
            assert_array_almost_equal(conv[b0, b1].data, conv_ref[b0, b1].data)

        # Matrix-valued GF
        g1 = self._make_test_keldysh_gf(self.tt_mesh, 1, (2, 2))
        g2 = self._make_test_keldysh_gf(self.tt_mesh, 2, (2, 2))
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=self.tt_mesh, target_shape=(2, 2))
        for i, k, j, m, l, n in product(*[range(self.n_t)] * 3,
                                        *[range(2)] * 3):
            conv_ref[FW, FW].data[i, j, m, n] += \
                g1[FW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, FW].data[k, j, l, n] - \
                g1[FW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, FW].data[k, j, l, n]
            conv_ref[FW, BW].data[i, j, m, n] += \
                g1[FW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, BW].data[k, j, l, n] - \
                g1[FW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, BW].data[k, j, l, n]
            conv_ref[BW, FW].data[i, j, m, n] += \
                g1[BW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, FW].data[k, j, l, n] - \
                g1[BW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, FW].data[k, j, l, n]
            conv_ref[BW, BW].data[i, j, m, n] += \
                g1[BW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, BW].data[k, j, l, n] - \
                g1[BW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, BW].data[k, j, l, n]

        for b0, b1 in product((FW, BW), repeat=2):
            assert_array_almost_equal(conv[b0, b1].data, conv_ref[b0, b1].data)

        # Square lattice
        bl = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])
        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(bl), n_k)
        ttk_mesh = MeshProduct(*self.tt_mesh.components, bz_mesh)

        # Scalar-valued GF with an extra k-mesh component
        g1 = self._make_test_keldysh_gf(ttk_mesh, 1)
        g2 = self._make_test_keldysh_gf(ttk_mesh, 2)
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk_mesh)
        for i, k, j, K in product(*[range(self.n_t)] * 3, range(len(bz_mesh))):
            conv_ref[FW, FW].data[i, j, K] += \
                g1[FW, FW].data[i, k, K] * w[k] * g2[FW, FW].data[k, j, K] - \
                g1[FW, BW].data[i, k, K] * w[k] * g2[BW, FW].data[k, j, K]
            conv_ref[FW, BW].data[i, j, K] += \
                g1[FW, FW].data[i, k, K] * w[k] * g2[FW, BW].data[k, j, K] - \
                g1[FW, BW].data[i, k, K] * w[k] * g2[BW, BW].data[k, j, K]
            conv_ref[BW, FW].data[i, j, K] += \
                g1[BW, FW].data[i, k, K] * w[k] * g2[FW, FW].data[k, j, K] - \
                g1[BW, BW].data[i, k, K] * w[k] * g2[BW, FW].data[k, j, K]
            conv_ref[BW, BW].data[i, j, K] += \
                g1[BW, FW].data[i, k, K] * w[k] * g2[FW, BW].data[k, j, K] - \
                g1[BW, BW].data[i, k, K] * w[k] * g2[BW, BW].data[k, j, K]

        for b0, b1 in product((FW, BW), repeat=2):
            assert_array_almost_equal(conv[b0, b1].data, conv_ref[b0, b1].data)

        # Matrix-valued GF with an extra k-mesh component
        g1 = self._make_test_keldysh_gf(ttk_mesh, 1, (2, 2))
        g2 = self._make_test_keldysh_gf(ttk_mesh, 2, (2, 2))
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk_mesh, target_shape=(2, 2))
        for i, k, j, K, m, l, n in product(*[range(self.n_t)] * 3,
                                           range(len(bz_mesh)),
                                           *[range(2)] * 3):
            conv_ref[FW, FW].data[i, j, K, m, n] += \
                g1[FW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K, l, n] - \
                g1[FW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K, l, n]
            conv_ref[FW, BW].data[i, j, K, m, n] += \
                g1[FW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K, l, n] - \
                g1[FW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K, l, n]
            conv_ref[BW, FW].data[i, j, K, m, n] += \
                g1[BW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K, l, n] - \
                g1[BW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K, l, n]
            conv_ref[BW, BW].data[i, j, K, m, n] += \
                g1[BW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K, l, n] - \
                g1[BW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K, l, n]

        for b0, b1 in product((FW, BW), repeat=2):
            assert_array_almost_equal(conv[b0, b1].data, conv_ref[b0, b1].data)

    def test_keldysh_vertex3(self):

        def make_time_piece(x):
            g = GfReTime(mesh=self.ttt_mesh, target_shape=())
            g.data[:] = x
            return g
        G = {(0, 1, 2): make_time_piece(1.0),
             (0, 2, 1): make_time_piece(2.0),
             (1, 0, 2): make_time_piece(3.0),
             (1, 2, 0): make_time_piece(4.0),
             (2, 0, 1): make_time_piece(5.0),
             (2, 1, 0): make_time_piece(6.0)}

        Lambda = KeldyshVertex3.from_G_perm_pieces(G)
        for a0, a1, a2 in product(Branch, repeat=3):
            for t0, t1, t2 in self.ttt_mesh:
                self.assertNotEqual(Lambda[CP(a0, t0), CP(a1, t1), CP(a2, t2)],
                                    0)

        t = next(iter(self.t_mesh))

        Lambda[CP(BW, t), CP(FW, t), CP(BW, t)] = 3.0
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 3.0)

        ones_time_mat = np.ones((self.n_t, self.n_t, self.n_t))
        Lambda[BW, FW, BW].data[:] = 2 * ones_time_mat
        assert_array_equal(Lambda[BW, FW, BW].data, 2 * ones_time_mat)

        # Multiplication by a scalar
        Lambda *= 3
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 6.0)
        self.assertEqual((Lambda * 2.0)[CP(BW, t), CP(FW, t), CP(BW, t)], 12.0)
        self.assertEqual((2.0 * Lambda)[CP(BW, t), CP(FW, t), CP(BW, t)], 12.0)

        # Addition
        Lambda += Lambda
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 12.0)
        self.assertEqual((Lambda + Lambda)[CP(BW, t), CP(FW, t), CP(BW, t)],
                         24.0)

        ## Subtraction
        Lambda -= 0.5 * Lambda
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 6.0)
        self.assertEqual((Lambda - Lambda)[CP(BW, t), CP(FW, t), CP(BW, t)],
                         0.0)

        # Unary minus
        self.assertEqual((-Lambda)[CP(BW, t), CP(FW, t), CP(BW, t)], -6.0)


if __name__ == '__main__':
    unittest.main()
