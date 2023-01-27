import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshProduct, MeshBrillouinZone
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import Branch, KeldyshGF, KeldyshVertex3
from tddt.diagrams import VertexLeg, vertex3_attach_leg
from tddt.util import simpsons_weights
from tddt.testing import assert_keldysh_vertex3_almost_equal

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

        # Simpson’s rule weights
        cls.W = simpsons_weights(cls.t_mesh[3])

    def _make_test_keldysh_gf(self, mesh, x, target_shape=()):
        g = KeldyshGF(mesh=mesh, target_shape=target_shape)
        for n, (b0, b1) in enumerate(product(Branch, repeat=2)):
            g_comp = g[b0, b1]
            s = g_comp.data.size
            g_comp.data[:] = x * np.arange(s).reshape(g_comp.data.shape) + n
        return g

    def _make_test_keldysh_vertex3(self, mesh, x, target_shape=()):
        Lambda = KeldyshVertex3(mesh=mesh, target_shape=target_shape)
        for n, (b0, b1, b2) in enumerate(product(Branch, repeat=3)):
            l_comp = Lambda[b0, b1, b2]
            s = l_comp.data.size
            l_comp.data[:] = x * np.arange(s).reshape(l_comp.data.shape) + n
        return Lambda

    def test_vertex3_attach_leg_scalar_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1])
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l in product(*self.t_ranges):
                f_ref[b0, b1, b2].data[i, j, k] += \
                    Lambda[b0, FW, b2].data[i, l, k] * self.W[l] *\
                    g[FW, b1].data[l, j] -\
                    Lambda[b0, BW, b2].data[i, l, k] * self.W[l] *\
                    g[BW, b1].data[l, j]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3])
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l in product(*self.t_ranges):
                f_ref[b0, b1, b2].data[i, j, k] += \
                    Lambda[FW, b1, b2].data[l, j, k] * self.W[l] *\
                    g[b0, FW].data[i, l] -\
                    Lambda[BW, b1, b2].data[l, j, k] * self.W[l] *\
                    g[b0, BW].data[i, l]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2])
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l in product(*self.t_ranges):
                f_ref[b0, b1, b2].data[i, j, k] += \
                    Lambda[b0, b1, FW].data[i, j, l] * self.W[l] *\
                    g[FW, b2].data[l, k] -\
                    Lambda[b0, b1, BW].data[i, j, l] * self.W[l] *\
                    g[BW, b2].data[l, k]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, x, y, z, w in product(*self.t_ranges,
                                                  *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, x, y, z] += \
                    Lambda[b0, FW, b2].data[i, l, k, x, w, z] * self.W[l] *\
                    g[FW, b1].data[l, j, w, y] -\
                    Lambda[b0, BW, b2].data[i, l, k, x, w, z] * self.W[l] *\
                    g[BW, b1].data[l, j, w, y]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, x, y, z, w in product(*self.t_ranges,
                                                  *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, x, y, z] += \
                    Lambda[FW, b1, b2].data[l, j, k, w, y, z] * self.W[l] *\
                    g[b0, FW].data[i, l, x, w] -\
                    Lambda[BW, b1, b2].data[l, j, k, w, y, z] * self.W[l] *\
                    g[b0, BW].data[i, l, x, w]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3]),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, x, y, z, w in product(*self.t_ranges,
                                                  *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, x, y, z] += \
                    Lambda[b0, b1, FW].data[i, j, l, x, y, w] * self.W[l] *\
                    g[FW, b2].data[l, k, w, z] -\
                    Lambda[b0, b1, BW].data[i, j, l, x, y, w] * self.W[l] *\
                    g[BW, b2].data[l, k, w, z]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K in product(*self.t_ranges, range(self.n_k)):
                f_ref[b0, b1, b2].data[i, j, k, K] += \
                    Lambda[b0, FW, b2].data[i, l, k] * self.W[l] *\
                    g[FW, b1].data[l, j, K] -\
                    Lambda[b0, BW, b2].data[i, l, k] * self.W[l] *\
                    g[BW, b1].data[l, j, K]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K in product(*self.t_ranges, range(self.n_k)):
                f_ref[b0, b1, b2].data[i, j, k, K] += \
                    Lambda[FW, b1, b2].data[l, j, k] * self.W[l] *\
                    g[b0, FW].data[i, l, K] -\
                    Lambda[BW, b1, b2].data[l, j, k] * self.W[l] *\
                    g[b0, BW].data[i, l, K]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K in product(*self.t_ranges, range(self.n_k)):
                f_ref[b0, b1, b2].data[i, j, k, K] += \
                    Lambda[b0, b1, FW].data[i, j, l] * self.W[l] *\
                    g[FW, b2].data[l, k, K] -\
                    Lambda[b0, b1, BW].data[i, j, l] * self.W[l] *\
                    g[BW, b2].data[l, k, K]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K, x, y, z, w in product(*self.t_ranges,
                                                     range(self.n_k),
                                                     *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K, x, y, z] += \
                    Lambda[b0, FW, b2].data[i, l, k, x, w, z] * self.W[l] *\
                    g[FW, b1].data[l, j, K, w, y] -\
                    Lambda[b0, BW, b2].data[i, l, k, x, w, z] * self.W[l] *\
                    g[BW, b1].data[l, j, K, w, y]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K, x, y, z, w in product(*self.t_ranges,
                                                     range(self.n_k),
                                                     *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K, x, y, z] += \
                    Lambda[FW, b1, b2].data[l, j, k, w, y, z] * self.W[l] *\
                    g[b0, FW].data[i, l, K, x, w] -\
                    Lambda[BW, b1, b2].data[l, j, k, w, y, z] * self.W[l] *\
                    g[b0, BW].data[i, l, K, x, w]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                self.bz_mesh),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K, x, y, z, w in product(*self.t_ranges,
                                                     range(self.n_k),
                                                     *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K, x, y, z] += \
                    Lambda[b0, b1, FW].data[i, j, l, x, y, w] * self.W[l] *\
                    g[FW, b2].data[l, k, K, w, z] -\
                    Lambda[b0, b1, BW].data[i, j, l, x, y, w] * self.W[l] *\
                    g[BW, b2].data[l, k, K, w, z]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_bz_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2 in product(*self.t_ranges,
                                              *[range(self.n_k)] * 2):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2] += \
                    Lambda[b0, FW, b2].data[i, l, k, K1] * self.W[l] *\
                    g[FW, b1].data[l, j, K2] -\
                    Lambda[b0, BW, b2].data[i, l, k, K1] * self.W[l] *\
                    g[BW, b1].data[l, j, K2]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_bz_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2 in product(*self.t_ranges,
                                              *[range(self.n_k)] * 2):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2] += \
                    Lambda[FW, b1, b2].data[l, j, k, K1] * self.W[l] *\
                    g[b0, FW].data[i, l, K2] -\
                    Lambda[BW, b1, b2].data[l, j, k, K1] * self.W[l] *\
                    g[b0, BW].data[i, l, K2]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_scalar_bz_bz_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0)

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2 in product(*self.t_ranges,
                                              *[range(self.n_k)] * 2):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2] += \
                    Lambda[b0, b1, FW].data[i, j, l, K1] * self.W[l] *\
                    g[FW, b2].data[l, k, K2] -\
                    Lambda[b0, b1, BW].data[i, j, l, K1] * self.W[l] *\
                    g[BW, b2].data[l, k, K2]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_bz_INBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 3, 2)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[1], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2, x, y, z, w in product(
                    *self.t_ranges, *[range(self.n_k)] * 2, *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2, x, y, z] += \
                    Lambda[b0, FW, b2].data[i, l, k, K1, x, w, z] * self.W[l] *\
                    g[FW, b1].data[l, j, K2, w, y] -\
                    Lambda[b0, BW, b2].data[i, l, k, K1, x, w, z] * self.W[l] *\
                    g[BW, b1].data[l, j, K2, w, y]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_bz_OUTBOUND(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (3, 1, 2)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[3], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.OUTBOUND)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2, x, y, z, w in product(
                    *self.t_ranges, *[range(self.n_k)] * 2, *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2, x, y, z] += \
                    Lambda[FW, b1, b2].data[l, j, k, K1, w, y, z] * self.W[l] *\
                    g[b0, FW].data[i, l, K2, x, w] -\
                    Lambda[BW, b1, b2].data[l, j, k, K1, w, y, z] * self.W[l] *\
                    g[b0, BW].data[i, l, K2, x, w]
        assert_keldysh_vertex3_almost_equal(f, f_ref)

    def test_vertex3_attach_leg_matrix_bz_bz_BOSON(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 1, 3)],
                                  self.bz_mesh)
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (2, 2, 2))
        g_mesh = MeshProduct(self.t_mesh[3], self.t_mesh[2], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (2, 2))

        f = vertex3_attach_leg(Lambda, g, VertexLeg.BOSON)

        f_ref = KeldyshVertex3(mesh=MeshProduct(*self.t_mesh[0:3],
                                                *[self.bz_mesh] * 2),
                               target_shape=(2, 2, 2))
        for b0, b1, b2 in product(Branch, repeat=3):
            for i, j, k, l, K1, K2, x, y, z, w in product(
                    *self.t_ranges, *[range(self.n_k)] * 2, *[range(2)] * 4):
                f_ref[b0, b1, b2].data[i, j, k, K1, K2, x, y, z] += \
                    Lambda[b0, b1, FW].data[i, j, l, K1, x, y, w] * self.W[l] *\
                    g[FW, b2].data[l, k, K2, w, z] -\
                    Lambda[b0, b1, BW].data[i, j, l, K1, x, y, w] * self.W[l] *\
                    g[BW, b2].data[l, k, K2, w, z]
        assert_keldysh_vertex3_almost_equal(f, f_ref)
