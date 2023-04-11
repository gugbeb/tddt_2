import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, GfReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.gf.descriptors import Function
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import (Branch,
                          ContourPoint,
                          contour_ordering,
                          KeldyshGF,
                          from_lesser_greater,
                          from_vertex3_pieces,
                          greater,
                          lesser,
                          retarded,
                          advanced,
                          retarded_mod,
                          ret2adv,
                          adv2ret)
from tddt.integration import GregoryIntegrator
from tddt.testing import assert_keldysh_gf_almost_equal

CP = ContourPoint
FW, BW = Branch.FORWARD, Branch.BACKWARD


class test_keldysh(unittest.TestCase):
    """Keldysh Green's functions and vertices"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 6.0
        cls.n_t1, cls.n_t2, cls.n_t3 = 7, 8, 9
        cls.t_mesh1 = MeshReTime(0, cls.t_max, cls.n_t1)
        cls.t_mesh2 = MeshReTime(0, cls.t_max, cls.n_t2)
        cls.t_mesh3 = MeshReTime(0, cls.t_max, cls.n_t3)
        cls.tt_mesh12 = MeshProduct(cls.t_mesh1, cls.t_mesh2)
        cls.tt_mesh23 = MeshProduct(cls.t_mesh2, cls.t_mesh3)
        cls.tt_mesh13 = MeshProduct(cls.t_mesh1, cls.t_mesh3)
        cls.ttt_mesh = MeshProduct(cls.t_mesh1, cls.t_mesh2, cls.t_mesh3)

        cls.bl = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # Square lattice

    def test_contour_ordering2(self):
        t1, t2 = list(self.t_mesh1)[2:4]

        data = [
            # (+,+)
            ((FW, t2), (FW, t1), (0, 1)),
            ((FW, t1), (FW, t1), (0, 1)),
            ((FW, t1), (FW, t2), (1, 0)),
            # (-,-)
            ((BW, t1), (BW, t2), (0, 1)),
            ((BW, t1), (BW, t1), (1, 0)),
            ((BW, t2), (BW, t1), (1, 0)),
            # (-,+)
            ((BW, t1), (FW, t2), (0, 1)),
            ((BW, t1), (FW, t1), (0, 1)),
            ((BW, t2), (FW, t1), (0, 1)),
            # (+,-)
            ((FW, t1), (BW, t2), (1, 0)),
            ((FW, t1), (BW, t1), (1, 0)),
            ((FW, t2), (BW, t1), (1, 0))
        ]

        for (ba, ta), (bb, tb), order in data:
            self.assertEqual(contour_ordering(CP(ba, ta), CP(bb, tb)), order)

    def test_contour_ordering3(self):
        t1, t2, t3 = list(self.t_mesh1)[2:5]

        data = [
            # (+,+,+)
            ((FW, t3), (FW, t2), (FW, t1), (0, 1, 2)),
            ((FW, t3), (FW, t1), (FW, t2), (0, 2, 1)),
            ((FW, t1), (FW, t2), (FW, t3), (2, 1, 0)),
            ((FW, t1), (FW, t3), (FW, t2), (1, 2, 0)),
            ((FW, t2), (FW, t3), (FW, t1), (1, 0, 2)),
            ((FW, t2), (FW, t1), (FW, t3), (2, 0, 1)),
            ((FW, t2), (FW, t2), (FW, t1), (0, 1, 2)),
            ((FW, t1), (FW, t1), (FW, t2), (2, 0, 1)),
            ((FW, t2), (FW, t1), (FW, t1), (0, 1, 2)),
            ((FW, t1), (FW, t2), (FW, t2), (1, 2, 0)),
            ((FW, t1), (FW, t2), (FW, t1), (1, 0, 2)),
            ((FW, t2), (FW, t1), (FW, t2), (0, 2, 1)),
            ((FW, t1), (FW, t1), (FW, t1), (0, 1, 2)),
            # (-,-,-)
            ((BW, t3), (BW, t2), (BW, t1), (2, 1, 0)),
            ((BW, t3), (BW, t1), (BW, t2), (1, 2, 0)),
            ((BW, t1), (BW, t2), (BW, t3), (0, 1, 2)),
            ((BW, t1), (BW, t3), (BW, t2), (0, 2, 1)),
            ((BW, t2), (BW, t3), (BW, t1), (2, 0, 1)),
            ((BW, t2), (BW, t1), (BW, t3), (1, 0, 2)),
            ((BW, t2), (BW, t2), (BW, t1), (2, 1, 0)),
            ((BW, t1), (BW, t1), (BW, t2), (1, 0, 2)),
            ((BW, t2), (BW, t1), (BW, t1), (2, 1, 0)),
            ((BW, t1), (BW, t2), (BW, t2), (0, 2, 1)),
            ((BW, t1), (BW, t2), (BW, t1), (2, 0, 1)),
            ((BW, t2), (BW, t1), (BW, t2), (1, 2, 0)),
            ((BW, t1), (BW, t1), (BW, t1), (2, 1, 0)),
            # (+,+,-)
            ((FW, t3), (FW, t2), (BW, t1), (2, 0, 1)),
            ((FW, t3), (FW, t1), (BW, t2), (2, 0, 1)),
            ((FW, t1), (FW, t2), (BW, t3), (2, 1, 0)),
            ((FW, t1), (FW, t3), (BW, t2), (2, 1, 0)),
            ((FW, t2), (FW, t3), (BW, t1), (2, 1, 0)),
            ((FW, t2), (FW, t1), (BW, t3), (2, 0, 1)),
            ((FW, t2), (FW, t2), (BW, t1), (2, 0, 1)),
            ((FW, t1), (FW, t1), (BW, t2), (2, 0, 1)),
            ((FW, t2), (FW, t1), (BW, t1), (2, 0, 1)),
            ((FW, t1), (FW, t2), (BW, t2), (2, 1, 0)),
            ((FW, t1), (FW, t2), (BW, t1), (2, 1, 0)),
            ((FW, t2), (FW, t1), (BW, t2), (2, 0, 1)),
            ((FW, t1), (FW, t1), (BW, t1), (2, 0, 1)),
            # (-,-,+)
            ((BW, t3), (BW, t2), (FW, t1), (1, 0, 2)),
            ((BW, t3), (BW, t1), (FW, t2), (1, 0, 2)),
            ((BW, t1), (BW, t2), (FW, t3), (0, 1, 2)),
            ((BW, t1), (BW, t3), (FW, t2), (0, 1, 2)),
            ((BW, t2), (BW, t3), (FW, t1), (0, 1, 2)),
            ((BW, t2), (BW, t1), (FW, t3), (1, 0, 2)),
            ((BW, t2), (BW, t2), (FW, t1), (1, 0, 2)),
            ((BW, t1), (BW, t1), (FW, t2), (1, 0, 2)),
            ((BW, t2), (BW, t1), (FW, t1), (1, 0, 2)),
            ((BW, t1), (BW, t2), (FW, t2), (0, 1, 2)),
            ((BW, t1), (BW, t2), (FW, t1), (0, 1, 2)),
            ((BW, t2), (BW, t1), (FW, t2), (1, 0, 2)),
            ((BW, t1), (BW, t1), (FW, t1), (1, 0, 2)),
            # (-,+,+)
            ((BW, t3), (FW, t2), (FW, t1), (0, 1, 2)),
            ((BW, t3), (FW, t1), (FW, t2), (0, 2, 1)),
            ((BW, t1), (FW, t2), (FW, t3), (0, 2, 1)),
            ((BW, t1), (FW, t3), (FW, t2), (0, 1, 2)),
            ((BW, t2), (FW, t3), (FW, t1), (0, 1, 2)),
            ((BW, t2), (FW, t1), (FW, t3), (0, 2, 1)),
            ((BW, t2), (FW, t2), (FW, t1), (0, 1, 2)),
            ((BW, t1), (FW, t1), (FW, t2), (0, 2, 1)),
            ((BW, t2), (FW, t1), (FW, t1), (0, 1, 2)),
            ((BW, t1), (FW, t2), (FW, t2), (0, 1, 2)),
            ((BW, t1), (FW, t2), (FW, t1), (0, 1, 2)),
            ((BW, t2), (FW, t1), (FW, t2), (0, 2, 1)),
            ((BW, t1), (FW, t1), (FW, t1), (0, 1, 2)),
            # (+,-,-)
            ((FW, t3), (BW, t2), (BW, t1), (2, 1, 0)),
            ((FW, t3), (BW, t1), (BW, t2), (1, 2, 0)),
            ((FW, t1), (BW, t2), (BW, t3), (1, 2, 0)),
            ((FW, t1), (BW, t3), (BW, t2), (2, 1, 0)),
            ((FW, t2), (BW, t3), (BW, t1), (2, 1, 0)),
            ((FW, t2), (BW, t1), (BW, t3), (1, 2, 0)),
            ((FW, t2), (BW, t2), (BW, t1), (2, 1, 0)),
            ((FW, t1), (BW, t1), (BW, t2), (1, 2, 0)),
            ((FW, t2), (BW, t1), (BW, t1), (2, 1, 0)),
            ((FW, t1), (BW, t2), (BW, t2), (2, 1, 0)),
            ((FW, t1), (BW, t2), (BW, t1), (2, 1, 0)),
            ((FW, t2), (BW, t1), (BW, t2), (1, 2, 0)),
            ((FW, t1), (BW, t1), (BW, t1), (2, 1, 0)),
            # (+,-,+)
            ((FW, t3), (BW, t2), (FW, t1), (1, 0, 2)),
            ((FW, t3), (BW, t1), (FW, t2), (1, 0, 2)),
            ((FW, t1), (BW, t2), (FW, t3), (1, 2, 0)),
            ((FW, t1), (BW, t3), (FW, t2), (1, 2, 0)),
            ((FW, t2), (BW, t3), (FW, t1), (1, 0, 2)),
            ((FW, t2), (BW, t1), (FW, t3), (1, 2, 0)),
            ((FW, t2), (BW, t2), (FW, t1), (1, 0, 2)),
            ((FW, t1), (BW, t1), (FW, t2), (1, 2, 0)),
            ((FW, t2), (BW, t1), (FW, t1), (1, 0, 2)),
            ((FW, t1), (BW, t2), (FW, t2), (1, 2, 0)),
            ((FW, t1), (BW, t2), (FW, t1), (1, 0, 2)),
            ((FW, t2), (BW, t1), (FW, t2), (1, 0, 2)),
            ((FW, t1), (BW, t1), (FW, t1), (1, 0, 2)),
            # (-,+,-)
            ((BW, t3), (FW, t2), (BW, t1), (2, 0, 1)),
            ((BW, t3), (FW, t1), (BW, t2), (2, 0, 1)),
            ((BW, t1), (FW, t2), (BW, t3), (0, 2, 1)),
            ((BW, t1), (FW, t3), (BW, t2), (0, 2, 1)),
            ((BW, t2), (FW, t3), (BW, t1), (2, 0, 1)),
            ((BW, t2), (FW, t1), (BW, t3), (0, 2, 1)),
            ((BW, t2), (FW, t2), (BW, t1), (2, 0, 1)),
            ((BW, t1), (FW, t1), (BW, t2), (0, 2, 1)),
            ((BW, t2), (FW, t1), (BW, t1), (2, 0, 1)),
            ((BW, t1), (FW, t2), (BW, t2), (0, 2, 1)),
            ((BW, t1), (FW, t2), (BW, t1), (2, 0, 1)),
            ((BW, t2), (FW, t1), (BW, t2), (2, 0, 1)),
            ((BW, t1), (FW, t1), (BW, t1), (2, 0, 1))
        ]

        for (ba, ta), (bb, tb), (bc, tc), order in data:
            self.assertEqual(
                contour_ordering(CP(ba, ta), CP(bb, tb), CP(bc, tc)),
                order
            )

    def _test_gf(self, g):
        # Check Aoki RMP Eq. (16)
        g11 = g[Branch.FORWARD, Branch.FORWARD]
        g12 = g[Branch.FORWARD, Branch.BACKWARD]
        g21 = g[Branch.BACKWARD, Branch.FORWARD]
        g22 = g[Branch.BACKWARD, Branch.BACKWARD]
        assert_array_almost_equal((g11 + g22).data, (g12 + g21).data)

        # Greater component
        assert_array_equal(greater(g), g21)
        # Lesser component
        assert_array_equal(lesser(g), g12)
        # Retarded component
        g_ret = retarded(g)
        for p in g_ret.mesh:
            t0, t1 = p[0], p[1]
            ref = g21[p] - g12[p] if t0.linear_index >= t1.linear_index else 0
            assert_array_equal(g_ret[p], ref)
        # Advanced component
        g_adv = advanced(g)
        for p in g_adv.mesh:
            t0, t1 = p[0], p[1]
            ref = g12[p] - g21[p] if t0.linear_index <= t1.linear_index else 0
            assert_array_equal(g_adv[p], ref)
        # Modified retarded component
        g_ret_mod = retarded_mod(g)
        for p in g_ret_mod.mesh:
            assert_array_equal(g_ret_mod[p], g21[p] - g12[p])

        non_t_shape = tuple(len(m) for m in g.mesh.components[2:]) \
            + g.target_shape

        t = next(iter(self.t_mesh1))

        if len(g.mesh.components) == 2:
            g[CP(BW, t), CP(FW, t)] = 3.0
        else:
            g[CP(BW, t), CP(FW, t)] = Function(lambda i: 3.0)
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           3.0 * np.ones(non_t_shape))

        if len(g.mesh.components) == 2:
            g[BW, FW, t, t] = 4.0
        else:
            g[BW, FW, t, t, :] = Function(lambda i: 4.0)
        assert_array_equal(g[CP(BW, t), CP(FW, t)].data,
                           4.0 * np.ones(non_t_shape))

        g[BW, FW].data[:] = 2 * np.ones((self.n_t1, self.n_t2, *non_t_shape))
        assert_array_equal(g[BW, FW].data,
                           2 * np.ones((self.n_t1, self.n_t2, *non_t_shape)))

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

    def test_keldysh_gf_n_args0(self):
        for mesh in (None, MeshProduct()):
            g = KeldyshGF(mesh=mesh)

            self.assertEqual(g.mesh, MeshProduct())
            self.assertEqual(g.time_mesh, MeshProduct())
            self.assertEqual(g.non_time_mesh, MeshProduct())
            self.assertEqual(g.n_args, 0)
            self.assertEqual(g.components.shape, ())
            # __getitem__()
            g.components[()].data[()] = 2.0
            self.assertEqual(g[()].data, 2.0)
            # __setitem__()
            g[()].data[()] = 3.0
            self.assertEqual(g.components[()].data[()], 3.0)

    def test_keldysh_gf_n_args0_bz(self):
        n_k = 10
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        g = KeldyshGF(mesh=MeshProduct(bz_mesh))

        self.assertEqual(g.mesh, MeshProduct(bz_mesh))
        self.assertEqual(g.time_mesh, MeshProduct())
        self.assertEqual(g.non_time_mesh, MeshProduct(bz_mesh))
        self.assertEqual(g.n_args, 0)
        self.assertEqual(g.components.shape, ())
        for i, k in enumerate(bz_mesh):
            # __getitem__()
            g.components[()].data[i] = i
            self.assertEqual(g[k], i)
            # __setitem__()
            g[k] = i + 1
            self.assertEqual(g.components[()].data[i], i + 1)

    def test_keldysh_gf_n_args1(self):
        for mesh in (self.t_mesh1, MeshProduct(self.t_mesh1)):
            g = KeldyshGF(mesh=mesh, target_subshapes=((3,),))

            self.assertEqual(g.mesh, MeshProduct(self.t_mesh1))
            self.assertEqual(g.time_mesh, MeshProduct(self.t_mesh1))
            self.assertEqual(g.non_time_mesh, MeshProduct())
            self.assertEqual(g.n_args, 1)
            self.assertEqual(g.components.shape, (2,))

            t = next(iter(self.t_mesh1))

            # __getitem__()
            g.components[1].data[:] = 2.0
            assert_array_equal(g[Branch.BACKWARD].data,
                               2.0 * np.ones((self.n_t1, 3)))
            g.components[1].data[:] = 3.0
            assert_array_equal(g[Branch.BACKWARD, t],
                               3.0 * np.ones(3))
            # __setitem__()
            g[Branch.FORWARD].data[:] = 4.0
            assert_array_equal(g.components[0].data,
                               4.0 * np.ones((self.n_t1, 3)))
            g[Branch.FORWARD, t] = 5.0
            assert_array_equal(g.components[0].data[0, :],
                               5.0 * np.ones(3))

    def test_keldysh_gf_n_args1_bz(self):
        n_k = 10
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(self.t_mesh1, bz_mesh)
        g = KeldyshGF(mesh=mesh, target_subshapes=((3,),))

        self.assertEqual(g.mesh, mesh)
        self.assertEqual(g.time_mesh, MeshProduct(self.t_mesh1))
        self.assertEqual(g.non_time_mesh, MeshProduct(bz_mesh))
        self.assertEqual(g.n_args, 1)
        self.assertEqual(g.components.shape, (2,))

        t = next(iter(self.t_mesh1))

        # __getitem__()
        g.components[1].data[:] = 2.0
        assert_array_equal(g[Branch.BACKWARD].data,
                           2.0 * np.ones((self.n_t1, n_k * n_k, 3)))
        # __setitem__()
        g[Branch.FORWARD].data[:] = 4.0
        assert_array_equal(g.components[0].data,
                           4.0 * np.ones((self.n_t1, n_k * n_k, 3)))

        for i, k in enumerate(bz_mesh):
            # __getitem__()
            g.components[1].data[:, i] = i
            assert_array_equal(g[Branch.BACKWARD, t, k], i * np.ones(3))
            # __setitem__()
            g[Branch.BACKWARD, t, k] = i
            assert_array_equal(g.components[1].data[0, i, :], i * np.ones(3))

    def test_keldysh_gf(self):
        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF
            g_l = Gf(mesh=self.tt_mesh12, target_shape=target_shape)
            g_g = Gf(mesh=self.tt_mesh12, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = from_lesser_greater(g_l, g_g)
            self.assertEqual(g.components.shape, (2, 2))

            for i, j in product(range(2), repeat=2):
                self.assertEqual(g.components[i, j].data.shape,
                                 (self.n_t1, self.n_t2) + target_shape)

            self._test_gf(g)

    def test_keldysh_gf_bz(self):
        n_k = 10
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        mesh = MeshProduct(*self.tt_mesh12.components, bz_mesh)

        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF with an extra
            # Brillouin zone mesh component
            g_l = Gf(mesh=mesh, target_shape=target_shape)
            g_g = Gf(mesh=mesh, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = from_lesser_greater(g_l, g_g)
            self.assertEqual(g.components.shape, (2, 2))

            for i, j in product(range(2), repeat=2):
                self.assertEqual(g.components[i, j].data.shape,
                                 (self.n_t1, self.n_t2, n_k**2) + target_shape)

            self._test_gf(g)

    def test_ret2adv(self):
        n_k = 3
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        mesh = MeshProduct(self.t_mesh1, self.t_mesh2)

        # Scalar-valued GF
        g_ret = Gf(mesh=mesh, target_shape=())
        g_ret.data[:] = 1j * np.arange(self.n_t1 * self.n_t2) \
            .reshape((self.n_t1, self.n_t2))
        g_adv = ret2adv(g_ret)
        for t1, t2 in g_adv.mesh:
            self.assertEqual(g_adv[t1, t2], np.conj(g_ret[t2, t1]))

        # Matrix-valued GF
        g_ret = Gf(mesh=mesh, target_shape=(2, 3))
        g_ret.data[:] = 1j * np.arange(self.n_t1 * self.n_t2 * 6) \
            .reshape((self.n_t1, self.n_t2, 2, 3))
        g_adv = ret2adv(g_ret)
        for (t1, t2), i, j in product(g_adv.mesh, range(3), range(2)):
            self.assertEqual(g_adv[t1, t2][i, j], np.conj(g_ret[t2, t1][j, i]))

        # Matrix-valued GF with an extra mesh component
        mesh = MeshProduct(self.t_mesh1, self.t_mesh2, bz_mesh)
        g_ret = Gf(mesh=mesh, target_shape=(2, 3))
        g_ret.data[:] = 1j * np.arange(self.n_t1 * self.n_t2 * 6 * n_k ** 2) \
            .reshape((self.n_t1, self.n_t2, n_k ** 2, 2, 3))
        g_adv = ret2adv(g_ret)
        for (t1, t2, K), i, j in product(g_adv.mesh, range(3), range(2)):
            self.assertEqual(g_adv[t1, t2, K][i, j],
                             np.conj(g_ret[t2, t1, K][j, i]))

        # Tensor-valued GF with an extra mesh component
        g_ret = Gf(mesh=mesh, target_shape=(2, 3, 4))
        g_ret.data[:] = 1j * np.arange(self.n_t1 * self.n_t2 * 24 * n_k ** 2) \
            .reshape((self.n_t1, self.n_t2, n_k ** 2, 2, 3, 4))
        g_adv = ret2adv(g_ret, n_left_indices=2)
        for (t1, t2, K), i, j, k in product(g_adv.mesh,
                                            range(4),
                                            range(2),
                                            range(3)):
            self.assertEqual(g_adv[t1, t2, K][i, j, k],
                             np.conj(g_ret[t2, t1, K][j, k, i]))

        g_ret2 = adv2ret(g_adv, n_left_indices=1)
        self.assertEqual(g_ret2.mesh, g_ret.mesh)
        self.assertEqual(g_ret2.target_shape, g_ret.target_shape)
        assert_array_equal(g_ret2.data, g_ret.data)

    def _make_test_keldysh_gf(self, mesh, x, target_subshapes=None):
        g = KeldyshGF(mesh=mesh, target_subshapes=target_subshapes)
        for n, (b0, b1) in enumerate(product(Branch, repeat=2)):
            g_comp = g[b0, b1]
            s = g_comp.data.size
            g_comp.data[:] = x * np.arange(s).reshape(g_comp.data.shape) + n
        return g

    def test_keldysh_gf_convolution(self):
        w = GregoryIntegrator(5).weights_conv(self.t_mesh2)

        # Scalar-valued GF
        g1 = self._make_test_keldysh_gf(self.tt_mesh12, 1)
        g2 = self._make_test_keldysh_gf(self.tt_mesh23, 2)
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=self.tt_mesh13)
        for i, k, j in product(*map(range, (self.n_t1, self.n_t2, self.n_t3))):
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

        assert_keldysh_gf_almost_equal(conv, conv_ref)

        # Matrix-valued GF
        g1 = self._make_test_keldysh_gf(self.tt_mesh12, 1, ((2,), (4,)))
        g2 = self._make_test_keldysh_gf(self.tt_mesh23, 2, ((4,), (1,)))
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=self.tt_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, m, l, n in product(range(self.n_t1),
                                        range(self.n_t2),
                                        range(self.n_t3),
                                        range(2), range(4), range(1)):
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

        assert_keldysh_gf_almost_equal(conv, conv_ref)

    def test_keldysh_gf_convolution_bz(self):
        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        ttk_mesh12 = MeshProduct(*self.tt_mesh12.components, bz_mesh)
        ttk_mesh23 = MeshProduct(*self.tt_mesh23.components, bz_mesh)
        ttk_mesh13 = MeshProduct(*self.tt_mesh13.components, bz_mesh)

        w = GregoryIntegrator(5).weights_conv(self.t_mesh2)

        # Scalar-valued GF with an extra k-mesh component
        g1 = self._make_test_keldysh_gf(ttk_mesh12, 1)
        g2 = self._make_test_keldysh_gf(ttk_mesh23, 2)
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk_mesh13)
        for i, k, j, K in product(range(self.n_t1),
                                  range(self.n_t2),
                                  range(self.n_t3),
                                  range(len(bz_mesh))):
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

        assert_keldysh_gf_almost_equal(conv, conv_ref)

        # Matrix-valued GF with an extra k-mesh component
        g1 = self._make_test_keldysh_gf(ttk_mesh12, 1, ((2,), (3,)))
        g2 = self._make_test_keldysh_gf(ttk_mesh23, 2, ((3,), (1,)))
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, K, m, l, n in product(range(self.n_t1),
                                           range(self.n_t2),
                                           range(self.n_t3),
                                           range(len(bz_mesh)),
                                           range(2), range(3), range(1)):
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

        assert_keldysh_gf_almost_equal(conv, conv_ref)

    def test_keldysh_gf_convolution_bz1_bz2(self):
        n_k1 = 4
        bz1_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k1)
        ttk1_mesh12 = MeshProduct(*self.tt_mesh12.components, bz1_mesh)

        n_k2 = 3
        bz2_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k2)
        ttk2_mesh23 = MeshProduct(*self.tt_mesh23.components, bz2_mesh)

        ttk12_mesh13 = MeshProduct(*self.tt_mesh13.components,
                                   bz1_mesh,
                                   bz2_mesh)

        w = GregoryIntegrator(5).weights_conv(self.t_mesh2)

        # Scalar-valued GFs with different k-mesh components
        g1 = self._make_test_keldysh_gf(ttk1_mesh12, 1)
        g2 = self._make_test_keldysh_gf(ttk2_mesh23, 2)
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk12_mesh13)
        for i, k, j, K1, K2 in product(range(self.n_t1),
                                       range(self.n_t2),
                                       range(self.n_t3),
                                       range(len(bz1_mesh)),
                                       range(len(bz2_mesh))):
            conv_ref[FW, FW].data[i, j, K1, K2] += \
                g1[FW, FW].data[i, k, K1] * w[k] * g2[FW, FW].data[k, j, K2] - \
                g1[FW, BW].data[i, k, K1] * w[k] * g2[BW, FW].data[k, j, K2]
            conv_ref[FW, BW].data[i, j, K1, K2] += \
                g1[FW, FW].data[i, k, K1] * w[k] * g2[FW, BW].data[k, j, K2] - \
                g1[FW, BW].data[i, k, K1] * w[k] * g2[BW, BW].data[k, j, K2]
            conv_ref[BW, FW].data[i, j, K1, K2] += \
                g1[BW, FW].data[i, k, K1] * w[k] * g2[FW, FW].data[k, j, K2] - \
                g1[BW, BW].data[i, k, K1] * w[k] * g2[BW, FW].data[k, j, K2]
            conv_ref[BW, BW].data[i, j, K1, K2] += \
                g1[BW, FW].data[i, k, K1] * w[k] * g2[FW, BW].data[k, j, K2] - \
                g1[BW, BW].data[i, k, K1] * w[k] * g2[BW, BW].data[k, j, K2]

        assert_keldysh_gf_almost_equal(conv, conv_ref)

        # Matrix-valued GFs with different k-mesh components
        g1 = self._make_test_keldysh_gf(ttk1_mesh12, 1, ((2,), (3,)))
        g2 = self._make_test_keldysh_gf(ttk2_mesh23, 2, ((3,), (1,)))
        conv = g1 @ g2

        conv_ref = KeldyshGF(mesh=ttk12_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, K1, K2, m, l, n in product(range(self.n_t1),
                                                range(self.n_t2),
                                                range(self.n_t3),
                                                range(len(bz1_mesh)),
                                                range(len(bz2_mesh)),
                                                range(2), range(3), range(1)):
            conv_ref[FW, FW].data[i, j, K1, K2, m, n] += \
                g1[FW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K2, l, n] - \
                g1[FW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K2, l, n]
            conv_ref[FW, BW].data[i, j, K1, K2, m, n] += \
                g1[FW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K2, l, n] - \
                g1[FW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K2, l, n]
            conv_ref[BW, FW].data[i, j, K1, K2, m, n] += \
                g1[BW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K2, l, n] - \
                g1[BW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K2, l, n]
            conv_ref[BW, BW].data[i, j, K1, K2, m, n] += \
                g1[BW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K2, l, n] - \
                g1[BW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K2, l, n]

        assert_keldysh_gf_almost_equal(conv, conv_ref)

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

        Lambda = from_vertex3_pieces(G)
        for a0, a1, a2 in product(Branch, repeat=3):
            for t0, t1, t2 in self.ttt_mesh:
                self.assertNotEqual(Lambda[CP(a0, t0), CP(a1, t1), CP(a2, t2)],
                                    0)

        t = next(iter(self.t_mesh1))

        Lambda[CP(BW, t), CP(FW, t), CP(BW, t)] = 3.0
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 3.0)

        ones_time_mat = np.ones((self.n_t1, self.n_t2, self.n_t3))
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
