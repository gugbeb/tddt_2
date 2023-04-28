import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from itertools import product
import numpy as np
from scipy.linalg import expm

from triqs.gf import MeshReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.gf.descriptors import Function
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.utility.comparison_tests import assert_gfs_are_close

from tddt.retime import conj
from tddt.keldysh import (Branch,
                          ContourPoint,
                          contour_ordering,
                          KeldyshGF,
                          target_dot,
                          from_lesser_greater,
                          from_vertex3_pieces,
                          greater,
                          lesser,
                          retarded,
                          advanced,
                          retarded_ext,
                          advanced_ext,
                          herm_conj,
                          is_hermitian)
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

        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice

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
        # Extended retarded component
        g_ret_ext = retarded_ext(g)
        for p in g_ret_ext.mesh:
            assert_array_equal(g_ret_ext[p], g21[p] - g12[p])
        # Extended advanced component
        g_adv_ext = advanced_ext(g)
        for p in g_adv_ext.mesh:
            assert_array_equal(g_adv_ext[p], g12[p] - g21[p])

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

        # Equality
        self.assertEqual(g, g)

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
                           2.0 * np.ones((self.n_t1, n_k, 3)))
        # __setitem__()
        g[Branch.FORWARD].data[:] = 4.0
        assert_array_equal(g.components[0].data,
                           4.0 * np.ones((self.n_t1, n_k, 3)))

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
                                 (self.n_t1, self.n_t2, n_k) + target_shape)

            self._test_gf(g)

    def test_target_dot(self):
        n_k = 3
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(self.t_mesh1, self.t_mesh1, self.t_mesh1, bz_mesh)

        g = KeldyshGF(mesh=mesh, target_subshapes=((2, 3), (4, 5, 6), (3, 2)))

        for a, idx in enumerate(np.ndindex(2, 2, 2)):
            c = g.components[idx]
            s = int(np.prod(c.data.shape))
            c.data[:] = a * np.arange(s).reshape(c.data.shape)

        x = np.arange(2 * 4 * 3 * 6 * 5).reshape((2, 4, 3, 6, 5))

        res = target_dot(g, x, 1, (1, 4, 3))

        ref = KeldyshGF(mesh=mesh, target_subshapes=((2, 3), (2, 3), (3, 2)))
        for br in product(Branch, repeat=g.n_args):
            for i, j, k, l, m in np.ndindex(4, 5, 6, 2, 3):
                ref[br].data[:, :, :, :, :, :, l, m, :, :] += \
                    g[br].data[:, :, :, :, :, :, i, j, k, :, :] \
                    * x[l, i, m, k, j]

        assert_keldysh_gf_almost_equal(res, ref, precision=1e-14)

    def test_hermitian(self):
        mesh = MeshProduct(self.t_mesh1, self.t_mesh1)

        # TODO: herm_conj(herm_conj(A)) == A
        # TODO: herm_conj(A @ B) == herm_conj(B) @ herm_conj(A)

        # Scalar-valued GF
        g_l = Gf(mesh=mesh, target_shape=())
        g_g = Gf(mesh=mesh, target_shape=())

        for t1, t2 in mesh:
            e = np.exp(-1j * 2.0 * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * -0.1 * e

        g = from_lesser_greater(g_l, g_g)
        self.assertTrue(is_hermitian(g))
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(greater(g), -conj(greater(g_hc)))
        assert_gfs_are_close(lesser(g), -conj(lesser(g_hc)))
        assert_gfs_are_close(retarded(g), conj(advanced(g_hc)))

        # Matrix-valued GF
        h_mat = np.array([[1.0, 0.5j], [-0.5j, 2.0]])

        g_l = Gf(mesh=mesh, target_shape=(2, 2))
        g_g = Gf(mesh=mesh, target_shape=(2, 2))

        for t1, t2 in mesh:
            e = expm(-1j * h_mat * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * -0.1 * e
        g = from_lesser_greater(g_l, g_g)
        self.assertTrue(is_hermitian(g))
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(greater(g), -conj(greater(g_hc)))
        assert_gfs_are_close(lesser(g), -conj(lesser(g_hc)))
        assert_gfs_are_close(retarded(g), conj(advanced(g_hc)))

        # Matrix-valued GF with an extra k-mesh component
        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(self.t_mesh1, self.t_mesh1, bz_mesh)

        g_l = Gf(mesh=mesh, target_shape=(2, 2))
        g_g = Gf(mesh=mesh, target_shape=(2, 2))
        for k in bz_mesh:
            eps = np.sum(k.value)
            h_mat = np.array([[eps, 0.5j], [-0.5j, 2.0 * eps]])
            for t1, t2 in MeshProduct(self.t_mesh1, self.t_mesh1):
                e = expm(-1j * h_mat * (t1.value - t2.value))
                g_g[t1, t2, k] = -1j * (1.0 - 0.1) * e
                g_l[t1, t2, k] = -1j * -0.1 * e
        g = from_lesser_greater(g_l, g_g)
        self.assertTrue(is_hermitian(g))
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(greater(g), -conj(greater(g_hc)))
        assert_gfs_are_close(lesser(g), -conj(lesser(g_hc)))
        assert_gfs_are_close(retarded(g), conj(advanced(g_hc)))

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
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=self.tt_mesh13)
        for i, k, j in product(*map(range, (self.n_t1, self.n_t2, self.n_t3))):
            g1g2_ref[FW, FW].data[i, j] += \
                g1[FW, FW].data[i, k] * w[k] * g2[FW, FW].data[k, j] - \
                g1[FW, BW].data[i, k] * w[k] * g2[BW, FW].data[k, j]
            g1g2_ref[FW, BW].data[i, j] += \
                g1[FW, FW].data[i, k] * w[k] * g2[FW, BW].data[k, j] - \
                g1[FW, BW].data[i, k] * w[k] * g2[BW, BW].data[k, j]
            g1g2_ref[BW, FW].data[i, j] += \
                g1[BW, FW].data[i, k] * w[k] * g2[FW, FW].data[k, j] - \
                g1[BW, BW].data[i, k] * w[k] * g2[BW, FW].data[k, j]
            g1g2_ref[BW, BW].data[i, j] += \
                g1[BW, FW].data[i, k] * w[k] * g2[FW, BW].data[k, j] - \
                g1[BW, BW].data[i, k] * w[k] * g2[BW, BW].data[k, j]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

        # Matrix-valued GF
        g1 = self._make_test_keldysh_gf(self.tt_mesh12, 1, ((2,), (4,)))
        g2 = self._make_test_keldysh_gf(self.tt_mesh23, 2, ((4,), (1,)))
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=self.tt_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, m, l, n in product(range(self.n_t1),
                                        range(self.n_t2),
                                        range(self.n_t3),
                                        range(2), range(4), range(1)):
            g1g2_ref[FW, FW].data[i, j, m, n] += \
                g1[FW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, FW].data[k, j, l, n] - \
                g1[FW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, FW].data[k, j, l, n]
            g1g2_ref[FW, BW].data[i, j, m, n] += \
                g1[FW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, BW].data[k, j, l, n] - \
                g1[FW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, BW].data[k, j, l, n]
            g1g2_ref[BW, FW].data[i, j, m, n] += \
                g1[BW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, FW].data[k, j, l, n] - \
                g1[BW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, FW].data[k, j, l, n]
            g1g2_ref[BW, BW].data[i, j, m, n] += \
                g1[BW, FW].data[i, k, m, l] * w[k] * \
                g2[FW, BW].data[k, j, l, n] - \
                g1[BW, BW].data[i, k, m, l] * w[k] * \
                g2[BW, BW].data[k, j, l, n]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

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
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=ttk_mesh13)
        for i, k, j, K in product(range(self.n_t1),
                                  range(self.n_t2),
                                  range(self.n_t3),
                                  range(len(bz_mesh))):
            g1g2_ref[FW, FW].data[i, j, K] += \
                g1[FW, FW].data[i, k, K] * w[k] * g2[FW, FW].data[k, j, K] - \
                g1[FW, BW].data[i, k, K] * w[k] * g2[BW, FW].data[k, j, K]
            g1g2_ref[FW, BW].data[i, j, K] += \
                g1[FW, FW].data[i, k, K] * w[k] * g2[FW, BW].data[k, j, K] - \
                g1[FW, BW].data[i, k, K] * w[k] * g2[BW, BW].data[k, j, K]
            g1g2_ref[BW, FW].data[i, j, K] += \
                g1[BW, FW].data[i, k, K] * w[k] * g2[FW, FW].data[k, j, K] - \
                g1[BW, BW].data[i, k, K] * w[k] * g2[BW, FW].data[k, j, K]
            g1g2_ref[BW, BW].data[i, j, K] += \
                g1[BW, FW].data[i, k, K] * w[k] * g2[FW, BW].data[k, j, K] - \
                g1[BW, BW].data[i, k, K] * w[k] * g2[BW, BW].data[k, j, K]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

        # Matrix-valued GF with an extra k-mesh component
        g1 = self._make_test_keldysh_gf(ttk_mesh12, 1, ((2,), (3,)))
        g2 = self._make_test_keldysh_gf(ttk_mesh23, 2, ((3,), (1,)))
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=ttk_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, K, m, l, n in product(range(self.n_t1),
                                           range(self.n_t2),
                                           range(self.n_t3),
                                           range(len(bz_mesh)),
                                           range(2), range(3), range(1)):
            g1g2_ref[FW, FW].data[i, j, K, m, n] += \
                g1[FW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K, l, n] - \
                g1[FW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K, l, n]
            g1g2_ref[FW, BW].data[i, j, K, m, n] += \
                g1[FW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K, l, n] - \
                g1[FW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K, l, n]
            g1g2_ref[BW, FW].data[i, j, K, m, n] += \
                g1[BW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K, l, n] - \
                g1[BW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K, l, n]
            g1g2_ref[BW, BW].data[i, j, K, m, n] += \
                g1[BW, FW].data[i, k, K, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K, l, n] - \
                g1[BW, BW].data[i, k, K, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K, l, n]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

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
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=ttk12_mesh13)
        for i, k, j, K1, K2 in product(range(self.n_t1),
                                       range(self.n_t2),
                                       range(self.n_t3),
                                       range(len(bz1_mesh)),
                                       range(len(bz2_mesh))):
            g1g2_ref[FW, FW].data[i, j, K1, K2] += \
                g1[FW, FW].data[i, k, K1] * w[k] * g2[FW, FW].data[k, j, K2] - \
                g1[FW, BW].data[i, k, K1] * w[k] * g2[BW, FW].data[k, j, K2]
            g1g2_ref[FW, BW].data[i, j, K1, K2] += \
                g1[FW, FW].data[i, k, K1] * w[k] * g2[FW, BW].data[k, j, K2] - \
                g1[FW, BW].data[i, k, K1] * w[k] * g2[BW, BW].data[k, j, K2]
            g1g2_ref[BW, FW].data[i, j, K1, K2] += \
                g1[BW, FW].data[i, k, K1] * w[k] * g2[FW, FW].data[k, j, K2] - \
                g1[BW, BW].data[i, k, K1] * w[k] * g2[BW, FW].data[k, j, K2]
            g1g2_ref[BW, BW].data[i, j, K1, K2] += \
                g1[BW, FW].data[i, k, K1] * w[k] * g2[FW, BW].data[k, j, K2] - \
                g1[BW, BW].data[i, k, K1] * w[k] * g2[BW, BW].data[k, j, K2]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

        # Matrix-valued GFs with different k-mesh components
        g1 = self._make_test_keldysh_gf(ttk1_mesh12, 1, ((2,), (3,)))
        g2 = self._make_test_keldysh_gf(ttk2_mesh23, 2, ((3,), (1,)))
        g1g2 = g1 @ g2

        g1g2_ref = KeldyshGF(mesh=ttk12_mesh13, target_subshapes=((2,), (1,)))
        for i, k, j, K1, K2, m, l, n in product(range(self.n_t1),
                                                range(self.n_t2),
                                                range(self.n_t3),
                                                range(len(bz1_mesh)),
                                                range(len(bz2_mesh)),
                                                range(2), range(3), range(1)):
            g1g2_ref[FW, FW].data[i, j, K1, K2, m, n] += \
                g1[FW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K2, l, n] - \
                g1[FW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K2, l, n]
            g1g2_ref[FW, BW].data[i, j, K1, K2, m, n] += \
                g1[FW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K2, l, n] - \
                g1[FW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K2, l, n]
            g1g2_ref[BW, FW].data[i, j, K1, K2, m, n] += \
                g1[BW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, FW].data[k, j, K2, l, n] - \
                g1[BW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, FW].data[k, j, K2, l, n]
            g1g2_ref[BW, BW].data[i, j, K1, K2, m, n] += \
                g1[BW, FW].data[i, k, K1, m, l] * w[k] * \
                g2[FW, BW].data[k, j, K2, l, n] - \
                g1[BW, BW].data[i, k, K1, m, l] * w[k] * \
                g2[BW, BW].data[k, j, K2, l, n]

        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref)

    def test_keldysh_vertex3(self):

        def make_time_piece(x):
            g = Gf(mesh=self.ttt_mesh, target_shape=())
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
