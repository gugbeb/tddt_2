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
                          Singular2PKeldyshGF,
                          target_dot,
                          herm_conj)
from tddt.testing import (assert_keldysh_gf_almost_equal,
                          assert_singular_2p_keldysh_gf_almost_equal)


CP = ContourPoint
FW, BW = Branch.FORWARD, Branch.BACKWARD


class TestContourPoint(unittest.TestCase):
    """Points on the Keldysh contour and their ordering"""

    def test_contour_ordering2(self):
        t_mesh = MeshReTime(0, 6.0, 7)
        t1, t2 = list(t_mesh)[2:4]

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
        t_mesh = MeshReTime(0, 6.0, 7)
        t1, t2, t3 = list(t_mesh)[2:5]

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


def single_state_g_l(occ, eps, dt):
    return (-1j * (1 - occ) * np.exp(-1j * eps * dt),
            -1j * (-occ) * np.exp(-1j * eps * dt))


class TestKeldyshGF(unittest.TestCase):
    """Keldysh Green's functions and vertices"""

    @classmethod
    def setUpClass(cls):
        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice

    def _test_gf(self, g):
        # Check Aoki RMP Eq. (16)
        g11 = g[Branch.FORWARD, Branch.FORWARD]
        g12 = g[Branch.FORWARD, Branch.BACKWARD]
        g21 = g[Branch.BACKWARD, Branch.FORWARD]
        g22 = g[Branch.BACKWARD, Branch.BACKWARD]
        assert_array_almost_equal((g11 + g22).data, (g12 + g21).data)

        # Greater component
        assert_array_equal(g.greater(), g21)
        # Lesser component
        assert_array_equal(g.lesser(), g12)
        # Retarded component
        g_ret = g.retarded()
        for p in g_ret.mesh:
            t0, t1 = p[0], p[1]
            ref = g21[p] - g12[p] if t0.linear_index >= t1.linear_index else 0
            assert_array_equal(g_ret[p], ref)
        # Advanced component
        g_adv = g.advanced()
        for p in g_adv.mesh:
            t0, t1 = p[0], p[1]
            ref = g12[p] - g21[p] if t0.linear_index <= t1.linear_index else 0
            assert_array_equal(g_adv[p], ref)
        # Extended retarded component
        g_ret_ext = g.retarded_ext()
        for p in g_ret_ext.mesh:
            assert_array_equal(g_ret_ext[p], g21[p] - g12[p])
        # Extended advanced component
        g_adv_ext = g.advanced_ext()
        for p in g_adv_ext.mesh:
            assert_array_equal(g_adv_ext[p], g12[p] - g21[p])

        non_t_shape = tuple(len(m) for m in g.mesh.components[2:]) \
            + g.target_shape

        t_mesh = MeshReTime(0, 6.0, 7)
        t = next(iter(t_mesh))

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

        g[BW, FW].data[:] = 2 * np.ones((7, 8, *non_t_shape))
        assert_array_equal(g[BW, FW].data, 2 * np.ones((7, 8, *non_t_shape)))

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

    def test_n_args1(self):
        t_mesh = MeshReTime(0, 6.0, 7)
        for mesh in (t_mesh, MeshProduct(t_mesh)):
            g = KeldyshGF(mesh=mesh, arg_index_shapes=((3,),))

            self.assertEqual(g.mesh, MeshProduct(t_mesh))
            self.assertEqual(g.time_mesh, MeshProduct(t_mesh))
            self.assertEqual(g.non_time_mesh, MeshProduct())
            self.assertEqual(g.n_args, 1)
            self.assertEqual(g.components.shape, (2,))

            t0, t1 = list(t_mesh)[:2]

            # __getitem__()
            g.components[1].data[:] = 2.0
            assert_array_equal(g[Branch.BACKWARD].data, 2.0 * np.ones((7, 3)))
            g.components[1].data[:] = 3.0
            assert_array_equal(g[Branch.BACKWARD, t0], 3.0 * np.ones(3))
            g.components[1].data[0, :] = 4.0
            assert_array_equal(g[CP(Branch.BACKWARD, t0)], 4.0 * np.ones(3))
            assert_array_equal(g[CP(Branch.BACKWARD, t1)], 3.0 * np.ones(3))

            # __setitem__()
            g[Branch.FORWARD].data[:] = 4.0
            assert_array_equal(g.components[0].data, 4.0 * np.ones((7, 3)))
            g[Branch.FORWARD, t0] = 5.0
            assert_array_equal(g.components[0].data[0, :], 5.0 * np.ones(3))
            g[CP(Branch.FORWARD, t0)] = 6.0
            assert_array_equal(g.components[0].data[0, :], 6.0 * np.ones(3))
            assert_array_equal(g.components[0].data[1, :], 4.0 * np.ones(3))

    def test_n_args1_bz(self):
        n_k = 10
        t_mesh = MeshReTime(0, 6.0, 7)
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(t_mesh, bz_mesh)

        g = KeldyshGF(mesh=mesh, arg_index_shapes=((3,),))

        self.assertEqual(g.mesh, mesh)
        self.assertEqual(g.time_mesh, MeshProduct(t_mesh))
        self.assertEqual(g.non_time_mesh, MeshProduct(bz_mesh))
        self.assertEqual(g.n_args, 1)
        self.assertEqual(g.components.shape, (2,))

        t0, t1 = list(t_mesh)[:2]

        # __getitem__()
        g.components[1].data[:] = 2.0
        assert_array_equal(g[Branch.BACKWARD].data, 2.0 * np.ones((7, n_k, 3)))
        g.components[1].data[0, :] = 3.0
        assert_array_equal(g[CP(Branch.BACKWARD, t0)].data,
                           3.0 * np.ones((n_k, 3)))
        assert_array_equal(g[CP(Branch.BACKWARD, t1)].data,
                           2.0 * np.ones((n_k, 3)))

        # __setitem__()
        g[Branch.FORWARD].data[:] = 4.0
        assert_array_equal(g.components[0].data,
                           4.0 * np.ones((7, n_k, 3)))

        for i, k in enumerate(bz_mesh):
            # __getitem__()
            g.components[1].data[:, i] = i
            assert_array_equal(g[Branch.BACKWARD, t0, k], i * np.ones(3))
            # __setitem__()
            g[Branch.BACKWARD, t0, k] = i
            assert_array_equal(g.components[1].data[0, i, :], i * np.ones(3))

    def test_common(self):
        tt_mesh = MeshProduct(MeshReTime(0, 6.0, 7), MeshReTime(0, 6.0, 8))

        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF
            g_l = Gf(mesh=tt_mesh, target_shape=target_shape)
            g_g = Gf(mesh=tt_mesh, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = KeldyshGF.from_lesser_greater(g_l, g_g)
            self.assertEqual(g.components.shape, (2, 2))

            for i, j in product(range(2), repeat=2):
                self.assertEqual(g.components[i, j].data.shape,
                                 (7, 8) + target_shape)

            self._test_gf(g)

    def test_from_arg_index_gen(self):
        ttt_mesh = MeshProduct(MeshReTime(0, 6.0, 7),
                               MeshReTime(0, 6.0, 8),
                               MeshReTime(0, 6.0, 9))
        arg_index_shapes = ((2, 3), (3, 2), (4,))

        def generator(ind1, ind2, ind3):
            g_el = KeldyshGF(mesh=ttt_mesh)
            for br1, br2, br3 in product(Branch, repeat=3):
                val = (ind1[0] + ind1[1]) * (ind2[0] - ind2[1]) * (ind3[0] + 3)
                g_el[br1, br2, br3].data[:] = val
            return g_el

        g = KeldyshGF.from_arg_index_gen(generator,
                                         mesh=ttt_mesh,
                                         arg_index_shapes=arg_index_shapes)

        index_ranges = product(*map(range, (2, 3, 3, 2, 4)))
        for i in index_ranges:
            val = (i[0] + i[1]) * (i[2] - i[3]) * (i[4] + 3)
            for br1, br2, br3 in product(Branch, repeat=3):
                assert_array_equal(
                    g[br1, br2, br3].data[:, :, :,
                                          i[0], i[1], i[2], i[3], i[4]],
                    val * np.ones((7, 8, 9)))

    def test_common_bz(self):
        tt_mesh = MeshProduct(MeshReTime(0, 6.0, 7), MeshReTime(0, 6.0, 8))
        n_k = 10
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        mesh = MeshProduct(*tt_mesh.components, bz_mesh)

        for target_shape in ((), (2, 2)):
            # Construct from lesser and greater GF with an extra
            # Brillouin zone mesh component
            g_l = Gf(mesh=mesh, target_shape=target_shape)
            g_g = Gf(mesh=mesh, target_shape=target_shape)

            g_l.data[:] = 2.0
            g_g.data[:] = 3.0
            g = KeldyshGF.from_lesser_greater(g_l, g_g)
            self.assertEqual(g.components.shape, (2, 2))

            for i, j in product(range(2), repeat=2):
                self.assertEqual(g.components[i, j].data.shape,
                                 (7, 8, n_k) + target_shape)

            self._test_gf(g)

    def test_target_dot(self):
        n_k = 3
        t_mesh = MeshReTime(0, 6.0, 7)
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(t_mesh, t_mesh, t_mesh, bz_mesh)

        g = KeldyshGF(mesh=mesh, arg_index_shapes=((2, 3), (4, 5, 6), (3, 2)))

        for a, idx in enumerate(np.ndindex(2, 2, 2)):
            c = g.components[idx]
            s = int(np.prod(c.data.shape))
            c.data[:] = a * np.arange(s).reshape(c.data.shape)

        x = np.arange(2 * 4 * 3 * 6 * 5).reshape((2, 4, 3, 6, 5))

        res = target_dot(g, x, 1, (1, 4, 3))

        ref = KeldyshGF(mesh=mesh, arg_index_shapes=((2, 3), (2, 3), (3, 2)))
        for br in product(Branch, repeat=g.n_args):
            for i, j, k, l, m in np.ndindex(4, 5, 6, 2, 3):
                ref[br].data[:, :, :, :, :, :, l, m, :, :] += \
                    g[br].data[:, :, :, :, :, :, i, j, k, :, :] \
                    * x[l, i, m, k, j]

        assert_keldysh_gf_almost_equal(res, ref, precision=1e-14)

    def test_hermitian(self):
        t_mesh = MeshReTime(0, 6.0, 7)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        # Scalar-valued GF
        g_l = Gf(mesh=tt_mesh, target_shape=())
        g_g = Gf(mesh=tt_mesh, target_shape=())

        for t1, t2 in tt_mesh:
            e = np.exp(-1j * 2.0 * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * -0.1 * e

        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self.assertTrue(g.is_hermitian())
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(g.greater(), -conj(g_hc.greater()))
        assert_gfs_are_close(g.lesser(), -conj(g_hc.lesser()))
        assert_gfs_are_close(g.retarded(), conj(g_hc.advanced()))

        # Matrix-valued GF
        h_mat = np.array([[1.0, 0.5j], [-0.5j, 2.0]])

        g_l = Gf(mesh=tt_mesh, target_shape=(2, 2))
        g_g = Gf(mesh=tt_mesh, target_shape=(2, 2))

        for t1, t2 in tt_mesh:
            e = expm(-1j * h_mat * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * -0.1 * e
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self.assertTrue(g.is_hermitian())
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(g.greater(), -conj(g_hc.greater()))
        assert_gfs_are_close(g.lesser(), -conj(g_hc.lesser()))
        assert_gfs_are_close(g.retarded(), conj(g_hc.advanced()))

        # Matrix-valued GF with an extra k-mesh component
        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(t_mesh, t_mesh, bz_mesh)

        g_l = Gf(mesh=mesh, target_shape=(2, 2))
        g_g = Gf(mesh=mesh, target_shape=(2, 2))
        for k in bz_mesh:
            eps = np.sum(k.value)
            h_mat = np.array([[eps, 0.5j], [-0.5j, 2.0 * eps]])
            for t1, t2 in MeshProduct(t_mesh, t_mesh):
                e = expm(-1j * h_mat * (t1.value - t2.value))
                g_g[t1, t2, k] = -1j * (1.0 - 0.1) * e
                g_l[t1, t2, k] = -1j * -0.1 * e
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self.assertTrue(g.is_hermitian())
        g_hc = herm_conj(g)
        assert_keldysh_gf_almost_equal(g_hc, g)
        assert_gfs_are_close(g.greater(), -conj(g_hc.greater()))
        assert_gfs_are_close(g.lesser(), -conj(g_hc.lesser()))
        assert_gfs_are_close(g.retarded(), conj(g_hc.advanced()))

        # Tensor-valued GF with an extra k-mesh component
        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        mesh = MeshProduct(t_mesh, t_mesh, bz_mesh)

        g_l = Gf(mesh=mesh, target_shape=(2, 3, 4))
        g_g = Gf(mesh=mesh, target_shape=(2, 3, 4))
        for k in bz_mesh:
            eps = np.sum(k.value)
            ten = np.arange(2 * 3 * 4).reshape((2, 3, 4))
            for t1, t2 in MeshProduct(t_mesh, t_mesh):
                e = np.exp(-1j * (t1.value - t2.value))
                g_g[t1, t2, k] = -1j * (1.0 - 0.1) * ten * e
                g_l[t1, t2, k] = -1j * -0.1 * ten * e
        g = KeldyshGF.from_lesser_greater(g_l, g_g, n_left_target_axes=1)
        self.assertFalse(g.is_hermitian())
        g_hc = herm_conj(g)
        assert_gfs_are_close(g.greater(),
                             -conj(g_hc.greater(), n_left_indices=2))
        assert_gfs_are_close(g.lesser(),
                             -conj(g_hc.lesser(), n_left_indices=2))
        assert_gfs_are_close(g.retarded(),
                             conj(g_hc.advanced(), n_left_indices=2))

    def _conv_g_l(self, occ1, occ2, eps1, eps2, t1, t2):
        dt = t1 - t2
        return (-1j / (eps1 - eps2) * (
                (1 - occ1) * np.exp(-1j * eps1 * dt)
                - (1 - occ2) * np.exp(-1j * eps2 * dt)
                + (occ1 - occ2) * np.exp(-1j * (eps1 * t1 - eps2 * t2))
                ),
                1j / (eps1 - eps2) * (
                occ1 * np.exp(-1j * eps1 * dt)
                - occ2 * np.exp(-1j * eps2 * dt)
                - (occ1 - occ2) * np.exp(-1j * (eps1 * t1 - eps2 * t2))
                )
                )

    def test_conv_scalar(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        occ = [0.1, 0.2]
        eps = [0.6, 0.7]

        g_l = [Gf(mesh=tt_mesh, target_shape=()) for _ in range(2)]
        g_g = [Gf(mesh=tt_mesh, target_shape=()) for _ in range(2)]
        for i in range(2):
            for t1, t2 in tt_mesh:
                dt = t1 - t2
                g_g[i][t1, t2], g_l[i][t1, t2] = single_state_g_l(occ[i],
                                                                  eps[i],
                                                                  dt)
        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)
        assert_keldysh_gf_almost_equal(herm_conj(g1g2),
                                       herm_conj(g2) @ herm_conj(g1))

        g1g2_ref_l = Gf(mesh=tt_mesh, target_shape=())
        g1g2_ref_g = Gf(mesh=tt_mesh, target_shape=())
        for t1, t2 in tt_mesh:
            g1g2_ref_g[t1, t2], g1g2_ref_l[t1, t2] = \
                self._conv_g_l(*occ, *eps, t1, t2)

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_conv_matrix(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        occ = [0.1, 0.2]
        eps = [0.6, 0.7]
        shapes = ((2, 4), (4, 1))

        g_l = [Gf(mesh=tt_mesh, target_shape=shapes[i]) for i in range(2)]
        g_g = [Gf(mesh=tt_mesh, target_shape=shapes[i]) for i in range(2)]
        for i in range(2):
            for (m, n) in np.ndindex(shapes[i]):
                for t1, t2 in tt_mesh:
                    dt = t1 - t2
                    g_val, l_val = single_state_g_l(occ[i], eps[i], dt)
                    g_l[i][t1, t2][m, n] = (m + 1) * (n + 1) * l_val
                    g_g[i][t1, t2][m, n] = (m + 1) * (n + 1) * g_val
        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)
        assert_keldysh_gf_almost_equal(herm_conj(g1g2),
                                       herm_conj(g2) @ herm_conj(g1))

        g1g2_ref_l = Gf(mesh=tt_mesh, target_shape=(2, 1))
        g1g2_ref_g = Gf(mesh=tt_mesh, target_shape=(2, 1))
        x = 4 * (1 + 4) * (1 + 2 * 4) / 6
        for (m, n) in np.ndindex(2, 1):
            for t1, t2 in tt_mesh:
                g_val, l_val = self._conv_g_l(*occ, *eps, t1, t2)
                g1g2_ref_g[t1, t2][m, n] = (m + 1) * x * (n + 1) * g_val
                g1g2_ref_l[t1, t2][m, n] = (m + 1) * x * (n + 1) * l_val

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_conv_scalar_bz(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        occ = [0.1, 0.2]
        eps_k = [np.array([0.6 + 0.01 * np.cos(k.value[0]) for k in bz_mesh]),
                 np.array([0.7 - 0.01 * np.cos(k.value[0]) for k in bz_mesh])]

        ttk_mesh = MeshProduct(*tt_mesh.components, bz_mesh)
        g_l = [Gf(mesh=ttk_mesh, target_shape=()) for _ in range(2)]
        g_g = [Gf(mesh=ttk_mesh, target_shape=()) for _ in range(2)]
        for i in range(2):
            for k, eps in zip(bz_mesh, eps_k[i]):
                for t1, t2 in tt_mesh:
                    dt = t1 - t2
                    g_g[i][t1, t2, k], g_l[i][t1, t2, k] = \
                        single_state_g_l(occ[i], eps, dt)
        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)
        assert_keldysh_gf_almost_equal(herm_conj(g1g2),
                                       herm_conj(g2) @ herm_conj(g1))

        g1g2_ref_l = Gf(mesh=ttk_mesh, target_shape=())
        g1g2_ref_g = Gf(mesh=ttk_mesh, target_shape=())
        for k, eps in zip(bz_mesh, zip(eps_k[0], eps_k[1])):
            for t1, t2 in tt_mesh:
                g1g2_ref_g[t1, t2, k], g1g2_ref_l[t1, t2, k] = \
                    self._conv_g_l(*occ, *eps, t1, t2)

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_conv_matrix_bz(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        n_k = 4
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        occ = [0.1, 0.2]
        eps_k = [np.array([0.6 + 0.01 * np.cos(k.value[0]) for k in bz_mesh]),
                 np.array([0.7 - 0.01 * np.cos(k.value[0]) for k in bz_mesh])]
        shapes = ((2, 4), (4, 1))

        ttk_mesh = MeshProduct(*tt_mesh.components, bz_mesh)
        g_l = [Gf(mesh=ttk_mesh, target_shape=shapes[i]) for i in range(2)]
        g_g = [Gf(mesh=ttk_mesh, target_shape=shapes[i]) for i in range(2)]
        for i in range(2):
            for (m, n) in np.ndindex(shapes[i]):
                for k, eps in zip(bz_mesh, eps_k[i]):
                    for t1, t2 in tt_mesh:
                        dt = t1 - t2
                        g_val, l_val = single_state_g_l(occ[i], eps, dt)
                        g_l[i][t1, t2, k][m, n] = (m + 1) * (n + 1) * l_val
                        g_g[i][t1, t2, k][m, n] = (m + 1) * (n + 1) * g_val

        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)

        g1g2_ref_l = Gf(mesh=ttk_mesh, target_shape=(2, 1))
        g1g2_ref_g = Gf(mesh=ttk_mesh, target_shape=(2, 1))
        x = 4 * (1 + 4) * (1 + 2 * 4) / 6
        for k, eps in zip(bz_mesh, zip(eps_k[0], eps_k[1])):
            for (m, n) in np.ndindex(2, 1):
                for t1, t2 in tt_mesh:
                    g_val, l_val = self._conv_g_l(*occ, *eps, t1, t2)
                    g1g2_ref_g[t1, t2, k][m, n] = (m + 1) * x * (n + 1) * g_val
                    g1g2_ref_l[t1, t2, k][m, n] = (m + 1) * x * (n + 1) * l_val

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_conv_scalar_bz1_bz2(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        n_k1, n_k2 = 4, 3
        bz_meshes = [MeshBrillouinZone(BrillouinZone(self.bl), n_k1),
                     MeshBrillouinZone(BrillouinZone(self.bl), n_k2)]
        ttk_meshes = [MeshProduct(*tt_mesh.components, bz_meshes[i])
                      for i in range(2)]

        occ = [0.1, 0.2]
        eps_k = [
            np.array([0.6 + 0.01 * np.cos(k.value[0]) for k in bz_meshes[0]]),
            np.array([0.7 - 0.01 * np.cos(k.value[0]) for k in bz_meshes[1]])
        ]

        g_l = [Gf(mesh=ttk_meshes[i], target_shape=()) for i in range(2)]
        g_g = [Gf(mesh=ttk_meshes[i], target_shape=()) for i in range(2)]
        for i in range(2):
            for k, eps in zip(bz_meshes[i], eps_k[i]):
                for t1, t2 in tt_mesh:
                    dt = t1 - t2
                    g_g[i][t1, t2, k], g_l[i][t1, t2, k] = \
                        single_state_g_l(occ[i], eps, dt)
        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)

        ttk1k2_mesh = MeshProduct(*tt_mesh.components, *bz_meshes)
        g1g2_ref_l = Gf(mesh=ttk1k2_mesh, target_shape=())
        g1g2_ref_g = Gf(mesh=ttk1k2_mesh, target_shape=())
        for (k1, eps1), (k2, eps2) in product(zip(bz_meshes[0], eps_k[0]),
                                              zip(bz_meshes[1], eps_k[1])):
            for t1, t2 in tt_mesh:
                g1g2_ref_g[t1, t2, k1, k2], g1g2_ref_l[t1, t2, k1, k2] = \
                    self._conv_g_l(*occ, eps1, eps2, t1, t2)

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_conv_matrix_bz1_bz2(self):
        t_max = 10.0
        n_t = 51
        t_mesh = MeshReTime(0.0, t_max, n_t)
        tt_mesh = MeshProduct(t_mesh, t_mesh)

        n_k1, n_k2 = 4, 3
        bz_meshes = [MeshBrillouinZone(BrillouinZone(self.bl), n_k1),
                     MeshBrillouinZone(BrillouinZone(self.bl), n_k2)]
        ttk_meshes = [MeshProduct(*tt_mesh.components, bz_meshes[i])
                      for i in range(2)]

        occ = [0.1, 0.2]
        eps_k = [
            np.array([0.6 + 0.01 * np.cos(k.value[0]) for k in bz_meshes[0]]),
            np.array([0.7 - 0.01 * np.cos(k.value[0]) for k in bz_meshes[1]])
        ]
        shapes = ((2, 4), (4, 1))

        g_l = [Gf(mesh=ttk_meshes[i], target_shape=shapes[i]) for i in range(2)]
        g_g = [Gf(mesh=ttk_meshes[i], target_shape=shapes[i]) for i in range(2)]
        for i in range(2):
            for k, eps in zip(bz_meshes[i], eps_k[i]):
                for (m, n) in np.ndindex(shapes[i]):
                    for t1, t2 in tt_mesh:
                        dt = t1 - t2
                        g_val, l_val = single_state_g_l(occ[i], eps, dt)
                        g_l[i][t1, t2, k][m, n] = (m + 1) * (n + 1) * l_val
                        g_g[i][t1, t2, k][m, n] = (m + 1) * (n + 1) * g_val

        g1 = KeldyshGF.from_lesser_greater(g_l[0], g_g[0])
        g2 = KeldyshGF.from_lesser_greater(g_l[1], g_g[1])

        g1g2 = g1 @ g2

        assert_keldysh_gf_almost_equal(herm_conj(herm_conj(g1g2)), g1g2)

        ttk1k2_mesh = MeshProduct(*tt_mesh.components, *bz_meshes)
        g1g2_ref_l = Gf(mesh=ttk1k2_mesh, target_shape=(2, 1))
        g1g2_ref_g = Gf(mesh=ttk1k2_mesh, target_shape=(2, 1))
        x = 4 * (1 + 4) * (1 + 2 * 4) / 6
        for (k1, eps1), (k2, eps2) in product(zip(bz_meshes[0], eps_k[0]),
                                              zip(bz_meshes[1], eps_k[1])):
            for (m, n) in np.ndindex(2, 1):
                for t1, t2 in tt_mesh:
                    g_val, l_val = self._conv_g_l(*occ, eps1, eps2, t1, t2)
                    g1g2_ref_g[t1, t2, k1, k2][m, n] = \
                        (m + 1) * x * (n + 1) * g_val
                    g1g2_ref_l[t1, t2, k1, k2][m, n] = \
                        (m + 1) * x * (n + 1) * l_val

        g1g2_ref = KeldyshGF.from_lesser_greater(g1g2_ref_l, g1g2_ref_g)
        assert_keldysh_gf_almost_equal(g1g2, g1g2_ref, 1e-10)

    def test_keldysh_vertex3(self):
        t_mesh1 = MeshReTime(0, 6.0, 7)
        t_mesh2 = MeshReTime(0, 6.0, 8)
        t_mesh3 = MeshReTime(0, 6.0, 9)
        ttt_mesh = MeshProduct(t_mesh1, t_mesh2, t_mesh3)

        def make_time_piece(x):
            g = Gf(mesh=ttt_mesh, target_shape=())
            g.data[:] = x
            return g
        G = {(0, 1, 2): make_time_piece(1.0),
             (0, 2, 1): make_time_piece(2.0),
             (1, 0, 2): make_time_piece(3.0),
             (1, 2, 0): make_time_piece(4.0),
             (2, 0, 1): make_time_piece(5.0),
             (2, 1, 0): make_time_piece(6.0)}

        Lambda = KeldyshGF.from_vertex3_pieces(G)
        for a0, a1, a2 in product(Branch, repeat=3):
            for t0, t1, t2 in ttt_mesh:
                self.assertNotEqual(Lambda[CP(a0, t0), CP(a1, t1), CP(a2, t2)],
                                    0)

        t = next(iter(t_mesh1))

        Lambda[CP(BW, t), CP(FW, t), CP(BW, t)] = 3.0
        self.assertEqual(Lambda[CP(BW, t), CP(FW, t), CP(BW, t)], 3.0)

        ones_time_mat = np.ones((7, 8, 9))
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


class TestSingular2PKeldyshGF(unittest.TestCase):
    """Singular 2-point Green's function"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 10.0
        cls.n_t = 7
        cls.t_mesh = MeshReTime(0.0, cls.t_max, cls.n_t)

        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice

        cls.n_k = [4, 3]
        cls.bz_meshes = [
            MeshBrillouinZone(BrillouinZone(cls.bl), n_k) for n_k in cls.n_k
        ]

    def test_basic(self):
        g = Singular2PKeldyshGF(mesh=self.t_mesh, arg_index_shapes=((3,), (4,)))

        self.assertEqual(g.mesh, MeshProduct(self.t_mesh))
        self.assertEqual(g.time_mesh, self.t_mesh)
        self.assertEqual(g.non_time_mesh, MeshProduct())
        self.assertEqual(g.components.shape, (2,))

        t0, t1 = list(self.t_mesh)[:2]

        # __getitem__()
        g.components[1].data[:] = 2.0
        assert_array_equal(g[Branch.BACKWARD].data, 2.0 * np.ones((7, 3, 4)))
        g.components[1].data[0, :] = 3.0
        assert_array_equal(g[CP(Branch.BACKWARD, t0)], 3.0 * np.ones((3, 4)))
        assert_array_equal(g[CP(Branch.BACKWARD, t1)], 2.0 * np.ones((3, 4)))

        # __setitem__()
        g[Branch.FORWARD].data[:] = 4.0
        assert_array_equal(g.components[0].data, 4.0 * np.ones((7, 3, 4)))
        g[CP(Branch.FORWARD, t0)] = 5.0
        assert_array_equal(g.components[0].data[0, :], 5.0 * np.ones((3, 4)))
        assert_array_equal(g.components[0].data[1, :], 4.0 * np.ones((3, 4)))

        # Arithmetics
        self.assertEqual(g, g)
        assert_array_equal((3 * g)[FW].data, 3 * g[FW].data)
        assert_array_equal((g * 3)[FW].data, 3 * g[FW].data)
        assert_array_equal((-g)[FW].data, -g[FW].data)
        assert_array_equal((g + g)[FW].data, 2 * g[FW].data)
        assert_array_equal((g - g)[FW].data, 0 * g[FW].data)

        # from_retime()
        eps_t = Gf(mesh=self.t_mesh, target_shape=())
        eps_t_m = Gf(mesh=self.t_mesh, target_shape=(3, 2))
        for t in self.t_mesh:
            eps_t[t] = np.cos(t.value)
            eps_t_m[t] = np.array([[1, 2], [3, 4], [5, 6]]) * np.cos(t.value)

        g = Singular2PKeldyshGF.from_retime(eps_t)
        self.assertEqual(g.arg_index_shapes, ((), ()))
        assert_array_equal(g.components[0].data, eps_t.data)
        assert_array_equal(g.components[1].data, eps_t.data)

        g_m1 = Singular2PKeldyshGF.from_retime(eps_t_m)
        self.assertEqual(g_m1.arg_index_shapes, ((3,), (2,)))
        assert_array_equal(g_m1.components[0].data, eps_t_m.data)
        assert_array_equal(g_m1.components[1].data, eps_t_m.data)

        g_m2 = Singular2PKeldyshGF.from_retime(eps_t_m, n_left_target_axes=2)
        self.assertEqual(g_m2.arg_index_shapes, ((3, 2), ()))
        assert_array_equal(g_m2.components[0].data, eps_t_m.data)
        assert_array_equal(g_m2.components[1].data, eps_t_m.data)

    def test_basic_bz(self):
        mesh = MeshProduct(self.t_mesh, self.bz_meshes[0])
        g = Singular2PKeldyshGF(mesh=mesh, arg_index_shapes=((3,), (4,)))

        self.assertEqual(g.mesh, mesh)
        self.assertEqual(g.time_mesh, self.t_mesh)
        self.assertEqual(g.non_time_mesh, MeshProduct(self.bz_meshes[0]))
        self.assertEqual(g.components.shape, (2,))

        t0, t1 = list(self.t_mesh)[:2]

        # __getitem__()
        g.components[1].data[:] = 2.0
        assert_array_equal(g[Branch.BACKWARD].data,
                           2.0 * np.ones((7, self.n_k[0], 3, 4)))
        g.components[1].data[0, :] = 3.0
        assert_array_equal(g[CP(Branch.BACKWARD, t0)].data,
                           3.0 * np.ones((self.n_k[0], 3, 4)))
        assert_array_equal(g[CP(Branch.BACKWARD, t1)].data,
                           2.0 * np.ones((self.n_k[0], 3, 4)))

        # __setitem__()
        g[Branch.FORWARD].data[:] = 4.0
        assert_array_equal(g.components[0].data,
                           4.0 * np.ones((7, self.n_k[0], 3, 4)))
        g[CP(Branch.FORWARD, t0)].data[:] = 5.0
        assert_array_equal(g.components[0].data[0, :],
                           5.0 * np.ones((self.n_k[0], 3, 4)))
        assert_array_equal(g.components[0].data[1, :],
                           4.0 * np.ones((self.n_k[0], 3, 4)))

        for i, k in enumerate(self.bz_meshes[0]):
            # __getitem__()
            g.components[1].data[:, i] = i
            assert_array_equal(g[Branch.BACKWARD, t0, k],
                               i * np.ones((3, 4)))
            # __setitem__()
            g[Branch.BACKWARD, t0, k] = i
            assert_array_equal(g.components[1].data[0, i, :],
                               i * np.ones((3, 4)))

        # Arithmetics
        self.assertEqual(g, g)
        assert_array_equal((3 * g)[FW].data, 3 * g[FW].data)
        assert_array_equal((g * 3)[FW].data, 3 * g[FW].data)
        assert_array_equal((-g)[FW].data, -g[FW].data)
        assert_array_equal((g + g)[FW].data, 2 * g[FW].data)
        assert_array_equal((g - g)[FW].data, 0 * g[FW].data)

        # from_retime()
        eps_t = Gf(mesh=mesh, target_shape=())
        eps_t_m = Gf(mesh=mesh, target_shape=(2, 1))
        for t, k in mesh:
            eps_t[t, k] = k[0] * np.cos(t.value)
            eps_t_m[t] = k[0] * np.array([[1], [2]]) * np.cos(t.value)

        g = Singular2PKeldyshGF.from_retime(eps_t)
        self.assertEqual(g.arg_index_shapes, ((), ()))
        assert_array_equal(g.components[0].data, eps_t.data)
        assert_array_equal(g.components[1].data, eps_t.data)

        g_m1 = Singular2PKeldyshGF.from_retime(eps_t_m)
        self.assertEqual(g_m1.arg_index_shapes, ((2,), (1,)))
        assert_array_equal(g_m1.components[0].data, eps_t_m.data)
        assert_array_equal(g_m1.components[1].data, eps_t_m.data)

        g_m2 = Singular2PKeldyshGF.from_retime(eps_t_m, n_left_target_axes=2)
        self.assertEqual(g_m2.arg_index_shapes, ((2, 1), ()))
        assert_array_equal(g_m2.components[0].data, eps_t_m.data)
        assert_array_equal(g_m2.components[1].data, eps_t_m.data)

    @classmethod
    def _test_conv(cls, eps_t1, eps_t2, eps_t12_ref):
        sg1 = Singular2PKeldyshGF.from_retime(eps_t1)
        sg2 = Singular2PKeldyshGF.from_retime(eps_t2)

        sg1sg2 = sg1 @ sg2

        sg1sg2_ref = Singular2PKeldyshGF.from_retime(eps_t12_ref)
        assert_singular_2p_keldysh_gf_almost_equal(sg1sg2, sg1sg2_ref, 1e-10)

    def test_conv_scalar(self):
        eps_t1 = Gf(mesh=self.t_mesh, target_shape=())
        eps_t2 = Gf(mesh=self.t_mesh, target_shape=())
        eps_t12_ref = Gf(mesh=MeshProduct(self.t_mesh), target_shape=())
        for t in self.t_mesh:
            eps_t1[t] = np.cos(t.value)
            eps_t2[t] = np.cos(2 * t.value)
            eps_t12_ref[t] = np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    def test_conv_matrix(self):
        eps_t1 = Gf(mesh=self.t_mesh, target_shape=(2, 4))
        eps_t2 = Gf(mesh=self.t_mesh, target_shape=(4, 1))
        eps_t12_ref = Gf(mesh=MeshProduct(self.t_mesh), target_shape=(2, 1))
        for t, (m, n) in product(self.t_mesh, np.ndindex((2, 4))):
            eps_t1[t][m, n] = (m + 1) * (n + 1) * np.cos(t.value)
        for t, (m, n) in product(self.t_mesh, np.ndindex((4, 1))):
            eps_t2[t][m, n] = (m + 1) * (n + 1) * np.cos(2 * t.value)
        for t, (m, n) in product(self.t_mesh, np.ndindex((2, 1))):
            eps_t12_ref[t][m, n] = (m + 1) * 30 * (n + 1) * \
                np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    def test_conv_scalar_bz(self):
        mesh = MeshProduct(self.t_mesh, self.bz_meshes[0])
        eps_t1 = Gf(mesh=mesh, target_shape=())
        eps_t2 = Gf(mesh=mesh, target_shape=())
        eps_t12_ref = Gf(mesh=mesh, target_shape=())
        for t, k in mesh:
            eps_t1[t, k] = k[0] * np.cos(t.value)
            eps_t2[t, k] = k[0] * np.cos(2 * t.value)
            eps_t12_ref[t, k] = k[0]**2 * np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    def test_conv_matrix_bz(self):
        mesh = MeshProduct(self.t_mesh, self.bz_meshes[0])
        eps_t1 = Gf(mesh=mesh, target_shape=(2, 4))
        eps_t2 = Gf(mesh=mesh, target_shape=(4, 1))
        eps_t12_ref = Gf(mesh=mesh, target_shape=(2, 1))
        for t, k in mesh:
            for m, n in np.ndindex((2, 4)):
                eps_t1[t, k][m, n] = \
                    (m + 1) * (n + 1) * k[0] * np.cos(t.value)
            for m, n in np.ndindex((4, 1)):
                eps_t2[t, k][m, n] = \
                    (m + 1) * (n + 1) * k[0] * np.cos(2 * t.value)
            for m, n in np.ndindex((2, 1)):
                eps_t12_ref[t, k][m, n] = \
                    (m + 1) * 30 * (n + 1) * k[0]**2 * \
                    np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    def test_conv_scalar_bz1_bz2(self):
        tk_meshes = [MeshProduct(self.t_mesh, bz_mesh)
                     for bz_mesh in self.bz_meshes]
        tk1k2_mesh = MeshProduct(self.t_mesh, *self.bz_meshes)
        eps_t1 = Gf(mesh=tk_meshes[0], target_shape=())
        eps_t2 = Gf(mesh=tk_meshes[1], target_shape=())
        eps_t12_ref = Gf(mesh=tk1k2_mesh, target_shape=())
        for t, k in tk_meshes[0]:
            eps_t1[t, k] = k[0] * np.cos(t.value)
        for t, k in tk_meshes[1]:
            eps_t2[t, k] = k[0] * np.cos(2 * t.value)
        for t, k1, k2 in tk1k2_mesh:
            eps_t12_ref[t, k1, k2] = \
                k1[0] * k2[0] * np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    def test_conv_matrix_bz1_bz2(self):
        tk_meshes = [MeshProduct(self.t_mesh, bz_mesh)
                     for bz_mesh in self.bz_meshes]
        tk1k2_mesh = MeshProduct(self.t_mesh, *self.bz_meshes)
        eps_t1 = Gf(mesh=tk_meshes[0], target_shape=(2, 4))
        eps_t2 = Gf(mesh=tk_meshes[1], target_shape=(4, 1))
        eps_t12_ref = Gf(mesh=tk1k2_mesh, target_shape=(2, 1))

        for t, k in tk_meshes[0]:
            for m, n in np.ndindex((2, 4)):
                eps_t1[t, k][m, n] = (m + 1) * (n + 1) * k[0] * np.cos(t.value)
        for t, k in tk_meshes[1]:
            for m, n in np.ndindex((4, 1)):
                eps_t2[t, k][m, n] = \
                    (m + 1) * (n + 1) * k[0] * np.cos(2 * t.value)
        for t, k1, k2 in tk1k2_mesh:
            for m, n in np.ndindex((2, 1)):
                eps_t12_ref[t, k1, k2][m, n] = \
                    (m + 1) * 30 * (n + 1) * k1[0] * k2[0] \
                    * np.cos(t.value) * np.cos(2 * t.value)
        self._test_conv(eps_t1, eps_t2, eps_t12_ref)

    @classmethod
    def _test_conv_reg_sing(cls, g, sg,
                            g_sg_ref_l, g_sg_ref_g,
                            sg_g_ref_l, sg_g_ref_g):
        g_sg = g @ sg
        sg_g = sg @ g

        assert_gfs_are_close(g_sg.lesser(), g_sg_ref_l)
        assert_gfs_are_close(g_sg.greater(), g_sg_ref_g)
        assert_gfs_are_close(sg_g.lesser(), sg_g_ref_l)
        assert_gfs_are_close(sg_g.greater(), sg_g_ref_g)

    def test_conv_reg_sing_scalar(self):
        sg = Singular2PKeldyshGF(mesh=self.t_mesh, arg_index_shapes=((), ()))
        for t in self.t_mesh:
            sg[FW][t] = np.cos(t.value)
            sg[BW][t] = np.sin(t.value)

        tt_mesh = MeshProduct(self.t_mesh, self.t_mesh)

        g_l = Gf(mesh=tt_mesh, target_shape=())
        g_g = Gf(mesh=tt_mesh, target_shape=())
        g_sg_ref_l = Gf(mesh=tt_mesh, target_shape=())
        g_sg_ref_g = Gf(mesh=tt_mesh, target_shape=())
        sg_g_ref_l = Gf(mesh=tt_mesh, target_shape=())
        sg_g_ref_g = Gf(mesh=tt_mesh, target_shape=())
        for t1, t2 in tt_mesh:
            dt = t1 - t2
            g_g[t1, t2], g_l[t1, t2] = single_state_g_l(0.1, 0.6, dt)
            g_sg_ref_l[t1, t2] = g_l[t1, t2] * sg[BW][t2]
            g_sg_ref_g[t1, t2] = g_g[t1, t2] * sg[FW][t2]
            sg_g_ref_l[t1, t2] = sg[FW][t1] * g_l[t1, t2]
            sg_g_ref_g[t1, t2] = sg[BW][t1] * g_g[t1, t2]
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)

    def test_conv_reg_sing_matrix(self):
        sg = Singular2PKeldyshGF(mesh=self.t_mesh,
                                 arg_index_shapes=((2,), (2,)))
        for t in self.t_mesh:
            sg[FW][t] = np.array([[2, 3], [4, 5]]) * np.cos(t.value)
            sg[BW][t] = np.array([[6, 7], [8, 9]]) * np.sin(t.value)

        tt_mesh = MeshProduct(self.t_mesh, self.t_mesh)

        g_l = Gf(mesh=tt_mesh, target_shape=(2, 2))
        g_g = Gf(mesh=tt_mesh, target_shape=(2, 2))
        g_sg_ref_l = Gf(mesh=tt_mesh, target_shape=(2, 2))
        g_sg_ref_g = Gf(mesh=tt_mesh, target_shape=(2, 2))
        sg_g_ref_l = Gf(mesh=tt_mesh, target_shape=(2, 2))
        sg_g_ref_g = Gf(mesh=tt_mesh, target_shape=(2, 2))
        for t1, t2 in tt_mesh:
            dt = t1 - t2
            g_val, l_val = single_state_g_l(0.1, 0.6, dt)
            g_g[t1, t2] = np.array([[10, 11], [12, 13]]) * g_val
            g_l[t1, t2] = np.array([[14, 15], [16, 17]]) * l_val
            g_sg_ref_l[t1, t2] = g_l[t1, t2] @ sg[BW][t2]
            g_sg_ref_g[t1, t2] = g_g[t1, t2] @ sg[FW][t2]
            sg_g_ref_l[t1, t2] = sg[FW][t1] @ g_l[t1, t2]
            sg_g_ref_g[t1, t2] = sg[BW][t1] @ g_g[t1, t2]
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)

    def test_conv_reg_sing_scalar_bz(self):
        bz_mesh = self.bz_meshes[0]
        tk_mesh = MeshProduct(self.t_mesh, bz_mesh)

        sg = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((), ()))
        for t, k in tk_mesh:
            sg[FW][t, k] = k[0] * np.cos(t.value)
            sg[BW][t, k] = k[0] * np.sin(t.value)

        ttk_mesh = MeshProduct(self.t_mesh, self.t_mesh, bz_mesh)

        g_l = Gf(mesh=ttk_mesh, target_shape=())
        g_g = Gf(mesh=ttk_mesh, target_shape=())
        g_sg_ref_l = Gf(mesh=ttk_mesh, target_shape=())
        g_sg_ref_g = Gf(mesh=ttk_mesh, target_shape=())
        sg_g_ref_l = Gf(mesh=ttk_mesh, target_shape=())
        sg_g_ref_g = Gf(mesh=ttk_mesh, target_shape=())
        for t1, t2, k in ttk_mesh:
            eps = 0.6 + 0.01 * np.cos(k.value[0])
            dt = t1 - t2
            g_g[t1, t2, k], g_l[t1, t2, k] = single_state_g_l(0.1, eps, dt)
            g_sg_ref_l[t1, t2, k] = g_l[t1, t2, k] * sg[BW][t2, k]
            g_sg_ref_g[t1, t2, k] = g_g[t1, t2, k] * sg[FW][t2, k]
            sg_g_ref_l[t1, t2, k] = sg[FW][t1, k] * g_l[t1, t2, k]
            sg_g_ref_g[t1, t2, k] = sg[BW][t1, k] * g_g[t1, t2, k]
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)

    def test_conv_reg_sing_matrix_bz(self):
        bz_mesh = self.bz_meshes[0]
        tk_mesh = MeshProduct(self.t_mesh, bz_mesh)

        sg = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
        for t, k in tk_mesh:
            sg[FW][t, k] = k[0] * np.array([[2, 3], [4, 5]]) * np.cos(t.value)
            sg[BW][t, k] = k[0] * np.array([[6, 7], [8, 9]]) * np.sin(t.value)

        ttk_mesh = MeshProduct(self.t_mesh, self.t_mesh, bz_mesh)

        g_l = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        g_g = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        g_sg_ref_l = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        g_sg_ref_g = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        sg_g_ref_l = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        sg_g_ref_g = Gf(mesh=ttk_mesh, target_shape=(2, 2))
        for t1, t2, k in ttk_mesh:
            eps = 0.6 + 0.01 * np.cos(k.value[0])
            dt = t1 - t2
            g_val, l_val = single_state_g_l(0.1, eps, dt)
            g_g[t1, t2, k] = np.array([[10, 11], [12, 13]]) * g_val
            g_l[t1, t2, k] = np.array([[14, 15], [16, 17]]) * l_val
            g_sg_ref_l[t1, t2, k] = g_l[t1, t2, k] @ sg[BW][t2, k]
            g_sg_ref_g[t1, t2, k] = g_g[t1, t2, k] @ sg[FW][t2, k]
            sg_g_ref_l[t1, t2, k] = sg[FW][t1, k] @ g_l[t1, t2, k]
            sg_g_ref_g[t1, t2, k] = sg[BW][t1, k] @ g_g[t1, t2, k]
        g = KeldyshGF.from_lesser_greater(g_l, g_g)
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)

    def test_conv_reg_sing_scalar_bz1_bz2(self):
        tk1_mesh = MeshProduct(self.t_mesh, self.bz_meshes[0])

        sg = Singular2PKeldyshGF(mesh=tk1_mesh, arg_index_shapes=((), ()))
        for t, k in tk1_mesh:
            sg[FW][t, k] = k[0] * np.cos(t.value)
            sg[BW][t, k] = k[0] * np.sin(t.value)

        ttk2_mesh = MeshProduct(self.t_mesh, self.t_mesh, self.bz_meshes[1])

        g_l = Gf(mesh=ttk2_mesh, target_shape=())
        g_g = Gf(mesh=ttk2_mesh, target_shape=())
        for t1, t2, k in ttk2_mesh:
            eps = 0.6 + 0.01 * np.cos(k.value[0])
            dt = t1 - t2
            g_g[t1, t2, k], g_l[t1, t2, k] = single_state_g_l(0.1, eps, dt)
        g = KeldyshGF.from_lesser_greater(g_l, g_g)

        ttk1k2_mesh = MeshProduct(self.t_mesh, self.t_mesh,
                                  self.bz_meshes[0], self.bz_meshes[1])
        ttk2k1_mesh = MeshProduct(self.t_mesh, self.t_mesh,
                                  self.bz_meshes[1], self.bz_meshes[0])
        g_sg_ref_l = Gf(mesh=ttk2k1_mesh, target_shape=())
        g_sg_ref_g = Gf(mesh=ttk2k1_mesh, target_shape=())
        sg_g_ref_l = Gf(mesh=ttk1k2_mesh, target_shape=())
        sg_g_ref_g = Gf(mesh=ttk1k2_mesh, target_shape=())
        for t1, t2, k1, k2 in ttk1k2_mesh:
            g_sg_ref_l[t1, t2, k2, k1] = g_l[t1, t2, k2] * sg[BW][t2, k1]
            g_sg_ref_g[t1, t2, k2, k1] = g_g[t1, t2, k2] * sg[FW][t2, k1]
            sg_g_ref_l[t1, t2, k1, k2] = sg[FW][t1, k1] * g_l[t1, t2, k2]
            sg_g_ref_g[t1, t2, k1, k2] = sg[BW][t1, k1] * g_g[t1, t2, k2]
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)

    def test_conv_reg_sing_matrix_bz1_bz2(self):
        tk1_mesh = MeshProduct(self.t_mesh, self.bz_meshes[0])

        sg = Singular2PKeldyshGF(mesh=tk1_mesh, arg_index_shapes=((2,), (2,)))
        for t, k in tk1_mesh:
            sg[FW][t, k] = np.array([[2, 3], [4, 5]]) * k[0] * np.cos(t.value)
            sg[BW][t, k] = np.array([[6, 7], [8, 9]]) * k[0] * np.sin(t.value)

        ttk2_mesh = MeshProduct(self.t_mesh, self.t_mesh, self.bz_meshes[1])

        g_l = Gf(mesh=ttk2_mesh, target_shape=(2, 2))
        g_g = Gf(mesh=ttk2_mesh, target_shape=(2, 2))
        for t1, t2, k in ttk2_mesh:
            eps = 0.6 + 0.01 * np.cos(k.value[0])
            dt = t1 - t2
            g_val, l_val = single_state_g_l(0.1, eps, dt)
            g_g[t1, t2, k] = np.array([[10, 11], [12, 13]]) * g_val
            g_l[t1, t2, k] = np.array([[14, 15], [16, 17]]) * l_val
        g = KeldyshGF.from_lesser_greater(g_l, g_g)

        ttk1k2_mesh = MeshProduct(self.t_mesh, self.t_mesh,
                                  self.bz_meshes[0], self.bz_meshes[1])
        ttk2k1_mesh = MeshProduct(self.t_mesh, self.t_mesh,
                                  self.bz_meshes[1], self.bz_meshes[0])
        g_sg_ref_l = Gf(mesh=ttk2k1_mesh, target_shape=(2, 2))
        g_sg_ref_g = Gf(mesh=ttk2k1_mesh, target_shape=(2, 2))
        sg_g_ref_l = Gf(mesh=ttk1k2_mesh, target_shape=(2, 2))
        sg_g_ref_g = Gf(mesh=ttk1k2_mesh, target_shape=(2, 2))
        for t1, t2, k1, k2 in ttk1k2_mesh:
            g_sg_ref_l[t1, t2, k2, k1] = g_l[t1, t2, k2] @ sg[BW][t2, k1]
            g_sg_ref_g[t1, t2, k2, k1] = g_g[t1, t2, k2] @ sg[FW][t2, k1]
            sg_g_ref_l[t1, t2, k1, k2] = sg[FW][t1, k1] @ g_l[t1, t2, k2]
            sg_g_ref_g[t1, t2, k1, k2] = sg[BW][t1, k1] @ g_g[t1, t2, k2]
        self._test_conv_reg_sing(g, sg,
                                 g_sg_ref_l, g_sg_ref_g, sg_g_ref_l, sg_g_ref_g)


if __name__ == '__main__':
    unittest.main()
