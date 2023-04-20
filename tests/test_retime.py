import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.utility.comparison_tests import assert_gfs_are_close

from tddt.retime import conj, conv_ret_l, conv_l_adv
from tddt.keldysh import (from_lesser_greater,
                          retarded,
                          retarded_ext,
                          advanced,
                          advanced_ext)
from tddt.integration import GregoryIntegrator


class test_retime(unittest.TestCase):
    """Auxiliary routines for Green's functions of real time"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 6.0
        cls.n_t = 7
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)

        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice
        cls.n_k = 5
        cls.bz_mesh = MeshBrillouinZone(BrillouinZone(cls.bl), cls.n_k)
        cls.eps_k = np.array([k.value[0] / np.pi + 1.0 for k in cls.bz_mesh])

    def _make_g_l_g_g_scalar(self, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh)
        g_l = Gf(mesh=mesh, target_shape=())
        g_g = Gf(mesh=mesh, target_shape=())
        for t1, t2 in mesh:
            e = np.exp(-1j * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1 * x) * e
            g_l[t1, t2] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def _make_g_l_g_g_matrix(self, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh)
        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        H = np.array([[1.0, 0.5, -0.1j],
                      [0.5, 2.0, 0.5],
                      [0.1j, 0.5, 3.0]])
        E, U = np.linalg.eig(H)
        for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
            dt = (t1.value - t2.value)
            e = U @ np.diag(np.exp(-1j * x * E * dt)) @ np.conj(U.T)
            g_g[t1, t2] = -1j * (1.0 - 0.1 * x) * e
            g_l[t1, t2] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def _make_g_l_g_g_matrix_bz(self, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh, self.bz_mesh)

        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        for k, eps in zip(self.bz_mesh, self.eps_k):
            H = np.array([[eps, 0.5, -0.1j],
                          [0.5, 2.0 * eps, 0.5],
                          [0.1j, 0.5, 3.0 * eps]])
            E, U = np.linalg.eig(H)
            for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
                dt = (t1.value - t2.value)
                e = U @ np.diag(np.exp(-1j * x * E * dt)) @ np.conj(U.T)
                g_g[t1, t2, k] = -1j * (1.0 - 0.1 * x) * e
                g_l[t1, t2, k] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def test_conj(self):
        # Scalar-valued GF
        g_l, g_g = self._make_g_l_g_g_scalar()
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

        # Matrix-valued GF
        g_l, g_g = self._make_g_l_g_g_matrix()
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

        # Matrix-valued GF with an extra mesh component
        g_l, g_g = self._make_g_l_g_g_matrix_bz()
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

    def test_conv(self):
        order = 5
        w = GregoryIntegrator(order).weights(self.t_mesh)

        # Scalar-valued GF
        a_l, a_g = self._make_g_l_g_g_scalar(1.0)
        b_l, b_g = self._make_g_l_g_g_scalar(2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_l(a_ret_ext, b_l)
        a_l_b_adv = conv_l_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=())
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=())
        for n, m in product(range(self.n_t), range(self.n_t)):
            a_ret_b_l_ref.data[n, m] = sum(
                w[n, j] * a_ret_ext.data[n, j] * b_l.data[j, m]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m] = sum(
                w[m, j] * a_l.data[n, j] * b_adv_ext.data[j, m]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        # Matrix-valued GF
        a_l, a_g = self._make_g_l_g_g_matrix(1.0)
        b_l, b_g = self._make_g_l_g_g_matrix(2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_l(a_ret_ext, b_l)
        a_l_b_adv = conv_l_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=(3, 3))
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=(3, 3))
        for n, m, a, b, c in product(range(self.n_t),
                                     range(self.n_t),
                                     *[range(3)] * 3):
            a_ret_b_l_ref.data[n, m, a, b] += sum(
                w[n, j] * a_ret_ext.data[n, j, a, c] * b_l.data[j, m, c, b]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m, a, b] += sum(
                w[m, j] * a_l.data[n, j, a, c] * b_adv_ext.data[j, m, c, b]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        # Matrix-valued GF with an extra mesh component
        a_l, a_g = self._make_g_l_g_g_matrix_bz(1.0)
        b_l, b_g = self._make_g_l_g_g_matrix_bz(2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_l(a_ret_ext, b_l)
        a_l_b_adv = conv_l_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=(3, 3))
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=(3, 3))
        for n, m, K, a, b, c in product(range(self.n_t),
                                        range(self.n_t),
                                        range(self.n_k),
                                        *[range(3)] * 3):
            a_ret_b_l_ref.data[n, m, K, a, b] += sum(
                w[n, j]
                * a_ret_ext.data[n, j, K, a, c] * b_l.data[j, m, K, c, b]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m, K, a, b] += sum(
                w[m, j]
                * a_l.data[n, j, K, a, c] * b_adv_ext.data[j, m, K, c, b]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)


if __name__ == '__main__':
    unittest.main()
