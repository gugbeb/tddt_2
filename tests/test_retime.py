import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np

from triqs.gf import MeshReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.retime import conj
from tddt.keldysh import from_lesser_greater, retarded, advanced


class test_retime(unittest.TestCase):
    """Auxiliary routines for Green's functions of real time"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 6.0
        cls.n_t = 7
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)

        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice

    def test_conj(self):
        mesh = MeshProduct(self.t_mesh, self.t_mesh)

        # Scalar-valued GF
        g_l = Gf(mesh=mesh, target_shape=())
        g_g = Gf(mesh=mesh, target_shape=())
        for t1, t2 in mesh:
            e = np.exp(-1j * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * (-0.1) * e
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_array_equal(g_adv.data, g_adv_ref.data)

        # Matrix-valued GF
        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        H = np.array([[1.0, 0.5, -0.1j],
                      [0.5, 2.0, 0.5],
                      [0.1j, 0.5, 3.0]])
        E, U = np.linalg.eig(H)
        for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
            dt = (t1.value - t2.value)
            e = U @ np.diag(np.exp(-1j * E * dt)) @ np.conj(U.T)
            g_g[t1, t2] = -1j * (1.0 - 0.1) * e
            g_l[t1, t2] = -1j * (-0.1) * e
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_array_almost_equal(g_adv.data, g_adv_ref.data, decimal=12)

        n_k = 3
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)
        eps_k = np.array([k.value[0] / np.pi + 1.0 for k in bz_mesh])
        mesh = MeshProduct(self.t_mesh, self.t_mesh, bz_mesh)

        # Matrix-valued GF with an extra mesh component
        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        for k, eps in zip(bz_mesh, eps_k):
            H = np.array([[eps, 0.5, -0.1j],
                          [0.5, 2.0 * eps, 0.5],
                          [0.1j, 0.5, 3.0 * eps]])
            E, U = np.linalg.eig(H)
            for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
                dt = (t1.value - t2.value)
                e = U @ np.diag(np.exp(-1j * E * dt)) @ np.conj(U.T)
                g_g[t1, t2, k] = -1j * (1.0 - 0.1) * e
                g_l[t1, t2, k] = -1j * (-0.1) * e
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_array_almost_equal(g_adv.data, g_adv_ref.data, decimal=12)


if __name__ == '__main__':
    unittest.main()
