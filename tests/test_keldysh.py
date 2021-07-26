import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from triqs.gf import MeshReTime, GfReTime, MeshProduct

from tddt.keldysh import *

class test_keldysh(unittest.TestCase):
    """Keldysh Green's functions and vertices"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 5.0
        cls.n_t = 6
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)
        cls.tt_mesh = MeshProduct(cls.t_mesh, cls.t_mesh)

    def test_keldysh_gf(self):
        # Construct from scalar-valued lesser and greater GF
        g_l = GfReTime(mesh = self.tt_mesh, target_shape = ())
        g_g = GfReTime(mesh = self.tt_mesh, target_shape = ())
        g = KeldyshGF(g_l, g_g)
        self.assertEqual(g.data.shape, (2, 2, self.n_t, self.n_t))

        # Construct from matrix-valued lesser and greater GF
        g_l = GfReTime(mesh = self.tt_mesh, target_shape = (2, 2))
        g_g = GfReTime(mesh = self.tt_mesh, target_shape = (2, 2))
        g_l.data[:] = 2.0
        g_g.data[:] = 3.0
        g = KeldyshGF(g_l[0, 1], g_g[1, 0])
        self.assertEqual(g.data.shape, (2, 2, self.n_t, self.n_t))

        # Check Aoki RMP Eq. (16)
        g11 = g[Branch.FORWARD, Branch.FORWARD]
        g12 = g[Branch.FORWARD, Branch.BACKWARD]
        g21 = g[Branch.BACKWARD, Branch.FORWARD]
        g22 = g[Branch.BACKWARD, Branch.BACKWARD]
        assert_array_almost_equal((g11 + g22).data, (g12 + g21).data)

        FW, BW = Branch.FORWARD, Branch.BACKWARD
        t = next(iter(self.t_mesh))

        g[BW, FW, t, t] = 3.0
        self.assertEqual(g[BW, FW, t, t], 3.0)


        g[BW, FW] = 2 * np.ones((self.n_t, self.n_t))
        assert_array_equal(g[BW, FW], 2*np.ones((self.n_t, self.n_t)))

        # Multiplication by a scalar
        g *= 3
        self.assertEqual(g[BW, FW, t, t], 6.0)

        # Addition
        g += g
        self.assertEqual(g[BW, FW, t, t], 12.0)
        self.assertEqual((g + g)[BW, FW, t, t], 24.0)

        # Subtraction
        g -= 0.5 * g
        self.assertEqual(g[BW, FW, t, t], 6.0)
        self.assertEqual((g - g)[BW, FW, t, t], 0.0)

        # Unary minus
        self.assertEqual((-g)[BW, FW, t, t], -6.0)
