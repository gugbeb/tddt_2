import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from triqs.gf import MeshReTime, GfReTime, MeshProduct

from tddt.keldysh import *

CP = ContourPoint

class test_keldysh(unittest.TestCase):
    """Keldysh Green's functions and vertices"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 5.0
        cls.n_t = 6
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)
        cls.tt_mesh = MeshProduct(cls.t_mesh, cls.t_mesh)
        cls.t_points = list(cls.t_mesh)

    def test_contour_ordering2(self):
        FW, BW = Branch.FORWARD, Branch.BACKWARD
        t1 = self.t_points[2]
        t2 = self.t_points[3]
        order = contour_ordering2

        # (+,+)
        self.assertEqual(order(CP(FW, t2), CP(FW, t1)), [0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t1)), [0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2)), [1, 0])
        # (-,-)
        self.assertEqual(order(CP(BW, t1), CP(BW, t2)), [0, 1])
        self.assertEqual(order(CP(BW, t1), CP(BW, t1)), [1, 0])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1)), [1, 0])
        # (-,+)
        self.assertEqual(order(CP(BW, t1), CP(FW, t2)), [0, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t1)), [0, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1)), [0, 1])
        # (+,-)
        self.assertEqual(order(CP(FW, t1), CP(BW, t2)), [1, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t1)), [1, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1)), [1, 0])

    def test_contour_ordering3(self):
        FW, BW = Branch.FORWARD, Branch.BACKWARD
        t1 = self.t_points[2]
        t2 = self.t_points[3]
        t3 = self.t_points[4]
        order = contour_ordering3

        # (+,+,+)
        self.assertEqual(order(CP(FW, t3), CP(FW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(FW, t3), CP(FW, t1), CP(FW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t3)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(FW, t3), CP(FW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(FW, t3), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t3)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t2), CP(FW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(FW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(FW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(FW, t1)), [0, 1, 2])
        # [-,-,-)
        self.assertEqual(order(CP(BW, t3), CP(BW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(BW, t3), CP(BW, t1), CP(BW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t3)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t3), CP(BW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(BW, t3), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t3)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(BW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(BW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(BW, t1)), [2, 1, 0])
        # [+,+,-)
        self.assertEqual(order(CP(FW, t3), CP(FW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t3), CP(FW, t1), CP(BW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t3)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(FW, t3), CP(BW, t2)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(FW, t3), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t3)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t2), CP(FW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(BW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t2)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(FW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(FW, t1), CP(BW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(FW, t1), CP(FW, t1), CP(BW, t1)), [2, 0, 1])
        # [-,-,+)
        self.assertEqual(order(CP(BW, t3), CP(BW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t3), CP(BW, t1), CP(FW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t3)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t3), CP(FW, t2)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t3), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t3)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(FW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t2)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(BW, t1), CP(FW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(BW, t1), CP(BW, t1), CP(FW, t1)), [1, 0, 2])
        # [-,+,+)
        self.assertEqual(order(CP(BW, t3), CP(FW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t3), CP(FW, t1), CP(FW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t3)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t3), CP(FW, t2)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(FW, t3), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t3)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(FW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t2)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(FW, t1)), [0, 1, 2])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(FW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(FW, t1)), [0, 1, 2])
        # [+,-,-)
        self.assertEqual(order(CP(FW, t3), CP(BW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t3), CP(BW, t1), CP(BW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t3)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t3), CP(BW, t2)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t3), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t3)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(BW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t2)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(BW, t1)), [2, 1, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(BW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(BW, t1)), [2, 1, 0])
        # [+,-,+)
        self.assertEqual(order(CP(FW, t3), CP(BW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t3), CP(BW, t1), CP(FW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t3)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t3), CP(FW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t3), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t3)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(FW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t2)), [1, 2, 0])
        self.assertEqual(order(CP(FW, t1), CP(BW, t2), CP(FW, t1)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t2), CP(BW, t1), CP(FW, t2)), [1, 0, 2])
        self.assertEqual(order(CP(FW, t1), CP(BW, t1), CP(FW, t1)), [1, 0, 2])
        # [-,+,-)
        self.assertEqual(order(CP(BW, t3), CP(FW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t3), CP(FW, t1), CP(BW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t3)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t3), CP(BW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t3), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t3)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(BW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t2)), [0, 2, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t2), CP(BW, t1)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t2), CP(FW, t1), CP(BW, t2)), [2, 0, 1])
        self.assertEqual(order(CP(BW, t1), CP(FW, t1), CP(BW, t1)), [2, 0, 1])

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

        g[CP(BW, t), CP(FW, t)] = 3.0
        self.assertEqual(g[CP(BW, t), CP(FW, t)], 3.0)


        g[BW, FW] = 2 * np.ones((self.n_t, self.n_t))
        assert_array_equal(g[BW, FW], 2*np.ones((self.n_t, self.n_t)))

        # Multiplication by a scalar
        g *= 3
        self.assertEqual(g[CP(BW, t), CP(FW, t)], 6.0)

        # Addition
        g += g
        self.assertEqual(g[CP(BW, t), CP(FW, t)], 12.0)
        self.assertEqual((g + g)[CP(BW, t), CP(FW, t)], 24.0)

        # Subtraction
        g -= 0.5 * g
        self.assertEqual(g[CP(BW, t), CP(FW, t)], 6.0)
        self.assertEqual((g - g)[CP(BW, t), CP(FW, t)], 0.0)

        # Unary minus
        self.assertEqual((-g)[CP(BW, t), CP(FW, t)], -6.0)
