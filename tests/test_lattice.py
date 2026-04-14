import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshCycLat, MeshBrZone, MeshProduct
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import Branch, KeldyshGF, Singular2PKeldyshGF
from tddt.lattice import local_part
from tddt.testing import (assert_keldysh_gf_almost_equal,
                          assert_singular_2p_keldysh_gf_almost_equal)


FW, BW = Branch.FORWARD, Branch.BACKWARD


class TestLattice(unittest.TestCase):
    """Functions and types related to lattice and Brillouin zone"""

    def test_local_part(self):
        t_mesh = MeshReTime(0.0, 10.0, 11)

        bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice
        n_k1 = 5
        n_k2 = 10
        k_mesh1 = MeshBrZone(BrillouinZone(bl), n_k1)
        k_mesh2 = MeshBrZone(BrillouinZone(bl), n_k2)
        n_r1 = 7
        n_r2 = 9
        r_mesh1 = MeshCycLat(bl, n_r1)
        r_mesh2 = MeshCycLat(bl, n_r2)

        # KeldyshGF
        mesh = MeshProduct(t_mesh, t_mesh, k_mesh1, r_mesh1, k_mesh2, r_mesh2)
        g = KeldyshGF(mesh=mesh, target_shape=(2, 2))
        data_shape = g[FW, FW].data.shape
        data_size = g[FW, FW].data.size
        for i, br in enumerate(product(Branch, repeat=2)):
            g[br].data[:] = np.arange(i, data_size + i).reshape(data_shape)

        mesh_ref = MeshProduct(t_mesh, t_mesh, r_mesh1, r_mesh2)
        g_loc_ref = KeldyshGF(mesh=mesh_ref, target_shape=(2, 2))
        for br1, br2 in product(Branch, repeat=2):
            for r1, r2 in product(range(n_r1), range(n_r2)):
                g_loc_ref[br1, br2].data[:, :, r1, r2, :, :] = sum(
                    g[br1, br2].data[:, :, k1, r1, k2, r2, :, :]
                    for k1, k2 in product(range(n_k1), range(n_k2))
                ) / (n_k1 * n_k2)
        assert_keldysh_gf_almost_equal(local_part(g), g_loc_ref, 1e-10)

        # Singular2PKeldyshGF
        mesh = MeshProduct(t_mesh, k_mesh1, r_mesh1, k_mesh2, r_mesh2)
        g = Singular2PKeldyshGF(mesh=mesh, target_shape=(2, 2))
        data_shape = g[FW].data.shape
        data_size = g[FW].data.size
        for i, br in enumerate(Branch):
            g[br].data[:] = np.arange(i, data_size + i).reshape(data_shape)

        mesh_ref = MeshProduct(t_mesh, r_mesh1, r_mesh2)
        g_loc_ref = Singular2PKeldyshGF(mesh=mesh_ref, target_shape=(2, 2))
        for br in Branch:
            for r1, r2 in product(range(n_r1), range(n_r2)):
                g_loc_ref[br].data[:, r1, r2, :, :] = sum(
                    g[br].data[:, k1, r1, k2, r2, :, :]
                    for k1, k2 in product(range(n_k1), range(n_k2))
                ) / (n_k1 * n_k2)
        assert_singular_2p_keldysh_gf_almost_equal(local_part(g),
                                                   g_loc_ref,
                                                   1e-10)


if __name__ == '__main__':
    unittest.main()
