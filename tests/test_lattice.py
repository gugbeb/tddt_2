# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild
#
# tddt is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# tddt is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# tddt. If not, see <http://www.gnu.org/licenses/>.
#
# ##############################################################################

import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshReFreq, MeshCycLat, MeshBrZone, MeshProduct
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import Branch, KeldyshGF, Singular2PKeldyshGF
from tddt.lattice import local_part, SpacialArgs, lattice_fourier
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

    def test_lattice_fourier_no_spacial(self):
        t_mesh = MeshReTime(0.0, 10.0, 5)
        w_mesh = MeshReFreq((-2, 2), 11)

        # Gf without spacial meshes
        mesh = MeshProduct(t_mesh, t_mesh, w_mesh)
        g = KeldyshGF(mesh=mesh, target_shape=(2, 3))
        g_s2p = Singular2PKeldyshGF(mesh=mesh, target_shape=(2, 3))
        for br in product(Branch, repeat=2):
            g_br = g[br]
            g_br.data[:] = np.arange(g_br.data.size).reshape(g_br.data.shape)
        for br in Branch:
            g_br = g_s2p[br]
            g_br.data[:] = np.arange(g_br.data.size).reshape(g_br.data.shape)

        for apply_to in SpacialArgs:
            assert_keldysh_gf_almost_equal(
                lattice_fourier(g, apply_to=apply_to), g, 1e-10
            )
            assert_singular_2p_keldysh_gf_almost_equal(
                lattice_fourier(g_s2p, apply_to=apply_to), g_s2p, 1e-10
            )

    def test_lattice_fourier(self):
        n_k = (4, 5, 1)
        n_r = (6, 7, 1)
        n_t = 3
        n_w = 4

        t_mesh = MeshReTime(0.0, 10.0, n_t)
        w_mesh = MeshReFreq((-2, 2), n_w)

        bl = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # Square lattice
        k_mesh = MeshBrZone(BrillouinZone(bl), n_k)
        r_mesh = MeshCycLat(bl, n_r)

        mesh = MeshProduct(t_mesh, t_mesh, k_mesh, w_mesh, r_mesh)
        g = KeldyshGF(mesh=mesh, target_shape=(2, 1))

        def theta(t1, t2):
            return float(t1.value >= t2.value)

        def n(w):
            return 1 / (1 + np.exp(w.value))

        def eps(k):
            return np.cos(k[0]) + np.cos(k[1]) \
                - 0.25 * np.cos(k[0]) * np.cos(k[1])

        for br in product(Branch, repeat=2):
            start = br[0].value + br[1].value
            target = np.arange(start, start + 2).reshape(2, 1)
            for t1, t2, k, w, r in g.mesh:
                g[br][t1, t2, k, w, r] = -1j * target * (theta(t1, t2) - n(w)) \
                    * np.exp(-1j * eps(k) * (t1.value - t2.value)) \
                    * np.exp(-0.5 * (r[0] + 0.8 * r[1]) ** 2)

        target_tmp = np.zeros(target.shape, dtype=complex)

        # apply_to == SpacialArgs.BRZONE
        mesh1 = MeshProduct(t_mesh, t_mesh, MeshCycLat(bl, n_k), w_mesh, r_mesh)
        g1_ref = KeldyshGF(mesh=mesh1, target_shape=(2, 1))
        for br in product(Branch, repeat=2):
            start = br[0].value + br[1].value
            target = np.arange(start, start + 2).reshape(2, 1)
            for t1, t2, rp, w, r in g1_ref.mesh:
                target_tmp[...] = 0
                for k in k_mesh:
                    target_tmp[...] += g[br][t1, t2, k, w, r] \
                        * np.exp(-1j * (rp[0] * k[0] + rp[1] * k[1]))
                target_tmp /= np.prod(n_k)
                g1_ref[br][t1, t2, rp, w, r] = target_tmp
        assert_keldysh_gf_almost_equal(
            lattice_fourier(g, apply_to=SpacialArgs.BRZONE), g1_ref, 1e-10
        )

        # apply_to == SpacialArgs.LATTICE
        mesh2 = MeshProduct(t_mesh, t_mesh, k_mesh, w_mesh,
                            MeshBrZone(BrillouinZone(bl), n_r))
        g2_ref = KeldyshGF(mesh=mesh2, target_shape=(2, 1))
        for br in product(Branch, repeat=2):
            start = br[0].value + br[1].value
            target = np.arange(start, start + 2).reshape(2, 1)
            for t1, t2, k, w, kp in g2_ref.mesh:
                target_tmp[...] = 0
                for r in r_mesh:
                    target_tmp[...] += g[br][t1, t2, k, w, r] \
                        * np.exp(1j * (r[0] * kp[0] + r[1] * kp[1]))
                g2_ref[br][t1, t2, k, w, kp] = target_tmp
        assert_keldysh_gf_almost_equal(
            lattice_fourier(g, apply_to=SpacialArgs.LATTICE), g2_ref, 1e-10
        )

        # apply_to = SpacialArgs.BOTH
        mesh3 = MeshProduct(t_mesh, t_mesh,
                            MeshCycLat(bl, n_k),
                            w_mesh,
                            MeshBrZone(BrillouinZone(bl), n_r))
        g3_ref = KeldyshGF(mesh=mesh3, target_shape=(2, 1))
        for br in product(Branch, repeat=2):
            start = br[0].value + br[1].value
            target = np.arange(start, start + 2).reshape(2, 1)
            for t1, t2, rp, w, kp in g3_ref.mesh:
                target_tmp[...] = 0
                for r in r_mesh:
                    target_tmp[...] += g1_ref[br][t1, t2, rp, w, r] \
                        * np.exp(1j * (r[0] * kp[0] + r[1] * kp[1]))
                g3_ref[br][t1, t2, rp, w, kp] = target_tmp
        assert_keldysh_gf_almost_equal(
            lattice_fourier(g, apply_to=SpacialArgs.BOTH), g3_ref, 1e-10
        )


if __name__ == '__main__':
    unittest.main()
