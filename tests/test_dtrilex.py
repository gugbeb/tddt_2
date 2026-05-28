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

from triqs.gf import MeshReTime, MeshProduct, MeshBrZone
from triqs.lattice import BravaisLattice, BrillouinZone

from tddt.keldysh import Branch, KeldyshGF
from tddt.dtrilex import (polarization_2nd_order,
                          selfenergy_2nd_order,
                          selfenergy_2nd_order_hf)
from tddt.integration import GregoryIntegrator
from tddt.testing import assert_keldysh_gf_almost_equal

FW, BW = Branch.FORWARD, Branch.BACKWARD


class TestDiagrams(unittest.TestCase):
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
        cls.bz_mesh = MeshBrZone(BrillouinZone(bl), cls.n_k)

        # Use a lower order quadrature rule so that is is compatible with small
        # time meshes.
        KeldyshGF.integrator = GregoryIntegrator(2)

    @classmethod
    def tearDownClass(cls):
        KeldyshGF.integrator = GregoryIntegrator(5)

    def _make_test_keldysh_gf(self, mesh, x, target_shape=None):
        g = KeldyshGF(mesh=mesh, target_shape=target_shape)
        for n, (b0, b1) in enumerate(product(Branch, repeat=2)):
            g_comp = g[b0, b1]
            s = g_comp.data.size
            g_comp.data[:] = x * np.arange(s).reshape(g_comp.data.shape) + n
        return g

    def _make_test_keldysh_vertex3(self, mesh, x, target_shape=None):
        Lambda = KeldyshGF(mesh=mesh, target_shape=target_shape)
        for n, (b0, b1, b2) in enumerate(product(Branch, repeat=3)):
            l_comp = Lambda[b0, b1, b2]
            s = l_comp.data.size
            l_comp.data[:] = x * np.arange(s).reshape(l_comp.data.shape) + n
        return Lambda

    def test_polarization_2nd_order_scalar(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 2.0)

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * g, 0.6 * g]
        )

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1], self.t_mesh[1]))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n in product(*[self.t_ranges[1]] * 2,
                                            *[self.t_ranges[0]] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j] += -1j * sign * \
                    Lambda[b2, b3, b0].data[k, l, i] * g[b5, b2].data[n, k] * \
                    W[k] * W[l] * W[m] * W[n] * \
                    g[b3, b4].data[l, m] * Lambda[b4, b5, b1].data[m, n, j]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_matrix(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 2.0, (1, 1))

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * g, 0.6 * g]
        )

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1]),
                           target_shape=(3, 3))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, x, y, w1, w2, w3, w4 in product(
                    *[self.t_ranges[1]] * 2,
                    *[self.t_ranges[0]] * 4,
                    *[range(3)] * 2, *[range(1)] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, x, y] += -1j * sign *\
                    Lambda[b2, b3, b0].data[k, l, i, w1, w2, x] *\
                    g[b5, b2].data[n, k, w4, w1] *\
                    W[k] * W[l] * W[m] * W[n] *\
                    g[b3, b4].data[l, m, w2, w3] *\
                    Lambda[b4, b5, b1].data[m, n, j, w3, w4, y]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_scalar_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 2.0)

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * g, 0.6 * g]
        )

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1],
                                            self.bz_mesh))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K in product(*[self.t_ranges[1]] * 2,
                                               *[self.t_ranges[0]] * 4,
                                               range(self.n_k)):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, K] += -1j * sign * \
                    Lambda[b2, b3, b0].data[k, l, i] * \
                    g[b5, b2].data[n, k, K] * \
                    W[k] * W[l] * W[m] * W[n] * \
                    g[b3, b4].data[l, m, K] * \
                    Lambda[b4, b5, b1].data[m, n, j]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_polarization_2nd_order_matrix_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 2.0, (1, 1))

        W = GregoryIntegrator(2).weights_conv(self.t_mesh[0])

        pi = polarization_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * g, 0.6 * g]
        )

        pi_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[1],
                                            self.t_mesh[1],
                                            self.bz_mesh),
                           target_shape=(3, 3))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K, x, y, w1, w2, w3, w4 in product(
                    *[self.t_ranges[1]] * 2,
                    *[self.t_ranges[0]] * 4,
                    range(self.n_k),
                    *[range(3)] * 2, *[range(1)] * 4):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                pi_ref[b0, b1].data[i, j, x, y, K] += -1j * sign *\
                    Lambda[b2, b3, b0].data[k, l, i, w1, w2, x] *\
                    g[b5, b2].data[n, k, K, w4, w1] *\
                    W[k] * W[l] * W[m] * W[n] *\
                    g[b3, b4].data[l, m, K, w2, w3] *\
                    Lambda[b4, b5, b1].data[m, n, j, w3, w4, y]

        assert_keldysh_gf_almost_equal(pi, pi_ref)

    def test_selfenergy_2nd_order_scalar(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 2.0)
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0)

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0], self.t_mesh[0]))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n in product(*[self.t_ranges[0]] * 4,
                                            *[self.t_ranges[1]] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j] += 1j * sign * \
                    Lambda[b0, b2, b4].data[i, k, m] * g[b2, b3].data[k, l] * \
                    W0[k] * W0[l] * W1[m] * W1[n] * \
                    w[b4, b5].data[m, n] * Lambda[b3, b1, b5].data[l, j, n]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-3)

    def test_selfenergy_2nd_order_matrix(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (1, 1))
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0, (3, 3))

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0]),
                              target_shape=(1, 1))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, x, y, w1, w2, w3, w4 in \
                product(*[self.t_ranges[0]] * 4,
                        *[self.t_ranges[1]] * 2,
                        *[range(1)] * 4, *[range(3)] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, x, y] += 1j * sign * \
                    Lambda[b0, b2, b4].data[i, k, m, x, w2, w4] * \
                    g[b2, b3].data[k, l, w1, w2] * \
                    W0[k] * W0[l] * W1[m] * W1[n] * \
                    w[b4, b5].data[m, n, w3, w4] * \
                    Lambda[b3, b1, b5].data[l, j, n, w2, y, w4]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-4)

    def test_selfenergy_2nd_order_scalar_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 2.0)
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1], self.bz_mesh)
        w = self._make_test_keldysh_gf(w_mesh, 3.0)

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0],
                                               self.bz_mesh))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K in product(*[self.t_ranges[0]] * 4,
                                               *[self.t_ranges[1]] * 2,
                                               range(self.n_k)):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, K] += 1j * sign * \
                    Lambda[b0, b2, b4].data[i, k, m] * \
                    g[b2, b3].data[k, l, K] * \
                    W0[k] * W0[l] * W1[m] * W1[n] * \
                    w[b4, b5].data[m, n, K] * \
                    Lambda[b3, b1, b5].data[l, j, n]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref)

    def test_selfenergy_2nd_order_matrix_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 2.0, (1, 1))
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1], self.bz_mesh)
        w = self._make_test_keldysh_gf(w_mesh, 3.0, (3, 3))

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0],
                                               self.bz_mesh),
                              target_shape=(1, 1))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K, x, y, w1, w2, w3, w4 in \
                product(*[self.t_ranges[0]] * 4,
                        *[self.t_ranges[1]] * 2,
                        range(self.n_k),
                        *[range(1)] * 4, *[range(3)] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, K, x, y] += 1j * sign * \
                    Lambda[b0, b2, b4].data[i, k, m, x, w2, w4] * \
                    g[b2, b3].data[k, l, K, w1, w2] * \
                    W0[k] * W0[l] * W1[m] * W1[n] * \
                    w[b4, b5].data[m, n, K, w3, w4] * \
                    Lambda[b3, b1, b5].data[l, j, n, w2, y, w4]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-2)

    def test_selfenergy_2nd_order_hf_scalar(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 2.0)
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0)

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order_hf(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0], self.t_mesh[0]))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n in product(*[self.t_ranges[0]] * 4,
                                            *[self.t_ranges[1]] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j] += -1j * sign * \
                    Lambda[b0, b1, b4].data[i, j, m] * \
                    W1[m] * w[b4, b5].data[m, n] * W1[n] * \
                    Lambda[b2, b3, b5].data[k, l, n] * W0[k] * W0[l] * \
                    g[b3, b2].data[l, k]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref)

    def test_selfenergy_2nd_order_hf_matrix(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0])
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (1, 1))
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0, (3, 3))

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order_hf(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0]),
                              target_shape=(1, 1))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, x, y, w1, w2, w3, w4 in \
                product(*[self.t_ranges[0]] * 4,
                        *[self.t_ranges[1]] * 2,
                        *[range(1)] * 4, *[range(3)] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, x, y] += -1j * sign * \
                    Lambda[b0, b1, b4].data[i, j, m, x, y, w3] * \
                    W1[m] * w[b4, b5].data[m, n, w3, w4] * W1[n] * \
                    Lambda[b2, b3, b5].data[k, l, n, w1, w2, w4] * \
                    W0[k] * W0[l] * g[b3, b2].data[l, k, w2, w1]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-4)

    def test_selfenergy_2nd_order_hf_scalar_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0)
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 2.0)
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0)

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order_hf(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0],
                                               self.bz_mesh))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K in product(*[self.t_ranges[0]] * 4,
                                               *[self.t_ranges[1]] * 2,
                                               range(self.n_k)):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, K] += -1j * sign * \
                    Lambda[b0, b1, b4].data[i, j, m] * \
                    W1[m] * w[b4, b5].data[m, n] * W1[n] * \
                    Lambda[b2, b3, b5].data[k, l, n] * W0[k] * W0[l] * \
                    g[b3, b2].data[l, k, K]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-4)

    def test_selfenergy_2nd_order_hf_matrix_bz(self):
        Lambda_mesh = MeshProduct(*[self.t_mesh[i] for i in (0, 0, 1)])
        Lambda = self._make_test_keldysh_vertex3(Lambda_mesh, 1.0, (1, 1, 3))
        g_mesh = MeshProduct(self.t_mesh[0], self.t_mesh[0], self.bz_mesh)
        g = self._make_test_keldysh_gf(g_mesh, 1.0, (1, 1))
        w_mesh = MeshProduct(self.t_mesh[1], self.t_mesh[1])
        w = self._make_test_keldysh_gf(w_mesh, 3.0, (3, 3))

        W0 = GregoryIntegrator(2).weights_conv(self.t_mesh[0])
        W1 = GregoryIntegrator(2).weights_conv(self.t_mesh[1])

        sigma = selfenergy_2nd_order_hf(
            Lambda, [0.3 * g, 0.7 * g], [0.4 * w, 0.6 * w]
        )

        sigma_ref = KeldyshGF(mesh=MeshProduct(self.t_mesh[0],
                                               self.t_mesh[0],
                                               self.bz_mesh),
                              target_shape=(1, 1))
        for b0, b1, b2, b3, b4, b5 in product(Branch, repeat=6):
            for i, j, k, l, m, n, K, x, y, w1, w2, w3, w4 in \
                product(*[self.t_ranges[0]] * 4,
                        *[self.t_ranges[1]] * 2,
                        range(self.n_k),
                        *[range(1)] * 4, *[range(3)] * 2):
                sign = (-1) ** (b2, b3, b4, b5).count(BW)
                sigma_ref[b0, b1].data[i, j, K, x, y] += -1j * sign * \
                    Lambda[b0, b1, b4].data[i, j, m, x, y, w3] * \
                    W1[m] * w[b4, b5].data[m, n, w3, w4] * W1[n] * \
                    Lambda[b2, b3, b5].data[k, l, n, w1, w2, w4] * \
                    W0[k] * W0[l] * g[b3, b2].data[l, k, K, w2, w1]

        assert_keldysh_gf_almost_equal(sigma, sigma_ref, precision=1e-4)


if __name__ == '__main__':
    unittest.main()
