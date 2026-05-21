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

from triqs.gf import MeshReTime, MeshBrZone, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone

from realevol.tinterp import TInterp as ti
import realevol.operators_tinterp as op

from tddt.keldysh import Branch, ContourPoint, KeldyshGF
from tddt.models import (spin_names,
                         SingleFermion,
                         FermionBand,
                         FermionFlatBand,
                         FiniteCluster)
from tddt.testing import assert_keldysh_gf_almost_equal


class TestFermion(unittest.TestCase):
    "Simple fermionic systems"

    t_mesh = MeshReTime(0, 20.0, 21)

    def test_SingleFermion(self):
        eps = 1.0
        n = 0.4
        T = 3.0

        sf = SingleFermion(eps)
        g_n = sf.gf(self.t_mesh, n=n)
        g_T = sf.gf(self.t_mesh, T=T)

        for br1, br2 in product(Branch, repeat=2):
            for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
                z1 = ContourPoint(br1, t1)
                z2 = ContourPoint(br2, t2)

                ex = np.exp(-1j * eps * (t1 - t2))
                if z1 < z2:
                    ref_n = -1j * (-n) * ex
                    ref_T = -1j * (-1 / (1 + np.exp(eps / T))) * ex
                else:
                    ref_n = -1j * (1.0 - n) * ex
                    ref_T = -1j * (1.0 - 1 / (1 + np.exp(eps / T))) * ex

                self.assertAlmostEqual(g_n[z1, z2], ref_n)
                self.assertAlmostEqual(g_T[z1, z2], ref_T)

    def test_FermionBand(self):
        bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice
        n_k = 10
        bz_mesh = MeshBrZone(BrillouinZone(bl), n_k)

        def eps_k(k):
            return np.cos(k[0])

        def n_k(k):
            return k[0] / (2 * np.pi)
        T = 3.0

        fb = FermionBand(bz_mesh, eps_k)
        g_n = fb.gf(self.t_mesh, n_k=n_k)
        g_T = fb.gf(self.t_mesh, T=T)

        for br1, br2 in product(Branch, repeat=2):
            for t1, t2, k in MeshProduct(self.t_mesh, self.t_mesh, bz_mesh):
                z1 = ContourPoint(br1, t1)
                z2 = ContourPoint(br2, t2)

                ex = np.exp(-1j * eps_k(k) * (t1 - t2))
                if z1 < z2:
                    ref_n = -1j * (-n_k(k)) * ex
                    ref_T = -1j * (-1 / (1 + np.exp(eps_k(k) / T))) * ex
                else:
                    ref_n = -1j * (1.0 - n_k(k)) * ex
                    ref_T = -1j * (1.0 - 1 / (1 + np.exp(eps_k(k) / T))) * ex

                self.assertAlmostEqual(g_n[z1, z2][k], ref_n)
                self.assertAlmostEqual(g_T[z1, z2][k], ref_T)

    def test_FermionFlatBand(self):
        t_points = list(self.t_mesh)
        t0, t1, t2 = t_points[0], t_points[10], t_points[20]

        # Reference values are produced by a Mathematica notebook

        ffb = FermionFlatBand(2, -3)

        g1_T0 = ffb.gf(self.t_mesh, T=0.0)
        self.assertAlmostEqual(g1_T0.greater()[t0, t0], 0.0)
        self.assertAlmostEqual(g1_T0.lesser()[t0, t0], 1j)
        self.assertAlmostEqual(g1_T0.greater()[t1, t0], 0.0)
        self.assertAlmostEqual(g1_T0.lesser()[t1, t0], 0.0451009 + 0.00704116j)
        self.assertAlmostEqual(g1_T0.greater()[t2, t0], 0.0)
        self.assertAlmostEqual(g1_T0.lesser()[t2, t0], 0.00567796 - 0.0177414j)
        g1 = ffb.gf(self.t_mesh, T=2.0)
        self.assertAlmostEqual(g1.greater()[t0, t0], -0.1975936j)
        self.assertAlmostEqual(g1.lesser()[t0, t0], 0.80240638j)
        self.assertAlmostEqual(g1.greater()[t1, t0], -0.00988 - 0.00430501j)
        self.assertAlmostEqual(g1.lesser()[t1, t0], 0.0352209 + 0.00273615j)
        self.assertAlmostEqual(g1.greater()[t2, t0], 0.00118654 + 0.00477684j)
        self.assertAlmostEqual(g1.lesser()[t2, t0], 0.0068645 - 0.0129646j)

        ffb = FermionFlatBand(2, -1)

        g2_T0 = ffb.gf(self.t_mesh, T=0.0)
        self.assertAlmostEqual(g2_T0.greater()[t0, t0], -0.25j)
        self.assertAlmostEqual(g2_T0.lesser()[t0, t0], 0.75j)
        self.assertAlmostEqual(g2_T0.greater()[t1, t0], -0.0459768 + 0.0136005j)
        self.assertAlmostEqual(g2_T0.lesser()[t1, t0], -0.0211437 - 0.0247008j)
        self.assertAlmostEqual(g2_T0.greater()[t2, t0], -0.007399 - 0.0114118j)
        self.assertAlmostEqual(g2_T0.lesser()[t2, t0], -0.0244052 - 0.00381013j)
        g2 = ffb.gf(self.t_mesh, T=2.0)
        self.assertAlmostEqual(g2.greater()[t0, t0], - 0.3863319j)
        self.assertAlmostEqual(g2.lesser()[t0, t0], + 0.6136681j)
        self.assertAlmostEqual(g2.greater()[t1, t0], -0.0134187 + 0.0132434j)
        self.assertAlmostEqual(g2.lesser()[t1, t0], 0.0114144 - 0.0250579j)
        self.assertAlmostEqual(g2.greater()[t2, t0], 0.00529361 - 0.00648333j)
        self.assertAlmostEqual(g2.lesser()[t2, t0], -0.0117126 + 0.00111836j)

        ffb = FermionFlatBand(2, 3)

        g3_T0 = ffb.gf(self.t_mesh, T=0.0)
        self.assertAlmostEqual(g3_T0.greater()[t0, t0], -1j)
        self.assertAlmostEqual(g3_T0.lesser()[t0, t0], 0)
        self.assertAlmostEqual(g3_T0.greater()[t1, t0], 0.0451009 - 0.00704116j)
        self.assertAlmostEqual(g3_T0.lesser()[t1, t0], 0)
        self.assertAlmostEqual(g3_T0.greater()[t2, t0], 0.00567796 + 0.0177414j)
        self.assertAlmostEqual(g3_T0.lesser()[t2, t0], 0)
        g3 = ffb.gf(self.t_mesh, T=2.0)
        self.assertAlmostEqual(g3.greater()[t0, t0], -0.8024064j)
        self.assertAlmostEqual(g3.lesser()[t0, t0], +0.1975936j)
        self.assertAlmostEqual(g3.greater()[t1, t0], 0.0352209 - 0.00273615j)
        self.assertAlmostEqual(g3.lesser()[t1, t0], -0.00988 + 0.00430501j)
        self.assertAlmostEqual(g3.greater()[t2, t0], 0.0068645 + 0.0129646j)
        self.assertAlmostEqual(g3.lesser()[t2, t0], 0.00118654 - 0.00477684j)


class TestFiniteCluster(unittest.TestCase):
    "A finite cluster"

    gf_struct_ref = [('up', 4), ('dn', 4)]
    fops_ref = set([('up', 0), ('up', 1), ('up', 2), ('up', 3),
                    ('dn', 0), ('dn', 1), ('dn', 2), ('dn', 3)])

    def test_zero(self):
        model = FiniteCluster([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
        self.assertEqual(model.N, 4)
        self.assertEqual(model.gf_struct, self.gf_struct_ref)
        self.assertEqual(model.fops, self.fops_ref)
        self.assertTrue(model.hamiltonian.is_zero())

    def test_static(self):
        hopping = [[1.0, 0.0, 0.0, 5j],
                   [0.0, 0.0, 0.5, 0.0],
                   [0.0, 0.5, 0.0, 0.0],
                   [-5j, 0.0, 0.0, 1.5]]
        local_int = [2.0, 0, 0, 3.0]
        nonlocal_int = np.zeros((4, 4))
        nonlocal_int[1, 2] = nonlocal_int[2, 1] = 7
        model = FiniteCluster(
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
            hopping=hopping,
            local_int=local_int,
            nonlocal_int=nonlocal_int,
            vector_potential=(0.5, -0.5, 0)
        )
        self.assertEqual(model.N, 4)
        self.assertEqual(model.gf_struct, self.gf_struct_ref)
        self.assertEqual(model.fops, self.fops_ref)

        h_ref = sum(1.0 * op.n(sn, 0) + 1.5 * op.n(sn, 3)
                    + 0.5 * op.c_dag(sn, 1) * op.c(sn, 2) * np.exp(1j)
                    + 0.5 * op.c_dag(sn, 2) * op.c(sn, 1) * np.exp(-1j)
                    + 5j * op.c_dag(sn, 0) * op.c(sn, 3)
                    - 5j * op.c_dag(sn, 3) * op.c(sn, 0)
                    for sn in spin_names)
        h_ref += 2.0 * op.n('up', 0) * op.n('dn', 0)
        h_ref += 3.0 * op.n('up', 3) * op.n('dn', 3)
        h_ref += 14 * (op.n('up', 1) + op.n('dn', 1)) \
                    * (op.n('up', 2) + op.n('dn', 2))

        self.assertTrue((model.hamiltonian - h_ref).is_zero())

    def test_dynamic(self):
        from cmath import exp, cos

        t_mesh = MeshReTime(0, 10.0, 1001)
        eps = ti(t_mesh, [t.value ** 2 for t in t_mesh])
        U = ti(t_mesh, [t.value ** 0.5 for t in t_mesh])
        A = (ti(t_mesh, [0.5 * cos(2.1 * t) for t in t_mesh]),
             ti(t_mesh, [-0.5 * cos(2.1 * t) for t in t_mesh]),
             0)
        # Peierls factors
        p12 = ti(t_mesh, [exp(1j * cos(2.1 * t)) for t in t_mesh])
        p21 = ti(t_mesh, [exp(-1j * cos(2.1 * t)) for t in t_mesh])

        hopping = [[eps, 0.0, 0.0, 5j],
                   [0.0, 0.0, 0.5, 0.0],
                   [0.0, 0.5, 0.0, 0.0],
                   [-5j, 0.0, 0.0, 1.5]]
        local_int = [U, 0, 0, 3.0]
        nonlocal_int = np.zeros((4, 4))
        nonlocal_int[1, 2] = nonlocal_int[2, 1] = 7
        model = FiniteCluster(
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
            hopping=hopping,
            local_int=local_int,
            nonlocal_int=nonlocal_int,
            vector_potential=A
        )
        self.assertEqual(model.N, 4)
        self.assertEqual(model.gf_struct, self.gf_struct_ref)
        self.assertEqual(model.fops, self.fops_ref)

        h_ref = sum(eps * op.n(sn, 0) + 1.5 * op.n(sn, 3)
                    + 0.5 * op.c_dag(sn, 1) * op.c(sn, 2) * p12
                    + 0.5 * op.c_dag(sn, 2) * op.c(sn, 1) * p21
                    + 5j * op.c_dag(sn, 0) * op.c(sn, 3)
                    - 5j * op.c_dag(sn, 3) * op.c(sn, 0)
                    for sn in spin_names)
        h_ref += U * op.n('up', 0) * op.n('dn', 0)
        h_ref += 3.0 * op.n('up', 3) * op.n('dn', 3)
        h_ref += 14 * (op.n('up', 1) + op.n('dn', 1)) \
                    * (op.n('up', 2) + op.n('dn', 2))

        self.assertTrue((model.hamiltonian - h_ref).is_zero())

    def test_hybridization(self):
        from cmath import cos, exp

        t_mesh = MeshReTime(0, 10.0, 101)
        tt_mesh = MeshProduct(t_mesh, t_mesh)
        ed = 2.0
        eps = [-1.0, 0.0, 1.0]
        V = [0.3, 0.4, 0.5]
        hopping = np.diag([ed, *eps]).astype(object)
        for bs, v in enumerate(V):
            v_t = ti(t_mesh, [v * cos(2 * t) for t in t_mesh])
            hopping[0, bs + 1] = hopping[bs + 1, 0] = v_t

        local_int = [2.0, 0, 0, 0]
        A = (0.1, 0.2, 0)
        T = 0.1

        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        model = FiniteCluster(coords,
                              hopping=hopping,
                              local_int=local_int,
                              vector_potential=A)
        Delta = model.hybridization(t_mesh, [0], [1, 2, 3], T=T)

        Delta_ref_g = Gf(mesh=tt_mesh, target_shape=(2, 1, 2, 1))
        Delta_ref_l = Gf(mesh=tt_mesh, target_shape=(2, 1, 2, 1))
        for e, v, c in zip(eps, V, coords[1:]):
            occ = 1 / (1 + exp(e / T))
            for time1, time2 in tt_mesh:
                ex = exp(-1j * e * (time1 - time2))
                v_t1 = v * cos(2 * time1)
                v_t2 = v * cos(2 * time2)
                val_g = -1j * (1.0 - occ) * ex * v_t1 * v_t2
                val_l = -1j * (-occ) * ex * v_t1 * v_t2
                for spin in range(2):
                    Delta_ref_g[time1, time2][spin, 0, spin, 0] += val_g
                    Delta_ref_l[time1, time2][spin, 0, spin, 0] += val_l
        Delta_ref = KeldyshGF.from_lesser_greater(Delta_ref_l, Delta_ref_g)

        assert_keldysh_gf_almost_equal(Delta, Delta_ref, precision=1e-9)


if __name__ == '__main__':
    unittest.main()
