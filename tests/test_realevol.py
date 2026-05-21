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
import pytest
from itertools import product
import numpy as np
from numpy.testing import assert_array_almost_equal

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, Gf

from realevol.texpr import TExpr as te
from realevol.tinterp import TInterp as ti
from realevol.operators_tinterp import c, c_dag, n
from realevol.init_state import make_equilibrium_init_state
from realevol.realevol import compute_expectval

from tddt.keldysh import Branch, KeldyshGF
from tddt.realevol import (
    is_zero,
    compute_keldysh_gf,
    compute_keldysh_gf_element,
    compute_keldysh_correlator_2t,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)
from tddt.testing import assert_keldysh_gf_almost_equal


class TestRealevol(unittest.TestCase):
    """Convenience wrapper functions around realevol"""

    @classmethod
    def setUpClass(cls):
        cls.spin_names = ('up', 'dn')
        cls.t_max = 5.0
        cls.n_t = 7
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)

        m_interp = MeshReTime(0, cls.t_max, 1001)

        # Model parameters
        U = 3.0
        mu = 0.5 * U
        eps = 0.2
        t = 0.3
        dt = ti(m_interp,
                np.array([0.1 * (1 - np.exp(-5 * x)) for x in m_interp]))

        fops = set(product(cls.spin_names, [0, 1]))

        # Initial Hamiltonian
        h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0)
        h0 += eps * (n('up', 1) + n('dn', 1))
        h0 += sum(-t * (c_dag(sn, 0) * c(sn, 1) + c_dag(sn, 1) * c(sn, 0))
                  for sn in cls.spin_names)

        cls.init_state = make_equilibrium_init_state(h0,
                                                     fermion_indices=fops,
                                                     boson_indices=set(),
                                                     temperature=0,
                                                     params={})

        # Hamiltonian after quench
        cls.h = h0 + \
            sum(dt * (c_dag(sn, 0) * c(sn, 1) + c_dag(sn, 1) * c(sn, 0))
                for sn in cls.spin_names)

        cls.params = {}
        cls.params['verbosity'] = 2
        cls.params['lanczos_min_matrix_size'] = 10000

    @pytest.mark.mpi
    def test_is_zero(self):
        self.assertTrue(is_zero(te(0)))
        self.assertTrue(is_zero(te("0", "0")))
        self.assertFalse(is_zero(te(1)))
        self.assertFalse(is_zero(te("t", "t")))
        self.assertTrue(is_zero(ti(self.t_mesh, np.zeros(self.n_t))))
        self.assertFalse(is_zero(ti(self.t_mesh, np.arange(self.n_t))))
        self.assertTrue(is_zero(0))
        self.assertTrue(is_zero(0.0))
        self.assertTrue(is_zero(0.0j))
        self.assertFalse(is_zero(1))
        self.assertFalse(is_zero(1.0j))
        self.assertFalse(is_zero(1.0j))

    @pytest.mark.mpi
    def test_compute_keldysh_gf(self):
        gf_struct = [('up', 2), ('dn', 2)]
        gf = compute_keldysh_gf(gf_struct,
                                self.init_state,
                                self.h,
                                self.t_mesh,
                                self.params)
        ref_blocks = set(['up', 'dn'])
        self.assertEqual(gf.keys(), ref_blocks)
        self.assertTrue(all(isinstance(g, KeldyshGF) for g in gf.values()))
        self.assertTrue(all(g.target_shape == (2, 2) for g in gf.values()))

        gf_el = compute_keldysh_gf_element(('dn', 0), ('dn', 1),
                                           self.init_state,
                                           self.h,
                                           self.t_mesh,
                                           self.params)
        # Iteration over Keldysh components
        for g, g_el in zip(gf['dn'].components.reshape(4),
                           gf_el.components.reshape(4)):
            self.assertEqual(g.mesh, g_el.mesh)
            assert_array_almost_equal(g[0, 1].data, g_el.data)

    @pytest.mark.mpi
    def test_compute_keldysh_correlator_2t(self):
        # Correlator <S_{z,0}(t) S_{z,0}(t')>
        Sz = (n('up', 0) - n('dn', 0)) / 2
        SzSz = compute_keldysh_correlator_2t(Sz, Sz,
                                             self.init_state,
                                             self.h,
                                             self.t_mesh,
                                             self.params)
        assert_array_almost_equal(
            SzSz[Branch.BACKWARD, Branch.FORWARD].data,
            np.transpose(SzSz[Branch.FORWARD, Branch.BACKWARD].data)
        )

        # GF of fermions
        g = compute_keldysh_correlator_2t(-1j * c('up', 0), c_dag('up', 1),
                                          self.init_state,
                                          self.h,
                                          self.t_mesh,
                                          self.params)

        g_ref = compute_keldysh_gf_element(('up', 0), ('up', 1),
                                           self.init_state,
                                           self.h,
                                           self.t_mesh,
                                           self.params)
        assert_keldysh_gf_almost_equal(g, g_ref)

    @pytest.mark.mpi
    def test_compute_keldysh_conn_correlator_2t(self):
        N0 = n('up', 0) + n('dn', 0)
        N1 = n('up', 1) + n('dn', 1)

        # Correlator <N_0(t) N_1(t')>
        N0N1 = compute_keldysh_correlator_2t(N0, N1,
                                             self.init_state,
                                             self.h,
                                             self.t_mesh,
                                             self.params)
        # Correlator <\rho_0(t) \rho_1(t')>
        rho0rho1 = compute_keldysh_conn_correlator_2t(N0, N1,
                                                      self.init_state,
                                                      self.h,
                                                      self.t_mesh,
                                                      self.params)

        # <N_0(t)>
        N0_aver = compute_expectval(N0, self.init_state,
                                    self.h,
                                    self.t_mesh,
                                    self.params)
        # <N_1(t)>
        N1_aver = compute_expectval(N1, self.init_state,
                                    self.h,
                                    self.t_mesh,
                                    self.params)

        N0_aver_N1_aver = Gf(mesh=N0N1.mesh,
                             target_shape=N0N1.target_shape)
        for t1, t2 in N0_aver_N1_aver.mesh:
            N0_aver_N1_aver[t1, t2] = N0_aver[t1] * N1_aver[t2]
        N1_aver_N0_aver = Gf(mesh=N0N1.mesh,
                             target_shape=N0N1.target_shape)
        for t2, t1 in N1_aver_N0_aver.mesh:
            N1_aver_N0_aver[t2, t1] = N0_aver[t1] * N1_aver[t2]
        N1_aver_N0_aver.data[:] = np.transpose(N1_aver_N0_aver.data)

        rho0rho1_ref = N0N1 - KeldyshGF.from_lesser_greater(N0_aver_N1_aver,
                                                            N1_aver_N0_aver)

        assert_keldysh_gf_almost_equal(rho0rho1, rho0rho1_ref)

    @pytest.mark.mpi
    def test_compute_keldysh_vertex3(self):
        compute_keldysh_vertex3(('up', 0),
                                ('up', 0),
                                n('dn', 1) + n('up', 1),
                                self.init_state,
                                self.h,
                                self.t_mesh,
                                self.params)


if __name__ == '__main__':
    unittest.main()
