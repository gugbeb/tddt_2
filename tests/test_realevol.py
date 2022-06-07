import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_array_almost_equal

import triqs.utility.mpi
from triqs.gf import MeshReTime, Gf

from realevol.tinterp import TInterp as ti
from realevol.operators_tinterp import *
from realevol.init_state import *
from realevol.realevol import compute_expectval

from tddt.keldysh import Branch, KeldyshGF
from tddt.realevol import (
    compute_keldysh_gf,
    compute_keldysh_gf_element,
    compute_keldysh_correlator_2t,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)

class test_realevol(unittest.TestCase):
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
        dt = ti(m_interp, np.array([0.1*(1-np.exp(-5*x)) for x in m_interp]))

        fops = set(product(cls.spin_names,[0,1]))

        # Initial Hamiltonian
        h0 = -mu*(n('up',0) + n('dn',0)) + U * n('up',0) * n('dn',0)
        h0 += eps*(n('up',1) + n('dn',1))
        h0 += sum(-t*(c_dag(sn,0) * c(sn,1) + c_dag(sn,1) * c(sn,0))
                  for sn in cls.spin_names)

        cls.init_state = make_equilibrium_init_state(h0,
                                                     fermion_indices = fops,
                                                     boson_indices = set(),
                                                     temperature = 0,
                                                     params = {})

        # Hamiltonian after quench
        cls.h = h0 + sum(dt*(c_dag(sn,0)*c(sn,1) + c_dag(sn,1)*c(sn,0))
                         for sn in cls.spin_names)

        cls.params = {}
        cls.params['verbosity'] = 2
        cls.params['lanczos_min_matrix_size'] = 10000

    def test_compute_keldysh_gf(self):
        gf_struct = [('up', [0,1]), ('dn', [0,1])]
        gf = compute_keldysh_gf(gf_struct,
                                self.init_state,
                                self.h,
                                self.t_mesh,
                                self.params)
        ref_keys = set([('up', 0, 0), ('up', 0, 1), ('up', 1, 0), ('up', 1, 1),
                        ('dn', 0, 0), ('dn', 0, 1), ('dn', 1, 0), ('dn', 1, 1)])
        self.assertEqual(gf.keys(), ref_keys)
        self.assertTrue(all(isinstance(g, KeldyshGF) for g in gf.values()))

        gf_el = compute_keldysh_gf_element(('dn', 0), ('dn', 1),
                                           self.init_state,
                                           self.h,
                                           self.t_mesh,
                                           self.params)
        for g, g_el in zip(gf[('dn', 0, 1)].data, gf_el.data):
            assert_array_almost_equal(g.mesh, g_el.mesh)
            assert_array_almost_equal(g.target_shape, g_el.target_shape)
            assert_array_almost_equal(g.data, g_el.data)

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
        for b0, b1 in product(Branch, Branch):
            assert_array_almost_equal(g[b0, b1].data, g_ref[b0, b1].data)

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

        N0_aver_N1_aver = Gf(mesh=N0N1.time_mesh,
                             target_shape=N0N1.target_shape)
        for t1, t2 in N0_aver_N1_aver.mesh:
            N0_aver_N1_aver[t1, t2] = N0_aver[t1] * N1_aver[t2]
        N1_aver_N0_aver = Gf(mesh=N0N1.time_mesh,
                             target_shape=N0N1.target_shape)
        for t2, t1 in N1_aver_N0_aver.mesh:
            N1_aver_N0_aver[t2, t1] = N0_aver[t1] * N1_aver[t2]
        N1_aver_N0_aver.data[:] = np.transpose(N1_aver_N0_aver.data)

        rho0rho1_ref = N0N1 - KeldyshGF(N0_aver_N1_aver, N1_aver_N0_aver)

        for b0, b1 in product(Branch, Branch):
            assert_array_almost_equal(rho0rho1[b0, b1].data,
                                      rho0rho1_ref[b0, b1].data)

    def test_compute_keldysh_vertex3(self):
        Lambda = compute_keldysh_vertex3(('up', 0),
                                         ('up', 0),
                                         n('dn', 1) + n('up', 1),
                                         self.init_state,
                                         self.h,
                                         self.t_mesh,
                                         self.params)

if __name__ == '__main__':
    unittest.main()
