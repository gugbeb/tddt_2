import unittest
import numpy as np

from triqs.gf import MeshReTime

from realevol.tinterp import TInterp as ti
import realevol.operators_tinterp as op

from tddt.models import spin_names, SquarePlaquette


class TestSquarePlaquette(unittest.TestCase):
    """Objects describing specific physical systems"""

    gf_struct_ref = [('up', 4), ('dn', 4)]
    fops_ref = set([('up', 0), ('up', 1), ('up', 2), ('up', 3),
                    ('dn', 0), ('dn', 1), ('dn', 2), ('dn', 3)])

    def test_zero(self):
        model = SquarePlaquette(2)
        self.assertEqual(model.N, 2)
        self.assertEqual(len(model.r_mesh), 4)
        self.assertEqual(len(model.k_mesh), 4)
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
        model = SquarePlaquette(
            N=2,
            hopping=hopping,
            local_int=local_int,
            nonlocal_int=nonlocal_int,
            vector_potential=(0.5, -0.5)
        )
        self.assertEqual(model.N, 2)
        self.assertEqual(len(model.r_mesh), 4)
        self.assertEqual(len(model.k_mesh), 4)
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
             ti(t_mesh, [-0.5 * cos(2.1 * t) for t in t_mesh]))
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
        model = SquarePlaquette(
            N=2,
            hopping=hopping,
            local_int=local_int,
            nonlocal_int=nonlocal_int,
            vector_potential=A
        )
        self.assertEqual(model.N, 2)
        self.assertEqual(len(model.r_mesh), 4)
        self.assertEqual(len(model.k_mesh), 4)
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


if __name__ == '__main__':
    unittest.main()
