from itertools import product, permutations
import numpy as np

# Inexplicably, the batch versions of `compute_*()` segfault in the cpp2py
# glue code (in `std::string to_string(PyObject * ob)`) if SciPy is not imported
# This can be an obscure memory corruption bug.
import scipy  # noqa: F401

from triqs.gf import Gf
from realevol import operators_texpr
from realevol import operators_tinterp
from realevol.realevol import (
    compute_expectval,
    compute_g_l,
    compute_g_g,
    compute_correlator_2t,
    compute_correlator_3t
)


from .keldysh import KeldyshGF, KeldyshVertex3


def _select_op_module(op):
    if isinstance(op, operators_texpr.Operator):
        return operators_texpr
    elif isinstance(op, operators_tinterp.Operator):
        return operators_tinterp
    else:
        raise TypeError("Unexpected operator type")


def compute_keldysh_gf(gf_struct, init_state, h, t_mesh, params):
    """Use realevol to compute a Keldysh Green's function"""
    g_l = compute_g_l(gf_struct, init_state, h, t_mesh, params)
    g_g = compute_g_g(gf_struct, init_state, h, t_mesh, params)
    gf = {}
    for bl, ind in gf_struct:
        for i, j in product(ind, ind):
            gf[(bl, i, j)] = KeldyshGF.from_g_l_g_g(g_l[bl][i, j],
                                                    g_g[bl][i, j])
    return gf


def compute_keldysh_gf_element(c_indices,
                               c_dag_indices,
                               init_state,
                               h,
                               t_mesh,
                               params):
    """
    Use realevol to compute a single matrix element of
    a Keldysh Green's function
    """
    op_module = _select_op_module(h)
    c_op = op_module.c(*c_indices)
    c_dag_op = op_module.c_dag(*c_dag_indices)
    g_g, g_l = compute_correlator_2t([(c_op, c_dag_op), (c_dag_op, c_op)],
                                     init_state,
                                     h,
                                     t_mesh,
                                     params)
    # Multiply by \pm i and swap time arguments of g_l
    g_g.data[:] *= -1j
    g_l.data[:] = 1j * np.transpose(g_l.data)
    return KeldyshGF.from_g_l_g_g(g_l, g_g)


def compute_keldysh_correlator_2t(A, B, init_state, h, t_mesh, params):
    """Use realevol to compute a 2-point correlator of operators A and B"""
    op_module = _select_op_module(A)
    assert op_module == _select_op_module(B)

    minus = (op_module.operator_stat(A) == "Fermion") and \
            (op_module.operator_stat(B) == "Fermion")

    verb = params.get('verbosity', 0) > 0

    if verb:
        print(f"Computing correlators <{A},{B}> and <{B},{A}>")
    AB_corr, BA_corr = compute_correlator_2t([(A, B), (B, A)],
                                             init_state,
                                             h,
                                             t_mesh,
                                             params)
    BA_corr.data[:] = (-1 if minus else 1) * np.transpose(BA_corr.data)
    return KeldyshGF.from_g_l_g_g(BA_corr, AB_corr)


def compute_keldysh_conn_correlator_2t(A, B, init_state, h, t_mesh, params):
    """
    Use realevol to compute a connected 2-point correlator of operators A and B,

        <T (A(t) - <A(t)>) (B(t') - <B(t')>)>
    """
    op_module = _select_op_module(A)
    assert op_module == _select_op_module(B)

    minus = (op_module.operator_stat(A) == "Fermion") and \
            (op_module.operator_stat(B) == "Fermion")

    verb = params.get('verbosity', 0) > 0

    if verb:
        print("Computing time-dependent expectation values of", A, "and", B)
    A_aver, B_aver = compute_expectval([A, B], init_state, h, t_mesh, params)

    if verb:
        print(f"Computing correlators <{A},{B}> and <{B},{A}>")
    AB_corr, BA_corr = compute_correlator_2t([(A, B), (B, A)],
                                             init_state,
                                             h,
                                             t_mesh,
                                             params)

    for (t1, t2) in AB_corr.mesh:
        AB_corr[t1, t2] -= A_aver[t1] * B_aver[t2]
    for (t2, t1) in BA_corr.mesh:
        BA_corr[t2, t1] -= A_aver[t1] * B_aver[t2]

    BA_corr.data[:] = (-1 if minus else 1) * np.transpose(BA_corr.data)
    return KeldyshGF.from_g_l_g_g(BA_corr, AB_corr)


def compute_keldysh_vertex3(c_indices,
                            c_dag_indices,
                            n_op,
                            init_state,
                            h,
                            t_mesh,
                            params):
    """Use realevol to compute a three-point vertex function"""

    op_module = _select_op_module(n_op)
    assert op_module.operator_stat(n_op) == "Boson"

    verb = params.get('verbosity', 0) > 0

    if verb:
        print("Computing time-dependent expectation value of", n_op)
    n_op_aver = compute_expectval(n_op, init_state, h, t_mesh, params)

    ops = (op_module.c(*c_indices), op_module.c_dag(*c_dag_indices), n_op)

    if verb:
        print(f"Computing correlator <{ops[0]},{ops[1]}>")
    c_c_dag_aver = compute_correlator_2t(ops[0],
                                         ops[1],
                                         init_state,
                                         h,
                                         t_mesh,
                                         params)
    if verb:
        print(f"Computing correlator <{ops[1]},{ops[0]}>")
    c_dag_c_aver = compute_correlator_2t(ops[1],
                                         ops[0],
                                         init_state,
                                         h,
                                         t_mesh,
                                         params)

    # Compute time pieces
    perms = list(permutations((0, 1, 2)))
    ops_perms = [(ops[p[0]], ops[p[1]], ops[p[2]]) for p in perms]

    if verb:
        print(f"Computing correlators of {ops_perms}")
    G_list = compute_correlator_3t(ops_perms, init_state, h, t_mesh, params)

    G = {}
    for p, G_p in zip(perms, G_list):
        # Are $c$ and $c^\dagger$ swapped by this permutation
        c_c_dag_swapped = p.index(1) < p.index(0)

        G[p] = Gf(mesh=G_p.mesh, target_shape=[])
        G_out = G[p]

        # Permute time arguments and subtract the disconnected part
        for t in G_out.mesh:
            t1, t2, t3 = t
            t_p1, t_p2, t_p3 = t[p[0]], t[p[1]], t[p[2]]
            G_out[t1, t2, t3] = G_p[t_p1, t_p2, t_p3]       \
                - n_op_aver[t3] * (c_dag_c_aver[t2, t1]
                                   if c_c_dag_swapped else
                                   c_c_dag_aver[t1, t2])
            G_out *= -1 * (-1 if c_c_dag_swapped else 1)

    return KeldyshVertex3.from_G_perm_pieces(G)
