from itertools import product
import numpy as np

# Inexplicably, the batch versions of `compute_*()` segfault in the cpp2py
# glue code (in `std::string to_string(PyObject * ob)`) if SciPy is not imported.
# This can be an obscure memory corruption bug.
import scipy

import triqs.utility.mpi
from realevol import operators_texpr
from realevol import operators_tinterp
from realevol.realevol import (
    compute_expectval,
    compute_g_l,
    compute_g_g,
    compute_correlator_2t,
    compute_correlator_3t
)

from .keldysh import KeldyshGF

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
            gf[(bl, i, j)] = KeldyshGF(g_l[bl][i, j], g_g[bl][i, j])
    return gf

def compute_keldysh_gf_element(i, j, init_state, h, t_mesh, params):
    """Use realevol to compute a single matrix element of a Keldysh Green's function"""
    op_module = _select_op_module(h)
    c_op = op_module.c(*i)
    c_dag_op = op_module.c_dag(*j)
    g_g, g_l = compute_correlator_2t([(c_op, c_dag_op), (c_dag_op, c_op)],
                                     init_state,
                                     h,
                                     t_mesh,
                                     params)
    # Multiply by \pm i and swap time arguments of g_l
    g_g.data[:] *= -1j
    g_l.data[:] = 1j * np.transpose(g_l.data)
    return KeldyshGF(g_l, g_g)
