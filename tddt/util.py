#
# Utilities
#

import numpy as np

"""Lists of subscripts for numpy.einsum()"""
subscripts = {'time': "abcdefgh",
              'nontime': "ABCDEFGH",
              'target': "ijklmnpqrstuvwxyz"}


def make_conv_res_nontime_mesh(a_nt_mesh_comps, b_nt_mesh_comps):
    """
    If the non-time components of the meshes agree, return either of them.
    Otherwise, return a concatenation of the component lists.
    """
    return a_nt_mesh_comps if (a_nt_mesh_comps == b_nt_mesh_comps) else \
        a_nt_mesh_comps + b_nt_mesh_comps


def make_conv_nontime_einsum_subscripts(a_nt_mesh_comps, b_nt_mesh_comps):
    """
    Return einsum() subscript strings corresponding to the non-time components
    of meshes to be used in convolution routines.
    """
    nts = subscripts['nontime']
    if a_nt_mesh_comps == b_nt_mesh_comps:
        # If the non-time components of the meshes agree, then we
        # use the same non-time mesh for the result
        ss = nts[:len(a_nt_mesh_comps)]
        return ss, ss, ss
    else:
        # Otherwise the result is defined on a direct product of the meshes.
        ss = nts[:len(a_nt_mesh_comps)]
        subs_a_nt = ss
        subs_res_nt = ss
        ss = nts[len(a_nt_mesh_comps):
                 len(a_nt_mesh_comps) + len(b_nt_mesh_comps)]
        subs_b_nt = ss
        subs_res_nt += ss
        return subs_a_nt, subs_b_nt, subs_res_nt


def fermi(x):
    """
    Fermi step function 1 / (1 + exp(x)) evaluated so that overflows are avoided
    for both positive and negative 'x'.
    """
    if x < 0:
        return 1.0 / (1.0 + np.exp(x))
    else:
        ex = np.exp(-x)
        return ex / (1.0 + ex)
