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

"""Utilities"""

from typing import Callable, Sequence
import numpy as np

"""Lists of subscripts for numpy.einsum()"""
subscripts = {'time': "abcdefgh",
              'nontime': "ABCDEFGH",
              'target': "ijklmnpqrstuvwxyz"}


def mapsum(f: Callable, s: Sequence):
    """
    A map-reduce style function that uses summation as a reduction operation,
    and computes the initial value from the first element of 's'.
    """
    it = iter(s)
    res = f(next(it))
    for x in it:
        res += f(x)
    return res


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
