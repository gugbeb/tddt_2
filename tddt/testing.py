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

"""Tools for unit testing"""

from itertools import product

from triqs.utility.comparison_tests import assert_gfs_are_close

from tddt.keldysh import Branch, KeldyshGF, Singular2PKeldyshGF


def assert_keldysh_gf_almost_equal(g: KeldyshGF,
                                   g_ref: KeldyshGF,
                                   precision: int = 1e-6,
                                   *,
                                   err_msg: str = ''):
    """
    Assert two KeldyshGF object having the same structure and being numerically
    close.
    """
    assert g.mesh == g_ref.mesh, err_msg
    assert g.arg_index_shapes == g_ref.arg_index_shapes, err_msg
    try:
        for br in product(Branch, repeat=g.n_args):
            assert_gfs_are_close(g[br], g_ref[br], precision)
    except AssertionError as e:
        if err_msg:
            raise AssertionError(err_msg)
        else:
            raise e


def assert_singular_2p_keldysh_gf_almost_equal(sg: Singular2PKeldyshGF,
                                               sg_ref: Singular2PKeldyshGF,
                                               precision: int = 1e-6,
                                               *,
                                               err_msg: str = ''):
    """
    Assert two Singular2PKeldyshGF object having the same structure and being
    numerically close.
    """
    assert sg.mesh == sg_ref.mesh, err_msg
    assert sg.arg_index_shapes == sg_ref.arg_index_shapes, err_msg
    try:
        for br in Branch:
            assert_gfs_are_close(sg[br], sg_ref[br], precision)
    except AssertionError as e:
        if err_msg:
            raise AssertionError(err_msg)
        else:
            raise e
