#
# Tools for unit testing
#

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
