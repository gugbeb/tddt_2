#
# Tools for unit testing
#

from itertools import product

from triqs.utility.comparison_tests import assert_gfs_are_close

from tddt.keldysh import Branch, KeldyshGF


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
