#
# Tools for unit testing
#

from itertools import product
from numpy.testing import assert_array_almost_equal

from tddt.keldysh import Branch, KeldyshGF, KeldyshVertex3


def assert_keldysh_gf_almost_equal(g: KeldyshGF, g_ref: KeldyshGF, decimal=6):
    """
    Assert two KeldyshGF object having the same structure and being numerically
    close.
    """
    assert g.mesh == g_ref.mesh
    assert g.target_shape == g_ref.target_shape
    for b0, b1 in product(Branch, repeat=2):
        assert g[b0, b1].mesh == g_ref[b0, b1].mesh
        assert g[b0, b1].target_shape == g_ref[b0, b1].target_shape
        assert_array_almost_equal(g[b0, b1].data, g_ref[b0, b1].data, decimal)


def assert_keldysh_vertex3_almost_equal(Lambda: KeldyshVertex3,
                                        Lambda_ref: KeldyshVertex3,
                                        decimal=6):
    """
    Assert two KeldyshVertex3 object having the same structure and being
    numerically close.
    """
    assert Lambda.mesh == Lambda_ref.mesh
    assert Lambda.target_shape == Lambda_ref.target_shape
    for b0, b1, b2 in product(Branch, repeat=3):
        assert Lambda[b0, b1, b2].mesh == Lambda_ref[b0, b1, b2].mesh
        assert Lambda[b0, b1, b2].target_shape == \
            Lambda_ref[b0, b1, b2].target_shape
        assert_array_almost_equal(Lambda[b0, b1, b2].data,
                                  Lambda_ref[b0, b1, b2].data,
                                  decimal)
