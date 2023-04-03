#
# Tools for unit testing
#

from itertools import product
from numpy.testing import assert_allclose

from tddt.keldysh import Branch, KeldyshGF


def assert_keldysh_gf_almost_equal(g: KeldyshGF, g_ref: KeldyshGF, **kwargs):
    """
    Assert two KeldyshGF object having the same structure and being numerically
    close.
    """
    assert g.mesh == g_ref.mesh
    assert g.target_subshapes == g_ref.target_subshapes
    for br in product(Branch, repeat=g.n_args):
        assert g[br].mesh == g_ref[br].mesh
        assert g[br].target_shape == g_ref[br].target_shape
        assert_allclose(g[br].data, g_ref[br].data, **kwargs)
