#
# Auxiliary routines for Green's functions of real time
#

import numpy as np

from triqs.gf import Gf, MeshReTime, MeshProduct


def conj(g: Gf, *, n_left_indices=None) -> Gf:
    r"""
    Given a 2-point real time Green's function G_{a, b}(t, t'), returns its
    Hermitian conjugate [G_{b, a}(t', t)]^*. The conjugation is performed
    independently for each point of the non-time components of G's mesh.

    g: Input Green's function.
    n_left_indices: Number of axes in G's target shape corresponding to the
                    multi-index 'a'. By default, a half of all axes.
    """

    assert len(g.mesh.components) >= 2
    assert isinstance(g.mesh.components[0], MeshReTime)
    assert isinstance(g.mesh.components[1], MeshReTime)

    mesh = MeshProduct(g.mesh.components[1],
                       g.mesh.components[0],
                       *g.mesh.components[2:])

    if n_left_indices is None:
        assert len(g.target_shape) % 2 == 0, \
            "n_left_indices must be provided when the target shape of the GF " \
            "has an odd number of dimensions"
        nli = len(g.target_shape) // 2
    else:
        nli = n_left_indices

    nri = len(g.target_shape) - nli
    ts = g.target_shape[nli:] + g.target_shape[:nli]

    g_conj = Gf(mesh=mesh, target_shape=ts)
    axes_from = [0, 1, *range(-1, - nri - 1, -1)]
    axes_to = [1, 0, *range(-1 - nli, - nri - nli - 1, -1)]
    g_conj.data[:] = np.conj(np.moveaxis(g.data, axes_from, axes_to))

    return g_conj
