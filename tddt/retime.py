#
# Auxiliary routines for 2-point Green's functions of real time
#

import numpy as np

from triqs.gf import Gf, MeshReTime, MeshProduct

from .util import subscripts
from .integration import GregoryIntegrator


def _is_2t_gf(g: Gf) -> bool:
    """Is g a 2-point real time GF?"""
    return len(g.mesh.components) >= 2 and \
        isinstance(g.mesh.components[0], MeshReTime) and \
        isinstance(g.mesh.components[1], MeshReTime)


def _extract_nli_nri(g: Gf, n_left_indices, assert_msg):
    """Return numbers of left and right indices in a real time GF."""
    if n_left_indices is None:
        assert len(g.target_shape) % 2 == 0, assert_msg
        nli = len(g.target_shape) // 2
    else:
        nli = n_left_indices

    nri = len(g.target_shape) - nli

    return nli, nri


def conj(g: Gf, *, n_left_indices=None) -> Gf:
    r"""
    Given a 2-point real time Green's function G_{a, b}(t, t'), returns its
    Hermitian conjugate [G_{b, a}(t', t)]^*. The conjugation is performed
    independently for each point of the non-time components of G's mesh.

    g: Input Green's function.
    n_left_indices: Number of axes in G's target shape corresponding to the
                    multi-index 'a'. By default, a half of all axes.
    """
    assert _is_2t_gf(g)

    mesh = MeshProduct(g.mesh.components[1],
                       g.mesh.components[0],
                       *g.mesh.components[2:])

    nli, nri = _extract_nli_nri(
        g,
        n_left_indices,
        "n_left_indices must be provided when the target shape of the GF "
        "has an odd number of dimensions"
    )

    ts = g.target_shape[nli:] + g.target_shape[:nli]

    g_conj = Gf(mesh=mesh, target_shape=ts)
    axes_from = [0, 1, *range(-1, - nri - 1, -1)]
    axes_to = [1, 0, *range(-1 - nli, - nri - nli - 1, -1)]
    g_conj.data[:] = np.conj(np.moveaxis(g.data, axes_from, axes_to))

    return g_conj


def _make_target_shape_subs(a_nli, a_nri, b_nli, b_nri):
    tgs = subscripts['target']
    subs_a_tg = tgs[:a_nli + a_nri]
    subs_b_tg = tgs[a_nli:(a_nli + b_nli + b_nri)]
    subs_res_tg = tgs[:a_nli]
    subs_res_tg += tgs[(a_nli + a_nri):(a_nli + b_nli + b_nri)]
    return subs_a_tg, subs_b_tg, subs_res_tg


def conv_ret_l(a_ret: Gf,
               b_l: Gf,
               a_ret_n_left_indices=None,
               gregory_order=5) -> Gf:
    r"""
    Compute a real-time convolution of an extended retarded function and a
    lesser function,

    F_{a, b}(t, t') =
        \sum_c \int_0^t d\bar t A^r_{a, c}(t, \bar t) B^<_{c, b}(\bar t, t').

    a_ret: The extended retarded function A^r(t, t').
    b_l: The lesser function B^<(t, t').
    a_ret_n_left_indices: Number of axes in A^r's target shape corresponding to
                          the multi-index 'a'. By default, a half of all axes.
    gregory_order: Order of the Gregory quadrature rule used to do the
                   convolution.
    """
    assert _is_2t_gf(a_ret)
    assert _is_2t_gf(b_l)

    assert a_ret.mesh.components[1] == b_l.mesh.components[0], \
           "Incompatible time meshes of a_ret and b_l"
    assert a_ret.mesh.components[2:] == b_l.mesh.components[2:], \
           "Different non-time meshes of a_ret and b_l"

    mesh = MeshProduct(a_ret.mesh.components[0],
                       b_l.mesh.components[1],
                       *a_ret.mesh.components[2:])

    a_ret_nli, a_ret_nri = _extract_nli_nri(
        a_ret,
        a_ret_n_left_indices,
        "a_ret_n_left_indices must be provided when the target shape of A^r "
        "has an odd number of dimensions"
    )

    b_l_nli = a_ret_nri
    b_l_nri = len(b_l.target_shape) - b_l_nli

    assert a_ret.target_shape[a_ret_nli:] == b_l.target_shape[:b_l_nli], \
        "Incompatible target shapes of a_ret and b_l"

    # Generate einsum() subscripts

    ts = subscripts['time']
    subs_a_ret_t = ts[0] + ts[2]
    subs_b_l_t = ts[2] + ts[1]
    subs_res_t = ts[0] + ts[1]
    subs_w = subs_a_ret_t

    subs_nt = subscripts['nontime'][:len(a_ret.mesh.components[2:])]

    subs_a_ret_tg, subs_b_l_tg, subs_res_tg = \
        _make_target_shape_subs(a_ret_nli, a_ret_nri, b_l_nli, b_l_nri)

    target_shape_res = \
        a_ret.target_shape[:a_ret_nli] + b_l.target_shape[b_l_nli:]

    # Perform summation

    subs_a_ret = subs_a_ret_t + subs_nt + subs_a_ret_tg
    subs_b_l = subs_b_l_t + subs_nt + subs_b_l_tg
    subs_res = subs_res_t + subs_nt + subs_res_tg

    subs = f"{subs_a_ret},{subs_w},{subs_b_l}->{subs_res}"

    res = Gf(mesh=mesh, target_shape=target_shape_res)

    w = GregoryIntegrator(gregory_order).weights(a_ret.mesh.components[1])
    res.data[:] = np.einsum(subs, a_ret.data, w, b_l.data, optimize="optimal")

    return res


def conv_l_adv(a_l: Gf,
               b_adv: Gf,
               a_l_n_left_indices=None,
               gregory_order=5) -> Gf:
    r"""
    Compute a real-time convolution of a lesser function and an extended
    advanced function,

    F_{a, b}(t, t') =
        \sum_c \int_0^{t'} d\bar t A^<_{a, c}(t, \bar t) B^a{c, b}(\bar t, t').

    a_l: The lesser function A^<(t, t').
    b_l: The extended advanced function B^a(t, t').
    a_l_n_left_indices: Number of axes in A^<'s target shape corresponding to
                        the multi-index 'a'. By default, a half of all axes.
    gregory_order: Order of the Gregory quadrature rule used to do the
                   convolution.
    """
    assert _is_2t_gf(a_l)
    assert _is_2t_gf(b_adv)

    assert a_l.mesh.components[1] == b_adv.mesh.components[0], \
           "Incompatible time meshes of a_l and b_adv"
    assert a_l.mesh.components[2:] == b_adv.mesh.components[2:], \
           "Different non-time meshes of a_l and b_adv"

    mesh = MeshProduct(a_l.mesh.components[0],
                       b_adv.mesh.components[1],
                       *a_l.mesh.components[2:])

    a_l_nli, a_l_nri = _extract_nli_nri(
        a_l,
        a_l_n_left_indices,
        "a_l_n_left_indices must be provided when the target shape of A^< "
        "has an odd number of dimensions"
    )

    b_adv_nli = a_l_nri
    b_adv_nri = len(b_adv.target_shape) - b_adv_nli

    assert a_l.target_shape[a_l_nli:] == b_adv.target_shape[:b_adv_nli], \
        "Incompatible target shapes of a_l and b_adv"

    # Generate einsum() subscripts

    ts = subscripts['time']
    subs_a_l_t = ts[0] + ts[2]
    subs_b_adv_t = ts[2] + ts[1]
    subs_res_t = ts[0] + ts[1]
    subs_w = ts[1] + ts[2]

    subs_nt = subscripts['nontime'][:len(a_l.mesh.components[2:])]

    subs_a_l_tg, subs_b_adv_tg, subs_res_tg = \
        _make_target_shape_subs(a_l_nli, a_l_nri, b_adv_nli, b_adv_nri)

    target_shape_res = \
        a_l.target_shape[:a_l_nli] + b_adv.target_shape[b_adv_nli:]

    # Perform summation

    subs_a_l = subs_a_l_t + subs_nt + subs_a_l_tg
    subs_b_adv = subs_b_adv_t + subs_nt + subs_b_adv_tg
    subs_res = subs_res_t + subs_nt + subs_res_tg

    subs = f"{subs_a_l},{subs_w},{subs_b_adv}->{subs_res}"

    res = Gf(mesh=mesh, target_shape=target_shape_res)

    w = GregoryIntegrator(gregory_order).weights(a_l.mesh.components[1])
    res.data[:] = np.einsum(subs, a_l.data, w, b_adv.data, optimize="optimal")

    return res
