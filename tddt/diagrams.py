#
# Evaluation of diagrams on Keldysh contour
#

from .keldysh import KeldyshGF, conv


def polarization_2nd_order(Lambda: KeldyshGF, g: KeldyshGF):
    r"""
    2nd order contribution to the polarization function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"

    # f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, \bar z, z_2) g(\bar z, z_1)
    f = conv(Lambda, g, [(1, 0)], free_args=([0, 2], [1]))
    # \Pi(z_1, z_2) = -i \int_C dz' dz'' f(z', z'', z_1) f(z'', z', z_2)
    return -1j * conv(f, f, [(0, 1), (1, 0)])


def selfenergy_2nd_order(Lambda: KeldyshGF, g: KeldyshGF, w: KeldyshGF):
    r"""
    2nd order contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    w: Bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"
    assert g.n_args == 2, "w must be a 2-point GF"

    # f1(z_1, z'''', z'') = \int_C dz' \Lambda(z_1, z', z'') g(z', z'''')
    f1 = conv(Lambda, g, [(1, 0)], free_args=([0, 2], [1]))
    # f2(z'''', z_2, z'') = \int_C dz''' \Lambda(z'''', z_2, z''') w(z'', z''')
    f2 = conv(Lambda, w, [(2, 1)], free_args=([0, 1], [2]))
    # \Sigma(z_1, z_2) = i \int_C dz'' dz'''' f1(z_1, z'''', z'')
    #                                         f2(z'''', z_2, z'')
    return 1j * conv(f1, f2, [(1, 0), (2, 2)])


def selfenergy_2nd_order_hf(Lambda: KeldyshGF, g: KeldyshGF, w: KeldyshGF):
    r"""
    Hartree-Fock contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    w: Bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"
    assert g.n_args == 2, "w must be a 2-point GF"

    # n(z'') = \int_C dz''' dz'''' \Lambda(z''', z'''', z'') g(z'''', z''')
    n = conv(Lambda, g, [(0, 1), (1, 0)])
    # wn(z') = \int_C dz'' w(z', z'') n(z'')
    wn = conv(w, n, [(1, 0)])
    # \Sigma_{HF}(z_1, z_2) = -i \int_C dz' \Lambda(z_1, z_2, z') wn(z')
    return -1j * conv(Lambda, wn, [(2, 0)])
