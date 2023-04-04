#
# Evaluation of diagrams on Keldysh contour
#

from .keldysh import KeldyshGF, conv


def polarization_2nd_order(Lambda: KeldyshGF, g: KeldyshGF):
    r"""
    2nd order contribution to the polarization function.

    Lambda - 3-point vertex.
    g - fermionic line
    """

    # f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, \bar z, z_2) g(\bar z, z_1)
    f = conv(Lambda, g, [(1, 0)], free_args=([0, 2], [1]))
    # \Pi(z_1, z_2) = -i \int_C dz' dz'' f(z', z'', z_1) f(z'', z', z_2)
    return -1j * conv(f, f, [(0, 1), (1, 0)])
