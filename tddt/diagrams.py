#
# Evaluation of diagrams on Keldysh contour
#

from enum import Enum
from itertools import product
from string import ascii_lowercase
from numpy import einsum

from triqs.gf import MeshProduct

from .keldysh import Branch, KeldyshGF, KeldyshVertex3
from .util import simpsons_weights


class VertexLeg(Enum):
    """Leg of a 3-point vertex"""
    INBOUND = 0
    OUTBOUND = 1
    BOSON = 2


def vertex3_attach_leg(Lambda: KeldyshVertex3, g: KeldyshGF, leg: VertexLeg):
    r"""
    Compute one of the following contour convolutions of a 3-point vertex
    \Lambda(z_0, z_1, z_2) and a fermionic/bosonic single-particle function
    g(z_0, z_1).

    leg = VertexLeg.INBOUND
    -----------------------
    f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, \bar z, z_2) g(\bar z, z_1)

    leg = VertexLeg.OUTBOUND
    ------------------------
    f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(\bar z, z_1, z_2) g(z_0, \bar z)

    leg = VertexLeg.BOSON
    ---------------------
    f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, z_1, \bar z) g(\bar z, z_2)
    """

    # Non-time components of the mesh
    non_t_mesh_comps = Lambda.mesh.components[3:] + g.mesh.components[2:]

    # einsum() subscripts corresponding to the resulting function
    # f(z_0, z_1, z_2) are as follows:
    #
    # * z_0 -> i
    # * z_1 -> j
    # * z_2 -> k
    # * Non-time components of the mesh -> a, b, c, ...
    # 3-dimensional target space -> x, y, z

    non_t_mesh_subscripts = ascii_lowercase[:len(non_t_mesh_comps)]
    assert non_t_mesh_subscripts == '' or non_t_mesh_subscripts[-1] < 'i'
    subscripts_res = "ijk" + non_t_mesh_subscripts
    if Lambda.target_shape != ():
        subscripts_res += "xyz"

    # Subscripts: Non-time components of Lambda's and g's mesh
    n_non_t_mesh_comps_Lambda = len(Lambda.mesh.components[3:])
    non_t_mesh_subscripts_Lambda, non_t_mesh_subscripts_g = \
        non_t_mesh_subscripts[:n_non_t_mesh_comps_Lambda], \
        non_t_mesh_subscripts[n_non_t_mesh_comps_Lambda:]

    FW, BW = Branch.FORWARD, Branch.BACKWARD

    # Either both Lambda and g are scalar-valued or they both are not
    assert (len(Lambda.target_shape) == 0 and len(g.target_shape) == 0) or \
           (len(Lambda.target_shape) == 3 and len(g.target_shape) == 2)

    if leg == VertexLeg.INBOUND:
        assert Lambda.time_mesh.components[1] == g.time_mesh.components[0]
        res_mesh = MeshProduct(Lambda.time_mesh.components[0],
                               g.time_mesh.components[1],
                               Lambda.time_mesh.components[2],
                               *non_t_mesh_comps)

        subscripts_Lambda = "ilk" + non_t_mesh_subscripts_Lambda
        subscripts_g = "lj" + non_t_mesh_subscripts_g

        if Lambda.target_shape == ():
            res_target_shape = ()
        else:
            assert Lambda.target_shape[1] == g.target_shape[0]
            res_target_shape = (Lambda.target_shape[0],
                                g.target_shape[1],
                                Lambda.target_shape[2])
            subscripts_Lambda += "xwz"
            subscripts_g += "wy"

        subscripts = f"{subscripts_Lambda},l,{subscripts_g}->{subscripts_res}"

        res = KeldyshVertex3(mesh=res_mesh, target_shape=res_target_shape)

        w = simpsons_weights(Lambda.time_mesh.components[1])
        for b0, b1, b2 in product(Branch, repeat=3):
            res[b0, b1, b2].data[:] = \
                einsum(subscripts, Lambda[b0, FW, b2].data, w, g[FW, b1].data) \
                - einsum(subscripts, Lambda[b0, BW, b2].data, w, g[BW, b1].data)
    elif leg == VertexLeg.OUTBOUND:
        assert Lambda.time_mesh.components[0] == g.time_mesh.components[1]
        res_mesh = MeshProduct(g.time_mesh.components[0],
                               Lambda.time_mesh.components[1],
                               Lambda.time_mesh.components[2],
                               *non_t_mesh_comps)

        subscripts_Lambda = "ljk" + non_t_mesh_subscripts_Lambda
        subscripts_g = "il" + non_t_mesh_subscripts_g

        if Lambda.target_shape == ():
            res_target_shape = ()
        else:
            assert Lambda.target_shape[0] == g.target_shape[1]
            res_target_shape = (g.target_shape[0],
                                Lambda.target_shape[1],
                                Lambda.target_shape[2])
            subscripts_Lambda += "wyz"
            subscripts_g += "xw"

        subscripts = f"{subscripts_Lambda},l,{subscripts_g}->{subscripts_res}"

        res = KeldyshVertex3(mesh=res_mesh, target_shape=res_target_shape)

        w = simpsons_weights(Lambda.time_mesh.components[0])
        for b0, b1, b2 in product(Branch, repeat=3):
            res[b0, b1, b2].data[:] = \
                einsum(subscripts, Lambda[FW, b1, b2].data, w, g[b0, FW].data) \
                - einsum(subscripts, Lambda[BW, b1, b2].data, w, g[b0, BW].data)
    elif leg == VertexLeg.BOSON:
        assert Lambda.time_mesh.components[2] == g.time_mesh.components[0]
        res_mesh = MeshProduct(Lambda.time_mesh.components[0],
                               Lambda.time_mesh.components[1],
                               g.time_mesh.components[1],
                               *non_t_mesh_comps)

        subscripts_Lambda = "ijl" + non_t_mesh_subscripts_Lambda
        subscripts_g = "lk" + non_t_mesh_subscripts_g

        if Lambda.target_shape == ():
            res_target_shape = ()
        else:
            assert Lambda.target_shape[0] == g.target_shape[1]
            res_target_shape = (g.target_shape[0],
                                Lambda.target_shape[1],
                                Lambda.target_shape[2])
            subscripts_Lambda += "xyw"
            subscripts_g += "wz"

        subscripts = f"{subscripts_Lambda},l,{subscripts_g}->{subscripts_res}"

        res = KeldyshVertex3(mesh=res_mesh, target_shape=res_target_shape)

        w = simpsons_weights(Lambda.time_mesh.components[2])
        for b0, b1, b2 in product(Branch, repeat=3):
            res[b0, b1, b2].data[:] = \
                einsum(subscripts, Lambda[b0, b1, FW].data, w, g[FW, b2].data) \
                - einsum(subscripts, Lambda[b0, b1, BW].data, w, g[BW, b2].data)
    else:
        raise RuntimeError("Unknown leg specification")

    return res


def polarization_2nd_order(Lambda: KeldyshVertex3, g: KeldyshGF):
    r"""
    2nd order contribution to the polarization function.

    Lambda - 3-point vertex.
    g - fermionic line
    """

    # f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, \bar z, z_2) g(\bar z, z_1)
    f = vertex3_attach_leg(Lambda, g, VertexLeg.INBOUND)

    # Evaluate \Pi(z_0, z_1) = \int_C d\bar z_0 \int_C d\bar z_1
    # f(\bar z_0, \bar z_1, z_0) f(\bar z_1, \bar z_0, z_1)

    # Non-time components of the mesh
    non_t_mesh_comps = f.mesh.components[3:] + f.mesh.components[3:]

    non_t_mesh_subscripts = ascii_lowercase[:len(non_t_mesh_comps)]
    assert non_t_mesh_subscripts == '' or non_t_mesh_subscripts[-1] < 'i'
    subscripts_res = "ij" + non_t_mesh_subscripts

    # Subscripts: Non-time components of f's mesh
    n_non_t_mesh_comps_f = len(f.mesh.components[3:])
    non_t_mesh_subscripts_f1, non_t_mesh_subscripts_f2 = \
        non_t_mesh_subscripts[:n_non_t_mesh_comps_f], \
        non_t_mesh_subscripts[n_non_t_mesh_comps_f:]

    subscripts_f1 = "kli" + non_t_mesh_subscripts_f1
    subscripts_f2 = "lkj" + non_t_mesh_subscripts_f2

    # Either f is scalar-valued or 3-tensor-valued
    assert len(f.target_shape) == 0 or len(f.target_shape) == 3

    if f.target_shape == ():
        res_target_shape = ()
    else:
        res_target_shape = (f.target_shape[2], f.target_shape[2])
        subscripts_f1 += "uwx"
        subscripts_f2 += "wuy"
        subscripts_res += "xy"

    subscripts = f"{subscripts_f1},k,l,{subscripts_f2}->{subscripts_res}"

    assert f.time_mesh.components[0] == f.time_mesh.components[1]
    res_mesh = MeshProduct(f.time_mesh.components[2],
                           f.time_mesh.components[2],
                           *non_t_mesh_comps)

    res = KeldyshGF(mesh=res_mesh, target_shape=res_target_shape)

    FW, BW = Branch.FORWARD, Branch.BACKWARD
    w = simpsons_weights(f.time_mesh.components[0])
    for b0, b1 in product(Branch, repeat=2):
        res[b0, b1].data[:] = \
            einsum(subscripts, f[FW, FW, b0].data, w, w, f[FW, FW, b1].data) -\
            einsum(subscripts, f[FW, BW, b0].data, w, w, f[BW, FW, b1].data) -\
            einsum(subscripts, f[BW, FW, b0].data, w, w, f[FW, BW, b1].data) +\
            einsum(subscripts, f[BW, BW, b0].data, w, w, f[BW, BW, b1].data)

    return res
