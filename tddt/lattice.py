#
# Functions and types related to lattice and Brillouin zone
#

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Union
import numpy as np

from triqs.gf import (MeshBrZone,
                      MeshCycLat,
                      MeshProduct,
                      make_adjoint_mesh,
                      Gf)

from .keldysh import Branch, KeldyshGF, Singular2PKeldyshGF


class SpacialArgs(Enum):
    "Kind of spacial arguments of a Green's function."
    LATTICE = 0
    """
    Real space (lattice) arguments
    """
    BRZONE = 1
    """
    Reciprocal space (Brillouin zone) arguments
    """
    BOTH = 2
    """
    Both real and reciprocal space arguments
    """


def local_part(g: Union[KeldyshGF, Singular2PKeldyshGF]) -> \
        Union[KeldyshGF, Singular2PKeldyshGF]:
    r"""
    Given a contour function 'g', compute the average over each MeshBrZone
    it is defined on.
    """
    mesh_comps_res = ()  # Components of resulting mesh
    avg_axes = ()        # Data axes to average over

    for i, m in enumerate(g.mesh.components):
        if isinstance(m, MeshBrZone):
            avg_axes += (i,)
        else:
            mesh_comps_res += (m,)

    res = type(g)(mesh=MeshProduct(*mesh_comps_res),
                  arg_index_shapes=g.arg_index_shapes)
    for g_comp, res_comp in zip(g.components.flat, res.components.flat):
        res_comp.data[:] = np.average(g_comp.data, axis=avg_axes)

    return res


def _fourier_impl(g_in: Gf, g_out: Gf, mesh_comp_pos: list[int], forward: bool):
    data_axes = ()
    in_data_shape = ()
    out_data_shape = ()
    axis = 0
    for i, m in enumerate(g_in.mesh.components):
        if i in mesh_comp_pos:
            data_axes += (axis, axis + 1, axis + 2)
            axis += 3
            in_data_shape += tuple(m.dims)
            out_data_shape += tuple(m.dims)
        else:
            axis += 1
            in_data_shape += (len(m),)
            out_data_shape += (len(m),)

    in_data_shape += g_in.target_shape
    out_data_shape += g_out.target_shape

    func = np.fft.fftn if forward else np.fft.ifftn
    func(np.reshape(g_in.data, in_data_shape),
         axes=data_axes,
         norm="forward",
         out=np.reshape(g_out.data, out_data_shape))


def lattice_fourier(g: Union[KeldyshGF, Singular2PKeldyshGF],
                    apply_to: SpacialArgs = SpacialArgs.BOTH) -> \
        Union[KeldyshGF, Singular2PKeldyshGF]:
    r"""
    Given a contour function 'g', apply a lattice Fourier transform to each
    MeshBrZone (if apply_to == SpacialArgs.BRZONE) or MeshCycLat
    (if apply_to == SpacialArgs.LATTICE) component of the mesh.
    apply_to == SpacialArgs.BOTH results in a transformation of both mesh types.
    """

    # Positions of MeshBrZone mesh components
    mesh_brzone_pos = []
    # Positions of MeshCycLat mesh components
    mesh_cyclat_pos = []

    for i, m in enumerate(g.mesh.components):
        if isinstance(m, MeshBrZone):
            mesh_brzone_pos.append(i)
        elif isinstance(m, MeshCycLat):
            mesh_cyclat_pos.append(i)

    g_tmp = g

    # Perform Fourier transform w.r.t. k-arguments
    if (apply_to in (SpacialArgs.BRZONE, SpacialArgs.BOTH)
            and len(mesh_brzone_pos) > 0):
        mesh_comps = [
            make_adjoint_mesh(m) if (i in mesh_brzone_pos) else m
            for i, m in enumerate(g_tmp.mesh.components)
        ]
        g_res = type(g_tmp)(mesh=MeshProduct(*mesh_comps),
                            arg_index_shapes=g_tmp.arg_index_shapes)
        for br in product(Branch, repeat=g.components.ndim):
            _fourier_impl(g_tmp[br], g_res[br], mesh_brzone_pos, True)
        g_tmp = g_res

    # Perform Fourier transform w.r.t. r-arguments
    if (apply_to in (SpacialArgs.LATTICE, SpacialArgs.BOTH)
            and len(mesh_cyclat_pos) > 0):
        mesh_comps = [
            make_adjoint_mesh(m) if (i in mesh_cyclat_pos) else m
            for i, m in enumerate(g_tmp.mesh.components)
        ]
        g_res = type(g_tmp)(mesh=MeshProduct(*mesh_comps),
                            arg_index_shapes=g_tmp.arg_index_shapes)
        for br in product(Branch, repeat=g.components.ndim):
            _fourier_impl(g_tmp[br], g_res[br], mesh_cyclat_pos, False)
        g_tmp = g_res

    return g_tmp
