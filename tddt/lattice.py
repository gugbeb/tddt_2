#
# Functions and types related to lattice and Brillouin zone
#

from __future__ import annotations

from typing import Union
import numpy as np

from triqs.gf import MeshBrZone, MeshProduct

from .keldysh import KeldyshGF, Singular2PKeldyshGF


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
