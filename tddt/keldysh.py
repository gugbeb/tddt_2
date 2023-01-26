#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
from itertools import product
from typing import Tuple, Dict
from numpy import einsum
from triqs.gf import Gf, MeshReTime, MeshPoint, MeshProduct

from .util import simpsons_weights


class Branch(Enum):
    """Branch of the Keldysh contour"""
    FORWARD = 0
    BACKWARD = 1


class ContourPoint:
    """
    Point on the Keldysh contour, combination of a branch and a real time point
    """

    def __init__(self, branch, t):
        assert isinstance(t, MeshPoint)
        self.branch = branch
        self.t = t

    def __lt__(self, other):
        """
        This function defines the comparison rule used by `contour_ordering2()`
        and `contour_ordering3()`.
        """
        if self.branch == other.branch:
            if self.branch == Branch.FORWARD:
                return self.t.linear_index < other.t.linear_index
            else:
                return self.t.linear_index >= other.t.linear_index
        else:
            return self.branch.value < other.branch.value


def contour_ordering2(*points):
    """
    Contour ordering of two points

    Takes two contour points and returns a permutation of integers (0, 1)
    describing the order of the points on the contour. A pair of coinciding
    points on the forward branch comes in the original order in the output
    permutation, while for the backward branch the order is reversed.
    """
    return tuple(sorted((0, 1), key=lambda n: points[n], reverse=True))


def contour_ordering3(*points):
    """
    Contour ordering of three points

    Takes three contour points and returns a permutation of integers (0, 1, 2)
    describing the order of the points on the contour. A pair of coinciding
    points on the forward branch comes in the original order in the output
    permutation, while for the backward branch the order is reversed.
    """
    return tuple(sorted((0, 1, 2), key=lambda n: points[n], reverse=True))


class KeldyshGF:
    """Single-particle Green's function on the Keldysh contour"""

    def __init__(self, mesh: MeshProduct, target_shape=()):
        # The mesh must at least have two real time components
        assert len(mesh.components) >= 2
        assert isinstance(mesh.components[0], MeshReTime)
        assert isinstance(mesh.components[1], MeshReTime)

        # The following precondition is relied upon by __matmul__()
        assert len(mesh.components[0]) % 2 == 1 and \
               len(mesh.components[1]) % 2 == 1, \
               "Time grid must have an odd number of nodes"

        self.mesh = mesh
        self.time_mesh = MeshProduct(mesh.components[0], mesh.components[1])
        self.target_shape = target_shape

        # 4 Keldysh components as real time GFs
        self.data = [Gf(mesh=mesh, target_shape=target_shape) for _ in range(4)]

    @classmethod
    def from_g_l_g_g(cls, g_l, g_g):
        """
        Construct a KeldyshGF instance from a pair of lesser and greater
        real time Green's functions.
        """
        assert g_l.mesh == g_g.mesh
        assert g_l.target_shape == g_g.target_shape
        assert g_l.mesh.components[0] == g_l.mesh.components[1]

        g = KeldyshGF(g_l.mesh, g_l.target_shape)

        #
        # Fill Keldysh components
        #

        def ordered(z0, z1):
            return contour_ordering2(z0, z1) == (0, 1)

        def slice_gf_2t(g: Gf, t0, t1):
            n_non_t_mesh_comp = len(g.mesh.components) - 2
            return g[(t0, t1) + (slice(None),) * n_non_t_mesh_comp]

        non_t_slice = (slice(None),) * (len(g.mesh.components) - 2)

        # Aoki RMP, Eqs. (17)
        g[Branch.BACKWARD, Branch.FORWARD] = g_g
        g[Branch.FORWARD, Branch.BACKWARD] = g_l
        # Aoki RMP, Eqs. (15)
        for t0, t1 in g.time_mesh:
            sl = (t0, t1) + non_t_slice
            z0 = ContourPoint(Branch.FORWARD, t0)
            z1 = ContourPoint(Branch.FORWARD, t1)
            g[z0, z1] = g_g[sl] if ordered(z0, z1) else g_l[sl]
            z0 = ContourPoint(Branch.BACKWARD, t0)
            z1 = ContourPoint(Branch.BACKWARD, t1)
            g[z0, z1] = g_g[sl] if ordered(z0, z1) else g_l[sl]

        return g

    def _ravel_branch_indices(self, b1, b2):
        return 2 * b1.value + b2.value

    def __getitem__(self, points):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            return self.data[self._ravel_branch_indices(*points)]
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            g = self.data[self._ravel_branch_indices(points[0].branch,
                                                     points[1].branch)]
            n_non_t_mesh_comp = len(g.mesh.components) - 2
            return g[(points[0].t, points[1].t)
                     + (slice(None),) * n_non_t_mesh_comp]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, points, value):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            self.data[self._ravel_branch_indices(*points)] = value
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            g = self.data[self._ravel_branch_indices(points[0].branch,
                                                     points[1].branch)]
            n_non_t_mesh_comp = len(g.mesh.components) - 2
            g[(points[0].t, points[1].t)
              + (slice(None),) * n_non_t_mesh_comp] = value
        else:
            raise IndexError("Unrecognized index format")

    #
    # Arithmetics
    #

    def __iadd__(self, other):
        assert self.mesh == other.mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data):
            sd += od
        return self

    def __isub__(self, other):
        assert self.mesh == other.mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data):
            sd -= od
        return self

    def __imul__(self, x):
        for sd in self.data:
            sd *= x
        return self

    def __add__(self, other):
        res = deepcopy(self)
        res += other
        return res

    def __sub__(self, other):
        res = deepcopy(self)
        res -= other
        return res

    def __mul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __rmul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __neg__(self):
        res = deepcopy(self)
        res *= -1
        return res

    def __matmul__(self, other):
        """Contour convolution"""
        assert self.mesh == other.mesh
        # Weights for Simpson’s rule
        w = simpsons_weights(self.mesh.components[0])

        res = deepcopy(self)

        target_shape = ()

        subscripts_self = "ik..."
        subscripts_other = "kj..."
        subscripts_res = "ij..."

        if len(self.target_shape) == 1:  # Vector-valued
            subscripts_self += "l"
        elif len(self.target_shape) == 2:  # Matrix-valued
            subscripts_self += "ml"
            subscripts_res += "m"
            target_shape = target_shape + (self.target_shape[0],)
        elif len(self.target_shape) > 2:
            raise RuntimeError("Contour convolution is not implemented "
                               " for target dimensions > 2")

        if len(other.target_shape) == 1:  # Vector-valued
            subscripts_other += "l"
        elif len(other.target_shape) == 2:  # Matrix-valued
            subscripts_other += "ln"
            subscripts_res += "n"
            target_shape = target_shape + (other.target_shape[1],)
        elif len(self.target_shape) > 2:
            raise RuntimeError("Contour convolution is not implemented "
                               " for target dimensions > 2")

        subscripts = f"{subscripts_self},k,{subscripts_other}->{subscripts_res}"

        res.target_shape = target_shape

        FW, BW = Branch.FORWARD, Branch.BACKWARD
        for b0, b1 in product(Branch, Branch):
            res[b0, b1].data[:] = \
                einsum(subscripts, self[b0, FW].data, w, other[FW, b1].data) - \
                einsum(subscripts, self[b0, BW].data, w, other[BW, b1].data)

        return res


class KeldyshVertex3:
    """Three-point vertex function <c c^+ \\rho> on the Keldysh contour"""

    def __init__(self, mesh: MeshProduct, target_shape=()):
        # The mesh must at least have three real time components
        assert len(mesh.components) >= 3
        assert isinstance(mesh.components[0], MeshReTime)
        assert isinstance(mesh.components[1], MeshReTime)
        assert isinstance(mesh.components[2], MeshReTime)

        self.mesh = mesh
        self.time_mesh = MeshProduct(mesh.components[0],
                                     mesh.components[1],
                                     mesh.components[2])
        self.target_shape = target_shape

        # 8 Keldysh components as real time GFs
        self.data = [Gf(mesh=mesh, target_shape=target_shape) for _ in range(8)]

    @classmethod
    def from_G_perm_pieces(cls, G: Dict[Tuple[int, int, int], Gf]):
        r"""
        Each element of dictionary G corresponds to one permutation of operators
        in the correlator,
        $$
        G_{ijk}(t_0, t_1, t_2) = -\xi_{ijk} <O_i(t_i) O_j(t_j) O_k(t_k)>,
        $$
        where $O_0(t_0) = c(t_0)$, $O_1(t_1) = c^\dagger(t_1)$,
        $O_2(t_2) = \rho(t_2)$. $\xi_{ijk} = -1$ if permutation (ijk) swaps
        indices 0 and 1, and +1 otherwise.

        Keys are 3! = 6 triplets (i, j, k), which are permutations of (0, 1, 2)
        indicating the respective order of $c$, $c^\dagger$ and $\rho$.
        """

        assert len(G) == 6
        mesh = next(iter(G.values())).mesh
        target_shape = next(iter(G.values())).target_shape
        assert all(p.mesh == mesh for p in G.values())
        assert all(p.target_shape == target_shape for p in G.values())

        Lambda = KeldyshVertex3(mesh, target_shape)

        #
        # Fill Keldysh components
        #

        for a0, a1, a2 in product(Branch, repeat=3):
            for t0, t1, t2 in Lambda.mesh:
                z0 = ContourPoint(a0, t0)
                z1 = ContourPoint(a1, t1)
                z2 = ContourPoint(a2, t2)
                order = contour_ordering3(z0, z1, z2)
                Lambda[z0, z1, z2] = G[order][t0, t1, t2]

        return Lambda

    def _ravel_branch_indices(self, b1, b2, b3):
        return 4 * b1.value + 2 * b2.value + b3.value

    def __getitem__(self, points):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            return self.data[self._ravel_branch_indices(*points)]
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            return self.data[self._ravel_branch_indices(points[0].branch,
                                                        points[1].branch,
                                                        points[2].branch)][
                points[0].t, points[1].t, points[2].t]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, points, value):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            self.data[self._ravel_branch_indices(*points)] = value
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            self.data[self._ravel_branch_indices(points[0].branch,
                                                 points[1].branch,
                                                 points[2].branch)][
                points[0].t, points[1].t, points[2].t] = value
        else:
            raise IndexError("Unrecognized index format")

    #
    # Arithmetics
    #

    def __iadd__(self, other):
        assert self.mesh == other.mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data):
            sd += od
        return self

    def __isub__(self, other):
        assert self.mesh == other.mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data):
            sd -= od
        return self

    def __imul__(self, x):
        for sd in self.data:
            sd *= x
        return self

    def __add__(self, other):
        res = deepcopy(self)
        res += other
        return res

    def __sub__(self, other):
        res = deepcopy(self)
        res -= other
        return res

    def __mul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __rmul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __neg__(self):
        res = deepcopy(self)
        res *= -1
        return res
