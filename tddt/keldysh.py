#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
from itertools import product
from typing import Tuple, Union, Dict
from numpy import zeros, ones
from triqs.gf import Gf

class Branch(Enum):
    """Branch of the Keldysh contour"""
    FORWARD = 0
    BACKWARD = 1

class ContourPoint:
    """Point on the Keldysh contour, combination of a branch and a real time point"""

    def __init__(self, branch, t):
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
    return tuple(sorted((0, 1), key = lambda n: points[n], reverse = True))

def contour_ordering3(*points):
    """
    Contour ordering of three points

    Takes three contour points and returns a permutation of integers (0, 1, 2)
    describing the order of the points on the contour. A pair of coinciding
    points on the forward branch comes in the original order in the output
    permutation, while for the backward branch the order is reversed.
    """
    return tuple(sorted((0, 1, 2), key = lambda n: points[n], reverse = True))

class KeldyshGF:
    """Single-particle Green's function on the Keldysh contour"""

    def __init__(self, g_l, g_g):
        assert g_l.mesh == g_g.mesh
        assert g_l.target_shape == g_g.target_shape
        self.time_mesh = g_l.mesh
        self.target_shape = g_l.target_shape

        # The following precondition is relied upon by __matmul__()
        assert len(g_l.mesh) % 2 == 1, \
               "Time grid must have an odd number of nodes"

        # 4 Keldysh components as real time GFs
        self.data = []
        for _ in range(4):
            self.data.append(
                Gf(mesh=self.time_mesh, target_shape=self.target_shape)
            )

        #
        # Fill Keldysh components
        #

        ordered = lambda z0, z1: contour_ordering2(z0, z1) == (0, 1)

        # Aoki RMP, Eqs. (17)
        self[Branch.BACKWARD, Branch.FORWARD] = g_g
        self[Branch.FORWARD, Branch.BACKWARD] = g_l
        # Aoki RMP, Eqs. (15)
        for t0, t1 in self.time_mesh:
            z0 = ContourPoint(Branch.FORWARD, t0)
            z1 = ContourPoint(Branch.FORWARD, t1)
            self[z0, z1] = g_g[t0, t1] if ordered(z0, z1) else g_l[t0, t1]
            z0 = ContourPoint(Branch.BACKWARD, t0)
            z1 = ContourPoint(Branch.BACKWARD, t1)
            self[z0, z1] = g_g[t0, t1] if ordered(z0, z1) else g_l[t0, t1]

    def _ravel_branch_indices(self, b1, b2):
        return 2 * b1.value + b2.value

    def __getitem__(self, points):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            return self.data[self._ravel_branch_indices(*points)]
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            return self.data[
                self._ravel_branch_indices(points[0].branch, points[1].branch)][
                points[0].t, points[1].t]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, points, value):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            self.data[self._ravel_branch_indices(*points)] = value
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            self.data[self._ravel_branch_indices(points[0].branch,
                                                 points[1].branch)][
                      points[0].t, points[1].t] = value
        else:
            raise IndexError("Unrecognized index format")

    #
    # Arithmetics
    #

    def __iadd__(self, other):
        assert self.time_mesh == other.time_mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data): sd += od
        return self

    def __isub__(self, other):
        assert self.time_mesh == other.time_mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data): sd -= od
        return self

    def __imul__(self, x):
        for sd in self.data: sd *= x
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
        assert self.time_mesh == other.time_mesh
        assert self.target_shape == other.target_shape
        # Weights for Simpson’s rule
        w = ones(len(self.time_mesh[0]))
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        w *= self.time_mesh[0].delta / 3
        res = deepcopy(self)
        # TODO: For now, we assume that all real-time GFs are scalar-valued
        # and self.data.ndim == 2.
        for b0, b1 in product(Branch, Branch):
            res[b0, b1].data[:] = \
                (self[b0, Branch.FORWARD].data * w) @ \
                 other[Branch.FORWARD, b1].data - \
                (self[b0, Branch.BACKWARD].data * w) @ \
                 other[Branch.BACKWARD, b1].data
        return res

class KeldyshVertex3:
    """Three-point vertex function <c c^+ \\rho> on the Keldysh contour"""

    def __init__(self, G: Dict[Tuple[int,int,int], Gf]):
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
        self.time_mesh = next(iter(G.values())).mesh
        self.target_shape = next(iter(G.values())).target_shape
        assert all(p.mesh == self.time_mesh for p in G.values())
        assert all(p.target_shape == self.target_shape for p in G.values())

        # 8 Keldysh components as real time GFs
        self.data = []
        for _ in range(8):
            self.data.append(
                Gf(mesh=self.time_mesh, target_shape=self.target_shape)
            )

        #
        # Fill Keldysh components
        #

        for a0, a1, a2 in product(Branch, repeat = 3):
            for t0, t1, t2 in self.time_mesh:
                z0 = ContourPoint(a0, t0)
                z1 = ContourPoint(a1, t1)
                z2 = ContourPoint(a2, t2)
                order = contour_ordering3(z0, z1, z2)
                self[z0, z1, z2] = G[order][t0, t1, t2]

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
                             points[0].t,
                             points[1].t,
                             points[2].t]
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
        assert self.time_mesh == other.time_mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data): sd += od
        return self

    def __isub__(self, other):
        assert self.time_mesh == other.time_mesh
        assert self.target_shape == other.target_shape
        for sd, od in zip(self.data, other.data): sd -= od
        return self

    def __imul__(self, x):
        for sd in self.data: sd *= x
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
