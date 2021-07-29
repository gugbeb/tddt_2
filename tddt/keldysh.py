#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
from itertools import product
from typing import Tuple, Union
from numpy import zeros
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
        self.time_mesh = g_l.mesh

        # 2 Keldysh indices, 2 real time indices
        self.data = zeros((2, 2, *self.time_mesh.size_of_components()),
                          dtype = complex)

        #
        # Fill Keldysh components
        #

        ordered = lambda z0, z1: contour_ordering2(z0, z1) == (0, 1)

        # Aoki RMP, Eqs. (17)
        self[Branch.BACKWARD, Branch.FORWARD] = g_g.data[:]
        self[Branch.FORWARD, Branch.BACKWARD] = g_l.data[:]
        # Aoki RMP, Eqs. (15)
        for t0, t1 in self.time_mesh:
            z0 = ContourPoint(Branch.FORWARD, t0)
            z1 = ContourPoint(Branch.FORWARD, t1)
            self[z0, z1] = g_g[t0, t1] if ordered(z0, z1) else g_l[t0, t1]
            z0 = ContourPoint(Branch.BACKWARD, t0)
            z1 = ContourPoint(Branch.BACKWARD, t1)
            self[z0, z1] = g_g[t0, t1] if ordered(z0, z1) else g_l[t0, t1]

    def __getitem__(self, points):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            return self.data[points[0].value, points[1].value, ...]
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            return self.data[points[0].branch.value,
                             points[1].branch.value,
                             points[0].t.linear_index,
                             points[1].t.linear_index]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, points, value):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            self.data[points[0].value, points[1].value, ...] = value
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            self.data[points[0].branch.value,
                      points[1].branch.value,
                      points[0].t.linear_index,
                      points[1].t.linear_index] = value
        else:
            raise IndexError("Unrecognized index format")

    #
    # Arithmetics
    #

    def __iadd__(self, other):
        assert self.time_mesh == other.time_mesh
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other):
        assert self.time_mesh == other.time_mesh
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, x):
        self.data[:] *= x
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
        res.data[:] = x * res.data[:]
        return res

    def __neg__(self):
        res = deepcopy(self)
        res.data *= -1
        return res

class KeldyshVertex3:
    """Three-point vertex function <c c^+ \\rho> on the Keldysh contour"""

    def __init__(self, G: dict[Tuple[int,int,int], Gf]):
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
        assert all(p.mesh == self.time_mesh for p in G.values())

        # 3 Keldysh indices, 3 real time indices
        self.data = zeros((2, 2, 2, *self.time_mesh.size_of_components()),
                          dtype = complex)

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

    def __getitem__(self, points):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            return self.data[points[0].value,
                             points[1].value,
                             points[2].value,
                             ...]
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            return self.data[points[0].branch.value,
                             points[1].branch.value,
                             points[2].branch.value,
                             points[0].t.linear_index,
                             points[1].t.linear_index,
                             points[2].t.linear_index]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, points, value):
        # Access one Keldysh block
        if all(isinstance(p, Branch) for p in points):
            self.data[points[0].value,
                      points[1].value,
                      points[2].value,
                      ...] = value
        # Access a single element
        elif all(isinstance(p, ContourPoint) for p in points):
            self.data[points[0].branch.value,
                      points[1].branch.value,
                      points[2].branch.value,
                      points[0].t.linear_index,
                      points[1].t.linear_index,
                      points[2].t.linear_index] = value
        else:
            raise IndexError("Unrecognized index format")

    #
    # Arithmetics
    #

    def __iadd__(self, other):
        assert self.time_mesh == other.time_mesh
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other):
        assert self.time_mesh == other.time_mesh
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, x):
        self.data[:] *= x
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
        res.data[:] = x * res.data[:]
        return res

    def __neg__(self):
        res = deepcopy(self)
        res.data *= -1
        return res
