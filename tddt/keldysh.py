#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
import numpy as np

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

    Takes two contour points and returns a permutation of integers [0, 1]
    describing the order of the points on the contour. A pair of coinciding
    points on the forward branch comes in the original order in the output
    permutation, while for the backward branch the order is reversed.
    """
    order = [0, 1]
    order.sort(key = lambda n: points[n], reverse = True)
    return order

def contour_ordering3(*points):
    """
    Contour ordering of three points

    Takes three contour points and returns a permutation of integers [0, 1, 2]
    describing the order of the points on the contour. A pair of coinciding
    points on the forward branch comes in the original order in the output
    permutation, while for the backward branch the order is reversed.
    """
    order = [0, 1, 2]
    order.sort(key = lambda n: points[n], reverse = True)
    return order

class KeldyshGF:
    """Single-particle Green's function on the Keldysh contour"""

    def __init__(self, g_l, g_g):
        assert g_l.mesh == g_g.mesh
        self.time_mesh = g_l.mesh

        # 2 Keldysh indices, 2 real time indices
        self.data = np.zeros((2, 2, *self.time_mesh.size_of_components()),
                             dtype = complex)

        #
        # Fill Keldysh components
        #

        # Aoki RMP, Eqs. (17)
        self[Branch.BACKWARD, Branch.FORWARD] = g_g.data[:]
        self[Branch.FORWARD, Branch.BACKWARD] = g_l.data[:]
        # Aoki RMP, Eqs. (15)
        for t1, t2 in self.time_mesh:
            i1, i2 = t1.linear_index, t2.linear_index
            self[Branch.FORWARD, Branch.FORWARD, t1, t2] = \
                g_g[t1, t2] if (i1 >= i2) else g_l[t1, t2]
            self[Branch.BACKWARD, Branch.BACKWARD, t1, t2] = \
                g_l[t1, t2] if (i1 >= i2) else g_g[t1, t2]

    def __getitem__(self, index):
        if len(index) == 2: # Specify the Keldysh indices
            return self.data[index[0].value, index[1].value, :, :]
        elif len(index) == 4: # Keldysh indices + real time points
            return self.data[index[0].value,
                             index[1].value,
                             index[2].linear_index,
                             index[3].linear_index]
        else:
            raise IndexError("Unrecognized index format")

    def __setitem__(self, index, value):
        if len(index) == 2: # Specify the Keldysh indices
            self.data[index[0].value, index[1].value, :, :] = value
        elif len(index) == 4: # Keldysh indices + real time points
            self.data[index[0].value,
                      index[1].value,
                      index[2].linear_index,
                      index[3].linear_index] = value
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
