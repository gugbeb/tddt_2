#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
from itertools import product, takewhile
from typing import Tuple, Dict, Union
from numpy import array, tril_indices, triu_indices
from triqs.gf import Gf, MeshReTime, MeshPoint, MeshProduct

from .integration import GregoryIntegrator


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
        This function defines the comparison rule used by `contour_ordering()`.
        """
        if self.branch == other.branch:
            if self.branch == Branch.FORWARD:
                return self.t.linear_index < other.t.linear_index
            else:
                return self.t.linear_index >= other.t.linear_index
        else:
            return self.branch.value < other.branch.value


def contour_ordering(*points):
    """
    Contour ordering of a list of points

    Takes a list of N contour points and returns a permutation of integers
    (0, 1, N-1) describing the order of the points on the contour. A pair of
    coinciding points on the forward branch comes in the original order in the
    output permutation, while for the backward branch the order is reversed.
    """
    return tuple(sorted(range(len(points)),
                        key=lambda n: points[n],
                        reverse=True))


class KeldyshGF:
    """Generic N-point Green's function defined on a 2-branch Keldysh contour"""

    """Integrator object for contour convolutions"""
    integrator = GregoryIntegrator(5)

    def __init__(self, *,
                 mesh: Union[MeshReTime, MeshProduct],
                 target_subshapes: Tuple[Tuple[int, ...], ...] = None):

        #
        # Process the supplied mesh
        #

        if mesh is None:  # Constant
            self.mesh = MeshProduct()
            self.time_mesh = MeshProduct()
            self.non_time_mesh = MeshProduct()
            self.n_args = 0

        elif isinstance(mesh, MeshReTime):  # Single-argument contour function
            self.mesh = MeshProduct(mesh)
            self.time_mesh = self.mesh
            self.non_time_mesh = MeshProduct()
            self.n_args = 1

        elif isinstance(mesh, MeshProduct):  # N-point Green's function
            self.mesh = mesh
            self.time_mesh = MeshProduct(
                *takewhile(lambda m: isinstance(m, MeshReTime), mesh.components)
            )
            self.n_args = len(self.time_mesh.components)
            if len(mesh.components[self.n_args:]) != 0:
                self.non_time_mesh = MeshProduct(*mesh.components[self.n_args:])
            else:
                self.non_time_mesh = MeshProduct()
        else:
            raise TypeError(f"Unsupported mesh type {type(mesh)}")

        #
        # Process the target subshapes
        #

        # All subshapes are 0-dimensional by default
        if target_subshapes is None:
            self.target_subshapes = ((),) * self.n_args
            self.target_shape = ()
        else:
            assert len(target_subshapes) == self.n_args, \
                f"target_subshapes must contain {self.n_args} elements for " \
                f"a {self.n_args}-point function"
            self.target_subshapes = target_subshapes
            self.target_shape = sum(target_subshapes, ())

        #
        # Allocate data storage
        #

        self.components = array(
            [Gf(mesh=self.mesh, target_shape=self.target_shape)
             for _ in range(2 ** self.n_args)]
        ).reshape((2,) * self.n_args)

    def __getitem__(self, args):
        args_t = args if isinstance(args, tuple) else (args,)

        assert len(args_t) >= self.n_args, \
            f"At least {self.n_args} arguments are required"

        if all(isinstance(a, Branch) for a in args_t[:self.n_args]):
            if len(args_t) == self.n_args:  # Access one Keldysh block
                return self.components[tuple(a.value for a in args_t)]
            else:  # Pass extra indices to the block
                g = self.components[
                    tuple(a.value for a in args_t[:self.n_args])
                ]
                return g[args_t[self.n_args:]]

        # Access a single point of the time mesh
        elif len(args_t) == self.n_args and \
                all(isinstance(a, ContourPoint) for a in args_t):
            g = self.components[tuple(a.branch.value for a in args_t)]
            return g[tuple(a.t for a in args_t)
                     + (slice(None),) * len(self.non_time_mesh.components)]

        else:
            raise IndexError(f"Unrecognized index format: {args}")

    def __setitem__(self, args, value):
        args_t = args if isinstance(args, tuple) else (args,)

        assert len(args_t) >= self.n_args, \
            f"At least {self.n_args} arguments are required"

        if all(isinstance(a, Branch) for a in args_t[:self.n_args]):
            if len(args_t) == self.n_args:  # Access one Keldysh block
                self.components[tuple(a.value for a in args_t)] = value
            else:  # Pass extra indices to the block
                g = self.components[
                    tuple(a.value for a in args_t[:self.n_args])
                ]
                g[args_t[self.n_args:]] = value

        # Access a single point of the time mesh
        elif len(args_t) == self.n_args and \
                all(isinstance(a, ContourPoint) for a in args_t):
            g = self.components[tuple(a.branch.value for a in args_t)]
            g[tuple(a.t for a in args_t)
              + (slice(None),) * len(self.non_time_mesh.components)] = value

        else:
            raise IndexError(f"Unrecognized index format: {args}")

    #
    # Simple arithmetic
    #

    def __eq__(self, other):
        return self.mesh == other.mesh and \
            self.target_subshapes == other.target_subshapes and \
            self.components == other.components

    def __iadd__(self, other):
        assert self.mesh == other.mesh
        assert self.target_subshapes == other.target_subshapes
        self.components += other.components
        return self

    def __isub__(self, other):
        assert self.mesh == other.mesh
        assert self.target_subshapes == other.target_subshapes
        self.components -= other.components
        return self

    def __imul__(self, x):
        self.components *= x
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

#
# Functions specific to the 2-point GFs
#


def greater(g: KeldyshGF) -> Gf:
    r"""Returns the greater component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    return g[Branch.BACKWARD, Branch.FORWARD]


def lesser(g: KeldyshGF) -> Gf:
    r"""Returns the lesser component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    return g[Branch.FORWARD, Branch.BACKWARD]


def retarded(g: KeldyshGF) -> Gf:
    r"""Returns the retarded component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_ret = Gf(mesh=g.mesh, target_shape=g.target_shape)
    tril_idx = tril_indices(len(g.time_mesh.components[0]))
    g_ret.data[tril_idx] = g_g.data[tril_idx] - g_l.data[tril_idx]
    return g_ret


def advanced(g: KeldyshGF) -> Gf:
    r"""Returns the advanced component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_adv = Gf(mesh=g.mesh, target_shape=g.target_shape)
    triu_idx = triu_indices(len(g.time_mesh.components[0]))
    g_adv.data[triu_idx] = g_l.data[triu_idx] - g_g.data[triu_idx]
    return g_adv


def from_lesser_greater(g_l: Gf, g_g: Gf, n_left_target_axes=None) -> KeldyshGF:
    r"""
    Construct a 2-point KeldyshGF object from a pair of lesser and greater
    real time Green's functions.
    """
    assert g_l.mesh == g_g.mesh
    assert g_l.target_shape == g_g.target_shape
    assert len(g_l.mesh.components) >= 2
    assert isinstance(g_l.mesh.components[0], MeshReTime) and \
           isinstance(g_l.mesh.components[1], MeshReTime)

    if n_left_target_axes is None:
        assert len(g_l.target_shape) % 2 == 0
        n_left_target_axes = len(g_l.target_shape) // 2

    target_subshapes = (g_l.target_shape[:n_left_target_axes],
                        g_l.target_shape[n_left_target_axes:])

    g = KeldyshGF(mesh=g_l.mesh, target_subshapes=target_subshapes)

    #
    # Fill Keldysh components
    #

    def ordered(z0, z1):
        return contour_ordering(z0, z1) == (0, 1)

    non_t_slice = (slice(None),) * len(g.non_time_mesh.components)

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

#
# Functions specific to the 3-point GFs
#


def from_vertex3_pieces(G: Dict[Tuple[int, int, int], Gf]) -> KeldyshGF:
    r"""
    Construct a 3-point vertex from 6 real-time correlators.

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

    G0 = next(iter(G.values()))
    assert all(p.mesh == G0.mesh for p in G.values())
    assert all(p.target_shape == G0.target_shape for p in G.values())

    ts_len = len(G0.target_shape)
    assert ts_len % 3 == 0, \
        "Target shape of the pieces must contain a multiple of 3 elements"
    target_subshapes = (
        G0.target_shape[:ts_len // 3],
        G0.target_shape[ts_len // 3: 2 * ts_len // 3],
        G0.target_shape[2 * ts_len // 3:]
    )

    Lambda = KeldyshGF(mesh=G0.mesh, target_subshapes=target_subshapes)

    #
    # Fill Keldysh components
    #

    for a0, a1, a2 in product(Branch, repeat=3):
        for t0, t1, t2 in Lambda.mesh:
            z0 = ContourPoint(a0, t0)
            z1 = ContourPoint(a1, t1)
            z2 = ContourPoint(a2, t2)
            order = contour_ordering(z0, z1, z2)
            Lambda[z0, z1, z2] = G[order][t0, t1, t2]

    return Lambda


# def __matmul__(self, other):
#     """Contour convolution"""
#     assert self.mesh == other.mesh
#
#     # Weights for quadrature rule
#     min_mesh_size = self.integrator.order + 1
#     assert len(self.mesh.components[1]) >= min_mesh_size, \
#         "Time grid must have at least %d nodes" % min_mesh_size
#     w = self.integrator.weights_conv(self.mesh.components[1])
#
#     res = deepcopy(self)
#
#     target_shape = ()
#
# subscripts_self = "ik..."
# subscripts_other = "kj..."
#     subscripts_res = "ij..."
#
#     if len(self.target_shape) == 1:  # Vector-valued
#         subscripts_self += "l"
#     elif len(self.target_shape) == 2:  # Matrix-valued
#     subscripts_self += "ml"
#         subscripts_res += "m"
#         target_shape = target_shape + (self.target_shape[0],)
#     elif len(self.target_shape) > 2:
#         raise RuntimeError("Contour convolution is not implemented "
#                         " for target dimensions > 2")
#
#     if len(other.target_shape) == 1:  # Vector-valued
#         subscripts_other += "l"
#     elif len(other.target_shape) == 2:  # Matrix-valued
#         subscripts_other += "ln"
#         subscripts_res += "n"
#         target_shape = target_shape + (other.target_shape[1],)
#     elif len(self.target_shape) > 2:
#         raise RuntimeError("Contour convolution is not implemented "
#                         " for target dimensions > 2")
#
#     subscripts = f"{subscripts_self},k,{subscripts_other}->{subscripts_res}"
#
#     res.target_shape = target_shape
#
#     FW, BW = Branch.FORWARD, Branch.BACKWARD
#     for b0, b1 in product(Branch, Branch):
#         res[b0, b1].data[:] = \
#             einsum(subscripts, self[b0, FW].data, w, other[FW, b1].data) - \
#             einsum(subscripts, self[b0, BW].data, w, other[BW, b1].data)
#
#     return res
