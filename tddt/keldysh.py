#
# Keldysh Green's functions and vertices
#

from enum import Enum
from copy import deepcopy
from itertools import product, takewhile, islice
from typing import Tuple, Dict, Union, Sequence
from triqs.gf import Gf, MeshReTime, MeshPoint, MeshProduct
import numpy as np

from .util import subscripts
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
                 target_shape: Tuple[int, ...] = None,
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
        if (target_shape is None) and (target_subshapes is None):
            self.target_subshapes = ((),) * self.n_args
            self.target_shape = ()

        elif (target_shape is None) and (target_subshapes is not None):
            assert len(target_subshapes) == self.n_args, \
                f"target_subshapes must contain {self.n_args} elements for " \
                f"a {self.n_args}-point function"
            self.target_subshapes = tuple(target_subshapes)
            self.target_shape = sum(target_subshapes, ())

        elif (target_shape is not None) and (target_subshapes is None):
            assert len(target_shape) % self.n_args == 0, \
                f"Target shape must have a multiple of {self.n_args} elements"
            self.target_shape = tuple(target_shape)
            tss_len = len(self.target_shape) // self.n_args
            self.target_subshapes = tuple(
                self.target_shape[i:i + tss_len]
                for i in range(0, len(self.target_shape), tss_len)
            )

        else:
            raise RuntimeError(
                "target_shape and target_subshapes are mutually exclusive"
            )

        #
        # Allocate data storage
        #

        self.components = np.array(
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

    def __matmul__(self, other):
        r"""
        Contour convolution over the last argument of 'self' and
        the first argument of 'other'.
        """
        return conv(self, other, [(-1, 0)])

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
    tril_idx = np.tril_indices(len(g.time_mesh.components[0]),
                               0,
                               len(g.time_mesh.components[1]))
    g_ret.data[tril_idx] = g_g.data[tril_idx] - g_l.data[tril_idx]
    return g_ret


def retarded_mod(g: KeldyshGF) -> Gf:
    r"""Returns the modified retarded component of a 2-point Keldysh
    Green's function. The modified version does not contain the time
    step-function in its definition."""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_ret = Gf(mesh=g.mesh, target_shape=g.target_shape)
    g_ret.data[...] = g_g.data - g_l.data
    return g_ret


def advanced(g: KeldyshGF) -> Gf:
    r"""Returns the advanced component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_adv = Gf(mesh=g.mesh, target_shape=g.target_shape)
    triu_idx = np.triu_indices(len(g.time_mesh.components[0]),
                               0,
                               len(g.time_mesh.components[1]))
    g_adv.data[triu_idx] = g_l.data[triu_idx] - g_g.data[triu_idx]
    return g_adv


def conj(g: Gf, *, n_left_indices=None) -> Gf:
    r"""
    Given a 2-point real time Green's function G_{a, b}(t, t'), returns its
    Hermitian conjugate [G_{b, a}(t', t)]^*. The conjugation is performed
    independently for each point of the non-time components of G's mesh.

    g: Input Green's function.
    n_left_indices: Number of axes in G's target shape corresponding to the
                    multi-index 'a'. By default, a half of all axes.
    """

    assert len(g.mesh.components) >= 2
    assert isinstance(g.mesh.components[0], MeshReTime)
    assert isinstance(g.mesh.components[1], MeshReTime)

    mesh = MeshProduct(g.mesh.components[1],
                       g.mesh.components[0],
                       *g.mesh.components[2:])

    if n_left_indices is None:
        assert len(g.target_shape) % 2 == 0, \
            "n_left_indices must be provided when the target shape of the GF " \
            "has an odd number of dimensions"
        nli = len(g.target_shape) // 2
    else:
        nli = n_left_indices

    nri = len(g.target_shape) - nli
    ts = g.target_shape[nli:] + g.target_shape[:nli]

    g_conj = Gf(mesh=mesh, target_shape=ts)
    axes_from = [0, 1, *range(-1, - nri - 1, -1)]
    axes_to = [1, 0, *range(-1 - nli, - nri - nli - 1, -1)]
    g_conj.data[:] = np.conj(np.moveaxis(g.data, axes_from, axes_to))

    return g_conj


def is_hermitian(g: KeldyshGF, *, atol=.0) -> bool:
    r"""Checks if a 2-point Keldysh Green's function is hermitian in the sense
    of the NESSi paper"""
    assert g.n_args == 2, "g must be a 2-point Green's function"

    if g.time_mesh.components[0] != g.time_mesh.components[1]:
        return False
    if g.target_subshapes[0] != g.target_subshapes[1]:
        return False

    nli = len(g.target_subshapes[0])

    axes_from = [0, 1, *range(-1, - nli - 1, -1)]
    axes_to = [1, 0, *range(-1 - nli, -2 * nli - 1, -1)]

    g_g = greater(g)
    g_l = lesser(g)
    g_ret_mod = retarded_mod(g)

    for comp in (g_g, g_l, g_ret_mod):
        if not np.allclose(comp.data,
                           -np.conj(np.moveaxis(comp.data, axes_from, axes_to)),
                           atol=atol):
            return False

    return True


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


def conv(a: KeldyshGF,  # noqa: C901
         b: KeldyshGF,
         coupled_args: Sequence[Tuple[int, int]] = [],
         *,
         free_args: Tuple[Sequence[int], Sequence[int]] = None,
         ) -> KeldyshGF:
    r"""
    Compute a contour convolution and a sum over its corresponding target
    indices of two contour function 'a' and 'b' w.r.t. one or more pairs
    of arguments.

    a: First function in the convolution.
    b: Second function in the convolution.
    coupled_args: Each element of this tuple is a pair of indices of the
                  arguments to integrate over (one index for 'a' and the other
                  for 'b'). Negative indices are interpreted as counting
                  from the end of the respective argument list.
    free_args: Specifies how arguments of the convolution result are distributed
               between 'free' (non-integrated) arguments of 'a' and 'b'.
               For example, coupled_args = (1, 2),
               free_args = ((0, 2, 3), (1, 4)) may results in the following
               convolution,

               f(z_0, z_1, z_2, z_3, z_4) = \inf_C dz'
                    a(z_0, z', z_2, z_3) b(z_1, z_4, z').
    """

    # Normalize the tuple of coupled arguments by resolving the negative
    # indices, removing duplicate pairs and sorting the resulting list
    conv_args = []
    for arg_a, arg_b in coupled_args:
        assert -a.n_args <= arg_a < a.n_args, \
            f"Wrong argument number {arg_a} for the first function with " \
            "{a.n_args} arguments"
        assert -b.n_args <= arg_b < b.n_args, \
            f"Wrong argument number {arg_b} for the second function with " \
            "{b.n_args} arguments"
        conv_args.append((arg_a if arg_a >= 0 else a.n_args + arg_a,
                          arg_b if arg_b >= 0 else b.n_args + arg_b))

    conv_args = sorted(list(set(conv_args)))
    n_conv_args = len(conv_args)

    n_args_res = a.n_args + b.n_args - 2 * n_conv_args
    assert n_args_res >= 0

    #
    # Process free_args
    #

    if free_args is None:
        # By default, arguments of the results are substituted into a and b
        # in the original left-to-right order.
        free_args_a = list(range(a.n_args - n_conv_args))
        free_args_b = list(range(a.n_args - n_conv_args, n_args_res))
    else:
        assert len(free_args) == 2, "Expected 2 elements in free_args"
        free_args_a, free_args_b = free_args
        assert len(free_args_a) == a.n_args - n_conv_args, \
            f"There must be exactly {a.n_args - n_conv_args} in " \
            f"free_args[0], got {len(free_args_a)}"
        assert len(free_args_b) == b.n_args - n_conv_args, \
            f"There must be exactly {b.n_args - n_conv_args} in " \
            f"free_args[1], got {len(free_args_b)}"

    free_args_all = set(free_args_a).union(free_args_b)
    assert len(free_args_all) == len(free_args_a) + len(free_args_b), \
        "No repeated numbers are allowed in free_args"
    assert free_args_all == set(range(n_args_res)), \
        f"Numbers in free_args must fully cover the {range(n_args_res)}"

    #
    # Label arguments with numbers
    #

    arg_indices_a = [None] * a.n_args
    arg_indices_b = [None] * b.n_args

    # Process coupled arguments
    for i, (arg_a, arg_b) in enumerate(conv_args):
        arg_indices_a[arg_a] = n_args_res + i
        arg_indices_b[arg_b] = n_args_res + i
    # Process free arguments of a
    i = 0
    for arg_a, ind_a in enumerate(arg_indices_a):
        if ind_a is None:
            arg_indices_a[arg_a] = free_args_a[i]
            i += 1
    # Process free arguments of b
    i = 0
    for arg_b, ind_b in enumerate(arg_indices_b):
        if ind_b is None:
            arg_indices_b[arg_b] = free_args_b[i]
            i += 1

    #
    # Handle the time components of the meshes
    #

    min_t_mesh_size = a.integrator.order + 1

    # Check mesh compatibility and compute integration weights
    w = []
    for arg_a, arg_b in conv_args:
        t_mesh_a = a.time_mesh.components[arg_a]
        t_mesh_b = b.time_mesh.components[arg_b]
        assert t_mesh_a == t_mesh_b, \
            f"Incompatible time mesh components {t_mesh_a} and {t_mesh_b} for" \
            f" the coupled argument pair ({arg_a}, {arg_b})"
        assert len(t_mesh_a) >= min_t_mesh_size, \
            f"Time mesh of a's argument {arg_a} must have " \
            f"at least {min_t_mesh_size} nodes"

        w.append(a.integrator.weights_conv(t_mesh_a))

    # Gather components of the resulting mesh
    mesh_comps_res = [None] * n_args_res
    for arg_a, arg_res in enumerate(arg_indices_a):
        if arg_res < n_args_res:
            mesh_comps_res[arg_res] = a.time_mesh.components[arg_a]
    for arg_b, arg_res in enumerate(arg_indices_b):
        if arg_res < n_args_res:
            mesh_comps_res[arg_res] = b.time_mesh.components[arg_b]

    # Generate einsum() subscripts
    ts = subscripts['time']
    subs_a_t = ''.join([ts[i] for i in arg_indices_a])
    subs_b_t = ''.join([ts[i] for i in arg_indices_b])
    subs_res_t = ts[:n_args_res]
    subs_w = [ts[n_args_res + i] for i in range(len(conv_args))]

    #
    # Handle the non-time components of the meshes
    #

    nt_mesh_a = a.non_time_mesh.components
    nt_mesh_b = b.non_time_mesh.components

    nts = subscripts['nontime']
    if nt_mesh_a == nt_mesh_b:
        # If the non-time components of the meshes of a and b agree, then we
        # use the same non-time mesh for the result
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_b_nt = ss
        subs_res_nt = ss

        mesh_comps_res += nt_mesh_a
    else:
        # Otherwise the result is defined on a direct product of the meshes.
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_res_nt = ss
        ss = nts[len(nt_mesh_a):len(nt_mesh_a) + len(nt_mesh_b)]
        subs_b_nt = ss
        subs_res_nt += ss

        mesh_comps_res += nt_mesh_a + nt_mesh_b

    #
    # Handle the targets
    #

    # Check subshapes compatibility
    for arg_a, arg_b in conv_args:
        subshape_a = a.target_subshapes[arg_a]
        subshape_b = b.target_subshapes[arg_b]
        assert subshape_a == subshape_b, \
            f"Incompatible target sub-shapes {subshape_a} and {subshape_b} for"\
            f" the coupled argument pair ({arg_a}, {arg_b})"

    # Gather components of the resulting subshapes
    subshapes_res = [None] * n_args_res
    for arg_a, arg_res in enumerate(arg_indices_a):
        if arg_res < n_args_res:
            subshapes_res[arg_res] = a.target_subshapes[arg_a]
    for arg_b, arg_res in enumerate(arg_indices_b):
        if arg_res < n_args_res:
            subshapes_res[arg_res] = b.target_subshapes[arg_b]

    # Collect sub-shapes of all n_args_res + n_conv_args arguments
    subshapes_all = subshapes_res[:]
    for arg_a, _ in conv_args:
        subshapes_all.append(a.target_subshapes[arg_a])

    # Compile a partitioned list of target subscripts
    tgs_it = iter(subscripts['target'])
    tgs = [''.join(islice(tgs_it, len(subshape))) for subshape in subshapes_all]

    # Generate einsum() subscripts
    subs_a_tg = ''.join([tgs[i] for i in arg_indices_a])
    subs_b_tg = ''.join([tgs[i] for i in arg_indices_b])
    subs_res_tg = ''.join(tgs[:n_args_res])

    #
    # Perform summation
    #

    subs_a = subs_a_t + subs_a_nt + subs_a_tg
    subs_b = subs_b_t + subs_b_nt + subs_b_tg
    subs_res = subs_res_t + subs_res_nt + subs_res_tg

    subs = f"{subs_a}," + ','.join(subs_w) + f",{subs_b}->{subs_res}"

    res = KeldyshGF(mesh=MeshProduct(*mesh_comps_res),
                    target_subshapes=subshapes_res)

    for br in product(Branch, repeat=n_args_res + n_conv_args):
        br_a = tuple(br[i] for i in arg_indices_a)
        br_b = tuple(br[i] for i in arg_indices_b)
        br_res = tuple(br[:n_args_res])
        sign = (-1) ** br[n_args_res:].count(Branch.BACKWARD)
        res[br_res].data[:] += sign * np.einsum(subs,
                                                a[br_a].data,
                                                *w,
                                                b[br_b].data,
                                                optimize="optimal")

    return res
