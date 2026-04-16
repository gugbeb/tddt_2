#
# Objects describing specific physical systems
#

from itertools import product
from typing import Tuple, Optional, Callable

import numpy as np

from triqs.gf import Gf, MeshReTime, MeshCycLat, MeshBrZone, MeshProduct
from triqs.lattice import BravaisLattice, BrillouinZone

from realevol.tinterp import TInterp as ti
import realevol.operators_tinterp as op
from realevol.init_state import make_equilibrium_init_state

from .keldysh import KeldyshGF
from .util import fermi


spin_names = ('up', 'dn')
"Names of two spin projections of an electron"


class SingleFermion:
    "Single fermionic degree of freedom with a given energy"

    def __init__(self, eps: float):
        "Construct a single fermion state with energy 'eps'."
        self.eps = eps

    def gf(self, t_mesh: MeshReTime, /,
           n: Optional[float] = None, T: Optional[float] = None):
        """
        Single-particle Green's function computed either for a fixed occupation
        'n' or at a fixed temperature 'T'. These two keyword arguments are
        mutually exclusive.
        """
        assert (n is None) ^ (T is None), \
            "Exactly one of 'n' and 'T' must be specified"

        tt_mesh = MeshProduct(t_mesh, t_mesh)
        g_l = Gf(mesh=tt_mesh, target_shape=())
        g_g = Gf(mesh=tt_mesh, target_shape=())

        occ = n if (n is not None) else fermi(self.eps / T)

        for time1, time2 in tt_mesh:
            ex = np.exp(-1j * self.eps * (time1 - time2))
            g_g[time1, time2] = -1j * (1.0 - occ) * ex
            g_l[time1, time2] = -1j * (-occ) * ex
        return KeldyshGF.from_lesser_greater(g_l, g_g)


class FermionBand:
    "Fermionic states forming a single band"

    def __init__(self, bz_mesh: MeshBrZone, eps_k: Callable):
        """
        Construct a band of fermionic states with dispersion law given by
        the function 'eps_k'.
        """
        self.bz_mesh = bz_mesh
        self.eps_k = eps_k

    def gf(self, t_mesh: MeshReTime, /,
           n_k: Optional[float] = None, T: Optional[float] = None):
        """
        Single-particle Green's function computed either for a fixed k-dependent
        occupation 'n_k' or at a fixed temperature 'T'. These two keyword
        arguments are mutually exclusive.
        """
        assert (n_k is None) ^ (T is None), \
            "Exactly one of 'n_k' and 'T' must be specified"

        tt_mesh = MeshProduct(t_mesh, t_mesh)
        ttk_mesh = MeshProduct(t_mesh, t_mesh, self.bz_mesh)
        g_l = Gf(mesh=ttk_mesh, target_shape=())
        g_g = Gf(mesh=ttk_mesh, target_shape=())

        for k in self.bz_mesh:
            eps_k = self.eps_k(k)
            occ_k = n_k(k) if (n_k is not None) else fermi(eps_k / T)
            for time1, time2 in tt_mesh:
                ex = np.exp(-1j * eps_k * (time1 - time2))
                g_g[time1, time2, k] = -1j * (1.0 - occ_k) * ex
                g_l[time1, time2, k] = -1j * (-occ_k) * ex
        return KeldyshGF.from_lesser_greater(g_l, g_g)


class FiniteSystem:
    "A finite system of fermions."
    def equilibrium_init_state(self, T: float, /, **kwargs):
        """
        Compute the equilibrium initial state of this model at temperature T
        using realevol. All keyword arguments are passed as parameter dictionary
        to make_equilibrium_init_state().
        """
        return make_equilibrium_init_state(self.hamiltonian,
                                           fermion_indices=self.fops,
                                           boson_indices=set(),
                                           temperature=T,
                                           params=kwargs)


class SquarePlaquette(FiniteSystem):
    """
    A square-shaped cluster of a 2D square lattice with local and non-local
    Hubbard interactions. Orientation of the Cartesian axes and the enumeration
    order of the cluster sites are shown below.

     ^ y
     |

    (N-1)------(2N-1)----- ... ---- ...
      |          |
     ...        ...
      |          |
     (2)-------(N+2)------ ... ---- ...
      |          |
     (1)-------(N+1)------ ... ---- ...
      |          |          |
     (0)--------(N)--------(2N)---- ... -> x
    """

    def __init__(self, N: int, *,
                 hopping: Optional[np.ndarray] = None,
                 local_int: Optional[np.ndarray] = None,
                 nonlocal_int: Optional[np.ndarray] = None,
                 vector_potential: Optional[Tuple[object, object]] = None):
        """
        Construct a square-shaped cluster.

        :param N: Number of sites along each axis.
        :type N: int

        :param hopping: Hopping matrix.
        :type hopping: np.ndarray, optional, default=zero array

        :param local_int: On-site Hubbard interaction constants (1D array).
        :type local_int: np.ndarray, optional, default=zero array

        :param nonlocal_int: Non-local interaction constants (2D array).
        :type nonlocal_int: np.ndarray, optional, default=zero array

        :param vector_potential: Vector potential
        :type vector_potential: Tuple[object, object],
                                optional, default = (0, 0)
        """
        self.N = N
        self.lattice = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])
        self.bz = BrillouinZone(self.lattice)
        self.r_mesh = MeshCycLat(self.lattice, self.N)
        self.k_mesh = MeshBrZone(self.bz, self.N)
        n_sites = len(self.r_mesh)

        # Hopping
        self.t = np.zeros((n_sites, n_sites), dtype=object) \
            if (hopping is None) else np.array(hopping)
        # Local interaction
        self.U = np.zeros((n_sites,), dtype=object) \
            if (local_int is None) else np.array(local_int)
        # Non-local interaction
        self.V = np.zeros((n_sites, n_sites), dtype=object) \
            if (nonlocal_int is None) else np.array(nonlocal_int)
        # Vector potential
        self.A = (0.0, 0.0) if (vector_potential is None) \
            else (*vector_potential,)

    @property
    def fops(self):
        "Fundamental operator set of this model"
        return set(product(spin_names, list(range(self.N ** 2))))

    @property
    def gf_struct(self):
        "Structure of TRIQS BlockGf object"
        return [(spin_names[0], self.N ** 2), (spin_names[1], self.N ** 2)]

    @property
    def hamiltonian(self):
        up, dn = spin_names
        n_sites = len(self.r_mesh)

        Ax, Ay = self.A

        h = op.Operator()
        # Hopping
        for (i, ri), (j, rj) in product(enumerate(self.r_mesh), repeat=2):
            # Compute Peierls prefactor
            dx, dy = (ri - rj)[:2]
            peierls = ti(Ax.mesh, np.exp(-1j * dx * Ax.data)) \
                if isinstance(Ax, ti) else np.exp(-1j * dx * Ax)
            peierls *= ti(Ay.mesh, np.exp(-1j * dy * Ay.data)) \
                if isinstance(Ay, ti) else np.exp(-1j * dy * Ay)

            h += sum(self.t[i, j] * peierls
                     * op.c_dag(sn, i) * op.c(sn, j) for sn in spin_names)

        # Interaction
        h += sum(self.U[i] * op.n(up, i) * op.n(dn, i) for i in range(n_sites))
        h += sum(self.V[i, j]
                 * (op.n(up, i) + op.n(dn, i)) * (op.n(up, j) + op.n(dn, j))
                 for i, j in product(range(n_sites), repeat=2))

        return h
