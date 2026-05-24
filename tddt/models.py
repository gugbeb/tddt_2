# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild
#
# tddt is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# tddt is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# tddt. If not, see <http://www.gnu.org/licenses/>.
#
# ##############################################################################


"""Objects describing specific physical systems"""

from itertools import product
from typing import Tuple, Optional, Callable

import numpy as np
from scipy.integrate import quad

from triqs.gf import Gf, MeshReTime, MeshBrZone, MeshProduct

from realevol.tinterp import TInterp as ti, is_constant
import realevol.operators_tinterp as op
from realevol.init_state import make_equilibrium_init_state

from .keldysh import Branch, KeldyshGF, Singular2PKeldyshGF
from .realevol import is_zero
from .util import fermi


spin_names = ('up', 'dn')
"Names of two spin projections of an electron"


class SingleFermion:
    "Single fermionic degree of freedom with a given energy"

    def __init__(self, eps: float):
        "Construct a single fermion state with energy 'eps'."
        self.eps = eps

    def gf(self, t_mesh: MeshReTime, /,
           n: Optional[float] = None, T: Optional[float] = None) -> KeldyshGF:
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
           n_k: Optional[float] = None, T: Optional[float] = None) -> KeldyshGF:
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


class FermionFlatBand:
    "Fermionic states forming a band with a flat density of states"

    def __init__(self, D: float, e0: float = 0.0):
        """
        Construct a flat band of fermionic states with the half-bandwidth 'D'
        centered on 'e0'.
        """
        assert D > 0, "The half-bandwidth must be positive"

        self.D = D
        self.e0 = e0

    def gf(self, t_mesh: MeshReTime, /, T: float) -> KeldyshGF:
        """
        Single-particle Green's function computed at a fixed temperature 'T'.
        """
        tt_mesh = MeshProduct(t_mesh, t_mesh)
        g_l = Gf(mesh=tt_mesh, target_shape=())
        g_g = Gf(mesh=tt_mesh, target_shape=())

        w1 = self.e0 - self.D
        w2 = self.e0 + self.D
        w_min_g, w_max_g = max(0, w1), max(0, w2)
        w_min_l, w_max_l = min(0, w1), min(0, w2)

        if T == 0.0:  # At zero temperature, we have simple expressions
            for time1, time2 in tt_mesh:
                dt = time1 - time2
                if dt == 0:
                    val_g = -1j * (w_max_g - w_min_g) / (2 * self.D)
                    val_l = 1j * (w_max_l - w_min_l) / (2 * self.D)
                else:
                    val_g = (np.exp(-1j * dt * w_max_g)
                             - np.exp(-1j * dt * w_min_g)) / (2 * self.D * dt)
                    val_l = -(np.exp(-1j * dt * w_max_l)
                              - np.exp(-1j * dt * w_min_l)) / (2 * self.D * dt)
                g_g[time1, time2] = val_g
                g_l[time1, time2] = val_l
        else:  # Use numerical integrator for energy integral
            x1, x2 = w1 / T, w2 / T

            def energy_integral(a, sign):
                return quad(lambda x: np.exp(-1j * a * x) * fermi(sign * x),
                            x1, x2, complex_func=True)[0]

            prefactor = T / (2 * self.D)
            for time1, time2 in tt_mesh:
                a = (time1 - time2) * T
                g_g[time1, time2] = -1j * prefactor * energy_integral(a, -1)
                g_l[time1, time2] = 1j * prefactor * energy_integral(a, 1)

        return KeldyshGF.from_lesser_greater(g_l, g_g)


class FiniteCluster:
    r"""
    A finite cluster with local and non-local Hubbard interactions.
    """

    def __init__(self,
                 coords: list[Tuple[float, float, float]], *,
                 hopping: Optional[np.ndarray] = None,
                 local_int: Optional[np.ndarray] = None,
                 nonlocal_int: Optional[np.ndarray] = None,
                 vector_potential:
                     Optional[Tuple[object, object, object]] = None):
        """
        Construct a cluster.

        :param coords: List of triplets -- Cartesian coordinates of sites
                       in the cluster.
        :type coords: list[tuple[float, float, float]]

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
        self.coords = coords
        self.N = len(coords)

        # Hopping
        self.hopping = np.zeros((self.N, self.N), dtype=object) \
            if (hopping is None) else np.array(hopping)

        # Local interaction
        self.local_int = np.zeros(self.N, dtype=object) \
            if (local_int is None) else np.array(local_int)

        # Non-local interaction
        self.nonlocal_int = np.zeros((self.N, self.N), dtype=object) \
            if (nonlocal_int is None) else np.array(nonlocal_int)

        # Vector potential
        self.peierls = np.zeros((self.N, self.N), dtype=object)
        self.vector_potential = (0.0, 0.0, 0.0) if (vector_potential is None) \
            else (*vector_potential,)

    def _compute_peierls(self):
        "Compute Peierls phases"
        Ax, Ay, Az = self.vector_potential
        for i, j in product(range(self.N), repeat=2):
            ri, rj = self.coords[i], self.coords[j]
            dx = ri[0] - rj[0]
            dy = ri[1] - rj[1]
            dz = ri[2] - rj[2]
            self.peierls[i, j] = ti(Ax.mesh, np.exp(-1j * dx * Ax.data)) \
                if isinstance(Ax, ti) else np.exp(-1j * dx * Ax)
            self.peierls[i, j] *= ti(Ay.mesh, np.exp(-1j * dy * Ay.data)) \
                if isinstance(Ay, ti) else np.exp(-1j * dy * Ay)
            self.peierls[i, j] *= ti(Az.mesh, np.exp(-1j * dz * Az.data)) \
                if isinstance(Az, ti) else np.exp(-1j * dz * Az)

    @property
    def vector_potential(self):
        "Vector potential"
        return self._A

    @vector_potential.setter
    def vector_potential(self, A: Tuple[object, object, object]):
        self._A = A
        self._compute_peierls()

    @property
    def fops(self):
        "Fundamental operator set of this model"
        return set(product(spin_names, list(range(self.N))))

    def spin_states(self, spin: int):
        r"""
        A list of (block index, inner index) pairs corresponding to the states
        with a certain spin projection (spin=0 -> up, spin=1 -> down).
        """
        assert 0 <= spin <= 1
        return [(spin_names[spin], i) for i in range(self.N)]

    @property
    def gf_struct(self):
        "Structure of TRIQS BlockGf object"
        return [(spin_names[0], self.N), (spin_names[1], self.N)]

    @property
    def hamiltonian(self):
        up, dn = spin_names

        h = op.Operator()
        # Hopping
        for i, j in product(range(self.N), repeat=2):
            h += sum(self.hopping[i, j] * self.peierls[i, j]
                     * op.c_dag(sn, i) * op.c(sn, j) for sn in spin_names)

        # Interaction
        h += sum(self.local_int[i] * op.n(up, i) * op.n(dn, i)
                 for i in range(self.N))
        h += sum(self.nonlocal_int[i, j]
                 * (op.n(up, i) + op.n(dn, i)) * (op.n(up, j) + op.n(dn, j))
                 for i, j in product(range(self.N), repeat=2))

        return h

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

    def hybridization(self,
                      t_mesh: MeshReTime,
                      imp_sites: list[int],
                      bath_sites: list[int],
                      *,
                      n: Optional[list[float]] = None,
                      T: Optional[float] = None) -> KeldyshGF:
        """
        Compute a hybridization function that describes effect of a subset of
        sites (bath sites) on another subset of sites (impurity sites).
        Occupations of the bath sites can be specified either directly or
        via the temperature of the bath.

        :param t_mesh: Time mesh used to construct the hybridization function.
        :type t_mesh: MeshReTime

        :param imp_sites: List of impurity sites.
        :type imp_sites: list[int]

        :param bath_sites: List of bath sites.
        :type bath_sites: list[int]

        :param n: Occupations of the bath, one element per site.
        :type n: list[float], optional

        :param T: Temperature of the bath.
        :type T: float, optional
        """
        assert (n is None) ^ (T is None), \
            "Exactly one of 'n' and 'T' must be specified"

        tt_mesh = MeshProduct(t_mesh, t_mesh)
        nimp = len(imp_sites)
        nbath = len(bath_sites)
        assert nimp > 0, "At least one impurity site is required"

        for ibath in bath_sites:
            if not is_zero(self.local_int[ibath]):
                raise RuntimeError(
                    f"Non-zero local interaction on bath site {ibath}"
                )
            if any((not is_zero(self.nonlocal_int[ibath, j]))
                   or (not is_zero(self.nonlocal_int[j, ibath]))
                   for j in range(self.N)):
                raise RuntimeError(
                    f"Non-zero nonlocal interaction on bath site {ibath}"
                )
        if any(not is_zero(self.hopping[bs1, bs2])
               for bs1, bs2 in product(bath_sites, repeat=2) if bs1 != bs2):
            raise RuntimeError("Coupled bath sites are not supported")

        # Compute bath GF
        def make_bath_gf(b1, b2):
            if b1 != b2:
                return KeldyshGF(mesh=tt_mesh)
            bs = bath_sites[b1[0]]
            eps = self.hopping[bs, bs]
            if isinstance(eps, ti):
                if is_constant(eps):
                    eps = eps(0)
                else:
                    raise RuntimeError("Bath state energies must be constant")
            sf = SingleFermion(eps)
            return sf.gf(t_mesh, T=T) \
                if (T is not None) else sf.gf(t_mesh, n=n[b1[0]])

        g_bath = KeldyshGF.from_arg_index_gen(
            make_bath_gf, mesh=tt_mesh, arg_index_shapes=((nbath,), (nbath,))
        )

        # Impurity-bath and bath-impurity hopping matrices
        Vib = Gf(mesh=t_mesh, target_shape=(nimp, nbath))
        Vbi = Gf(mesh=t_mesh, target_shape=(nbath, nimp))
        for (i, imps), (b, bs) in product(enumerate(imp_sites),
                                          enumerate(bath_sites)):
            vib = self.hopping[imps, bs] * self.peierls[imps, bs]
            vbi = self.hopping[bs, imps] * self.peierls[bs, imps]
            for t in t_mesh:
                Vib[t][i, b] = vib(t.value) if isinstance(vib, ti) else vib
                Vbi[t][b, i] = vbi(t.value) if isinstance(vbi, ti) else vbi

        # Hybridization function (single spin)
        delta = Singular2PKeldyshGF.from_retime(Vib) \
            @ g_bath \
            @ Singular2PKeldyshGF.from_retime(Vbi)

        # Add spin indices to the hybridization function
        Delta = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2, nimp), (2, nimp)))
        for br in product(Branch, repeat=2):
            for spin in range(2):
                for i1, i2 in product(range(nimp), repeat=2):
                    Delta[br][spin, i1, spin, i2] = delta[br][i1, i2]

        return Delta
