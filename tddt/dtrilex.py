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

"""Dual TRILEX theory"""

from copy import deepcopy
from enum import Enum
from itertools import product
from typing import Union, Sequence
import numpy as np

from triqs.gf import MeshReTime, MeshBrZone, MeshProduct, Gf

import realevol.operators_tinterp as op

from .keldysh import (Branch,
                      KeldyshGF,
                      Singular2PKeldyshGF,
                      herm_conj,
                      conv,
                      target_dot)
from .models import FiniteCluster
from .lattice import local_part
from .realevol import (
    compute_keldysh_gf,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)
from .vie2 import solve_vie2


IndicesType = tuple[Union[int, str], Union[int, str]]


class Channel(Enum):
    "Bosonic channel."
    CHARGE = 0
    "Charge channel"
    SPIN = 1
    "Spin channel"


class DualTRILEX:
    r"""
    Implementation of the "single-shot" dual TRILEX theory.
    """

    def __init__(self, system_ref: FiniteCluster,
                 t_mesh: MeshReTime,
                 *,
                 imp_states_up: list[IndicesType],
                 imp_states_dn: list[IndicesType]):
        r"""
        Initialize a dual TRILEX calculation.

        ref_system: Model object representing the reference system.
        t_mesh: Real time mesh used to define correlation functions.
        imp_states_up: A list of (block index, inner index) pairs corresponding
                       to the spin-up impurity manifold of the reference system.
        imp_states_dn: A list of (block index, inner index) pairs corresponding
                       to the spin-down impurity manifold of the reference
                       system.
        """
        self.system_ref = system_ref
        self.t_mesh = t_mesh
        self.tt_mesh = MeshProduct(t_mesh, t_mesh)
        self.ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)

        assert len(imp_states_up) == len(imp_states_dn), \
            "The numbers of spin-up and spin-down impurity states must be equal"
        self.imp_states_up = imp_states_up
        self.imp_states_dn = imp_states_dn

    def compute_ref_init_state(self, T: float, /, **kwargs):
        r"""
        Call realevol to compute the initial thermal state of the reference
        system.

        T: Temperature.
        kwargs: Parameters to be passed to make_equilibrium_init_state().
        """

        # Compute the initial state
        self.init_state_ref = self.system_ref.equilibrium_init_state(
            T, **kwargs
        )

    def compute_ref_correlators(self, **kwargs):
        r"""
        Call realevol to compute correlation functions of the reference system.
        The computed functions are stored as object attributes:

        self.g_ref: Single-particle Green's function of the reference system.
        self.g_imp: Single-particle Green's function of the impurity subsystem.
        self.chi_imp: Susceptibility of the impurity subsystem.
        self.Lambda: Three-point vertex of the impurity subsystem.

        kwargs: Parameters to be passed to compute_*() methods of realevol.
        """
        N = self.system_ref.N
        Nimp = len(self.imp_states_up)
        imp_states = [self.imp_states_up, self.imp_states_dn]

        def make_n_imp(spin, iimp1, iimp2):
            return op.c_dag(*imp_states[spin][iimp1]) \
                * op.c(*imp_states[spin][iimp2])

        #
        # Reference system GF
        #

        g_ref = compute_keldysh_gf(self.system_ref.gf_struct,
                                   self.init_state_ref,
                                   self.system_ref.hamiltonian,
                                   self.t_mesh,
                                   params=kwargs)

        self.g_ref = KeldyshGF(mesh=self.tt_mesh,
                               arg_index_shapes=((2, N), (2, N)))
        # Impurity GF
        self.g_imp = KeldyshGF(mesh=self.tt_mesh,
                               arg_index_shapes=((2, Nimp), (2, Nimp)))

        for spin1, spin2 in product(range(2), repeat=2):
            indices1 = self.system_ref.spin_states(spin1)
            indices2 = self.system_ref.spin_states(spin2)
            for (n1, ind1), (n2, ind2) in product(enumerate(indices1),
                                                  enumerate(indices2)):
                bl1, i1 = ind1
                bl2, i2 = ind2
                # Skip elements corresponding to different blocks of g_ref
                if bl1 != bl2:
                    continue

                for br in product(Branch, repeat=2):
                    self.g_ref[br][spin1, n1, spin2, n2] \
                        << g_ref[bl1][br][i1, i2]

                if (ind1 in imp_states[spin1]) and (ind2 in imp_states[spin2]):
                    iimp1 = imp_states[spin1].index(ind1)
                    iimp2 = imp_states[spin2].index(ind2)
                    for br in product(Branch, repeat=2):
                        self.g_imp[br][spin1, iimp1, spin2, iimp2] << \
                            self.g_ref[br][spin1, n1, spin2, n2]

        #
        # Impurity susceptibility
        #

        # Generator of scalar-valued elements of impurity susceptibility
        def generator_chi_imp(ind1, ind2):
            chan1, iimp1, iimp2 = ind1
            chan2, iimp3, iimp4 = ind2

            s1 = 1 if (chan1 == Channel.CHARGE.value) else -1
            op1 = make_n_imp(0, iimp1, iimp2) + s1 * make_n_imp(1, iimp1, iimp2)
            s2 = 1 if (chan2 == Channel.CHARGE.value) else -1
            op2 = make_n_imp(0, iimp3, iimp4) + s2 * make_n_imp(1, iimp3, iimp4)

            return -1j * compute_keldysh_conn_correlator_2t(
                op1, op2,
                self.init_state_ref, self.system_ref.hamiltonian, self.t_mesh,
                params=kwargs
            )

        self.chi_imp = KeldyshGF.from_arg_index_gen(
            generator_chi_imp,
            mesh=self.tt_mesh,
            arg_index_shapes=((2, Nimp, Nimp), (2, Nimp, Nimp))
        )

        #
        # Three-point correlator of the impurity
        #

        # Generator of scalar-valued elements of the correlator
        def generator_corr_3t_imp(ind1, ind2, ind3):
            spin1, iimp1 = ind1
            spin2, iimp2 = ind2
            chan, iimp3, iimp4 = ind3

            c_index = imp_states[spin1][iimp1]
            c_dag_index = imp_states[spin2][iimp2]

            s = 1 if (chan == Channel.CHARGE.value) else -1
            n_op = make_n_imp(0, iimp3, iimp4) + s * make_n_imp(1, iimp3, iimp4)

            return compute_keldysh_vertex3(
                c_index, c_dag_index, n_op,
                self.init_state_ref,
                self.system_ref.hamiltonian,
                self.t_mesh,
                params=kwargs
            )

        self.corr_3t_imp = KeldyshGF.from_arg_index_gen(
            generator_corr_3t_imp,
            mesh=self.ttt_mesh,
            arg_index_shapes=((2, Nimp), (2, Nimp), (2, Nimp, Nimp))
        )

    def compute_bare_lines_vertex(
        self, eps_tk: Gf, Delta: KeldyshGF, U_tq: Gf, U_dc: np.array
    ):
        r"""
        Compute bare dual lines and the three-point vertex function necessary
        to evaluate D-TRILEX diagrams.

        eps_tk: Lattice dispersion \eps_{\sigma,l,\sigma',l'}(t, k).
        Delta: Hybridization function \Delta(\sigma,l,\sigma',l')(t, t').
        U_tq: Interaction U^\varsigma_{l_1,l_2,l_3,l_4}(t, q).
        U_dc: Double-counting interaction shifts U^\varsigma_{l_1,l_2,l_3,l_4}.
        """
        Nimp = len(self.imp_states_up)

        assert (isinstance(eps_tk.mesh, MeshProduct)
                and len(eps_tk.mesh.components) == 2
                and isinstance(eps_tk.mesh.components[0], MeshReTime)
                and isinstance(eps_tk.mesh.components[1], MeshBrZone)), \
            "'eps_tk' must be defined on MeshProduct(MeshReTime, MeshBrZone)"
        assert eps_tk.target_shape == (2, Nimp, 2, Nimp), \
            f"For {Nimp} impurity state(s), eps_tk.target_shape " \
            f"must be {(2, Nimp, 2, Nimp)}"

        assert Delta.mesh == self.tt_mesh, \
            "'Delta' must be defined on the same time mesh as this object"
        assert Delta.arg_index_shapes == ((2, Nimp), (2, Nimp)), \
            f"For {Nimp} impurity state(s), Delta.arg_index_shapes " \
            f"must be {((2, Nimp), (2, Nimp))}"

        assert eps_tk.mesh == U_tq.mesh, \
            "'eps_tk' and 'U_tq' must be defined on the same mesh"

        assert U_tq.target_shape == (2, Nimp, Nimp, Nimp, Nimp), \
            f"For {Nimp} impurity state(s), U_tq.target_shape " \
            f"must be {(2, Nimp, Nimp, Nimp, Nimp)}."

        assert U_dc.shape == (2, Nimp, Nimp, Nimp, Nimp), \
            f"For {Nimp} impurity state(s), U_dc.shape " \
            f"must be {(2, Nimp, Nimp, Nimp, Nimp)}."

        self.eps_tk = Singular2PKeldyshGF.from_retime(eps_tk)
        eps_loc = local_part(self.eps_tk)
        for br in Branch:
            for t, k in self.eps_tk.mesh:
                self.eps_tk[br][t, k] = self.eps_tk[br][t, k] - eps_loc[br][t,]

        self.U_tq = Singular2PKeldyshGF(
            mesh=U_tq.mesh,
            arg_index_shapes=((2, Nimp, Nimp), (2, Nimp, Nimp))
        )
        self.tilde_U_tq = deepcopy(self.U_tq)

        for br in Branch:
            for t, k in self.U_tq.mesh:
                for ch in range(2):
                    self.U_tq[br][t, k][ch, :, :, ch, :, :] = U_tq[t, k][ch]
                    self.tilde_U_tq[br][t, k][ch, :, :, ch, :, :] = \
                        U_tq[t, k][ch] - 0.5 * U_dc[ch]

        eps_gimp = self.eps_tk @ self.g_imp
        gimp_eps = self.g_imp @ self.eps_tk
        eps_gimp_eps = eps_gimp @ self.eps_tk

        Delta_gimp = Delta @ self.g_imp
        eps_gimp_Delta = eps_gimp @ Delta
        Delta_gimp_eps = Delta @ gimp_eps
        Delta_gimp_Delta = Delta_gimp @ Delta

        self.k_mesh = eps_tk.mesh.components[1]
        ttk_mesh = MeshProduct(self.t_mesh, self.t_mesh, self.k_mesh)

        # Compute bare lines (fermions)
        Gd0_Q = KeldyshGF(mesh=ttk_mesh,
                          arg_index_shapes=Delta.arg_index_shapes)
        for br in product(Branch, repeat=2):
            for t1, t2, k in Gd0_Q.mesh:
                Gd0_Q[br][t1, t2, k] = eps_gimp_eps[br][t1, t2, k] \
                    - Delta_gimp_eps[br][t1, t2, k] \
                    - eps_gimp_Delta[br][t1, t2, k] \
                    + Delta_gimp_Delta[br][t1, t2]

        # FIXME: To silence the hermiticity check
        Gd0_Q = 0.5 * (Gd0_Q + herm_conj(Gd0_Q))

        Gd0_F = KeldyshGF(mesh=ttk_mesh,
                          arg_index_shapes=eps_gimp.arg_index_shapes)
        for br in product(Branch, repeat=2):
            for t1, t2, k in Gd0_F.mesh:
                Gd0_F[br][t1, t2, k] = - eps_gimp[br][t1, t2, k] \
                    + Delta_gimp[br][t1, t2]

        self.Gd0_reg = solve_vie2(Gd0_F, Gd0_Q)
        for br in product(Branch, repeat=2):
            for t1, t2, k in self.Gd0_reg.mesh:
                self.Gd0_reg[br][t1, t2, k] -= Delta[br][t1, t2]

        # Compute impurity polarization operator
        # U_dc as a diagonal matrix w.r.t. the channel indices
        U_dc_ch_mat = np.einsum("cijkl,cd->cijdkl", U_dc, np.eye(2))

        chi_imp_U = target_dot(self.chi_imp, U_dc_ch_mat, 1, (0, 1, 2))
        self.pi_imp = solve_vie2(chi_imp_U, self.chi_imp)

        # Compute impurity vertex
        U_pi_imp = target_dot(self.pi_imp, U_dc_ch_mat, 0, (3, 4, 5))
        self.Lambda = self.corr_3t_imp \
            - conv(self.corr_3t_imp, U_pi_imp, [(2, 0)])

        U_tq_pi_imp = self.U_tq @ self.pi_imp
        U_tq_pi_imp_U_tq = U_tq_pi_imp @ self.U_tq

        # FIXME: To silence the hermiticity check
        U_tq_pi_imp_U_tq = 0.5 * (U_tq_pi_imp_U_tq
                                  + herm_conj(U_tq_pi_imp_U_tq))
        self.W0prime = solve_vie2(-U_tq_pi_imp, U_tq_pi_imp_U_tq)


def polarization_2nd_order(Lambda: KeldyshGF,
                           g1: Sequence[KeldyshGF],
                           g2: Sequence[KeldyshGF]):
    r"""
    2nd order contribution to the polarization function.

    Lambda: 3-point vertex.
    g1: A list of all contributions to the 1st fermionic line.
    g2: A list of all contributions to the 2nd fermionic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert len(g1) > 0, "g1 must not be empty"
    assert len(g2) > 0, "g2 must not be empty"
    assert all(g1_i.n_args == 2 for g1_i in g1), \
        "All elements of g1 must be 2-point GFs"
    assert all(g2_i.n_args == 2 for g2_i in g2), \
        "All elements of g2 must be 2-point GFs"

    # f1(z_0, z_1, z_2) =
    #     \sum_i \int_C d\bar z \Lambda(z_0, \bar z, z_2) g1[i](\bar z, z_1)
    f1 = conv(Lambda, g1[0], [(1, 0)], free_args=([0, 2], [1]))
    for g1_i in g1[1:]:
        f1 += conv(Lambda, g1_i, [(1, 0)], free_args=([0, 2], [1]))

    # f2(z_0, z_1, z_2) =
    #     \sum_i \int_C d\bar z g2[i](\bar z, z_1) \Lambda(z_0, \bar z, z_2)
    f2 = conv(g2[0], Lambda, [(0, 1)], free_args=([1], [0, 2]))
    for g2_i in g2[1:]:
        f2 += conv(g2_i, Lambda, [(0, 1)], free_args=([1], [0, 2]))

    # \Pi(z_1, z_2) = -i \int_C dz' dz'' f1(z', z'', z_1) f2(z'', z', z_2)
    return -1j * conv(f1, f2, [(0, 1), (1, 0)])


def selfenergy_2nd_order(Lambda: KeldyshGF,
                         g: Sequence[KeldyshGF],
                         w: Sequence[KeldyshGF]):
    r"""
    2nd order contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: A list of all contributions to the fermionic line.
    w: A list of all contributions to the bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert len(g) > 0, "g must not be empty"
    assert len(w) > 0, "w must not be empty"
    assert all(g_i.n_args == 2 for g_i in g), \
        "All elements of g must be 2-point GFs"
    assert all(w_i.n_args == 2 for w_i in w), \
        "All elements of w must be 2-point GFs"

    # f1(z_1, z'''', z'') =
    #     \sum_i \int_C dz' \Lambda(z_1, z', z'') g[i](z', z'''')
    f1 = conv(Lambda, g[0], [(1, 0)], free_args=([0, 2], [1]))
    for g_i in g[1:]:
        f1 += conv(Lambda, g_i, [(1, 0)], free_args=([0, 2], [1]))

    # f2(z'''', z_2, z'') =
    #     \sum_i \int_C dz''' \Lambda(z'''', z_2, z''') w[i](z'', z''')
    f2 = conv(Lambda, w[0], [(2, 1)], free_args=([0, 1], [2]))
    for w_i in w[1:]:
        f2 += conv(Lambda, w_i, [(2, 1)], free_args=([0, 1], [2]))

    # \Sigma(z_1, z_2) = i \int_C dz'' dz'''' f1(z_1, z'''', z'')
    #                                         f2(z'''', z_2, z'')
    return 1j * conv(f1, f2, [(1, 0), (2, 2)])


def selfenergy_2nd_order_hf(Lambda: KeldyshGF,
                            g: Sequence[KeldyshGF],
                            w: Sequence[KeldyshGF]):
    r"""
    Hartree-Fock contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: A list of all contributions to the fermionic line.
    w: A list of all contributions to the bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert len(g) > 0, "g must not be empty"
    assert len(w) > 0, "w must not be empty"
    assert all(g_i.n_args == 2 for g_i in g), \
        "All elements of g must be 2-point GFs"
    assert all(w_i.n_args == 2 for w_i in w), \
        "All elements of w must be 2-point GFs"

    # n(z'') = \sum_i \int_C dz''' dz''''
    #     \Lambda(z''', z'''', z'') g[i](z'''', z''')
    n = conv(Lambda, g[0], [(0, 1), (1, 0)])
    for g_i in g[1:]:
        n += conv(Lambda, g_i, [(0, 1), (1, 0)])

    # wn(z') = sum_i \int_C dz'' w[i](z', z'') n(z'')
    wn = conv(w[0], n, [(1, 0)])
    for w_i in w[1:]:
        wn += conv(w_i, n, [(1, 0)])

    # \Sigma_{HF}(z_1, z_2) = -i \int_C dz' \Lambda(z_1, z_2, z') wn(z')
    return -1j * conv(Lambda, wn, [(2, 0)])
